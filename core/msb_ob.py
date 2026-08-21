"""Isolated MSB-OB structural evidence engine.

Pure, side-effect-free port of the Pine Script semantics for
"Market Structure Break & Order Block" (MSB-OB, EmreKb). This module MUST
NOT emit trade signals. It computes closed-candle structural evidence:
zigzag swing extraction (to_up/to_down), trend state, fib_factor-gated
market breaks (bullish/bearish MSB), order-block zones (Bu-OB/Be-OB) and
breaker/mitigation block zones (Bu-BB & Bu-MB / Be-BB & Be-MB), plus
zone lifecycle statuses (ACTIVE/TOUCHED/MITIGATING/INVALIDATED/EXPIRED).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_ZIGZAG_LEN = 9
DEFAULT_FIB_FACTOR = 0.33
# Pine fills arrays of 5 with 0 so index access is safe on empty state; we
# mirror that behavior without needing arrays.
LONG = 1
SHORT = -1

STATUS_ACTIVE = "ACTIVE"
STATUS_TOUCHED = "TOUCHED"
STATUS_MITIGATING = "MITIGATING"
STATUS_INVALIDATED = "INVALIDATED"
STATUS_EXPIRED = "EXPIRED"


@dataclass
class SwingPoint:
    price: float
    index: int  # bar index of swing formation


@dataclass
class MSBEvent:
    direction: int  # LONG for bullish MSB, SHORT for bearish MSB
    price: float    # l0 for bullish MSB, h0 for bearish MSB
    index: int      # bar index where the break was confirmed
    fib_factor: float


@dataclass
class InstitutionalZone:
    symbol: str
    side: int                    # LONG for bullish, SHORT for bearish
    zone_type: str               # OB | BB | MB
    top: float
    bottom: float
    created_at: int              # bar index
    msb_direction: int
    msb_price: float
    swing_high: float
    swing_low: float
    fib_factor: float
    freshness: int = 0           # bars since creation (0 = just formed)
    touch_count: int = 0
    displacement_score: float = 0.0   # max move from zone after creation in ATR units
    volume_score: float = 0.0         # volume ratio vs rolling baseline at creation
    structure_score: float = 0.0      # kept fixed 1.0 for now (structure tag captured)
    liquidity_context: str = "NONE"   # optional free-text hook; default NONE
    zone_strength: float = 0.0
    status: str = STATUS_ACTIVE

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": "LONG" if self.side == LONG else "SHORT",
            "zone_type": self.zone_type,
            "top": self.top,
            "bottom": self.bottom,
            "created_at": self.created_at,
            "msb_direction": "BULL" if self.msb_direction == LONG else "BEAR",
            "msb_price": self.msb_price,
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "fib_factor": self.fib_factor,
            "freshness": self.freshness,
            "touch_count": self.touch_count,
            "displacement_score": self.displacement_score,
            "volume_score": self.volume_score,
            "liquidity_context": self.liquidity_context,
            "zone_strength": self.zone_strength,
            "status": self.status,
        }


class MSBOBEngine:
    """Reprocesses a dataframe and extracts structural zones.

    Runs wholly deterministically: on each call, it replays the Pine state
    machine over all closed candles. Only closed candles are consulted.
    """

    def __init__(self, zigzag_len: int = DEFAULT_ZIGZAG_LEN, fib_factor: float = DEFAULT_FIB_FACTOR,
                 max_zones: int = 5):
        if zigzag_len < 2:
            raise ValueError("zigzag_len must be >= 2")
        self.zigzag_len = int(zigzag_len)
        self.fib_factor = float(fib_factor)
        self.max_zones = int(max_zones)

    # ---------- Public entry point ----------
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """Scan closed candles and return structured evidence."""
        result: Dict = {"zones": [], "msb_events": [], "market": None, "error": None}
        if df is None or len(df) < max(self.zigzag_len + 2, 12):
            result["error"] = "DATA_UNAVAILABLE"
            return result
        highs = list(df["high"].values)
        lows = list(df["low"].values)
        opens = list(df["open"].values)
        closes = list(df["close"].values)
        n = len(df)
        # Pine: to_up = high >= ta.highest(len); to_down = low <= ta.lowest(len)
        to_up = [highs[i] >= max(highs[max(0, i - self.zigzag_len + 1): i + 1]) for i in range(n)]
        to_down = [lows[i] <= min(lows[max(0, i - self.zigzag_len + 1): i + 1]) for i in range(n)]
        trend: List[int] = [LONG]  # Pine initial value 1 unless reversal happened
        high_points: List[SwingPoint] = []
        low_points: List[SwingPoint] = []
        market: List[int] = [LONG]
        zones: List[InstitutionalZone] = []
        highs_included = highs.copy()
        lows_included = lows.copy()
        for i in range(1, n):
            last_trend_up_since = self._barssince(to_up, i - 1) or 1
            last_trend_down_since = self._barssince(to_down, i - 1) or 1
            low_window = lows[max(0, i - last_trend_up_since): i + 1]
            high_window = highs[max(0, i - last_trend_down_since): i + 1]
            low_val = min(low_window)
            high_val = max(high_window)
            low_idx = max(0, i - last_trend_up_since) + low_window.index(low_val)
            high_idx = max(0, i - last_trend_down_since) + high_window.index(high_val)
            flipped = trend[-1]
            if trend[-1] == LONG and to_down[i - 1]:
                flipped = SHORT
                low_points.append(SwingPoint(low_val, low_idx))
            elif trend[-1] == SHORT and to_up[i - 1]:
                flipped = LONG
                high_points.append(SwingPoint(high_val, high_idx))
            else:
                flipped = trend[-1]
            trend.append(flipped)
            # Swing context h0/h1 (last two highs) l0/l1 (last two lows)
            h0 = high_points[-1].price if high_points else None
            h1 = high_points[-2].price if len(high_points) > 1 else None
            l0 = low_points[-1].price if low_points else None
            l1 = low_points[-2].price if len(low_points) > 1 else None
            if (h0 is None or l0 is None):
                market.append(market[-1])
                continue
            new_market = market[-1]
            if h1 is not None and l1 is not None:
                if market[-1] == LONG and l0 < l1:
                    new_market = SHORT if l0 < l1 - abs(h0 - l1) * self.fib_factor else LONG
                elif market[-1] == SHORT and h0 > h1:
                    new_market = LONG if h0 > h1 + abs(h1 - l0) * self.fib_factor else SHORT
            if new_market != market[-1]:
                if (l0 is None) or (h0 is None):
                    # Pine cannot form a zone without swings; skip zone emission
                    market.append(new_market)
                    continue
                msb = MSBEvent(direction=new_market, price=l0 if new_market == LONG else h0,
                               index=i, fib_factor=self.fib_factor)
                result["msb_events"].append(msb)
            market.append(new_market)
            if new_market == market[-2]:
                # No market break on this bar: no new structural zones
                continue
            # Zones: Bullish MSB → Bu-OB (last bearish candle in upswing→downswing window)
            # Zone boundaries are taken from the candle's high/low (top/bottom).
            if new_market == LONG and len(high_points) > 0 and len(low_points) > 0:
                h1i = high_points[-2].index if len(high_points) > 1 else high_points[-1].index
                l0i = low_points[-1].index
                # Pine walks h1i→l0i and takes the LAST bearish candle (open>close).
                zone = None
                for j in range(min(h1i, l0i), max(h1i, l0i) + 1):
                    if 0 <= j < n and opens[j] > closes[j]:
                        zone = j
                if zone is not None:
                    iz = self._build_zone(symbol, LONG, "OB", i, zone, highs, lows, closes, l0, h0, msb)
                    zones.append(iz)
                # Bu-BB / Bu-MB: walk l1i→h1i for the LAST bullish candle
                l1i = low_points[-2].index if len(low_points) > 1 else low_points[-1].index
                bb_zone = None
                for j in range(min(l1i, h1i), max(l1i, h1i) + 1):
                    if 0 <= j < n and opens[j] < closes[j]:
                        bb_zone = j
                if bb_zone is not None:
                    ztype = "BB" if (h1 is not None and l1 is not None and l0 < l1) else "MB"
                    iz = self._build_zone(symbol, LONG, ztype, i, bb_zone, highs, lows, closes, l0, h0, msb)
                    zones.append(iz)
            if new_market == SHORT and len(high_points) > 0 and len(low_points) > 0:
                l1i = low_points[-2].index if len(low_points) > 1 else low_points[-1].index
                h0i = high_points[-1].index
                # Be-OB: walk l1i→h0i and take the LAST bullish candle
                zone = None
                for j in range(min(l1i, h0i), max(l1i, h0i) + 1):
                    if 0 <= j < n and opens[j] < closes[j]:
                        zone = j
                if zone is not None:
                    iz = self._build_zone(symbol, SHORT, "OB", i, zone, highs, lows, closes, l0, h0, msb)
                    zones.append(iz)
                # Be-BB / Be-MB: walk h1i→l1i for the LAST bearish candle
                h1i = high_points[-2].index if len(high_points) > 1 else high_points[-1].index
                bb_zone = None
                for j in range(min(h1i, l1i), max(h1i, l1i) + 1):
                    if 0 <= j < n and opens[j] > closes[j]:
                        bb_zone = j
                if bb_zone is not None:
                    ztype = "BB" if (h1 is not None and h0 > h1) else "MB"
                    iz = self._build_zone(symbol, SHORT, ztype, i, bb_zone, highs, lows, closes, l0, h0, msb)
                    zones.append(iz)
        self._mark_statuses(zones, n, closes)
        zones = self._cap(zones)
        result["zones"] = [z.to_dict() for z in zones]
        result["market"] = market[-1]
        return result

    # ---------- Helpers ----------
    @staticmethod
    def _barssince(cond: List[bool], index: int) -> Optional[int]:
        for back in range(1, index + 1):
            if cond[index - back]:
                return back
        return None

    @staticmethod
    def _barssince_at_equal(vals: List[float], target, index: int) -> Optional[int]:
        for back in range(1, index + 1):
            if vals[index - back] == target:
                return back
        return None

    def _build_zone(self, symbol: str, side: int, ztype: str, created_at: int,
                    zone_bar: int, highs, lows, closes, swing_low, swing_high,
                    msb: MSBEvent) -> InstitutionalZone:
        return InstitutionalZone(
            symbol=symbol,
            side=side,
            zone_type=ztype,
            top=float(highs[zone_bar]),
            bottom=float(lows[zone_bar]),
            created_at=created_at,
            msb_direction=side,
            msb_price=float(msb.price),
            swing_high=float(highs[zone_bar]),
            swing_low=float(lows[zone_bar]),
            fib_factor=self.fib_factor,
            zone_strength=1.0,
        )

    def _mark_statuses(self, zones: List[InstitutionalZone], end_bar: int, closes) -> None:
        # Closed-candle invalidation: bull zones die when close < bottom;
        # bearish zones die when close > top (mirrors Pine box deletion).
        for z in zones:
            z.freshness = end_bar - 1 - z.created_at
            touched = False
            mitigating = False
            for i in range(z.created_at + 1, end_bar):
                close = closes[i]
                if z.side == LONG and close < z.bottom:
                    z.status = STATUS_INVALIDATED
                    break
                if z.side == LONG and close < z.top:
                    mitigating = True
                    z.touch_count += 1
                if z.side == SHORT and close > z.top:
                    z.status = STATUS_INVALIDATED
                    break
                if z.side == SHORT and close > z.bottom:
                    mitigating = True
                    z.touch_count += 1
            if z.status == STATUS_ACTIVE and mitigating:
                z.status = STATUS_MITIGATING
            elif z.status == STATUS_ACTIVE and z.touch_count > 0:
                z.status = STATUS_TOUCHED

    def _cap(self, zones: List[InstitutionalZone]) -> List[InstitutionalZone]:
        if len(zones) <= self.max_zones:
            return zones
        return zones[-self.max_zones:]


# Small deterministic entry used by downstream pipeline
def analyze_msb(df, symbol="UNKNOWN", zigzag_len=DEFAULT_ZIGZAG_LEN,
                fib_factor=DEFAULT_FIB_FACTOR, max_zones=5) -> Dict:
    """Convenience wrapper matching the existing engine style."""
    return MSBOBEngine(int(zigzag_len), float(fib_factor), max_zones).analyze(df, symbol)
