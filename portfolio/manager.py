"""Runtime portfolio orchestration for multiple independent positions.

The legacy trading brain is single-position internally. This manager isolates
that state per symbol and activates one context at a time, allowing the same
proven management code to service up to N independent positions without
duplicating the strategy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any
import copy
import os
import threading
import time

from portfolio.risk import PortfolioRiskGuard


@dataclass
class PositionContext:
    symbol: str
    state: dict
    trade_state: dict
    live_manager: Any
    paper_position: Any = None
    opened_at: float = 0.0


class PortfolioManager:
    def __init__(self, max_positions: int = 6, engine=None):
        self.max_positions = max(1, int(max_positions))
        self.engine = engine
        self.contexts: Dict[str, PositionContext] = {}
        self.active_symbol: Optional[str] = None
        self._lock = threading.RLock()
        self._base_state = None
        self._base_trade_state = None
        self.risk_guard = PortfolioRiskGuard(engine)
        self._last_perf_trade_count = 0
        # If an engine is supplied at construction time, bind immediately.
        # The runtime creates PortfolioManager(MAX_OPEN_POSITIONS, E), so
        # waiting for a later explicit bind leaves the blank-state snapshots
        # as None and causes STATE.update(None) during the first activation.
        if engine is not None:
            self.bind(engine)

    def bind(self, engine):
        self.engine = engine
        self.risk_guard.engine = engine
        if self._base_state is None:
            self._base_state = copy.deepcopy(engine.STATE)
        if self._base_trade_state is None:
            self._base_trade_state = copy.deepcopy(engine.TRADE_STATE)

    def count(self) -> int:
        return len(self.contexts)

    def symbols(self):
        return list(self.contexts.keys())

    @staticmethod
    def _asset_class(symbol: str, explicit: str | None = None) -> str:
        if explicit:
            return str(explicit).upper()
        text = str(symbol or "").upper()
        if any(x in text for x in ("XAU", "GOLD")):
            return "GOLD"
        if any(x in text for x in ("WTI", "BRENT", "OIL", "CRUDE")):
            return "OIL"
        if any(x in text for x in ("SP500", "US500", "NASDAQ", "USTECH", "US30", "DAX", "FTSE", "CAC", "NIKKEI", "INDEX")):
            return "INDEX"
        stock_hints = {"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "META", "TSLA", "JPM", "ARM", "INTC", "CRCL", "COIN", "PLTR"}
        base = text.replace(":USDT", "").replace("/USDT", "").replace("-USDT", "")
        if base in stock_hints:
            return "STOCK"
        return "CRYPTO"

    def can_open(self, symbol: str, asset_class: str | None = None) -> bool:
        with self._lock:
            if symbol in self.contexts or len(self.contexts) >= self.max_positions:
                return False
            if not self.risk_guard.can_open(len(self.contexts)):
                return False
            max_per_class = max(1, int(os.getenv("MAX_POSITIONS_PER_ASSET_CLASS", "2")))
            cls = self._asset_class(symbol, asset_class)
            current = sum(1 for pos in self.contexts.values() if self._asset_class(pos.symbol) == cls)
            return current < max_per_class

    def _capture(self):
        if not self.engine or not self.active_symbol:
            return
        ctx = self.contexts.get(self.active_symbol)
        if ctx is None:
            return
        ctx.state = copy.deepcopy(self.engine.STATE)
        ctx.trade_state = copy.deepcopy(self.engine.TRADE_STATE)
        ctx.live_manager = self.engine._live_manager
        paper = getattr(self.engine, "paper", None)
        if isinstance(paper, dict):
            ctx.paper_position = copy.deepcopy(paper.get("position"))

    def _blank(self):
        # Defensive fallback for callers that construct the manager before
        # the engine has initialized its state dictionaries.
        if self._base_state is None:
            self._base_state = copy.deepcopy(getattr(self.engine, "STATE", {}) or {})
        if self._base_trade_state is None:
            self._base_trade_state = copy.deepcopy(
                getattr(self.engine, "TRADE_STATE", {}) or {}
            )

        self.engine.STATE.clear()
        self.engine.STATE.update(copy.deepcopy(self._base_state or {}))
        self.engine.TRADE_STATE.clear()
        self.engine.TRADE_STATE.update(copy.deepcopy(self._base_trade_state or {}))

        paper = getattr(self.engine, "paper", None)
        if isinstance(paper, dict):
            paper["position"] = None

    def activate(self, symbol: Optional[str]):
        if not self.engine:
            raise RuntimeError("PortfolioManager is not bound to core.engine")
        with self._lock:
            self._capture()
            self.active_symbol = symbol
            if symbol is None:
                self._blank()
                return
            ctx = self.contexts.get(symbol)
            if ctx is None:
                self._blank()
                # Each new position gets its own LiveTradeManager state.
                ctx_manager = self.engine.LiveTradeManager(
                    self.engine._event_bus,
                    self.engine._exchange_sync,
                    self.engine._recovery_guard,
                )
                self.engine._live_manager = ctx_manager
                return
            self.engine.STATE.clear()
            self.engine.STATE.update(copy.deepcopy(ctx.state))
            self.engine.TRADE_STATE.clear()
            self.engine.TRADE_STATE.update(copy.deepcopy(ctx.trade_state))
            self.engine._live_manager = ctx.live_manager
            paper = getattr(self.engine, "paper", None)
            if isinstance(paper, dict):
                paper["position"] = copy.deepcopy(ctx.paper_position)

    def deactivate(self):
        with self._lock:
            self._capture()
            self.active_symbol = None
            self._blank()

    def _store_after_open(self, symbol: str, manager):
        self.contexts[symbol] = PositionContext(
            symbol=symbol,
            state=copy.deepcopy(self.engine.STATE),
            trade_state=copy.deepcopy(self.engine.TRADE_STATE),
            live_manager=manager,
            paper_position=copy.deepcopy(
                self.engine.paper.get("position")
            ) if isinstance(getattr(self.engine, "paper", None), dict) else None,
            opened_at=time.time(),
        )

    def open_candidate(self, candidate: dict) -> bool:
        """Open one ranked candidate using the preserved execution brain."""
        symbol = candidate["symbol"]
        if not self.can_open(symbol, candidate.get("asset_class")):
            return False

        self.activate(symbol)
        try:
            ok = bool(self.engine.execute_entry(
                candidate["side"], symbol, candidate["price"],
                candidate["sl"], candidate["tp1"], candidate["tp2"],
                candidate["score"],
                f"DEEP:{candidate.get('scenario','UNKNOWN')}|NEWS:{candidate.get('news',{}).get('bias','NEUTRAL')}",
                candidate["atr"],
                "INSTITUTIONAL",
                "DEEP_SCANNER",
                "SNIPER",
            ))
            if ok and self.engine.STATE.get("open"):
                self._store_after_open(symbol, self.engine._live_manager)
                self.engine.log_execution(
                    f"[PORTFOLIO] Opened {symbol} {candidate['side']} | "
                    f"slot {len(self.contexts)}/{self.max_positions}",
                    "SUCCESS",
                )
                return True
            return False
        finally:
            self.deactivate()

    def open_top(self, candidates, slots: Optional[int] = None) -> int:
        opened = 0
        target = self.max_positions - self.count() if slots is None else min(
            int(slots), self.max_positions - self.count()
        )
        if target <= 0:
            return 0

        for candidate in candidates:
            if opened >= target:
                break
            if not self.can_open(candidate["symbol"], candidate.get("asset_class")):
                continue
            # Avoid stacking highly correlated same-symbol/same-side duplicates.
            if self.open_candidate(candidate):
                opened += 1
        return opened

    def manage_all(self):
        """Synchronously service every active position through the legacy brain."""
        if not self.engine:
            return

        for symbol in list(self.contexts.keys()):
            self.activate(symbol)
            try:
                if not self.engine.STATE.get("open"):
                    self.contexts.pop(symbol, None)
                    continue

                self.engine.sync_position_state(symbol)
                if self.engine.STATE.get("open"):
                    self.engine._live_manager.manage_live_trade()

                if self.engine.STATE.get("open"):
                    price = self.engine.get_ticker_safe(symbol)
                    if price:
                        df = self.engine.get_ohlcv_safe(symbol, 50)
                        if df is not None:
                            try:
                                if self.engine.council_exit(df, price):
                                    self.engine.finalize_trade_with_reality(symbol)
                            except Exception as exc:
                                self.engine.log_execution(
                                    f"[PORTFOLIO] council_exit {symbol}: {exc}", "WARN"
                                )

                if not self.engine.STATE.get("open"):
                    # close_position_full() already performs finalization.
                    self.contexts.pop(symbol, None)
                else:
                    self._capture()
            except Exception as exc:
                self.engine.log_execution(f"[PORTFOLIO] manage {symbol}: {exc}", "ERROR")
            finally:
                self.risk_guard.sync_closed_trades()
                self.engine.MEMORY["portfolio_risk"] = self.risk_guard.snapshot(self.count())

                self.deactivate()

    def risk_snapshot(self):
        return self.risk_guard.snapshot(self.count())

    def close_symbol(self, symbol: str) -> bool:
        if symbol not in self.contexts:
            return False
        self.activate(symbol)
        try:
            if self.engine.STATE.get("open"):
                self.engine.close_position_full()
            closed = not self.engine.STATE.get("open")
            if closed:
                self.contexts.pop(symbol, None)
            else:
                self._capture()
            return closed
        finally:
            self.deactivate()

    def snapshot(self):
        out = []
        for symbol in list(self.contexts.keys()):
            ctx = self.contexts[symbol]
            s = ctx.state
            out.append({
                "symbol": symbol,
                "side": s.get("side"),
                "entry": s.get("entry", 0.0),
                "mark_price": s.get("mark_price", 0.0),
                "qty": s.get("qty", 0.0),
                "remaining_qty": s.get("remaining_qty", 0.0),
                "roe_pct": s.get("roe_pct", 0.0),
                "pnl_usdt": s.get("unrealized_pnl_usdt", 0.0),
                "sl": s.get("synthetic_sl", s.get("sl", 0.0)),
                "tp1": s.get("synthetic_tp1", s.get("tp1_price", 0.0)),
                "tp2": s.get("tp2_price", 0.0),
                "tp1_hit": bool(s.get("tp1_hit", False)),
                "tp2_hit": bool(s.get("tp2_hit", False)),
                "trailing_active": bool(s.get("trail_activated", False)),
                "confidence": s.get("current_confidence", 0.0),
                "regime": s.get("market_regime", "UNKNOWN"),
                "trade_state": s.get("trade_state", "UNKNOWN"),
            })
        return out
