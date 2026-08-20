"""Dynamic multi-market discovery + continuous institutional watchlist.

Pipeline:
    venue discovery -> lightweight global radar -> TOP-N watchlist
    -> rotating deep analysis -> institutional execution queue.

The scanner discovers and analyzes opportunities but never places orders.
Portfolio/execution remain the only order authorities.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import core.engine as E
from news.service import NewsService, news_state_for_side
from strategy.engine import StrategyEngine
from scanner.universe import build_balanced


class DeepScanner:
    ASSET_CLASSES = ("CRYPTO", "GOLD", "OIL", "INDEX", "STOCK")

    def __init__(
        self,
        max_symbols: int = 60,
        *,
        exchange=None,
        market_loader: Optional[Callable[[], dict]] = None,
    ):
        self.watchlist_limit = int(os.getenv("DEEP_WATCHLIST_SIZE", str(max_symbols)))
        # Optional dependency injection keeps tests deterministic without changing
        # the production default, which continues to use the canonical exchange.
        self.exchange = exchange
        self._market_loader = market_loader
        self.radar_symbol_override = self._parse_radar_symbols(os.getenv("RADAR_SYMBOLS"))
        self.radar_symbols = int(os.getenv("DEEP_SCAN_RADAR_SYMBOLS", "240"))
        self.watch_batch_size = int(os.getenv("WATCHLIST_DEEP_BATCH_SIZE", "10"))
        self.watch_interval = float(os.getenv("WATCHLIST_DEEP_INTERVAL_SEC", "20"))
        self.discovery_interval = float(os.getenv("GLOBAL_SCAN_INTERVAL_SEC", "1200"))
        self.strategy = StrategyEngine()
        self.news = NewsService()
        self.last_scan = 0.0
        self.last_watch_update = 0.0
        self.last_result: List[dict] = []
        self.last_radar: List[dict] = []
        self.watch_symbols: List[str] = []
        self.watch_cursor = 0
        self.cycle_id = 0
        self.status = {
            "universe": "INIT",
            "radar": "INIT",
            "watchlist": "INIT",
            "universe_error": None,
            "radar_error": None,
        }
        self.stats = {
            "universe": 0,
            "radar_scanned": 0,
            "watchlist_target": self.watchlist_limit,
            "watchlist_active": 0,
            "deep_analyzed": 0,
            "queue_candidates": 0,
            "errors": 0,
            "last_discovery": 0.0,
            "last_watch_update": 0.0,
        }

    @staticmethod
    def _parse_radar_symbols(raw: Optional[str]) -> List[str]:
        """Parse the optional RADAR_SYMBOLS override without making it mandatory."""
        if not raw or not str(raw).strip():
            return []
        return [item.strip() for item in str(raw).split(",") if item.strip()]

    def _load_markets(self) -> dict:
        """Load markets through an injectable boundary while preserving E.ex in production."""
        if self._market_loader is not None:
            markets = self._market_loader()
            return markets if isinstance(markets, dict) else {}
        exchange = self.exchange or E.ex
        exchange.load_markets()
        markets = getattr(exchange, "markets", {}) or {}
        return markets if isinstance(markets, dict) else {}

    def _select_radar_rows(self, rows: List[dict]) -> List[dict]:
        """Apply RADAR_SYMBOLS only as an optional filter/override."""
        if not self.radar_symbol_override:
            return rows
        wanted = set(self.radar_symbol_override)
        return [row for row in rows if row.get("symbol") in wanted]

    def _publish_status(self) -> None:
        """Publish explicit scanner state without changing existing list contracts."""
        E.MEMORY["deep_scanner_status"] = dict(self.status)
        E.MEMORY["deep_universe_status"] = self.status["universe"]
        E.MEMORY["deep_radar_status"] = self.status["radar"]

    def _discover(self) -> List[dict]:
        self.status["universe_error"] = None
        try:
            markets = self._load_markets()
        except Exception as exc:
            self.stats["errors"] += 1
            self.status["universe"] = "PROVIDER_FAILURE"
            self.status["universe_error"] = str(exc)
            self._publish_status()
            E.log_execution(f"[DEEP] market discovery failed: {exc}", "WARN")
            return []

        if not markets:
            self.status["universe"] = "DATA_UNAVAILABLE"
            self._publish_status()
            E.log_execution("[DEEP] market discovery returned no markets", "WARN")
            return []

        radar_limit = len(markets) if self.radar_symbols <= 0 else min(self.radar_symbols, len(markets))
        rows = build_balanced(markets, radar_limit=radar_limit)
        rows = self._select_radar_rows(rows)
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["asset_class"]] = counts.get(row["asset_class"], 0) + 1

        self.stats["universe"] = len(rows)
        self.status["universe"] = "HEALTHY" if rows else "NO_OPPORTUNITY"
        self._publish_status()
        E.MEMORY["deep_universe_counts"] = counts
        E.MEMORY["deep_universe_size"] = len(rows)
        E.log_execution(
            f"[DEEP] Dynamic venue universe={len(rows)} | {counts}",
            "INFO",
            debounce_key="deep_universe_counts",
            debounce_sec=60,
        )
        return rows

    @staticmethod
    def _zone_context(sym: str, df) -> dict:
        """Return nearest support/resistance context with explicit zone_status.

        A computation failure is reported as ZONE_ERROR, an empty result as
        NO_VALID_ZONE, and missing frames as DATA_UNAVAILABLE — never silently
        folded into a legitimate distance/score.
        """
        status = "OK"
        if df is None or len(df) < 2:
            status = "DATA_UNAVAILABLE"
            zones = {"buy_zones": [], "sell_zones": []}
        else:
            try:
                zones = E.get_smart_zones(sym, df, None)
            except Exception as exc:
                zones = {"buy_zones": [], "sell_zones": []}
                status = "ZONE_ERROR"
                E.log_execution(f"[RADAR] zone computation failed for {sym}: {exc}", "WARN",
                                debounce_key=f"zone_err_{sym}", debounce_sec=300)
            if status == "OK" and not zones.get("buy_zones") and not zones.get("sell_zones"):
                status = "NO_VALID_ZONE"
        result = {
            "zone_status": status,
            "buy_zone": None,
            "sell_zone": None,
            "buy_distance": 999.0,
            "sell_distance": 999.0,
            "buy_strength": 0.0,
            "sell_strength": 0.0,
        }
        if df is None or len(df) < 2:
            return result
        price = float(df["close"].iloc[-1])
        buy = zones.get("buy_zones", [])
        sell = zones.get("sell_zones", [])
        buy_near = min(
            buy,
            key=lambda z: abs(price - float(z["price"])) / price,
            default=None,
        )
        sell_near = min(
            sell,
            key=lambda z: abs(price - float(z["price"])) / price,
            default=None,
        )
        result["buy_zone"] = buy_near
        result["sell_zone"] = sell_near
        result["buy_distance"] = (
            abs(price - float(buy_near["price"])) / price if buy_near else 999.0
        )
        result["sell_distance"] = (
            abs(price - float(sell_near["price"])) / price if sell_near else 999.0
        )
        result["buy_strength"] = float(buy_near.get("strength", 0)) if buy_near else 0.0
        result["sell_strength"] = float(sell_near.get("strength", 0)) if sell_near else 0.0
        return result

    def _radar(self, rows: List[dict]) -> List[dict]:
        radar = []
        attempted = 0
        data_failures = 0
        analysis_failures = 0
        scan_count = len(rows) if self.radar_symbols <= 0 else min(self.radar_symbols, len(rows))
        for row in rows[:scan_count]:
            attempted += 1
            sym = row["symbol"]
            try:
                df = E.get_ohlcv_safe(sym, 120)
                if df is None or len(df) < 40:
                    data_failures += 1
                    continue
                df.symbol = sym
                price = float(df["close"].iloc[-1])
                atr = float(E.compute_atr(df).iloc[-1])
                adx = float(E.compute_adx(df).iloc[-1])
                if price <= 0 or atr <= 0:
                    continue

                atr_pct = atr / price * 100
                rf = E.RFEngine(period=20, multiplier=3.5).compute(df)
                vol_ma = float(df["volume"].iloc[-20:].mean()) if "volume" in df else 0.0
                vol_ratio = float(df["volume"].iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
                momentum = E.MomentumFlowEngine.analyze_momentum_flow(df)
                smart = E.SmartMoneyEngine.analyze_smart_money(df)
                # Publish the freshly classified market regime so the Intent
                # Engine's layer-9 adaptive weights consume a live value
                # (evaluate_with_narrative is not reached by the current
                # watch-list-driven pipeline).
                try:
                    regime = E.detect_market_regime(df)
                    E.MEMORY["regime"] = regime
                except Exception:
                    pass  # regime publishing must never break the scan
                zones = self._zone_context(sym, df)

                # Lightweight radar score: discovery only. No entry decision here.
                score = 0.0
                score += min(3.0, max(0.0, (adx - 18.0) / 4.0))
                score += min(2.0, max(0.0, vol_ratio - 0.7))
                score += 1.5 if rf.get("triggered") else max(
                    0.0, 1.0 - abs(float(rf.get("distance", 1.0))) / 0.02
                )
                score += 1.5 if momentum.get("trend_expansion") else 0.0
                score += 1.5 if smart.get("smart_money_dominant") else 0.0

                # Discovery is explicitly zone-aware: prefer symbols near a strong
                # support/resistance/order-block proxy, rather than random movers.
                if zones["buy_zone"] is not None and zones["buy_distance"] <= 0.01:
                    score += min(2.5, zones["buy_strength"] / 40.0)
                if zones["sell_zone"] is not None and zones["sell_distance"] <= 0.01:
                    score += min(2.5, zones["sell_strength"] / 40.0)

                if smart.get("distribution_risk", 0) > 70:
                    score -= 2.0

                # Direction is only a watchlist hypothesis. The watchlist re-checks
                # both BUY and SELL continuously before queue promotion.
                candidates = []
                if rf.get("signal") in ("BUY", "SELL"):
                    candidates.append(rf["signal"])
                if smart.get("institutional_bias") in ("BUY", "SELL"):
                    candidates.append(smart["institutional_bias"])
                if momentum.get("flow_bias") in ("BUY", "SELL"):
                    candidates.append(momentum["flow_bias"])

                if zones["buy_distance"] < zones["sell_distance"]:
                    candidate_side = "BUY"
                elif zones["sell_distance"] < zones["buy_distance"]:
                    candidate_side = "SELL"
                elif candidates:
                    candidate_side = max(
                        set(candidates), key=candidates.count
                    )
                else:
                    candidate_side = "BUY" if float(df["close"].iloc[-1]) >= float(df["close"].iloc[-5]) else "SELL"

                radar.append(
                    {
                        "symbol": sym,
                        "asset_class": row["asset_class"],
                        "price": price,
                        "adx": round(adx, 1),
                        "atr_pct": round(atr_pct, 3),
                        "vol_ratio": round(vol_ratio, 2),
                        "rf_signal": rf.get("signal"),
                        "candidate_side": candidate_side,
                        "radar_score": round(max(0.0, score), 3),
                        "institutional_bias": smart.get("institutional_bias", "NEUTRAL"),
                        "flow_bias": momentum.get("flow_bias", "NEUTRAL"),
                        "buy_distance": round(zones["buy_distance"] * 100, 3),
                        "sell_distance": round(zones["sell_distance"] * 100, 3),
                        "buy_zone_strength": round(zones["buy_strength"], 1),
                        "sell_zone_strength": round(zones["sell_strength"], 1),
                        "zone_status": zones["zone_status"],
                        "data_quality": "OK" if zones["zone_status"] == "OK" else zones["zone_status"],
                    }
                )
            except Exception as exc:
                analysis_failures += 1
                self.stats["errors"] += 1
                E.log_execution(
                    f"[DEEP-RADAR] {sym} failed: {exc}",
                    "WARN",
                    debounce_key=f"deep_radar_{sym}",
                    debounce_sec=300,
                )

        radar.sort(key=lambda x: x["radar_score"], reverse=True)
        if radar:
            self.status["radar"] = "HEALTHY" if (data_failures == 0 and analysis_failures == 0) else "DEGRADED"
            self.status["radar_error"] = None
        elif attempted == 0:
            self.status["radar"] = "NO_OPPORTUNITY"
            self.status["radar_error"] = None
        elif data_failures == attempted:
            self.status["radar"] = "DATA_UNAVAILABLE"
            self.status["radar_error"] = f"No usable OHLCV for {attempted} candidate(s)"
        else:
            self.status["radar"] = "PROVIDER_FAILURE"
            self.status["radar_error"] = f"{analysis_failures} analysis failure(s)"
        self._publish_status()
        return radar

    def _seed_watchlist(self, radar: List[dict]) -> List[dict]:
        top = radar[: self.watchlist_limit]
        self.cycle_id += 1
        now = time.time()
        active = {}

        for item in top:
            sym = item["symbol"]
            side = item.get("candidate_side", "BUY")
            active[sym] = {
                "symbol": sym,
                "side": side,
                "score": round(float(item["radar_score"]), 2),
                "radar_score": round(float(item["radar_score"]), 2),
                "state": "DETECTED",
                "strength": "WEAK",
                "reasons": ["GLOBAL_RADAR"],
                "trade_type": "TREND",
                "asset_class": item["asset_class"],
                "price": item["price"],
                "rf_signal": item.get("rf_signal"),
                "institutional_bias": item.get("institutional_bias", "NEUTRAL"),
                "flow_bias": item.get("flow_bias", "NEUTRAL"),
                "buy_distance": item.get("buy_distance", 999),
                "sell_distance": item.get("sell_distance", 999),
                "buy_zone_strength": item.get("buy_zone_strength", 0),
                "sell_zone_strength": item.get("sell_zone_strength", 0),
                "deep_score": round(float(item["radar_score"]), 2),
                "deep_analyzed": False,
                "news": {"available": False, "risk": 0, "bias": "NEUTRAL", "headlines": []},
                "cycle_id": self.cycle_id,
                "last_update": now,
            }

        E.MEMORY["watchlist"] = active
        self.watch_symbols = list(active.keys())
        self.watch_cursor = 0
        self.stats["watchlist_active"] = len(active)
        E.MEMORY["watchlist_cycle_id"] = self.cycle_id
        E.MEMORY["watchlist_target"] = self.watchlist_limit
        E.MEMORY["watchlist_active"] = len(active)
        E.MEMORY["watchlist_last_seed"] = now
        return top

    def scan(self, force: bool = False) -> List[dict]:
        """Run the 20-minute whole-venue discovery cycle and seed TOP-N watchlist."""
        if not force and self.last_result and time.time() - self.last_scan < self.discovery_interval:
            return self.last_result

        rows = self._discover()
        radar = self._radar(rows) if rows else []
        self.last_radar = radar

        # A transient provider/data failure must not erase a valid watchlist.
        # Only an explicit successful scan is allowed to replace the active set.
        if self.status["universe"] in {"PROVIDER_FAILURE", "DATA_UNAVAILABLE"} or self.status["radar"] in {"PROVIDER_FAILURE", "DATA_UNAVAILABLE"}:
            top = self.last_result if self.last_result else []
            self.status["watchlist"] = "PRESERVED_DEGRADED" if top else "UNAVAILABLE"
        else:
            top = self._seed_watchlist(radar)
            self.status["watchlist"] = "HEALTHY" if top else "NO_OPPORTUNITY"
        self._publish_status()

        self.last_result = top
        self.last_scan = time.time()
        self.stats["radar_scanned"] = len(radar)
        self.stats["last_discovery"] = self.last_scan

        E.MEMORY["deep_radar"] = radar[:50]
        E.MEMORY["deep_scanner"] = top[: self.watchlist_limit]
        E.MEMORY["deep_scanner_last_scan"] = self.last_scan
        E.MEMORY["deep_radar_last_scan"] = self.last_scan
        E.MEMORY["deep_discovery_count"] = len(radar)
        E.MEMORY["scanned_count"] = len(radar)
        E.MEMORY["last_scan"] = self.last_scan
        E.log_execution(
            f"[DEEP] Discovery complete: universe={len(rows)} radar={len(radar)} "
            f"watchlist={len(top)} cycle={self.cycle_id}",
            "INFO",
        )
        return top

    @staticmethod
    def _fvg_context(df, side: str) -> dict:
        """Lightweight three-candle fair-value-gap context.

        This implements the useful SMC part from Vibe Trading without adding a
        third-party dependency. It is a watchlist evidence signal, never a
        standalone entry trigger.
        """
        if df is None or len(df) < 5:
            return {"present": False, "distance": 999.0, "type": "NONE"}

        price = float(df["close"].iloc[-1])
        found = None
        for i in range(len(df) - 1, 1, -1):
            a = df.iloc[i - 2]
            c = df.iloc[i]
            if float(c["low"]) > float(a["high"]):
                low, high, typ = float(a["high"]), float(c["low"]), "BULLISH"
            elif float(c["high"]) < float(a["low"]):
                low, high, typ = float(c["high"]), float(a["low"]), "BEARISH"
            else:
                continue
            if (side == "BUY" and typ == "BULLISH") or (side == "SELL" and typ == "BEARISH"):
                distance = 0.0 if low <= price <= high else min(
                    abs(price - low) / price, abs(price - high) / price
                )
                found = {"present": True, "distance": distance, "type": typ, "low": low, "high": high}
                break
        return found or {"present": False, "distance": 999.0, "type": "NONE"}

    @staticmethod
    def _orderbook_imbalance(ob) -> float:
        """Top-5 bid/ask depth imbalance in [-1, 1]."""
        try:
            bids = ob.get("bids", [])[:5]
            asks = ob.get("asks", [])[:5]
            bid_qty = sum(float(x[1]) for x in bids if len(x) >= 2)
            ask_qty = sum(float(x[1]) for x in asks if len(x) >= 2)
            total = bid_qty + ask_qty
            return (bid_qty - ask_qty) / total if total > 0 else 0.0
        except Exception:
            return 0.0

    def _analyze_symbol(self, entry: dict) -> dict | None:
        sym = entry["symbol"]
        asset = entry.get("asset_class", "CRYPTO")
        try:
            df = E.get_ohlcv_safe(sym, 150)
            if df is None or len(df) < 60:
                return None
            df.symbol = sym
            ob = E.get_orderbook_cached(sym, limit=10)
            news = self.news.assess(sym, asset)
            analysis_age = max(0.0, time.time() - float(entry.get("last_update", time.time()) or 0))
            data_age = 0.0
            try:
                if "timestamp" in df.columns:
                    ts_last = df["timestamp"].iloc[-1]
                    ts_last = ts_last.timestamp() if hasattr(ts_last, "timestamp") else float(ts_last) / 1000.0
                    data_age = max(0.0, time.time() - ts_last)
            except Exception:
                data_age = 0.0
            data_quality = "OK"
            if data_age > 900:
                data_quality = "STALE"
            elif ob is None:
                data_quality = "DEGRADED_NO_ORDERBOOK"
            ob_imbalance = self._orderbook_imbalance(ob)

            analyses = []
            for side in ("BUY", "SELL"):
                analysis = self.strategy.analyze(sym, side, df, ob)
                analysis["df"] = df
                score = float(analysis.get("score", 0.0))
                score += float(entry.get("radar_score", 0.0)) * 0.75

                fvg = self._fvg_context(df, side)
                if fvg.get("present") and float(fvg.get("distance", 999)) <= 0.005:
                    score += 0.75
                if (side == "BUY" and ob_imbalance >= 0.15) or (side == "SELL" and ob_imbalance <= -0.15):
                    score += 0.50

                if news.bias == "BULLISH" and side == "BUY":
                    score += 0.5
                elif news.bias == "BEARISH" and side == "SELL":
                    score += 0.5
                score -= float(news.risk) / 20.0
                analysis["watch_score"] = max(0.0, score)
                analyses.append(analysis)

            best = max(analyses, key=lambda x: float(x.get("watch_score", 0.0)))
            narrative = best.get("narrative") or {}
            score = float(best.get("watch_score", 0.0))
            reasons = []
            for key, label in (
                ("sweep", "Liquidity Sweep"),
                ("choch_bos", "BOS/CHoCH"),
                ("retest", "OB/Zone Retest"),
                ("rejection", "Rejection"),
                ("displacement", "Displacement"),
                ("volume_confirmation", "Volume"),
                ("rf_alignment", "RF"),
            ):
                if narrative.get(key):
                    reasons.append(label)

            state = "DETECTED"
            if narrative.get("retest"):
                state = "RETEST"
            if narrative.get("rejection"):
                state = "REJECTION"
            if narrative.get("displacement"):
                state = "DISPLACEMENT"
            if narrative.get("sweep") and narrative.get("choch_bos") and narrative.get("retest") and narrative.get("rejection"):
                state = "CONFIRMED"

            if news.risk >= float(os.getenv("NEWS_RISK_BLOCK", "80")):
                state = "NEWS_RISK"
            if fvg.get("present") and float(fvg.get("distance", 999)) <= 0.005:
                reasons.append("FVG")
            if (best["side"] == "BUY" and ob_imbalance >= 0.15) or (best["side"] == "SELL" and ob_imbalance <= -0.15):
                reasons.append("LOB Imbalance")

            strength = "STRONG" if score >= 8 else "MEDIUM" if score >= 5 else "WEAK"
            smart = best.get("smart_money") or {}
            momentum = best.get("momentum") or {}
            intent_details = best.get("intent_details") or {}
            fvg = self._fvg_context(df, best["side"])

            # Attach the canonical engine-owned zone for the winning side.
            zone_payload = None
            zone_status = "NO_VALID_ZONE"
            try:
                zmap = E.get_smart_zones(sym, df, ob)
                side_zones = zmap.get("buy_zones") if best["side"] == "BUY" else zmap.get("sell_zones")
                if side_zones:
                    z = side_zones[0]
                    zone_payload = {
                        "side": best["side"],
                        "price": float(z.get("price", z.get("level", 0.0))),
                        "strength": float(z.get("strength", 0.0)),
                        "type": z.get("type", "ZONE"),
                        "reaction_count": int((z.get("details") or {}).get("reaction_count", 0)),
                        "institutional_score": float((z.get("details") or {}).get("institutional_score", 0.0)),
                    }
                    zone_status = "OK"
            except Exception as exc:
                zone_status = "ZONE_ERROR"
                self.stats["errors"] += 1
                E.log_execution(
                    f"[WATCHLIST] {sym} zone attach failed: {exc}",
                    "WARN",
                    debounce_key=f"watch_zone_{sym}",
                    debounce_sec=300,
                )

            entry.update(
                {
                    "side": best["side"],
                    "price": best["price"],
                    "score": round(score, 3),
                    "deep_score": round(score, 3),
                    "narrative_score": round(float(best.get("narrative_score", 0)), 3),
                    "intent_score": round(float(best.get("intent_score", 0)), 2),
                    "intent_status": best.get("intent_status", "NEUTRAL"),
                    "intent_details": intent_details,
                    "state": state,
                    "strength": strength,
                    "reasons": reasons or ["Deep Analysis"],
                    "trade_type": "REVERSAL" if (narrative.get("sweep") or narrative.get("retest")) else "TREND",
                    "smart_money_bias": smart.get("institutional_bias", "NEUTRAL"),
                    "smart_money_bias_detailed": smart.get("institutional_bias_detailed", "NEUTRAL"),
                    "distribution_risk": round(float(smart.get("distribution_risk", 0)), 1),
                    "accumulation": round(float(smart.get("accumulation_strength", 0)), 1),
                    "momentum_expansion": bool(momentum.get("trend_expansion")),
                    "momentum_decay": bool(momentum.get("momentum_decay")),
                    "exhaustion_risk": round(float(momentum.get("exhaustion_risk", 0)), 1),
                    "continuation_strength": round(float(momentum.get("continuation_strength", 0)), 1),
                    "narrative": narrative,
                    "smart_money": smart,
                    "momentum": momentum,
                    "news": news.as_dict(),
                    "news_risk": float(news.risk),
                    "news_bias": news.bias,
                    "news_state": news_state_for_side(
                        news, best["side"], float(os.getenv("NEWS_RISK_BLOCK", "80"))
                    ),
                    "zone": zone_payload,
                    "zone_status": zone_status,
                    "orderbook_imbalance": round(ob_imbalance, 4),
                    "fvg": fvg,
                    "deep_analyzed": True,
                    "analysis_age": round(analysis_age, 1),
                    "data_age": round(data_age, 1),
                    "data_quality": data_quality,
                    "last_update": time.time(),
                }
            )
            return entry
        except Exception as exc:
            self.stats["errors"] += 1
            E.log_execution(
                f"[WATCHLIST] {sym} deep analysis failed: {exc}",
                "WARN",
                debounce_key=f"watch_deep_{sym}",
                debounce_sec=120,
            )
            return None

    def monitor_watchlist(self, force: bool = False) -> List[dict]:
        """Continuously deep-analyze a rotating batch of active watchlist symbols."""
        now = time.time()
        if not force and now - self.last_watch_update < self.watch_interval:
            return []

        watch = E.MEMORY.get("watchlist", {})
        if not isinstance(watch, dict) or not watch:
            self.stats["watchlist_active"] = 0
            return []

        self.watch_symbols = [s for s in self.watch_symbols if s in watch]
        if not self.watch_symbols:
            self.watch_symbols = list(watch.keys())
            self.watch_cursor = 0

        batch = []
        for _ in range(min(self.watch_batch_size, len(self.watch_symbols))):
            if not self.watch_symbols:
                break
            sym = self.watch_symbols[self.watch_cursor % len(self.watch_symbols)]
            self.watch_cursor = (self.watch_cursor + 1) % len(self.watch_symbols)
            batch.append(sym)

        updated = []
        for sym in batch:
            result = self._analyze_symbol(watch[sym])
            if result:
                watch[sym] = result
                updated.append(result)

        # Keep the watchlist dynamic: stale entries are removed only after they
        # have been rechecked, while the next global cycle can replace them.
        self.stats["deep_analyzed"] = self.stats.get("deep_analyzed", 0) + len(updated)
        self.stats["watchlist_active"] = len(watch)
        self.stats["last_watch_update"] = now
        E.MEMORY["watchlist"] = watch
        E.MEMORY["watchlist_active"] = len(watch)
        E.MEMORY["watchlist_last_update"] = now
        E.MEMORY["watchlist_deep_analyzed"] = self.stats["deep_analyzed"]

        # Keep dashboard deep scanner as the current ranked watchlist snapshot.
        ranked = sorted(
            watch.values(),
            key=lambda x: float(x.get("score", 0)),
            reverse=True,
        )
        E.MEMORY["deep_scanner"] = ranked[: self.watchlist_limit]
        self.last_watch_update = now
        return updated

    def top(self, limit: int = 6) -> List[dict]:
        self.monitor_watchlist(force=True)
        ranked = sorted(
            E.MEMORY.get("watchlist", {}).values(),
            key=lambda x: float(x.get("score", 0)),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]
