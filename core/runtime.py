"""Runtime orchestration, exchange reconciliation and portfolio supervision."""
from __future__ import annotations

import os
import threading
import time
import traceback
import requests

import core.engine as E
import scanner.scanner as S
from portfolio.manager import PortfolioManager
from portfolio.allocator import GlobalAssetAllocator
from scanner.deep_scanner import DeepScanner

# Compatibility exports: legacy modules still call these names through E.
globals().update({k: v for k, v in vars(E).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(S).items() if not k.startswith("__")})

MAX_OPEN_POSITIONS = max(1, int(os.getenv("MAX_OPEN_POSITIONS", "6")))
PORTFOLIO = PortfolioManager(MAX_OPEN_POSITIONS, E)
ALLOCATOR = GlobalAssetAllocator(PORTFOLIO, E)
DEEP_SCANNER = DeepScanner()


def sync_position_with_exchange(symbol):
    try:
        if hasattr(ex, "fetch_positions"):
            positions = safe_api_call(ex.fetch_positions, [normalize_symbol(symbol)])
        elif hasattr(ex, "fetch_open_positions"):
            positions = safe_api_call(ex.fetch_open_positions, [normalize_symbol(symbol)])
        else:
            return None
        if not positions:
            return None
        for pos in positions:
            pos_sym = pos.get("symbol", pos.get("info", {}).get("symbol", ""))
            if normalize_symbol(symbol) in pos_sym and float(pos.get("contracts", 0)) > 0:
                return pos
        return None
    except Exception as e:
        log_execution(f"[SYNC] error: {e}", "ERROR")
        return None


def get_realized_pnl(symbol, limit=100):
    try:
        trades = safe_api_call(ex.fetch_my_trades, normalize_symbol(symbol), limit=limit)
        if not trades:
            return 0.0, 0.0
        buy_value = sell_value = 0.0
        for trade in trades:
            qty = float(trade.get("amount", 0) or 0)
            price = float(trade.get("price", 0) or 0)
            cost = qty * price
            if str(trade.get("side", "")).lower() == "buy":
                buy_value += cost
            elif str(trade.get("side", "")).lower() == "sell":
                sell_value += cost
        pnl = sell_value - buy_value
        balance = get_balance_safe()
        return pnl, (pnl / balance * 100) if balance > 0 else 0.0
    except Exception as e:
        log_execution(f"[SYNC] error in get_realized_pnl: {e}", "ERROR")
        return 0.0, 0.0


def validate_position_state(local_pos, symbol):
    return local_pos if sync_position_with_exchange(symbol) is not None else None


def sync_all_states():
    """Compatibility sync for single-state callers.

    The production loop uses PORTFOLIO.manage_all(), which isolates state per
    symbol. This function remains for dashboard/manual API compatibility.
    """
    if E.PAPER_MODE:
        MEMORY["position_status"] = "OPEN" if STATE.get("open") else "CLOSED"
        MEMORY["total_pnl"] = PERF.get("total_pnl_usdt", 0.0)
        MEMORY["total_pnl_pct"] = PERF.get("total_pnl_pct", 0.0) * 100
        return

    symbols = PORTFOLIO.symbols()
    if symbols:
        PORTFOLIO.manage_all()
        MEMORY["position_status"] = "OPEN"
        MEMORY["current_positions"] = PORTFOLIO.snapshot()
    else:
        real_pos = sync_position_with_exchange(DEFAULT_SYMBOL)
        MEMORY["position_status"] = "OPEN" if real_pos else "CLOSED"
        MEMORY["current_position"] = real_pos
        real_pnl, real_pnl_pct = get_realized_pnl(DEFAULT_SYMBOL)
        MEMORY["total_pnl"] = real_pnl
        MEMORY["total_pnl_pct"] = real_pnl_pct


def _publish_portfolio_dashboard(dashboard_module):
    positions = PORTFOLIO.snapshot()
    E.DASHBOARD_STATE["positions"] = positions
    E.DASHBOARD_STATE["portfolio"] = {
        "open_positions": len(positions),
        "max_positions": PORTFOLIO.max_positions,
        "capacity": PORTFOLIO.max_positions - len(positions),
        "asset_classes": {},
        "risk": PORTFOLIO.risk_snapshot(),
    }
    for pos in positions:
        base = str(pos.get("symbol", "")).upper()
        cls = "CRYPTO"
        if any(x in base for x in ("XAU", "GOLD")): cls = "GOLD"
        elif any(x in base for x in ("WTI", "BRENT", "OIL", "CRUDE")): cls = "OIL"
        elif any(x in base for x in ("SPX", "SP500", "NAS", "US30", "DAX", "FTSE", "CAC", "NIKKEI")): cls = "INDEX"
        E.DASHBOARD_STATE["portfolio"]["asset_classes"][cls] = E.DASHBOARD_STATE["portfolio"]["asset_classes"].get(cls, 0) + 1
    if positions:
        primary = positions[0]
        E.DASHBOARD_STATE["position"] = primary
    else:
        E.DASHBOARD_STATE["position"] = None


def _run_discovery():
    """Run the 20-minute global discovery cycle.

    Legacy RF/Smart scanners remain auxiliary evidence providers. The canonical
    market-wide candidate source is DeepScanner, which seeds the dynamic
    50-60-symbol watchlist and never executes orders.
    """
    try:
        run_scanner_v2()
    except Exception as exc:
        log_execution(f"[SCANNER] v2: {exc}", "WARN")

    try:
        cands = scan_market_rf(top_n=40)
        MEMORY["rf_candidates"] = cands
        # Rebuild the RF dashboard from the candidates just scanned (shape-
        # compatible with the legacy build_rf_dashboard output) without a
        # second market scan.
        MEMORY["rf_dashboard"] = [
            {
                "symbol": c.get("symbol"),
                "status": c.get("status", "PROXIMITY"),
                "icon": "🔔" if c.get("rf_triggered") else "📡",
                "score": c.get("score", 0),
                "adx": c.get("adx", 0),
                "rsi": c.get("rsi", 0),
                "atrp": c.get("atrp", 0),
                "signal": c.get("rf_signal") or "N/A",
            }
            for c in (cands or [])
        ]
        MEMORY["rf_last_scan"] = time.time()
    except Exception as exc:
        log_execution(f"[SCANNER] RF auxiliary: {exc}", "WARN")

    try:
        candidates = DEEP_SCANNER.scan(force=True)
        MEMORY["top_candidates"] = candidates[:60]
        MEMORY["scanned_count"] = DEEP_SCANNER.stats.get("radar_scanned", 0)
        MEMORY["last_scan"] = DEEP_SCANNER.last_scan
    except Exception as exc:
        log_execution(f"[SCANNER] dynamic deep discovery: {traceback.format_exc()}", "ERROR")


def _service_watchlist_and_queue():
    """Continuously analyze watchlist, promote mature setups and re-evaluate queue."""
    try:
        updated = DEEP_SCANNER.monitor_watchlist()
        if updated:
            log_execution(
                f"[WATCHLIST] Deep batch analyzed={len(updated)} | "
                f"active={len(MEMORY.get('watchlist', {}))}",
                "INFO",
                debounce_key="watchlist_batch",
                debounce_sec=20,
            )
    except Exception as exc:
        log_execution(f"[WATCHLIST] monitor error: {exc}", "WARN")

    try:
        promoted = promote_to_queue()
        if promoted:
            log_execution(
                f"[QUEUE] Promoted {promoted} mature watchlist candidate(s)",
                "INFO",
                debounce_key="queue_promotions",
                debounce_sec=20,
            )
    except Exception as exc:
        log_execution(f"[QUEUE] promotion error: {exc}", "WARN")

    # Lifecycle cleanup lives here, never in dashboard read routes.
    try:
        cleanup_watchlist(float(os.getenv("WATCHLIST_EXPIRE_AFTER_SEC", "600")))
    except Exception as exc:
        log_execution(f"[WATCHLIST] cleanup error: {exc}", "WARN")

    MEMORY["queue_status"] = queue.get_status() if USE_EXECUTION_QUEUE else {}

    # Continuously evaluate global allocation over the current queue snapshot
    # (read-only publication; the actual gate still only fires on READY).
    try:
        if USE_EXECUTION_QUEUE:
            queue_snapshot = [c.to_dict() for c in queue._candidates.values()]
            report = ALLOCATOR.allocate(queue_snapshot, limit=PORTFOLIO.max_positions)
            MEMORY["portfolio_allocation"] = report.to_dict()
    except Exception as exc:
        log_execution(f"[PORTFOLIO] allocation evaluation failed: {exc}", "WARN")

def _execute_ready_queue_candidate():
    """Execute only a READY queue candidate through PortfolioManager.

    This preserves the intended pipeline:
    Discovery -> Watchlist -> Institutional Queue -> READY -> Portfolio execution.
    The scanner itself never places an order.
    """
    if not USE_EXECUTION_QUEUE:
        return False

    # Safety barrier: emergency kill switch (daily drawdown / loss limit) must
    # actually gate new entries, not merely exist as dead code.
    try:
        if emergency_kill_switch_active():
            log_execution(
                "[SAFETY] Kill switch active — new entries blocked",
                "WARN",
                debounce_key="kill_switch_active",
                debounce_sec=300,
            )
            return False
    except Exception as exc:
        log_execution(f"[SAFETY] kill-switch check failed: {exc}", "WARN")

    slots = PORTFOLIO.max_positions - PORTFOLIO.count()
    if slots <= 0:
        return False

    try:
        best = queue.get_best_candidate()
        # Unified with zone_score threshold: READY and EXECUTION both at 75.
        if best is None or best.priority_score < float(os.getenv("QUEUE_MIN_READY_SCORE", "75")):
            return False

        # News is a risk gate even after queue promotion; the candidate remains
        # visible until it either clears or is invalidated by queue re-evaluation.
        news_risk = 0.0
        watch = MEMORY.get("watchlist", {}).get(best.symbol, {})
        if isinstance(watch, dict):
            news_risk = float(watch.get("news_risk", 0) or 0)
        if news_risk >= float(os.getenv("NEWS_RISK_BLOCK", "80")):
            return False

        candidate = {
            "symbol": best.symbol,
            "side": best.side,
            "price": best.price,
            "sl": best.stop_loss,
            "tp1": best.take_profit_1,
            "tp2": best.take_profit_2,
            "score": best.priority_score,
            "atr": best.atr,
            "scenario": best.opportunity_type.value,
            "asset_class": watch.get("asset_class", "CRYPTO") if isinstance(watch, dict) else "CRYPTO",
            "news": watch.get("news", {}) if isinstance(watch, dict) else {},
        }

        # Global portfolio-allocator gate: rejects when class/direction caps
        # would be violated, and records the reason for dashboard explain.
        queue_snapshot = [c.to_dict() for c in queue._candidates.values()]
        alloc_report = ALLOCATOR.allocate(queue_snapshot + [candidate], limit=PORTFOLIO.max_positions)
        for decision in alloc_report.decisions:
            if decision.symbol == candidate["symbol"]:
                if not decision.allowed:
                    MEMORY["portfolio_allocation"] = alloc_report.to_dict()
                    log_execution(
                        f"[PORTFOLIO] {candidate['symbol']} rejected: {decision.reason}",
                        "INFO",
                        debounce_key=f"alloc_reject_{candidate['symbol']}",
                        debounce_sec=60,
                    )
                    return False
                break

        MEMORY["portfolio_allocation"] = alloc_report.to_dict()
        if PORTFOLIO.open_candidate(candidate):
            with queue._lock:
                if best.symbol in queue._candidates:
                    queue._candidates[best.symbol].state = ExecutionState.EXECUTED
                    queue.total_executed += 1
            log_execution(
                f"[QUEUE] READY -> EXECUTED {best.symbol} {best.side} "
                f"priority={best.priority_score:.1f}",
                "SUCCESS",
            )
            return True
    except Exception as exc:
        log_execution(f"[QUEUE] ready execution error: {exc}", "ERROR")
    return False


def portfolio_loop(dashboard_module=None):
    last_discovery = 0.0
    last_watch_service = 0.0
    last_snapshot = 0.0
    last_queue_eval = 0.0
    discovery_interval = float(os.getenv("GLOBAL_SCAN_INTERVAL_SEC", "1200"))
    watch_interval = float(os.getenv("WATCHLIST_SERVICE_INTERVAL_SEC", "20"))
    snapshot_interval = float(os.getenv("SNAPSHOT_INTERVAL", "60"))

    try:
        ex.load_markets()
        log_execution("Markets loaded", "INFO")
    except Exception as exc:
        log_execution(f"Failed to load markets: {exc}", "ERROR")

    tg_start(get_balance_safe(), "LIVE" if MODE_LIVE else "PAPER")

    while True:
        try:
            now = time.time()

            # Existing positions are serviced first.
            if PORTFOLIO.count():
                PORTFOLIO.manage_all()

            # Scanner/queue work runs outside any active portfolio context.
            PORTFOLIO.activate(None)
            try:
                if now - last_discovery >= discovery_interval:
                    _run_discovery()
                    last_discovery = now

                if now - last_watch_service >= watch_interval:
                    _service_watchlist_and_queue()
                    last_watch_service = now

                # Queue is re-evaluated frequently, but not on every main-loop
                # tick, to avoid hammering the exchange API.
                if USE_EXECUTION_QUEUE and now - last_queue_eval >= float(os.getenv("QUEUE_RE_EVAL_INTERVAL", "5")):
                    try:
                        queue.re_evaluate_all(lambda sym: get_ohlcv_safe(sym, 100))
                        MEMORY["queue_last_evaluation"] = time.time()
                        last_queue_eval = now
                    except Exception as exc:
                        log_execution(f"[QUEUE] fast re-evaluation error: {exc}", "WARN")

                _execute_ready_queue_candidate()
                if USE_EXECUTION_QUEUE:
                    queue.cleanup()
                    MEMORY["queue_status"] = queue.get_status()

                # Keep the institutional flow panel alive independently.
                if now - MEMORY.get("last_flow_update", 0) >= 60:
                    try:
                        update_institutional_flow_scanner()
                    except Exception as exc:
                        log_execution(f"[SCANNER] flow updater: {exc}", "WARN")
                    MEMORY["last_flow_update"] = now
            finally:
                PORTFOLIO.deactivate()

            if dashboard_module:
                _publish_portfolio_dashboard(dashboard_module)

            if time.time() - last_snapshot >= snapshot_interval:
                try:
                    if dashboard_module and hasattr(dashboard_module, "print_snapshot"):
                        dashboard_module.print_snapshot()
                except Exception as exc:
                    log_execution(f"[SNAPSHOT] {exc}", "WARN")
                last_snapshot = time.time()

            try:
                if dashboard_module and hasattr(dashboard_module, "hourly_cleanup"):
                    dashboard_module.hourly_cleanup()
            except Exception:
                pass

            time.sleep(max(1, BASE_SLEEP))

        except Exception:
            log_execution(f"Portfolio loop error: {traceback.format_exc()}", "ERROR")
            time.sleep(max(2, BASE_SLEEP))


def safe_main_loop(dashboard_module=None):
    while True:
        try:
            portfolio_loop(dashboard_module)
        except Exception:
            tb = traceback.format_exc()
            print(f"CRITICAL EXCEPTION: {tb}")
            try:
                log_execution(f"CRITICAL EXCEPTION: {tb}", "ERROR")
            except Exception:
                pass
            time.sleep(5)


def keep_alive():
    interval = int(os.environ.get("KEEPALIVE_INTERVAL", "50"))
    url = os.environ.get("KEEPALIVE_URL", "").strip()
    while True:
        try:
            if url:
                requests.get(url, timeout=5)
        except Exception as exc:
            try:
                log_execution(f"[KEEPALIVE] {exc}", "WARN")
            except Exception:
                pass
        time.sleep(max(10, interval))


RUNTIME_EXPORTS = [
    "sync_position_with_exchange",
    "get_realized_pnl",
    "validate_position_state",
    "sync_all_states",
    "safe_main_loop",
    "keep_alive",
]
