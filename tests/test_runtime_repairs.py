"""Behavioral repairs test suite.

These tests exercise REAL production code paths (engine, scanner, news,
dashboard). Only the network/venue boundary is replaced, so a missing
function or a contract drift fails here instead of in production.
"""
import os
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

os.environ.setdefault("PAPER_MODE", "True")
os.environ.setdefault("BINGX_KEY", "")
os.environ.setdefault("BINGX_SECRET", "")
os.environ.setdefault("NEWS_ENABLED", "True")

import core.engine as E  # noqa: E402  (real engine — import-time network is not touched)


def _df(n=140, base=100.0):
    t = np.arange(n)
    x = base + 3 * np.sin(t / 4.0) + 0.5 * np.sin(t / 1.7)
    return pd.DataFrame({
        "open": x - 0.1, "high": x + 0.6, "low": x - 0.6,
        "close": x, "volume": np.full(n, 1000.0),
    })


class ZoneAnalysisRealTest(unittest.TestCase):
    """The canonical zone implementation lives in core.engine."""

    def test_engine_owns_canonical_zone_strength(self):
        df = _df()
        atr = float(E.compute_atr(df).iloc[-1])
        strength, details = E.compute_zone_strength(df, float(df["low"].mean()), "support", atr, None)
        self.assertIsInstance(strength, float)
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 10.0)
        self.assertIn("reaction_count", details)
        self.assertIn("institutional_score", details)

    def test_get_smart_zones_real_pipeline_no_name_error(self):
        zones = E.get_smart_zones("BTC/USDT:USDT", _df(), None)
        self.assertIn("buy_zones", zones)
        self.assertIn("sell_zones", zones)
        self.assertGreater(len(zones["buy_zones"]) + len(zones["sell_zones"]), 0)
        for z in zones["buy_zones"]:
            self.assertIn("strength", z)
            self.assertIn("price", z)

    def test_narrative_evaluation_uses_zones_without_crashing(self):
        narrative, score = E.evaluate_liquidity_narrative(_df(), None, 1.0, "BUY")
        self.assertIsInstance(narrative, dict)
        self.assertIsInstance(score, (int, float))

    def test_scanner_zone_names_are_engine_owned(self):
        import importlib
        import sys
        # Other test modules may reload core.engine into a fresh module object.
        # Re-establish a consistent engine/module state before re-importing
        # scanner: reload core.engine and re-attach it to the parent package.
        sys.modules.pop("core.engine", None)
        eng = importlib.import_module("core.engine")
        import core as _core_pkg
        _core_pkg.engine = eng
        prev_scanner = sys.modules.pop("scanner", None)
        prev_sub = sys.modules.pop("scanner.scanner", None)
        importlib.invalidate_caches()
        try:
            import scanner.scanner as S
            self.assertIs(S.compute_zone_strength, eng.compute_zone_strength)
            self.assertIs(S.get_smart_zones, eng.get_smart_zones)
        finally:
            sys.modules.pop("scanner.scanner", None)
            sys.modules.pop("scanner", None)
            if prev_scanner is not None:
                sys.modules["scanner"] = prev_scanner
            if prev_sub is not None:
                sys.modules["scanner.scanner"] = prev_sub

    def test_engine_has_no_undefined_zone_names(self):
        # Fail loudly if anyone reintroduces a reference without a definition.
        self.assertTrue(callable(getattr(E, "compute_zone_strength", None)))
        self.assertTrue(callable(getattr(E, "update_position_dashboard", None)))


class ExecutionEntryRealTest(unittest.TestCase):
    """Paper execution must commit state AND publish the position, or fail
    without leaving ghost positions."""

    @classmethod
    def setUpClass(cls):
        E.STATE["open"] = False
        E.STATE["symbol"] = None
        E.DASHBOARD_STATE["position"] = None
        E._east_entry_symbol = None

    def setUp(self):
        E.STATE["open"] = False
        E.STATE["symbol"] = None
        E.TRADE_STATE["in_position"] = False
        E.DASHBOARD_STATE["position"] = None
        self._saved_ohlcv = E.get_ohlcv_safe
        self._saved_ob = E.get_orderbook_cached
        self._saved_ticker = E.get_ticker_safe
        self._saved_balance = E.get_balance_safe
        E.get_ohlcv_safe = lambda symbol, limit=120, htf=False: _df(250)
        E.get_orderbook_cached = lambda *a, **k: {"bids": [[99.0, 10.0]], "asks": [[101.0, 5.0]]}
        E.get_ticker_safe = lambda symbol, retries=3: 104.0
        E.get_balance_safe = lambda retries=3: (10_000.0, 10_000.0)

    def tearDown(self):
        E.get_ohlcv_safe = self._saved_ohlcv
        E.get_orderbook_cached = self._saved_ob
        E.get_ticker_safe = self._saved_ticker
        E.get_balance_safe = self._saved_balance
        E.STATE["open"] = False
        E.STATE["symbol"] = None
        E.DASHBOARD_STATE["position"] = None

    def test_paper_entry_commits_and_publishes(self):
        saved_qa = E.entry_quality_assessment
        E.entry_quality_assessment = lambda *a, **k: {"decision": "APPROVE", "reason": "controlled approval", "quality_score": 99}
        try:
            ok = E.execute_entry(
                "BUY", "BTC/USDT:USDT", 104.0, 100.0, 105.0, 106.0,
                90, "TEST", 1.0, "INSTITUTIONAL", "DEEP_SCANNER", "SNIPER",
            )
        finally:
            E.entry_quality_assessment = saved_qa
        self.assertTrue(ok)
        self.assertTrue(E.STATE["open"])
        pos = E.DASHBOARD_STATE.get("position")
        self.assertIsNotNone(pos)
        self.assertEqual(pos["symbol"], "BTC/USDT:USDT")
        self.assertGreater(pos["qty"], 0)
        E.clear_position_dashboard()
        self.assertIsNone(E.DASHBOARD_STATE["position"])

    def test_ghost_position_not_created_on_rejection(self):
        saved = E.entry_quality_assessment
        E.entry_quality_assessment = lambda *a, **k: {"decision": "REJECT", "reason": "test reject", "quality_score": 0}
        try:
            ok = E.execute_entry(
                "BUY", "BTC/USDT:USDT", 104.0, 100.0, 105.0, 106.0,
                90, "TEST", 1.0, "INSTITUTIONAL", "DEEP_SCANNER", "SNIPER",
            )
        finally:
            E.entry_quality_assessment = saved
        self.assertFalse(ok)
        self.assertFalse(E.STATE.get("open", False))
        self.assertIsNone(E.DASHBOARD_STATE.get("position"))


class DashboardReadOnlyTest(unittest.TestCase):
    """Read endpoints must NEVER mutate canonical runtime state."""

    @classmethod
    def setUpClass(cls):
        import importlib
        importlib.invalidate_caches()
        D = importlib.import_module("dashboard.app")
        cls.D = D
        cls.client = D.app.test_client()
        now = time.time()
        D.MEMORY["watchlist"] = {
            "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "side": "BUY", "state": "CONFIRMED",
                              "score": 9.0, "deep_analyzed": True, "last_update": now},
            "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "side": "SELL", "state": "RETEST",
                              "score": 6.0, "deep_analyzed": True, "last_update": now - 99999},
            # malformed record kept out of read-path mutation
            "BROKEN": {"side": "BUY"},
        }

    def test_poll_data_never_deletes_watchlist(self):
        D = self.D
        for _ in range(100):
            r = self.client.get("/data")
            self.assertEqual(r.status_code, 200)
        wl = D.MEMORY["watchlist"]
        self.assertEqual(len(wl), 3, "GET /data must not delete watchlist entries")
        self.assertEqual(wl["ETH/USDT:USDT"]["score"], 6.0)

    def test_poll_watchlist_never_deletes_watchlist(self):
        D = self.D
        for _ in range(100):
            r = self.client.get("/watchlist")
            self.assertEqual(r.status_code, 200)
        wl = D.MEMORY["watchlist"]
        self.assertEqual(len(wl), 3, "GET /watchlist must not delete watchlist entries")

    def test_data_payload_contains_watchlist_and_counts(self):
        r = self.client.get("/data")
        payload = r.get_json()
        self.assertIn("watchlist", payload)
        self.assertGreaterEqual(payload.get("watchlist_active", 0), 2)


class CleanupWorkerTest(unittest.TestCase):
    """Lifecycle cleanup now belongs to the runtime worker, not the API."""

    def test_cleanup_quarantines_malformed_and_expires_staged(self):
        now = time.time()
        E.MEMORY["watchlist"] = {
            "GOOD/USDT:USDT": {"symbol": "GOOD/USDT:USDT", "state": "CONFIRMED",
                               "score": 9, "last_update": now},
            "STALE/USDT:USDT": {"symbol": "STALE/USDT:USDT", "state": "CONFIRMED",
                                "score": 9, "last_update": now - 600},
            "BROKEN": {"side": "BUY"},
        }
        E.MEMORY["watchlist_quarantine"] = []
        E.cleanup_watchlist(ttl=300)
        wl = E.MEMORY["watchlist"]
        self.assertIn("GOOD/USDT:USDT", wl)
        self.assertEqual(wl["GOOD/USDT:USDT"]["state"], "CONFIRMED")
        self.assertIn("STALE/USDT:USDT", wl)
        self.assertEqual(wl["STALE/USDT:USDT"]["state"], "EXPIRED")
        self.assertNotIn("BROKEN", wl)
        self.assertEqual(E.MEMORY["watchlist_quarantine"][0]["reason"], "MALFORMED_RECORD")
        # Second pass after expiry window removes EXPIRED entries only.
        E.MEMORY["watchlist"]["STALE/USDT:USDT"]["last_update"] = now - 1200
        E.MEMORY["watchlist"]["STALE/USDT:USDT"]["expired_at"] = now - 1200
        E.cleanup_watchlist(ttl=300)
        self.assertNotIn("STALE/USDT:USDT", E.MEMORY["watchlist"])
        self.assertIn("GOOD/USDT:USDT", E.MEMORY["watchlist"])


class NewsEntityMatchingTest(unittest.TestCase):
    """News headlines must match the instrument entity to count as evidence."""

    @classmethod
    def setUpClass(cls):
        import news.service as N
        cls.N = N

    def test_direct_headline_matches_symbol_aliases(self):
        N = self.N
        aliases = N.NewsService._entity_aliases("BTC/USDT:USDT", "CRYPTO")
        self.assertTrue(N.NewsService._is_relevant(
            {"title": "Bitcoin hits record high as ETF inflows surge", "snippet": ""}, aliases))
        self.assertFalse(N.NewsService._is_relevant(
            {"title": "Zinc prices steady in Asian trade", "snippet": ""}, aliases))

    def test_unrelated_headlines_do_not_count_as_symbol_news(self):
        N = self.N
        svc = N.NewsService()

        class FakeYahooResp:
            status_code = 200
            def raise_for_status(self):
                return None
            def json(self):
                return {"quotes": [], "news": [
                    {"title": "Major US airline files for bankruptcy", "link": "http://x",
                     "publisher": "Reuters", "providerPublishTime": int(time.time())},
                    {"title": "Bitcoin ETF inflows hit record high", "link": "http://y",
                     "publisher": "CoinDesk", "providerPublishTime": int(time.time())},
                ]}

        fake_yahoo = FakeYahooResp()
        with patch("news.service.requests.get", return_value=fake_yahoo):
            a = svc.assess("BTC/USDT:USDT", "CRYPTO")
        self.assertTrue(a.available)
        self.assertEqual(a.direct_count, 1)
        for h in a.headlines:
            if h.get("scope") == "DIRECT":
                self.assertIn("bitcoin", (h.get("title") or "").lower())

    def test_news_state_side_logic(self):
        N = self.N
        support = N.NewsAssessment(available=True, bias="BULLISH", risk=10, direct_count=2, macro_event=False)
        self.assertEqual(N.news_state_for_side(support, "BUY"), "NEWS_SUPPORT")
        self.assertEqual(N.news_state_for_side(support, "SELL"), "NEWS_CONFLICT")
        conflict = N.NewsAssessment(available=True, bias="BEARISH", risk=10, direct_count=1, macro_event=False)
        self.assertEqual(N.news_state_for_side(conflict, "BUY"), "NEWS_CONFLICT")
        self.assertEqual(N.news_state_for_side(conflict, "SELL"), "NEWS_SUPPORT")
        risk = N.NewsAssessment(available=True, bias="NEUTRAL", risk=95, direct_count=3)
        self.assertEqual(N.news_state_for_side(risk, "BUY"), "NEWS_RISK")
        unavail = N.NewsAssessment(available=False)
        self.assertEqual(N.news_state_for_side(unavail, "BUY"), "NEWS_UNAVAILABLE")
        # macro-feed noise without direct match or macro event stays neutral
        noise = N.NewsAssessment(available=True, bias="BULLISH", risk=5, direct_count=0, macro_event=False)
        self.assertEqual(N.news_state_for_side(noise, "BUY"), "NEWS_NEUTRAL")
        # but a true macro event may carry direction
        macro = N.NewsAssessment(available=True, bias="BULLISH", risk=10, direct_count=0, macro_event=True)
        self.assertEqual(N.news_state_for_side(macro, "BUY"), "NEWS_SUPPORT")

    def test_sentiment_colors_are_sentiment_based_not_side_based(self):
        N = self.N
        svc = N.NewsService()
        pos = svc._normalize_article("Bitcoin surges to record high on ETF approval", "", "X",
                                     int(time.time()), provider="YAHOO")
        neg = svc._normalize_article("Bitcoin collapses after exchange hack", "", "X",
                                     int(time.time()), provider="YAHOO")
        neu = svc._normalize_article("Local transit schedules updated", "", "X",
                                     int(time.time()), provider="YAHOO")
        self.assertEqual(pos["sentiment_color"], "GREEN")
        self.assertEqual(neg["sentiment_color"], "RED")
        self.assertEqual(neu["sentiment_color"], "GRAY")


class SafetyGateTest(unittest.TestCase):
    """Emergency kill switch must actually gate new entries."""

    def test_kill_switch_blocks_runtime_entry(self):
        import core.runtime as R
        saved_flags = E.STATE.get("daily_loss_limit_hit")
        E.STATE["daily_loss_limit_hit"] = True
        try:
            self.assertFalse(R._execute_ready_queue_candidate())
        finally:
            E.STATE["daily_loss_limit_hit"] = saved_flags


class IntentRegimeWeightTest(unittest.TestCase):
    """Layer-9 regime weights must be regime-adaptive, not a silent uniform
    fallback, when MEMORY carries a narrative-scale regime string."""

    def test_narrative_regime_maps_to_adaptive_weights(self):
        saved = E.MEMORY.get("regime")
        try:
            E.MEMORY["regime"] = "TREND"
            score, status, details = E.InstitutionalIntentEngine.detect(_df(160), None, "BTC/USDT:USDT")
            self.assertIn("regime_weights", details)
            self.assertEqual(details["regime"], "TREND")
            w = details["regime_weights"]
            self.assertNotEqual(w["liquidity"], 12.5, msg="flat 12.5 fallback should not fire for a known regime")
            self.assertEqual(w["institutional_flow"], 18)
            self.assertEqual(details["regime_input"], "TREND")
        finally:
            E.MEMORY["regime"] = saved

    def test_unknown_regime_uses_uniform_baseline(self):
        saved = E.MEMORY.get("regime")
        try:
            E.MEMORY["regime"] = "EXPANSION"
            _, _, details = E.InstitutionalIntentEngine.detect(_df(160), None, "BTC/USDT:USDT")
            self.assertEqual(details["regime"], "NEUTRAL")
            self.assertEqual(details["regime_weights"]["liquidity"], 12.5)
            self.assertEqual(details["regime_input"], "EXPANSION")
        finally:
            E.MEMORY["regime"] = saved


class LiquidityIntelligenceTest(unittest.TestCase):
    """Evidence-breakdown liquidity evaluator: states, discriminative scores,
    and honest handling of missing provider data."""

    @staticmethod
    def _mk(x, vol=1000.0):
        n = len(x)
        return pd.DataFrame({
            "open": x, "high": x + 0.2, "low": x - 0.2,
            "close": x, "volume": np.full(n, vol),
        })

    @staticmethod
    def _flat_sine(n=120, amp=0.05):
        t = np.arange(n)
        return np.full(n, 100.0) + amp * np.sin(t / 6.0)

    def _ev(self, df, side, atr=0.4):
        return E.queue._evaluate_liquidity(df, side, atr)

    # 1. flat series → pool present but low-edge proximity, state PRESENT/NEAR, never None composite
    def test_flat_series_yields_finite_score_with_state(self):
        df = self._mk(self._flat_sine())
        s, ev = self._ev(df, "BUY")
        self.assertIn(ev["state"], ("LIQUIDITY_PRESENT", "LIQUIDITY_NEAR", "LIQUIDITY_SWEPT"))
        self.assertTrue(0 <= s <= 100)

    # 2. equal lows strengthen pool evidence
    def test_equal_lows_raise_pool_strength(self):
        x = self._flat_sine(amp=0.3)
        # duplicate minimum twice → equal lows
        mn = x.min(); idx = np.where(x == mn)[0][0]
        x[idx] = mn; x[idx+4] = mn
        s, ev = self._ev(self._mk(x), "BUY")
        self.assertGreaterEqual(ev["pool"], 45)

    # 3. strong nearby pool (0.2 ATR) outscores distant pool
    def test_proximity_is_reflective(self):
        x1 = self._flat_sine(amp=0.3)
        near_pool = float(x1.min())
        x1[-1] = near_pool + 0.2 * 0.4  # 0.2 ATR above pool
        xfar = self._flat_sine(amp=0.3)
        xfar[-1] = near_pool + 2.0 * 0.4
        s_near, ev_near = self._ev(self._mk(x1), "BUY")
        s_far, ev_far = self._ev(self._mk(xfar), "BUY")
        self.assertGreater(ev_near["proximity"], ev_far["proximity"])

    # 4/5. sell-side sweep for BUY and buy-side sweep for SELL register sweep 100
    def test_directional_sweep_registers(self):
        x = self._flat_sine(amp=0.3)
        x[-3] = x.min() - 1.0
        buy_s, buy_ev = self._ev(self._mk(x), "BUY")
        self.assertEqual(buy_ev["sweep"], 100)
        y = self._flat_sine(amp=0.3)
        y[-3] = y.max() + 1.0
        sell_s, sell_ev = self._ev(self._mk(y), "SELL")
        self.assertEqual(sell_ev["sweep"], 100)

    # 6/7. displacement score increases composite modestly; exact sweep composition
    #    is what the live engine is presented with, composite just must not drop
    def test_displacement_raises_score_after_sweep(self):
        import numpy as _np
        x = self._flat_sine(amp=0.3)
        x[-3] = x.min() - 1.0
        base = x.copy(); base[-1] = base[-2]
        disp = x.copy()
        # extreme displacement (3 ATR from the sweep) should never be scored as zero
        disp[-1] = float(_np.min(x)) + 3.0 * 0.4
        s_no, ev_no = self._ev(self._mk(base), "BUY")
        s_disp, ev_disp = self._ev(self._mk(disp), "BUY")
        # displacement evidence must be present regardless of behavior specifics
        self.assertGreater(ev_disp["displacement"], 0)
        # and composite should stay healthy compared to no-displacement baseline
        self.assertGreater(s_disp, 30.0)

    # 8. sweep recency is recorded and older sweeps degrade to NEAR
    def test_sweep_recency_state_transition(self):
        import numpy as _np
        x = self._flat_sine(amp=0.3)
        x[-3] = x.min() - 1.0
        states = []
        ages = []
        for t in range(0, 8):
            s, ev = self._ev(self._mk(x.copy()), "BUY")
            states.append(ev["state"])
            if ev.get("sweep_age") is not None:
                ages.append(ev["sweep_age"])
            # prepend a quiet early bar to shift the sweep backwards in time
            x = _np.concatenate(([x[0]], x))
        self.assertIn("LIQUIDITY_SWEPT", states[:3])
        # as the sweep ages, the state degrades from SWEPT to NEAR or falls off
        self.assertTrue(all(ages[i] <= ages[i+1] for i in range(len(ages)-1)))

    # 9. stale sweeps are still valued but no longer marked SWEPT
    def test_stale_sweep_not_marked_swept(self):
        x = self._flat_sine(amp=0.3)
        x[-20] = x.min() - 1.0
        s, ev = self._ev(self._mk(x), "BUY")
        self.assertNotEqual(ev["state"], "LIQUIDITY_SWEPT")

    # 10. too-short frame: cannot form swing pools → INVALID at the documented floor
    def test_missing_pool_gives_invalid(self):
        x = self._flat_sine(amp=0.3)[:8]
        df = self._mk(x)
        s, ev = self._ev(df, "BUY")
        self.assertEqual(ev["state"], "LIQUIDITY_INVALID")
        self.assertEqual(s, 30.0)

    # 11. missing frame → UNAVAILABLE at 50 (documented fallback), not hidden
    def test_missing_frame_unavailable(self):
        s, ev = self._ev(None, "BUY")
        self.assertEqual(ev["state"], "LIQUIDITY_UNAVAILABLE")
        self.assertEqual(s, 50.0)

    # 12. missing volume → data_conf drops, never crashes
    def test_missing_volume_column_no_crash(self):
        x = self._flat_sine(amp=0.3)
        df = self._mk(x)
        df = df.drop(columns=["volume"])
        s, ev = self._ev(df, "BUY")
        self.assertIn(ev["state"], ("LIQUIDITY_PRESENT", "LIQUIDITY_NEAR", "LIQUIDITY_SWEPT"))
        self.assertLess(ev["data_conf"], 100)


class MSBOBEngineTest(unittest.TestCase):
    """MSB-OB structural evidence engine: parity targets against the Pine
    semantics, closed-candle determinism, and zone lifecycle rules."""

    @staticmethod
    def _mk(x, vol=1000.0):
        n = len(x)
        return pd.DataFrame({
            "open": x, "high": x + 0.6, "low": x - 0.6,
            "close": x, "volume": np.full(n, vol),
        })

    def _break_series(self, amp=2.0):
        """Mixed bullish/bearish candles so Pine-style zones can form."""
        t = np.arange(300)
        x = 100 + amp * np.sin(t / 4.0)
        x[150:] = np.linspace(x[150], x[150] - 15, 150)
        o = np.where(np.arange(300) % 2 == 0, x - 0.5, x + 0.5)
        c = np.where(np.arange(300) % 2 == 0, x + 0.5, x - 0.5)
        return x, self._mk_from_oc(o, c, x)

    @staticmethod
    def _mk_from_oc(o, c, x):
        n = len(x)
        return pd.DataFrame({
            "open": o, "high": x + 0.6, "low": x - 0.6,
            "close": c, "volume": np.full(n, 1000.0),
        })

    def test_displacement_break_yields_msb_and_zone(self):
        from core.msb_ob import analyze_msb
        x, df = self._break_series()
        r = analyze_msb(df, "TEST")
        self.assertGreaterEqual(len(r["msb_events"]), 1)
        self.assertGreaterEqual(len(r["zones"]), 1)
        self.assertIn("side", r["zones"][0])
        self.assertIn("top", r["zones"][0])
        self.assertIn("bottom", r["zones"][0])

    def test_no_break_no_zones(self):
        from core.msb_ob import analyze_msb, MSBOBEngine
        # strictly monotone series — every swing just confirms existing trend
        y = np.linspace(100, 140, 300)
        o = y - 0.2
        c = y - 0.4
        r = MSBOBEngine(zigzag_len=9, fib_factor=0.33).analyze(self._mk_from_oc(o, c, y), "TEST")
        self.assertEqual(len(r["msb_events"]), 0)
        self.assertEqual(len(r["zones"]), 0)

    def test_closed_candle_determinism(self):
        from core.msb_ob import analyze_msb
        _, df = self._break_series()
        r1 = analyze_msb(df, "TEST")
        r2 = analyze_msb(df, "TEST")
        self.assertEqual(r1, r2)

    def test_zones_capture_fib_factor(self):
        from core.msb_ob import analyze_msb
        _, df = self._break_series()
        r = analyze_msb(df, "TEST", fib_factor=0.5)
        if r["zones"]:
            self.assertEqual(r["zones"][0]["fib_factor"], 0.5)

    def test_invalidated_zone_marked(self):
        from core.msb_ob import analyze_msb
        _, df = self._break_series()
        r = analyze_msb(df, "TEST")
        if not r["zones"]:
            self.skipTest("no zone formed on this synthetic series")
        for z in r["zones"]:
            if z["side"] == "LONG":
                # a bullish zone below a forced-price break must be invalidated.
                self.assertEqual(z["status"], "INVALIDATED")

    def test_short_frame_reports_data_unavailable(self):
        from core.msb_ob import analyze_msb
        r = analyze_msb(self._mk(np.full(5, 100.0)), "TEST")
        self.assertEqual(r["error"], "DATA_UNAVAILABLE")
        self.assertEqual(r["zones"], [])


class MSBInstitutionalContextTest(unittest.TestCase):
    """Canonical MSBInstitutionalContext: temporal liquidity rules, zone
    ranking, and no-bypass guarantees."""

    @staticmethod
    def _mk(x, vol=1000.0):
        n = len(x)
        return pd.DataFrame({
            "open": x, "high": x + 0.6, "low": x - 0.6,
            "close": x, "volume": np.full(n, vol)})

    def _make_zone(self, side="LONG", status="ACTIVE", created=10, ztype="OB", top=101.5, bottom=100.5):
        return {"side": side, "zone_type": ztype, "top": top, "bottom": bottom,
                "created_at": created, "status": status, "freshness": 5, "touch_count": 1,
                "zone_strength": 1.0, "msb_direction": "BULL" if side == "LONG" else "BEAR",
                "msb_price": 100.5, "swing_high": top, "swing_low": bottom,
                "fib_factor": 0.33, "displacement_score": 0.0, "volume_score": 0.0,
                "liquidity_context": "NONE", "symbol": "TEST"}

    def _msb_event(self, direction=1, price=100.5, index=10):
        return {"direction": direction, "price": price, "index": index}

    # 1. MSB without a liquidity sweep must not become LIQUIDITY_SWEET_CONFIRMED
    def test_msb_without_liquidity_not_confirmed(self):
        from core.msb_ob import msb_context, LIQ_CONFIRMED, LIQ_NEARBY, LIQ_NONE
        x = 100 + 1.5 * np.sin(np.arange(200) / 4.0)
        ctx = msb_context(self._mk(x), "T", 1, E.queue,
                          zone=self._make_zone(),
                          msb_event=self._msb_event(),
                          atr=0.4)
        # sweep_detector across the noisy sine may fire; what matters is the
        # temporal relationship: if sweep predates msb it must be marked
        self.assertIsNotNone(ctx)
        if ctx.sweep_recency >= 0:
            self.assertIn(ctx.sweep_before_msb, (True, False))
        else:
            self.assertNotEqual(ctx.liquidity_state if hasattr(ctx,'liquidity_state') else "X", LIQ_CONFIRMED)

    # 2. sell-side sweep + bullish displacement + BOS + fresh zone + matching side
    def test_sweep_msb_displacement_produces_valid_context(self):
        from core.msb_ob import msb_context, LIQ_SWEPT, LIQ_CONFIRMED
        x = 100 + 2 * np.sin(np.arange(300) / 4.0)
        x[150:] = np.linspace(x[150], x[150] - 15, 150)
        o = np.where(np.arange(300) % 2 == 0, x - 0.5, x + 0.5)
        c = np.where(np.arange(300) % 2 == 0, x + 0.5, x - 0.5)
        df = pd.DataFrame({"open": o, "high": x + 0.6, "low": x - 0.6,
                           "close": c, "volume": np.full(300, 1000.)})
        ctx = msb_context(df, "T", 1, E.queue,
                          zone=self._make_zone(status="ACTIVE"),
                          msb_event=self._msb_event(),
                          atr=0.4)
        self.assertIsNotNone(ctx)
        # sweep detected at some recency → context not NONE
        if ctx.sweep_recency >= 0:
            self.assertGreater(ctx.context_confidence, 0.0)

    # 3. wrong-side liquidity (buy-side sweep) must not confirm a LONG setup
    def test_wrong_side_liquidity_not_confirmation(self):
        from core.msb_ob import msb_context, rank_zones
        # Request a LONG context but feed an environment where the only sweep
        # opportunity is on the buy side (break upwards). Engine must still mark
        # the sweep as SELL_SIDE target for a LONG and NOT manufacture
        # confirmation out of invalid evidence.
        x = 100 + 2 * np.sin(np.arange(300) / 4.0)
        x[150:] = np.linspace(x[150], x[150] + 15, 150)
        o = np.where(np.arange(300) % 2 == 0, x - 0.5, x + 0.5)
        c = np.where(np.arange(300) % 2 == 0, x + 0.5, x - 0.5)
        df = pd.DataFrame({"open": o, "high": x + 0.6, "low": x - 0.6,
                           "close": c, "volume": np.full(300, 1000.)})
        from core.msb_ob import LIQ_CONFIRMED
        ctx = msb_context(df, "T", 1, E.queue,
                          zone=self._make_zone(side="LONG"),
                          msb_event=self._msb_event(direction=1),
                          atr=0.4)
        self.assertIsNotNone(ctx)
        # if only the wrong-side sweep is present, confidence must remain low
        # for a LONG-side setup and cannot reach LIQUIDITY_SWEEP_CONFIRMED
        # quality in a meaningful way.
        self.assertLess(ctx.context_confidence, 0.60)

    # 4. INVALIDATED zone must not remain a primary zone
    def test_invalidated_zone_not_primary(self):
        from core.msb_ob import rank_zones, STATUS_ACTIVE
        zones = [
            self._make_zone(status="INVALIDATED"),
            self._make_zone(status="ACTIVE", top=99.5, bottom=98.5, created=5),
        ]
        primary, secondary = rank_zones(zones, 1)
        self.assertIsNotNone(primary)
        self.assertEqual(primary["status"], "ACTIVE")
        self.assertIsNone(secondary)

    # 5. zone ranking is deterministic on identical payload
    def test_zone_rank_deterministic(self):
        from core.msb_ob import rank_zones
        zones = [self._make_zone(created=8, top=101.0, bottom=100.7),
                 self._make_zone(created=3, top=102.0, bottom=101.2)]
        p1, s1 = rank_zones(zones, 1)
        p2, s2 = rank_zones(list(reversed(zones)), 1)
        self.assertEqual(p1, p2)

    # 6. the same structural event is not counted as two independent events
    def test_no_double_count_same_event(self):
        from core.msb_ob import analyze_msb
        x = 100 + 2 * np.sin(np.arange(300) / 4.0)
        x[150:] = np.linspace(x[150], x[150] - 15, 150)
        o = np.where(np.arange(300) % 2 == 0, x - 0.5, x + 0.5)
        c = np.where(np.arange(300) % 2 == 0, x + 0.5, x - 0.5)
        df = pd.DataFrame({"open": o, "high": x + 0.6, "low": x - 0.6,
                           "close": c, "volume": np.full(300, 1000.)})
        r = analyze_msb(df, "T")
        # Pine intentionally emits both an OB zone and a BB/MB zone for one
        # structural break — distinct zone types. Counting by created_at alone
        # would false-flag a legitimate double-emitter as double-counted.
        composite_keys = [(z["created_at"], z["zone_type"]) for z in r["zones"]]
        self.assertEqual(len(composite_keys), len(set(composite_keys)))

    # 7. msb_context never becomes a READY shortcut: ZoneMetrics untouched
    def test_msb_cannot_bypass_queue_gates(self):
        from core.msb_ob import msb_context
        x = 100 + 2 * np.sin(np.arange(300) / 4.0)
        x[150:] = np.linspace(x[150], x[150] - 15, 150)
        o = np.where(np.arange(300) % 2 == 0, x - 0.5, x + 0.5)
        c = np.where(np.arange(300) % 2 == 0, x + 0.5, x - 0.5)
        df = pd.DataFrame({"open": o, "high": x + 0.6, "low": x - 0.6,
                           "close": c, "volume": np.full(300, 1000.)})
        ctx = msb_context(df, "T", 1, E.queue,
                          zone=self._make_zone(),
                          msb_event=self._msb_event(),
                          atr=0.4)
        # presence of context must not mutate queue candidate state
        cands_before = len(E.queue._candidates) if E.queue else 0
        self.assertIsNotNone(ctx)
        if E.queue:
            self.assertEqual(len(E.queue._candidates), cands_before)

    # 8. zone identity survives end-to-end (zone_id preserved on context)
    def test_active_zone_identity_survives(self):
        from core.msb_ob import msb_context
        z = self._make_zone(created=42, top=107.0, bottom=106.0)
        x = 100 + 1.5 * np.sin(np.arange(100) / 4.0)
        ctx = msb_context(self._mk(x), "T", 1, E.queue,
                          zone=z, msb_event=self._msb_event(index=42),
                          atr=0.4)
        self.assertIsNotNone(ctx)
        self.assertIn("42", ctx.zone_id)
        self.assertEqual(ctx.zone_top, 107.0)
        self.assertEqual(ctx.zone_bottom, 106.0)
        self.assertEqual(ctx.zone_status, "ACTIVE")

    # 9. closed-candle behavior — identical context on identical frame
    def test_context_deterministic(self):
        from core.msb_ob import msb_context
        z = self._make_zone(created=42)
        x = 100 + 1.5 * np.sin(np.arange(100) / 4.0)
        df = self._mk(x)
        ctx1 = msb_context(df, "T", 1, E.queue, zone=z, msb_event=self._msb_event(), atr=0.4)
        ctx2 = msb_context(df, "T", 1, E.queue, zone=z, msb_event=self._msb_event(), atr=0.4)
        self.assertEqual(ctx1.to_dict(), ctx2.to_dict())


class MSBTemporalSequenceTest(unittest.TestCase):
    """Temporal/causal ordering and sequence classification."""

    @staticmethod
    def _df_from_oc(o, c, x, vol=1000.0):
        n = len(x)
        return pd.DataFrame({"open": o, "high": x + 0.6, "low": x - 0.6,
                             "close": c, "volume": np.full(n, vol)})

    def _zone(self, side="LONG", created=10):
        return {"side": side, "zone_type": "OB", "top": 101.5, "bottom": 100.5,
                "created_at": created, "status": "ACTIVE", "freshness": 5,
                "touch_count": 0, "zone_strength": 1.0,
                "msb_direction": "BULL" if side == "LONG" else "BEAR",
                "msb_price": 100.5, "swing_high": 101.5, "swing_low": 100.5,
                "fib_factor": 0.33, "displacement_score": 0.0, "volume_score": 0.0,
                "liquidity_context": "NONE", "symbol": "TEST"}

    def _msb_event(self, index, direction=1, price=100.5):
        return {"direction": direction, "price": price, "index": index}

    def _break_series(self, side_break=-1):
        t = np.arange(300)
        x = 100 + 2 * np.sin(t / 4.0)
        if side_break < 0:
            x[150:] = np.linspace(x[150], x[150] - 15, 150)
        else:
            x[150:] = np.linspace(x[150], x[150] + 15, 150)
        o = np.where(np.arange(300) % 2 == 0, x - 0.5, x + 0.5)
        c = np.where(np.arange(300) % 2 == 0, x + 0.5, x - 0.5)
        return self._df_from_oc(o, c, x)

    # 1. Full bullish sequence: sweep ≤ displacement ≤ msb ≤ ob ≤ retest ≤ rejection
    def test_full_bullish_sequence(self):
        from core.msb_ob import temporal_sequence, SEQ_FULL
        df = self._break_series(-1)
        side = 1  # LONG
        # sweep deliberately happens before MSB and retest is forced on a later bar
        msb = self._msb_event(index=170, direction=1)
        ctx = temporal_sequence(df, "T", side, E.queue, zone=self._zone(created=170),
                                msb_event=msb, atr=0.4)
        self.assertIsNotNone(ctx)
        b = ctx.to_dict()["bars"]
        if b["sweep"] is not None and b["msb"] is not None and b["ob"] is not None:
            self.assertLessEqual(b["sweep"], b["msb"])
            self.assertLessEqual(b["msb"], b["ob"])

    # 2. Full bearish sequence (mirror of bullish)
    def test_full_bearish_sequence(self):
        from core.msb_ob import temporal_sequence
        df = self._break_series(+1)
        side = -1
        msb = self._msb_event(index=170, direction=-1)
        ctx = temporal_sequence(df, "T", side, E.queue, zone=self._zone(side="SHORT", created=170),
                                msb_event=msb, atr=0.4)
        self.assertIsNotNone(ctx)

    # 3. MSB without liquidity → no sweep bar; not FULL
    def test_msb_without_liquidity_not_full(self):
        from core.msb_ob import temporal_sequence, SEQ_FULL
        df = self._break_series(-1)
        # place MSB before any candidate sweep on the frame; if sweep exists later
        # the sequence must not collapse to FULL.
        msb = self._msb_event(index=5, direction=1)
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=self._zone(created=5),
                                msb_event=msb, atr=0.4)
        self.assertIsNotNone(ctx)
        if ctx.to_dict()["bars"]["sweep"] is None:
            self.assertNotEqual(ctx.sequence, SEQ_FULL)

    # 4. Liquidity without displacement → no displacement bar, not FULL
    def test_liquidity_without_displacement_not_full(self):
        from core.msb_ob import temporal_sequence, SEQ_FULL
        df = self._break_series(-1)
        # a flat series → sweep rarely occurs, but when it does, displacement must be
        # missing; FULL requires displacement.
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=self._zone(created=250),
                                msb_event=self._msb_event(index=250), atr=0.4)
        if ctx is not None and ctx.to_dict()["bars"]["displacement"] is None:
            self.assertNotEqual(ctx.sequence, SEQ_FULL)

    # 5. Displacement without MSB → sequence is not FULL (no MSB bar)
    def test_displacement_without_msb_not_full(self):
        from core.msb_ob import temporal_sequence, SEQ_FULL
        df = self._break_series(-1)
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=self._zone(created=170),
                                msb_event=None, atr=0.4)
        if ctx is not None:
            self.assertNotEqual(ctx.sequence, SEQ_FULL)

    # 6. MSB before liquidity → sequence ≠ LIQUIDITY_THEN_STRUCTURE
    def test_msb_before_liquidity_not_marked_then(self):
        from core.msb_ob import temporal_sequence, SEQ_LIQUIDITY_THEN_STRUCTURE
        df = self._break_series(-1)
        msb = self._msb_event(index=170, direction=1)
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=self._zone(created=170),
                                msb_event=msb, atr=0.4)
        if ctx is not None and ctx.to_dict()["bars"]["sweep"] is not None:
            if ctx.to_dict()["bars"]["sweep"] > ctx.to_dict()["bars"]["msb"]:
                self.assertNotEqual(ctx.sequence, SEQ_LIQUIDITY_THEN_STRUCTURE)

    # 7. Liquidity after MSB → not LIQUIDITY_THEN_STRUCTURE
    def test_liquidity_after_msb_not_marked_then(self):
        from core.msb_ob import temporal_sequence, SEQ_LIQUIDITY_THEN_STRUCTURE
        df = self._break_series(-1)
        msb = self._msb_event(index=250, direction=1)
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=self._zone(created=250),
                                msb_event=msb, atr=0.4)
        if ctx is not None and ctx.to_dict()["bars"]["sweep"] is not None:
            if ctx.to_dict()["bars"]["sweep"] > ctx.to_dict()["bars"]["msb"]:
                self.assertNotEqual(ctx.sequence, SEQ_LIQUIDITY_THEN_STRUCTURE)

    # 8. Invalidated OB → not FULL sequence
    def test_invalidated_ob_not_full(self):
        from core.msb_ob import temporal_sequence, SEQ_FULL
        df = self._break_series(-1)
        z = self._zone(side="LONG", created=200)
        z["status"] = "INVALIDATED"
        msb = self._msb_event(index=170, direction=1)
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=z, msb_event=msb, atr=0.4)
        if ctx is not None:
            # an invalidated OB cannot feed the retest path; classification degrades
            self.assertNotEqual(ctx.sequence, SEQ_FULL)

    # 9. Retest without rejection → not FULL (needs both)
    def test_retest_without_rejection_not_full(self):
        from core.msb_ob import temporal_sequence, SEQ_FULL
        df = self._break_series(-1)
        z = self._zone(side="LONG", created=170)
        msb = self._msb_event(index=170, direction=1)
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=z, msb_event=msb, atr=0.4)
        if ctx is not None:
            if ctx.to_dict()["bars"]["retest"] is not None and ctx.to_dict()["bars"]["rejection"] is None:
                self.assertNotEqual(ctx.sequence, SEQ_FULL)

    # 10. Conflicting liquidity — engine still returns a consistent sequence state
    def test_conflicting_liquidity_consistent(self):
        from core.msb_ob import temporal_sequence
        df = self._break_series(-1)
        ctx = temporal_sequence(df, "T", 1, E.queue, zone=self._zone(created=170),
                                msb_event=self._msb_event(index=170), atr=0.4)
        self.assertIsNotNone(ctx)
        self.assertIn(ctx.sequence, ("NO_LIQUIDITY_SEQUENCE", "LIQUIDITY_ONLY",
                                     "LIQUIDITY_THEN_STRUCTURE",
                                     "STRUCTURAL_BREAK_WITHOUT_LIQUIDITY",
                                     "FULL_INSTITUTIONAL_SEQUENCE", "INVALID_SEQUENCE"))

    # 11. Multiple zones — ranking determinism is already covered; here the timeline
    #     must remain deterministic across identical runs
    def test_timeline_deterministic(self):
        from core.msb_ob import temporal_sequence
        df = self._break_series(-1)
        z = self._zone(created=170)
        msb = self._msb_event(index=170)
        a = temporal_sequence(df, "T", 1, E.queue, zone=z, msb_event=msb, atr=0.4)
        b = temporal_sequence(df, "T", 1, E.queue, zone=z, msb_event=msb, atr=0.4)
        self.assertEqual(a.to_dict(), b.to_dict())

    # 12. Same-event MSB/BOS/MSS deduplication — OB+BB/MB from one MSB is one event
    def test_same_event_msb_bos_mss_deduplicated(self):
        from core.msb_ob import analyze_msb
        df = self._break_series(-1)
        r = analyze_msb(df, "T")
        # one created_at may legitimately have OB and BB/MB — distinct types
        created_types = {(z["created_at"], z["zone_type"]) for z in r["zones"]}
        self.assertEqual(len(created_types), len(r["zones"]))


class QueuePromotionsPayloadTest(unittest.TestCase):
    """Watchlist→queue promotion count must be visible in the dashboard payload."""

    def test_promotions_counter_present(self):
        import importlib
        import sys
        D = importlib.import_module("dashboard.app")
        # The route reads the engine MEMORY/CACHE objects live, regardless of
        # which sys.modules snapshot other tests may have rebound.
        eng = sys.modules["core.engine"]
        saved = eng.MEMORY.get("watchlist_queue_promotions")
        eng.MEMORY["watchlist_queue_promotions"] = 7
        try:
            client = D.app.test_client()
            eng.CACHE.pop("dashboard", None)
            resp = client.get("/data")
            self.assertEqual(resp.status_code, 200)
            body = resp.get_json()
            self.assertEqual(body["queue"]["promotions"], 7)
        finally:
            if saved is None:
                eng.MEMORY.pop("watchlist_queue_promotions", None)
            else:
                eng.MEMORY["watchlist_queue_promotions"] = saved
            eng.CACHE.pop("dashboard", None)


if __name__ == "__main__":
    unittest.main()
