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


if __name__ == "__main__":
    unittest.main()
