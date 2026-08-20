import unittest

from portfolio.manager import PortfolioManager, PositionContext


class FakeManager:
    def __init__(self):
        self.STATE = {"open": False, "current_symbol": None}
        self.TRADE_STATE = {"in_position": False, "symbol": None}
        self.paper = {"position": None}
        self._live_manager = "default"
        self.LiveTradeManager = lambda *args: "new-manager"
        self._event_bus = None
        self._exchange_sync = None
        self._recovery_guard = None


class PortfolioIsolationTest(unittest.TestCase):
    def test_symbol_contexts_are_isolated(self):
        e = FakeManager()
        p = PortfolioManager(2, e)
        p.bind(e)
        p.contexts["BTC"] = PositionContext(
            "BTC",
            {"open": True, "current_symbol": "BTC", "side": "BUY"},
            {"in_position": True, "symbol": "BTC"},
            "BTC-MGR",
        )
        p.contexts["ETH"] = PositionContext(
            "ETH",
            {"open": True, "current_symbol": "ETH", "side": "SELL"},
            {"in_position": True, "symbol": "ETH"},
            "ETH-MGR",
        )

        p.activate("BTC")
        self.assertEqual(e.STATE["current_symbol"], "BTC")
        self.assertEqual(e._live_manager, "BTC-MGR")

        p.activate("ETH")
        self.assertEqual(e.STATE["current_symbol"], "ETH")
        self.assertEqual(e._live_manager, "ETH-MGR")

        p.deactivate()
        self.assertFalse(e.STATE["open"])


if __name__ == "__main__":
    unittest.main()

class PortfolioCapacityPolicyTest(unittest.TestCase):
    def test_asset_class_cap_is_enforced(self):
        e = FakeManager()
        p = PortfolioManager(6, e)
        p.bind(e)
        p.contexts["BTC"] = PositionContext("BTC", {}, {}, "mgr")
        p.contexts["ETH"] = PositionContext("ETH", {}, {}, "mgr")
        self.assertFalse(p.can_open("SOL", "CRYPTO"))
        self.assertTrue(p.can_open("AAPL", "STOCK"))
