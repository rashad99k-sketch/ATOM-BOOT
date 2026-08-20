import os
import time
import unittest

from portfolio.risk import PortfolioRiskGuard


class FakeRiskEngine:
    def __init__(self):
        self.PERF = {"trades": 0, "last_trade": None}
        self.balance = 1000.0

    def get_balance_safe(self):
        return self.balance


class PortfolioRiskGuardTest(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        os.environ["POSITION_MARGIN_PCT"] = "0.10"
        os.environ["PORTFOLIO_MARGIN_CAP_PCT"] = "0.60"
        os.environ["MAX_DAILY_LOSS_PCT"] = "5"
        os.environ["MAX_CONSECUTIVE_LOSSES"] = "3"
        os.environ["COOLDOWN_MINUTES_LOSS"] = "1"
        os.environ["COOLDOWN_MINUTES_DRAWDOWN"] = "2"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_margin_cap_blocks_seventh_position(self):
        e = FakeRiskEngine()
        guard = PortfolioRiskGuard(e)
        self.assertTrue(guard.can_open(5))
        self.assertFalse(guard.can_open(6))

    def test_daily_drawdown_blocks_new_entries(self):
        e = FakeRiskEngine()
        guard = PortfolioRiskGuard(e)
        self.assertTrue(guard.can_open(0))
        e.balance = 940.0
        status = guard.status(0)
        self.assertFalse(status.allowed)
        self.assertEqual(status.reason, "DAILY_DRAWDOWN_LIMIT")

    def test_three_losses_arm_longer_cooldown(self):
        e = FakeRiskEngine()
        guard = PortfolioRiskGuard(e)
        guard.status(0)
        for n in range(1, 4):
            e.PERF["trades"] = n
            e.PERF["last_trade"] = {"result": "LOSS", "pnl_pct": -1.0}
            guard.sync_closed_trades()
        status = guard.status(0)
        self.assertFalse(status.allowed)
        self.assertEqual(status.reason, "LOSS_COOLDOWN")
        self.assertEqual(status.consecutive_losses, 3)
        self.assertGreater(status.cooldown_until, time.time())


if __name__ == "__main__":
    unittest.main()
