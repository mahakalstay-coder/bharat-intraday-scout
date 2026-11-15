import json, os, sqlite3, tempfile, unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import adaptive_journal as aj
import market_regime as mr
import trading_calendar as tc

TZ = ZoneInfo("Asia/Kolkata")


class CalendarTests(unittest.TestCase):
    def test_weekend_holiday_and_normal_day(self):
        h = {"2026-12-25"}
        self.assertFalse(tc.is_trading_day(date(2026, 11, 14), h))   # Saturday
        self.assertFalse(tc.is_trading_day(date(2026, 12, 25), h))   # listed holiday
        self.assertTrue(tc.is_trading_day(date(2026, 11, 16), h))    # Monday

    def test_missing_file_never_blocks(self):
        self.assertEqual(tc.load_holidays(Path("/nonexistent/x.json")), set())

    def test_shipped_file_is_valid(self):
        self.assertIn("2026-11-24", tc.load_holidays())


class RegimeTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(mr.classify(0.9, 12)["nifty_trend"], "UP")
        self.assertEqual(mr.classify(-0.9, 20)["vix_bucket"], "HIGH")
        self.assertEqual(mr.classify(0.1, 15)["nifty_trend"], "FLAT")
        self.assertEqual(mr.classify(None, None)["nifty_trend"], "UNKNOWN")

    def test_failure_returns_unknown_and_never_raises(self):
        mr._CACHE.update(at=0.0, day=None, value=None)
        def boom(sym): raise RuntimeError("feed down")
        self.assertEqual(mr.current_regime(datetime(2026, 11, 16, 10, 0, tzinfo=TZ), boom)["nifty_trend"], "UNKNOWN")

    def test_stale_data_is_not_guessed(self):
        mr._CACHE.update(at=0.0, day=None, value=None)
        idx = pd.to_datetime(["2026-11-12", "2026-11-13"])
        frame = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
        out = mr.current_regime(datetime(2026, 11, 16, 10, 0, tzinfo=TZ), lambda s: frame)
        self.assertEqual(out["nifty_trend"], "UNKNOWN")   # last bar is not today


class BootstrapTests(unittest.TestCase):
    def test_lucky_small_sample_is_rejected(self):
        # passes the old gate (mean >= 0.10R, win rate >= 40%) but rests on a few big winners
        vals = [-1] * 7 + [4.0, 3.0, 2.0, 1.0, 0.5]
        self.assertGreaterEqual(sum(vals) / len(vals), 0.10)
        self.assertGreaterEqual(sum(v > 0 for v in vals) / len(vals), 0.40)
        self.assertLess(aj._bootstrap_lcb(vals), 0.0)

    def test_consistent_edge_passes(self):
        vals = [0.8, 1.2, -0.4, 1.0, 0.9, 1.1, -0.3, 1.0, 0.7, 1.3, 0.6, 1.0]
        self.assertGreater(aj._bootstrap_lcb(vals), 0.0)

    def test_deterministic_and_small_n(self):
        self.assertEqual(aj._bootstrap_lcb([1, 2, 3]), aj._bootstrap_lcb([1, 2, 3]))
        self.assertIsNone(aj._bootstrap_lcb([1.0]))


class ReplayWeightsTests(unittest.TestCase):
    def test_replay_weights_are_not_live_by_default(self):
        src = Path("scanner.py").read_text()
        self.assertIn('ALLOW_REPLAY_WEIGHTS", "false"', src)


if __name__ == "__main__":
    unittest.main()


class MeaningfulTargetFloorTests(unittest.TestCase):
    """Learned levels must never shrink the target just to make the win rate look better."""

    def build(self, price=500.0, atr_value=1.0, t1_r=None):
        import scanner as sc
        sc.load_filters = lambda: {"mode": "proven", "total_resolved": 999,
                                   "levels": {"BUY": {"t1_r": t1_r, "stop_scale": 0.7}} if t1_r else {}}
        sig = sc.build_signal_dict(symbol="T", side="BUY", price=price, atr_value=atr_value, score=80, rsi_value=60, mfi_value=60,
                                   adx_val=25, bb_bw=1.2, relvol=2.0, cumulative_relvol=1.5, sessions_compared=5, daily={},
                                   vwap_value=price, breakout_level=price, day_move=1.0, last_idx=pd.Timestamp("2026-11-13 10:00", tz="Asia/Kolkata"),
                                   body_ratio=0.6, setup_label="t", tier="TIER-1", pillars_matched=[])
        return sig, abs(sig["target"] - sig["price"]) / sig["price"] * 100.0

    def test_learned_tiny_target_is_floored(self):
        sig, pct = self.build(t1_r=0.05)          # learner "found" a 0.05R target
        self.assertGreaterEqual(sig["t1_r"], 2.0); self.assertGreaterEqual(pct, 1.5)

    def test_default_target_unchanged_and_meaningful(self):
        sig, pct = self.build()
        self.assertAlmostEqual(sig["t1_r"], 2.5, places=2); self.assertGreaterEqual(pct, 1.9)

    def test_floor_applies_to_tight_stops_too(self):
        sig, pct = self.build(price=2000.0, atr_value=0.1, t1_r=1.0)
        self.assertGreaterEqual(pct, 1.5)

    def test_high_probability_cannot_override_gate(self):
        sector = dict(sector="IT",count=5,change_pct=1,advancing_pct=80)
        stock = dict(sector="IT",peer_count=3,peer_relative_pct=1)
        signal = dict(ml_target_probability=.9,side="BUY",tier="TIER-1",score=92,relative_volume=3,
                      cumulative_relative_volume=2,forecast=dict(model="Kronos-mini",direction_aligned=True))
        for mode in ("RESEARCH_ONLY","VALIDATED_PAPER_MODE"):
            for changes,context in [({"score":50},stock),({},dict(stock,peer_relative_pct=-1)),({},dict(stock,sector="Unknown"))]:
                approved,_ = aj.confirmed_gate(dict(signal,**changes,market_context=context),{"sectors":[sector]},{"status":mode})
                self.assertFalse(approved)
