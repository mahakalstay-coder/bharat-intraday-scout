import json, os, tempfile, unittest
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import adaptive_journal as aj
import feature_labels as fl
import feature_learner as lrn
import features_pro as fp
import news_events as ne


def bars(closes, spread=0.2, vol=1000.0, start="2026-11-12 09:15", tz="Asia/Kolkata", opens=None):
    idx = pd.date_range(start, periods=len(closes), freq="5min", tz=tz)
    c = np.array(closes, float)
    o = np.array(opens, float) if opens is not None else np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                         "close": c, "volume": np.full(len(c), vol)}, index=idx)


def one_session(n=60, base=100.0):
    return bars(base + np.linspace(0, 3, n) + np.sin(np.arange(n)) * 0.2)


class VolumeProfileTests(unittest.TestCase):
    def test_poc_where_most_volume_traded(self):
        df = bars([100] * 30 + [105] * 30, spread=0.05)
        df.loc[df.index[:30], "volume"] = 5000.0   # heavy volume near 100
        vp = fp.volume_profile(df)
        self.assertAlmostEqual(vp["poc"], 100, delta=0.6)
        self.assertEqual(vp["price_vs_value"] in ("ABOVE_VALUE", "INSIDE_VALUE", "BELOW_VALUE"), True)
        self.assertLessEqual(vp["val"], vp["poc"]); self.assertGreaterEqual(vp["vah"], vp["poc"])

    def test_flat_market_returns_empty(self):
        self.assertEqual(fp.volume_profile(bars([100.0] * 50, spread=0.0)), {})


class CandleTests(unittest.TestCase):
    def mk(self, rows):
        base = bars([100.0] * 40)
        t = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.date_range(
            base.index[-1] + pd.Timedelta(minutes=5), periods=len(rows), freq="5min", tz="Asia/Kolkata"))
        t["volume"] = 1000.0
        return pd.concat([base, t])

    def test_bullish_engulfing(self):
        self.assertIn("bullish_engulfing", fp.candlestick_patterns(self.mk([(101, 101.2, 99.8, 100.0), (99.8, 102.2, 99.7, 102.0)])))

    def test_bearish_engulfing(self):
        self.assertIn("bearish_engulfing", fp.candlestick_patterns(self.mk([(100, 101.2, 99.9, 101.0), (101.2, 101.3, 99.0, 99.2)])))

    def test_hammer_and_shooting_star(self):
        self.assertIn("hammer", fp.candlestick_patterns(self.mk([(100, 100.1, 98.0, 100.0 + 0.2)])))
        self.assertIn("shooting_star", fp.candlestick_patterns(self.mk([(100, 102.1, 99.9, 100.0 - 0.1)])))

    def test_doji_and_marubozu(self):
        self.assertIn("doji", fp.candlestick_patterns(self.mk([(100, 101, 99, 100.05)])))
        self.assertIn("marubozu_bull", fp.candlestick_patterns(self.mk([(100, 102.0, 100, 102.0)])))

    def test_morning_star(self):
        rows = [(102, 102.1, 99.9, 100.0), (99.9, 100.1, 99.7, 99.95), (100.0, 102.0, 99.95, 101.9)]
        self.assertIn("morning_star", fp.candlestick_patterns(self.mk(rows)))


class ChartPatternTests(unittest.TestCase):
    def test_double_top_breakdown(self):
        path = list(np.linspace(100, 104, 12)) + list(np.linspace(104, 101, 8)) + list(np.linspace(101, 104, 8)) \
            + list(np.linspace(104, 99.5, 12))
        pats = fp.chart_patterns(bars(path, spread=0.05))
        self.assertIn("double_top_breakdown", pats)

    def test_double_bottom_breakout(self):
        path = list(np.linspace(104, 100, 12)) + list(np.linspace(100, 103, 8)) + list(np.linspace(103, 100, 8)) \
            + list(np.linspace(100, 104.5, 12))
        self.assertIn("double_bottom_breakout", fp.chart_patterns(bars(path, spread=0.05)))

    def test_range_breakout_and_breakdown(self):
        up = fp.chart_patterns(bars([100 + 0.02 * (i % 3) for i in range(40)] + [103.0], spread=0.05))
        dn = fp.chart_patterns(bars([100 + 0.02 * (i % 3) for i in range(40)] + [97.0], spread=0.05))
        self.assertIn("range_breakout_up", up); self.assertIn("range_breakdown", dn)


class IndicatorAndComputeTests(unittest.TestCase):
    def test_extra_indicators_finite(self):
        df = pd.concat([one_session(60, 100), bars(100 + np.linspace(3, 5, 60), start="2026-11-13 09:15")])
        ind = fp.extra_indicators(df)
        self.assertEqual(ind["ema_stack"], "BULL_STACK")
        self.assertEqual(ind["prev_day_break"], "ABOVE_PDH")
        for k in ("stoch_k", "cci20", "obv_slope_norm", "atr14_pct", "price_vs_vwap_pct"):
            self.assertIsNotNone(ind[k], k)

    def test_compute_buy_uptrend_confirms_and_sell_conflicts(self):
        df = pd.concat([one_session(60, 100), bars(100 + np.linspace(3, 6, 60), start="2026-11-13 09:15")])
        buy, sell = fp.compute(df, "BUY"), fp.compute(df, "SELL")
        self.assertGreaterEqual(buy["confluence"], 2)
        self.assertGreater(sell["conflict_count"], buy["conflict_count"])

    def test_too_few_bars_and_garbage_never_raise(self):
        self.assertEqual(fp.compute(bars([100.0] * 10), "BUY"), {})
        self.assertEqual(fp.compute(pd.DataFrame(), "BUY"), {})
        self.assertEqual(fp.compute(None, "BUY"), {})


class NewsEventTests(unittest.TestCase):
    def test_classify(self):
        r = ne.classify(["Company X bags order worth Rs 500 crore", "Q2 net profit jumps 30%"])
        self.assertEqual(r["sentiment"], "POSITIVE"); self.assertIn("results", r["tags"]); self.assertIn("orders", r["tags"])
        n = ne.classify(["SEBI probe into promoter; shares plunge"])
        self.assertEqual(n["sentiment"], "NEGATIVE"); self.assertIn("regulatory", n["tags"])
        self.assertEqual(ne.classify([])["sentiment"], "NONE")
        self.assertTrue(ne.confirms_side("POSITIVE", "BUY")); self.assertFalse(ne.confirms_side("POSITIVE", "SELL"))
        self.assertIsNone(ne.confirms_side("NONE", "BUY"))

    def test_calendar_and_earnings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.json"; p.write_text(json.dumps({"events": [{"date": "2026-12-05", "name": "RBI policy"}]}))
            self.assertEqual(ne.calendar_events(date(2026, 12, 5), p)[0]["name"], "RBI policy")
            self.assertEqual(ne.calendar_events(date(2026, 12, 6), p), [])
        self.assertEqual(ne.calendar_events(date(2026, 1, 1), Path("/nope.json")), [])
        self.assertTrue(ne.earnings_today("ZZTEST1", date(2026, 11, 16), lambda s: {"Earnings Date": [date(2026, 11, 16)]}))
        self.assertIsNone(ne.earnings_today("ZZTEST2", date(2026, 11, 16), lambda s: (_ for _ in ()).throw(RuntimeError())))


class LabelTests(unittest.TestCase):
    def test_labels(self):
        p = {"side": "BUY", "day_move_pct": 1.2, "relative_volume": 2.5, "candle_time": "2026-11-13T09:45:00+05:30",
             "pro": {"confirms": {"candles": True}, "conflicts": {"volume_profile": True}, "confluence": 3, "conflict_count": 1,
                     "candles": ["hammer"], "chart_patterns": [], "volume_profile": {"price_vs_value": "ABOVE_VALUE"}, "indicators": {}},
             "news_events": {"sentiment": "POSITIVE", "tags": ["orders"], "macro_events": []}}
        labels = set(fl.extra_labels(p))
        for want in ("Candles confirms side", "Volume profile against side", "Pro confluence >= 3", "Candle: hammer",
                     "News confirms side", "News event: orders", "Not chased (day move <= 2.8%)", "RVOL >= 2.2x"):
            self.assertIn(want, labels)
        self.assertNotIn("Midday window (10:00-13:00)", labels)
        self.assertEqual(fl.extra_labels({})[:0], [])


class FeatureLearnerTests(unittest.TestCase):
    def rows(self, n_days, helpful_bonus):
        rng = np.random.default_rng(7); out = []
        for d in range(n_days):
            for k in range(4):
                tag = k < 2
                out.append({"day": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}", "net_r": float(rng.normal(0, 0.3) + (helpful_bonus if tag else -0.1)),
                            "labels": {"Pattern: x"} if tag else {"Other"}})
        return out

    def test_collecting_until_enough_sessions(self):
        self.assertEqual(lrn.learn_from_rows(self.rows(10, 0.8))["status"], "collecting")

    def test_promotes_stable_helpful_feature(self):
        r = lrn.learn_from_rows(self.rows(40, 0.8))
        self.assertEqual(r["status"], "validated"); self.assertIn("Pattern: x", r["positive"]); self.assertIn("Other", r["negative"])

    def test_noise_rarely_promotes(self):
        # 30 trials of pure noise with 10 candidate labels: expect only occasional false promotions
        false = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            rows = [{"day": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}", "net_r": float(rng.normal(0, 1.1)),
                     "labels": {f"L{j}" for j in range(10) if rng.random() < 0.5}} for d in range(40) for _ in range(4)]
            r = lrn.learn_from_rows(rows)
            false += bool(r["positive"] or r["negative"])
        self.assertLessEqual(false / 30, 0.20)

    def test_strong_real_effect_is_usually_found(self):
        found = 0
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            rows = []
            for d in range(60):
                for _ in range(6):
                    labs = {f"L{j}" for j in range(10) if rng.random() < 0.5}
                    rows.append({"day": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}",
                                 "net_r": float(rng.normal(0, 1.1) + (0.8 if "L0" in labs else 0)), "labels": labs})
            found += "L0" in lrn.learn_from_rows(rows)["positive"]
        self.assertGreaterEqual(found / 20, 0.6)

    def test_adjustment_is_bounded_and_zero_by_default(self):
        self.assertEqual(lrn.score_adjustment(None, ["x"]), 0)
        v = {"positive": {"a": {}, "b": {}, "c": {}}, "negative": {"d": {}}}
        self.assertEqual(lrn.score_adjustment(v, ["a", "b", "c"]), 6)
        self.assertEqual(lrn.score_adjustment(v, ["d"]), -3)

    def test_load_requires_validated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.json"; p.write_text(json.dumps({"status": "collecting"}))
            self.assertIsNone(lrn.load(p)); self.assertIsNone(lrn.load(Path(d) / "missing.json"))


class KronosAdvisoryTests(unittest.TestCase):
    def sig(self):
        return {"score": 90, "tier": "TIER-1", "relative_volume": 2.5, "cumulative_relative_volume": 1.5, "side": "BUY",
                "forecast": {"model": "Kronos-mini", "direction_aligned": False},
                "market_context": {"sector": "IT", "peer_relative_pct": 0.5, "peer_count": 4}}

    def market(self):
        return {"sectors": [{"sector": "IT", "count": 8, "change_pct": 0.6, "advancing_pct": 70}]}

    def test_disagreeing_kronos_blocks_only_when_required(self):
        act = {"status": "RESEARCH_ONLY", "minimum_rvol_for_validated_alert": 1.8}
        os.environ["KRONOS_REQUIRED"] = "true"
        self.assertFalse(aj.confirmed_gate(self.sig(), self.market(), act)[0])
        os.environ["KRONOS_REQUIRED"] = "false"
        try:
            self.assertTrue(aj.confirmed_gate(self.sig(), self.market(), act)[0])
        finally:
            os.environ.pop("KRONOS_REQUIRED", None)


if __name__ == "__main__":
    unittest.main()
