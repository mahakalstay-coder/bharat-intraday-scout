import json, os, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import backtest_history as bt
import scanner as sc


def make_symbol(seed, sessions=45, start="2026-06-01", price0=400.0):
    rng = np.random.default_rng(seed); rows = []; px = price0
    for d in pd.bdate_range(start, periods=sessions, tz="Asia/Kolkata"):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min")
        trend, sgn = rng.random() < 0.30, rng.choice([-1, 1]); px *= 1 + rng.normal(0, 0.004)
        for k, ts in enumerate(idx):
            drift = sgn * 0.0009 if (trend and 6 <= k <= 45) else 0.0
            vm = (2.6 if (trend and 6 <= k <= 45) else 1.0) * (1.6 if k < 6 else 1.0)
            o = px; c = o * (1 + rng.normal(drift, 0.0016))
            h = max(o, c) * (1 + abs(rng.normal(0, 0.0007))); l = min(o, c) * (1 - abs(rng.normal(0, 0.0007)))
            rows.append((ts, o, h, l, c, max(100, rng.normal(2600, 500)) * vm)); px = c
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def fake_trades(n_days=120, per_day=3, mean_r=0.0, seed=0, buy_bonus=0.0):
    rng = np.random.default_rng(seed); out = []
    for d in range(n_days):
        day = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for k in range(per_day):
            side = "BUY" if rng.random() < 0.5 else "SELL"
            r = float(rng.normal(mean_r + (buy_bonus if side == "BUY" else 0), 1.1))
            grid = {f"{s}|{t}": float(rng.normal(0.0, 1.0)) for s in bt.STOPS_PCT for t in bt.TARGETS_PCT}
            out.append({"day": day, "time": f"10:{k:02d}", "symbol": f"S{k}", "side": side, "tier": "TIER-1", "window": "BEFORE_10AM",
                        "formulas": ["F1_X"], "net_r": r, "net_pct": r * 0.8, "risk_pct": 0.8, "target_pct": 2.0,
                        "result": "T1" if r > 0 else "STOP", "mfe_pct": abs(float(rng.normal(1.5, 1))), "mfe_pre_pct": abs(float(rng.normal(1.0, 1))),
                        "mirror_net_r": float(rng.normal(0, 1.1)), "mirror_result": "STOP", "grid": grid,
                        "payload": {"side": side, "pro": {}, "day_move_pct": 1.0, "relative_volume": 2.0,
                                    "candle_time": f"{day}T10:00:00+05:30", "tier": "TIER-1", "pillars_matched": ["F1_X"], "score": 80}})
    return out


class LoadingTests(unittest.TestCase):
    def frame(self):
        idx = pd.date_range("2026-11-12 09:15", periods=60, freq="5min")
        return pd.DataFrame({"Date": idx.strftime("%Y-%m-%d %H:%M:%S"), "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000})

    def test_naive_strings_are_ist(self):
        df = bt.standardise(self.frame())
        self.assertEqual(str(df.index.tz), "Asia/Kolkata"); self.assertEqual(df.index[0].strftime("%H:%M"), "09:15")

    def test_epoch_timestamps_in_any_unit_are_converted_to_ist(self):
        base = pd.to_datetime(self.frame()["Date"]).dt.tz_localize("Asia/Kolkata")
        secs = (base - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
        for mult in (1, 1000, 1_000_000, 1_000_000_000):          # s, ms, us, ns
            f = self.frame(); f["Date"] = secs * mult
            df = bt.standardise(f)
            self.assertEqual(df.index[0].strftime("%H:%M"), "09:15", mult); self.assertEqual(len(df), 60)

    def test_one_minute_resampled_to_five(self):
        idx = pd.date_range("2026-11-12 09:15", periods=60, freq="1min")
        f = pd.DataFrame({"datetime": idx.strftime("%Y-%m-%d %H:%M"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10})
        df = bt.standardise(f)
        self.assertEqual(len(df), 12); self.assertEqual(df["volume"].iloc[0], 50)

    def test_missing_columns_rejected(self):
        self.assertIsNone(bt.standardise(pd.DataFrame({"a": [1], "b": [2]})))

    def test_loads_folder_and_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            make_symbol(1, 30).to_csv(Path(d) / "abc.NS_5m.csv", index=False)
            self.assertIn("ABC", bt.load_history(d))
            both = pd.concat([make_symbol(1, 30).assign(symbol="AAA"), make_symbol(2, 30).assign(symbol="BBB")])
            both.to_csv(Path(d) / "all.csv", index=False)
            self.assertEqual(set(bt.load_history(str(Path(d) / "all.csv"))), {"AAA", "BBB"})


class GradingTests(unittest.TestCase):
    def after(self, rows):
        return pd.DataFrame(rows, columns=["high", "low", "close"])

    def test_target_hit_and_stop_hit(self):
        win = bt.grade_grid("BUY", 100.0, self.after([(100.5, 99.9, 100.2), (102.2, 100.0, 102.0), (102, 101, 101)]), 0.2)
        self.assertAlmostEqual(win["1.0|2.0"], 2.0 - 0.2)
        loss = bt.grade_grid("BUY", 100.0, self.after([(100.5, 98.9, 99.0)]), 0.2)
        self.assertAlmostEqual(loss["1.0|2.0"], -1.0 - 0.2)

    def test_stop_wins_a_same_bar_tie(self):
        tie = bt.grade_grid("BUY", 100.0, self.after([(102.5, 98.5, 100.0)]), 0.0)
        self.assertAlmostEqual(tie["1.0|2.0"], -1.0)

    def test_sell_side_and_end_of_day_exit(self):
        sell = bt.grade_grid("SELL", 100.0, self.after([(100.2, 97.8, 98.0)]), 0.0)
        self.assertAlmostEqual(sell["1.0|2.0"], 2.0)
        flat = bt.grade_grid("BUY", 100.0, self.after([(100.4, 99.7, 100.3)]), 0.0)
        self.assertAlmostEqual(flat["1.0|2.0"], 0.3)

    def test_grid_targets_are_all_meaningful_by_default(self):
        self.assertGreaterEqual(min(bt.TARGETS_PCT), 1.5)


class MoveBeforeStopTests(unittest.TestCase):
    def test_counts_only_the_move_before_the_stop_and_is_not_capped(self):
        a = pd.DataFrame([(101.0, 99.8), (104.0, 100.5), (104.5, 98.9), (110.0, 99.0)], columns=["high", "low"])
        self.assertAlmostEqual(bt.mfe_before_stop("BUY", 100.0, 99.0, a), 4.0)      # stop hit on bar 3; the 10% rally after is ignored
        b = pd.DataFrame([(100.2, 99.0), (100.4, 95.0)], columns=["high", "low"])
        self.assertAlmostEqual(bt.mfe_before_stop("SELL", 100.0, 101.0, b), 5.0)    # never stopped -> best 5% drop
        c = pd.DataFrame([(101.5, 98.5)], columns=["high", "low"])
        self.assertEqual(bt.mfe_before_stop("BUY", 100.0, 99.0, c), 0.0)           # stopped on the first bar

    def test_replayed_moves_can_exceed_the_first_target(self):
        r = bt.replay_symbol(("S", bt.standardise(make_symbol(1, 45)), {"cost_bps": 20, "scan_every": 5}))
        self.assertTrue(any(t["mfe_pre_pct"] > t["target_pct"] for t in r["trades"]))


class FastVolumeEquivalence(unittest.TestCase):
    def test_matches_live_functions_with_gaps_and_zero_volume(self):
        raw = make_symbol(3, 40)
        raw = raw.drop(raw.sample(frac=0.03, random_state=1).index)
        raw.loc[raw.sample(frac=0.03, random_state=2).index, "volume"] = 0.0
        df = bt.standardise(raw)
        days = sorted(set(df.index.date)); dates = np.array([t.date() for t in df.index])
        bounds = {d: (int(np.argmax(dates == d)), int(len(dates) - np.argmax(dates[::-1] == d))) for d in days}
        fv, bad = bt.FastVolume(df, days, bounds), 0
        for di, day in enumerate(days):
            if di < 10:
                continue
            s, e = bounds[day]; ws = bounds[days[max(0, di - 8)]][0]
            for pos in range(s, e, 3):
                prefix, ts = df.iloc[ws:pos + 1], df.index[pos]
                r1, _ = sc.same_slot_relative_volume(prefix, ts, 3); c1, _ = sc.session_cumulative_relative_volume(prefix, ts, 3)
                r2, c2 = fv.values(day, pos)
                bad += abs(r1 - r2) > 1e-9 or abs(c1 - c2) > 1e-9
        self.assertEqual(bad, 0)


class ReplayIntegrity(unittest.TestCase):
    opts = {"cost_bps": 20, "scan_every": 5}

    def test_fast_path_equals_exact_path(self):
        df = bt.standardise(make_symbol(1, 32))
        fast = bt.replay_symbol(("S", df, self.opts)); exact = bt.replay_symbol(("S", df, {**self.opts, "exact": True}))
        key = lambda r: [(t["day"], t["time"], t["side"], t["net_r"]) for t in r["trades"]]
        self.assertEqual(key(fast), key(exact))

    def test_no_look_ahead_future_data_cannot_change_past_signals(self):
        raw = make_symbol(2, 50)
        full = bt.replay_symbol(("S", bt.standardise(raw), self.opts))
        cut_day = sorted(set(pd.to_datetime(raw["datetime"]).dt.date))[39]
        trunc = bt.replay_symbol(("S", bt.standardise(raw[pd.to_datetime(raw["datetime"]).dt.date <= cut_day]), self.opts))
        key = lambda r: [(t["day"], t["time"], t["side"], t["net_r"], t["result"]) for t in r["trades"] if t["day"] <= str(cut_day)]
        self.assertEqual(key(full), key(trunc))
        self.assertGreater(len(key(full)), 0)

    def test_daily_meta_uses_only_earlier_sessions(self):
        df = bt.standardise(make_symbol(4, 45))
        meta = bt.daily_meta(df, sc)
        later = df.copy(); later.iloc[-75:, later.columns.get_loc("close")] *= 3.0     # corrupt the LAST session
        meta2 = bt.daily_meta(later, sc)
        days = sorted(meta)
        for d in days[:-1]:
            self.assertEqual(meta[d], meta2[d])
        self.assertNotEqual(meta[days[-1]]["previous_close"], 0)

    def test_signals_have_meaningful_targets_and_costs(self):
        r = bt.replay_symbol(("S", bt.standardise(make_symbol(1, 45)), self.opts))
        self.assertGreater(len(r["trades"]), 0)
        for t in r["trades"]:
            self.assertGreaterEqual(t["target_pct"], 1.5); self.assertGreater(t["cost_r"], 0)
            self.assertAlmostEqual(t["net_r"], t["gross_r"] - t["cost_r"], places=3)


class AnalysisTests(unittest.TestCase):
    def test_too_few_trades(self):
        self.assertIn("Too few", bt.analyze(fake_trades(5, 2))["verdict"])

    def test_edge_is_detected_on_unseen_sessions(self):
        rep = bt.analyze(fake_trades(150, 3, mean_r=0.45, seed=1))
        self.assertIn("Positive expectancy", rep["verdict"]); self.assertGreater(rep["holdout_lower_bound_net_r"], 0)

    def test_no_edge_is_reported_honestly(self):
        rep = bt.analyze(fake_trades(150, 3, mean_r=-0.25, seed=2))
        self.assertIn("No edge", rep["verdict"])

    def test_pure_noise_is_not_called_an_edge(self):
        wrong = sum("Positive expectancy" in bt.analyze(fake_trades(120, 3, mean_r=0.0, seed=s))["verdict"] for s in range(12))
        self.assertLessEqual(wrong, 2)

    def test_learner_validates_a_real_side_effect(self):
        rep = bt.analyze(fake_trades(150, 4, mean_r=-0.1, seed=3, buy_bonus=0.9))
        self.assertIn("Side: BUY", rep["validated_features"]["positive"])

    def test_grid_respects_target_floor_and_holdout_columns(self):
        rep = bt.analyze(fake_trades(100, 3, seed=4), min_target_pct=2.5)
        self.assertTrue(all(g["target_pct"] >= 2.5 for g in rep["grid_top_by_train"]))
        self.assertIn("holdout_net_pct", rep["grid_top_by_train"][0])
        self.assertIn("Targets below 2.5%", rep["grid_note"])

    def test_report_has_risk_and_meaningful_move_tables(self):
        rep = bt.analyze(fake_trades(100, 3, seed=5))
        self.assertIn("max_drawdown_r", rep["risk"]); self.assertIn(">= 2.0%", rep["meaningful_moves"]); self.assertIn("by_month", rep)


class CliEndToEnd(unittest.TestCase):
    def test_main_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d) / "h", Path(d) / "o"; data.mkdir()
            for i in range(3):
                make_symbol(i, 40).to_csv(data / f"SYM{i}.csv", index=False)
            self.assertEqual(bt.main(["--data", str(data), "--out", str(out), "--workers", "1"]), 0)
            for f in ("trades.jsonl", "trades.csv", "report.json", "report.md"):
                self.assertTrue((out / f).exists(), f)
            self.assertIn("Verdict", (out / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
