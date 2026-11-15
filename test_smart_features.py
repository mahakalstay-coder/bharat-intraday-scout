import json, sqlite3, unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import features_pro as fp
import feature_labels as fl
import feature_learner as lrn
import mirror_study as ms
import news_events as ne

TZ = ZoneInfo("Asia/Kolkata")


def ohlc(rows, start="2026-11-13 09:15"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    return df


def flat(n, px=100.0, spread=0.15):
    return [(px, px + spread, px - spread, px)] * n


class AnatomyAndLevels(unittest.TestCase):
    def test_wick_and_body(self):
        d = ohlc(flat(40) + [(100.0, 100.1, 98.0, 100.1)])
        a = fp.bar_anatomy(d)
        self.assertEqual(a["rejection"], "LOWER_WICK"); self.assertGreater(a["lower_wick_ratio"], 0.8)
        d2 = ohlc(flat(40) + [(100.0, 102.0, 99.9, 100.0)])
        self.assertEqual(fp.bar_anatomy(d2)["rejection"], "UPPER_WICK")

    def test_support_zone_touched_twice_then_tested(self):
        path = []
        for _ in range(3):  # three dips to ~99 with rallies between -> support zone with several touches
            path += [(101, 101.2, 100.8, 101)] * 4 + [(100.5, 100.6, 99.0, 99.4)] + [(99.4, 99.9, 99.2, 99.8)] * 2 + [(99.8, 101.3, 99.7, 101.2)] * 3
        path += [(100.8, 101.0, 100.7, 100.9)] * 4 + [(100.0, 100.1, 99.05, 99.9)]  # latest bar tests ~99
        z = fp.sr_zones(ohlc(path))
        self.assertTrue(z["at_support"], z); self.assertGreaterEqual(z["support_touches"], 2)
        self.assertFalse(z["at_resistance"])


class PatternTests(unittest.TestCase):
    def swings(self, peaks, troughs):
        """alternate trough/peak legs, linear between points"""
        pts = []
        for t, p in zip(troughs, peaks):
            pts += [t, p]
        pts.append(troughs[-1])
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            out += list(np.linspace(a, b, 7)[1:])
        return out

    def test_head_and_shoulders_breakdown(self):
        path = self.swings([103, 106, 103.05], [100, 101, 101.0]) + list(np.linspace(101.0, 98.5, 8))
        c = np.array(path)
        d = ohlc([(a, max(a, b) + 0.05, min(a, b) - 0.05, b) for a, b in zip(np.r_[c[0], c[:-1]], c)])
        self.assertIn("head_shoulders_breakdown", fp.chart_patterns(d))

    def test_inverse_head_and_shoulders_breakout(self):
        pts = [103, 100, 103, 97, 103, 100.05, 104.5]      # left shoulder, head, right shoulder, then breakout
        path = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            path += list(np.linspace(a, b, 7)[1:])
        c = np.array(path)
        d = ohlc([(a, max(a, b) + 0.05, min(a, b) - 0.05, b) for a, b in zip(np.r_[c[0], c[:-1]], c)])
        self.assertIn("inverse_head_shoulders_breakout", fp.chart_patterns(d))

    def test_head_shoulders_not_found_in_random_walk_often(self):
        hits = 0
        for seed in range(60):
            c = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.15, 90))
            d = ohlc([(a, max(a, b) + 0.1, min(a, b) - 0.1, b) for a, b in zip(np.r_[c[0], c[:-1]], c)])
            hits += any("head_shoulders" in x for x in fp.chart_patterns(d))
        self.assertLess(hits / 60, 0.15)   # strict geometry: should be rare on noise


class SmartMoneyTests(unittest.TestCase):
    def test_bullish_fvg_retest(self):
        rows = flat(30) + [(100, 100.2, 99.9, 100.1), (100.1, 101.5, 100.1, 101.4), (101.4, 102.2, 101.0, 102.0)]  # gap 100.2 -> 101.0
        rows += [(102.0, 102.1, 101.9, 102.0)] * 3 + [(101.8, 101.9, 100.7, 101.6)]  # dips back into the gap, not through
        f = fp.fair_value_gaps(ohlc(rows))
        self.assertTrue(f["in_bull_fvg"]); self.assertGreaterEqual(f["open_bull_gaps"], 1)

    def test_filled_gap_is_not_a_retest(self):
        rows = flat(30) + [(100, 100.2, 99.9, 100.1), (100.1, 101.5, 100.1, 101.4), (101.4, 102.2, 101.0, 102.0)]
        rows += [(102, 102.1, 99.5, 99.8)] + [(99.8, 100.0, 99.7, 99.9)] * 3
        self.assertFalse(fp.fair_value_gaps(ohlc(rows))["in_bull_fvg"])

    def test_bullish_order_block_first_retest(self):
        rows = flat(20) + [(100.3, 100.4, 99.6, 99.7)]                       # last down candle (the block 99.6-100.4)
        rows += [(99.7, 101.6, 99.7, 101.5), (101.5, 102.8, 101.4, 102.7), (102.7, 103.5, 102.6, 103.4)]  # displacement + break
        rows += [(103.4, 103.6, 103.2, 103.5)] * 4 + [(103.0, 103.1, 100.3, 100.9)]   # first return into the block
        ob = fp.order_blocks(ohlc(rows))
        self.assertTrue(ob["at_bull_ob"], ob)

    def test_no_order_block_in_quiet_market(self):
        self.assertFalse(fp.order_blocks(ohlc(flat(60)))["at_bull_ob"])


class ComputeIntegration(unittest.TestCase):
    def test_compute_exposes_new_groups(self):
        c = 100 + np.cumsum(np.random.default_rng(2).normal(0.02, 0.25, 160))
        rows = [(a, max(a, b) + 0.2, min(a, b) - 0.2, b) for a, b in zip(np.r_[c[0], c[:-1]], c)]
        idx = pd.concat([pd.Series(pd.date_range("2026-11-12 09:15", periods=80, freq="5min", tz="Asia/Kolkata")),
                         pd.Series(pd.date_range("2026-11-13 09:15", periods=80, freq="5min", tz="Asia/Kolkata"))])
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx)); df["volume"] = 1000.0
        out = fp.compute(df, "BUY")
        for k in ("levels", "anatomy", "fvg", "order_blocks"):
            self.assertIn(k, out)
        self.assertEqual(out["groups"], 7); self.assertEqual(set(out["confirms"]), set(out["conflicts"]))
        self.assertTrue(0 <= out["confluence"] <= 7)


class ComboLabelTests(unittest.TestCase):
    def test_hammer_at_support_labels(self):
        p = {"side": "BUY", "candle_time": "2026-11-13T10:30:00+05:30",
             "pro": {"groups": 7, "confluence": 4, "conflict_count": 0,
                     "confirms": {"candles": True, "levels": True, "smart_money": False},
                     "conflicts": {}, "candles": ["hammer"], "chart_patterns": [],
                     "anatomy": {"rejection": "LOWER_WICK", "body_x_avg": 0.4},
                     "levels": {"at_support": True}, "fvg": {}, "order_blocks": {}}}
        labels = set(fl.extra_labels(p))
        for want in ("Candle: hammer", "Level: at support", "Combo: candle + level confirm", "Combo: wick rejection at level",
                     "Wick: lower wick rejection", "Pro confluence >= 4 of 7", "Body: weak (< 0.6x avg)"):
            self.assertIn(want, labels)
        sell = dict(p, side="SELL")
        self.assertNotIn("Combo: wick rejection at level", fl.extra_labels(sell))


class PairLearningTests(unittest.TestCase):
    def test_discovers_combination_but_not_noise_singles(self):
        rng = np.random.default_rng(1); rows = []
        for d in range(100):
            for _ in range(6):
                labs = {f"F{j}:v" for j in range(12) if rng.random() < 0.5}
                eff = 1.2 if ("F0:v" in labs and "F1:v" in labs) else 0.0
                rows.append({"day": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}", "net_r": float(rng.normal(0, 1.1) + eff), "labels": labs})
        r = lrn.learn_from_rows(rows)
        self.assertTrue(any(lrn.SEP in k and "F0:v" in k and "F1:v" in k for k in r["positive"]), r["positive"].keys())

    def test_pair_adjustment_needs_both_labels(self):
        v = {"positive": {"A:x" + lrn.SEP + "B:y": {}}, "negative": {}}
        self.assertEqual(lrn.score_adjustment(v, ["A:x", "B:y"]), 3)
        self.assertEqual(lrn.score_adjustment(v, ["A:x"]), 0)

    def test_same_family_not_paired(self):
        rows = [{"day": f"2026-01-{1 + d % 28:02d}", "net_r": 0.0, "labels": {"News: Positive", "News confirms side"}} for d in range(60)]
        r = lrn.learn_from_rows(rows)
        self.assertEqual(r["status"], "no_stable_features")


class MirrorTests(unittest.TestCase):
    def test_mirror_math_buy_and_sell(self):
        b = ms.make_mirror_payload({"symbol": "X", "side": "BUY", "price": 100.0, "stop": 99.0, "target": 102.5, "candle_time": "t"})
        self.assertEqual((b["side"], b["stop"], b["target"]), ("SELL", 101.0, 97.5))
        s = ms.make_mirror_payload({"symbol": "X", "side": "SELL", "price": 100.0, "stop": 101.0, "target": 97.5, "candle_time": "t"})
        self.assertEqual((s["side"], s["stop"], s["target"]), ("BUY", 99.0, 102.5))
        self.assertIsNone(ms.make_mirror_payload({"symbol": "X", "side": "BUY", "price": 100, "stop": 101, "target": 99, "candle_time": "t"}))
        self.assertIsNone(ms.make_mirror_payload({"symbol": "X"}))

    def test_record_resolve_report_end_to_end(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, created_at TEXT, symbol TEXT, side TEXT, payload TEXT)")
        conn.execute("CREATE TABLE outcomes (signal_id INTEGER, result TEXT, r_result REAL)")
        now = datetime(2026, 11, 16, 16, 0, tzinfo=TZ)
        for i in range(40):
            p = {"symbol": f"S{i}", "side": "BUY", "price": 100.0, "stop": 99.2, "target": 102.0, "candle_time": "2026-11-13T10:00:00+05:30",
                 "paper_cost_bps": 20, "shadow_only": False}
            conn.execute("INSERT INTO signals VALUES (?,?,?,?,?)", (i + 1, "2026-11-13T10:00:00+05:30", p["symbol"], "BUY", json.dumps(p)))
            conn.execute("INSERT INTO outcomes VALUES (?,?,?)", (i + 1, "T1", 2.5))   # originals all win
            self.assertTrue(ms.record(conn, i + 1, p, datetime(2026, 11, 13, 10, 0, tzinfo=TZ)))
        self.assertFalse(ms.record(conn, 1, p, now))                                  # no double insert
        fake = lambda mp, frame: {"result": "STOP", "r_result": -1.0}                  # mirrors all stop out
        n = ms.resolve_pending(conn, {f"S{i}": pd.DataFrame({"x": [1]}) for i in range(40)}, fake, lambda f: f, now)
        self.assertEqual(n, 40)
        rep = ms.report(conn)
        self.assertEqual(rep["status"], "evaluated"); self.assertGreater(rep["direction_edge_r"], 3)
        self.assertEqual(rep["original_beats_mirror_pct"], 100.0); self.assertGreater(rep["edge_lower_bound_r"], 0)
        self.assertIn("Directional edge", rep["reading"])
        self.assertIn("pairs", ms.digest_line(rep))

    def test_no_edge_reading_when_mirror_equals_original(self):
        pairs = [{"orig_net": x, "mirror_net": x, "orig_result": "STOP", "mirror_result": "STOP"} for x in np.linspace(-1.2, 1.5, 40)]
        r = ms.compare(pairs)
        self.assertEqual(r["status"], "evaluated"); self.assertIn("No directional edge", r["reading"])

    def test_backfill_recent_only_and_idempotent(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, created_at TEXT, symbol TEXT, side TEXT, payload TEXT)")
        mk = lambda sym, ct: json.dumps({"symbol": sym, "side": "BUY", "price": 100.0, "stop": 99.0, "target": 102.5, "candle_time": ct})
        conn.execute("INSERT INTO signals VALUES (1,'x','NEW','BUY',?)", (mk("NEW", "2026-11-12T10:00:00+05:30"),))
        conn.execute("INSERT INTO signals VALUES (2,'x','OLD','BUY',?)", (mk("OLD", "2026-10-01T10:00:00+05:30"),))
        now = datetime(2026, 11, 14, 16, 0, tzinfo=TZ)
        self.assertEqual(ms.backfill(conn, now), 1)
        self.assertEqual(ms.backfill(conn, now), 0)
        self.assertEqual(conn.execute("SELECT symbol FROM mirror_trades").fetchall(), [("NEW",)])

    def test_pending_session_is_not_graded_and_flag_off(self):
        conn = sqlite3.connect(":memory:")
        p = {"symbol": "S", "side": "SELL", "price": 50.0, "stop": 50.5, "target": 48.75, "candle_time": "t"}
        self.assertTrue(ms.record(conn, 1, p, datetime(2026, 11, 16, 10, 0, tzinfo=TZ)))
        n = ms.resolve_pending(conn, {}, lambda *a: {"result": "T1", "r_result": 2.5}, lambda f: f, datetime(2026, 11, 16, 11, 0, tzinfo=TZ))
        self.assertEqual(n, 0)
        import os; os.environ["ENABLE_MIRROR_TRADES"] = "false"
        try: self.assertFalse(ms.record(conn, 2, p, datetime(2026, 11, 16, 10, 0, tzinfo=TZ)))
        finally: os.environ.pop("ENABLE_MIRROR_TRADES", None)


RSS = """<rss><channel>
<item><title>ABC bags order worth Rs 400 crore - Economic Times</title><pubDate>{fresh}</pubDate><source>Economic Times</source></item>
<item><title>ABC shares plunge on SEBI probe - Mint</title><pubDate>{old}</pubDate><source>Mint</source></item>
<item><title>ABC bags order worth Rs 400 crore - Economic Times</title><pubDate>{fresh}</pubDate><source>Economic Times</source></item>
<item><title>Ancient ABC news falls</title><pubDate>{ancient}</pubDate></item>
</channel></rss>"""


class NewsUpgradeTests(unittest.TestCase):
    def test_recency_weighting_dedupe_and_age_cutoff(self):
        now = datetime(2026, 11, 13, 12, 0, tzinfo=TZ)
        fmt = lambda dt: dt.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        xml = RSS.format(fresh=fmt(now - timedelta(hours=2)), old=fmt(now - timedelta(hours=30)), ancient=fmt(now - timedelta(days=9)))
        items = ne.parse_rss(xml)
        self.assertEqual(items[0]["title"], "ABC bags order worth Rs 400 crore")
        r = ne.classify_items(items, now)
        self.assertEqual(r["headline_count"], 2)           # duplicate and 9-day-old dropped
        self.assertEqual(r["fresh_count"], 1)
        self.assertEqual(r["sentiment"], "POSITIVE")       # fresh +2 (order) beats older -1 (probe/plunge x1 weight)
        self.assertIn("regulatory", r["tags"]); self.assertIn("orders", r["tags"])

    def test_fetch_failure_and_empty(self):
        self.assertEqual(ne.fetch_google_news("ABC", fetch=lambda u: (_ for _ in ()).throw(RuntimeError())), [])
        self.assertEqual(ne.fetch_google_news("ABC", fetch=lambda u: ""), [])
        self.assertEqual(ne.classify_items([], datetime.now(timezone.utc))["sentiment"], "NONE")

    def test_fetch_builds_query_url(self):
        seen = []
        ne.fetch_google_news("PNCINFRA", fetch=lambda u: seen.append(u) or "")
        self.assertIn("PNCINFRA", seen[0]); self.assertIn("news.google.com/rss/search", seen[0])


if __name__ == "__main__":
    unittest.main()
