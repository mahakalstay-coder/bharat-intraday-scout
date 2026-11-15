import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import ml_calibrator as ml


def samples(noise=False):
    rng = np.random.default_rng(44)
    x = rng.normal(size=500)
    y = rng.normal(size=500)>0 if noise else x>0
    days = pd.date_range("2024-01-01",periods=100)
    return [dict(day=str(days[i//5].date()),net_r=1 if y[i] else -1,cost_source="entry snapshot",
                 result="T1" if y[i] else "STOP",payload={"pro":{"x":x[i]}}) for i in range(500)]


class CalibratorTests(unittest.TestCase):
    def test_signal(self):
        rows = samples()
        a = ml.train(rows)
        self.assertEqual(a["status"],"validated")
        self.assertGreaterEqual(a["holdout_auc"],.7)
        self.assertEqual(a["feature_importance_top10"][0][0],"pro.x")
        self.assertEqual(len(a["cv"]),5)
        self.assertEqual(json.dumps(a),json.dumps(ml.train(rows)))
        self.assertGreater(ml.predict({"pro":{"x":2}},a),.5)

    def test_noise(self):
        with TemporaryDirectory() as path:
            dest = Path(path)/"model.json"
            r = ml.train(samples(True),dest)
            self.assertEqual(r["status"],"no_stable_change")
            self.assertAlmostEqual(r["holdout_auc"],.5,delta=.15)
            self.assertNotIn("booster",json.loads(dest.read_text()))

    def test_small_and_missing(self):
        self.assertEqual(ml.train(samples()[:200])["status"],"collecting")
        with patch.dict("sys.modules",lightgbm=None):
            self.assertEqual(ml.train([])["status"],"missing_dependency")

    def test_logged_probability_uses_a_model_available_at_entry(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.executescript("CREATE TABLE signals(id,created_at,payload); CREATE TABLE app_state(key,value)")
        model = ml.train(samples(),now="2025-02-01T00:00:00+00:00")
        conn.execute("INSERT INTO app_state VALUES('v5_calibrator',?)",(json.dumps(model),))
        for sid,day in enumerate(["2025-01-01","2025-02-02"]):
            payload = {"pro":{"x":2,"renko":{}},"validation":{"approved":False}}
            conn.execute("INSERT INTO signals VALUES(?,?,?)",(sid,day+"T10:00:00+05:30",json.dumps(payload)))
        ml.annotate(conn,Path("/unused"))
        old,new = [json.loads(v[0]) for v in conn.execute("SELECT payload FROM signals ORDER BY id")]
        self.assertIsNone(old["ml_target_probability"])
        self.assertGreater(new["ml_target_probability"],.5)
        self.assertFalse(new["validation"]["approved"])
        conn.close()
