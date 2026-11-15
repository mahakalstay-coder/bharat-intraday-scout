import unittest
import numpy as np
import pandas as pd
import features_renko as renko


def frame(c):
    c = np.asarray(c, dtype=float)
    return pd.DataFrame(dict(close=c, open=c, high=c+.5, low=c-.5, volume=100),
                        index=pd.date_range("2025-01-01", periods=len(c), freq="5min"))


class RenkoTests(unittest.TestCase):
    def test_atr_reversal(self):
        d = frame(np.r_[np.arange(50)*.5+100, 124,123.5,123,122.5,122,121.5,121])
        # ATR14 is exactly 1; the final up-brick closes at 124.5.
        self.assertEqual(renko.compute(d.iloc[:50])["renko_trend"], "UP")
        r = renko.compute(d)
        self.assertEqual(r["renko_trend"], "FLAT")
        self.assertEqual(r["renko_reversal_within"], 3)
        self.assertAlmostEqual(r["renko_distance_atr"], 3.5)
        self.assertEqual(renko.compute(frame(np.arange(60)), mode="classic")["renko_trend"], "UP")

import features_smc as smc


class StructureTests(unittest.TestCase):
    def test_structure(self):
        c = np.r_[np.ones(40)*100, 102,100,98,100,100,100,100,104]
        self.assertTrue(smc.compute(frame(c))["bos_recent"])
        self.assertTrue(smc.compute(frame(np.r_[c,96]))["choch_recent"])
        d = frame(c)
        d.iloc[-1,d.columns.get_loc("close")] = 101
        self.assertTrue(smc.compute(d)["liquidity_sweep"])
        self.assertGreaterEqual(smc.compute(d)["equal_highs"], 2)

import features_cycles as cycles


class SpectralTests(unittest.TestCase):
    def test_sine(self):
        d = frame(100+2*np.sin(np.arange(150)*2*np.pi/16))
        for i in range(128,140):
            r = cycles.compute(d.iloc[:i])
            self.assertAlmostEqual(r["fft_dominant_period_bars"],16,delta=1)
            self.assertAlmostEqual(r["hilbert_amplitude_pct"]*d.close.iloc[i-1]/100,2)

import features_gann as gann
from features_pro import _swings


class GeometryTests(unittest.TestCase):
    def test_anchors(self):
        d = frame(np.interp(np.arange(96),[0,22,60,95],[100,90,110,100]))
        d.index = pd.date_range("2025-01-01",periods=len(d))
        self.assertEqual(_swings(d.high.values,d.low.values,20),([60],[22]))
        r = gann.compute(d)
        self.assertAlmostEqual(r["gann_retracement_pct"],50)
        self.assertEqual(r["gann_time_days_from_swing"],35)

    def test_incomplete(self):
        d = frame(np.arange(45)+100)
        for module in (renko,smc,cycles,gann):
            for x in (d, d.iloc[:2],d.assign(volume=0),d.assign(close=np.nan),d.drop(columns="open")):
                facts = module.compute(x,"SELL")
                self.assertEqual(set(facts),set(module.KEYS))
                self.assertTrue(all(v is None or isinstance(v,str) or np.isfinite(v) for v in facts.values()))
