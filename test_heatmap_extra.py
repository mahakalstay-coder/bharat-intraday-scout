import unittest
import numpy as np
import pandas as pd
import heatmap_extra as h


class HeatmapTests(unittest.TestCase):
    def test_quadrants(self):
        for prev,now,name in [(102,110,"LEADING"),(110,102,"WEAKENING"),(90,98,"IMPROVING"),(98,90,"LAGGING")]:
            c = pd.Series(np.ones(26)*100)
            c.iloc[-6],c.iloc[-1] = prev,now
            self.assertEqual(h.rs(c,pd.Series(np.ones(26)*100))["quadrant"],name)

    def test_atr_rank(self):
        for spread,rank in [(np.arange(1,41),100),(np.arange(40,0,-1),0)]:
            d = pd.DataFrame(dict(close=100,high=100+spread,low=100-spread))
            self.assertEqual(h.compression(d)["atr_pct_rank_20d"],rank)

    def test_missing(self):
        self.assertIsNone(h.rs(pd.Series([1]),pd.Series(dtype=float)))
        html = h.sections({})
        self.assertEqual(html.count("<section>"),3)
        self.assertEqual(html.count("insufficient_data"),3)

    def test_sector_rvol_and_rendered_tiles(self):
        from unittest.mock import patch
        dates = pd.bdate_range("2026-11-02",periods=6,tz="Asia/Kolkata")+pd.Timedelta(hours=10)
        d = pd.DataFrame(dict(open=100,high=101,low=99,close=100,volume=[100]*5+[300]),index=dates)
        market = {"stocks":{"A":{"sector":"Bank"},"B":{"sector":"Bank"}}}
        with patch.object(h,"daily_history",return_value={}):
            out = h.collect({"A":d,"B":d.assign(volume=[100]*5+[100])},{},market,dates[-1]+pd.Timedelta(minutes=5))
        self.assertEqual(out["rvol_by_sector"],[dict(sector="Bank",n=2,median_rvol=2.,p90_rvol=2.8)])
        self.assertIn("<td>Bank</td>",h.sections(out))
        self.assertEqual(out["rs_vs_nifty"],[])
