import inspect
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import scanner as sc
import feature_labels
import mirror_study
from test_features_pro_v5 import frame


class IsolationTests(unittest.TestCase):
    def test_protected_sources(self):
        text = "\n".join(inspect.getsource(f) for f in
                         (sc.aj.confirmed_gate,sc.aj._confidence_report,sc.aj._bootstrap_lcb,mirror_study))
        for name in ("features_renko","features_gann","features_smc","features_cycles","ml_calibrator","heatmap_extra"):
            self.assertNotIn(name,text)

    def test_research_labels_are_opt_in(self):
        p = dict(pro={"renko":{"renko_trend":"UP"}},side="BUY")
        self.assertNotIn("Renko: up trend",feature_labels.extra_labels(p))
        self.assertIn("Renko: up trend",feature_labels.extra_labels(p,research=True))

    def test_scan_and_daily_complete_with_failed_research(self):
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            root = Path(t)
            (root/"data").mkdir()
            cache = root/"symbols.json"
            cache.write_text(json.dumps(dict(scan_symbols=[dict(symbol="X")])))
            def stub(obj,name,**kw):
                return stack.enter_context(patch.object(obj,name,**kw))
            for k,v in dict(ROOT=root,DB_PATH=root/"test.sqlite",UNIVERSE_CACHE=cache).items():
                stub(sc,k,new=v)
            stack.enter_context(patch.dict(os.environ,ENABLE_GOOGLE_NEWS="false",ENABLE_EARNINGS_LOOKUP="false",GITHUB_ACTIONS="false"))
            stack.enter_context(patch.dict(sc.CFG,KRONOS_REQUIRED=False,TRADING_R1_REQUIRED=False,KRONOS_ENABLED=False))
            now = datetime(2026,11,16,16,tzinfo=sc.TZ)
            stub(sc,"datetime").now.return_value = now
            d = frame(100+np.sin(np.arange(128)))
            d.index = pd.date_range("2026-11-16 09:15",periods=len(d),freq="5min",tz=sc.TZ)
            signal = dict(symbol="X",side="BUY",tier="TIER-1",score=92,price=100,stop=99,target=103,
                          relative_volume=2.4,cumulative_relative_volume=2,candle_time=d.index[50].isoformat(),
                          forecast={"model":"Kronos-mini","direction_aligned":True})
            market = dict(generated_at=now.isoformat(),scope="test",stocks={"X":dict(symbol="X",sector="IT",industry="IT",
                          peer_count=3,peer_relative_pct=1,change_pct=1,peers=[])},
                          sectors=[dict(sector="IT",count=5,change_pct=1,advancing_pct=80)])
            for name,val in dict(download_batch={"X":d},is_entry_window=True,get_window_signal_counts=("A",0,2,0),
                                 deliver_pending_signals=0,rank_pre_explosion_candidates=[],evaluate_symbol=signal,
                                 learned_gate={"ok":True},fetch_stock_news_headlines=[],trading_r1_evaluate_candidate={},
                                 resolve_pending_outcomes=[],study_movers=[],recompute_filters={"mode":"warmup"}).items():
                stub(sc,name,return_value=val)
            stub(sc.ml,"snapshot",return_value=market)
            stub(sc.market_regime,"current_regime",return_value={})
            stub(sc.research,"sector_update",return_value=False)
            inserted = stub(sc,"add_signal",return_value=True)
            failed = []
            for module in (sc.features_smc,sc.features_renko,sc.features_cycles,sc.features_gann):
                failed.append(stub(module,"compute",side_effect=RuntimeError("test")))
            failed.append(stub(sc.heatmap_extra,"collect",side_effect=RuntimeError("test")))
            failed.append(stub(sc.ml_calibrator,"refresh",side_effect=RuntimeError("test")))
            summary = sc.run_scan_cycle()
            self.assertEqual(summary["new_validated_alerts_queued"],1)
            self.assertFalse(inserted.call_args.kwargs["shadow_only"])
            self.assertIn("confirms",signal["pro"])
            self.assertEqual(sc.run_daily_learning(send_telegram=False)["day"],"2026-11-16")
            self.assertTrue(all(mock.called for mock in failed))
