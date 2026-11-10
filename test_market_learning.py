import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import pandas as pd
import market_learning as ml
import scanner as s

class LearningTests(unittest.TestCase):
    def bars(self):
        idx=pd.date_range('2026-11-09 10:05','2026-11-09 15:25',freq='5min',tz=ml.TZ)
        return pd.DataFrame({'open':100.,'high':100.5,'low':99.5,'close':100.,'volume':100},index=idx)
    def payload(self):
        return dict(side='BUY',price=100,stop=99,target=102,candle_time='2026-11-09T10:00:00+05:30')
    def test_partial_session_not_graded(self):
        self.assertIsNone(s.resolve_outcome(self.payload(), self.bars().iloc[:-1]))
    def test_missing_bar_not_graded(self):
        self.assertIsNone(s.resolve_outcome(self.payload(), self.bars().drop(self.bars().index[3])))
    def test_same_bar_stop_wins(self):
        b=self.bars();b.iloc[0,b.columns.get_loc('high')]=103;b.iloc[0,b.columns.get_loc('low')]=98
        self.assertEqual(s.resolve_outcome(self.payload(),b)['r_result'],-1)
    def test_gap_stop(self):
        b=self.bars();b.iloc[0]=[97,98,96,97,100]
        self.assertEqual(s.resolve_outcome(self.payload(),b)['r_result'],-3)
    def test_sell_target(self):
        p=self.payload();p.update(side='SELL',stop=101,target=98)
        b=self.bars();b.iloc[0,b.columns.get_loc('low')]=97
        self.assertEqual(s.resolve_outcome(p,b)['r_result'],2)
    def test_no_data_not_loss(self):
        c=sqlite3.connect(':memory:');c.executescript('CREATE TABLE signals(id,created_at,payload); CREATE TABLE outcomes(signal_id,result,r_result);')
        p=self.payload();p.update(symbol='X',paper_cost_bps=20)
        c.execute('INSERT INTO signals VALUES(1,?,?)',('2026-11-09',json.dumps(p)))
        c.execute("INSERT INTO outcomes VALUES(1,'NO_DATA',NULL)")
        self.assertIsNone(ml.ledger(c)[0]['net_r'])
    def test_costs(self):
        c=sqlite3.connect(':memory:');c.executescript('CREATE TABLE signals(id,created_at,payload); CREATE TABLE outcomes(signal_id,result,r_result);')
        p=self.payload();p.update(symbol='X',paper_cost_bps=20)
        c.execute('INSERT INTO signals VALUES(1,?,?)',('2026-11-09',json.dumps(p)))
        c.execute("INSERT INTO outcomes VALUES(1,'T1',2)")
        self.assertAlmostEqual(ml.ledger(c)[0]['net_r'],1.8)
    def test_fresh_peers(self):
        b=self.bars();old=b.iloc[:1].copy();old.index=old.index-pd.Timedelta(days=1)
        frame=pd.concat([old,b]);now=datetime(2026,11,9,15,30,tzinfo=ml.TZ)
        md={k:dict(sector='IT',industry='Software',updated_at=now.isoformat()) for k in ['X','Y']}
        snap=ml.snapshot({'X':frame,'Y':frame,'STALE':old},md,now)
        self.assertEqual(snap['fresh'],2);self.assertEqual(snap['stocks']['X']['peer_count'],1)
    def test_insufficient_data_no_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            r=ml.tune(sqlite3.connect(':memory:'),[],{'F':1},Path(d)/'w.json',datetime.now(ml.TZ))
            self.assertEqual(r['status'],'collecting');self.assertEqual(r['formula_weights'],{'F':1})
    def test_holdout_reversal_no_promotion(self):
        rows=[]
        for day in range(1,31):
            for i in range(5):
                rows.append(dict(day=f'2026-10-{day:02}',net_r=1 if day<22 else -1,formulas=['F'],cost_source='entry snapshot'))
        with tempfile.TemporaryDirectory() as d:
            r=ml.tune(sqlite3.connect(':memory:'),rows,{'F':1},Path(d)/'w.json',datetime.now(ml.TZ))
            self.assertEqual(r['status'],'no_stable_change')
    def test_stable_promotion(self):
        rows=[dict(day=f'2026-10-{day:02}',net_r=1,formulas=['F'],cost_source='entry snapshot') for day in range(1,31) for _ in range(5)]
        with tempfile.TemporaryDirectory() as d:
            r=ml.tune(sqlite3.connect(':memory:'),rows,{'F':1},Path(d)/'w.json',datetime.now(ml.TZ))
            self.assertEqual(r['formula_weights']['F'],1.15)

if __name__=='__main__': unittest.main()
