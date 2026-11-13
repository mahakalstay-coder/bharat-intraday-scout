from datetime import datetime,timedelta
import sqlite3
import unittest
from unittest.mock import Mock
import pandas as pd
import adaptive_journal as a
import scout_research as sr

class ResearchTests(unittest.TestCase):
    def test_no_future_or_missing_bars(self):
        idx=pd.date_range('2026-11-12 09:15',periods=8,freq='5min',tz=a.TZ)
        close=pd.Series([100,101,102,103,104,105,106,1],index=idx)
        f=sr.drift_forecast(close,idx[6],6)
        self.assertAlmostEqual(f['forecast_change_pct'],600/106,places=4)
        self.assertEqual(sr.drift_forecast(close.drop(idx[3]),idx[6],6),{})

    def test_drift_only_in_research(self):
        t='2026-11-12T10:00:00+05:30'
        p=dict(tier='TIER-1',score=94,side='SELL',relative_volume=2,cumulative_relative_volume=1.6,
               market_context=dict(sector='Energy',peer_count=3,peer_relative_pct=-1),
               forecast=dict(model='Kronos-mini',direction_aligned=False,forecast_change_pct=.03,horizon_bars=6),
               drift_forecast=dict(model='RandomWalkWithDrift',forecast_change_pct=-.30,horizon_bars=6,asof=t),
               candle_time=t)
        m=dict(sectors=[dict(sector='Energy',count=5,change_pct=-1,advancing_pct=25)])
        self.assertTrue(a.confirmed_gate(p,m,dict(status='RESEARCH_ONLY'))[0])
        self.assertFalse(a.confirmed_gate(p,m,dict(status='VALIDATED_PAPER_MODE'))[0])
        p['forecast']['forecast_change_pct']=.2
        self.assertFalse(a.confirmed_gate(p,m,dict(status='RESEARCH_ONLY'))[0])

    def test_rotation_margin_cooldown_and_dedup(self):
        c=sqlite3.connect(':memory:')
        c.execute('create table app_state(key text primary key,value text)')
        base=datetime(2026,11,12,10,0,tzinfo=a.TZ)
        send=Mock(return_value=True)
        def cycle(minutes,a_gain,b_gain):
            now=base+timedelta(minutes=minutes)
            stocks={}
            for name,g in [('Energy',a_gain),('Technology',b_gain)]:
                for i in range(3):
                    stocks[name+str(i)]=dict(asof=(now-timedelta(minutes=5)).isoformat(),sector=name,
                        industry='same',change_pct=g+i*.1)
            sectors=[dict(sector=n,change_pct=g+.1,count=3,advancing_pct=100)
                     for n,g in [('Energy',a_gain),('Technology',b_gain)]]
            return sr.sector_update(c,dict(stocks=stocks,sectors=sectors),now,send)
        self.assertFalse(cycle(0,1,.5))
        self.assertTrue(cycle(5,1,.5))
        self.assertIn('Energy2',send.call_args.args[0])
        self.assertFalse(cycle(5,1,.5))
        self.assertFalse(cycle(10,1,2))
        self.assertFalse(cycle(15,1,2))
        self.assertFalse(cycle(30,1,1.2))
        self.assertFalse(cycle(35,1,1.2))
        self.assertFalse(cycle(40,1,2))
        self.assertTrue(cycle(45,1,2))
        self.assertIn('Previous leader: Energy',send.call_args.args[0])
        self.assertFalse(cycle(50,1,2))
        self.assertEqual(send.call_count,2)
