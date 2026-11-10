import json
import sqlite3
import unittest
from datetime import datetime, timedelta
import pandas as pd
import adaptive_journal as a

class AdaptiveTests(unittest.TestCase):
    def db(self):
        conn=sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE signals (id INTEGER PRIMARY KEY, created_at TEXT, symbol TEXT, payload TEXT)')
        conn.execute('CREATE TABLE outcomes (signal_id INTEGER PRIMARY KEY, result TEXT, r_result REAL)')
        a.init(conn)
        return conn

    def sample(self,conn, n_days=14, trades_per_day=4, positive=True):
        first=datetime(2026,10,1,tzinfo=a.TZ)
        for d in range(n_days):
            day=first+timedelta(days=d)
            for k in range(trades_per_day):
                sid=d*trades_per_day+k+1
                p={'symbol':f'S{sid}','side':'BUY','price':100,'stop':99,'target':102.5,
                   'relative_volume':1.8+(k%3)*.2,'cumulative_relative_volume':1.5,
                   'paper_cost_bps':20,'tier':'TIER-1','score':88,
                   'forecast':{'model':'Kronos-mini','direction_aligned':True},
                   'market_context':{'sector':'IT','peer_count':3,'peer_relative_pct':1.0},
                   'sector_context':{'sector':'IT','change_pct':1.0,'advancing_pct':75,'count':5},
                   'pillars_matched':['F1']}
                conn.execute('INSERT INTO signals VALUES(?,?,?,?)',
                    (sid,day.isoformat(),p['symbol'],json.dumps(p)))
                net=1.5 if positive else -1
                conn.execute('INSERT INTO outcomes VALUES(?,?,?)',(sid,'T1' if net>0 else 'STOP',net))
        # forecasts: 25 correct of 25
        for sid in range(1,26):
            conn.execute('INSERT INTO kronos_accuracy VALUES (?,?,?,?,?,?)',
                         (sid,'2026-10-01',0.4,0.5,1,0.1))
        conn.commit()

    def test_learning_not_promoted_without_evidence(self):
        c=self.db()
        self.assertEqual(a.mode(c,datetime(2026,11,10,tzinfo=a.TZ))['status'],'RESEARCH_ONLY')
        c.close()

    def test_strong_holdout_promotes_qualified_paper_mode(self):
        c=self.db();self.sample(c)
        r=a.mode(c,datetime(2026,11,10,tzinfo=a.TZ))
        self.assertEqual(r['status'],'VALIDATED_PAPER_MODE')
        self.assertEqual(r['completed_costed_trades'],56)
        c.close()

    def test_negative_holdout_rejects(self):
        c=self.db();self.sample(c,positive=False)
        r=a.mode(c,datetime(2026,11,10,tzinfo=a.TZ))
        self.assertEqual(r['status'],'RESEARCH_ONLY')
        self.assertFalse(r['checks']['holdout_net_expectancy'])
        c.close()

    def test_confirmed_gate_requires_sector_and_peers(self):
        activation={'status':'VALIDATED_PAPER_MODE'}
        signal={'score':89,'tier':'TIER-1','side':'BUY','relative_volume':2.1,
                'cumulative_relative_volume':1.6,'forecast':{'model':'Kronos-mini','direction_aligned':True},
                'market_context':{'sector':'IT','peer_count':3,'peer_relative_pct':1}}
        market={'sectors':[{'sector':'IT','count':5,'change_pct':1.2,'advancing_pct':75}]}
        self.assertTrue(a.confirmed_gate(signal,market,activation)[0])
        signal['market_context']['peer_relative_pct']=-1
        self.assertFalse(a.confirmed_gate(signal,market,activation)[0])

    def test_research_preview_delivers_before_validation(self):
        activation = {'status': 'RESEARCH_ONLY'}
        signal = {
            'score': 90, 'tier': 'TIER-1', 'side': 'BUY',
            'relative_volume': 2.3, 'cumulative_relative_volume': 1.6,
            'forecast': {'model': 'Kronos-mini', 'direction_aligned': True},
            'market_context': {'sector': 'IT', 'peer_count': 3, 'peer_relative_pct': 1.0},
        }
        market = {'sectors': [{'sector': 'IT', 'count': 5, 'change_pct': 1.2, 'advancing_pct': 75}]}
        ok, reason = a.confirmed_gate(signal, market, activation)
        self.assertTrue(ok)
        self.assertTrue(reason.startswith('RESEARCH PREVIEW'))
        signal['market_context']['peer_relative_pct'] = -1
        self.assertFalse(a.confirmed_gate(signal, market, activation)[0])

    def test_kronos_future_close_scored_no_data_not_loss(self):
        c=self.db()
        p={'symbol':'X','side':'BUY','price':100,'stop':99,'target':102.5,
           'candle_time':'2026-11-09T10:00:00+05:30',
           'forecast':{'model':'Kronos-mini','forecast_change_pct':1,'horizon_bars':6}}
        c.execute('INSERT INTO signals VALUES(1,?,?,?)',('2026-11-09', 'X', json.dumps(p)))
        b=pd.DataFrame({'close':101.0},index=[pd.Timestamp('2026-11-09T10:30:00+05:30')])
        graded=a.resolve_kronos(c,{'X':b})
        self.assertEqual(len(graded),1)
        self.assertTrue(graded[0]['correct'])
        self.assertEqual(len(a.resolve_kronos(c,{'X':b})),0)
        c.close()

    def test_shadow_watch_no_false_outcome_without_full_session(self):
        c=self.db();now=datetime(2026,11,10,10,35,tzinfo=a.TZ)
        a.record_watch(c,[{'symbol':'X','side':'BUY','price':100}],now)
        b=pd.DataFrame({'high':[103.0],'low':[99.]},index=[pd.Timestamp('2026-11-10T10:40:00+05:30')])
        self.assertEqual(a.resolve_watch(c,{'X':b},now),[])
        c.close()

if __name__=='__main__': unittest.main()

class RepairedScannerSmokeTests(unittest.TestCase):
    def test_scan_cycle_without_market_feed_returns_cleanly(self):
        """Ensures the restored scan entry point resolves its dependencies."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        import scanner as s
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            universe=root/'universe.json'
            universe.write_text(json.dumps({'scan_symbols':[{'symbol':'ABC','avg_turnover':1e8}]}))
            with patch.object(s,'DB_PATH',root/'fresh.sqlite3'), \
                 patch.object(s,'UNIVERSE_CACHE',universe), \
                 patch.object(s,'is_entry_window',return_value=True), \
                 patch.object(s,'download_batch',return_value={'ABC':pd.DataFrame()}), \
                 patch.object(s,'persist_state_to_git',return_value=None), \
                 patch.object(s.ml,'export_reports',return_value={}), \
                 patch.object(s.aj,'record_watch',return_value=None):
                result=s.run_scan_cycle()
                self.assertEqual(result['status'],'complete')
                self.assertEqual(result['new_research_signals_logged'],0)

class ShadowDeliverySafetyTests(unittest.TestCase):
    def test_shadow_never_sent_and_approved_sent_once(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import scanner as s
        now=datetime(2026,11,10,10,10,tzinfo=a.TZ)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(s,'DB_PATH',Path(temp)/'signals.sqlite3'):
                s.init_db()
                base={'symbol':'X','side':'BUY','price':105,'stop':104,'target':108,
                      'score':90,'validation':{'approved':False,'reason':'learning'}}
                self.assertTrue(s.add_signal(base,now,shadow_only=True))
                with patch.object(s,'send_telegram_message') as fake_send:
                    self.assertEqual(s.deliver_pending_signals(),0)
                    fake_send.assert_not_called()
                good={**base,'symbol':'X','validation':{'approved':True,'reason':'qualified'},
                      'forecast':{'forecast_change_pct':0.5,'horizon_bars':6}}
                self.assertTrue(s.add_signal(good,now,shadow_only=False))
                with patch.object(s,'send_telegram_message',return_value=True) as fake_send:
                    self.assertEqual(s.deliver_pending_signals(),1)
                    self.assertEqual(s.deliver_pending_signals(),0)
                    self.assertEqual(fake_send.call_count,1)
                with s.db_connect() as c:
                    self.assertEqual(c.execute("SELECT delivered FROM signals WHERE symbol='X' AND dedupe_key LIKE '%SHADOW'").fetchone()[0],2)

    def test_research_preview_delivery_header(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import scanner as s
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(s, 'DB_PATH', Path(temp) / 'signals.sqlite3'):
                s.init_db()
                signal = {'symbol': 'FAKE', 'side': 'BUY', 'price': 100,
                          'stop': 99, 'target': 102, 'score': 90,
                          'validation': {'approved': True,
                                         'reason': 'RESEARCH PREVIEW — not validated'}}
                self.assertTrue(s.add_signal(signal, datetime(2026, 11, 10, 10, 10, tzinfo=a.TZ), shadow_only=False))
                with patch.object(s, 'send_telegram_message', return_value=True) as send:
                    self.assertEqual(s.deliver_pending_signals(), 1)
                    self.assertTrue(send.call_args.args[0].startswith('🔬 RESEARCH PREVIEW'))
                    self.assertEqual(s.deliver_pending_signals(), 0)
                    self.assertEqual(send.call_count, 1)
