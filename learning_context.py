"""Additive, timestamped market/sector evidence. Never gates existing strategies."""
from __future__ import annotations
import io
import json
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from pathlib import Path
from statistics import median
import pandas as pd
import requests

LOG = logging.getLogger(__name__)
# Official constituent files; quote symbols are provider identifiers, not guarantees of coverage.
SECTORS = {
    'Nifty Bank': ('ind_niftybanklist.csv', '^NSEBANK'),
    'Nifty IT': ('ind_niftyitlist.csv', '^CNXIT'),
    'Nifty Auto': ('ind_niftyautolist.csv', '^CNXAUTO'),
    'Nifty Pharma': ('ind_niftypharmalist.csv', '^CNXPHARMA'),
    'Nifty Metal': ('ind_niftymetallist.csv', '^CNXMETAL'),
    'Nifty FMCG': ('ind_niftyfmcglist.csv', '^CNXFMCG'),
    'Nifty Realty': ('ind_niftyrealtylist.csv', '^CNXREALTY'),
    'Nifty Financial Services': ('ind_niftyfinancelist.csv', '^CNXFINANCE'),
    'Nifty PSU Bank': ('ind_niftypsubanklist.csv', '^CNXPSUBANK'),
    'Nifty Private Bank': ('ind_nifty_privatebanklist.csv', '^NIFTYPVTBANK'),
    'Nifty Oil & Gas': ('ind_niftyoilgaslist.csv', '^NIFTYOILGAS'),
    'Nifty Healthcare': ('ind_niftyhealthcarelist.csv', '^NIFTYHEALTHCARE'),
}


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.href=None; self.parts=[]
    def handle_starttag(self, tag, attrs):
        if tag=='a': self.href=dict(attrs).get('href'); self.parts=[]
    def handle_data(self, data):
        if self.href: self.parts.append(data)
    def handle_endtag(self, tag):
        if tag=='a' and self.href:
            self.links.append((self.href,' '.join(' '.join(self.parts).split())))
            self.href=None


def _get_text(url):
    response=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=8)
    response.raise_for_status();return response.text


def refresh_memberships(root, now, fetch=None, batch=3):
    """Bounded weekly refresh; retain last good mapping, explicitly age it at use."""
    path = Path(root) / 'data/sector_memberships.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    specs=dict(SECTORS)
    catalog=cache.get('_catalog',{})
    # Discover new official sector pages as NSE expands its index family.
    # Unsupported quote identifiers remain UNKNOWN; never invent an index return.
    try: catalog_age=(pd.Timestamp(now)-pd.Timestamp(catalog['checked_at'])).total_seconds()/86400
    except (KeyError,ValueError):catalog_age=999
    if fetch is None and catalog_age>=7:
        try:
            page='https://www.niftyindices.com/indices/equity/sectoral-indices'
            parser=_Links();parser.feed(_get_text(page));pages={}
            for href,name in parser.links:
                url=urljoin(page,href)
                if urlparse(url).netloc=='www.niftyindices.com' and '/indices/equity/sectoral-indices/' in urlparse(url).path and name:
                    pages[name]=url
            if pages:catalog=dict(checked_at=now.isoformat(),pages=pages)
        except Exception as exc:
            LOG.warning('Official sector catalog unavailable: %s',type(exc).__name__)
    cache['_catalog']=catalog
    known={' '.join(n.casefold().split()) for n in specs}
    for name,page in catalog.get('pages',{}).items():
        if ' '.join(name.casefold().split()) not in known:specs[name]=(page,None)
    due = []
    for name in specs:
        rec = cache.get(name, {})
        try: age = (pd.Timestamp(now) - pd.Timestamp(rec['asof'])).total_seconds()/86400
        except (KeyError, ValueError): age = 999
        if age >= 7: due.append(name)
    # Oldest first means repeated failed endpoints cannot starve all other indices.
    due.sort(key=lambda name: cache.get(name, {}).get('attempted_at', ''))
    for name in due[:batch]:
        csv_name, quote = specs[name]
        url = csv_name if csv_name.startswith('https://') else 'https://www.niftyindices.com/IndexConstituent/' + csv_name
        try:
            if not url.lower().endswith('.csv'):
                parser=_Links();parser.feed((fetch or _get_text)(url))
                links=[urljoin(url,href) for href,_ in parser.links if re.search(r'IndexConstituent/[^?]+\.csv',href,re.I)]
                links=[link for link in links if urlparse(link).netloc=='www.niftyindices.com']
                if not links:raise ValueError('Official sector page has no constituent CSV')
                url=links[0]
            if fetch is None:
                body = _get_text(url)
            else: body = fetch(url)
            table = pd.read_csv(io.StringIO(body))
            table.columns = [str(col).strip() for col in table.columns]
            if 'Symbol' not in table: raise ValueError('No Symbol column in constituent CSV')
            members = []
            for _, row in table.iterrows():
                symbol = str(row['Symbol']).strip()
                if symbol and symbol != 'nan':
                    industry = row.get('Industry', row.get('Basic Industry', 'Unknown'))
                    members.append({'symbol':symbol, 'industry':str(industry).strip()})
            if len(members) < 3: raise ValueError('Incomplete constituent CSV')
            cache[name] = dict(asof=now.isoformat(), source=url, quote_symbol=quote, members=members)
        except Exception as exc:
            LOG.warning('Sector membership %s unavailable: %s', name, type(exc).__name__)
            cache.setdefault(name, {})['last_error'] = type(exc).__name__
        cache.setdefault(name, {})['attempted_at'] = now.isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(cache, indent=2)); tmp.replace(path)
    return cache


def completed(frame, now):
    if frame is None or frame.empty: return pd.DataFrame()
    d = frame.copy()
    d.columns = [str(c).lower() for c in d.columns]
    idx = pd.to_datetime(d.index)
    d.index = idx.tz_localize('Asia/Kolkata') if idx.tz is None else idx.tz_convert('Asia/Kolkata')
    d = d[~d.index.duplicated()].sort_index()
    return d[d.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]


def measure(frame, now):
    d = completed(frame, now)
    if d.empty or 'close' not in d: return {'status':'UNKNOWN'}
    today = d[d.index.date == now.date()].between_time('09:15','15:25')
    prior = d[d.index.date < now.date()]
    if today.empty or (pd.Timestamp(now)-today.index[-1]).total_seconds()>900:
        return {'status':'UNKNOWN'}
    closes = today['close'].astype(float)
    if not pd.notna(closes.iloc[-1]) or closes.iloc[-1] <= 0: return {'status':'UNKNOWN'}
    price = float(closes.iloc[-1])
    prev = float(prior['close'].iloc[-1]) if not prior.empty else None
    last15 = today.tail(3)
    move15 = (price/float(last15['open'].iloc[0])-1)*100 if 'open' in today else None
    vol = today['volume'].astype(float).clip(lower=0) if 'volume' in today else pd.Series(dtype=float)
    vwap = None
    if len(vol) and vol.sum()>0:
        typ = (today['high']+today['low']+today['close'])/3
        vwap = float((typ*vol).sum()/vol.sum())
    slope = (price/float(closes.iloc[-4])-1)*100 if len(closes)>=4 else None
    if vwap is None or slope is None or len(last15)<3:
        trend = 'UNKNOWN'
    elif price>vwap and slope>0.05 and move15>0: trend='UP'
    elif price<vwap and slope<-0.05 and move15<0: trend='DOWN'
    else: trend='SIDEWAYS'
    background = 'UNKNOWN'
    if len(prior)>=30:
        daily = prior['close'].resample('1D').last().dropna()
        if len(daily)>=5:
            avg = float(daily.tail(5).mean())
            background = 'UP' if daily.iloc[-1]>avg else 'DOWN' if daily.iloc[-1]<avg else 'SIDEWAYS'
    return dict(status='ok', asof=today.index[-1].isoformat(), price=price,
                change_pct=round((price/prev-1)*100,4) if prev and prev>0 else None,
                change_15m_pct=move15, vwap=vwap, intraday_raw=trend, background=background)


def snapshot(root, conn, stock_frames, market, now, fetch_benchmarks):
    path = Path(root)/'data/sector_memberships.json'
    memberships = json.loads(path.read_text()) if path.exists() else {}
    # Only reference data acquired by this point in time can be used.
    memberships = {n:r for n,r in memberships.items() if r.get('asof') and
                   0 <= (pd.Timestamp(now)-pd.Timestamp(r['asof'])).total_seconds() <= 30*86400}
    symbols = ['^NSEI', '^INDIAVIX'] + list(dict.fromkeys(r['quote_symbol'] for r in memberships.values() if r.get('quote_symbol')))
    cached = conn.execute("SELECT value FROM app_state WHERE key='four_context_cache'").fetchone()
    saved = json.loads(cached[0]) if cached else {}
    if saved.get('at') and 0 <= (pd.Timestamp(now)-pd.Timestamp(saved['at'])).total_seconds()<300:
        benchmarks = saved['benchmarks']
    else:
        frames = fetch_benchmarks(symbols)
        benchmarks = {symbol:measure(frames.get(symbol),now) for symbol in symbols}
        saved = {'at':now.isoformat(),'benchmarks':benchmarks}
        conn.execute("INSERT OR REPLACE INTO app_state VALUES('four_context_cache',?)",(json.dumps(saved),))
    nifty = benchmarks.get('^NSEI', {'status':'UNKNOWN'})
    vix = benchmarks.get('^INDIAVIX', {}).get('price')
    vix_bucket = 'UNKNOWN' if vix is None else 'LOW' if vix<13 else 'HIGH' if vix>18 else 'NORMAL'
    raw = nifty.get('intraday_raw','UNKNOWN')
    oldrow = conn.execute("SELECT value FROM app_state WHERE key='four_regime_state'").fetchone()
    old = json.loads(oldrow[0]) if oldrow else {}
    slot = nifty.get('asof')
    if old.get('day') != now.date().isoformat(): old = {}
    count = old.get('count',0)
    if slot != old.get('slot'): count = count+1 if raw==old.get('candidate') else 1
    stable = old.get('stable','UNKNOWN')
    if raw=='UNKNOWN': stable='UNKNOWN'
    elif count>=2: stable=raw
    state = dict(day=now.date().isoformat(),slot=slot,candidate=raw,count=count,stable=stable)
    conn.execute("INSERT OR REPLACE INTO app_state VALUES('four_regime_state',?)",(json.dumps(state),))
    stocks = market.get('stocks',{})
    measured = {s:measure(frame,now) for s,frame in stock_frames.items()}
    sector_views={}; peers={}
    for name, rec in memberships.items():
        all_members = [m['symbol'] for m in rec['members']]
        available = [s for s in all_members if measured.get(s,{}).get('status')=='ok']
        index = benchmarks.get(rec['quote_symbol'],{'status':'UNKNOWN'})
        above = [measured[s]['price']>measured[s]['vwap'] for s in available if measured[s].get('vwap')]
        sector_views[name]=dict(index=index, total_members=len(all_members), sampled_members=len(available),
                               above_vwap_pct=100*sum(above)/len(above) if above else None,
                               scope='sampled members; not full index breadth', source=rec['source'])
    for symbol, stock in stocks.items():
        own=measured.get(symbol,{})
        # Prefer industry-matched peers; membership alone does not make stocks industry peers.
        industry=stock.get('industry','Unknown')
        group=[s for s,v in stocks.items() if s!=symbol and industry!='Unknown' and v.get('industry')==industry
               and measured.get(s,{}).get('status')=='ok' and measured[s].get('change_pct') is not None
               and measured[s].get('asof')==own.get('asof')]
        returns=[measured[s]['change_pct'] for s in group]
        member_of=[n for n,r in memberships.items() if any(m['symbol']==symbol for m in r['members'])]
        # Broader liquid-universe stocks can use industry mappings even when not index constituents.
        if not member_of and industry!='Unknown':
            member_of=[n for n,r in memberships.items() if any(stocks.get(m['symbol'],{}).get('industry')==industry for m in r['members'])]
        peers[symbol]=dict(official_sector_indices=member_of, industry=industry, peer_count=len(group),
                           peer_symbols=group, peer_median_pct=median(returns) if returns else None,
                           stock_vs_peers_pp=own.get('change_pct')-median(returns) if returns and own.get('change_pct') is not None else None,
                           stock_asof=own.get('asof'), sector_views={n:sector_views[n] for n in member_of})
        peers[symbol]['relative_to_indices'] = {}
        for name in member_of:
            index = sector_views[name]['index']
            aligned = own.get('asof') == index.get('asof') and own.get('change_pct') is not None and index.get('change_pct') is not None
            nifty_aligned = index.get('asof') == nifty.get('asof') and nifty.get('change_pct') is not None and index.get('change_pct') is not None
            peers[symbol]['relative_to_indices'][name] = dict(
                stock_vs_sector_pp=own['change_pct']-index['change_pct'] if aligned else None,
                sector_vs_nifty_pp=index['change_pct']-nifty['change_pct'] if nifty_aligned else None,
                aligned_timestamps=aligned)
    conn.commit()
    broad=[v for v in measured.values() if v.get('status')=='ok' and v.get('change_pct') is not None]
    return dict(asof=now.isoformat(), nifty=nifty, intraday_regime=stable,
                india_vix=vix, vix_bucket=vix_bucket,
                sampled_breadth=dict(count=len(broad),advancing_pct=100*sum(v['change_pct']>0 for v in broad)/len(broad) if broad else None,
                                     scope='sampled liquid universe, not all NSE'),
                regime_pending=raw if raw!=stable else None, sectors=sector_views, stocks=peers,
                advisory_only=True)
