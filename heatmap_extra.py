"""Research tiles. Daily metrics use completed sessions, five-minute RVOL uses the current completed slot."""
import html
import json
import logging
import numpy as np
import pandas as pd

TILES = {
    "rvol_by_sector": ("RVOL by sector", ("sector","n","median_rvol","p90_rvol")),
    "rs_vs_nifty": ("20-session relative strength vs Nifty", ("symbol","rs_20d_pct","quadrant")),
    "atr_compression": ("Daily ATR compression", ("symbol","atr14_pct","atr_pct_rank_20d","bb_width_pct")),
}


def optional(call):
    try:
        return call()
    except Exception:
        logging.getLogger(__name__).warning("Optional heatmap research unavailable", exc_info=True)
        return None


def rs(stock, benchmark):
    paired = pd.concat([stock, benchmark], axis=1, join="inner").dropna()
    if len(paired)<26 or (paired.iloc[-26:]<=0).any().any():
        return None
    ratio = paired.iloc[:,0]/paired.iloc[:,1]
    strength = ratio.pct_change(20)*100
    value, prev = strength.iloc[-1],strength.iloc[-6]
    quadrant = ("LEADING" if value>=prev else "WEAKENING") if value>=0 else ("IMPROVING" if value>=prev else "LAGGING")
    return dict(rs_20d_pct=round(value,4),quadrant=quadrant)


def compression(d):
    c = d.close
    tr = pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
    a = (100*tr.rolling(14).mean()/c).tail(20)
    if len(a)<20 or not np.isfinite(a).all() or c.tail(20).mean()<=0:
        return None
    return dict(atr14_pct=round(a.iloc[-1],4),atr_pct_rank_20d=round(100*(a<a.iloc[-1]).sum()/19,2),
                bb_width_pct=round(400*c.tail(20).std(ddof=0)/c.tail(20).mean(),4))


def daily_history(symbols, now):
    import scanner
    import yfinance as yf
    key = "v5_heatmap_daily"
    with scanner.db_connect() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
        cache = json.loads(row[0]) if row else {}
        if cache.get("day")!=str(now.date()):
            cache = {"day":str(now.date()), "frames":{}}
        frames = cache["frames"]
        tickers = {s:s+".NS" for s in symbols}
        tickers["^NSEI"] = "^NSEI"
        missing = [s for s in tickers if s not in frames]
        for start in range(0,len(missing),80):
            group = missing[start:start+80]
            raw = yf.download([tickers[s] for s in group],period="3mo",interval="1d",group_by="ticker",
                              auto_adjust=True,progress=False,timeout=8)
            for s in group:
                d = scanner.extract_ticker_frame(raw,tickers[s],single=len(group)==1)
                frames[s] = []
                if d.empty:
                    continue
                d.columns = d.columns.str.lower()
                d = d.loc[d.index.date<now.date(),["close","high","low"]].dropna().tail(45).copy()
                d.index = d.index.strftime("%Y-%m-%d")
                frames[s] = d.rename_axis("day").reset_index().to_dict("records")
        if missing:
            conn.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES(?,?)",(key,json.dumps(cache)))
    return {s:pd.DataFrame(frames[s]).set_index("day") for s in tickers if frames[s]}


def collect(intraday_all, daily_map, market, now):
    import scanner
    stocks = market.get("stocks",{})
    daily = optional(lambda: daily_history(stocks, now)) if stocks else {}
    daily = daily or {}
    out = {k:[] for k in TILES}
    sectors = {}
    for symbol, meta in stocks.items():
        d = scanner.completed_bars(scanner.normalize_intraday(intraday_all[symbol]),now)
        if not d.empty:
            v,n = scanner.same_slot_relative_volume(d,d.index[-1],5)
            if n>=5 and np.isfinite(v):
                sectors.setdefault(meta["sector"],[]).append(v)
        if symbol in daily:
            history = daily[symbol]
            if len(history)>=21:
                slope = history.close.ewm(span=20,adjust=False).mean().diff().iloc[-1]
                intraday_all[symbol].attrs["prior_daily_ema20"] = (str(history.index[-1]),float(slope))
            a = compression(history)
            if a:
                out["atr_compression"].append(dict(symbol=symbol,**a))
            if "^NSEI" in daily:
                r = rs(daily[symbol].close,daily["^NSEI"].close)
                if r:
                    out["rs_vs_nifty"].append(dict(symbol=symbol,**r))
    out["atr_compression"].sort(key=lambda r:(r["atr_pct_rank_20d"],r["atr14_pct"],r["symbol"]))
    for sector,v in sorted(sectors.items()):
        out["rvol_by_sector"].append(dict(sector=sector,n=len(v),median_rvol=round(np.median(v),3),
                                          p90_rvol=round(np.percentile(v,90),3)))
    return out


def cached_tiles(conn, asof, extras):
    """Keep the latest tiles with v4's existing persisted scanner state."""
    if extras is not None:
        conn.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES('v5_heatmap_tiles',?)",
                     (json.dumps({"asof":asof,"tiles":extras}),))
    else:
        row = conn.execute("SELECT value FROM app_state WHERE key='v5_heatmap_tiles'").fetchone()
        cached = json.loads(row[0]) if row else {}
        if cached.get("asof")==asof:
            return cached["tiles"]
    return extras


def sections(extras):
    parts = []
    for key,(title,cols) in TILES.items():
        rows = (extras or {}).get(key,[])
        table = "<table><thead><tr>"+ "".join("<th>"+c+"</th>" for c in cols)+"</tr></thead><tbody>"
        for row in rows:
            table += "<tr>"+"".join("<td>"+html.escape(str(row[c]))+"</td>" for c in cols)+"</tr>"
        parts.append("<section><h2>"+title+"</h2>"+(table+"</tbody></table>" if rows else "<p>insufficient_data</p>")+"</section>")
    return "".join(parts)
