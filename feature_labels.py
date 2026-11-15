"""Turns a stored signal payload into plain-English feature labels the journal can score."""
from __future__ import annotations
from typing import Any, Dict, List

GROUP_NAMES = {"volume_profile": "Volume profile", "candles": "Candles", "chart_patterns": "Chart pattern",
               "trend_stack": "EMA trend stack", "flow": "OBV+VWAP flow", "levels": "Support/resistance",
               "smart_money": "FVG/Order block"}


def extra_labels(p: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    side = p.get("side")
    pro = p.get("pro") or {}
    if pro:
        conf, bad = pro.get("confirms") or {}, pro.get("conflicts") or {}
        for k, nm in GROUP_NAMES.items():
            if conf.get(k):
                labels.append(f"{nm} confirms side")
            elif bad.get(k):
                labels.append(f"{nm} against side")
        n, groups = pro.get("confluence"), pro.get("groups", 5)
        if n is not None:
            if groups >= 7:
                labels.append("Pro confluence >= 4 of 7" if n >= 4 else "Pro confluence <= 2 of 7" if n <= 2 else "Pro confluence 3 of 7")
            else:
                labels.append("Pro confluence >= 3" if n >= 3 else "Pro confluence <= 1" if n <= 1 else "Pro confluence 2")
        if pro.get("conflict_count", 0) >= 2:
            labels.append("Pro conflicts >= 2")
        # named combinations (also discovered automatically as pairs by the learner)
        an, lv = pro.get("anatomy") or {}, pro.get("levels") or {}
        if conf.get("candles") and conf.get("levels"):
            labels.append("Combo: candle + level confirm")
        if (side == "BUY" and an.get("rejection") == "LOWER_WICK" and lv.get("at_support")) or \
           (side == "SELL" and an.get("rejection") == "UPPER_WICK" and lv.get("at_resistance")):
            labels.append("Combo: wick rejection at level")
        if conf.get("smart_money") and conf.get("volume_profile"):
            labels.append("Combo: FVG/OB retest + volume profile")
        if conf.get("chart_patterns") and conf.get("trend_stack"):
            labels.append("Combo: pattern + trend stack")
        if conf.get("levels") and conf.get("smart_money"):
            labels.append("Combo: level + FVG/OB retest")
        if lv.get("at_support"):
            labels.append("Level: at support")
        if lv.get("at_resistance"):
            labels.append("Level: at resistance")
        if an.get("rejection") in ("LOWER_WICK", "UPPER_WICK"):
            labels.append("Wick: " + an["rejection"].replace("_", " ").lower() + " rejection")
        bx = an.get("body_x_avg")
        if bx is not None:
            labels.append("Body: strong (>= 1.5x avg)" if bx >= 1.5 else "Body: weak (< 0.6x avg)" if bx < 0.6 else "Body: normal")
        fvg, ob = pro.get("fvg") or {}, pro.get("order_blocks") or {}
        if fvg.get("in_bull_fvg"):
            labels.append("SmartMoney: bullish FVG retest")
        if fvg.get("in_bear_fvg"):
            labels.append("SmartMoney: bearish FVG retest")
        if ob.get("at_bull_ob"):
            labels.append("SmartMoney: bullish order block retest")
        if ob.get("at_bear_ob"):
            labels.append("SmartMoney: bearish order block retest")
        vp = pro.get("volume_profile") or {}
        if vp.get("price_vs_value"):
            labels.append("Value area: " + vp["price_vs_value"].replace("_", " ").title())
        for c in pro.get("candles") or []:
            labels.append("Candle: " + c)
        for c in pro.get("chart_patterns") or []:
            labels.append("Pattern: " + c)
        pdb = (pro.get("indicators") or {}).get("prev_day_break")
        if pdb in ("ABOVE_PDH", "BELOW_PDL"):
            labels.append("Prev-day level: " + pdb)
    news = p.get("news_events") or {}
    if news:
        s = news.get("sentiment", "NONE")
        labels.append("News: " + s.title())
        if (s == "POSITIVE") == (side == "BUY") and s in ("POSITIVE", "NEGATIVE"):
            labels.append("News confirms side")
        elif s in ("POSITIVE", "NEGATIVE"):
            labels.append("News against side")
        for t in news.get("tags") or []:
            labels.append("News event: " + t)
        if news.get("fresh_count"):
            labels.append("News: fresh (< 6h)")
        if news.get("earnings_today"):
            labels.append("Results day")
        if news.get("macro_events"):
            labels.append("Macro event day")
    # shadow rules suggested by the first 37-trade analysis (logged, never hard gates)
    try:
        labels.append("Not chased (day move <= 2.8%)" if abs(float(p.get("day_move_pct") or 0)) <= 2.8
                      else "Chased (day move > 2.8%)")
        labels.append("RVOL >= 2.2x" if float(p.get("relative_volume") or 0) >= 2.2 else "RVOL < 2.2x")
        hh = int(str(p.get("candle_time", ""))[11:13])
        if 10 <= hh < 13:
            labels.append("Midday window (10:00-13:00)")
    except (ValueError, TypeError):
        pass
    return labels
