"""Keyword-based news sentiment/event tags and a user-editable macro event calendar.

Deliberately simple and transparent: headline keyword matching, no model calls.
Tags are logged with each signal and judged later on real paper outcomes.
"""
from __future__ import annotations
import json
import re
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

EVENT_FILE = Path(__file__).resolve().parent / "data" / "event_calendar.json"

POSITIVE = ["surge", "surges", "jumps", "rallies", "soars", "upgrade", "upgrades", "beats", "record profit",
            "bags order", "wins order", "order win", "approval", "buyback", "strong results", "profit rises",
            "profit jumps", "raises target", "target raised", "outperform", "bonus issue", "special dividend"]
NEGATIVE = ["falls", "plunges", "slumps", "tumbles", "downgrade", "downgrades", "misses", "loss widens",
            "probe", "fraud", "penalty", "default", "resigns", "raid", "ban", "cuts target", "target cut",
            "weak results", "profit falls", "profit drops", "underperform", "show-cause", "insolvency"]
EVENT_TAGS = {
    "results": r"\b(q[1-4]|results|earnings|quarterly|net profit|revenue)\b",
    "corporate_action": r"\b(dividend|bonus|stock split|split|buyback|rights issue)\b",
    "mna": r"\b(acquire|acquires|acquisition|merger|takeover|stake|demerger)\b",
    "orders": r"\b(order|orders|contract|bags|wins)\b",
    "regulatory": r"\b(sebi|probe|investigation|penalty|notice|cbi|show-cause)\b",
    "rating": r"\b(upgrade|downgrade|target price|initiates coverage|brokerage)\b",
    "fundraise": r"\b(qip|ipo|fund raise|fundraise|preferential|ofs)\b",
    "management": r"\b(resigns|appoints|ceo|cfo|chairman)\b",
}


def _counts(text: str):
    pos = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in POSITIVE)
    neg = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in NEGATIVE)
    return pos, neg


def parse_rss(xml_text: str, max_items: int = 10) -> List[Dict[str, Any]]:
    """Parse RSS items into {'title','published'(aware datetime|None),'source'}."""
    out: List[Dict[str, Any]] = []
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        src = (item.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):
            title = title[: -len(" - " + src)].strip()
        pub = None
        raw = item.findtext("pubDate")
        if raw:
            try:
                pub = parsedate_to_datetime(raw)
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pub = None
        out.append({"title": title, "published": pub, "source": src})
    return out


def fetch_google_news(symbol: str, max_items: int = 8, fetch: Optional[Callable[[str], str]] = None) -> List[Dict[str, Any]]:
    """Free Google News RSS search for an NSE symbol. Best-effort; any failure returns []."""
    url = f"https://news.google.com/rss/search?q={quote_plus(symbol + ' NSE stock')}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        if fetch is None:
            import requests
            def fetch(u):
                r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                return r.text if r.status_code == 200 else ""
        text = fetch(url)
        return parse_rss(text, max_items) if text else []
    except Exception:
        return []


def classify_items(items: List[Dict[str, Any]], now: datetime) -> Dict[str, Any]:
    """Recency-weighted sentiment over de-duplicated items (<=6h x2, <=24h x1, <=48h x0.5, older ignored,
    undated x1). Same keyword method as `classify`; keeps the tags of the items it uses."""
    seen, kept = set(), []
    for it in items or []:
        title = (it.get("title") or "").strip()
        key = re.sub(r"[^a-z0-9]", "", title.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        pub, w, fresh = it.get("published"), 1.0, False
        if pub is not None:
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_h = (now - pub).total_seconds() / 3600.0
            if age_h < -1 or age_h > 48:
                continue
            w, fresh = (2.0, True) if age_h <= 6 else (1.0, False) if age_h <= 24 else (0.5, False)
        kept.append((title, w, fresh))
    if not kept:
        return {"sentiment": "NONE", "score": 0, "tags": [], "headline_count": 0, "fresh_count": 0}
    score, any_pos, tags = 0.0, False, set()
    for title, w, _ in kept:
        low = title.lower()
        pos, neg = _counts(low)
        score += w * (pos - neg)
        any_pos = any_pos or pos > 0
        tags.update(name for name, rx in EVENT_TAGS.items() if re.search(rx, low))
    sentiment = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else ("MIXED" if any_pos else "NEUTRAL")
    return {"sentiment": sentiment, "score": round(score, 2), "tags": sorted(tags),
            "headline_count": len(kept), "fresh_count": sum(1 for k in kept if k[2])}


def classify(headlines: List[str]) -> Dict[str, Any]:
    text = " | ".join(h.lower() for h in (headlines or []) if h)
    if not text:
        return {"sentiment": "NONE", "score": 0, "tags": [], "headline_count": 0}
    pos = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in POSITIVE)
    neg = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in NEGATIVE)
    sentiment = "POSITIVE" if pos > neg else "NEGATIVE" if neg > pos else ("MIXED" if pos else "NEUTRAL")
    tags = [name for name, rx in EVENT_TAGS.items() if re.search(rx, text)]
    return {"sentiment": sentiment, "score": pos - neg, "tags": tags, "headline_count": len(headlines)}


def calendar_events(day: date, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Macro events (RBI policy, budget, expiry, ...) the user lists in data/event_calendar.json."""
    try:
        data = json.loads((path or EVENT_FILE).read_text(encoding="utf-8"))
        return [e for e in data.get("events", []) if str(e.get("date")) == day.isoformat()]
    except Exception:
        return []


_EARN_CACHE: Dict[str, Optional[bool]] = {}


def earnings_today(symbol: str, day: date, fetch: Optional[Callable[[str], Any]] = None) -> Optional[bool]:
    """True/False if a results date is known for `symbol`, None if unknown. Best-effort, cached."""
    key = f"{day.isoformat()}|{symbol}"
    if key in _EARN_CACHE:
        return _EARN_CACHE[key]
    result: Optional[bool] = None
    try:
        if fetch is None:
            import yfinance as yf
            fetch = lambda s: yf.Ticker(s + ".NS").calendar
        cal = fetch(symbol)
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            result = any(getattr(d, "isoformat", lambda: str(d))()[:10] == day.isoformat() for d in dates)
    except Exception:
        result = None
    _EARN_CACHE[key] = result
    return result


def confirms_side(sentiment: str, side: str) -> Optional[bool]:
    if sentiment in ("POSITIVE", "NEGATIVE"):
        return (sentiment == "POSITIVE") == (side == "BUY")
    return None
