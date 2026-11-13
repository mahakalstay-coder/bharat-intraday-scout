"""NSE trading-day check driven by an editable holiday file.

`data/nse_holidays.json` is a plain list of ISO dates. It is NOT authoritative:
verify it against NSE's official circular each year. An unreadable or missing
file never blocks trading (weekends are still excluded).
"""
from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Set

HOLIDAY_FILE = Path(__file__).resolve().parent / "data" / "nse_holidays.json"


def load_holidays(path: Optional[Path] = None) -> Set[str]:
    try:
        data = json.loads((path or HOLIDAY_FILE).read_text(encoding="utf-8"))
        return {str(d) for d in data.get("holidays", [])}
    except Exception:
        return set()


def is_trading_day(day: Optional[date] = None, holidays: Optional[Set[str]] = None) -> bool:
    day = day or datetime.now().date()
    if day.weekday() >= 5:
        return False
    return day.isoformat() not in (holidays if holidays is not None else load_holidays())
