"""
Macro Event Calendar -- HR-010
Loads the FOMC/CPI/NFP event calendar and provides blackout-window checks.

Rule HR-010: Do not open new short premium positions within 2 trading days
before a high-impact macro event that falls within the 45-day holding window.

Usage:
    from .macro_calendar import check_macro_blackout, get_upcoming_events

    # Hard-block check (used by decision engine)
    blocked, events = check_macro_blackout(days_before=2)

    # Info display (used by dashboard)
    upcoming = get_upcoming_events(days_ahead=45)
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import yaml


_CALENDAR_PATH = Path(__file__).parent.parent / 'knowledge_base' / 'macro_calendar.yaml'

# Cached after first load
_events_cache: Optional[list] = None


def _load_events() -> list:
    """Load and cache the macro calendar YAML."""
    global _events_cache
    if _events_cache is not None:
        return _events_cache

    try:
        with open(_CALENDAR_PATH, 'r') as f:
            data = yaml.safe_load(f)
        raw = data.get('events', [])
        parsed = []
        for e in raw:
            try:
                d = date.fromisoformat(str(e['date']))
                parsed.append({
                    'date':        d,
                    'date_str':    str(e['date']),
                    'type':        e.get('type', 'UNKNOWN'),
                    'description': e.get('description', ''),
                    'impact':      e.get('impact', 'high'),
                })
            except Exception:
                continue
        _events_cache = parsed
        return _events_cache
    except Exception:
        return []


def get_upcoming_events(
    from_date: Optional[date] = None,
    days_ahead: int = 45,
    impact_filter: str = 'high',
) -> list[dict]:
    """
    Return events between from_date and from_date + days_ahead (inclusive).

    Args:
        from_date:     Start date. Defaults to today.
        days_ahead:    How many calendar days forward to look.
        impact_filter: 'high' returns only high-impact events;
                       'all' returns everything.

    Returns list of event dicts, sorted by date, with a 'days_away' key added.
    """
    from_date = from_date or date.today()
    cutoff    = from_date + timedelta(days=days_ahead)
    events    = _load_events()

    results = []
    for e in events:
        if e['date'] < from_date or e['date'] > cutoff:
            continue
        if impact_filter == 'high' and e['impact'] != 'high':
            continue
        results.append({**e, 'days_away': (e['date'] - from_date).days})

    results.sort(key=lambda x: x['date'])
    return results


def check_macro_blackout(days_before: int = 2) -> tuple[bool, list[dict]]:
    """
    Check whether today is within `days_before` calendar days of a
    high-impact macro event.

    This is the HR-010 hard-block check used by the decision engine.

    Returns:
        (is_blocked: bool, events: list[dict])
        is_blocked is True if ANY high-impact event occurs within the window.
        events contains the specific triggering event(s).
    """
    imminent = get_upcoming_events(
        from_date=date.today(),
        days_ahead=days_before,
        impact_filter='high',
    )
    return bool(imminent), imminent


def next_event_summary(days_ahead: int = 45) -> str:
    """
    Returns a short human-readable summary of the next upcoming event.
    Useful for dashboard display.

    Example: "FOMC in 8 days (2026-03-18)"
    """
    upcoming = get_upcoming_events(days_ahead=days_ahead)
    if not upcoming:
        return f"No high-impact events in the next {days_ahead} days"
    e = upcoming[0]
    days = e['days_away']
    if days == 0:
        return f"{e['type']} TODAY ({e['date_str']})"
    elif days == 1:
        return f"{e['type']} TOMORROW ({e['date_str']})"
    else:
        return f"{e['type']} in {days} days ({e['date_str']})"
