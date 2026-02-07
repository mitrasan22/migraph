from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """
    Returns current UTC time.
    """
    return datetime.now(timezone.utc)


def iso_timestamp(dt: datetime | None = None) -> str:
    """
    Returns ISO-8601 timestamp string.
    """
    if dt is None:
        dt = utc_now()
    return dt.isoformat()
