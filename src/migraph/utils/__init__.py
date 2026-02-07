"""
Utility functions for migraph.

This module contains low-level helpers used across the system.
No domain logic should live here.
"""

from migraph.utils.text import normalize_text, hash_text
from migraph.utils.time import utc_now, iso_timestamp
from migraph.utils.helpers import safe_mean, safe_std, clamp

__all__ = [
    "normalize_text",
    "hash_text",
    "utc_now",
    "iso_timestamp",
    "safe_mean",
    "safe_std",
    "clamp",
]
