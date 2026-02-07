from __future__ import annotations

import re
import hashlib


def normalize_text(text: str) -> str:
    """
    Normalizes text for deterministic processing.

    Used for hashing, embeddings, and memory keys.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_text(text: str) -> str:
    """
    Stable hash of normalized text.
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
