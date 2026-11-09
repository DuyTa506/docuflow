"""Credentials for the library-integration API.

A library system authenticates with an API key (header ``X-API-Key``); DocuFlow
keeps only its SHA-256, so a leaked database does not leak working keys.

Result callbacks are signed the way Stripe/GitHub webhooks are: the receiver
recomputes ``HMAC-SHA256(secret, "<timestamp>." + body)`` and compares it with
``X-DocuFlow-Signature: sha256=<hex>``; the timestamp (``X-DocuFlow-Timestamp``)
bounds replays.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Optional, Tuple

KEY_PREFIX = "dfk_"
SIGNATURE_TOLERANCE_S = 300


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> Tuple[str, str, str]:
    """Return (key shown once, display prefix, hash to store)."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, key[:12], hash_api_key(key)


def generate_webhook_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(32)


def sign_callback(secret: str, timestamp: int, body: bytes) -> str:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify_callback(
    secret: str, timestamp: str, body: bytes, signature: str, now: Optional[float] = None
) -> bool:
    """What a receiving library system runs on each callback."""
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs((time.time() if now is None else now) - ts) > SIGNATURE_TOLERANCE_S:
        return False
    return hmac.compare_digest(sign_callback(secret, ts, body), signature or "")
