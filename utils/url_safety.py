"""Where the server may be told to send a request.

A library system registers a webhook, so a private-network address is fine — the
library is on the LAN. What must never be reachable is the host itself: its
loopback services (database, object store, model servers) and the cloud metadata
address answer without any credentials.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


class UnsafeUrl(ValueError):
    """The URL points somewhere the server must not call."""


def _resolve(host: str) -> list[str]:
    try:
        return [info[4][0] for info in socket.getaddrinfo(host, None)]
    except socket.gaierror as exc:
        raise UnsafeUrl(f"callback host '{host}' cannot be resolved") from exc


def _is_forbidden(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    return (
        ip.is_loopback
        or ip.is_unspecified
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
    )


def validate_callback_url(url: str) -> str:
    """Return ``url`` unchanged if the server may POST to it, else raise UnsafeUrl."""
    parts = urlsplit((url or "").strip())
    if parts.scheme not in ("http", "https"):
        raise UnsafeUrl("callback URL must be http or https")
    if parts.username or parts.password:
        raise UnsafeUrl("callback URL must not embed credentials")
    host = parts.hostname
    if not host:
        raise UnsafeUrl("callback URL has no host")
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        addresses = _resolve(host)
    if any(_is_forbidden(a) for a in addresses):
        raise UnsafeUrl("callback URL points at the server itself or a reserved address")
    return url.strip()
