"""Transport-level protections that sit in front of every route.

``SecurityHeadersMiddleware`` stamps the headers a scanner looks for;
``LoginLimiter`` slows password guessing. Both are process-local: the API runs as
one uvicorn process, and a restart clearing the counters only helps a patient
attacker by minutes.
"""

from __future__ import annotations

import threading
import time

# The Angular build ships its own scripts and inline component styles, and loads
# nothing from another origin; the API docs page (dev only) pulls Swagger UI from
# a CDN, so it is left out of the policy rather than breaking it.
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    # The Angular build defers its stylesheet with a stock inline handler
    # (<link media="print" onload="this.media='all'">); without this hash the page
    # loads unstyled. A hash allows exactly that string, not inline script in general.
    "script-src-attr 'unsafe-hashes' 'sha256-MhtPZXr7+LpJUY5qtMutB+qWfQtMaPccfe7QXtCcEYc='; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)
_CSP_EXEMPT = ("/docs", "/redoc")
_HSTS = "max-age=31536000; includeSubDomains"


class SecurityHeadersMiddleware:
    """Pure ASGI, so streaming responses (SSE, file downloads) are not buffered."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        https = scope.get("scheme") == "https"

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                present = {name.lower() for name, _ in headers}

                def add(name: bytes, value: str) -> None:
                    if name not in present:
                        headers.append((name, value.encode("latin-1")))

                add(b"x-content-type-options", "nosniff")
                add(b"x-frame-options", "DENY")
                add(b"referrer-policy", "no-referrer")
                add(b"permissions-policy", "camera=(), microphone=(), geolocation=()")
                add(b"cross-origin-opener-policy", "same-origin")
                add(b"cross-origin-resource-policy", "same-origin")
                add(b"cross-origin-embedder-policy", "require-corp")
                if not path.startswith(_CSP_EXEMPT):
                    add(b"content-security-policy", _CSP)
                if path.startswith(("/api/", "/health")) or path == "/api":
                    add(b"cache-control", "no-store")
                if https:
                    add(b"strict-transport-security", _HSTS)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class NulByteGuard:
    """Refuse a NUL byte in the path or query with a 400.

    PostgreSQL cannot store one, so left alone it becomes a 500 deep in a query —
    which a scanner reports as a server error and a log reader as an incident.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            raw = scope.get("raw_path") or scope.get("path", "").encode()
            query = scope.get("query_string", b"")
            if b"\x00" in raw or b"%00" in raw.lower() or b"\x00" in query or b"%00" in query.lower():
                body = b'{"detail":"Invalid character in request"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 400,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


class LoginLimiter:
    """Locks a (client, username) pair — and a client overall — after repeated failures."""

    def __init__(self, max_failures: int = 5, window_seconds: int = 900, per_client: int = 30):
        self.max_failures = max_failures
        self.per_client = per_client
        self.window = window_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _keys(client: str, username: str) -> tuple[str, str]:
        return f"{client}|{username.strip().lower()}", f"{client}|*"

    def _recent(self, key: str, now: float) -> list[float]:
        recent = [t for t in self._failures.get(key, []) if now - t < self.window]
        if recent:
            self._failures[key] = recent
        else:
            self._failures.pop(key, None)
        return recent

    def retry_after(self, client: str, username: str) -> int:
        """Seconds until another attempt is allowed; 0 when it is allowed now."""
        now = time.monotonic()
        pair, overall = self._keys(client, username)
        with self._lock:
            waits = []
            for key, limit in ((pair, self.max_failures), (overall, self.per_client)):
                recent = self._recent(key, now)
                if len(recent) >= limit:
                    waits.append(int(self.window - (now - recent[0])) + 1)
            return max(waits, default=0)

    def record_failure(self, client: str, username: str) -> None:
        now = time.monotonic()
        with self._lock:
            for key in self._keys(client, username):
                self._failures.setdefault(key, []).append(now)

    def record_success(self, client: str, username: str) -> None:
        with self._lock:
            self._failures.pop(self._keys(client, username)[0], None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


login_limiter = LoginLimiter()
