# Security notes

Source package prepared for a security review (static source scan + dynamic scan).
Tests, fixtures, sample documents and agent tooling are intentionally not included.

## Exposure
- One listening port beyond loopback: the API + bundled SPA on `8022`.
- PostgreSQL, MinIO, Temporal, vLLM (OCR) and llama.cpp bind `127.0.0.1` only.
- `deploy/check-exposure.sh` lists non-loopback listeners; `deploy/harden-firewall.sh` sets the host firewall.

## Controls
- JWT bearer auth; role and account status are read from the database on every request.
- Document-scoped routes go through `get_authorized_document` (owner or ADMIN); someone else's document answers 404, exactly like a missing one. Sub-resources (summaries, translations, extractions, ...) are looked up by `(id, document_id)`.
- Login: bcrypt, constant-time user lookup, 5 failures / 15 min per client+user then HTTP 429, 8-72 character passwords.
- Response headers: CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, COOP/CORP, `Cache-Control: no-store` on API/health (`serving/security.py`). HSTS is sent when served over HTTPS.
- CORS: closed by default (same-origin SPA); `CORS_ALLOW_ORIGINS` opts origins in.
- `/docs`, `/redoc`, `/openapi.json` are disabled when `DOCUFLOW_PROD=1`; with it set, startup refuses default JWT/MinIO/DB/admin credentials.
- `/health/live` and `/health/ready` are anonymous and reveal only up/down per dependency; `/health/capacity` is ADMIN only.
- Integration webhooks: `http(s)` only, no embedded credentials, loopback / link-local / reserved targets refused at submit and again at delivery.
- OCR output is parsed with `ast.literal_eval`, never `eval`.

## Scan results (2026-11-09)
- Port scan (all 65,535 TCP ports, from outside the host firewall): only `8022` open.
- OWASP ZAP API scan, authenticated, active rules (SQLi, XSS, command injection, path traversal, SSRF, ...): 0 failures; 0 HTTP 5xx over ~8.5k requests.
- NUL bytes (`%00`) in URLs or JSON answer 400, never 500.
- Bandit: 0 high, 0 medium in the shipped code.

## Known, accepted
- HTTP only on the LAN (no TLS termination in this package); terminate TLS in front of `8022` if required.
- The SSE endpoint `/api/v2/events` takes the JWT as a query parameter (EventSource cannot set headers).
- JWTs are stateless: changing a password does not revoke tokens already issued (they expire; account status is re-checked per request).
- `python-jose` and `PyPDF2` have advisories with no fixed release.
- Development defaults for the database / MinIO credentials exist in `config/settings.py` and are rejected under `DOCUFLOW_PROD=1`.
