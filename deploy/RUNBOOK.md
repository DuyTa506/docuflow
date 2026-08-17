# Single-host operations (LAN)

One GPU host, one PostgreSQL, one MinIO, one Temporal. No HA. Durability comes
from backups, admission control, and workflow resume — not failover.

## URLs

- App + API (the only LAN entrypoint): `http://<host>:8022/`
- Do not publish Postgres `:5433`, MinIO `:9000/:9001`, Temporal `:7233/:8088`
  to the office Wi-Fi. Compose binds them to `127.0.0.1`.

## RPO / RTO (accepted single-host limits)

- RPO: last successful `scripts/backup.sh` (run daily; copy off-host).
- RTO: restore Postgres + MinIO, start systemd units, confirm `/health/ready`,
  then one OCR/digest smoke. Expect tens of minutes, not seconds.

## GPU / vLLM / llama.cpp die

- API stays up (`start.sh` watchdog respawns vLLM only).
- OCR/extract fails until vLLM `/health` is 200. Digest/translation use llama.cpp
  on `:5011` and do not need `:8000`.
- Docling extraction takes the `docling` GPU lease. If OCR is starved after an
  extract, check `/health/capacity` and `nvidia-smi`.

## Disk / MinIO / Postgres unavailable

- `/health/ready` returns 503. Stop submitting work.
- `GET /health/capacity` shows open Task slots.
- Orphan objects: `python scripts/reconcile_storage.py` (add `--apply` to delete).

## Deploy drain

```bash
sudo systemctl stop docuflow-temporal-worker docuflow-extraction-worker
# TimeoutStopSec=360 — workers finish or checkpoint, then SIGKILL
# restart after pulling code
sudo systemctl start docuflow-temporal-worker docuflow-extraction-worker
```

Long summarize/main-content activities resume from node/chapter checkpoints.
Do not kill -9 the worker unless it is wedged.

## Backup

```bash
bash scripts/backup.sh /mnt/nas/docuflow/$(date +%Y%m%d)
# restore (API/workers stopped):
bash scripts/restore.sh /mnt/nas/docuflow/YYYYMMDD
```

## Deferred LAN security (accepted until dated)

- HTTPS on the LAN (P1). Risk: Wi-Fi users can sniff JWTs on HTTP.
- Short-lived SSE ticket instead of JWT query string (P1).
- Login rate limit (P1).
- Owner: ops. Revisit if guest Wi-Fi or VPN is added.
