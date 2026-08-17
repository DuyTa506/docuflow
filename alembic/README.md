# Alembic vs startup DDL

DocuFlow currently creates tables with SQLAlchemy `create_all` and then applies
the frozen `_ADDITIVE_COLUMNS` map in `data/database.py` on every boot. That is
the source of truth for existing LAN hosts.

## Rule from here on

- Do **not** add keys to `_ADDITIVE_COLUMNS`.
- New schema changes: `alembic revision -m "reason"` (autogenerate if useful),
  review the file, then `alembic upgrade head`.
- One-shot `scripts/migrate_*.py` files are historical. Do not add new ones
  for additive columns.

## Stamping a known-good database

If this host already matches the models and you want Alembic to take over:

```bash
alembic stamp head
alembic current
```

`alembic current` must equal `head` after a production deploy that uses
migrations. Keep `create_all` on startup until the first stamp so a fresh
dev database still boots without a manual upgrade.
