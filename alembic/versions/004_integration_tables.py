"""Library-system integration: API clients and their jobs.

Revision ID: 004_integration_tables
Revises: 003_extraction_uniques
Create Date: 2026-09-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004_integration_tables"
down_revision: Union[str, None] = "003_extraction_uniques"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent: create_all on API/worker startup may have created them first.
    tables = set(sa.inspect(op.get_bind()).get_table_names())

    if "integration_clients" not in tables:
        op.create_table(
            "integration_clients",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("api_key_prefix", sa.String(), nullable=False),
            sa.Column("api_key_hash", sa.String(), nullable=False),
            sa.Column("webhook_secret", sa.String(), nullable=False),
            sa.Column("callback_url", sa.String(), nullable=True),
            sa.Column("service_user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("last_used_at", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_integration_clients_api_key_hash",
            "integration_clients",
            ["api_key_hash"],
            unique=True,
        )

    if "integration_jobs" not in tables:
        op.create_table(
            "integration_jobs",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column(
                "client_id", sa.String(), sa.ForeignKey("integration_clients.id"), nullable=False
            ),
            sa.Column("external_id", sa.String(), nullable=True),
            sa.Column("document_id", sa.String(), sa.ForeignKey("documents.id"), nullable=True),
            sa.Column("tasks", sa.JSON(), nullable=False),
            sa.Column("stages", sa.JSON(), nullable=True),
            sa.Column("translation_id", sa.String(), nullable=True),
            sa.Column("callback_url", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("delivery", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_integration_jobs_client_external",
            "integration_jobs",
            ["client_id", "external_id"],
        )

    op.execute(
        """
        INSERT INTO id_sequences (table_name, prefix, current_value)
        SELECT v.table_name, v.prefix, 0
        FROM (VALUES ('integration_clients', 'CLI'), ('integration_jobs', 'JOB'))
             AS v(table_name, prefix)
        WHERE NOT EXISTS (
            SELECT 1 FROM id_sequences s WHERE s.table_name = v.table_name
        )
        """
    )


def downgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "integration_jobs" in tables:
        op.drop_table("integration_jobs")
    if "integration_clients" in tables:
        op.drop_table("integration_clients")
    op.execute(
        "DELETE FROM id_sequences WHERE table_name IN ('integration_clients', 'integration_jobs')"
    )
