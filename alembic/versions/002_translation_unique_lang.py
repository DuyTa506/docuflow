"""Enforce one translation row per (document_id, target_language).

Revision ID: 002_translation_unique_lang
Revises: 001_storage_redesign
Create Date: 2026-07-08
"""

from typing import Sequence, Union

from alembic import op

revision: str = "002_translation_unique_lang"
down_revision: Union[str, None] = "001_storage_redesign"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop older duplicate rows per (document_id, target_language), keeping the
    # most recently created one, before the unique constraint can be added.
    op.execute(
        """
        DELETE FROM translations t
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY document_id, target_language
                       ORDER BY created_at DESC, id DESC
                   ) AS rn
            FROM translations
        ) ranked
        WHERE t.id = ranked.id AND ranked.rn > 1
        """
    )
    op.create_unique_constraint(
        "uq_translations_doc_lang", "translations", ["document_id", "target_language"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_translations_doc_lang", "translations", type_="unique")
