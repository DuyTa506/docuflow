"""Unique pages per document+page_number and one DigitizedText per document.

Revision ID: 003_extraction_uniques
Revises: 002_translation_unique_lang
Create Date: 2026-08-24
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003_extraction_uniques"
down_revision: Union[str, None] = "002_translation_unique_lang"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_names(inspector, table: str) -> set[str]:
    return {ix["name"] for ix in inspector.get_indexes(table) if ix.get("name")}


def _unique_names(inspector, table: str) -> set[str]:
    return {uc["name"] for uc in inspector.get_unique_constraints(table) if uc.get("name")}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Layout elements reference pages without ON DELETE CASCADE — drop orphans
    # belonging to duplicate pages first, then the duplicate page rows.
    op.execute(
        """
        DELETE FROM layout_elements le
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY document_id, page_number
                       ORDER BY created_at DESC, id DESC
                   ) AS rn
            FROM pages
        ) ranked
        WHERE le.page_id = ranked.id AND ranked.rn > 1
        """
    )
    op.execute(
        """
        DELETE FROM pages p
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY document_id, page_number
                       ORDER BY created_at DESC, id DESC
                   ) AS rn
            FROM pages
        ) ranked
        WHERE p.id = ranked.id AND ranked.rn > 1
        """
    )
    op.execute(
        """
        DELETE FROM digitized_texts dt
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY document_id
                       ORDER BY created_at DESC, id DESC
                   ) AS rn
            FROM digitized_texts
        ) ranked
        WHERE dt.id = ranked.id AND ranked.rn > 1
        """
    )

    # Replace the non-unique (document_id, page_number) index with a unique constraint.
    # Idempotent: create_all on a fresh host may already have the unique, or may
    # never have created the legacy index.
    inspector = sa.inspect(bind)
    if "ix_pages_document_page_number" in _index_names(inspector, "pages"):
        op.drop_index("ix_pages_document_page_number", table_name="pages")
    if "uq_pages_document_page_number" not in _unique_names(inspector, "pages"):
        op.create_unique_constraint(
            "uq_pages_document_page_number", "pages", ["document_id", "page_number"]
        )

    inspector = sa.inspect(bind)
    if "uq_digitized_texts_document_id" not in _unique_names(inspector, "digitized_texts"):
        op.create_unique_constraint(
            "uq_digitized_texts_document_id", "digitized_texts", ["document_id"]
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "uq_digitized_texts_document_id" in _unique_names(inspector, "digitized_texts"):
        op.drop_constraint(
            "uq_digitized_texts_document_id", "digitized_texts", type_="unique"
        )
    if "uq_pages_document_page_number" in _unique_names(inspector, "pages"):
        op.drop_constraint("uq_pages_document_page_number", "pages", type_="unique")
    if "ix_pages_document_page_number" not in _index_names(inspector, "pages"):
        op.create_index(
            "ix_pages_document_page_number", "pages", ["document_id", "page_number"]
        )
