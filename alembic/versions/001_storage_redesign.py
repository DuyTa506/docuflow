"""Storage redesign additive columns.

Revision ID: 001_storage_redesign
Revises:
Create Date: 2026-06-26
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "001_storage_redesign"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("image_key", sa.String(), nullable=True))
    op.add_column("layout_elements", sa.Column("crop_image_key", sa.String(), nullable=True))
    op.add_column("digitized_texts", sa.Column("ocr_content_key", sa.String(), nullable=True))
    op.add_column(
        "digitized_texts", sa.Column("normalized_content_key", sa.String(), nullable=True)
    )
    op.add_column("translations", sa.Column("translated_content_key", sa.String(), nullable=True))
    op.add_column("translations", sa.Column("translated_elements_key", sa.String(), nullable=True))
    op.add_column("tree_indices", sa.Column("tree_data_key", sa.String(), nullable=True))
    op.alter_column("tree_indices", "tree_data", existing_type=sa.JSON(), nullable=True)


def downgrade() -> None:
    op.alter_column("tree_indices", "tree_data", existing_type=sa.JSON(), nullable=False)
    op.drop_column("tree_indices", "tree_data_key")
    op.drop_column("translations", "translated_elements_key")
    op.drop_column("translations", "translated_content_key")
    op.drop_column("digitized_texts", "normalized_content_key")
    op.drop_column("digitized_texts", "ocr_content_key")
    op.drop_column("layout_elements", "crop_image_key")
    op.drop_column("pages", "image_key")
