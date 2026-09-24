#!/usr/bin/env python3
"""
Initialize the DocuFlow database.

Creates all tables and seeds ID sequences.
Supports --drop-existing to start fresh.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.database import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Initialize DocuFlow database")
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help=f"Database URL (default: {settings.database_url})",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating new ones (WARNING: destroys data!)",
    )

    args = parser.parse_args()

    # Create database manager (uses DATABASE_URL from .env when not passed)
    db_manager = DatabaseManager(args.database_url or settings.database_url)

    print("=" * 60)
    print("DocuFlow Database Initialization")
    print("=" * 60)
    print(f"Database URL: {db_manager.database_url}")
    print()

    # Drop tables if requested
    if args.drop_existing:
        confirm = input("Drop existing tables? This will DELETE ALL DATA! (yes/no): ")
        if confirm.lower() == "yes":
            db_manager.drop_tables()
            print()
        else:
            print("Aborted.")
            return

    # Create tables
    db_manager.create_tables()

    # Apply Alembic migrations (dedupe + unique constraints, etc.)
    _run_alembic_upgrade()

    # Seed ID sequences
    db_manager.seed_sequences()

    # Create default admin user
    _create_default_admin(db_manager)

    # Seed the research-direction catalog
    _seed_research_directions(db_manager)

    print()
    print("Database initialized successfully!")
    print()
    print("Tables created:")
    print("  - id_sequences")
    print("  - users")
    print("  - documents")
    print("  - pages")
    print("  - layout_elements")
    print("  - digitized_texts")
    print("  - translations")
    print("  - summaries")
    print("  - main_contents")
    print("  - keywords")
    print("  - document_keywords")
    print("  - keyword_extractions")
    print("  - research_directions")
    print("  - document_research_directions")
    print("  - research_extractions")
    print("  - tree_indices")
    print("  - tree_nodes")
    print("  - tasks")
    print("  - task_eta_observations")
    print("  - task_eta_profiles")
    print()
    print("You can now:")
    print("  1. Start the API server: uvicorn serving.workflow_api:app --port 8002")
    print("  2. Process documents via API or CLI")
    print()


def _run_alembic_upgrade() -> None:
    """Bring the DB to Alembic head (idempotent for create_all hosts)."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text

    root = Path(__file__).parent.parent
    cfg = Config(str(root / "alembic.ini"))
    url = settings.database_url
    cfg.set_main_option("sqlalchemy.url", url)

    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'alembic_version'"
                )
            ).scalar()
            if row is None:
                # create_all path: tables exist, migrations never stamped.
                # Stamp just before 003 so only the new unique-constraint
                # revision runs (001/002 columns already come from create_all /
                # _ADDITIVE_COLUMNS).
                print("Alembic: stamping 002_translation_unique_lang (first run)…")
                command.stamp(cfg, "002_translation_unique_lang")
    finally:
        engine.dispose()

    print("Alembic: upgrading to head…")
    command.upgrade(cfg, "head")
    print("Alembic: at head")


def _create_default_admin(db_manager: DatabaseManager):
    """Create a default ADMIN user if none exists."""
    from data.db_models import User
    from data.id_generator import IdGenerator

    with db_manager.session() as session:
        existing = session.query(User).filter(User.role == "ADMIN").first()
        if existing:
            print(f"Admin user already exists: {existing.username}")
            return

        try:
            from passlib.context import CryptContext

            password = settings.admin_password
            prod = __import__("os").environ.get("DOCUFLOW_PROD", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if prod and password == "admin":
                raise SystemExit(
                    "ADMIN_PASSWORD is still 'admin'. Set a real password in .env "
                    "before initializing a production database."
                )
            pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
            admin_id = IdGenerator.next_id(session, "users")
            admin = User(
                id=admin_id,
                username="admin",
                password_hash=pwd_ctx.hash(password),
                full_name="System Administrator",
                group="LIBRARY",
                role="ADMIN",
                status="ACTIVE",
            )
            session.add(admin)
            session.flush()
            if password == "admin":
                print("Default admin user created (username=admin, password=admin)")
            else:
                print("Admin user created (username=admin, password from ADMIN_PASSWORD)")
        except ImportError:
            print("passlib not installed — skipping default admin creation")


def _seed_research_directions(db_manager: DatabaseManager):
    """Seed the predefined research-direction catalog from the ngành catalog.

    Without this the catalog is empty, the §3 prompt reads "(empty catalog)",
    and every direction the model returns is marked as new — which is how
    "hướng nghiên cứu" ended up unusable.

    The seed is the catalog's nhóm ngành: official names, and already filtered
    to the Academy's scope. It replaces a hand-written list of 18 group names
    that the catalog file itself recorded as unverified.
    """
    from data.db_models import ResearchDirection
    from utils.ctdt_catalog import load_catalog, research_area_names

    names = research_area_names(load_catalog())
    if not names:
        print("No research areas in the ngành catalog — skipping seed")
        return

    with db_manager.session() as session:
        existing = {r.direction_name for r in session.query(ResearchDirection.direction_name).all()}
        added = 0
        for name in names:
            if name in existing:
                continue
            session.add(ResearchDirection(direction_name=name, is_predefined=True))
            added += 1
        session.flush()

    print(f"Research area catalog: {added} added, {len(names) - added} already present")


if __name__ == "__main__":
    main()
