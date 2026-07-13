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

    # Seed ID sequences
    db_manager.seed_sequences()

    # Create default admin user
    _create_default_admin(db_manager)

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
    print()
    print("You can now:")
    print("  1. Start the API server: uvicorn serving.workflow_api:app --port 8002")
    print("  2. Process documents via API or CLI")
    print()


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

            pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
            admin_id = IdGenerator.next_id(session, "users")
            admin = User(
                id=admin_id,
                username="admin",
                password_hash=pwd_ctx.hash("admin"),
                full_name="System Administrator",
                group="LIBRARY",
                role="ADMIN",
                status="ACTIVE",
            )
            session.add(admin)
            session.flush()
            print(f"Default admin user created (username=admin, password=admin)")
        except ImportError:
            print("passlib not installed — skipping default admin creation")


if __name__ == "__main__":
    main()
