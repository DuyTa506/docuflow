#!/usr/bin/env python3
"""
Initialize the OCR document database.

Creates all necessary tables for storing documents, pages, layout elements,
and tree indices.
"""
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.database import DatabaseManager, get_db_manager
from data.db_models import Base


def main():
    parser = argparse.ArgumentParser(
        description='Initialize OCR document database'
    )
    parser.add_argument(
        '--database-url',
        type=str,
        default=None,
        help=f'Database URL (default: sqlite:///{DEFAULT_DB_PATH})'
    )
    parser.add_argument(
        '--drop-existing',
        action='store_true',
        help='Drop existing tables before creating new ones (WARNING: destroys data!)'
    )
    
    args = parser.parse_args()
    
    # Create database manager
    db_manager = DatabaseManager(args.database_url)
    
    print("=" * 60)
    print("OCR Document Database Initialization")
    print("=" * 60)
    print(f"Database URL: {db_manager.database_url}")
    print()
    
    # Drop tables if requested
    if args.drop_existing:
        confirm = input("⚠️  Drop existing tables? This will DELETE ALL DATA! (yes/no): ")
        if confirm.lower() == 'yes':
            db_manager.drop_tables()
            print()
        else:
            print("Aborted.")
            return
    
    # Create tables
    db_manager.create_tables()
    
    print()
    print("✓ Database initialized successfully!")
    print()
    print("Tables created:")
    print("  - documents")
    print("  - pages")
    print("  - layout_elements")
    print("  - tree_indices")
    print("  - tree_nodes")
    print()
    print("You can now:")
    print("  1. Start the API server: uvicorn serving.api:app --port 8001")
    print("  2. Process documents via API or CLI")
    print()


if __name__ == '__main__':
    main()
