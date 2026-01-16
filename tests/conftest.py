"""
Pytest configuration and global fixtures.
"""
import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db_models import Base


@pytest.fixture(scope="session")
def test_db_engine():
    """Create in-memory SQLite engine for tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def test_db_session(test_db_engine):
    """Create fresh database session for each test."""
    Session = sessionmaker(bind=test_db_engine)
    session = Session()
    
    yield session
    
    # Rollback any uncommitted changes and close
    session.rollback()
    session.close()


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample test image."""
    from PIL import Image
    
    img_path = temp_dir / "test_image.png"
    img = Image.new('RGB', (800, 600), color='white')
    img.save(img_path)
    
    return str(img_path)


@pytest.fixture
def sample_base64_image():
    """Provide base64 encoded sample image."""
    import base64
    from io import BytesIO
    from PIL import Image
    
    img = Image.new('RGB', (100, 100), color='blue')
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()
