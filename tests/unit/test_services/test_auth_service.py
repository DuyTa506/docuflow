"""
Unit tests for AuthService.update_profile and AuthService.change_password.

RED phase: written before implementation — must fail until service methods are added.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, User
from services.auth_service import AuthService


@pytest.fixture(scope="module")
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def auth():
    return AuthService()


def _seed_user(db, uid, username, email="seed@example.com"):
    user = User(
        id=uid,
        username=username,
        password_hash=AuthService.hash_password("secret123"),
        full_name="Seed User",
        email=email,
        group="TEACHER",
        role="MEMBER",
        status="ACTIVE",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class TestUpdateProfile:
    def test_updates_full_name_and_email(self, db, auth):
        user = _seed_user(db, "USR_p001", "user_p001", "p001@example.com")
        updated = auth.update_profile(db, user.id, full_name="New Name", email="new001@example.com")
        assert updated.full_name == "New Name"
        assert updated.email == "new001@example.com"

    def test_partial_update_only_full_name_leaves_email(self, db, auth):
        user = _seed_user(db, "USR_p002", "user_p002", "p002@example.com")
        updated = auth.update_profile(db, user.id, full_name="Only Name", email=None)
        assert updated.full_name == "Only Name"
        assert updated.email == "p002@example.com"

    def test_email_conflict_raises(self, db, auth):
        _seed_user(db, "USR_p003", "user_p003", "taken@example.com")
        victim = _seed_user(db, "USR_p004", "user_p004", "victim@example.com")
        with pytest.raises(ValueError, match="already in use"):
            auth.update_profile(db, victim.id, full_name=None, email="taken@example.com")

    def test_same_email_as_own_does_not_raise(self, db, auth):
        user = _seed_user(db, "USR_p005", "user_p005", "own@example.com")
        updated = auth.update_profile(db, user.id, full_name="Fine", email="own@example.com")
        assert updated.email == "own@example.com"


class TestChangePassword:
    def test_success_updates_hash(self, db, auth):
        user = _seed_user(db, "USR_c001", "user_c001", "c001@example.com")
        auth.change_password(db, user.id, "secret123", "newpass456")
        db.refresh(user)
        assert AuthService.verify_password("newpass456", user.password_hash)
        assert not AuthService.verify_password("secret123", user.password_hash)

    def test_wrong_current_password_raises(self, db, auth):
        user = _seed_user(db, "USR_c002", "user_c002", "c002@example.com")
        with pytest.raises(ValueError, match="[Ii]ncorrect"):
            auth.change_password(db, user.id, "wrongpass", "newpass456")

    def test_user_not_found_raises(self, db, auth):
        with pytest.raises(ValueError, match="[Nn]ot found"):
            auth.change_password(db, "USR_MISSING", "any", "newpass456")
