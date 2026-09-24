from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, User
from data.id_generator import IdGenerator
from services.auth_service import AuthService


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(session)
    session.commit()
    return session


def _patch_hash(monkeypatch):
    monkeypatch.setattr(AuthService, "hash_password", staticmethod(lambda p: f"hashed:{p}"))


def test_teacher_pending_when_approval_required(monkeypatch):
    from services import auth_service as auth_mod

    _patch_hash(monkeypatch)
    monkeypatch.setattr(auth_mod.settings, "require_registration_approval", True)
    db = _db()
    user = AuthService().register_user(db, "teacher1", "secret123", group="TEACHER")
    assert user.status == "PENDING_APPROVAL"


def test_teacher_active_when_approval_disabled(monkeypatch):
    from services import auth_service as auth_mod

    _patch_hash(monkeypatch)
    monkeypatch.setattr(auth_mod.settings, "require_registration_approval", False)
    db = _db()
    user = AuthService().register_user(db, "teacher2", "secret123", group="TEACHER")
    assert user.status == "ACTIVE"


def test_library_always_pending(monkeypatch):
    from services import auth_service as auth_mod

    _patch_hash(monkeypatch)
    monkeypatch.setattr(auth_mod.settings, "require_registration_approval", False)
    db = _db()
    user = AuthService().register_user(db, "lib1", "secret123", group="LIBRARY")
    assert user.status == "PENDING_APPROVAL"
