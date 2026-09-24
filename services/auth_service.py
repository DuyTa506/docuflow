"""
Authentication & user management service.

Handles:
- User registration with password hashing
- JWT token creation / validation
- User approval / deactivation (admin flows)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config.settings import settings
from data.db_models import User
from data.id_generator import IdGenerator

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Stateless authentication helper — all state lives in the DB."""

    # ── Password utilities ──────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)

    # ── Token utilities ─────────────────────────────────────────────

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (
            expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
        )
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        """Return payload dict or ``None`` if token is invalid/expired."""
        try:
            return jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm],
            )
        except JWTError:
            return None

    # ── User CRUD ───────────────────────────────────────────────────

    def register_user(
        self,
        db: Session,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        group: str = "TEACHER",
        role: str = "MEMBER",
    ) -> User:
        """
        Create a new user.

        Registration rules:
        - TEACHER group → immediately ACTIVE (self-service, no approval needed)
        - LIBRARY group → PENDING_APPROVAL (admin must activate before login)
        - ADMIN role    → cannot be created via this endpoint (raises ValueError)

        Args:
            db: Database session
            username: Unique username (3–50 chars)
            password: Plain-text password (will be hashed)
            full_name: Display name (optional)
            email: Email address (optional, must be unique if provided)
            group: User group — "TEACHER" or "LIBRARY"
            role: Permission level — only "MEMBER" allowed for self-registration

        Returns:
            Newly created User

        Raises:
            ValueError: if username is taken or role is ADMIN
        """
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise ValueError(f"Username '{username}' already taken")

        role_upper = role.upper()
        if role_upper == "ADMIN":
            raise ValueError(
                "ADMIN accounts cannot be created via self-registration. "
                "Contact a system administrator."
            )
        if role_upper not in ("MEMBER",):
            raise ValueError(f"Invalid role '{role}'. Allowed values: MEMBER")

        group_upper = group.upper()
        if group_upper not in ("TEACHER", "LIBRARY"):
            raise ValueError(f"Invalid group '{group}'. Allowed values: TEACHER, LIBRARY")

        # TEACHER used to skip approval; on a LAN with many machines that is
        # the path that floods GPU queues. Default: everyone waits for admin.
        if settings.require_registration_approval:
            initial_status = "PENDING_APPROVAL"
        else:
            initial_status = "ACTIVE" if group_upper == "TEACHER" else "PENDING_APPROVAL"

        user_id = IdGenerator.next_id(db, "users")
        user = User(
            id=user_id,
            username=username,
            password_hash=self.hash_password(password),
            full_name=full_name,
            email=email,
            group=group_upper,
            role=role_upper,
            status=initial_status,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def authenticate(self, db: Session, username: str, password: str) -> Optional[User]:
        """Verify credentials; returns ``User`` or ``None``."""
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            return None
        if not self.verify_password(password, user.password_hash):
            return None
        return user

    def approve_user(self, db: Session, user_id: str) -> User:
        """Set user status → ACTIVE (admin operation)."""
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise ValueError("User not found")
        user.status = "ACTIVE"
        db.commit()
        db.refresh(user)
        return user

    def deactivate_user(self, db: Session, user_id: str) -> User:
        """Set user status → DEACTIVATED (admin operation)."""
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise ValueError("User not found")
        user.status = "DEACTIVATED"
        db.commit()
        db.refresh(user)
        return user

    def update_profile(
        self,
        db: Session,
        user_id: str,
        full_name: Optional[str],
        email: Optional[str],
    ) -> User:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise ValueError("User not found")
        if email is not None and email != user.email:
            conflict = db.query(User).filter(User.email == email).first()
            if conflict:
                raise ValueError(f"Email '{email}' is already in use")
            user.email = email
        if full_name is not None:
            user.full_name = full_name
        db.commit()
        db.refresh(user)
        return user

    def change_password(
        self,
        db: Session,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> User:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise ValueError("User not found")
        if not self.verify_password(current_password, user.password_hash):
            raise ValueError("Incorrect current password")
        user.password_hash = self.hash_password(new_password)
        db.commit()
        db.refresh(user)
        return user

    def list_users(self, db: Session, username: Optional[str] = None) -> list:
        """List users, optionally filtered by username (case-insensitive partial match)."""
        q = db.query(User).order_by(User.created_at.desc())
        if username and username.strip():
            q = q.filter(User.username.ilike(f"%{username.strip()}%"))
        return q.all()

    def delete_user(
        self,
        db: Session,
        user_id: str,
        *,
        requesting_user_id: Optional[str] = None,
    ) -> None:
        """Permanently delete a user and their documents (admin operation)."""
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise ValueError("User not found")
        if requesting_user_id and user_id == requesting_user_id:
            raise ValueError("Cannot delete your own account")
        if user.role == "ADMIN":
            admin_count = db.query(User).filter(User.role == "ADMIN").count()
            if admin_count <= 1:
                raise ValueError("Cannot delete the last admin account")
        from data.db_models import Document
        from services.storage_lifecycle import cleanup_document_artifacts

        doc_ids = [
            row[0] for row in db.query(Document.id).filter(Document.user_id == user_id).all()
        ]
        db.delete(user)
        db.commit()
        for doc_id in doc_ids:
            try:
                cleanup_document_artifacts(doc_id)
            except Exception:
                logger.warning("Storage cleanup failed for %s", doc_id, exc_info=True)
