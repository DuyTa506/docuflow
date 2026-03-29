"""
Authentication & user management service.

Handles:
- User registration with password hashing
- JWT token creation / validation
- User approval / deactivation (admin flows)
"""
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config.settings import settings
from data.db_models import User
from data.id_generator import IdGenerator

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
            expires_delta
            or timedelta(minutes=settings.jwt_access_token_expire_minutes)
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
        role: str = "TEACHER",
    ) -> User:
        """
        Create a new user.

        Registration rules:
        - TEACHER   → immediately ACTIVE (self-service, no approval needed)
        - LIBRARIAN → PENDING_APPROVAL (admin must approve before login is allowed)
        - ADMIN     → cannot be created via this endpoint (raises ValueError)

        Args:
            db: Database session
            username: Unique username (3–50 chars)
            password: Plain-text password (will be hashed)
            full_name: Display name (optional)
            role: One of "TEACHER" or "LIBRARIAN"

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
        if role_upper not in ("TEACHER", "LIBRARIAN"):
            raise ValueError(
                f"Invalid role '{role}'. Allowed values: TEACHER, LIBRARIAN"
            )

        # TEACHER is active immediately; LIBRARIAN requires admin approval
        initial_status = "ACTIVE" if role_upper == "TEACHER" else "PENDING_APPROVAL"

        user_id = IdGenerator.next_id(db, "users")
        user = User(
            id=user_id,
            username=username,
            password_hash=self.hash_password(password),
            full_name=full_name,
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

    def list_users(self, db: Session) -> list:
        return db.query(User).order_by(User.created_at.desc()).all()
