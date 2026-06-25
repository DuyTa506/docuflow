"""
API Dependencies — Dependency injection for FastAPI.

Provides reusable dependencies for:
- Database sessions
- JWT authentication
- Role-based access control
- LLM client factory
"""
from typing import Generator, Callable, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from data.database import get_db_manager
from data.db_models import Document, Task, User
from config.settings import settings

# OAuth2 scheme — token URL matches the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v2/auth/login", auto_error=False)


# ── Database dependency ─────────────────────────────────────────────

def get_db() -> Generator:
    """Dependency for database session."""
    db_manager = get_db_manager()
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Authentication dependencies ─────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Decode JWT and return the authenticated User.

    Raises 401 if token is missing / invalid / user not found.
    """
    from services.auth_service import AuthService

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth = AuthService()
    payload = auth.decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if user.status != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status}",
        )

    return user


def get_authorized_document(document_id: str, user: User, db: Session) -> Document:
    """Return document if it exists and the user may access it (owner or ADMIN)."""
    from data.repositories import DocumentRepository

    doc = DocumentRepository(db).get(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if user.role != "ADMIN" and doc.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return doc


def sanitize_task_payload(payload: dict, user: User) -> dict:
    """Hide verbose tracebacks from non-admin task consumers."""
    if user.role == "ADMIN" or not payload.get("error"):
        return payload
    err = payload["error"]
    first_line = err.splitlines()[0] if err else ""
    sanitized = dict(payload)
    sanitized["error"] = first_line[:500] if first_line else "Task failed"
    return sanitized


def list_authorized_tasks(
    db: Session,
    user: User,
    document_id: Optional[str] = None,
) -> list:
    """List tasks visible to the user."""
    query = db.query(Task)
    if document_id:
        get_authorized_document(document_id, user, db)
        query = query.filter(Task.document_id == document_id)
    elif user.role != "ADMIN":
        query = query.join(Document, Task.document_id == Document.id).filter(
            Document.user_id == user.id
        )
    tasks = query.order_by(Task.created_at.desc()).all()
    return [
        {
            "task_id": t.id,
            "document_id": t.document_id,
            "task_type": t.task_type,
            "status": t.status,
            "progress": t.progress,
            "message": t.message,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }
        for t in tasks
    ]


def require_role(*roles: str) -> Callable:
    """
    Returns a dependency that ensures the current user has one of the
    specified roles.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(require_role("ADMIN"))])
        async def admin_endpoint(): ...
    """
    async def _check_role(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' not authorized. Required: {', '.join(roles)}",
            )
        return user

    return _check_role


# ── LLM client dependency ───────────────────────────────────────────

_llm_cache: dict = {}


def get_llm_client():
    """
    Return a cached LLM client for AI services (translation, summarisation, etc.).

    The client is cached by (provider, model, ollama_base_url) so repeated calls
    within the same process never re-initialise the underlying HTTP session.
    """
    from core.pageindex.llm.client_factory import LLMClientFactory

    key = (settings.ai_provider, settings.ai_model, settings.ai_ollama_base_url, settings.ai_openai_base_url)
    if key not in _llm_cache:
        _llm_cache[key] = LLMClientFactory.create_client(
            provider=settings.ai_provider,
            model=settings.ai_model,
            ollama_base_url=settings.ai_ollama_base_url,
            openai_base_url=settings.ai_openai_base_url,
        )
    return _llm_cache[key]
