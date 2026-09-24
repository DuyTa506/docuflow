"""
Authentication & user management endpoints.

POST  /api/v2/auth/register
POST  /api/v2/auth/login
GET   /api/v2/auth/me
PATCH /api/v2/auth/me
PUT   /api/v2/auth/me/password
POST  /api/v2/auth/approve/{user_id}
GET   /api/v2/auth/users?q=...          — list / search by username (partial)
DELETE /api/v2/auth/users/{user_id}    — delete user (ADMIN)
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from api.dependencies import get_current_user, get_db, require_role
from api.schemas import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from data.db_models import User
from services.auth_service import AuthService

router = APIRouter(prefix="/api/v2/auth", tags=["auth"])

_auth = AuthService()


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        email=user.email,
        group=user.group,
        role=user.role,
        status=user.status,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


# ── Register ────────────────────────────────────────────────────────


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user. Librarians start in PENDING_APPROVAL status."""
    try:
        user = _auth.register_user(
            db,
            username=body.username,
            password=body.password,
            full_name=body.full_name,
            email=body.email,
            group=body.group,
            role=body.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _to_user_response(user)


# ── Login ───────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and receive a JWT access token."""
    user = _auth.authenticate(db, body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    if user.status != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status}",
        )
    token = _auth.create_access_token({"sub": user.id, "role": user.role, "group": user.group})
    return TokenResponse(access_token=token)


# ── Current user profile ────────────────────────────────────────────


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    """Return the current authenticated user profile."""
    return _to_user_response(user)


# ── Update own profile ──────────────────────────────────────────────


@router.patch("/me", response_model=UserResponse)
async def update_profile(
    body: UpdateProfileRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Update the current user's full_name and/or email."""
    try:
        updated = _auth.update_profile(db, user.id, body.full_name, body.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _to_user_response(updated)


# ── Change own password ─────────────────────────────────────────────


@router.put("/me/password", status_code=204)
async def change_password(
    body: ChangePasswordRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Change the current user's password. Requires the current password."""
    try:
        _auth.change_password(db, user.id, body.current_password, body.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return Response(status_code=204)


# ── Approve user (ADMIN) ────────────────────────────────────────────


@router.post("/approve/{user_id}", response_model=UserResponse)
async def approve_user(
    user_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("ADMIN")),
):
    """Approve a pending user (ADMIN only)."""
    try:
        user = _auth.approve_user(db, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return _to_user_response(user)


# ── List / search users (ADMIN) ───────────────────────────────────────


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    q: Optional[str] = Query(
        None,
        min_length=1,
        description="Filter by username (case-insensitive partial match)",
    ),
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("ADMIN")),
):
    """List all users, or search by username when ``q`` is provided (ADMIN only)."""
    users = _auth.list_users(db, username=q)
    return [_to_user_response(u) for u in users]


# ── Delete user (ADMIN) ─────────────────────────────────────────────


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("ADMIN")),
):
    """Permanently delete a user and their documents (ADMIN only)."""
    from data.db_models import Document
    from services.pipeline.temporal_client import terminate_document_workflows

    doc_ids = [
        row[0] for row in db.query(Document.id).filter(Document.user_id == user_id).all()
    ]
    for doc_id in doc_ids:
        await terminate_document_workflows(doc_id)
    try:
        _auth.delete_user(db, user_id, requesting_user_id=admin.id)
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    return Response(status_code=204)
