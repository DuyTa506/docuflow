"""
Authentication & user management endpoints.

POST  /api/v2/auth/register
POST  /api/v2/auth/login
GET   /api/v2/auth/me
PATCH /api/v2/auth/me
PUT   /api/v2/auth/me/password
POST  /api/v2/auth/approve/{user_id}
GET   /api/v2/auth/users
"""
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_current_user, require_role
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
    return UserResponse(
        id=updated.id,
        username=updated.username,
        full_name=updated.full_name,
        email=updated.email,
        group=updated.group,
        role=updated.role,
        status=updated.status,
        created_at=updated.created_at.isoformat() if updated.created_at else None,
    )


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


# ── List users (ADMIN) ──────────────────────────────────────────────

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("ADMIN")),
):
    """List all users (ADMIN only)."""
    users = _auth.list_users(db)
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            full_name=u.full_name,
            email=u.email,
            group=u.group,
            role=u.role,
            status=u.status,
            created_at=u.created_at.isoformat() if u.created_at else None,
        )
        for u in users
    ]
