"""
SentinXFL - Authentication API Routes
=======================================

Simple JWT-based authentication for dual dashboard system.
- Client Bank users: login, view reports, upload data
- SentinXFL Employees: login, global dashboard, management

Author: Anshuman Bakshi
"""

import bcrypt as _bcrypt
import secrets
import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from sentinxfl.core.logging import get_logger


def _hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its bcrypt hash."""
    return _bcrypt.checkpw(password.encode(), hashed.encode())

log = get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# ============================================
# Login rate limiting (in-memory)
# ============================================
_login_attempts: dict[str, list[float]] = defaultdict(list)
_MAX_LOGIN_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 300  # 5 minutes


def _check_rate_limit(email: str) -> None:
    """Raise 429 if too many login attempts for this email."""
    now = time.time()
    window_start = now - _LOGIN_WINDOW_SECONDS
    _login_attempts[email] = [t for t in _login_attempts[email] if t > window_start]
    if len(_login_attempts[email]) >= _MAX_LOGIN_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again later.",
        )


def _record_attempt(email: str) -> None:
    _login_attempts[email].append(time.time())


# ============================================
# In-memory user store (production: use DB)
# ============================================
_users: dict[str, dict] = {}
_tokens: dict[str, dict] = {}  # token -> {user_id, role, bank_id, expires}

# Seed default users — bcrypt-hashed passwords
# In production these would come from a database, not source code.
_DEFAULT_USERS = [
    {
        "user_id": "admin",
        "email": "admin@sentinxfl.com",
        "password_hash": _hash_password("admin123"),
        "role": "employee",
        "display_name": "SentinXFL Admin",
        "bank_id": None,
    },
    {
        "user_id": "bank_demo",
        "email": "demo@bankdemo.com",
        "password_hash": _hash_password("bank123"),
        "role": "client",
        "display_name": "Demo Bank User",
        "bank_id": "bank-demo-001",
    },
]

for u in _DEFAULT_USERS:
    _users[u["user_id"]] = u


# ============================================
# Request/Response Models
# ============================================
class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)
    display_name: str
    role: str = Field(default="client", pattern="^(client|employee)$")
    bank_id: Optional[str] = None


class LoginResponse(BaseModel):
    token: str
    user_id: str
    email: str
    role: str
    display_name: str
    bank_id: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    email: str
    role: str
    display_name: str
    bank_id: Optional[str] = None


# ============================================
# Auth Helpers
# ============================================
def _generate_token(user: dict) -> str:
    """Generate a simple token (production: use proper JWT)."""
    token = secrets.token_urlsafe(48)
    _tokens[token] = {
        "user_id": user["user_id"],
        "role": user["role"],
        "bank_id": user.get("bank_id"),
        "expires": time.time() + 86400,  # 24 hours
    }
    return token


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict | None:
    """Validate token and return user info."""
    if not credentials:
        return None
    token = credentials.credentials
    token_data = _tokens.get(token)
    if not token_data:
        return None
    if token_data["expires"] < time.time():
        del _tokens[token]
        return None
    user = _users.get(token_data["user_id"])
    return user


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Require valid authentication."""
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def require_employee(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Require employee role."""
    user = await require_auth(credentials)
    if user["role"] != "employee":
        raise HTTPException(status_code=403, detail="Employee access required")
    return user


# ============================================
# Auth Endpoints
# ============================================


@router.post("/auth/login", tags=["auth"])
async def login(req: LoginRequest):
    """Login with email and password."""
    _check_rate_limit(req.email)

    user = None
    for u in _users.values():
        if u["email"] == req.email:
            user = u
            break

    if not user or not _verify_password(req.password, user["password_hash"]):
        _record_attempt(req.email)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _generate_token(user)
    return LoginResponse(
        token=token,
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
        display_name=user["display_name"],
        bank_id=user.get("bank_id"),
    )


@router.post("/auth/register", tags=["auth"])
async def register(req: RegisterRequest):
    """Register a new user."""
    # Check duplicate email
    for u in _users.values():
        if u["email"] == req.email:
            raise HTTPException(status_code=400, detail="Email already registered")

    user_id = f"user-{secrets.token_hex(8)}"
    user = {
        "user_id": user_id,
        "email": req.email,
        "password_hash": _hash_password(req.password),
        "role": req.role,
        "display_name": req.display_name,
        "bank_id": req.bank_id,
    }
    _users[user_id] = user

    token = _generate_token(user)
    return LoginResponse(
        token=token,
        user_id=user_id,
        email=req.email,
        role=user["role"],
        display_name=req.display_name,
        bank_id=req.bank_id,
    )


@router.get("/auth/me", tags=["auth"])
async def get_me(user: dict = Depends(require_auth)):
    """Get current user info."""
    return UserInfo(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
        display_name=user["display_name"],
        bank_id=user.get("bank_id"),
    )


@router.post("/auth/logout", tags=["auth"])
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout and invalidate token."""
    if credentials and credentials.credentials in _tokens:
        del _tokens[credentials.credentials]
    return {"message": "Logged out"}


@router.get("/auth/users", tags=["auth"])
async def list_users(user: dict = Depends(require_employee)):
    """List all users (employee only)."""
    return {
        "users": [
            UserInfo(
                user_id=u["user_id"],
                email=u["email"],
                role=u["role"],
                display_name=u["display_name"],
                bank_id=u.get("bank_id"),
            ).model_dump()
            for u in _users.values()
        ]
    }
