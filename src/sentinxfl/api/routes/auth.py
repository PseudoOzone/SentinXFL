"""
SentinXFL - Authentication API Routes
=======================================

Simple JWT-based authentication for dual dashboard system.
- Client Bank users: login, view reports, upload data
- SentinXFL Employees: login, global dashboard, management

Author: Anshuman Bakshi
"""

import hashlib
import secrets
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from sentinxfl.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)


# ============================================
# In-memory user store (production: use DB)
# ============================================
_users: dict[str, dict] = {}
_tokens: dict[str, dict] = {}  # token -> {user_id, role, bank_id, expires}

# Seed default users
_DEFAULT_USERS = [
    {
        "user_id": "admin",
        "email": "admin@sentinxfl.com",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "employee",
        "display_name": "SentinXFL Admin",
        "bank_id": None,
    },
    {
        "user_id": "bank_demo",
        "email": "demo@bankdemo.com",
        "password_hash": hashlib.sha256("bank123".encode()).hexdigest(),
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
    password_hash = hashlib.sha256(req.password.encode()).hexdigest()

    user = None
    for u in _users.values():
        if u["email"] == req.email and u["password_hash"] == password_hash:
            user = u
            break

    if not user:
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
        "password_hash": hashlib.sha256(req.password.encode()).hexdigest(),
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
