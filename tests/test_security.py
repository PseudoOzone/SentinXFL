"""
Security tests for SentinXFL authentication and hardening.

Covers:
- Password hashing (bcrypt, not SHA-256)
- Login rate limiting
- Path traversal prevention in uploads
- Secret key validation
- CORS configuration
- Error message sanitization
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure src/ is on the path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="module")
def client():
    """Create a test HTTP client against the FastAPI app.

    Uses lazy import — if heavy ML/LLM deps aren't installed the
    TestClient-dependent tests are skipped automatically.
    """
    try:
        from sentinxfl.api.app import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        return TestClient(app)
    except ImportError as exc:
        pytest.skip(f"Cannot create test client — missing dependency: {exc}")


# ───────────────────────────────────────────────
# AUTH: Password hashing
# ───────────────────────────────────────────────
class TestPasswordSecurity:
    """Verify bcrypt is used and SHA-256 is not."""

    def test_login_with_correct_credentials(self, client):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "admin@sentinxfl.com", "password": "admin123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["role"] == "employee"

    def test_login_with_wrong_password(self, client):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "admin@sentinxfl.com", "password": "wrong"},
        )
        assert resp.status_code == 401

    def test_stored_hashes_are_bcrypt(self):
        try:
            from sentinxfl.api.routes.auth import _users
        except ImportError:
            pytest.skip("Cannot import auth module")

        for user in _users.values():
            h = user["password_hash"]
            # bcrypt hashes start with $2b$ or $2a$
            assert h.startswith(("$2b$", "$2a$")), (
                f"User {user['user_id']} password is not bcrypt-hashed"
            )

    def test_register_stores_bcrypt(self, client):
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test_security@test.com",
                "password": "strongpass123",
                "display_name": "Test",
                "role": "client",
            },
        )
        assert resp.status_code == 200

        try:
            from sentinxfl.api.routes.auth import _users
        except ImportError:
            pytest.skip("Cannot import auth module")

        new_user = None
        for u in _users.values():
            if u["email"] == "test_security@test.com":
                new_user = u
                break
        assert new_user is not None
        assert new_user["password_hash"].startswith(("$2b$", "$2a$"))


# ───────────────────────────────────────────────
# AUTH: Rate limiting
# ───────────────────────────────────────────────
class TestLoginRateLimit:
    def test_rate_limit_after_max_attempts(self, client):
        """After N failed attempts, further logins should be rejected."""
        try:
            from sentinxfl.api.routes.auth import (
                _login_attempts,
                _MAX_LOGIN_ATTEMPTS,
            )
        except ImportError:
            pytest.skip("Cannot import auth module")

        email = "ratelimit_test@example.com"
        _login_attempts[email] = [time.time()] * _MAX_LOGIN_ATTEMPTS

        resp = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "anything"},
        )
        assert resp.status_code == 429

        # cleanup
        del _login_attempts[email]


# ───────────────────────────────────────────────
# UPLOADS: Path traversal
# ───────────────────────────────────────────────
class TestUploadSecurity:
    def _get_auth_header(self, client) -> dict:
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "admin@sentinxfl.com", "password": "admin123"},
        )
        token = resp.json()["token"]
        return {"Authorization": f"Bearer {token}"}

    def test_filename_sanitized(self, client):
        headers = self._get_auth_header(client)
        # Attempt path traversal via filename
        import io

        file_content = b"col1,col2\n1,2\n"
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("../../etc/passwd.csv", io.BytesIO(file_content), "text/csv")},
            data={"bank_id": "test-bank"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Filename should be sanitized — no path separators
        assert "/" not in data["filename"]
        assert "\\" not in data["filename"]
        assert ".." not in data["filename"]


# ───────────────────────────────────────────────
# CONFIG: Secret key validation
# ───────────────────────────────────────────────
class TestSecretKeyValidation:
    def test_weak_key_rejected_in_production(self):
        """Settings should reject weak keys in non-dev environments."""
        from sentinxfl.core.config import Settings

        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            Settings(
                environment="production",
                secret_key="dev-secret-key-change-in-production",
            )


# ───────────────────────────────────────────────
# CORS: No wildcard
# ───────────────────────────────────────────────
class TestCORSConfig:
    def test_no_wildcard_cors(self):
        from sentinxfl.core.config import get_settings

        s = get_settings()
        assert "*" not in s.cors_origins


# ───────────────────────────────────────────────
# API: Security headers present
# ───────────────────────────────────────────────
class TestSecurityHeaders:
    def test_health_endpoint_has_security_headers(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert "X-Request-ID" in resp.headers

    def test_error_responses_no_stack_trace(self, client):
        """500 responses should not leak internal details."""
        resp = client.post(
            "/api/v1/load",
            json={"dataset_type": "credit_card_fraud", "sample_frac": 0.001},
        )
        # Even if this endpoint fails, the detail should NOT contain Python tracebacks
        if resp.status_code >= 500:
            detail = resp.json().get("detail", "")
            assert "Traceback" not in detail
            assert "File \"" not in detail
