"""API route modules."""

from sentinxfl.api.routes import data, privacy, ml, fl

try:
    from sentinxfl.api.routes import llm, knowledge, auth, upload
except ImportError:
    llm = None  # type: ignore
    knowledge = None  # type: ignore
    auth = None  # type: ignore
    upload = None  # type: ignore

__all__ = ["data", "privacy", "ml", "fl", "llm", "knowledge", "auth", "upload"]
