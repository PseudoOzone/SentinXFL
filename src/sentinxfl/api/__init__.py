"""API module for SentinXFL."""


def get_app():
    """Lazily import the app to avoid circular imports and early side effects."""
    from sentinxfl.api.app import app
    return app


__all__ = ["get_app"]
