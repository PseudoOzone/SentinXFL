"""
SentinXFL - FastAPI Application
================================

Main application setup with all routes registered.

Author: Anshuman Bakshi
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    logger.info("Starting SentinXFL API server")
    yield
    logger.info("Shutting down SentinXFL API server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SentinXFL",
        description="Privacy-First Federated Fraud Detection Platform",
        version="2.0.0",
        lifespan=lifespan,
    )
    
    # ── Security Headers Middleware ──────────────────────────────
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
            response.headers["Cache-Control"] = "no-store"
            # Correlation / request ID
            request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
            response.headers["X-Request-ID"] = request_id
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    # ── Request Body Size Limit (10 MB default, uploads use their own limit) ──
    class BodySizeLimitMiddleware(BaseHTTPMiddleware):
        MAX_BODY = 10 * 1024 * 1024  # 10 MB

        async def dispatch(self, request: Request, call_next):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.MAX_BODY:
                # Allow upload endpoint to handle its own size check
                if "/upload" not in request.url.path:
                    return JSONResponse(
                        {"detail": "Request body too large"},
                        status_code=413,
                    )
            return await call_next(request)

    app.add_middleware(BodySizeLimitMiddleware)

    # ── CORS middleware — restricted origins ────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    
    # Import and register routes
    from sentinxfl.api.routes import data, privacy, ml, fl
    
    app.include_router(data.router, prefix="/api/v1")
    app.include_router(privacy.router, prefix="/api/v1")
    app.include_router(ml.router, prefix="/api/v1")
    app.include_router(fl.router, prefix="/api/v1")
    
    try:
        from sentinxfl.api.routes import llm, knowledge, auth, upload
        app.include_router(llm.router, prefix="/api/v1")
        app.include_router(knowledge.router, prefix="/api/v1")
        app.include_router(auth.router, prefix="/api/v1")
        app.include_router(upload.router, prefix="/api/v1")
    except ImportError:
        pass  # Optional routes when chromadb/other deps are missing
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": "2.0.0",
            "project": "SentinXFL",
        }
    
    @app.get("/api/v1/health")
    async def health_check_v1():
        return {
            "status": "healthy",
            "version": "2.0.0",
            "project": "SentinXFL",
        }
    
    @app.get("/")
    async def root():
        # Serve dashboard if built, otherwise API info
        dist_dir = Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "dist"
        index = dist_dir / "index.html"
        if index.exists():
            return FileResponse(index)
        return {
            "message": "SentinXFL - Privacy-First Federated Fraud Detection",
            "docs": "/docs",
            "health": "/health",
        }
    
    # Serve dashboard static files (built Vite output)
    dist_dir = Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "dist"
    if dist_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="static-assets")
        
        # Catch-all for SPA routing — must be last
        # Only serve SPA for non-API paths so API 404s still work correctly
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Don't intercept API routes — let them 404 naturally
            if full_path.startswith("api/"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            file_path = dist_dir / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(dist_dir / "index.html")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "sentinxfl.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
