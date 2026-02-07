"""
SentinXFL - FastAPI Application
================================

Main application setup with all routes registered.

Author: Anshuman Bakshi
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import and register routes
    from sentinxfl.api.routes import data, privacy, ml, fl, llm, knowledge, auth, upload
    
    app.include_router(data.router, prefix="/api/v1")
    app.include_router(privacy.router, prefix="/api/v1")
    app.include_router(ml.router, prefix="/api/v1")
    app.include_router(fl.router, prefix="/api/v1")
    app.include_router(llm.router, prefix="/api/v1")
    app.include_router(knowledge.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(upload.router, prefix="/api/v1")
    
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
        return {
            "message": "SentinXFL - Privacy-First Federated Fraud Detection",
            "docs": "/docs",
            "health": "/health",
        }
    
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
