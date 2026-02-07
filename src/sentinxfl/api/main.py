"""
SentinXFL FastAPI Application
==============================

Main entry point for the REST API with CORS, health checks, and routers.

Author: Anshuman Bakshi
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from sentinxfl.core.config import settings
from sentinxfl.core.logging import setup_logging, log


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    setup_logging()
    log.info(f"Starting {settings.app_name} v{settings.app_version}")
    log.info(f"Environment: {settings.environment}")
    log.info(f"Debug mode: {settings.debug}")
    log.info(f"VRAM budget: {settings.total_vram_gb}GB (LLM: {settings.llm_vram_gb}GB, TabNet: {settings.tabnet_vram_gb}GB)")

    # Initialize data directory
    settings.data_dir_abs.mkdir(parents=True, exist_ok=True)
    settings.processed_dir_abs.mkdir(parents=True, exist_ok=True)
    settings.models_dir_abs.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    log.info("Shutting down SentinXFL...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Privacy-First Federated Fraud Detection Platform",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000

    log.debug(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )

    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response


# ============================================
# Health Check Endpoints
# ============================================
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs" if settings.debug else "disabled",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns component status and system info.
    """
    import torch
    import platform

    gpu_available = torch.cuda.is_available()
    gpu_info = None
    if gpu_available:
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment,
        "system": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "processor": platform.processor(),
        },
        "gpu": {
            "available": gpu_available,
            "info": gpu_info,
        },
        "components": {
            "api": "operational",
            "database": "pending",  # Will be updated when DuckDB is connected
            "ml_models": "pending",
            "fl_server": "pending",
            "llm": "pending",
        },
    }


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes-style readiness probe."""
    # Check if critical components are ready
    # For now, always ready during development
    return {"ready": True}


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {"alive": True}


# ============================================
# Include Routers
# ============================================
from sentinxfl.api.routes import data, privacy, ml

app.include_router(data.router, prefix="/api/v1/data", tags=["Data"])
app.include_router(privacy.router, prefix="/api/v1/privacy", tags=["Privacy"])
app.include_router(ml.router, prefix="/api/v1", tags=["Machine Learning"])

# Future routers (to be added in Sprint 3+)
# app.include_router(fl.router, prefix="/api/v1/fl", tags=["Federated Learning"])
# app.include_router(llm.router, prefix="/api/v1/llm", tags=["LLM Explainability"])
