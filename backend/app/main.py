"""Kernle Backend API - FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .config import get_settings
from .rate_limit import limiter
from .routes import auth_router, memories_router, sync_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"Starting Kernle Backend API (debug={settings.debug})")
    yield
    # Shutdown
    print("Shutting down Kernle Backend API")


app = FastAPI(
    title="Kernle Backend API",
    description="Railway API backend for Kernle memory sync",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(sync_router)
app.include_router(memories_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "kernle-backend",
        "version": "0.1.0",
        "status": "ok",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "connected",  # TODO: Add actual DB check
    }
