"""Kernle Backend API - FastAPI application."""

from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .rate_limit import limiter
from .routes import admin_router, auth_router, embeddings_router, memories_router, sync_router


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection middleware using Origin header validation.

    Validates that state-changing requests (POST, PUT, DELETE, PATCH)
    originate from allowed origins when using cookie-based auth.
    Works in conjunction with SameSite=Strict cookies for defense-in-depth.

    CSRF is only relevant for cookie auth because:
    - API keys can't be forged by malicious sites (they're not auto-sent)
    - Cookies ARE auto-sent, making them vulnerable to CSRF
    """

    STATE_CHANGING_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
    COOKIE_NAME = "kernle_auth"

    def __init__(self, app, allowed_origins: list[str]):
        super().__init__(app)
        # Normalize origins to just scheme://host(:port) for comparison
        self.allowed_origins = set()
        for origin in allowed_origins:
            parsed = urlparse(origin)
            normalized = f"{parsed.scheme}://{parsed.netloc}"
            self.allowed_origins.add(normalized.lower())

    async def dispatch(self, request: Request, call_next):
        # Only check state-changing methods
        if request.method not in self.STATE_CHANGING_METHODS:
            return await call_next(request)

        # Check if request uses API key auth (Bearer header with knl_ prefix)
        # API key requests don't need CSRF protection - they can't be forged
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer knl_"):
            return await call_next(request)

        # Check if request has auth cookie (cookie-based auth needs CSRF protection)
        has_auth_cookie = self.COOKIE_NAME in request.cookies

        # No auth cookie = not using cookie auth, let it through
        # (will fail auth check later if auth is required)
        if not has_auth_cookie:
            return await call_next(request)

        # Cookie auth detected - validate Origin header
        origin = request.headers.get("origin")
        if not origin:
            referer = request.headers.get("referer")
            if referer:
                parsed = urlparse(referer)
                origin = f"{parsed.scheme}://{parsed.netloc}"

        # Cookie auth without Origin = potential CSRF attack
        if not origin:
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF validation failed: missing Origin header"},
            )

        # Validate origin against allowed list
        origin_lower = origin.lower()
        if origin_lower not in self.allowed_origins:
            return JSONResponse(
                status_code=403,
                content={"detail": f"CSRF validation failed: origin '{origin}' not allowed"},
            )

        return await call_next(request)


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
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# CSRF protection middleware (validates Origin header for state-changing requests)
app.add_middleware(CSRFMiddleware, allowed_origins=settings.allowed_origins)

# Include routers
app.include_router(auth_router)
app.include_router(sync_router)
app.include_router(memories_router)
app.include_router(embeddings_router)
app.include_router(admin_router)


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
    """Detailed health check with actual database verification."""
    from .database import get_supabase_client

    db_status = "disconnected"
    try:
        db = get_supabase_client()
        # Simple query to verify connection
        db.table("agents").select("id").limit(1).execute()
        db_status = "connected"
    except Exception as e:
        import logging

        logging.getLogger("kernle.health").error(
            f"Health check database error: {type(e).__name__}: {e}"
        )
        db_status = "error"

    overall_status = "healthy" if db_status == "connected" else "degraded"

    return {
        "status": overall_status,
        "database": db_status,
    }
