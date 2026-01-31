"""API routes."""

from .admin import router as admin_router
from .auth import router as auth_router
from .commerce import escrow_router, jobs_router, maintenance_router, skills_router, wallets_router
from .embeddings import router as embeddings_router
from .memories import router as memories_router
from .sync import router as sync_router

__all__ = [
    "admin_router",
    "auth_router",
    "sync_router",
    "memories_router",
    "embeddings_router",
    "wallets_router",
    "jobs_router",
    "skills_router",
    "escrow_router",
    "maintenance_router",
]
