"""Sync routes for local-to-cloud memory synchronization."""

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from ..auth import CurrentAgent
from ..database import (
    Database,
    delete_memory,
    get_agent_by_user_and_name,
    get_changes_since,
    update_agent_last_sync,
    upsert_memory,
)
from ..embeddings import create_embedding, extract_text_for_embedding
from ..logging_config import get_logger, log_sync_operation
from ..models import (
    SyncOperation,
    SyncPullRequest,
    SyncPullResponse,
    SyncPushRequest,
    SyncPushResponse,
)
from ..rate_limit import limiter

logger = get_logger("kernle.sync")
router = APIRouter(prefix="/sync", tags=["sync"])

# Fields that clients are NEVER allowed to set (server-controlled)
# These are stripped from any incoming sync data to prevent mass assignment attacks
SERVER_CONTROLLED_FIELDS = frozenset(
    {
        "agent_ref",  # Server sets based on authenticated user (FK integrity)
        "deleted",  # Server controls soft-delete state
        "version",  # Server manages versioning
        "id",  # Server assigns/validates record IDs
        "embedding",  # Server generates embeddings (already stripped separately)
        "synced_at",  # Server timestamp
        "server_updated_at",  # Server timestamp
    }
)


@router.post("/push", response_model=SyncPushResponse)
@limiter.limit("60/minute")
async def push_changes(
    request: Request,
    sync_request: SyncPushRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Push local changes to the cloud.

    Processes operations in order:
    - insert/update: Upsert the record
    - delete: Soft-delete the record

    Returns count of synced operations and any conflicts.

    Agent ID namespacing:
    - Client can send just project_name (e.g., "claire")
    - Server uses agent_id from JWT for DB (maintains FK integrity)
    - user_id is available for future multi-tenant queries
    """
    # Use agent_id from JWT (maintains FK to agents table)
    agent_id = auth.agent_id
    log_prefix = f"{auth.user_id}/{agent_id}" if auth.user_id else agent_id
    logger.info(f"PUSH | {log_prefix} | {len(sync_request.operations)} operations")

    # Look up agent UUID for FK (multi-tenant: use user_id + agent_id)
    agent_ref = None
    if auth.user_id:
        agent = await get_agent_by_user_and_name(db, auth.user_id, agent_id)
        if agent:
            agent_ref = agent.get("id")

    synced = 0
    conflicts = []

    for op in sync_request.operations:
        try:
            if op.operation == "delete":
                await delete_memory(db, agent_id, op.table, op.record_id)
                log_sync_operation(log_prefix, "delete", op.table, op.record_id, True)
            else:
                # insert or update
                if op.data is None:
                    log_sync_operation(
                        log_prefix, op.operation, op.table, op.record_id, False, "Missing data"
                    )
                    conflicts.append(
                        {
                            "record_id": op.record_id,
                            "error": "Missing data for insert/update",
                        }
                    )
                    continue

                # SECURITY: Strip server-controlled fields to prevent mass assignment
                # Client cannot set agent_ref, deleted, version, id, etc.
                stripped_fields = [k for k in op.data.keys() if k in SERVER_CONTROLLED_FIELDS]
                if stripped_fields:
                    logger.warning(
                        f"Stripped server-controlled fields from {op.table}/{op.record_id}: {stripped_fields}"
                    )

                # Filter to only allowed fields (exclude server-controlled AND embedding)
                sanitized_data = {
                    k: v for k, v in op.data.items() if k not in SERVER_CONTROLLED_FIELDS
                }

                # Server-side re-embedding: generate OpenAI embedding
                # Client uses 384-dim HashEmbedder locally; server uses 1536-dim OpenAI
                # This makes semantic search a subscription feature (uses our OpenAI key)
                data_with_embedding = sanitized_data.copy()

                text_content = extract_text_for_embedding(op.table, sanitized_data)
                if text_content:
                    embedding = await create_embedding(text_content)
                    if embedding:
                        data_with_embedding["embedding"] = embedding
                        logger.debug(f"Generated 1536-dim embedding for {op.table}/{op.record_id}")
                    else:
                        logger.warning(
                            f"Failed to generate embedding for {op.table}/{op.record_id} - storing without embedding"
                        )

                await upsert_memory(
                    db, agent_id, op.table, op.record_id, data_with_embedding, agent_ref=agent_ref
                )
                log_sync_operation(log_prefix, op.operation, op.table, op.record_id, True)
            synced += 1
        except ValueError as e:
            # Log full error server-side for debugging
            logger.warning(
                f"Validation error during {op.operation} on {op.table}/{op.record_id}: {e}"
            )
            log_sync_operation(log_prefix, op.operation, op.table, op.record_id, False, str(e))
            # Return generic message to client to avoid leaking internal details
            conflicts.append(
                {
                    "record_id": op.record_id,
                    "error": "Validation error: invalid operation data",
                }
            )
        except Exception as e:
            # Log full error server-side for debugging
            logger.error(f"Database error during {op.operation} on {op.table}/{op.record_id}: {e}")
            log_sync_operation(log_prefix, op.operation, op.table, op.record_id, False, str(e))
            # Return generic message to client to avoid leaking internal details
            conflicts.append(
                {
                    "record_id": op.record_id,
                    "error": "Database error: operation failed",
                }
            )

    # Update agent's last sync time
    await update_agent_last_sync(db, agent_id)

    logger.info(f"PUSH COMPLETE | {log_prefix} | synced={synced} conflicts={len(conflicts)}")

    return SyncPushResponse(
        synced=synced,
        conflicts=conflicts,
        server_time=datetime.now(timezone.utc),
    )


@router.post("/pull", response_model=SyncPullResponse)
@limiter.limit("60/minute")
async def pull_changes(
    request: Request,
    pull_request: SyncPullRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Pull changes from the cloud since the given timestamp.

    Used for:
    - Initial sync (since=None gets all records)
    - Incremental sync (since=last_sync_at)
    """
    agent_id = auth.agent_id
    log_prefix = f"{auth.user_id}/{agent_id}" if auth.user_id else agent_id
    logger.info(f"PULL | {log_prefix} | since={pull_request.since}")

    since_str = pull_request.since.isoformat() if pull_request.since else None
    changes, has_more = await get_changes_since(db, agent_id, since_str)

    operations = []
    for change in changes:
        data = change.get("data", {})
        # Strip embedding from response - client will re-embed locally with HashEmbedder
        # Server stores 1536-dim OpenAI embeddings; client uses 384-dim locally
        data_without_embedding = {k: v for k, v in data.items() if k != "embedding"}

        # Parse local_updated_at or use created_at or current time
        local_updated = (
            data.get("local_updated_at") or data.get("created_at") or datetime.now(timezone.utc)
        )
        if isinstance(local_updated, str):
            try:
                local_updated = datetime.fromisoformat(local_updated.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                local_updated = datetime.now(timezone.utc)

        operations.append(
            SyncOperation(
                operation=change["operation"],
                table=change["table"],
                record_id=change["record_id"],
                data=data_without_embedding if change["operation"] != "delete" else None,
                local_updated_at=local_updated,
                version=data.get("version", 1),
            )
        )

    logger.info(f"PULL COMPLETE | {log_prefix} | {len(operations)} operations, has_more={has_more}")

    return SyncPullResponse(
        operations=operations,
        server_time=datetime.now(timezone.utc),
        has_more=has_more,
    )


@router.post("/full", response_model=SyncPullResponse)
@limiter.limit("10/minute")
async def full_sync(
    request: Request,
    auth: CurrentAgent,
    db: Database,
):
    """
    Perform a full sync - returns all records for the agent.

    Use for:
    - Initial setup on new device
    - Recovery after data loss
    """
    agent_id = auth.agent_id
    # Full sync uses a very high limit to get everything
    changes, _ = await get_changes_since(db, agent_id, None, limit=100000)

    operations = []
    for change in changes:
        if change["data"].get("deleted"):
            continue  # Skip deleted records in full sync
        # Strip embedding from response - client will re-embed locally
        data = {k: v for k, v in change["data"].items() if k != "embedding"}

        # Parse local_updated_at with same logic as pull_changes
        local_updated = (
            data.get("local_updated_at") or data.get("created_at") or datetime.now(timezone.utc)
        )
        if isinstance(local_updated, str):
            try:
                local_updated = datetime.fromisoformat(local_updated.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                local_updated = datetime.now(timezone.utc)

        operations.append(
            SyncOperation(
                operation="update",  # Full sync is always "update" (upsert on client)
                table=change["table"],
                record_id=change["record_id"],
                data=data,
                local_updated_at=local_updated,
                version=change["data"].get("version", 1),
            )
        )

    return SyncPullResponse(
        operations=operations,
        server_time=datetime.now(timezone.utc),
        has_more=False,
    )
