"""Sync routes for local-to-cloud memory synchronization."""

from datetime import datetime, timezone

from fastapi import APIRouter

from ..auth import CurrentAgent
from ..database import (
    Database,
    delete_memory,
    get_changes_since,
    update_agent_last_sync,
    upsert_memory,
)
from ..logging_config import get_logger, log_sync_operation
from ..models import (
    SyncOperation,
    SyncPullRequest,
    SyncPullResponse,
    SyncPushRequest,
    SyncPushResponse,
)

logger = get_logger("kernle.sync")
router = APIRouter(prefix="/sync", tags=["sync"])


@router.post("/push", response_model=SyncPushResponse)
async def push_changes(
    request: SyncPushRequest,
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
    logger.info(f"PUSH | {log_prefix} | {len(request.operations)} operations")
    synced = 0
    conflicts = []

    for op in request.operations:
        try:
            if op.operation == "delete":
                await delete_memory(db, agent_id, op.table, op.record_id)
                log_sync_operation(log_prefix, "delete", op.table, op.record_id, True)
            else:
                # insert or update
                if op.data is None:
                    log_sync_operation(log_prefix, op.operation, op.table, op.record_id, False, "Missing data")
                    conflicts.append({
                        "record_id": op.record_id,
                        "error": "Missing data for insert/update",
                    })
                    continue
                await upsert_memory(db, agent_id, op.table, op.record_id, op.data)
                log_sync_operation(log_prefix, op.operation, op.table, op.record_id, True)
            synced += 1
        except ValueError as e:
            log_sync_operation(log_prefix, op.operation, op.table, op.record_id, False, str(e))
            conflicts.append({
                "record_id": op.record_id,
                "error": str(e),
            })
        except Exception as e:
            # Log full error server-side for debugging
            logger.error(f"Database error during {op.operation} on {op.table}/{op.record_id}: {e}")
            log_sync_operation(log_prefix, op.operation, op.table, op.record_id, False, str(e))
            # Return generic message to client to avoid leaking internal details
            conflicts.append({
                "record_id": op.record_id,
                "error": "Database error: operation failed",
            })

    # Update agent's last sync time
    await update_agent_last_sync(db, agent_id)

    logger.info(f"PUSH COMPLETE | {log_prefix} | synced={synced} conflicts={len(conflicts)}")

    return SyncPushResponse(
        synced=synced,
        conflicts=conflicts,
        server_time=datetime.now(timezone.utc),
    )


@router.post("/pull", response_model=SyncPullResponse)
async def pull_changes(
    request: SyncPullRequest,
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
    logger.info(f"PULL | {log_prefix} | since={request.since}")

    since_str = request.since.isoformat() if request.since else None
    changes = await get_changes_since(db, agent_id, since_str)

    operations = []
    for change in changes:
        data = change.get("data", {})
        # Parse local_updated_at or use created_at or current time
        local_updated = data.get("local_updated_at") or data.get("created_at") or datetime.now(timezone.utc)
        if isinstance(local_updated, str):
            try:
                local_updated = datetime.fromisoformat(local_updated.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                local_updated = datetime.now(timezone.utc)

        operations.append(SyncOperation(
            operation=change["operation"],
            table=change["table"],
            record_id=change["record_id"],
            data=data if change["operation"] != "delete" else None,
            local_updated_at=local_updated,
            version=data.get("version", 1),
        ))

    logger.info(f"PULL COMPLETE | {log_prefix} | {len(operations)} operations")

    return SyncPullResponse(
        operations=operations,
        server_time=datetime.now(timezone.utc),
        has_more=len(operations) >= 1000,  # Simple pagination indicator
    )


@router.post("/full", response_model=SyncPullResponse)
async def full_sync(
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
    changes = await get_changes_since(db, agent_id, None)

    operations = [
        SyncOperation(
            operation="update",  # Full sync is always "update" (upsert on client)
            table=change["table"],
            record_id=change["record_id"],
            data=change["data"],
            local_updated_at=change["data"].get("local_updated_at", datetime.now(timezone.utc)),
            version=change["data"].get("version", 1),
        )
        for change in changes
        if not change["data"].get("deleted")  # Skip deleted records in full sync
    ]

    return SyncPullResponse(
        operations=operations,
        server_time=datetime.now(timezone.utc),
        has_more=False,
    )
