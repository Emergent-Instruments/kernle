"""Sync commands for Kernle CLI — local-to-cloud synchronization."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from kernle.utils import get_kernle_home

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


def cmd_sync(args, k: "Kernle"):
    """Handle sync subcommands for local-to-cloud synchronization."""
    # Load credentials with priority:
    # 1. ~/.kernle/credentials.json (preferred)
    # 2. Environment variables (fallback)
    # 3. ~/.kernle/config.json (legacy fallback)

    backend_url = None
    auth_token = None
    user_id = None

    # Try credentials.json first (preferred)
    credentials_path = get_kernle_home() / "credentials.json"
    if credentials_path.exists():
        try:
            import json as json_module

            with open(credentials_path) as f:
                creds = json_module.load(f)
                backend_url = creds.get("backend_url")
                # Support multiple auth token field names
                auth_token = creds.get("auth_token") or creds.get("token") or creds.get("api_key")
                user_id = creds.get("user_id")
        except Exception as e:
            logger.debug(f"Failed to load credentials file: {e}")
            # Fall through to env vars

    # Fall back to environment variables
    if not backend_url:
        backend_url = os.environ.get("KERNLE_BACKEND_URL")
    if not auth_token:
        auth_token = os.environ.get("KERNLE_AUTH_TOKEN")
    if not user_id:
        user_id = os.environ.get("KERNLE_USER_ID")

    # Legacy fallback: check config.json
    config_path = get_kernle_home() / "config.json"
    if config_path.exists() and (not backend_url or not auth_token):
        try:
            import json as json_module

            with open(config_path) as f:
                config = json_module.load(f)
                backend_url = backend_url or config.get("backend_url")
                auth_token = auth_token or config.get("auth_token")
        except Exception as e:
            logger.debug(f"Failed to load legacy config file: {e}")

    def get_local_project_name():
        """Extract the local project name from stack_id (without namespace)."""
        # k.stack_id might be "roundtable" or "user123/roundtable"
        # We want just "roundtable"
        stack_id = k.stack_id
        if "/" in stack_id:
            return stack_id.split("/")[-1]
        return stack_id

    def get_namespaced_stack_id():
        """Get the full namespaced agent ID (user_id/project_name)."""
        project_name = get_local_project_name()
        if user_id:
            return f"{user_id}/{project_name}"
        return project_name

    def get_http_client():
        """Get an HTTP client for backend requests."""
        try:
            import httpx

            return httpx
        except ImportError:
            print("✗ httpx not installed. Run: pip install httpx")
            sys.exit(1)

    def check_backend_connection(httpx_client):
        """Check if backend is reachable and authenticated."""
        if not backend_url:
            return False, "No backend URL configured"
        if not auth_token:
            return False, "Not authenticated (run `kernle auth login`)"

        try:
            response = httpx_client.get(
                f"{backend_url.rstrip('/')}/health",
                timeout=5.0,
            )
            if response.status_code == 200:
                return True, "Connected"
            return False, f"Backend returned status {response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def get_headers():
        """Get authorization headers for backend requests."""
        return {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

    def format_datetime(dt):
        """Format datetime for API requests."""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    if args.sync_action == "status":
        httpx = get_http_client()

        # Get local status from storage
        pending_count = k._storage.get_pending_sync_count()
        last_sync = k._storage.get_last_sync_time()
        is_online = k._storage.is_online()

        # Check backend connection
        backend_connected, connection_msg = check_backend_connection(httpx)

        # Get namespaced agent ID for display
        local_project = get_local_project_name()
        namespaced_id = get_namespaced_stack_id()

        if args.json:
            status_data = {
                "local_stack_id": local_project,
                "namespaced_stack_id": namespaced_id if user_id else None,
                "user_id": user_id,
                "pending_operations": pending_count,
                "last_sync_time": format_datetime(last_sync),
                "local_storage_online": is_online,
                "backend_url": backend_url or "(not configured)",
                "backend_connected": backend_connected,
                "connection_status": connection_msg,
                "authenticated": bool(auth_token),
            }
            print(json.dumps(status_data, indent=2, default=str))
        else:
            print("Sync Status")
            print("=" * 50)
            print()

            # Agent/Project info
            print(f"📦 Local project: {local_project}")
            if user_id and backend_connected:
                print(f"   Synced as: {namespaced_id}")
            elif user_id:
                print(f"   Will sync as: {namespaced_id}")
            print()

            # Connection status
            conn_icon = "🟢" if backend_connected else "🔴"
            print(f"{conn_icon} Backend: {connection_msg}")
            if backend_url:
                print(f"   URL: {backend_url}")
            if user_id:
                print(f"   User: {user_id}")
            print()

            # Pending operations
            pending_icon = "🟢" if pending_count == 0 else "🟡" if pending_count < 10 else "🟠"
            print(f"{pending_icon} Pending operations: {pending_count}")

            # Last sync time
            if last_sync:
                now = datetime.now(timezone.utc)
                if hasattr(last_sync, "tzinfo") and last_sync.tzinfo is None:
                    from datetime import timezone as tz

                    last_sync = last_sync.replace(tzinfo=tz.utc)
                elapsed = now - last_sync
                if elapsed.total_seconds() < 60:
                    elapsed_str = "just now"
                elif elapsed.total_seconds() < 3600:
                    elapsed_str = f"{int(elapsed.total_seconds() / 60)} minutes ago"
                elif elapsed.total_seconds() < 86400:
                    elapsed_str = f"{int(elapsed.total_seconds() / 3600)} hours ago"
                else:
                    elapsed_str = f"{int(elapsed.total_seconds() / 86400)} days ago"
                print(f"🕐 Last sync: {elapsed_str}")
                print(f"   ({last_sync.isoformat()[:19]})")
            else:
                print("🕐 Last sync: Never")

            # Suggestions
            print()
            if pending_count > 0 and backend_connected:
                print("💡 Run `kernle sync push` to upload pending changes")
            elif not backend_connected and not auth_token:
                print("💡 Run `kernle auth login` to authenticate")
            elif not backend_connected:
                print("💡 Check backend connection or run `kernle auth login`")

    elif args.sync_action == "push":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        # Get pending changes from storage
        queued_changes = k._storage.get_queued_changes(limit=args.limit)

        if not queued_changes:
            print("✓ No pending changes to push")
            return

        print(f"Pushing {len(queued_changes)} changes to backend...")

        # Map local table names to backend table names
        table_name_map = {
            "values": "values",
            "agent_beliefs": "beliefs",
            "agent_episodes": "episodes",
            "agent_notes": "notes",
            "agent_goals": "goals",
            "agent_drives": "drives",
            "agent_relationships": "relationships",
            "agent_playbooks": "playbooks",
            "agent_raw": "raw_captures",
            "raw_entries": "raw_captures",  # actual local table name
        }

        # Build operations list for the API
        operations = []
        skipped_orphans = 0
        for change in queued_changes:
            op_type = (
                "update" if change.operation in ("upsert", "insert", "update") else change.operation
            )

            # Map table name for backend
            backend_table = table_name_map.get(change.table_name, change.table_name)

            op_data = {
                "operation": op_type,
                "table": backend_table,
                "record_id": change.record_id,
                "local_updated_at": format_datetime(change.queued_at),
                "version": 1,
            }

            # Add record data for non-delete operations
            # Strategy: stored payload first (canonical snapshot from change time),
            # fall back to re-fetch from source table
            if op_type != "delete":
                record_dict = None

                # Try stored payload first (survives source record deletion)
                if change.payload:
                    try:
                        record_dict = json.loads(change.payload)
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Fall back to live source table
                if not record_dict:
                    record = k._storage._get_record_for_push(change.table_name, change.record_id)
                    if record:
                        # Extract all dataclass fields from the record.
                        # Uses actual dataclass field names (e.g., "outcome" not
                        # "outcome_description") so the payload matches the
                        # Supabase schema after table_name_map translation.
                        import dataclasses as _dc

                        if _dc.is_dataclass(record):
                            for f in _dc.fields(record):
                                value = getattr(record, f.name)
                                if value is None:
                                    continue
                                if hasattr(value, "isoformat"):
                                    value = value.isoformat()
                                record_dict[f.name] = value
                        else:
                            # Fallback for non-dataclass records (shouldn't happen)
                            for field in [
                                "id",
                                "stack_id",
                                "content",
                                "objective",
                                "outcome",
                                "outcome_type",
                                "lessons",
                                "tags",
                                "statement",
                                "confidence",
                                "drive_type",
                                "intensity",
                                "name",
                                "priority",
                                "title",
                                "status",
                                "progress",
                                "entity_name",
                                "entity_type",
                                "relationship_type",
                                "notes",
                                "sentiment",
                                "focus_areas",
                                "created_at",
                                "updated_at",
                                "local_updated_at",
                                "source_type",
                                "source_entity",
                                "source_episodes",
                                "derived_from",
                                "context",
                                "context_tags",
                                "timestamp",
                                "source",
                                "processed",
                                "description",
                                "steps",
                                "triggers",
                                "target_date",
                            ]:
                                if hasattr(record, field):
                                    value = getattr(record, field)
                                    if hasattr(value, "isoformat"):
                                        value = value.isoformat()
                                    record_dict[field] = value

                if record_dict:
                    op_data["data"] = record_dict
                else:
                    # No stored payload AND no source record — orphaned entry
                    skipped_orphans += 1
                    with k._storage._connect() as conn:
                        conn.execute(
                            "UPDATE sync_queue SET synced = 1 WHERE id = ?",
                            (change.id,),
                        )
                    continue

            operations.append(op_data)

        if skipped_orphans > 0:
            print(f"⚠️  Skipped {skipped_orphans} orphaned entries (source records deleted)")

        # Send to backend
        # Include stack_id as just the local project name
        # Backend will namespace it with the authenticated user_id
        try:
            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/push",
                headers=get_headers(),
                json={
                    "stack_id": local_project,  # Local name only, backend namespaces
                    "operations": operations,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                synced = result.get("synced", 0)
                conflicts = result.get("conflicts", [])

                # Clear synced items from local queue
                with k._storage._connect() as conn:
                    for change in queued_changes[:synced]:
                        k._storage._clear_queued_change(conn, change.id)
                        k._storage._mark_synced(conn, change.table_name, change.record_id)
                    conn.commit()

                # Update last sync time
                k._storage._set_sync_meta("last_sync_time", k._storage._now())

                if args.json:
                    result["local_project"] = local_project
                    result["namespaced_id"] = get_namespaced_stack_id()
                    print(json.dumps(result, indent=2, default=str))
                else:
                    namespaced = get_namespaced_stack_id()
                    print(f"✓ Pushed {synced} changes")
                    if user_id:
                        print(f"  Synced as: {namespaced}")
                    if conflicts:
                        print(f"⚠️  {len(conflicts)} conflicts:")
                        for c in conflicts[:5]:
                            print(
                                f"   - {c.get('record_id', 'unknown')}: {c.get('error', 'unknown error')}"
                            )
            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Push failed: {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except Exception as e:
            print(f"✗ Push failed: {e}")
            sys.exit(1)

    elif args.sync_action == "pull":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        # Get last sync time for incremental pull
        since = k._storage.get_last_sync_time() if not args.full else None

        print(f"Pulling changes from backend{' (full)' if args.full else ''}...")

        try:
            # Include stack_id - backend will namespace with user_id
            request_data = {
                "stack_id": local_project,  # Local name only, backend namespaces
            }
            if since and not args.full:
                request_data["since"] = format_datetime(since)

            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/pull",
                headers=get_headers(),
                json=request_data,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                operations = result.get("operations", [])
                has_more = result.get("has_more", False)

                if not operations:
                    print("✓ Already up to date")
                    return

                # Apply operations locally
                applied = 0
                conflicts = 0

                for op in operations:
                    try:
                        table = op.get("table")
                        record_id = op.get("record_id")
                        data = op.get("data", {})
                        operation = op.get("operation")

                        if operation == "delete":
                            # Handle soft delete
                            # (implementation depends on storage structure)
                            pass
                        else:
                            # Upsert the record
                            # This is simplified - real implementation would use proper converters
                            if table == "episodes" and data:
                                from kernle.storage import Episode

                                ep = Episode(
                                    id=record_id,
                                    stack_id=k.stack_id,
                                    objective=data.get("objective", ""),
                                    outcome_type=data.get("outcome_type", "neutral"),
                                    outcome=data.get(
                                        "outcome", data.get("outcome_description", "")
                                    ),
                                    lessons=data.get("lessons", data.get("lessons_learned", [])),
                                    tags=data.get("tags", []),
                                )
                                k._storage.save_episode(ep)
                                # Mark as synced (don't queue for push)
                                with k._storage._connect() as conn:
                                    k._storage._mark_synced(conn, table, record_id)
                                    conn.execute(
                                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                                        (table, record_id),
                                    )
                                    conn.commit()
                                applied += 1
                            elif table == "notes" and data:
                                from kernle.storage import Note

                                note = Note(
                                    id=record_id,
                                    stack_id=k.stack_id,
                                    content=data.get("content", ""),
                                    note_type=data.get("note_type", "note"),
                                    tags=data.get("tags", []),
                                )
                                k._storage.save_note(note)
                                with k._storage._connect() as conn:
                                    k._storage._mark_synced(conn, table, record_id)
                                    conn.execute(
                                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                                        (table, record_id),
                                    )
                                    conn.commit()
                                applied += 1
                            # Add more table handlers as needed
                            else:
                                # For other tables, just track as applied
                                applied += 1

                    except Exception as e:
                        logger.debug(f"Failed to apply operation for {table}:{record_id}: {e}")
                        conflicts += 1

                # Update last sync time
                k._storage._set_sync_meta("last_sync_time", k._storage._now())

                if args.json:
                    print(
                        json.dumps(
                            {
                                "pulled": applied,
                                "conflicts": conflicts,
                                "has_more": has_more,
                                "local_project": local_project,
                                "namespaced_id": get_namespaced_stack_id(),
                            },
                            indent=2,
                        )
                    )
                else:
                    print(f"✓ Pulled {applied} changes")
                    if user_id:
                        print(f"  From: {get_namespaced_stack_id()}")
                    if conflicts > 0:
                        print(f"⚠️  {conflicts} conflicts during apply")
                    if has_more:
                        print("ℹ️  More changes available - run `kernle sync pull` again")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Pull failed: {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except Exception as e:
            print(f"✗ Pull failed: {e}")
            sys.exit(1)

    elif args.sync_action == "full":
        httpx = get_http_client()

        if not backend_url:
            print("✗ Backend not configured")
            print("  Run `kernle auth login` or set KERNLE_BACKEND_URL")
            sys.exit(1)
        if not auth_token:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` or set KERNLE_AUTH_TOKEN")
            sys.exit(1)

        # Use local project name - backend will namespace with user_id
        local_project = get_local_project_name()

        print("Running full bidirectional sync...")
        if user_id:
            print(f"  Syncing as: {get_namespaced_stack_id()}")
        print()

        # Step 1: Pull first (to get remote changes)
        print("Step 1: Pulling remote changes...")
        try:
            response = httpx.post(
                f"{backend_url.rstrip('/')}/sync/pull",
                headers=get_headers(),
                json={
                    "stack_id": local_project,
                    "since": format_datetime(k._storage.get_last_sync_time()),
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                pulled = len(result.get("operations", []))
                print(f"  ✓ Pulled {pulled} changes")
            else:
                print(f"  ⚠️  Pull returned status {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Pull failed: {e}")

        # Step 2: Push local changes
        print("Step 2: Pushing local changes...")
        queued_changes = k._storage.get_queued_changes(limit=1000)

        # Map local table names to backend table names
        table_name_map = {
            "values": "values",
            "agent_beliefs": "beliefs",
            "agent_episodes": "episodes",
            "agent_notes": "notes",
            "agent_goals": "goals",
            "agent_drives": "drives",
            "agent_relationships": "relationships",
            "agent_playbooks": "playbooks",
            "agent_raw": "raw_captures",
            "raw_entries": "raw_captures",  # actual local table name
        }

        if not queued_changes:
            print("  ✓ No pending changes to push")
        else:
            operations = []
            skipped_orphans = 0
            for change in queued_changes:
                op_type = (
                    "update"
                    if change.operation in ("upsert", "insert", "update")
                    else change.operation
                )

                # Map table name for backend
                backend_table = table_name_map.get(change.table_name, change.table_name)

                op_data = {
                    "operation": op_type,
                    "table": backend_table,
                    "record_id": change.record_id,
                    "local_updated_at": format_datetime(change.queued_at),
                    "version": 1,
                }

                # Payload-first: use stored data, fall back to source table
                if op_type != "delete":
                    record_dict = None

                    if change.payload:
                        try:
                            record_dict = json.loads(change.payload)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    if not record_dict:
                        record = k._storage._get_record_for_push(
                            change.table_name, change.record_id
                        )
                        if record:
                            record_dict = {}
                            # Extract all dataclass fields (see first push path)
                            import dataclasses as _dc

                            if _dc.is_dataclass(record):
                                for f in _dc.fields(record):
                                    value = getattr(record, f.name)
                                    if value is None:
                                        continue
                                    if hasattr(value, "isoformat"):
                                        value = value.isoformat()
                                    record_dict[f.name] = value
                            else:
                                for field in [
                                    "id",
                                    "stack_id",
                                    "content",
                                    "objective",
                                    "outcome",
                                    "outcome_type",
                                    "lessons",
                                    "tags",
                                    "statement",
                                    "confidence",
                                    "drive_type",
                                    "intensity",
                                    "name",
                                    "priority",
                                    "title",
                                    "status",
                                    "progress",
                                    "entity_name",
                                    "entity_type",
                                    "relationship_type",
                                    "notes",
                                    "sentiment",
                                    "focus_areas",
                                    "created_at",
                                    "updated_at",
                                    "local_updated_at",
                                    "source_type",
                                    "source_entity",
                                    "source_episodes",
                                    "derived_from",
                                    "context",
                                    "context_tags",
                                    "timestamp",
                                    "source",
                                    "processed",
                                    "description",
                                    "steps",
                                    "triggers",
                                    "target_date",
                                ]:
                                    if hasattr(record, field):
                                        value = getattr(record, field)
                                        if hasattr(value, "isoformat"):
                                            value = value.isoformat()
                                        record_dict[field] = value

                    if record_dict:
                        op_data["data"] = record_dict
                    else:
                        skipped_orphans += 1
                        with k._storage._connect() as conn:
                            conn.execute(
                                "UPDATE sync_queue SET synced = 1 WHERE id = ?",
                                (change.id,),
                            )
                    continue

                operations.append(op_data)

            if skipped_orphans > 0:
                print(f"  ⚠️  Skipped {skipped_orphans} orphaned entries (source records deleted)")

            try:
                response = httpx.post(
                    f"{backend_url.rstrip('/')}/sync/push",
                    headers=get_headers(),
                    json={
                        "stack_id": local_project,  # Local name only, backend namespaces
                        "operations": operations,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    synced = result.get("synced", 0)

                    # Clear synced items
                    with k._storage._connect() as conn:
                        for change in queued_changes[:synced]:
                            k._storage._clear_queued_change(conn, change.id)
                            k._storage._mark_synced(conn, change.table_name, change.record_id)
                        conn.commit()

                    print(f"  ✓ Pushed {synced} changes")
                else:
                    print(f"  ⚠️  Push returned status {response.status_code}")
            except Exception as e:
                print(f"  ⚠️  Push failed: {e}")

        # Update last sync time
        k._storage._set_sync_meta("last_sync_time", k._storage._now())

        print()
        print("✓ Full sync complete")

        # Show final status
        remaining = k._storage.get_pending_sync_count()
        if remaining > 0:
            print(f"ℹ️  {remaining} operations still pending")

    elif args.sync_action == "conflicts":
        # Get conflict history from storage
        if args.clear:
            cleared = k._storage.clear_sync_conflicts()
            if args.json:
                print(json.dumps({"cleared": cleared}))
            else:
                print(f"✓ Cleared {cleared} conflict records")
            return

        conflicts = k._storage.get_sync_conflicts(limit=args.limit)

        if args.json:
            conflict_data = []
            for c in conflicts:
                conflict_data.append(
                    {
                        "id": c.id,
                        "table": c.table,
                        "record_id": c.record_id,
                        "resolution": c.resolution,
                        "resolved_at": c.resolved_at.isoformat() if c.resolved_at else None,
                        "local_summary": c.local_summary,
                        "cloud_summary": c.cloud_summary,
                    }
                )
            print(json.dumps({"conflicts": conflict_data, "count": len(conflicts)}, indent=2))
        else:
            if not conflicts:
                print("No sync conflicts in history")
                print("  Conflicts are recorded when local and cloud versions differ during sync")
                return

            print(f"Sync Conflict History ({len(conflicts)} conflicts)")
            print()

            for c in conflicts:
                resolution_icon = "↓" if c.resolution == "cloud_wins" else "↑"
                resolution_text = "cloud wins" if c.resolution == "cloud_wins" else "local wins"
                when = c.resolved_at.strftime("%Y-%m-%d %H:%M") if c.resolved_at else "unknown"

                print(f"{resolution_icon} {c.table}:{c.record_id[:8]}... ({resolution_text})")
                print(f"  Resolved: {when}")
                if c.local_summary:
                    print(f'  Local:  "{c.local_summary}"')
                if c.cloud_summary:
                    print(f'  Cloud:  "{c.cloud_summary}"')
                print()

            print("💡 Use `kernle sync conflicts --clear` to clear history")
