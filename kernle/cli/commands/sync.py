"""Sync commands for Kernle CLI — local-to-cloud synchronization."""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from kernle.utils import get_kernle_home

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)

PULL_POISON_META_KEY = "pull_poison_operations"


def _validate_backend_url(url: str) -> "str | None":
    """Validate backend URL to avoid leaking auth tokens over plaintext HTTP.

    Mirrors the validation in CloudClient._validate_backend_url().
    """
    if not url:
        return None
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        logger.warning("Invalid backend_url scheme; only http/https allowed.")
        return None
    if not parsed.netloc:
        logger.warning("Invalid backend_url; missing host.")
        return None
    if parsed.scheme == "http":
        host = parsed.hostname or ""
        if host not in {"localhost", "127.0.0.1"}:
            logger.warning("Refusing non-local http backend_url for security.")
            return None
    return url


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

    # Validate backend URL security
    if backend_url:
        backend_url = _validate_backend_url(backend_url)

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

    def _operation_identity(table, record_id):
        """Build a stable identity tuple for sync operation matching."""
        if not record_id:
            return None
        return (table or "", str(record_id))

    def _extract_identities(items):
        """Extract operation identities from response/conflict structures."""
        identities = set()
        if not isinstance(items, list):
            return identities

        for item in items:
            if isinstance(item, dict):
                table = item.get("table") or item.get("table_name")
                record_id = item.get("record_id") or item.get("id")
                identity = _operation_identity(table, record_id)
                if identity:
                    identities.add(identity)
            elif isinstance(item, str):
                # Support record_id-only formats
                identities.add(("", item))
        return identities

    def _identity_matches(identity, identities):
        """Match full identity first, then record_id-only fallback."""
        if not identity:
            return False
        return identity in identities or ("", identity[1]) in identities

    def _resolve_acked_changes(result, operations, op_identity_to_change):
        """Resolve acknowledged queue items using operation identity, not position."""
        if not operations:
            return []

        explicit_ack_keys = (
            "acknowledged_operations",
            "acknowledged",
            "acked_operations",
            "synced_operations",
            "applied_operations",
            "applied",
            "synced_records",
            "record_ids",
        )

        ack_identities = set()
        for key in explicit_ack_keys:
            value = result.get(key)
            if isinstance(value, list):
                ack_identities = _extract_identities(value)
                if ack_identities:
                    break

        if ack_identities:
            acknowledged = []
            seen = set()
            for identity, change in op_identity_to_change.items():
                if identity in seen:
                    continue
                if _identity_matches(identity, ack_identities):
                    acknowledged.append(change)
                    seen.add(identity)
            return acknowledged

        # Fallback inference: treat conflict identities as failed and acknowledge the rest.
        conflict_identities = _extract_identities(result.get("conflicts", []))
        inferred_identities = []
        for op in operations:
            identity = _operation_identity(op.get("table"), op.get("record_id"))
            if identity and not _identity_matches(identity, conflict_identities):
                inferred_identities.append(identity)

        # Safety guard: if server count disagrees with identity inference, do not ack blindly.
        synced_count = result.get("synced")
        if isinstance(synced_count, int) and synced_count != len(inferred_identities):
            logger.warning(
                "Push acknowledgement mismatch: synced=%s inferred=%s; refusing positional ack.",
                synced_count,
                len(inferred_identities),
            )
            return []

        acknowledged = []
        seen = set()
        for identity in inferred_identities:
            if identity in seen:
                continue
            change = op_identity_to_change.get(identity)
            if change:
                acknowledged.append(change)
                seen.add(identity)
        return acknowledged

    def _build_push_operations(queued_changes):
        """Build push payload operations and map them back to queue identities."""
        operations = []
        skipped_orphans = 0
        op_identity_to_change = {}

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

            # Payload-first: use stored data, fall back to source table
            if op_type != "delete":
                record_dict = None

                if change.payload:
                    try:
                        record_dict = json.loads(change.payload)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if not record_dict:
                    record = k._storage._get_record_for_push(change.table_name, change.record_id)
                    if record:
                        # Extract dataclass fields directly to keep payload schema-aligned.
                        import dataclasses as _dc

                        record_dict = {}
                        for f in _dc.fields(record):
                            value = getattr(record, f.name)
                            if value is None:
                                continue
                            if hasattr(value, "isoformat"):
                                value = value.isoformat()
                            record_dict[f.name] = value

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
            identity = _operation_identity(backend_table, change.record_id)
            if identity:
                op_identity_to_change[identity] = change

        return operations, skipped_orphans, op_identity_to_change

    def _serialize_json(value):
        """Serialize to stable JSON for metadata storage/fingerprinting."""
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return json.dumps(str(value))

    def _pull_poison_key(op):
        """Build a stable key for a failed pull operation."""
        table = str(op.get("table") or "")
        record_id = str(op.get("record_id") or "")
        operation = str(op.get("operation") or "")
        if table and record_id and operation:
            return f"{table}:{record_id}:{operation}"
        digest = hashlib.sha256(_serialize_json(op).encode("utf-8")).hexdigest()[:16]
        return f"invalid:{digest}"

    def _load_pull_poison_records():
        """Load quarantined pull operations from sync metadata."""
        raw = k._storage._get_sync_meta(PULL_POISON_META_KEY)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Corrupt pull poison metadata; resetting.")
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(key): value for key, value in parsed.items() if isinstance(value, dict)}

    def _save_pull_poison_records(records):
        """Persist quarantined pull operations."""
        k._storage._set_sync_meta(PULL_POISON_META_KEY, _serialize_json(records or {}))

    def _save_pull_apply_conflict(op, error):
        """Persist a failed pull apply as conflict history for observability."""
        from kernle.types import SyncConflict

        table = str(op.get("table") or "unknown")
        record_id = str(op.get("record_id") or f"unknown-{uuid4()}")
        summary = f"pull apply failed: {error}"[:200]
        conflict = SyncConflict(
            id=str(uuid4()),
            table=table,
            record_id=record_id,
            local_version={},
            cloud_version={"operation": op},
            resolution="pull_apply_failed",
            resolved_at=datetime.now(timezone.utc),
            local_summary="apply failed",
            cloud_summary=summary,
        )
        try:
            k._storage.save_sync_conflict(conflict)
        except Exception as exc:
            logger.debug(
                "Failed to persist pull apply conflict for %s:%s: %s", table, record_id, exc
            )

    def _apply_single_pull_operation(op):
        """Apply a single pulled operation; return (applied, error_message)."""
        table = op.get("table")
        record_id = op.get("record_id")
        operation = op.get("operation")
        data = op.get("data", {})

        try:
            if not table or not record_id or not operation:
                return False, "invalid payload (missing table/record_id/operation)"

            if operation == "delete":
                # Delete merge behavior is not implemented in this CLI path yet.
                return False, "delete pull operations are not supported in CLI sync"

            if operation not in ("upsert", "insert", "update"):
                return False, f"unsupported operation type: {operation}"

            # Upsert known record types
            if table == "episodes" and data:
                from kernle.storage import Episode

                ep = Episode(
                    id=record_id,
                    stack_id=k.stack_id,
                    objective=data.get("objective", ""),
                    outcome_type=data.get("outcome_type", "neutral"),
                    outcome=data.get("outcome", data.get("outcome_description", "")),
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
                return True, None

            if table == "notes" and data:
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
                return True, None

            return False, f"unhandled pull operation for table={table}"

        except Exception as e:
            return False, str(e)

    def _retry_poisoned_pull_operations(poison_records):
        """Retry previously quarantined pull operations."""
        attempted = 0
        recovered = 0

        for key, entry in list(poison_records.items()):
            op = entry.get("operation_payload")
            if not isinstance(op, dict):
                continue

            attempted += 1
            applied, error = _apply_single_pull_operation(op)
            if applied:
                recovered += 1
                poison_records.pop(key, None)
                continue

            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["last_seen_at"] = k._storage._now()
            entry["last_error"] = (error or "unknown error")[:500]
            poison_records[key] = entry

        return attempted, recovered

    def _apply_pull_operations(operations):
        """Apply pulled operations locally.

        Returns:
            tuple[int, list[dict], set[str]]: applied count, failed op details, applied keys.
        """
        applied = 0
        failed_ops = []
        applied_keys = set()

        for op in operations:
            key = _pull_poison_key(op)
            applied_ok, error = _apply_single_pull_operation(op)
            if applied_ok:
                applied += 1
                applied_keys.add(key)
                continue

            logger.debug(
                "Failed to apply pull operation for %s:%s (%s): %s",
                op.get("table"),
                op.get("record_id"),
                op.get("operation"),
                error,
            )
            failed_ops.append(
                {
                    "key": key,
                    "operation": op,
                    "error": (error or "unknown error")[:500],
                }
            )

        return applied, failed_ops, applied_keys

    def _quarantine_failed_pull_operations(failed_ops, poison_records):
        """Store failed pull ops in sync metadata and conflict history."""
        newly_poisoned = 0

        for failed in failed_ops:
            op = failed["operation"]
            key = failed["key"]
            error = failed["error"]
            digest = _serialize_json(op)

            existing = poison_records.get(key)
            first_seen = existing.get("first_seen_at") if existing else k._storage._now()
            attempts = int(existing.get("attempts", 0)) + 1 if existing else 1
            payload_changed = bool(existing and existing.get("payload_digest") != digest)

            poison_records[key] = {
                "table": op.get("table"),
                "record_id": op.get("record_id"),
                "operation": op.get("operation"),
                "attempts": attempts,
                "first_seen_at": first_seen,
                "last_seen_at": k._storage._now(),
                "last_error": error,
                "payload_digest": digest,
                "operation_payload": op,
            }

            if existing is None or payload_changed:
                _save_pull_apply_conflict(op, error)
                newly_poisoned += 1

        return newly_poisoned

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
        operations, skipped_orphans, op_identity_to_change = _build_push_operations(queued_changes)

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
                conflicts = result.get("conflicts", [])
                acknowledged_changes = _resolve_acked_changes(
                    result, operations, op_identity_to_change
                )
                synced = len(acknowledged_changes)

                # Clear synced items from local queue
                with k._storage._connect() as conn:
                    for change in acknowledged_changes:
                        k._storage._clear_queued_change(conn, change.id)
                        k._storage._mark_synced(conn, change.table_name, change.record_id)
                    conn.commit()
                result["synced"] = synced

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

                poison_records = _load_pull_poison_records()
                _, recovered_from_poison = _retry_poisoned_pull_operations(poison_records)
                newly_poisoned = 0

                if operations:
                    # Apply fresh operations locally.
                    applied_now, failed_ops, applied_keys = _apply_pull_operations(operations)
                    for key in applied_keys:
                        poison_records.pop(key, None)
                    newly_poisoned = _quarantine_failed_pull_operations(failed_ops, poison_records)
                    k._storage._set_sync_meta("last_sync_time", k._storage._now())
                else:
                    applied_now = 0
                    failed_ops = []

                _save_pull_poison_records(poison_records)

                applied = applied_now + recovered_from_poison
                conflicts = len(failed_ops)
                poison_pending = len(poison_records)

                if not operations and applied == 0:
                    print("✓ Already up to date")
                    if poison_pending > 0:
                        print(f"⚠️  {poison_pending} pull operations remain quarantined for retry")
                    return

                if args.json:
                    print(
                        json.dumps(
                            {
                                "pulled": applied,
                                "conflicts": conflicts,
                                "has_more": has_more,
                                "poisoned_new": newly_poisoned,
                                "poisoned_pending": poison_pending,
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
                    if poison_pending > 0:
                        print(f"⚠️  {poison_pending} pull operations quarantined for retry")
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
        pulled = 0
        pull_conflicts = 0
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
                pulled_ops = result.get("operations", [])
                poison_records = _load_pull_poison_records()
                _, recovered_from_poison = _retry_poisoned_pull_operations(poison_records)
                if pulled_ops:
                    pulled_now, failed_ops, applied_keys = _apply_pull_operations(pulled_ops)
                    for key in applied_keys:
                        poison_records.pop(key, None)
                    _quarantine_failed_pull_operations(failed_ops, poison_records)
                    pulled = pulled_now + recovered_from_poison
                    pull_conflicts = len(failed_ops)
                    k._storage._set_sync_meta("last_sync_time", k._storage._now())
                else:
                    pulled = recovered_from_poison
                    pull_conflicts = 0
                _save_pull_poison_records(poison_records)
                print(f"  ✓ Pulled {pulled} changes")
                if pull_conflicts > 0:
                    print(f"  ⚠️  {pull_conflicts} conflicts during apply")
                if len(poison_records) > 0:
                    print(f"  ⚠️  {len(poison_records)} pull operations quarantined for retry")
            else:
                print(f"  ⚠️  Pull returned status {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Pull failed: {e}")

        # Step 2: Push local changes
        print("Step 2: Pushing local changes...")
        queued_changes = k._storage.get_queued_changes(limit=1000)

        if not queued_changes:
            print("  ✓ No pending changes to push")
        else:
            operations, skipped_orphans, op_identity_to_change = _build_push_operations(
                queued_changes
            )

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
                    acknowledged_changes = _resolve_acked_changes(
                        result, operations, op_identity_to_change
                    )
                    synced = len(acknowledged_changes)

                    # Clear synced items
                    with k._storage._connect() as conn:
                        for change in acknowledged_changes:
                            k._storage._clear_queued_change(conn, change.id)
                            k._storage._mark_synced(conn, change.table_name, change.record_id)
                        conn.commit()

                    print(f"  ✓ Pushed {synced} changes")
                else:
                    print(f"  ⚠️  Push returned status {response.status_code}")
            except Exception as e:
                print(f"  ⚠️  Push failed: {e}")

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
                if c.resolution == "cloud_wins":
                    resolution_icon = "↓"
                    resolution_text = "cloud wins"
                elif c.resolution == "local_wins":
                    resolution_icon = "↑"
                    resolution_text = "local wins"
                elif c.resolution == "pull_apply_failed":
                    resolution_icon = "!"
                    resolution_text = "pull apply failed"
                else:
                    resolution_icon = "?"
                    resolution_text = c.resolution
                when = c.resolved_at.strftime("%Y-%m-%d %H:%M") if c.resolved_at else "unknown"

                print(f"{resolution_icon} {c.table}:{c.record_id[:8]}... ({resolution_text})")
                print(f"  Resolved: {when}")
                if c.local_summary:
                    print(f'  Local:  "{c.local_summary}"')
                if c.cloud_summary:
                    print(f'  Cloud:  "{c.cloud_summary}"')
                print()

            print("💡 Use `kernle sync conflicts --clear` to clear history")
