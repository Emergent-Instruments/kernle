"""Synchronization operations for Kernle."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SyncMixin:
    """Sync operations for Kernle."""

    def sync(self) -> Dict[str, Any]:
        """Sync local changes with cloud storage.

        Returns:
            Sync results including counts and any errors
        """
        result = self._storage.sync()
        return {
            "pushed": result.pushed,
            "pulled": result.pulled,
            "conflicts": result.conflicts,
            "errors": result.errors,
            "success": result.success,
        }

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status.

        Returns:
            Sync status including pending count and connectivity
        """
        return {
            "pending": self._storage.get_pending_sync_count(),
            "online": self._storage.is_online(),
        }

    def _sync_before_load(self) -> Dict[str, Any]:
        """Pull remote changes before loading local state.

        Called automatically by load() when auto_sync is enabled.
        Non-blocking: logs errors but doesn't fail the load.

        Returns:
            Dict with pull result or error info
        """
        result = {
            "attempted": False,
            "pulled": 0,
            "conflicts": 0,
            "errors": [],
        }

        try:
            # Check if sync is available
            if not self._storage.is_online():
                logger.debug("Sync before load: offline, skipping pull")
                return result

            result["attempted"] = True
            pull_result = self._storage.pull_changes()
            result["pulled"] = pull_result.pulled
            result["conflicts"] = pull_result.conflicts
            result["errors"] = pull_result.errors

            if pull_result.pulled > 0:
                logger.info(f"Sync before load: pulled {pull_result.pulled} changes")
            if pull_result.errors:
                logger.warning(
                    f"Sync before load: {len(pull_result.errors)} errors: {pull_result.errors[:3]}"
                )

        except Exception as e:
            # Don't fail the load on sync errors
            logger.warning(f"Sync before load failed (continuing with local data): {e}")
            result["errors"].append(str(e))

        return result

    def _sync_after_checkpoint(self) -> Dict[str, Any]:
        """Push local changes after saving a checkpoint.

        Called automatically by checkpoint() when auto_sync is enabled.
        Non-blocking: logs errors but doesn't fail the checkpoint save.

        Returns:
            Dict with push result or error info
        """
        result = {
            "attempted": False,
            "pushed": 0,
            "conflicts": 0,
            "errors": [],
        }

        try:
            # Check if sync is available
            if not self._storage.is_online():
                logger.debug("Sync after checkpoint: offline, changes queued for later")
                result["errors"].append("Offline - changes queued")
                return result

            result["attempted"] = True
            sync_result = self._storage.sync()
            result["pushed"] = sync_result.pushed
            result["conflicts"] = sync_result.conflicts
            result["errors"] = sync_result.errors

            if sync_result.pushed > 0:
                logger.info(f"Sync after checkpoint: pushed {sync_result.pushed} changes")
            if sync_result.errors:
                logger.warning(
                    f"Sync after checkpoint: {len(sync_result.errors)} errors: {sync_result.errors[:3]}"
                )

        except Exception as e:
            # Don't fail the checkpoint on sync errors
            logger.warning(f"Sync after checkpoint failed (local save succeeded): {e}")
            result["errors"].append(str(e))

        return result
