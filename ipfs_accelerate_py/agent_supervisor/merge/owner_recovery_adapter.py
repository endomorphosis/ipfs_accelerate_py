"""Bind native merge recovery to separately admitted queue and runtime peers.

Only Git worktrees and recomputable local caches use the bound scratch directory.
Consumer leases, scan cursors and train receipt versions remain in the queue
owner. A failed observation never becomes an empty cursor or missing receipt.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import threading
from typing import Any, Mapping
import uuid

from .owner_merge_queue_adapter import OwnerMergeQueueAdapter

CURSOR_STAGES = frozenset(
    {
        "priority_task_cids",
        "completed_requests",
        "false_completed_requests",
        "false_pending_requests",
        "false_processing_requests",
        "pending_requests",
        "quarantined_requests",
        "processing_requests",
    }
)


class OwnerRecoveryRuntimeError(RuntimeError):
    """The exact admitted recovery runtime cannot perform this operation."""


class OwnerRecoveryCursorConflict(OwnerRecoveryRuntimeError):
    """A reconstructed consumer observed a newer durable cursor revision."""


class OwnerMergeRecoveryRuntime:
    """Native factory binding; this class neither issues grants nor installs schema."""

    def __init__(
        self,
        queue: OwnerMergeQueueAdapter,
        recovery_client: Any,
        *,
        repository_root: Path,
        attempt_root: Path,
        board_namespace: str,
        config_cid: str,
        plan_cid: str,
        lane_id: str,
    ):
        from .checkout_lock import checkout_repository_id
        from .owner_recovery_runtime import OwnerRecoveryRuntimeClient

        if (
            type(queue) is not OwnerMergeQueueAdapter
            or type(recovery_client) is not OwnerRecoveryRuntimeClient
        ):
            raise OwnerRecoveryRuntimeError(
                "native recovery requires the exact admitted clients"
            )
        self.queue = queue
        self.client = recovery_client
        self.repository_root = Path(repository_root).resolve(strict=True)
        self.attempt_root = Path(attempt_root).absolute()
        if self.attempt_root.resolve(strict=False) != self.attempt_root:
            raise OwnerRecoveryRuntimeError(
                "native recovery scratch namespace contains a symlink"
            )
        expected = {
            "board_namespace": board_namespace,
            "config_cid": config_cid,
            "plan_cid": plan_cid,
            "lane_id": lane_id,
            "attempt_root": str(self.attempt_root),
        }
        if not all(type(value) is str and value for value in expected.values()):
            raise OwnerRecoveryRuntimeError(
                "exact native lane/config/plan binding is required"
            )
        if dict(recovery_client.describe_scope()) != expected:
            raise OwnerRecoveryRuntimeError(
                "native recovery namespace differs from owner admission"
            )
        queue_client = queue._client
        if (
            dict(queue_client.connection.identity)
            != dict(recovery_client.connection.identity)
            or queue_client.repository_id
            != checkout_repository_id(self.repository_root)
            or queue_client.repository_id != recovery_client.repository_id
            or queue_client.target_branch != recovery_client.target_branch
            or queue_client.consumer_id != recovery_client.consumer_id
        ):
            raise OwnerRecoveryRuntimeError(
                "queue and recovery peers do not share one exact owner/consumer"
            )
        self._binding = dict(expected)
        self.consumer_id = queue_client.consumer_id
        self.target_branch = queue_client.target_branch
        self.local_state_dir = self.attempt_root / "merge-train-local-state"
        self._state_lock = threading.RLock()
        self._consumer_lock = threading.RLock()
        self._lease: dict[str, Any] | None = None
        self._lease_depth = 0
        self._lease_thread_id: int | None = None
        self._pending_acquire_operation_id: str | None = None
        self._pending_release_operation_id: str | None = None
        self._lease_uncertain = False
        self._cursors: dict[str, Any] | None = None
        self._receipt_heads: dict[str, Mapping[str, Any] | None] = {}
        queue.bind_recovery_runtime(self)

    def validate_factory_binding(
        self,
        *,
        repository_root,
        attempt_root,
        board_namespace,
        lane_id,
        target_branch,
        admitted_config_cid,
        admitted_plan_cid,
    ):
        if (
            Path(repository_root).resolve(strict=True) != self.repository_root
            or Path(attempt_root).absolute() != self.attempt_root
            or board_namespace != self._binding["board_namespace"]
            or str(lane_id) != self._binding["lane_id"]
            or target_branch != self.target_branch
            or admitted_config_cid != self._binding["config_cid"]
            or admitted_plan_cid != self._binding["plan_cid"]
            or dict(self.client.describe_scope()) != self._binding
        ):
            raise OwnerRecoveryRuntimeError(
                "factory differs from its admitted native recovery namespace"
            )

    def validate_train(
        self, queue, *, repository_root, target_branch, owner_id, state_dir
    ):
        if (
            queue is not self.queue
            or Path(repository_root).resolve(strict=True) != self.repository_root
            or target_branch != self.target_branch
            or owner_id not in (None, self.consumer_id)
            or (
                state_dir is not None
                and Path(state_dir).absolute() != self.local_state_dir
            )
        ):
            raise OwnerRecoveryRuntimeError(
                "merge train differs from admitted owner runtime"
            )
        if dict(self.client.describe_scope()) != self._binding:
            raise OwnerRecoveryRuntimeError("owner recovery namespace changed")

    @staticmethod
    def _cursor_map(value):
        if (
            not isinstance(value, Mapping)
            or set(value) != CURSOR_STAGES
            or any(
                type(item) is not str
                or len(item.encode("utf-8")) > 4096
                or any(ord(char) < 32 for char in item)
                for item in value.values()
            )
        ):
            raise OwnerRecoveryRuntimeError("owner cursor map is malformed")
        return dict(value)

    def load_cursors(self):
        with self._state_lock:
            current = dict(self.client.load_cursors())
            values = self._cursor_map(current.get("cursors"))
            if (
                type(current.get("revision")) is not int
                or current["revision"] < 0
                or not current.get("state_cid")
            ):
                raise OwnerRecoveryRuntimeError("owner cursor revision is unavailable")
            self._cursors = {**current, "cursors": values}
            return dict(values)

    def save_cursors(self, cursors):
        values = self._cursor_map(cursors)
        with self._state_lock:
            if self._cursors is None:
                self.load_cursors()
            current = self._cursors
            if values == current["cursors"]:
                return
            result = dict(
                self.client.cas_cursors(
                    expected_revision=current["revision"],
                    expected_state_cid=current["state_cid"],
                    cursors=values,
                    operation_id=uuid.uuid4().hex,
                )
            )
            if result.get("conflict") is True:
                self._cursors = None
                raise OwnerRecoveryCursorConflict(
                    "durable recovery cursor revision changed"
                )
            if self._cursor_map(result.get("cursors")) != values:
                self._cursors = None
                raise OwnerRecoveryRuntimeError("owner cursor post-state differs")
            self._cursors = result

    @contextmanager
    def consumer_lease(self):
        if not self._consumer_lock.acquire(blocking=False):
            yield False
            return
        outer = self._lease_depth == 0
        acquired = False
        try:
            if self._lease_uncertain:
                raise OwnerRecoveryRuntimeError(
                    "unknown merge callback retains its owner consumer lease"
                )
            if outer:
                if self._pending_release_operation_id is not None:
                    # The previous synchronous callback returned normally.
                    # Finish only its exact release before admitting new work.
                    self._release_completed_callback()
                if self._pending_acquire_operation_id is None:
                    self._pending_acquire_operation_id = uuid.uuid4().hex
                result = dict(
                    self.client.acquire_consumer_lease(
                        operation_id=self._pending_acquire_operation_id, ttl_seconds=300
                    )
                )
                if result.get("acquired") is not True:
                    if result.get("acquired") is not False:
                        raise OwnerRecoveryRuntimeError(
                            "owner consumer lease admission is unavailable"
                        )
                    self._pending_acquire_operation_id = None
                    yield False
                    return
                if (
                    type(result.get("lease_id")) is not str
                    or not result["lease_id"]
                    or type(result.get("fence_epoch")) is not int
                    or result["fence_epoch"] < 1
                ):
                    raise OwnerRecoveryRuntimeError(
                        "owner consumer acquisition identity is unavailable"
                    )
                self._pending_acquire_operation_id = None
                self._lease = result
                self._lease_thread_id = threading.get_ident()
            self._lease_depth += 1
            acquired = True
            try:
                yield True
            except BaseException:
                # Do not infer callback closure or make the lease stealable.
                self._lease_uncertain = True
                raise
            finally:
                self._lease_depth -= 1
            if outer:
                if self._lease_uncertain:
                    raise OwnerRecoveryRuntimeError(
                        "nested unknown callback retains its owner consumer lease"
                    )
                self._pending_release_operation_id = uuid.uuid4().hex
                self._release_completed_callback()
        finally:
            if outer and acquired and self._lease_depth != 0:
                self._lease_uncertain = True
            self._consumer_lock.release()

    def _release_completed_callback(self):
        # No callback is entered while this exact release response is pending.
        # Failure preserves the lease and operation ID for the same runtime;
        # a new owner/client still needs the backend's current custody checks.
        if (
            self._lease_uncertain
            or self._lease is None
            or self._pending_release_operation_id is None
        ):
            raise OwnerRecoveryRuntimeError(
                "no known completed callback release is pending"
            )
        lease = self._lease
        result = self.client.release_consumer_lease(
            lease_id=lease["lease_id"],
            fence_epoch=lease["fence_epoch"],
            operation_id=self._pending_release_operation_id,
        )
        if (
            result.get("released") is not True
            or result.get("lease_id") != lease["lease_id"]
            or result.get("fence_epoch") != lease["fence_epoch"]
        ):
            raise OwnerRecoveryRuntimeError("owner consumer release was not admitted")
        self._pending_release_operation_id = None
        self._lease = None
        self._lease_thread_id = None

    def _held_lease(self):
        if (
            self._lease is None
            or self._lease_uncertain
            or self._lease_depth < 1
            or self._lease_thread_id != threading.get_ident()
        ):
            raise OwnerRecoveryRuntimeError(
                "receipt publication requires the exact active train lease"
            )
        lease = dict(self._lease)
        result = self.client.renew_consumer_lease(
            lease_id=lease["lease_id"],
            fence_epoch=lease["fence_epoch"],
            ttl_seconds=300,
            operation_id=uuid.uuid4().hex,
        )
        if (
            result.get("renewed") is not True
            or result.get("lease_id") != lease["lease_id"]
            or result.get("fence_epoch") != lease["fence_epoch"]
        ):
            self._lease_uncertain = True
            raise OwnerRecoveryRuntimeError("owner consumer renewal was not admitted")
        return lease

    def read_receipt(self, key):
        value = self.read_optional_receipt(key)
        return {} if value is None else value

    def diagnostics(self):
        """Local recovery progress only; this is not an owner settlement proof."""
        if self._lease_uncertain:
            state = "unknown_callback_custody_retained"
        elif self._pending_release_operation_id is not None:
            state = "exact_release_reply_pending"
        elif self._pending_acquire_operation_id is not None:
            state = "exact_pre_callback_acquire_reply_pending"
        elif self._lease is not None:
            state = "local_consumer_callback_active"
        else:
            state = "no_local_consumer_lease"
        return {
            "consumer_custody": state,
            "recovery_scope_cid": self.client.recovery_scope_cid,
            "completion_authority": False,
            "settlement_authority": False,
        }

    def read_optional_receipt(self, key):
        with self._state_lock:
            head = self.client.get_receipt(key)
            self._receipt_heads[key] = head
            if head is None:
                return None
            receipt = head.get("receipt")
            if not isinstance(receipt, Mapping):
                raise OwnerRecoveryRuntimeError("owner train receipt is malformed")
            return dict(receipt)

    def write_receipt(self, key, payload):
        # Preserve existing JSON values, but never stringify unsupported objects.
        value = json.loads(json.dumps(dict(payload), allow_nan=False))
        with self._state_lock:
            lease = self._held_lease()
            if key not in self._receipt_heads:
                self.read_receipt(key)
            current = self._receipt_heads[key]
            result = self.client.publish_receipt(
                key,
                value,
                expected_revision=0 if current is None else current["revision"],
                expected_receipt_cid="" if current is None else current["receipt_cid"],
                lease_id=lease["lease_id"],
                fence_epoch=lease["fence_epoch"],
                operation_id=uuid.uuid4().hex,
            )
            if result.get("conflict") is True:
                self._receipt_heads.pop(key, None)
                raise OwnerRecoveryRuntimeError(
                    "owner receipt head changed before publication"
                )
            head = result.get("head")
            if not isinstance(head, Mapping) or head.get("receipt") != value:
                raise OwnerRecoveryRuntimeError(
                    "owner receipt publication post-state differs"
                )
            self._receipt_heads[key] = dict(head)
            return head["receipt_cid"]
