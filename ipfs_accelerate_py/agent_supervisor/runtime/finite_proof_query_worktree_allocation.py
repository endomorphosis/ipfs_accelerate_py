"""Native allocation custody for the finite proof-query coding boundary.

The private Portal task CID is derived from the owner-verified projection. It
is never substituted for the native task CID. A serialized allocation cannot
authorize dispatch. This module reads current ownership; it creates no claim,
worktree, database schema, proof, or process.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import time

from ..merge import worktree_lifecycle as lifecycle
from ..planning import finite_integer_source_custody as custody
from ..task_sources.control_plane_contracts import canonical_json_bytes
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

SCHEMA = "finite-proof-query-native-worktree-allocation@1"
_SEAL = object()
_FALSE = {name: False for name in ("proof_authority", "completion_authority",
    "convergence_proved", "process_origin_attested", "independent_attempt_lease_checked")}


class FiniteProofQueryAllocationError(ValueError):
    """The exact current native allocation could not be authenticated."""


def _need(value, message):
    if not value:
        raise FiniteProofQueryAllocationError(message)


def _native_lifecycle_bytes(value):
    """Bind native float timestamps without changing control-plane encoding."""
    _need(type(value) is dict, "exact native lifecycle record required")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise FiniteProofQueryAllocationError(
            "native lifecycle record cannot be serialized") from error
    _need(len(raw) <= 262_144, "native lifecycle record exceeds custody bound")
    return raw


def _regular(path, checkpoint, *, bound=262_144):
    metadata, raw = custody._read(Path(path), role="finite-proof-query:allocation",
                                 bound=bound, checkpoint=checkpoint)
    actual = Path(path).lstat()
    _need(actual.st_uid == os.geteuid() and actual.st_nlink == 1,
          "native allocation metadata is not an owner regular file")
    return metadata, raw


def _native_projection(broker, request, claim, checkpoint):
    """Rebuild the exact sealed Portal binding from the owner's current row."""
    from ..task_sources.intent_repository import IntentRepository
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..task_sources.board_control_plane import infer_board_namespace
    from ..todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, DatabaseTaskAttempt

    checkpoint()
    scope, runtime = broker._scope, broker._runtime
    repository = scope._owner.repository
    _head, raw = _regular(repository / ".git" / "HEAD", checkpoint, bound=4096)
    prefix = "ref: refs/heads/"
    text = raw.decode("ascii").strip()
    _need(text.startswith(prefix) and len(text) > len(prefix),
          "native allocation requires the canonical named baseline branch")
    target = text[len(prefix):]
    source = DatabaseTaskSource(intent=IntentRepository(
        bound_connection=broker._server._connection, install_schema=False), install_schema=False)
    record = source.get_task(request["database_task_cid"])
    _need(record is not None and record.status == "in_progress"
          and record.revision == claim["task_revision"],
          "native allocation requires its current admitted task")
    receipt = record.body.get("completion_receipt", {})
    attempt = DatabaseTaskAttempt(task_cid=record.task_cid, task_alias=record.task_alias,
        **{key: receipt[key] for key in ("attempt_id", "claim_id", "attempt_number",
            "owner_session_id", "fencing_token", "fence_epoch", "lease_id")},
        committed_phase=receipt["attempt_execution_phase"], status="running",
        started_at_ms=0, revision=receipt["attempt_execution_revision"], body={})
    attempt = replace(attempt, body={"control_binding":
        DatabaseImplementationDaemon._control_claim_binding(attempt, record)})

    def no_provider(*_args):
        raise FiniteProofQueryAllocationError("allocation custody cannot dispatch a provider")

    bridge = DatabasePortalExecutionBridge(task_source=source,
        attempt_root=Path(runtime.state) / "run" / "admitted_database_portal_attempts",
        portal_factory=no_provider, repo_root=repository,
        board_namespace=infer_board_namespace(merge_target_branch=target,
            todo_path=Path(broker._server.config.database_path), state_prefix="admitted"),
        merge_target_branch=target, task_header_prefix="## " + record.task_alias)
    paths = bridge._paths(attempt)
    bridge._seal_attempt_directory(paths, attempt_id=attempt.attempt_id, create=False)
    _regular(paths.binding, checkpoint)
    binding = bridge._strict_binding(paths.binding)
    expected = bridge._binding(attempt, record, bridge._render_projection(attempt, record),
                               schema=binding["schema"])
    _need(binding == expected, "native allocation Portal binding differs from the native claim")
    _regular(paths.task_projection, checkpoint)
    identity = bridge._prior_projection_identity(paths, binding)
    _need(identity["task_id"] == request["task_id"],
          "native allocation Portal identity differs from the native alias")
    return paths, binding, identity, target


class FrozenFiniteProofQueryWorktreeAllocation:
    """Exact owner-created native allocation; mappings and subclasses are inert."""

    def __init__(self, seal, *, broker, request, claim, peer_identity, checkpoint):
        from .finite_proof_query_worker_dispatch import OwnerFiniteProofQueryDispatchBroker
        _need(seal is _SEAL and type(self) is FrozenFiniteProofQueryWorktreeAllocation
              and type(broker) is OwnerFiniteProofQueryDispatchBroker,
              "native allocation requires its exact live owner broker")
        broker._identity_current()
        self._seal, self._broker, self._scope = seal, broker, broker._scope
        self._request = json.loads(canonical_json_bytes(request))
        self._claim = json.loads(canonical_json_bytes(claim))
        self._peer = tuple(peer_identity)
        self._request_bytes = canonical_json_bytes(self._request)
        from .finite_proof_query_worker_dispatch import _stable_native_dispatch_claim
        self._claim_bytes = canonical_json_bytes(_stable_native_dispatch_claim(self._claim))
        self._initial_claim_bytes = canonical_json_bytes(self._claim)
        self._last_grant_expiry = self._claim["grant_expires_at_ms"]
        self._paths, binding, identity, target = _native_projection(
            broker, self._request, self._claim, checkpoint)
        self._binding_bytes = canonical_json_bytes(binding)
        self._portal_identity_bytes = canonical_json_bytes(identity)
        repository = Path(self._scope._owner.repository)
        self._store = lifecycle.WorktreeLifecycleStore(
            repo_root=repository, store_dir=repository / ".git" / "agent-worktree-lifecycle")
        self._worktree = Path(self._request["worktree"])
        self._target = target
        record = self._read_lifecycle(checkpoint)
        self._frozen_lifecycle = record.to_dict()
        self._last_fence = record.fence
        self._last_updated_at, self._last_expiry = record.updated_at, record.expires_at
        self._fields = (broker, broker._scope, broker._runtime, broker._server,
            broker._server._connection, self._paths, self._store, self._worktree,
            self._request_bytes, self._initial_claim_bytes, self._binding_bytes,
            self._portal_identity_bytes, _native_lifecycle_bytes(self._frozen_lifecycle))
        material = {"schema": SCHEMA, "repository_root": str(repository),
            "baseline_source_commit": self._scope._custody._manifest.snapshot.git_commit,
            "worktree_path": str(self._worktree), "branch": record.branch,
            "merge_target": target, "database_task_cid": claim["task_cid"],
            "database_attempt_id": claim["attempt_id"], "database_claim_id": claim["claim_id"],
            "database_attempt_number": claim["attempt_number"],
            "portal_canonical_task_cid": identity["canonical_task_cid"],
            "portal_projection_identity_cid": cid_for_structured(identity),
            "portal_binding_id": binding["binding_id"],
            "lifecycle_record_id": record.record_id, "lifecycle_attempt": record.attempt,
            "lifecycle_initial_fence": record.fence,
            "lifecycle_lease_identity_cid": cid_for_structured({"lease_id": record.lease_id}),
            "lifecycle_owner_birth": record.owner.to_dict(),
            "native_worktree_lease_checked": True, **_FALSE}
        material["allocation_cid"] = cid_for_structured(material)
        self._material = canonical_json_bytes(material)
        self._original_material = self._material
        self._fields += (self._material,)
        self.require_current(checkpoint=checkpoint)

    @property
    def material_binding(self):
        return json.loads(self._material)

    def _read_lifecycle(self, checkpoint):
        checkpoint()
        path = self._worktree
        root = Path(self._broker._runtime.manifest.get("worker_worktree_root")
                    or self._broker._runtime.state / "worktrees")
        _need(root.is_absolute() and root.resolve() == root and path.is_absolute()
              and path.resolve() == path and path != root and path.is_relative_to(root)
              and path.is_dir() and not path.is_symlink(),
              "native allocation path differs from the bound worker root")
        _regular(path / ".git", checkpoint, bound=4096)
        _regular(self._store.workspace_path_for(path), checkpoint)
        record = self._store._load_strict_workspace_record(path)
        identity = json.loads(self._portal_identity_bytes)
        prefix = (1 << 52) | (self._claim["attempt_number"] * (1 << 16))
        local_attempt = record.attempt - prefix
        _need(record.state is lifecycle.WorkspaceLifecycleState.ACTIVE
              and record.record_id == record.compute_record_id()
              and record.task_id == self._request["task_id"]
              and record.canonical_task_cid == identity["canonical_task_cid"]
              and 1 <= local_attempt < (1 << 16) and record.attempt == prefix | local_attempt
              and record.workspace_path == str(path)
              and record.repo_root == str(self._scope._owner.repository)
              and record.state_dir == str(self._paths.root)
              and record.lane_id.startswith(str(self._paths.root) + ":")
              and record.merge_target == self._target and not record.terminal_reason,
              "native allocation lifecycle differs from its exact Portal task and attempt")
        birth = lifecycle.read_process_birth(self._peer[0])
        _need(birth is not None and birth == record.owner
              and birth.start_time_ticks == self._peer[2]
              and os.stat("/proc/" + str(birth.pid)).st_uid == self._peer[1]
              and record.expires_at > time.time() + 5.0,
              "native allocation owner birth or live worktree lease differs")
        index = self._store.task_index_path_for(canonical_task_cid=record.canonical_task_cid,
            task_id=record.task_id, attempt=record.attempt)
        _regular(index, checkpoint)
        self._store._require_exact_task_index(record, index_path=index)
        return record

    def require_current(self, *, checkpoint):
        from .finite_proof_query_worker_dispatch import (
            _require_native_dispatch_claim, _stable_native_dispatch_claim,
        )
        _need(type(self) is FrozenFiniteProofQueryWorktreeAllocation and self._seal is _SEAL,
              "exact sealed native allocation required")
        broker = self._broker
        broker._identity_current()
        current = (broker, broker._scope, broker._runtime, broker._server,
            broker._server._connection, self._paths, self._store, self._worktree,
            canonical_json_bytes(self._request), canonical_json_bytes(self._claim),
            self._binding_bytes, self._portal_identity_bytes,
            _native_lifecycle_bytes(self._frozen_lifecycle), self._material)
        _need(len(current) == len(self._fields)
              and all(a is b if n < 7 else a == b for n, (a, b) in enumerate(zip(current, self._fields)))
              and self._claim_bytes == canonical_json_bytes(_stable_native_dispatch_claim(self._claim))
              and self._scope is broker._scope and self._material == self._original_material
              and self._store.repo_root == self._scope._owner.repository
              and self._store.store_dir == self._scope._owner.repository / ".git" / "agent-worktree-lifecycle",
              "native allocation cached ownership was rebound")
        claim = _require_native_dispatch_claim(server=broker._server,
            expected_population=broker._population, request=self._request,
            peer_identity=self._peer, client_id=broker._runtime.client_id,
            store_id=broker._runtime.owner_identity.store_id)
        _need(canonical_json_bytes(_stable_native_dispatch_claim(claim)) == self._claim_bytes
              and claim["grant_expires_at_ms"] >= self._last_grant_expiry
              and claim["grant_expires_at_ms"] >= self._claim["grant_expires_at_ms"],
              "native allocation claim or active grant changed")
        self._last_grant_expiry = claim["grant_expires_at_ms"]
        _regular(self._paths.binding, checkpoint)
        from ..todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
        binding = DatabasePortalExecutionBridge._strict_binding(self._paths.binding)
        _need(canonical_json_bytes(binding) == self._binding_bytes,
              "native allocation sealed Portal binding changed")
        _regular(self._paths.task_projection, checkpoint)
        DatabasePortalExecutionBridge._verify_projection(self._paths, binding)
        record = self._read_lifecycle(checkpoint)
        actual, expected = record.to_dict(), dict(self._frozen_lifecycle)
        for key in ("fence", "updated_at", "expires_at"):
            actual.pop(key)
            expected.pop(key)
        _need(_native_lifecycle_bytes(actual) == _native_lifecycle_bytes(expected)
              and record.fence >= self._last_fence
              and record.fence >= self._frozen_lifecycle["fence"]
              and record.updated_at >= self._last_updated_at
              and record.updated_at >= self._frozen_lifecycle["updated_at"]
              and record.expires_at >= self._last_expiry
              and record.expires_at >= self._frozen_lifecycle["expires_at"],
              "native allocation changed ownership or rolled back its worktree lease")
        self._last_fence, self._last_updated_at, self._last_expiry = (
            record.fence, record.updated_at, record.expires_at)
        return self.material_binding


def _capture_owner_finite_proof_query_worktree_allocation(*, broker, request, claim,
                                                        peer_identity, checkpoint):
    return FrozenFiniteProofQueryWorktreeAllocation(_SEAL, broker=broker, request=request,
        claim=claim, peer_identity=peer_identity, checkpoint=checkpoint)


__all__ = ["FiniteProofQueryAllocationError", "FrozenFiniteProofQueryWorktreeAllocation"]
