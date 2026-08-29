"""Interruption recovery and idempotent resume (PCCE-024).

Persists and replays lifecycle checkpoints so resume after interruption
reuses completed valid stages, repairs ambiguous effects, and never
reinvokes a terminal adapter or publishes through an expired fence.

This module coordinates checkpoints over an injected kit-facing
generation-fenced CAS. It does not implement a write-ahead log and does
not infer success from process exit. Importing this module performs no
I/O, network, process, or filesystem mutation and does not bind a model
provider or search sibling checkouts.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.proof_context.errors import (
    ERRORS,
    BoundaryViolationError,
    MalformedError,
    ProofContextError,
    SchemaMismatchError,
    UnknownFieldError,
)
from ipfs_accelerate_py.proof_context.lifecycle import (
    APPLY_STAGE,
    CHECKPOINT_SCHEMA,
    COMPATIBILITY_MATRIX_CONTENT_ID,
    CONTRACT_VERSION,
    DISPOSITION_STAGE,
    LIFECYCLE_CID,
    LIFECYCLE_RECORD_SCHEMA,
    PCCE_006_CONTENT_ID,
    SEAL_STAGE,
    STAGE_ARTIFACT_SCHEMA,
    STAGES,
    VERIFY_STAGE,
    LifecycleIdentities,
    LifecyclePorts,
    LifecycleRecord,
    PatchLifecycle,
    StageArtifact,
    merge_identities,
)
from ipfs_accelerate_py.proof_context.policy import (
    LIVE_MODES,
    MODES,
    POLICY_CID,
    PolicyError,
    admit_cid,
    admit_mode,
)
from ipfs_accelerate_py.proof_context.results import (
    RESULT_STATE_CID,
    STATUSES,
    admit_status,
    is_terminal,
)

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
RECOVERY_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/recovery"
CHECKPOINT_RECORD_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/recovery-checkpoint"
RECOVERY_RECORD_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/recovery-record"
REPAIR_RECEIPT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/repair-receipt"
IDEMPOTENCY_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/idempotency-key"
CONTRACT_SCHEMA_PREFIX: Final[str] = "pcce/proof-context/v0.1/"
PROVIDER_BOUND: Final[bool] = False
SIBLING_LAYOUT_REQUIRED: Final[bool] = False
SOLE_LIFECYCLE_AUTHORITY: Final[bool] = True
SECOND_WAL: Final[bool] = False
WAL_IMPLEMENTATION: Final[None] = None
INFER_SUCCESS_FROM_PROCESS_EXIT: Final[bool] = False
PERSISTENCE_AUTHORITY: Final[str] = "injected-kit-checkpoint-store"
RESTART_AWARE: Final[bool] = True

CRASH_POSITIONS: Final[tuple[str, ...]] = ("before", "during", "after")
EFFECTFUL_STAGES: Final[tuple[str, ...]] = (APPLY_STAGE, VERIFY_STAGE, SEAL_STAGE)
TERMINAL_ADAPTER_STAGES: Final[tuple[str, ...]] = (SEAL_STAGE, DISPOSITION_STAGE)
LEASE_BOUNDARY: Final[str] = "acquire-lease"
FENCE_BOUNDARY: Final[str] = "acquire-fence"
PUBLISH_BOUNDARY: Final[str] = "publish"
BOUNDARIES: Final[tuple[str, ...]] = (LEASE_BOUNDARY, FENCE_BOUNDARY, *STAGES, PUBLISH_BOUNDARY)
CRASH_MATRIX: Final[tuple[tuple[str, str], ...]] = tuple(
    (stage, position) for stage in EFFECTFUL_STAGES for position in CRASH_POSITIONS
)
VALID_TERMINAL_STATUSES: Final[tuple[str, ...]] = tuple(
    status for status in STATUSES if is_terminal(status)
) + ("repair_required",)

_BYPASS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "skip",
        "skip_stages",
        "start_at",
        "bypass",
        "bypass_stages",
        "self_approved",
        "adapter_approved",
    }
)


class RecoveryError(ProofContextError):
    """Fail-closed recovery error. Never claims publication."""

    code = "repair_required"


class StaleWriterError(RecoveryError):
    """A fenced-out or generation-lagged writer cannot persist or publish."""

    code = "stale_root"


class CrashInterrupt(BaseException):
    """Simulated process death at a lifecycle boundary.

    This is not a Python exception the coordinator may translate into
    success. Process exit is never evidence that a stage completed.
    """

    def __init__(self, stage: str, position: str) -> None:
        super().__init__(f"crash {position} {stage}")
        self.stage = stage
        self.position = position


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze(item) for item in sorted(value, key=repr))
    return value


def _canonicalize(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        parts = []
        for key in sorted(str(item) for item in value):
            parts.append(
                json.dumps(str(key), ensure_ascii=False, separators=(",", ":"))
                + ":"
                + _canonicalize(value[key] if key in value else value[str(key)])
            )
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonicalize(item) for item in value) + "]"
    raise RecoveryError(
        f"unsupported recovery canonicalization type {type(value).__name__}",
        code="malformed",
    )


def mint_recovery_cid(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonicalize(value).encode("utf-8")).digest()
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def _admit_recovery_cid(value: str) -> str:
    try:
        return admit_cid(value)
    except PolicyError as exc:
        raise RecoveryError(
            str(exc),
            code=exc.reason if exc.reason in ERRORS else "pseudo_cid",
        ) from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise MalformedError("payload must be a mapping")


def admit_position(position: Any) -> str:
    if not isinstance(position, str) or position not in CRASH_POSITIONS:
        raise UnknownFieldError(f"unknown crash position {position!r}")
    return position


def admit_boundary(boundary: Any) -> str:
    if not isinstance(boundary, str) or boundary not in BOUNDARIES:
        raise UnknownFieldError(f"unknown recovery boundary {boundary!r}")
    return boundary


def mint_idempotency_key(
    *,
    attempt_id: str,
    run_id: str,
    trace_id: str,
    stage: str,
    position: str,
    inbound_cid: str | None,
    generation: int,
) -> str:
    """Content-addressed idempotency key for one lifecycle boundary."""

    admit_position(position)
    if stage not in BOUNDARIES:
        raise UnknownFieldError(f"unknown recovery boundary {stage!r}")
    return mint_recovery_cid(
        {
            "schema": IDEMPOTENCY_SCHEMA,
            "attempt_id": attempt_id,
            "run_id": run_id,
            "trace_id": trace_id,
            "stage": stage,
            "position": position,
            "inbound_cid": inbound_cid or "",
            "generation": generation,
        }
    )


def _default_clock() -> int:
    return 0


@dataclass(frozen=True)
class AttemptIdentity:
    """Supervisor attempt binding for fenced recovery."""

    attempt_id: str
    writer_id: str
    writer_generation: int
    fence_token: str
    lease_id: str
    fence_id: str
    identities: LifecycleIdentities
    lease_expires_at: int = 2147483647

    def __post_init__(self) -> None:
        for name in (
            "attempt_id",
            "writer_id",
            "fence_token",
            "lease_id",
            "fence_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MalformedError(f"attempt field {name} is required")
        if not isinstance(self.writer_generation, int) or self.writer_generation < 1:
            raise MalformedError("writer_generation must be a positive integer")
        if not isinstance(self.lease_expires_at, int):
            raise MalformedError("lease_expires_at must be an integer timestamp")
        if not isinstance(self.identities, LifecycleIdentities):
            if type(self.identities).__name__ != "LifecycleIdentities":
                raise MalformedError("attempt identities must be LifecycleIdentities")

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "attempt_id": self.attempt_id,
                "writer_id": self.writer_id,
                "writer_generation": self.writer_generation,
                "fence_token": self.fence_token,
                "lease_id": self.lease_id,
                "fence_id": self.fence_id,
                "lease_expires_at": self.lease_expires_at,
                "identities": dict(self.identities.to_mapping()),
            }
        )


@runtime_checkable
class CheckpointStore(Protocol):
    """Kit-facing generation-fenced CAS for recovery checkpoints. Not a WAL."""

    def put(
        self,
        record: Mapping[str, Any],
        *,
        writer_id: str,
        generation: int,
        fence_token: str,
    ) -> str: ...

    def latest(self, attempt_id: str) -> Mapping[str, Any] | None: ...

    def history(self, attempt_id: str) -> Sequence[Mapping[str, Any]]: ...

    def current_generation(self, attempt_id: str) -> int | None: ...

    def current_token(self, attempt_id: str) -> str | None: ...

    def current_writer(self, attempt_id: str) -> str | None: ...

    def record_invocation(self, attempt_id: str, stage: str) -> int: ...

    def invocation_count(self, attempt_id: str, stage: str) -> int: ...

    def reclaim(
        self,
        attempt_id: str,
        *,
        writer_id: str,
        fence_token: str,
    ) -> int: ...


class FencedCheckpointStore:
    """In-process generation-fenced checkpoint CAS.

    Snapshots are content-addressed and idempotent by key. There is no
    operation log, LSN, truncate, or WAL replay. Production durability is
    the injected kit store this type stands in for during hermetic tests.
    """

    second_wal = False

    def __init__(self) -> None:
        self._history: dict[str, list[Mapping[str, Any]]] = {}
        self._by_key: dict[str, Mapping[str, Any]] = {}
        self._generation: dict[str, int] = {}
        self._token: dict[str, str] = {}
        self._writer: dict[str, str] = {}
        self._invocations: dict[tuple[str, str], int] = {}

    def put(
        self,
        record: Mapping[str, Any],
        *,
        writer_id: str,
        generation: int,
        fence_token: str,
    ) -> str:
        payload = _as_mapping(record)
        attempt_id = str(payload.get("attempt_id") or "")
        if not attempt_id:
            raise MalformedError("checkpoint attempt_id is required")
        key = str(payload.get("idempotency_key") or "")
        if not key:
            raise MalformedError("checkpoint idempotency_key is required")
        _admit_recovery_cid(key)
        if attempt_id not in self._generation:
            self._generation[attempt_id] = generation
            self._token[attempt_id] = fence_token
            self._writer[attempt_id] = writer_id
        if generation != self._generation[attempt_id]:
            raise StaleWriterError(
                "stale writer generation cannot persist a checkpoint",
                details={"reason": "generation"},
            )
        if fence_token != self._token[attempt_id]:
            raise StaleWriterError(
                "expired or replaced fence cannot persist a checkpoint",
                details={"reason": "fence_token"},
            )
        if writer_id != self._writer[attempt_id]:
            raise StaleWriterError(
                "stale writer identity cannot persist a checkpoint",
                details={"reason": "writer_id"},
            )
        existing = self._by_key.get(key)
        if existing is not None:
            return str(existing["checkpoint_cid"])
        body = {str(name): payload[name] for name in payload}
        checkpoint_cid = mint_recovery_cid(
            {name: body[name] for name in body if name != "checkpoint_cid"}
        )
        stored = MappingProxyType({**body, "checkpoint_cid": checkpoint_cid})
        self._by_key[key] = stored
        self._history.setdefault(attempt_id, []).append(stored)
        return checkpoint_cid

    def latest(self, attempt_id: str) -> Mapping[str, Any] | None:
        records = self._history.get(attempt_id) or []
        return records[-1] if records else None

    def history(self, attempt_id: str) -> Sequence[Mapping[str, Any]]:
        return tuple(self._history.get(attempt_id) or ())

    def current_generation(self, attempt_id: str) -> int | None:
        return self._generation.get(attempt_id)

    def current_token(self, attempt_id: str) -> str | None:
        return self._token.get(attempt_id)

    def current_writer(self, attempt_id: str) -> str | None:
        return self._writer.get(attempt_id)

    def record_invocation(self, attempt_id: str, stage: str) -> int:
        key = (attempt_id, stage)
        self._invocations[key] = self._invocations.get(key, 0) + 1
        return self._invocations[key]

    def invocation_count(self, attempt_id: str, stage: str) -> int:
        return self._invocations.get((attempt_id, stage), 0)

    def reclaim(
        self,
        attempt_id: str,
        *,
        writer_id: str,
        fence_token: str,
    ) -> int:
        if not writer_id or not fence_token:
            raise MalformedError("reclaim requires writer_id and fence_token")
        new_generation = int(self._generation.get(attempt_id) or 0) + 1
        self._generation[attempt_id] = new_generation
        self._token[attempt_id] = fence_token
        self._writer[attempt_id] = writer_id
        return new_generation

    def invalidate_fence(self, attempt_id: str) -> None:
        """Expire the current fence token without granting a replacement writer."""

        self._token[attempt_id] = mint_recovery_cid(
            {"kind": "expired-fence", "attempt_id": attempt_id}
        )


def _artifact_mapping(artifact: Any) -> Mapping[str, Any] | None:
    if artifact is None:
        return None
    if isinstance(artifact, StageArtifact):
        return dict(artifact.to_mapping())
    if hasattr(artifact, "to_mapping"):
        mapped = artifact.to_mapping()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    if isinstance(artifact, Mapping):
        return dict(artifact)
    return None


@dataclass(frozen=True)
class RecoveryRecord:
    """Settled recovery outcome. Publication is fenced and fail-closed."""

    schema: str
    status: str
    identities: LifecycleIdentities
    attempt_id: str
    writer_id: str
    writer_generation: int
    mode: str
    published: bool
    sealed: bool
    evidence_cid: str
    error: str | None = None
    replay_trace: tuple[Mapping[str, Any], ...] = ()
    idempotency_keys: Mapping[str, str] = field(default_factory=dict)
    invoked_stages: Mapping[str, int] = field(default_factory=dict)
    checkpoint: Mapping[str, Any] | None = None
    repair_receipt: Mapping[str, Any] | None = None
    lifecycle: Mapping[str, Any] | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    accepted: bool = field(init=False, default=False)
    settled: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "replay_trace", tuple(_freeze(item) for item in self.replay_trace))
        object.__setattr__(self, "idempotency_keys", _freeze(self.idempotency_keys))
        object.__setattr__(self, "invoked_stages", _freeze(self.invoked_stages))
        object.__setattr__(self, "checkpoint", _freeze(self.checkpoint))
        object.__setattr__(self, "repair_receipt", _freeze(self.repair_receipt))
        object.__setattr__(self, "lifecycle", _freeze(self.lifecycle))
        object.__setattr__(self, "payload", _freeze(self.payload))
        if self.schema != RECOVERY_RECORD_SCHEMA:
            raise SchemaMismatchError(
                f"recovery record schema {self.schema!r} is not {RECOVERY_RECORD_SCHEMA}"
            )
        admit_status(self.status)
        admit_mode(self.mode)
        object.__setattr__(self, "evidence_cid", _admit_recovery_cid(self.evidence_cid))
        if self.error is not None and self.error not in ERRORS:
            raise UnknownFieldError(f"unknown error {self.error!r}")
        if self.published and not (
            self.mode in LIVE_MODES
            and self.status == "succeeded"
            and self.sealed
            and self.error is None
        ):
            raise BoundaryViolationError(
                "recovery publication requires a sealed live success under a valid fence"
            )
        object.__setattr__(
            self,
            "accepted",
            bool(self.published and self.status == "succeeded"),
        )
        object.__setattr__(self, "settled", True)

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "contract_version": CONTRACT_VERSION,
                "status": self.status,
                "mode": self.mode,
                "published": self.published,
                "sealed": self.sealed,
                "accepted": self.accepted,
                "settled": self.settled,
                "error": self.error,
                "attempt_id": self.attempt_id,
                "writer_id": self.writer_id,
                "writer_generation": self.writer_generation,
                "identities": dict(self.identities.to_mapping()),
                "evidence_cid": self.evidence_cid,
                "replay_trace": [dict(item) for item in self.replay_trace],
                "idempotency_keys": dict(self.idempotency_keys),
                "invoked_stages": dict(self.invoked_stages),
                "checkpoint": dict(self.checkpoint) if isinstance(self.checkpoint, Mapping) else None,
                "repair_receipt": dict(self.repair_receipt)
                if isinstance(self.repair_receipt, Mapping)
                else None,
                "lifecycle": dict(self.lifecycle) if isinstance(self.lifecycle, Mapping) else None,
                "payload": dict(self.payload) if isinstance(self.payload, Mapping) else self.payload,
                "recovery_cid": RECOVERY_CID,
                "lifecycle_cid": LIFECYCLE_CID,
                "policy_cid": POLICY_CID,
                "result_state_cid": RESULT_STATE_CID,
                "second_wal": SECOND_WAL,
                "infer_success_from_process_exit": INFER_SUCCESS_FROM_PROCESS_EXIT,
            }
        )


def replay_trace(store: CheckpointStore, attempt_id: str) -> tuple[Mapping[str, Any], ...]:
    """Ordered durable checkpoint trace for an attempt."""

    trace = []
    for item in store.history(attempt_id):
        trace.append(
            MappingProxyType(
                {
                    "stage": item.get("stage"),
                    "position": item.get("position"),
                    "idempotency_key": item.get("idempotency_key"),
                    "status": item.get("status"),
                    "in_flight": item.get("in_flight"),
                    "published": item.get("published"),
                    "settled": item.get("settled"),
                    "checkpoint_cid": item.get("checkpoint_cid"),
                }
            )
        )
    return tuple(trace)


def _completed_artifacts(history: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    completed: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in history:
        if item.get("position") != "after":
            continue
        stage = item.get("stage")
        if stage not in STAGES or stage in seen:
            continue
        artifact = item.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        completed.append(dict(artifact))
        seen.add(str(stage))
    return completed


def _has_after(history: Sequence[Mapping[str, Any]], stage: str) -> bool:
    return any(
        item.get("stage") == stage and item.get("position") == "after" for item in history
    )


def _has_unresolved_inflight(history: Sequence[Mapping[str, Any]], stage: str) -> bool:
    saw_during = False
    for item in history:
        if item.get("stage") != stage:
            continue
        if item.get("position") == "during":
            saw_during = True
        if item.get("position") == "after":
            saw_during = False
    return saw_during


def _settled_record(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for item in reversed(history):
        if item.get("settled") is True and isinstance(item.get("record"), Mapping):
            return item
    return None


def _idempotency_keys_from_history(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    keys: dict[str, str] = {}
    for item in history:
        stage = item.get("stage")
        position = item.get("position")
        key = item.get("idempotency_key")
        if stage in BOUNDARIES and position == "after" and isinstance(key, str) and key:
            keys[str(stage)] = key
    return keys


def _invoked_from_store(store: CheckpointStore, attempt_id: str) -> dict[str, int]:
    return {stage: store.invocation_count(attempt_id, stage) for stage in STAGES}


class _PortGuard:
    def __init__(self, inner: Any, recovery: RecoveryCoordinator) -> None:
        self._inner = inner
        self._recovery = recovery


class _RecoveringOperator(_PortGuard):
    def identify(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "identify-operator",
            lambda: self._inner.identify(identities, repository),
        )


class _RecoveringRepository(_PortGuard):
    def resolve(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "resolve-repository",
            lambda: self._inner.resolve(identities, repository),
        )


class _RecoveringSemantic(_PortGuard):
    def scan(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "scan-semantic", lambda: self._inner.scan(identities, repository)
        )

    def invalidate(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "invalidate", lambda: self._inner.invalidate(identities, repository)
        )

    def context_pack(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "context-pack", lambda: self._inner.context_pack(identities, repository)
        )

    def sufficiency(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "sufficiency", lambda: self._inner.sufficiency(identities, repository)
        )

    def impact(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "impact", lambda: self._inner.impact(identities, repository)
        )

    def escalate(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "escalate", lambda: self._inner.escalate(identities, repository)
        )


class _RecoveringRoute(_PortGuard):
    def route(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "route", lambda: self._inner.route(identities, repository)
        )


class _RecoveringProposal(_PortGuard):
    def propose(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        return self._recovery._invoke_stage(
            "proposal",
            lambda: self._inner.propose(identities, repository, proposal),
        )


class _RecoveringScope(_PortGuard):
    def check(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        return self._recovery._invoke_stage(
            "scope-check",
            lambda: self._inner.check(identities, repository, proposal),
        )


class _RecoveringWorktree(_PortGuard):
    def apply(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        return self._recovery._invoke_stage(
            APPLY_STAGE,
            lambda: self._inner.apply(identities, repository, proposal),
        )

    def discard(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        return self._inner.discard(identities, repository)


class _RecoveringVerification(_PortGuard):
    def verify(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            VERIFY_STAGE, lambda: self._inner.verify(identities, repository)
        )


class _RecoveringAssurance(_PortGuard):
    def assure(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            "assurance", lambda: self._inner.assure(identities, repository)
        )


class _RecoveringSealing(_PortGuard):
    def seal(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            SEAL_STAGE, lambda: self._inner.seal(identities, repository)
        )


class _RecoveringDisposition(_PortGuard):
    def decide(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact:
        return self._recovery._invoke_stage(
            DISPOSITION_STAGE, lambda: self._inner.decide(identities, repository)
        )


class _RecoveringGovernance(_PortGuard):
    def acquire_lease(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        recovery = self._recovery
        recovery._write(stage=LEASE_BOUNDARY, position="before")
        raw = dict(self._inner.acquire_lease(identities, repository))
        raw["lease_id"] = recovery._attempt.lease_id
        raw["expires_at"] = recovery._attempt.lease_expires_at
        raw["writer_id"] = recovery._attempt.writer_id
        raw["generation"] = recovery._attempt.writer_generation
        raw["valid"] = bool(raw.get("valid", True)) and recovery._lease_valid()
        recovery._write(
            stage=LEASE_BOUNDARY,
            position="after",
            extra={"lease": raw},
            status="succeeded" if raw.get("valid") else "stale",
        )
        return raw

    def acquire_fence(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        recovery = self._recovery
        recovery._write(stage=FENCE_BOUNDARY, position="before")
        raw = dict(self._inner.acquire_fence(identities, repository))
        raw["fence_id"] = recovery._attempt.fence_id
        raw["token"] = recovery._attempt.fence_token
        raw["generation"] = recovery._attempt.writer_generation
        raw["writer_id"] = recovery._attempt.writer_id
        raw["valid"] = bool(raw.get("valid", True)) and recovery._fence_valid()
        recovery._write(
            stage=FENCE_BOUNDARY,
            position="after",
            extra={"fence": raw},
            status="succeeded" if raw.get("valid") else "stale",
        )
        return raw

    def admit_schedule(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        return self._inner.admit_schedule(identities, repository)

    def check_cancellation(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]:
        recovery = self._recovery
        if not recovery._lease_valid() or not recovery._fence_valid():
            return {"status": "stale", "error": "stale_root"}
        return self._inner.check_cancellation(identities, repository)


class _RecoveringPersistence(_PortGuard):
    def persist(
        self,
        artifact: StageArtifact | Mapping[str, Any],
        *,
        published: bool,
    ) -> Mapping[str, Any]:
        recovery = self._recovery
        allowed_publish = bool(published) and recovery._may_publish()
        if published and not allowed_publish:
            recovery._publication_blocked = True
        receipt = self._inner.persist(artifact, published=allowed_publish)
        stage = getattr(artifact, "stage", None)
        if stage in STAGES:
            already_after = _has_after(
                recovery._store.history(recovery._attempt.attempt_id),
                str(stage),
            )
            recovery._write(
                stage=str(stage),
                position="after",
                artifact=artifact,
                status=str(getattr(artifact, "status", "succeeded")),
                published=False,
            )
            # Process death after persist must not replay on resume: a
            # durable after-record is the completed checkpoint.
            if recovery._should_crash(str(stage), "after") and not already_after:
                raise CrashInterrupt(str(stage), "after")
        return receipt


class RecoveryCoordinator:
    """Persist, fence, and replay PatchLifecycle checkpoints."""

    schema = RECOVERY_SCHEMA
    contract_version = CONTRACT_VERSION
    provider_bound = PROVIDER_BOUND
    sibling_layout_required = SIBLING_LAYOUT_REQUIRED
    second_wal = SECOND_WAL
    infer_success_from_process_exit = INFER_SUCCESS_FROM_PROCESS_EXIT

    def __init__(
        self,
        repository: Path,
        *,
        ports: LifecyclePorts,
        identities: LifecycleIdentities,
        attempt: AttemptIdentity,
        store: CheckpointStore,
        mode: str,
        clock: Callable[[], int],
    ) -> None:
        self._repository = repository
        self._inner_ports = ports
        self._identities = identities
        self._attempt = attempt
        self._store = store
        self._mode = mode
        self._clock = clock
        self._crash: tuple[str, str] | None = None
        self._publication_blocked = False
        self._inbound: str | None = None
        self._ports = self._wrap_ports(ports)

    @classmethod
    def open(
        cls,
        repository: str | Path,
        *,
        ports: LifecyclePorts,
        identities: LifecycleIdentities,
        attempt: AttemptIdentity,
        store: CheckpointStore,
        mode: str = "production",
        clock: Callable[[], int] | None = None,
    ) -> RecoveryCoordinator:
        try:
            admitted_mode = admit_mode(mode)
        except PolicyError as exc:
            raise RecoveryError(
                str(exc),
                code=exc.reason if exc.reason in ERRORS else "unknown_field",
            ) from exc
        root = Path(repository)
        if not root.is_dir():
            raise RecoveryError(
                "repository must be an ordinary directory",
                code="malformed",
            )
        bound = merge_identities(identities, attempt.identities) if attempt.identities else identities
        if bound.run_id != identities.run_id or bound.trace_id != identities.trace_id:
            # merge_identities already rejects drift; keep the constructor identities.
            bound = identities
        return cls(
            root,
            ports=ports,
            identities=bound,
            attempt=attempt,
            store=store,
            mode=admitted_mode,
            clock=clock or _default_clock,
        )

    @property
    def repository(self) -> Path:
        return self._repository

    @property
    def store(self) -> CheckpointStore:
        return self._store

    @property
    def attempt(self) -> AttemptIdentity:
        return self._attempt

    @property
    def identities(self) -> LifecycleIdentities:
        return self._identities

    @property
    def mode(self) -> str:
        return self._mode

    def inject_crash(self, stage: str, position: str) -> None:
        if stage not in EFFECTFUL_STAGES:
            raise UnknownFieldError(f"crash injection is limited to {EFFECTFUL_STAGES}")
        self._crash = (stage, admit_position(position))

    def _wrap_ports(self, ports: LifecyclePorts) -> LifecyclePorts:
        return LifecyclePorts(
            operator=_RecoveringOperator(ports.operator, self),
            repository=_RecoveringRepository(ports.repository, self),
            semantic=_RecoveringSemantic(ports.semantic, self),
            route=_RecoveringRoute(ports.route, self),
            proposal=_RecoveringProposal(ports.proposal, self),
            scope=_RecoveringScope(ports.scope, self),
            worktree=_RecoveringWorktree(ports.worktree, self),
            verification=_RecoveringVerification(ports.verification, self),
            assurance=_RecoveringAssurance(ports.assurance, self),
            sealing=_RecoveringSealing(ports.sealing, self),
            disposition=_RecoveringDisposition(ports.disposition, self),
            governance=_RecoveringGovernance(ports.governance, self),
            persistence=_RecoveringPersistence(ports.persistence, self),
        )

    def _should_crash(self, stage: str, position: str) -> bool:
        return self._crash == (stage, position)

    def _lease_valid(self) -> bool:
        return self._clock() < self._attempt.lease_expires_at

    def _fence_valid(self) -> bool:
        current_generation = self._store.current_generation(self._attempt.attempt_id)
        current_token = self._store.current_token(self._attempt.attempt_id)
        if current_generation is None:
            return True
        return (
            current_generation == self._attempt.writer_generation
            and current_token == self._attempt.fence_token
        )

    def _may_write(self) -> bool:
        return self._fence_valid() and self._lease_valid()

    def _may_publish(self) -> bool:
        if self._mode not in LIVE_MODES:
            return False
        if not self._lease_valid() or not self._fence_valid():
            return False
        current_writer = self._store.current_writer(self._attempt.attempt_id)
        if current_writer not in {None, self._attempt.writer_id}:
            return False
        return not self._publication_blocked

    def _reject_bypass(self, payload: Mapping[str, Any] | None) -> None:
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise MalformedError("payload must be a mapping")
        for key in _BYPASS_KEYS:
            if key in payload and payload[key]:
                raise BoundaryViolationError(
                    "adapters cannot bypass or self-approve a lifecycle stage",
                    details={"reason": key},
                )

    def _write(
        self,
        *,
        stage: str,
        position: str,
        artifact: Any = None,
        status: str = "succeeded",
        published: bool = False,
        settled: bool = False,
        in_flight: bool = False,
        extra: Mapping[str, Any] | None = None,
        record: Mapping[str, Any] | None = None,
    ) -> str:
        key = mint_idempotency_key(
            attempt_id=self._attempt.attempt_id,
            run_id=self._identities.run_id,
            trace_id=self._identities.trace_id,
            stage=stage,
            position=position,
            inbound_cid=self._inbound,
            generation=self._attempt.writer_generation,
        )
        artifact_map = _artifact_mapping(artifact)
        payload: dict[str, Any] = {
            "schema": CHECKPOINT_RECORD_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "attempt_id": self._attempt.attempt_id,
            "writer_id": self._attempt.writer_id,
            "writer_generation": self._attempt.writer_generation,
            "fence_token": self._attempt.fence_token,
            "lease_id": self._attempt.lease_id,
            "fence_id": self._attempt.fence_id,
            "lease_expires_at": self._attempt.lease_expires_at,
            "mode": self._mode,
            "stage": stage,
            "position": position,
            "idempotency_key": key,
            "in_flight": in_flight,
            "published": published,
            "settled": settled,
            "status": status,
            "identities": dict(self._identities.to_mapping()),
            "artifact": artifact_map,
            "inbound_cid": self._inbound,
            "governance": self._governance_mapping(),
        }
        if extra:
            payload.update(dict(extra))
        if record is not None:
            payload["record"] = dict(record)
        cid = self._store.put(
            payload,
            writer_id=self._attempt.writer_id,
            generation=self._attempt.writer_generation,
            fence_token=self._attempt.fence_token,
        )
        if artifact_map and artifact_map.get("artifact_cid"):
            self._inbound = str(artifact_map["artifact_cid"])
        return cid

    def _governance_mapping(self) -> dict[str, Any]:
        return {
            "lease": {
                "lease_id": self._attempt.lease_id,
                "valid": self._lease_valid(),
                "expires_at": self._attempt.lease_expires_at,
                "writer_id": self._attempt.writer_id,
                "generation": self._attempt.writer_generation,
            },
            "fence": {
                "fence_id": self._attempt.fence_id,
                "valid": self._fence_valid(),
                "token": self._attempt.fence_token,
                "generation": self._attempt.writer_generation,
                "writer_id": self._attempt.writer_id,
            },
            "worktree": {},
            "schedule": {"admitted": True},
        }

    def _invoke_stage(self, stage: str, fn: Callable[[], StageArtifact]) -> StageArtifact:
        history = self._store.history(self._attempt.attempt_id)
        if _has_after(history, stage):
            raise BoundaryViolationError(
                "cannot reinvoke a completed lifecycle stage",
                details={"stage": stage, "reason": "idempotency"},
            )
        self._write(stage=stage, position="before")
        if self._should_crash(stage, "before"):
            raise CrashInterrupt(stage, "before")
        invocations = self._store.invocation_count(self._attempt.attempt_id, stage)
        if invocations and stage in EFFECTFUL_STAGES:
            raise RecoveryError(
                "cannot reinvoke an ambiguous effectful adapter",
                code="repair_required",
                details={"stage": stage},
            )
        if stage in TERMINAL_ADAPTER_STAGES and invocations:
            raise BoundaryViolationError(
                "cannot reinvoke a terminal adapter",
                details={"stage": stage},
            )
        self._store.record_invocation(self._attempt.attempt_id, stage)
        self._write(stage=stage, position="during", in_flight=True)
        result = fn()
        if self._should_crash(stage, "during"):
            raise CrashInterrupt(stage, "during")
        return result

    def _lifecycle(self) -> PatchLifecycle:
        return PatchLifecycle.open(
            self._repository,
            ports=self._ports,
            identities=self._identities,
            mode=self._mode,
        )

    def _lifecycle_checkpoint(self) -> Mapping[str, Any] | None:
        history = self._store.history(self._attempt.attempt_id)
        completed = _completed_artifacts(history)
        if not completed:
            return None
        apply_artifact = next(
            (item for item in completed if item.get("stage") == APPLY_STAGE),
            None,
        )
        worktree: dict[str, Any] = {}
        if isinstance(apply_artifact, Mapping):
            payload = apply_artifact.get("payload")
            if isinstance(payload, Mapping):
                worktree = {
                    "worktree_id": payload.get("worktree_id"),
                    "disposable": payload.get("disposable", True),
                    "canonical_mutated": False,
                    "canonical_head": payload.get("canonical_head"),
                    "receipt_cid": apply_artifact.get("artifact_cid"),
                }
        identities = completed[-1].get("identities")
        bound = self._identities
        if isinstance(identities, Mapping):
            bound = merge_identities(bound, LifecycleIdentities.from_mapping(identities))
            self._identities = bound
        last_status = str(completed[-1].get("status") or "succeeded")
        inbound = completed[-1].get("artifact_cid")
        if isinstance(inbound, str) and inbound:
            self._inbound = inbound
        return {
            "schema": CHECKPOINT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "mode": self._mode,
            "identities": dict(bound.to_mapping()),
            "completed": completed,
            "governance": {
                **self._governance_mapping(),
                "worktree": worktree,
            },
            "status": last_status,
            "published": False,
        }

    def run(self, proposal: Mapping[str, Any] | None = None) -> RecoveryRecord:
        self._reject_bypass(proposal)
        return self._execute(proposal, resume=False)

    def resume(self, checkpoint: Mapping[str, Any] | None = None) -> RecoveryRecord:
        self._reject_bypass(checkpoint)
        # Crash injection models process death. A restarted coordinator
        # must not keep the dying process's crash point.
        self._crash = None
        return self._execute(None, resume=True, checkpoint=checkpoint)

    def _execute(
        self,
        proposal: Mapping[str, Any] | None,
        *,
        resume: bool,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> RecoveryRecord:
        history = self._store.history(self._attempt.attempt_id)
        settled = _settled_record(history)
        if settled is not None and isinstance(settled.get("record"), Mapping):
            return self._record_from_mapping(settled["record"])
        latest = self._store.latest(self._attempt.attempt_id)
        if latest is not None and latest.get("stage") in EFFECTFUL_STAGES:
            if latest.get("position") == "during" and not _has_after(history, str(latest["stage"])):
                return self._repair_required(str(latest["stage"]))
        if not self._fence_valid():
            if resume and latest is not None:
                return self._stale("expired or replaced fence cannot publish")
            if not resume:
                return self._stale("expired or replaced fence cannot publish")
        if resume and not self._lease_valid():
            in_flight = any(
                latest is not None
                and latest.get("position") == "during"
                and latest.get("stage") in EFFECTFUL_STAGES
                for _ in (0,)
            )
            if in_flight:
                return self._repair_required(str(latest.get("stage") if latest else APPLY_STAGE))
            return self._stale("expired lease cannot publish")
        loaded = checkpoint if checkpoint is not None else self._lifecycle_checkpoint()
        try:
            lifecycle = self._lifecycle()
            if loaded is not None:
                record = lifecycle.resume(loaded)
            else:
                record = lifecycle.run(proposal)
        except CrashInterrupt:
            raise
        except ProofContextError as exc:
            return self._from_error(exc)
        except PolicyError as exc:
            code = exc.reason if exc.reason in ERRORS else "boundary_violation"
            return self._from_error(RecoveryError(str(exc), code=code))
        return self._from_lifecycle(record)

    def _repair_required(self, stage: str) -> RecoveryRecord:
        discarded = False
        try:
            result = _as_mapping(
                self._inner_ports.worktree.discard(self._identities, self._repository)
            )
            discarded = result.get("discarded") is True
        except Exception:  # noqa: BLE001 - discard failure remains auditable
            discarded = False
        history = self._store.history(self._attempt.attempt_id)
        repair = {
            "schema": REPAIR_RECEIPT_SCHEMA,
            "stage": stage,
            "position": "during",
            "ambiguous": True,
            "published": False,
            "discarded": discarded,
            "status": "repair_required",
            "error": "repair_required",
            "infer_success_from_process_exit": False,
            "invoked": self._store.invocation_count(self._attempt.attempt_id, stage),
        }
        evidence_cid = mint_recovery_cid(
            {
                "kind": "repair",
                "attempt_id": self._attempt.attempt_id,
                "stage": stage,
                "run_id": self._identities.run_id,
            }
        )
        record = RecoveryRecord(
            schema=RECOVERY_RECORD_SCHEMA,
            status="repair_required",
            identities=self._identities,
            attempt_id=self._attempt.attempt_id,
            writer_id=self._attempt.writer_id,
            writer_generation=self._attempt.writer_generation,
            mode=self._mode,
            published=False,
            sealed=False,
            evidence_cid=evidence_cid,
            error="repair_required",
            replay_trace=replay_trace(self._store, self._attempt.attempt_id),
            idempotency_keys=_idempotency_keys_from_history(history),
            invoked_stages=_invoked_from_store(self._store, self._attempt.attempt_id),
            checkpoint=dict(self._store.latest(self._attempt.attempt_id) or {}),
            repair_receipt=repair,
            payload={
                "ambiguous_stage": stage,
                "completed": [item.get("stage") for item in _completed_artifacts(history)],
            },
        )
        if self._may_write():
            self._write(
                stage=PUBLISH_BOUNDARY,
                position="after",
                status="repair_required",
                published=False,
                settled=True,
                extra={"repair": True},
                record=record.to_mapping(),
            )
        return record

    def _stale(self, message: str) -> RecoveryRecord:
        history = self._store.history(self._attempt.attempt_id)
        evidence_cid = mint_recovery_cid(
            {
                "kind": "stale",
                "attempt_id": self._attempt.attempt_id,
                "run_id": self._identities.run_id,
                "message": message,
            }
        )
        record = RecoveryRecord(
            schema=RECOVERY_RECORD_SCHEMA,
            status="stale",
            identities=self._identities,
            attempt_id=self._attempt.attempt_id,
            writer_id=self._attempt.writer_id,
            writer_generation=self._attempt.writer_generation,
            mode=self._mode,
            published=False,
            sealed=False,
            evidence_cid=evidence_cid,
            error="stale_root",
            replay_trace=replay_trace(self._store, self._attempt.attempt_id),
            idempotency_keys=_idempotency_keys_from_history(history),
            invoked_stages=_invoked_from_store(self._store, self._attempt.attempt_id),
            checkpoint=dict(self._store.latest(self._attempt.attempt_id) or {}),
            payload={"message": message, "published": False},
        )
        return record

    def _from_error(self, exc: ProofContextError) -> RecoveryRecord:
        history = self._store.history(self._attempt.attempt_id)
        evidence_cid = self._identities.evidence_cid or mint_recovery_cid(
            {
                "kind": "error",
                "attempt_id": self._attempt.attempt_id,
                "code": exc.code,
                "run_id": self._identities.run_id,
            }
        )
        record = RecoveryRecord(
            schema=RECOVERY_RECORD_SCHEMA,
            status=exc.status,
            identities=self._identities,
            attempt_id=self._attempt.attempt_id,
            writer_id=self._attempt.writer_id,
            writer_generation=self._attempt.writer_generation,
            mode=self._mode,
            published=False,
            sealed=False,
            evidence_cid=evidence_cid,
            error=exc.code,
            replay_trace=replay_trace(self._store, self._attempt.attempt_id),
            idempotency_keys=_idempotency_keys_from_history(history),
            invoked_stages=_invoked_from_store(self._store, self._attempt.attempt_id),
            checkpoint=dict(self._store.latest(self._attempt.attempt_id) or {}),
            payload={"message": str(exc), "published": False},
        )
        if self._may_write():
            self._write(
                stage=PUBLISH_BOUNDARY,
                position="after",
                status=exc.status,
                published=False,
                settled=True,
                record=record.to_mapping(),
            )
        return record

    def _from_lifecycle(self, record: LifecycleRecord) -> RecoveryRecord:
        published = bool(record.published and self._may_publish())
        status = record.status
        error = record.error
        if record.published and not published:
            status = "stale"
            error = "stale_root"
            published = False
        evidence_cid = record.evidence_cid
        self._identities = record.identities
        history = self._store.history(self._attempt.attempt_id)
        publish_key = mint_idempotency_key(
            attempt_id=self._attempt.attempt_id,
            run_id=self._identities.run_id,
            trace_id=self._identities.trace_id,
            stage=PUBLISH_BOUNDARY,
            position="after",
            inbound_cid=self._inbound,
            generation=self._attempt.writer_generation,
        )
        keys = _idempotency_keys_from_history(history)
        keys[PUBLISH_BOUNDARY] = publish_key
        trace = list(replay_trace(self._store, self._attempt.attempt_id))
        trace.append(
            MappingProxyType(
                {
                    "stage": PUBLISH_BOUNDARY,
                    "position": "after",
                    "idempotency_key": publish_key,
                    "status": status,
                    "in_flight": False,
                    "published": published,
                    "settled": True,
                    "checkpoint_cid": None,
                }
            )
        )
        recovered = RecoveryRecord(
            schema=RECOVERY_RECORD_SCHEMA,
            status=status,
            identities=record.identities,
            attempt_id=self._attempt.attempt_id,
            writer_id=self._attempt.writer_id,
            writer_generation=self._attempt.writer_generation,
            mode=self._mode,
            published=published,
            sealed=record.sealed,
            evidence_cid=evidence_cid,
            error=None if published else error,
            replay_trace=tuple(trace),
            idempotency_keys=keys,
            invoked_stages=_invoked_from_store(self._store, self._attempt.attempt_id),
            checkpoint=dict(self._store.latest(self._attempt.attempt_id) or {}),
            lifecycle=dict(record.to_mapping()),
            payload={
                "published": published,
                "applied": bool(record.payload.get("applied"))
                if isinstance(record.payload, Mapping)
                else False,
                "second_wal": False,
            },
        )
        if self._may_write():
            self._write(
                stage=PUBLISH_BOUNDARY,
                position="after",
                status=status,
                published=published,
                settled=True,
                record=recovered.to_mapping(),
            )
        return recovered

    def _record_from_mapping(self, payload: Mapping[str, Any]) -> RecoveryRecord:
        identities_raw = payload.get("identities")
        if not isinstance(identities_raw, Mapping):
            raise MalformedError("settled recovery record identities are required")
        return RecoveryRecord(
            schema=str(payload.get("schema") or RECOVERY_RECORD_SCHEMA),
            status=str(payload.get("status") or ""),
            identities=LifecycleIdentities.from_mapping(identities_raw),
            attempt_id=str(payload.get("attempt_id") or self._attempt.attempt_id),
            writer_id=str(payload.get("writer_id") or self._attempt.writer_id),
            writer_generation=int(
                payload.get("writer_generation") or self._attempt.writer_generation
            ),
            mode=str(payload.get("mode") or self._mode),
            published=bool(payload.get("published")),
            sealed=bool(payload.get("sealed")),
            evidence_cid=str(payload.get("evidence_cid") or ""),
            error=_optional_str(payload.get("error")),
            replay_trace=tuple(payload.get("replay_trace") or ()),
            idempotency_keys=payload.get("idempotency_keys")
            if isinstance(payload.get("idempotency_keys"), Mapping)
            else {},
            invoked_stages=payload.get("invoked_stages")
            if isinstance(payload.get("invoked_stages"), Mapping)
            else {},
            checkpoint=payload.get("checkpoint")
            if isinstance(payload.get("checkpoint"), Mapping)
            else None,
            repair_receipt=payload.get("repair_receipt")
            if isinstance(payload.get("repair_receipt"), Mapping)
            else None,
            lifecycle=payload.get("lifecycle")
            if isinstance(payload.get("lifecycle"), Mapping)
            else None,
            payload=payload.get("payload") if isinstance(payload.get("payload"), Mapping) else {},
        )


def lifecycle_checkpoint_from_store(
    store: CheckpointStore,
    attempt_id: str,
    *,
    mode: str,
    identities: LifecycleIdentities,
) -> Mapping[str, Any] | None:
    """Rebuild a PatchLifecycle checkpoint from durable recovery snapshots."""

    completed = _completed_artifacts(store.history(attempt_id))
    if not completed:
        return None
    return MappingProxyType(
        {
            "schema": CHECKPOINT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "mode": mode,
            "identities": dict(identities.to_mapping()),
            "completed": completed,
            "status": completed[-1].get("status"),
            "published": False,
        }
    )


_DESCRIPTOR_BODY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": RECOVERY_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_schema_prefix": CONTRACT_SCHEMA_PREFIX,
        "boundaries": BOUNDARIES,
        "crash_positions": CRASH_POSITIONS,
        "crash_matrix": CRASH_MATRIX,
        "effectful_stages": EFFECTFUL_STAGES,
        "terminal_adapter_stages": TERMINAL_ADAPTER_STAGES,
        "stages": STAGES,
        "modes": MODES,
        "second_wal": SECOND_WAL,
        "wal_implementation": WAL_IMPLEMENTATION,
        "infer_success_from_process_exit": INFER_SUCCESS_FROM_PROCESS_EXIT,
        "persistence_authority": PERSISTENCE_AUTHORITY,
        "restart_aware": RESTART_AWARE,
        "sole_lifecycle_authority": SOLE_LIFECYCLE_AUTHORITY,
        "sibling_layout_required": SIBLING_LAYOUT_REQUIRED,
        "provider_bound": PROVIDER_BOUND,
        "pcce_006_content_id": PCCE_006_CONTENT_ID,
        "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
        "policy_cid": POLICY_CID,
        "lifecycle_cid": LIFECYCLE_CID,
        "result_state_cid": RESULT_STATE_CID,
        "lifecycle_record_schema": LIFECYCLE_RECORD_SCHEMA,
        "stage_artifact_schema": STAGE_ARTIFACT_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
    }
)
RECOVERY_CID: Final[str] = mint_recovery_cid(_DESCRIPTOR_BODY)
RECOVERY_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**dict(_DESCRIPTOR_BODY), "cid": RECOVERY_CID}
)


def recovery_descriptor() -> Mapping[str, Any]:
    return RECOVERY_DESCRIPTOR


def recovery_cid() -> str:
    return RECOVERY_CID


def crash_matrix() -> tuple[tuple[str, str], ...]:
    return CRASH_MATRIX


__all__ = [
    "APPLY_STAGE",
    "BOUNDARIES",
    "CHECKPOINT_RECORD_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "COMPATIBILITY_MATRIX_CONTENT_ID",
    "CONTRACT_SCHEMA_PREFIX",
    "CONTRACT_VERSION",
    "CRASH_MATRIX",
    "CRASH_POSITIONS",
    "DISPOSITION_STAGE",
    "EFFECTFUL_STAGES",
    "IDEMPOTENCY_SCHEMA",
    "INFER_SUCCESS_FROM_PROCESS_EXIT",
    "LEASE_BOUNDARY",
    "FENCE_BOUNDARY",
    "LIFECYCLE_CID",
    "PCCE_006_CONTENT_ID",
    "PERSISTENCE_AUTHORITY",
    "POLICY_CID",
    "PROVIDER_BOUND",
    "PUBLISH_BOUNDARY",
    "RECOVERY_CID",
    "RECOVERY_DESCRIPTOR",
    "RECOVERY_RECORD_SCHEMA",
    "RECOVERY_SCHEMA",
    "REPAIR_RECEIPT_SCHEMA",
    "RESTART_AWARE",
    "RESULT_STATE_CID",
    "SCHEMA",
    "SEAL_STAGE",
    "SECOND_WAL",
    "SIBLING_LAYOUT_REQUIRED",
    "SOLE_LIFECYCLE_AUTHORITY",
    "STAGES",
    "TERMINAL_ADAPTER_STAGES",
    "VALID_TERMINAL_STATUSES",
    "VERIFY_STAGE",
    "WAL_IMPLEMENTATION",
    "AttemptIdentity",
    "CheckpointStore",
    "CrashInterrupt",
    "FencedCheckpointStore",
    "RecoveryCoordinator",
    "RecoveryError",
    "RecoveryRecord",
    "StaleWriterError",
    "admit_boundary",
    "admit_position",
    "crash_matrix",
    "lifecycle_checkpoint_from_store",
    "mint_idempotency_key",
    "mint_recovery_cid",
    "recovery_cid",
    "recovery_descriptor",
    "replay_trace",
]
