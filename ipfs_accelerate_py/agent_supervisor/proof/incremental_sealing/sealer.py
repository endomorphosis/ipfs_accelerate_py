"""Atomic WAL-backed seal publication and current-root CAS (IPS-040).

``IncrementalProofSealer`` is the sole accelerate coordinator over kit WAL and
current-seal compare-and-swap.  A full or delta seal becomes current only when
every pre-CAS phase succeeds, the seal bytes are persisted, transition evidence
verifies, and expected-parent CAS wins.

Fail-closed guarantees:

* any pre-CAS failure leaves the old current pointer unchanged;
* exactly one concurrent valid writer publishes a given generation;
* post-CAS recovery recognizes a committed pointer and finalizes cleanup;
* a stale parent returns ``stale_parent`` without overwrite.

Interfaces: ``IncrementalProofSealer``, ``SealPublicationResult``,
``publish_full_checkpoint``, ``publish_delta_seal``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

def _ensure_nested_package_paths() -> None:
    """Make sibling datasets/kit packages importable from the superproject root."""

    repo_root = Path(__file__).resolve().parents[4]
    for name in ("ipfs_datasets_py", "ipfs_kit_py"):
        candidate = repo_root / name
        package = candidate / name
        if package.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_ensure_nested_package_paths()

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    DeltaSeal,
    DeltaTransitionStatement,
    DeltaUnitEvidence,
    ParentSealView,
    build_delta_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FullCheckpointSeal,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import SealStatus
from ipfs_kit_py.proof_seal_store.contracts import (
    ArtifactKind,
    CurrentSealPointer,
    ExplicitRootRequiredError,
    ProofSealStoreContractError,
    SealTransitionPhase,
    SealTransitionRecord,
    SealTransitionState,
    StoreRoot,
    validate_explicit_root_path,
)
from ipfs_kit_py.proof_seal_store.local_store import HermeticProofSealStore
from ipfs_kit_py.proof_seal_store.pointer import (
    CurrentSealRepository,
    PointerDisposition,
    PointerReason,
)
from ipfs_kit_py.proof_seal_store.recovery import (
    RecoveryDisposition,
    RecoveryReason,
    RecoveryReport,
    recover_seal_transitions,
)
from ipfs_kit_py.proof_seal_store.wal import (
    PHASE_ORDER,
    SealTransitionWal,
    SealTransitionWalCrash,
    SealTransitionWalError,
    abort_transition,
    begin_transition,
    commit_transition,
    record_phase,
)

EVIDENCE_SUBSET: Final[str] = "ips/atomic-transition@1"
SEALER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "atomic-sealer@1"
)
SEALER_INTERFACE: Final[str] = "IncrementalProofSealer@1"
PUBLICATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "seal-publication-result@1"
)
CONTRACT_VERSION: Final[int] = 1

_FULL_SEAL_DOMAIN: Final[str] = "ips.full_checkpoint.seal.v1"
_DELTA_SEAL_DOMAIN: Final[str] = "ips.delta_seal.seal.v1"

# Closed ordered publication workflow phases (plan §8.3 / §9).
PUBLICATION_PHASES: Final[tuple[SealTransitionPhase, ...]] = PHASE_ORDER


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class PublicationKind(str, Enum):
    """Closed kinds of seal publication workflows."""

    FULL_CHECKPOINT = "full_checkpoint"
    DELTA_SEAL = "delta_seal"


class PublicationReason(str, Enum):
    """Stable reason codes for atomic publication outcomes."""

    OK = "ok"
    SEALED = "sealed"
    STALE_PARENT = "stale_parent"
    PROOF_FAILED = "proof_failed"
    VERIFICATION_FAILED = "verification_failed"
    INCOMPLETE_MANIFEST = "incomplete_manifest"
    INVALID_CACHE = "invalid_cache"
    SIMULATED_ONLY = "simulated_only"
    FULL_REPROOF_REQUIRED = "full_reproof_required"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    PRE_CAS_FAILURE = "pre_cas_failure"
    CAS_ERROR = "cas_error"
    PERSISTENCE_FAILED = "persistence_failed"
    WAL_ERROR = "wal_error"
    CRASH_INJECTED = "crash_injected"
    NOT_SEALED = "not_sealed"
    RECOVERED_SUCCESS = "recovered_success"
    RECOVERED_STALE = "recovered_stale"
    RECOVERED_OPEN = "recovered_open"


class SealerError(ValueError):
    """Fail-closed atomic sealer contract violation."""

    def __init__(
        self,
        message: str,
        *,
        reason: PublicationReason = PublicationReason.UNKNOWN,
    ) -> None:
        super().__init__(message)
        self.reason = reason


class SealerCrash(SealerError):
    """Raised by an optional test crash injector at a named publication boundary."""

    def __init__(self, boundary: str) -> None:
        super().__init__(
            f"injected seal publication crash at {boundary}",
            reason=PublicationReason.CRASH_INJECTED,
        )
        self.boundary = boundary


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SealPublicationResult:
    """Structured outcome of one atomic seal publication attempt."""

    schema: str
    evidence_subset: str
    status: SealStatus
    reason: PublicationReason
    published: bool
    publication_kind: PublicationKind
    repository_id: str
    branch_id: str
    transition_id: str
    seal_cid: str
    previous_seal_cid: str
    generation: int
    phase_reached: SealTransitionPhase
    pointer: CurrentSealPointer | None = None
    full_seal: FullCheckpointSeal | None = None
    delta_seal: DeltaSeal | None = None
    recovery_disposition: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != PUBLICATION_RESULT_SCHEMA:
            raise SealerError(f"schema must be {PUBLICATION_RESULT_SCHEMA}")
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise SealerError(f"evidence_subset must be {EVIDENCE_SUBSET}")
        if self.published and self.status not in {
            SealStatus.SEALED_FULL,
            SealStatus.SEALED_INCREMENTAL,
        }:
            raise SealerError(
                "published=True requires sealed_full or sealed_incremental"
            )
        if self.published and self.pointer is None:
            raise SealerError("published=True requires a current pointer")

    def __bool__(self) -> bool:
        return self.published

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "contract_version": CONTRACT_VERSION,
            "status": self.status.value,
            "reason": self.reason.value,
            "published": self.published,
            "publication_kind": self.publication_kind.value,
            "repository_id": self.repository_id,
            "branch_id": self.branch_id,
            "transition_id": self.transition_id,
            "seal_cid": self.seal_cid,
            "previous_seal_cid": self.previous_seal_cid,
            "generation": self.generation,
            "phase_reached": self.phase_reached.value,
            "pointer": None if self.pointer is None else self.pointer.to_dict(),
            "full_seal_cid": (
                None if self.full_seal is None else self.full_seal.seal_cid()
            ),
            "delta_seal_cid": (
                None if self.delta_seal is None else self.delta_seal.seal_cid()
            ),
            "recovery_disposition": self.recovery_disposition,
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    # Match full_checkpoint/delta_seal seal_cid encoding exactly.
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _as_store_root(
    root: StoreRoot | str | Path | None,
) -> StoreRoot:
    if root is None:
        raise ExplicitRootRequiredError(
            "IncrementalProofSealer requires an explicit StoreRoot; "
            "no default user-state or daemon root exists"
        )
    if isinstance(root, StoreRoot):
        store_root = root
    else:
        store_root = StoreRoot.require(root)
    validate_explicit_root_path(store_root.root_path, field_name="root_path")
    return store_root


def _require_branch(branch_id: str) -> str:
    if type(branch_id) is not str or not branch_id.strip():
        raise SealerError("branch_id must be a non-empty string")
    text = branch_id.strip()
    if text != branch_id:
        raise SealerError("branch_id must not have surrounding whitespace")
    return text


def _transition_id(
    *,
    repository_id: str,
    branch_id: str,
    generation: int,
    seal_kind: str,
    nonce: str | None = None,
) -> str:
    material = {
        "domain": "ips.atomic_sealer.transition.v1",
        "repository_id": repository_id,
        "branch_id": branch_id,
        "generation": generation,
        "seal_kind": seal_kind,
        "nonce": nonce or uuid.uuid4().hex,
    }
    digest = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()[:24]
    return f"txn:{digest}"


def _seal_envelope_bytes(
    *,
    domain: str,
    canonical: Mapping[str, Any],
) -> bytes:
    return _canonical_json_bytes({"domain": domain, "payload": dict(canonical)})


def _persistable_seal_payload(
    *,
    domain: str,
    seal_cid: str,
    canonical: Mapping[str, Any],
) -> bytes:
    """Return exact seal envelope bytes that rehash to ``seal_cid``."""

    envelope = _seal_envelope_bytes(domain=domain, canonical=canonical)
    recomputed = "sha256:" + hashlib.sha256(envelope).hexdigest()
    if recomputed != seal_cid:
        raise SealerError(
            "seal envelope bytes do not rehash to seal_cid",
            reason=PublicationReason.PERSISTENCE_FAILED,
        )
    return envelope


def _seal_status_to_reason(status: SealStatus) -> PublicationReason:
    mapping: dict[SealStatus, PublicationReason] = {
        SealStatus.SEALED_FULL: PublicationReason.SEALED,
        SealStatus.SEALED_INCREMENTAL: PublicationReason.SEALED,
        SealStatus.STALE_PARENT: PublicationReason.STALE_PARENT,
        SealStatus.PROOF_FAILED: PublicationReason.PROOF_FAILED,
        SealStatus.VERIFICATION_FAILED: PublicationReason.VERIFICATION_FAILED,
        SealStatus.INCOMPLETE_MANIFEST: PublicationReason.INCOMPLETE_MANIFEST,
        SealStatus.INVALID_CACHE: PublicationReason.INVALID_CACHE,
        SealStatus.SIMULATED_ONLY: PublicationReason.SIMULATED_ONLY,
        SealStatus.FULL_REPROOF_REQUIRED: PublicationReason.FULL_REPROOF_REQUIRED,
        SealStatus.UNAVAILABLE: PublicationReason.UNAVAILABLE,
        SealStatus.TIMEOUT: PublicationReason.TIMEOUT,
        SealStatus.CANCELLED: PublicationReason.CANCELLED,
        SealStatus.UNKNOWN: PublicationReason.UNKNOWN,
    }
    return mapping.get(status, PublicationReason.NOT_SEALED)


def _unit_proof_cids(
    units: Sequence[RequiredUnitEvidence | DeltaUnitEvidence | Mapping[str, Any]],
) -> tuple[str, ...]:
    cids: list[str] = []
    seen: set[str] = set()
    for raw in units:
        if isinstance(raw, (RequiredUnitEvidence, DeltaUnitEvidence)):
            cid = raw.proof_object_cid
        elif isinstance(raw, Mapping):
            cid = str(raw.get("proof_object_cid", "") or "")
        else:
            continue
        if cid and cid not in seen:
            seen.add(cid)
            cids.append(cid)
    return tuple(cids)


# ---------------------------------------------------------------------------
# Sealer
# ---------------------------------------------------------------------------


class IncrementalProofSealer:
    """Coordinate full/delta seal construction with kit WAL and current-root CAS.

    Construction requires an explicit store root.  There is no default under
    ``~``, ``$XDG_*``, ``~/.ipfs``, or any daemon path.
    """

    __test__ = False

    def __init__(
        self,
        root: StoreRoot | str | Path | None,
        *,
        branch_id: str = "main",
        create: bool = True,
        crash_injector: Callable[..., Any] | None = None,
        store: HermeticProofSealStore | None = None,
        wal: SealTransitionWal | None = None,
        pointers: CurrentSealRepository | None = None,
    ) -> None:
        store_root = _as_store_root(root)
        self._root = store_root
        self._root_path = Path(store_root.root_path)
        self._branch_id = _require_branch(branch_id)
        self._crash_injector = crash_injector

        if self._root_path.exists() and self._root_path.is_symlink():
            raise SealerError(
                "store root must not be a symlink",
                reason=PublicationReason.WAL_ERROR,
            )

        self._store = store or HermeticProofSealStore(store_root, create=create)
        self._wal = wal or SealTransitionWal(
            store_root,
            create=create,
            crash_injector=self._wal_crash_bridge if crash_injector else None,
        )
        self._pointers = pointers or CurrentSealRepository(
            store_root, create=create
        )

    # -- surface ------------------------------------------------------------

    @property
    def root(self) -> StoreRoot:
        return self._root

    @property
    def root_path(self) -> Path:
        return self._root_path

    @property
    def branch_id(self) -> str:
        return self._branch_id

    @property
    def store(self) -> HermeticProofSealStore:
        return self._store

    @property
    def wal(self) -> SealTransitionWal:
        return self._wal

    @property
    def pointers(self) -> CurrentSealRepository:
        return self._pointers

    def get_current_seal(
        self, repository_id: str, *, branch_id: str | None = None
    ) -> CurrentSealPointer | None:
        """Read the repository/branch current-seal pointer."""

        return self._pointers.get_current_seal(
            repository_id, branch_id or self._branch_id
        )

    def close(self) -> None:
        """Close the underlying WAL segment handle."""

        self._wal.close()

    # -- public publication APIs --------------------------------------------

    def publish_full_checkpoint(
        self,
        repository_state: RepositoryStateView | Mapping[str, Any],
        verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
        *,
        units: Sequence[RequiredUnitEvidence | Mapping[str, Any]] = (),
        expected_unit_ids: Sequence[str] | None = None,
        parent_seal_cid: str | None = None,
        fallback_reasons: Sequence[str] = (),
        expected_repository_proof_root: str | None = None,
        branch_id: str | None = None,
        transition_id: str | None = None,
        fail_before_phase: SealTransitionPhase | str | None = None,
        expected_current: CurrentSealPointer | None = None,
    ) -> SealPublicationResult:
        """Build a full checkpoint and publish it via WAL + expected-parent CAS.

        ``expected_current`` pins the CAS expected pointer (for concurrent
        same-parent races).  When omitted, the live pointer is read once at
        the start of the workflow.
        """

        branch = _require_branch(branch_id or self._branch_id)
        # Intentionally unlocked across the full workflow: kit WAL and pointer
        # CAS provide their own fencing so concurrent same-parent writers race
        # at CAS and exactly one publishes.
        return self._publish(
            publication_kind=PublicationKind.FULL_CHECKPOINT,
            repository_state=repository_state,
            verification_policy=verification_policy,
            units=units,
            branch_id=branch,
            transition_id=transition_id,
            fail_before_phase=fail_before_phase,
            expected_unit_ids=expected_unit_ids,
            parent_seal_cid=parent_seal_cid,
            fallback_reasons=fallback_reasons,
            expected_repository_proof_root=expected_repository_proof_root,
            expected_current=expected_current,
        )

    def publish_delta_seal(
        self,
        parent: ParentSealView | Mapping[str, Any],
        new_repository_state: RepositoryStateView | Mapping[str, Any],
        verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
        transition: DeltaTransitionStatement | Mapping[str, Any],
        *,
        units: Sequence[DeltaUnitEvidence | Mapping[str, Any]] = (),
        branch_id: str | None = None,
        transition_id: str | None = None,
        fail_before_phase: SealTransitionPhase | str | None = None,
        expected_current: CurrentSealPointer | None = None,
    ) -> SealPublicationResult:
        """Build a parent-bound delta seal and publish it via WAL + CAS."""

        branch = _require_branch(branch_id or self._branch_id)
        return self._publish(
            publication_kind=PublicationKind.DELTA_SEAL,
            repository_state=new_repository_state,
            verification_policy=verification_policy,
            units=units,
            branch_id=branch,
            transition_id=transition_id,
            fail_before_phase=fail_before_phase,
            parent=parent,
            transition=transition,
            expected_current=expected_current,
        )

    def recover_publication(
        self,
        *,
        apply_mutations: bool = True,
    ) -> RecoveryReport:
        """Run deterministic kit recovery over open seal transitions.

        Post-CAS success is recognized when the current pointer equals the
        journaled new seal; cleanup is finalized idempotently.  A stale parent
        after pre-CAS seal persistence rejects publication without overwrite.
        """

        return recover_seal_transitions(
            self._root,
            wal=self._wal,
            store=self._store,
            pointers=self._pointers,
            policy={"apply_mutations": apply_mutations},
        )

    # -- core workflow ------------------------------------------------------

    def _publish(
        self,
        *,
        publication_kind: PublicationKind,
        repository_state: RepositoryStateView | Mapping[str, Any],
        verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
        units: Sequence[Any],
        branch_id: str,
        transition_id: str | None,
        fail_before_phase: SealTransitionPhase | str | None,
        expected_unit_ids: Sequence[str] | None = None,
        parent_seal_cid: str | None = None,
        fallback_reasons: Sequence[str] = (),
        expected_repository_proof_root: str | None = None,
        parent: ParentSealView | Mapping[str, Any] | None = None,
        transition: DeltaTransitionStatement | Mapping[str, Any] | None = None,
        expected_current: CurrentSealPointer | None = None,
    ) -> SealPublicationResult:
        fail_phase = self._coerce_fail_phase(fail_before_phase)

        # Resolve repository identity early for failure results.
        if isinstance(repository_state, RepositoryStateView):
            repository_id = repository_state.repository_id
        elif isinstance(repository_state, Mapping):
            repository_id = str(repository_state.get("repository_id", "") or "")
        else:
            raise SealerError("repository_state must be RepositoryStateView or mapping")
        if not repository_id:
            raise SealerError("repository_id is required")

        if expected_current is not None:
            if not isinstance(expected_current, CurrentSealPointer):
                raise SealerError("expected_current must be a CurrentSealPointer")
            if (
                expected_current.repository_id != repository_id
                or expected_current.branch_id != branch_id
            ):
                raise SealerError(
                    "expected_current namespace must match repository/branch",
                    reason=PublicationReason.STALE_PARENT,
                )
            current = expected_current
        else:
            current = self._pointers.get_current_seal(repository_id, branch_id)
        previous_seal_cid = "" if current is None else current.seal_cid
        expected_parent_for_cas = previous_seal_cid
        generation = 0 if current is None else current.generation + 1

        # Delta seals require the declared parent to still be current.
        if publication_kind is PublicationKind.DELTA_SEAL:
            parent_view = self._coerce_parent_view(parent)
            if current is None or current.seal_cid != parent_view.seal_cid:
                return self._result(
                    status=SealStatus.STALE_PARENT,
                    reason=PublicationReason.STALE_PARENT,
                    published=False,
                    publication_kind=publication_kind,
                    repository_id=repository_id,
                    branch_id=branch_id,
                    transition_id=transition_id or "",
                    seal_cid="",
                    previous_seal_cid=previous_seal_cid,
                    generation=generation,
                    phase_reached=SealTransitionPhase.INTENT,
                    pointer=current,
                    diagnostics={
                        "detail": "declared parent is not the current seal pointer",
                        "declared_parent": parent_view.seal_cid,
                        "current_seal": previous_seal_cid,
                    },
                )
            if parent_view.branch_id and parent_view.branch_id != branch_id:
                return self._result(
                    status=SealStatus.STALE_PARENT,
                    reason=PublicationReason.STALE_PARENT,
                    published=False,
                    publication_kind=publication_kind,
                    repository_id=repository_id,
                    branch_id=branch_id,
                    transition_id=transition_id or "",
                    seal_cid="",
                    previous_seal_cid=previous_seal_cid,
                    generation=generation,
                    phase_reached=SealTransitionPhase.INTENT,
                    pointer=current,
                    diagnostics={"detail": "branch mismatch against parent seal"},
                )

        tid = transition_id or _transition_id(
            repository_id=repository_id,
            branch_id=branch_id,
            generation=generation,
            seal_kind=publication_kind.value,
        )

        phase_reached = SealTransitionPhase.INTENT
        full_seal: FullCheckpointSeal | None = None
        delta_seal: DeltaSeal | None = None
        seal_cid = ""
        seal_kind = (
            ArtifactKind.CHECKPOINT_SEAL
            if publication_kind is PublicationKind.FULL_CHECKPOINT
            else ArtifactKind.DELTA_SEAL
        )

        try:
            self._maybe_fail(fail_phase, SealTransitionPhase.INTENT, when="before")

            intent = SealTransitionRecord(
                transition_id=tid,
                repository_id=repository_id,
                branch_id=branch_id,
                phase=SealTransitionPhase.INTENT,
                state=SealTransitionState.OPEN,
                expected_parent_seal_cid=expected_parent_for_cas,
                generation=generation,
            )
            begin_transition(self._wal, intent)
            phase_reached = SealTransitionPhase.INTENT
            self._boundary("after_begin", tid)

            # Construct the seal only after durable intent.
            if publication_kind is PublicationKind.FULL_CHECKPOINT:
                full_seal = create_full_checkpoint(
                    repository_state,
                    verification_policy,
                    units=units,
                    expected_unit_ids=expected_unit_ids,
                    parent_seal_cid=parent_seal_cid,
                    fallback_reasons=fallback_reasons,
                    expected_repository_proof_root=expected_repository_proof_root,
                )
                if not full_seal.sealed:
                    abort_transition(self._wal, tid, phase=phase_reached)
                    return self._result(
                        status=full_seal.seal_status,
                        reason=_seal_status_to_reason(full_seal.seal_status),
                        published=False,
                        publication_kind=publication_kind,
                        repository_id=repository_id,
                        branch_id=branch_id,
                        transition_id=tid,
                        seal_cid="",
                        previous_seal_cid=previous_seal_cid,
                        generation=generation,
                        phase_reached=phase_reached,
                        pointer=current,
                        full_seal=full_seal,
                        diagnostics={"detail": "full checkpoint not sealed"},
                    )
                seal_cid = full_seal.seal_cid()
                proof_cids = _unit_proof_cids(units)
                forest_cid = full_seal.repository_proof_root
                aggregate_cid = full_seal.aggregation_root
                domain = _FULL_SEAL_DOMAIN
                canonical = full_seal.to_canonical()
            else:
                assert parent is not None and transition is not None
                delta_seal = build_delta_seal(
                    parent,
                    repository_state,
                    verification_policy,
                    transition,
                    units=units,
                )
                if not delta_seal.sealed:
                    abort_transition(self._wal, tid, phase=phase_reached)
                    return self._result(
                        status=delta_seal.seal_status,
                        reason=_seal_status_to_reason(delta_seal.seal_status),
                        published=False,
                        publication_kind=publication_kind,
                        repository_id=repository_id,
                        branch_id=branch_id,
                        transition_id=tid,
                        seal_cid="",
                        previous_seal_cid=previous_seal_cid,
                        generation=generation,
                        phase_reached=phase_reached,
                        pointer=current,
                        delta_seal=delta_seal,
                        diagnostics={"detail": "delta seal not sealed"},
                    )
                seal_cid = delta_seal.seal_cid()
                proof_cids = _unit_proof_cids(units)
                forest_cid = delta_seal.new_forest_root_cid
                aggregate_cid = delta_seal.new_aggregation_root
                domain = _DELTA_SEAL_DOMAIN
                canonical = delta_seal.to_canonical()

            # Pre-CAS phase journal: proof -> receipt -> forest -> aggregate.
            phase_reached = self._advance_pre_seal_phases(
                tid,
                proof_cids=proof_cids,
                forest_cid=forest_cid,
                aggregate_cid=aggregate_cid,
                fail_phase=fail_phase,
            )

            # SEAL_PERSISTENCE: durable seal bytes before CAS.
            self._maybe_fail(
                fail_phase, SealTransitionPhase.SEAL_PERSISTENCE, when="before"
            )
            envelope = _persistable_seal_payload(
                domain=domain,
                seal_cid=seal_cid,
                canonical=canonical,
            )
            put = self._store.put_immutable(
                seal_kind,
                envelope,
                claimed_cid=seal_cid,
            )
            if put.cid != seal_cid:
                abort_transition(self._wal, tid, phase=phase_reached)
                return self._result(
                    status=SealStatus.VERIFICATION_FAILED,
                    reason=PublicationReason.PERSISTENCE_FAILED,
                    published=False,
                    publication_kind=publication_kind,
                    repository_id=repository_id,
                    branch_id=branch_id,
                    transition_id=tid,
                    seal_cid=seal_cid,
                    previous_seal_cid=previous_seal_cid,
                    generation=generation,
                    phase_reached=phase_reached,
                    pointer=current,
                    full_seal=full_seal,
                    delta_seal=delta_seal,
                    diagnostics={"detail": "persisted seal cid mismatch"},
                )
            record_phase(
                self._wal,
                tid,
                SealTransitionPhase.SEAL_PERSISTENCE,
                artifact_cids=(seal_cid,),
                new_seal_cid=seal_cid,
                new_seal_kind=seal_kind,
                generation=generation,
            )
            phase_reached = SealTransitionPhase.SEAL_PERSISTENCE
            self._boundary("after_seal_persistence", tid)

            # CURRENT_ROOT_CAS: only now may the pointer move.
            self._maybe_fail(
                fail_phase, SealTransitionPhase.CURRENT_ROOT_CAS, when="before"
            )
            new_pointer = CurrentSealPointer(
                repository_id=repository_id,
                branch_id=branch_id,
                seal_cid=seal_cid,
                seal_kind=seal_kind,
                generation=generation,
                parent_seal_cid=expected_parent_for_cas,
            )
            cas = self._pointers.compare_and_swap_current_seal_result(
                current, new_pointer
            )
            if cas.disposition is PointerDisposition.STALE or (
                cas.reason is PointerReason.STALE_PARENT and not cas.swapped
            ):
                abort_transition(
                    self._wal,
                    tid,
                    phase=SealTransitionPhase.SEAL_PERSISTENCE,
                )
                live = self._pointers.get_current_seal(repository_id, branch_id)
                return self._result(
                    status=SealStatus.STALE_PARENT,
                    reason=PublicationReason.STALE_PARENT,
                    published=False,
                    publication_kind=publication_kind,
                    repository_id=repository_id,
                    branch_id=branch_id,
                    transition_id=tid,
                    seal_cid=seal_cid,
                    previous_seal_cid=previous_seal_cid,
                    generation=generation,
                    phase_reached=SealTransitionPhase.SEAL_PERSISTENCE,
                    pointer=live,
                    full_seal=full_seal,
                    delta_seal=delta_seal,
                    diagnostics={
                        "detail": "expected parent no longer current",
                        "cas_reason": cas.reason.value,
                        "cas_disposition": cas.disposition.value,
                        **dict(cas.diagnostics),
                    },
                )
            if not cas.swapped or cas.pointer is None:
                abort_transition(
                    self._wal,
                    tid,
                    phase=SealTransitionPhase.SEAL_PERSISTENCE,
                )
                live = self._pointers.get_current_seal(repository_id, branch_id)
                return self._result(
                    status=SealStatus.VERIFICATION_FAILED,
                    reason=PublicationReason.CAS_ERROR,
                    published=False,
                    publication_kind=publication_kind,
                    repository_id=repository_id,
                    branch_id=branch_id,
                    transition_id=tid,
                    seal_cid=seal_cid,
                    previous_seal_cid=previous_seal_cid,
                    generation=generation,
                    phase_reached=SealTransitionPhase.SEAL_PERSISTENCE,
                    pointer=live,
                    full_seal=full_seal,
                    delta_seal=delta_seal,
                    diagnostics={
                        "detail": "CAS failed closed",
                        "cas_reason": cas.reason.value,
                        "cas_disposition": cas.disposition.value,
                        **dict(cas.diagnostics),
                    },
                )

            record_phase(
                self._wal,
                tid,
                SealTransitionPhase.CURRENT_ROOT_CAS,
                new_seal_cid=seal_cid,
                new_seal_kind=seal_kind,
                generation=generation,
            )
            phase_reached = SealTransitionPhase.CURRENT_ROOT_CAS
            self._boundary("after_current_root_cas", tid)

            # CLEANUP / commit: pointer already matches the new seal.
            self._maybe_fail(fail_phase, SealTransitionPhase.CLEANUP, when="before")
            commit_transition(
                self._wal,
                tid,
                new_seal_cid=seal_cid,
                new_seal_kind=seal_kind,
                phase=SealTransitionPhase.CLEANUP,
                generation=generation,
            )
            phase_reached = SealTransitionPhase.CLEANUP
            self._boundary("after_cleanup", tid)

            published_pointer = self._pointers.get_current_seal(
                repository_id, branch_id
            )
            status = (
                SealStatus.SEALED_FULL
                if publication_kind is PublicationKind.FULL_CHECKPOINT
                else SealStatus.SEALED_INCREMENTAL
            )
            return self._result(
                status=status,
                reason=PublicationReason.SEALED,
                published=True,
                publication_kind=publication_kind,
                repository_id=repository_id,
                branch_id=branch_id,
                transition_id=tid,
                seal_cid=seal_cid,
                previous_seal_cid=previous_seal_cid,
                generation=generation,
                phase_reached=phase_reached,
                pointer=published_pointer,
                full_seal=full_seal,
                delta_seal=delta_seal,
            )

        except SealerCrash:
            # Injected crash: leave WAL open.  Pre-CAS leaves the old pointer;
            # post-CAS success is recognized by recover_publication.
            raise
        except SealTransitionWalCrash:
            raise
        except SealerError as exc:
            return self._fail_closed_after_exception(
                exc,
                reason=exc.reason,
                publication_kind=publication_kind,
                repository_id=repository_id,
                branch_id=branch_id,
                transition_id=tid,
                seal_cid=seal_cid,
                previous_seal_cid=previous_seal_cid,
                generation=generation,
                phase_reached=phase_reached,
                full_seal=full_seal,
                delta_seal=delta_seal,
            )
        except (SealTransitionWalError, ProofSealStoreContractError, OSError) as exc:
            return self._fail_closed_after_exception(
                exc,
                reason=PublicationReason.PRE_CAS_FAILURE,
                publication_kind=publication_kind,
                repository_id=repository_id,
                branch_id=branch_id,
                transition_id=tid,
                seal_cid=seal_cid,
                previous_seal_cid=previous_seal_cid,
                generation=generation,
                phase_reached=phase_reached,
                full_seal=full_seal,
                delta_seal=delta_seal,
            )

    def _advance_pre_seal_phases(
        self,
        transition_id: str,
        *,
        proof_cids: Sequence[str],
        forest_cid: str,
        aggregate_cid: str,
        fail_phase: SealTransitionPhase | None,
    ) -> SealTransitionPhase:
        """Journal proof/receipt/forest/aggregate phases before seal persistence."""

        # PROOF_EXECUTION
        self._maybe_fail(
            fail_phase, SealTransitionPhase.PROOF_EXECUTION, when="before"
        )
        record_phase(
            self._wal,
            transition_id,
            SealTransitionPhase.PROOF_EXECUTION,
            artifact_cids=tuple(proof_cids[:16]),
        )
        self._boundary("after_proof_execution", transition_id)

        # RECEIPT_PERSISTENCE
        self._maybe_fail(
            fail_phase, SealTransitionPhase.RECEIPT_PERSISTENCE, when="before"
        )
        receipt_cids = tuple(proof_cids[:8])
        record_phase(
            self._wal,
            transition_id,
            SealTransitionPhase.RECEIPT_PERSISTENCE,
            artifact_cids=receipt_cids,
        )
        self._boundary("after_receipt_persistence", transition_id)

        # FOREST_UPDATE
        self._maybe_fail(
            fail_phase, SealTransitionPhase.FOREST_UPDATE, when="before"
        )
        forest_artifacts = (forest_cid,) if forest_cid else ()
        record_phase(
            self._wal,
            transition_id,
            SealTransitionPhase.FOREST_UPDATE,
            artifact_cids=forest_artifacts,
        )
        self._boundary("after_forest_update", transition_id)

        # AGGREGATE_GENERATION
        self._maybe_fail(
            fail_phase, SealTransitionPhase.AGGREGATE_GENERATION, when="before"
        )
        aggregate_artifacts = (aggregate_cid,) if aggregate_cid else ()
        record_phase(
            self._wal,
            transition_id,
            SealTransitionPhase.AGGREGATE_GENERATION,
            artifact_cids=aggregate_artifacts,
        )
        self._boundary("after_aggregate_generation", transition_id)
        return SealTransitionPhase.AGGREGATE_GENERATION

    # -- recovery helpers ---------------------------------------------------

    def recognize_post_cas_success(
        self, transition_id: str
    ) -> SealPublicationResult:
        """Recover a post-CAS transition and report whether success was recognized."""

        report = self.recover_publication(apply_mutations=True)
        try:
            decision = report.decision_for(transition_id)
        except Exception as exc:
            raise SealerError(
                f"no recovery decision for {transition_id!r}: {exc}",
                reason=PublicationReason.UNKNOWN,
            ) from exc

        record = self._wal.get_transition(transition_id)
        pointer = None
        if record is not None:
            pointer = self._pointers.get_current_seal(
                record.repository_id, record.branch_id
            )

        recognized = (
            decision.pointer_recognized
            or decision.reason is RecoveryReason.POINTER_MATCHES_SEAL
            or (
                decision.disposition is RecoveryDisposition.REPAIR
                and record is not None
                and record.state is SealTransitionState.COMMITTED
            )
        )
        stale = (
            decision.reason is RecoveryReason.STALE_PARENT
            or decision.publication_rejected
        )

        if recognized:
            status = SealStatus.SEALED_FULL
            if record is not None and record.new_seal_kind is ArtifactKind.DELTA_SEAL:
                status = SealStatus.SEALED_INCREMENTAL
            return self._result(
                status=status,
                reason=PublicationReason.RECOVERED_SUCCESS,
                published=True,
                publication_kind=(
                    PublicationKind.DELTA_SEAL
                    if status is SealStatus.SEALED_INCREMENTAL
                    else PublicationKind.FULL_CHECKPOINT
                ),
                repository_id="" if record is None else record.repository_id,
                branch_id="" if record is None else record.branch_id,
                transition_id=transition_id,
                seal_cid="" if record is None else record.new_seal_cid,
                previous_seal_cid=(
                    "" if record is None else record.expected_parent_seal_cid
                ),
                generation=0 if record is None else record.generation,
                phase_reached=(
                    SealTransitionPhase.CLEANUP
                    if record is None
                    else record.phase
                ),
                pointer=pointer,
                recovery_disposition=decision.disposition.value,
                diagnostics={
                    "recovery_reason": decision.reason.value,
                    "applied": decision.applied,
                    "pointer_recognized": decision.pointer_recognized,
                },
            )

        if stale:
            return self._result(
                status=SealStatus.STALE_PARENT,
                reason=PublicationReason.RECOVERED_STALE,
                published=False,
                publication_kind=PublicationKind.FULL_CHECKPOINT,
                repository_id="" if record is None else record.repository_id,
                branch_id="" if record is None else record.branch_id,
                transition_id=transition_id,
                seal_cid="" if record is None else record.new_seal_cid,
                previous_seal_cid=(
                    "" if record is None else record.expected_parent_seal_cid
                ),
                generation=0 if record is None else record.generation,
                phase_reached=(
                    SealTransitionPhase.INTENT if record is None else record.phase
                ),
                pointer=pointer,
                recovery_disposition=decision.disposition.value,
                diagnostics={
                    "recovery_reason": decision.reason.value,
                    "publication_rejected": decision.publication_rejected,
                },
            )

        return self._result(
            status=SealStatus.UNKNOWN,
            reason=PublicationReason.RECOVERED_OPEN,
            published=False,
            publication_kind=PublicationKind.FULL_CHECKPOINT,
            repository_id="" if record is None else record.repository_id,
            branch_id="" if record is None else record.branch_id,
            transition_id=transition_id,
            seal_cid="" if record is None else record.new_seal_cid,
            previous_seal_cid=(
                "" if record is None else record.expected_parent_seal_cid
            ),
            generation=0 if record is None else record.generation,
            phase_reached=(
                SealTransitionPhase.INTENT if record is None else record.phase
            ),
            pointer=pointer,
            recovery_disposition=decision.disposition.value,
            diagnostics={
                "recovery_reason": decision.reason.value,
                "disposition": decision.disposition.value,
            },
        )

    # -- internals ----------------------------------------------------------

    def _result(
        self,
        *,
        status: SealStatus,
        reason: PublicationReason,
        published: bool,
        publication_kind: PublicationKind,
        repository_id: str,
        branch_id: str,
        transition_id: str,
        seal_cid: str,
        previous_seal_cid: str,
        generation: int,
        phase_reached: SealTransitionPhase,
        pointer: CurrentSealPointer | None = None,
        full_seal: FullCheckpointSeal | None = None,
        delta_seal: DeltaSeal | None = None,
        recovery_disposition: str = "",
        diagnostics: Mapping[str, Any] | None = None,
    ) -> SealPublicationResult:
        return SealPublicationResult(
            schema=PUBLICATION_RESULT_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            status=status,
            reason=reason,
            published=published,
            publication_kind=publication_kind,
            repository_id=repository_id,
            branch_id=branch_id,
            transition_id=transition_id,
            seal_cid=seal_cid,
            previous_seal_cid=previous_seal_cid,
            generation=generation,
            phase_reached=phase_reached,
            pointer=pointer,
            full_seal=full_seal,
            delta_seal=delta_seal,
            recovery_disposition=recovery_disposition,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )

    def _fail_closed_after_exception(
        self,
        exc: BaseException,
        *,
        reason: PublicationReason,
        publication_kind: PublicationKind,
        repository_id: str,
        branch_id: str,
        transition_id: str,
        seal_cid: str,
        previous_seal_cid: str,
        generation: int,
        phase_reached: SealTransitionPhase,
        full_seal: FullCheckpointSeal | None,
        delta_seal: DeltaSeal | None,
    ) -> SealPublicationResult:
        """Abort only when CAS has not already published the new seal."""

        live = self._pointers.get_current_seal(repository_id, branch_id)
        cas_already_won = bool(
            seal_cid
            and live is not None
            and live.seal_cid == seal_cid
            and live.generation == generation
        )
        if cas_already_won:
            # Leave the transition open for post-CAS recovery/cleanup.
            status = (
                SealStatus.SEALED_FULL
                if publication_kind is PublicationKind.FULL_CHECKPOINT
                else SealStatus.SEALED_INCREMENTAL
            )
            return self._result(
                status=status,
                reason=PublicationReason.SEALED,
                published=True,
                publication_kind=publication_kind,
                repository_id=repository_id,
                branch_id=branch_id,
                transition_id=transition_id,
                seal_cid=seal_cid,
                previous_seal_cid=previous_seal_cid,
                generation=generation,
                phase_reached=phase_reached,
                pointer=live,
                full_seal=full_seal,
                delta_seal=delta_seal,
                diagnostics={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "post_cas_open": True,
                    "cleanup_pending": True,
                },
            )

        try:
            open_rec = self._wal.get_transition(transition_id)
            if open_rec is not None and open_rec.state not in {
                SealTransitionState.COMMITTED,
                SealTransitionState.ABORTED,
                SealTransitionState.FAILED,
            }:
                abort_transition(self._wal, transition_id, phase=phase_reached)
        except Exception:
            pass
        live = self._pointers.get_current_seal(repository_id, branch_id)
        return self._result(
            status=SealStatus.VERIFICATION_FAILED,
            reason=reason,
            published=False,
            publication_kind=publication_kind,
            repository_id=repository_id,
            branch_id=branch_id,
            transition_id=transition_id,
            seal_cid=seal_cid,
            previous_seal_cid=previous_seal_cid,
            generation=generation,
            phase_reached=phase_reached,
            pointer=live,
            full_seal=full_seal,
            delta_seal=delta_seal,
            diagnostics={"error": str(exc), "error_type": type(exc).__name__},
        )

    def _coerce_parent_view(
        self, parent: ParentSealView | Mapping[str, Any] | None
    ) -> ParentSealView:
        if isinstance(parent, ParentSealView):
            return parent
        if isinstance(parent, Mapping):
            seal_cid = str(parent.get("seal_cid") or parent.get("parent_seal_cid") or "")
            branch = str(parent.get("branch_id") or "main")
            repo_id = str(parent.get("repository_id") or "")
            if not seal_cid or not repo_id:
                raise SealerError(
                    "parent mapping requires seal_cid and repository_id",
                    reason=PublicationReason.STALE_PARENT,
                )
            # Minimal view for pre-CAS current-pointer binding only.  Full
            # parent validation is performed by build_delta_seal.
            return ParentSealView(
                seal_cid=seal_cid,
                accepted=bool(parent.get("accepted", True)),
                seal_status=str(
                    parent.get("seal_status") or SealStatus.SEALED_FULL.value
                ),
                repository_id=repo_id,
                branch_id=branch,
                revision=str(parent.get("revision") or "n/a"),
                source_root_cid=str(parent.get("source_root_cid") or ("sha256:" + ("00" * 32))),
                repository_state_cid=str(
                    parent.get("repository_state_cid") or ("sha256:" + ("00" * 32))
                ),
                environment_cid=str(
                    parent.get("environment_cid") or ("sha256:" + ("00" * 32))
                ),
                policy_cid=str(parent.get("policy_cid") or ("sha256:" + ("00" * 32))),
                manifest_root_cid=str(
                    parent.get("manifest_root_cid") or ("sha256:" + ("00" * 32))
                ),
                forest_root_cid=str(
                    parent.get("forest_root_cid")
                    or parent.get("repository_proof_root")
                    or ("sha256:" + ("00" * 32))
                ),
                aggregation_root=str(
                    parent.get("aggregation_root") or ("sha256:" + ("00" * 32))
                ),
                required_unit_ids=tuple(
                    str(item) for item in parent.get("required_unit_ids", ())
                ),
                unit_proof_cids={
                    str(key): str(value)
                    for key, value in dict(parent.get("unit_proof_cids", {})).items()
                },
            )
        raise SealerError("parent must be ParentSealView or mapping")

    def _coerce_fail_phase(
        self, value: SealTransitionPhase | str | None
    ) -> SealTransitionPhase | None:
        if value is None:
            return None
        if isinstance(value, SealTransitionPhase):
            return value
        if isinstance(value, str):
            try:
                return SealTransitionPhase(value)
            except ValueError as exc:
                raise SealerError(
                    f"unknown fail_before_phase: {value!r}",
                    reason=PublicationReason.UNKNOWN,
                ) from exc
        raise SealerError("fail_before_phase must be SealTransitionPhase or str")

    def _maybe_fail(
        self,
        fail_phase: SealTransitionPhase | None,
        current: SealTransitionPhase,
        *,
        when: str,
    ) -> None:
        if fail_phase is not None and fail_phase is current and when == "before":
            raise SealerCrash(f"before_{current.value}")

    def _boundary(self, name: str, transition_id: str) -> None:
        if self._crash_injector is None:
            return
        try:
            self._crash_injector(name, transition_id)
        except TypeError:
            try:
                self._crash_injector(name)
            except TypeError:
                return
        except SealerCrash:
            raise
        except SealTransitionWalCrash:
            raise

    def _wal_crash_bridge(
        self,
        name: str,
        transition_id: str | None = None,
        phase: SealTransitionPhase | None = None,
    ) -> None:
        if self._crash_injector is None:
            return
        try:
            if phase is not None:
                self._crash_injector(name, transition_id, phase)
            else:
                self._crash_injector(name, transition_id)
        except TypeError:
            try:
                self._crash_injector(name, transition_id)
            except TypeError:
                self._crash_injector(name)


# ---------------------------------------------------------------------------
# Module-level facades
# ---------------------------------------------------------------------------


def publish_full_checkpoint(
    root: StoreRoot | str | Path | None,
    repository_state: RepositoryStateView | Mapping[str, Any],
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
    *,
    units: Sequence[RequiredUnitEvidence | Mapping[str, Any]] = (),
    expected_unit_ids: Sequence[str] | None = None,
    parent_seal_cid: str | None = None,
    fallback_reasons: Sequence[str] = (),
    expected_repository_proof_root: str | None = None,
    branch_id: str = "main",
    transition_id: str | None = None,
    fail_before_phase: SealTransitionPhase | str | None = None,
    sealer: IncrementalProofSealer | None = None,
) -> SealPublicationResult:
    """Publish a full checkpoint under WAL-backed expected-parent CAS."""

    owns = sealer is None
    coordinator = sealer or IncrementalProofSealer(root, branch_id=branch_id)
    try:
        return coordinator.publish_full_checkpoint(
            repository_state,
            verification_policy,
            units=units,
            expected_unit_ids=expected_unit_ids,
            parent_seal_cid=parent_seal_cid,
            fallback_reasons=fallback_reasons,
            expected_repository_proof_root=expected_repository_proof_root,
            branch_id=branch_id,
            transition_id=transition_id,
            fail_before_phase=fail_before_phase,
        )
    finally:
        if owns:
            coordinator.close()


def publish_delta_seal(
    root: StoreRoot | str | Path | None,
    parent: ParentSealView | Mapping[str, Any],
    new_repository_state: RepositoryStateView | Mapping[str, Any],
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
    transition: DeltaTransitionStatement | Mapping[str, Any],
    *,
    units: Sequence[DeltaUnitEvidence | Mapping[str, Any]] = (),
    branch_id: str = "main",
    transition_id: str | None = None,
    fail_before_phase: SealTransitionPhase | str | None = None,
    sealer: IncrementalProofSealer | None = None,
) -> SealPublicationResult:
    """Publish a parent-bound delta seal under WAL-backed expected-parent CAS."""

    owns = sealer is None
    coordinator = sealer or IncrementalProofSealer(root, branch_id=branch_id)
    try:
        return coordinator.publish_delta_seal(
            parent,
            new_repository_state,
            verification_policy,
            transition,
            units=units,
            branch_id=branch_id,
            transition_id=transition_id,
            fail_before_phase=fail_before_phase,
        )
    finally:
        if owns:
            coordinator.close()


__all__ = (
    "CONTRACT_VERSION",
    "EVIDENCE_SUBSET",
    "PUBLICATION_PHASES",
    "PUBLICATION_RESULT_SCHEMA",
    "SEALER_INTERFACE",
    "SEALER_SCHEMA",
    "IncrementalProofSealer",
    "PublicationKind",
    "PublicationReason",
    "SealPublicationResult",
    "SealerCrash",
    "SealerError",
    "publish_delta_seal",
    "publish_full_checkpoint",
)
