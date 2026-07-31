"""Bounded production Hammer proof and reconstruction coordinator (LPR-012).

Extends :class:`IpfsDatasetsLogicProvider` rather than bypassing it.  After an
explicit :class:`NativeExecutionAuthorizationGate` permit the coordinator:

1. intersects supervisor / request / provider resource policies;
2. checks a pinned environment lock;
3. uses deterministic premise selection by default (learned is opt-in,
   digest-pinned, ranking-only);
4. runs the allowlisted bounded portfolio through the existing provider;
5. normalizes proof/counterexample evidence with exact translation-map and
   :class:`ProgramLogicNativeGoalBinding` provenance;
6. reconstructs only through matching native kernel verification;
7. independently validates countermodels (raw diagnostics stay non-authoritative
   until deterministic LogicIR replay or proof of negation);
8. persists a complete Hammer receipt binding; and
9. maps every outcome exactly.

Cancellation cleans owned temp directories and never leaves provider import
temps untracked.  Stale/cross-root/corpus/environment results, malformed
traces, timeouts, denials, and unavailable kernels are non-conclusive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    ProgramLogicAuthorityRoots,
    ProgramLogicNativeGoalBinding,
)
from ..integrations.ipfs_datasets_logic_provider import (
    HAMMER_ADAPTER_SCHEMA_VERSION,
    HAMMER_IMPORT_ISOLATION,
    HammerAdapterStatus,
    HammerSupervisorPolicy,
    IpfsDatasetsLogicProvider,
    IsolatedHammerLoader,
    get_isolated_hammer_loader,
)
from ..proof.formal_verification_contracts import (
    CodeProofObligation,
    ResourceBudget,
    canonical_json,
)
from ..proof.formal_verification_provider import (
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
    dispatch_provider_request,
)
from ..validation.hammer_native_execution_gate import (
    NativeExecutionAuthorizationGate,
    NativeExecutionDecision,
    NativeExecutionDisposition,
    NativeExecutionLane,
    NativeExecutionOperation,
    NativeExecutionPermit,
    PolicyIntersection,
    ResourceEnforcementReport,
    ResourcePolicySlice,
    intersect_resource_policies,
    probe_resource_enforcement,
)


# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

TACTICIAN_HAMMER_COORDINATOR_INTERFACE: Final = "TacticianHammerCoordinator@1"
HAMMER_COORDINATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-coordination-receipt@1"
)
HAMMER_RECEIPT_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-receipt-binding@1"
)
COUNTERMODEL_VALIDATOR_ID: Final = "countermodel-validator@1"
COORDINATOR_PRODUCER_ID: Final = "tactician-hammer-coordinator@1"
COORDINATOR_VERSION: Final = 1

# Exact outcome vocabulary required by LPR-012 acceptance.
COORDINATION_OUTCOMES: Final = frozenset(
    {
        "verified",
        "candidate",
        "counterexample",
        "timeout",
        "unsupported",
        "unavailable",
        "policy_denied",
        "unknown",
        "stale",
        "error",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class HammerCoordinationOutcome(str, Enum):
    """Exact, non-promotable coordination outcomes."""

    VERIFIED = "verified"
    CANDIDATE = "candidate"
    COUNTEREXAMPLE = "counterexample"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    POLICY_DENIED = "policy_denied"
    UNKNOWN = "unknown"
    STALE = "stale"
    ERROR = "error"


class PremiseSelectorMode(str, Enum):
    """Premise selection policy.  Deterministic is the production default."""

    DETERMINISTIC = "deterministic"
    LEARNED_RANKING_ONLY = "learned_ranking_only"


class CoordinationConclusiveness(str, Enum):
    """Whether an outcome may advance proof/refutation authority."""

    CONCLUSIVE_PROOF = "conclusive_proof"
    CONCLUSIVE_REFUTATION = "conclusive_refutation"
    NON_CONCLUSIVE = "non_conclusive"
    DIAGNOSTIC = "diagnostic"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HammerCoordinationError(ValueError):
    """Raised when coordination contracts are malformed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise HammerCoordinationError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise HammerCoordinationError(f"{field_name} must not be empty")
    return result


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HammerCoordinationError(f"{field_name} must be an object")
    return {str(k): v for k, v in value.items()}


def _digest(payload: Mapping[str, Any] | Sequence[Any], *, prefix: str) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def map_provider_status_to_outcome(
    status: Any,
    *,
    proof_success: bool = False,
    kernel_checked: bool = False,
    authoritative_assurance: str = "",
) -> HammerCoordinationOutcome:
    """Map provider/adapter statuses onto the exact LPR-012 vocabulary."""

    raw = str(getattr(status, "value", status) or "").strip().lower()
    if proof_success and kernel_checked and authoritative_assurance == "kernel_verified":
        return HammerCoordinationOutcome.VERIFIED
    if raw in {"verified", "accepted"} and kernel_checked and proof_success:
        return HammerCoordinationOutcome.VERIFIED
    mapping = {
        "verified": HammerCoordinationOutcome.VERIFIED,
        "candidate": HammerCoordinationOutcome.CANDIDATE,
        "counterexample": HammerCoordinationOutcome.COUNTEREXAMPLE,
        "timeout": HammerCoordinationOutcome.TIMEOUT,
        "timed_out": HammerCoordinationOutcome.TIMEOUT,
        "unsupported": HammerCoordinationOutcome.UNSUPPORTED,
        "unsupported_translation": HammerCoordinationOutcome.UNSUPPORTED,
        "unavailable": HammerCoordinationOutcome.UNAVAILABLE,
        "policy_denied": HammerCoordinationOutcome.POLICY_DENIED,
        "unknown": HammerCoordinationOutcome.UNKNOWN,
        "stale": HammerCoordinationOutcome.STALE,
        "error": HammerCoordinationOutcome.ERROR,
        "translated": HammerCoordinationOutcome.UNKNOWN,
        "rejected": HammerCoordinationOutcome.CANDIDATE,
    }
    return mapping.get(raw, HammerCoordinationOutcome.UNKNOWN)


def conclusiveness_for(
    outcome: HammerCoordinationOutcome,
    *,
    countermodel_validated: bool = False,
) -> CoordinationConclusiveness:
    """Stale/timeout/denial/unavailable/error are never conclusive."""

    if outcome is HammerCoordinationOutcome.VERIFIED:
        return CoordinationConclusiveness.CONCLUSIVE_PROOF
    if (
        outcome is HammerCoordinationOutcome.COUNTEREXAMPLE
        and countermodel_validated
    ):
        return CoordinationConclusiveness.CONCLUSIVE_REFUTATION
    if outcome is HammerCoordinationOutcome.COUNTEREXAMPLE:
        return CoordinationConclusiveness.DIAGNOSTIC
    if outcome is HammerCoordinationOutcome.CANDIDATE:
        return CoordinationConclusiveness.NON_CONCLUSIVE
    return CoordinationConclusiveness.NON_CONCLUSIVE


# ---------------------------------------------------------------------------
# Countermodel validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CountermodelValidator:
    """Separates raw solver countermodels from LogicIR-replay authority."""

    producer_id: str = COUNTERMODEL_VALIDATOR_ID

    def validate(
        self,
        *,
        roots: ProgramLogicAuthorityRoots | Mapping[str, Any],
        solver_countermodel_id: str,
        translation_map_id: str,
        originating_logic_ir_id: str,
        raw_diagnostic_refs: Sequence[str] = (),
        replay_result: Mapping[str, Any] | None = None,
        proof_of_negation_id: str = "",
        assumption_refs: Sequence[str] = (),
        resource_policy_ref: str = "",
        invalidation_refs: Sequence[str] = (),
    ) -> CountermodelValidationReceipt:
        """Build a CountermodelValidationReceipt@1.

        Raw countermodels remain ``diagnostic_only`` until a deterministic
        replay against the original LogicIR semantics succeeds or a proof of
        negation is supplied.
        """

        if not isinstance(roots, ProgramLogicAuthorityRoots):
            roots = ProgramLogicAuthorityRoots.from_dict(_mapping(roots, field_name="roots"))

        raw_refs = tuple(
            _text(item, field_name="raw_diagnostic_refs")
            for item in raw_diagnostic_refs
        )
        if not raw_refs and not proof_of_negation_id and not replay_result:
            raise HammerCoordinationError(
                "countermodel validation requires raw diagnostics, replay, "
                "or proof of negation"
            )

        inv = tuple(
            _text(item, field_name="invalidation_refs")
            for item in (invalidation_refs or (roots.tree_id,))
        )

        replayed: list[str] = []
        replay_method = ""
        disposition = CountermodelDisposition.DIAGNOSTIC_ONLY

        if proof_of_negation_id:
            disposition = CountermodelDisposition.VALIDATED
            replay_method = "proof_of_negation"
            # proof_of_negation_id is not a replayed evidence ref; keep disjoint.
        elif replay_result is not None:
            replay = _mapping(replay_result, field_name="replay_result")
            status = str(replay.get("status") or replay.get("disposition") or "").lower()
            method = str(
                replay.get("replay_method")
                or replay.get("method")
                or "deterministic_logic_ir_replay"
            )
            evidence_id = str(
                replay.get("evidence_id")
                or replay.get("replay_id")
                or _digest(replay, prefix="replay")
            )
            if status in {"validated", "accepted", "unsat", "rejected_hypothesis", "ok"}:
                disposition = CountermodelDisposition.VALIDATED
                replay_method = method
                replayed.append(evidence_id)
            elif status in {"stale", "cross_root", "environment_mismatch", "corpus_mismatch"}:
                disposition = CountermodelDisposition.STALE
                replay_method = method
            elif status in {"unsupported"}:
                disposition = CountermodelDisposition.UNSUPPORTED
                replay_method = method
            else:
                disposition = CountermodelDisposition.REPLAY_FAILED
                replay_method = method

        if not raw_refs and disposition is CountermodelDisposition.DIAGNOSTIC_ONLY:
            # Validated path without raw refs is allowed when proof_of_negation.
            if disposition is not CountermodelDisposition.VALIDATED:
                raise HammerCoordinationError(
                    "diagnostic-only countermodels require raw diagnostic refs"
                )

        # Diagnostic-only requires raw refs; validated may omit them.
        if disposition is CountermodelDisposition.DIAGNOSTIC_ONLY and not raw_refs:
            raw_refs = ("diag:unspecified",)

        receipt_id = _digest(
            {
                "solver_countermodel_id": solver_countermodel_id,
                "translation_map_id": translation_map_id,
                "originating_logic_ir_id": originating_logic_ir_id,
                "disposition": disposition.value,
                "raw": list(raw_refs),
                "replayed": list(replayed),
                "proof_of_negation_id": proof_of_negation_id,
            },
            prefix="countermodel-validation",
        )
        return CountermodelValidationReceipt(
            roots=roots,
            receipt_id=receipt_id,
            solver_countermodel_id=_text(
                solver_countermodel_id, field_name="solver_countermodel_id"
            ),
            translation_map_id=_text(
                translation_map_id, field_name="translation_map_id"
            ),
            originating_logic_ir_id=_text(
                originating_logic_ir_id, field_name="originating_logic_ir_id"
            ),
            disposition=disposition,
            raw_diagnostic_refs=raw_refs,
            replayed_rejection_evidence_refs=tuple(replayed),
            proof_of_negation_id=_text(
                proof_of_negation_id,
                field_name="proof_of_negation_id",
                required=False,
            ),
            replay_method=replay_method,
            assumption_refs=tuple(
                _text(item, field_name="assumption_refs") for item in assumption_refs
            ),
            resource_policy_ref=_text(
                resource_policy_ref,
                field_name="resource_policy_ref",
                required=False,
            ),
            invalidation_refs=inv,
        )


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HammerReceiptBinding:
    """Binds a complete Hammer receipt to authority roots and policy."""

    binding_id: str
    hammer_receipt_id: str
    request_id: str
    obligation_id: str
    translation_map_id: str
    environment_lock_id: str
    policy_id: str
    corpus_revision: str
    tree_id: str
    native_goal_binding_id: str = ""
    reconstruction_id: str = ""
    countermodel_validation_id: str = ""
    outcome: HammerCoordinationOutcome = HammerCoordinationOutcome.UNKNOWN
    conclusive: CoordinationConclusiveness = CoordinationConclusiveness.NON_CONCLUSIVE
    persisted_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _text(self.binding_id, field_name="binding_id")
        )
        object.__setattr__(
            self,
            "hammer_receipt_id",
            _text(self.hammer_receipt_id, field_name="hammer_receipt_id"),
        )
        if not isinstance(self.outcome, HammerCoordinationOutcome):
            object.__setattr__(
                self, "outcome", HammerCoordinationOutcome(str(self.outcome))
            )
        if not isinstance(self.conclusive, CoordinationConclusiveness):
            object.__setattr__(
                self,
                "conclusive",
                CoordinationConclusiveness(str(self.conclusive)),
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HAMMER_RECEIPT_BINDING_SCHEMA,
            "binding_id": self.binding_id,
            "hammer_receipt_id": self.hammer_receipt_id,
            "request_id": self.request_id,
            "obligation_id": self.obligation_id,
            "translation_map_id": self.translation_map_id,
            "environment_lock_id": self.environment_lock_id,
            "policy_id": self.policy_id,
            "corpus_revision": self.corpus_revision,
            "tree_id": self.tree_id,
            "native_goal_binding_id": self.native_goal_binding_id,
            "reconstruction_id": self.reconstruction_id,
            "countermodel_validation_id": self.countermodel_validation_id,
            "outcome": self.outcome.value,
            "conclusive": self.conclusive.value,
            "persisted_path": self.persisted_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HammerCoordinationReceipt:
    """Complete, state-bound coordination receipt for one Hammer attempt."""

    receipt_id: str
    outcome: HammerCoordinationOutcome
    conclusiveness: CoordinationConclusiveness
    gate_decision: Mapping[str, Any]
    policy_intersection: Mapping[str, Any]
    resource_enforcement: Mapping[str, Any]
    selector_mode: PremiseSelectorMode
    translation_map_id: str
    environment_lock_id: str
    obligation_id: str
    request_id: str
    provider_result: Mapping[str, Any] = field(default_factory=dict)
    native_goal_binding_id: str = ""
    countermodel_validation: Mapping[str, Any] | None = None
    receipt_binding: Mapping[str, Any] | None = None
    reason_codes: tuple[str, ...] = ()
    import_isolation: str = HAMMER_IMPORT_ISOLATION
    learned_selector_model_digest: str = ""
    proof_success: bool = False
    kernel_checked: bool = False
    cancelled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome.value not in COORDINATION_OUTCOMES:
            raise HammerCoordinationError(
                f"outcome must be one of {sorted(COORDINATION_OUTCOMES)}"
            )
        object.__setattr__(self, "gate_decision", MappingProxyType(dict(self.gate_decision)))
        object.__setattr__(
            self, "policy_intersection", MappingProxyType(dict(self.policy_intersection))
        )
        object.__setattr__(
            self,
            "resource_enforcement",
            MappingProxyType(dict(self.resource_enforcement)),
        )
        object.__setattr__(
            self, "provider_result", MappingProxyType(dict(self.provider_result))
        )
        if self.countermodel_validation is not None:
            object.__setattr__(
                self,
                "countermodel_validation",
                MappingProxyType(dict(self.countermodel_validation)),
            )
        if self.receipt_binding is not None:
            object.__setattr__(
                self,
                "receipt_binding",
                MappingProxyType(dict(self.receipt_binding)),
            )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not isinstance(self.selector_mode, PremiseSelectorMode):
            object.__setattr__(
                self, "selector_mode", PremiseSelectorMode(str(self.selector_mode))
            )
        if not isinstance(self.outcome, HammerCoordinationOutcome):
            object.__setattr__(
                self, "outcome", HammerCoordinationOutcome(str(self.outcome))
            )
        if not isinstance(self.conclusiveness, CoordinationConclusiveness):
            object.__setattr__(
                self,
                "conclusiveness",
                CoordinationConclusiveness(str(self.conclusiveness)),
            )

    @property
    def is_conclusive(self) -> bool:
        return self.conclusiveness in {
            CoordinationConclusiveness.CONCLUSIVE_PROOF,
            CoordinationConclusiveness.CONCLUSIVE_REFUTATION,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HAMMER_COORDINATION_RECEIPT_SCHEMA,
            "interface": TACTICIAN_HAMMER_COORDINATOR_INTERFACE,
            "producer_id": COORDINATOR_PRODUCER_ID,
            "coordinator_version": COORDINATOR_VERSION,
            "receipt_id": self.receipt_id,
            "outcome": self.outcome.value,
            "conclusiveness": self.conclusiveness.value,
            "is_conclusive": self.is_conclusive,
            "gate_decision": dict(self.gate_decision),
            "policy_intersection": dict(self.policy_intersection),
            "resource_enforcement": dict(self.resource_enforcement),
            "selector_mode": self.selector_mode.value,
            "translation_map_id": self.translation_map_id,
            "environment_lock_id": self.environment_lock_id,
            "obligation_id": self.obligation_id,
            "request_id": self.request_id,
            "provider_result": dict(self.provider_result),
            "native_goal_binding_id": self.native_goal_binding_id,
            "countermodel_validation": (
                dict(self.countermodel_validation)
                if self.countermodel_validation is not None
                else None
            ),
            "receipt_binding": (
                dict(self.receipt_binding)
                if self.receipt_binding is not None
                else None
            ),
            "reason_codes": list(self.reason_codes),
            "import_isolation": self.import_isolation,
            "learned_selector_model_digest": self.learned_selector_model_digest,
            "proof_success": self.proof_success,
            "kernel_checked": self.kernel_checked,
            "cancelled": self.cancelled,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


@dataclass
class TacticianHammerCoordinator:
    """Supervisor-owned coordinator over the production Hammer provider."""

    provider: IpfsDatasetsLogicProvider
    gate: NativeExecutionAuthorizationGate
    countermodel_validator: CountermodelValidator = field(
        default_factory=CountermodelValidator
    )
    receipt_store_dir: str | Path | None = None
    loader: IsolatedHammerLoader | None = None
    _temps: list[str] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _cancelled: threading.Event = field(
        default_factory=threading.Event, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.provider, IpfsDatasetsLogicProvider):
            raise HammerCoordinationError(
                "provider must be an IpfsDatasetsLogicProvider"
            )
        if not isinstance(self.gate, NativeExecutionAuthorizationGate):
            raise HammerCoordinationError(
                "gate must be a NativeExecutionAuthorizationGate"
            )
        if self.loader is None:
            self.loader = get_isolated_hammer_loader()
        if self.receipt_store_dir is not None:
            path = Path(self.receipt_store_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.receipt_store_dir = path

    # -- cancellation / lifecycle -----------------------------------------

    def cancel(self) -> None:
        """Signal cancellation and release owned temps (no child/temp leak)."""

        self._cancelled.set()
        self.cleanup()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cleanup(self) -> None:
        with self._lock:
            for path in list(self._temps):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.isfile(path):
                        os.unlink(path)
                except OSError:
                    pass
            self._temps.clear()
            if self.loader is not None:
                self.loader.cleanup_temps()

    def _owned_tempdir(self, prefix: str = "hammer-coord-") -> str:
        path = tempfile.mkdtemp(prefix=prefix)
        with self._lock:
            self._temps.append(path)
        return path

    # -- policy helpers ---------------------------------------------------

    def _provider_policy_slice(self) -> ResourcePolicySlice:
        policy = self.provider.policy
        return ResourcePolicySlice(
            allowed_solvers=tuple(policy.allowed_solvers),
            timeout_ms=policy.timeout_ms,
            cpu_time_ms=policy.cpu_time_ms,
            memory_bytes=policy.memory_bytes,
            max_premises=policy.max_premises,
            max_parallel_processes=policy.max_parallel_processes,
            network_allowed=policy.network_allowed,
            native_execution_allowed=False,
            model_execution_allowed=False,
            learned_selector_allowed=False,
            require_supply_chain_integrity=False,
        )

    def _stale_checks(
        self,
        *,
        obligation: CodeProofObligation,
        payload: Mapping[str, Any],
        expected_tree_id: str = "",
        expected_corpus_revision: str = "",
        expected_environment_id: str = "",
    ) -> list[str]:
        reasons: list[str] = []
        tree = obligation.repository_tree_id
        if expected_tree_id and tree != expected_tree_id:
            reasons.append("cross_root_tree")
        corpus = str(
            payload.get("corpus_revision")
            or obligation.metadata.get("corpus_revision")
            or ""
        )
        if expected_corpus_revision and corpus != expected_corpus_revision:
            reasons.append("stale_corpus")
        env = str(
            payload.get("environment_id")
            or obligation.metadata.get("environment_id")
            or ""
        )
        if expected_environment_id and env and env != expected_environment_id:
            reasons.append("stale_environment")
        return reasons

    # -- receipt persistence ----------------------------------------------

    def persist_receipt(
        self,
        receipt: HammerCoordinationReceipt,
        *,
        full_hammer_receipt: Mapping[str, Any] | None = None,
    ) -> HammerReceiptBinding:
        """Assemble and persist a complete state-bound Hammer receipt."""

        store = Path(self.receipt_store_dir or self._owned_tempdir("hammer-receipts-"))
        store.mkdir(parents=True, exist_ok=True)
        body = receipt.to_dict()
        if full_hammer_receipt is not None:
            body["hammer_receipt"] = dict(full_hammer_receipt)
        receipt_id = receipt.receipt_id
        path = store / f"{receipt_id.replace(':', '_')}.json"
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(body, sort_keys=True, indent=2, default=str)
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, path)

        binding = HammerReceiptBinding(
            binding_id=_digest(
                {
                    "receipt_id": receipt_id,
                    "request_id": receipt.request_id,
                    "obligation_id": receipt.obligation_id,
                    "path": str(path),
                },
                prefix="hammer-receipt-binding",
            ),
            hammer_receipt_id=receipt_id,
            request_id=receipt.request_id,
            obligation_id=receipt.obligation_id,
            translation_map_id=receipt.translation_map_id,
            environment_lock_id=receipt.environment_lock_id,
            policy_id=str(
                receipt.policy_intersection.get("policy_id")
                or receipt.gate_decision.get("permit_id")
                or ""
            ),
            corpus_revision=str(
                receipt.metadata.get("corpus_revision")
                or receipt.provider_result.get("provenance", {}).get(
                    "semantic_bindings", {}
                ).get("corpus_revision")
                or ""
            )
            if isinstance(receipt.provider_result.get("provenance"), Mapping)
            else str(receipt.metadata.get("corpus_revision") or ""),
            tree_id=str(receipt.metadata.get("tree_id") or ""),
            native_goal_binding_id=receipt.native_goal_binding_id,
            reconstruction_id=str(
                (receipt.provider_result.get("kernel_verification") or {}).get(
                    "kernel_receipt_id"
                )
                if isinstance(receipt.provider_result.get("kernel_verification"), Mapping)
                else ""
            ),
            countermodel_validation_id=str(
                (receipt.countermodel_validation or {}).get("receipt_id") or ""
            ),
            outcome=receipt.outcome,
            conclusive=receipt.conclusiveness,
            persisted_path=str(path),
        )
        binding_path = store / f"{binding.binding_id.replace(':', '_')}.binding.json"
        binding_path.write_text(
            json.dumps(binding.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return binding

    # -- main coordination entry ------------------------------------------

    def coordinate(
        self,
        *,
        obligation: CodeProofObligation | Mapping[str, Any],
        premises: Sequence[Mapping[str, Any]] | None = None,
        permit: NativeExecutionPermit | Mapping[str, Any] | None = None,
        environment_lock: Mapping[str, Any] | None = None,
        translations: Sequence[Mapping[str, Any]] | None = None,
        translation_map_id: str = "",
        translation_map: Mapping[str, Any] | None = None,
        native_goal_binding: ProgramLogicNativeGoalBinding
        | Mapping[str, Any]
        | None = None,
        goal_snapshot: Mapping[str, Any] | None = None,
        native_source: str = "",
        kernel_id: str = "",
        toolchain_id: str = "",
        proof_candidate: Mapping[str, Any] | None = None,
        selector_mode: PremiseSelectorMode | str = PremiseSelectorMode.DETERMINISTIC,
        learned_selector_model_digest: str = "",
        premise_selection: Mapping[str, Any] | None = None,
        corpus_manifest: Mapping[str, Any] | None = None,
        request_policy: ResourcePolicySlice | Mapping[str, Any] | None = None,
        resource_budget: ResourceBudget | None = None,
        roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None = None,
        expected_tree_id: str = "",
        expected_corpus_revision: str = "",
        expected_environment_id: str = "",
        countermodel_raw: Mapping[str, Any] | None = None,
        countermodel_replay: Mapping[str, Any] | None = None,
        proof_of_negation_id: str = "",
        operation: NativeExecutionOperation | str = NativeExecutionOperation.PORTFOLIO,
        reconstruct: bool = False,
        persist: bool = True,
        portfolio_runner_result: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> HammerCoordinationReceipt:
        """Run one bounded, gated Hammer coordination attempt."""

        if self.cancelled:
            return self._terminal(
                outcome=HammerCoordinationOutcome.UNKNOWN,
                reason_codes=("cancelled",),
                gate_decision={},
                policy_intersection={},
                obligation_id="",
                request_id="",
                cancelled=True,
            )

        try:
            if isinstance(obligation, CodeProofObligation):
                obl = obligation
            else:
                obl = CodeProofObligation.from_dict(
                    _mapping(obligation, field_name="obligation")
                )
        except (TypeError, ValueError) as exc:
            return self._terminal(
                outcome=HammerCoordinationOutcome.ERROR,
                reason_codes=("malformed_obligation", str(exc)),
                gate_decision={},
                policy_intersection={},
                obligation_id="",
                request_id="",
            )

        if not isinstance(selector_mode, PremiseSelectorMode):
            selector_mode = PremiseSelectorMode(str(selector_mode))
        if (
            selector_mode is PremiseSelectorMode.LEARNED_RANKING_ONLY
            and not learned_selector_model_digest
        ):
            return self._terminal(
                outcome=HammerCoordinationOutcome.POLICY_DENIED,
                reason_codes=(
                    "learned_selector_requires_pinned_model_digest",
                ),
                gate_decision={},
                policy_intersection={},
                obligation_id=obl.obligation_id,
                request_id="",
            )

        # Build payload for the existing provider (never bypass it).
        payload: dict[str, Any] = {
            "obligation": obl.to_dict(),
            "premises": [dict(item) for item in (premises or ())],
        }
        lock = dict(environment_lock or self.provider.policy.environment_lock or {})
        if lock:
            payload["environment_lock"] = lock
        if translations is not None:
            payload["translations"] = [dict(item) for item in translations]
        if translation_map is not None:
            payload["translation_map"] = dict(translation_map)
        if translation_map_id:
            payload["translation_map_id"] = translation_map_id
        if native_goal_binding is not None:
            if hasattr(native_goal_binding, "to_record"):
                payload["native_goal_binding"] = dict(
                    native_goal_binding.to_record()  # type: ignore[union-attr]
                )
            elif hasattr(native_goal_binding, "to_dict"):
                payload["native_goal_binding"] = dict(
                    native_goal_binding.to_dict()  # type: ignore[union-attr]
                )
            else:
                payload["native_goal_binding"] = dict(
                    _mapping(native_goal_binding, field_name="native_goal_binding")
                )
        if goal_snapshot is not None:
            payload["goal_snapshot"] = dict(goal_snapshot)
        if native_source:
            payload["native_source"] = native_source
        if kernel_id:
            payload["kernel_id"] = kernel_id
        if toolchain_id:
            payload["toolchain_id"] = toolchain_id
        if proof_candidate is not None:
            payload["proof_candidate"] = dict(proof_candidate)
        if premise_selection is not None:
            if selector_mode is PremiseSelectorMode.LEARNED_RANKING_ONLY:
                # Ranking-only: never author theorem premises; digest is pinned.
                selection = dict(premise_selection)
                selection["learned_model_digest"] = learned_selector_model_digest
                selection["ranking_only"] = True
                selection["allow_learned"] = True
                payload["premise_selection"] = selection
            else:
                payload["premise_selection"] = dict(premise_selection)
        if corpus_manifest is not None:
            payload["corpus_manifest"] = dict(corpus_manifest)

        stale = self._stale_checks(
            obligation=obl,
            payload=payload,
            expected_tree_id=expected_tree_id,
            expected_corpus_revision=expected_corpus_revision,
            expected_environment_id=expected_environment_id,
        )
        if stale:
            return self._terminal(
                outcome=HammerCoordinationOutcome.STALE,
                reason_codes=tuple(stale),
                gate_decision={},
                policy_intersection={},
                obligation_id=obl.obligation_id,
                request_id="",
                metadata={"stale_reasons": list(stale)},
            )

        # Gate: defaults disabled; require exact permit + environment + policy.
        op = (
            operation
            if isinstance(operation, NativeExecutionOperation)
            else NativeExecutionOperation(str(operation))
        )
        if reconstruct:
            op = NativeExecutionOperation.RECONSTRUCTION

        # Ensure gate sees provider policy bounds.
        gate = self.gate
        if not gate.provider_policy.allowed_solvers and self.provider.policy.allowed_solvers:
            gate = NativeExecutionAuthorizationGate(
                default_permit=gate.default_permit,
                resource_enforcement=gate.resource_enforcement,
                supervisor_policy=gate.supervisor_policy
                if gate.supervisor_policy.allowed_solvers
                else ResourcePolicySlice(
                    allowed_solvers=tuple(self.provider.policy.allowed_solvers),
                    timeout_ms=self.provider.policy.timeout_ms,
                    cpu_time_ms=self.provider.policy.cpu_time_ms,
                    memory_bytes=self.provider.policy.memory_bytes,
                    max_premises=self.provider.policy.max_premises,
                    max_parallel_processes=self.provider.policy.max_parallel_processes,
                    network_allowed=self.provider.policy.network_allowed,
                    native_execution_allowed=True,
                ),
                provider_policy=self._provider_policy_slice(),
                require_environment_lock=gate.require_environment_lock,
            )

        decision = gate.authorize(
            op,
            permit=permit,
            environment_lock=lock,
            request_policy=request_policy,
            required_solvers=tuple(self.provider.policy.allowed_solvers)
            if op
            in {
                NativeExecutionOperation.SOLVER,
                NativeExecutionOperation.PORTFOLIO,
            }
            else None,
        )
        if not decision.authorized:
            outcome = (
                HammerCoordinationOutcome.POLICY_DENIED
                if decision.disposition
                in {
                    NativeExecutionDisposition.POLICY_DENIED,
                    NativeExecutionDisposition.DISABLED_BY_DEFAULT,
                    NativeExecutionDisposition.PERMIT_MISMATCH,
                    NativeExecutionDisposition.PERMIT_MISSING,
                    NativeExecutionDisposition.SUPPLY_CHAIN_DENIED,
                    NativeExecutionDisposition.RESOURCE_UNENFORCEABLE,
                }
                else HammerCoordinationOutcome.UNAVAILABLE
                if decision.disposition
                is NativeExecutionDisposition.ENVIRONMENT_MISMATCH
                else HammerCoordinationOutcome.ERROR
                if decision.disposition is NativeExecutionDisposition.MALFORMED
                else HammerCoordinationOutcome.POLICY_DENIED
            )
            return self._terminal(
                outcome=outcome,
                reason_codes=decision.reason_codes,
                gate_decision=decision.to_dict(),
                policy_intersection=decision.policy_intersection.to_dict(),
                obligation_id=obl.obligation_id,
                request_id="",
                environment_lock_id=decision.environment_lock_id,
            )

        if self.cancelled:
            return self._terminal(
                outcome=HammerCoordinationOutcome.UNKNOWN,
                reason_codes=("cancelled",),
                gate_decision=decision.to_dict(),
                policy_intersection=decision.policy_intersection.to_dict(),
                obligation_id=obl.obligation_id,
                request_id="",
                cancelled=True,
            )

        budget = resource_budget or ResourceBudget(
            wall_time_ms=decision.policy_intersection.timeout_ms,
            cpu_time_ms=decision.policy_intersection.cpu_time_ms,
            memory_bytes=decision.policy_intersection.memory_bytes,
            max_processes=decision.policy_intersection.max_parallel_processes,
            max_premises=decision.policy_intersection.max_premises,
            network_allowed=decision.policy_intersection.network_allowed,
        )

        # Dispatch through the production provider facade.
        operation_name = "reconstruct" if reconstruct else "prove"
        if portfolio_runner_result is not None and not reconstruct:
            # Test/injection path: still go through provider but with injected runner.
            injected = IpfsDatasetsLogicProvider(
                self.provider.policy,
                portfolio_runner=lambda _inv: dict(portfolio_runner_result),
                verification_cache=self.provider.verification_cache,
                kernel_verifier=self.provider.kernel_verifier,
            )
            active_provider = injected
        else:
            active_provider = self.provider

        request = ProviderRequest(
            request_id=_digest(
                {
                    "obligation_id": obl.obligation_id,
                    "operation": operation_name,
                    "permit_id": decision.permit_id,
                },
                prefix="hammer-coord-request",
            ),
            operation=operation_name,
            payload=payload,
            resource_budget=budget,
            network_allowed=decision.policy_intersection.network_allowed,
        )

        try:
            response = dispatch_provider_request(active_provider, request)
        except Exception as exc:  # noqa: BLE001 - map to exact error outcome
            return self._terminal(
                outcome=HammerCoordinationOutcome.ERROR,
                reason_codes=("provider_dispatch_error", type(exc).__name__),
                gate_decision=decision.to_dict(),
                policy_intersection=decision.policy_intersection.to_dict(),
                obligation_id=obl.obligation_id,
                request_id=request.request_id,
                environment_lock_id=decision.environment_lock_id,
                metadata={"error": str(exc)},
            )

        provider_result: dict[str, Any]
        if isinstance(response, ProviderResponse):
            if not response.ok:
                code = response.error.code if response.error else None
                details = dict(response.error.details) if response.error else {}
                status = details.get("status") or (
                    code.value if code is not None else "error"
                )
                if code is ProviderFailureCode.TIMED_OUT:
                    outcome = HammerCoordinationOutcome.TIMEOUT
                elif code is ProviderFailureCode.UNSUPPORTED:
                    outcome = HammerCoordinationOutcome.UNSUPPORTED
                elif code is ProviderFailureCode.UNAVAILABLE:
                    outcome = HammerCoordinationOutcome.UNAVAILABLE
                elif code is ProviderFailureCode.RESOURCE_EXHAUSTED:
                    outcome = HammerCoordinationOutcome.POLICY_DENIED
                else:
                    outcome = map_provider_status_to_outcome(status)
                provider_result = {
                    "ok": False,
                    "error": {
                        "code": code.value if code is not None else "error",
                        "message": (
                            response.error.message if response.error else ""
                        ),
                        "details": details,
                    },
                }
                return self._finalize(
                    outcome=outcome,
                    decision=decision,
                    selector_mode=selector_mode,
                    learned_selector_model_digest=learned_selector_model_digest,
                    translation_map_id=str(
                        payload.get("translation_map_id")
                        or details.get("translation_map_id")
                        or ""
                    ),
                    obligation=obl,
                    request_id=request.request_id,
                    provider_result=provider_result,
                    native_goal_binding=payload.get("native_goal_binding"),
                    roots=roots,
                    countermodel_raw=countermodel_raw,
                    countermodel_replay=countermodel_replay,
                    proof_of_negation_id=proof_of_negation_id,
                    persist=persist,
                    reason_codes=(
                        str(details.get("reason_code") or code.value if code else "error"),
                    ),
                    metadata=dict(metadata or {}),
                )
            provider_result = dict(response.require_result())
        else:
            provider_result = dict(response)

        proof_success = bool(provider_result.get("proof_success"))
        kernel_checked = bool(provider_result.get("kernel_checked"))
        assurance = str(provider_result.get("authoritative_assurance") or "")
        outcome = map_provider_status_to_outcome(
            provider_result.get("status"),
            proof_success=proof_success,
            kernel_checked=kernel_checked,
            authoritative_assurance=assurance,
        )
        # Only matching native kernel reconstruction may prove a claim.
        if outcome is HammerCoordinationOutcome.VERIFIED and not (
            proof_success and kernel_checked and assurance == "kernel_verified"
        ):
            outcome = HammerCoordinationOutcome.CANDIDATE
            proof_success = False

        return self._finalize(
            outcome=outcome,
            decision=decision,
            selector_mode=selector_mode,
            learned_selector_model_digest=learned_selector_model_digest,
            translation_map_id=str(
                provider_result.get("translation_map_id")
                or payload.get("translation_map_id")
                or ""
            ),
            obligation=obl,
            request_id=str(
                (provider_result.get("provenance") or {}).get("request_id")
                or request.request_id
            ),
            provider_result=provider_result,
            native_goal_binding=payload.get("native_goal_binding")
            or provider_result.get("native_goal_binding"),
            roots=roots,
            countermodel_raw=countermodel_raw
            or self._extract_countermodel(provider_result),
            countermodel_replay=countermodel_replay,
            proof_of_negation_id=proof_of_negation_id,
            persist=persist,
            proof_success=proof_success,
            kernel_checked=kernel_checked,
            metadata={
                **dict(metadata or {}),
                "tree_id": obl.repository_tree_id,
                "corpus_revision": str(
                    payload.get("corpus_revision")
                    or obl.metadata.get("corpus_revision")
                    or ""
                ),
            },
        )

    def _extract_countermodel(
        self, provider_result: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        if str(provider_result.get("status") or "") != "counterexample":
            return None
        hammer_result = provider_result.get("hammer_result") or {}
        if not isinstance(hammer_result, Mapping):
            return None
        for key in ("counterexample", "countermodel", "model"):
            value = hammer_result.get(key)
            if isinstance(value, Mapping):
                return value
        return {
            "solver_countermodel_id": str(
                hammer_result.get("counterexample_id")
                or hammer_result.get("request_id")
                or "solver-cm:unknown"
            ),
            "raw_diagnostic_refs": ["diag:provider-counterexample"],
        }

    def _finalize(
        self,
        *,
        outcome: HammerCoordinationOutcome,
        decision: NativeExecutionDecision,
        selector_mode: PremiseSelectorMode,
        learned_selector_model_digest: str,
        translation_map_id: str,
        obligation: CodeProofObligation,
        request_id: str,
        provider_result: Mapping[str, Any],
        native_goal_binding: Any,
        roots: ProgramLogicAuthorityRoots | Mapping[str, Any] | None,
        countermodel_raw: Mapping[str, Any] | None,
        countermodel_replay: Mapping[str, Any] | None,
        proof_of_negation_id: str,
        persist: bool,
        reason_codes: Sequence[str] = (),
        proof_success: bool = False,
        kernel_checked: bool = False,
        metadata: Mapping[str, Any] | None = None,
        cancelled: bool = False,
    ) -> HammerCoordinationReceipt:
        cm_receipt_dict: dict[str, Any] | None = None
        cm_validated = False
        if (
            outcome is HammerCoordinationOutcome.COUNTEREXAMPLE
            or countermodel_raw is not None
            or proof_of_negation_id
        ):
            try:
                cm_roots = roots
                if cm_roots is None and isinstance(native_goal_binding, Mapping):
                    cm_roots = native_goal_binding.get("roots")
                if cm_roots is None:
                    # Minimal roots from obligation metadata when caller omitted them.
                    meta = obligation.metadata
                    cm_roots = {
                        "repository_id": obligation.repository_id,
                        "objective_id": str(meta.get("objective_id") or "objective:coord"),
                        "trace_id": str(meta.get("trace_id") or "trace:coord"),
                        "change_id": str(meta.get("change_id") or "change:coord"),
                        "consumer_id": str(meta.get("consumer_id") or "consumer:coord"),
                        "forest_id": str(meta.get("forest_id") or "forest:coord"),
                        "tree_id": obligation.repository_tree_id,
                        "overlay_id": str(meta.get("overlay_id") or "overlay:coord"),
                        "graph_id": str(meta.get("graph_id") or "graph:coord"),
                        "index_id": str(meta.get("index_id") or "index:coord"),
                        "corpus_id": str(
                            meta.get("corpus_revision") or meta.get("corpus_id") or "corpus:coord"
                        ),
                        "model_id": str(meta.get("model_id") or "model:none"),
                        "translator_id": str(meta.get("translator_id") or "translator:coord"),
                        "toolchain_id": str(meta.get("toolchain_id") or "toolchain:coord"),
                        "policy_id": str(
                            decision.policy_intersection.policy_id or "policy:coord"
                        ),
                        "environment_id": str(
                            decision.environment_lock_id or "environment:coord"
                        ),
                    }
                raw = dict(countermodel_raw or {})
                cm = self.countermodel_validator.validate(
                    roots=cm_roots,
                    solver_countermodel_id=str(
                        raw.get("solver_countermodel_id")
                        or raw.get("counterexample_id")
                        or f"solver-cm:{request_id or obligation.obligation_id}"
                    ),
                    translation_map_id=translation_map_id
                    or str(raw.get("translation_map_id") or "translation-map:missing"),
                    originating_logic_ir_id=str(
                        raw.get("originating_logic_ir_id")
                        or (
                            native_goal_binding.get("logic_ir_obligation_id")
                            if isinstance(native_goal_binding, Mapping)
                            else obligation.obligation_id
                        )
                    ),
                    raw_diagnostic_refs=tuple(
                        raw.get("raw_diagnostic_refs")
                        or ("diag:solver-countermodel",)
                    ),
                    replay_result=countermodel_replay,
                    proof_of_negation_id=proof_of_negation_id,
                    resource_policy_ref=decision.policy_intersection.policy_id,
                    invalidation_refs=(obligation.repository_tree_id,),
                )
                cm_receipt_dict = cm.to_dict() if hasattr(cm, "to_dict") else cm.to_record()
                cm_validated = cm.may_reject_hypothesis
                if outcome is not HammerCoordinationOutcome.COUNTEREXAMPLE:
                    if cm_validated:
                        outcome = HammerCoordinationOutcome.COUNTEREXAMPLE
            except (TypeError, ValueError, HammerCoordinationError) as exc:
                reason_codes = tuple(reason_codes) + (
                    "countermodel_validation_error",
                    str(exc),
                )
                cm_receipt_dict = {
                    "disposition": CountermodelDisposition.REPLAY_FAILED.value,
                    "error": str(exc),
                }

        conclusive = conclusiveness_for(
            outcome, countermodel_validated=cm_validated
        )
        # Candidate without kernel reconstruction is never conclusive proof.
        if outcome is HammerCoordinationOutcome.CANDIDATE:
            conclusive = CoordinationConclusiveness.NON_CONCLUSIVE
            proof_success = False

        binding_id = ""
        if isinstance(native_goal_binding, Mapping):
            binding_id = str(native_goal_binding.get("binding_id") or "")
        elif hasattr(native_goal_binding, "binding_id"):
            binding_id = str(getattr(native_goal_binding, "binding_id"))

        receipt_id = _digest(
            {
                "request_id": request_id,
                "obligation_id": obligation.obligation_id,
                "outcome": outcome.value,
                "translation_map_id": translation_map_id,
                "gate_decision_id": decision.decision_id,
            },
            prefix="hammer-coordination",
        )
        receipt = HammerCoordinationReceipt(
            receipt_id=receipt_id,
            outcome=outcome,
            conclusiveness=conclusive,
            gate_decision=decision.to_dict(),
            policy_intersection=decision.policy_intersection.to_dict(),
            resource_enforcement=decision.resource_enforcement.to_dict(),
            selector_mode=selector_mode,
            translation_map_id=translation_map_id,
            environment_lock_id=decision.environment_lock_id,
            obligation_id=obligation.obligation_id,
            request_id=request_id,
            provider_result=dict(provider_result),
            native_goal_binding_id=binding_id,
            countermodel_validation=cm_receipt_dict,
            reason_codes=tuple(reason_codes),
            import_isolation=(
                self.loader.import_isolation
                if self.loader is not None
                else HAMMER_IMPORT_ISOLATION
            ),
            learned_selector_model_digest=learned_selector_model_digest,
            proof_success=proof_success,
            kernel_checked=kernel_checked,
            cancelled=cancelled,
            metadata=dict(metadata or {}),
        )
        if persist and not cancelled:
            binding = self.persist_receipt(receipt)
            receipt = HammerCoordinationReceipt(
                receipt_id=receipt.receipt_id,
                outcome=receipt.outcome,
                conclusiveness=receipt.conclusiveness,
                gate_decision=dict(receipt.gate_decision),
                policy_intersection=dict(receipt.policy_intersection),
                resource_enforcement=dict(receipt.resource_enforcement),
                selector_mode=receipt.selector_mode,
                translation_map_id=receipt.translation_map_id,
                environment_lock_id=receipt.environment_lock_id,
                obligation_id=receipt.obligation_id,
                request_id=receipt.request_id,
                provider_result=dict(receipt.provider_result),
                native_goal_binding_id=receipt.native_goal_binding_id,
                countermodel_validation=dict(receipt.countermodel_validation)
                if receipt.countermodel_validation
                else None,
                receipt_binding=binding.to_dict(),
                reason_codes=receipt.reason_codes,
                import_isolation=receipt.import_isolation,
                learned_selector_model_digest=receipt.learned_selector_model_digest,
                proof_success=receipt.proof_success,
                kernel_checked=receipt.kernel_checked,
                cancelled=receipt.cancelled,
                metadata=dict(receipt.metadata),
            )
        return receipt

    def _terminal(
        self,
        *,
        outcome: HammerCoordinationOutcome,
        reason_codes: Sequence[str],
        gate_decision: Mapping[str, Any],
        policy_intersection: Mapping[str, Any],
        obligation_id: str,
        request_id: str,
        environment_lock_id: str = "",
        cancelled: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> HammerCoordinationReceipt:
        enforcement = (
            self.gate.resource_enforcement.to_dict()
            if self.gate.resource_enforcement is not None
            else probe_resource_enforcement().to_dict()
        )
        receipt_id = _digest(
            {
                "outcome": outcome.value,
                "reason_codes": list(reason_codes),
                "obligation_id": obligation_id,
                "request_id": request_id,
            },
            prefix="hammer-coordination",
        )
        return HammerCoordinationReceipt(
            receipt_id=receipt_id,
            outcome=outcome,
            conclusiveness=conclusiveness_for(outcome),
            gate_decision=dict(gate_decision),
            policy_intersection=dict(policy_intersection),
            resource_enforcement=enforcement,
            selector_mode=PremiseSelectorMode.DETERMINISTIC,
            translation_map_id="",
            environment_lock_id=environment_lock_id,
            obligation_id=obligation_id,
            request_id=request_id,
            reason_codes=tuple(reason_codes),
            import_isolation=HAMMER_IMPORT_ISOLATION,
            cancelled=cancelled,
            metadata=dict(metadata or {}),
        )


def create_tactician_hammer_coordinator(
    *,
    policy: HammerSupervisorPolicy | None = None,
    permit: NativeExecutionPermit | None = None,
    portfolio_runner: Callable[[Any], Any] | None = None,
    kernel_verifier: Any = None,
    receipt_store_dir: str | Path | None = None,
    supervisor_policy: ResourcePolicySlice | None = None,
    resource_enforcement: ResourceEnforcementReport | None = None,
) -> TacticianHammerCoordinator:
    """Factory wiring provider + fail-closed native gate."""

    provider = IpfsDatasetsLogicProvider(
        policy,
        portfolio_runner=portfolio_runner,
        kernel_verifier=kernel_verifier,
    )
    sup = supervisor_policy or ResourcePolicySlice(
        allowed_solvers=tuple((policy or HammerSupervisorPolicy()).allowed_solvers),
        timeout_ms=(policy or HammerSupervisorPolicy()).timeout_ms,
        cpu_time_ms=(policy or HammerSupervisorPolicy()).cpu_time_ms,
        memory_bytes=(policy or HammerSupervisorPolicy()).memory_bytes,
        max_premises=(policy or HammerSupervisorPolicy()).max_premises,
        max_parallel_processes=(
            (policy or HammerSupervisorPolicy()).max_parallel_processes
        ),
        network_allowed=(policy or HammerSupervisorPolicy()).network_allowed,
        native_execution_allowed=True,
    )
    gate = NativeExecutionAuthorizationGate(
        default_permit=permit or NativeExecutionPermit.disabled(),
        resource_enforcement=resource_enforcement or probe_resource_enforcement(),
        supervisor_policy=sup,
        provider_policy=ResourcePolicySlice(
            allowed_solvers=tuple(provider.policy.allowed_solvers),
            timeout_ms=provider.policy.timeout_ms,
            cpu_time_ms=provider.policy.cpu_time_ms,
            memory_bytes=provider.policy.memory_bytes,
            max_premises=provider.policy.max_premises,
            max_parallel_processes=provider.policy.max_parallel_processes,
            network_allowed=provider.policy.network_allowed,
        ),
    )
    return TacticianHammerCoordinator(
        provider=provider,
        gate=gate,
        receipt_store_dir=receipt_store_dir,
    )


__all__ = [
    "TACTICIAN_HAMMER_COORDINATOR_INTERFACE",
    "HAMMER_COORDINATION_RECEIPT_SCHEMA",
    "HAMMER_RECEIPT_BINDING_SCHEMA",
    "COUNTERMODEL_VALIDATOR_ID",
    "COORDINATOR_PRODUCER_ID",
    "COORDINATOR_VERSION",
    "COORDINATION_OUTCOMES",
    "HammerCoordinationOutcome",
    "PremiseSelectorMode",
    "CoordinationConclusiveness",
    "HammerCoordinationError",
    "map_provider_status_to_outcome",
    "conclusiveness_for",
    "CountermodelValidator",
    "HammerReceiptBinding",
    "HammerCoordinationReceipt",
    "TacticianHammerCoordinator",
    "create_tactician_hammer_coordinator",
]
