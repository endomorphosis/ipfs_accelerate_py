"""Program-world production Hammer integration (SAWM-018).

Interface: ``ProgramWorldProofSearch@1``
Evidence: ``sawm/tactician-hammer@1``

Run the landed production Hammer against compiled program-world goals, then
admit a result only after:

* native kernel reconstruction of a proof candidate, or
* deterministic LogicIR replay of a countermodel (or a proof of negation).

This module does not create a second prover, proof store, tactic engine, logic
family, or acceptance gate.  Solver traces, model nominations, and claimed
``verified`` flags remain non-authoritative until reconstruction or replay
succeeds against the current tree, environment, toolchain, and theorem
binding.

Contradictions abstain.  Timeout, unsupported logic, and missing kernels are
typed.  Models may nominate premises or candidates and cannot create proof,
countermodel truth, or authority.

Importing this module performs no I/O and starts no threads.  Optional Hammer
and kernel packages are imported only when an injected backend is absent and
a live search is requested.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ..analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    ProgramLogicAuthorityError,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
)
from .formal_verification_contracts import content_identity
from .program_world_tactician import (
    PROGRAM_WORLD_PROOF_SEARCH_INTERFACE,
    SAWM_TACTICIAN_HAMMER_EVIDENCE,
    ProgramWorldCapabilityKind,
    ProgramWorldCapabilityReceipt,
    ProgramWorldCompilationDisposition,
    ProgramWorldGoalCompilation,
    ProgramWorldReasonCode,
    ProgramWorldStaleBindingError,
    ProgramWorldTacticianError,
    _bool,
    _enum,
    _identifier,
    _ids,
    _mapping,
    _roots,
    _text,
    probe_program_world_proof_capabilities,
)


PROGRAM_WORLD_HAMMER_INTERFACE: Final[str] = "ProgramWorldHammer@1"
PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE: Final[str] = "ProgramWorldProofAdmission@1"

PROGRAM_WORLD_HAMMER_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-hammer-candidate@1"
)
PROGRAM_WORLD_RECONSTRUCTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-reconstruction@1"
)
PROGRAM_WORLD_PROOF_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-proof-admission@1"
)
PROGRAM_WORLD_PROOF_SEARCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-proof-search@1"
)
PROGRAM_WORLD_COUNTERMODEL_REPLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-countermodel-replay@1"
)

HAMMER_PRODUCER_ID: Final[str] = "program-world-hammer@1"
HAMMER_CONTRACT_VERSION: Final[int] = 1
MAX_CANDIDATES: Final[int] = 32
MAX_TIMEOUT_MS: Final[int] = 600_000

_MODEL_ORIGINS: Final[frozenset[str]] = frozenset(
    {"model", "llm", "learned", "nomination", "vector", "knowledge_graph"}
)
_AUTHORITY_ORIGINS: Final[frozenset[str]] = frozenset(
    {"kernel", "reconstruction", "replay", "proof_of_negation"}
)


class ProgramWorldHammerError(ProgramWorldTacticianError):
    """Closed program-world Hammer contract violation."""


class ProgramWorldProofDisposition(str, Enum):
    """Closed proof-search outcomes.  Only reconstructed/replayed evidence admits."""

    ADMITTED_PROOF = "admitted_proof"
    ADMITTED_COUNTERMODEL = "admitted_countermodel"
    CANDIDATE = "candidate"
    ABSTAINED = "abstained"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    REJECTED = "rejected"
    DIAGNOSTIC = "diagnostic"


class ProgramWorldCandidateKind(str, Enum):
    """Closed Hammer candidate families.  None is admission by itself."""

    PROOF = "proof"
    COUNTERMODEL = "countermodel"
    UNKNOWN = "unknown"


class ProgramWorldReconstructionStatus(str, Enum):
    """Closed native reconstruction outcomes."""

    KERNEL_ACCEPTED = "kernel_accepted"
    KERNEL_REJECTED = "kernel_rejected"
    THEOREM_MISMATCH = "theorem_mismatch"
    TREE_MISMATCH = "tree_mismatch"
    SORRY_PRESENT = "sorry_present"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Protocols (injected backends; never a second prover)
# ---------------------------------------------------------------------------


class ProgramWorldHammerBackend(Protocol):
    """Landed or test-injected Hammer search.  Results are candidates only."""

    def search(self, request: "ProgramWorldHammerSearchRequest") -> "ProgramWorldHammerCandidate":
        ...


class ProgramWorldKernelReconstructor(Protocol):
    """Landed or test-injected native kernel reconstruction."""

    def reconstruct(
        self,
        candidate: "ProgramWorldHammerCandidate",
        *,
        compilation: ProgramWorldGoalCompilation,
        goal: ProgramLogicGoal,
    ) -> "ProgramWorldReconstructionReceipt":
        ...


class ProgramWorldCountermodelReplayer(Protocol):
    """Deterministic LogicIR replay of a raw solver countermodel."""

    def replay(
        self,
        candidate: "ProgramWorldHammerCandidate",
        *,
        compilation: ProgramWorldGoalCompilation,
        goal: ProgramLogicGoal,
    ) -> Mapping[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramWorldHammerSearchRequest:
    """Bounded Hammer search against one compiled program-world goal."""

    compilation: ProgramWorldGoalCompilation
    goal_id: str
    timeout_ms: int = 1_000
    logic_family: str = "fol"

    def __post_init__(self) -> None:
        if not isinstance(self.compilation, ProgramWorldGoalCompilation):
            raise ProgramWorldHammerError(
                "compilation must be ProgramWorldGoalCompilation"
            )
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        if isinstance(self.timeout_ms, bool) or not isinstance(self.timeout_ms, int):
            raise ProgramWorldHammerError("timeout_ms must be a positive integer")
        if self.timeout_ms <= 0 or self.timeout_ms > MAX_TIMEOUT_MS:
            raise ProgramWorldHammerError("timeout_ms is outside the supported bound")
        object.__setattr__(self, "logic_family", _text(self.logic_family, "logic_family"))

    @property
    def goal(self) -> ProgramLogicGoal:
        for item in self.compilation.goals:
            if item.goal_id == self.goal_id:
                return item
        raise ProgramWorldHammerError(f"unknown goal_id {self.goal_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "compilation_cid": self.compilation.compilation_cid,
            "goal_id": self.goal_id,
            "timeout_ms": self.timeout_ms,
            "logic_family": self.logic_family,
            "tree_id": self.compilation.roots.tree_id,
            "corpus_cid": self.compilation.corpus.corpus_cid,
        }


@dataclass(frozen=True)
class ProgramWorldHammerCandidate:
    """Untrusted Hammer/solver/model candidate.  Never admission evidence."""

    candidate_id: str
    kind: ProgramWorldCandidateKind
    origin: str
    status: str
    theorem_id: str
    tree_id: str
    environment_id: str = ""
    toolchain_id: str = ""
    translation_map_id: str = ""
    originating_logic_ir_id: str = ""
    kernel_script_ref: str = ""
    countermodel_id: str = ""
    raw_diagnostic_refs: tuple[str, ...] = ()
    proof_success_claimed: bool = False
    kernel_accepted_claimed: bool = False
    timeout: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_HAMMER_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramWorldCandidateKind, "kind")
        )
        object.__setattr__(self, "origin", _text(self.origin, "origin"))
        object.__setattr__(self, "status", _text(self.status, "status"))
        object.__setattr__(self, "theorem_id", _identifier(self.theorem_id, "theorem_id"))
        object.__setattr__(self, "tree_id", _identifier(self.tree_id, "tree_id"))
        object.__setattr__(
            self,
            "environment_id",
            _text(self.environment_id, "environment_id", required=False),
        )
        object.__setattr__(
            self, "toolchain_id", _text(self.toolchain_id, "toolchain_id", required=False)
        )
        object.__setattr__(
            self,
            "translation_map_id",
            _text(self.translation_map_id, "translation_map_id", required=False),
        )
        object.__setattr__(
            self,
            "originating_logic_ir_id",
            _text(
                self.originating_logic_ir_id,
                "originating_logic_ir_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "kernel_script_ref",
            _text(self.kernel_script_ref, "kernel_script_ref", required=False),
        )
        object.__setattr__(
            self,
            "countermodel_id",
            _text(self.countermodel_id, "countermodel_id", required=False),
        )
        object.__setattr__(
            self,
            "raw_diagnostic_refs",
            _ids(self.raw_diagnostic_refs, "raw_diagnostic_refs"),
        )
        object.__setattr__(
            self,
            "proof_success_claimed",
            _bool(self.proof_success_claimed, "proof_success_claimed"),
        )
        object.__setattr__(
            self,
            "kernel_accepted_claimed",
            _bool(self.kernel_accepted_claimed, "kernel_accepted_claimed"),
        )
        object.__setattr__(self, "timeout", _bool(self.timeout, "timeout"))
        object.__setattr__(
            self, "metadata", MappingProxyType(_mapping(self.metadata, "metadata"))
        )

    @property
    def model_origin(self) -> bool:
        return self.origin in _MODEL_ORIGINS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "origin": self.origin,
            "status": self.status,
            "theorem_id": self.theorem_id,
            "tree_id": self.tree_id,
            "environment_id": self.environment_id,
            "toolchain_id": self.toolchain_id,
            "translation_map_id": self.translation_map_id,
            "originating_logic_ir_id": self.originating_logic_ir_id,
            "kernel_script_ref": self.kernel_script_ref,
            "countermodel_id": self.countermodel_id,
            "raw_diagnostic_refs": list(self.raw_diagnostic_refs),
            "proof_success_claimed": self.proof_success_claimed,
            "kernel_accepted_claimed": self.kernel_accepted_claimed,
            "timeout": self.timeout,
            "model_origin": self.model_origin,
            "authoritative": False,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProgramWorldReconstructionReceipt:
    """Independent native reconstruction check.  The only proof-promotion path."""

    reconstruction_id: str
    theorem_id: str
    kernel_id: str
    tree_id: str
    environment_id: str
    toolchain_id: str
    status: ProgramWorldReconstructionStatus
    kernel_accepted: bool
    matching_theorem: bool
    sorry_free: bool
    native_source_digest: str = ""
    environment_lock_id: str = ""
    diagnostic: str = ""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_RECONSTRUCTION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reconstruction_id",
            _identifier(self.reconstruction_id, "reconstruction_id"),
        )
        object.__setattr__(self, "theorem_id", _identifier(self.theorem_id, "theorem_id"))
        object.__setattr__(self, "kernel_id", _identifier(self.kernel_id, "kernel_id"))
        object.__setattr__(self, "tree_id", _identifier(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "environment_id", _identifier(self.environment_id, "environment_id")
        )
        object.__setattr__(
            self, "toolchain_id", _identifier(self.toolchain_id, "toolchain_id")
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, ProgramWorldReconstructionStatus, "status"),
        )
        object.__setattr__(
            self, "kernel_accepted", _bool(self.kernel_accepted, "kernel_accepted")
        )
        object.__setattr__(
            self, "matching_theorem", _bool(self.matching_theorem, "matching_theorem")
        )
        object.__setattr__(self, "sorry_free", _bool(self.sorry_free, "sorry_free"))
        object.__setattr__(
            self,
            "native_source_digest",
            _text(self.native_source_digest, "native_source_digest", required=False),
        )
        object.__setattr__(
            self,
            "environment_lock_id",
            _text(self.environment_lock_id, "environment_lock_id", required=False),
        )
        object.__setattr__(
            self, "diagnostic", _text(self.diagnostic, "diagnostic", required=False)
        )
        if self.kernel_accepted and (
            not self.matching_theorem
            or not self.sorry_free
            or self.status is not ProgramWorldReconstructionStatus.KERNEL_ACCEPTED
        ):
            raise ProgramWorldHammerError(
                "kernel_accepted requires matching theorem, sorry-free source, "
                "and kernel_accepted status"
            )

    @property
    def conclusive_proof(self) -> bool:
        return (
            self.kernel_accepted
            and self.matching_theorem
            and self.sorry_free
            and self.status is ProgramWorldReconstructionStatus.KERNEL_ACCEPTED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "reconstruction_id": self.reconstruction_id,
            "theorem_id": self.theorem_id,
            "kernel_id": self.kernel_id,
            "tree_id": self.tree_id,
            "environment_id": self.environment_id,
            "toolchain_id": self.toolchain_id,
            "status": self.status.value,
            "kernel_accepted": self.kernel_accepted,
            "matching_theorem": self.matching_theorem,
            "sorry_free": self.sorry_free,
            "native_source_digest": self.native_source_digest,
            "environment_lock_id": self.environment_lock_id,
            "diagnostic": self.diagnostic,
            "conclusive_proof": self.conclusive_proof,
        }


@dataclass(frozen=True)
class ProgramWorldProofAdmission:
    """Independent admission of reconstructed/replayed current-tree evidence."""

    admission_id: str
    roots: ProgramLogicAuthorityRoots
    disposition: ProgramWorldProofDisposition
    reason_code: ProgramWorldReasonCode
    goal_id: str
    corpus_cid: str
    admitted: bool
    reconstruction: ProgramWorldReconstructionReceipt | None = None
    countermodel: CountermodelValidationReceipt | None = None
    candidate: ProgramWorldHammerCandidate | None = None
    diagnostic: str = ""
    semantic_authority: bool = False

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PROOF_ADMISSION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "admission_id", _identifier(self.admission_id, "admission_id")
        )
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ProgramWorldProofDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, ProgramWorldReasonCode, "reason_code"),
        )
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(self, "corpus_cid", _identifier(self.corpus_cid, "corpus_cid"))
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted"))
        if self.reconstruction is not None and not isinstance(
            self.reconstruction, ProgramWorldReconstructionReceipt
        ):
            raise ProgramWorldHammerError(
                "reconstruction must be ProgramWorldReconstructionReceipt"
            )
        if self.countermodel is not None and not isinstance(
            self.countermodel, CountermodelValidationReceipt
        ):
            raise ProgramWorldHammerError(
                "countermodel must be CountermodelValidationReceipt"
            )
        if self.candidate is not None and not isinstance(
            self.candidate, ProgramWorldHammerCandidate
        ):
            raise ProgramWorldHammerError(
                "candidate must be ProgramWorldHammerCandidate"
            )
        object.__setattr__(
            self, "diagnostic", _text(self.diagnostic, "diagnostic", required=False)
        )
        if self.semantic_authority is not False:
            raise ProgramLogicAuthorityError(
                "proof admission cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        if self.admitted and self.disposition not in {
            ProgramWorldProofDisposition.ADMITTED_PROOF,
            ProgramWorldProofDisposition.ADMITTED_COUNTERMODEL,
        }:
            raise ProgramWorldHammerError(
                "admitted receipts must use admitted_proof or admitted_countermodel"
            )
        if self.disposition is ProgramWorldProofDisposition.ADMITTED_PROOF:
            if self.reconstruction is None or not self.reconstruction.conclusive_proof:
                raise ProgramWorldHammerError(
                    "admitted proof requires conclusive native reconstruction"
                )
            if self.reconstruction.tree_id != self.roots.tree_id:
                raise ProgramWorldStaleBindingError(
                    "reconstruction tree_id does not match current roots"
                )
        if self.disposition is ProgramWorldProofDisposition.ADMITTED_COUNTERMODEL:
            if (
                self.countermodel is None
                or self.countermodel.disposition is not CountermodelDisposition.VALIDATED
            ):
                raise ProgramWorldHammerError(
                    "admitted countermodel requires validated replay or proof of negation"
                )
        if self.admitted and self.candidate is not None and self.candidate.model_origin:
            if self.disposition is ProgramWorldProofDisposition.ADMITTED_PROOF:
                if self.reconstruction is None or not self.reconstruction.conclusive_proof:
                    raise ProgramWorldHammerError(
                        "model candidates cannot create proof authority"
                    )
            if self.disposition is ProgramWorldProofDisposition.ADMITTED_COUNTERMODEL:
                if (
                    self.countermodel is None
                    or self.countermodel.disposition is not CountermodelDisposition.VALIDATED
                ):
                    raise ProgramWorldHammerError(
                        "model candidates cannot create countermodel truth"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "admission_id": self.admission_id,
            "roots": self.roots.to_dict(),
            "disposition": self.disposition.value,
            "reason_code": self.reason_code.value,
            "goal_id": self.goal_id,
            "corpus_cid": self.corpus_cid,
            "admitted": self.admitted,
            "reconstruction": None
            if self.reconstruction is None
            else self.reconstruction.to_dict(),
            "countermodel": None if self.countermodel is None else self.countermodel.to_dict(),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "diagnostic": self.diagnostic,
            "semantic_authority": False,
            "producer_id": HAMMER_PRODUCER_ID,
        }


@dataclass(frozen=True)
class ProgramWorldProofSearchResult:
    """Hammer search result plus independent admission decision."""

    request: ProgramWorldHammerSearchRequest
    admission: ProgramWorldProofAdmission
    capabilities: tuple[ProgramWorldCapabilityReceipt, ...] = ()
    selected_axiom_ids: tuple[str, ...] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PROOF_SEARCH_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProgramWorldHammerSearchRequest):
            raise ProgramWorldHammerError(
                "request must be ProgramWorldHammerSearchRequest"
            )
        if not isinstance(self.admission, ProgramWorldProofAdmission):
            raise ProgramWorldHammerError(
                "admission must be ProgramWorldProofAdmission"
            )
        capabilities = tuple(self.capabilities)
        for item in capabilities:
            if not isinstance(item, ProgramWorldCapabilityReceipt):
                raise ProgramWorldHammerError(
                    "capabilities must be ProgramWorldCapabilityReceipt values"
                )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "selected_axiom_ids",
            _ids(self.selected_axiom_ids, "selected_axiom_ids"),
        )

    @property
    def result_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": PROGRAM_WORLD_PROOF_SEARCH_INTERFACE,
            "evidence": SAWM_TACTICIAN_HAMMER_EVIDENCE,
            "request": self.request.to_dict(),
            "admission": self.admission.to_dict(),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "selected_axiom_ids": list(self.selected_axiom_ids),
            "producer_id": HAMMER_PRODUCER_ID,
            "contract_version": HAMMER_CONTRACT_VERSION,
        }


# ---------------------------------------------------------------------------
# Admission helpers
# ---------------------------------------------------------------------------


def _capability(
    receipts: Sequence[ProgramWorldCapabilityReceipt],
    kind: ProgramWorldCapabilityKind,
) -> ProgramWorldCapabilityReceipt | None:
    for item in receipts:
        if item.kind is kind:
            return item
    return None


def _admission_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        content_identity(payload).encode("utf-8")
    ).hexdigest()
    return f"admission:sha256:{digest}"


def _unavailable_reconstruction(*, roots: ProgramLogicAuthorityRoots, theorem_id: str, diagnostic: str) -> ProgramWorldReconstructionReceipt:
    return ProgramWorldReconstructionReceipt(
        reconstruction_id=f"reconstruction:unavailable:{theorem_id}",
        theorem_id=theorem_id,
        kernel_id="kernel:unavailable",
        tree_id=roots.tree_id,
        environment_id=roots.environment_id,
        toolchain_id=roots.toolchain_id,
        status=ProgramWorldReconstructionStatus.UNAVAILABLE,
        kernel_accepted=False,
        matching_theorem=False,
        sorry_free=False,
        diagnostic=diagnostic,
    )


def _select_axioms(compilation: ProgramWorldGoalCompilation) -> tuple[str, ...]:
    if compilation.tactic_plan is not None:
        selected = tuple(
            item
            for item in compilation.tactic_plan.selected_premise_ids
            if item not in set(compilation.corpus.nominated_premise_ids)
        )
        if selected:
            return selected
    return compilation.corpus.axiom_premise_ids


def admit_program_world_proof(
    *,
    compilation: ProgramWorldGoalCompilation,
    goal: ProgramLogicGoal,
    candidate: ProgramWorldHammerCandidate | None,
    reconstruction: ProgramWorldReconstructionReceipt | None,
    countermodel: CountermodelValidationReceipt | None,
    reason_code: ProgramWorldReasonCode,
    disposition: ProgramWorldProofDisposition,
    diagnostic: str = "",
) -> ProgramWorldProofAdmission:
    """Fail-closed admission: only reconstructed/replayed current-tree evidence."""

    admitted = disposition in {
        ProgramWorldProofDisposition.ADMITTED_PROOF,
        ProgramWorldProofDisposition.ADMITTED_COUNTERMODEL,
    }
    payload = {
        "goal_id": goal.goal_id,
        "disposition": disposition.value,
        "reason_code": reason_code.value,
        "corpus_cid": compilation.corpus.corpus_cid,
        "tree_id": compilation.roots.tree_id,
        "reconstruction_id": ""
        if reconstruction is None
        else reconstruction.reconstruction_id,
        "countermodel_id": "" if countermodel is None else countermodel.receipt_id,
        "candidate_id": "" if candidate is None else candidate.candidate_id,
    }
    return ProgramWorldProofAdmission(
        admission_id=_admission_id(payload),
        roots=compilation.roots,
        disposition=disposition,
        reason_code=reason_code,
        goal_id=goal.goal_id,
        corpus_cid=compilation.corpus.corpus_cid,
        admitted=admitted,
        reconstruction=reconstruction,
        countermodel=countermodel,
        candidate=candidate,
        diagnostic=diagnostic,
        semantic_authority=False,
    )


def _replay_countermodel(
    *,
    compilation: ProgramWorldGoalCompilation,
    goal: ProgramLogicGoal,
    candidate: ProgramWorldHammerCandidate,
    replayer: ProgramWorldCountermodelReplayer | None,
) -> CountermodelValidationReceipt:
    raw_refs = candidate.raw_diagnostic_refs or (
        (candidate.countermodel_id,) if candidate.countermodel_id else (candidate.candidate_id,)
    )
    translation_map_id = candidate.translation_map_id or "translation:unspecified"
    originating_ir = candidate.originating_logic_ir_id or goal.positive_statement_ref
    solver_id = candidate.countermodel_id or candidate.candidate_id

    replay_result: Mapping[str, Any] | None = None
    proof_of_negation_id = ""
    if replayer is not None:
        replay_result = dict(
            replayer.replay(candidate, compilation=compilation, goal=goal)
        )
        proof_of_negation_id = str(replay_result.get("proof_of_negation_id") or "")
    elif candidate.model_origin and candidate.proof_success_claimed:
        # Model-claimed countermodel truth without replay stays diagnostic.
        replay_result = None

    disposition = CountermodelDisposition.DIAGNOSTIC_ONLY
    replayed: tuple[str, ...] = ()
    replay_method = ""
    if proof_of_negation_id:
        disposition = CountermodelDisposition.VALIDATED
        replay_method = "proof_of_negation"
    elif replay_result is not None:
        status = str(
            replay_result.get("status") or replay_result.get("disposition") or ""
        ).lower()
        replay_method = str(
            replay_result.get("replay_method")
            or replay_result.get("method")
            or "deterministic_logic_ir_replay"
        )
        evidence_id = str(
            replay_result.get("evidence_id")
            or replay_result.get("replay_id")
            or f"replay:{candidate.candidate_id}"
        )
        if status in {"validated", "accepted", "unsat", "rejected_hypothesis", "ok"}:
            if str(replay_result.get("tree_id") or compilation.roots.tree_id) != compilation.roots.tree_id:
                disposition = CountermodelDisposition.STALE
            elif str(replay_result.get("originating_logic_ir_id") or originating_ir) != originating_ir:
                disposition = CountermodelDisposition.INCONSISTENT
            else:
                disposition = CountermodelDisposition.VALIDATED
                replayed = (evidence_id,)
        elif status in {"stale", "cross_root", "environment_mismatch"}:
            disposition = CountermodelDisposition.STALE
        elif status == "unsupported":
            disposition = CountermodelDisposition.UNSUPPORTED
        else:
            disposition = CountermodelDisposition.REPLAY_FAILED

    receipt_id = (
        "countermodel-validation:"
        + hashlib.sha256(
            f"{solver_id}:{disposition.value}:{originating_ir}".encode("utf-8")
        ).hexdigest()
    )
    return CountermodelValidationReceipt(
        roots=compilation.roots,
        receipt_id=receipt_id,
        solver_countermodel_id=solver_id,
        translation_map_id=translation_map_id,
        originating_logic_ir_id=originating_ir,
        disposition=disposition,
        raw_diagnostic_refs=raw_refs if disposition is CountermodelDisposition.DIAGNOSTIC_ONLY or raw_refs else raw_refs,
        replayed_rejection_evidence_refs=replayed,
        proof_of_negation_id=proof_of_negation_id,
        replay_method=replay_method,
        invalidation_refs=(compilation.roots.tree_id,),
    )


class ProgramWorldHammer:
    """Operational Hammer adapter with independent reconstruction/replay admission."""

    INTERFACE: ClassVar[str] = PROGRAM_WORLD_HAMMER_INTERFACE

    def __init__(
        self,
        *,
        backend: ProgramWorldHammerBackend | None = None,
        reconstructor: ProgramWorldKernelReconstructor | None = None,
        replayer: ProgramWorldCountermodelReplayer | None = None,
    ) -> None:
        self._backend = backend
        self._reconstructor = reconstructor
        self._replayer = replayer

    def search(
        self,
        compilation: ProgramWorldGoalCompilation,
        *,
        goal_id: str = "",
        timeout_ms: int = 1_000,
        logic_family: str = "",
        expected_tree_id: str = "",
        capabilities: Sequence[ProgramWorldCapabilityReceipt] | None = None,
    ) -> ProgramWorldProofSearchResult:
        if expected_tree_id and expected_tree_id != compilation.roots.tree_id:
            raise ProgramWorldStaleBindingError("search tree_id is not current")
        if not compilation.goals:
            raise ProgramWorldHammerError("compilation contains no goals")
        selected_goal_id = goal_id or compilation.goals[0].goal_id
        family = logic_family or compilation.goals[0].family.value
        request = ProgramWorldHammerSearchRequest(
            compilation=compilation,
            goal_id=selected_goal_id,
            timeout_ms=timeout_ms,
            logic_family=family if family else "fol",
        )
        goal = request.goal
        probed = (
            tuple(capabilities)
            if capabilities is not None
            else tuple(compilation.capabilities)
            or probe_program_world_proof_capabilities()
        )
        axioms = _select_axioms(compilation)

        if compilation.disposition is ProgramWorldCompilationDisposition.CONFLICT:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=None,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.CONTRADICTION,
                disposition=ProgramWorldProofDisposition.CONFLICT,
                diagnostic="contradictory obligations abstain from proof admission",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if compilation.disposition is ProgramWorldCompilationDisposition.STALE:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=None,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.STALE_TREE,
                disposition=ProgramWorldProofDisposition.STALE,
                diagnostic="stale compilation cannot be admitted",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if compilation.disposition is ProgramWorldCompilationDisposition.UNSUPPORTED:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=None,
                reconstruction=None,
                countermodel=None,
                reason_code=compilation.reason_code,
                disposition=ProgramWorldProofDisposition.UNSUPPORTED,
                diagnostic=compilation.diagnostic,
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )

        hammer_cap = _capability(probed, ProgramWorldCapabilityKind.HAMMER)
        kernel_cap = _capability(probed, ProgramWorldCapabilityKind.NATIVE_RECONSTRUCTION)
        backend_cap = _capability(probed, ProgramWorldCapabilityKind.BACKEND)

        if self._backend is None:
            if hammer_cap is None or not hammer_cap.available:
                reason = ProgramWorldReasonCode.HAMMER_UNAVAILABLE
                diagnostic = (
                    hammer_cap.diagnostic
                    if hammer_cap is not None
                    else "production Hammer is not available"
                )
            elif backend_cap is None or not backend_cap.available:
                reason = ProgramWorldReasonCode.BACKEND_UNAVAILABLE
                diagnostic = (
                    backend_cap.diagnostic
                    if backend_cap is not None
                    else "no SMT/ATP backend on PATH"
                )
            elif kernel_cap is None or not kernel_cap.available:
                reason = ProgramWorldReasonCode.KERNEL_UNAVAILABLE
                diagnostic = (
                    kernel_cap.diagnostic
                    if kernel_cap is not None
                    else "native kernel is not available on PATH"
                )
            else:
                reason = ProgramWorldReasonCode.HAMMER_UNAVAILABLE
                diagnostic = (
                    "production Hammer execution is not auto-invoked without an "
                    "admitted native backend"
                )
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=None,
                reconstruction=None,
                countermodel=None,
                reason_code=reason,
                disposition=ProgramWorldProofDisposition.UNAVAILABLE,
                diagnostic=diagnostic,
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )

        candidate = self._backend.search(request)
        if candidate is None:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=None,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.HAMMER_UNAVAILABLE,
                disposition=ProgramWorldProofDisposition.UNAVAILABLE,
                diagnostic="Hammer search returned no candidate",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if not isinstance(candidate, ProgramWorldHammerCandidate):
            raise ProgramWorldHammerError(
                "Hammer backend must return ProgramWorldHammerCandidate"
            )
        if candidate.tree_id != compilation.roots.tree_id:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.STALE_TREE,
                disposition=ProgramWorldProofDisposition.STALE,
                diagnostic="Hammer candidate tree_id is not current",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if candidate.timeout or candidate.status == "timeout":
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.TIMEOUT,
                disposition=ProgramWorldProofDisposition.TIMEOUT,
                diagnostic="Hammer search timed out",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if candidate.status in {"unsupported", "unsupported_translation"}:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.UNSUPPORTED_TRANSLATION,
                disposition=ProgramWorldProofDisposition.UNSUPPORTED,
                diagnostic="Hammer translation or logic family is unsupported",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if candidate.status == "unavailable":
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.HAMMER_UNAVAILABLE,
                disposition=ProgramWorldProofDisposition.UNAVAILABLE,
                diagnostic="Hammer reported unavailability",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )

        if candidate.kind is ProgramWorldCandidateKind.COUNTERMODEL or candidate.status in {
            "counterexample",
            "countermodel",
        }:
            replay_receipt = _replay_countermodel(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                replayer=self._replayer,
            )
            if replay_receipt.disposition is CountermodelDisposition.VALIDATED:
                if candidate.model_origin and self._replayer is None:
                    admission = admit_program_world_proof(
                        compilation=compilation,
                        goal=goal,
                        candidate=candidate,
                        reconstruction=None,
                        countermodel=replay_receipt,
                        reason_code=ProgramWorldReasonCode.MODEL_CANNOT_CREATE_COUNTERMODEL_TRUTH,
                        disposition=ProgramWorldProofDisposition.REJECTED,
                        diagnostic="model-nominated countermodels cannot self-validate",
                    )
                else:
                    admission = admit_program_world_proof(
                        compilation=compilation,
                        goal=goal,
                        candidate=candidate,
                        reconstruction=None,
                        countermodel=replay_receipt,
                        reason_code=ProgramWorldReasonCode.REPLAYED,
                        disposition=ProgramWorldProofDisposition.ADMITTED_COUNTERMODEL,
                        diagnostic="countermodel replayed against originating LogicIR",
                    )
            elif replay_receipt.disposition is CountermodelDisposition.STALE:
                admission = admit_program_world_proof(
                    compilation=compilation,
                    goal=goal,
                    candidate=candidate,
                    reconstruction=None,
                    countermodel=replay_receipt,
                    reason_code=ProgramWorldReasonCode.STALE_TREE,
                    disposition=ProgramWorldProofDisposition.STALE,
                    diagnostic="countermodel replay bound a stale tree",
                )
            else:
                admission = admit_program_world_proof(
                    compilation=compilation,
                    goal=goal,
                    candidate=candidate,
                    reconstruction=None,
                    countermodel=replay_receipt,
                    reason_code=ProgramWorldReasonCode.DIAGNOSTIC_ONLY,
                    disposition=ProgramWorldProofDisposition.DIAGNOSTIC,
                    diagnostic="raw solver countermodel is diagnostic until replayed",
                )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )

        reconstruction: ProgramWorldReconstructionReceipt | None = None
        if self._reconstructor is not None:
            reconstruction = self._reconstructor.reconstruct(
                candidate, compilation=compilation, goal=goal
            )
            if not isinstance(reconstruction, ProgramWorldReconstructionReceipt):
                raise ProgramWorldHammerError(
                    "reconstructor must return ProgramWorldReconstructionReceipt"
                )
        elif kernel_cap is None or not kernel_cap.available:
            reconstruction = _unavailable_reconstruction(
                roots=compilation.roots,
                theorem_id=goal.positive_statement_ref,
                diagnostic=(
                    kernel_cap.diagnostic
                    if kernel_cap is not None
                    else "native kernel is not available on PATH"
                ),
            )
            if candidate.proof_success_claimed or candidate.kernel_accepted_claimed:
                admission = admit_program_world_proof(
                    compilation=compilation,
                    goal=goal,
                    candidate=candidate,
                    reconstruction=reconstruction,
                    countermodel=None,
                    reason_code=(
                        ProgramWorldReasonCode.MODEL_CANNOT_CREATE_PROOF
                        if candidate.model_origin
                        else ProgramWorldReasonCode.CANDIDATE_NOT_ADMITTED
                    ),
                    disposition=ProgramWorldProofDisposition.CANDIDATE,
                    diagnostic="solver/model claims cannot admit without kernel reconstruction",
                )
                return ProgramWorldProofSearchResult(
                    request=request,
                    admission=admission,
                    capabilities=probed,
                    selected_axiom_ids=axioms,
                )
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=reconstruction,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.KERNEL_UNAVAILABLE,
                disposition=ProgramWorldProofDisposition.UNAVAILABLE,
                diagnostic=reconstruction.diagnostic,
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )

        assert reconstruction is not None
        if reconstruction.tree_id != compilation.roots.tree_id:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=reconstruction,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.STALE_TREE,
                disposition=ProgramWorldProofDisposition.STALE,
                diagnostic="reconstruction tree_id is not current",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if reconstruction.theorem_id not in {
            goal.positive_statement_ref,
            candidate.theorem_id,
            goal.goal_id,
        } or not reconstruction.matching_theorem:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=reconstruction,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.CANDIDATE_NOT_ADMITTED,
                disposition=ProgramWorldProofDisposition.REJECTED,
                diagnostic="reconstructed theorem does not match the compiled goal",
            )
            return ProgramWorldProofSearchResult(
                request=request,
                admission=admission,
                capabilities=probed,
                selected_axiom_ids=axioms,
            )
        if reconstruction.conclusive_proof:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=reconstruction,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.RECONSTRUCTED,
                disposition=ProgramWorldProofDisposition.ADMITTED_PROOF,
                diagnostic="native kernel accepted the reconstructed proof",
            )
        elif candidate.model_origin:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=reconstruction,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.MODEL_CANNOT_CREATE_PROOF,
                disposition=ProgramWorldProofDisposition.REJECTED,
                diagnostic="model-nominated proof candidates cannot self-admit",
            )
        else:
            admission = admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=reconstruction,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.CANDIDATE_NOT_ADMITTED,
                disposition=ProgramWorldProofDisposition.CANDIDATE,
                diagnostic="Hammer candidate remains non-authoritative",
            )
        return ProgramWorldProofSearchResult(
            request=request,
            admission=admission,
            capabilities=probed,
            selected_axiom_ids=axioms,
        )

    def replay_countermodel(
        self,
        compilation: ProgramWorldGoalCompilation,
        candidate: ProgramWorldHammerCandidate,
        *,
        goal_id: str = "",
    ) -> ProgramWorldProofAdmission:
        if not compilation.goals:
            raise ProgramWorldHammerError("compilation contains no goals")
        selected_goal_id = goal_id or compilation.goals[0].goal_id
        goal = next(
            item for item in compilation.goals if item.goal_id == selected_goal_id
        )
        if candidate.tree_id != compilation.roots.tree_id:
            return admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=None,
                countermodel=None,
                reason_code=ProgramWorldReasonCode.STALE_TREE,
                disposition=ProgramWorldProofDisposition.STALE,
                diagnostic="countermodel tree_id is not current",
            )
        replay_receipt = _replay_countermodel(
            compilation=compilation,
            goal=goal,
            candidate=candidate,
            replayer=self._replayer,
        )
        if replay_receipt.disposition is CountermodelDisposition.VALIDATED:
            if candidate.model_origin and self._replayer is None:
                return admit_program_world_proof(
                    compilation=compilation,
                    goal=goal,
                    candidate=candidate,
                    reconstruction=None,
                    countermodel=replay_receipt,
                    reason_code=ProgramWorldReasonCode.MODEL_CANNOT_CREATE_COUNTERMODEL_TRUTH,
                    disposition=ProgramWorldProofDisposition.REJECTED,
                    diagnostic="model-nominated countermodels cannot self-validate",
                )
            return admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=None,
                countermodel=replay_receipt,
                reason_code=ProgramWorldReasonCode.REPLAYED,
                disposition=ProgramWorldProofDisposition.ADMITTED_COUNTERMODEL,
                diagnostic="countermodel replayed against originating LogicIR",
            )
        if replay_receipt.disposition is CountermodelDisposition.STALE:
            return admit_program_world_proof(
                compilation=compilation,
                goal=goal,
                candidate=candidate,
                reconstruction=None,
                countermodel=replay_receipt,
                reason_code=ProgramWorldReasonCode.STALE_TREE,
                disposition=ProgramWorldProofDisposition.STALE,
                diagnostic="countermodel replay bound a stale tree",
            )
        return admit_program_world_proof(
            compilation=compilation,
            goal=goal,
            candidate=candidate,
            reconstruction=None,
            countermodel=replay_receipt,
            reason_code=ProgramWorldReasonCode.DIAGNOSTIC_ONLY,
            disposition=ProgramWorldProofDisposition.DIAGNOSTIC,
            diagnostic="raw solver countermodel is diagnostic until replayed",
        )


def search_program_world_proof(
    compilation: ProgramWorldGoalCompilation,
    *,
    goal_id: str = "",
    timeout_ms: int = 1_000,
    logic_family: str = "",
    expected_tree_id: str = "",
    backend: ProgramWorldHammerBackend | None = None,
    reconstructor: ProgramWorldKernelReconstructor | None = None,
    replayer: ProgramWorldCountermodelReplayer | None = None,
    capabilities: Sequence[ProgramWorldCapabilityReceipt] | None = None,
) -> ProgramWorldProofSearchResult:
    """Search with production Hammer and admit only reconstructed/replayed evidence."""

    return ProgramWorldHammer(
        backend=backend,
        reconstructor=reconstructor,
        replayer=replayer,
    ).search(
        compilation,
        goal_id=goal_id,
        timeout_ms=timeout_ms,
        logic_family=logic_family,
        expected_tree_id=expected_tree_id,
        capabilities=capabilities,
    )


def replay_program_world_countermodel(
    compilation: ProgramWorldGoalCompilation,
    candidate: ProgramWorldHammerCandidate,
    *,
    goal_id: str = "",
    replayer: ProgramWorldCountermodelReplayer | None = None,
) -> ProgramWorldProofAdmission:
    """Deterministically replay a solver countermodel before refutation admission."""

    return ProgramWorldHammer(replayer=replayer).replay_countermodel(
        compilation,
        candidate,
        goal_id=goal_id,
    )


__all__ = [
    "HAMMER_PRODUCER_ID",
    "PROGRAM_WORLD_HAMMER_INTERFACE",
    "PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE",
    "ProgramWorldCandidateKind",
    "ProgramWorldCountermodelReplayer",
    "ProgramWorldHammer",
    "ProgramWorldHammerBackend",
    "ProgramWorldHammerCandidate",
    "ProgramWorldHammerError",
    "ProgramWorldHammerSearchRequest",
    "ProgramWorldKernelReconstructor",
    "ProgramWorldProofAdmission",
    "ProgramWorldProofDisposition",
    "ProgramWorldProofSearchResult",
    "ProgramWorldReconstructionReceipt",
    "ProgramWorldReconstructionStatus",
    "admit_program_world_proof",
    "replay_program_world_countermodel",
    "search_program_world_proof",
]
