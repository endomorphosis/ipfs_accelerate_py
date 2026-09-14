"""Program-world production Hammer integration (SAWM-018).

Interface: ``ProgramWorldHammer@1``
Evidence: ``sawm/tactician-hammer@1``

Run the current production Hammer over compiled program-world obligations,
then reconstruct proofs in native kernels or deterministically replay
countermodels *before* admission.  This module does not create a second
prover, proof store, tactic engine, logic family, or acceptance gate.

Normative constraints:

* Only reconstructed or independently replayed current-tree evidence is
  admitted.
* Solver SAT, model drafts, and raw diagnostics never become proof,
  countermodel truth, or authority.
* Contradictions abstain (no ex-falso).
* Timeout, unsupported logic, and missing backends are typed.
* Unavailability is judged against the sealed validation PATH, not a
  provider-side probe.
* Importing this module performs no I/O, starts no subprocess, and does
  not import optional Hammer kernels.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ..analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    CountermodelValidationReceipt,
    ProgramLogicAuthorityRoots,
    ProofStatus,
)
from .formal_verification_contracts import CanonicalContract, content_identity
from .program_world_tactician import (
    PROGRAM_WORLD_TACTICIAN_INTERFACE,
    SAWM_TACTICIAN_HAMMER_EVIDENCE,
    ProgramWorldCompilationDisposition,
    ProgramWorldGoalCompilation,
    ProgramWorldTacticianAuthorityError,
    ProgramWorldTacticianError,
    _roots,
    _text,
)


PROGRAM_WORLD_HAMMER_INTERFACE: Final[str] = "ProgramWorldHammer@1"
PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE: Final[str] = "ProgramWorldProofAdmission@1"

PROGRAM_WORLD_BACKEND_PROBE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-backend-probe@1"
)
PROGRAM_WORLD_RECONSTRUCTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-reconstruction@1"
)
PROGRAM_WORLD_COUNTERMODEL_REPLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-countermodel-replay@1"
)
PROGRAM_WORLD_PROOF_SEARCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-proof-search@1"
)
PROGRAM_WORLD_PROOF_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-proof-admission@1"
)

PRODUCER_ID: Final[str] = "program-world-hammer@1"
HAMMER_VERSION: Final[str] = "1"

ITP_EXECUTABLES: Final[tuple[str, ...]] = ("lean", "coqc", "isabelle")
HAMMER_MODULES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py.logic.hammers.reconstruction",
    "ipfs_datasets_py.logic.hammers.portfolio",
    "ipfs_datasets_py.logic.hammers.premise_selection",
)
VALIDATION_PATH: Final[str] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"

HammerSearchFn = Callable[[ProgramWorldGoalCompilation, Mapping[str, Any]], Mapping[str, Any]]
ReconstructionFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ReplayFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class CountermodelValidateFn(Protocol):
    """Independent countermodel replay adapter.  Never proof authority."""

    def validate(self, **kwargs: Any) -> CountermodelValidationReceipt: ...


class ProgramWorldHammerError(ProgramWorldTacticianError):
    """Hammer search or replay inputs cannot produce trustworthy evidence."""


class ProgramWorldHammerAuthorityError(ProgramWorldTacticianAuthorityError):
    """A model, solver, or stale result attempted to mint proof authority."""


class ProgramWorldSearchOutcome(str, Enum):
    """Closed search outcomes.  Verified/refuted require independent evidence."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    CANDIDATE = "candidate"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"
    ABSTAINED = "abstained"
    STALE = "stale"
    ERROR = "error"


class ProgramWorldAdmissionDisposition(str, Enum):
    """Closed admission outcomes for reconstructed/replayed evidence."""

    ADMITTED_PROOF = "admitted_proof"
    ADMITTED_REFUTATION = "admitted_refutation"
    REJECTED = "rejected"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


class ProgramWorldBackendStatus(str, Enum):
    """Typed backend availability.  Absent tools are never 'supported'."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramWorldHammerError(f"{field_name} must be a boolean")
    return value


def _reason_codes(*groups: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
    return tuple(ordered)


def _compact_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def _validate_countermodel(
    *,
    roots: ProgramLogicAuthorityRoots,
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
    """Replay-gate a solver countermodel.  Raw diagnostics stay non-authoritative."""

    raw_refs = tuple(
        _text(item, "raw_diagnostic_refs") for item in (raw_diagnostic_refs or ())
    )
    if not raw_refs and not proof_of_negation_id and not replay_result:
        raise ProgramWorldHammerError(
            "countermodel validation requires raw diagnostics, replay, or proof of negation"
        )
    inv = tuple(
        _text(item, "invalidation_refs")
        for item in (invalidation_refs or (roots.tree_id,))
    )
    replayed: list[str] = []
    replay_method = ""
    disposition = CountermodelDisposition.DIAGNOSTIC_ONLY
    if proof_of_negation_id:
        disposition = CountermodelDisposition.VALIDATED
        replay_method = "proof_of_negation"
    elif replay_result is not None:
        replay = _mapping(replay_result, "replay_result")
        status = str(replay.get("status") or replay.get("disposition") or "").lower()
        method = str(
            replay.get("replay_method")
            or replay.get("method")
            or "deterministic_logic_ir_replay"
        )
        evidence_id = str(
            replay.get("evidence_id")
            or replay.get("replay_id")
            or _compact_id(str(sorted(replay.items())), prefix="replay")
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
    if disposition is CountermodelDisposition.DIAGNOSTIC_ONLY and not raw_refs:
        raw_refs = (
            _text(solver_countermodel_id, "raw_diagnostic_refs"),
        )
    receipt_id = _compact_id(
        f"{solver_countermodel_id}:{translation_map_id}:{originating_logic_ir_id}:{disposition.value}",
        prefix="countermodel-validation",
    )
    return CountermodelValidationReceipt(
        roots=roots,
        receipt_id=receipt_id,
        solver_countermodel_id=_text(solver_countermodel_id, "solver_countermodel_id"),
        translation_map_id=_text(translation_map_id, "translation_map_id"),
        originating_logic_ir_id=_text(
            originating_logic_ir_id, "originating_logic_ir_id"
        ),
        disposition=disposition,
        raw_diagnostic_refs=raw_refs,
        replayed_rejection_evidence_refs=tuple(replayed),
        proof_of_negation_id=_text(
            proof_of_negation_id, "proof_of_negation_id", required=False
        ),
        replay_method=replay_method,
        assumption_refs=tuple(
            _text(item, "assumption_refs") for item in assumption_refs
        ),
        resource_policy_ref=_text(
            resource_policy_ref, "resource_policy_ref", required=False
        ),
        invalidation_refs=inv,
    )


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramWorldHammerError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _find_spec_without_import(module: str) -> Any | None:
    import importlib.machinery

    parts = [part for part in str(module).split(".") if part]
    if not parts:
        return None
    spec = None
    parent = ""
    for index, part in enumerate(parts):
        name = part if not parent else f"{parent}.{part}"
        path = None if spec is None else getattr(spec, "submodule_search_locations", None)
        if index == 0:
            spec = importlib.machinery.PathFinder.find_spec(name)
        else:
            if not path:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        parent = name
    return spec


def _which(executable: str, *, path: str | None = None) -> str | None:
    return shutil.which(executable, path=path)


@dataclass(frozen=True)
class ProgramWorldBackendProbe(CanonicalContract):
    """Exact, non-authoritative observation of Hammer/ITP availability."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_BACKEND_PROBE_SCHEMA

    kernel_status: ProgramWorldBackendStatus
    hammer_status: ProgramWorldBackendStatus
    solver_status: ProgramWorldBackendStatus
    reconstruction_ready: bool
    kernel_executables: tuple[str, ...] = ()
    hammer_modules: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_status",
            self.kernel_status
            if isinstance(self.kernel_status, ProgramWorldBackendStatus)
            else ProgramWorldBackendStatus(str(self.kernel_status)),
        )
        object.__setattr__(
            self,
            "hammer_status",
            self.hammer_status
            if isinstance(self.hammer_status, ProgramWorldBackendStatus)
            else ProgramWorldBackendStatus(str(self.hammer_status)),
        )
        object.__setattr__(
            self,
            "solver_status",
            self.solver_status
            if isinstance(self.solver_status, ProgramWorldBackendStatus)
            else ProgramWorldBackendStatus(str(self.solver_status)),
        )
        object.__setattr__(
            self, "reconstruction_ready", _bool(self.reconstruction_ready, "reconstruction_ready")
        )
        object.__setattr__(
            self,
            "kernel_executables",
            tuple(_text(item, "kernel_executables") for item in (self.kernel_executables or ())),
        )
        object.__setattr__(
            self,
            "hammer_modules",
            tuple(_text(item, "hammer_modules") for item in (self.hammer_modules or ())),
        )
        object.__setattr__(
            self, "missing", tuple(_text(item, "missing") for item in (self.missing or ()))
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(item, "reason_codes") for item in (self.reason_codes or ())),
        )
        object.__setattr__(self, "path", _text(self.path, "path", required=False))
        if self.reconstruction_ready and self.kernel_status is not ProgramWorldBackendStatus.AVAILABLE:
            raise ProgramWorldHammerAuthorityError(
                "reconstruction_ready requires an available kernel backend"
            )

    @property
    def available(self) -> bool:
        return (
            self.kernel_status is ProgramWorldBackendStatus.AVAILABLE
            and self.hammer_status is ProgramWorldBackendStatus.AVAILABLE
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "kernel_status": self.kernel_status.value,
            "hammer_status": self.hammer_status.value,
            "solver_status": self.solver_status.value,
            "reconstruction_ready": self.reconstruction_ready,
            "kernel_executables": list(self.kernel_executables),
            "hammer_modules": list(self.hammer_modules),
            "missing": list(self.missing),
            "reason_codes": list(self.reason_codes),
            "path": self.path,
            "proof_success": False,
        }


def probe_program_world_backends(
    *,
    path: str | None = None,
    which: Callable[[str], str | None] | None = None,
    find_spec: Callable[[str], Any | None] | None = None,
    probe_solvers: bool = True,
) -> ProgramWorldBackendProbe:
    """Observe Hammer/ITP availability against the sealed validation PATH.

    This receipt never sets proof success and never installs tools.
    """

    search_path = path if path is not None else VALIDATION_PATH
    finder = find_spec or _find_spec_without_import
    locator = which or (lambda name: _which(name, path=search_path))

    kernel_hits: list[str] = []
    missing: list[str] = []
    reasons: list[str] = []
    for executable in ITP_EXECUTABLES:
        located = locator(executable)
        if located:
            kernel_hits.append(f"{executable}:{located}")
        else:
            missing.append(f"executable:{executable}")
    hammer_hits: list[str] = []
    for module in HAMMER_MODULES:
        spec = finder(module)
        if spec is not None:
            location = str(getattr(spec, "origin", "") or "")
            hammer_hits.append(module if not location else f"{module}:{location}")
        else:
            missing.append(f"module:{module}")

    solver_status = ProgramWorldBackendStatus.UNAVAILABLE
    if probe_solvers:
        try:
            from .solver_readiness import (
                SolverBackendFamily,
                SolverReadinessStatus,
                probe_solver_readiness,
            )

            report = probe_solver_readiness()
            hammer = next(
                item
                for item in report.backends
                if item.family is SolverBackendFamily.HAMMER
            )
            if hammer.status is SolverReadinessStatus.UNSUPPORTED:
                solver_status = ProgramWorldBackendStatus.UNSUPPORTED
                reasons.append("solver_hammer_unsupported")
            elif hammer.supported:
                solver_status = ProgramWorldBackendStatus.AVAILABLE
            else:
                reasons.append("solver_hammer_unavailable")
        except Exception:
            reasons.append("solver_readiness_unavailable")
            solver_status = ProgramWorldBackendStatus.UNAVAILABLE

    kernel_status = (
        ProgramWorldBackendStatus.AVAILABLE
        if kernel_hits
        else ProgramWorldBackendStatus.UNAVAILABLE
    )
    hammer_status = (
        ProgramWorldBackendStatus.AVAILABLE
        if len(hammer_hits) == len(HAMMER_MODULES)
        else ProgramWorldBackendStatus.UNAVAILABLE
    )
    if kernel_status is ProgramWorldBackendStatus.UNAVAILABLE:
        reasons.append("kernel_unavailable")
    if hammer_status is ProgramWorldBackendStatus.UNAVAILABLE:
        reasons.append("hammer_modules_unavailable")
    reconstruction_ready = (
        kernel_status is ProgramWorldBackendStatus.AVAILABLE
        and hammer_status is ProgramWorldBackendStatus.AVAILABLE
    )
    if not reconstruction_ready:
        reasons.append("reconstruction_not_ready")
    return ProgramWorldBackendProbe(
        kernel_status=kernel_status,
        hammer_status=hammer_status,
        solver_status=solver_status,
        reconstruction_ready=reconstruction_ready,
        kernel_executables=tuple(kernel_hits),
        hammer_modules=tuple(hammer_hits),
        missing=tuple(missing),
        reason_codes=_reason_codes(reasons),
        path=search_path,
    )


@dataclass(frozen=True)
class ProgramWorldReconstructionReceipt(CanonicalContract):
    """Independent native-kernel reconstruction of a proof candidate."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_RECONSTRUCTION_SCHEMA

    reconstruction_id: str
    tree_id: str
    obligation_id: str
    kernel_id: str
    kernel_accepted: bool
    native_source_id: str = ""
    reason_code: str = ""
    model_authored: bool = False
    solver_sat: bool = False
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reconstruction_id", _text(self.reconstruction_id, "reconstruction_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(self, "kernel_id", _text(self.kernel_id, "kernel_id"))
        object.__setattr__(
            self, "kernel_accepted", _bool(self.kernel_accepted, "kernel_accepted")
        )
        object.__setattr__(
            self,
            "native_source_id",
            _text(self.native_source_id, "native_source_id", required=False),
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", required=False)
        )
        object.__setattr__(
            self, "model_authored", _bool(self.model_authored, "model_authored")
        )
        object.__setattr__(self, "solver_sat", _bool(self.solver_sat, "solver_sat"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
        if self.kernel_accepted and not self.native_source_id:
            raise ProgramWorldHammerAuthorityError(
                "kernel-accepted reconstructions require a native source identity"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "reconstruction_id": self.reconstruction_id,
            "tree_id": self.tree_id,
            "obligation_id": self.obligation_id,
            "kernel_id": self.kernel_id,
            "kernel_accepted": self.kernel_accepted,
            "native_source_id": self.native_source_id,
            "reason_code": self.reason_code,
            "model_authored": self.model_authored,
            "solver_sat": self.solver_sat,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProgramWorldCountermodelReplay(CanonicalContract):
    """Independent countermodel replay bound to current-tree LogicIR."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_COUNTERMODEL_REPLAY_SCHEMA

    replay_id: str
    tree_id: str
    validation: CountermodelValidationReceipt
    model_authored: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "replay_id", _text(self.replay_id, "replay_id"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        if not isinstance(self.validation, CountermodelValidationReceipt):
            raise ProgramWorldHammerError(
                "validation must be CountermodelValidationReceipt"
            )
        object.__setattr__(
            self, "model_authored", _bool(self.model_authored, "model_authored")
        )
        if self.validation.roots.tree_id != self.tree_id:
            raise ProgramWorldHammerAuthorityError(
                "countermodel replay must bind the current tree"
            )

    @property
    def validated(self) -> bool:
        return self.validation.disposition is CountermodelDisposition.VALIDATED

    def _payload(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "tree_id": self.tree_id,
            "validation": self.validation.to_dict(),
            "model_authored": self.model_authored,
            "validated": self.validated,
        }


@dataclass(frozen=True)
class ProgramWorldProofAdmission(CanonicalContract):
    """Fail-closed admission of reconstructed/replayed current-tree evidence."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PROOF_ADMISSION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE

    admission_id: str
    disposition: ProgramWorldAdmissionDisposition
    tree_id: str
    compilation_id: str
    proof_status: ProofStatus
    reason_codes: tuple[str, ...] = ()
    reconstruction_id: str = ""
    replay_id: str = ""
    model_authored: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "admission_id", _text(self.admission_id, "admission_id"))
        disposition = (
            self.disposition
            if isinstance(self.disposition, ProgramWorldAdmissionDisposition)
            else ProgramWorldAdmissionDisposition(str(self.disposition))
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "compilation_id", _text(self.compilation_id, "compilation_id")
        )
        status = (
            self.proof_status
            if isinstance(self.proof_status, ProofStatus)
            else ProofStatus(str(self.proof_status))
        )
        object.__setattr__(self, "proof_status", status)
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(item, "reason_codes") for item in (self.reason_codes or ())),
        )
        object.__setattr__(
            self,
            "reconstruction_id",
            _text(self.reconstruction_id, "reconstruction_id", required=False),
        )
        object.__setattr__(
            self, "replay_id", _text(self.replay_id, "replay_id", required=False)
        )
        object.__setattr__(
            self, "model_authored", _bool(self.model_authored, "model_authored")
        )
        if disposition is ProgramWorldAdmissionDisposition.ADMITTED_PROOF:
            if status is not ProofStatus.KERNEL_VERIFIED:
                raise ProgramWorldHammerAuthorityError(
                    "admitted proofs require kernel-verified proof status"
                )
            if not self.reconstruction_id:
                raise ProgramWorldHammerAuthorityError(
                    "admitted proofs require a reconstruction identity"
                )
        if disposition is ProgramWorldAdmissionDisposition.ADMITTED_REFUTATION:
            if status is not ProofStatus.VALIDATED_REFUTED:
                raise ProgramWorldHammerAuthorityError(
                    "admitted refutations require validated-refuted proof status"
                )
            if not self.replay_id:
                raise ProgramWorldHammerAuthorityError(
                    "admitted refutations require a replay identity"
                )
        if self.model_authored and disposition in {
            ProgramWorldAdmissionDisposition.ADMITTED_PROOF,
            ProgramWorldAdmissionDisposition.ADMITTED_REFUTATION,
        }:
            raise ProgramWorldHammerAuthorityError(
                "model-authored candidates cannot be admitted as proof or refutation"
            )

    @property
    def admitted(self) -> bool:
        return self.disposition in {
            ProgramWorldAdmissionDisposition.ADMITTED_PROOF,
            ProgramWorldAdmissionDisposition.ADMITTED_REFUTATION,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "admission_id": self.admission_id,
            "disposition": self.disposition.value,
            "tree_id": self.tree_id,
            "compilation_id": self.compilation_id,
            "proof_status": self.proof_status.value,
            "reason_codes": list(self.reason_codes),
            "reconstruction_id": self.reconstruction_id,
            "replay_id": self.replay_id,
            "model_authored": self.model_authored,
            "admitted": self.admitted,
        }


@dataclass(frozen=True)
class ProgramWorldProofSearchReceipt(CanonicalContract):
    """One Hammer search plus independent reconstruction/replay/admission."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PROOF_SEARCH_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_HAMMER_INTERFACE

    receipt_id: str
    outcome: ProgramWorldSearchOutcome
    compilation_id: str
    tree_id: str
    premise_corpus_cid: str
    tactic_plan_id: str
    admission: ProgramWorldProofAdmission
    backend: ProgramWorldBackendProbe
    reconstruction: ProgramWorldReconstructionReceipt | None = None
    replay: ProgramWorldCountermodelReplay | None = None
    reason_codes: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, "receipt_id"))
        outcome = (
            self.outcome
            if isinstance(self.outcome, ProgramWorldSearchOutcome)
            else ProgramWorldSearchOutcome(str(self.outcome))
        )
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self, "compilation_id", _text(self.compilation_id, "compilation_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "premise_corpus_cid", _text(self.premise_corpus_cid, "premise_corpus_cid")
        )
        object.__setattr__(
            self, "tactic_plan_id", _text(self.tactic_plan_id, "tactic_plan_id")
        )
        if not isinstance(self.admission, ProgramWorldProofAdmission):
            raise ProgramWorldHammerError("admission must be ProgramWorldProofAdmission")
        if not isinstance(self.backend, ProgramWorldBackendProbe):
            raise ProgramWorldHammerError("backend must be ProgramWorldBackendProbe")
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(item, "reason_codes") for item in (self.reason_codes or ())),
        )
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        if (
            outcome is ProgramWorldSearchOutcome.VERIFIED
            and self.admission.disposition
            is not ProgramWorldAdmissionDisposition.ADMITTED_PROOF
        ):
            raise ProgramWorldHammerAuthorityError(
                "verified outcomes require admitted reconstructed proof"
            )
        if (
            outcome is ProgramWorldSearchOutcome.REFUTED
            and self.admission.disposition
            is not ProgramWorldAdmissionDisposition.ADMITTED_REFUTATION
        ):
            raise ProgramWorldHammerAuthorityError(
                "refuted outcomes require admitted replayed countermodels"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence": SAWM_TACTICIAN_HAMMER_EVIDENCE,
            "producer_id": self.producer_id,
            "hammer_version": HAMMER_VERSION,
            "receipt_id": self.receipt_id,
            "outcome": self.outcome.value,
            "compilation_id": self.compilation_id,
            "tree_id": self.tree_id,
            "premise_corpus_cid": self.premise_corpus_cid,
            "tactic_plan_id": self.tactic_plan_id,
            "admission": self.admission.to_dict(),
            "backend": self.backend.to_dict(),
            "reconstruction": (
                None if self.reconstruction is None else self.reconstruction.to_dict()
            ),
            "replay": None if self.replay is None else self.replay.to_dict(),
            "reason_codes": list(self.reason_codes),
            "tactician_interface": PROGRAM_WORLD_TACTICIAN_INTERFACE,
        }


def _decode_compilation(
    compilation: ProgramWorldGoalCompilation | Mapping[str, Any],
) -> ProgramWorldGoalCompilation:
    if isinstance(compilation, ProgramWorldGoalCompilation):
        return compilation
    if isinstance(compilation, Mapping):
        raise ProgramWorldHammerError(
            "proof search requires a typed ProgramWorldGoalCompilation"
        )
    raise ProgramWorldHammerError("compilation must be ProgramWorldGoalCompilation")


def _stale_tree(
    roots: ProgramLogicAuthorityRoots, expected_tree_id: str | None
) -> bool:
    expected = _text(expected_tree_id or "", "expected_tree_id", required=False)
    return bool(expected) and expected != roots.tree_id


def reconstruct_program_world_proof(
    compilation: ProgramWorldGoalCompilation,
    candidate: Mapping[str, Any],
    *,
    reconstructor: ReconstructionFn | None = None,
    expected_tree_id: str | None = None,
    backend: ProgramWorldBackendProbe | None = None,
) -> ProgramWorldReconstructionReceipt:
    """Reconstruct a proof candidate in a pinned kernel or typed-unavailable."""

    payload = _mapping(candidate, "candidate")
    tree_id = compilation.roots.tree_id
    if _stale_tree(compilation.roots, expected_tree_id):
        return ProgramWorldReconstructionReceipt(
            reconstruction_id=_compact_id(tree_id, prefix="recon"),
            tree_id=tree_id,
            obligation_id=_text(
                payload.get("obligation_id")
                or (compilation.obligations[0].obligation_id if compilation.obligations else "obl:none"),
                "obligation_id",
            ),
            kernel_id="kernel:none",
            kernel_accepted=False,
            reason_code="stale_tree",
            model_authored=bool(payload.get("model_authored")),
            solver_sat=bool(payload.get("solver_sat")),
        )
    model_authored = bool(payload.get("model_authored"))
    solver_sat = bool(payload.get("solver_sat") or payload.get("sat"))
    obligation_id = _text(
        payload.get("obligation_id")
        or (compilation.obligations[0].obligation_id if compilation.obligations else "obl:none"),
        "obligation_id",
    )
    if reconstructor is not None:
        raw = _mapping(reconstructor(payload), "reconstruction")
        accepted = bool(raw.get("kernel_accepted") or raw.get("accepted"))
        native = _text(
            raw.get("native_source_id") or raw.get("native_source") or "",
            "native_source_id",
            required=False,
        )
        kernel_id = _text(
            raw.get("kernel_id") or payload.get("kernel_id") or "kernel:injected",
            "kernel_id",
        )
        reason = _text(
            raw.get("reason_code") or ("kernel_accepted" if accepted else "kernel_rejected"),
            "reason_code",
            required=False,
        )
        if accepted and not native:
            accepted = False
            reason = "missing_native_source"
        return ProgramWorldReconstructionReceipt(
            reconstruction_id=_compact_id(
                f"{tree_id}:{obligation_id}:{native}:{accepted}", prefix="recon"
            ),
            tree_id=tree_id,
            obligation_id=obligation_id,
            kernel_id=kernel_id,
            kernel_accepted=accepted,
            native_source_id=native,
            reason_code=reason,
            model_authored=model_authored,
            solver_sat=solver_sat,
            metadata={"injected": True, "solver_claim": str(payload.get("outcome") or "")},
        )

    probe = backend or probe_program_world_backends()
    if not probe.reconstruction_ready:
        return ProgramWorldReconstructionReceipt(
            reconstruction_id=_compact_id(f"{tree_id}:{obligation_id}", prefix="recon"),
            tree_id=tree_id,
            obligation_id=obligation_id,
            kernel_id="kernel:unavailable",
            kernel_accepted=False,
            reason_code="kernel_unavailable",
            model_authored=model_authored,
            solver_sat=solver_sat,
            metadata={"backend": probe.to_dict()},
        )
    # Native reconstruction is available but must still be invoked through the
    # landed datasets Hammer reconstructor.  Without an injected adapter this
    # path remains non-admitted: solver SAT is not proof.
    return ProgramWorldReconstructionReceipt(
        reconstruction_id=_compact_id(f"{tree_id}:{obligation_id}", prefix="recon"),
        tree_id=tree_id,
        obligation_id=obligation_id,
        kernel_id="kernel:pending-native",
        kernel_accepted=False,
        reason_code="native_reconstruction_requires_adapter_or_digest_bound_kernel",
        model_authored=model_authored,
        solver_sat=solver_sat,
        metadata={"backend": probe.to_dict()},
    )


def replay_program_world_countermodel(
    compilation: ProgramWorldGoalCompilation | ProgramLogicAuthorityRoots | Mapping[str, Any],
    countermodel: Mapping[str, Any],
    *,
    replay: ReplayFn | None = None,
    expected_tree_id: str | None = None,
    validator: CountermodelValidateFn | None = None,
) -> ProgramWorldCountermodelReplay:
    """Independently replay a solver countermodel before refutation admission."""

    if isinstance(compilation, ProgramWorldGoalCompilation):
        roots = compilation.roots
    else:
        roots = _roots(compilation)
    payload = _mapping(countermodel, "countermodel")
    if _stale_tree(roots, expected_tree_id):
        raise ProgramWorldHammerAuthorityError(
            "countermodel replay refused for stale tree bindings"
        )
    model_authored = bool(payload.get("model_authored"))
    replay_result = payload.get("replay_result")
    if replay is not None:
        replay_result = replay(payload)
    raw_refs = tuple(
        str(item)
        for item in (
            payload.get("raw_diagnostic_refs")
            or payload.get("raw_diagnostics")
            or ()
        )
    )
    if not raw_refs and not replay_result and not payload.get("proof_of_negation_id"):
        raw_refs = (
            _text(
                payload.get("solver_countermodel_id") or "diag:raw-countermodel",
                "raw_diagnostic_refs",
            ),
        )
    validate_kwargs = {
        "roots": roots,
        "solver_countermodel_id": _text(
            payload.get("solver_countermodel_id") or payload.get("countermodel_id") or "cm:unknown",
            "solver_countermodel_id",
        ),
        "translation_map_id": _text(
            payload.get("translation_map_id") or "translation-map:program-world",
            "translation_map_id",
        ),
        "originating_logic_ir_id": _text(
            payload.get("originating_logic_ir_id")
            or payload.get("logic_ir_id")
            or "logic-ir:program-world",
            "originating_logic_ir_id",
        ),
        "raw_diagnostic_refs": raw_refs,
        "replay_result": replay_result if isinstance(replay_result, Mapping) else None,
        "proof_of_negation_id": _text(
            payload.get("proof_of_negation_id") or "",
            "proof_of_negation_id",
            required=False,
        ),
        "assumption_refs": tuple(str(item) for item in (payload.get("assumption_refs") or ())),
        "resource_policy_ref": _text(
            payload.get("resource_policy_ref") or "",
            "resource_policy_ref",
            required=False,
        ),
        "invalidation_refs": (roots.tree_id, roots.toolchain_id, roots.policy_id),
    }
    if validator is not None:
        validation = validator.validate(**validate_kwargs)
    else:
        validation = _validate_countermodel(**validate_kwargs)
    return ProgramWorldCountermodelReplay(
        replay_id=_compact_id(validation.receipt_id, prefix="replay"),
        tree_id=roots.tree_id,
        validation=validation,
        model_authored=model_authored,
    )


def _admit(
    compilation: ProgramWorldGoalCompilation,
    *,
    reconstruction: ProgramWorldReconstructionReceipt | None,
    replay: ProgramWorldCountermodelReplay | None,
    outcome_hint: str,
    model_authored: bool,
    reason_codes: Sequence[str],
) -> ProgramWorldProofAdmission:
    reasons = list(reason_codes)
    tree_id = compilation.roots.tree_id
    if compilation.disposition is ProgramWorldCompilationDisposition.CONFLICT:
        return ProgramWorldProofAdmission(
            admission_id=_compact_id(compilation.compilation_id, prefix="admission"),
            disposition=ProgramWorldAdmissionDisposition.ABSTAINED,
            tree_id=tree_id,
            compilation_id=compilation.compilation_id,
            proof_status=ProofStatus.INCONCLUSIVE,
            reason_codes=_reason_codes(reasons, ("contradiction", "abstain")),
            model_authored=False,
        )
    if reconstruction is not None and reconstruction.reason_code == "stale_tree":
        return ProgramWorldProofAdmission(
            admission_id=_compact_id(compilation.compilation_id, prefix="admission"),
            disposition=ProgramWorldAdmissionDisposition.STALE,
            tree_id=tree_id,
            compilation_id=compilation.compilation_id,
            proof_status=ProofStatus.STALE,
            reason_codes=_reason_codes(reasons, ("stale_tree",)),
            reconstruction_id=reconstruction.reconstruction_id,
            model_authored=model_authored,
        )
    proof_ok = (
        reconstruction is not None
        and reconstruction.kernel_accepted
        and reconstruction.tree_id == tree_id
    )
    refute_ok = (
        replay is not None
        and replay.validated
        and replay.tree_id == tree_id
    )
    if proof_ok and refute_ok:
        return ProgramWorldProofAdmission(
            admission_id=_compact_id(compilation.compilation_id, prefix="admission"),
            disposition=ProgramWorldAdmissionDisposition.ABSTAINED,
            tree_id=tree_id,
            compilation_id=compilation.compilation_id,
            proof_status=ProofStatus.INCONCLUSIVE,
            reason_codes=_reason_codes(
                reasons, ("contradiction", "proof_and_refutation")
            ),
            reconstruction_id=reconstruction.reconstruction_id if reconstruction else "",
            replay_id=replay.replay_id if replay else "",
            model_authored=False,
        )
    if proof_ok and reconstruction is not None:
        return ProgramWorldProofAdmission(
            admission_id=_compact_id(compilation.compilation_id, prefix="admission"),
            disposition=ProgramWorldAdmissionDisposition.ADMITTED_PROOF,
            tree_id=tree_id,
            compilation_id=compilation.compilation_id,
            proof_status=ProofStatus.KERNEL_VERIFIED,
            reason_codes=_reason_codes(reasons, ("kernel_reconstructed",)),
            reconstruction_id=reconstruction.reconstruction_id,
            model_authored=False,
        )
    if refute_ok and replay is not None:
        return ProgramWorldProofAdmission(
            admission_id=_compact_id(compilation.compilation_id, prefix="admission"),
            disposition=ProgramWorldAdmissionDisposition.ADMITTED_REFUTATION,
            tree_id=tree_id,
            compilation_id=compilation.compilation_id,
            proof_status=ProofStatus.VALIDATED_REFUTED,
            reason_codes=_reason_codes(reasons, ("countermodel_replayed",)),
            replay_id=replay.replay_id,
            model_authored=False,
        )
    if model_authored or (reconstruction is not None and reconstruction.model_authored):
        reasons.append("model_authored_not_authority")
    if reconstruction is not None and reconstruction.solver_sat:
        reasons.append("solver_sat_not_proof")
    if outcome_hint in {"timeout", ProgramWorldSearchOutcome.TIMEOUT.value}:
        disposition = ProgramWorldAdmissionDisposition.REJECTED
        status = ProofStatus.INCONCLUSIVE
        reasons.append("timeout")
    elif outcome_hint in {"unsupported", ProgramWorldSearchOutcome.UNSUPPORTED.value}:
        disposition = ProgramWorldAdmissionDisposition.REJECTED
        status = ProofStatus.UNSUPPORTED
        reasons.append("unsupported_logic")
    elif outcome_hint in {"unavailable", ProgramWorldSearchOutcome.UNAVAILABLE.value}:
        disposition = ProgramWorldAdmissionDisposition.UNAVAILABLE
        status = ProofStatus.INCONCLUSIVE
        reasons.append("backend_unavailable")
    else:
        disposition = ProgramWorldAdmissionDisposition.REJECTED
        status = ProofStatus.CANDIDATE
        reasons.append("reconstruction_or_replay_required")
    return ProgramWorldProofAdmission(
        admission_id=_compact_id(compilation.compilation_id, prefix="admission"),
        disposition=disposition,
        tree_id=tree_id,
        compilation_id=compilation.compilation_id,
        proof_status=status,
        reason_codes=_reason_codes(reasons),
        reconstruction_id=(
            reconstruction.reconstruction_id if reconstruction is not None else ""
        ),
        replay_id=replay.replay_id if replay is not None else "",
        model_authored=model_authored,
    )


def _outcome_for(
    admission: ProgramWorldProofAdmission,
    *,
    hint: str,
    reasons: Sequence[str],
) -> ProgramWorldSearchOutcome:
    if admission.disposition is ProgramWorldAdmissionDisposition.ADMITTED_PROOF:
        return ProgramWorldSearchOutcome.VERIFIED
    if admission.disposition is ProgramWorldAdmissionDisposition.ADMITTED_REFUTATION:
        return ProgramWorldSearchOutcome.REFUTED
    if admission.disposition is ProgramWorldAdmissionDisposition.ABSTAINED:
        return ProgramWorldSearchOutcome.ABSTAINED
    if admission.disposition is ProgramWorldAdmissionDisposition.STALE:
        return ProgramWorldSearchOutcome.STALE
    if admission.disposition is ProgramWorldAdmissionDisposition.UNAVAILABLE:
        return ProgramWorldSearchOutcome.UNAVAILABLE
    if hint == "timeout" or "timeout" in reasons:
        return ProgramWorldSearchOutcome.TIMEOUT
    if hint == "unsupported" or "unsupported_logic" in reasons:
        return ProgramWorldSearchOutcome.UNSUPPORTED
    if hint == "conflict" or "contradiction" in reasons:
        return ProgramWorldSearchOutcome.CONFLICT
    if hint == "error":
        return ProgramWorldSearchOutcome.ERROR
    return ProgramWorldSearchOutcome.CANDIDATE


class ProgramWorldHammer:
    """Production Hammer search with independent reconstruction and replay."""

    interface: ClassVar[str] = PROGRAM_WORLD_HAMMER_INTERFACE
    evidence: ClassVar[str] = SAWM_TACTICIAN_HAMMER_EVIDENCE
    version: ClassVar[str] = HAMMER_VERSION

    def __init__(
        self,
        *,
        search: HammerSearchFn | None = None,
        reconstructor: ReconstructionFn | None = None,
        replay: ReplayFn | None = None,
        validator: CountermodelValidateFn | None = None,
        which: Callable[[str], str | None] | None = None,
        find_spec: Callable[[str], Any | None] | None = None,
        path: str | None = None,
        probe_solvers: bool = True,
    ) -> None:
        self._search = search
        self._reconstructor = reconstructor
        self._replay = replay
        self._validator = validator
        self._which = which
        self._find_spec = find_spec
        self._path = path
        self._probe_solvers = probe_solvers

    def probe_backends(self) -> ProgramWorldBackendProbe:
        return probe_program_world_backends(
            path=self._path,
            which=self._which,
            find_spec=self._find_spec,
            probe_solvers=self._probe_solvers,
        )

    def search(
        self,
        compilation: ProgramWorldGoalCompilation,
        *,
        expected_tree_id: str | None = None,
        candidate: Mapping[str, Any] | None = None,
    ) -> ProgramWorldProofSearchReceipt:
        return search_program_world_proof(
            compilation,
            expected_tree_id=expected_tree_id,
            candidate=candidate,
            search=self._search,
            reconstructor=self._reconstructor,
            replay=self._replay,
            validator=self._validator,
            which=self._which,
            find_spec=self._find_spec,
            path=self._path,
            probe_solvers=self._probe_solvers,
        )

    def replay_countermodel(
        self,
        compilation: ProgramWorldGoalCompilation,
        countermodel: Mapping[str, Any],
        *,
        expected_tree_id: str | None = None,
    ) -> ProgramWorldCountermodelReplay:
        return replay_program_world_countermodel(
            compilation,
            countermodel,
            replay=self._replay,
            expected_tree_id=expected_tree_id,
            validator=self._validator,
        )


def search_program_world_proof(
    compilation: ProgramWorldGoalCompilation | Mapping[str, Any],
    *,
    expected_tree_id: str | None = None,
    candidate: Mapping[str, Any] | None = None,
    search: HammerSearchFn | None = None,
    reconstructor: ReconstructionFn | None = None,
    replay: ReplayFn | None = None,
    validator: CountermodelValidateFn | None = None,
    which: Callable[[str], str | None] | None = None,
    find_spec: Callable[[str], Any | None] | None = None,
    path: str | None = None,
    probe_solvers: bool = True,
) -> ProgramWorldProofSearchReceipt:
    """Search with production Hammer; admit only reconstructed/replayed evidence."""

    compiled = _decode_compilation(compilation)
    backend = probe_program_world_backends(
        path=path,
        which=which,
        find_spec=find_spec,
        probe_solvers=probe_solvers,
    )
    reasons: list[str] = list(compiled.reason_codes)
    tree_id = compiled.roots.tree_id
    if _stale_tree(compiled.roots, expected_tree_id):
        admission = ProgramWorldProofAdmission(
            admission_id=_compact_id(compiled.compilation_id, prefix="admission"),
            disposition=ProgramWorldAdmissionDisposition.STALE,
            tree_id=tree_id,
            compilation_id=compiled.compilation_id,
            proof_status=ProofStatus.STALE,
            reason_codes=("stale_tree",),
        )
        return ProgramWorldProofSearchReceipt(
            receipt_id=_compact_id(compiled.compilation_id, prefix="search"),
            outcome=ProgramWorldSearchOutcome.STALE,
            compilation_id=compiled.compilation_id,
            tree_id=tree_id,
            premise_corpus_cid=compiled.premise_corpus_cid,
            tactic_plan_id=compiled.tactic_plan.plan_id,
            admission=admission,
            backend=backend,
            reason_codes=("stale_tree",),
        )

    if compiled.disposition in {
        ProgramWorldCompilationDisposition.CONFLICT,
        ProgramWorldCompilationDisposition.ABSTAINED,
    }:
        admission = _admit(
            compiled,
            reconstruction=None,
            replay=None,
            outcome_hint=(
                "conflict"
                if compiled.disposition is ProgramWorldCompilationDisposition.CONFLICT
                else "abstained"
            ),
            model_authored=False,
            reason_codes=reasons,
        )
        if compiled.disposition is ProgramWorldCompilationDisposition.ABSTAINED and not admission.admitted:
            admission = ProgramWorldProofAdmission(
                admission_id=_compact_id(compiled.compilation_id, prefix="admission"),
                disposition=ProgramWorldAdmissionDisposition.ABSTAINED,
                tree_id=tree_id,
                compilation_id=compiled.compilation_id,
                proof_status=ProofStatus.INCONCLUSIVE,
                reason_codes=_reason_codes(reasons, ("empty_or_abstained_inventory",)),
            )
        return ProgramWorldProofSearchReceipt(
            receipt_id=_compact_id(compiled.compilation_id, prefix="search"),
            outcome=ProgramWorldSearchOutcome.ABSTAINED,
            compilation_id=compiled.compilation_id,
            tree_id=tree_id,
            premise_corpus_cid=compiled.premise_corpus_cid,
            tactic_plan_id=compiled.tactic_plan.plan_id,
            admission=admission,
            backend=backend,
            reason_codes=admission.reason_codes,
        )

    payload: dict[str, Any] = dict(candidate or {})
    if search is not None:
        payload.update(_mapping(search(compiled, MappingProxyType(payload)), "search"))
    elif not payload:
        if compiled.disposition is ProgramWorldCompilationDisposition.ABSTAINED:
            payload = {"outcome": "abstained"}
        elif compiled.unsupported_goal_ids and not [
            goal
            for goal in compiled.goals
            if goal.goal_id not in compiled.unsupported_goal_ids
            and goal.goal_id not in compiled.residual_goal_ids
        ]:
            payload = {"outcome": "unsupported"}
        elif not backend.reconstruction_ready and reconstructor is None:
            payload = {"outcome": "unavailable", "reason": "kernel_unavailable"}
        else:
            payload = {"outcome": "candidate"}

    hint = str(payload.get("outcome") or payload.get("status") or "candidate").lower()
    model_authored = bool(payload.get("model_authored") or payload.get("from_model"))
    reconstruction: ProgramWorldReconstructionReceipt | None = None
    replay_receipt: ProgramWorldCountermodelReplay | None = None

    if hint in {"timeout", "timed_out"}:
        hint = "timeout"
    elif hint in {"unsupported", "unsupported_translation", "unsupported_logic"}:
        hint = "unsupported"
    elif hint in {"unavailable", "policy_denied"}:
        hint = "unavailable"

    needs_recon = hint in {"verified", "candidate", "accepted", "proved"} or bool(
        payload.get("proof_candidate") or payload.get("proof")
    )
    needs_replay = hint in {"counterexample", "refuted", "countermodel"} or bool(
        payload.get("countermodel") or payload.get("solver_countermodel_id")
    )

    if needs_recon and hint not in {"timeout", "unsupported", "unavailable"}:
        reconstruction = reconstruct_program_world_proof(
            compiled,
            payload.get("proof_candidate") or payload,
            reconstructor=reconstructor,
            expected_tree_id=expected_tree_id,
            backend=backend,
        )
        if reconstruction.reason_code:
            reasons.append(reconstruction.reason_code)
    if needs_replay and hint not in {"timeout", "unsupported", "unavailable"}:
        try:
            replay_receipt = replay_program_world_countermodel(
                compiled,
                payload.get("countermodel") or payload,
                replay=replay,
                expected_tree_id=expected_tree_id,
                validator=validator,
            )
            if replay_receipt.validation.disposition.value:
                reasons.append(replay_receipt.validation.disposition.value)
        except ProgramWorldHammerAuthorityError as exc:
            detail = str(exc).strip().splitlines()[0][:120]
            reasons.append(detail.replace(" ", "_") if detail else "replay_rejected")

    admission = _admit(
        compiled,
        reconstruction=reconstruction,
        replay=replay_receipt,
        outcome_hint=hint,
        model_authored=model_authored,
        reason_codes=reasons,
    )
    outcome = _outcome_for(admission, hint=hint, reasons=admission.reason_codes)
    if hint == "timeout":
        outcome = ProgramWorldSearchOutcome.TIMEOUT
    elif hint == "unsupported" and not admission.admitted:
        outcome = ProgramWorldSearchOutcome.UNSUPPORTED
    elif hint == "unavailable" and not admission.admitted:
        outcome = ProgramWorldSearchOutcome.UNAVAILABLE
    return ProgramWorldProofSearchReceipt(
        receipt_id=_compact_id(
            f"{compiled.compilation_id}:{outcome.value}:{admission.admission_id}",
            prefix="search",
        ),
        outcome=outcome,
        compilation_id=compiled.compilation_id,
        tree_id=tree_id,
        premise_corpus_cid=compiled.premise_corpus_cid,
        tactic_plan_id=compiled.tactic_plan.plan_id,
        admission=admission,
        backend=backend,
        reconstruction=reconstruction,
        replay=replay_receipt,
        reason_codes=admission.reason_codes,
    )


def map_hammer_outcome(status: Any) -> ProgramWorldSearchOutcome:
    """Map landed Hammer coordination outcomes onto program-world vocabulary."""

    raw = str(getattr(status, "value", status) or "").strip().lower()
    mapping = {
        "verified": ProgramWorldSearchOutcome.VERIFIED,
        "candidate": ProgramWorldSearchOutcome.CANDIDATE,
        "counterexample": ProgramWorldSearchOutcome.CANDIDATE,
        "timeout": ProgramWorldSearchOutcome.TIMEOUT,
        "unsupported": ProgramWorldSearchOutcome.UNSUPPORTED,
        "unavailable": ProgramWorldSearchOutcome.UNAVAILABLE,
        "policy_denied": ProgramWorldSearchOutcome.UNAVAILABLE,
        "unknown": ProgramWorldSearchOutcome.CANDIDATE,
        "stale": ProgramWorldSearchOutcome.STALE,
        "error": ProgramWorldSearchOutcome.ERROR,
    }
    return mapping.get(raw, ProgramWorldSearchOutcome.CANDIDATE)


__all__ = [
    "HAMMER_MODULES",
    "ITP_EXECUTABLES",
    "PROGRAM_WORLD_HAMMER_INTERFACE",
    "PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE",
    "SAWM_TACTICIAN_HAMMER_EVIDENCE",
    "VALIDATION_PATH",
    "ProgramWorldAdmissionDisposition",
    "ProgramWorldBackendProbe",
    "ProgramWorldBackendStatus",
    "ProgramWorldCountermodelReplay",
    "ProgramWorldHammer",
    "ProgramWorldHammerAuthorityError",
    "ProgramWorldHammerError",
    "ProgramWorldProofAdmission",
    "ProgramWorldProofSearchReceipt",
    "ProgramWorldReconstructionReceipt",
    "ProgramWorldSearchOutcome",
    "map_hammer_outcome",
    "probe_program_world_backends",
    "reconstruct_program_world_proof",
    "replay_program_world_countermodel",
    "search_program_world_proof",
]
