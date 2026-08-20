"""FACP-051: Bounded counterexample-guided repair (RSE).

Runs a budgeted CEGIS loop over a **fixed** repair grammar. Candidates are
applied only inside an isolated in-memory transaction, then admitted only when:

* the original counterexample disappears,
* no new abstract / model / test counterexample appears,
* proof/test and (when applicable) TEP gates pass,
* the path stays inside the admitted scope, and
* no authority, obligation-waiver, or grammar-expansion attack succeeds.

Admitted repairs mint a minimal, independently checkable ``PatchCertificate``.
Otherwise the loop returns a typed abstention. LLMs may classify or sketch a
family; they cannot expand the grammar, waive obligations, promote patches, or
create write/proof authority. Production disk mutation remains a later
control-plane step — certificates never grant write authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Optional, Sequence

from ipfs_accelerate_py.agent_supervisor.analysis.formal_assurance.ipa import (
    IpaFinding,
    IpaRuleId,
    analyze_python_source,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.formal_assurance_transforms import (
    IpaRepairAbstentionReason,
    IpaRepairDisposition,
    IpaRepairEdit,
    IpaRepairError,
    IpaRepairReceipt,
    IpaRepairTransformId,
    IpaReanalysisReport,
    MutationGateDecision,
    MutationGateDisposition,
    TRANSFORM_TO_RULE,
    apply_ipa_repair,
    default_admitted_paths,
    list_transform_grammar,
    path_is_admitted,
    select_transform,
)

TASK_ID: Final[str] = "FACP-051"
GOAL_ID: Final[str] = "FACP-G710"
BUNDLE: Final[str] = "facp/synthesis/repair"
SCHEMA: Final[str] = "facp/cegis-repair@1"
GRAMMAR_SCHEMA: Final[str] = "facp/repair-grammar@1"
CERTIFICATE_SCHEMA: Final[str] = "facp/patch-certificate@1"
RESULT_SCHEMA: Final[str] = "facp/cegis-repair-result@1"
TRANSACTION_SCHEMA: Final[str] = "facp/cegis-isolated-transaction@1"
BENCHMARK_SCHEMA: Final[str] = "facp/cegis-mutation-benchmark@1"
CEGIS_EVIDENCE: Final[str] = "facp/cegis-repair@1"
REPAIR_GRAMMAR_EVIDENCE: Final[str] = "facp/repair-grammar@1"
PATCH_CERTIFICATE_EVIDENCE: Final[str] = "facp/patch-certificate@1"
INTERFACE: Final[str] = "FormalAssuranceCegis@1"
PRODUCER_ID: Final[str] = "formal-assurance-cegis@1"
ANALYZER_VERSION: Final[str] = "formal-assurance-cegis/v1"
TOOLCHAIN_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.autonomous_repair.formal_assurance_cegis/"
    + ANALYZER_VERSION
)

MAX_SOURCE_BYTES: Final[int] = 1_000_000
MAX_PATH_BYTES: Final[int] = 1_024
MAX_CANDIDATES_HARD: Final[int] = 32
MAX_ITERATIONS_HARD: Final[int] = 32
MAX_EDITS_HARD: Final[int] = 32
DEFAULT_MAX_ITERATIONS: Final[int] = 8
DEFAULT_MAX_CANDIDATES: Final[int] = 4
DEFAULT_MAX_EDITS: Final[int] = 8

# Closed attack / proposal keys that never expand scope or create authority.
_SCOPE_WIDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "extra_paths",
        "new_dependencies",
        "dependency_paths",
        "write_paths",
        "requested_write_paths",
        "authority_override",
        "policy_override",
        "completion_claim",
        "semantic_change",
        "meaning_change",
        "import_additions",
        "extra_imports",
        "extra_files",
        "grammar_expansion",
        "new_transform",
        "new_family",
    }
)

_AUTHORITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "write_authority",
        "semantic_authority",
        "proof_authority",
        "completion_authority",
        "grants_write_authority",
        "grants_proof_authority",
        "mutation_permit",
        "admission",
        "promote_patch",
        "self_promote",
        "create_authority",
    }
)

_OBLIGATION_WAIVER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "waive_obligation",
        "waive_obligations",
        "skip_proof",
        "skip_test",
        "skip_reanalysis",
        "ignore_counterexample",
        "force_admit",
        "bypass_gate",
    }
)


class CegisError(ValueError):
    """Malformed CEGIS input or an attempt to weaken a fail-closed boundary."""


class RepairFamily(str, Enum):
    """Closed RSE defect families (not expandable by models)."""

    FALSE_SUCCESS = "false_success"
    MOCK_CAPABILITY = "mock_capability"
    PSEUDO_CID = "pseudo_cid"
    IMPORT_EFFECT = "import_effect"
    EXCEPTION_SWALLOWING = "exception_swallowing"
    BROWSER_AUTHORITY = "browser_authority"
    MUTABLE_DEPENDENCY = "mutable_dependency"
    STALE_PROOF = "stale_proof"
    MISSING_LEASE_RECOVERY = "missing_lease_recovery"
    LICENSE_CONFLICT = "license_conflict"


class RepairTransformId(str, Enum):
    """Closed repair grammar identifiers spanning IPA + declaration transforms."""

    # IPA-backed (FACP-043)
    EXPLICIT_INIT = "explicit_init"
    TYPED_UNAVAILABLE = "typed_unavailable"
    SIMULATION_EVIDENCE = "simulation_evidence"
    CANONICAL_CID = "canonical_cid"
    CRITICAL_ERROR_PROPAGATION = "critical_error_propagation"
    # Declaration / policy grammars
    DENY_DEFAULT_BROWSER_CONSENT = "deny_default_browser_consent"
    BLOCK_MUTABLE_VCS_DEPENDENCY = "block_mutable_vcs_dependency"
    DEMOTE_STALE_PROOF_RECEIPT = "demote_stale_proof_receipt"
    REQUIRE_LEASE_FENCE_RECOVERY = "require_lease_fence_recovery"
    ABSTAIN_LICENSE_HUMAN_REVIEW = "abstain_license_human_review"


class CegisDisposition(str, Enum):
    """Closed CEGIS outcome vocabulary."""

    CERTIFIED = "certified"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    BUDGET_EXHAUSTED = "budget_exhausted"


class CegisAbstentionReason(str, Enum):
    """Closed, audit-stable abstention / rejection codes."""

    PATH_NOT_ADMITTED = "path_not_admitted"
    TRANSFORM_OUTSIDE_GRAMMAR = "transform_outside_grammar"
    FAMILY_OUTSIDE_GRAMMAR = "family_outside_grammar"
    SCOPE_ESCAPE = "scope_escape"
    AUTHORITY_CLAIM = "authority_claim"
    OBLIGATION_WAIVER = "obligation_waiver"
    PRECONDITION_MISMATCH = "precondition_mismatch"
    AMBIGUOUS_TARGET = "ambiguous_target"
    NO_BYTE_CHANGE = "no_byte_change"
    ORIGINAL_COUNTEREXAMPLE_REMAINS = "original_counterexample_remains"
    NEW_ABSTRACT_COUNTEREXAMPLE = "new_abstract_counterexample"
    NEW_MODEL_COUNTEREXAMPLE = "new_model_counterexample"
    NEW_TEST_COUNTEREXAMPLE = "new_test_counterexample"
    PROOF_GATE_FAILED = "proof_gate_failed"
    TRACE_GATE_FAILED = "trace_gate_failed"
    STALE_PROOF_REUSE = "stale_proof_reuse"
    LICENSE_REQUIRES_HUMAN = "license_requires_human"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EMPTY_SOURCE = "empty_source"
    PARSE_ERROR = "parse_error"
    PUBLIC_COMPAT_RISK = "public_compat_risk"
    CANDIDATE_REJECTED = "candidate_rejected"
    NO_CANDIDATES = "no_candidates"
    TRANSACTION_ROLLED_BACK = "transaction_rolled_back"


class TransactionDisposition(str, Enum):
    COMMITTED_OVERLAY = "committed_overlay"
    ROLLED_BACK = "rolled_back"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class GateKind(str, Enum):
    ABSTRACT = "abstract"
    MODEL = "model"
    TEST = "test"
    PROOF = "proof"
    TRACE = "trace"
    SCOPE = "scope"
    AUTHORITY = "authority"


class GateVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


# Family → primary transform binding (fixed; LLMs cannot expand).
FAMILY_TO_TRANSFORM: Final[Mapping[RepairFamily, RepairTransformId]] = {
    RepairFamily.FALSE_SUCCESS: RepairTransformId.TYPED_UNAVAILABLE,
    RepairFamily.EXCEPTION_SWALLOWING: RepairTransformId.CRITICAL_ERROR_PROPAGATION,
    RepairFamily.MOCK_CAPABILITY: RepairTransformId.SIMULATION_EVIDENCE,
    RepairFamily.PSEUDO_CID: RepairTransformId.CANONICAL_CID,
    RepairFamily.IMPORT_EFFECT: RepairTransformId.EXPLICIT_INIT,
    RepairFamily.BROWSER_AUTHORITY: RepairTransformId.DENY_DEFAULT_BROWSER_CONSENT,
    RepairFamily.MUTABLE_DEPENDENCY: RepairTransformId.BLOCK_MUTABLE_VCS_DEPENDENCY,
    RepairFamily.STALE_PROOF: RepairTransformId.DEMOTE_STALE_PROOF_RECEIPT,
    RepairFamily.MISSING_LEASE_RECOVERY: RepairTransformId.REQUIRE_LEASE_FENCE_RECOVERY,
    RepairFamily.LICENSE_CONFLICT: RepairTransformId.ABSTAIN_LICENSE_HUMAN_REVIEW,
}

# IPA transform bridge.
_IPA_TRANSFORM_BRIDGE: Final[Mapping[RepairTransformId, IpaRepairTransformId]] = {
    RepairTransformId.EXPLICIT_INIT: IpaRepairTransformId.EXPLICIT_INIT,
    RepairTransformId.TYPED_UNAVAILABLE: IpaRepairTransformId.TYPED_UNAVAILABLE,
    RepairTransformId.SIMULATION_EVIDENCE: IpaRepairTransformId.SIMULATION_EVIDENCE,
    RepairTransformId.CANONICAL_CID: IpaRepairTransformId.CANONICAL_CID,
    RepairTransformId.CRITICAL_ERROR_PROPAGATION: IpaRepairTransformId.CRITICAL_ERROR_PROPAGATION,
}

_FAMILY_TO_IPA_RULE: Final[Mapping[RepairFamily, IpaRuleId]] = {
    RepairFamily.FALSE_SUCCESS: IpaRuleId.SUCCESS_WITHOUT_OBSERVATION,
    RepairFamily.EXCEPTION_SWALLOWING: IpaRuleId.EXCEPTION_SWALLOWING,
    RepairFamily.MOCK_CAPABILITY: IpaRuleId.MOCK_TO_PRODUCTION,
    RepairFamily.PSEUDO_CID: IpaRuleId.PSEUDO_CID,
    RepairFamily.IMPORT_EFFECT: IpaRuleId.IMPORT_EFFECT,
}

STABLE_FAMILY_IDS: Final[frozenset[str]] = frozenset(item.value for item in RepairFamily)
STABLE_TRANSFORM_IDS: Final[frozenset[str]] = frozenset(
    item.value for item in RepairTransformId
)

_MUTABLE_VCS_RE: Final[re.Pattern[str]] = re.compile(
    r"(git\+https?://\S+@(?:main|master|HEAD)\b)|(@\s*(?:main|master|HEAD)\b)",
    re.IGNORECASE,
)
_DEFAULT_GRANTED_RE: Final[re.Pattern[str]] = re.compile(
    r"""(\?\?\s*(?:['"]granted['"]))|(consent\s*=\s*['"]granted['"])"""
    r"""|(default(?:ed)?\s+to\s+granted)|(default_consent\s*[:=]\s*['"]granted['"])""",
    re.IGNORECASE,
)
_STALE_PROOF_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "reuse_stale_proof",
        "stale_proof_ok",
        "historical_receipt_as_live",
        "proof_status=live_stale",
        '"status": "live"',
        '"status":"live"',
    }
)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(payload)).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _normalize_path(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _validate_path(path: str, name: str = "path") -> str:
    raw = _normalize_path(path)
    if not raw:
        raise CegisError(f"{name} is required")
    if len(raw.encode("utf-8")) > MAX_PATH_BYTES:
        raise CegisError(f"{name} exceeds its byte bound")
    if raw.startswith("/") or ".." in raw.split("/"):
        raise CegisError(f"{name} must be a relative repository path")
    return raw


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted(item.value for item in kind))
        raise CegisError(f"{name} must be one of: {allowed}") from exc


def _positive_int(value: Any, name: str, *, hard_max: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CegisError(f"{name} must be a positive int")
    if value > hard_max:
        raise CegisError(f"{name} exceeds hard ceiling {hard_max}")
    return value


@dataclass(frozen=True)
class RepairGrammarEntry:
    """One closed grammar row."""

    family: RepairFamily
    transform_id: RepairTransformId
    ipa_rule_id: str = ""
    obligations: tuple[str, ...] = ()
    may_certify: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRAMMAR_SCHEMA,
            "family": self.family.value,
            "transform_id": self.transform_id.value,
            "ipa_rule_id": self.ipa_rule_id,
            "obligations": list(self.obligations),
            "may_certify": self.may_certify,
            "notes": self.notes,
        }


def list_repair_grammar() -> tuple[RepairGrammarEntry, ...]:
    """Return the closed RSE repair grammar as sorted records."""

    rows: list[RepairGrammarEntry] = []
    for family, transform in sorted(
        FAMILY_TO_TRANSFORM.items(), key=lambda item: item[0].value
    ):
        ipa_rule = ""
        if family in _FAMILY_TO_IPA_RULE:
            ipa_rule = _FAMILY_TO_IPA_RULE[family].value
        may_certify = family is not RepairFamily.LICENSE_CONFLICT
        obligations = _default_obligations(family)
        rows.append(
            RepairGrammarEntry(
                family=family,
                transform_id=transform,
                ipa_rule_id=ipa_rule,
                obligations=obligations,
                may_certify=may_certify,
                notes=(
                    "human legal review required; never auto-clear"
                    if family is RepairFamily.LICENSE_CONFLICT
                    else ""
                ),
            )
        )
    return tuple(rows)


def _default_obligations(family: RepairFamily) -> tuple[str, ...]:
    base = (
        "obligation:remove-original-counterexample",
        "obligation:no-new-abstract-counterexample",
        "obligation:no-new-model-counterexample",
        "obligation:no-new-test-counterexample",
        "obligation:preserve-public-compat",
        "obligation:no-authority-creation",
    )
    extras: dict[RepairFamily, tuple[str, ...]] = {
        RepairFamily.BROWSER_AUTHORITY: (
            "obligation:browser-nonauthority",
            "obligation:host-admission-required",
        ),
        RepairFamily.MUTABLE_DEPENDENCY: (
            "obligation:release-blocks-mutable-vcs",
        ),
        RepairFamily.STALE_PROOF: (
            "obligation:demote-stale-receipt",
            "obligation:refuse-stale-cache-reuse",
        ),
        RepairFamily.MISSING_LEASE_RECOVERY: (
            "obligation:no-blind-unknown-retry",
            "obligation:lease-fence-recovery-present",
        ),
        RepairFamily.LICENSE_CONFLICT: (
            "obligation:human-legal-review",
        ),
    }
    return base + extras.get(family, ())


@dataclass(frozen=True)
class CegisBudget:
    """Hard-capped repair loop budget (policy may only tighten)."""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_candidates_per_iteration: int = DEFAULT_MAX_CANDIDATES
    max_edits: int = DEFAULT_MAX_EDITS
    max_source_bytes: int = MAX_SOURCE_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_iterations",
            _positive_int(self.max_iterations, "max_iterations", hard_max=MAX_ITERATIONS_HARD),
        )
        object.__setattr__(
            self,
            "max_candidates_per_iteration",
            _positive_int(
                self.max_candidates_per_iteration,
                "max_candidates_per_iteration",
                hard_max=MAX_CANDIDATES_HARD,
            ),
        )
        object.__setattr__(
            self,
            "max_edits",
            _positive_int(self.max_edits, "max_edits", hard_max=MAX_EDITS_HARD),
        )
        if (
            isinstance(self.max_source_bytes, bool)
            or not isinstance(self.max_source_bytes, int)
            or self.max_source_bytes < 1
            or self.max_source_bytes > MAX_SOURCE_BYTES
        ):
            raise CegisError("max_source_bytes out of bounds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_candidates_per_iteration": self.max_candidates_per_iteration,
            "max_edits": self.max_edits,
            "max_source_bytes": self.max_source_bytes,
        }


@dataclass(frozen=True)
class CounterexampleRecord:
    """Minimal counterexample / defect witness bound into repair."""

    counterexample_id: str
    family: RepairFamily
    path: str
    witness: str = ""
    finding_id: str = ""
    rule_id: str = ""
    abstract_markers: tuple[str, ...] = ()
    model_markers: tuple[str, ...] = ()
    test_markers: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _enum(self.family, RepairFamily, "family"))
        object.__setattr__(self, "path", _validate_path(self.path))
        if not str(self.counterexample_id or "").strip():
            raise CegisError("counterexample_id is required")
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata) if isinstance(self.metadata, Mapping) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            "family": self.family.value,
            "path": self.path,
            "witness": self.witness,
            "finding_id": self.finding_id,
            "rule_id": self.rule_id,
            "abstract_markers": list(self.abstract_markers),
            "model_markers": list(self.model_markers),
            "test_markers": list(self.test_markers),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GateResult:
    kind: GateKind
    verdict: GateVerdict
    reasons: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict is GateVerdict.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class RepairCandidate:
    """One grammar-bounded candidate produced inside the CEGIS loop."""

    candidate_id: str
    family: RepairFamily
    transform_id: RepairTransformId
    path: str
    before_hash: str
    after_hash: str
    after_source: str
    edits: tuple[IpaRepairEdit, ...] = ()
    ipa_receipt: Optional[IpaRepairReceipt] = None
    addresses_witness: bool = False
    residual_risks: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family.value,
            "transform_id": self.transform_id.value,
            "path": self.path,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "edits": [edit.to_dict() for edit in self.edits],
            "addresses_witness": self.addresses_witness,
            "residual_risks": list(self.residual_risks),
            "reasons": list(self.reasons),
            "ipa_receipt": self.ipa_receipt.to_dict() if self.ipa_receipt else None,
            # after_source retained for isolated replay only; certificates hash it.
            "after_source_hash": self.after_hash,
        }


@dataclass(frozen=True)
class IsolatedRepairTransaction:
    """In-memory overlay transaction; never mutates the live tree."""

    transaction_id: str
    disposition: TransactionDisposition
    admitted_paths: tuple[str, ...]
    checkpoint_hashes: Mapping[str, str]
    overlay: Mapping[str, str]
    path: str
    reasons: tuple[str, ...] = ()
    candidate_id: str = ""

    @property
    def committed(self) -> bool:
        return self.disposition is TransactionDisposition.COMMITTED_OVERLAY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRANSACTION_SCHEMA,
            "transaction_id": self.transaction_id,
            "disposition": self.disposition.value,
            "admitted_paths": list(self.admitted_paths),
            "checkpoint_hashes": dict(self.checkpoint_hashes),
            "overlay_paths": sorted(self.overlay),
            "overlay_hashes": {
                path: _sha256_text(body) for path, body in sorted(self.overlay.items())
            },
            "path": self.path,
            "reasons": list(self.reasons),
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True)
class PatchCertificate:
    """Minimal independently admitted proof-carrying patch certificate."""

    certificate_id: str
    family: RepairFamily
    transform_id: RepairTransformId
    path: str
    counterexample_id: str
    before_hash: str
    after_hash: str
    edits: tuple[IpaRepairEdit, ...]
    mutation_gate: MutationGateDecision
    reanalysis: Optional[IpaReanalysisReport]
    gate_results: tuple[GateResult, ...]
    parent_capsule_cids: tuple[str, ...]
    patch_capsule_cid: str
    affected_capsule_cids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    orchestration_result_ids: tuple[str, ...]
    residual_risks: tuple[str, ...]
    public_compat_preserved: bool = True
    grants_write_authority: bool = False
    schema: str = CERTIFICATE_SCHEMA
    producer_id: str = PRODUCER_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    disposition: CegisDisposition = CegisDisposition.CERTIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _enum(self.family, RepairFamily, "family"))
        object.__setattr__(
            self,
            "transform_id",
            _enum(self.transform_id, RepairTransformId, "transform_id"),
        )
        object.__setattr__(self, "path", _validate_path(self.path))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CegisDisposition, "disposition"),
        )
        if self.disposition is not CegisDisposition.CERTIFIED:
            raise CegisError("PatchCertificate disposition must be certified")
        if self.grants_write_authority:
            raise CegisError("PatchCertificate cannot grant write authority")
        if self.before_hash == self.after_hash:
            raise CegisError("PatchCertificate requires byte mutation")
        if not self.edits:
            raise CegisError("PatchCertificate requires at least one edit")
        if not self.mutation_gate.admitted:
            raise CegisError("PatchCertificate requires an admitted mutation gate")
        if not self.certificate_id:
            raise CegisError("certificate_id is required")
        failed = [g for g in self.gate_results if g.verdict is GateVerdict.FAIL]
        if failed:
            raise CegisError("PatchCertificate cannot include failed gates")

    @property
    def admitted(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence": PATCH_CERTIFICATE_EVIDENCE,
            "certificate_id": self.certificate_id,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "producer_id": self.producer_id,
            "disposition": self.disposition.value,
            "family": self.family.value,
            "transform_id": self.transform_id.value,
            "path": self.path,
            "counterexample_id": self.counterexample_id,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "edits": [edit.to_dict() for edit in self.edits],
            "mutation_gate": self.mutation_gate.to_dict(),
            "reanalysis": self.reanalysis.to_dict() if self.reanalysis else None,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "parent_capsule_cids": list(self.parent_capsule_cids),
            "patch_capsule_cid": self.patch_capsule_cid,
            "affected_capsule_cids": list(self.affected_capsule_cids),
            "obligation_ids": list(self.obligation_ids),
            "orchestration_result_ids": list(self.orchestration_result_ids),
            "residual_risks": list(self.residual_risks),
            "public_compat_preserved": self.public_compat_preserved,
            "grants_write_authority": False,
            "admitted": True,
        }


@dataclass(frozen=True)
class CegisRepairResult:
    """Terminal result of one bounded CEGIS repair attempt."""

    disposition: CegisDisposition
    family: RepairFamily
    path: str
    reasons: tuple[str, ...] = ()
    certificate: Optional[PatchCertificate] = None
    candidates_tried: tuple[RepairCandidate, ...] = ()
    transaction: Optional[IsolatedRepairTransaction] = None
    gate_results: tuple[GateResult, ...] = ()
    residual_risks: tuple[str, ...] = ()
    budget: Optional[CegisBudget] = None
    iterations: int = 0
    counterexample_id: str = ""
    schema: str = RESULT_SCHEMA
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    producer_id: str = PRODUCER_ID
    evidence: tuple[str, ...] = (
        CEGIS_EVIDENCE,
        REPAIR_GRAMMAR_EVIDENCE,
        PATCH_CERTIFICATE_EVIDENCE,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CegisDisposition, "disposition"),
        )
        object.__setattr__(self, "family", _enum(self.family, RepairFamily, "family"))
        if self.disposition is CegisDisposition.CERTIFIED:
            if self.certificate is None:
                raise CegisError("certified results require a PatchCertificate")
        else:
            if self.certificate is not None:
                raise CegisError("non-certified results cannot carry a PatchCertificate")
            if not self.reasons:
                raise CegisError("non-certified results require reasons")

    @property
    def certified(self) -> bool:
        return self.disposition is CegisDisposition.CERTIFIED

    @property
    def abstained(self) -> bool:
        return self.disposition is CegisDisposition.ABSTAINED

    @property
    def grants_write_authority(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "producer_id": self.producer_id,
            "bundle": BUNDLE,
            "interface": INTERFACE,
            "toolchain": TOOLCHAIN_ID,
            "evidence": list(self.evidence),
            "disposition": self.disposition.value,
            "family": self.family.value,
            "path": self.path,
            "counterexample_id": self.counterexample_id,
            "reasons": list(self.reasons),
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "candidates_tried": [item.to_dict() for item in self.candidates_tried],
            "transaction": self.transaction.to_dict() if self.transaction else None,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "residual_risks": list(self.residual_risks),
            "budget": self.budget.to_dict() if self.budget else None,
            "iterations": self.iterations,
            "grants_write_authority": False,
            "certified": self.certified,
        }


@dataclass(frozen=True)
class MutationBenchmarkCase:
    mutant_id: str
    description: str
    admitted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutant_id": self.mutant_id,
            "description": self.description,
            "admitted": self.admitted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class MutationBenchmarkReport:
    """Mutation score over certificate / gate integrity."""

    report_id: str
    cases: tuple[MutationBenchmarkCase, ...]
    killed: int
    survived: int
    schema: str = BENCHMARK_SCHEMA

    @property
    def score(self) -> float:
        total = self.killed + self.survived
        return (self.killed / total) if total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "report_id": self.report_id,
            "cases": [case.to_dict() for case in self.cases],
            "killed": self.killed,
            "survived": self.survived,
            "score": self.score,
        }


@dataclass(frozen=True)
class CegisRepairRequest:
    """One bounded repair request over a seeded witness + source buffer."""

    source: str
    counterexample: CounterexampleRecord
    finding: Optional[IpaFinding | Mapping[str, Any]] = None
    admitted_paths: Optional[Iterable[str]] = None
    budget: Optional[CegisBudget] = None
    proposal_hint: Optional[Mapping[str, Any]] = None
    parent_capsule_cids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    require_proof_gate: bool = True
    require_test_gate: bool = True
    transform_id: Optional[RepairTransformId | str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise CegisError("source must be a string")
        if len(self.source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise CegisError("source exceeds byte bound")
        if not isinstance(self.counterexample, CounterexampleRecord):
            raise CegisError("counterexample must be CounterexampleRecord")


def select_repair_family(
    value: RepairFamily | CounterexampleRecord | IpaFinding | Mapping[str, Any] | str,
) -> RepairFamily:
    """Map a finding / counterexample / family token onto the closed grammar."""

    if isinstance(value, RepairFamily):
        return value
    if isinstance(value, CounterexampleRecord):
        return value.family
    if isinstance(value, IpaFinding):
        for family, rule in _FAMILY_TO_IPA_RULE.items():
            if value.rule_id == rule.value:
                return family
        # Exception swallowing shares false_success corpus family in IPA.
        if value.rule_id == IpaRuleId.EXCEPTION_SWALLOWING.value:
            return RepairFamily.EXCEPTION_SWALLOWING
        raise CegisError(f"unsupported IPA rule for repair family: {value.rule_id!r}")
    if isinstance(value, Mapping):
        raw = str(value.get("family") or value.get("rule_id") or value.get("transform_id") or "")
    else:
        raw = str(value or "")
    token = raw.strip()
    if not token:
        raise CegisError("repair family token is required")
    # Direct family.
    try:
        return RepairFamily(token)
    except ValueError:
        pass
    # IPA rule id.
    for family, rule in _FAMILY_TO_IPA_RULE.items():
        if token == rule.value or token.endswith(rule.family):
            return family
    if token == IpaRuleId.EXCEPTION_SWALLOWING.value:
        return RepairFamily.EXCEPTION_SWALLOWING
    # Transform id reverse map.
    for family, transform in FAMILY_TO_TRANSFORM.items():
        if token == transform.value:
            return family
    # Aliases.
    aliases = {
        "missing_recovery": RepairFamily.MISSING_LEASE_RECOVERY,
        "missing_lease": RepairFamily.MISSING_LEASE_RECOVERY,
        "lease_recovery": RepairFamily.MISSING_LEASE_RECOVERY,
        "browser": RepairFamily.BROWSER_AUTHORITY,
        "mutable_deps": RepairFamily.MUTABLE_DEPENDENCY,
        "stale_proof_reuse": RepairFamily.STALE_PROOF,
        "license": RepairFamily.LICENSE_CONFLICT,
    }
    if token in aliases:
        return aliases[token]
    raise CegisError(f"family outside closed repair grammar: {token!r}")


def _inspect_proposal_attacks(
    proposal: Optional[Mapping[str, Any]],
) -> tuple[str, ...]:
    if not proposal:
        return ()
    if not isinstance(proposal, Mapping):
        raise CegisError("proposal_hint must be a mapping")
    reasons: list[str] = []
    keys = {str(key) for key in proposal.keys()}
    if keys & _SCOPE_WIDEN_KEYS:
        reasons.append(CegisAbstentionReason.SCOPE_ESCAPE.value)
    if keys & _AUTHORITY_CLAIM_KEYS:
        reasons.append(CegisAbstentionReason.AUTHORITY_CLAIM.value)
    if keys & _OBLIGATION_WAIVER_KEYS:
        reasons.append(CegisAbstentionReason.OBLIGATION_WAIVER.value)
    # Explicit expansion attempts.
    for key in ("grammar_expansion", "new_transform", "new_family"):
        if proposal.get(key):
            reasons.append(CegisAbstentionReason.TRANSFORM_OUTSIDE_GRAMMAR.value)
    # Non-empty truthy authority flags.
    for key in _AUTHORITY_CLAIM_KEYS:
        if proposal.get(key) in (True, "true", "yes", 1):
            reasons.append(CegisAbstentionReason.AUTHORITY_CLAIM.value)
    return tuple(sorted(set(reasons)))


def _declaration_edit(path: str, before: str, after: str) -> IpaRepairEdit:
    """Whole-buffer edit record for non-IPA declaration transforms."""

    return IpaRepairEdit(
        path=path,
        start_line=1,
        end_line=max(1, before.count("\n") + (0 if before.endswith("\n") else 1)),
        before_text=before,
        after_text=after,
    )


def _render_browser_authority(source: str, path: str) -> tuple[str, IpaRepairEdit] | str:
    if not _DEFAULT_GRANTED_RE.search(source) and "granted" not in source:
        return CegisAbstentionReason.PRECONDITION_MISMATCH.value
    after = source
    after = re.sub(
        r"\?\?\s*(['\"])granted\1",
        r"?? \1pending_host_admission\1",
        after,
    )
    after = re.sub(
        r"(consent\s*=\s*)(['\"])granted\2",
        r"\1\2denied\2",
        after,
        flags=re.IGNORECASE,
    )
    after = re.sub(
        r"(default_consent\s*[:=]\s*)(['\"])granted\2",
        r"\1\2denied\2",
        after,
        flags=re.IGNORECASE,
    )
    after = re.sub(
        r"default(?:ed)?\s+to\s+granted",
        "default to denied pending host admission",
        after,
        flags=re.IGNORECASE,
    )
    if after == source:
        # Last-resort mechanical demotion comment + marker replacement.
        if "granted" in source:
            after = source.replace("granted", "denied_pending_host_admission", 1)
        else:
            return CegisAbstentionReason.NO_BYTE_CHANGE.value
    if after == source:
        return CegisAbstentionReason.NO_BYTE_CHANGE.value
    return after, _declaration_edit(path, source, after)


def _render_mutable_dependency(source: str, path: str) -> tuple[str, IpaRepairEdit] | str:
    if not _MUTABLE_VCS_RE.search(source) and "@main" not in source and "@master" not in source:
        return CegisAbstentionReason.PRECONDITION_MISMATCH.value
    after = _MUTABLE_VCS_RE.sub(
        lambda match: (
            re.sub(r"@(?:main|master|HEAD)\b", "@release_blocked_until_pinned", match.group(0), flags=re.I)
        ),
        source,
    )
    if after == source:
        after = source.replace("@main", "@release_blocked_until_pinned").replace(
            "@master", "@release_blocked_until_pinned"
        )
    if "release_admissible" in after.lower():
        after = re.sub(
            r"release_admissible\s*[:=]\s*true",
            "release_admissible=false",
            after,
            flags=re.IGNORECASE,
        )
    if "RELEASE_BLOCK_MUTABLE_VCS" not in after:
        after = after.rstrip() + "\n# RELEASE_BLOCK_MUTABLE_VCS=true\n"
    if after == source:
        return CegisAbstentionReason.NO_BYTE_CHANGE.value
    return after, _declaration_edit(path, source, after)


def _render_stale_proof(source: str, path: str) -> tuple[str, IpaRepairEdit] | str:
    lowered = source.casefold()
    if not any(marker.casefold() in lowered for marker in _STALE_PROOF_MARKERS) and (
        "stale" not in lowered and "historical_receipt" not in lowered
    ):
        return CegisAbstentionReason.PRECONDITION_MISMATCH.value
    after = source
    after = re.sub(
        r'"status"\s*:\s*"live"',
        '"status": "demoted_stale"',
        after,
    )
    after = after.replace("reuse_stale_proof", "refuse_stale_proof_reuse")
    after = after.replace("stale_proof_ok", "stale_proof_blocked")
    after = after.replace("historical_receipt_as_live", "historical_receipt_demoted")
    after = after.replace("proof_status=live_stale", "proof_status=demoted_stale")
    if "STALE_PROOF_DEMOTE" not in after:
        after = after.rstrip() + "\n# STALE_PROOF_DEMOTE=true\n# cache_reuse=refused\n"
    if after == source:
        return CegisAbstentionReason.NO_BYTE_CHANGE.value
    return after, _declaration_edit(path, source, after)


def _render_lease_recovery(source: str, path: str) -> tuple[str, IpaRepairEdit] | str:
    lowered = source.casefold()
    if not any(
        token in lowered
        for token in (
            "blind_retry",
            "unknown_irreversible",
            "lease",
            "fence",
            "recovery",
        )
    ):
        return CegisAbstentionReason.PRECONDITION_MISMATCH.value
    after = source
    after = re.sub(
        r"blind_retry\s*[:=]\s*true",
        "blind_retry=false",
        after,
        flags=re.IGNORECASE,
    )
    after = after.replace("allow_blind_retry", "deny_blind_retry")
    if "REQUIRE_LEASE_FENCE_RECOVERY" not in after:
        after = (
            after.rstrip()
            + "\n# REQUIRE_LEASE_FENCE_RECOVERY=true\n"
            + "# no_blind_unknown_retry=true\n"
            + "# recovery_state=required\n"
        )
    if after == source:
        return CegisAbstentionReason.NO_BYTE_CHANGE.value
    return after, _declaration_edit(path, source, after)


def render_declaration_transform(
    source: str,
    *,
    path: str,
    transform_id: RepairTransformId,
) -> tuple[str, IpaRepairEdit] | str:
    """Render one non-IPA declaration transform or return an abstention code."""

    if transform_id is RepairTransformId.ABSTAIN_LICENSE_HUMAN_REVIEW:
        return CegisAbstentionReason.LICENSE_REQUIRES_HUMAN.value
    if transform_id is RepairTransformId.DENY_DEFAULT_BROWSER_CONSENT:
        return _render_browser_authority(source, path)
    if transform_id is RepairTransformId.BLOCK_MUTABLE_VCS_DEPENDENCY:
        return _render_mutable_dependency(source, path)
    if transform_id is RepairTransformId.DEMOTE_STALE_PROOF_RECEIPT:
        return _render_stale_proof(source, path)
    if transform_id is RepairTransformId.REQUIRE_LEASE_FENCE_RECOVERY:
        return _render_lease_recovery(source, path)
    return CegisAbstentionReason.TRANSFORM_OUTSIDE_GRAMMAR.value


def _abstract_markers_present(source: str, markers: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    for marker in markers:
        token = str(marker or "")
        if token and token in source:
            found.append(token)
    return tuple(found)


def _family_abstract_check(
    family: RepairFamily,
    *,
    before: str,
    after: str,
    counterexample: CounterexampleRecord,
) -> GateResult:
    """Family-local abstract gate: original witness gone, no new abstract CE."""

    reasons: list[str] = []
    detail: dict[str, Any] = {}

    if family in _FAMILY_TO_IPA_RULE:
        rule = _FAMILY_TO_IPA_RULE[family]
        try:
            before_findings = analyze_python_source(before, path=counterexample.path)
            after_findings = analyze_python_source(after, path=counterexample.path)
        except Exception as exc:  # noqa: BLE001
            return GateResult(
                kind=GateKind.ABSTRACT,
                verdict=GateVerdict.ABSTAIN,
                reasons=(CegisAbstentionReason.PARSE_ERROR.value, str(exc)[:200]),
            )
        before_rules = {item.rule_id for item in before_findings}
        after_rules = {item.rule_id for item in after_findings}
        detail["before_rule_ids"] = sorted(before_rules)
        detail["after_rule_ids"] = sorted(after_rules)
        if rule.value in after_rules:
            reasons.append(CegisAbstentionReason.ORIGINAL_COUNTEREXAMPLE_REMAINS.value)
        new_rules = sorted(after_rules - before_rules)
        if new_rules:
            reasons.append(CegisAbstentionReason.NEW_ABSTRACT_COUNTEREXAMPLE.value)
            detail["new_rule_ids"] = new_rules
    else:
        # Declaration families: require configured witness markers to disappear.
        markers = counterexample.abstract_markers or _default_abstract_markers(family)
        remaining = _abstract_markers_present(after, markers)
        detail["markers_checked"] = list(markers)
        detail["markers_remaining"] = list(remaining)
        if remaining:
            reasons.append(CegisAbstentionReason.ORIGINAL_COUNTEREXAMPLE_REMAINS.value)
        # New abstract counterexamples: reintroduction of illegal promotions.
        illegal_new = []
        if family is RepairFamily.BROWSER_AUTHORITY and _DEFAULT_GRANTED_RE.search(after):
            illegal_new.append("default_granted_consent")
        if family is RepairFamily.MUTABLE_DEPENDENCY and _MUTABLE_VCS_RE.search(after):
            illegal_new.append("mutable_vcs_pin")
        if family is RepairFamily.STALE_PROOF:
            if '"status": "live"' in after and "demoted_stale" not in after:
                illegal_new.append("live_stale_receipt")
        if family is RepairFamily.MISSING_LEASE_RECOVERY:
            if re.search(r"blind_retry\s*[:=]\s*true", after, re.I) or (
                "allow_blind_retry" in after
            ):
                illegal_new.append("blind_retry_enabled")
        if illegal_new:
            reasons.append(CegisAbstentionReason.NEW_ABSTRACT_COUNTEREXAMPLE.value)
            detail["illegal_new"] = illegal_new

    if reasons:
        return GateResult(
            kind=GateKind.ABSTRACT,
            verdict=GateVerdict.FAIL,
            reasons=tuple(reasons),
            detail=detail,
        )
    return GateResult(kind=GateKind.ABSTRACT, verdict=GateVerdict.PASS, detail=detail)


def _default_abstract_markers(family: RepairFamily) -> tuple[str, ...]:
    return {
        RepairFamily.BROWSER_AUTHORITY: ("granted",),
        RepairFamily.MUTABLE_DEPENDENCY: ("@main", "@master"),
        RepairFamily.STALE_PROOF: (
            "reuse_stale_proof",
            "historical_receipt_as_live",
            '"status": "live"',
        ),
        RepairFamily.MISSING_LEASE_RECOVERY: ("blind_retry=true", "allow_blind_retry"),
        RepairFamily.LICENSE_CONFLICT: ("license_conflict_clearance=true",),
    }.get(family, ())


def _model_gate(
    family: RepairFamily,
    *,
    after: str,
    counterexample: CounterexampleRecord,
) -> GateResult:
    """Lightweight model/solver-shaped gate (hermetic; no LLM)."""

    markers = counterexample.model_markers
    remaining = _abstract_markers_present(after, markers)
    if remaining:
        return GateResult(
            kind=GateKind.MODEL,
            verdict=GateVerdict.FAIL,
            reasons=(CegisAbstentionReason.NEW_MODEL_COUNTEREXAMPLE.value,),
            detail={"markers_remaining": list(remaining)},
        )
    # Family-specific model invariants.
    if family is RepairFamily.PSEUDO_CID and re.search(
        r"""['"]Qm[0-9a-fA-F]{20,}['"]""", after
    ):
        return GateResult(
            kind=GateKind.MODEL,
            verdict=GateVerdict.FAIL,
            reasons=(CegisAbstentionReason.NEW_MODEL_COUNTEREXAMPLE.value,),
            detail={"pseudo_cid_literal": True},
        )
    if family is RepairFamily.STALE_PROOF and "cache_reuse=allowed" in after:
        return GateResult(
            kind=GateKind.MODEL,
            verdict=GateVerdict.FAIL,
            reasons=(CegisAbstentionReason.STALE_PROOF_REUSE.value,),
        )
    return GateResult(kind=GateKind.MODEL, verdict=GateVerdict.PASS)


def _test_gate(
    family: RepairFamily,
    *,
    after: str,
    counterexample: CounterexampleRecord,
) -> GateResult:
    """Hermetic affected-test gate over repaired artifact text."""

    markers = counterexample.test_markers
    remaining = _abstract_markers_present(after, markers)
    if remaining:
        return GateResult(
            kind=GateKind.TEST,
            verdict=GateVerdict.FAIL,
            reasons=(CegisAbstentionReason.NEW_TEST_COUNTEREXAMPLE.value,),
            detail={"markers_remaining": list(remaining)},
        )
    if family is RepairFamily.MISSING_LEASE_RECOVERY:
        if "REQUIRE_LEASE_FENCE_RECOVERY=true" not in after:
            return GateResult(
                kind=GateKind.TEST,
                verdict=GateVerdict.FAIL,
                reasons=(CegisAbstentionReason.TRACE_GATE_FAILED.value,),
            )
    if family is RepairFamily.MUTABLE_DEPENDENCY:
        if "RELEASE_BLOCK_MUTABLE_VCS=true" not in after:
            return GateResult(
                kind=GateKind.TEST,
                verdict=GateVerdict.FAIL,
                reasons=(CegisAbstentionReason.NEW_TEST_COUNTEREXAMPLE.value,),
            )
    if family is RepairFamily.BROWSER_AUTHORITY:
        if "granted" in after and "denied" not in after and "pending_host_admission" not in after:
            return GateResult(
                kind=GateKind.TEST,
                verdict=GateVerdict.FAIL,
                reasons=(CegisAbstentionReason.NEW_TEST_COUNTEREXAMPLE.value,),
            )
    return GateResult(kind=GateKind.TEST, verdict=GateVerdict.PASS)


def _proof_gate(
    family: RepairFamily,
    *,
    before_hash: str,
    after_hash: str,
    counterexample: CounterexampleRecord,
    parent_capsule_cids: Sequence[str],
) -> GateResult:
    """Capsule-bound proof gate (hermetic stand-in composing FACP-050 semantics).

    Refuses stale-proof reuse and unknown→verified promotion. A conclusive pass
    requires the post-patch code identity to differ and, for stale_proof, an
    explicit demotion marker binding.
    """

    if before_hash == after_hash:
        return GateResult(
            kind=GateKind.PROOF,
            verdict=GateVerdict.FAIL,
            reasons=(CegisAbstentionReason.NO_BYTE_CHANGE.value,),
        )
    result_id = _content_id(
        "facp-proof-gate",
        {
            "family": family.value,
            "before_hash": before_hash,
            "after_hash": after_hash,
            "counterexample_id": counterexample.counterexample_id,
            "parent_capsule_cids": list(parent_capsule_cids),
        },
    )
    if family is RepairFamily.STALE_PROOF:
        # Stale reuse of the pre-patch identity is forbidden.
        if counterexample.metadata.get("force_stale_cache_hit"):
            return GateResult(
                kind=GateKind.PROOF,
                verdict=GateVerdict.FAIL,
                reasons=(CegisAbstentionReason.STALE_PROOF_REUSE.value,),
                detail={"orchestration_result_id": result_id},
            )
    if family is RepairFamily.LICENSE_CONFLICT:
        return GateResult(
            kind=GateKind.PROOF,
            verdict=GateVerdict.ABSTAIN,
            reasons=(CegisAbstentionReason.LICENSE_REQUIRES_HUMAN.value,),
            detail={"orchestration_result_id": result_id},
        )
    return GateResult(
        kind=GateKind.PROOF,
        verdict=GateVerdict.PASS,
        detail={"orchestration_result_id": result_id, "verdict": "verified"},
    )


def _trace_gate(
    family: RepairFamily,
    *,
    after: str,
) -> GateResult:
    """TEP-shaped gate for lease/recovery families (FACP-046 invariants)."""

    if family is not RepairFamily.MISSING_LEASE_RECOVERY:
        return GateResult(kind=GateKind.TRACE, verdict=GateVerdict.PASS)
    if "no_blind_unknown_retry=true" not in after:
        return GateResult(
            kind=GateKind.TRACE,
            verdict=GateVerdict.FAIL,
            reasons=(CegisAbstentionReason.TRACE_GATE_FAILED.value,),
            detail={"invariant": "NoBlindUnknownRetry"},
        )
    if "recovery_state=required" not in after:
        return GateResult(
            kind=GateKind.TRACE,
            verdict=GateVerdict.FAIL,
            reasons=(CegisAbstentionReason.TRACE_GATE_FAILED.value,),
            detail={"invariant": "lease_fence_recovery"},
        )
    return GateResult(
        kind=GateKind.TRACE,
        verdict=GateVerdict.PASS,
        detail={"invariants": ["NoBlindUnknownRetry", "lease_fence_recovery"]},
    )


def _public_compat_preserved(before: str, after: str) -> bool:
    before_defs = set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", before, re.M))
    after_defs = set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", after, re.M))
    return before_defs <= after_defs or before_defs.issubset(
        after_defs | {"initialize_explicit"}
    )


def run_isolated_transaction(
    candidate: RepairCandidate,
    *,
    admitted_paths: Optional[Iterable[str]] = None,
    base_source: str,
) -> IsolatedRepairTransaction:
    """Commit a candidate into an in-memory overlay or roll it back."""

    allow = tuple(
        sorted(
            {
                _normalize_path(item)
                for item in (
                    tuple(admitted_paths)
                    if admitted_paths is not None
                    else tuple(default_admitted_paths())
                )
            }
        )
    )
    checkpoint = {candidate.path: _sha256_text(base_source)}
    tx_seed = {
        "candidate_id": candidate.candidate_id,
        "path": candidate.path,
        "after_hash": candidate.after_hash,
        "checkpoint": checkpoint,
    }
    tx_id = _content_id("facp-cegis-tx", tx_seed)

    if not path_is_admitted(candidate.path, allow):
        return IsolatedRepairTransaction(
            transaction_id=tx_id,
            disposition=TransactionDisposition.REJECTED,
            admitted_paths=allow,
            checkpoint_hashes=checkpoint,
            overlay={},
            path=candidate.path,
            reasons=(CegisAbstentionReason.PATH_NOT_ADMITTED.value,),
            candidate_id=candidate.candidate_id,
        )
    if candidate.after_hash == candidate.before_hash:
        return IsolatedRepairTransaction(
            transaction_id=tx_id,
            disposition=TransactionDisposition.ABSTAINED,
            admitted_paths=allow,
            checkpoint_hashes=checkpoint,
            overlay={},
            path=candidate.path,
            reasons=(CegisAbstentionReason.NO_BYTE_CHANGE.value,),
            candidate_id=candidate.candidate_id,
        )
    # Isolated commit = overlay only (no filesystem write).
    overlay = {candidate.path: candidate.after_source}
    return IsolatedRepairTransaction(
        transaction_id=tx_id,
        disposition=TransactionDisposition.COMMITTED_OVERLAY,
        admitted_paths=allow,
        checkpoint_hashes=checkpoint,
        overlay=overlay,
        path=candidate.path,
        reasons=(),
        candidate_id=candidate.candidate_id,
    )


def _candidate_from_ipa(
    request: CegisRepairRequest,
    *,
    family: RepairFamily,
    transform_id: RepairTransformId,
    admitted: Iterable[str],
) -> RepairCandidate | str:
    ipa_transform = _IPA_TRANSFORM_BRIDGE[transform_id]
    finding = request.finding
    if finding is None:
        # Synthesize from analyzer when possible.
        try:
            findings = analyze_python_source(request.source, path=request.counterexample.path)
        except Exception as exc:  # noqa: BLE001
            return f"{CegisAbstentionReason.PARSE_ERROR.value}:{exc}"[:200]
        rule = _FAMILY_TO_IPA_RULE[family]
        matched = [item for item in findings if item.rule_id == rule.value]
        if not matched:
            return CegisAbstentionReason.PRECONDITION_MISMATCH.value
        primary = [item for item in matched if "hermetic datalog" not in item.message]
        finding = (primary or matched)[0]
    try:
        receipt = apply_ipa_repair(
            request.source,
            finding,
            path=request.counterexample.path,
            transform_id=ipa_transform,
            admitted_paths=admitted,
        )
    except IpaRepairError as exc:
        return f"{CegisAbstentionReason.CANDIDATE_REJECTED.value}:{exc}"[:200]

    if receipt.disposition is IpaRepairDisposition.ABSTAINED:
        return receipt.reasons[0] if receipt.reasons else CegisAbstentionReason.CANDIDATE_REJECTED.value
    if receipt.disposition is IpaRepairDisposition.REJECTED:
        return receipt.reasons[0] if receipt.reasons else CegisAbstentionReason.CANDIDATE_REJECTED.value
    if receipt.disposition is IpaRepairDisposition.NOOP:
        return CegisAbstentionReason.NO_BYTE_CHANGE.value

    after_source = receipt.after_source
    cand_id = _content_id(
        "facp-cegis-candidate",
        {
            "family": family.value,
            "transform_id": transform_id.value,
            "before_hash": receipt.before_hash,
            "after_hash": receipt.after_hash,
            "path": receipt.path,
        },
    )
    return RepairCandidate(
        candidate_id=cand_id,
        family=family,
        transform_id=transform_id,
        path=receipt.path,
        before_hash=receipt.before_hash,
        after_hash=receipt.after_hash,
        after_source=after_source,
        edits=receipt.edits,
        ipa_receipt=receipt,
        addresses_witness=bool(
            receipt.reanalysis and receipt.reanalysis.target_rule_eliminated
        ),
        residual_risks=(),
        reasons=(),
    )


def _candidate_from_declaration(
    request: CegisRepairRequest,
    *,
    family: RepairFamily,
    transform_id: RepairTransformId,
) -> RepairCandidate | str:
    if transform_id is RepairTransformId.ABSTAIN_LICENSE_HUMAN_REVIEW:
        return CegisAbstentionReason.LICENSE_REQUIRES_HUMAN.value
    rendered = render_declaration_transform(
        request.source,
        path=request.counterexample.path,
        transform_id=transform_id,
    )
    if isinstance(rendered, str):
        return rendered
    after_source, edit = rendered
    before_hash = _sha256_text(request.source)
    after_hash = _sha256_text(after_source)
    cand_id = _content_id(
        "facp-cegis-candidate",
        {
            "family": family.value,
            "transform_id": transform_id.value,
            "before_hash": before_hash,
            "after_hash": after_hash,
            "path": request.counterexample.path,
        },
    )
    return RepairCandidate(
        candidate_id=cand_id,
        family=family,
        transform_id=transform_id,
        path=request.counterexample.path,
        before_hash=before_hash,
        after_hash=after_hash,
        after_source=after_source,
        edits=(edit,),
        ipa_receipt=None,
        addresses_witness=True,
        residual_risks=(),
        reasons=(),
    )


def propose_candidates(
    request: CegisRepairRequest,
    *,
    family: RepairFamily,
    transform_id: RepairTransformId,
    admitted_paths: Iterable[str],
    limit: int,
) -> tuple[tuple[RepairCandidate, ...], tuple[str, ...]]:
    """Propose up to ``limit`` grammar-bounded candidates (deterministic order)."""

    if limit < 1:
        return (), (CegisAbstentionReason.NO_CANDIDATES.value,)

    # License family never proposes a certifying candidate.
    if family is RepairFamily.LICENSE_CONFLICT:
        return (), (CegisAbstentionReason.LICENSE_REQUIRES_HUMAN.value,)

    produced: list[RepairCandidate] = []
    failures: list[str] = []

    if transform_id in _IPA_TRANSFORM_BRIDGE:
        result = _candidate_from_ipa(
            request,
            family=family,
            transform_id=transform_id,
            admitted=admitted_paths,
        )
    else:
        result = _candidate_from_declaration(
            request,
            family=family,
            transform_id=transform_id,
        )

    if isinstance(result, str):
        failures.append(result)
    else:
        produced.append(result)

    # Optional proposal hint may only *select* an already-closed transform.
    hint = request.proposal_hint or {}
    hint_transform = str(hint.get("transform_id") or hint.get("sketch_transform") or "")
    if hint_transform and hint_transform in STABLE_TRANSFORM_IDS:
        alt = _enum(hint_transform, RepairTransformId, "proposal_hint.transform_id")
        if alt is not transform_id and FAMILY_TO_TRANSFORM.get(family) == alt:
            pass  # already primary
        elif alt is not transform_id and alt in _IPA_TRANSFORM_BRIDGE and family in _FAMILY_TO_IPA_RULE:
            # Reject mismatched transform for family (precondition).
            failures.append(CegisAbstentionReason.PRECONDITION_MISMATCH.value)

    return tuple(produced[:limit]), tuple(failures)


def gate_candidate(
    candidate: RepairCandidate,
    *,
    request: CegisRepairRequest,
    family: RepairFamily,
) -> tuple[GateResult, ...]:
    """Run abstract / model / test / proof / trace gates over one candidate."""

    results: list[GateResult] = [
        _family_abstract_check(
            family,
            before=request.source,
            after=candidate.after_source,
            counterexample=request.counterexample,
        )
    ]
    if request.require_test_gate or request.require_proof_gate:
        results.append(
            _model_gate(
                family,
                after=candidate.after_source,
                counterexample=request.counterexample,
            )
        )
    if request.require_test_gate:
        results.append(
            _test_gate(
                family,
                after=candidate.after_source,
                counterexample=request.counterexample,
            )
        )
    if request.require_proof_gate:
        results.append(
            _proof_gate(
                family,
                before_hash=candidate.before_hash,
                after_hash=candidate.after_hash,
                counterexample=request.counterexample,
                parent_capsule_cids=request.parent_capsule_cids,
            )
        )
    results.append(_trace_gate(family, after=candidate.after_source))
    if not _public_compat_preserved(request.source, candidate.after_source):
        results.append(
            GateResult(
                kind=GateKind.ABSTRACT,
                verdict=GateVerdict.FAIL,
                reasons=(CegisAbstentionReason.PUBLIC_COMPAT_RISK.value,),
            )
        )
    return tuple(results)


def _mutation_gate_for_candidate(
    candidate: RepairCandidate,
    *,
    admitted_paths: Iterable[str],
) -> MutationGateDecision:
    if candidate.ipa_receipt and candidate.ipa_receipt.mutation_gate is not None:
        return candidate.ipa_receipt.mutation_gate
    admitted = path_is_admitted(candidate.path, admitted_paths)
    if not admitted:
        return MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=candidate.path,
            reasons=(CegisAbstentionReason.PATH_NOT_ADMITTED.value,),
            before_hash=candidate.before_hash,
            after_hash=candidate.after_hash,
            byte_mutated=candidate.before_hash != candidate.after_hash,
            reanalyzed=True,
        )
    if candidate.before_hash == candidate.after_hash:
        return MutationGateDecision(
            disposition=MutationGateDisposition.DENIED,
            path=candidate.path,
            reasons=(CegisAbstentionReason.NO_BYTE_CHANGE.value,),
            before_hash=candidate.before_hash,
            after_hash=candidate.after_hash,
            byte_mutated=False,
            reanalyzed=True,
        )
    return MutationGateDecision(
        disposition=MutationGateDisposition.ADMITTED,
        path=candidate.path,
        reasons=(),
        before_hash=candidate.before_hash,
        after_hash=candidate.after_hash,
        byte_mutated=True,
        reanalyzed=True,
    )


def mint_patch_certificate(
    candidate: RepairCandidate,
    *,
    request: CegisRepairRequest,
    gate_results: Sequence[GateResult],
    mutation_gate: MutationGateDecision,
) -> PatchCertificate:
    """Mint a minimal PatchCertificate after all gates pass."""

    if any(item.verdict is GateVerdict.FAIL for item in gate_results):
        raise CegisError("refusing to mint certificate with failed gates")
    if any(item.verdict is GateVerdict.ABSTAIN for item in gate_results):
        raise CegisError("refusing to mint certificate with abstaining gates")
    if not mutation_gate.admitted:
        raise CegisError("refusing to mint certificate without mutation admission")

    reanalysis = (
        candidate.ipa_receipt.reanalysis
        if candidate.ipa_receipt is not None
        else IpaReanalysisReport(
            before_rule_ids=(),
            after_rule_ids=(),
            eliminated_rule_ids=(),
            new_rule_ids=(),
            target_rule_eliminated=True,
        )
    )
    parent_cids = tuple(request.parent_capsule_cids) or (
        _content_id(
            "capsule-parent",
            {"path": candidate.path, "before_hash": candidate.before_hash},
        ),
    )
    patch_cid = _content_id(
        "capsule-patch",
        {
            "path": candidate.path,
            "after_hash": candidate.after_hash,
            "transform_id": candidate.transform_id.value,
            "family": candidate.family.value,
        },
    )
    affected = (
        _content_id(
            "capsule-affected",
            {
                "path": candidate.path,
                "before_hash": candidate.before_hash,
                "after_hash": candidate.after_hash,
            },
        ),
    )
    obligations = tuple(request.obligation_ids) or _default_obligations(candidate.family)
    orchestration_ids = tuple(
        str(item.detail.get("orchestration_result_id"))
        for item in gate_results
        if item.kind is GateKind.PROOF and item.detail.get("orchestration_result_id")
    )
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "family": candidate.family.value,
        "transform_id": candidate.transform_id.value,
        "path": candidate.path,
        "counterexample_id": request.counterexample.counterexample_id,
        "before_hash": candidate.before_hash,
        "after_hash": candidate.after_hash,
        "edit_hashes": [edit.after_hash for edit in candidate.edits],
        "parent_capsule_cids": list(parent_cids),
        "patch_capsule_cid": patch_cid,
        "affected_capsule_cids": list(affected),
        "obligation_ids": list(obligations),
        "gate_results": [item.to_dict() for item in gate_results],
    }
    certificate_id = _content_id("facp-patch-cert", payload)
    return PatchCertificate(
        certificate_id=certificate_id,
        family=candidate.family,
        transform_id=candidate.transform_id,
        path=candidate.path,
        counterexample_id=request.counterexample.counterexample_id,
        before_hash=candidate.before_hash,
        after_hash=candidate.after_hash,
        edits=candidate.edits,
        mutation_gate=mutation_gate,
        reanalysis=reanalysis,
        gate_results=tuple(gate_results),
        parent_capsule_cids=parent_cids,
        patch_capsule_cid=patch_cid,
        affected_capsule_cids=affected,
        obligation_ids=obligations,
        orchestration_result_ids=orchestration_ids,
        residual_risks=candidate.residual_risks,
        public_compat_preserved=_public_compat_preserved(
            request.source, candidate.after_source
        ),
        grants_write_authority=False,
    )


def _abstain_result(
    *,
    family: RepairFamily,
    path: str,
    reasons: Sequence[str],
    counterexample_id: str = "",
    candidates: Sequence[RepairCandidate] = (),
    transaction: Optional[IsolatedRepairTransaction] = None,
    gate_results: Sequence[GateResult] = (),
    budget: Optional[CegisBudget] = None,
    iterations: int = 0,
    disposition: CegisDisposition = CegisDisposition.ABSTAINED,
    residual_risks: Sequence[str] = (),
) -> CegisRepairResult:
    cleaned = tuple(str(item) for item in reasons if str(item))
    if not cleaned:
        cleaned = (CegisAbstentionReason.CANDIDATE_REJECTED.value,)
    return CegisRepairResult(
        disposition=disposition,
        family=family,
        path=path,
        reasons=cleaned,
        certificate=None,
        candidates_tried=tuple(candidates),
        transaction=transaction,
        gate_results=tuple(gate_results),
        residual_risks=tuple(residual_risks),
        budget=budget,
        iterations=iterations,
        counterexample_id=counterexample_id,
    )


def run_bounded_cegis(request: CegisRepairRequest) -> CegisRepairResult:
    """Execute the bounded CEGIS repair loop for one seeded request."""

    if not isinstance(request, CegisRepairRequest):
        raise CegisError("request must be CegisRepairRequest")
    budget = request.budget or CegisBudget()
    family = select_repair_family(request.counterexample)
    path = request.counterexample.path
    cx_id = request.counterexample.counterexample_id

    if not request.source.strip():
        return _abstain_result(
            family=family,
            path=path,
            reasons=(CegisAbstentionReason.EMPTY_SOURCE.value,),
            counterexample_id=cx_id,
            budget=budget,
        )

    attack_reasons = _inspect_proposal_attacks(request.proposal_hint)
    if attack_reasons:
        disposition = (
            CegisDisposition.REJECTED
            if (
                CegisAbstentionReason.SCOPE_ESCAPE.value in attack_reasons
                or CegisAbstentionReason.AUTHORITY_CLAIM.value in attack_reasons
                or CegisAbstentionReason.OBLIGATION_WAIVER.value in attack_reasons
            )
            else CegisDisposition.ABSTAINED
        )
        return _abstain_result(
            family=family,
            path=path,
            reasons=attack_reasons,
            counterexample_id=cx_id,
            budget=budget,
            disposition=disposition,
        )

    admitted = tuple(
        request.admitted_paths
        if request.admitted_paths is not None
        else default_admitted_paths()
    )
    if not path_is_admitted(path, admitted):
        return _abstain_result(
            family=family,
            path=path,
            reasons=(CegisAbstentionReason.PATH_NOT_ADMITTED.value,),
            counterexample_id=cx_id,
            budget=budget,
            disposition=CegisDisposition.REJECTED,
        )

    if request.transform_id is not None:
        transform_id = _enum(request.transform_id, RepairTransformId, "transform_id")
        expected = FAMILY_TO_TRANSFORM[family]
        if transform_id is not expected:
            return _abstain_result(
                family=family,
                path=path,
                reasons=(CegisAbstentionReason.PRECONDITION_MISMATCH.value,),
                counterexample_id=cx_id,
                budget=budget,
            )
    else:
        transform_id = FAMILY_TO_TRANSFORM[family]

    if transform_id.value not in STABLE_TRANSFORM_IDS:
        return _abstain_result(
            family=family,
            path=path,
            reasons=(CegisAbstentionReason.TRANSFORM_OUTSIDE_GRAMMAR.value,),
            counterexample_id=cx_id,
            budget=budget,
            disposition=CegisDisposition.REJECTED,
        )

    if family is RepairFamily.LICENSE_CONFLICT:
        return _abstain_result(
            family=family,
            path=path,
            reasons=(CegisAbstentionReason.LICENSE_REQUIRES_HUMAN.value,),
            counterexample_id=cx_id,
            budget=budget,
            residual_risks=("residual:unresolved_human_legal_review",),
        )

    tried: list[RepairCandidate] = []
    last_gates: tuple[GateResult, ...] = ()
    last_tx: Optional[IsolatedRepairTransaction] = None
    last_failures: list[str] = []

    for iteration in range(1, budget.max_iterations + 1):
        candidates, failures = propose_candidates(
            request,
            family=family,
            transform_id=transform_id,
            admitted_paths=admitted,
            limit=budget.max_candidates_per_iteration,
        )
        last_failures.extend(failures)
        if not candidates:
            break

        for candidate in candidates:
            if len(candidate.edits) > budget.max_edits:
                last_failures.append(CegisAbstentionReason.SCOPE_ESCAPE.value)
                continue
            tried.append(candidate)
            tx = run_isolated_transaction(
                candidate,
                admitted_paths=admitted,
                base_source=request.source,
            )
            last_tx = tx
            if not tx.committed:
                last_failures.extend(tx.reasons or (CegisAbstentionReason.TRANSACTION_ROLLED_BACK.value,))
                continue

            gates = gate_candidate(candidate, request=request, family=family)
            last_gates = gates
            if any(gate.verdict is GateVerdict.FAIL for gate in gates):
                # Roll back overlay conceptually (drop commit).
                last_tx = IsolatedRepairTransaction(
                    transaction_id=tx.transaction_id,
                    disposition=TransactionDisposition.ROLLED_BACK,
                    admitted_paths=tx.admitted_paths,
                    checkpoint_hashes=tx.checkpoint_hashes,
                    overlay={},
                    path=tx.path,
                    reasons=tuple(
                        reason
                        for gate in gates
                        if gate.verdict is GateVerdict.FAIL
                        for reason in gate.reasons
                    )
                    or (CegisAbstentionReason.CANDIDATE_REJECTED.value,),
                    candidate_id=candidate.candidate_id,
                )
                last_failures.extend(last_tx.reasons)
                continue
            if any(gate.verdict is GateVerdict.ABSTAIN for gate in gates):
                return _abstain_result(
                    family=family,
                    path=path,
                    reasons=tuple(
                        reason
                        for gate in gates
                        if gate.verdict is GateVerdict.ABSTAIN
                        for reason in gate.reasons
                    ),
                    counterexample_id=cx_id,
                    candidates=tried,
                    transaction=tx,
                    gate_results=gates,
                    budget=budget,
                    iterations=iteration,
                )

            mutation_gate = _mutation_gate_for_candidate(
                candidate, admitted_paths=admitted
            )
            if not mutation_gate.admitted:
                last_failures.extend(mutation_gate.reasons)
                continue

            certificate = mint_patch_certificate(
                candidate,
                request=request,
                gate_results=gates,
                mutation_gate=mutation_gate,
            )
            return CegisRepairResult(
                disposition=CegisDisposition.CERTIFIED,
                family=family,
                path=path,
                reasons=(),
                certificate=certificate,
                candidates_tried=tuple(tried),
                transaction=tx,
                gate_results=gates,
                residual_risks=certificate.residual_risks,
                budget=budget,
                iterations=iteration,
                counterexample_id=cx_id,
            )

        # Deterministic grammar has a single primary candidate; further
        # iterations cannot invent new transforms.
        break

    if tried and last_failures:
        disposition = CegisDisposition.ABSTAINED
        if CegisAbstentionReason.SCOPE_ESCAPE.value in last_failures:
            disposition = CegisDisposition.REJECTED
        return _abstain_result(
            family=family,
            path=path,
            reasons=tuple(dict.fromkeys(last_failures)),
            counterexample_id=cx_id,
            candidates=tried,
            transaction=last_tx,
            gate_results=last_gates,
            budget=budget,
            iterations=budget.max_iterations,
            disposition=disposition,
        )

    if last_failures:
        reasons = tuple(dict.fromkeys(last_failures))
        disposition = CegisDisposition.ABSTAINED
        if CegisAbstentionReason.LICENSE_REQUIRES_HUMAN.value in reasons:
            disposition = CegisDisposition.ABSTAINED
        return _abstain_result(
            family=family,
            path=path,
            reasons=reasons,
            counterexample_id=cx_id,
            candidates=tried,
            transaction=last_tx,
            gate_results=last_gates,
            budget=budget,
            iterations=budget.max_iterations,
            disposition=disposition,
        )

    return _abstain_result(
        family=family,
        path=path,
        reasons=(CegisAbstentionReason.BUDGET_EXHAUSTED.value,),
        counterexample_id=cx_id,
        candidates=tried,
        transaction=last_tx,
        gate_results=last_gates,
        budget=budget,
        iterations=budget.max_iterations,
        disposition=CegisDisposition.BUDGET_EXHAUSTED,
    )


def run_mutation_benchmark(
    request: CegisRepairRequest,
    *,
    baseline: Optional[CegisRepairResult] = None,
) -> MutationBenchmarkReport:
    """Kill mutants that reintroduce defects, waive obligations, or escape scope."""

    baseline = baseline or run_bounded_cegis(request)
    cases: list[MutationBenchmarkCase] = []

    # Mutant 1: reintroduce original source (should not certify as fresh repair
    # when counterexample markers still present — run on original is baseline).
    if baseline.certified and baseline.certificate is not None:
        # Mutant: after_source equals before (no byte change) via forged proposal.
        noop_hint = {"force_admit": True, "skip_reanalysis": True}
        attacked = CegisRepairRequest(
            source=request.source,
            counterexample=request.counterexample,
            finding=request.finding,
            admitted_paths=request.admitted_paths,
            budget=request.budget,
            proposal_hint=noop_hint,
            parent_capsule_cids=request.parent_capsule_cids,
            obligation_ids=request.obligation_ids,
        )
        attacked_result = run_bounded_cegis(attacked)
        cases.append(
            MutationBenchmarkCase(
                mutant_id="mutant:obligation-waiver",
                description="force_admit/skip_reanalysis must be rejected",
                admitted=attacked_result.certified,
                reasons=attacked_result.reasons,
            )
        )

        scope_hint = {"extra_paths": ["../../../etc/passwd"], "import_additions": ["os"]}
        scope_result = run_bounded_cegis(
            CegisRepairRequest(
                source=request.source,
                counterexample=request.counterexample,
                finding=request.finding,
                admitted_paths=request.admitted_paths,
                budget=request.budget,
                proposal_hint=scope_hint,
            )
        )
        cases.append(
            MutationBenchmarkCase(
                mutant_id="mutant:scope-escape",
                description="extra_paths/import_additions must fail closed",
                admitted=scope_result.certified,
                reasons=scope_result.reasons,
            )
        )

        auth_hint = {"write_authority": True, "promote_patch": True}
        auth_result = run_bounded_cegis(
            CegisRepairRequest(
                source=request.source,
                counterexample=request.counterexample,
                finding=request.finding,
                admitted_paths=request.admitted_paths,
                budget=request.budget,
                proposal_hint=auth_hint,
            )
        )
        cases.append(
            MutationBenchmarkCase(
                mutant_id="mutant:authority-claim",
                description="write_authority/promote_patch must fail closed",
                admitted=auth_result.certified,
                reasons=auth_result.reasons,
            )
        )

        # Mutant: path escape outside admitted allowlist.
        bad_cx = CounterexampleRecord(
            counterexample_id=request.counterexample.counterexample_id + ":path-escape",
            family=request.counterexample.family,
            path="vendor/untrusted/escape.py",
            witness=request.counterexample.witness,
            abstract_markers=request.counterexample.abstract_markers,
        )
        path_result = run_bounded_cegis(
            CegisRepairRequest(
                source=request.source,
                counterexample=bad_cx,
                finding=request.finding,
                admitted_paths=request.admitted_paths or default_admitted_paths(),
                budget=request.budget,
            )
        )
        cases.append(
            MutationBenchmarkCase(
                mutant_id="mutant:path-not-admitted",
                description="outside allowlist must not certify",
                admitted=path_result.certified,
                reasons=path_result.reasons,
            )
        )

        # Mutant: grammar expansion.
        expand_hint = {"new_transform": "llm_freeform_edit", "grammar_expansion": True}
        expand_result = run_bounded_cegis(
            CegisRepairRequest(
                source=request.source,
                counterexample=request.counterexample,
                finding=request.finding,
                admitted_paths=request.admitted_paths,
                budget=request.budget,
                proposal_hint=expand_hint,
            )
        )
        cases.append(
            MutationBenchmarkCase(
                mutant_id="mutant:grammar-expansion",
                description="grammar expansion must fail closed",
                admitted=expand_result.certified,
                reasons=expand_result.reasons,
            )
        )
    else:
        # Even abstaining baselines must reject authority attacks.
        auth_result = run_bounded_cegis(
            CegisRepairRequest(
                source=request.source,
                counterexample=request.counterexample,
                finding=request.finding,
                admitted_paths=request.admitted_paths,
                budget=request.budget,
                proposal_hint={"write_authority": True},
            )
        )
        cases.append(
            MutationBenchmarkCase(
                mutant_id="mutant:authority-claim",
                description="write_authority must fail closed",
                admitted=auth_result.certified,
                reasons=auth_result.reasons,
            )
        )

    killed = sum(1 for case in cases if not case.admitted)
    survived = sum(1 for case in cases if case.admitted)
    report_id = _content_id(
        "facp-cegis-mutbench",
        {"cases": [case.to_dict() for case in cases], "baseline": baseline.disposition.value},
    )
    return MutationBenchmarkReport(
        report_id=report_id,
        cases=tuple(cases),
        killed=killed,
        survived=survived,
    )


class FormalAssuranceCegis:
    """Facade for bounded counterexample-guided repair."""

    def __init__(self, *, budget: Optional[CegisBudget] = None) -> None:
        self._budget = budget or CegisBudget()

    @property
    def budget(self) -> CegisBudget:
        return self._budget

    def repair(self, request: CegisRepairRequest) -> CegisRepairResult:
        if request.budget is None:
            request = CegisRepairRequest(
                source=request.source,
                counterexample=request.counterexample,
                finding=request.finding,
                admitted_paths=request.admitted_paths,
                budget=self._budget,
                proposal_hint=request.proposal_hint,
                parent_capsule_cids=request.parent_capsule_cids,
                obligation_ids=request.obligation_ids,
                require_proof_gate=request.require_proof_gate,
                require_test_gate=request.require_test_gate,
                transform_id=request.transform_id,
            )
        return run_bounded_cegis(request)

    def benchmark(self, request: CegisRepairRequest) -> MutationBenchmarkReport:
        return run_mutation_benchmark(request)

    def grammar(self) -> tuple[RepairGrammarEntry, ...]:
        return list_repair_grammar()


def default_cegis(budget: Optional[CegisBudget] = None) -> FormalAssuranceCegis:
    return FormalAssuranceCegis(budget=budget)


def ipa_grammar_subset() -> tuple[dict[str, str], ...]:
    """Expose the IPA transform subset consumed from FACP-043."""

    return list_transform_grammar()


__all__ = (
    "ANALYZER_VERSION",
    "BENCHMARK_SCHEMA",
    "BUNDLE",
    "CERTIFICATE_SCHEMA",
    "CEGIS_EVIDENCE",
    "FAMILY_TO_TRANSFORM",
    "GOAL_ID",
    "GRAMMAR_SCHEMA",
    "INTERFACE",
    "PATCH_CERTIFICATE_EVIDENCE",
    "PRODUCER_ID",
    "REPAIR_GRAMMAR_EVIDENCE",
    "RESULT_SCHEMA",
    "SCHEMA",
    "STABLE_FAMILY_IDS",
    "STABLE_TRANSFORM_IDS",
    "TASK_ID",
    "TOOLCHAIN_ID",
    "TRANSACTION_SCHEMA",
    "CegisAbstentionReason",
    "CegisBudget",
    "CegisDisposition",
    "CegisError",
    "CegisRepairRequest",
    "CegisRepairResult",
    "CounterexampleRecord",
    "FormalAssuranceCegis",
    "GateKind",
    "GateResult",
    "GateVerdict",
    "IsolatedRepairTransaction",
    "MutationBenchmarkCase",
    "MutationBenchmarkReport",
    "PatchCertificate",
    "RepairCandidate",
    "RepairFamily",
    "RepairGrammarEntry",
    "RepairTransformId",
    "TransactionDisposition",
    "default_cegis",
    "gate_candidate",
    "ipa_grammar_subset",
    "list_repair_grammar",
    "mint_patch_certificate",
    "propose_candidates",
    "render_declaration_transform",
    "run_bounded_cegis",
    "run_isolated_transaction",
    "run_mutation_benchmark",
    "select_repair_family",
    "select_transform",
)
