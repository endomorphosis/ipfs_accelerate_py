"""Provider-free orchestration for proof-directed supervisor decisions.

The repository already has strict contracts for decisions, contexts, plan
admission, and exact execution permits.  This module composes those contracts
at one live boundary.  Optional model, dataset, graph, and prover providers are
deliberately injected; importing or inspecting this module never resolves one.

``OFF`` and ``SHADOW`` are compatibility modes.  They never manufacture
execution or completion authority.  ``ENFORCE`` is fail closed: a complete
current decision, admitted plan, context witness, and permit are required
immediately before every effect.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .control.control_contracts import EventCursor
from .decision_context import ContextCompletenessWitness
from .decision_contracts import (
    ApplicabilityFact,
    DecisionKind,
    DecisionRequest,
    DecisionStage,
    EffectEnvelope,
    MANDATORY_SEMANTIC_ROOT_KINDS,
    SemanticRoot,
)
from .control.execution_permit import (
    ExecutionAttempt,
    ExecutionEvidence,
    ExecutionPermit,
    ExecutionPermitIssuer,
    PermitUseReceipt,
    PermitVerificationError,
)
from .event_log import (
    CursorReplayError,
    SemanticChange,
    SemanticChangeIntegrityError,
    SemanticChangeKind,
    read_jsonl_event_page,
    read_semantic_change_page,
)
from .formal_verification_contracts import canonical_json_bytes, content_identity
from .ir_constraint_compiler import (
    PlanAdmissionReceipt,
    PlanAdmissionRequest,
    compile_plan_admission,
)


DECISION_RUNTIME_VERSION: Final[int] = 1
DECISION_RUNTIME_CONFIG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/decision-runtime-config@1"
)
DECISION_RUNTIME_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/decision-runtime-receipt@1"
)
EFFECT_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/effect-observation@1"
)
RUNTIME_INVALIDATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/runtime-invalidation-receipt@1"
)
INCREMENTAL_REVALIDATION_REQUIREMENT_ID: Final[str] = (
    "asi-138:dependency-local-invalidation-reproof-recovery"
)
DEFAULT_RUNTIME_CALLER: Final[str] = "agent-supervisor:decision-runtime"
DEFAULT_RUNTIME_POLICY_ID: Final[str] = "policy:decision-runtime"
DEFAULT_RUNTIME_POLICY_REVISION: Final[str] = "sha256:decision-runtime-v1"


class DecisionRuntimeError(RuntimeError):
    """Base error raised at the unified decision boundary."""


class DecisionRuntimeConfigurationError(DecisionRuntimeError):
    """The provider-free runtime configuration is malformed or incomplete."""


class DecisionRuntimeDenied(DecisionRuntimeError):
    """An enforced decision did not establish authority."""

    def __init__(self, reason_codes: Sequence[str], message: str = "") -> None:
        self.reason_codes = tuple(sorted({str(item) for item in reason_codes}))
        super().__init__(
            message
            or "decision runtime denied operation: "
            + ", ".join(self.reason_codes)
        )


class DecisionRuntimeBypassError(DecisionRuntimeDenied):
    """A caller tried to reach an effect without a runtime decision."""


class DecisionRuntimeCancelled(DecisionRuntimeDenied):
    """Cancellation was observed before an authority-bearing effect."""


class DecisionRuntimeEffectMismatch(DecisionRuntimeDenied):
    """Observed effects are not exactly the admitted expected effects."""

    def __init__(self, receipt: "EffectObservationReceipt") -> None:
        self.receipt = receipt
        super().__init__(
            receipt.reason_codes or ("observed_effect_mismatch",),
            "observed effects do not exactly match the admitted effects",
        )


class DecisionRuntimeMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"

    @property
    def blocks(self) -> bool:
        return self is DecisionRuntimeMode.ENFORCE


class DecisionBoundary(str, Enum):
    TASK_PROPOSAL = "task_proposal"
    ANALYSIS_REQUEST = "analysis_request"
    PLAN_SELECTION = "plan_selection"
    IMPLEMENTATION_CONTEXT = "implementation_context"
    RETRY = "retry"
    EXPANSION = "expansion"
    VALIDATION_SELECTION = "validation_selection"
    VALIDATION_EXECUTION = "validation_execution"
    FILE_MUTATION = "file_mutation"
    TASK_BOARD_MUTATION = "task_board_mutation"
    COMMAND_INVOCATION = "command_invocation"
    TOOL_INVOCATION = "tool_invocation"
    COMMIT = "commit"
    MERGE = "merge"
    COMPLETION = "completion"

    @property
    def mutating(self) -> bool:
        return self in {
            DecisionBoundary.VALIDATION_EXECUTION,
            DecisionBoundary.FILE_MUTATION,
            DecisionBoundary.TASK_BOARD_MUTATION,
            DecisionBoundary.COMMAND_INVOCATION,
            DecisionBoundary.TOOL_INVOCATION,
            DecisionBoundary.COMMIT,
            DecisionBoundary.MERGE,
        }


# Compatibility spelling used by early integration callers.
DecisionRuntimeBoundary = DecisionBoundary


class DecisionOutcome(str, Enum):
    OFF = "off"
    SHADOW_ALLOWED = "shadow_allowed"
    SHADOW_WOULD_BLOCK = "shadow_would_block"
    ADMITTED = "admitted"
    DENIED = "denied"
    CANCELLED = "cancelled"
    EFFECT_MISMATCH = "effect_mismatch"
    COMPLETION_ADMITTED = "completion_admitted"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "to_record") and callable(value.to_record):
        return _plain(value.to_record())
    return value


def _text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise DecisionRuntimeConfigurationError(f"{name} must not be empty")
    if "\x00" in result:
        raise DecisionRuntimeConfigurationError(f"{name} must not contain NUL")
    return result


def _cancelled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if callable(value):
        return bool(value())
    checker = getattr(value, "is_set", None)
    if callable(checker):
        return bool(checker())
    raise TypeError("cancellation must be a boolean, predicate, event, or None")


def _boundary(value: DecisionBoundary | str) -> DecisionBoundary:
    if isinstance(value, DecisionBoundary):
        return value
    try:
        return DecisionBoundary(str(value))
    except ValueError as exc:
        raise DecisionRuntimeConfigurationError(
            f"unknown decision boundary {value!r}"
        ) from exc


_CHANGE_SCOPE_KINDS: Final[Mapping[SemanticChangeKind, str]] = MappingProxyType(
    {
        SemanticChangeKind.WORKTREE: "program_snapshot",
        SemanticChangeKind.AST: "ast_edge",
        SemanticChangeKind.EFFECT: "effect",
        SemanticChangeKind.INTENT_IR: "ir_root",
        SemanticChangeKind.LEGAL_IR: "ir_root",
        SemanticChangeKind.SECURITY_IR: "ir_root",
        SemanticChangeKind.POLICY: "policy",
        SemanticChangeKind.TOOL_CATALOG: "tool_operation",
        SemanticChangeKind.CAPABILITY: "assumption",
        SemanticChangeKind.PROOF: "premise",
        SemanticChangeKind.MONITOR: "assumption",
        SemanticChangeKind.LEASE: "execution_permit",
        SemanticChangeKind.OBSERVED_EFFECT: "effect",
    }
)

_CHANGE_ROOT_KEYS: Final[Mapping[SemanticChangeKind, tuple[str, ...]]] = (
    MappingProxyType(
        {
            SemanticChangeKind.WORKTREE: ("dirty_worktree", "worktree"),
            SemanticChangeKind.AST: ("program", "ast"),
            SemanticChangeKind.EFFECT: ("program", "effect"),
            SemanticChangeKind.INTENT_IR: ("intent_ir",),
            SemanticChangeKind.LEGAL_IR: ("legal_ir",),
            SemanticChangeKind.SECURITY_IR: ("security_ir",),
            SemanticChangeKind.POLICY: ("policy",),
            SemanticChangeKind.TOOL_CATALOG: ("tool_catalog",),
            SemanticChangeKind.CAPABILITY: ("capability",),
            SemanticChangeKind.PROOF: ("proof",),
            SemanticChangeKind.MONITOR: ("monitor",),
            SemanticChangeKind.LEASE: ("lease",),
            SemanticChangeKind.OBSERVED_EFFECT: ("program", "observed_effect"),
        }
    )
)


def canonical_dependency_change(
    kind: SemanticChangeKind | str,
    *,
    subject_id: str,
    previous_root_id: str,
    current_root_id: str,
    scope_value: str = "",
    scope_kind: str = "",
    repository_id: str = "",
    tree_id: str = "",
    semantic_dependency_ids: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> SemanticChange:
    """Create the canonical proof-scope event for any authority input change."""

    selected = (
        kind
        if isinstance(kind, SemanticChangeKind)
        else SemanticChangeKind(str(kind))
    )
    return SemanticChange(
        kind=selected,
        subject_id=subject_id,
        previous_root_id=previous_root_id,
        current_root_id=current_root_id,
        scope_kind=scope_kind or _CHANGE_SCOPE_KINDS[selected],
        scope_value=scope_value or subject_id,
        repository_id=repository_id,
        tree_id=tree_id,
        semantic_dependency_ids=tuple(semantic_dependency_ids),
        metadata=metadata or {},
    )


DependencyChangeEvent = SemanticChange
CanonicalDependencyChange = SemanticChange


@dataclass(frozen=True)
class DecisionRuntimeConfig:
    """Exact, transport-neutral runtime inputs.

    In enforced mode the configured semantic roots are the only roots a
    decision may use.  Generic prompt policy and edit scope live here and are
    therefore included in ``config_id``; callers should additionally place
    them in the policy root and action/effect scope of each DecisionRequest.
    """

    mode: DecisionRuntimeMode | str = DecisionRuntimeMode.OFF
    semantic_roots: tuple[SemanticRoot, ...] = ()
    applicability_facts: tuple[ApplicabilityFact, ...] = ()
    generic_prompt_policy: tuple[str, ...] = ()
    allowed_edit_paths: tuple[str, ...] = ()
    protected_edit_paths: tuple[str, ...] = ()
    caller: str = DEFAULT_RUNTIME_CALLER
    policy_id: str = DEFAULT_RUNTIME_POLICY_ID
    policy_revision: str = DEFAULT_RUNTIME_POLICY_REVISION
    permit_ttl_ms: int = 30_000
    deterministic_degradation: bool = True

    def __post_init__(self) -> None:
        try:
            mode = (
                self.mode
                if isinstance(self.mode, DecisionRuntimeMode)
                else DecisionRuntimeMode(str(self.mode))
            )
        except ValueError as exc:
            raise DecisionRuntimeConfigurationError(
                f"unknown runtime mode {self.mode!r}"
            ) from exc
        object.__setattr__(self, "mode", mode)
        roots = tuple(
            item if isinstance(item, SemanticRoot) else SemanticRoot.from_dict(item)
            for item in self.semantic_roots
        )
        roots = tuple(sorted(roots, key=lambda item: item.kind.value))
        if len({item.kind for item in roots}) != len(roots):
            raise DecisionRuntimeConfigurationError(
                "runtime semantic roots contain duplicate roles"
            )
        if mode is DecisionRuntimeMode.ENFORCE and {
            item.kind for item in roots
        } != set(MANDATORY_SEMANTIC_ROOT_KINDS):
            raise DecisionRuntimeConfigurationError(
                "enforced runtime requires the exact mandatory semantic roots"
            )
        object.__setattr__(self, "semantic_roots", roots)
        facts = tuple(
            item
            if isinstance(item, ApplicabilityFact)
            else ApplicabilityFact.from_dict(item)
            for item in self.applicability_facts
        )
        facts = tuple(sorted(facts, key=lambda item: item.fact_id))
        if len({item.fact_id for item in facts}) != len(facts):
            raise DecisionRuntimeConfigurationError(
                "runtime applicability facts contain duplicate IDs"
            )
        if mode is DecisionRuntimeMode.ENFORCE and not facts:
            raise DecisionRuntimeConfigurationError(
                "enforced runtime requires exact applicability facts"
            )
        object.__setattr__(self, "applicability_facts", facts)
        for name in (
            "generic_prompt_policy",
            "allowed_edit_paths",
            "protected_edit_paths",
        ):
            values = tuple(str(item).strip() for item in getattr(self, name))
            if any(not item for item in values):
                raise DecisionRuntimeConfigurationError(
                    f"{name} contains an empty value"
                )
            if len(values) != len(set(values)):
                raise DecisionRuntimeConfigurationError(
                    f"{name} contains duplicates"
                )
            object.__setattr__(self, name, tuple(sorted(values)))
        for path in (*self.allowed_edit_paths, *self.protected_edit_paths):
            parsed = PurePosixPath(path)
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or "\\" in path
                or "\x00" in path
                or path in {".", ""}
            ):
                raise DecisionRuntimeConfigurationError(
                    f"edit scope path must be repository-relative: {path!r}"
                )
        overlap = set(self.allowed_edit_paths).intersection(
            self.protected_edit_paths
        )
        if overlap:
            raise DecisionRuntimeConfigurationError(
                "protected edit paths cannot also be allowed: "
                + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "caller", _text(self.caller, "caller"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        if (
            isinstance(self.permit_ttl_ms, bool)
            or not isinstance(self.permit_ttl_ms, int)
            or self.permit_ttl_ms < 1
            or self.permit_ttl_ms > 300_000
        ):
            raise DecisionRuntimeConfigurationError(
                "permit_ttl_ms must be between 1 and 300000"
            )
        if not isinstance(self.deterministic_degradation, bool):
            raise DecisionRuntimeConfigurationError(
                "deterministic_degradation must be boolean"
            )

    @property
    def config_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DECISION_RUNTIME_CONFIG_SCHEMA,
            "version": DECISION_RUNTIME_VERSION,
            "mode": self.mode.value,
            "semantic_roots": [item.to_dict() for item in self.semantic_roots],
            "applicability_facts": [
                item.to_dict() for item in self.applicability_facts
            ],
            "generic_prompt_policy": list(self.generic_prompt_policy),
            "allowed_edit_paths": list(self.allowed_edit_paths),
            "protected_edit_paths": list(self.protected_edit_paths),
            "caller": self.caller,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "permit_ttl_ms": self.permit_ttl_ms,
            "deterministic_degradation": self.deterministic_degradation,
        }
        if include_identity:
            payload["config_id"] = self.config_id
        return payload

    to_record = to_dict

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionRuntimeConfig":
        if not isinstance(value, Mapping):
            raise DecisionRuntimeConfigurationError(
                "decision runtime config must be an object"
            )
        if value.get("schema") != DECISION_RUNTIME_CONFIG_SCHEMA:
            raise DecisionRuntimeConfigurationError(
                "unsupported decision runtime config schema"
            )
        if value.get("version") != DECISION_RUNTIME_VERSION:
            raise DecisionRuntimeConfigurationError(
                "unsupported decision runtime config version"
            )
        allowed = {
            "schema",
            "version",
            "config_id",
            "mode",
            "semantic_roots",
            "applicability_facts",
            "generic_prompt_policy",
            "allowed_edit_paths",
            "protected_edit_paths",
            "caller",
            "policy_id",
            "policy_revision",
            "permit_ttl_ms",
            "deterministic_degradation",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise DecisionRuntimeConfigurationError(
                "runtime config contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        result = cls(
            mode=value.get("mode", DecisionRuntimeMode.OFF.value),
            semantic_roots=tuple(value.get("semantic_roots", ())),
            applicability_facts=tuple(value.get("applicability_facts", ())),
            generic_prompt_policy=tuple(value.get("generic_prompt_policy", ())),
            allowed_edit_paths=tuple(value.get("allowed_edit_paths", ())),
            protected_edit_paths=tuple(value.get("protected_edit_paths", ())),
            caller=value.get("caller", DEFAULT_RUNTIME_CALLER),
            policy_id=value.get("policy_id", DEFAULT_RUNTIME_POLICY_ID),
            policy_revision=value.get(
                "policy_revision", DEFAULT_RUNTIME_POLICY_REVISION
            ),
            permit_ttl_ms=value.get("permit_ttl_ms", 30_000),
            deterministic_degradation=value.get(
                "deterministic_degradation", True
            ),
        )
        claimed = value.get("config_id")
        if claimed is not None and claimed != result.config_id:
            raise DecisionRuntimeConfigurationError(
                "decision runtime config identity mismatch"
            )
        return result

    @classmethod
    def from_json(cls, value: str) -> "DecisionRuntimeConfig":
        import json

        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DecisionRuntimeConfigurationError(
                "decision runtime config JSON is malformed"
            ) from exc
        return cls.from_dict(decoded)

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True)
class DecisionRuntimeInput:
    """Already compiled provider-free inputs for one runtime decision."""

    boundary: DecisionBoundary | str
    decision_request: DecisionRequest
    graph: Any = None
    retrieval_receipt: Any = None
    context_compilation: Any = None
    admission_request: PlanAdmissionRequest | None = None
    admission_receipt: PlanAdmissionReceipt | None = None
    evidence_receipts: tuple[ExecutionEvidence, ...] = ()
    acceptance: Any = None
    validation: Any = None
    failure_behavior: Any = None
    completion_evidence: Mapping[str, Any] | None = None
    prior_decision_request_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary", _boundary(self.boundary))
        if not isinstance(self.decision_request, DecisionRequest):
            raise DecisionRuntimeConfigurationError(
                "runtime input requires a DecisionRequest"
            )
        if self.admission_request is not None and not isinstance(
            self.admission_request, PlanAdmissionRequest
        ):
            raise DecisionRuntimeConfigurationError(
                "admission_request must be a PlanAdmissionRequest"
            )
        if self.admission_receipt is not None and not isinstance(
            self.admission_receipt, PlanAdmissionReceipt
        ):
            raise DecisionRuntimeConfigurationError(
                "admission_receipt must be a PlanAdmissionReceipt"
            )
        evidence = tuple(
            item
            if isinstance(item, ExecutionEvidence)
            else ExecutionEvidence.from_dict(item)
            for item in self.evidence_receipts
        )
        object.__setattr__(self, "evidence_receipts", evidence)
        if self.completion_evidence is not None:
            object.__setattr__(
                self,
                "completion_evidence",
                MappingProxyType(dict(_plain(self.completion_evidence))),
            )


@dataclass(frozen=True)
class DecisionRuntimeReceipt:
    runtime_id: str
    config_id: str
    boundary: DecisionBoundary
    mode: DecisionRuntimeMode
    outcome: DecisionOutcome
    decision_request_id: str = ""
    context_witness_id: str = ""
    admission_receipt_id: str = ""
    permit_id: str = ""
    reason_codes: tuple[str, ...] = ()
    authoritative: bool = False
    completion_authoritative: bool = False
    sequence: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary", _boundary(self.boundary))
        object.__setattr__(
            self,
            "mode",
            self.mode
            if isinstance(self.mode, DecisionRuntimeMode)
            else DecisionRuntimeMode(str(self.mode)),
        )
        object.__setattr__(
            self,
            "outcome",
            self.outcome
            if isinstance(self.outcome, DecisionOutcome)
            else DecisionOutcome(str(self.outcome)),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({str(item) for item in self.reason_codes if str(item)})),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(_plain(self.metadata)))
        )
        if self.mode is not DecisionRuntimeMode.ENFORCE and (
            self.authoritative or self.completion_authoritative
        ):
            raise DecisionRuntimeConfigurationError(
                "off/shadow receipts cannot claim authority"
            )
        if self.completion_authoritative and self.boundary is not DecisionBoundary.COMPLETION:
            raise DecisionRuntimeConfigurationError(
                "completion authority is valid only at the completion boundary"
            )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def admitted(self) -> bool:
        return self.outcome in {
            DecisionOutcome.ADMITTED,
            DecisionOutcome.COMPLETION_ADMITTED,
            DecisionOutcome.OFF,
            DecisionOutcome.SHADOW_ALLOWED,
            DecisionOutcome.SHADOW_WOULD_BLOCK,
        }

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DECISION_RUNTIME_RECEIPT_SCHEMA,
            "version": DECISION_RUNTIME_VERSION,
            "runtime_id": self.runtime_id,
            "config_id": self.config_id,
            "boundary": self.boundary.value,
            "mode": self.mode.value,
            "outcome": self.outcome.value,
            "decision_request_id": self.decision_request_id,
            "context_witness_id": self.context_witness_id,
            "admission_receipt_id": self.admission_receipt_id,
            "permit_id": self.permit_id,
            "reason_codes": list(self.reason_codes),
            "authoritative": self.authoritative,
            "completion_authoritative": self.completion_authoritative,
            "sequence": self.sequence,
            "metadata": _plain(self.metadata),
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    to_record = to_dict

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "DecisionRuntimeReceipt":
        if not isinstance(value, Mapping):
            raise DecisionRuntimeConfigurationError(
                "decision runtime receipt must be an object"
            )
        allowed = {
            "schema",
            "version",
            "receipt_id",
            "runtime_id",
            "config_id",
            "boundary",
            "mode",
            "outcome",
            "decision_request_id",
            "context_witness_id",
            "admission_receipt_id",
            "permit_id",
            "reason_codes",
            "authoritative",
            "completion_authoritative",
            "sequence",
            "metadata",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise DecisionRuntimeConfigurationError(
                "runtime receipt contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if (
            value.get("schema") != DECISION_RUNTIME_RECEIPT_SCHEMA
            or value.get("version") != DECISION_RUNTIME_VERSION
        ):
            raise DecisionRuntimeConfigurationError(
                "unsupported decision runtime receipt schema"
            )
        result = cls(
            runtime_id=value.get("runtime_id", ""),
            config_id=value.get("config_id", ""),
            boundary=value.get("boundary", ""),
            mode=value.get("mode", ""),
            outcome=value.get("outcome", ""),
            decision_request_id=value.get("decision_request_id", ""),
            context_witness_id=value.get("context_witness_id", ""),
            admission_receipt_id=value.get("admission_receipt_id", ""),
            permit_id=value.get("permit_id", ""),
            reason_codes=tuple(value.get("reason_codes", ())),
            authoritative=value.get("authoritative", False),
            completion_authoritative=value.get(
                "completion_authoritative", False
            ),
            sequence=value.get("sequence", 0),
            metadata=value.get("metadata", {}),
        )
        claimed = value.get("receipt_id")
        if claimed is not None and claimed != result.receipt_id:
            raise DecisionRuntimeConfigurationError(
                "decision runtime receipt identity mismatch"
            )
        return result


@dataclass(frozen=True)
class DecisionRuntimeDecision:
    receipt: DecisionRuntimeReceipt
    decision_request: DecisionRequest | None = None
    context_compilation: Any = None
    admission_receipt: PlanAdmissionReceipt | None = None
    permit: ExecutionPermit | None = None

    @property
    def admitted(self) -> bool:
        return self.receipt.admitted

    @property
    def authoritative(self) -> bool:
        return self.receipt.authoritative

    @property
    def decision_id(self) -> str:
        return self.receipt.receipt_id


@dataclass(frozen=True)
class ObservedEffect:
    effect_id: str
    kind: str
    authority: str
    target_ids: tuple[str, ...]
    repository_paths: tuple[str, ...]
    description: str
    verification: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _text(self.effect_id, "effect_id"))
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(self, "authority", _text(self.authority, "authority"))
        object.__setattr__(
            self,
            "target_ids",
            tuple(sorted({_text(item, "target_id") for item in self.target_ids})),
        )
        object.__setattr__(
            self,
            "repository_paths",
            tuple(
                sorted(
                    {
                        _text(item, "repository_path")
                        for item in self.repository_paths
                    }
                )
            ),
        )
        object.__setattr__(
            self, "description", _text(self.description, "description")
        )
        object.__setattr__(
            self, "verification", MappingProxyType(dict(_plain(self.verification)))
        )

    @classmethod
    def from_effect(cls, value: EffectEnvelope) -> "ObservedEffect":
        return cls(
            effect_id=value.effect_id,
            kind=value.kind.value,
            authority=value.authority.value,
            target_ids=value.target_ids,
            repository_paths=value.repository_paths,
            description=value.description,
            verification=value.verification,
        )

    @classmethod
    def from_value(cls, value: Any) -> "ObservedEffect":
        if isinstance(value, cls):
            return value
        if isinstance(value, EffectEnvelope):
            return cls.from_effect(value)
        if not isinstance(value, Mapping):
            raise DecisionRuntimeConfigurationError(
                "observed effect must be an EffectEnvelope or mapping"
            )
        return cls(
            effect_id=value.get("effect_id", ""),
            kind=str(getattr(value.get("kind"), "value", value.get("kind", ""))),
            authority=str(
                getattr(value.get("authority"), "value", value.get("authority", ""))
            ),
            target_ids=tuple(value.get("target_ids", ())),
            repository_paths=tuple(value.get("repository_paths", ())),
            description=value.get("description", ""),
            verification=value.get("verification", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "kind": self.kind,
            "authority": self.authority,
            "target_ids": list(self.target_ids),
            "repository_paths": list(self.repository_paths),
            "description": self.description,
            "verification": _plain(self.verification),
        }


@dataclass(frozen=True)
class EffectObservationReceipt:
    runtime_id: str
    decision_receipt_id: str
    permit_use_receipt_id: str
    expected_effects: tuple[ObservedEffect, ...]
    observed_effects: tuple[ObservedEffect, ...]
    matched: bool
    reason_codes: tuple[str, ...]
    pre_root_id: str = ""
    post_root_id: str = ""

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": EFFECT_OBSERVATION_SCHEMA,
            "version": DECISION_RUNTIME_VERSION,
            "runtime_id": self.runtime_id,
            "decision_receipt_id": self.decision_receipt_id,
            "permit_use_receipt_id": self.permit_use_receipt_id,
            "expected_effects": [
                item.to_dict() for item in self.expected_effects
            ],
            "observed_effects": [
                item.to_dict() for item in self.observed_effects
            ],
            "matched": self.matched,
            "reason_codes": list(self.reason_codes),
            "pre_root_id": self.pre_root_id,
            "post_root_id": self.post_root_id,
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    to_record = to_dict

    def semantic_changes(self) -> tuple[SemanticChange, ...]:
        """Project mismatches/root movement into canonical invalidation inputs."""

        if self.matched and (
            not self.pre_root_id
            or not self.post_root_id
            or self.pre_root_id == self.post_root_id
        ):
            return ()
        previous = self.pre_root_id or content_identity(
            [item.to_dict() for item in self.expected_effects]
        )
        current = self.post_root_id or content_identity(
            [item.to_dict() for item in self.observed_effects]
        )
        subject = (
            self.decision_receipt_id
            or self.receipt_id
        )
        return (
            canonical_dependency_change(
                SemanticChangeKind.OBSERVED_EFFECT,
                subject_id=subject,
                previous_root_id=previous,
                current_root_id=current,
                scope_value=subject,
                metadata={
                    "effect_observation_receipt_id": self.receipt_id,
                    "reason_codes": list(self.reason_codes),
                },
            ),
        )


@dataclass(frozen=True)
class DecisionExecutionResult:
    value: Any
    decision: DecisionRuntimeDecision
    permit_use: PermitUseReceipt | None
    effect_observation: EffectObservationReceipt | None


@dataclass(frozen=True)
class RuntimeInvalidationReceipt:
    """Exact dependency-local invalidation and minimum revalidation closure."""

    runtime_id: str
    change_ids: tuple[str, ...]
    previous_roots: Mapping[str, str]
    current_roots: Mapping[str, str]
    event_cursor: EventCursor | None = None
    proof_index_id: str = ""
    cas_transaction_ids: tuple[str, ...] = ()
    retrieval_ids: tuple[str, ...] = ()
    context_ids: tuple[str, ...] = ()
    plan_ids: tuple[str, ...] = ()
    plan_suffix_ids: tuple[str, ...] = ()
    permit_ids: tuple[str, ...] = ()
    proof_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()
    cache_ids: tuple[str, ...] = ()
    merge_receipt_ids: tuple[str, ...] = ()
    completion_receipt_ids: tuple[str, ...] = ()
    preserved_artifact_ids: tuple[str, ...] = ()
    recomputed_artifact_ids: tuple[str, ...] = ()
    fencing_epoch: int = 0
    authoritative: bool = False
    reason_codes: tuple[str, ...] = ()
    requirement_id: str = INCREMENTAL_REVALIDATION_REQUIREMENT_ID

    def __post_init__(self) -> None:
        if self.requirement_id != INCREMENTAL_REVALIDATION_REQUIREMENT_ID:
            raise DecisionRuntimeConfigurationError(
                "runtime invalidation receipt requirement mismatch"
            )
        object.__setattr__(self, "runtime_id", _text(self.runtime_id, "runtime_id"))
        object.__setattr__(
            self,
            "proof_index_id",
            _text(self.proof_index_id, "proof_index_id"),
        )
        for name in (
            "change_ids",
            "cas_transaction_ids",
            "retrieval_ids",
            "context_ids",
            "plan_ids",
            "plan_suffix_ids",
            "permit_ids",
            "proof_ids",
            "obligation_ids",
            "validation_ids",
            "cache_ids",
            "merge_receipt_ids",
            "completion_receipt_ids",
            "preserved_artifact_ids",
            "recomputed_artifact_ids",
            "reason_codes",
        ):
            values = tuple(
                sorted({str(item) for item in getattr(self, name) if str(item)})
            )
            if len(values) != len(tuple(getattr(self, name))):
                raise DecisionRuntimeConfigurationError(
                    f"{name} contains empty or duplicate identities"
                )
            object.__setattr__(self, name, values)
        if not self.change_ids:
            raise DecisionRuntimeConfigurationError(
                "invalidation receipt requires a semantic change"
            )
        previous = {
            _text(str(key), "previous root kind"):
            _text(str(item), "previous root identity")
            for key, item in sorted(self.previous_roots.items())
        }
        current = {
            _text(str(key), "current root kind"):
            _text(str(item), "current root identity")
            for key, item in sorted(self.current_roots.items())
        }
        if not previous or not current or previous == current:
            raise DecisionRuntimeConfigurationError(
                "invalidation receipt must bind a changed semantic root set"
            )
        object.__setattr__(
            self, "previous_roots", MappingProxyType(previous)
        )
        object.__setattr__(
            self, "current_roots", MappingProxyType(current)
        )
        if (
            isinstance(self.fencing_epoch, bool)
            or not isinstance(self.fencing_epoch, int)
            or self.fencing_epoch < 0
        ):
            raise DecisionRuntimeConfigurationError(
                "fencing_epoch must be a nonnegative integer"
            )
        if self.event_cursor is not None and not isinstance(
            self.event_cursor, EventCursor
        ):
            raise DecisionRuntimeConfigurationError(
                "event_cursor must be an EventCursor or None"
            )
        if not set(self.plan_suffix_ids).issubset(self.plan_ids):
            raise DecisionRuntimeConfigurationError(
                "plan suffix contains plans outside the invalidation closure"
            )
        if set(self.invalidated_artifact_ids).intersection(
            self.preserved_artifact_ids
        ):
            raise DecisionRuntimeConfigurationError(
                "invalidation receipt preserves an invalidated artifact"
            )
        if self.authoritative and self.reason_codes:
            raise DecisionRuntimeConfigurationError(
                "an authoritative revalidation receipt cannot contain failures"
            )

    @property
    def invalidated_artifact_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.retrieval_ids,
                    *self.context_ids,
                    *self.plan_ids,
                    *self.permit_ids,
                    *self.proof_ids,
                    *self.obligation_ids,
                    *self.validation_ids,
                    *self.cache_ids,
                    *self.merge_receipt_ids,
                    *self.completion_receipt_ids,
                }
            )
        )

    @property
    def minimum_reproof_ids(self) -> tuple[str, ...]:
        return tuple(sorted({*self.obligation_ids, *self.proof_ids}))

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": RUNTIME_INVALIDATION_RECEIPT_SCHEMA,
            "version": 1,
            "requirement_id": self.requirement_id,
            "runtime_id": self.runtime_id,
            "change_ids": list(self.change_ids),
            "previous_roots": dict(self.previous_roots),
            "current_roots": dict(self.current_roots),
            "event_cursor": (
                self.event_cursor.to_record()
                if self.event_cursor is not None
                else None
            ),
            "proof_index_id": self.proof_index_id,
            "cas_transaction_ids": list(self.cas_transaction_ids),
            "retrieval_ids": list(self.retrieval_ids),
            "context_ids": list(self.context_ids),
            "plan_ids": list(self.plan_ids),
            "plan_suffix_ids": list(self.plan_suffix_ids),
            "permit_ids": list(self.permit_ids),
            "proof_ids": list(self.proof_ids),
            "obligation_ids": list(self.obligation_ids),
            "validation_ids": list(self.validation_ids),
            "cache_ids": list(self.cache_ids),
            "merge_receipt_ids": list(self.merge_receipt_ids),
            "completion_receipt_ids": list(
                self.completion_receipt_ids
            ),
            "preserved_artifact_ids": list(self.preserved_artifact_ids),
            "recomputed_artifact_ids": list(
                self.recomputed_artifact_ids
            ),
            "fencing_epoch": self.fencing_epoch,
            "authoritative": self.authoritative,
            "reason_codes": list(self.reason_codes),
            "invalidated_artifact_ids": list(
                self.invalidated_artifact_ids
            ),
            "minimum_reproof_ids": list(self.minimum_reproof_ids),
        }
        if include_identity:
            value["receipt_id"] = self.receipt_id
        return value

    to_record = to_dict

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "RuntimeInvalidationReceipt":
        if (
            value.get("schema") != RUNTIME_INVALIDATION_RECEIPT_SCHEMA
            or value.get("version") != 1
        ):
            raise DecisionRuntimeConfigurationError(
                "unsupported runtime invalidation receipt schema"
            )
        cursor_value = value.get("event_cursor")
        result = cls(
            runtime_id=str(value.get("runtime_id") or ""),
            change_ids=tuple(value.get("change_ids") or ()),
            previous_roots=value.get("previous_roots") or {},
            current_roots=value.get("current_roots") or {},
            event_cursor=(
                EventCursor.from_dict(cursor_value)
                if isinstance(cursor_value, Mapping)
                else None
            ),
            proof_index_id=str(value.get("proof_index_id") or ""),
            cas_transaction_ids=tuple(
                value.get("cas_transaction_ids") or ()
            ),
            retrieval_ids=tuple(value.get("retrieval_ids") or ()),
            context_ids=tuple(value.get("context_ids") or ()),
            plan_ids=tuple(value.get("plan_ids") or ()),
            plan_suffix_ids=tuple(value.get("plan_suffix_ids") or ()),
            permit_ids=tuple(value.get("permit_ids") or ()),
            proof_ids=tuple(value.get("proof_ids") or ()),
            obligation_ids=tuple(value.get("obligation_ids") or ()),
            validation_ids=tuple(value.get("validation_ids") or ()),
            cache_ids=tuple(value.get("cache_ids") or ()),
            merge_receipt_ids=tuple(
                value.get("merge_receipt_ids") or ()
            ),
            completion_receipt_ids=tuple(
                value.get("completion_receipt_ids") or ()
            ),
            preserved_artifact_ids=tuple(
                value.get("preserved_artifact_ids") or ()
            ),
            recomputed_artifact_ids=tuple(
                value.get("recomputed_artifact_ids") or ()
            ),
            fencing_epoch=value.get("fencing_epoch", 0),
            authoritative=bool(value.get("authoritative", False)),
            reason_codes=tuple(value.get("reason_codes") or ()),
            requirement_id=str(
                value.get("requirement_id")
                or INCREMENTAL_REVALIDATION_REQUIREMENT_ID
            ),
        )
        if value.get("receipt_id") not in (None, result.receipt_id):
            raise DecisionRuntimeConfigurationError(
                "runtime invalidation receipt identity mismatch"
            )
        return result


@dataclass(frozen=True)
class RuntimeInvalidationResult:
    """Updated proof index plus its exact invalidation/revalidation receipt."""

    proof_index: Any
    receipt: RuntimeInvalidationReceipt
    recomputed: Mapping[str, Any] = field(default_factory=dict)

    @property
    def invalidation_receipt(self) -> RuntimeInvalidationReceipt:
        return self.receipt

    def __iter__(self):
        yield self.proof_index
        yield self.receipt


DecisionResolver = Callable[
    [DecisionBoundary, Mapping[str, Any]], DecisionRuntimeInput | DecisionRuntimeDecision
]


class DecisionRuntime:
    """One decision, permit, effect, and completion spine for live paths."""

    def __init__(
        self,
        config: DecisionRuntimeConfig | Mapping[str, Any] | None = None,
        *,
        resolver: DecisionResolver | None = None,
        issuer: ExecutionPermitIssuer | None = None,
        cancellation: Any = None,
        clock_ms: Callable[[], int] | None = None,
        runtime_id: str | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, DecisionRuntimeConfig)
            else DecisionRuntimeConfig.from_dict(config)
            if config is not None
            else DecisionRuntimeConfig()
        )
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._resolver = resolver
        self._cancellation = cancellation
        self._issuer = issuer or ExecutionPermitIssuer(clock_ms=self._clock_ms)
        self._verifier = self._issuer.verifier()
        self._lock = threading.RLock()
        self._sequence = 0
        self._receipts: list[DecisionRuntimeReceipt] = []
        self._effect_receipts: list[EffectObservationReceipt] = []
        self._invalidation_receipts: list[RuntimeInvalidationReceipt] = []
        self._invalidated_permit_ids: set[str] = set()
        self._invalidated_decision_receipt_ids: set[str] = set()
        self._issued_permits: dict[str, ExecutionPermit] = {}
        self._seen_change_ids: set[str] = set()
        self._invalidation_quarantine_reasons: set[str] = set()
        self._event_cursor: EventCursor | None = None
        self._fencing_epoch = 0
        self._semantic_root_state: dict[str, str] = {
            item.kind.value: item.artifact.cid_v1
            for item in self.config.semantic_roots
        }
        self.runtime_id = runtime_id or content_identity(
            {
                "kind": "decision-runtime",
                "config_id": self.config.config_id,
                "caller": self.config.caller,
                "policy_id": self.config.policy_id,
            }
        )

    @property
    def receipts(self) -> tuple[DecisionRuntimeReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    @property
    def effect_receipts(self) -> tuple[EffectObservationReceipt, ...]:
        with self._lock:
            return tuple(self._effect_receipts)

    @property
    def invalidation_receipts(self) -> tuple[RuntimeInvalidationReceipt, ...]:
        with self._lock:
            return tuple(self._invalidation_receipts)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "runtime_id": self.runtime_id,
                "config": self.config.to_dict(),
                "decision_count": len(self._receipts),
                "effect_observation_count": len(self._effect_receipts),
                "invalidation_count": len(self._invalidation_receipts),
                "invalidated_permit_count": len(
                    self._invalidated_permit_ids
                ),
                "invalidation_quarantined": bool(
                    self._invalidation_quarantine_reasons
                ),
                "invalidation_quarantine_reasons": sorted(
                    self._invalidation_quarantine_reasons
                ),
                "semantic_roots": dict(self._semantic_root_state),
                "event_cursor": (
                    self._event_cursor.to_record()
                    if self._event_cursor is not None
                    else None
                ),
                "fencing_epoch": self._fencing_epoch,
                "optional_providers_loaded": False,
                "processes_started": False,
            }

    discover = status

    def _check_cancelled(self, stage: str) -> None:
        if _cancelled(self._cancellation):
            raise DecisionRuntimeCancelled(
                ("cancelled", f"cancelled_{stage}"),
                f"decision runtime cancelled at {stage}",
            )

    def _record(
        self,
        *,
        boundary: DecisionBoundary,
        outcome: DecisionOutcome,
        decision_request_id: str = "",
        context_witness_id: str = "",
        admission_receipt_id: str = "",
        permit_id: str = "",
        reason_codes: Sequence[str] = (),
        authoritative: bool = False,
        completion_authoritative: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> DecisionRuntimeReceipt:
        with self._lock:
            self._sequence += 1
            receipt = DecisionRuntimeReceipt(
                runtime_id=self.runtime_id,
                config_id=self.config.config_id,
                boundary=boundary,
                mode=self.config.mode,
                outcome=outcome,
                decision_request_id=decision_request_id,
                context_witness_id=context_witness_id,
                admission_receipt_id=admission_receipt_id,
                permit_id=permit_id,
                reason_codes=tuple(reason_codes),
                authoritative=authoritative,
                completion_authoritative=completion_authoritative,
                sequence=self._sequence,
                metadata=metadata or {},
            )
            self._receipts.append(receipt)
            return receipt

    @staticmethod
    def _witness(compilation: Any) -> ContextCompletenessWitness | None:
        if compilation is None:
            return None
        witness = getattr(compilation, "witness", None)
        if witness is None:
            witness = getattr(compilation, "completeness_witness", None)
        return witness if isinstance(witness, ContextCompletenessWitness) else None

    def _validate_bindings(self, request: DecisionRequest) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.config.semantic_roots and request.semantic_roots != self.config.semantic_roots:
            reasons.append("semantic_roots_mismatch")
        if (
            self.config.applicability_facts
            and request.applicability_facts != self.config.applicability_facts
        ):
            reasons.append("applicability_facts_mismatch")
        if request.authority.principal_id == "":
            reasons.append("principal_missing")
        if request.action.authority.value == "mutation":
            if request.authority.lease_id is None:
                reasons.append("lease_missing")
            if request.authority.fencing_epoch is None:
                reasons.append("fencing_epoch_missing")
            if request.authority.idempotency_key is None:
                reasons.append("idempotency_key_missing")
        paths = {
            path
            for target in request.action.targets
            for path in target.repository_paths
        }
        paths.update(
            path
            for effect in request.expected_effects
            for path in effect.repository_paths
        )
        protected = paths.intersection(self.config.protected_edit_paths)
        if protected:
            reasons.append("protected_path_targeted")
        if request.action.authority.value == "mutation":
            if paths != set(self.config.allowed_edit_paths):
                reasons.append("edit_scope_mismatch")
        return tuple(sorted(set(reasons)))

    def compile_context(
        self,
        request: DecisionRequest,
        graph: Any,
        retrieval_receipt: Any,
        *,
        compiler: Any = None,
        acceptance: Any = None,
        validation: Any = None,
        failure_behavior: Any = None,
        artifact_store: Any = None,
        overflow_behavior: Any = "split",
    ) -> Any:
        self._check_cancelled("context")
        if compiler is None:
            from .context_compiler import DecisionContextCompiler
            from .context_contracts import ContextBudget

            compiler = DecisionContextCompiler(
                ContextBudget(
                    max_input_tokens=request.budget.max_input_tokens,
                    reserved_output_tokens=request.budget.max_output_tokens,
                    reserved_tool_tokens=0,
                    max_items=request.budget.max_items,
                    max_item_bytes=request.budget.max_artifact_bytes,
                    max_serialized_bytes=request.budget.max_serialized_bytes,
                    max_depth=request.budget.max_depth,
                    max_text_bytes=request.budget.max_text_bytes,
                )
            )
        result = compiler.compile(
            request,
            graph,
            retrieval_receipt,
            artifact_store=artifact_store,
            acceptance=acceptance,
            validation=validation,
            failure_behavior=failure_behavior,
            overflow_behavior=overflow_behavior,
        )
        self._check_cancelled("context")
        return result

    def retry_context(
        self, compiler: Any, parent: Any, /, **changes: Any
    ) -> Any:
        self._check_cancelled("retry")
        result = compiler.compile_retry(parent, **changes)
        self._check_cancelled("retry")
        return result

    def expand_context(
        self,
        compiler: Any,
        parent: Any,
        request: Any,
        resolver: Any,
        /,
        **current: Any,
    ) -> Any:
        self._check_cancelled("expansion")
        result = compiler.expand_decision_context(
            parent,
            request,
            resolver,
            cancelled=self._cancellation,
            **current,
        )
        self._check_cancelled("expansion")
        return result

    @staticmethod
    def _result_identity(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,) if value else ()
        if isinstance(value, Mapping):
            for name in (
                "artifact_id",
                "receipt_id",
                "content_id",
                "result_id",
            ):
                if value.get(name):
                    return (str(value[name]),)
            return tuple(
                sorted(
                    {
                        item
                        for nested in value.values()
                        for item in DecisionRuntime._result_identity(nested)
                    }
                )
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return tuple(
                sorted(
                    {
                        item
                        for nested in value
                        for item in DecisionRuntime._result_identity(nested)
                    }
                )
            )
        for name in ("artifact_id", "receipt_id", "content_id", "result_id"):
            selected = getattr(value, name, "")
            if selected:
                return (str(selected),)
        return ()

    @staticmethod
    def _runtime_artifact_family(record: Any) -> str:
        if record is None:
            return ""
        identity = getattr(record, "identity", None)
        text = " ".join(
            (
                str(getattr(identity, "namespace", "")),
                str(getattr(identity, "artifact_kind", "")),
                str(getattr(identity, "payload_schema", "")),
            )
        ).casefold()
        payload = getattr(record, "payload", None)
        if isinstance(payload, Mapping):
            text += " " + " ".join(
                str(payload.get(name) or "")
                for name in (
                    "kind",
                    "artifact_kind",
                    "receipt_kind",
                    "cache_kind",
                    "stage",
                )
            ).casefold()
        for family in (
            "completion",
            "retrieval",
            "context",
            "plan",
            "permit",
            "proof",
            "monitor",
            "validation",
            "merge",
            "cache",
        ):
            if family in text:
                return family
        return "cache"

    @staticmethod
    def _replacement_is_current(
        value: Any, current_roots: Mapping[str, str]
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        values = (
            value
            if isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, Mapping))
            else (value,)
        )
        current_values = set(current_roots.values())
        for item in values:
            if isinstance(item, Mapping):
                if item.get("authoritative") is False:
                    reasons.append("replacement_not_authoritative")
                bound = str(
                    item.get("root_id")
                    or item.get("tree_id")
                    or item.get("roots_id")
                    or ""
                )
                if bound and bound not in current_values and bound != content_identity(
                    dict(current_roots)
                ):
                    reasons.append("replacement_root_mismatch")
        return tuple(sorted(set(reasons)))

    def apply_dependency_change(
        self,
        change: SemanticChange | Mapping[str, Any],
        proof_index: Any,
        *,
        cas: Any = None,
        event_cursor: EventCursor | Mapping[str, Any] | str | None = None,
        recompute: Mapping[str, Callable[[tuple[str, ...], SemanticChange, Any], Any]]
        | None = None,
        root_reader: Callable[[], Mapping[str, str]] | None = None,
    ) -> RuntimeInvalidationResult:
        """Invalidate exactly the reverse proof/CAS closure and revalidate it.

        Recompute callbacks use the closed keys ``context``, ``plan``,
        ``proof``, and ``validation`` and are invoked only for non-empty
        affected populations. Independent artifacts are never passed to a
        producer and remain reusable.
        """

        selected = (
            change
            if isinstance(change, SemanticChange)
            else SemanticChange.from_dict(change)
        )
        selected_cursor: EventCursor | None
        if event_cursor is None:
            selected_cursor = None
        elif isinstance(event_cursor, EventCursor):
            selected_cursor = event_cursor
        elif isinstance(event_cursor, str):
            selected_cursor = EventCursor.from_token(event_cursor)
        elif isinstance(event_cursor, Mapping):
            selected_cursor = EventCursor.from_dict(event_cursor)
        else:
            raise DecisionRuntimeConfigurationError(
                "event_cursor must be an EventCursor, record, token, or None"
            )
        with self._lock:
            if selected.change_id in self._seen_change_ids:
                raise DecisionRuntimeDenied(
                    ("duplicate_semantic_change",),
                    "semantic change was already applied",
                )
            if self._event_cursor is not None and selected_cursor is not None:
                if (
                    selected_cursor.stream_id != self._event_cursor.stream_id
                    or selected_cursor.snapshot_id
                    != self._event_cursor.snapshot_id
                    or selected_cursor.position <= self._event_cursor.position
                ):
                    raise DecisionRuntimeDenied(
                        ("event_cursor_reordered",)
                    )
            root_key = selected.subject_id
            known_root = self._semantic_root_state.get(root_key)
            if known_root is None:
                for candidate in _CHANGE_ROOT_KEYS[selected.kind]:
                    known_root = self._semantic_root_state.get(candidate)
                    if known_root is not None:
                        root_key = candidate
                        break
            if (
                known_root is not None
                and selected.previous_root_id != known_root
            ):
                raise DecisionRuntimeDenied(
                    ("missed_or_reordered_semantic_change", "stale_root")
                )
            previous_roots = dict(self._semantic_root_state)
            if root_key not in previous_roots:
                previous_roots[root_key] = selected.previous_root_id
            target_roots = dict(previous_roots)
            target_roots[root_key] = selected.current_root_id
            before_snapshot = (
                {
                    str(key): str(item)
                    for key, item in root_reader().items()
                }
                if root_reader is not None
                else None
            )

            from .proof_scope_index import (
                ProofInputKind,
                ProofScopeIndex,
                ProofScopeKey,
                invalidate_proof_scope_inputs,
            )

            if not isinstance(proof_index, ProofScopeIndex):
                raise DecisionRuntimeConfigurationError(
                    "proof_index must be a verified ProofScopeIndex"
                )
            try:
                key = ProofScopeKey(
                    ProofInputKind(selected.scope_kind),
                    selected.scope_value,
                )
                dependents = proof_index.reverse_dependents(key)
                changed_index = invalidate_proof_scope_inputs(
                    proof_index, (key,)
                )
            except Exception as exc:
                raise DecisionRuntimeDenied(
                    ("corrupt_or_unusable_proof_index", type(exc).__name__)
                ) from exc

            artifacts = {
                item.artifact_id: item for item in proof_index.artifacts
            }
            retrieval_ids: set[str] = set()
            cache_ids = set(dependents.cache_ids)
            for artifact_id in dependents.cache_ids:
                artifact = artifacts.get(artifact_id)
                text = (
                    artifact_id
                    + " "
                    + str(getattr(artifact, "payload", {}))
                ).casefold()
                if "retriev" in text:
                    retrieval_ids.add(artifact_id)
            merge_ids: set[str] = set()
            completion_ids: set[str] = set()
            for artifact_id in dependents.merge_ids:
                artifact = artifacts.get(artifact_id)
                text = (
                    artifact_id
                    + " "
                    + str(getattr(artifact, "payload", {}))
                ).casefold()
                if "completion" in text:
                    completion_ids.add(artifact_id)
                else:
                    merge_ids.add(artifact_id)
            context_ids = set(dependents.context_ids)
            plan_ids = set(dependents.plan_ids)
            permit_ids = set(dependents.permit_ids)
            proof_ids = set(dependents.proof_ids)
            validation_ids = set(dependents.validation_ids)
            obligation_ids = set(dependents.obligation_ids)
            # Indexed receipts are proof evidence rather than retrieval output.
            proof_ids.update(dependents.receipt_ids)
            preserved = set(changed_index.active_artifact_ids)
            cas_transaction_ids: list[str] = []
            if cas is not None:
                dependency_ids = set(selected.semantic_dependency_ids)
                if not dependency_ids:
                    resolver = getattr(cas, "semantic_dependency_ids", None)
                    if callable(resolver):
                        dependency_ids.update(
                            resolver(
                                namespace=selected.scope_kind,
                                key=selected.scope_value,
                                revision=selected.previous_root_id,
                            )
                        )
                        dependency_ids.update(
                            resolver(
                                key=selected.scope_value,
                                digest=selected.previous_root_id,
                            )
                        )
                artifact_roots = tuple(
                    item
                    for item in (
                        *context_ids,
                        *plan_ids,
                        *permit_ids,
                        *proof_ids,
                        *validation_ids,
                        *cache_ids,
                        *merge_ids,
                        *completion_ids,
                    )
                    if str(item).startswith("runtime-artifact:sha256:")
                )
                if dependency_ids or artifact_roots:
                    transaction = cas.invalidate_batch(
                        artifact_ids=artifact_roots,
                        semantic_dependency_ids=tuple(dependency_ids),
                        reason=f"{selected.kind.value}_changed",
                        roots_id=content_identity(target_roots),
                        event_cursor=(
                            selected_cursor.to_token()
                            if selected_cursor is not None
                            else ""
                        ),
                    )
                    cas_transaction_ids.append(transaction.transaction_id)
                    preserved.update(transaction.preserved_artifact_ids)
                    for artifact_id in transaction.invalidated_artifact_ids:
                        record = cas.inspect_artifact(artifact_id)
                        family = self._runtime_artifact_family(record)
                        {
                            "retrieval": retrieval_ids,
                            "context": context_ids,
                            "plan": plan_ids,
                            "permit": permit_ids,
                            "proof": proof_ids,
                            "monitor": proof_ids,
                            "validation": validation_ids,
                            "merge": merge_ids,
                            "completion": completion_ids,
                            "cache": cache_ids,
                        }[family].add(artifact_id)

            # A live permit is dependent only when one of its exact root,
            # evidence, lease, plan, or closure bindings changed.
            for permit_id, permit in self._issued_permits.items():
                evidence_ids = {
                    item.receipt_id for item in permit.evidence_receipts
                }
                if (
                    permit_id in permit_ids
                    or selected.previous_root_id
                    in {
                        permit.repository_tree_id,
                        permit.worktree_root_id,
                        *permit.semantic_roots.values(),
                    }
                    or selected.subject_id
                    in {
                        permit.lease_id,
                        permit.candidate_plan_id,
                        permit.mandatory_closure.closure_id,
                        *evidence_ids,
                    }
                    or evidence_ids.intersection(proof_ids)
                ):
                    permit_ids.add(permit_id)

            plan_suffix_ids = tuple(sorted(plan_ids))
            callbacks = dict(recompute or {})
            recomputed: dict[str, Any] = {}
            recomputed_ids: set[str] = set()
            reasons: list[str] = []
            closure = (
                ("context", tuple(sorted(context_ids))),
                ("plan", plan_suffix_ids),
                (
                    "proof",
                    tuple(sorted({*obligation_ids, *proof_ids})),
                ),
                ("validation", tuple(sorted(validation_ids))),
            )
            for family, identities in closure:
                if not identities:
                    continue
                callback = callbacks.get(family)
                if callback is None:
                    reasons.append(f"{family}_recompute_missing")
                    continue
                value = callback(identities, selected, changed_index)
                recomputed[family] = value
                replacement_ids = self._result_identity(value)
                if not replacement_ids:
                    reasons.append(f"{family}_recompute_empty")
                recomputed_ids.update(replacement_ids)
                reasons.extend(
                    self._replacement_is_current(value, target_roots)
                )
            after_snapshot = (
                {
                    str(key): str(item)
                    for key, item in root_reader().items()
                }
                if root_reader is not None
                else None
            )
            if (
                before_snapshot is not None
                and after_snapshot != before_snapshot
            ):
                reasons.append("root_race")
            if (
                after_snapshot is not None
                and selected.current_root_id
                not in after_snapshot.values()
            ):
                reasons.append("current_root_not_observed")

            self._invalidated_permit_ids.update(permit_ids)
            for item in self._receipts:
                if item.permit_id in permit_ids:
                    self._invalidated_decision_receipt_ids.add(
                        item.receipt_id
                    )
            self._seen_change_ids.add(selected.change_id)
            self._semantic_root_state = target_roots
            if selected_cursor is not None:
                self._event_cursor = selected_cursor
            self._fencing_epoch += 1
            cache_ids.difference_update(retrieval_ids)
            receipt = RuntimeInvalidationReceipt(
                runtime_id=self.runtime_id,
                change_ids=(selected.change_id,),
                previous_roots=previous_roots,
                current_roots=target_roots,
                event_cursor=selected_cursor,
                proof_index_id=changed_index.index_id,
                cas_transaction_ids=tuple(cas_transaction_ids),
                retrieval_ids=tuple(retrieval_ids),
                context_ids=tuple(context_ids),
                plan_ids=tuple(plan_ids),
                plan_suffix_ids=plan_suffix_ids,
                permit_ids=tuple(permit_ids),
                proof_ids=tuple(proof_ids),
                obligation_ids=tuple(obligation_ids),
                validation_ids=tuple(validation_ids),
                cache_ids=tuple(cache_ids),
                merge_receipt_ids=tuple(merge_ids),
                completion_receipt_ids=tuple(completion_ids),
                preserved_artifact_ids=tuple(preserved),
                recomputed_artifact_ids=tuple(recomputed_ids),
                fencing_epoch=self._fencing_epoch,
                authoritative=not reasons,
                reason_codes=tuple(sorted(set(reasons))),
            )
            self._invalidation_receipts.append(receipt)
            if reasons:
                self._invalidation_quarantine_reasons.update(
                    receipt.reason_codes
                )
            return RuntimeInvalidationResult(
                changed_index,
                receipt,
                MappingProxyType(recomputed),
            )

    invalidate_dependency_change = apply_dependency_change
    apply_semantic_change = apply_dependency_change

    def replay_dependency_events(
        self,
        event_log_path: Any,
        cursor: EventCursor | Mapping[str, Any] | str,
        proof_index: Any,
        *,
        cas: Any = None,
        recompute: Mapping[str, Callable[[tuple[str, ...], SemanticChange, Any], Any]]
        | None = None,
        root_reader: Callable[[], Mapping[str, str]] | None = None,
        page_size: int = 256,
        max_events: int = 4096,
    ) -> tuple[Any, tuple[RuntimeInvalidationReceipt, ...], EventCursor]:
        """Strictly replay canonical dependency changes from one checkpoint."""

        selected_cursor = (
            cursor
            if isinstance(cursor, EventCursor)
            else EventCursor.from_token(cursor)
            if isinstance(cursor, str)
            else EventCursor.from_dict(cursor)
        )
        index = proof_index
        receipts: list[RuntimeInvalidationReceipt] = []
        consumed = 0
        if page_size < 1 or max_events < 1:
            raise DecisionRuntimeConfigurationError(
                "page_size and max_events must be positive integers"
            )
        while True:
            page_limit = min(page_size, max_events - consumed)
            try:
                page = read_semantic_change_page(
                    event_log_path,
                    selected_cursor,
                    limit=page_limit,
                    known_change_ids=self._seen_change_ids,
                    expected_roots=self._semantic_root_state,
                )
                physical_page = read_jsonl_event_page(
                    event_log_path, selected_cursor, limit=page_limit
                )
            except (SemanticChangeIntegrityError, CursorReplayError) as exc:
                raise DecisionRuntimeDenied(
                    ("semantic_event_replay_failed", type(exc).__name__)
                ) from exc
            # Each logical change shares the page cursor only if it is the last
            # semantic event. Reconstruct its exact physical cursor from the
            # event sequence and identity for checkpoint-safe application.
            for change, event_id in zip(page.changes, page.event_ids):
                position = selected_cursor.position + 1
                # Non-semantic events may precede this change. The content ID
                # is enough to locate the exact sequence in the verified page.
                physical = next(
                    event
                    for event in physical_page.events
                    if str(event.get("event_id") or "") == event_id
                )
                position = int(physical["sequence"])
                change_cursor = EventCursor(
                    stream_id=page.next_cursor.stream_id,
                    snapshot_id=page.next_cursor.snapshot_id,
                    position=position,
                    last_event_id=event_id,
                )
                result = self.apply_dependency_change(
                    change,
                    index,
                    cas=cas,
                    event_cursor=change_cursor,
                    recompute=recompute,
                    root_reader=root_reader,
                )
                index = result.proof_index
                receipts.append(result.receipt)
            consumed += len(physical_page.events)
            selected_cursor = page.next_cursor
            self._event_cursor = selected_cursor
            if not page.has_more:
                break
            if consumed >= max_events:
                raise DecisionRuntimeDenied(
                    ("semantic_event_replay_bound_exceeded",)
                )
        return index, tuple(receipts), selected_cursor

    def decide(self, value: DecisionRuntimeInput) -> DecisionRuntimeDecision:
        self._check_cancelled("decision")
        if not isinstance(value, DecisionRuntimeInput):
            raise DecisionRuntimeConfigurationError(
                "decide requires DecisionRuntimeInput"
            )
        boundary = value.boundary
        request = value.decision_request
        if self.config.mode is DecisionRuntimeMode.OFF:
            receipt = self._record(
                boundary=boundary,
                outcome=DecisionOutcome.OFF,
                decision_request_id=request.request_id,
                reason_codes=("runtime_off",),
            )
            return DecisionRuntimeDecision(receipt, request)

        reasons = list(self._validate_bindings(request))
        if self._invalidation_quarantine_reasons:
            reasons.extend(
                (
                    "invalidation_quarantined",
                    *sorted(self._invalidation_quarantine_reasons),
                )
            )
        compilation = value.context_compilation
        if compilation is None and value.graph is not None and value.retrieval_receipt is not None:
            try:
                compilation = self.compile_context(
                    request,
                    value.graph,
                    value.retrieval_receipt,
                    acceptance=value.acceptance,
                    validation=value.validation,
                    failure_behavior=value.failure_behavior,
                )
            except Exception as exc:
                reasons.extend(
                    ("context_compilation_failed", type(exc).__name__)
                )
        witness = self._witness(compilation)
        if witness is None:
            reasons.append("context_witness_missing")
        elif witness.decision_request_id != request.request_id:
            reasons.append("context_witness_request_mismatch")

        admission_request = value.admission_request
        admission_receipt = value.admission_receipt
        if admission_request is not None:
            try:
                compiled_admission = compile_plan_admission(admission_request)
            except Exception as exc:
                reasons.extend(("plan_admission_failed", type(exc).__name__))
            else:
                if (
                    admission_receipt is not None
                    and admission_receipt != compiled_admission
                ):
                    reasons.append("admission_receipt_mismatch")
                admission_receipt = compiled_admission
                if not admission_receipt.admitted:
                    reasons.append("plan_not_admitted")
        elif boundary.mutating:
            reasons.append("admission_request_missing")

        if boundary is DecisionBoundary.COMPLETION:
            return self._decide_completion(
                value,
                compilation=compilation,
                admission_receipt=admission_receipt,
                reasons=reasons,
            )

        if boundary.mutating and request.decision_kind is DecisionKind.COMPLETE:
            reasons.append("completion_request_cannot_authorize_effect")
        if boundary.mutating and request.action.authority.value != "mutation":
            reasons.append("mutation_authority_missing")

        normalized_reasons = tuple(sorted(set(reasons)))
        if self.config.mode is DecisionRuntimeMode.SHADOW:
            receipt = self._record(
                boundary=boundary,
                outcome=(
                    DecisionOutcome.SHADOW_WOULD_BLOCK
                    if normalized_reasons
                    else DecisionOutcome.SHADOW_ALLOWED
                ),
                decision_request_id=request.request_id,
                context_witness_id=witness.content_id if witness else "",
                admission_receipt_id=(
                    admission_receipt.receipt_id if admission_receipt else ""
                ),
                reason_codes=normalized_reasons or ("shadow_non_authoritative",),
            )
            return DecisionRuntimeDecision(
                receipt, request, compilation, admission_receipt
            )

        if normalized_reasons:
            receipt = self._record(
                boundary=boundary,
                outcome=DecisionOutcome.DENIED,
                decision_request_id=request.request_id,
                context_witness_id=witness.content_id if witness else "",
                admission_receipt_id=(
                    admission_receipt.receipt_id if admission_receipt else ""
                ),
                reason_codes=normalized_reasons,
            )
            raise DecisionRuntimeDenied(normalized_reasons)

        permit: ExecutionPermit | None = None
        if boundary.mutating:
            assert admission_request is not None
            assert admission_receipt is not None
            assert witness is not None
            now = self._clock_ms()
            permit = self._issuer.issue(
                admission_request,
                admission_receipt,
                witness,
                caller=self.config.caller,
                policy_id=self.config.policy_id,
                policy_revision=self.config.policy_revision,
                expires_at_ms=now + self.config.permit_ttl_ms,
                evidence_receipts=value.evidence_receipts,
                issued_at_ms=now,
            )
            self._issued_permits[permit.permit_id] = permit
        receipt = self._record(
            boundary=boundary,
            outcome=DecisionOutcome.ADMITTED,
            decision_request_id=request.request_id,
            context_witness_id=witness.content_id if witness else "",
            admission_receipt_id=(
                admission_receipt.receipt_id if admission_receipt else ""
            ),
            permit_id=permit.permit_id if permit else "",
            authoritative=bool(boundary.mutating and permit is not None),
        )
        return DecisionRuntimeDecision(
            receipt, request, compilation, admission_receipt, permit
        )

    def _decide_completion(
        self,
        value: DecisionRuntimeInput,
        *,
        compilation: Any,
        admission_receipt: PlanAdmissionReceipt | None,
        reasons: list[str],
    ) -> DecisionRuntimeDecision:
        request = value.decision_request
        witness = self._witness(compilation)
        if request.decision_kind is not DecisionKind.COMPLETE:
            reasons.append("completion_kind_required")
        if request.stage is not DecisionStage.COMPLETION:
            reasons.append("completion_stage_required")
        if not value.prior_decision_request_id:
            reasons.append("prior_decision_request_id_missing")
        elif value.prior_decision_request_id == request.request_id:
            reasons.append("fresh_completion_decision_required")
        evidence = dict(value.completion_evidence or {})
        if not evidence:
            reasons.append("merged_tree_evidence_missing")
        else:
            expected_roots = {
                request.repository_root.cid_v1,
                request.dirty_worktree_root.cid_v1,
            }
            observed_roots = {
                str(evidence.get(key) or "")
                for key in (
                    "repository_tree_id",
                    "merged_tree_id",
                    "program_root_id",
                    "dirty_worktree_root_id",
                )
                if evidence.get(key)
            }
            if not observed_roots or not observed_roots.intersection(expected_roots):
                reasons.append("merged_tree_evidence_root_mismatch")
            if evidence.get("passed") is not True:
                reasons.append("merged_tree_evidence_not_passed")
            if evidence.get("completion_authoritative") is not True:
                reasons.append("completion_evidence_not_authoritative")
        normalized_reasons = tuple(sorted(set(reasons)))
        if self.config.mode is DecisionRuntimeMode.SHADOW:
            receipt = self._record(
                boundary=DecisionBoundary.COMPLETION,
                outcome=(
                    DecisionOutcome.SHADOW_WOULD_BLOCK
                    if normalized_reasons
                    else DecisionOutcome.SHADOW_ALLOWED
                ),
                decision_request_id=request.request_id,
                context_witness_id=witness.content_id if witness else "",
                reason_codes=normalized_reasons or ("shadow_non_authoritative",),
            )
            return DecisionRuntimeDecision(receipt, request, compilation)
        if normalized_reasons:
            self._record(
                boundary=DecisionBoundary.COMPLETION,
                outcome=DecisionOutcome.DENIED,
                decision_request_id=request.request_id,
                context_witness_id=witness.content_id if witness else "",
                reason_codes=normalized_reasons,
            )
            raise DecisionRuntimeDenied(normalized_reasons)
        receipt = self._record(
            boundary=DecisionBoundary.COMPLETION,
            outcome=DecisionOutcome.COMPLETION_ADMITTED,
            decision_request_id=request.request_id,
            context_witness_id=witness.content_id if witness else "",
            admission_receipt_id=(
                admission_receipt.receipt_id if admission_receipt else ""
            ),
            authoritative=False,
            completion_authoritative=True,
            metadata={
                "prior_decision_request_id": value.prior_decision_request_id,
                "completion_evidence_id": content_identity(evidence),
            },
        )
        return DecisionRuntimeDecision(receipt, request, compilation)

    def route(
        self,
        boundary: DecisionBoundary | str,
        payload: Mapping[str, Any] | None = None,
        *,
        decision_input: DecisionRuntimeInput | None = None,
    ) -> DecisionRuntimeDecision:
        selected = _boundary(boundary)
        self._check_cancelled("route")
        if decision_input is not None:
            if decision_input.boundary is not selected:
                raise DecisionRuntimeConfigurationError(
                    "decision input boundary differs from route"
                )
            return self.decide(decision_input)
        if self._resolver is not None:
            resolved = self._resolver(
                selected, MappingProxyType(dict(_plain(payload or {})))
            )
            if isinstance(resolved, DecisionRuntimeDecision):
                if resolved.receipt.boundary is not selected:
                    raise DecisionRuntimeConfigurationError(
                        "resolver returned a decision for another boundary"
                    )
                return resolved
            if not isinstance(resolved, DecisionRuntimeInput):
                raise DecisionRuntimeConfigurationError(
                    "resolver must return DecisionRuntimeInput or DecisionRuntimeDecision"
                )
            if resolved.boundary is not selected:
                raise DecisionRuntimeConfigurationError(
                    "resolver returned inputs for another boundary"
                )
            return self.decide(resolved)
        if self.config.mode is DecisionRuntimeMode.ENFORCE:
            self._record(
                boundary=selected,
                outcome=DecisionOutcome.DENIED,
                reason_codes=("decision_resolver_missing",),
            )
            raise DecisionRuntimeBypassError(
                ("decision_resolver_missing", "direct_call_bypass")
            )
        outcome = (
            DecisionOutcome.OFF
            if self.config.mode is DecisionRuntimeMode.OFF
            else DecisionOutcome.SHADOW_WOULD_BLOCK
        )
        receipt = self._record(
            boundary=selected,
            outcome=outcome,
            reason_codes=(
                ("runtime_off",)
                if outcome is DecisionOutcome.OFF
                else ("decision_resolver_missing", "shadow_non_authoritative")
            ),
            metadata={"payload_id": content_identity(_plain(payload or {}))},
        )
        return DecisionRuntimeDecision(receipt)

    prepare = route

    def verify_observed_effects(
        self,
        decision: DecisionRuntimeDecision,
        observed_effects: Sequence[Any],
        *,
        permit_use: PermitUseReceipt | None = None,
        pre_root_id: str = "",
        post_root_id: str = "",
    ) -> EffectObservationReceipt:
        if decision.decision_request is None:
            raise DecisionRuntimeBypassError(
                ("decision_request_missing", "direct_call_bypass")
            )
        expected = tuple(
            ObservedEffect.from_effect(item)
            for item in decision.decision_request.expected_effects
        )
        observed = tuple(
            sorted(
                (ObservedEffect.from_value(item) for item in observed_effects),
                key=lambda item: item.effect_id,
            )
        )
        expected = tuple(sorted(expected, key=lambda item: item.effect_id))
        reasons: list[str] = []
        expected_ids = {item.effect_id for item in expected}
        observed_ids = {item.effect_id for item in observed}
        if observed_ids.difference(expected_ids):
            reasons.append("unexpected_effect")
        if expected_ids.difference(observed_ids):
            reasons.append("missing_expected_effect")
        if len(observed_ids) != len(observed):
            reasons.append("duplicate_observed_effect")
        by_expected = {item.effect_id: item for item in expected}
        for item in observed:
            expected_item = by_expected.get(item.effect_id)
            if expected_item is not None and item != expected_item:
                reasons.append(f"changed_effect:{item.effect_id}")
        if pre_root_id and post_root_id and pre_root_id == post_root_id and any(
            item.authority == "mutation" for item in expected
        ):
            reasons.append("mutation_root_unchanged")
        matched = not reasons
        receipt = EffectObservationReceipt(
            runtime_id=self.runtime_id,
            decision_receipt_id=decision.receipt.receipt_id,
            permit_use_receipt_id=(
                getattr(permit_use, "receipt_id", "")
                or getattr(permit_use, "content_id", "")
                if permit_use is not None
                else ""
            ),
            expected_effects=expected,
            observed_effects=observed,
            matched=matched,
            reason_codes=tuple(sorted(set(reasons))),
            pre_root_id=str(pre_root_id or ""),
            post_root_id=str(post_root_id or ""),
        )
        with self._lock:
            self._effect_receipts.append(receipt)
        return receipt

    def authorize_mutation(
        self,
        decision: DecisionRuntimeDecision,
        dispatch: Callable[[], Any],
        *,
        observe_effects: Callable[[Any], Sequence[Any]] | None = None,
        decision_request: DecisionRequest | None = None,
        repository_tree_id: str | None = None,
        semantic_roots: Mapping[str, str] | None = None,
        active_lease_id: str | None = None,
        current_fencing_epoch: int | None = None,
        actual_paths: Sequence[str] | None = None,
        now_ms: int | None = None,
        pre_root_id: str = "",
        post_root: Callable[[Any], str] | None = None,
    ) -> DecisionExecutionResult:
        if not callable(dispatch):
            raise TypeError("dispatch must be callable")
        self._check_cancelled("pre_effect")
        permit_use: PermitUseReceipt | None = None
        if self.config.mode is DecisionRuntimeMode.ENFORCE:
            if self._invalidation_quarantine_reasons:
                raise DecisionRuntimeDenied(
                    (
                        "invalidation_quarantined",
                        *tuple(
                            sorted(self._invalidation_quarantine_reasons)
                        ),
                    )
                )
            if (
                not isinstance(decision, DecisionRuntimeDecision)
                or decision.receipt.runtime_id != self.runtime_id
                or not decision.receipt.authoritative
                or decision.permit is None
            ):
                raise DecisionRuntimeBypassError(
                    ("current_permit_missing", "direct_call_bypass")
                )
            permit = decision.permit
            if (
                permit.permit_id in self._invalidated_permit_ids
                or decision.receipt.receipt_id
                in self._invalidated_decision_receipt_ids
            ):
                self._record(
                    boundary=decision.receipt.boundary,
                    outcome=DecisionOutcome.DENIED,
                    decision_request_id=decision.receipt.decision_request_id,
                    permit_id=permit.permit_id,
                    reason_codes=(
                        "dependency_invalidated",
                        "current_permit_rejected",
                    ),
                )
                raise DecisionRuntimeDenied(
                    ("dependency_invalidated", "current_permit_rejected")
                )
            attempt = ExecutionAttempt.from_permit(
                permit,
                now_ms=self._clock_ms() if now_ms is None else now_ms,
                decision_request=decision_request or permit.decision_request,
                repository_tree_id=(
                    repository_tree_id or permit.repository_tree_id
                ),
                semantic_roots=semantic_roots or permit.semantic_roots,
                active_lease_id=active_lease_id or permit.lease_id,
                current_fencing_epoch=(
                    permit.fencing_epoch
                    if current_fencing_epoch is None
                    else current_fencing_epoch
                ),
                actual_paths=(
                    tuple(actual_paths)
                    if actual_paths is not None
                    else permit.declared_paths
                ),
            )
            # This is intentionally adjacent to dispatch.  No transport,
            # backend, or caller can split verification from the effect.
            self._check_cancelled("permit_verification")
            try:
                permit_use = self._verifier.verify(permit, attempt)
            except PermitVerificationError as exc:
                code = str(getattr(exc, "code", "permit_rejected"))
                reason = str(getattr(getattr(exc, "code", None), "value", code))
                self._record(
                    boundary=decision.receipt.boundary,
                    outcome=DecisionOutcome.DENIED,
                    decision_request_id=decision.receipt.decision_request_id,
                    permit_id=decision.receipt.permit_id,
                    reason_codes=("current_permit_rejected", reason),
                )
                raise DecisionRuntimeDenied(
                    ("current_permit_rejected", reason)
                ) from exc
            self._check_cancelled("dispatch")
        value = dispatch()
        observed: Sequence[Any] = ()
        if observe_effects is not None:
            observed = observe_effects(value)
        elif isinstance(value, Mapping) and isinstance(
            value.get("observed_effects"), Sequence
        ):
            observed = value["observed_effects"]
        observation: EffectObservationReceipt | None = None
        if decision.decision_request is not None and (
            observe_effects is not None
            or observed
            or self.config.mode is DecisionRuntimeMode.ENFORCE
        ):
            post_root_id = post_root(value) if post_root is not None else ""
            observation = self.verify_observed_effects(
                decision,
                observed,
                permit_use=permit_use,
                pre_root_id=pre_root_id,
                post_root_id=post_root_id,
            )
            if (
                self.config.mode is DecisionRuntimeMode.ENFORCE
                and not observation.matched
            ):
                self._record(
                    boundary=decision.receipt.boundary,
                    outcome=DecisionOutcome.EFFECT_MISMATCH,
                    decision_request_id=decision.receipt.decision_request_id,
                    permit_id=decision.receipt.permit_id,
                    reason_codes=observation.reason_codes,
                    metadata={
                        "effect_observation_receipt_id": observation.receipt_id
                    },
                )
                raise DecisionRuntimeEffectMismatch(observation)
        return DecisionExecutionResult(value, decision, permit_use, observation)

    execute_mutation = authorize_mutation

    def execute(
        self,
        boundary: DecisionBoundary | str,
        dispatch: Callable[[], Any],
        *,
        payload: Mapping[str, Any] | None = None,
        decision_input: DecisionRuntimeInput | None = None,
        observe_effects: Callable[[Any], Sequence[Any]] | None = None,
        **current: Any,
    ) -> DecisionExecutionResult:
        decision = self.route(
            boundary, payload, decision_input=decision_input
        )
        selected = _boundary(boundary)
        if selected.mutating:
            return self.authorize_mutation(
                decision,
                dispatch,
                observe_effects=observe_effects,
                **current,
            )
        self._check_cancelled("dispatch")
        return DecisionExecutionResult(dispatch(), decision, None, None)

    def admit_completion(
        self, value: DecisionRuntimeInput
    ) -> DecisionRuntimeDecision:
        if _boundary(value.boundary) is not DecisionBoundary.COMPLETION:
            raise DecisionRuntimeConfigurationError(
                "completion admission requires the completion boundary"
            )
        return self.decide(value)


def runtime_config_from_control_parameters(
    parameters: Mapping[str, Any],
) -> DecisionRuntimeConfig | None:
    """Decode one canonical runtime config shared by Python, CLI, and MCP."""

    if not isinstance(parameters, Mapping):
        raise DecisionRuntimeConfigurationError(
            "control parameters must be an object"
        )
    value = parameters.get("decision_runtime")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DecisionRuntimeConfigurationError(
            "parameters.decision_runtime must be an object"
        )
    return DecisionRuntimeConfig.from_dict(value)


__all__ = [
    "DECISION_RUNTIME_CONFIG_SCHEMA",
    "DECISION_RUNTIME_RECEIPT_SCHEMA",
    "DECISION_RUNTIME_VERSION",
    "EFFECT_OBSERVATION_SCHEMA",
    "INCREMENTAL_REVALIDATION_REQUIREMENT_ID",
    "RUNTIME_INVALIDATION_RECEIPT_SCHEMA",
    "CanonicalDependencyChange",
    "DependencyChangeEvent",
    "DecisionBoundary",
    "DecisionExecutionResult",
    "DecisionOutcome",
    "DecisionRuntime",
    "DecisionRuntimeBoundary",
    "DecisionRuntimeBypassError",
    "DecisionRuntimeCancelled",
    "DecisionRuntimeConfig",
    "DecisionRuntimeConfigurationError",
    "DecisionRuntimeDecision",
    "DecisionRuntimeDenied",
    "DecisionRuntimeEffectMismatch",
    "DecisionRuntimeError",
    "DecisionRuntimeInput",
    "DecisionRuntimeMode",
    "DecisionRuntimeReceipt",
    "EffectObservationReceipt",
    "ObservedEffect",
    "RuntimeInvalidationReceipt",
    "RuntimeInvalidationResult",
    "SemanticChange",
    "SemanticChangeKind",
    "canonical_dependency_change",
    "runtime_config_from_control_parameters",
]
