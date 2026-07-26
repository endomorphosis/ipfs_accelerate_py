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
from .execution_permit import (
    ExecutionAttempt,
    ExecutionEvidence,
    ExecutionPermit,
    ExecutionPermitIssuer,
    PermitUseReceipt,
    PermitVerificationError,
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


@dataclass(frozen=True)
class DecisionExecutionResult:
    value: Any
    decision: DecisionRuntimeDecision
    permit_use: PermitUseReceipt | None
    effect_observation: EffectObservationReceipt | None


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

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "runtime_id": self.runtime_id,
                "config": self.config.to_dict(),
                "decision_count": len(self._receipts),
                "effect_observation_count": len(self._effect_receipts),
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
    "runtime_config_from_control_parameters",
]
