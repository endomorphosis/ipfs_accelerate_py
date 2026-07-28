"""Cross-lane information-flow models and bounded verification.

Single executions cannot demonstrate noninterference: the relevant question is
whether changing a high/confidential input can change a low observation.  This
module therefore models supervisor information-flow requirements as two-trace
HyperLTL properties and provides two deliberately separate evidence paths:

* HyperLTL, AutoHyper, and MCHyper adapters become available only after a
  bounded executable conformance fixture has passed; and
* deterministic bounded self-composition supplies useful, explicitly
  non-authoritative test evidence when no conformant engine is available.

Private inputs are comparison-only values.  They have no public serialization
method and are never copied into reports or counterexamples.  Counterexamples
contain only approved observation paths and digests of differing values, are
reduced to one violating trace pair, and bind the exact observation policy.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    _canonical_value,
    canonical_json_bytes,
    content_identity,
)
from .prover_matrix_registry import (
    CommandRequest,
    CommandResult,
    _default_command_runner as _bounded_command_runner,
)


HYPERPROPERTY_VERIFICATION_VERSION = 1
OBSERVATION_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hyperproperty-observation-policy@1"
)
HYPERPROPERTY_MODEL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hyperproperty-model@1"
)
HYPERTRACE_COUNTEREXAMPLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hypertrace-counterexample@1"
)
HYPERPROPERTY_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hyperproperty-result@1"
)
ENGINE_CONFORMANCE_FIXTURE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hyperproperty-engine-fixture@1"
)
ENGINE_CONFORMANCE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hyperproperty-engine-conformance@1"
)
ENGINE_CAPABILITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hyperproperty-engine-capability@1"
)

DEFAULT_MAX_COMPOSITION_TRACES = 32
DEFAULT_MAX_COMPOSITION_PAIRS = 256
DEFAULT_ENGINE_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_ENGINE_OUTPUT_BYTES = 64 * 1024
DEFAULT_MAX_EXECUTABLE_BYTES = 16 * 1024 * 1024
REDACTED_VALUE = "<redacted>"
MISSING_VALUE = "<missing>"


class HyperpropertyValidationError(ContractValidationError):
    """Raised when a hyperproperty contract could leak or is malformed."""


class HyperpropertyKind(str, Enum):
    """The reviewed cross-lane requirements."""

    PROMPT_ISOLATION = "prompt_isolation"
    WORKTREE_ISOLATION = "worktree_isolation"
    LOG_REDACTION = "log_redaction"
    PROVIDER_ROUTING = "provider_routing"
    ZKP_WITNESS_NONINTERFERENCE = "zkp_witness_noninterference"
    CROSS_TASK_CACHE_SEPARATION = "cross_task_cache_separation"


class HyperpropertyEngine(str, Enum):
    HYPERLTL = "hyperltl"
    AUTOHYPER = "autohyper"
    MCHYPER = "mchyper"


# Compatibility spelling for callers that use the shorter noun.
EngineKind = HyperpropertyEngine


class ConformanceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    NOT_RUN = "not_run"


class EngineCapabilityStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    CONFORMANT = "conformant"


class HyperpropertyVerdict(str, Enum):
    HOLDS = "holds"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class HyperpropertyEvidenceKind(str, Enum):
    ENGINE = "hyperproperty_engine"
    BOUNDED_SELF_COMPOSITION = "bounded_self_composition"
    NONE = "none"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise HyperpropertyValidationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise HyperpropertyValidationError(f"{name} must not be empty")
    return result


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise HyperpropertyValidationError(f"unsupported {name}") from exc


def _strings(
    values: Sequence[Any] | None, name: str, *, required: bool = False
) -> tuple[str, ...]:
    if values is None:
        result: tuple[str, ...] = ()
    elif isinstance(values, (str, bytes, bytearray)):
        raise HyperpropertyValidationError(f"{name} must be a sequence")
    else:
        result = tuple(sorted({_text(item, name) for item in values}))
    if required and not result:
        raise HyperpropertyValidationError(f"{name} must not be empty")
    return result


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise HyperpropertyValidationError(
            f"{name} must be an object with string keys"
        )
    try:
        result = _canonical_value(dict(value))
    except ContractValidationError as exc:
        raise HyperpropertyValidationError(str(exc)) from exc
    assert isinstance(result, dict)
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HyperpropertyValidationError(f"{name} must be a positive integer")
    return value


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise HyperpropertyValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _claimed_identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id") or payload.get("identity")
    if claimed and claimed != actual:
        raise HyperpropertyValidationError(f"{noun} identity does not match payload")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return MISSING_VALUE
        current = current[component]
    return current


def _projection(value: Mapping[str, Any], paths: Sequence[str]) -> tuple[Any, ...]:
    return tuple(_path_value(value, path) for path in paths)


@dataclass(frozen=True)
class ObservationPolicy(CanonicalContract):
    """Exact low-input and low-observation boundary for one model.

    Field paths are relative to :class:`Hypertrace` ``public_inputs`` and
    ``observations`` mappings.  Secret field names intentionally do not form
    part of the public policy: all values in ``private_inputs`` are high.
    """

    SCHEMA = OBSERVATION_POLICY_SCHEMA

    policy_id: str
    version: str
    low_input_fields: tuple[str, ...]
    observation_fields: tuple[str, ...]
    subject_fields: tuple[str, ...] = ("task_id",)
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        object.__setattr__(
            self,
            "low_input_fields",
            _strings(self.low_input_fields, "low_input_fields"),
        )
        object.__setattr__(
            self,
            "observation_fields",
            _strings(self.observation_fields, "observation_fields", required=True),
        )
        allowed_subjects = {"task_id", "lane_id", "worktree_id"}
        subjects = _strings(self.subject_fields, "subject_fields")
        if any(item not in allowed_subjects for item in subjects):
            raise HyperpropertyValidationError(
                "subject_fields may contain only task_id, lane_id, and worktree_id"
            )
        object.__setattr__(self, "subject_fields", subjects)
        object.__setattr__(
            self, "description", _text(self.description, "description", required=False)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "hyperproperty_version": HYPERPROPERTY_VERIFICATION_VERSION,
            "policy_id": self.policy_id,
            "version": self.version,
            "low_input_fields": self.low_input_fields,
            "observation_fields": self.observation_fields,
            "subject_fields": self.subject_fields,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservationPolicy":
        if not isinstance(payload, Mapping):
            raise HyperpropertyValidationError("observation policy must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            policy_id=payload.get("policy_id", ""),
            version=payload.get("version", ""),
            low_input_fields=tuple(payload.get("low_input_fields") or ()),
            observation_fields=tuple(payload.get("observation_fields") or ()),
            subject_fields=tuple(payload.get("subject_fields") or ()),
            description=payload.get("description", ""),
        )
        _claimed_identity(payload, result.content_id, "observation policy")
        return result


@dataclass(frozen=True)
class HyperpropertyModel(CanonicalContract):
    """Reviewed two-trace HyperLTL model."""

    SCHEMA = HYPERPROPERTY_MODEL_SCHEMA

    model_id: str
    version: str
    kind: HyperpropertyKind
    observation_policy: ObservationPolicy
    hyperltl_formula: str
    description: str
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        object.__setattr__(
            self, "kind", _enum(self.kind, HyperpropertyKind, "hyperproperty kind")
        )
        if isinstance(self.observation_policy, Mapping):
            object.__setattr__(
                self,
                "observation_policy",
                ObservationPolicy.from_dict(self.observation_policy),
            )
        if not isinstance(self.observation_policy, ObservationPolicy):
            raise HyperpropertyValidationError(
                "observation_policy must be an ObservationPolicy"
            )
        formula = _text(self.hyperltl_formula, "hyperltl_formula")
        compact = " ".join(formula.split()).lower()
        if compact.count("forall") < 2 or "->" not in compact:
            raise HyperpropertyValidationError(
                "a noninterference model must be an explicit two-trace implication"
            )
        object.__setattr__(self, "hyperltl_formula", formula)
        object.__setattr__(self, "description", _text(self.description, "description"))
        object.__setattr__(
            self, "assumptions", _strings(self.assumptions, "assumptions")
        )

    @property
    def observation_policy_id(self) -> str:
        return self.observation_policy.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "hyperproperty_version": HYPERPROPERTY_VERIFICATION_VERSION,
            "model_id": self.model_id,
            "version": self.version,
            "kind": self.kind,
            "observation_policy": self.observation_policy,
            "observation_policy_id": self.observation_policy_id,
            "hyperltl_formula": self.hyperltl_formula,
            "description": self.description,
            "assumptions": self.assumptions,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HyperpropertyModel":
        if not isinstance(payload, Mapping):
            raise HyperpropertyValidationError("hyperproperty model must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            model_id=payload.get("model_id", ""),
            version=payload.get("version", ""),
            kind=payload.get("kind", ""),
            observation_policy=payload.get("observation_policy") or {},
            hyperltl_formula=payload.get("hyperltl_formula", ""),
            description=payload.get("description", ""),
            assumptions=tuple(payload.get("assumptions") or ()),
        )
        claimed_policy = payload.get("observation_policy_id")
        if claimed_policy and claimed_policy != result.observation_policy_id:
            raise HyperpropertyValidationError(
                "model observation_policy_id does not match embedded policy"
            )
        _claimed_identity(payload, result.content_id, "hyperproperty model")
        return result


@dataclass(frozen=True)
class Hypertrace:
    """One execution used in self-composition.

    ``private_inputs`` is intentionally absent from :meth:`to_public_dict`,
    ``repr``, equality, and hashing.  It exists only long enough to decide
    whether two executions vary a high input.
    """

    trace_id: str
    task_id: str
    lane_id: str
    worktree_id: str
    public_inputs: Mapping[str, Any]
    observations: Mapping[str, Any]
    private_inputs: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        for name in ("trace_id", "task_id", "lane_id", "worktree_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "public_inputs", _mapping(self.public_inputs, "public_inputs")
        )
        object.__setattr__(
            self, "observations", _mapping(self.observations, "observations")
        )
        object.__setattr__(
            self, "private_inputs", _mapping(self.private_inputs, "private_inputs")
        )

    def subject_projection(self, fields: Sequence[str]) -> tuple[str, ...]:
        return tuple(str(getattr(self, field)) for field in fields)

    @property
    def public_ref(self) -> str:
        # Bind the execution without publishing its values.  Including public
        # and observed digests distinguishes two traces that share the same
        # task/lane labels while still keeping the counterexample value-free.
        return content_identity(
            {
                "labels": {
                    "trace_id": self.trace_id,
                    "task_id": self.task_id,
                    "lane_id": self.lane_id,
                    "worktree_id": self.worktree_id,
                },
                "public_inputs_sha256": _digest(self.public_inputs),
                "observations_sha256": _digest(self.observations),
                "private_inputs_redacted": True,
            }
        )

    def to_public_dict(self) -> dict[str, Any]:
        return _canonical_value(
            {
                "trace_ref": content_identity(
                    {
                        "trace_id": self.trace_id,
                        "task_id": self.task_id,
                        "lane_id": self.lane_id,
                        "worktree_id": self.worktree_id,
                    }
                ),
                "task_ref": content_identity({"task_id": self.task_id}),
                "lane_ref": content_identity({"lane_id": self.lane_id}),
                "worktree_ref": content_identity({"worktree_id": self.worktree_id}),
                "private_inputs_redacted": True,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Compatibility alias that remains a redacted public projection."""

        return self.to_public_dict()


# Friendly name used in design documents.
ExecutionTrace = Hypertrace


@dataclass(frozen=True)
class ObservationDifference(CanonicalContract):
    """Value-free difference at one policy-approved observation path."""

    SCHEMA = ""

    field: str
    left_digest: str
    right_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _text(self.field, "field"))
        for name in ("left_digest", "right_digest"):
            value = _text(getattr(self, name), name)
            if not value.startswith("sha256:"):
                raise HyperpropertyValidationError(f"{name} must be a sha256 digest")
            object.__setattr__(self, name, value)

    def _payload(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "left_digest": self.left_digest,
            "right_digest": self.right_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(self._payload())


@dataclass(frozen=True)
class HypertraceCounterexample(CanonicalContract):
    """Minimal, redacted two-trace counterexample."""

    SCHEMA = HYPERTRACE_COUNTEREXAMPLE_SCHEMA

    model_id: str
    model_identity: str
    observation_policy_id: str
    observation_policy_name: str
    observed_fields: tuple[str, ...]
    trace_refs: tuple[str, str]
    differences: tuple[ObservationDifference, ...]
    redacted: bool = True
    minimized: bool = True

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_identity",
            "observation_policy_id",
            "observation_policy_name",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        fields = _strings(self.observed_fields, "observed_fields", required=True)
        object.__setattr__(self, "observed_fields", fields)
        refs = tuple(self.trace_refs)
        if len(refs) != 2 or any(not str(item).startswith("b") for item in refs):
            raise HyperpropertyValidationError(
                "counterexample must bind exactly two content-addressed trace refs"
            )
        object.__setattr__(self, "trace_refs", tuple(str(item) for item in refs))
        differences = tuple(self.differences)
        if not differences or any(
            not isinstance(item, ObservationDifference) for item in differences
        ):
            raise HyperpropertyValidationError(
                "counterexample requires observation differences"
            )
        if any(item.field not in fields for item in differences):
            raise HyperpropertyValidationError(
                "counterexample difference is outside its observation policy"
            )
        object.__setattr__(
            self, "differences", tuple(sorted(differences, key=lambda item: item.field))
        )
        if self.redacted is not True or self.minimized is not True:
            raise HyperpropertyValidationError(
                "hypertrace counterexamples must be redacted and minimized"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "hyperproperty_version": HYPERPROPERTY_VERIFICATION_VERSION,
            "model_id": self.model_id,
            "model_identity": self.model_identity,
            "observation_policy_id": self.observation_policy_id,
            "observation_policy_name": self.observation_policy_name,
            "observed_fields": self.observed_fields,
            "trace_refs": self.trace_refs,
            "differences": self.differences,
            "redacted": True,
            "minimized": True,
            "contains_private_inputs": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypertraceCounterexample":
        if not isinstance(payload, Mapping):
            raise HyperpropertyValidationError("counterexample must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            model_id=payload.get("model_id", ""),
            model_identity=payload.get("model_identity", ""),
            observation_policy_id=payload.get("observation_policy_id", ""),
            observation_policy_name=payload.get("observation_policy_name", ""),
            observed_fields=tuple(payload.get("observed_fields") or ()),
            trace_refs=tuple(payload.get("trace_refs") or ()),  # type: ignore[arg-type]
            differences=tuple(
                ObservationDifference(
                    field=item.get("field", ""),
                    left_digest=item.get("left_digest", ""),
                    right_digest=item.get("right_digest", ""),
                )
                for item in (payload.get("differences") or ())
                if isinstance(item, Mapping)
            ),
            redacted=payload.get("redacted", False),
            minimized=payload.get("minimized", False),
        )
        if payload.get("contains_private_inputs") not in (None, False):
            raise HyperpropertyValidationError(
                "counterexample claims to contain private inputs"
            )
        _claimed_identity(payload, result.content_id, "counterexample")
        return result


@dataclass(frozen=True)
class HyperpropertyVerificationResult(CanonicalContract):
    """Bound result with an explicit evidence/authority boundary."""

    SCHEMA = HYPERPROPERTY_RESULT_SCHEMA

    model_id: str
    model_identity: str
    observation_policy_id: str
    verdict: HyperpropertyVerdict
    evidence_kind: HyperpropertyEvidenceKind
    authoritative: bool
    bounded: bool
    explored_traces: int
    explored_pairs: int
    maximum_pairs: int
    reason: str
    engine: HyperpropertyEngine | None = None
    engine_capability_id: str | None = None
    counterexample: HypertraceCounterexample | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "model_identity", "observation_policy_id", "reason"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "verdict", _enum(self.verdict, HyperpropertyVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "evidence_kind",
            _enum(self.evidence_kind, HyperpropertyEvidenceKind, "evidence kind"),
        )
        if not isinstance(self.authoritative, bool) or not isinstance(
            self.bounded, bool
        ):
            raise HyperpropertyValidationError(
                "authoritative and bounded must be booleans"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.explored_traces,
                self.explored_pairs,
                self.maximum_pairs,
            )
        ):
            raise HyperpropertyValidationError("result counts must be non-negative")
        if self.explored_pairs > self.maximum_pairs:
            raise HyperpropertyValidationError("explored_pairs exceeds maximum_pairs")
        if self.evidence_kind is HyperpropertyEvidenceKind.BOUNDED_SELF_COMPOSITION:
            if self.authoritative or not self.bounded or self.engine is not None:
                raise HyperpropertyValidationError(
                    "self-composition evidence must be bounded and non-authoritative"
                )
        if self.evidence_kind is HyperpropertyEvidenceKind.ENGINE:
            if self.engine is None or not self.engine_capability_id:
                raise HyperpropertyValidationError(
                    "engine evidence requires an exact conformant capability binding"
                )
        elif self.engine is not None or self.engine_capability_id is not None:
            raise HyperpropertyValidationError(
                "non-engine evidence cannot claim an engine binding"
            )
        if self.counterexample is not None:
            if self.verdict is not HyperpropertyVerdict.VIOLATED:
                raise HyperpropertyValidationError(
                    "only a violated result may contain a counterexample"
                )
            if self.counterexample.model_identity != self.model_identity:
                raise HyperpropertyValidationError(
                    "counterexample belongs to a different model"
                )
            if (
                self.counterexample.observation_policy_id
                != self.observation_policy_id
            ):
                raise HyperpropertyValidationError(
                    "counterexample belongs to a different observation policy"
                )
        elif self.verdict is HyperpropertyVerdict.VIOLATED:
            raise HyperpropertyValidationError(
                "a violated result requires a redacted counterexample"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "hyperproperty_version": HYPERPROPERTY_VERIFICATION_VERSION,
            "model_id": self.model_id,
            "model_identity": self.model_identity,
            "observation_policy_id": self.observation_policy_id,
            "verdict": self.verdict,
            "evidence_kind": self.evidence_kind,
            "authoritative": self.authoritative,
            "bounded": self.bounded,
            "explored_traces": self.explored_traces,
            "explored_pairs": self.explored_pairs,
            "maximum_pairs": self.maximum_pairs,
            "reason": self.reason,
            "engine": self.engine,
            "engine_capability_id": self.engine_capability_id,
            "counterexample": self.counterexample,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "HyperpropertyVerificationResult":
        if not isinstance(payload, Mapping):
            raise HyperpropertyValidationError("verification result must be an object")
        _schema(payload, cls.SCHEMA)
        raw_counterexample = payload.get("counterexample")
        result = cls(
            model_id=payload.get("model_id", ""),
            model_identity=payload.get("model_identity", ""),
            observation_policy_id=payload.get("observation_policy_id", ""),
            verdict=payload.get("verdict", ""),
            evidence_kind=payload.get("evidence_kind", ""),
            authoritative=payload.get("authoritative", False),
            bounded=payload.get("bounded", False),
            explored_traces=payload.get("explored_traces", -1),
            explored_pairs=payload.get("explored_pairs", -1),
            maximum_pairs=payload.get("maximum_pairs", -1),
            reason=payload.get("reason", ""),
            engine=(
                None if payload.get("engine") is None else payload.get("engine")
            ),
            engine_capability_id=payload.get("engine_capability_id"),
            counterexample=(
                HypertraceCounterexample.from_dict(raw_counterexample)
                if isinstance(raw_counterexample, Mapping)
                else None
            ),
        )
        _claimed_identity(payload, result.content_id, "verification result")
        return result


@dataclass(frozen=True)
class EngineConformanceFixture(CanonicalContract):
    """Executable semantic fixture required before an adapter is available."""

    SCHEMA = ENGINE_CONFORMANCE_FIXTURE_SCHEMA

    engine: HyperpropertyEngine
    fixture_id: str
    model_text: str
    file_name: str
    args: tuple[str, ...]
    expected_output_any: tuple[str, ...]
    translator_id: str
    semantic_profile_id: str
    auxiliary_files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "engine", _enum(self.engine, HyperpropertyEngine, "engine")
        )
        for name in (
            "fixture_id",
            "model_text",
            "file_name",
            "translator_id",
            "semantic_profile_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if Path(self.file_name).name != self.file_name:
            raise HyperpropertyValidationError("fixture file_name must be a basename")
        args = tuple(_text(item, "args") for item in self.args)
        if "{fixture}" not in args:
            raise HyperpropertyValidationError("fixture args must include {fixture}")
        auxiliary: dict[str, str] = {}
        if not isinstance(self.auxiliary_files, Mapping):
            raise HyperpropertyValidationError("auxiliary_files must be an object")
        for raw_name, raw_text in self.auxiliary_files.items():
            name = _text(raw_name, "auxiliary file name")
            if Path(name).name != name:
                raise HyperpropertyValidationError(
                    "auxiliary file names must be basenames"
                )
            auxiliary[name] = _text(raw_text, f"auxiliary_files[{name}]")
        expected_placeholders = {f"{{auxiliary:{name}}}" for name in auxiliary}
        supplied_placeholders = {
            item for item in args if item.startswith("{auxiliary:")
        }
        if supplied_placeholders != expected_placeholders:
            raise HyperpropertyValidationError(
                "fixture args must bind every auxiliary file exactly"
            )
        object.__setattr__(self, "args", args)
        object.__setattr__(
            self, "auxiliary_files", dict(sorted(auxiliary.items()))
        )
        markers = tuple(
            sorted(
                {
                    _text(item, "expected_output_any").casefold()
                    for item in self.expected_output_any
                }
            )
        )
        if not markers:
            raise HyperpropertyValidationError(
                "fixture requires a semantic success marker"
            )
        object.__setattr__(self, "expected_output_any", markers)

    def _payload(self) -> dict[str, Any]:
        return {
            "hyperproperty_version": HYPERPROPERTY_VERIFICATION_VERSION,
            "engine": self.engine,
            "fixture_id": self.fixture_id,
            "model_text_sha256": _digest(self.model_text),
            "file_name": self.file_name,
            "args": self.args,
            "expected_output_any": self.expected_output_any,
            "translator_id": self.translator_id,
            "semantic_profile_id": self.semantic_profile_id,
            "auxiliary_files": {
                name: _digest(text)
                for name, text in sorted(self.auxiliary_files.items())
            },
        }


@dataclass(frozen=True)
class EngineConformanceReceipt(CanonicalContract):
    """Output-free receipt for an actually executed conformance fixture."""

    SCHEMA = ENGINE_CONFORMANCE_RECEIPT_SCHEMA

    engine: HyperpropertyEngine
    status: ConformanceStatus
    fixture_id: str
    fixture_identity: str
    executable_path: str
    executable_identity: str
    executable_version: str
    command_identity: str
    returncode: int | None
    timed_out: bool
    output_sha256: str
    marker_matched: bool
    duration_ms: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "engine", _enum(self.engine, HyperpropertyEngine, "engine")
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, ConformanceStatus, "conformance status"),
        )
        for name in (
            "fixture_id",
            "fixture_identity",
            "executable_path",
            "executable_identity",
            "executable_version",
            "command_identity",
            "output_sha256",
            "reason",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if isinstance(self.duration_ms, bool) or self.duration_ms < 0:
            raise HyperpropertyValidationError("duration_ms must be non-negative")
        if not isinstance(self.timed_out, bool) or not isinstance(
            self.marker_matched, bool
        ):
            raise HyperpropertyValidationError(
                "timed_out and marker_matched must be booleans"
            )
        if self.returncode is not None and (
            isinstance(self.returncode, bool) or not isinstance(self.returncode, int)
        ):
            raise HyperpropertyValidationError("returncode must be an integer or null")
        for name in ("executable_identity", "output_sha256"):
            if not str(getattr(self, name)).startswith("sha256:"):
                raise HyperpropertyValidationError(f"{name} must be a sha256 digest")
        if self.status is ConformanceStatus.PASSED and (
            self.returncode != 0 or self.timed_out or not self.marker_matched
        ):
            raise HyperpropertyValidationError(
                "passing conformance requires execution success and a semantic marker"
            )
        if self.status is not ConformanceStatus.PASSED and self.marker_matched:
            raise HyperpropertyValidationError(
                "failed conformance cannot claim a semantic marker"
            )

    @property
    def passed(self) -> bool:
        return self.status is ConformanceStatus.PASSED

    def _payload(self) -> dict[str, Any]:
        return {
            "hyperproperty_version": HYPERPROPERTY_VERIFICATION_VERSION,
            "engine": self.engine,
            "status": self.status,
            "fixture_id": self.fixture_id,
            "fixture_identity": self.fixture_identity,
            "executable_path": self.executable_path,
            "executable_identity": self.executable_identity,
            "executable_version": self.executable_version,
            "command_identity": self.command_identity,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "output_sha256": self.output_sha256,
            "marker_matched": self.marker_matched,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
            "raw_output_included": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineConformanceReceipt":
        if not isinstance(payload, Mapping):
            raise HyperpropertyValidationError(
                "engine conformance receipt must be an object"
            )
        _schema(payload, cls.SCHEMA)
        if payload.get("raw_output_included") not in (None, False):
            raise HyperpropertyValidationError(
                "engine conformance receipt must not include raw output"
            )
        result = cls(
            engine=payload.get("engine", ""),
            status=payload.get("status", ""),
            fixture_id=payload.get("fixture_id", ""),
            fixture_identity=payload.get("fixture_identity", ""),
            executable_path=payload.get("executable_path", ""),
            executable_identity=payload.get("executable_identity", ""),
            executable_version=payload.get("executable_version", ""),
            command_identity=payload.get("command_identity", ""),
            returncode=payload.get("returncode"),
            timed_out=payload.get("timed_out", False),
            output_sha256=payload.get("output_sha256", ""),
            marker_matched=payload.get("marker_matched", False),
            duration_ms=payload.get("duration_ms", -1),
            reason=payload.get("reason", ""),
        )
        _claimed_identity(payload, result.content_id, "engine conformance receipt")
        return result


@dataclass(frozen=True)
class EngineCapability(CanonicalContract):
    """Fail-closed runtime adapter capability."""

    SCHEMA = ENGINE_CAPABILITY_SCHEMA

    engine: HyperpropertyEngine
    status: EngineCapabilityStatus
    reason: str
    executable_path: str | None = None
    executable_version: str | None = None
    conformance_receipt: EngineConformanceReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "engine", _enum(self.engine, HyperpropertyEngine, "engine")
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, EngineCapabilityStatus, "capability status"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.status is EngineCapabilityStatus.CONFORMANT:
            receipt = self.conformance_receipt
            if (
                receipt is None
                or not receipt.passed
                or receipt.engine is not self.engine
                or receipt.executable_path != self.executable_path
                or not self.executable_version
            ):
                raise HyperpropertyValidationError(
                    "conformant capability requires its exact passing executable receipt"
                )
        elif self.conformance_receipt is not None and self.conformance_receipt.passed:
            raise HyperpropertyValidationError(
                "passing receipt cannot be reported as unavailable"
            )

    @property
    def available(self) -> bool:
        return self.status is EngineCapabilityStatus.CONFORMANT

    @property
    def conformance_passed(self) -> bool:
        return bool(self.conformance_receipt and self.conformance_receipt.passed)

    def _payload(self) -> dict[str, Any]:
        return {
            "hyperproperty_version": HYPERPROPERTY_VERIFICATION_VERSION,
            "engine": self.engine,
            "status": self.status,
            "available": self.available,
            "reason": self.reason,
            "executable_path": self.executable_path,
            "executable_version": self.executable_version,
            "conformance_receipt": self.conformance_receipt,
            "discovery_is_conformance": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineCapability":
        if not isinstance(payload, Mapping):
            raise HyperpropertyValidationError("engine capability must be an object")
        _schema(payload, cls.SCHEMA)
        if payload.get("discovery_is_conformance") not in (None, False):
            raise HyperpropertyValidationError(
                "engine discovery cannot be represented as conformance"
            )
        raw_receipt = payload.get("conformance_receipt")
        result = cls(
            engine=payload.get("engine", ""),
            status=payload.get("status", ""),
            reason=payload.get("reason", ""),
            executable_path=payload.get("executable_path"),
            executable_version=payload.get("executable_version"),
            conformance_receipt=(
                EngineConformanceReceipt.from_dict(raw_receipt)
                if isinstance(raw_receipt, Mapping)
                else None
            ),
        )
        claimed_available = payload.get("available")
        if claimed_available is not None and claimed_available is not result.available:
            raise HyperpropertyValidationError(
                "engine availability does not match conformance status"
            )
        _claimed_identity(payload, result.content_id, "engine capability")
        return result


CommandRunner = Callable[[CommandRequest], CommandResult | Mapping[str, Any]]
ExecutableFinder = Callable[[str], str | None]


def _default_command_runner(request: CommandRequest) -> CommandResult:
    # Reuse the registry's process-group, deadline, and streaming-output
    # implementation.  In particular, ``subprocess.run(capture_output=True)``
    # would place no in-memory bound on a hostile executable's output.
    return _bounded_command_runner(request)


def _command_result(value: CommandResult | Mapping[str, Any]) -> CommandResult:
    if isinstance(value, CommandResult):
        return value
    if not isinstance(value, Mapping):
        return CommandResult(returncode=None, error="malformed runner result")
    return CommandResult(
        returncode=value.get("returncode"),
        stdout=str(value.get("stdout") or ""),
        stderr=str(value.get("stderr") or ""),
        timed_out=bool(value.get("timed_out", False)),
        error=(str(value["error"]) if value.get("error") else None),
        output_truncated=bool(value.get("output_truncated", False)),
    )


def _executable_identity(path: Path, maximum_bytes: int) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > maximum_bytes:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


class HyperpropertyEngineAdapter:
    """Bounded executable probe for one hyperproperty engine."""

    engine: HyperpropertyEngine
    executable_candidates: tuple[str, ...]
    fixture: EngineConformanceFixture

    def __init__(
        self,
        *,
        which: ExecutableFinder | None = None,
        command_runner: CommandRunner | None = None,
        timeout_seconds: float = DEFAULT_ENGINE_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_ENGINE_OUTPUT_BYTES,
        max_executable_bytes: int = DEFAULT_MAX_EXECUTABLE_BYTES,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise HyperpropertyValidationError("timeout_seconds must be positive")
        self.which = which or shutil.which
        self.command_runner = command_runner or _default_command_runner
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = _positive_int(max_output_bytes, "max_output_bytes")
        self.max_executable_bytes = _positive_int(
            max_executable_bytes, "max_executable_bytes"
        )
        self.monotonic = monotonic or time.monotonic

    def _discover(self) -> str | None:
        for candidate in self.executable_candidates:
            try:
                found = self.which(candidate)
            except BaseException:
                continue
            if found:
                return str(Path(found).resolve())
        return None

    def _run(self, request: CommandRequest) -> CommandResult:
        try:
            return _command_result(self.command_runner(request))
        except BaseException as exc:
            return CommandResult(returncode=None, error=type(exc).__name__)

    def probe(self, *, run_conformance: bool = True) -> EngineCapability:
        executable = self._discover()
        if executable is None:
            return EngineCapability(
                engine=self.engine,
                status=EngineCapabilityStatus.UNAVAILABLE,
                reason="executable not discovered",
            )
        executable_identity = _executable_identity(
            Path(executable), self.max_executable_bytes
        )
        if executable_identity is None:
            return EngineCapability(
                engine=self.engine,
                status=EngineCapabilityStatus.UNAVAILABLE,
                executable_path=executable,
                reason="executable identity could not be bounded and pinned",
            )

        version_result = self._run(
            CommandRequest(
                command=(executable, "--version"),
                stdin_text=None,
                cwd=None,
                timeout_seconds=min(2.0, self.timeout_seconds),
                max_output_bytes=min(4096, self.max_output_bytes),
            )
        )
        version_output = (version_result.stdout + "\n" + version_result.stderr).strip()
        if (
            version_result.returncode != 0
            or version_result.timed_out
            or not version_output
        ):
            return EngineCapability(
                engine=self.engine,
                status=EngineCapabilityStatus.UNAVAILABLE,
                executable_path=executable,
                reason="executable did not provide a bounded version",
            )
        version = version_output.splitlines()[0][:256]
        if not run_conformance:
            return EngineCapability(
                engine=self.engine,
                status=EngineCapabilityStatus.UNAVAILABLE,
                executable_path=executable,
                executable_version=version,
                reason="executable discovered but conformance fixture was not run",
            )

        receipt = self._run_conformance(
            executable, executable_identity, version
        )
        return EngineCapability(
            engine=self.engine,
            status=(
                EngineCapabilityStatus.CONFORMANT
                if receipt.passed
                else EngineCapabilityStatus.UNAVAILABLE
            ),
            executable_path=executable,
            executable_version=version,
            conformance_receipt=receipt,
            reason=receipt.reason,
        )

    # Compatibility spelling used by other capability probes.
    capability = probe
    probe_capability = probe
    check_capability = probe

    def render_model(self, model: HyperpropertyModel) -> str:
        """Render the reviewed HyperLTL formula without weakening its policy.

        System-format translation is engine specific and deliberately remains
        outside this method.  The returned model is exactly the formula whose
        content identity binds ``model.observation_policy``.
        """

        if not isinstance(model, HyperpropertyModel):
            raise HyperpropertyValidationError("model must be a HyperpropertyModel")
        return model.hyperltl_formula + (
            "" if model.hyperltl_formula.endswith("\n") else "\n"
        )

    def _run_conformance(
        self, executable: str, executable_identity: str, version: str
    ) -> EngineConformanceReceipt:
        started = self.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"{self.engine.value}-conformance-") as raw:
            fixture_path = Path(raw) / self.fixture.file_name
            fixture_path.write_text(self.fixture.model_text, encoding="utf-8")
            auxiliary_paths: dict[str, str] = {}
            for name, text in self.fixture.auxiliary_files.items():
                auxiliary_path = Path(raw) / name
                auxiliary_path.write_text(text, encoding="utf-8")
                auxiliary_paths[f"{{auxiliary:{name}}}"] = str(auxiliary_path)
            command = (executable,) + tuple(
                (
                    str(fixture_path)
                    if item == "{fixture}"
                    else auxiliary_paths.get(item, item)
                )
                for item in self.fixture.args
            )
            result = self._run(
                CommandRequest(
                    command=command,
                    stdin_text=None,
                    cwd=raw,
                    timeout_seconds=self.timeout_seconds,
                    max_output_bytes=self.max_output_bytes,
                )
            )
        duration_ms = max(0, round((self.monotonic() - started) * 1000))
        output = (result.stdout + "\n" + result.stderr).encode(
            "utf-8", errors="replace"
        )
        folded = output.decode("utf-8", errors="replace").casefold()
        marker = (
            result.returncode == 0
            and not result.timed_out
            and any(value in folded for value in self.fixture.expected_output_any)
        )
        if marker:
            status = ConformanceStatus.PASSED
            reason = "executable conformance fixture passed"
        elif result.timed_out:
            status = ConformanceStatus.TIMED_OUT
            reason = "executable conformance fixture timed out"
        elif result.error:
            status = ConformanceStatus.ERROR
            reason = "executable conformance fixture could not execute"
        else:
            status = ConformanceStatus.FAILED
            reason = "executable conformance fixture did not establish expected semantics"
        return EngineConformanceReceipt(
            engine=self.engine,
            status=status,
            fixture_id=self.fixture.fixture_id,
            fixture_identity=self.fixture.content_id,
            executable_path=executable,
            executable_identity=executable_identity,
            executable_version=version,
            command_identity=content_identity({"command": command}),
            returncode=result.returncode,
            timed_out=result.timed_out,
            output_sha256="sha256:" + hashlib.sha256(output).hexdigest(),
            marker_matched=marker,
            duration_ms=duration_ms,
            reason=reason,
        )


_CONFORMANCE_FORMULA = (
    'forall A. forall B. G (({"low"_A = "low"_B}) -> '
    '({"obs"_A = "obs"_B}))'
)

_AUTOHYPER_CONFORMANCE_SYSTEM = """\
Variables: ("low" Bool) ("obs" Bool)
Init: 0 1
--BODY--
State: 0 {("low" false) ("obs" false)}
0
State: 1 {("low" true) ("obs" true)}
1
--END--
"""


class HyperLTLAdapter(HyperpropertyEngineAdapter):
    engine = HyperpropertyEngine.HYPERLTL
    executable_candidates = ("hyperltl",)
    fixture = EngineConformanceFixture(
        engine=engine,
        fixture_id="supervisor-hyperltl-noninterference@1",
        model_text=_CONFORMANCE_FORMULA + "\n",
        file_name="conformance.hltl",
        args=("{fixture}",),
        expected_output_any=("holds", "verified", "true"),
        translator_id="supervisor-hyperltl@1",
        semantic_profile_id="two-trace-noninterference@1",
    )


class AutoHyperAdapter(HyperpropertyEngineAdapter):
    engine = HyperpropertyEngine.AUTOHYPER
    executable_candidates = ("AutoHyper", "autohyper")
    fixture = EngineConformanceFixture(
        engine=engine,
        fixture_id="supervisor-autohyper-noninterference@1",
        model_text=_CONFORMANCE_FORMULA + "\n",
        file_name="conformance.hltl",
        args=("--explicit", "{auxiliary:conformance.explicit}", "{fixture}"),
        expected_output_any=("sat",),
        translator_id="supervisor-autohyper@1",
        semantic_profile_id="two-trace-noninterference@1",
        auxiliary_files={"conformance.explicit": _AUTOHYPER_CONFORMANCE_SYSTEM},
    )


class MCHyperAdapter(HyperpropertyEngineAdapter):
    engine = HyperpropertyEngine.MCHYPER
    executable_candidates = ("mchyper",)
    fixture = EngineConformanceFixture(
        engine=engine,
        fixture_id="supervisor-mchyper-noninterference@1",
        model_text=_CONFORMANCE_FORMULA + "\n",
        file_name="conformance.hltl",
        args=("{fixture}",),
        expected_output_any=("holds", "verified", "true"),
        translator_id="supervisor-mchyper@1",
        semantic_profile_id="two-trace-noninterference@1",
    )


DEFAULT_ENGINE_ADAPTER_TYPES = (
    HyperLTLAdapter,
    AutoHyperAdapter,
    MCHyperAdapter,
)


def probe_hyperproperty_engines(
    adapters: Sequence[HyperpropertyEngineAdapter] | None = None,
    *,
    run_conformance: bool = True,
) -> tuple[EngineCapability, ...]:
    """Probe every engine independently; discovery never implies availability."""

    selected = tuple(adapters) if adapters is not None else tuple(
        adapter_type() for adapter_type in DEFAULT_ENGINE_ADAPTER_TYPES
    )
    capabilities = tuple(
        adapter.probe(run_conformance=run_conformance) for adapter in selected
    )
    engines = [item.engine for item in capabilities]
    if len(engines) != len(set(engines)):
        raise HyperpropertyValidationError("engine adapters must be unique")
    return capabilities


class BoundedSelfCompositionChecker:
    """Deterministically compare low-equivalent traces under explicit bounds."""

    def __init__(
        self,
        *,
        max_traces: int = DEFAULT_MAX_COMPOSITION_TRACES,
        max_pairs: int = DEFAULT_MAX_COMPOSITION_PAIRS,
    ) -> None:
        self.max_traces = _positive_int(max_traces, "max_traces")
        self.max_pairs = _positive_int(max_pairs, "max_pairs")

    def check(
        self,
        model: HyperpropertyModel,
        traces: Sequence[Hypertrace],
    ) -> HyperpropertyVerificationResult:
        if not isinstance(model, HyperpropertyModel):
            raise HyperpropertyValidationError("model must be a HyperpropertyModel")
        values = tuple(traces)
        if any(not isinstance(item, Hypertrace) for item in values):
            raise HyperpropertyValidationError("traces must contain Hypertrace values")
        selected = tuple(sorted(values, key=lambda item: item.public_ref))[
            : self.max_traces
        ]
        policy = model.observation_policy
        pairs = 0
        eligible_pairs = 0
        possible_pairs = len(selected) * max(0, len(selected) - 1) // 2
        bound_hit = len(values) > len(selected) or possible_pairs > self.max_pairs
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                if pairs >= self.max_pairs:
                    bound_hit = True
                    break
                pairs += 1
                if left.subject_projection(policy.subject_fields) != right.subject_projection(
                    policy.subject_fields
                ):
                    continue
                if _projection(
                    left.public_inputs, policy.low_input_fields
                ) != _projection(right.public_inputs, policy.low_input_fields):
                    continue
                if left.private_inputs == right.private_inputs:
                    continue
                eligible_pairs += 1
                differences = tuple(
                    ObservationDifference(
                        field=field_name,
                        left_digest=_digest(_path_value(left.observations, field_name)),
                        right_digest=_digest(_path_value(right.observations, field_name)),
                    )
                    for field_name in policy.observation_fields
                    if _path_value(left.observations, field_name)
                    != _path_value(right.observations, field_name)
                )
                if differences:
                    # One approved differing observation is sufficient to
                    # witness the violation.  Keeping only the lexicographically
                    # first delta gives a deterministic 1-minimal explanation.
                    differences = differences[:1]
                    counterexample = HypertraceCounterexample(
                        model_id=model.model_id,
                        model_identity=model.content_id,
                        observation_policy_id=policy.content_id,
                        observation_policy_name=policy.policy_id,
                        observed_fields=policy.observation_fields,
                        trace_refs=(left.public_ref, right.public_ref),
                        differences=differences,
                    )
                    return HyperpropertyVerificationResult(
                        model_id=model.model_id,
                        model_identity=model.content_id,
                        observation_policy_id=policy.content_id,
                        verdict=HyperpropertyVerdict.VIOLATED,
                        evidence_kind=HyperpropertyEvidenceKind.BOUNDED_SELF_COMPOSITION,
                        authoritative=False,
                        bounded=True,
                        explored_traces=len(selected),
                        explored_pairs=pairs,
                        maximum_pairs=self.max_pairs,
                        reason="bounded self-composition found a low-observable difference",
                        counterexample=counterexample,
                    )
            if pairs >= self.max_pairs:
                break

        if eligible_pairs == 0:
            verdict = HyperpropertyVerdict.INCONCLUSIVE
            reason = "no low-equivalent trace pair varied a private input"
        elif bound_hit:
            verdict = HyperpropertyVerdict.INCONCLUSIVE
            reason = "no violation observed before a self-composition bound was reached"
        else:
            verdict = HyperpropertyVerdict.HOLDS
            reason = (
                "all bounded low-equivalent trace pairs preserved approved observations"
            )
        return HyperpropertyVerificationResult(
            model_id=model.model_id,
            model_identity=model.content_id,
            observation_policy_id=policy.content_id,
            verdict=verdict,
            evidence_kind=HyperpropertyEvidenceKind.BOUNDED_SELF_COMPOSITION,
            authoritative=False,
            bounded=True,
            explored_traces=len(selected),
            explored_pairs=pairs,
            maximum_pairs=self.max_pairs,
            reason=reason,
        )


def bounded_self_composition(
    model: HyperpropertyModel,
    traces: Sequence[Hypertrace],
    *,
    max_traces: int = DEFAULT_MAX_COMPOSITION_TRACES,
    max_pairs: int = DEFAULT_MAX_COMPOSITION_PAIRS,
) -> HyperpropertyVerificationResult:
    """Functional entry point for the non-authoritative fallback."""

    return BoundedSelfCompositionChecker(
        max_traces=max_traces, max_pairs=max_pairs
    ).check(model, traces)


class HyperpropertyVerifier:
    """Capability-gated verifier with deterministic fallback evidence.

    This coordinator does not turn an engine's presence into a proof.  Engine
    execution is intentionally a separate integration boundary; until such a
    result is supplied by a conformant adapter, trace checking remains bounded
    fallback evidence.
    """

    def __init__(
        self,
        adapters: Sequence[HyperpropertyEngineAdapter] | None = None,
        *,
        fallback_checker: BoundedSelfCompositionChecker | None = None,
    ) -> None:
        self.adapters = (
            tuple(adapters)
            if adapters is not None
            else tuple(adapter_type() for adapter_type in DEFAULT_ENGINE_ADAPTER_TYPES)
        )
        self.fallback_checker = fallback_checker or BoundedSelfCompositionChecker()

    def capabilities(
        self, *, run_conformance: bool = True
    ) -> tuple[EngineCapability, ...]:
        return probe_hyperproperty_engines(
            self.adapters, run_conformance=run_conformance
        )

    def verify(
        self,
        model: HyperpropertyModel,
        traces: Sequence[Hypertrace],
        *,
        capabilities: Sequence[EngineCapability] | None = None,
    ) -> HyperpropertyVerificationResult:
        # Even a conformant engine requires a separately executed model-check
        # receipt.  This trace-oriented API therefore always provides the
        # documented bounded fallback, while capability reports tell a caller
        # whether it may route a full transition-system model to an adapter.
        if capabilities is not None:
            for capability in capabilities:
                if not isinstance(capability, EngineCapability):
                    raise HyperpropertyValidationError(
                        "capabilities must contain EngineCapability values"
                    )
        return self.fallback_checker.check(model, traces)


def _formula_variable(field_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", field_name)


def _default_formula(
    low_inputs: Sequence[str], observations: Sequence[str]
) -> str:
    def equal_across_traces(field_name: str) -> str:
        variable = _formula_variable(field_name)
        return f'{{"{variable}"_pi1 = "{variable}"_pi2}}'

    low = " & ".join(equal_across_traces(field) for field in low_inputs) or "1"
    observed = " & ".join(
        equal_across_traces(field) for field in observations
    )
    return f"forall pi1. forall pi2. G((({low}) -> ({observed})))"


def _default_model(
    kind: HyperpropertyKind,
    observations: tuple[str, ...],
    description: str,
    *,
    low_inputs: tuple[str, ...] = ("request.operation", "request.policy_id"),
) -> HyperpropertyModel:
    policy = ObservationPolicy(
        policy_id=f"supervisor.{kind.value}.observations",
        version="1",
        low_input_fields=low_inputs,
        observation_fields=observations,
        subject_fields=("task_id",),
        description=f"Approved low observations for {kind.value}.",
    )
    return HyperpropertyModel(
        model_id=f"supervisor.{kind.value}",
        version="1",
        kind=kind,
        observation_policy=policy,
        hyperltl_formula=_default_formula(low_inputs, observations),
        description=description,
        assumptions=(
            "trace subjects use canonical task identities",
            "private inputs vary independently of reviewed low inputs",
        ),
    )


DEFAULT_HYPERPROPERTY_MODELS: tuple[HyperpropertyModel, ...] = (
    _default_model(
        HyperpropertyKind.PROMPT_ISOLATION,
        ("prompt.text_digest", "prompt.capsule_id"),
        "Unrelated lane prompts and transcripts cannot affect a task prompt.",
    ),
    _default_model(
        HyperpropertyKind.WORKTREE_ISOLATION,
        ("worktree.visible_tree_id", "worktree.allowed_path_digest"),
        "Changes in another worktree cannot affect the subject worktree view.",
    ),
    _default_model(
        HyperpropertyKind.LOG_REDACTION,
        ("log.public_digest", "log.redaction_shape"),
        "Secrets and private lane values cannot affect public log observations.",
    ),
    _default_model(
        HyperpropertyKind.PROVIDER_ROUTING,
        ("routing.provider_id", "routing.isolation_mode"),
        "Unrelated private inputs cannot influence provider selection or isolation.",
        low_inputs=(
            "request.operation",
            "request.policy_id",
            "request.property_kind",
        ),
    ),
    _default_model(
        HyperpropertyKind.ZKP_WITNESS_NONINTERFERENCE,
        ("zkp.public_statement_id", "zkp.verification_verdict", "log.public_digest"),
        "A private ZKP witness cannot influence non-approved public channels.",
        low_inputs=("request.operation", "request.policy_id", "zkp.statement_id"),
    ),
    _default_model(
        HyperpropertyKind.CROSS_TASK_CACHE_SEPARATION,
        ("cache.result_id", "cache.hit", "cache.namespace_id"),
        "Another task's cache contents cannot influence a subject task lookup.",
        low_inputs=(
            "request.operation",
            "request.policy_id",
            "cache.key",
            "cache.scope_id",
        ),
    ),
)

DEFAULT_HYPERPROPERTY_MODELS_BY_KIND: Mapping[
    HyperpropertyKind, HyperpropertyModel
] = {model.kind: model for model in DEFAULT_HYPERPROPERTY_MODELS}

PROMPT_ISOLATION_MODEL = DEFAULT_HYPERPROPERTY_MODELS_BY_KIND[
    HyperpropertyKind.PROMPT_ISOLATION
]
WORKTREE_ISOLATION_MODEL = DEFAULT_HYPERPROPERTY_MODELS_BY_KIND[
    HyperpropertyKind.WORKTREE_ISOLATION
]
LOG_REDACTION_MODEL = DEFAULT_HYPERPROPERTY_MODELS_BY_KIND[
    HyperpropertyKind.LOG_REDACTION
]
PROVIDER_ROUTING_MODEL = DEFAULT_HYPERPROPERTY_MODELS_BY_KIND[
    HyperpropertyKind.PROVIDER_ROUTING
]
ZKP_WITNESS_NONINTERFERENCE_MODEL = DEFAULT_HYPERPROPERTY_MODELS_BY_KIND[
    HyperpropertyKind.ZKP_WITNESS_NONINTERFERENCE
]
CROSS_TASK_CACHE_SEPARATION_MODEL = DEFAULT_HYPERPROPERTY_MODELS_BY_KIND[
    HyperpropertyKind.CROSS_TASK_CACHE_SEPARATION
]


def default_hyperproperty_models() -> tuple[HyperpropertyModel, ...]:
    """Return all six reviewed versioned models."""

    return DEFAULT_HYPERPROPERTY_MODELS


def model_for(kind: HyperpropertyKind | str) -> HyperpropertyModel:
    """Return the reviewed model for ``kind``."""

    normalized = _enum(kind, HyperpropertyKind, "hyperproperty kind")
    return DEFAULT_HYPERPROPERTY_MODELS_BY_KIND[normalized]


# Compatibility names that keep external API intent obvious.
HyperpropertyResult = HyperpropertyVerificationResult
CounterexampleHypertrace = HypertraceCounterexample
SelfCompositionChecker = BoundedSelfCompositionChecker
HyperLTLEngineAdapter = HyperLTLAdapter
AutoHyperEngineAdapter = AutoHyperAdapter
MCHyperEngineAdapter = MCHyperAdapter
verify_hyperproperty = bounded_self_composition


__all__ = [
    "AutoHyperAdapter",
    "AutoHyperEngineAdapter",
    "BoundedSelfCompositionChecker",
    "ConformanceStatus",
    "CounterexampleHypertrace",
    "CROSS_TASK_CACHE_SEPARATION_MODEL",
    "DEFAULT_ENGINE_ADAPTER_TYPES",
    "DEFAULT_ENGINE_TIMEOUT_SECONDS",
    "DEFAULT_HYPERPROPERTY_MODELS",
    "DEFAULT_HYPERPROPERTY_MODELS_BY_KIND",
    "DEFAULT_MAX_COMPOSITION_PAIRS",
    "DEFAULT_MAX_COMPOSITION_TRACES",
    "EngineCapability",
    "EngineCapabilityStatus",
    "EngineConformanceFixture",
    "EngineConformanceReceipt",
    "EngineKind",
    "ExecutionTrace",
    "HYPERPROPERTY_VERIFICATION_VERSION",
    "HyperLTLAdapter",
    "HyperLTLEngineAdapter",
    "HyperpropertyEngine",
    "HyperpropertyEngineAdapter",
    "HyperpropertyEvidenceKind",
    "HyperpropertyKind",
    "HyperpropertyModel",
    "HyperpropertyResult",
    "HyperpropertyValidationError",
    "HyperpropertyVerdict",
    "HyperpropertyVerificationResult",
    "HyperpropertyVerifier",
    "Hypertrace",
    "HypertraceCounterexample",
    "MCHyperAdapter",
    "MCHyperEngineAdapter",
    "LOG_REDACTION_MODEL",
    "ObservationDifference",
    "ObservationPolicy",
    "PROMPT_ISOLATION_MODEL",
    "PROVIDER_ROUTING_MODEL",
    "SelfCompositionChecker",
    "WORKTREE_ISOLATION_MODEL",
    "ZKP_WITNESS_NONINTERFERENCE_MODEL",
    "bounded_self_composition",
    "default_hyperproperty_models",
    "model_for",
    "probe_hyperproperty_engines",
    "verify_hyperproperty",
]
