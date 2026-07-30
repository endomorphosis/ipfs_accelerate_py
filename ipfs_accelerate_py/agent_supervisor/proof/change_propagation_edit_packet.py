"""Fail-closed multi-edit packets bound to one admitted atomic propagation plan.

``AtomicPropagationPlan@1`` / ``PropagationPlanAdmission`` grant the only
mutation authority for transitive change propagation.  This module turns a
*current, admitted, non-abstaining* plan into a compact provider hand-off that:

* partitions deterministic analytical steps from behavior-complete
  model-required (``llm_bounded``) steps;
* binds roots, plan/SCC/dependency order, exact read/write allowlists,
  before hashes, selected value sources, required behavior, minimal
  counterexamples, proof/index/graph refs, unsupported limits, per-edit and
  fixed-point postconditions, focused commands, and bounded expansion
  handles; and
* refuses alternatives, source/proof bodies, secrets, unknown semantics,
  stale/forged/partial/abstaining plans, and path mismatches.

The packet never embeds source, AST, solver, or proof bodies.  Expansion
handles are content-addressed pointers only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import (
    AnalyticalTransform,
    AtomicPropagationPlan,
    PlanDisposition,
    PlanStepKind,
    PropagationAuthorityRoots,
    PropagationPlanStep,
    PropagationSCCGroup,
    RequiredBehaviorContract,
    TransformDisposition,
)
from ..planning.change_propagation_plan import (
    PlanEvidenceBundle,
    PlanPathSpan,
    PlanValidationCommand,
    PropagationPlanAdmission,
)
from .missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)
from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)


# ---------------------------------------------------------------------------
# Schema / bounds
# ---------------------------------------------------------------------------

CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE: Final[str] = "ChangePropagationEditPacket@1"
CHANGE_PROPAGATION_EDIT_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-edit-packet@1"
)
CHANGE_PROPAGATION_EDIT_PACKET_VERSION: Final[int] = 1
PROPAGATION_EDIT_STEP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-edit-step@1"
)
PROPAGATION_EXPANSION_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-expansion-handle@1"
)
PATH_BEFORE_HASH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-path-before-hash@1"
)
SELECTED_VALUE_SOURCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-selected-value-source@1"
)

PRODUCER_ID: Final[str] = "change-propagation-edit-packet@1"

MAX_PACKET_BYTES: Final[int] = 262_144
MAX_STEPS: Final[int] = 512
MAX_PATHS: Final[int] = 1_024
MAX_REFERENCES: Final[int] = 1_024
MAX_HANDLES: Final[int] = 128
MAX_COMMANDS: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024

_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "ast_nodes",
        "body",
        "code",
        "contents",
        "content",
        "file_content",
        "file_contents",
        "file_text",
        "full_receipt",
        "gold_ir",
        "gold_ir_body",
        "kernel_proof_body",
        "lean_source",
        "private_witness",
        "proof_body",
        "proof_text",
        "raw_ast",
        "receipt_body",
        "secret",
        "secrets",
        "snippet",
        "solver_trace",
        "source",
        "source_body",
        "source_text",
        "witness",
    }
)

_FORBIDDEN_HANDLE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "source",
        "source_body",
        "proof_body",
        "ast_body",
        "secret",
        "secrets",
        "witness",
        "private_witness",
    }
)


# ---------------------------------------------------------------------------
# Errors and closed rejection vocabulary
# ---------------------------------------------------------------------------


class ChangePropagationEditPacketError(ContractValidationError):
    """A packet would weaken the admitted plan authority boundary."""


class ChangePropagationEditPacketReason(str, Enum):
    """Stable, machine-readable materialization failures."""

    NOT_CURRENT = "not_current"
    NOT_ADMITTED = "not_admitted"
    ABSTAINING = "abstaining"
    PARTIAL = "partial"
    STALE_ROOTS = "stale_roots"
    FORGED = "forged"
    PATH_MISMATCH = "path_mismatch"
    SCOPE_BROADENING = "scope_broadening"
    FORBIDDEN_BODY = "forbidden_body"
    UNKNOWN_SEMANTICS = "unknown_semantics"
    ALTERNATIVES = "alternatives"
    MALFORMED = "malformed"
    MISSING_BEHAVIOR = "missing_behavior"
    MISSING_VALUE = "missing_value"


class PropagationEditStepKind(str, Enum):
    """Packet-level partition of plan steps."""

    ANALYTICAL = "analytical"
    MODEL_REQUIRED = "model_required"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"


def _plan_step_kind_to_edit_kind(kind: PlanStepKind) -> PropagationEditStepKind:
    if kind is PlanStepKind.ANALYTICAL:
        return PropagationEditStepKind.ANALYTICAL
    if kind is PlanStepKind.LLM_BOUNDED:
        return PropagationEditStepKind.MODEL_REQUIRED
    if kind is PlanStepKind.VALIDATION:
        return PropagationEditStepKind.VALIDATION
    if kind is PlanStepKind.CHECKPOINT:
        return PropagationEditStepKind.CHECKPOINT
    if kind is PlanStepKind.ROLLBACK:
        return PropagationEditStepKind.ROLLBACK
    raise ChangePropagationEditPacketError(
        f"unknown plan step kind cannot materialize: {kind}"
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ChangePropagationEditPacketError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ChangePropagationEditPacketError(f"{name} is required")
    if "\x00" in result:
        raise ChangePropagationEditPacketError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > limit:
        raise ChangePropagationEditPacketError(f"{name} exceeds its byte bound")
    return result


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True, limit=MAX_TEXT_BYTES)
    if any(char.isspace() for char in result):
        raise ChangePropagationEditPacketError(f"{name} must be a compact identifier")
    return result


def _path(value: Any, name: str = "path") -> str:
    raw = _text(value, name, required=True, limit=MAX_PATH_BYTES).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or raw in {".", ""}
        or raw.startswith("./")
        or any(char in raw for char in "*?[]{}")
    ):
        raise ChangePropagationEditPacketError(
            f"{name} must be an exact repository-relative path"
        )
    if raw != candidate.as_posix():
        raise ChangePropagationEditPacketError(
            f"{name} must be a normalized repository-relative path"
        )
    return raw


def _paths(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_PATHS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationEditPacketError(f"{name} must be a sequence of paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ChangePropagationEditPacketError(f"{name} must not be empty")
    if len(ordered) > limit:
        raise ChangePropagationEditPacketError(f"{name} exceeds path bound")
    return tuple(ordered)


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCES,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationEditPacketError(f"{name} must be a sequence of identifiers")
    if preserve_order:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in values:
            ident = _identifier(item, name)
            if ident not in seen:
                seen.add(ident)
                ordered.append(ident)
        result = tuple(ordered)
    else:
        result = tuple(sorted({_identifier(item, name) for item in values}))
    if required and not result:
        raise ChangePropagationEditPacketError(f"{name} must not be empty")
    if len(result) > limit:
        raise ChangePropagationEditPacketError(f"{name} exceeds identifier bound")
    return result


def _commands(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ChangePropagationEditPacketError(f"{name} must be a sequence of commands")
    result = tuple(
        sorted({_text(value, name, limit=MAX_TEXT_BYTES) for value in values})
    )
    if not result:
        raise ChangePropagationEditPacketError(f"{name} must not be empty")
    if len(result) > MAX_COMMANDS or any("\n" in item or "\r" in item for item in result):
        raise ChangePropagationEditPacketError(
            f"{name} must contain bounded one-line commands"
        )
    return result


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ChangePropagationEditPacketError(f"unknown {name}: {value}") from exc
    raise ChangePropagationEditPacketError(f"{name} must be a {enum_cls.__name__}")


def _roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            PropagationAuthorityRoots.from_dict(value)
            if "schema" in value
            else PropagationAuthorityRoots(**value)
        )
    raise ChangePropagationEditPacketError("roots must be PropagationAuthorityRoots")


def _reject_forbidden_keys(payload: Mapping[str, Any], *, where: str) -> None:
    for key in payload:
        normalized = str(key).casefold().replace("-", "_")
        if normalized in _FORBIDDEN_BODY_KEYS or "secret" in normalized:
            raise ChangePropagationEditPacketError(
                f"{where} cannot embed {key} (forbidden body/secret)"
            )


# ---------------------------------------------------------------------------
# Nested packet records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class PathBeforeHash:
    """Exact path identity plus the pre-edit content hash from plan spans."""

    path: str
    before_hash: str
    artifact_id: str = ""
    schema: str = PATH_BEFORE_HASH_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(
            self, "before_hash", _text(self.before_hash, "before_hash", required=False)
        )
        object.__setattr__(
            self, "artifact_id", _text(self.artifact_id, "artifact_id", required=False)
        )
        if self.schema != PATH_BEFORE_HASH_SCHEMA:
            raise ChangePropagationEditPacketError("unsupported path before-hash schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "before_hash": self.before_hash,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathBeforeHash":
        if not isinstance(payload, Mapping):
            raise ChangePropagationEditPacketError("path before-hash must be an object")
        _reject_forbidden_keys(payload, where="path before-hash")
        allowed = {"schema", "path", "before_hash", "artifact_id"}
        if set(payload).difference(allowed):
            raise ChangePropagationEditPacketError(
                "path before-hash contains unsupported fields"
            )
        return cls(
            path=payload.get("path", ""),
            before_hash=payload.get("before_hash", ""),
            artifact_id=payload.get("artifact_id", ""),
            schema=payload.get("schema", PATH_BEFORE_HASH_SCHEMA),
        )


@dataclass(frozen=True, order=True)
class SelectedValueSource:
    """Compact proved value mapping projection (never alternatives)."""

    requirement_id: str
    consumer_id: str
    candidate_id: str
    expression_ref: str = ""
    type_ref: str = ""
    proof_id: str = ""
    schema: str = SELECTED_VALUE_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_id", _identifier(self.requirement_id, "requirement_id")
        )
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "expression_ref", _text(self.expression_ref, "expression_ref", required=False)
        )
        object.__setattr__(
            self, "type_ref", _text(self.type_ref, "type_ref", required=False)
        )
        object.__setattr__(
            self, "proof_id", _text(self.proof_id, "proof_id", required=False)
        )
        if self.schema != SELECTED_VALUE_SOURCE_SCHEMA:
            raise ChangePropagationEditPacketError(
                "unsupported selected value source schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requirement_id": self.requirement_id,
            "consumer_id": self.consumer_id,
            "candidate_id": self.candidate_id,
            "expression_ref": self.expression_ref,
            "type_ref": self.type_ref,
            "proof_id": self.proof_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectedValueSource":
        if not isinstance(payload, Mapping):
            raise ChangePropagationEditPacketError(
                "selected value source must be an object"
            )
        _reject_forbidden_keys(payload, where="selected value source")
        allowed = {
            "schema",
            "requirement_id",
            "consumer_id",
            "candidate_id",
            "expression_ref",
            "type_ref",
            "proof_id",
        }
        if set(payload).difference(allowed):
            raise ChangePropagationEditPacketError(
                "selected value source contains unsupported fields"
            )
        return cls(
            requirement_id=payload["requirement_id"],
            consumer_id=payload["consumer_id"],
            candidate_id=payload["candidate_id"],
            expression_ref=payload.get("expression_ref", ""),
            type_ref=payload.get("type_ref", ""),
            proof_id=payload.get("proof_id", ""),
            schema=payload.get("schema", SELECTED_VALUE_SOURCE_SCHEMA),
        )


@dataclass(frozen=True, order=True)
class PropagationExpansionHandle:
    """Bounded pointer to more evidence; never contains the evidence body."""

    handle_id: str
    kind: str
    reference_id: str
    permitted_paths: tuple[str, ...] = ()
    schema: str = PROPAGATION_EXPANSION_HANDLE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "handle_id", _identifier(self.handle_id, "handle_id"))
        object.__setattr__(self, "kind", _identifier(self.kind, "handle.kind"))
        kind_norm = self.kind.casefold().replace("-", "_")
        if kind_norm in _FORBIDDEN_HANDLE_KINDS:
            raise ChangePropagationEditPacketError(
                "expansion handles may not name embedded bodies or secrets"
            )
        object.__setattr__(
            self, "reference_id", _identifier(self.reference_id, "handle.reference_id")
        )
        object.__setattr__(
            self,
            "permitted_paths",
            _paths(self.permitted_paths, "handle.permitted_paths", required=False),
        )
        if self.schema != PROPAGATION_EXPANSION_HANDLE_SCHEMA:
            raise ChangePropagationEditPacketError(
                "unsupported expansion handle schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "handle_id": self.handle_id,
            "kind": self.kind,
            "reference_id": self.reference_id,
            "permitted_paths": list(self.permitted_paths),
            "body_embedded": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationExpansionHandle":
        if not isinstance(payload, Mapping):
            raise ChangePropagationEditPacketError(
                "expansion handle must be an object"
            )
        _reject_forbidden_keys(payload, where="expansion handle")
        allowed = {
            "schema",
            "handle_id",
            "kind",
            "reference_id",
            "permitted_paths",
            "body_embedded",
        }
        if set(payload).difference(allowed):
            raise ChangePropagationEditPacketError(
                "expansion handle contains unsupported fields"
            )
        if payload.get("body_embedded", False) is not False:
            raise ChangePropagationEditPacketError(
                "expansion handle cannot embed a body"
            )
        return cls(
            handle_id=payload.get("handle_id"),
            kind=payload.get("kind"),
            reference_id=payload.get("reference_id"),
            permitted_paths=tuple(payload.get("permitted_paths", ())),
            schema=payload.get("schema", PROPAGATION_EXPANSION_HANDLE_SCHEMA),
        )


@dataclass(frozen=True)
class PropagationEditStep(CanonicalContract):
    """One plan-bound edit step projected into the multi-edit packet."""

    SCHEMA: ClassVar[str] = PROPAGATION_EDIT_STEP_SCHEMA

    step_id: str
    kind: PropagationEditStepKind
    plan_step_kind: PlanStepKind
    obligation_ids: tuple[str, ...]
    dependency_step_ids: tuple[str, ...] = ()
    scc_group_id: str = ""
    transform_id: str = ""
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    before_hashes: tuple[PathBeforeHash, ...] = ()
    selected_value_sources: tuple[SelectedValueSource, ...] = ()
    required_behavior_ids: tuple[str, ...] = ()
    counterexample_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    precondition_refs: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    unsupported_limits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, PropagationEditStepKind, "kind")
        )
        object.__setattr__(
            self,
            "plan_step_kind",
            _enum(self.plan_step_kind, PlanStepKind, "plan_step_kind"),
        )
        expected_kind = _plan_step_kind_to_edit_kind(self.plan_step_kind)
        if self.kind is not expected_kind:
            raise ChangePropagationEditPacketError(
                "edit step kind must partition the plan step kind exactly"
            )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", required=True),
        )
        object.__setattr__(
            self,
            "dependency_step_ids",
            _ids(
                self.dependency_step_ids,
                "dependency_step_ids",
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self, "scc_group_id", _text(self.scc_group_id, "scc_group_id", required=False)
        )
        object.__setattr__(
            self, "transform_id", _text(self.transform_id, "transform_id", required=False)
        )
        object.__setattr__(self, "read_paths", _paths(self.read_paths, "read_paths"))
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        if self.kind is PropagationEditStepKind.ANALYTICAL and not self.transform_id:
            raise ChangePropagationEditPacketError(
                "analytical edit steps require a transform_id"
            )
        if self.kind is PropagationEditStepKind.MODEL_REQUIRED and not self.write_paths:
            raise ChangePropagationEditPacketError(
                "model-required steps require exact write path authority"
            )
        if not isinstance(self.before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.before_hashes
        ):
            raise ChangePropagationEditPacketError(
                "before_hashes must be PathBeforeHash values"
            )
        hashes = tuple(sorted(self.before_hashes, key=lambda item: item.path))
        if len({item.path for item in hashes}) != len(hashes):
            raise ChangePropagationEditPacketError(
                "before_hashes must have unique paths"
            )
        for item in hashes:
            if item.path not in set(self.read_paths) | set(self.write_paths):
                raise ChangePropagationEditPacketError(
                    "before_hash path must stay inside step read/write allowlists"
                )
        object.__setattr__(self, "before_hashes", hashes)
        if not isinstance(self.selected_value_sources, Sequence) or not all(
            isinstance(item, SelectedValueSource) for item in self.selected_value_sources
        ):
            raise ChangePropagationEditPacketError(
                "selected_value_sources must be SelectedValueSource values"
            )
        values = tuple(
            sorted(
                self.selected_value_sources,
                key=lambda item: (item.requirement_id, item.candidate_id),
            )
        )
        object.__setattr__(self, "selected_value_sources", values)
        object.__setattr__(
            self,
            "required_behavior_ids",
            _ids(self.required_behavior_ids, "required_behavior_ids"),
        )
        object.__setattr__(
            self, "counterexample_refs", _ids(self.counterexample_refs, "counterexample_refs")
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "precondition_refs", _ids(self.precondition_refs, "precondition_refs")
        )
        object.__setattr__(
            self, "postcondition_refs", _ids(self.postcondition_refs, "postcondition_refs")
        )
        object.__setattr__(
            self, "validation_refs", _ids(self.validation_refs, "validation_refs")
        )
        object.__setattr__(
            self, "unsupported_limits", _ids(self.unsupported_limits, "unsupported_limits")
        )
        if self.step_id in self.dependency_step_ids:
            raise ChangePropagationEditPacketError(
                "edit step cannot depend on itself"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_EDIT_PACKET_VERSION,
            "step_id": self.step_id,
            "kind": self.kind.value,
            "plan_step_kind": self.plan_step_kind.value,
            "obligation_ids": list(self.obligation_ids),
            "dependency_step_ids": list(self.dependency_step_ids),
            "scc_group_id": self.scc_group_id,
            "transform_id": self.transform_id,
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "before_hashes": [item.to_dict() for item in self.before_hashes],
            "selected_value_sources": [
                item.to_dict() for item in self.selected_value_sources
            ],
            "required_behavior_ids": list(self.required_behavior_ids),
            "counterexample_refs": list(self.counterexample_refs),
            "proof_refs": list(self.proof_refs),
            "precondition_refs": list(self.precondition_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "validation_refs": list(self.validation_refs),
            "unsupported_limits": list(self.unsupported_limits),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PropagationEditStep":
        if not isinstance(payload, Mapping):
            raise ChangePropagationEditPacketError("edit step payload must be an object")
        _reject_forbidden_keys(payload, where="edit step")
        fields = {
            "schema",
            "contract_version",
            "content_id",
            "step_id",
            "kind",
            "plan_step_kind",
            "obligation_ids",
            "dependency_step_ids",
            "scc_group_id",
            "transform_id",
            "read_paths",
            "write_paths",
            "before_hashes",
            "selected_value_sources",
            "required_behavior_ids",
            "counterexample_refs",
            "proof_refs",
            "precondition_refs",
            "postcondition_refs",
            "validation_refs",
            "unsupported_limits",
        }
        if set(payload).difference(fields):
            raise ChangePropagationEditPacketError(
                "edit step contains unsupported fields"
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise ChangePropagationEditPacketError("edit step has unsupported schema")
        try:
            step = cls(
                step_id=payload["step_id"],
                kind=payload["kind"],
                plan_step_kind=payload["plan_step_kind"],
                obligation_ids=tuple(payload["obligation_ids"]),
                dependency_step_ids=tuple(payload.get("dependency_step_ids", ())),
                scc_group_id=payload.get("scc_group_id", ""),
                transform_id=payload.get("transform_id", ""),
                read_paths=tuple(payload.get("read_paths", ())),
                write_paths=tuple(payload.get("write_paths", ())),
                before_hashes=tuple(
                    PathBeforeHash.from_dict(item)
                    for item in payload.get("before_hashes", ())
                ),
                selected_value_sources=tuple(
                    SelectedValueSource.from_dict(item)
                    for item in payload.get("selected_value_sources", ())
                ),
                required_behavior_ids=tuple(payload.get("required_behavior_ids", ())),
                counterexample_refs=tuple(payload.get("counterexample_refs", ())),
                proof_refs=tuple(payload.get("proof_refs", ())),
                precondition_refs=tuple(payload.get("precondition_refs", ())),
                postcondition_refs=tuple(payload.get("postcondition_refs", ())),
                validation_refs=tuple(payload.get("validation_refs", ())),
                unsupported_limits=tuple(payload.get("unsupported_limits", ())),
            )
        except ChangePropagationEditPacketError:
            raise
        except (KeyError, TypeError, ContractValidationError, ValueError) as exc:
            raise ChangePropagationEditPacketError(
                "edit step payload is malformed"
            ) from exc
        claimed = payload.get("content_id")
        if claimed not in (None, "") and claimed != step.content_id:
            raise ChangePropagationEditPacketError(
                "edit step content identity is forged"
            )
        return step


# ---------------------------------------------------------------------------
# Packet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangePropagationEditPacket(CanonicalContract):
    """Content-addressed multi-edit packet bound to one admitted plan."""

    SCHEMA: ClassVar[str] = CHANGE_PROPAGATION_EDIT_PACKET_SCHEMA

    roots: PropagationAuthorityRoots
    admission_id: str
    plan_id: str
    plan_content_id: str
    evidence_bundle_id: str
    change_set_id: str
    delta_id: str
    impact_closure_id: str
    obligation_set_id: str
    step_order: tuple[str, ...]
    scc_group_ids: tuple[str, ...]
    steps: tuple[PropagationEditStep, ...]
    analytical_step_ids: tuple[str, ...]
    model_required_step_ids: tuple[str, ...]
    permitted_read_paths: tuple[str, ...]
    permitted_write_paths: tuple[str, ...]
    before_hashes: tuple[PathBeforeHash, ...]
    selected_value_sources: tuple[SelectedValueSource, ...]
    required_behavior_ids: tuple[str, ...]
    counterexample_refs: tuple[str, ...]
    proof_refs: tuple[str, ...]
    index_refs: tuple[str, ...]
    graph_refs: tuple[str, ...]
    unsupported_limits: tuple[str, ...]
    per_edit_postcondition_refs: tuple[str, ...]
    fixed_point_obligation_ref: str
    fixed_point_postcondition_refs: tuple[str, ...]
    validation_commands: tuple[str, ...]
    checkpoint_strategy_ref: str
    rollback_strategy_ref: str
    invalidation_refs: tuple[str, ...]
    expansion_handles: tuple[PropagationExpansionHandle, ...] = ()
    scc_groups: tuple[PropagationSCCGroup, ...] = ()
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        for name in (
            "admission_id",
            "plan_id",
            "plan_content_id",
            "evidence_bundle_id",
            "change_set_id",
            "delta_id",
            "impact_closure_id",
            "obligation_set_id",
            "fixed_point_obligation_ref",
            "checkpoint_strategy_ref",
            "rollback_strategy_ref",
            "producer_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "step_order",
            _ids(self.step_order, "step_order", required=True, preserve_order=True, limit=MAX_STEPS),
        )
        object.__setattr__(
            self,
            "scc_group_ids",
            _ids(self.scc_group_ids, "scc_group_ids", preserve_order=True, limit=MAX_STEPS),
        )
        if not isinstance(self.steps, Sequence) or not self.steps:
            raise ChangePropagationEditPacketError("packet requires plan steps")
        if not all(isinstance(item, PropagationEditStep) for item in self.steps):
            raise ChangePropagationEditPacketError(
                "steps must be PropagationEditStep values"
            )
        if len(self.steps) > MAX_STEPS:
            raise ChangePropagationEditPacketError("steps exceed bound")
        step_ids = [item.step_id for item in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ChangePropagationEditPacketError("steps must have unique step_ids")
        if set(step_ids) != set(self.step_order) or len(self.step_order) != len(step_ids):
            raise ChangePropagationEditPacketError(
                "step_order must list every step exactly once"
            )
        for step in self.steps:
            missing = set(step.dependency_step_ids) - set(step_ids)
            if missing:
                raise ChangePropagationEditPacketError(
                    "step dependencies must reference packet steps"
                )
        object.__setattr__(self, "steps", tuple(self.steps))

        analytical = _ids(
            self.analytical_step_ids, "analytical_step_ids", preserve_order=True, limit=MAX_STEPS
        )
        model_required = _ids(
            self.model_required_step_ids,
            "model_required_step_ids",
            preserve_order=True,
            limit=MAX_STEPS,
        )
        expected_analytical = tuple(
            item.step_id
            for item in self.steps
            if item.kind is PropagationEditStepKind.ANALYTICAL
        )
        expected_model = tuple(
            item.step_id
            for item in self.steps
            if item.kind is PropagationEditStepKind.MODEL_REQUIRED
        )
        if set(analytical) != set(expected_analytical):
            raise ChangePropagationEditPacketError(
                "analytical_step_ids must partition analytical steps exactly"
            )
        if set(model_required) != set(expected_model):
            raise ChangePropagationEditPacketError(
                "model_required_step_ids must partition model-required steps exactly"
            )
        if set(analytical) & set(model_required):
            raise ChangePropagationEditPacketError(
                "analytical and model-required partitions must be disjoint"
            )
        object.__setattr__(self, "analytical_step_ids", analytical)
        object.__setattr__(self, "model_required_step_ids", model_required)

        object.__setattr__(
            self,
            "permitted_read_paths",
            _paths(self.permitted_read_paths, "permitted_read_paths"),
        )
        object.__setattr__(
            self,
            "permitted_write_paths",
            _paths(self.permitted_write_paths, "permitted_write_paths", required=True),
        )
        for step in self.steps:
            if not set(step.read_paths).issubset(self.permitted_read_paths):
                raise ChangePropagationEditPacketError(
                    "step read paths must stay inside packet read allowlist"
                )
            if not set(step.write_paths).issubset(self.permitted_write_paths):
                raise ChangePropagationEditPacketError(
                    "step write paths must stay inside packet write allowlist"
                )

        if not isinstance(self.before_hashes, Sequence) or not all(
            isinstance(item, PathBeforeHash) for item in self.before_hashes
        ):
            raise ChangePropagationEditPacketError(
                "before_hashes must be PathBeforeHash values"
            )
        before = tuple(sorted(self.before_hashes, key=lambda item: item.path))
        if len({item.path for item in before}) != len(before):
            raise ChangePropagationEditPacketError("before_hashes must have unique paths")
        allowed_paths = set(self.permitted_read_paths) | set(self.permitted_write_paths)
        for item in before:
            if item.path not in allowed_paths:
                raise ChangePropagationEditPacketError(
                    "before_hash path must stay inside packet path allowlists"
                )
        object.__setattr__(self, "before_hashes", before)

        if not isinstance(self.selected_value_sources, Sequence) or not all(
            isinstance(item, SelectedValueSource) for item in self.selected_value_sources
        ):
            raise ChangePropagationEditPacketError(
                "selected_value_sources must be SelectedValueSource values"
            )
        selected = tuple(
            sorted(
                self.selected_value_sources,
                key=lambda item: (item.requirement_id, item.candidate_id),
            )
        )
        object.__setattr__(self, "selected_value_sources", selected)

        object.__setattr__(
            self,
            "required_behavior_ids",
            _ids(self.required_behavior_ids, "required_behavior_ids"),
        )
        object.__setattr__(
            self, "counterexample_refs", _ids(self.counterexample_refs, "counterexample_refs")
        )
        object.__setattr__(
            self, "proof_refs", _ids(self.proof_refs, "proof_refs", required=True)
        )
        index_refs = _ids(self.index_refs, "index_refs", required=True)
        if self.roots.index_id not in index_refs:
            raise ChangePropagationEditPacketError(
                "index_refs must bind the plan index root"
            )
        object.__setattr__(self, "index_refs", index_refs)
        graph_refs = _ids(self.graph_refs, "graph_refs", required=True)
        if self.roots.graph_id not in graph_refs:
            raise ChangePropagationEditPacketError(
                "graph_refs must bind the plan graph root"
            )
        object.__setattr__(self, "graph_refs", graph_refs)
        object.__setattr__(
            self, "unsupported_limits", _ids(self.unsupported_limits, "unsupported_limits")
        )
        object.__setattr__(
            self,
            "per_edit_postcondition_refs",
            _ids(
                self.per_edit_postcondition_refs,
                "per_edit_postcondition_refs",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "fixed_point_postcondition_refs",
            _ids(
                self.fixed_point_postcondition_refs,
                "fixed_point_postcondition_refs",
                required=True,
            ),
        )
        if self.fixed_point_obligation_ref not in self.fixed_point_postcondition_refs:
            object.__setattr__(
                self,
                "fixed_point_postcondition_refs",
                tuple(
                    sorted(
                        {
                            *self.fixed_point_postcondition_refs,
                            self.fixed_point_obligation_ref,
                        }
                    )
                ),
            )
        object.__setattr__(
            self, "validation_commands", _commands(self.validation_commands, "validation_commands")
        )
        object.__setattr__(
            self,
            "invalidation_refs",
            _ids(self.invalidation_refs, "invalidation_refs", required=True),
        )

        if not isinstance(self.expansion_handles, Sequence) or not all(
            isinstance(item, PropagationExpansionHandle) for item in self.expansion_handles
        ):
            raise ChangePropagationEditPacketError(
                "expansion_handles must be PropagationExpansionHandle values"
            )
        handles = tuple(sorted(self.expansion_handles, key=lambda item: item.handle_id))
        if len(handles) > MAX_HANDLES or len({item.handle_id for item in handles}) != len(
            handles
        ):
            raise ChangePropagationEditPacketError(
                "expansion_handles must be unique and bounded"
            )
        reference_ids = set(self.proof_refs) | set(self.counterexample_refs) | set(
            self.index_refs
        ) | set(self.graph_refs)
        reference_ids.update(item.proof_id for item in self.selected_value_sources if item.proof_id)
        reference_ids.update(self.required_behavior_ids)
        reference_ids.update(self.per_edit_postcondition_refs)
        reference_ids.update(self.fixed_point_postcondition_refs)
        reference_ids.add(self.fixed_point_obligation_ref)
        reference_ids.add(self.plan_content_id)
        reference_ids.add(self.admission_id)
        for step in self.steps:
            reference_ids.update(step.proof_refs)
            reference_ids.update(step.counterexample_refs)
            reference_ids.update(step.postcondition_refs)
            reference_ids.update(step.precondition_refs)
            reference_ids.update(step.validation_refs)
            reference_ids.update(item.proof_id for item in step.selected_value_sources if item.proof_id)
        for handle in handles:
            if handle.reference_id not in reference_ids:
                raise ChangePropagationEditPacketError(
                    "an expansion handle must point to packet-bound evidence"
                )
            if not set(handle.permitted_paths).issubset(self.permitted_read_paths):
                raise ChangePropagationEditPacketError(
                    "an expansion handle cannot expand read scope"
                )
        object.__setattr__(self, "expansion_handles", handles)

        if not isinstance(self.scc_groups, Sequence) or not all(
            isinstance(item, PropagationSCCGroup) for item in self.scc_groups
        ):
            raise ChangePropagationEditPacketError(
                "scc_groups must be PropagationSCCGroup values"
            )
        groups = tuple(self.scc_groups)
        if len(groups) != len(self.scc_group_ids):
            # scc_group_ids is the admission order; groups may be empty when no SCCs.
            if self.scc_group_ids and not groups:
                raise ChangePropagationEditPacketError(
                    "scc_group_ids require scc_groups bindings"
                )
        group_ids = {item.group_id for item in groups}
        if self.scc_group_ids and set(self.scc_group_ids) != group_ids:
            raise ChangePropagationEditPacketError(
                "scc_groups must match scc_group_ids exactly"
            )
        for group in groups:
            if set(group.step_ids) - set(step_ids):
                raise ChangePropagationEditPacketError(
                    "scc group steps must reference packet steps"
                )
        object.__setattr__(self, "scc_groups", groups)

        if len(canonical_json_bytes(self._payload())) > MAX_PACKET_BYTES:
            raise ChangePropagationEditPacketError(
                "packet exceeds its serialized byte bound"
            )

    @property
    def interface(self) -> str:
        return CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE

    @property
    def packet_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CHANGE_PROPAGATION_EDIT_PACKET_VERSION,
            "interface": self.interface,
            "producer_id": self.producer_id,
            "roots": self.roots.to_dict(),
            "admission_id": self.admission_id,
            "plan_id": self.plan_id,
            "plan_content_id": self.plan_content_id,
            "evidence_bundle_id": self.evidence_bundle_id,
            "change_set_id": self.change_set_id,
            "delta_id": self.delta_id,
            "impact_closure_id": self.impact_closure_id,
            "obligation_set_id": self.obligation_set_id,
            "step_order": list(self.step_order),
            "scc_group_ids": list(self.scc_group_ids),
            "scc_groups": [item.to_dict() for item in self.scc_groups],
            "steps": [item.to_dict() for item in self.steps],
            "analytical_step_ids": list(self.analytical_step_ids),
            "model_required_step_ids": list(self.model_required_step_ids),
            "permitted_read_paths": list(self.permitted_read_paths),
            "permitted_write_paths": list(self.permitted_write_paths),
            "before_hashes": [item.to_dict() for item in self.before_hashes],
            "selected_value_sources": [
                item.to_dict() for item in self.selected_value_sources
            ],
            "required_behavior_ids": list(self.required_behavior_ids),
            "counterexample_refs": list(self.counterexample_refs),
            "proof_refs": list(self.proof_refs),
            "index_refs": list(self.index_refs),
            "graph_refs": list(self.graph_refs),
            "unsupported_limits": list(self.unsupported_limits),
            "per_edit_postcondition_refs": list(self.per_edit_postcondition_refs),
            "fixed_point_obligation_ref": self.fixed_point_obligation_ref,
            "fixed_point_postcondition_refs": list(self.fixed_point_postcondition_refs),
            "validation_commands": list(self.validation_commands),
            "checkpoint_strategy_ref": self.checkpoint_strategy_ref,
            "rollback_strategy_ref": self.rollback_strategy_ref,
            "invalidation_refs": list(self.invalidation_refs),
            "expansion_handles": [item.to_dict() for item in self.expansion_handles],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChangePropagationEditPacket":
        if not isinstance(payload, Mapping):
            raise ChangePropagationEditPacketError("packet payload must be an object")
        _reject_forbidden_keys(payload, where="packet")
        fields = {
            "schema",
            "contract_version",
            "interface",
            "content_id",
            "producer_id",
            "roots",
            "admission_id",
            "plan_id",
            "plan_content_id",
            "evidence_bundle_id",
            "change_set_id",
            "delta_id",
            "impact_closure_id",
            "obligation_set_id",
            "step_order",
            "scc_group_ids",
            "scc_groups",
            "steps",
            "analytical_step_ids",
            "model_required_step_ids",
            "permitted_read_paths",
            "permitted_write_paths",
            "before_hashes",
            "selected_value_sources",
            "required_behavior_ids",
            "counterexample_refs",
            "proof_refs",
            "index_refs",
            "graph_refs",
            "unsupported_limits",
            "per_edit_postcondition_refs",
            "fixed_point_obligation_ref",
            "fixed_point_postcondition_refs",
            "validation_commands",
            "checkpoint_strategy_ref",
            "rollback_strategy_ref",
            "invalidation_refs",
            "expansion_handles",
        }
        if set(payload).difference(fields):
            raise ChangePropagationEditPacketError(
                "packet has unsupported fields or schema"
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise ChangePropagationEditPacketError("packet has an unsupported schema")
        if payload.get("contract_version") not in (
            None,
            CHANGE_PROPAGATION_EDIT_PACKET_VERSION,
        ):
            raise ChangePropagationEditPacketError(
                "packet has an unsupported contract version"
            )
        if payload.get("interface") not in (
            None,
            "",
            CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE,
        ):
            raise ChangePropagationEditPacketError(
                "packet has an unsupported interface"
            )
        try:
            packet = cls(
                roots=_roots(payload["roots"]),
                admission_id=payload["admission_id"],
                plan_id=payload["plan_id"],
                plan_content_id=payload["plan_content_id"],
                evidence_bundle_id=payload["evidence_bundle_id"],
                change_set_id=payload["change_set_id"],
                delta_id=payload["delta_id"],
                impact_closure_id=payload["impact_closure_id"],
                obligation_set_id=payload["obligation_set_id"],
                step_order=tuple(payload["step_order"]),
                scc_group_ids=tuple(payload.get("scc_group_ids", ())),
                steps=tuple(
                    PropagationEditStep.from_dict(item) for item in payload["steps"]
                ),
                analytical_step_ids=tuple(payload.get("analytical_step_ids", ())),
                model_required_step_ids=tuple(
                    payload.get("model_required_step_ids", ())
                ),
                permitted_read_paths=tuple(payload["permitted_read_paths"]),
                permitted_write_paths=tuple(payload["permitted_write_paths"]),
                before_hashes=tuple(
                    PathBeforeHash.from_dict(item)
                    for item in payload.get("before_hashes", ())
                ),
                selected_value_sources=tuple(
                    SelectedValueSource.from_dict(item)
                    for item in payload.get("selected_value_sources", ())
                ),
                required_behavior_ids=tuple(payload.get("required_behavior_ids", ())),
                counterexample_refs=tuple(payload.get("counterexample_refs", ())),
                proof_refs=tuple(payload["proof_refs"]),
                index_refs=tuple(payload["index_refs"]),
                graph_refs=tuple(payload["graph_refs"]),
                unsupported_limits=tuple(payload.get("unsupported_limits", ())),
                per_edit_postcondition_refs=tuple(
                    payload["per_edit_postcondition_refs"]
                ),
                fixed_point_obligation_ref=payload["fixed_point_obligation_ref"],
                fixed_point_postcondition_refs=tuple(
                    payload["fixed_point_postcondition_refs"]
                ),
                validation_commands=tuple(payload["validation_commands"]),
                checkpoint_strategy_ref=payload["checkpoint_strategy_ref"],
                rollback_strategy_ref=payload["rollback_strategy_ref"],
                invalidation_refs=tuple(payload["invalidation_refs"]),
                expansion_handles=tuple(
                    PropagationExpansionHandle.from_dict(item)
                    for item in payload.get("expansion_handles", ())
                ),
                scc_groups=tuple(
                    PropagationSCCGroup.from_dict(item)
                    if isinstance(item, Mapping) and "schema" in item
                    else PropagationSCCGroup(**item)
                    for item in payload.get("scc_groups", ())
                ),
                producer_id=payload.get("producer_id", PRODUCER_ID),
            )
        except ChangePropagationEditPacketError:
            raise
        except (KeyError, TypeError, ContractValidationError, ValueError) as exc:
            raise ChangePropagationEditPacketError(
                "packet payload is malformed"
            ) from exc
        claimed = payload.get("content_id")
        if claimed not in (None, "") and claimed != packet.content_id:
            raise ChangePropagationEditPacketError(
                "packet content identity is forged"
            )
        return packet


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _before_hashes_from_spans(
    spans: Sequence[PlanPathSpan],
    *,
    allowed_paths: set[str],
) -> tuple[PathBeforeHash, ...]:
    by_path: dict[str, PathBeforeHash] = {}
    for span in spans:
        if span.path not in allowed_paths:
            continue
        # Prefer the first non-empty before_hash for a path.
        existing = by_path.get(span.path)
        if existing is None or (not existing.before_hash and span.before_hash):
            by_path[span.path] = PathBeforeHash(
                path=span.path,
                before_hash=span.before_hash,
                artifact_id=span.artifact_id,
            )
    return tuple(sorted(by_path.values(), key=lambda item: item.path))


def _selected_values_from_proofs(
    proofs: Sequence[ValueMappingProof],
) -> tuple[SelectedValueSource, ...]:
    selected: list[SelectedValueSource] = []
    for proof in proofs:
        if not isinstance(proof, ValueMappingProof):
            raise ChangePropagationEditPacketError(
                "value_mapping_proofs must be ValueMappingProof values"
            )
        if proof.disposition is SynthesisDisposition.UNIQUE_PROVED:
            if len(proof.proved_candidate_ids) != 1:
                raise ChangePropagationEditPacketError(
                    "unique_proved mapping must select exactly one value source"
                )
            selected.append(
                SelectedValueSource(
                    requirement_id=proof.requirement_id,
                    consumer_id=proof.consumer_id,
                    candidate_id=proof.proved_candidate_ids[0],
                    expression_ref=proof.expression_ref,
                    type_ref=proof.type_ref,
                    proof_id=proof.proof_id,
                )
            )
        elif proof.disposition is SynthesisDisposition.AMBIGUOUS:
            raise ChangePropagationEditPacketError(
                "alternatives from ambiguous value mappings cannot broaden packet scope"
            )
        elif proof.disposition in {
            SynthesisDisposition.UNKNOWN,
            SynthesisDisposition.UNSUPPORTED,
            SynthesisDisposition.TIMEOUT,
        }:
            # Unknown semantics cannot be materialised as authority.
            raise ChangePropagationEditPacketError(
                "unknown or unsupported value semantics cannot broaden packet scope"
            )
        # REFUTED / non-unique non-success that was already plan-discharged may
        # be omitted (no selected source).  Competing alternatives never enter.
    return tuple(
        sorted(selected, key=lambda item: (item.requirement_id, item.candidate_id))
    )


def _validation_commands_from_evidence(
    evidence: PlanEvidenceBundle | None,
    admission: PropagationPlanAdmission,
    *,
    override: Sequence[str] | None,
) -> tuple[str, ...]:
    if override is not None:
        commands = _commands(override, "validation_commands")
        return commands
    if evidence is not None and evidence.validation_commands:
        rendered: list[str] = []
        for command in evidence.validation_commands:
            if not isinstance(command, PlanValidationCommand):
                raise ChangePropagationEditPacketError(
                    "validation_commands must be PlanValidationCommand values"
                )
            rendered.append(" ".join(command.argv))
        return _commands(rendered, "validation_commands")
    if admission.validation_command_ids:
        # Focused command identities from admission when argv is unavailable.
        return _commands(
            tuple(f"command:{item}" for item in admission.validation_command_ids),
            "validation_commands",
        )
    raise ChangePropagationEditPacketError(
        "focused validation commands are required for packet materialization"
    )


def _project_step(
    plan_step: PropagationPlanStep,
    *,
    before_by_path: Mapping[str, PathBeforeHash],
    selected_by_consumer: Mapping[str, tuple[SelectedValueSource, ...]],
    behavior_by_obligation: Mapping[str, tuple[str, ...]],
    transform_proofs: Mapping[str, tuple[str, ...]],
    counterexample_refs: Sequence[str],
    unsupported_limits: Sequence[str],
    obligation_consumers: Mapping[str, str],
) -> PropagationEditStep:
    kind = _plan_step_kind_to_edit_kind(plan_step.kind)
    step_paths = set(plan_step.read_paths) | set(plan_step.write_paths)
    before = tuple(
        before_by_path[path] for path in sorted(step_paths) if path in before_by_path
    )
    consumers = {
        obligation_consumers[oid]
        for oid in plan_step.obligation_ids
        if oid in obligation_consumers
    }
    selected: list[SelectedValueSource] = []
    for consumer_id in sorted(consumers):
        selected.extend(selected_by_consumer.get(consumer_id, ()))
    behaviors: list[str] = []
    for oid in plan_step.obligation_ids:
        behaviors.extend(behavior_by_obligation.get(oid, ()))
    proof_refs: list[str] = []
    if plan_step.transform_id and plan_step.transform_id in transform_proofs:
        proof_refs.extend(transform_proofs[plan_step.transform_id])
    # Model-required steps must already be behavior-complete.
    if kind is PropagationEditStepKind.MODEL_REQUIRED and not behaviors:
        # Placement-backed LLM steps may carry behavior via precondition refs.
        placement_behaviors = [
            ref
            for ref in plan_step.precondition_refs
            if ref.startswith("pre:placement:")
        ]
        if not placement_behaviors:
            raise ChangePropagationEditPacketError(
                "model-required steps require behavior-complete required behavior bindings"
            )
        behaviors.extend(
            ref.removeprefix("pre:placement:") for ref in placement_behaviors
        )

    # Counterexamples are plan-level; bind the full admitted set only when the
    # step is model-required (provider-visible). Analytical steps stay body-free.
    if kind is PropagationEditStepKind.MODEL_REQUIRED:
        step_counterexamples = tuple(sorted(counterexample_refs))
    else:
        step_counterexamples = ()

    return PropagationEditStep(
        step_id=plan_step.step_id,
        kind=kind,
        plan_step_kind=plan_step.kind,
        obligation_ids=plan_step.obligation_ids,
        dependency_step_ids=plan_step.dependency_step_ids,
        scc_group_id=plan_step.scc_group_id,
        transform_id=plan_step.transform_id,
        read_paths=plan_step.read_paths,
        write_paths=plan_step.write_paths,
        before_hashes=before,
        selected_value_sources=tuple(selected),
        required_behavior_ids=tuple(sorted(set(behaviors))),
        counterexample_refs=step_counterexamples,
        proof_refs=tuple(sorted(set(proof_refs))),
        precondition_refs=plan_step.precondition_refs,
        postcondition_refs=plan_step.postcondition_refs,
        validation_refs=plan_step.validation_refs,
        unsupported_limits=tuple(sorted(set(unsupported_limits))),
    )


def materialize_change_propagation_edit_packet(
    admission: PropagationPlanAdmission,
    *,
    roots: PropagationAuthorityRoots | None = None,
    evidence: PlanEvidenceBundle | None = None,
    value_mapping_proofs: Sequence[ValueMappingProof] = (),
    analytical_transforms: Sequence[AnalyticalTransform] = (),
    required_behaviors: Sequence[RequiredBehaviorContract] = (),
    counterexample_refs: Sequence[str] = (),
    unsupported_limits: Sequence[str] = (),
    validation_commands: Sequence[str] | None = None,
    expansion_handles: Sequence[PropagationExpansionHandle] = (),
) -> ChangePropagationEditPacket:
    """Materialize one multi-edit packet from a current admitted plan.

    Only a :class:`PropagationPlanAdmission` with ``PlanDisposition.ADMITTED``
    may materialize.  Bare, abstaining, partial, stale, or forged plans fail
    closed.  Packet content cannot broaden admitted path or semantic scope.
    """

    if not isinstance(admission, PropagationPlanAdmission):
        raise ChangePropagationEditPacketError(
            "a current PropagationPlanAdmission is required"
        )
    if admission.disposition is not PlanDisposition.ADMITTED or not admission.admitted:
        raise ChangePropagationEditPacketError(
            "only a current admitted non-abstaining plan may materialize"
        )
    if admission.reason_codes:
        raise ChangePropagationEditPacketError(
            "admitted plan materialization cannot carry abstention reason codes"
        )
    if admission.alternative_plan_ids:
        raise ChangePropagationEditPacketError(
            "alternatives cannot broaden packet scope"
        )

    plan = admission.plan
    if not isinstance(plan, AtomicPropagationPlan):
        raise ChangePropagationEditPacketError(
            "admission must carry AtomicPropagationPlan@1"
        )
    if plan.disposition is not PlanDisposition.ADMITTED:
        raise ChangePropagationEditPacketError(
            "only a current admitted non-abstaining plan may materialize"
        )
    if not plan.steps:
        raise ChangePropagationEditPacketError(
            "partial plans without steps cannot materialize"
        )
    if not plan.permitted_write_paths:
        raise ChangePropagationEditPacketError(
            "admitted plan lacks exact write path authority"
        )
    if not plan.fixed_point_obligation_ref:
        raise ChangePropagationEditPacketError(
            "admitted plan lacks fixed-point postcondition authority"
        )
    if not plan.proof_refs:
        raise ChangePropagationEditPacketError(
            "admitted plan lacks proof refs"
        )
    if not plan.checkpoint_strategy_ref or not plan.rollback_strategy_ref:
        raise ChangePropagationEditPacketError(
            "admitted plan lacks checkpoint/rollback strategy refs"
        )

    current_roots = roots if roots is not None else plan.roots
    current_roots = _roots(current_roots)
    if current_roots != plan.roots:
        raise ChangePropagationEditPacketError(
            "roots are stale relative to the admitted plan"
        )

    if evidence is not None:
        if not isinstance(evidence, PlanEvidenceBundle):
            raise ChangePropagationEditPacketError(
                "evidence must be PlanEvidenceBundle when provided"
            )
        if evidence.content_id != admission.evidence_bundle_id:
            raise ChangePropagationEditPacketError(
                "evidence bundle is not the current admitted evidence"
            )
        if evidence.roots != plan.roots:
            raise ChangePropagationEditPacketError(
                "evidence roots are stale relative to the admitted plan"
            )
        if evidence.delta_id != plan.delta_id or evidence.change_set_id != plan.change_set_id:
            raise ChangePropagationEditPacketError(
                "evidence change/delta identity does not match the admitted plan"
            )
        if evidence.expected_roots is not None and evidence.expected_roots != plan.roots:
            raise ChangePropagationEditPacketError(
                "evidence expected roots are stale"
            )
        # Explicit caller projections take precedence so fail-closed tests can
        # re-bind values/transforms without inventing a second evidence bundle.
        proofs = (
            tuple(value_mapping_proofs)
            if value_mapping_proofs
            else evidence.value_mapping_proofs
        )
        transforms = (
            tuple(analytical_transforms)
            if analytical_transforms
            else evidence.analytical_transforms
        )
    else:
        proofs = tuple(value_mapping_proofs)
        transforms = tuple(analytical_transforms)

    # Selected values: only unique-proved; reject ambiguous/unknown.
    selected_values = _selected_values_from_proofs(proofs)
    selected_by_consumer: dict[str, list[SelectedValueSource]] = {}
    for item in selected_values:
        selected_by_consumer.setdefault(item.consumer_id, []).append(item)
    selected_by_consumer_t = {
        key: tuple(value) for key, value in selected_by_consumer.items()
    }

    # Analytical transforms must be admitted and path-aligned when supplied.
    transform_proofs: dict[str, tuple[str, ...]] = {}
    for transform in transforms:
        if not isinstance(transform, AnalyticalTransform):
            raise ChangePropagationEditPacketError(
                "analytical_transforms must be AnalyticalTransform values"
            )
        if transform.roots != plan.roots:
            raise ChangePropagationEditPacketError(
                "analytical transform roots are stale"
            )
        if transform.disposition is not TransformDisposition.ADMITTED:
            # Non-admitted transforms cannot grant authority or broaden scope.
            continue
        if not set(transform.target_paths).issubset(plan.permitted_write_paths):
            raise ChangePropagationEditPacketError(
                "transform target paths mismatch plan write authority"
            )
        transform_proofs[transform.transform_id] = transform.proof_refs

    # Required behaviors: only independently sourced contracts bound to roots.
    behavior_ids: list[str] = []
    for behavior in required_behaviors:
        if not isinstance(behavior, RequiredBehaviorContract):
            raise ChangePropagationEditPacketError(
                "required_behaviors must be RequiredBehaviorContract values"
            )
        if behavior.roots != plan.roots:
            raise ChangePropagationEditPacketError(
                "required behavior roots are stale"
            )
        if behavior.implementation_hypothesis:
            raise ChangePropagationEditPacketError(
                "implementation hypotheses cannot broaden required behavior authority"
            )
        behavior_ids.append(behavior.behavior_id)
    behavior_ids_t = tuple(sorted(set(behavior_ids)))

    # Also harvest behavior ids already present on obligations.
    behavior_by_obligation: dict[str, tuple[str, ...]] = {}
    for obligation in plan.obligations:
        # Bind obligation-declared behavior ids only; never invent.
        behavior_by_obligation[obligation.obligation_id] = tuple(
            sorted(set(obligation.behavior_contract_ids))
        )

    obligation_consumers = {
        item.obligation_id: item.consumer_id for item in plan.obligations
    }

    allowed_paths = set(plan.permitted_read_paths) | set(plan.permitted_write_paths)
    # Path mismatch: every step path must remain inside plan authority.
    for step in plan.steps:
        if not set(step.read_paths).issubset(plan.permitted_read_paths):
            raise ChangePropagationEditPacketError(
                "plan step read path mismatch against plan allowlist"
            )
        if not set(step.write_paths).issubset(plan.permitted_write_paths):
            raise ChangePropagationEditPacketError(
                "plan step write path mismatch against plan allowlist"
            )

    spans = tuple(admission.permitted_read_spans) + tuple(
        admission.permitted_write_spans
    )
    if evidence is not None:
        spans = tuple(evidence.read_spans) + tuple(evidence.write_spans) + spans
    before_hashes = _before_hashes_from_spans(spans, allowed_paths=allowed_paths)
    before_by_path = {item.path: item for item in before_hashes}

    counterexamples = _ids(counterexample_refs, "counterexample_refs")
    # Allow explicit counterexample refs that callers supply as already admitted
    # evidence identifiers, but reject free-form path-like broadening.
    for ref in counterexamples:
        if "/" in ref or ref.startswith(".."):
            raise ChangePropagationEditPacketError(
                "counterexample refs cannot broaden path scope"
            )

    limits = _ids(unsupported_limits, "unsupported_limits")

    # Project steps in admission dependency order.
    plan_steps_by_id = {item.step_id: item for item in plan.steps}
    if admission.step_order:
        ordered_ids = list(admission.step_order)
        if set(ordered_ids) != set(plan_steps_by_id):
            raise ChangePropagationEditPacketError(
                "admission step_order does not match plan steps"
            )
    else:
        ordered_ids = [item.step_id for item in plan.steps]

    projected: list[PropagationEditStep] = []
    for step_id in ordered_ids:
        projected.append(
            _project_step(
                plan_steps_by_id[step_id],
                before_by_path=before_by_path,
                selected_by_consumer=selected_by_consumer_t,
                behavior_by_obligation=behavior_by_obligation,
                transform_proofs=transform_proofs,
                counterexample_refs=counterexamples,
                unsupported_limits=limits,
                obligation_consumers=obligation_consumers,
            )
        )

    analytical_step_ids = tuple(
        item.step_id
        for item in projected
        if item.kind is PropagationEditStepKind.ANALYTICAL
    )
    model_required_step_ids = tuple(
        item.step_id
        for item in projected
        if item.kind is PropagationEditStepKind.MODEL_REQUIRED
    )

    per_edit_posts = tuple(
        sorted(
            {
                ref
                for step in projected
                for ref in step.postcondition_refs
            }
        )
    )
    if not per_edit_posts:
        # Every admitted plan step carries postconditions from the planner; if
        # somehow empty, fail closed rather than invent them.
        raise ChangePropagationEditPacketError(
            "per-edit postconditions are required"
        )

    fixed_point_posts = tuple(
        sorted({plan.fixed_point_obligation_ref, "post:fixed-point:closure"})
    )

    commands = _validation_commands_from_evidence(
        evidence, admission, override=validation_commands
    )

    handles = tuple(expansion_handles)
    for handle in handles:
        if not isinstance(handle, PropagationExpansionHandle):
            raise ChangePropagationEditPacketError(
                "expansion_handles must be PropagationExpansionHandle values"
            )

    # Packet-level required behaviors: union of step bindings and explicit set.
    packet_behavior_ids = tuple(
        sorted(
            {
                *behavior_ids_t,
                *(
                    bid
                    for step in projected
                    for bid in step.required_behavior_ids
                ),
            }
        )
    )

    return ChangePropagationEditPacket(
        roots=plan.roots,
        admission_id=admission.content_id,
        plan_id=plan.plan_id,
        plan_content_id=plan.content_id,
        evidence_bundle_id=admission.evidence_bundle_id,
        change_set_id=plan.change_set_id,
        delta_id=plan.delta_id,
        impact_closure_id=plan.impact_closure_id,
        obligation_set_id=plan.obligation_set_id,
        step_order=tuple(ordered_ids),
        scc_group_ids=admission.scc_group_ids or tuple(
            group.group_id for group in plan.scc_groups
        ),
        steps=tuple(projected),
        analytical_step_ids=analytical_step_ids,
        model_required_step_ids=model_required_step_ids,
        permitted_read_paths=plan.permitted_read_paths,
        permitted_write_paths=plan.permitted_write_paths,
        before_hashes=before_hashes,
        selected_value_sources=selected_values,
        required_behavior_ids=packet_behavior_ids,
        counterexample_refs=counterexamples,
        proof_refs=plan.proof_refs,
        index_refs=(plan.roots.index_id,),
        graph_refs=(plan.roots.graph_id,),
        unsupported_limits=limits,
        per_edit_postcondition_refs=per_edit_posts,
        fixed_point_obligation_ref=plan.fixed_point_obligation_ref,
        fixed_point_postcondition_refs=fixed_point_posts,
        validation_commands=commands,
        checkpoint_strategy_ref=plan.checkpoint_strategy_ref,
        rollback_strategy_ref=plan.rollback_strategy_ref,
        invalidation_refs=plan.invalidation_refs,
        expansion_handles=handles,
        scc_groups=plan.scc_groups,
    )


build_change_propagation_edit_packet = materialize_change_propagation_edit_packet


__all__ = [
    "CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE",
    "CHANGE_PROPAGATION_EDIT_PACKET_SCHEMA",
    "CHANGE_PROPAGATION_EDIT_PACKET_VERSION",
    "ChangePropagationEditPacket",
    "ChangePropagationEditPacketError",
    "ChangePropagationEditPacketReason",
    "PRODUCER_ID",
    "PathBeforeHash",
    "PropagationEditStep",
    "PropagationEditStepKind",
    "PropagationExpansionHandle",
    "SelectedValueSource",
    "build_change_propagation_edit_packet",
    "materialize_change_propagation_edit_packet",
]
