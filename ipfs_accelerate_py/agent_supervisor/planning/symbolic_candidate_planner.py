"""Deterministic-first candidate portfolios for proof-directed planning.

The planner in this module is deliberately a composition layer.  It consumes
the reviewed ``ObligationGraph@1`` produced by
:mod:`obligation_graph_compiler`, derives bounded task sets by backward
chaining through its AND/OR refinements, compiles dependency-valid partial
orders, and delegates the final hard-gated quality/cost decision to the
existing adaptive planner.

Optional model output is only a nomination of task identifiers over the same
frozen request.  Authority, scope, safety, and proof decisions are recomputed
locally and remain non-compensable.

DCR-062 additionally exposes a finite operator-bound portfolio admission path
via :mod:`deterministic_candidate_portfolio` (``RepairCandidate@1`` /
``CandidateAdmission@1``): unique winners are admitted, while ties and
unknowns abstain, and every candidate must bind current evidence plus exact
operator CIDs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import queue
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from ..proof.formal_verification_contracts import canonical_json, content_identity
from .adaptive_planner import (
    AdaptivePlanCandidate,
    AdaptivePlanSelectionReceipt,
    AdaptivePlanner,
    FrozenPlanningGoal,
    HardConstraintReceipt,
    HardGateEvaluator,
    deterministic_hard_gate_receipts,
)
from .obligation_graph_compiler import (
    IssueSeverity,
    ObligationGraph,
    ObligationNode,
    ObligationNodeKind,
    ObligationStatus,
    RefinementKind,
    TaskCandidate,
)
from .plan_evaluator import EvidenceAwarePlanCandidate, PlanBranch
from .plan_failure_memory import (
    FailureMemoryScope,
    PlanFailureMemory,
    PlanFailureMemorySnapshot,
)
from .task_proposal_router import (
    CandidateGenerationBounds,
    FrozenCandidateGenerationRequest,
)


SYMBOLIC_CANDIDATE_PLANNER_INTERFACE: Final[str] = "SymbolicCandidatePlanner@1"
SYMBOLIC_CANDIDATE_PLANNER_VERSION: Final[int] = 1
SYMBOLIC_CANDIDATE_PORTFOLIO_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-candidate-portfolio@1"
)
SYMBOLIC_CANDIDATE_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-candidate-request@1"
)
SYMBOLIC_CANDIDATE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-candidate@1"
)
SYMBOLIC_CANDIDATE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-candidate-snapshot@1"
)
SYMBOLIC_PROVIDER_USAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-provider-usage@1"
)
SYMBOLIC_SCHEDULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-partial-order-schedule@1"
)

_SAFE_ID = re.compile(r"^[^\x00\r\n\t]{1,2048}$")
_PATH = re.compile(r"^[A-Za-z0-9._-][A-Za-z0-9._/+-]*$")


class SymbolicCandidatePlanningError(ValueError):
    """Frozen symbolic inputs cannot produce a trustworthy portfolio."""


class SymbolicCandidateSource(str, Enum):
    DETERMINISTIC_BASELINE = "deterministic_baseline"
    DETERMINISTIC_ALTERNATIVE = "deterministic_alternative"
    MODEL_PROPOSAL = "model_proposal"


class SymbolicProviderStatus(str, Enum):
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    MALFORMED = "malformed"
    BUDGET_REJECTED = "budget_rejected"


def _id(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SymbolicCandidatePlanningError(f"{name} must be a string")
    result = value.strip()
    if not result and allow_empty:
        return ""
    if not result or not _SAFE_ID.fullmatch(result):
        raise SymbolicCandidatePlanningError(f"{name} must be a bounded identifier")
    return result


def _ids(
    value: Iterable[Any],
    name: str,
    *,
    allow_empty: bool = True,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise SymbolicCandidatePlanningError(f"{name} must be an array")
    result: list[str] = []
    for item in value:
        normalized = _id(item, name)
        if normalized not in result:
            result.append(normalized)
    if not result and not allow_empty:
        raise SymbolicCandidatePlanningError(f"{name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SymbolicCandidatePlanningError(
            f"{name} must be an integer of at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise SymbolicCandidatePlanningError(
            f"{name} must be no greater than {maximum}"
        )
    return value


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        raise SymbolicCandidatePlanningError("symbolic context exceeds depth bound")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise SymbolicCandidatePlanningError(
            "symbolic context must use integer fixed-point values"
        )
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            _id(str(key), "context key"): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise SymbolicCandidatePlanningError(
        f"symbolic context contains unsupported type {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SymbolicCandidateBounds:
    """Finite aggregate search and optional-model limits."""

    candidate_count: int = 4
    max_search_states: int = 4_096
    max_tasks_per_candidate: int = 512
    max_model_candidates: int = 1
    max_provider_response_bytes: int = 256_000
    max_provider_input_tokens: int = 12_288
    max_provider_output_tokens: int = 4_096
    provider_timeout_milliseconds: int = 30_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_count", _integer(self.candidate_count, "candidate_count", 1, 32)
        )
        for name in (
            "max_search_states",
            "max_tasks_per_candidate",
            "max_provider_response_bytes",
            "max_provider_input_tokens",
            "max_provider_output_tokens",
            "provider_timeout_milliseconds",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        object.__setattr__(
            self,
            "max_model_candidates",
            min(
                _integer(
                    self.max_model_candidates,
                    "max_model_candidates",
                    0,
                ),
                self.candidate_count - 1,
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicCandidateBounds":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise SymbolicCandidatePlanningError(
                "symbolic candidate bounds must use the closed schema"
            )
        return cls(**dict(payload))


@dataclass(frozen=True)
class FrozenSymbolicCandidateRequest:
    """Exact graph, policy, context, template and failure-memory bindings."""

    obligation_graph: ObligationGraph
    frozen_goal: FrozenPlanningGoal
    context: Mapping[str, Any]
    bounds: SymbolicCandidateBounds
    reviewed_template_ids: tuple[str, ...]
    failure_memory_state_id: str
    failure_memory_scope_id: str = ""
    interface: str = SYMBOLIC_CANDIDATE_PLANNER_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_graph, ObligationGraph):
            object.__setattr__(
                self,
                "obligation_graph",
                ObligationGraph.from_dict(self.obligation_graph),
            )
        if not isinstance(self.frozen_goal, FrozenPlanningGoal):
            object.__setattr__(
                self, "frozen_goal", FrozenPlanningGoal.from_dict(self.frozen_goal)
            )
        if not isinstance(self.bounds, SymbolicCandidateBounds):
            object.__setattr__(
                self, "bounds", SymbolicCandidateBounds.from_dict(self.bounds)
            )
        if (
            self.obligation_graph.current_root_id
            and self.obligation_graph.current_root_id
            != self.frozen_goal.repository_tree_id
        ):
            raise SymbolicCandidatePlanningError(
                "obligation graph is detached from the frozen repository tree"
            )
        plain_context = _plain(self.context)
        if not isinstance(plain_context, Mapping):
            raise SymbolicCandidatePlanningError("context must be a mapping")
        if len(_json_bytes(plain_context)) > self.bounds.max_provider_input_tokens * 4:
            raise SymbolicCandidatePlanningError(
                "frozen symbolic context exceeds the input budget"
            )
        object.__setattr__(self, "context", _freeze(plain_context))
        object.__setattr__(
            self,
            "reviewed_template_ids",
            _ids(self.reviewed_template_ids, "reviewed_template_ids", allow_empty=False),
        )
        object.__setattr__(
            self,
            "failure_memory_state_id",
            _id(self.failure_memory_state_id, "failure_memory_state_id"),
        )
        object.__setattr__(
            self,
            "failure_memory_scope_id",
            _id(
                self.failure_memory_scope_id,
                "failure_memory_scope_id",
                allow_empty=True,
            ),
        )
        object.__setattr__(self, "interface", _id(self.interface, "interface"))
        if self.interface != SYMBOLIC_CANDIDATE_PLANNER_INTERFACE:
            raise SymbolicCandidatePlanningError(
                "unsupported symbolic candidate planner interface"
            )

    @property
    def graph_id(self) -> str:
        return self.obligation_graph.graph_id

    @property
    def context_id(self) -> str:
        return _digest(_plain(self.context))

    @property
    def request_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_CANDIDATE_REQUEST_SCHEMA,
            "interface": self.interface,
            "obligation_graph": self.obligation_graph.to_dict(),
            "frozen_goal": self.frozen_goal.to_dict(),
            "context": _plain(self.context),
            "context_id": self.context_id,
            "bounds": self.bounds.to_dict(),
            "reviewed_template_ids": list(self.reviewed_template_ids),
            "failure_memory_state_id": self.failure_memory_state_id,
            "failure_memory_scope_id": self.failure_memory_scope_id,
        }
        if include_identity:
            payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenSymbolicCandidateRequest":
        expected = {
            "schema",
            "interface",
            "obligation_graph",
            "frozen_goal",
            "context",
            "context_id",
            "bounds",
            "reviewed_template_ids",
            "failure_memory_state_id",
            "failure_memory_scope_id",
            "request_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise SymbolicCandidatePlanningError(
                "frozen symbolic request must use the closed schema"
            )
        if payload.get("schema") != SYMBOLIC_CANDIDATE_REQUEST_SCHEMA:
            raise SymbolicCandidatePlanningError("unsupported symbolic request schema")
        result = cls(
            obligation_graph=ObligationGraph.from_dict(
                payload.get("obligation_graph") or {}
            ),
            frozen_goal=FrozenPlanningGoal.from_dict(payload.get("frozen_goal") or {}),
            context=payload.get("context") or {},
            bounds=SymbolicCandidateBounds.from_dict(payload.get("bounds") or {}),
            reviewed_template_ids=tuple(payload.get("reviewed_template_ids") or ()),
            failure_memory_state_id=payload.get("failure_memory_state_id", ""),
            failure_memory_scope_id=payload.get("failure_memory_scope_id", ""),
            interface=payload.get("interface", ""),
        )
        if payload.get("context_id") != result.context_id:
            raise SymbolicCandidatePlanningError("symbolic context identity mismatch")
        if payload.get("request_id") != result.request_id:
            raise SymbolicCandidatePlanningError("symbolic request identity mismatch")
        return result


@dataclass(frozen=True)
class PartialOrderSchedule:
    """A stable topological wave decomposition of selected task candidates."""

    waves: tuple[tuple[str, ...], ...]
    dependency_edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        waves: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for index, wave in enumerate(self.waves):
            normalized = _ids(
                wave,
                f"waves[{index}]",
                allow_empty=False,
            )
            if seen.intersection(normalized):
                raise SymbolicCandidatePlanningError(
                    "a scheduled task may appear in exactly one wave"
                )
            seen.update(normalized)
            waves.append(normalized)
        if not waves:
            raise SymbolicCandidatePlanningError("schedule must contain work")
        edges = tuple(
            sorted(
                {
                    (
                        _id(item[0], "dependency predecessor"),
                        _id(item[1], "dependency successor"),
                    )
                    for item in self.dependency_edges
                }
            )
        )
        positions = {
            task_id: index
            for index, wave in enumerate(waves)
            for task_id in wave
        }
        if any(
            before not in positions
            or after not in positions
            or positions[before] >= positions[after]
            for before, after in edges
        ):
            raise SymbolicCandidatePlanningError(
                "partial-order schedule violates a dependency edge"
            )
        object.__setattr__(self, "waves", tuple(waves))
        object.__setattr__(self, "dependency_edges", edges)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(item for wave in self.waves for item in wave)

    @property
    def schedule_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_SCHEDULE_SCHEMA,
            "waves": [list(wave) for wave in self.waves],
            "dependency_edges": [list(item) for item in self.dependency_edges],
        }
        if include_identity:
            payload["schedule_id"] = self.schedule_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PartialOrderSchedule":
        expected = {"schema", "waves", "dependency_edges", "schedule_id"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise SymbolicCandidatePlanningError(
                "partial-order schedule must use the closed schema"
            )
        if payload.get("schema") != SYMBOLIC_SCHEDULE_SCHEMA:
            raise SymbolicCandidatePlanningError("unsupported schedule schema")
        result = cls(
            waves=tuple(tuple(item) for item in payload.get("waves") or ()),
            dependency_edges=tuple(
                tuple(item) for item in payload.get("dependency_edges") or ()
            ),
        )
        if payload.get("schedule_id") != result.schedule_id:
            raise SymbolicCandidatePlanningError("schedule identity mismatch")
        return result


@dataclass(frozen=True)
class SymbolicCandidateRecord:
    """Auditable symbolic derivation paired with an evaluator declaration."""

    request_id: str
    source: SymbolicCandidateSource
    strategy_ids: tuple[str, ...]
    task_candidate_ids: tuple[str, ...]
    covered_obligation_ids: tuple[str, ...]
    schedule: PartialOrderSchedule
    plan: EvidenceAwarePlanCandidate
    expected_information_gain_millionths: int
    proof_feasible: bool
    constraint_solution_id: str
    failure_memory_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _id(self.request_id, "request_id"))
        object.__setattr__(self, "source", SymbolicCandidateSource(self.source))
        object.__setattr__(
            self,
            "strategy_ids",
            _ids(self.strategy_ids, "strategy_ids", allow_empty=False),
        )
        object.__setattr__(
            self,
            "task_candidate_ids",
            _ids(self.task_candidate_ids, "task_candidate_ids", allow_empty=False),
        )
        if set(self.schedule.task_ids) != set(self.task_candidate_ids):
            raise SymbolicCandidatePlanningError(
                "schedule must cover the exact symbolic task population"
            )
        object.__setattr__(
            self,
            "covered_obligation_ids",
            _ids(
                self.covered_obligation_ids,
                "covered_obligation_ids",
                allow_empty=False,
            ),
        )
        plan = (
            self.plan
            if isinstance(self.plan, EvidenceAwarePlanCandidate)
            else EvidenceAwarePlanCandidate.from_dict(self.plan)
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(
            self,
            "expected_information_gain_millionths",
            _integer(
                self.expected_information_gain_millionths,
                "expected_information_gain_millionths",
                0,
                1_000_000,
            ),
        )
        if not isinstance(self.proof_feasible, bool):
            raise SymbolicCandidatePlanningError("proof_feasible must be boolean")
        if self.plan.proof_feasible != self.proof_feasible:
            raise SymbolicCandidatePlanningError(
                "symbolic and evaluator proof feasibility disagree"
            )
        object.__setattr__(
            self,
            "constraint_solution_id",
            _id(self.constraint_solution_id, "constraint_solution_id"),
        )
        object.__setattr__(
            self,
            "failure_memory_record_ids",
            _ids(
                self.failure_memory_record_ids,
                "failure_memory_record_ids",
            ),
        )

    @property
    def candidate_id(self) -> str:
        return self.plan.candidate_id

    @property
    def record_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_CANDIDATE_RECORD_SCHEMA,
            "request_id": self.request_id,
            "source": self.source.value,
            "strategy_ids": list(self.strategy_ids),
            "task_candidate_ids": list(self.task_candidate_ids),
            "covered_obligation_ids": list(self.covered_obligation_ids),
            "schedule": self.schedule.to_dict(),
            "plan": self.plan.to_dict(profile_g=True),
            "expected_information_gain_millionths": (
                self.expected_information_gain_millionths
            ),
            "proof_feasible": self.proof_feasible,
            "constraint_solution_id": self.constraint_solution_id,
            "failure_memory_record_ids": list(self.failure_memory_record_ids),
        }
        if include_identity:
            payload["record_id"] = self.record_id
            payload["candidate_id"] = self.candidate_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicCandidateRecord":
        expected = {
            "schema",
            "request_id",
            "source",
            "strategy_ids",
            "task_candidate_ids",
            "covered_obligation_ids",
            "schedule",
            "plan",
            "expected_information_gain_millionths",
            "proof_feasible",
            "constraint_solution_id",
            "failure_memory_record_ids",
            "record_id",
            "candidate_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate record must use the closed schema"
            )
        if payload.get("schema") != SYMBOLIC_CANDIDATE_RECORD_SCHEMA:
            raise SymbolicCandidatePlanningError(
                "unsupported symbolic candidate record schema"
            )
        from .adaptive_planner import _decode_profile_candidate

        result = cls(
            request_id=payload.get("request_id", ""),
            source=payload.get("source", ""),
            strategy_ids=tuple(payload.get("strategy_ids") or ()),
            task_candidate_ids=tuple(payload.get("task_candidate_ids") or ()),
            covered_obligation_ids=tuple(
                payload.get("covered_obligation_ids") or ()
            ),
            schedule=PartialOrderSchedule.from_dict(payload.get("schedule") or {}),
            plan=_decode_profile_candidate(payload.get("plan") or {}),
            expected_information_gain_millionths=payload.get(
                "expected_information_gain_millionths", -1
            ),
            proof_feasible=payload.get("proof_feasible"),
            constraint_solution_id=payload.get("constraint_solution_id", ""),
            failure_memory_record_ids=tuple(
                payload.get("failure_memory_record_ids") or ()
            ),
        )
        if payload.get("candidate_id") != result.candidate_id:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate identity projection is inconsistent"
            )
        if payload.get("record_id") != result.record_id:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate record identity mismatch"
            )
        return result


@dataclass(frozen=True)
class SymbolicProviderUsageReceipt:
    provider_id: str
    request_id: str
    status: SymbolicProviderStatus
    attempted: bool
    candidate_ids: tuple[str, ...] = ()
    reason_code: str = ""
    request_bytes: int = 0
    request_sha256: str = ""
    response_bytes: int = 0
    response_sha256: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_milliseconds: int = 0

    def __post_init__(self) -> None:
        for name in ("provider_id", "request_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "status", SymbolicProviderStatus(self.status))
        if not isinstance(self.attempted, bool):
            raise SymbolicCandidatePlanningError("attempted must be boolean")
        object.__setattr__(
            self, "candidate_ids", _ids(self.candidate_ids, "candidate_ids")
        )
        object.__setattr__(
            self,
            "reason_code",
            _id(self.reason_code, "reason_code", allow_empty=True),
        )
        for name in (
            "request_bytes",
            "response_bytes",
            "input_tokens",
            "output_tokens",
            "latency_milliseconds",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 0))
        for name in ("request_sha256", "response_sha256"):
            object.__setattr__(
                self, name, _id(getattr(self, name), name, allow_empty=True)
            )
        if self.status is SymbolicProviderStatus.SUCCEEDED and not self.candidate_ids:
            raise SymbolicCandidatePlanningError(
                "successful provider usage requires admitted candidates"
            )
        if self.status is not SymbolicProviderStatus.SUCCEEDED and self.candidate_ids:
            raise SymbolicCandidatePlanningError(
                "degraded provider usage cannot claim candidates"
            )

    @property
    def usage_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_PROVIDER_USAGE_SCHEMA,
            "provider_id": self.provider_id,
            "request_id": self.request_id,
            "status": self.status.value,
            "attempted": self.attempted,
            "candidate_ids": list(self.candidate_ids),
            "reason_code": self.reason_code,
            "request_bytes": self.request_bytes,
            "request_sha256": self.request_sha256,
            "response_bytes": self.response_bytes,
            "response_sha256": self.response_sha256,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_milliseconds": self.latency_milliseconds,
        }
        if include_identity:
            payload["usage_id"] = self.usage_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicProviderUsageReceipt":
        expected = {
            "schema",
            "provider_id",
            "request_id",
            "status",
            "attempted",
            "candidate_ids",
            "reason_code",
            "request_bytes",
            "request_sha256",
            "response_bytes",
            "response_sha256",
            "input_tokens",
            "output_tokens",
            "latency_milliseconds",
            "usage_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise SymbolicCandidatePlanningError(
                "provider usage must use the closed schema"
            )
        if payload.get("schema") != SYMBOLIC_PROVIDER_USAGE_SCHEMA:
            raise SymbolicCandidatePlanningError("unsupported provider usage schema")
        values = {key: value for key, value in payload.items() if key not in {"schema", "usage_id"}}
        result = cls(**values)
        if payload.get("usage_id") != result.usage_id:
            raise SymbolicCandidatePlanningError("provider usage identity mismatch")
        return result


@dataclass(frozen=True)
class SymbolicCandidateSnapshot:
    symbolic_candidate: SymbolicCandidateRecord
    adaptive_candidate: AdaptivePlanCandidate
    disposition: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.symbolic_candidate, SymbolicCandidateRecord):
            raise SymbolicCandidatePlanningError(
                "symbolic_candidate must be SymbolicCandidateRecord"
            )
        if not isinstance(self.adaptive_candidate, AdaptivePlanCandidate):
            object.__setattr__(
                self,
                "adaptive_candidate",
                AdaptivePlanCandidate.from_dict(self.adaptive_candidate),
            )
        if (
            self.symbolic_candidate.candidate_id
            != self.adaptive_candidate.candidate_id
        ):
            raise SymbolicCandidatePlanningError(
                "symbolic and adaptive candidate identities disagree"
            )
        disposition = _id(self.disposition, "disposition")
        if disposition not in {"selected", "rejected"}:
            raise SymbolicCandidatePlanningError(
                "snapshot disposition must be selected or rejected"
            )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "reason_codes",
            _ids(
                self.reason_codes,
                "reason_codes",
                allow_empty=disposition == "selected",
            ),
        )
        if disposition == "selected" and self.reason_codes:
            raise SymbolicCandidatePlanningError(
                "selected snapshot cannot contain rejection reasons"
            )

    @property
    def candidate_id(self) -> str:
        return self.symbolic_candidate.candidate_id

    @property
    def candidate_snapshot_id(self) -> str:
        return self.adaptive_candidate.snapshot_id

    @property
    def snapshot_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_CANDIDATE_SNAPSHOT_SCHEMA,
            "symbolic_candidate": self.symbolic_candidate.to_dict(),
            "adaptive_candidate": self.adaptive_candidate.to_dict(),
            "candidate_id": self.candidate_id,
            "candidate_snapshot_id": self.candidate_snapshot_id,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
        }
        if include_identity:
            payload["snapshot_id"] = self.snapshot_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicCandidateSnapshot":
        expected = {
            "schema",
            "symbolic_candidate",
            "adaptive_candidate",
            "candidate_id",
            "candidate_snapshot_id",
            "disposition",
            "reason_codes",
            "snapshot_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate snapshot must use the closed schema"
            )
        if payload.get("schema") != SYMBOLIC_CANDIDATE_SNAPSHOT_SCHEMA:
            raise SymbolicCandidatePlanningError(
                "unsupported symbolic candidate snapshot schema"
            )
        result = cls(
            symbolic_candidate=SymbolicCandidateRecord.from_dict(
                payload.get("symbolic_candidate") or {}
            ),
            adaptive_candidate=AdaptivePlanCandidate.from_dict(
                payload.get("adaptive_candidate") or {}
            ),
            disposition=payload.get("disposition", ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        if payload.get("candidate_id") != result.candidate_id:
            raise SymbolicCandidatePlanningError(
                "snapshot candidate identity projection is inconsistent"
            )
        if payload.get("candidate_snapshot_id") != result.candidate_snapshot_id:
            raise SymbolicCandidatePlanningError(
                "adaptive snapshot identity projection is inconsistent"
            )
        if payload.get("snapshot_id") != result.snapshot_id:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate snapshot identity mismatch"
            )
        return result


@dataclass(frozen=True)
class SymbolicCandidatePortfolio:
    """Complete content-addressed generation, gate and selection receipt."""

    request: FrozenSymbolicCandidateRequest
    snapshots: tuple[SymbolicCandidateSnapshot, ...]
    provider_usage: SymbolicProviderUsageReceipt
    adaptive_selection: AdaptivePlanSelectionReceipt
    interface: str = SYMBOLIC_CANDIDATE_PLANNER_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.request, FrozenSymbolicCandidateRequest):
            object.__setattr__(
                self,
                "request",
                FrozenSymbolicCandidateRequest.from_dict(self.request),
            )
        snapshots = tuple(self.snapshots)
        if not snapshots:
            raise SymbolicCandidatePlanningError(
                "candidate portfolio requires its deterministic baseline"
            )
        if len(snapshots) > self.request.bounds.candidate_count:
            raise SymbolicCandidatePlanningError(
                "candidate portfolio exceeds candidate_count"
            )
        if (
            snapshots[0].symbolic_candidate.source
            is not SymbolicCandidateSource.DETERMINISTIC_BASELINE
        ):
            raise SymbolicCandidatePlanningError(
                "first portfolio member must be the deterministic baseline"
            )
        ids = [item.candidate_id for item in snapshots]
        if len(ids) != len(set(ids)):
            raise SymbolicCandidatePlanningError(
                "candidate portfolio contains duplicate identities"
            )
        if any(
            item.symbolic_candidate.request_id != self.request.request_id
            for item in snapshots
        ):
            raise SymbolicCandidatePlanningError(
                "candidate is detached from the frozen symbolic request"
            )
        selected = self.adaptive_selection.selected_candidate_id
        selected_snapshots = [
            item for item in snapshots if item.disposition == "selected"
        ]
        if selected is None:
            if selected_snapshots:
                raise SymbolicCandidatePlanningError(
                    "abstaining selection cannot mark a selected snapshot"
                )
        elif (
            len(selected_snapshots) != 1
            or selected_snapshots[0].candidate_id != selected
        ):
            raise SymbolicCandidatePlanningError(
                "snapshot selection projection is inconsistent"
            )
        if any(
            item.disposition
            != ("selected" if item.candidate_id == selected else "rejected")
            for item in snapshots
        ):
            raise SymbolicCandidatePlanningError(
                "every non-selected candidate must have a rejected snapshot"
            )
        evaluated = {
            item.candidate_id
            for item in self.adaptive_selection.evaluation.ranked
        }
        if evaluated != set(ids):
            raise SymbolicCandidatePlanningError(
                "adaptive selection must cover the exact symbolic population"
            )
        if self.provider_usage.request_id != self.request.request_id:
            raise SymbolicCandidatePlanningError(
                "provider usage is detached from the frozen request"
            )
        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "interface", _id(self.interface, "interface"))
        if self.interface != SYMBOLIC_CANDIDATE_PLANNER_INTERFACE:
            raise SymbolicCandidatePlanningError(
                "unsupported symbolic portfolio interface"
            )

    @property
    def selected(self) -> SymbolicCandidateSnapshot | None:
        return next(
            (item for item in self.snapshots if item.disposition == "selected"),
            None,
        )

    @property
    def rejected(self) -> tuple[SymbolicCandidateSnapshot, ...]:
        return tuple(item for item in self.snapshots if item.disposition == "rejected")

    @property
    def baseline(self) -> SymbolicCandidateSnapshot:
        return self.snapshots[0]

    @property
    def selected_snapshot_id(self) -> str:
        return self.selected.snapshot_id if self.selected is not None else ""

    @property
    def rejected_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(item.snapshot_id for item in self.rejected)

    @property
    def portfolio_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SYMBOLIC_CANDIDATE_PORTFOLIO_SCHEMA,
            "planner_version": SYMBOLIC_CANDIDATE_PLANNER_VERSION,
            "interface": self.interface,
            "request": self.request.to_dict(),
            "snapshots": [item.to_dict() for item in self.snapshots],
            "provider_usage": self.provider_usage.to_dict(),
            "adaptive_selection": self.adaptive_selection.to_dict(),
            "baseline_snapshot_id": self.baseline.snapshot_id,
            "selected_snapshot_id": self.selected_snapshot_id,
            "rejected_snapshot_ids": list(self.rejected_snapshot_ids),
        }
        if include_identity:
            payload["portfolio_id"] = self.portfolio_id
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolicCandidatePortfolio":
        expected = {
            "schema",
            "planner_version",
            "interface",
            "request",
            "snapshots",
            "provider_usage",
            "adaptive_selection",
            "baseline_snapshot_id",
            "selected_snapshot_id",
            "rejected_snapshot_ids",
            "portfolio_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate portfolio must use the closed schema"
            )
        if (
            payload.get("schema") != SYMBOLIC_CANDIDATE_PORTFOLIO_SCHEMA
            or payload.get("planner_version")
            != SYMBOLIC_CANDIDATE_PLANNER_VERSION
        ):
            raise SymbolicCandidatePlanningError(
                "unsupported symbolic candidate portfolio version"
            )
        result = cls(
            request=FrozenSymbolicCandidateRequest.from_dict(
                payload.get("request") or {}
            ),
            snapshots=tuple(
                SymbolicCandidateSnapshot.from_dict(item)
                for item in payload.get("snapshots") or ()
            ),
            provider_usage=SymbolicProviderUsageReceipt.from_dict(
                payload.get("provider_usage") or {}
            ),
            adaptive_selection=AdaptivePlanSelectionReceipt.from_dict(
                payload.get("adaptive_selection") or {}
            ),
            interface=payload.get("interface", ""),
        )
        expected_projections = result.to_dict(include_identity=False)
        for name in (
            "baseline_snapshot_id",
            "selected_snapshot_id",
            "rejected_snapshot_ids",
        ):
            if payload.get(name) != expected_projections[name]:
                raise SymbolicCandidatePlanningError(
                    f"symbolic portfolio {name} projection is inconsistent"
                )
        if payload.get("portfolio_id") != result.portfolio_id:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate portfolio identity mismatch"
            )
        return result

    @classmethod
    def from_json(cls, value: str) -> "SymbolicCandidatePortfolio":
        try:
            payload = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise SymbolicCandidatePlanningError(
                "symbolic candidate portfolio JSON is malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise SymbolicCandidatePlanningError(
                "symbolic candidate portfolio JSON must be an object"
            )
        return cls.from_dict(payload)


def _failure_snapshot(
    value: PlanFailureMemory | PlanFailureMemorySnapshot | None,
) -> PlanFailureMemorySnapshot:
    if value is None:
        return PlanFailureMemory().snapshot()
    if isinstance(value, PlanFailureMemory):
        return value.snapshot()
    if isinstance(value, PlanFailureMemorySnapshot):
        return value
    raise SymbolicCandidatePlanningError(
        "failure_memory must be PlanFailureMemory or PlanFailureMemorySnapshot"
    )


def _task_metadata(context: Mapping[str, Any], task_id: str) -> Mapping[str, Any]:
    all_metadata = context.get("task_metadata", {})
    if not isinstance(all_metadata, Mapping):
        raise SymbolicCandidatePlanningError("task_metadata must be a mapping")
    value = all_metadata.get(task_id, {})
    if not isinstance(value, Mapping):
        raise SymbolicCandidatePlanningError(
            f"task_metadata[{task_id}] must be a mapping"
        )
    return value


def _paths(context: Mapping[str, Any], tasks: Sequence[TaskCandidate]) -> tuple[str, ...]:
    values: list[str] = []
    for task in tasks:
        metadata = _task_metadata(context, task.candidate_id)
        for name in ("predicted_files", "outputs", "paths", "scope_paths"):
            raw = metadata.get(name, ())
            if isinstance(raw, str):
                raw = (raw,)
            if isinstance(raw, Sequence):
                for item in raw:
                    path = (
                        item.get("path", "")
                        if isinstance(item, Mapping)
                        else str(item)
                    ).strip()
                    if path and _PATH.fullmatch(path) and ".." not in path.split("/"):
                        values.append(path)
        for reference in task.provenance_refs:
            if _PATH.fullmatch(reference) and "/" in reference and ":" not in reference:
                values.append(reference)
    if not values:
        for name in ("repository_paths", "predicted_files", "outputs"):
            raw = context.get(name, ())
            if isinstance(raw, str):
                raw = (raw,)
            if isinstance(raw, Sequence):
                for item in raw:
                    path = (
                        item.get("path", "")
                        if isinstance(item, Mapping)
                        else str(item)
                    ).strip()
                    if path and _PATH.fullmatch(path) and ".." not in path.split("/"):
                        values.append(path)
    result = tuple(sorted(set(values)))
    protected = set(str(item) for item in context.get("protected_paths", ()) or ())
    result = tuple(item for item in result if item not in protected)
    if not result:
        raise SymbolicCandidatePlanningError(
            "no codebase-derived, in-scope path supports the deterministic baseline"
        )
    return result


def _schedule(
    selected: Iterable[str],
    by_id: Mapping[str, TaskCandidate],
    *,
    max_tasks: int,
) -> tuple[PartialOrderSchedule, tuple[TaskCandidate, ...]]:
    pending = set(selected)
    frontier = list(pending)
    while frontier:
        task_id = frontier.pop()
        task = by_id.get(task_id)
        if task is None:
            raise SymbolicCandidatePlanningError(
                f"candidate depends on unknown task {task_id}"
            )
        for dependency in task.depends_on_candidate_ids:
            if dependency not in by_id:
                raise SymbolicCandidatePlanningError(
                    f"candidate dependency {dependency} is unavailable"
                )
            if dependency not in pending:
                pending.add(dependency)
                frontier.append(dependency)
        if len(pending) > max_tasks:
            raise SymbolicCandidatePlanningError(
                "candidate exceeds max_tasks_per_candidate"
            )
    edges = tuple(
        sorted(
            (dependency, task_id)
            for task_id in pending
            for dependency in by_id[task_id].depends_on_candidate_ids
            if dependency in pending
        )
    )
    remaining = set(pending)
    completed: set[str] = set()
    waves: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(
                task_id
                for task_id in remaining
                if set(by_id[task_id].depends_on_candidate_ids) <= completed
            )
        )
        if not ready:
            raise SymbolicCandidatePlanningError(
                "task candidate dependencies contain a cycle"
            )
        waves.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)
    return (
        PartialOrderSchedule(tuple(waves), edges),
        tuple(by_id[item] for item in sorted(pending)),
    )


def _candidate_task_sets(
    graph: ObligationGraph,
    *,
    limit: int,
    max_states: int,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate stable plans by backward chaining through AND/OR refinements."""

    nodes = {item.obligation_id: item for item in graph.nodes}
    tasks_by_obligation: dict[str, tuple[str, ...]] = {}
    for node_id in nodes:
        tasks_by_obligation[node_id] = tuple(
            item.candidate_id
            for item in graph.task_candidates
            if node_id in item.closes_obligation_ids
        )
    refinements = {
        node_id: graph.refinements_for(node_id)
        for node_id in nodes
    }
    states = 0
    memo: dict[str, tuple[frozenset[str], ...]] = {}
    active: set[str] = set()

    def combine(
        groups: Sequence[Sequence[frozenset[str]]],
    ) -> tuple[frozenset[str], ...]:
        nonlocal states
        result: list[frozenset[str]] = [frozenset()]
        for group in groups:
            next_result: list[frozenset[str]] = []
            for left in result:
                for right in group:
                    states += 1
                    if states > max_states:
                        raise SymbolicCandidatePlanningError(
                            "symbolic backward-chaining state budget exceeded"
                        )
                    merged = left | right
                    if merged not in next_result:
                        next_result.append(merged)
                    if len(next_result) >= limit:
                        break
                if len(next_result) >= limit:
                    break
            result = next_result
        return tuple(result)

    def solve(node_id: str) -> tuple[frozenset[str], ...]:
        if node_id in memo:
            return memo[node_id]
        if node_id in active:
            raise SymbolicCandidatePlanningError(
                "obligation graph contains a backward-chaining cycle"
            )
        active.add(node_id)
        node = nodes[node_id]
        if node.status is ObligationStatus.DISCHARGED:
            result: tuple[frozenset[str], ...] = (frozenset(),)
        else:
            direct = tuple(
                frozenset((task_id,)) for task_id in tasks_by_obligation[node_id]
            )
            node_refinements = refinements[node_id]
            if node.kind is ObligationNodeKind.PRODUCER:
                base = direct
                if not base:
                    result = ()
                else:
                    and_children = [
                        refinement
                        for refinement in node_refinements
                        if refinement.kind is RefinementKind.AND
                    ]
                    groups: list[Sequence[frozenset[str]]] = [base]
                    for refinement in and_children:
                        groups.extend(
                            solve(child)
                            for child in refinement.child_obligation_ids
                        )
                    result = combine(groups)
            else:
                choices: list[frozenset[str]] = list(direct)
                for refinement in node_refinements:
                    if refinement.kind is RefinementKind.OR:
                        for child in refinement.child_obligation_ids:
                            choices.extend(solve(child))
                    else:
                        choices.extend(
                            combine(
                                [
                                    solve(child)
                                    for child in refinement.child_obligation_ids
                                ]
                            )
                        )
                result = tuple(dict.fromkeys(choices))
        active.remove(node_id)
        memo[node_id] = result[:limit]
        return memo[node_id]

    root_groups = [solve(item) for item in graph.root_obligation_ids]
    if any(not group for group in root_groups):
        raise SymbolicCandidatePlanningError(
            "obligation graph has no task-backed plan for every root"
        )
    plans = combine(root_groups)
    normalized = sorted(
        {tuple(sorted(item)) for item in plans if item},
        key=lambda item: (len(item), item),
    )
    if not normalized:
        raise SymbolicCandidatePlanningError(
            "obligation graph is already complete; no task candidate is invented"
        )
    return tuple(normalized[:limit])


def _failure_prior(
    snapshot: PlanFailureMemorySnapshot,
    scope: FailureMemoryScope | None,
    task_ids: set[str],
    obligation_ids: set[str],
) -> tuple[int, tuple[str, ...]]:
    if scope is None:
        return 0, ()
    matched = []
    weight = 0
    for record in snapshot.records:
        features = record.features
        if features.scope != scope:
            continue
        if not (
            task_ids.intersection(features.step_ids)
            or obligation_ids.intersection(features.obligation_ids)
            or task_ids.intersection(features.alternative_ids)
        ):
            continue
        matched.append(record.diagnostic_id)
        weight += min(250_000, record.occurrence_count * 50_000)
    return min(900_000, weight), tuple(sorted(matched))


def _make_symbolic_record(
    request: FrozenSymbolicCandidateRequest,
    task_ids: Sequence[str],
    *,
    source: SymbolicCandidateSource,
    ordinal: int,
    failure_snapshot: PlanFailureMemorySnapshot,
    failure_scope: FailureMemoryScope | None,
) -> SymbolicCandidateRecord:
    graph = request.obligation_graph
    by_id = {item.candidate_id: item for item in graph.task_candidates}
    schedule, tasks = _schedule(
        task_ids,
        by_id,
        max_tasks=request.bounds.max_tasks_per_candidate,
    )
    covered = tuple(
        sorted(
            {
                obligation_id
                for task in tasks
                for obligation_id in task.closes_obligation_ids
            }
        )
    )
    paths = _paths(request.context, tasks)
    metadata = [_task_metadata(request.context, item.candidate_id) for item in tasks]
    validations = tuple(
        sorted(
            {
                str(command).strip()
                for item in metadata
                for command in (
                    item.get("validation_commands")
                    or item.get("validations")
                    or ()
                )
                if str(command).strip()
            }
        )
    )
    if not validations:
        validations = tuple(
            sorted(
                {
                    requirement
                    for node_id in covered
                    for requirement in graph.node(node_id).validation_requirement_refs
                }
            )
        )
    if not validations:
        validations = ("validation:independent-admission-required",)
    proof_requirements = tuple(
        sorted(
            {
                requirement
                for node_id in covered
                for requirement in graph.node(node_id).proof_requirement_refs
            }
        )
    )
    explicitly_infeasible = set(
        str(item)
        for item in request.context.get("proof_infeasible_obligation_ids", ())
    )
    proof_feasible = (
        not explicitly_infeasible.intersection(covered)
        and not any(
            issue.severity is IssueSeverity.ERROR
            for issue in graph.issues
        )
        and bool(proof_requirements or not request.frozen_goal.policy.require_proof)
    )
    denied_tasks = set(
        str(item)
        for item in request.context.get("authority_denied_task_ids", ())
    )
    unsafe_tasks = set(
        str(item)
        for item in request.context.get("unsafe_task_ids", ())
    )
    selected_ids = set(task_ids)
    prior, memory_ids = _failure_prior(
        failure_snapshot,
        failure_scope,
        selected_ids,
        set(covered),
    )
    evidence_population = {
        reference
        for task in tasks
        for reference in task.provenance_refs
    } | set(proof_requirements) | set(validations)
    denominator = max(
        1,
        len(graph.source_refs)
        + sum(
            len(item.provenance_refs)
            + len(item.proof_requirement_refs)
            + len(item.validation_requirement_refs)
            for item in graph.nodes
        ),
    )
    information_gain = min(
        1_000_000,
        max(1, len(evidence_population)) * 1_000_000 // denominator,
    )
    source_name = source.value
    signature = {
        "request_id": request.request_id,
        "source": source_name,
        "task_candidate_ids": list(sorted(task_ids)),
        "schedule_id": schedule.schedule_id,
    }
    branch_id = "symbolic-" + hashlib.sha256(_json_bytes(signature)).hexdigest()
    dependencies = tuple(
        sorted(
            {
                dependency
                for task in tasks
                for dependency in task.depends_on_candidate_ids
            }
        )
    )
    policy = request.frozen_goal.policy
    allowed = policy.allowed_scopes
    declared_scopes = tuple(
        sorted(
            {
                str(scope)
                for item in metadata
                for scope in (item.get("scope_ids") or ())
                if str(scope).strip()
            }
        )
    )
    changed_scopes = declared_scopes or allowed or paths
    authorized_scopes = allowed or changed_scopes
    resource_classes = tuple(
        sorted(
            {
                str(resource)
                for item in metadata
                for resource in (item.get("resource_classes") or ())
                if str(resource).strip()
            }
        )
    ) or policy.available_resource_classes
    estimated_tokens = sum(
        int(item.get("estimated_tokens", 128)) for item in metadata
    )
    estimated_runtime_ms = sum(
        int(item.get("estimated_runtime_milliseconds", 1_000))
        for item in metadata
    )
    cost_millionths = sum(
        int(item.get("estimated_cost_millionths", 100_000))
        for item in metadata
    )
    risk = min(
        1_000_000,
        prior
        + (250_000 if unsafe_tasks.intersection(selected_ids) else 0),
    )
    branch = PlanBranch(
        branch_id=branch_id,
        summary=(
            "Execute a codebase-derived obligation plan using reviewed producers, "
            "backward chaining, partial-order scheduling, and constraint checks."
        ),
        predicted_files=paths,
        predicted_symbols=tuple(
            sorted(
                {
                    str(symbol)
                    for item in metadata
                    for symbol in (item.get("predicted_symbols") or ())
                    if str(symbol).strip()
                }
            )
        )
        or tuple(f"task:{item.candidate_id}" for item in tasks),
        dependencies=dependencies,
        validation_commands=validations,
        validation_proof=proof_requirements
        or ("proof:independent-admission-required",),
        estimated_cost=cost_millionths / 1_000_000,
        risk=risk / 1_000_000,
        expected_objective_delta=information_gain / 1_000_000,
        source=source_name,
    )
    plan = EvidenceAwarePlanCandidate(
        branch=branch,
        covered_acceptance_criteria=policy.acceptance_criteria,
        covered_evidence_terms=policy.evidence_terms,
        assumptions=policy.trusted_assumptions,
        validated_assumptions=policy.trusted_assumptions,
        semantic_requirements=policy.supported_semantics,
        supported_semantics=policy.supported_semantics,
        dependencies=dependencies,
        critical_path=tuple(
            dependency
            for wave in schedule.waves[:-1]
            for dependency in wave[:1]
            if dependency in dependencies
        ),
        unresolved_conflicts=tuple(
            f"unsafe_task:{item}"
            for item in sorted(unsafe_tasks.intersection(selected_ids))
        ),
        changed_scopes=changed_scopes,
        authorized_scopes=authorized_scopes,
        authority_violations=tuple(
            f"authority_denied_task:{item}"
            for item in sorted(denied_tasks.intersection(selected_ids))
        ),
        validation_feasible=bool(validations),
        proof_feasible=proof_feasible,
        novelty=max(policy.min_novelty, information_gain / 1_000_000),
        resource_classes=resource_classes,
        estimated_resource_cost=cost_millionths / 1_000_000,
        estimated_tokens=estimated_tokens,
        estimated_runtime_seconds=estimated_runtime_ms / 1_000,
    )
    strategies = (
        request.reviewed_template_ids[ordinal % len(request.reviewed_template_ids)],
        "backward_chaining",
        "htn_and_or_refinement",
        "partial_order_scheduling",
        "constraint_solving",
        "proof_feasibility",
        "expected_information_gain",
        "failure_memory",
    )
    return SymbolicCandidateRecord(
        request_id=request.request_id,
        source=source,
        strategy_ids=strategies,
        task_candidate_ids=tuple(task_ids),
        covered_obligation_ids=covered,
        schedule=schedule,
        plan=plan,
        expected_information_gain_millionths=information_gain,
        proof_feasible=proof_feasible,
        constraint_solution_id=content_identity(
            {
                "request_id": request.request_id,
                "schedule_id": schedule.schedule_id,
                "covered_obligation_ids": covered,
                "proof_feasible": proof_feasible,
                "failure_memory_record_ids": memory_ids,
            }
        ),
        failure_memory_record_ids=memory_ids,
    )


def _call_provider(
    provider: Callable[[FrozenSymbolicCandidateRequest], Any],
    request: FrozenSymbolicCandidateRequest,
) -> tuple[bool, Any, int]:
    output: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)
    started = time.monotonic()

    def invoke() -> None:
        try:
            output.put_nowait((True, provider(request)))
        except BaseException as exc:  # provider isolation boundary
            output.put_nowait((False, exc))

    worker = threading.Thread(
        target=invoke,
        name="symbolic-candidate-provider",
        daemon=True,
    )
    worker.start()
    worker.join(request.bounds.provider_timeout_milliseconds / 1_000)
    elapsed = max(0, int((time.monotonic() - started) * 1_000))
    if worker.is_alive():
        return False, TimeoutError("symbolic provider exceeded timeout"), elapsed
    try:
        succeeded, value = output.get_nowait()
    except queue.Empty:
        return False, RuntimeError("symbolic provider returned no result"), elapsed
    return succeeded, value, elapsed


class SymbolicCandidatePlanner:
    """Generate and hard-gate one deterministic-first candidate portfolio."""

    interface = SYMBOLIC_CANDIDATE_PLANNER_INTERFACE

    def __init__(
        self,
        *,
        bounds: SymbolicCandidateBounds | None = None,
        reviewed_template_ids: Sequence[str] = (
            "reviewed-template:obligation-task@1",
            "reviewed-template:dependency-wave@1",
            "reviewed-template:proof-validation@1",
        ),
    ) -> None:
        self.bounds = bounds or SymbolicCandidateBounds()
        self.reviewed_template_ids = _ids(
            reviewed_template_ids,
            "reviewed_template_ids",
            allow_empty=False,
        )

    def plan(
        self,
        obligation_graph: ObligationGraph,
        frozen_goal: FrozenPlanningGoal,
        context: Mapping[str, Any],
        *,
        failure_memory: PlanFailureMemory | PlanFailureMemorySnapshot | None = None,
        failure_scope: FailureMemoryScope | None = None,
        model_provider: Callable[[FrozenSymbolicCandidateRequest], Any] | None = None,
        provider_id: str = "model-proposal",
        allow_model: bool = True,
        hard_gate_evaluator: HardGateEvaluator = deterministic_hard_gate_receipts,
    ) -> SymbolicCandidatePortfolio:
        graph = (
            obligation_graph
            if isinstance(obligation_graph, ObligationGraph)
            else ObligationGraph.from_dict(obligation_graph)
        )
        if graph.planning_blocked:
            reasons = ",".join(item.reason_code for item in graph.issues)
            raise SymbolicCandidatePlanningError(
                "blocked obligation graph cannot generate candidates"
                + (f": {reasons}" if reasons else "")
            )
        if graph.review_required:
            raise SymbolicCandidatePlanningError(
                "review-required obligation graph cannot be silently admitted"
            )
        snapshot = _failure_snapshot(failure_memory)
        if failure_scope is not None and not isinstance(
            failure_scope, FailureMemoryScope
        ):
            raise SymbolicCandidatePlanningError(
                "failure_scope must be FailureMemoryScope"
            )
        request = FrozenSymbolicCandidateRequest(
            obligation_graph=graph,
            frozen_goal=frozen_goal,
            context=context,
            bounds=self.bounds,
            reviewed_template_ids=self.reviewed_template_ids,
            failure_memory_state_id=snapshot.state_id,
            failure_memory_scope_id=(
                failure_scope.scope_id if failure_scope is not None else ""
            ),
        )
        reserve_model = (
            min(self.bounds.max_model_candidates, self.bounds.candidate_count - 1)
            if allow_model and model_provider is not None
            else 0
        )
        deterministic_limit = self.bounds.candidate_count - reserve_model
        task_sets = _candidate_task_sets(
            graph,
            limit=max(1, deterministic_limit),
            max_states=self.bounds.max_search_states,
        )
        records = [
            _make_symbolic_record(
                request,
                task_ids,
                source=(
                    SymbolicCandidateSource.DETERMINISTIC_BASELINE
                    if index == 0
                    else SymbolicCandidateSource.DETERMINISTIC_ALTERNATIVE
                ),
                ordinal=index,
                failure_snapshot=snapshot,
                failure_scope=failure_scope,
            )
            for index, task_ids in enumerate(task_sets[:deterministic_limit])
        ]

        request_bytes = _json_bytes(request.to_dict())
        request_sha = "sha256:" + hashlib.sha256(request_bytes).hexdigest()
        usage = SymbolicProviderUsageReceipt(
            provider_id=provider_id,
            request_id=request.request_id,
            status=(
                SymbolicProviderStatus.DISABLED
                if not allow_model
                else SymbolicProviderStatus.UNAVAILABLE
            ),
            attempted=False,
            reason_code=(
                "model_policy_disabled"
                if not allow_model
                else "provider_not_configured"
            ),
            request_bytes=len(request_bytes),
            request_sha256=request_sha,
        )
        if reserve_model:
            succeeded, raw, elapsed = _call_provider(model_provider, request)
            if not succeeded:
                usage = replace(
                    usage,
                    status=(
                        SymbolicProviderStatus.TIMED_OUT
                        if isinstance(raw, TimeoutError)
                        else SymbolicProviderStatus.FAILED
                    ),
                    attempted=True,
                    reason_code=(
                        "provider_timeout"
                        if isinstance(raw, TimeoutError)
                        else "provider_exception"
                    ),
                    latency_milliseconds=elapsed,
                )
            else:
                try:
                    response_bytes = _json_bytes(raw)
                    if len(response_bytes) > self.bounds.max_provider_response_bytes:
                        raise OverflowError("provider_response_bytes")
                    proposals = (
                        raw.get("candidates", ())
                        if isinstance(raw, Mapping)
                        else raw
                    )
                    if isinstance(proposals, Mapping):
                        proposals = (proposals,)
                    if isinstance(proposals, (str, bytes, bytearray)) or not isinstance(
                        proposals, Iterable
                    ):
                        raise TypeError("provider candidates must be an array")
                    admitted: list[SymbolicCandidateRecord] = []
                    deterministic_task_sets = {
                        item.task_candidate_ids for item in records
                    }
                    for proposal in tuple(proposals)[:reserve_model]:
                        if not isinstance(proposal, Mapping):
                            raise TypeError("provider candidate must be an object")
                        allowed = {
                            "request_id",
                            "task_candidate_ids",
                            "input_tokens",
                            "output_tokens",
                        }
                        if set(proposal) - allowed:
                            raise ValueError("provider candidate has unknown fields")
                        if proposal.get("request_id") != request.request_id:
                            raise ValueError("provider candidate is stale")
                        record = _make_symbolic_record(
                            request,
                            tuple(proposal.get("task_candidate_ids") or ()),
                            source=SymbolicCandidateSource.MODEL_PROPOSAL,
                            ordinal=len(records) + len(admitted),
                            failure_snapshot=snapshot,
                            failure_scope=failure_scope,
                        )
                        if (
                            record.task_candidate_ids not in deterministic_task_sets
                            and record.task_candidate_ids
                            not in {
                                item.task_candidate_ids for item in admitted
                            }
                        ):
                            admitted.append(record)
                    input_tokens = int(
                        raw.get("input_tokens", 0)
                        if isinstance(raw, Mapping)
                        else 0
                    )
                    output_tokens = int(
                        raw.get("output_tokens", 0)
                        if isinstance(raw, Mapping)
                        else 0
                    )
                    if (
                        input_tokens > self.bounds.max_provider_input_tokens
                        or output_tokens > self.bounds.max_provider_output_tokens
                    ):
                        raise OverflowError("provider_token_budget")
                    if admitted:
                        records.extend(admitted)
                        usage = SymbolicProviderUsageReceipt(
                            provider_id=provider_id,
                            request_id=request.request_id,
                            status=SymbolicProviderStatus.SUCCEEDED,
                            attempted=True,
                            candidate_ids=tuple(
                                item.candidate_id for item in admitted
                            ),
                            reason_code="bounded_proposals_admitted",
                            request_bytes=len(request_bytes),
                            request_sha256=request_sha,
                            response_bytes=len(response_bytes),
                            response_sha256=(
                                "sha256:"
                                + hashlib.sha256(response_bytes).hexdigest()
                            ),
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            latency_milliseconds=elapsed,
                        )
                    else:
                        usage = replace(
                            usage,
                            status=SymbolicProviderStatus.MALFORMED,
                            attempted=True,
                            reason_code="no_novel_bounded_candidate",
                            response_bytes=len(response_bytes),
                            response_sha256=(
                                "sha256:"
                                + hashlib.sha256(response_bytes).hexdigest()
                            ),
                            latency_milliseconds=elapsed,
                        )
                except OverflowError:
                    usage = replace(
                        usage,
                        status=SymbolicProviderStatus.BUDGET_REJECTED,
                        attempted=True,
                        reason_code="provider_budget_exceeded",
                        latency_milliseconds=elapsed,
                    )
                except (TypeError, ValueError, SymbolicCandidatePlanningError):
                    usage = replace(
                        usage,
                        status=SymbolicProviderStatus.MALFORMED,
                        attempted=True,
                        reason_code="malformed_or_stale_provider_proposal",
                        latency_milliseconds=elapsed,
                    )

        generation_request = FrozenCandidateGenerationRequest.freeze(
            frozen_goal,
            {
                "symbolic_request_id": request.request_id,
                "obligation_graph_id": graph.graph_id,
                "context_id": request.context_id,
            },
            bounds=CandidateGenerationBounds(
                max_candidates_per_provider=max(1, self.bounds.max_model_candidates),
                max_total_candidates=self.bounds.candidate_count,
                max_input_tokens=self.bounds.max_provider_input_tokens,
                max_output_tokens=self.bounds.max_provider_output_tokens,
                max_response_bytes=self.bounds.max_provider_response_bytes,
                timeout_seconds=self.bounds.provider_timeout_milliseconds / 1_000,
            ),
        )
        adaptive_candidates: list[AdaptivePlanCandidate] = []
        for record in records:
            raw_receipts = hard_gate_evaluator(
                record.plan,
                frozen_goal,
                generation_request,
            )
            if isinstance(raw_receipts, Mapping):
                from .adaptive_planner import _normalize_gate_receipts

                receipts = _normalize_gate_receipts(
                    raw_receipts,
                    plan=record.plan,
                    frozen_goal=frozen_goal,
                    request=generation_request,
                )
            else:
                receipts = tuple(
                    item
                    if isinstance(item, HardConstraintReceipt)
                    else HardConstraintReceipt.from_dict(item)
                    for item in raw_receipts
                )
            adaptive_candidates.append(
                AdaptivePlanCandidate(
                    plan=record.plan,
                    goal_content_id=frozen_goal.goal_content_id,
                    repository_tree_id=frozen_goal.repository_tree_id,
                    policy_digest=frozen_goal.policy_digest,
                    hard_constraint_receipts=receipts,
                )
            )
        selection = AdaptivePlanner(
            max_candidates=self.bounds.candidate_count
        ).select(frozen_goal, adaptive_candidates)
        non_selection = selection.evaluation.non_selection_reasons
        snapshots = tuple(
            SymbolicCandidateSnapshot(
                symbolic_candidate=record,
                adaptive_candidate=adaptive,
                disposition=(
                    "selected"
                    if selection.selected_candidate_id == adaptive.candidate_id
                    else "rejected"
                ),
                reason_codes=(
                    ()
                    if selection.selected_candidate_id == adaptive.candidate_id
                    else non_selection.get(
                        adaptive.candidate_id,
                        ("not_selected",),
                    )
                ),
            )
            for record, adaptive in zip(records, adaptive_candidates)
        )
        return SymbolicCandidatePortfolio(
            request=request,
            snapshots=snapshots,
            provider_usage=usage,
            adaptive_selection=selection,
        )

    generate = plan
    generate_portfolio = plan


def plan_symbolic_candidates(
    obligation_graph: ObligationGraph,
    frozen_goal: FrozenPlanningGoal,
    context: Mapping[str, Any],
    **kwargs: Any,
) -> SymbolicCandidatePortfolio:
    """Functional entry point for ``SymbolicCandidatePlanner@1``."""

    bounds = kwargs.pop("bounds", None)
    reviewed_template_ids = kwargs.pop("reviewed_template_ids", None)
    planner_kwargs: dict[str, Any] = {}
    if bounds is not None:
        planner_kwargs["bounds"] = bounds
    if reviewed_template_ids is not None:
        planner_kwargs["reviewed_template_ids"] = reviewed_template_ids
    return SymbolicCandidatePlanner(**planner_kwargs).plan(
        obligation_graph,
        frozen_goal,
        context,
        **kwargs,
    )


generate_symbolic_candidate_portfolio = plan_symbolic_candidates
compile_symbolic_candidate_portfolio = plan_symbolic_candidates


def admit_operator_candidate_portfolio(
    nominations: Sequence[Any],
    *,
    current_evidence_cid: str,
    portfolio_id: str = "portfolio:symbolic-dcr062",
    **kwargs: Any,
) -> Any:
    """DCR-062 bridge: build and uniquely admit a finite operator portfolio.

    Delegates to :func:`deterministic_candidate_portfolio.build_and_admit_candidate_portfolio`
    so the symbolic planner surface and the deterministic portfolio share one
    unique-admission policy (ties/unknowns abstain; evidence + operator CIDs
    are mandatory).
    """

    from .deterministic_candidate_portfolio import (
        build_and_admit_candidate_portfolio,
    )

    return build_and_admit_candidate_portfolio(
        nominations,
        current_evidence_cid=current_evidence_cid,
        portfolio_id=portfolio_id,
        **kwargs,
    )


# DCR-062 surface: re-export finite portfolio admission symbols so the
# predicted planner package exports remain one import path.
from .deterministic_candidate_portfolio import (  # noqa: E402
    CANDIDATE_ADMISSION_INTERFACE,
    DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
    REPAIR_CANDIDATE_INTERFACE,
    CandidateAdmission,
    CandidateFacts,
    CandidatePortfolio,
    RepairCandidate,
    admit_candidate_portfolio,
    build_deterministic_candidate_portfolio,
)


__all__ = [
    "SYMBOLIC_CANDIDATE_PLANNER_INTERFACE",
    "SYMBOLIC_CANDIDATE_PLANNER_VERSION",
    "SYMBOLIC_CANDIDATE_PORTFOLIO_SCHEMA",
    "CANDIDATE_ADMISSION_INTERFACE",
    "DCR_CANDIDATE_PORTFOLIO_EVIDENCE",
    "REPAIR_CANDIDATE_INTERFACE",
    "CandidateAdmission",
    "CandidateFacts",
    "CandidatePortfolio",
    "FrozenSymbolicCandidateRequest",
    "PartialOrderSchedule",
    "RepairCandidate",
    "SymbolicCandidateBounds",
    "SymbolicCandidatePlanner",
    "SymbolicCandidatePlanningError",
    "SymbolicCandidatePortfolio",
    "SymbolicCandidateRecord",
    "SymbolicCandidateSnapshot",
    "SymbolicCandidateSource",
    "SymbolicProviderStatus",
    "SymbolicProviderUsageReceipt",
    "admit_candidate_portfolio",
    "admit_operator_candidate_portfolio",
    "build_deterministic_candidate_portfolio",
    "compile_symbolic_candidate_portfolio",
    "generate_symbolic_candidate_portfolio",
    "plan_symbolic_candidates",
]
