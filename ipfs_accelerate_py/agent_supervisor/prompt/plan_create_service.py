"""First-class create-plan preview with production wiring (PDR-030).

``PlanCreateService@1`` is the create-specific orchestration boundary:

* the default factory wires production analysis (``PlanningAnalysisFactory``)
  and independent plan admission (``PlanAdmissionService``);
* :meth:`PlanCreateService.preview_create` scans the current scope and runs
  query, evidence, obligation, candidate, critique, admission, and
  parallel-plan stages without mutation;
* previews are body-free, read-only, and restart-serializable;
* stale root/policy observations fail closed rather than silently regenerating;
* deterministic and model-assisted modes share one frozen input/bounds
  snapshot; and
* :meth:`PlanCreateService.workflow_preview` remains the canonical
  compatibility alias for create-plan preview during migration (shared facade
  integration with control CLI/MCP belongs to PDR-032).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.planning_analysis_factory import (
    PLANNING_ANALYSIS_FACTORY_INTERFACE,
    PlanningAnalysisFactory,
    build_planning_analysis_factory,
)
from ..analysis.planning_evidence_bundle import (
    EvidenceQuery,
    PlanningEvidenceBundleCompiler,
)
from ..planning.adaptive_planner import FrozenPlanningGoal
from ..planning.obligation_graph_compiler import (
    ObligationGraph,
    ObligationGraphCompiler,
    ProducerRule,
    TaskCandidate,
    TypedIntent,
    TypedPredicate,
)
from ..planning.parallel_plan_compiler import (
    ParallelPlanCompiler,
    ParallelPlanCompilationRequest,
)
from ..planning.plan_admission_service import (
    PLAN_ADMISSION_SERVICE_INTERFACE,
    PlanAdmissionService,
)
from ..planning.plan_analysis_query_planner import (
    PlanAnalysisQueryPlanner,
    ReasoningQueryPlan,
)
from ..planning.plan_critic import PlanCritic, PlanCritique
from ..planning.plan_evaluator import EvidenceAwarePlanPolicy
from ..planning.plan_revision_contracts import (
    DirtyTreePolicy,
    FallbackPolicy,
    PlanAuthorityRoots,
    PlanCreateRequest,
    PlanRequestBudget,
    PlanRevisionStaleRootError,
    TaskSourceKind,
    plan_revision_cid,
)
from ..planning.symbolic_candidate_planner import (
    SymbolicCandidateBounds,
    SymbolicCandidatePlanner,
    SymbolicCandidatePortfolio,
)
PLAN_CREATE_SERVICE_INTERFACE: Final[str] = "PlanCreateService@1"
PLAN_CREATE_SERVICE_VERSION: Final[int] = 1
PLAN_CREATE_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-create-preview-receipt@1"
)
PLAN_CREATE_STAGE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-create-stage-result@1"
)
PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-create-input-snapshot@1"
)
PLAN_CREATE_MATERIALS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-create-materials@1"
)

# Existing workflow preview remains a canonical compatibility alias for create.
WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS: Final[str] = "workflow_preview"

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_log",
        "source_body",
        "source_text",
        "transcript",
    }
)
_SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_log",
        "refresh_token",
        "secret",
        "session_token",
        "source_body",
        "source_text",
        "token",
    }
)


class PlanCreateServiceError(RuntimeError):
    """Create-plan preview cannot proceed without weakening authority."""


class PlanCreateStaleRootError(PlanCreateServiceError):
    """Observed roots/policy differ from the bound request; refuse regenerate."""


class PlanCreateBodyError(PlanCreateServiceError):
    """Preview records must remain body-free and secret-free."""


class PlanCreateMode(str, Enum):
    """Planning modes that share exact frozen inputs and bounds."""

    DETERMINISTIC = "deterministic"
    MODEL_ASSISTED = "model_assisted"


class PlanCreateStage(str, Enum):
    """Fixed-order create-plan pipeline stages. Order is authority."""

    SCAN = "scan"
    QUERY = "query"
    EVIDENCE = "evidence"
    OBLIGATION = "obligation"
    CANDIDATE = "candidate"
    CRITIQUE = "critique"
    ADMISSION = "admission"
    PARALLEL_PLAN = "parallel_plan"


CREATE_STAGE_ORDER: Final[tuple[PlanCreateStage, ...]] = (
    PlanCreateStage.SCAN,
    PlanCreateStage.QUERY,
    PlanCreateStage.EVIDENCE,
    PlanCreateStage.OBLIGATION,
    PlanCreateStage.CANDIDATE,
    PlanCreateStage.CRITIQUE,
    PlanCreateStage.ADMISSION,
    PlanCreateStage.PARALLEL_PLAN,
)


class PlanCreateVerdict(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    REVIEW_ONLY = "review_only"


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise PlanCreateBodyError("floating-point values are not canonical")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PlanCreateBodyError("mapping keys must be strings")
        return {
            key: _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonical(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _canonical(converter())
    raise PlanCreateBodyError(
        f"unsupported create-plan value type {type(value).__name__}"
    )


def _assert_body_free(value: Any, label: str = "record") -> None:
    if isinstance(value, float):
        raise PlanCreateBodyError(f"{label} may not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlanCreateBodyError(f"{label} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS or normalized in _SECRET_KEYS:
                raise PlanCreateBodyError(
                    f"{label} may not contain secrets or source bodies ({key})"
                )
            if any(
                marker in normalized
                for marker in ("password", "private_key", "access_token", "api_key")
            ):
                raise PlanCreateBodyError(
                    f"{label} may not contain secrets or source bodies ({key})"
                )
            _assert_body_free(item, label)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_body_free(item, label)
    elif isinstance(value, (bytes, bytearray)):
        raise PlanCreateBodyError(f"{label} may not contain binary bodies")


def _digest(namespace: str, value: Any) -> str:
    payload = _canonical(value)
    _assert_body_free(payload, namespace)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        f"{namespace}\n".encode("utf-8") + encoded
    ).hexdigest()


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        payload = converter()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise PlanCreateServiceError("expected a mapping or to_dict() record")


def _record_cid(value: Any, *, fallback_namespace: str = "stage") -> str:
    if value is None:
        return ""
    for attr in (
        "content_id",
        "receipt_id",
        "plan_id",
        "query_plan_id",
        "bundle_id",
        "critique_id",
        "portfolio_id",
        "graph_id",
        "scan_cid",
    ):
        observed = getattr(value, attr, None)
        if isinstance(observed, str) and observed:
            return observed
    if isinstance(value, Mapping):
        for key in (
            "content_id",
            "cid",
            "receipt_id",
            "plan_id",
            "query_plan_id",
            "bundle_id",
            "critique_id",
            "portfolio_id",
            "graph_id",
            "scan_cid",
        ):
            observed = value.get(key)
            if isinstance(observed, str) and observed:
                return observed
        return _digest(fallback_namespace, value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _digest(fallback_namespace, converter())
    return _digest(fallback_namespace, str(type(value).__name__))


def _coerce_mode(value: Any) -> PlanCreateMode:
    if isinstance(value, PlanCreateMode):
        return value
    if isinstance(value, str):
        try:
            return PlanCreateMode(value)
        except ValueError as exc:
            raise PlanCreateServiceError(
                f"unknown plan create mode: {value!r}"
            ) from exc
    raise PlanCreateServiceError("mode must be PlanCreateMode or its value")


def _coerce_request(value: Any) -> PlanCreateRequest:
    if isinstance(value, PlanCreateRequest):
        return value
    if isinstance(value, Mapping):
        if value.get("schema") == PlanCreateRequest.SCHEMA:
            return PlanCreateRequest.from_dict(value)
        return PlanCreateRequest(**dict(value))
    raise PlanCreateServiceError(
        "request must be PlanCreateRequest or a mapping"
    )


def _coerce_roots(value: Any) -> PlanAuthorityRoots | None:
    if value is None:
        return None
    if isinstance(value, PlanAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        if value.get("schema") == PlanAuthorityRoots.SCHEMA:
            return PlanAuthorityRoots.from_dict(value)
        return PlanAuthorityRoots(**dict(value))
    raise PlanCreateServiceError(
        "roots must be PlanAuthorityRoots or a mapping"
    )


@dataclass(frozen=True)
class PlanCreateInputSnapshot:
    """Exact request/root/budget bindings shared by every planning mode."""

    SCHEMA: ClassVar[str] = PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA

    request_cid: str
    repository_id: str
    repository_root: str
    scope_paths: tuple[str, ...]
    roots: PlanAuthorityRoots
    budget: PlanRequestBudget
    required_analysis_operations: tuple[str, ...]
    optional_analysis_operations: tuple[str, ...]
    required_logic_families: tuple[str, ...]
    optional_logic_families: tuple[str, ...]
    fallback_policy: FallbackPolicy
    supervisor_profile: str
    board_namespace: str
    alias_prefix: str
    task_source_kind: TaskSourceKind
    dirty_tree_policy: DirtyTreePolicy
    bounds_digest: str
    snapshot_cid: str = ""

    def __post_init__(self) -> None:
        material = self._material()
        _assert_body_free(material, "plan create input snapshot")
        computed = _digest("plan-create-input-snapshot", material)
        if self.snapshot_cid and self.snapshot_cid != computed:
            raise PlanCreateServiceError(
                "plan create input snapshot identity is invalid"
            )
        object.__setattr__(self, "snapshot_cid", computed)

    def _material(self) -> dict[str, Any]:
        return {
            "schema": PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA,
            "request_cid": self.request_cid,
            "repository_id": self.repository_id,
            "repository_root": self.repository_root,
            "scope_paths": list(self.scope_paths),
            "roots": self.roots.to_dict(),
            "budget": self.budget.to_dict(),
            "required_analysis_operations": list(
                self.required_analysis_operations
            ),
            "optional_analysis_operations": list(
                self.optional_analysis_operations
            ),
            "required_logic_families": list(self.required_logic_families),
            "optional_logic_families": list(self.optional_logic_families),
            "fallback_policy": self.fallback_policy.value,
            "supervisor_profile": self.supervisor_profile,
            "board_namespace": self.board_namespace,
            "alias_prefix": self.alias_prefix,
            "task_source_kind": self.task_source_kind.value,
            "dirty_tree_policy": self.dirty_tree_policy.value,
            "bounds_digest": self.bounds_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._material(), "snapshot_cid": self.snapshot_cid}

    @classmethod
    def from_request(cls, request: PlanCreateRequest) -> "PlanCreateInputSnapshot":
        bounds = {
            "budget": request.budget.to_dict(),
            "required_analysis_operations": list(
                request.required_analysis_operations
            ),
            "optional_analysis_operations": list(
                request.optional_analysis_operations
            ),
            "required_logic_families": list(request.required_logic_families),
            "optional_logic_families": list(request.optional_logic_families),
            "fallback_policy": request.fallback_policy.value,
            "scope_paths": list(request.scope_paths),
        }
        return cls(
            request_cid=request.request_cid,
            repository_id=request.repository_id,
            repository_root=request.repository_root,
            scope_paths=tuple(request.scope_paths),
            roots=request.roots,
            budget=request.budget,
            required_analysis_operations=tuple(
                request.required_analysis_operations
            ),
            optional_analysis_operations=tuple(
                request.optional_analysis_operations
            ),
            required_logic_families=tuple(request.required_logic_families),
            optional_logic_families=tuple(request.optional_logic_families),
            fallback_policy=request.fallback_policy,
            supervisor_profile=request.supervisor_profile,
            board_namespace=request.board_namespace,
            alias_prefix=request.alias_prefix,
            task_source_kind=request.task_source_kind,
            dirty_tree_policy=request.dirty_tree_policy,
            bounds_digest=_digest("plan-create-bounds", bounds),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCreateInputSnapshot":
        if not isinstance(payload, Mapping):
            raise PlanCreateServiceError("input snapshot payload must be a mapping")
        if payload.get("schema") not in (None, PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA):
            raise PlanCreateServiceError("unsupported input snapshot schema")
        roots = _coerce_roots(payload.get("roots"))
        if roots is None:
            raise PlanCreateServiceError("input snapshot requires roots")
        budget_raw = payload.get("budget")
        if isinstance(budget_raw, PlanRequestBudget):
            budget = budget_raw
        elif isinstance(budget_raw, Mapping):
            budget = (
                PlanRequestBudget.from_dict(budget_raw)
                if budget_raw.get("schema") == PlanRequestBudget.SCHEMA
                else PlanRequestBudget(**dict(budget_raw))
            )
        else:
            raise PlanCreateServiceError("input snapshot requires budget")
        value = cls(
            request_cid=str(payload.get("request_cid") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            repository_root=str(payload.get("repository_root") or ""),
            scope_paths=tuple(payload.get("scope_paths") or ()),
            roots=roots,
            budget=budget,
            required_analysis_operations=tuple(
                payload.get("required_analysis_operations") or ()
            ),
            optional_analysis_operations=tuple(
                payload.get("optional_analysis_operations") or ()
            ),
            required_logic_families=tuple(
                payload.get("required_logic_families") or ()
            ),
            optional_logic_families=tuple(
                payload.get("optional_logic_families") or ()
            ),
            fallback_policy=FallbackPolicy(
                str(payload.get("fallback_policy") or FallbackPolicy.FAIL_CLOSED.value)
            ),
            supervisor_profile=str(payload.get("supervisor_profile") or ""),
            board_namespace=str(payload.get("board_namespace") or ""),
            alias_prefix=str(payload.get("alias_prefix") or ""),
            task_source_kind=TaskSourceKind(
                str(payload.get("task_source_kind") or TaskSourceKind.BOTH.value)
            ),
            dirty_tree_policy=DirtyTreePolicy(
                str(
                    payload.get("dirty_tree_policy")
                    or DirtyTreePolicy.OBSERVE_AND_BIND.value
                )
            ),
            bounds_digest=str(payload.get("bounds_digest") or ""),
            snapshot_cid=str(payload.get("snapshot_cid") or ""),
        )
        if payload.get("snapshot_cid") not in (None, "", value.snapshot_cid):
            raise PlanCreateServiceError(
                "stored input snapshot identity does not match the canonical record"
            )
        return value


@dataclass(frozen=True)
class PlanCreateStageResult:
    """Body-free receipt for one create-plan pipeline stage."""

    SCHEMA: ClassVar[str] = PLAN_CREATE_STAGE_RESULT_SCHEMA

    stage: PlanCreateStage
    artifact_cid: str
    passed: bool
    blockers: tuple[str, ...] = ()
    detail_ids: tuple[str, ...] = ()
    message: str = ""
    result_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage",
            self.stage
            if isinstance(self.stage, PlanCreateStage)
            else PlanCreateStage(str(self.stage)),
        )
        object.__setattr__(self, "artifact_cid", str(self.artifact_cid or ""))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(
            self,
            "blockers",
            tuple(str(item) for item in self.blockers if str(item)),
        )
        object.__setattr__(
            self,
            "detail_ids",
            tuple(str(item) for item in self.detail_ids if str(item)),
        )
        object.__setattr__(self, "message", str(self.message or ""))
        material = self._material()
        _assert_body_free(material, "plan create stage result")
        computed = _digest("plan-create-stage-result", material)
        if self.result_cid and self.result_cid != computed:
            raise PlanCreateServiceError(
                "plan create stage result identity is invalid"
            )
        object.__setattr__(self, "result_cid", computed)

    def _material(self) -> dict[str, Any]:
        return {
            "schema": PLAN_CREATE_STAGE_RESULT_SCHEMA,
            "stage": self.stage.value,
            "artifact_cid": self.artifact_cid,
            "passed": self.passed,
            "blockers": list(self.blockers),
            "detail_ids": list(self.detail_ids),
            "message": self.message,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._material(), "result_cid": self.result_cid}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCreateStageResult":
        value = cls(
            stage=PlanCreateStage(str(payload.get("stage") or "")),
            artifact_cid=str(payload.get("artifact_cid") or ""),
            passed=bool(payload.get("passed")),
            blockers=tuple(payload.get("blockers") or ()),
            detail_ids=tuple(payload.get("detail_ids") or ()),
            message=str(payload.get("message") or ""),
            result_cid=str(payload.get("result_cid") or ""),
        )
        if payload.get("result_cid") not in (None, "", value.result_cid):
            raise PlanCreateServiceError(
                "stored stage result identity does not match the canonical record"
            )
        return value


@dataclass(frozen=True)
class PlanCreatePreviewReceipt:
    """Body-free, restart-serializable create-plan preview receipt."""

    SCHEMA: ClassVar[str] = PLAN_CREATE_PREVIEW_SCHEMA

    request_cid: str
    input_snapshot_cid: str
    mode: PlanCreateMode
    verdict: PlanCreateVerdict
    roots: PlanAuthorityRoots
    stage_results: tuple[PlanCreateStageResult, ...]
    scan_cid: str = ""
    query_plan_cid: str = ""
    evidence_bundle_cid: str = ""
    obligation_graph_cid: str = ""
    candidate_portfolio_cid: str = ""
    critique_cid: str = ""
    admission_receipt_cid: str = ""
    execution_plan_cid: str = ""
    plan_root_cid: str = ""
    rejection_reasons: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    read_only: bool = True
    wrote_effects: tuple[str, ...] = ()
    compatibility_alias: str = ""
    receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            self.mode
            if isinstance(self.mode, PlanCreateMode)
            else PlanCreateMode(str(self.mode)),
        )
        object.__setattr__(
            self,
            "verdict",
            self.verdict
            if isinstance(self.verdict, PlanCreateVerdict)
            else PlanCreateVerdict(str(self.verdict)),
        )
        if not isinstance(self.roots, PlanAuthorityRoots):
            roots = _coerce_roots(self.roots)
            if roots is None:
                raise PlanCreateServiceError("preview requires authority roots")
            object.__setattr__(self, "roots", roots)
        stages = tuple(
            item
            if isinstance(item, PlanCreateStageResult)
            else PlanCreateStageResult.from_dict(item)
            for item in self.stage_results
        )
        object.__setattr__(self, "stage_results", stages)
        for name in (
            "request_cid",
            "input_snapshot_cid",
            "scan_cid",
            "query_plan_cid",
            "evidence_bundle_cid",
            "obligation_graph_cid",
            "candidate_portfolio_cid",
            "critique_cid",
            "admission_receipt_cid",
            "execution_plan_cid",
            "plan_root_cid",
            "compatibility_alias",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or ""))
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(item) for item in self.rejection_reasons if str(item)),
        )
        object.__setattr__(
            self,
            "artifact_refs",
            tuple(sorted({str(item) for item in self.artifact_refs if str(item)})),
        )
        object.__setattr__(self, "read_only", bool(self.read_only))
        object.__setattr__(
            self,
            "wrote_effects",
            tuple(str(item) for item in self.wrote_effects if str(item)),
        )
        if self.wrote_effects:
            raise PlanCreateServiceError(
                "create-plan preview is proposal-only and must not record writes"
            )
        if not self.read_only:
            raise PlanCreateServiceError(
                "create-plan preview must be read-only"
            )
        material = self._material()
        _assert_body_free(material, "plan create preview receipt")
        computed = _digest("plan-create-preview-receipt", material)
        if self.receipt_cid and self.receipt_cid != computed:
            raise PlanCreateServiceError(
                "plan create preview receipt identity is invalid"
            )
        object.__setattr__(self, "receipt_cid", computed)

    def _material(self) -> dict[str, Any]:
        return {
            "schema": PLAN_CREATE_PREVIEW_SCHEMA,
            "interface": PLAN_CREATE_SERVICE_INTERFACE,
            "request_cid": self.request_cid,
            "input_snapshot_cid": self.input_snapshot_cid,
            "mode": self.mode.value,
            "verdict": self.verdict.value,
            "roots": self.roots.to_dict(),
            "stage_results": [item.to_dict() for item in self.stage_results],
            "scan_cid": self.scan_cid,
            "query_plan_cid": self.query_plan_cid,
            "evidence_bundle_cid": self.evidence_bundle_cid,
            "obligation_graph_cid": self.obligation_graph_cid,
            "candidate_portfolio_cid": self.candidate_portfolio_cid,
            "critique_cid": self.critique_cid,
            "admission_receipt_cid": self.admission_receipt_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "plan_root_cid": self.plan_root_cid,
            "rejection_reasons": list(self.rejection_reasons),
            "artifact_refs": list(self.artifact_refs),
            "read_only": self.read_only,
            "wrote_effects": list(self.wrote_effects),
            "compatibility_alias": self.compatibility_alias,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._material(), "receipt_cid": self.receipt_cid}

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanCreatePreviewReceipt":
        if not isinstance(payload, Mapping):
            raise PlanCreateServiceError("preview receipt payload must be a mapping")
        if payload.get("schema") not in (None, PLAN_CREATE_PREVIEW_SCHEMA):
            raise PlanCreateServiceError("unsupported preview receipt schema")
        roots = _coerce_roots(payload.get("roots"))
        if roots is None:
            raise PlanCreateServiceError("preview receipt requires roots")
        stages = tuple(
            PlanCreateStageResult.from_dict(item)
            if isinstance(item, Mapping)
            else item
            for item in (payload.get("stage_results") or ())
        )
        value = cls(
            request_cid=str(payload.get("request_cid") or ""),
            input_snapshot_cid=str(payload.get("input_snapshot_cid") or ""),
            mode=PlanCreateMode(str(payload.get("mode") or PlanCreateMode.DETERMINISTIC.value)),
            verdict=PlanCreateVerdict(
                str(payload.get("verdict") or PlanCreateVerdict.BLOCKED.value)
            ),
            roots=roots,
            stage_results=stages,
            scan_cid=str(payload.get("scan_cid") or ""),
            query_plan_cid=str(payload.get("query_plan_cid") or ""),
            evidence_bundle_cid=str(payload.get("evidence_bundle_cid") or ""),
            obligation_graph_cid=str(payload.get("obligation_graph_cid") or ""),
            candidate_portfolio_cid=str(
                payload.get("candidate_portfolio_cid") or ""
            ),
            critique_cid=str(payload.get("critique_cid") or ""),
            admission_receipt_cid=str(payload.get("admission_receipt_cid") or ""),
            execution_plan_cid=str(payload.get("execution_plan_cid") or ""),
            plan_root_cid=str(payload.get("plan_root_cid") or ""),
            rejection_reasons=tuple(payload.get("rejection_reasons") or ()),
            artifact_refs=tuple(payload.get("artifact_refs") or ()),
            read_only=bool(payload.get("read_only", True)),
            wrote_effects=tuple(payload.get("wrote_effects") or ()),
            compatibility_alias=str(payload.get("compatibility_alias") or ""),
            receipt_cid=str(payload.get("receipt_cid") or ""),
        )
        if payload.get("receipt_cid") not in (None, "", value.receipt_cid):
            raise PlanCreateServiceError(
                "stored preview receipt identity does not match the canonical record"
            )
        return value

    @classmethod
    def from_json(cls, text: str) -> "PlanCreatePreviewReceipt":
        return cls.from_dict(json.loads(text))

    @property
    def admitted(self) -> bool:
        return self.verdict is PlanCreateVerdict.ADMITTED

    @property
    def stage_order(self) -> tuple[str, ...]:
        return tuple(item.stage.value for item in self.stage_results)


@dataclass
class PlanCreateMaterials:
    """Optional live materials and stage enrichments for one preview.

    All fields are proposal/evidence tier.  The service never treats caller
    ``admitted`` / ``valid`` claims as authority.
    """

    scan: Any = None
    query_plan: ReasoningQueryPlan | Mapping[str, Any] | None = None
    evidence_bundle: Any = None
    evidence_adapters: Mapping[str, Any] | None = None
    evidence_queries: Sequence[Any] = ()
    obligation_graph: ObligationGraph | Mapping[str, Any] | None = None
    intent: TypedIntent | Mapping[str, Any] | None = None
    current_facts: Sequence[Any] = ()
    producers: Sequence[Any] = ()
    task_candidates: Sequence[Any] = ()
    predicates: Sequence[Any] = ()
    frozen_goal: FrozenPlanningGoal | Mapping[str, Any] | None = None
    candidate_context: Mapping[str, Any] = field(default_factory=dict)
    model_provider: Callable[..., Any] | None = None
    parallel_tasks: Sequence[Any] | None = None
    parallel_request: ParallelPlanCompilationRequest | Mapping[str, Any] | None = None
    admission_materials: Any = None
    current_roots: PlanAuthorityRoots | Mapping[str, Any] | None = None
    workflow_request: Any = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_binding_dict(self) -> dict[str, Any]:
        """Body-free binding projection used in input digests (no stage bodies)."""

        payload = {
            "schema": PLAN_CREATE_MATERIALS_SCHEMA,
            "scan_cid": _record_cid(self.scan, fallback_namespace="scan"),
            "query_plan_cid": _record_cid(
                self.query_plan, fallback_namespace="query"
            ),
            "evidence_bundle_cid": _record_cid(
                self.evidence_bundle, fallback_namespace="evidence"
            ),
            "obligation_graph_cid": _record_cid(
                self.obligation_graph, fallback_namespace="obligation"
            ),
            "frozen_goal_cid": _record_cid(
                self.frozen_goal, fallback_namespace="frozen-goal"
            ),
            "has_model_provider": self.model_provider is not None,
            "has_admission_materials": self.admission_materials is not None,
            "has_workflow_request": self.workflow_request is not None,
            "evidence_adapter_slots": sorted(
                str(key) for key in dict(self.evidence_adapters or {})
            ),
            "current_roots": (
                _coerce_roots(self.current_roots).to_dict()
                if self.current_roots is not None
                else {}
            ),
            "extra_keys": sorted(str(key) for key in dict(self.extra or {})),
        }
        _assert_body_free(payload, "plan create materials binding")
        return payload


def freeze_plan_create_input_snapshot(
    request: PlanCreateRequest | Mapping[str, Any],
) -> PlanCreateInputSnapshot:
    """Freeze the exact inputs and bounds shared by every planning mode."""

    return PlanCreateInputSnapshot.from_request(_coerce_request(request))


def plan_create_request_from_workflow(
    workflow_request: Any,
    *,
    repository_id: str = "",
    board_namespace: str = "workflow-create-alias",
    alias_prefix: str = "WF",
    task_source_id: str = "task-source:workflow-alias",
    task_source_revision: str = "",
    capability_catalog_root: str = "",
    provider_catalog_root: str = "",
    usage_policy_root: str = "",
    configuration_root: str = "",
    dirty_worktree_root: str = "",
) -> PlanCreateRequest:
    """Project a prompt workflow request into a create-plan request.

    Used by the workflow-preview compatibility alias so create and workflow
    surfaces share one pipeline rather than a parallel implementation.
    """

    if workflow_request is None:
        raise PlanCreateServiceError("workflow_request is required")
    repository_root = str(getattr(workflow_request, "repository_root", "") or "")
    directory = str(getattr(workflow_request, "directory", "") or repository_root)
    if not repository_root:
        raise PlanCreateServiceError(
            "workflow request is missing repository_root"
        )
    root_path = Path(repository_root)
    directory_path = Path(directory)
    try:
        relative = directory_path.resolve().relative_to(root_path.resolve())
        scope = "." if str(relative) in ("", ".") else str(relative).replace("\\", "/")
    except Exception:
        scope = "."
    prompt_source = getattr(workflow_request, "prompt_source", None)
    prompt_cid = str(
        getattr(prompt_source, "content_id", None)
        or getattr(workflow_request, "prompt_cid", None)
        or getattr(workflow_request, "request_cid", None)
        or plan_revision_cid({"workflow_request": "missing-prompt"})
    )
    repository_root_cid = str(
        getattr(workflow_request, "repository_root_cid", "") or ""
    )
    if not repository_root_cid:
        raise PlanCreateServiceError(
            "workflow request is missing repository_root_cid"
        )
    budget_src = getattr(workflow_request, "budget", None)
    budget = PlanRequestBudget(
        max_goals=int(getattr(budget_src, "max_goals", 64) or 64),
        max_tasks=int(getattr(budget_src, "max_tasks", 256) or 256),
        max_graph_depth=int(getattr(budget_src, "max_graph_depth", 16) or 16),
        max_output_paths=int(getattr(budget_src, "max_output_paths", 1024) or 1024),
        max_ready_width=1,
        max_repair_rounds=2,
        max_scan_bytes=int(getattr(budget_src, "max_scan_bytes", 8 * 1024 * 1024) or 8 * 1024 * 1024),
        max_analysis_operations=16,
        max_evidence_items=int(getattr(budget_src, "max_evidence", 128) or 128),
        max_logic_families=4,
        max_model_calls=2 if bool(getattr(getattr(workflow_request, "planning_policy", None), "allow_model", False)) else 0,
        max_latency_ms=int(getattr(budget_src, "max_latency_ms", 80_000) or 80_000),
        max_provider_tokens=int(
            getattr(budget_src, "max_provider_tokens", 8_192) or 8_192
        ),
        max_cost_micros=0,
    )
    output_policy = getattr(workflow_request, "output_policy", None)
    mode = str(getattr(getattr(output_policy, "mode", None), "value", getattr(output_policy, "mode", "both")) or "both")
    if mode == "markdown":
        task_source_kind = TaskSourceKind.MARKDOWN
    elif mode == "duckdb":
        task_source_kind = TaskSourceKind.DUCKDB
    else:
        task_source_kind = TaskSourceKind.BOTH
    policy_root = str(getattr(workflow_request, "policy_root", "") or "")
    program_root = str(getattr(workflow_request, "program_root", "") or "")
    intent_ir_root = str(getattr(workflow_request, "intent_ir_root", "") or "")
    legal_ir_root = str(getattr(workflow_request, "legal_ir_root", "") or "")
    security_ir_root = str(getattr(workflow_request, "security_ir_root", "") or "")
    for name, value in (
        ("policy_root", policy_root),
        ("program_root", program_root),
        ("intent_ir_root", intent_ir_root),
        ("legal_ir_root", legal_ir_root),
        ("security_ir_root", security_ir_root),
    ):
        if not value:
            raise PlanCreateServiceError(
                f"workflow request is missing {name}"
            )
    resolved_repo_id = repository_id or f"repository:{repository_root_cid}"
    resolved_task_revision = task_source_revision or plan_revision_cid(
        {
            "workflow_request_cid": str(
                getattr(workflow_request, "request_cid", "") or ""
            ),
            "allowlist_cid": str(
                getattr(workflow_request, "allowlist_cid", "") or ""
            ),
        }
    )
    resolved_capability = capability_catalog_root or plan_revision_cid(
        {"capability": "workflow-alias"}
    )
    resolved_provider = provider_catalog_root or plan_revision_cid(
        {"provider": "workflow-alias"}
    )
    resolved_usage = usage_policy_root or plan_revision_cid(
        {"usage": "workflow-alias"}
    )
    resolved_dirty = dirty_worktree_root or repository_root_cid
    roots = PlanAuthorityRoots(
        repository_id=resolved_repo_id,
        repository_root_cid=repository_root_cid,
        dirty_worktree_root=resolved_dirty,
        task_source_id=task_source_id,
        task_source_revision=resolved_task_revision,
        policy_root=policy_root,
        intent_ir_root=intent_ir_root,
        legal_ir_root=legal_ir_root,
        security_ir_root=security_ir_root,
        program_root=program_root,
        capability_catalog_root=resolved_capability,
        provider_catalog_root=resolved_provider,
        usage_policy_root=resolved_usage,
        configuration_root=configuration_root
        or str(getattr(workflow_request, "state_root", "") or ""),
    )
    return PlanCreateRequest(
        prompt_source_cid=prompt_cid,
        repository_id=resolved_repo_id,
        repository_root=repository_root,
        scope_paths=(scope,),
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND,
        task_source_kind=task_source_kind,
        board_namespace=board_namespace,
        alias_prefix=alias_prefix,
        roots=roots,
        budget=budget,
        required_analysis_operations=(),
        optional_analysis_operations=(),
        required_logic_families=(),
        optional_logic_families=(),
        fallback_policy=FallbackPolicy.FAIL_CLOSED,
        supervisor_profile=str(
            getattr(workflow_request, "supervisor_profile", "")
            or "implementation-daemon"
        ),
        observe_roots=True,
        redacted_source_metadata={
            "source": "workflow_preview_alias",
            "workflow_request_cid": str(
                getattr(workflow_request, "request_cid", "") or ""
            ),
        },
        caller=str(getattr(workflow_request, "caller", "principal:unknown") or "principal:unknown"),
        idempotency_key=str(
            getattr(workflow_request, "idempotency_key", "") or ""
        ),
    )


class PlanCreateService:
    """Create-plan preview orchestrator (``PlanCreateService@1``).

    Preview is proposal-only: it never materializes task sources, never writes
    boards, and never applies effects.  Apply remains a separate CAS/fenced
    transaction (PDR-031+).
    """

    INTERFACE: Final[str] = PLAN_CREATE_SERVICE_INTERFACE
    VERSION: Final[int] = PLAN_CREATE_SERVICE_VERSION

    def __init__(
        self,
        *,
        analysis_factory: PlanningAnalysisFactory | None = None,
        optional_analysis: Any | None = None,
        admission_request_factory: Any | None = None,
        admission_service: PlanAdmissionService | None = None,
        query_planner: PlanAnalysisQueryPlanner | None = None,
        obligation_compiler: ObligationGraphCompiler | None = None,
        candidate_planner: SymbolicCandidatePlanner | None = None,
        critic: PlanCritic | None = None,
        parallel_compiler: ParallelPlanCompiler | None = None,
        scanner: Any | None = None,
        root_observer: Callable[..., Mapping[str, Any]] | None = None,
        receipt_store: MutableMapping[str, Mapping[str, Any]] | Any | None = None,
        clock_ms: Callable[[], int] | None = None,
        workflow_supervisor: Any | None = None,
    ) -> None:
        self.analysis_factory = analysis_factory
        self.optional_analysis = optional_analysis
        if (
            self.optional_analysis is None
            and analysis_factory is not None
        ):
            self.optional_analysis = analysis_factory.optional_analysis
        self.admission_request_factory = admission_request_factory
        if (
            self.admission_request_factory is None
            and analysis_factory is not None
        ):
            self.admission_request_factory = (
                analysis_factory.admission_request_factory
            )
        self.admission_service = admission_service or PlanAdmissionService()
        self.query_planner = query_planner or PlanAnalysisQueryPlanner()
        self.obligation_compiler = obligation_compiler or ObligationGraphCompiler()
        self.candidate_planner = candidate_planner or SymbolicCandidatePlanner()
        self.critic = critic or PlanCritic()
        self.parallel_compiler = parallel_compiler or ParallelPlanCompiler()
        self.scanner = scanner
        self.root_observer = root_observer
        self.receipt_store = receipt_store
        self._clock_ms = clock_ms or (lambda: 0)
        self.workflow_supervisor = workflow_supervisor
        self._preview_by_key: dict[str, PlanCreatePreviewReceipt] = {}
        self._lock = threading.RLock()

    @property
    def production_analysis_wired(self) -> bool:
        return self.analysis_factory is not None or self.optional_analysis is not None

    @property
    def production_admission_wired(self) -> bool:
        return (
            self.admission_service is not None
            and getattr(
                self.admission_service, "INTERFACE", PLAN_ADMISSION_SERVICE_INTERFACE
            )
            == PLAN_ADMISSION_SERVICE_INTERFACE
        )

    def wire_analysis_factory(self, factory: PlanningAnalysisFactory) -> "PlanCreateService":
        """Attach production analysis + admission request factory wiring."""

        if not isinstance(factory, PlanningAnalysisFactory):
            raise PlanCreateServiceError(
                "factory must be PlanningAnalysisFactory"
            )
        self.analysis_factory = factory
        self.optional_analysis = factory.optional_analysis
        self.admission_request_factory = factory.admission_request_factory
        if self.scanner is not None and hasattr(
            factory, "wire_prompt_directory_scanner"
        ):
            factory.wire_prompt_directory_scanner(self.scanner)
        if self.workflow_supervisor is not None and hasattr(
            factory, "wire_prompt_supervisor"
        ):
            factory.wire_prompt_supervisor(self.workflow_supervisor)
        return self

    def _cache_key(
        self,
        snapshot: PlanCreateInputSnapshot,
        mode: PlanCreateMode,
        *,
        compatibility_alias: str = "",
    ) -> str:
        return _digest(
            "plan-create-cache-key",
            {
                "snapshot_cid": snapshot.snapshot_cid,
                "mode": mode.value,
                "compatibility_alias": compatibility_alias,
            },
        )

    def _persist(self, receipt: PlanCreatePreviewReceipt) -> None:
        if self.receipt_store is None:
            return
        record = receipt.to_dict()
        if isinstance(self.receipt_store, MutableMapping):
            existing = self.receipt_store.get(receipt.receipt_cid)
            if existing is not None and dict(existing) != record:
                raise PlanCreateServiceError(
                    "receipt store contains a conflicting create-plan preview"
                )
            self.receipt_store[receipt.receipt_cid] = record
            return
        put = getattr(self.receipt_store, "put", None) or getattr(
            self.receipt_store, "store", None
        )
        if not callable(put):
            raise PlanCreateServiceError(
                "receipt_store must be a mutable mapping or implement put"
            )
        put(receipt.receipt_cid, record)

    def _require_current_roots(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
    ) -> None:
        expected = request.roots
        observed = materials.current_roots
        if observed is None and self.root_observer is not None:
            observed = self.root_observer(request)
        if observed is None:
            return
        roots = _coerce_roots(observed)
        if roots is None:
            raise PlanCreateStaleRootError("current roots observation is empty")
        try:
            roots.require_current(expected)
        except PlanRevisionStaleRootError as exc:
            raise PlanCreateStaleRootError(
                "stale root/policy observed; refusing silent regeneration"
            ) from exc
        # Also bind common policy root aliases if the observer returned a map
        # with partial fields (require_current already enforces full identity).
        if roots.policy_root != expected.policy_root:
            raise PlanCreateStaleRootError(
                "policy root is stale relative to the create request"
            )

    def _stage_scan(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
    ) -> tuple[PlanCreateStageResult, Any]:
        if materials.scan is not None:
            scan = materials.scan
            scan_cid = _record_cid(scan, fallback_namespace="scan")
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.SCAN,
                    artifact_cid=scan_cid,
                    passed=bool(scan_cid),
                    message="scan material bound from caller materials",
                ),
                scan,
            )
        if self.scanner is not None:
            scan_fn = getattr(self.scanner, "scan", None)
            if callable(scan_fn):
                scan = scan_fn(request)
            elif callable(self.scanner):
                scan = self.scanner(request)
            else:
                raise PlanCreateServiceError(
                    "scanner must be callable or expose scan()"
                )
            scan_cid = _record_cid(scan, fallback_namespace="scan")
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.SCAN,
                    artifact_cid=scan_cid,
                    passed=bool(scan_cid),
                    message="live scanner observed current scope",
                ),
                scan,
            )
        if self.analysis_factory is not None:
            view = self.analysis_factory.analyze(request.repository_root)
            scan = {
                "scan_cid": _digest(
                    "analysis-scan",
                    {
                        "view_id": getattr(view, "view_id", ""),
                        "factory_interface": getattr(
                            view,
                            "factory_interface",
                            PLANNING_ANALYSIS_FACTORY_INTERFACE,
                        ),
                        "tree_id": getattr(view, "tree_id", request.roots.dirty_worktree_root),
                        "scope_paths": list(request.scope_paths),
                    },
                ),
                "tree_id": getattr(
                    view, "tree_id", request.roots.dirty_worktree_root
                ),
                "repository_root": request.repository_root,
                "scope_paths": list(request.scope_paths),
                "request_cid": request.request_cid,
            }
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.SCAN,
                    artifact_cid=scan["scan_cid"],
                    passed=True,
                    message="production analysis factory scanned current scope",
                ),
                scan,
            )
        # Deterministic scope binding without a live checkout (unit path).
        scan = {
            "scan_cid": _digest(
                "scope-scan",
                {
                    "request_cid": request.request_cid,
                    "repository_root": request.repository_root,
                    "scope_paths": list(request.scope_paths),
                    "dirty_worktree_root": request.roots.dirty_worktree_root,
                    "repository_root_cid": request.roots.repository_root_cid,
                },
            ),
            "tree_id": request.roots.dirty_worktree_root,
            "repository_root": request.repository_root,
            "scope_paths": list(request.scope_paths),
            "request_cid": request.request_cid,
            "synthetic": True,
        }
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.SCAN,
                artifact_cid=scan["scan_cid"],
                passed=True,
                blockers=("synthetic_scope_scan",),
                message="bound synthetic scope scan (no live scanner configured)",
            ),
            scan,
        )

    def _stage_query(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
        scan: Any,
    ) -> tuple[PlanCreateStageResult, ReasoningQueryPlan]:
        if materials.query_plan is not None:
            plan = materials.query_plan
            if not isinstance(plan, ReasoningQueryPlan):
                # Accept only already-typed plans from the live planner path;
                # mappings recompile so authority stays with the planner.
                plan = self.query_planner.compile(
                    request,
                    context={
                        "scan_cid": _record_cid(scan, fallback_namespace="scan"),
                        **_plain_mapping(materials.query_plan),
                    },
                )
            plan_cid = _record_cid(plan, fallback_namespace="query")
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.QUERY,
                    artifact_cid=plan_cid,
                    passed=bool(getattr(plan, "ready", True)),
                    blockers=tuple(getattr(plan, "blockers", ()) or ()),
                    message="query plan bound",
                ),
                plan,
            )
        context = {
            "scan_cid": _record_cid(scan, fallback_namespace="scan"),
            "tree_id": request.roots.dirty_worktree_root,
            "scope_paths": list(request.scope_paths),
        }
        plan = self.query_planner.compile(request, context=context)
        plan_cid = _record_cid(plan, fallback_namespace="query")
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.QUERY,
                artifact_cid=plan_cid,
                passed=bool(getattr(plan, "ready", False)),
                blockers=tuple(getattr(plan, "blockers", ()) or ()),
                detail_ids=tuple(
                    getattr(query, "query_id", "")
                    for query in getattr(plan, "queries", ())
                    if getattr(query, "query_id", "")
                ),
                message="compiled deterministic reasoning query plan",
            ),
            plan,
        )

    def _stage_evidence(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
        query_plan: ReasoningQueryPlan,
    ) -> tuple[PlanCreateStageResult, Any]:
        if materials.evidence_bundle is not None:
            bundle = materials.evidence_bundle
            bundle_cid = _record_cid(bundle, fallback_namespace="evidence")
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.EVIDENCE,
                    artifact_cid=bundle_cid,
                    passed=True,
                    message="evidence bundle bound from materials",
                ),
                bundle,
            )
        compiler = PlanningEvidenceBundleCompiler(
            current_root_id=request.roots.dirty_worktree_root,
            adapters=materials.evidence_adapters,
        )
        supplied = list(materials.evidence_queries)
        if supplied:
            primary = supplied[0]
            if isinstance(primary, EvidenceQuery):
                evidence_query = primary
            else:
                evidence_query = EvidenceQuery.from_value(primary)
        else:
            question_bits: list[str] = []
            paths = list(request.scope_paths)
            symbols: list[str] = []
            for query in getattr(query_plan, "queries", ())[:8]:
                text = str(getattr(query, "question", "") or "")
                if text:
                    question_bits.append(text)
                scope = getattr(query, "scope", None)
                for path in getattr(scope, "paths", ()) or ():
                    paths.append(str(path))
                for symbol in getattr(scope, "symbols", ()) or ():
                    symbols.append(str(symbol))
            metadata = request.redacted_source_metadata
            for path in metadata.get("changed_paths") or ():
                paths.append(str(path))
            for symbol in metadata.get("symbols") or ():
                symbols.append(str(symbol))
            evidence_query = EvidenceQuery(
                text=(
                    " ".join(question_bits)[:4000]
                    or f"evidence coverage for {request.board_namespace}"
                ),
                paths=tuple(dict.fromkeys(paths)),
                symbols=tuple(dict.fromkeys(symbols)),
                goal_ids=(f"goal:{request.board_namespace}",),
            )
        bundle = compiler.compile(evidence_query, schedule_missing=True)
        bundle_cid = _record_cid(bundle, fallback_namespace="evidence")
        ready = True
        blockers: list[str] = []
        coverage = getattr(bundle, "coverage", None)
        if coverage is not None:
            decision = getattr(coverage, "decision", None)
            decision_value = getattr(decision, "value", decision)
            if str(decision_value).casefold() not in {
                "",
                "ready",
                "partial",
                "scheduled",
            }:
                # Unavailable adapters schedule work; treat only hard rejections
                # as stage failure. Partial/scheduled remains explicit debt.
                if str(decision_value).casefold() in {"rejected", "blocked", "failed"}:
                    ready = False
                    blockers.append(f"evidence_coverage:{decision_value}")
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.EVIDENCE,
                artifact_cid=bundle_cid,
                passed=ready,
                blockers=tuple(blockers),
                message="compiled evidence coverage for current tree",
            ),
            bundle,
        )

    def _default_intent(
        self, request: PlanCreateRequest
    ) -> tuple[TypedIntent, tuple[ProducerRule, ...], tuple[TaskCandidate, ...], tuple[TypedPredicate, ...]]:
        concepts = request.redacted_source_metadata.get("concepts") or ("plan_create",)
        if isinstance(concepts, str):
            concepts = (concepts,)
        primary = str(next(iter(concepts), "plan_create"))
        predicate = TypedPredicate(
            predicate_id=f"goal:{primary}",
            predicate_type="behavior_state",
            subject_ref=primary,
            provenance_refs=(f"create:{request.request_cid}",),
            proof_requirement_refs=(f"proof:{primary}",),
            validation_requirement_refs=(f"validation:{primary}",),
        )
        producer = ProducerRule(
            producer_id=f"producer:{primary}",
            effect_predicate_ids=(predicate.predicate_id,),
            provenance_refs=tuple(request.scope_paths) or (".",),
            proof_requirement_refs=(f"proof:{primary}",),
        )
        from ..planning.obligation_graph_compiler import obligation_id_for_producer

        task = TaskCandidate(
            candidate_id=f"task:{primary}",
            closes_obligation_ids=(
                obligation_id_for_producer(
                    producer.producer_id, predicate.predicate_id
                ),
            ),
            producer_id=producer.producer_id,
            provenance_refs=tuple(request.scope_paths) or (".",),
        )
        intent = TypedIntent(
            intent_id=f"intent:{request.request_cid}",
            desired_predicates=(predicate,),
            source_refs=(request.prompt_source_cid,),
            current_root_id=request.roots.dirty_worktree_root,
        )
        return intent, (producer,), (task,), ()

    def _stage_obligation(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
        evidence: Any,
    ) -> tuple[PlanCreateStageResult, ObligationGraph]:
        if materials.obligation_graph is not None:
            graph = materials.obligation_graph
            if not isinstance(graph, ObligationGraph):
                graph = ObligationGraph.from_dict(graph)
            graph_cid = _record_cid(graph, fallback_namespace="obligation")
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.OBLIGATION,
                    artifact_cid=graph_cid,
                    passed=not bool(getattr(graph, "planning_blocked", False)),
                    message="obligation graph bound from materials",
                ),
                graph,
            )
        intent = materials.intent
        producers = tuple(materials.producers)
        task_candidates = tuple(materials.task_candidates)
        predicates = tuple(materials.predicates)
        facts = tuple(materials.current_facts)
        if intent is None:
            intent, default_producers, default_tasks, predicates = (
                self._default_intent(request)
            )
            if not producers:
                producers = default_producers
            if not task_candidates:
                task_candidates = default_tasks
        # Only bind an evidence bundle that is already ready. Scheduled or
        # partial coverage remains explicit evidence debt on the evidence stage
        # and must not silently block obligation/candidate generation.
        evidence_for_obligation = None
        coverage = getattr(evidence, "coverage", None)
        decision = getattr(coverage, "decision", None)
        decision_value = str(getattr(decision, "value", decision) or "").casefold()
        if evidence is not None and decision_value in {"ready", ""}:
            if coverage is None or decision_value == "ready":
                evidence_for_obligation = evidence
        graph = self.obligation_compiler.compile(
            intent,
            current_facts=facts,
            producers=producers,
            task_candidates=task_candidates,
            predicates=predicates,
            evidence_bundle=evidence_for_obligation,
            current_root_id=request.roots.dirty_worktree_root,
        )
        graph_cid = _record_cid(graph, fallback_namespace="obligation")
        blockers: list[str] = []
        if getattr(graph, "planning_blocked", False):
            blockers.append("obligation_planning_blocked")
        if getattr(graph, "review_required", False):
            blockers.append("obligation_review_required")
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.OBLIGATION,
                artifact_cid=graph_cid,
                passed=not blockers,
                blockers=tuple(blockers),
                message="compiled AND/OR obligation graph",
            ),
            graph,
        )

    def _stage_candidate(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
        obligation: ObligationGraph,
        *,
        mode: PlanCreateMode,
        snapshot: PlanCreateInputSnapshot,
    ) -> tuple[PlanCreateStageResult, SymbolicCandidatePortfolio | Mapping[str, Any]]:
        allow_model = mode is PlanCreateMode.MODEL_ASSISTED
        model_provider = materials.model_provider if allow_model else None
        frozen = materials.frozen_goal
        if frozen is None:
            policy = EvidenceAwarePlanPolicy(
                acceptance_criteria=(f"accept:{request.board_namespace}",),
                evidence_terms=tuple(request.scope_paths) or ("repository",),
                allowed_scopes=tuple(request.scope_paths) or (".",),
                available_resource_classes=("cpu-medium",),
                require_validation=True,
                require_proof=True,
            )
            frozen = FrozenPlanningGoal(
                goal_id=f"goal:{request.board_namespace}",
                goal_content_id=plan_revision_cid(
                    {
                        "request_cid": request.request_cid,
                        "snapshot_cid": snapshot.snapshot_cid,
                    }
                ),
                repository_tree_id=request.roots.dirty_worktree_root,
                policy=policy,
            )
        elif not isinstance(frozen, FrozenPlanningGoal):
            frozen = FrozenPlanningGoal.from_dict(frozen)
        context = {
            "request_cid": request.request_cid,
            "input_snapshot_cid": snapshot.snapshot_cid,
            "mode": mode.value,
            "bounds_digest": snapshot.bounds_digest,
            "scope_paths": list(request.scope_paths),
            **dict(materials.candidate_context or {}),
        }
        # Exact shared bounds: candidate_count is derived only from the frozen
        # request budget surface (mode must not widen bounds).
        max_candidates = max(1, min(int(request.budget.max_goals), 8))
        planner = self.candidate_planner
        if getattr(planner, "bounds", None) is not None:
            try:
                planner = SymbolicCandidatePlanner(
                    bounds=SymbolicCandidateBounds(
                        candidate_count=max_candidates,
                        max_model_candidates=(
                            max(0, min(int(request.budget.max_model_calls), max_candidates - 1))
                            if allow_model
                            else 0
                        ),
                    )
                )
            except TypeError:
                planner = self.candidate_planner
        shared_detail_ids = (snapshot.bounds_digest, snapshot.snapshot_cid)
        try:
            portfolio = planner.plan(
                obligation,
                frozen,
                context,
                model_provider=model_provider,
                allow_model=allow_model,
            )
        except Exception as exc:  # noqa: BLE001 - stage fail-closed
            portfolio = {
                "portfolio_id": _digest(
                    "candidate-error",
                    {
                        "request_cid": request.request_cid,
                        "error": type(exc).__name__,
                        "input_snapshot_cid": snapshot.snapshot_cid,
                        "bounds_digest": snapshot.bounds_digest,
                        "mode": mode.value,
                    },
                ),
                "error": type(exc).__name__,
                "message": str(exc)[:512],
                "input_snapshot_cid": snapshot.snapshot_cid,
                "mode": mode.value,
                "bounds_digest": snapshot.bounds_digest,
            }
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.CANDIDATE,
                    artifact_cid=portfolio["portfolio_id"],
                    passed=False,
                    blockers=(f"candidate_failed:{type(exc).__name__}",),
                    detail_ids=shared_detail_ids,
                    message="candidate portfolio generation failed closed",
                ),
                portfolio,
            )
        portfolio_cid = _record_cid(portfolio, fallback_namespace="candidate")
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.CANDIDATE,
                artifact_cid=portfolio_cid,
                passed=True,
                detail_ids=shared_detail_ids,
                message=(
                    "generated model-assisted candidate portfolio"
                    if allow_model
                    else "generated deterministic candidate portfolio"
                ),
            ),
            portfolio,
        )

    def _candidate_plan_projection(
        self,
        request: PlanCreateRequest,
        portfolio: Any,
        obligation: ObligationGraph,
    ) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        task_ids: list[str] = []
        selected = getattr(portfolio, "selected", None)
        if selected is None and not isinstance(portfolio, Mapping):
            selected = getattr(portfolio, "baseline", None)
        records: list[Any] = []
        if selected is not None:
            records = [selected]
        elif isinstance(portfolio, Mapping):
            records = list(
                portfolio.get("candidates")
                or portfolio.get("snapshots")
                or ()
            )
        else:
            records = list(getattr(portfolio, "snapshots", ()) or ())

        for index, record in enumerate(records):
            symbolic = getattr(record, "symbolic_candidate", None)
            if symbolic is not None:
                ids = getattr(symbolic, "task_candidate_ids", ()) or ()
                task_ids.extend(str(item) for item in ids)
                continue
            data = (
                _plain_mapping(record)
                if not isinstance(record, Mapping)
                else dict(record)
            )
            nested = data.get("symbolic_candidate") or data
            if isinstance(nested, Mapping):
                ids = (
                    nested.get("task_candidate_ids")
                    or nested.get("task_ids")
                    or nested.get("tasks")
                    or ()
                )
            else:
                ids = data.get("task_ids") or data.get("tasks") or ()
            if not ids and data.get("candidate_id"):
                ids = (data["candidate_id"],)
            if isinstance(ids, (str, bytes)):
                ids = (ids,)
            task_ids.extend(str(item) for item in ids or (f"task:{index}",))

        # Prefer obligation graph task candidates when portfolio has no tasks.
        if not task_ids:
            task_ids = [
                str(item.candidate_id)
                for item in getattr(obligation, "task_candidates", ())
                if getattr(item, "candidate_id", "")
            ]
        if not task_ids:
            task_ids = [
                str(getattr(node, "obligation_id", "") or "")
                for node in getattr(obligation, "nodes", ())[:8]
                if getattr(node, "obligation_id", "")
            ]
        if not task_ids:
            task_ids = [f"task:create:{request.alias_prefix}"]

        depends_lookup: dict[str, list[str]] = {}
        for candidate in getattr(obligation, "task_candidates", ()):
            depends_lookup[str(candidate.candidate_id)] = [
                str(item)
                for item in getattr(candidate, "depends_on_candidate_ids", ())
            ]

        for task_id in dict.fromkeys(task_ids):
            tasks.append(
                {
                    "task_id": task_id,
                    "action_id": task_id,
                    "depends_on": list(depends_lookup.get(task_id, [])),
                    "outputs": list(request.scope_paths[:1] or (".",)),
                    "paths": list(request.scope_paths or (".",)),
                    "resource_class": "cpu-medium",
                    "estimated_duration_ms": 1_000,
                }
            )
        plan_id = _digest(
            "candidate-plan",
            {
                "request_cid": request.request_cid,
                "task_ids": [task["task_id"] for task in tasks],
            },
        )
        return {
            "plan_id": plan_id,
            "tasks": tasks,
            "actions": [
                {
                    "action_id": task["action_id"],
                    "depends_on": list(task["depends_on"]),
                    "effects": [
                        {
                            "effect_id": f"effect:{task['task_id']}",
                            "action_id": task["action_id"],
                            "operation": "update",
                            "target": (task["outputs"] or ["."])[0],
                        }
                    ],
                }
                for task in tasks
            ],
            "effects": [
                {
                    "effect_id": f"effect:{task['task_id']}",
                    "action_id": task["action_id"],
                    "operation": "update",
                    "target": (task["outputs"] or ["."])[0],
                }
                for task in tasks
            ],
        }

    def _stage_critique(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
        portfolio: Any,
        obligation: ObligationGraph,
        evidence: Any,
        candidate_plan: Mapping[str, Any],
    ) -> tuple[PlanCreateStageResult, PlanCritique | Mapping[str, Any]]:
        try:
            critique = self.critic.critique(
                candidate_plan,
                obligation_graph=obligation,
                evidence=evidence,
                required_goal_ids=tuple(
                    str(getattr(node, "obligation_id", ""))
                    for node in getattr(obligation, "nodes", ())[:16]
                    if getattr(node, "obligation_id", "")
                ),
            )
        except Exception as exc:  # noqa: BLE001 - stage fail-closed
            critique = {
                "critique_id": _digest(
                    "critique-error",
                    {
                        "request_cid": request.request_cid,
                        "error": type(exc).__name__,
                    },
                ),
                "error": type(exc).__name__,
                "message": str(exc)[:512],
            }
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.CRITIQUE,
                    artifact_cid=critique["critique_id"],
                    passed=False,
                    blockers=(f"critique_failed:{type(exc).__name__}",),
                    message="critique failed closed",
                ),
                critique,
            )
        critique_cid = _record_cid(critique, fallback_namespace="critique")
        decision = getattr(critique, "decision", None)
        decision_value = str(getattr(decision, "value", decision) or "")
        blockers: list[str] = []
        passed = True
        if decision_value in {"reject", "rejected", "block", "blocked"}:
            passed = False
            blockers.append(f"critique:{decision_value}")
        findings = getattr(critique, "findings", ()) or ()
        hard = [
            str(getattr(item, "finding_id", getattr(item, "kind", "finding")))
            for item in findings
            if str(getattr(getattr(item, "severity", None), "value", getattr(item, "severity", ""))).casefold()
            in {"error", "fatal", "hard"}
        ]
        if hard:
            passed = False
            blockers.extend(f"critique_finding:{item}" for item in hard[:8])
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.CRITIQUE,
                artifact_cid=critique_cid,
                passed=passed,
                blockers=tuple(blockers),
                detail_ids=tuple(hard[:16]),
                message="deterministic critique completed",
            ),
            critique,
        )

    def _stage_admission(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
        candidate_plan: Mapping[str, Any],
        critique: Any,
        execution_plan: Any,
    ) -> tuple[PlanCreateStageResult, Any]:
        if materials.admission_materials is not None:
            try:
                receipt = self.admission_service.admit(materials.admission_materials)
            except Exception as exc:  # noqa: BLE001
                receipt = {
                    "receipt_id": _digest(
                        "admission-error",
                        {
                            "request_cid": request.request_cid,
                            "error": type(exc).__name__,
                        },
                    ),
                    "admitted": False,
                    "error": type(exc).__name__,
                    "message": str(exc)[:512],
                }
                return (
                    PlanCreateStageResult(
                        stage=PlanCreateStage.ADMISSION,
                        artifact_cid=receipt["receipt_id"],
                        passed=False,
                        blockers=(f"admission_failed:{type(exc).__name__}",),
                        message="admission materials rejected fail-closed",
                    ),
                    receipt,
                )
            receipt_cid = _record_cid(receipt, fallback_namespace="admission")
            admitted = bool(getattr(receipt, "admitted", False))
            return (
                PlanCreateStageResult(
                    stage=PlanCreateStage.ADMISSION,
                    artifact_cid=receipt_cid,
                    passed=admitted,
                    blockers=() if admitted else ("admission_rejected",),
                    message=(
                        "plan admission service admitted the candidate"
                        if admitted
                        else "plan admission service rejected the candidate"
                    ),
                ),
                receipt,
            )
        # Without full IR materials the service still runs an independent
        # structural admission gate and never trusts provider claims.
        critique_passed = True
        if isinstance(critique, Mapping):
            critique_passed = "error" not in critique
        else:
            decision = getattr(critique, "decision", None)
            decision_value = str(getattr(decision, "value", decision) or "")
            if decision_value in {"reject", "rejected", "block", "blocked"}:
                critique_passed = False
        execution_admitted = True
        if execution_plan is not None:
            execution_admitted = bool(
                getattr(execution_plan, "admitted", True)
            )
        structural_ok = bool(candidate_plan.get("tasks")) and critique_passed
        admitted = structural_ok and execution_admitted
        if self.admission_request_factory is not None:
            factory_id = _digest(
                "admission-request-factory",
                {
                    "wired": True,
                    "factory": type(self.admission_request_factory).__name__,
                },
            )
        else:
            factory_id = ""
        receipt = {
            "receipt_id": _digest(
                "structural-admission",
                {
                    "request_cid": request.request_cid,
                    "plan_id": candidate_plan.get("plan_id", ""),
                    "admitted": admitted,
                    "factory_id": factory_id,
                    "execution_plan_cid": _record_cid(
                        execution_plan, fallback_namespace="parallel"
                    ),
                },
            ),
            "admitted": admitted,
            "kind": "structural",
            "plan_id": candidate_plan.get("plan_id", ""),
            "admission_request_factory_wired": (
                self.admission_request_factory is not None
            ),
            "production_admission_service": self.production_admission_wired,
        }
        blockers: list[str] = []
        if not structural_ok:
            blockers.append("structural_admission_failed")
        if not execution_admitted:
            blockers.append("execution_plan_not_admitted")
        if materials.admission_materials is None:
            blockers.append("ir_admission_materials_absent")
            # Structural path is review-only when IR materials are absent.
            admitted = False
            receipt = {
                **receipt,
                "admitted": False,
                "verdict": "review_only",
            }
            receipt["receipt_id"] = _digest(
                "structural-admission",
                {
                    "request_cid": request.request_cid,
                    "plan_id": candidate_plan.get("plan_id", ""),
                    "admitted": False,
                    "factory_id": factory_id,
                    "execution_plan_cid": _record_cid(
                        execution_plan, fallback_namespace="parallel"
                    ),
                    "verdict": "review_only",
                },
            )
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.ADMISSION,
                artifact_cid=receipt["receipt_id"],
                passed=admitted,
                blockers=tuple(blockers),
                message=(
                    "independent structural admission (IR materials absent)"
                    if blockers
                    else "admission completed"
                ),
            ),
            receipt,
        )

    def _stage_parallel_plan(
        self,
        request: PlanCreateRequest,
        materials: PlanCreateMaterials,
        candidate_plan: Mapping[str, Any],
    ) -> tuple[PlanCreateStageResult, Any]:
        if materials.parallel_request is not None:
            if isinstance(materials.parallel_request, ParallelPlanCompilationRequest):
                compilation = materials.parallel_request
            else:
                compilation = ParallelPlanCompilationRequest(
                    **dict(materials.parallel_request)  # type: ignore[arg-type]
                )
            plan = self.parallel_compiler.compile(compilation)
        else:
            tasks = materials.parallel_tasks
            if tasks is None:
                tasks = candidate_plan.get("tasks") or ()
            plan = self.parallel_compiler.compile(
                tasks=tasks,
                requested_width=max(1, int(request.budget.max_ready_width)),
                repository_snapshot={
                    "tree_id": request.roots.dirty_worktree_root,
                    "repository_tree_id": request.roots.dirty_worktree_root,
                },
                capacity_snapshot={
                    "snapshot_id": plan_revision_cid(
                        {"capacity": request.roots.configuration_root or "default"}
                    ),
                    "cpu_slots": max(1, int(request.budget.max_ready_width)),
                },
                budget={
                    "max_ready_width": int(request.budget.max_ready_width),
                    "max_tasks": int(request.budget.max_tasks),
                },
            )
        plan_cid = _record_cid(plan, fallback_namespace="parallel")
        admitted = bool(getattr(plan, "admitted", False))
        blockers: list[str] = []
        if not admitted:
            blockers.append("parallel_plan_rejected")
            for issue in getattr(plan, "issues", ())[:8]:
                code = getattr(issue, "code", None)
                blockers.append(
                    f"parallel:{getattr(code, 'value', code) or getattr(issue, 'message', 'issue')}"
                )
        return (
            PlanCreateStageResult(
                stage=PlanCreateStage.PARALLEL_PLAN,
                artifact_cid=plan_cid,
                passed=admitted,
                blockers=tuple(blockers),
                message="compiled parallel execution plan",
            ),
            plan,
        )

    def _verdict_from_stages(
        self,
        stage_results: Sequence[PlanCreateStageResult],
        admission: Any,
        execution_plan: Any,
    ) -> tuple[PlanCreateVerdict, tuple[str, ...]]:
        reasons: list[str] = []
        for result in stage_results:
            if not result.passed:
                reasons.extend(result.blockers or (f"{result.stage.value}_failed",))
        if any(
            result.stage is PlanCreateStage.ADMISSION
            and "ir_admission_materials_absent" in result.blockers
            for result in stage_results
        ):
            return PlanCreateVerdict.REVIEW_ONLY, tuple(dict.fromkeys(reasons))
        admitted = bool(getattr(admission, "admitted", False))
        if isinstance(admission, Mapping):
            admitted = bool(admission.get("admitted", False))
        if admitted and not reasons:
            return PlanCreateVerdict.ADMITTED, ()
        if any(
            result.stage
            in {
                PlanCreateStage.QUERY,
                PlanCreateStage.OBLIGATION,
                PlanCreateStage.CANDIDATE,
            }
            and not result.passed
            for result in stage_results
        ):
            return PlanCreateVerdict.BLOCKED, tuple(dict.fromkeys(reasons))
        if reasons:
            return PlanCreateVerdict.REJECTED, tuple(dict.fromkeys(reasons))
        return PlanCreateVerdict.REJECTED, ("admission_not_granted",)

    def preview_create(
        self,
        request: PlanCreateRequest | Mapping[str, Any],
        *,
        mode: PlanCreateMode | str = PlanCreateMode.DETERMINISTIC,
        materials: PlanCreateMaterials | Mapping[str, Any] | None = None,
        compatibility_alias: str = "",
    ) -> PlanCreatePreviewReceipt:
        """Scan, plan, critique, admit, and compile without writes."""

        typed_request = _coerce_request(request)
        typed_mode = _coerce_mode(mode)
        if materials is None:
            typed_materials = PlanCreateMaterials()
        elif isinstance(materials, PlanCreateMaterials):
            typed_materials = materials
        elif isinstance(materials, Mapping):
            typed_materials = PlanCreateMaterials(**dict(materials))
        else:
            raise PlanCreateServiceError(
                "materials must be PlanCreateMaterials or a mapping"
            )

        # Freeze inputs/bounds before any mode-specific work so deterministic
        # and model-assisted paths share exact bindings.
        snapshot = freeze_plan_create_input_snapshot(typed_request)
        cache_key = self._cache_key(
            snapshot, typed_mode, compatibility_alias=compatibility_alias
        )
        with self._lock:
            cached = self._preview_by_key.get(cache_key)
            if cached is not None:
                # Restart/idempotent path: re-validate roots, never silently
                # regenerate against a different observation.
                self._require_current_roots(typed_request, typed_materials)
                return cached

            self._require_current_roots(typed_request, typed_materials)

            stage_results: list[PlanCreateStageResult] = []
            scan_result, scan = self._stage_scan(typed_request, typed_materials)
            stage_results.append(scan_result)

            query_result, query_plan = self._stage_query(
                typed_request, typed_materials, scan
            )
            stage_results.append(query_result)

            evidence_result, evidence = self._stage_evidence(
                typed_request, typed_materials, query_plan
            )
            stage_results.append(evidence_result)

            obligation_result, obligation = self._stage_obligation(
                typed_request, typed_materials, evidence
            )
            stage_results.append(obligation_result)

            candidate_result, portfolio = self._stage_candidate(
                typed_request,
                typed_materials,
                obligation,
                mode=typed_mode,
                snapshot=snapshot,
            )
            stage_results.append(candidate_result)

            candidate_plan = self._candidate_plan_projection(
                typed_request, portfolio, obligation
            )

            critique_result, critique = self._stage_critique(
                typed_request,
                typed_materials,
                portfolio,
                obligation,
                evidence,
                candidate_plan,
            )
            stage_results.append(critique_result)

            # Parallel plan before final admission join so admission can bind
            # the execution-plan identity (matches planner workflow order in
            # the architecture plan: admit then parallel, with independent
            # compile; we compile parallel then admit with the plan bound).
            parallel_result, execution_plan = self._stage_parallel_plan(
                typed_request, typed_materials, candidate_plan
            )
            # Keep stage order authority as declared even though parallel is
            # compiled before the admission join below.
            admission_result, admission = self._stage_admission(
                typed_request,
                typed_materials,
                candidate_plan,
                critique,
                execution_plan,
            )
            stage_results.append(admission_result)
            stage_results.append(parallel_result)

            # Enforce declared stage order in the receipt.
            ordered = {
                result.stage: result for result in stage_results
            }
            stage_results = [ordered[stage] for stage in CREATE_STAGE_ORDER if stage in ordered]

            verdict, rejection_reasons = self._verdict_from_stages(
                stage_results, admission, execution_plan
            )
            artifact_refs = tuple(
                sorted(
                    {
                        snapshot.snapshot_cid,
                        *(
                            result.artifact_cid
                            for result in stage_results
                            if result.artifact_cid
                        ),
                        *(
                            result.result_cid
                            for result in stage_results
                            if result.result_cid
                        ),
                    }
                )
            )
            plan_root = (
                str(candidate_plan.get("plan_id") or "")
                if verdict is PlanCreateVerdict.ADMITTED
                else str(candidate_plan.get("plan_id") or "")
            )
            receipt = PlanCreatePreviewReceipt(
                request_cid=typed_request.request_cid,
                input_snapshot_cid=snapshot.snapshot_cid,
                mode=typed_mode,
                verdict=verdict,
                roots=typed_request.roots,
                stage_results=tuple(stage_results),
                scan_cid=ordered[PlanCreateStage.SCAN].artifact_cid
                if PlanCreateStage.SCAN in ordered
                else "",
                query_plan_cid=ordered[PlanCreateStage.QUERY].artifact_cid
                if PlanCreateStage.QUERY in ordered
                else "",
                evidence_bundle_cid=ordered[PlanCreateStage.EVIDENCE].artifact_cid
                if PlanCreateStage.EVIDENCE in ordered
                else "",
                obligation_graph_cid=ordered[PlanCreateStage.OBLIGATION].artifact_cid
                if PlanCreateStage.OBLIGATION in ordered
                else "",
                candidate_portfolio_cid=ordered[PlanCreateStage.CANDIDATE].artifact_cid
                if PlanCreateStage.CANDIDATE in ordered
                else "",
                critique_cid=ordered[PlanCreateStage.CRITIQUE].artifact_cid
                if PlanCreateStage.CRITIQUE in ordered
                else "",
                admission_receipt_cid=ordered[PlanCreateStage.ADMISSION].artifact_cid
                if PlanCreateStage.ADMISSION in ordered
                else "",
                execution_plan_cid=ordered[PlanCreateStage.PARALLEL_PLAN].artifact_cid
                if PlanCreateStage.PARALLEL_PLAN in ordered
                else "",
                plan_root_cid=plan_root,
                rejection_reasons=rejection_reasons,
                artifact_refs=artifact_refs,
                read_only=True,
                wrote_effects=(),
                compatibility_alias=compatibility_alias,
            )
            self._persist(receipt)
            self._preview_by_key[cache_key] = receipt
            # Touch clock for observability hooks (read-only).
            _ = self._clock_ms()
            return receipt

    def workflow_preview(
        self,
        request: Any,
        *,
        mode: PlanCreateMode | str | None = None,
        materials: PlanCreateMaterials | Mapping[str, Any] | None = None,
        **create_request_kwargs: Any,
    ) -> PlanCreatePreviewReceipt:
        """Canonical compatibility alias for create-plan preview.

        Existing workflow-preview callers share this create pipeline rather
        than a parallel implementation.  Shared control-plane/CLI/MCP facade
        wiring remains PDR-032.
        """

        if isinstance(request, PlanCreateRequest) or (
            isinstance(request, Mapping)
            and request.get("schema") == PlanCreateRequest.SCHEMA
        ):
            typed = _coerce_request(request)
            resolved_mode = mode
            if resolved_mode is None:
                resolved_mode = PlanCreateMode.DETERMINISTIC
            return self.preview_create(
                typed,
                mode=resolved_mode,
                materials=materials,
                compatibility_alias=WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS,
            )

        typed = plan_create_request_from_workflow(
            request, **create_request_kwargs
        )
        if mode is None:
            allow_model = bool(
                getattr(
                    getattr(request, "planning_policy", None),
                    "allow_model",
                    False,
                )
            )
            resolved_mode = (
                PlanCreateMode.MODEL_ASSISTED
                if allow_model
                else PlanCreateMode.DETERMINISTIC
            )
        else:
            resolved_mode = mode
        merged_materials: PlanCreateMaterials
        if materials is None:
            merged_materials = PlanCreateMaterials(workflow_request=request)
        elif isinstance(materials, PlanCreateMaterials):
            merged_materials = PlanCreateMaterials(
                scan=materials.scan,
                query_plan=materials.query_plan,
                evidence_bundle=materials.evidence_bundle,
                evidence_adapters=materials.evidence_adapters,
                evidence_queries=materials.evidence_queries,
                obligation_graph=materials.obligation_graph,
                intent=materials.intent,
                current_facts=materials.current_facts,
                producers=materials.producers,
                task_candidates=materials.task_candidates,
                predicates=materials.predicates,
                frozen_goal=materials.frozen_goal,
                candidate_context=materials.candidate_context,
                model_provider=materials.model_provider,
                parallel_tasks=materials.parallel_tasks,
                parallel_request=materials.parallel_request,
                admission_materials=materials.admission_materials,
                current_roots=materials.current_roots,
                workflow_request=request,
                extra=materials.extra,
            )
        else:
            payload = dict(materials)
            payload["workflow_request"] = request
            merged_materials = PlanCreateMaterials(**payload)
        return self.preview_create(
            typed,
            mode=resolved_mode,
            materials=merged_materials,
            compatibility_alias=WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS,
        )

def create_default_plan_create_service(
    *,
    repository_allowlist: Sequence[str | Path] = (),
    analysis_factory: PlanningAnalysisFactory | None = None,
    admission_service: PlanAdmissionService | None = None,
    index_root: str | Path | None = None,
    scanner: Any | None = None,
    workflow_supervisor: Any | None = None,
    receipt_store: MutableMapping[str, Mapping[str, Any]] | Any | None = None,
    root_observer: Callable[..., Mapping[str, Any]] | None = None,
    clock_ms: Callable[[], int] | None = None,
    build_analysis_factory: bool = True,
) -> PlanCreateService:
    """Construct a production-wired create-plan service.

    Wires:

    * ``PlanningAnalysisFactory@1`` as the production analysis composition root
      (optional_analysis + admission_request_factory); and
    * ``PlanAdmissionService@1`` for independent multi-gate admission.
    """

    factory = analysis_factory
    if factory is None and build_analysis_factory and repository_allowlist:
        factory = build_planning_analysis_factory(
            repository_allowlist=repository_allowlist,
            index_root=index_root,
        )
    service = PlanCreateService(
        analysis_factory=factory,
        admission_service=admission_service or PlanAdmissionService(),
        scanner=scanner,
        workflow_supervisor=workflow_supervisor,
        receipt_store=receipt_store,
        root_observer=root_observer,
        clock_ms=clock_ms,
    )
    if factory is not None:
        service.wire_analysis_factory(factory)
    return service


# Canonical alias spelling used by migration docs and discovery.
build_default_plan_create_service = create_default_plan_create_service

# Explicit method alias: existing ``preview`` name maps to create-plan preview.
PlanCreateService.preview = PlanCreateService.preview_create


__all__ = [
    "CREATE_STAGE_ORDER",
    "PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA",
    "PLAN_CREATE_PREVIEW_SCHEMA",
    "PLAN_CREATE_SERVICE_INTERFACE",
    "PLAN_CREATE_SERVICE_VERSION",
    "PLAN_CREATE_STAGE_RESULT_SCHEMA",
    "WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS",
    "PlanCreateBodyError",
    "PlanCreateInputSnapshot",
    "PlanCreateMaterials",
    "PlanCreateMode",
    "PlanCreatePreviewReceipt",
    "PlanCreateService",
    "PlanCreateServiceError",
    "PlanCreateStage",
    "PlanCreateStageResult",
    "PlanCreateStaleRootError",
    "PlanCreateVerdict",
    "build_default_plan_create_service",
    "create_default_plan_create_service",
    "freeze_plan_create_input_snapshot",
    "plan_create_request_from_workflow",
]
