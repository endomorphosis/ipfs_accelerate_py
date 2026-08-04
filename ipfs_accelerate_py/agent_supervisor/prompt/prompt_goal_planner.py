"""Bounded prompt-to-goal planning through the shared ``llm_router`` adapter.

The model-facing shape in this module is deliberately smaller than
``PromptGoalGraph``.  A provider proposes local keys and references only
already-admitted evidence handles.  This module validates the complete
proposal and then assigns the canonical goal/task CIDs owned by
``prompt_workflow``.  Provider text is never retained in a receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .prompt_workflow import (
    DirectoryScanReceipt,
    EvidenceAuthority,
    PromptAcceptanceRecord,
    PromptEvidenceRecord,
    PromptGoalGraph,
    PromptGoalRecord,
    PromptGraphError,
    PromptOutputRecord,
    PromptTaskRecord,
    PromptValidationRecord,
    PromptWorkflowBoundsError,
    PromptWorkflowContractError,
    PromptWorkflowRequest,
)


PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prompt-goal-provider-request@1"
)
PROMPT_GOAL_PROPOSAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prompt-goal-proposal@1"
)
PROMPT_GOAL_PLANNING_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prompt-goal-planning-receipt@1"
)
PROMPT_GOAL_CANDIDATE_PORTFOLIO_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prompt-goal-candidate-portfolio@1"
)
PROMPT_GOAL_PLANNER_VERSION = "1"

DEFAULT_MAX_PROVIDER_REQUEST_BYTES = 64 * 1024
DEFAULT_MAX_PROVIDER_RESPONSE_BYTES = 256 * 1024
DEFAULT_MAX_SELECTED_EVIDENCE = 64
DEFAULT_MAX_SUMMARY_BYTES = 1_024
DEFAULT_RESOURCE_CLASSES = (
    "cpu-small",
    "cpu-medium",
    "cpu-large",
    "provider-llm",
)
DEFAULT_VALIDATION_PREFIXES = (
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("pytest",),
)
_SAFE_FALLBACK_BEHAVIORS = frozenset(
    {"fail_closed", "deterministic_retry", "manual_review", "quarantine"}
)
_SAFE_OUTPUT_EFFECTS = frozenset({"create", "write", "modify"})
_SAFE_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/octet-stream",
        "text/markdown",
        "text/plain",
        "text/x-python",
        "text/yaml",
    }
)
_PROTECTED_DEFAULTS = frozenset(
    {
        "docs/architecture/agent_supervisor_self_improvement.todo.md",
        "docs/architecture/agent_supervisor_self_improvement.objectives.md",
    }
)
_SHELL_META_RE = re.compile(r"(?:[;&|`]|\$\(|\$\{|(?:^|\s)(?:>|<){1,2}(?:\s|$))")
_INSTRUCTION_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:the\s+)?(?:policy|authority|constraints?|instructions?)|"
    r"grant\s+(?:me|model|provider|task)\s+authority|"
    r"(?:mark|declare|claim)\s+(?:the\s+)?(?:task|goal|work)\s+(?:as\s+)?complete|"
    r"(?:sudo|/bin/(?:ba)?sh|sh\s+-c|bash\s+-c|rm\s+-rf|eval\s*\(|exec\s*\()|"
    r"```"
    r")",
    re.IGNORECASE,
)
_CONSTRAINT_KEYS = frozenset(
    {
        "allowed_paths",
        "protected_paths",
        "validation_commands",
        "policy_roots",
        "proof_handles",
        "constraint_summaries",
    }
)
_CAPABILITY_KEYS = frozenset(
    {
        "available",
        "operations",
        "output_effects",
        "resource_classes",
        "validation_prefixes",
        "max_parallelism",
        "capability_roots",
    }
)


class PromptGoalPlannerError(RuntimeError):
    """A provider request or proposal failed the goal-planning boundary."""

    def __init__(self, message: str, *, reason_code: str = "planner_error") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "planner_error")


class PromptGoalProposalError(PromptGoalPlannerError):
    """Strict provider JSON was malformed, unsafe, or graph-invalid."""


class PromptGoalProviderRequestError(PromptGoalPlannerError):
    """The bounded model request could not be compiled."""


@dataclass(frozen=True)
class PromptGoalPlannerConfig:
    """Runtime and policy bounds for one optional provider call."""

    repo_root: Path | None = None
    provider: str | None = None
    model: str = "gpt-5.3-codex-spark"
    timeout_seconds: int = 300
    max_new_tokens: int | None = None
    temperature: float = 0.0
    allow_local_fallback: bool = False
    max_provider_request_bytes: int = DEFAULT_MAX_PROVIDER_REQUEST_BYTES
    max_provider_response_bytes: int = DEFAULT_MAX_PROVIDER_RESPONSE_BYTES
    max_selected_evidence: int = DEFAULT_MAX_SELECTED_EVIDENCE
    max_summary_bytes: int = DEFAULT_MAX_SUMMARY_BYTES
    allowed_validation_prefixes: tuple[tuple[str, ...], ...] = (
        DEFAULT_VALIDATION_PREFIXES
    )
    allowed_resource_classes: tuple[str, ...] = DEFAULT_RESOURCE_CLASSES
    protected_paths: tuple[str, ...] = tuple(sorted(_PROTECTED_DEFAULTS))

    def __post_init__(self) -> None:
        for name in (
            "timeout_seconds",
            "max_provider_request_bytes",
            "max_provider_response_bytes",
            "max_selected_evidence",
            "max_summary_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_new_tokens is not None and (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not self.allowed_validation_prefixes or any(
            not prefix or any(not isinstance(item, str) or not item for item in prefix)
            for prefix in self.allowed_validation_prefixes
        ):
            raise ValueError("allowed_validation_prefixes must contain token prefixes")
        if not self.allowed_resource_classes:
            raise ValueError("allowed_resource_classes must not be empty")
        for path in self.protected_paths:
            _safe_relative_path(path, "protected_paths")


@dataclass(frozen=True)
class PromptGoalProviderReceipt:
    attempted: bool
    status: str
    reason_code: str
    provider_id: str
    model_id: str
    request_bytes: int
    request_sha256: str
    response_bytes: int = 0
    response_sha256: str = ""
    timeout_ms: int = 0
    max_new_tokens: int = 0
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "latency_ms": self.latency_ms,
            "max_new_tokens": self.max_new_tokens,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "reason_code": self.reason_code,
            "request_bytes": self.request_bytes,
            "request_sha256": self.request_sha256,
            "response_bytes": self.response_bytes,
            "response_sha256": self.response_sha256,
            "status": self.status,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True)
class PromptGoalParseReceipt:
    attempted: bool
    status: str
    reason_code: str
    proposal_schema: str
    response_bytes: int
    response_sha256: str
    goal_count: int = 0
    task_count: int = 0
    evidence_count: int = 0
    plan_root_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "evidence_count": self.evidence_count,
            "goal_count": self.goal_count,
            "plan_root_cid": self.plan_root_cid,
            "proposal_schema": self.proposal_schema,
            "reason_code": self.reason_code,
            "response_bytes": self.response_bytes,
            "response_sha256": self.response_sha256,
            "status": self.status,
            "task_count": self.task_count,
        }


@dataclass(frozen=True)
class PromptGoalFallbackReceipt:
    used: bool
    status: str
    reason_code: str
    planner_id: str = "prompt-goal-deterministic@1"
    plan_root_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_root_cid": self.plan_root_cid,
            "planner_id": self.planner_id,
            "reason_code": self.reason_code,
            "status": self.status,
            "used": self.used,
        }


@dataclass(frozen=True)
class PromptGoalPlanningReceipt:
    request_cid: str
    scan_cid: str
    plan_root_cid: str
    outcome: str
    provider: PromptGoalProviderReceipt
    parse: PromptGoalParseReceipt
    fallback: PromptGoalFallbackReceipt
    schema: str = PROMPT_GOAL_PLANNING_RECEIPT_SCHEMA
    receipt_version: str = PROMPT_GOAL_PLANNER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "fallback": self.fallback.to_dict(),
            "outcome": self.outcome,
            "parse": self.parse.to_dict(),
            "plan_root_cid": self.plan_root_cid,
            "provider": self.provider.to_dict(),
            "receipt_version": self.receipt_version,
            "request_cid": self.request_cid,
            "scan_cid": self.scan_cid,
            "schema": self.schema,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PromptGoalPlanningResult:
    graph: PromptGoalGraph
    receipt: PromptGoalPlanningReceipt
    portfolio: Any = None

    @property
    def used_fallback(self) -> bool:
        return self.receipt.fallback.used

    @property
    def provider_succeeded(self) -> bool:
        return not self.used_fallback and self.receipt.provider.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "graph": self.graph.to_dict(),
            "receipt": self.receipt.to_dict(),
        }
        if self.portfolio is not None:
            payload["portfolio"] = self.portfolio.to_dict()
        return payload


RouterCallable = Callable[[str], str]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _response_fingerprint(text: str | None) -> tuple[int, str]:
    if text is None:
        return 0, ""
    data = text.encode("utf-8", errors="surrogatepass")
    return len(data), _sha256(data)


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PromptGoalProposalError(
            f"{name} must be a non-empty canonical repository-relative path",
            reason_code="invalid_path",
        )
    candidate = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != value
        or value in {".", ".git"}
        or value.startswith(".git/")
    ):
        raise PromptGoalProposalError(
            f"{name} must be a canonical repository-relative path",
            reason_code="invalid_path",
        )
    return value


def _scan_scope(request: PromptWorkflowRequest) -> str:
    root = PurePosixPath(request.repository_root)
    directory = PurePosixPath(request.directory)
    relative = directory.relative_to(root).as_posix()
    return "" if relative == "." else relative


def _within_scope(path: str, scope: str) -> bool:
    return not scope or path == scope or path.startswith(scope + "/")


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PromptGoalProposalError(
            f"{name} must be non-empty trimmed text",
            reason_code="invalid_schema",
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise PromptGoalProposalError(
            f"{name} exceeds its safe text bound",
            reason_code="output_too_large",
        )
    if _INSTRUCTION_RE.search(value):
        raise PromptGoalProposalError(
            f"{name} contains a forbidden code, authority, policy, or completion instruction",
            reason_code="forbidden_instruction",
        )
    return value


def _string_list(
    value: Any,
    name: str,
    *,
    maximum: int,
    item_bytes: int,
    allow_empty: bool = True,
    paths: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PromptGoalProposalError(
            f"{name} must be an array", reason_code="invalid_schema"
        )
    if len(value) > maximum or (not allow_empty and not value):
        raise PromptGoalProposalError(
            f"{name} violates its item bound", reason_code="output_too_large"
        )
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(
            _safe_relative_path(item, f"{name}[{index}]")
            if paths
            else _bounded_text(item, f"{name}[{index}]", item_bytes)
        )
    if len(set(result)) != len(result):
        raise PromptGoalProposalError(
            f"{name} contains duplicate values", reason_code="duplicate_value"
        )
    return tuple(result)


def _json_depth(value: Any, depth: int = 1) -> int:
    if isinstance(value, Mapping):
        return max((depth, *(_json_depth(item, depth + 1) for item in value.values())))
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return max((depth, *(_json_depth(item, depth + 1) for item in value)))
    return depth


def _strict_object(
    value: Any,
    name: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptGoalProposalError(
            f"{name} must be an object", reason_code="invalid_schema"
        )
    keys = {str(key) for key in value}
    if keys != fields:
        missing = sorted(fields - keys)
        unknown = sorted(keys - fields)
        details = (
            f"missing {', '.join(missing)}"
            if missing
            else f"unknown {', '.join(unknown)}"
        )
        raise PromptGoalProposalError(
            f"{name} fields do not match the strict schema: {details}",
            reason_code="unknown_or_missing_field",
        )
    return value


def _strict_array(value: Any, name: str, maximum: int, *, nonempty: bool = True) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PromptGoalProposalError(
            f"{name} must be an array", reason_code="invalid_schema"
        )
    if len(value) > maximum or (nonempty and not value):
        raise PromptGoalProposalError(
            f"{name} violates its item bound", reason_code="output_too_large"
        )
    return value


def _frozen_summary_map(
    value: Mapping[str, Any] | None,
    *,
    allowed: frozenset[str],
    maximum_bytes: int,
    noun: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PromptGoalProviderRequestError(
            f"{noun} must be an object", reason_code="invalid_request_context"
        )
    unknown = set(value).difference(allowed)
    if unknown:
        raise PromptGoalProviderRequestError(
            f"{noun} contains unsupported fields: {', '.join(sorted(unknown))}",
            reason_code="invalid_request_context",
        )

    def freeze(item: Any, depth: int = 0) -> Any:
        if depth > 6:
            raise PromptGoalProviderRequestError(
                f"{noun} exceeds nesting bound", reason_code="request_over_budget"
            )
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                raise PromptGoalProviderRequestError(
                    f"{noun} contains a non-finite number",
                    reason_code="invalid_request_context",
                )
            return item
        if isinstance(item, str):
            if len(item.encode("utf-8")) > maximum_bytes or _INSTRUCTION_RE.search(item):
                raise PromptGoalProviderRequestError(
                    f"{noun} contains unsafe or overlong text",
                    reason_code="invalid_request_context",
                )
            return item
        if isinstance(item, Mapping):
            if len(item) > 64:
                raise PromptGoalProviderRequestError(
                    f"{noun} exceeds mapping bound", reason_code="request_over_budget"
                )
            return {
                str(key): freeze(member, depth + 1)
                for key, member in sorted(item.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            if len(item) > 256:
                raise PromptGoalProviderRequestError(
                    f"{noun} exceeds sequence bound", reason_code="request_over_budget"
                )
            return [freeze(member, depth + 1) for member in item]
        raise PromptGoalProviderRequestError(
            f"{noun} contains unsupported values",
            reason_code="invalid_request_context",
        )

    frozen = freeze(value)
    encoded = _canonical_json(frozen).encode("utf-8")
    if len(encoded) > maximum_bytes * 16:
        raise PromptGoalProviderRequestError(
            f"{noun} exceeds its serialized bound", reason_code="request_over_budget"
        )
    return frozen


def _prompt_evidence(request: PromptWorkflowRequest) -> PromptEvidenceRecord:
    metadata = request.prompt_source.redacted_metadata
    summary = ""
    if isinstance(metadata, Mapping):
        for key in ("summary", "objective", "title", "description"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                summary = value.strip()
                break
    if not summary:
        summary = "Immutable user prompt request."
    if len(summary.encode("utf-8")) > DEFAULT_MAX_SUMMARY_BYTES:
        summary = "Immutable user prompt request."
    return PromptEvidenceRecord(
        evidence_key="prompt:request",
        source_kind="prompt",
        artifact_cid=request.prompt_cid,
        summary=summary,
        repository_paths=(),
        claim_keys=("claim:prompt-objective",),
        authority=EvidenceAuthority.PROMPT,
        provenance={"request_cid": request.request_cid},
    )


def _evidence_terms(request: PromptWorkflowRequest) -> frozenset[str]:
    values: list[str] = []
    metadata = request.prompt_source.redacted_metadata
    if isinstance(metadata, Mapping):
        values.extend(str(item) for item in metadata.values() if isinstance(item, str))
    terms = {
        token
        for value in values
        for token in re.findall(r"[a-z0-9_]{3,}", value.lower())
    }
    return frozenset(terms)


def _compact_prompt_metadata(
    request: PromptWorkflowRequest, maximum_bytes: int
) -> dict[str, Any]:
    """Keep useful redacted scalars without forwarding an arbitrary metadata tree."""

    metadata = request.prompt_source.to_dict()["redacted_metadata"]
    if not isinstance(metadata, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in sorted(metadata):
        if len(compact) >= 16:
            break
        value = metadata[key]
        if not isinstance(key, str) or len(key.encode("utf-8")) > 128:
            continue
        if isinstance(value, (bool, int, float)) or value is None:
            compact[key] = value
        elif isinstance(value, str) and len(value.encode("utf-8")) <= maximum_bytes:
            compact[key] = value
    return compact


def _select_evidence(
    request: PromptWorkflowRequest,
    scan: DirectoryScanReceipt,
    config: PromptGoalPlannerConfig,
) -> tuple[PromptEvidenceRecord, ...]:
    terms = _evidence_terms(request)

    def rank(item: PromptEvidenceRecord) -> tuple[int, str, str]:
        haystack = " ".join(
            (item.summary, item.evidence_key, *item.repository_paths)
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        return (-score, item.evidence_key, item.evidence_cid)

    maximum = min(
        config.max_selected_evidence,
        request.budget.max_evidence,
        scan.budget.max_evidence,
    )
    ranked = sorted(scan.evidence, key=rank)
    positively_ranked = tuple(item for item in ranked if rank(item)[0] < 0)
    candidates = positively_ranked if terms and positively_ranked else tuple(ranked)
    selected = candidates[: max(0, maximum - 1)]
    prompt_item = _prompt_evidence(request)
    unique = {item.evidence_cid: item for item in (prompt_item, *selected)}
    return tuple(sorted(unique.values(), key=lambda item: item.evidence_cid))


def _validate_request_scan_pair(
    request: PromptWorkflowRequest, scan: DirectoryScanReceipt
) -> None:
    if scan.request_cid != request.request_cid:
        raise PromptGoalProviderRequestError(
            "scan receipt is bound to a different request",
            reason_code="identity_mismatch",
        )
    if scan.repository_root != request.repository_root:
        raise PromptGoalProviderRequestError(
            "scan repository root differs from request",
            reason_code="identity_mismatch",
        )
    if scan.directory != request.directory:
        raise PromptGoalProviderRequestError(
            "scan directory differs from request",
            reason_code="identity_mismatch",
        )
    if scan.repository_root_cid != request.repository_root_cid:
        raise PromptGoalProviderRequestError(
            "scan repository identity differs from request",
            reason_code="identity_mismatch",
        )
    if scan.program_root != request.program_root:
        raise PromptGoalProviderRequestError(
            "scan program root differs from request",
            reason_code="identity_mismatch",
        )


def _proposal_schema(request: PromptWorkflowRequest) -> dict[str, Any]:
    budget = request.budget
    strings = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
    }
    acceptance = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "criterion_key",
            "criterion",
            "evidence_cids",
            "validation_keys",
        ],
        "properties": {
            "criterion_key": {"type": "string", "minLength": 1},
            "criterion": {"type": "string", "minLength": 1},
            "evidence_cids": {**strings, "minItems": 1},
            "validation_keys": strings,
        },
    }
    output = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_OUTPUT_FIELDS),
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "effect": {"enum": sorted(_SAFE_OUTPUT_EFFECTS)},
            "media_type": {"enum": sorted(_SAFE_MEDIA_TYPES)},
        },
    }
    validation = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_VALIDATION_FIELDS),
        "properties": {
            "validation_key": {"type": "string", "minLength": 1},
            "argv": {**strings, "minItems": 1, "maxItems": 256},
            "cwd": {"type": "string", "minLength": 1},
            "expected_exit_codes": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "integer", "minimum": 0, "maximum": 255},
            },
        },
    }
    goal = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_GOAL_FIELDS),
        "properties": {
            "goal_key": {"type": "string", "minLength": 1},
            "parent_goal_key": {"type": "string"},
            "dependency_goal_keys": strings,
            "title": {"type": "string", "minLength": 1},
            "objective": {"type": "string", "minLength": 1},
            "rationale": {"type": "string", "minLength": 1},
            "scope_paths": {**strings, "minItems": 1},
            "acceptance": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/definitions/acceptance"},
            },
            "evidence_cids": {**strings, "minItems": 1},
            "risks": strings,
            "assumptions": strings,
        },
    }
    task = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_TASK_FIELDS),
        "properties": {
            "task_key": {"type": "string", "minLength": 1},
            "goal_key": {"type": "string", "minLength": 1},
            "dependency_task_keys": strings,
            "objective": {"type": "string", "minLength": 1},
            "rationale": {"type": "string", "minLength": 1},
            "scope_paths": {**strings, "minItems": 1},
            "outputs": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/definitions/output"},
            },
            "validations": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/definitions/validation"},
            },
            "acceptance": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/definitions/acceptance"},
            },
            "evidence_cids": {**strings, "minItems": 1},
            "priority": {"type": "string", "minLength": 1},
            "track": {"type": "string", "minLength": 1},
            "bundle": {"type": "string"},
            "parallel_lane": {"type": "string"},
            "resource_class": {
                "type": "string",
                "minLength": 1,
            },
            "predicted_files": {**strings, "minItems": 1},
            "risks": strings,
            "assumptions": strings,
            "fallback_behavior": {"enum": sorted(_SAFE_FALLBACK_BEHAVIORS)},
        },
    }
    return {
        "$id": PROMPT_GOAL_PROPOSAL_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "proposal_version",
            "root_goal_key",
            "goals",
            "tasks",
            "unresolved_questions",
            "uncertainty_debt",
        ],
        "bounds": {
            "max_goals": budget.max_goals,
            "max_tasks": budget.max_tasks,
            "max_graph_depth": budget.max_graph_depth,
            "max_serialized_bytes": min(
                budget.max_serialized_bytes,
                budget.max_provider_tokens * 4,
            ),
        },
        "definitions": {
            "acceptance": acceptance,
            "goal": goal,
            "output": output,
            "task": task,
            "validation": validation,
        },
        "properties": {
            "schema": {"const": PROMPT_GOAL_PROPOSAL_SCHEMA},
            "proposal_version": {"const": PROMPT_GOAL_PLANNER_VERSION},
            "root_goal_key": {"type": "string", "minLength": 1},
            "goals": {
                "type": "array",
                "minItems": 1,
                "maxItems": budget.max_goals,
                "items": {"$ref": "#/definitions/goal"},
            },
            "tasks": {
                "type": "array",
                "minItems": 1,
                "maxItems": budget.max_tasks,
                "items": {"$ref": "#/definitions/task"},
            },
            "unresolved_questions": strings,
            "uncertainty_debt": strings,
        },
    }


def build_prompt_goal_provider_request(
    request: PromptWorkflowRequest,
    scan: DirectoryScanReceipt,
    *,
    capabilities: Mapping[str, Any] | None = None,
    constraint_summaries: Mapping[str, Any] | None = None,
    config: PromptGoalPlannerConfig | None = None,
) -> str:
    """Compile a bounded, body-free canonical request for ``llm_router``."""

    if not isinstance(request, PromptWorkflowRequest):
        raise TypeError("request must be PromptWorkflowRequest")
    if not isinstance(scan, DirectoryScanReceipt):
        raise TypeError("scan must be DirectoryScanReceipt")
    _validate_request_scan_pair(request, scan)
    resolved = config or PromptGoalPlannerConfig()
    evidence = _select_evidence(request, scan, resolved)
    scope = _scan_scope(request)
    frozen_capabilities = _frozen_summary_map(
        capabilities,
        allowed=_CAPABILITY_KEYS,
        maximum_bytes=resolved.max_summary_bytes,
        noun="capabilities",
    )
    frozen_constraints = _frozen_summary_map(
        constraint_summaries,
        allowed=_CONSTRAINT_KEYS,
        maximum_bytes=resolved.max_summary_bytes,
        noun="constraint_summaries",
    )
    payload = {
        "acceptance": {
            "exact_schema": PROMPT_GOAL_PROPOSAL_SCHEMA,
            "json_only": True,
            "local_keys_only": True,
            "non_authoritative": True,
            "require_acceptance": request.planning_policy.require_acceptance,
            "require_validation": request.planning_policy.require_validation,
        },
        "budgets": {
            "candidate_count": request.planning_policy.candidate_count,
            "max_evidence": min(
                len(evidence),
                request.budget.max_evidence,
            ),
            "max_goals": request.budget.max_goals,
            "max_graph_depth": request.budget.max_graph_depth,
            "max_provider_tokens": request.budget.max_provider_tokens,
            "max_serialized_bytes": min(
                request.budget.max_serialized_bytes,
                resolved.max_provider_response_bytes,
            ),
            "max_tasks": request.budget.max_tasks,
        },
        "capabilities": frozen_capabilities,
        "constraints": {
            **frozen_constraints,
            "allowed_output_effects": sorted(_SAFE_OUTPUT_EFFECTS),
            "allowed_resource_classes": sorted(
                set(resolved.allowed_resource_classes)
            ),
            "allowed_validation_prefixes": [
                list(item) for item in resolved.allowed_validation_prefixes
            ],
            "completion_authoritative": False,
            "directory_scope": scope,
            "policy_roots": sorted(
                {
                    request.policy_root,
                    request.intent_ir_root,
                    request.legal_ir_root,
                    request.security_ir_root,
                }
            ),
            "protected_paths": sorted(
                set(resolved.protected_paths)
                | set(
                    item
                    for item in frozen_constraints.get("protected_paths", [])
                    if isinstance(item, str)
                )
            ),
            "shell_allowed": False,
        },
        "evidence_handles": [
            {
                "artifact_cid": item.artifact_cid,
                "authority": item.authority.value,
                "evidence_cid": item.evidence_cid,
                "evidence_key": item.evidence_key,
                "repository_paths": list(item.repository_paths),
                "source_kind": item.source_kind,
                "summary": (
                    item.summary
                    if len(item.summary.encode("utf-8")) <= resolved.max_summary_bytes
                    else "Bounded evidence handle."
                ),
            }
            for item in evidence
        ],
        "provider_instructions": [
            "Return one JSON object only, with no prose, Markdown, comments, code, or duplicate fields.",
            "Use only listed evidence CIDs, paths inside directory_scope, "
            "validation prefixes, resources, and output effects.",
            "Propose local dependency keys; do not claim policy, authority, proof, execution, or completion.",
        ],
        "request_core": {
            "allowlist_cid": request.allowlist_cid,
            "caller": request.caller,
            "directory_scope": scope,
            "intent_ir_root": request.intent_ir_root,
            "legal_ir_root": request.legal_ir_root,
            "output_policy_cid": request.output_policy.content_id,
            "planning_policy_cid": request.planning_policy.content_id,
            "policy_root": request.policy_root,
            "program_root": request.program_root,
            "prompt_cid": request.prompt_cid,
            "prompt_metadata": _compact_prompt_metadata(
                request, resolved.max_summary_bytes
            ),
            "repository_root_cid": request.repository_root_cid,
            "request_cid": request.request_cid,
            "scan_cid": scan.scan_cid,
            "security_ir_root": request.security_ir_root,
        },
        "response_schema": _proposal_schema(request),
        "schema": PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA,
        "stage": "prompt_goal_planning",
        "version": PROMPT_GOAL_PLANNER_VERSION,
    }
    encoded = _canonical_json(payload)
    encoded_bytes = encoded.encode("utf-8")
    token_char_bound = request.budget.max_prompt_tokens * 4
    maximum = min(
        resolved.max_provider_request_bytes,
        request.budget.max_serialized_bytes,
        token_char_bound,
    )
    if len(encoded_bytes) > maximum:
        raise PromptGoalProviderRequestError(
            f"provider request exceeds bounded input: {len(encoded_bytes)} > {maximum}",
            reason_code="request_over_budget",
        )
    return encoded


_TOP_FIELDS = frozenset(
    {
        "schema",
        "proposal_version",
        "root_goal_key",
        "goals",
        "tasks",
        "unresolved_questions",
        "uncertainty_debt",
    }
)
_GOAL_FIELDS = frozenset(
    {
        "goal_key",
        "parent_goal_key",
        "dependency_goal_keys",
        "title",
        "objective",
        "rationale",
        "scope_paths",
        "acceptance",
        "evidence_cids",
        "risks",
        "assumptions",
    }
)
_TASK_FIELDS = frozenset(
    {
        "task_key",
        "goal_key",
        "dependency_task_keys",
        "objective",
        "rationale",
        "scope_paths",
        "outputs",
        "validations",
        "acceptance",
        "evidence_cids",
        "priority",
        "track",
        "bundle",
        "parallel_lane",
        "resource_class",
        "predicted_files",
        "risks",
        "assumptions",
        "fallback_behavior",
    }
)
_ACCEPTANCE_FIELDS = frozenset(
    {"criterion_key", "criterion", "evidence_cids", "validation_keys"}
)
_OUTPUT_FIELDS = frozenset({"path", "effect", "media_type"})
_VALIDATION_FIELDS = frozenset(
    {"validation_key", "argv", "cwd", "expected_exit_codes"}
)


def _decode_strict_json(text: str, maximum: int, maximum_depth: int) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text:
        raise PromptGoalProposalError(
            "llm_router returned an empty response", reason_code="malformed"
        )
    data = text.encode("utf-8", errors="surrogatepass")
    if len(data) > maximum:
        raise PromptGoalProposalError(
            "llm_router response exceeds the output budget",
            reason_code="response_over_budget",
        )
    if text != text.strip() or not text.startswith("{") or not text.endswith("}"):
        raise PromptGoalProposalError(
            "llm_router response must be one unwrapped JSON object",
            reason_code="prose_wrapper",
        )

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PromptGoalProposalError(
                    f"duplicate JSON field {key!r}", reason_code="duplicate_key"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number {value}")
            ),
        )
    except PromptGoalProposalError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise PromptGoalProposalError(
            "llm_router response is not strict JSON", reason_code="malformed"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PromptGoalProposalError(
            "llm_router response must be an object", reason_code="invalid_schema"
        )
    if _json_depth(payload) > maximum_depth:
        raise PromptGoalProposalError(
            "llm_router response exceeds the nesting bound",
            reason_code="response_over_budget",
        )
    return payload


def _path_policy(
    path: Any,
    name: str,
    *,
    scope: str,
    protected_paths: frozenset[str],
    allowed_paths: tuple[str, ...],
) -> str:
    normalized = _safe_relative_path(path, name)
    if not _within_scope(normalized, scope):
        raise PromptGoalProposalError(
            f"{name} is outside the scanned directory",
            reason_code="invalid_path",
        )
    if normalized in protected_paths:
        raise PromptGoalProposalError(
            f"{name} targets a protected path", reason_code="protected_path"
        )
    if allowed_paths and not any(
        normalized == root or normalized.startswith(root.rstrip("/") + "/")
        for root in allowed_paths
    ):
        raise PromptGoalProposalError(
            f"{name} is outside allowed paths", reason_code="invalid_path"
        )
    return normalized


def _allowed_validation(
    argv: tuple[str, ...],
    *,
    prefixes: tuple[tuple[str, ...], ...],
    exact_commands: tuple[tuple[str, ...], ...],
) -> None:
    if exact_commands and argv not in exact_commands:
        raise PromptGoalProposalError(
            "validation argv is not in the pinned command capability",
            reason_code="forbidden_instruction",
        )
    if not any(argv[: len(prefix)] == prefix for prefix in prefixes):
        raise PromptGoalProposalError(
            "validation argv is outside the closed validation prefixes",
            reason_code="forbidden_instruction",
        )
    forbidden_tokens = {
        "-c",
        "--command",
        "bash",
        "sh",
        "zsh",
        "fish",
        "powershell",
        "pwsh",
        "sudo",
    }
    if any(item in forbidden_tokens or _SHELL_META_RE.search(item) for item in argv):
        raise PromptGoalProposalError(
            "validation argv contains shell or code execution",
            reason_code="forbidden_instruction",
        )
    if any(
        "\x00" in item
        or "\\" in item
        or PurePosixPath(item).is_absolute()
        or ".." in PurePosixPath(item.split("::", 1)[0]).parts
        for item in argv
    ):
        raise PromptGoalProposalError(
            "validation argv contains an invalid or escaping path",
            reason_code="invalid_path",
        )


def _parse_acceptance(
    value: Any,
    name: str,
    *,
    maximum: int,
    evidence_cids: frozenset[str],
) -> tuple[PromptAcceptanceRecord, ...]:
    records: list[PromptAcceptanceRecord] = []
    for index, raw in enumerate(_strict_array(value, name, maximum)):
        item = _strict_object(raw, f"{name}[{index}]", _ACCEPTANCE_FIELDS)
        refs = _string_list(
            item["evidence_cids"],
            f"{name}[{index}].evidence_cids",
            maximum=maximum,
            item_bytes=256,
            allow_empty=False,
        )
        if not set(refs).issubset(evidence_cids):
            raise PromptGoalProposalError(
                "acceptance references unknown evidence",
                reason_code="orphan_reference",
            )
        records.append(
            PromptAcceptanceRecord(
                criterion_key=_bounded_text(
                    item["criterion_key"], f"{name}[{index}].criterion_key", 256
                ),
                criterion=_bounded_text(
                    item["criterion"], f"{name}[{index}].criterion", 4_096
                ),
                evidence_cids=refs,
                validation_keys=_string_list(
                    item["validation_keys"],
                    f"{name}[{index}].validation_keys",
                    maximum=maximum,
                    item_bytes=256,
                ),
            )
        )
    if len({item.criterion_key for item in records}) != len(records):
        raise PromptGoalProposalError(
            f"{name} contains duplicate criterion keys",
            reason_code="duplicate_value",
        )
    return tuple(records)


def _coerce_string_commands(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    commands: list[tuple[str, ...]] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            tokens = tuple(str(token) for token in item)
            if tokens:
                commands.append(tokens)
    return tuple(commands)


def _local_graph_depth(
    nodes: Mapping[str, tuple[str, ...]],
) -> int:
    depths: dict[str, int] = {}

    def depth(node: str) -> int:
        if node not in depths:
            dependencies = nodes[node]
            depths[node] = (
                1
                if not dependencies
                else 1 + max(depth(dependency) for dependency in dependencies)
            )
        return depths[node]

    return max((depth(node) for node in nodes), default=0)


def parse_prompt_goal_graph(
    text: str,
    request: PromptWorkflowRequest,
    scan: DirectoryScanReceipt,
    *,
    config: PromptGoalPlannerConfig | None = None,
    capabilities: Mapping[str, Any] | None = None,
    constraint_summaries: Mapping[str, Any] | None = None,
) -> PromptGoalGraph:
    """Parse strict provider JSON and compile local keys to canonical graph CIDs."""

    resolved = config or PromptGoalPlannerConfig()
    _validate_request_scan_pair(request, scan)
    response_limit = min(
        resolved.max_provider_response_bytes,
        request.budget.max_serialized_bytes,
        request.budget.max_provider_tokens * 4,
    )
    payload = _decode_strict_json(
        text,
        response_limit,
        min(32, max(10, request.budget.max_graph_depth + 8)),
    )
    top = _strict_object(payload, "proposal", _TOP_FIELDS)
    if (
        top["schema"] != PROMPT_GOAL_PROPOSAL_SCHEMA
        or str(top["proposal_version"]) != PROMPT_GOAL_PLANNER_VERSION
    ):
        raise PromptGoalProposalError(
            "provider proposal has an unsupported schema",
            reason_code="invalid_schema",
        )
    selected_evidence = _select_evidence(request, scan, resolved)
    evidence_by_cid = {item.evidence_cid: item for item in selected_evidence}
    evidence_cids = frozenset(evidence_by_cid)
    scope = _scan_scope(request)
    constraints = _frozen_summary_map(
        constraint_summaries,
        allowed=_CONSTRAINT_KEYS,
        maximum_bytes=resolved.max_summary_bytes,
        noun="constraint_summaries",
    )
    frozen_capabilities = _frozen_summary_map(
        capabilities,
        allowed=_CAPABILITY_KEYS,
        maximum_bytes=resolved.max_summary_bytes,
        noun="capabilities",
    )
    protected = frozenset(
        set(resolved.protected_paths)
        | {
            _safe_relative_path(item, "protected_paths")
            for item in constraints.get("protected_paths", [])
            if isinstance(item, str)
        }
    )
    allowed_paths = tuple(
        _safe_relative_path(item, "allowed_paths")
        for item in constraints.get("allowed_paths", [])
        if isinstance(item, str)
    )
    exact_commands = _coerce_string_commands(
        constraints.get("validation_commands", ())
    )
    capability_prefixes = _coerce_string_commands(
        frozen_capabilities.get("validation_prefixes", ())
    )
    validation_prefixes = (
        tuple(
            prefix
            for prefix in resolved.allowed_validation_prefixes
            if prefix in capability_prefixes
        )
        if capability_prefixes
        else resolved.allowed_validation_prefixes
    )
    if not validation_prefixes:
        raise PromptGoalProposalError(
            "no configured validation prefix is available",
            reason_code="unsupported_capability",
        )
    capability_resources = {
        str(item)
        for item in frozen_capabilities.get("resource_classes", ())
        if isinstance(item, str)
    }
    allowed_resources = (
        set(resolved.allowed_resource_classes).intersection(capability_resources)
        if capability_resources
        else set(resolved.allowed_resource_classes)
    )
    if not allowed_resources:
        raise PromptGoalProposalError(
            "no configured resource class is available",
            reason_code="unsupported_capability",
        )

    raw_goals = _strict_array(top["goals"], "goals", request.budget.max_goals)
    raw_tasks = _strict_array(top["tasks"], "tasks", request.budget.max_tasks)
    root_key = _bounded_text(top["root_goal_key"], "root_goal_key", 256)

    goal_specs: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_goals):
        item = _strict_object(raw, f"goals[{index}]", _GOAL_FIELDS)
        key = _bounded_text(item["goal_key"], f"goals[{index}].goal_key", 256)
        if key in goal_specs:
            raise PromptGoalProposalError(
                "goals contain duplicate local keys", reason_code="duplicate_value"
            )
        goal_specs[key] = item
    if root_key not in goal_specs:
        raise PromptGoalProposalError(
            "root_goal_key does not identify a goal",
            reason_code="orphan_reference",
        )

    goal_records: dict[str, PromptGoalRecord] = {}
    while len(goal_records) < len(goal_specs):
        progressed = False
        for key in sorted(set(goal_specs) - set(goal_records)):
            item = goal_specs[key]
            parent = _bounded_text(
                item["parent_goal_key"],
                f"goals[{key}].parent_goal_key",
                256,
            ) if item["parent_goal_key"] else ""
            dependencies = _string_list(
                item["dependency_goal_keys"],
                f"goals[{key}].dependency_goal_keys",
                maximum=request.budget.max_goals,
                item_bytes=256,
            )
            references = tuple(filter(None, (parent, *dependencies)))
            if any(reference not in goal_specs for reference in references):
                raise PromptGoalProposalError(
                    "goal references an unknown local key",
                    reason_code="orphan_reference",
                )
            if key in references:
                raise PromptGoalProposalError(
                    "goal graph contains a self-cycle", reason_code="cycle"
                )
            if any(reference not in goal_records for reference in references):
                continue
            refs = _string_list(
                item["evidence_cids"],
                f"goals[{key}].evidence_cids",
                maximum=request.budget.max_evidence,
                item_bytes=256,
                allow_empty=False,
            )
            if not set(refs).issubset(evidence_cids):
                raise PromptGoalProposalError(
                    "goal references unknown evidence",
                    reason_code="orphan_reference",
                )
            paths = tuple(
                _path_policy(
                    path,
                    f"goals[{key}].scope_paths",
                    scope=scope,
                    protected_paths=protected,
                    allowed_paths=allowed_paths,
                )
                for path in _string_list(
                    item["scope_paths"],
                    f"goals[{key}].scope_paths",
                    maximum=256,
                    item_bytes=1_024,
                    allow_empty=False,
                    paths=True,
                )
            )
            goal_records[key] = PromptGoalRecord(
                goal_key=key,
                parent_goal_cid=goal_records[parent].goal_cid if parent else "",
                dependency_goal_cids=tuple(
                    goal_records[dependency].goal_cid for dependency in dependencies
                ),
                title=_bounded_text(item["title"], f"goals[{key}].title", 1_024),
                objective=_bounded_text(
                    item["objective"], f"goals[{key}].objective", 4_096
                ),
                rationale=_bounded_text(
                    item["rationale"], f"goals[{key}].rationale", 4_096
                ),
                scope_paths=paths,
                acceptance=_parse_acceptance(
                    item["acceptance"],
                    f"goals[{key}].acceptance",
                    maximum=request.budget.max_tasks,
                    evidence_cids=evidence_cids,
                ),
                evidence_cids=refs,
                risks=_string_list(
                    item["risks"],
                    f"goals[{key}].risks",
                    maximum=64,
                    item_bytes=1_024,
                ),
                assumptions=_string_list(
                    item["assumptions"],
                    f"goals[{key}].assumptions",
                    maximum=64,
                    item_bytes=1_024,
                ),
                provenance={
                    "planner": "llm_router",
                    "request_cid": request.request_cid,
                    "scan_cid": scan.scan_cid,
                },
            )
            progressed = True
        if not progressed:
            raise PromptGoalProposalError(
                "goal graph contains a cycle", reason_code="cycle"
            )
    roots = [
        key
        for key, item in goal_specs.items()
        if not item["parent_goal_key"]
    ]
    if roots != [root_key]:
        raise PromptGoalProposalError(
            "goal graph requires exactly the declared root",
            reason_code="orphan_node",
        )
    goal_edges = {
        key: tuple(
            filter(
                None,
                (
                    str(item["parent_goal_key"]),
                    *tuple(str(value) for value in item["dependency_goal_keys"]),
                ),
            )
        )
        for key, item in goal_specs.items()
    }
    if _local_graph_depth(goal_edges) > request.budget.max_graph_depth:
        raise PromptGoalProposalError(
            "goal graph exceeds the request depth budget",
            reason_code="graph_over_budget",
        )

    task_specs: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_tasks):
        item = _strict_object(raw, f"tasks[{index}]", _TASK_FIELDS)
        key = _bounded_text(item["task_key"], f"tasks[{index}].task_key", 256)
        if key in task_specs:
            raise PromptGoalProposalError(
                "tasks contain duplicate local keys", reason_code="duplicate_value"
            )
        task_specs[key] = item

    policy_roots = tuple(
        sorted(
            {
                request.policy_root,
                request.intent_ir_root,
                request.legal_ir_root,
                request.security_ir_root,
            }
        )
    )
    task_records: dict[str, PromptTaskRecord] = {}
    while len(task_records) < len(task_specs):
        progressed = False
        for key in sorted(set(task_specs) - set(task_records)):
            item = task_specs[key]
            goal_key = _bounded_text(
                item["goal_key"], f"tasks[{key}].goal_key", 256
            )
            if goal_key not in goal_records:
                raise PromptGoalProposalError(
                    "task references an unknown goal key",
                    reason_code="orphan_reference",
                )
            dependencies = _string_list(
                item["dependency_task_keys"],
                f"tasks[{key}].dependency_task_keys",
                maximum=request.budget.max_tasks,
                item_bytes=256,
            )
            if any(dependency not in task_specs for dependency in dependencies):
                raise PromptGoalProposalError(
                    "task references an unknown dependency key",
                    reason_code="orphan_reference",
                )
            if key in dependencies:
                raise PromptGoalProposalError(
                    "task graph contains a self-cycle", reason_code="cycle"
                )
            if any(dependency not in task_records for dependency in dependencies):
                continue
            refs = _string_list(
                item["evidence_cids"],
                f"tasks[{key}].evidence_cids",
                maximum=request.budget.max_evidence,
                item_bytes=256,
                allow_empty=False,
            )
            if not set(refs).issubset(evidence_cids):
                raise PromptGoalProposalError(
                    "task references unknown evidence",
                    reason_code="orphan_reference",
                )
            outputs: list[PromptOutputRecord] = []
            for output_index, raw_output in enumerate(
                _strict_array(
                    item["outputs"],
                    f"tasks[{key}].outputs",
                    256,
                )
            ):
                output = _strict_object(
                    raw_output,
                    f"tasks[{key}].outputs[{output_index}]",
                    _OUTPUT_FIELDS,
                )
                effect = _bounded_text(
                    output["effect"],
                    f"tasks[{key}].outputs[{output_index}].effect",
                    32,
                )
                if effect not in _SAFE_OUTPUT_EFFECTS:
                    raise PromptGoalProposalError(
                        "task output requests a forbidden effect",
                        reason_code="forbidden_instruction",
                    )
                media_type = _bounded_text(
                    output["media_type"],
                    f"tasks[{key}].outputs[{output_index}].media_type",
                    128,
                )
                if media_type not in _SAFE_MEDIA_TYPES:
                    raise PromptGoalProposalError(
                        "task output media type is not admitted",
                        reason_code="invalid_schema",
                    )
                outputs.append(
                    PromptOutputRecord(
                        path=_path_policy(
                            output["path"],
                            f"tasks[{key}].outputs[{output_index}].path",
                            scope=scope,
                            protected_paths=protected,
                            allowed_paths=allowed_paths,
                        ),
                        effect=effect,
                        media_type=media_type,
                    )
                )
            validations: list[PromptValidationRecord] = []
            for validation_index, raw_validation in enumerate(
                _strict_array(
                    item["validations"],
                    f"tasks[{key}].validations",
                    64,
                )
            ):
                validation = _strict_object(
                    raw_validation,
                    f"tasks[{key}].validations[{validation_index}]",
                    _VALIDATION_FIELDS,
                )
                argv = _string_list(
                    validation["argv"],
                    f"tasks[{key}].validations[{validation_index}].argv",
                    maximum=256,
                    item_bytes=1_024,
                    allow_empty=False,
                )
                _allowed_validation(
                    argv,
                    prefixes=validation_prefixes,
                    exact_commands=exact_commands,
                )
                codes_raw = validation["expected_exit_codes"]
                if isinstance(codes_raw, (str, bytes)) or not isinstance(
                    codes_raw, Sequence
                ):
                    raise PromptGoalProposalError(
                        "expected_exit_codes must be an array",
                        reason_code="invalid_schema",
                    )
                codes: list[int] = []
                for code in codes_raw:
                    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255:
                        raise PromptGoalProposalError(
                            "expected_exit_codes must contain byte-sized integers",
                            reason_code="invalid_schema",
                        )
                    codes.append(code)
                if not codes:
                    raise PromptGoalProposalError(
                        "expected_exit_codes must not be empty",
                        reason_code="missing_validation",
                    )
                cwd = validation["cwd"]
                if cwd != ".":
                    cwd = _path_policy(
                        cwd,
                        f"tasks[{key}].validations[{validation_index}].cwd",
                        scope=scope,
                        protected_paths=protected,
                        allowed_paths=allowed_paths,
                    )
                validations.append(
                    PromptValidationRecord(
                        validation_key=_bounded_text(
                            validation["validation_key"],
                            f"tasks[{key}].validations[{validation_index}].validation_key",
                            256,
                        ),
                        argv=argv,
                        cwd=cwd,
                        expected_exit_codes=tuple(codes),
                        policy_cid=request.policy_root,
                    )
                )
            paths = tuple(
                _path_policy(
                    path,
                    f"tasks[{key}].scope_paths",
                    scope=scope,
                    protected_paths=protected,
                    allowed_paths=allowed_paths,
                )
                for path in _string_list(
                    item["scope_paths"],
                    f"tasks[{key}].scope_paths",
                    maximum=256,
                    item_bytes=1_024,
                    allow_empty=False,
                    paths=True,
                )
            )
            predicted = tuple(
                _path_policy(
                    path,
                    f"tasks[{key}].predicted_files",
                    scope=scope,
                    protected_paths=protected,
                    allowed_paths=allowed_paths,
                )
                for path in _string_list(
                    item["predicted_files"],
                    f"tasks[{key}].predicted_files",
                    maximum=256,
                    item_bytes=1_024,
                    allow_empty=False,
                    paths=True,
                )
            )
            output_paths = {output.path for output in outputs}
            if not output_paths.issubset(set(predicted)):
                raise PromptGoalProposalError(
                    "every task output must be a predicted file",
                    reason_code="invalid_schema",
                )
            resource_class = _bounded_text(
                item["resource_class"], f"tasks[{key}].resource_class", 128
            )
            if resource_class not in allowed_resources:
                raise PromptGoalProposalError(
                    "task resource class is not available",
                    reason_code="unsupported_resource",
                )
            fallback_behavior = _bounded_text(
                item["fallback_behavior"], f"tasks[{key}].fallback_behavior", 128
            )
            if fallback_behavior not in _SAFE_FALLBACK_BEHAVIORS:
                raise PromptGoalProposalError(
                    "task fallback behavior is not closed",
                    reason_code="forbidden_instruction",
                )
            task_records[key] = PromptTaskRecord(
                task_key=key,
                goal_cid=goal_records[goal_key].goal_cid,
                dependency_task_cids=tuple(
                    task_records[dependency].task_cid for dependency in dependencies
                ),
                objective=_bounded_text(
                    item["objective"], f"tasks[{key}].objective", 4_096
                ),
                rationale=_bounded_text(
                    item["rationale"], f"tasks[{key}].rationale", 4_096
                ),
                scope_paths=paths,
                outputs=tuple(outputs),
                validations=tuple(validations),
                acceptance=_parse_acceptance(
                    item["acceptance"],
                    f"tasks[{key}].acceptance",
                    maximum=256,
                    evidence_cids=evidence_cids,
                ),
                evidence_cids=refs,
                policy_roots=policy_roots,
                priority=_bounded_text(
                    item["priority"], f"tasks[{key}].priority", 32
                ),
                track=_bounded_text(item["track"], f"tasks[{key}].track", 128),
                bundle=_bounded_text(
                    item["bundle"], f"tasks[{key}].bundle", 256
                ) if item["bundle"] else "",
                parallel_lane=_bounded_text(
                    item["parallel_lane"], f"tasks[{key}].parallel_lane", 256
                ) if item["parallel_lane"] else "",
                resource_class=resource_class,
                predicted_files=predicted,
                risks=_string_list(
                    item["risks"],
                    f"tasks[{key}].risks",
                    maximum=64,
                    item_bytes=1_024,
                ),
                assumptions=_string_list(
                    item["assumptions"],
                    f"tasks[{key}].assumptions",
                    maximum=64,
                    item_bytes=1_024,
                ),
                fallback_behavior=fallback_behavior,
                provenance={
                    "planner": "llm_router",
                    "request_cid": request.request_cid,
                    "scan_cid": scan.scan_cid,
                },
            )
            progressed = True
        if not progressed:
            raise PromptGoalProposalError(
                "task graph contains a cycle", reason_code="cycle"
            )
    task_edges = {
        key: tuple(str(value) for value in item["dependency_task_keys"])
        for key, item in task_specs.items()
    }
    if _local_graph_depth(task_edges) > request.budget.max_graph_depth:
        raise PromptGoalProposalError(
            "task graph exceeds the request depth budget",
            reason_code="graph_over_budget",
        )
    task_validation_keys = {
        validation.validation_key
        for task in task_records.values()
        for validation in task.validations
    }
    if any(
        not set(criterion.validation_keys).issubset(task_validation_keys)
        for goal in goal_records.values()
        for criterion in goal.acceptance
    ):
        raise PromptGoalProposalError(
            "goal acceptance references an unknown task validation",
            reason_code="orphan_reference",
        )

    parent_keys = {
        str(item["parent_goal_key"])
        for item in goal_specs.values()
        if item["parent_goal_key"]
    }
    leaf_keys = set(goal_specs) - parent_keys
    tasked_goal_keys = {
        _bounded_text(item["goal_key"], "task.goal_key", 256)
        for item in task_specs.values()
    }
    if not leaf_keys.issubset(tasked_goal_keys):
        raise PromptGoalProposalError(
            "one or more leaf goals have no task",
            reason_code="orphan_node",
        )
    unresolved = _string_list(
        top["unresolved_questions"],
        "unresolved_questions",
        maximum=256,
        item_bytes=2_048,
    )
    uncertainty = _string_list(
        top["uncertainty_debt"],
        "uncertainty_debt",
        maximum=256,
        item_bytes=2_048,
    )
    try:
        graph = PromptGoalGraph(
            request_cid=request.request_cid,
            scan_cid=scan.scan_cid,
            program_root=request.program_root,
            policy_roots=policy_roots,
            goals=tuple(goal_records.values()),
            tasks=tuple(task_records.values()),
            evidence=selected_evidence,
            unresolved_questions=unresolved,
            uncertainty_debt=uncertainty,
        )
    except (PromptGraphError, PromptWorkflowContractError) as exc:
        raise PromptGoalProposalError(
            str(exc),
            reason_code="cycle" if "cycle" in str(exc).lower() else "invalid_graph",
        ) from exc
    if len(graph.canonical_bytes()) > request.budget.max_serialized_bytes:
        raise PromptGoalProposalError(
            "compiled graph exceeds serialized byte budget",
            reason_code="graph_over_budget",
        )
    if len(graph.goals) > request.budget.max_goals or len(graph.tasks) > request.budget.max_tasks:
        raise PromptGoalProposalError(
            "compiled graph exceeds count budget", reason_code="graph_over_budget"
        )
    return graph


def _fallback_output_paths(
    request: PromptWorkflowRequest,
    evidence: Sequence[PromptEvidenceRecord],
    protected: frozenset[str],
) -> tuple[str, ...]:
    scope = _scan_scope(request)
    terms = _evidence_terms(request)
    ranked: list[tuple[int, str]] = []
    for item in evidence:
        haystack = " ".join(
            (item.summary, item.evidence_key, *item.repository_paths)
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        ranked.extend(
            (-score, path)
            for path in item.repository_paths
            if _within_scope(path, scope) and path not in protected
        )
    candidates = tuple(path for _score, path in sorted(set(ranked)))
    if not candidates:
        raise PromptWorkflowContractError(
            "deterministic planning requires an in-scope, non-protected "
            "codebase evidence path"
        )
    return candidates[: request.budget.max_tasks]


def deterministic_prompt_goal_graph(
    request: PromptWorkflowRequest,
    scan: DirectoryScanReceipt,
    *,
    config: PromptGoalPlannerConfig | None = None,
    reason_code: str = "deterministic_requested",
) -> PromptGoalGraph:
    """Create a stable schema-equivalent graph without provider inference."""

    resolved = config or PromptGoalPlannerConfig()
    _validate_request_scan_pair(request, scan)
    evidence = _select_evidence(request, scan, resolved)
    prompt_item = next(
        item for item in evidence if item.evidence_key == "prompt:request"
    )
    scope = _scan_scope(request)
    output_paths = _fallback_output_paths(
        request, evidence, frozenset(resolved.protected_paths)
    )
    scope_path = scope or str(PurePosixPath(output_paths[0]).parent)
    test_paths = tuple(
        path
        for path in output_paths
        if "test" in PurePosixPath(path).name.casefold()
    )

    def validation_for(path: str, index: int) -> PromptValidationRecord:
        targets = test_paths or (path,)
        return PromptValidationRecord(
            validation_key=f"validation:deterministic:{index}",
            argv=("python", "-m", "pytest", *targets, "-q"),
            cwd=".",
            policy_cid=request.policy_root,
        )

    validations = tuple(
        validation_for(path, index)
        for index, path in enumerate(output_paths, start=1)
    )
    acceptance = PromptAcceptanceRecord(
        criterion_key="criterion:bounded-plan",
        criterion="The proposed scoped change passes the declared deterministic validation.",
        evidence_cids=(prompt_item.evidence_cid,),
        validation_keys=tuple(item.validation_key for item in validations),
    )
    objective = prompt_item.summary
    if _INSTRUCTION_RE.search(objective):
        objective = "Produce a bounded implementation plan for the immutable prompt request."
    root = PromptGoalRecord(
        goal_key="goal:root",
        parent_goal_cid="",
        dependency_goal_cids=(),
        title="Plan the immutable prompt request",
        objective=objective,
        rationale="A deterministic plan preserves progress without provider authority.",
        scope_paths=(scope_path,),
        acceptance=(acceptance,),
        evidence_cids=(prompt_item.evidence_cid,),
        risks=("Repository semantics may require later admission-time refinement.",),
        assumptions=("Pinned request and scan identities remain current.",),
        provenance={
            "planner": "deterministic",
            "reason_code": reason_code,
            "request_cid": request.request_cid,
            "scan_cid": scan.scan_cid,
        },
    )
    policy_roots = tuple(
        sorted(
            {
                request.policy_root,
                request.intent_ir_root,
                request.legal_ir_root,
                request.security_ir_root,
            }
        )
    )
    tasks: list[PromptTaskRecord] = []
    for index, (output_path, validation) in enumerate(
        zip(output_paths, validations), start=1
    ):
        supporting = tuple(
            item.evidence_cid
            for item in evidence
            if output_path in item.repository_paths
        ) or (prompt_item.evidence_cid,)
        tasks.append(
            PromptTaskRecord(
                task_key=f"task:codebase-evidence:{index}",
                goal_cid=root.goal_cid,
                dependency_task_cids=(),
                objective=(
                    f"Implement the evidence-backed change at {output_path} "
                    "and satisfy its focused validation."
                ),
                rationale=(
                    "The pinned repository scan nominates this exact path; "
                    "independent admission still decides authority and proof."
                ),
                scope_paths=(output_path,),
                outputs=(
                    PromptOutputRecord(
                        path=output_path,
                        effect="modify",
                        media_type=(
                            "text/x-python"
                            if output_path.endswith(".py")
                            else "text/markdown"
                            if output_path.endswith(".md")
                            else "text/plain"
                        ),
                    ),
                ),
                validations=(validation,),
                acceptance=(acceptance,),
                evidence_cids=supporting,
                policy_roots=policy_roots,
                priority="P1",
                track="prompt-goal-planning",
                bundle="prompt-workflow/deterministic",
                parallel_lane=f"prompt-goal-deterministic-{index}",
                resource_class="cpu-small",
                predicted_files=(output_path,),
                risks=(
                    "Static scan evidence may omit a dependent path; admission "
                    "must check impact closure.",
                ),
                assumptions=(
                    "Admission will independently verify scope, policy, and proof obligations.",
                ),
                fallback_behavior="fail_closed",
                provenance={
                    "planner": "deterministic",
                    "strategy": "codebase-evidence-template",
                    "reason_code": reason_code,
                    "request_cid": request.request_cid,
                    "scan_cid": scan.scan_cid,
                },
            )
        )
    graph = PromptGoalGraph(
        request_cid=request.request_cid,
        scan_cid=scan.scan_cid,
        program_root=request.program_root,
        policy_roots=policy_roots,
        goals=(root,),
        tasks=tuple(tasks),
        evidence=evidence,
        unresolved_questions=(),
        uncertainty_debt=(
            "Provider-independent admission must verify the proposed task against current repository semantics.",
        ),
    )
    if len(graph.canonical_bytes()) > request.budget.max_serialized_bytes:
        raise PromptWorkflowBoundsError(
            "deterministic graph exceeds max_serialized_bytes"
        )
    return graph


def _default_router(
    prompt: str,
    request: PromptWorkflowRequest,
    config: PromptGoalPlannerConfig,
) -> str:
    # Keep the dependency optional and reuse the existing isolated adapter.
    # ASI-168: envelope + gateway when usage mode is not off; off mode is
    # unchanged for focused-suite compatibility and never loads the migration
    # stack.
    import os

    from ..todo_daemon.llm import LlmRouterInvocation, call_llm_router

    provider = config.provider
    if provider is None and request.planning_policy.provider_preferences:
        provider = request.planning_policy.provider_preferences[0]
    model = config.model
    if request.planning_policy.model_preferences and config.model == "gpt-5.3-codex-spark":
        model = request.planning_policy.model_preferences[0]
    tokens = min(
        config.max_new_tokens or request.budget.max_provider_tokens,
        request.budget.max_provider_tokens,
    )
    invocation = LlmRouterInvocation(
        repo_root=config.repo_root or Path(request.repository_root),
        provider=provider,
        model_name=model,
        allow_local_fallback=config.allow_local_fallback,
        timeout_seconds=min(
            config.timeout_seconds,
            max(1, request.budget.max_latency_ms // 1_000),
        ),
        max_new_tokens=tokens,
        max_prompt_chars=min(
            config.max_provider_request_bytes,
            request.budget.max_prompt_tokens * 4,
        ),
        temperature=config.temperature,
        reject_effective_provider_name=(
            None if config.allow_local_fallback else "local_hf"
        ),
    )

    def _invoke() -> str:
        return call_llm_router(prompt, invocation)

    mode_raw = str(
        os.environ.get("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", "off")
    ).strip().casefold()
    if mode_raw in {"", "off"}:
        return _invoke()

    from ..provider_usage_migration import (
        ConsumerId,
        build_consumer_call_context,
        dispatch_migrated_provider_call,
        resolve_usage_mode,
        retain_last_call_result,
    )

    mode = resolve_usage_mode(mode_raw)
    provider_id = str(provider or "llm_router:auto")
    context = build_consumer_call_context(
        consumer_id=ConsumerId.PROMPT_GOAL_PLANNER,
        provider_id=provider_id,
        stage="prompt_goal_planning",
        task_id=str(getattr(request, "request_id", "") or "prompt_goal"),
        goal_id=str(
            getattr(request, "goal_id", "") or "goal:prompt-goal-planner"
        ),
        objective_id="prompt_goal_planner",
        tree_id=str(
            getattr(request, "repository_tree_id", "") or "tree:unknown"
        ),
        estimated_output_tokens=max(0, int(tokens or 0)),
        estimated_input_tokens=max(
            0, int(getattr(request.budget, "max_prompt_tokens", 0) or 0)
        ),
        metadata={"model": str(model)},
    )
    migrated = dispatch_migrated_provider_call(
        context=context,
        invoke=_invoke,
        mode=mode,
    )
    retain_last_call_result(ConsumerId.PROMPT_GOAL_PLANNER, migrated)
    return migrated.text


def _failure_kind(exc: BaseException) -> str:
    reason = getattr(exc, "reason_code", "")
    if reason in {
        "response_over_budget",
        "graph_over_budget",
        "output_too_large",
    }:
        return "over_budget"
    message = f"{type(exc).__name__}: {exc}".lower()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "unavailable" in message:
        return "unavailable"
    if isinstance(exc, PromptGoalProposalError):
        return "malformed"
    return "failed"


def _generate_prompt_goal_graph_single(
    request: PromptWorkflowRequest,
    scan: DirectoryScanReceipt,
    *,
    router: RouterCallable | None = None,
    capabilities: Mapping[str, Any] | None = None,
    constraint_summaries: Mapping[str, Any] | None = None,
    config: PromptGoalPlannerConfig | None = None,
) -> PromptGoalPlanningResult:
    """Generate one strict graph or a deterministic schema-equivalent fallback."""

    resolved = config or PromptGoalPlannerConfig()
    provider_id = resolved.provider or (
        request.planning_policy.provider_preferences[0]
        if request.planning_policy.provider_preferences
        else "llm_router:auto"
    )
    model_id = (
        request.planning_policy.model_preferences[0]
        if request.planning_policy.model_preferences
        and resolved.model == "gpt-5.3-codex-spark"
        else resolved.model
    )
    tokens = min(
        resolved.max_new_tokens or request.budget.max_provider_tokens,
        request.budget.max_provider_tokens,
    )
    timeout_ms = min(
        resolved.timeout_seconds * 1_000,
        request.budget.max_latency_ms,
    )
    try:
        provider_prompt = build_prompt_goal_provider_request(
            request,
            scan,
            capabilities=capabilities,
            constraint_summaries=constraint_summaries,
            config=resolved,
        )
    except PromptGoalProviderRequestError as exc:
        if exc.reason_code != "request_over_budget":
            raise
        graph = deterministic_prompt_goal_graph(
            request,
            scan,
            config=resolved,
            reason_code="request_over_budget",
        )
        identity_bytes = _canonical_json(
            {
                "request_cid": request.request_cid,
                "scan_cid": scan.scan_cid,
                "schema": PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA,
            }
        ).encode("utf-8")
        provider_receipt = PromptGoalProviderReceipt(
            attempted=False,
            status="over_budget",
            reason_code="request_over_budget",
            provider_id=provider_id,
            model_id=model_id,
            request_bytes=0,
            request_sha256=_sha256(identity_bytes),
            timeout_ms=timeout_ms,
            max_new_tokens=tokens,
        )
        parse_receipt = PromptGoalParseReceipt(
            attempted=False,
            status="not_attempted",
            reason_code="request_over_budget",
            proposal_schema=PROMPT_GOAL_PROPOSAL_SCHEMA,
            response_bytes=0,
            response_sha256="",
        )
        fallback_receipt = PromptGoalFallbackReceipt(
            used=True,
            status="succeeded",
            reason_code="request_over_budget",
            plan_root_cid=graph.plan_root_cid,
        )
        return PromptGoalPlanningResult(
            graph=graph,
            receipt=PromptGoalPlanningReceipt(
                request_cid=request.request_cid,
                scan_cid=scan.scan_cid,
                plan_root_cid=graph.plan_root_cid,
                outcome="fallback",
                provider=provider_receipt,
                parse=parse_receipt,
                fallback=fallback_receipt,
            ),
        )
    request_data = provider_prompt.encode("utf-8")
    request_hash = _sha256(request_data)
    if not request.planning_policy.allow_model:
        graph = deterministic_prompt_goal_graph(
            request, scan, config=resolved, reason_code="policy_disabled"
        )
        provider_receipt = PromptGoalProviderReceipt(
            attempted=False,
            status="disabled",
            reason_code="policy_disabled",
            provider_id=provider_id,
            model_id=model_id,
            request_bytes=len(request_data),
            request_sha256=request_hash,
            timeout_ms=timeout_ms,
            max_new_tokens=tokens,
        )
        parse_receipt = PromptGoalParseReceipt(
            attempted=False,
            status="not_attempted",
            reason_code="policy_disabled",
            proposal_schema=PROMPT_GOAL_PROPOSAL_SCHEMA,
            response_bytes=0,
            response_sha256="",
        )
        fallback_receipt = PromptGoalFallbackReceipt(
            used=True,
            status="succeeded",
            reason_code="policy_disabled",
            plan_root_cid=graph.plan_root_cid,
        )
        return PromptGoalPlanningResult(
            graph=graph,
            receipt=PromptGoalPlanningReceipt(
                request_cid=request.request_cid,
                scan_cid=scan.scan_cid,
                plan_root_cid=graph.plan_root_cid,
                outcome="fallback",
                provider=provider_receipt,
                parse=parse_receipt,
                fallback=fallback_receipt,
            ),
        )
    if isinstance(capabilities, Mapping) and capabilities.get("available") is False:
        graph = deterministic_prompt_goal_graph(
            request,
            scan,
            config=resolved,
            reason_code="capability_unavailable",
        )
        provider_receipt = PromptGoalProviderReceipt(
            attempted=False,
            status="unavailable",
            reason_code="capability_unavailable",
            provider_id=provider_id,
            model_id=model_id,
            request_bytes=len(request_data),
            request_sha256=request_hash,
            timeout_ms=timeout_ms,
            max_new_tokens=tokens,
        )
        parse_receipt = PromptGoalParseReceipt(
            attempted=False,
            status="not_attempted",
            reason_code="capability_unavailable",
            proposal_schema=PROMPT_GOAL_PROPOSAL_SCHEMA,
            response_bytes=0,
            response_sha256="",
        )
        fallback_receipt = PromptGoalFallbackReceipt(
            used=True,
            status="succeeded",
            reason_code="capability_unavailable",
            plan_root_cid=graph.plan_root_cid,
        )
        return PromptGoalPlanningResult(
            graph=graph,
            receipt=PromptGoalPlanningReceipt(
                request_cid=request.request_cid,
                scan_cid=scan.scan_cid,
                plan_root_cid=graph.plan_root_cid,
                outcome="fallback",
                provider=provider_receipt,
                parse=parse_receipt,
                fallback=fallback_receipt,
            ),
        )

    started = time.monotonic()
    response: str | None = None
    try:
        response = (
            router(provider_prompt)
            if router is not None
            else _default_router(provider_prompt, request, resolved)
        )
        if not isinstance(response, str):
            raise PromptGoalProposalError(
                "llm_router returned a non-text response",
                reason_code="malformed",
            )
        response_bytes, response_hash = _response_fingerprint(response)
        graph = parse_prompt_goal_graph(
            response,
            request,
            scan,
            config=resolved,
            capabilities=capabilities,
            constraint_summaries=constraint_summaries,
        )
        latency_ms = max(0, int((time.monotonic() - started) * 1_000))
        provider_receipt = PromptGoalProviderReceipt(
            attempted=True,
            status="succeeded",
            reason_code="provider_graph_accepted",
            provider_id=provider_id,
            model_id=model_id,
            request_bytes=len(request_data),
            request_sha256=request_hash,
            response_bytes=response_bytes,
            response_sha256=response_hash,
            timeout_ms=timeout_ms,
            max_new_tokens=tokens,
            latency_ms=latency_ms,
        )
        parse_receipt = PromptGoalParseReceipt(
            attempted=True,
            status="succeeded",
            reason_code="strict_graph_accepted",
            proposal_schema=PROMPT_GOAL_PROPOSAL_SCHEMA,
            response_bytes=response_bytes,
            response_sha256=response_hash,
            goal_count=len(graph.goals),
            task_count=len(graph.tasks),
            evidence_count=len(graph.evidence),
            plan_root_cid=graph.plan_root_cid,
        )
        fallback_receipt = PromptGoalFallbackReceipt(
            used=False,
            status="not_used",
            reason_code="provider_graph_accepted",
        )
        return PromptGoalPlanningResult(
            graph=graph,
            receipt=PromptGoalPlanningReceipt(
                request_cid=request.request_cid,
                scan_cid=scan.scan_cid,
                plan_root_cid=graph.plan_root_cid,
                outcome="provider",
                provider=provider_receipt,
                parse=parse_receipt,
                fallback=fallback_receipt,
            ),
        )
    except Exception as exc:
        failure = _failure_kind(exc)
        reason = str(getattr(exc, "reason_code", "") or failure)
        response_bytes, response_hash = _response_fingerprint(response)
        latency_ms = max(0, int((time.monotonic() - started) * 1_000))
        graph = deterministic_prompt_goal_graph(
            request, scan, config=resolved, reason_code=reason
        )
        provider_receipt = PromptGoalProviderReceipt(
            attempted=True,
            status=failure,
            reason_code=reason,
            provider_id=provider_id,
            model_id=model_id,
            request_bytes=len(request_data),
            request_sha256=request_hash,
            response_bytes=response_bytes,
            response_sha256=response_hash,
            timeout_ms=timeout_ms,
            max_new_tokens=tokens,
            latency_ms=latency_ms,
        )
        parse_attempted = response is not None
        parse_receipt = PromptGoalParseReceipt(
            attempted=parse_attempted,
            status="rejected" if parse_attempted else "not_attempted",
            reason_code=reason,
            proposal_schema=PROMPT_GOAL_PROPOSAL_SCHEMA,
            response_bytes=response_bytes,
            response_sha256=response_hash,
        )
        fallback_receipt = PromptGoalFallbackReceipt(
            used=True,
            status="succeeded",
            reason_code=reason,
            plan_root_cid=graph.plan_root_cid,
        )
        return PromptGoalPlanningResult(
            graph=graph,
            receipt=PromptGoalPlanningReceipt(
                request_cid=request.request_cid,
                scan_cid=scan.scan_cid,
                plan_root_cid=graph.plan_root_cid,
                outcome="fallback",
                provider=provider_receipt,
                parse=parse_receipt,
                fallback=fallback_receipt,
            ),
        )


@dataclass(frozen=True)
class PromptGoalCandidateSnapshot:
    """One request/scan-bound graph with a content-addressed disposition."""

    graph: PromptGoalGraph
    source: str
    disposition: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.graph, PromptGoalGraph):
            raise PromptGoalPlannerError("candidate graph must be PromptGoalGraph")
        source = str(self.source or "").strip()
        if source not in {"deterministic_baseline", "model_proposal"}:
            raise PromptGoalPlannerError("candidate source is unsupported")
        disposition = str(self.disposition or "").strip()
        if disposition not in {"selected", "rejected"}:
            raise PromptGoalPlannerError("candidate disposition is unsupported")
        reasons = tuple(
            sorted({str(item).strip() for item in self.reason_codes if str(item).strip()})
        )
        if disposition == "selected" and reasons:
            raise PromptGoalPlannerError(
                "selected prompt candidate cannot contain rejection reasons"
            )
        if disposition == "rejected" and not reasons:
            raise PromptGoalPlannerError(
                "rejected prompt candidate requires a typed reason"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def snapshot_id(self) -> str:
        return _sha256(
            _canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "graph": self.graph.to_dict(),
            "plan_root_cid": self.graph.plan_root_cid,
            "source": self.source,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
        }
        if include_identity:
            payload["snapshot_id"] = self.snapshot_id
        return payload


@dataclass(frozen=True)
class PromptGoalCandidatePortfolio:
    """Bounded deterministic-first prompt graph portfolio."""

    request_cid: str
    scan_cid: str
    candidate_count: int
    snapshots: tuple[PromptGoalCandidateSnapshot, ...]
    provider_receipts: tuple[PromptGoalProviderReceipt, ...] = ()
    schema: str = PROMPT_GOAL_CANDIDATE_PORTFOLIO_SCHEMA

    def __post_init__(self) -> None:
        for name in ("request_cid", "scan_cid"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise PromptGoalPlannerError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or not 1 <= self.candidate_count <= 32
        ):
            raise PromptGoalPlannerError("candidate_count must be in [1, 32]")
        snapshots = tuple(self.snapshots)
        if not snapshots or len(snapshots) > self.candidate_count:
            raise PromptGoalPlannerError(
                "portfolio must retain a baseline within candidate_count"
            )
        if snapshots[0].source != "deterministic_baseline":
            raise PromptGoalPlannerError(
                "first prompt candidate must be the deterministic baseline"
            )
        if any(
            item.graph.request_cid != self.request_cid
            or item.graph.scan_cid != self.scan_cid
            for item in snapshots
        ):
            raise PromptGoalPlannerError(
                "prompt candidate is detached from frozen request or scan"
            )
        ids = [item.graph.plan_root_cid for item in snapshots]
        if len(ids) != len(set(ids)):
            raise PromptGoalPlannerError("prompt portfolio contains duplicate graphs")
        selected = [item for item in snapshots if item.disposition == "selected"]
        if len(selected) != 1:
            raise PromptGoalPlannerError(
                "prompt portfolio requires exactly one selected candidate"
            )
        if any(
            item.disposition
            != (
                "selected"
                if item.graph.plan_root_cid == selected[0].graph.plan_root_cid
                else "rejected"
            )
            for item in snapshots
        ):
            raise PromptGoalPlannerError(
                "every non-selected prompt candidate must be rejected"
            )
        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "provider_receipts", tuple(self.provider_receipts))
        if self.schema != PROMPT_GOAL_CANDIDATE_PORTFOLIO_SCHEMA:
            raise PromptGoalPlannerError("unsupported prompt portfolio schema")

    @property
    def selected(self) -> PromptGoalCandidateSnapshot:
        return next(item for item in self.snapshots if item.disposition == "selected")

    @property
    def baseline(self) -> PromptGoalCandidateSnapshot:
        return self.snapshots[0]

    @property
    def rejected(self) -> tuple[PromptGoalCandidateSnapshot, ...]:
        return tuple(item for item in self.snapshots if item.disposition == "rejected")

    @property
    def portfolio_id(self) -> str:
        return _sha256(
            _canonical_json(self.to_dict(include_identity=False)).encode("utf-8")
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "request_cid": self.request_cid,
            "scan_cid": self.scan_cid,
            "candidate_count": self.candidate_count,
            "snapshots": [item.to_dict() for item in self.snapshots],
            "provider_usage": [
                {
                    **item.to_dict(),
                    "usage_id": _sha256(
                        _canonical_json(item.to_dict()).encode("utf-8")
                    ),
                }
                for item in self.provider_receipts
            ],
            "baseline_snapshot_id": self.baseline.snapshot_id,
            "selected_snapshot_id": self.selected.snapshot_id,
            "rejected_snapshot_ids": [
                item.snapshot_id for item in self.rejected
            ],
        }
        if include_identity:
            payload["portfolio_id"] = self.portfolio_id
        return payload


def generate_prompt_goal_candidate_portfolio(
    request: PromptWorkflowRequest,
    scan: DirectoryScanReceipt,
    *,
    router: RouterCallable | None = None,
    capabilities: Mapping[str, Any] | None = None,
    constraint_summaries: Mapping[str, Any] | None = None,
    config: PromptGoalPlannerConfig | None = None,
) -> PromptGoalCandidatePortfolio:
    """Generate exactly one baseline plus bounded, deduplicated proposals.

    The legacy singular helper remains available, but this is the authoritative
    ``candidate_count`` path: the configured count is an aggregate ceiling and
    includes the mandatory codebase-derived baseline.
    """

    resolved = config or PromptGoalPlannerConfig()
    baseline = deterministic_prompt_goal_graph(
        request,
        scan,
        config=resolved,
        reason_code="mandatory_deterministic_baseline",
    )
    graphs: list[tuple[PromptGoalGraph, str]] = [
        (baseline, "deterministic_baseline")
    ]
    provider_receipts: list[PromptGoalProviderReceipt] = []
    if request.planning_policy.allow_model:
        for _index in range(request.planning_policy.candidate_count - 1):
            result = _generate_prompt_goal_graph_single(
                request,
                scan,
                router=router,
                capabilities=capabilities,
                constraint_summaries=constraint_summaries,
                config=resolved,
            )
            provider_receipts.append(result.receipt.provider)
            if (
                result.provider_succeeded
                and result.graph.plan_root_cid
                not in {item.plan_root_cid for item, _source in graphs}
            ):
                graphs.append((result.graph, "model_proposal"))
            if len(graphs) >= request.planning_policy.candidate_count:
                break

    # All provider graphs have already crossed strict scope/schema/command
    # admission.  Rank deterministic, reproducible structural utility only;
    # no provider confidence or prose participates.
    ranked = sorted(
        graphs,
        key=lambda item: (
            -len(item[0].tasks),
            -sum(len(task.evidence_cids) for task in item[0].tasks),
            len(item[0].unresolved_questions),
            item[0].plan_root_cid,
        ),
    )
    selected_id = ranked[0][0].plan_root_cid
    snapshots = tuple(
        PromptGoalCandidateSnapshot(
            graph=graph,
            source=source,
            disposition=(
                "selected" if graph.plan_root_cid == selected_id else "rejected"
            ),
            reason_codes=(
                ()
                if graph.plan_root_cid == selected_id
                else ("lower_deterministic_structural_utility",)
            ),
        )
        for graph, source in graphs
    )
    return PromptGoalCandidatePortfolio(
        request_cid=request.request_cid,
        scan_cid=scan.scan_cid,
        candidate_count=request.planning_policy.candidate_count,
        snapshots=snapshots,
        provider_receipts=tuple(provider_receipts),
    )


def generate_prompt_goal_graph(
    request: PromptWorkflowRequest,
    scan: DirectoryScanReceipt,
    *,
    router: RouterCallable | None = None,
    capabilities: Mapping[str, Any] | None = None,
    constraint_summaries: Mapping[str, Any] | None = None,
    config: PromptGoalPlannerConfig | None = None,
) -> PromptGoalPlanningResult:
    """Generate the selected graph, using a real portfolio when requested.

    A count of one retains the historical singular provider behavior for API
    compatibility.  The explicit portfolio helper treats one as baseline-only,
    and every count above one flows through that deterministic-first contract.
    """

    if request.planning_policy.candidate_count == 1:
        return _generate_prompt_goal_graph_single(
            request,
            scan,
            router=router,
            capabilities=capabilities,
            constraint_summaries=constraint_summaries,
            config=config,
        )
    portfolio = generate_prompt_goal_candidate_portfolio(
        request,
        scan,
        router=router,
        capabilities=capabilities,
        constraint_summaries=constraint_summaries,
        config=config,
    )
    selected = portfolio.selected
    matching_receipt = next(
        (
            receipt
            for receipt in portfolio.provider_receipts
            if selected.source == "model_proposal"
            and receipt.status == "succeeded"
        ),
        None,
    )
    if matching_receipt is None:
        provider = PromptGoalProviderReceipt(
            attempted=bool(portfolio.provider_receipts),
            status=(
                portfolio.provider_receipts[-1].status
                if portfolio.provider_receipts
                else "disabled"
            ),
            reason_code="deterministic_baseline_selected",
            provider_id=(
                portfolio.provider_receipts[-1].provider_id
                if portfolio.provider_receipts
                else "llm_router:auto"
            ),
            model_id=(
                portfolio.provider_receipts[-1].model_id
                if portfolio.provider_receipts
                else (config or PromptGoalPlannerConfig()).model
            ),
            request_bytes=(
                portfolio.provider_receipts[-1].request_bytes
                if portfolio.provider_receipts
                else 0
            ),
            request_sha256=(
                portfolio.provider_receipts[-1].request_sha256
                if portfolio.provider_receipts
                else _sha256(
                    _canonical_json(
                        {
                            "request_cid": request.request_cid,
                            "scan_cid": scan.scan_cid,
                        }
                    ).encode("utf-8")
                )
            ),
        )
        outcome = "fallback"
        fallback = PromptGoalFallbackReceipt(
            used=True,
            status="succeeded",
            reason_code="deterministic_baseline_selected",
            plan_root_cid=selected.graph.plan_root_cid,
        )
        parse = PromptGoalParseReceipt(
            attempted=False,
            status="not_attempted",
            reason_code="deterministic_baseline_selected",
            proposal_schema=PROMPT_GOAL_PROPOSAL_SCHEMA,
            response_bytes=0,
            response_sha256="",
        )
    else:
        provider = matching_receipt
        outcome = "provider"
        fallback = PromptGoalFallbackReceipt(
            used=False,
            status="not_used",
            reason_code="provider_graph_selected",
        )
        parse = PromptGoalParseReceipt(
            attempted=True,
            status="succeeded",
            reason_code="strict_graph_selected",
            proposal_schema=PROMPT_GOAL_PROPOSAL_SCHEMA,
            response_bytes=provider.response_bytes,
            response_sha256=provider.response_sha256,
            goal_count=len(selected.graph.goals),
            task_count=len(selected.graph.tasks),
            evidence_count=len(selected.graph.evidence),
            plan_root_cid=selected.graph.plan_root_cid,
        )
    return PromptGoalPlanningResult(
        graph=selected.graph,
        receipt=PromptGoalPlanningReceipt(
            request_cid=request.request_cid,
            scan_cid=scan.scan_cid,
            plan_root_cid=selected.graph.plan_root_cid,
            outcome=outcome,
            provider=provider,
            parse=parse,
            fallback=fallback,
        ),
        portfolio=portfolio,
    )


# Descriptive compatibility aliases for callers that name the planning stage.
build_goal_planning_prompt = build_prompt_goal_provider_request
parse_goal_planning_response = parse_prompt_goal_graph
deterministic_goal_planner = deterministic_prompt_goal_graph
plan_prompt_goal_graph = generate_prompt_goal_graph


# ---------------------------------------------------------------------------
# Residual-only LLM repair (PDR-025) — PlannerDoctorContextCapsule integration
# ---------------------------------------------------------------------------

RESIDUAL_ONLY_REPAIR_STAGE = "residual_only_llm_repair"


@dataclass(frozen=True)
class ResidualOnlyRepairReceipt:
    """Body-free receipt for residual-only prompt-goal repair.

    When deterministic closure already exists, ``llm_attempted`` is false and
    no provider text is retained.  Otherwise the receipt binds residual budget
    usage and admission of rejected-record replacements only.
    """

    capsule_id: str
    disposition: str
    llm_attempted: bool
    outcome: str
    reason_code: str
    planning_receipt: PromptGoalPlanningReceipt | None = None
    residual_usage: Mapping[str, Any] = field(default_factory=dict)
    admitted_record_ids: tuple[str, ...] = ()
    request_sha256: str = ""
    response_sha256: str = ""
    schema: str = (
        "ipfs_accelerate_py/agent-supervisor/residual-only-repair-receipt@1"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capsule_id": self.capsule_id,
            "disposition": self.disposition,
            "llm_attempted": self.llm_attempted,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "planning_receipt": (
                self.planning_receipt.to_dict()
                if self.planning_receipt is not None
                else None
            ),
            "residual_usage": dict(self.residual_usage),
            "admitted_record_ids": list(self.admitted_record_ids),
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "completion_authority": False,
            "proof_authority": False,
            "stage": RESIDUAL_ONLY_REPAIR_STAGE,
        }


def _import_planner_doctor_context():
    """Lazy import to keep prompt package import-light for cold paths."""

    from ..context.planner_doctor_context import (
        PlannerDoctorContextCapsule,
        ResidualRepairDisposition,
        ResidualProposalError,
        admit_residual_proposal,
        build_residual_provider_request,
        open_residual_repair_session,
    )

    return (
        PlannerDoctorContextCapsule,
        ResidualRepairDisposition,
        ResidualProposalError,
        admit_residual_proposal,
        build_residual_provider_request,
        open_residual_repair_session,
    )


def build_residual_only_provider_request(capsule: Any) -> str:
    """Compile residual-only provider JSON from a PlannerDoctor context capsule."""

    (
        PlannerDoctorContextCapsule,
        _Disposition,
        _ProposalError,
        _admit,
        build_residual_provider_request,
        _open,
    ) = _import_planner_doctor_context()
    if not isinstance(capsule, PlannerDoctorContextCapsule):
        raise PromptGoalProviderRequestError(
            "capsule must be a PlannerDoctorContextCapsule",
            reason_code="invalid_capsule",
        )
    try:
        return build_residual_provider_request(capsule)
    except Exception as exc:
        reason = str(getattr(exc, "reason_code", "") or "request_error")
        raise PromptGoalProviderRequestError(
            f"residual-only provider request failed: {exc}",
            reason_code=reason,
        ) from exc


def generate_residual_only_repair(
    capsule: Any,
    *,
    router: RouterCallable | None = None,
    request: PromptWorkflowRequest | None = None,
    scan: DirectoryScanReceipt | None = None,
    config: PromptGoalPlannerConfig | None = None,
) -> ResidualOnlyRepairReceipt:
    """Run residual-only LLM repair, or skip when deterministic closure holds.

    The model may replace only rejected/repairable proposal records.  Prompt
    and repository instructions are inert; malformed/scope-widening/authority/
    completion output fails closed.  Maximum residual call/token/round/cost
    budgets are enforced by the capsule session.
    """

    (
        PlannerDoctorContextCapsule,
        ResidualRepairDisposition,
        ResidualProposalError,
        admit_residual_proposal,
        build_residual_provider_request,
        open_residual_repair_session,
    ) = _import_planner_doctor_context()

    if not isinstance(capsule, PlannerDoctorContextCapsule):
        raise PromptGoalPlannerError(
            "capsule must be a PlannerDoctorContextCapsule",
            reason_code="invalid_capsule",
        )

    session = open_residual_repair_session(capsule)
    if session.disposition is ResidualRepairDisposition.DETERMINISTIC_CLOSED:
        return ResidualOnlyRepairReceipt(
            capsule_id=capsule.capsule_id,
            disposition=session.disposition.value,
            llm_attempted=False,
            outcome="deterministic_closed",
            reason_code="deterministic_closure_exists",
            residual_usage=session.usage.to_dict(),
        )
    if session.disposition is ResidualRepairDisposition.BLOCKED:
        return ResidualOnlyRepairReceipt(
            capsule_id=capsule.capsule_id,
            disposition=session.disposition.value,
            llm_attempted=False,
            outcome="blocked",
            reason_code="residual_blocked",
            residual_usage=session.usage.to_dict(),
        )

    residual_request = build_residual_provider_request(capsule)
    request_hash = _sha256(residual_request.encode("utf-8"))

    # Optional: if a full prompt workflow is bound, also compile the standard
    # planning receipt path for portfolio continuity — still residual-only.
    planning_receipt: PromptGoalPlanningReceipt | None = None
    if request is not None and scan is not None:
        # Inject residual constraints into the standard planner path without
        # replaying full repository dumps.
        constraint_summaries = {
            "allowed_paths": list(capsule.allowed_paths),
            "protected_paths": [],
            "validation_commands": list(capsule.validation_commands),
            "proof_handles": list(capsule.satisfied_proof_handles),
            "constraint_summaries": [
                "residual_only",
                "replace_rejected_records_only",
                "prompt_instructions_inert",
            ],
        }
        result = generate_prompt_goal_graph(
            request,
            scan,
            router=router,
            capabilities={"available": router is not None},
            constraint_summaries=constraint_summaries,
            config=config,
        )
        planning_receipt = result.receipt
        if result.used_fallback or not result.provider_succeeded:
            return ResidualOnlyRepairReceipt(
                capsule_id=capsule.capsule_id,
                disposition=session.disposition.value,
                llm_attempted=result.receipt.provider.attempted,
                outcome="fallback",
                reason_code=result.receipt.provider.reason_code
                or "provider_fallback",
                planning_receipt=planning_receipt,
                residual_usage=session.usage.to_dict(),
                request_sha256=request_hash,
                response_sha256=result.receipt.provider.response_sha256,
            )

    if router is None:
        return ResidualOnlyRepairReceipt(
            capsule_id=capsule.capsule_id,
            disposition=session.disposition.value,
            llm_attempted=False,
            outcome="skipped",
            reason_code="no_router",
            planning_receipt=planning_receipt,
            residual_usage=session.usage.to_dict(),
            request_sha256=request_hash,
        )

    response: str | None = None
    try:
        response = router(residual_request)
        if not isinstance(response, str):
            raise ResidualProposalError(
                "residual router returned non-text",
                reason_code="malformed",
            )
        response_bytes, response_hash = _response_fingerprint(response)
        try:
            proposal = json.loads(response)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ResidualProposalError(
                "residual proposal is not valid JSON",
                reason_code="malformed",
            ) from exc
        admission, charged = admit_residual_proposal(
            capsule,
            proposal,
            session=session,
            response_tokens=max(1, response_bytes // 4),
        )
        return ResidualOnlyRepairReceipt(
            capsule_id=capsule.capsule_id,
            disposition=session.disposition.value,
            llm_attempted=True,
            outcome="accepted"
            if admission.decision.value == "accepted"
            else "rejected",
            reason_code=(
                "residual_proposal_accepted"
                if admission.decision.value == "accepted"
                else (admission.reason_codes[0] if admission.reason_codes else "rejected")
            ),
            planning_receipt=planning_receipt,
            residual_usage=charged.usage.to_dict(),
            admitted_record_ids=admission.admitted_record_ids,
            request_sha256=request_hash,
            response_sha256=response_hash,
        )
    except Exception as exc:
        reason = str(getattr(exc, "reason_code", "") or _failure_kind(exc))
        response_bytes, response_hash = _response_fingerprint(response)
        return ResidualOnlyRepairReceipt(
            capsule_id=capsule.capsule_id,
            disposition=session.disposition.value,
            llm_attempted=response is not None,
            outcome="rejected",
            reason_code=reason,
            planning_receipt=planning_receipt,
            residual_usage=session.usage.to_dict(),
            request_sha256=request_hash,
            response_sha256=response_hash,
        )


__all__ = [
    "PROMPT_GOAL_PLANNER_VERSION",
    "PROMPT_GOAL_CANDIDATE_PORTFOLIO_SCHEMA",
    "PROMPT_GOAL_PLANNING_RECEIPT_SCHEMA",
    "PROMPT_GOAL_PROPOSAL_SCHEMA",
    "PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA",
    "RESIDUAL_ONLY_REPAIR_STAGE",
    "PromptGoalFallbackReceipt",
    "PromptGoalCandidatePortfolio",
    "PromptGoalCandidateSnapshot",
    "PromptGoalParseReceipt",
    "PromptGoalPlannerConfig",
    "PromptGoalPlannerError",
    "PromptGoalPlanningReceipt",
    "PromptGoalPlanningResult",
    "PromptGoalProposalError",
    "PromptGoalProviderReceipt",
    "PromptGoalProviderRequestError",
    "ResidualOnlyRepairReceipt",
    "build_goal_planning_prompt",
    "build_prompt_goal_provider_request",
    "build_residual_only_provider_request",
    "deterministic_goal_planner",
    "deterministic_prompt_goal_graph",
    "generate_prompt_goal_graph",
    "generate_prompt_goal_candidate_portfolio",
    "generate_residual_only_repair",
    "parse_goal_planning_response",
    "parse_prompt_goal_graph",
    "plan_prompt_goal_graph",
]
