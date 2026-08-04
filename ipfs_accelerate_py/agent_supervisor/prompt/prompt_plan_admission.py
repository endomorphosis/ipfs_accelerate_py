"""Fail-closed admission for prompt-generated supervisor plans.

This module is an adapter over the existing formal and cross-IR compilers.  It
does not score away a hard failure and it does not infer authority, proofs, or
validation results.  Final workflow CIDs are published only after every gate
has accepted the same canonical action/effect graph.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..planning.formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
    PlanCompilationResult,
)
from ..proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceFreshness,
    ProofVerdict,
    content_identity,
)
from ..proof.ir_constraint_compiler import (
    PlanAdmissionReceipt,
    PlanAdmissionRequest,
)
from .prompt_workflow import (
    DirectoryScanReceipt,
    PromptGoalGraph,
    PromptWorkflowRequest,
    prompt_workflow_cid,
)


PROMPT_PLAN_ADMISSION_VERSION: Final[int] = 1
PROMPT_PLAN_ADMISSION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/prompt-plan-admission-receipt@1"
)
PROMPT_PLAN_ADMISSION_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/prompt-plan-admission-policy@1"
)

_SHELL_META_RE = re.compile(
    r"(?:[;&|`]|\$\(|\$\{|(?:^|\s)(?:>|<){1,2}(?:\s|$))"
)
_SHELL_TOKENS = frozenset(
    {
        "-c",
        "--command",
        "bash",
        "cmd",
        "cmd.exe",
        "eval",
        "exec",
        "fish",
        "powershell",
        "pwsh",
        "sh",
        "sudo",
        "zsh",
    }
)
_DEFAULT_PROTECTED_PATHS = (
    "docs/architecture/agent_supervisor_self_improvement.objectives.md",
    "docs/architecture/agent_supervisor_self_improvement.todo.md",
)


class PromptPlanAdmissionVerdict(str, Enum):
    ADMITTED = "admitted"
    ACCEPTED = "admitted"
    REJECTED = "rejected"
    DENIED = "rejected"


class PromptPlanAdmissionCode(str, Enum):
    MALFORMED_GRAPH = "malformed_graph"
    STALE_ROOT = "stale_root"
    UNKNOWN_MANDATORY_STATE = "unknown_mandatory_state"
    DISCONNECTED_GRAPH = "disconnected_graph"
    CYCLIC_GRAPH = "cyclic_graph"
    UNSTABLE_TOPOLOGY = "unstable_topology"
    ACCEPTANCE_UNCOVERED = "acceptance_uncovered"
    TASK_TOO_BROAD = "task_too_broad"
    CONFLICT_UNORDERED = "conflict_unordered"
    RESOURCE_INFEASIBLE = "resource_infeasible"
    OUTPUT_FORBIDDEN = "output_forbidden"
    UNBOUND_PATH = "unbound_path"
    SHELL_VALIDATION = "shell_bearing_validation"
    VALIDATION_FORBIDDEN = "validation_forbidden"
    EVIDENCE_UNTRACED = "evidence_untraced"
    FORMAL_REJECTED = "formal_compilation_rejected"
    IR_BINDING_MISMATCH = "ir_binding_mismatch"
    IR_REJECTED = "ir_admission_rejected"
    MISSING_PROOF = "missing_proof"
    MISSING_VALIDATION = "missing_validation"
    UNDECLARED_EFFECT = "undeclared_effect"


@dataclass(frozen=True)
class PromptPlanAdmissionPolicy:
    """Closed quality, path, validation, and granularity policy."""

    allowed_path_roots: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = _DEFAULT_PROTECTED_PATHS
    allowed_output_effects: tuple[str, ...] = ("create", "modify", "write")
    allowed_resource_classes: tuple[str, ...] = (
        "cpu-large",
        "cpu-medium",
        "cpu-small",
        "io-artifact",
        "provider-llm",
    )
    allowed_media_types: tuple[str, ...] = (
        "application/json",
        "application/octet-stream",
        "text/markdown",
        "text/plain",
        "text/x-python",
        "text/yaml",
    )
    allowed_validation_prefixes: tuple[tuple[str, ...], ...] = (
        ("pytest",),
        ("python", "-m", "pytest"),
        ("python3", "-m", "pytest"),
    )
    exact_validation_argv: tuple[tuple[str, ...], ...] = ()
    max_acceptance_per_task: int = 12
    max_outputs_per_task: int = 16
    max_scope_paths_per_task: int = 32
    max_dependencies_per_task: int = 32
    require_mandatory_closure: bool = True
    require_evidence_path_binding: bool = True

    def __post_init__(self) -> None:
        for name in ("allowed_path_roots", "protected_paths"):
            values = _canonical_paths(getattr(self, name), name)
            object.__setattr__(self, name, values)
        for name in (
            "allowed_output_effects",
            "allowed_resource_classes",
            "allowed_media_types",
        ):
            values = _strings(getattr(self, name), name)
            if not values:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, values)
        for name in ("allowed_validation_prefixes", "exact_validation_argv"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ValueError(f"{name} must be a sequence")
            values = tuple(
                sorted(
                    {
                        tuple(_argv(part, f"{name}[{index}]"))
                        for index, part in enumerate(raw)
                    }
                )
            )
            if name == "allowed_validation_prefixes" and not values:
                raise ValueError("allowed_validation_prefixes must not be empty")
            object.__setattr__(self, name, values)
        for name in (
            "max_acceptance_per_task",
            "max_outputs_per_task",
            "max_scope_paths_per_task",
            "max_dependencies_per_task",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "require_mandatory_closure",
            "require_evidence_path_binding",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def policy_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROMPT_PLAN_ADMISSION_POLICY_SCHEMA,
            "version": PROMPT_PLAN_ADMISSION_VERSION,
            "allowed_path_roots": list(self.allowed_path_roots),
            "protected_paths": list(self.protected_paths),
            "allowed_output_effects": list(self.allowed_output_effects),
            "allowed_resource_classes": list(self.allowed_resource_classes),
            "allowed_media_types": list(self.allowed_media_types),
            "allowed_validation_prefixes": [
                list(item) for item in self.allowed_validation_prefixes
            ],
            "exact_validation_argv": [
                list(item) for item in self.exact_validation_argv
            ],
            "max_acceptance_per_task": self.max_acceptance_per_task,
            "max_outputs_per_task": self.max_outputs_per_task,
            "max_scope_paths_per_task": self.max_scope_paths_per_task,
            "max_dependencies_per_task": self.max_dependencies_per_task,
            "require_mandatory_closure": self.require_mandatory_closure,
            "require_evidence_path_binding": self.require_evidence_path_binding,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PromptPlanAdmissionPolicy":
        allowed = {
            "schema",
            "version",
            "policy_id",
            "allowed_path_roots",
            "protected_paths",
            "allowed_output_effects",
            "allowed_resource_classes",
            "allowed_media_types",
            "allowed_validation_prefixes",
            "exact_validation_argv",
            "max_acceptance_per_task",
            "max_outputs_per_task",
            "max_scope_paths_per_task",
            "max_dependencies_per_task",
            "require_mandatory_closure",
            "require_evidence_path_binding",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "prompt-plan admission policy contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if payload.get("schema") != PROMPT_PLAN_ADMISSION_POLICY_SCHEMA:
            raise ValueError("unsupported prompt-plan admission policy schema")
        if payload.get("version") != PROMPT_PLAN_ADMISSION_VERSION:
            raise ValueError("unsupported prompt-plan admission policy version")
        result = cls(
            allowed_path_roots=tuple(payload.get("allowed_path_roots") or ()),
            protected_paths=tuple(payload.get("protected_paths") or ()),
            allowed_output_effects=tuple(
                payload.get("allowed_output_effects") or ()
            ),
            allowed_resource_classes=tuple(
                payload.get("allowed_resource_classes") or ()
            ),
            allowed_media_types=tuple(
                payload.get("allowed_media_types") or ()
            ),
            allowed_validation_prefixes=tuple(
                tuple(item)
                for item in payload.get("allowed_validation_prefixes") or ()
            ),
            exact_validation_argv=tuple(
                tuple(item)
                for item in payload.get("exact_validation_argv") or ()
            ),
            max_acceptance_per_task=payload.get(
                "max_acceptance_per_task", 0
            ),
            max_outputs_per_task=payload.get("max_outputs_per_task", 0),
            max_scope_paths_per_task=payload.get(
                "max_scope_paths_per_task", 0
            ),
            max_dependencies_per_task=payload.get(
                "max_dependencies_per_task", 0
            ),
            require_mandatory_closure=payload.get(
                "require_mandatory_closure", True
            ),
            require_evidence_path_binding=payload.get(
                "require_evidence_path_binding", True
            ),
        )
        claimed = str(payload.get("policy_id") or "")
        if claimed and claimed != result.policy_id:
            raise ValueError("prompt-plan admission policy identity does not match")
        return result


@dataclass(frozen=True)
class PromptPlanAdmissionFinding:
    code: str
    domain: str
    path: str
    message: str
    action_id: str = ""
    effect_id: str = ""
    source_ids: tuple[str, ...] = ()
    counterexample: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "code",
            "domain",
            "path",
            "message",
            "action_id",
            "effect_id",
        ):
            value = str(getattr(self, name) or "").strip()
            if name in {"code", "domain", "path", "message"} and not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        canonical = _canonical_value(self.counterexample)
        if not isinstance(canonical, dict):
            raise ValueError("counterexample must be a mapping")
        object.__setattr__(
            self,
            "counterexample",
            MappingProxyType(canonical),
        )

    @property
    def finding_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "domain": self.domain,
            "path": self.path,
            "message": self.message,
            "action_id": self.action_id,
            "effect_id": self.effect_id,
            "source_ids": list(self.source_ids),
            "counterexample": dict(self.counterexample),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PromptPlanAdmissionFinding":
        result = cls(
            code=payload.get("code", ""),
            domain=payload.get("domain", ""),
            path=payload.get("path", ""),
            message=payload.get("message", ""),
            action_id=payload.get("action_id", ""),
            effect_id=payload.get("effect_id", ""),
            source_ids=tuple(payload.get("source_ids") or ()),
            counterexample=payload.get("counterexample") or {},
        )
        claimed = str(payload.get("finding_id") or "")
        if claimed and claimed != result.finding_id:
            raise ValueError("prompt-plan finding identity does not match")
        return result


@dataclass(frozen=True)
class PromptPlanAdmissionReceipt:
    candidate_plan_cid: str
    repository_tree_id: str
    policy_id: str
    verdict: PromptPlanAdmissionVerdict
    candidate_task_cids: tuple[str, ...] = ()
    topological_task_cids: tuple[str, ...] = ()
    topology_id: str = ""
    formal_plan_id: str = ""
    formal_source_identity: str = ""
    ir_request_id: str = ""
    ir_receipt_id: str = ""
    final_plan_cid: str = ""
    final_task_cids: tuple[str, ...] = ()
    findings: tuple[PromptPlanAdmissionFinding, ...] = ()
    invariants: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", PromptPlanAdmissionVerdict(self.verdict))
        for name in (
            "candidate_plan_cid",
            "repository_tree_id",
            "policy_id",
            "topology_id",
            "formal_plan_id",
            "formal_source_identity",
            "ir_request_id",
            "ir_receipt_id",
            "final_plan_cid",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in (
            "candidate_task_cids",
            "final_task_cids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        object.__setattr__(
            self,
            "topological_task_cids",
            _ordered_strings(
                self.topological_task_cids, "topological_task_cids"
            ),
        )
        values = tuple(
            sorted(
                {
                    item.finding_id: item
                    for item in self.findings
                }.values(),
                key=lambda item: item.finding_id,
            )
        )
        object.__setattr__(self, "findings", values)
        frozen_invariants = {
            str(key): bool(value)
            for key, value in sorted(self.invariants.items())
        }
        object.__setattr__(
            self, "invariants", MappingProxyType(frozen_invariants)
        )
        if self.admitted:
            if self.findings or not self.final_plan_cid:
                raise ValueError("admitted receipt requires final IDs and no findings")
            if self.final_task_cids != self.candidate_task_cids:
                raise ValueError("admitted receipt must publish every candidate task CID")
            if not all(self.invariants.values()):
                raise ValueError("admitted receipt requires every invariant")
        elif self.final_plan_cid or self.final_task_cids:
            raise ValueError("rejected receipt cannot publish final plan/task CIDs")

    @property
    def admitted(self) -> bool:
        return self.verdict is PromptPlanAdmissionVerdict.ADMITTED

    @property
    def authorizes_execution(self) -> bool:
        return False

    @property
    def candidate_order_independent(self) -> bool:
        return True

    @property
    def irrelevant_corpus_independent(self) -> bool:
        return True

    @property
    def plan_root_cid(self) -> str:
        return self.final_plan_cid

    @property
    def task_cids(self) -> tuple[str, ...]:
        return self.final_task_cids

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(sorted({item.code for item in self.findings}))

    @property
    def rejection_reasons(self) -> tuple[PromptPlanAdmissionFinding, ...]:
        return self.findings

    @property
    def counterexamples(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item.counterexample
            for item in self.findings
            if item.counterexample
        )

    @property
    def receipt_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROMPT_PLAN_ADMISSION_RECEIPT_SCHEMA,
            "version": PROMPT_PLAN_ADMISSION_VERSION,
            "candidate_plan_cid": self.candidate_plan_cid,
            "repository_tree_id": self.repository_tree_id,
            "policy_id": self.policy_id,
            "verdict": self.verdict.value,
            "admitted": self.admitted,
            "candidate_task_cids": list(self.candidate_task_cids),
            "topological_task_cids": list(self.topological_task_cids),
            "topology_id": self.topology_id,
            "formal_plan_id": self.formal_plan_id,
            "formal_source_identity": self.formal_source_identity,
            "ir_request_id": self.ir_request_id,
            "ir_receipt_id": self.ir_receipt_id,
            "final_plan_cid": self.final_plan_cid,
            "final_task_cids": list(self.final_task_cids),
            "findings": [
                {**item.to_dict(), "finding_id": item.finding_id}
                for item in self.findings
            ],
            "reason_codes": list(self.reason_codes),
            "invariants": dict(self.invariants),
            "authorizes_execution": False,
            "candidate_order_independent": self.candidate_order_independent,
            "irrelevant_corpus_independent": self.irrelevant_corpus_independent,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def to_json(self) -> str:
        return self.canonical_bytes.decode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromptPlanAdmissionReceipt":
        allowed = {
            "schema",
            "version",
            "receipt_id",
            "candidate_plan_cid",
            "repository_tree_id",
            "policy_id",
            "verdict",
            "admitted",
            "candidate_task_cids",
            "topological_task_cids",
            "topology_id",
            "formal_plan_id",
            "formal_source_identity",
            "ir_request_id",
            "ir_receipt_id",
            "final_plan_cid",
            "final_task_cids",
            "findings",
            "reason_codes",
            "invariants",
            "authorizes_execution",
            "candidate_order_independent",
            "irrelevant_corpus_independent",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "prompt-plan admission receipt contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if payload.get("schema") != PROMPT_PLAN_ADMISSION_RECEIPT_SCHEMA:
            raise ValueError("unsupported prompt-plan admission receipt schema")
        if payload.get("version") != PROMPT_PLAN_ADMISSION_VERSION:
            raise ValueError("unsupported prompt-plan admission receipt version")
        result = cls(
            candidate_plan_cid=payload.get("candidate_plan_cid", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            verdict=payload.get("verdict", PromptPlanAdmissionVerdict.REJECTED),
            candidate_task_cids=tuple(payload.get("candidate_task_cids") or ()),
            topological_task_cids=tuple(
                payload.get("topological_task_cids") or ()
            ),
            topology_id=payload.get("topology_id", ""),
            formal_plan_id=payload.get("formal_plan_id", ""),
            formal_source_identity=payload.get("formal_source_identity", ""),
            ir_request_id=payload.get("ir_request_id", ""),
            ir_receipt_id=payload.get("ir_receipt_id", ""),
            final_plan_cid=payload.get("final_plan_cid", ""),
            final_task_cids=tuple(payload.get("final_task_cids") or ()),
            findings=tuple(
                PromptPlanAdmissionFinding.from_dict(item)
                for item in payload.get("findings") or ()
            ),
            invariants=payload.get("invariants") or {},
        )
        if payload.get("receipt_id") != result.receipt_id:
            raise ValueError("prompt-plan admission receipt identity does not match")
        if payload.get("admitted") is not result.admitted:
            raise ValueError("prompt-plan admitted flag does not match verdict")
        if tuple(payload.get("reason_codes") or ()) != result.reason_codes:
            raise ValueError("prompt-plan reason codes do not match findings")
        if bool(payload.get("authorizes_execution", False)):
            raise ValueError("prompt-plan admission cannot authorize execution")
        if payload.get("candidate_order_independent") is not True:
            raise ValueError("receipt must retain candidate-order independence")
        if payload.get("irrelevant_corpus_independent") is not True:
            raise ValueError("receipt must retain irrelevant-corpus independence")
        return result


@dataclass(frozen=True)
class PromptPlanAdmissionRequest:
    graph: PromptGoalGraph
    repository_tree_id: str
    ir_request: PlanAdmissionRequest
    workflow_request: PromptWorkflowRequest | None = None
    scan_receipt: DirectoryScanReceipt | None = None
    policy: PromptPlanAdmissionPolicy = field(
        default_factory=PromptPlanAdmissionPolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.graph, PromptGoalGraph):
            raise TypeError("graph must be PromptGoalGraph")
        tree_id = str(self.repository_tree_id or "").strip()
        if not tree_id:
            raise ValueError("repository_tree_id is required")
        object.__setattr__(self, "repository_tree_id", tree_id)
        if not isinstance(self.ir_request, PlanAdmissionRequest):
            raise TypeError("ir_request must be PlanAdmissionRequest")
        if self.workflow_request is not None and not isinstance(
            self.workflow_request, PromptWorkflowRequest
        ):
            raise TypeError("workflow_request must be PromptWorkflowRequest")
        if self.scan_receipt is not None and not isinstance(
            self.scan_receipt, DirectoryScanReceipt
        ):
            raise TypeError("scan_receipt must be DirectoryScanReceipt")
        if not isinstance(self.policy, PromptPlanAdmissionPolicy):
            raise TypeError("policy must be PromptPlanAdmissionPolicy")


@dataclass(frozen=True)
class PromptPlanAdmissionResult:
    receipt: PromptPlanAdmissionReceipt
    formal_compilation: PlanCompilationResult | None = None
    ir_receipt: PlanAdmissionReceipt | None = None
    admitted_graph: PromptGoalGraph | None = None

    def __post_init__(self) -> None:
        if self.receipt.admitted != (self.admitted_graph is not None):
            raise ValueError("admitted graph must agree with receipt verdict")
        if self.receipt.admitted and (
            self.formal_compilation is None or self.ir_receipt is None
        ):
            raise ValueError("admission requires formal and IR receipts")

    @property
    def admitted(self) -> bool:
        return self.receipt.admitted

    @property
    def accepted(self) -> bool:
        return self.admitted

    @property
    def verdict(self) -> PromptPlanAdmissionVerdict:
        return self.receipt.verdict

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return self.receipt.reason_codes

    @property
    def plan_root_cid(self) -> str:
        return self.receipt.plan_root_cid

    @property
    def task_cids(self) -> tuple[str, ...]:
        return self.receipt.task_cids

    @property
    def formal_result(self) -> PlanCompilationResult | None:
        return self.formal_compilation

    @property
    def admission_receipt(self) -> PromptPlanAdmissionReceipt:
        return self.receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "formal_compilation": (
                self.formal_compilation.to_dict()
                if self.formal_compilation is not None
                else None
            ),
            "ir_receipt": (
                self.ir_receipt.to_dict() if self.ir_receipt is not None else None
            ),
            "admitted_graph": (
                self.admitted_graph.to_dict()
                if self.admitted_graph is not None
                else None
            ),
        }


def _strings(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = []
    for item in values:
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(f"{name} must contain non-empty trimmed strings")
        result.append(item)
    return tuple(sorted(set(result)))


def _ordered_strings(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(values)
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in result
    ):
        raise ValueError(f"{name} must contain non-empty trimmed strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _canonical_paths(values: Any, name: str) -> tuple[str, ...]:
    return tuple(_canonical_path(item, name) for item in _strings(values, name))


def _canonical_path(value: str, name: str) -> str:
    candidate = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != value
        or value in {"", ".", ".git"}
        or value.startswith(".git/")
    ):
        raise ValueError(f"{name} contains a non-canonical repository path")
    return value


def _argv(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an argv sequence")
    result = tuple(value)
    if not result or any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in result
    ):
        raise ValueError(f"{name} must contain non-empty trimmed argv tokens")
    return result


def _canonical_value(value: Any) -> Any:
    def plain(item: Any) -> Any:
        if isinstance(item, Enum):
            return item.value
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        if isinstance(item, Mapping):
            return {
                str(key): plain(member)
                for key, member in sorted(
                    item.items(), key=lambda pair: str(pair[0])
                )
            }
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return [plain(member) for member in item]
        converter = getattr(item, "to_dict", None)
        if callable(converter):
            return plain(converter())
        raise TypeError(f"unsupported canonical value {type(item).__name__}")

    return json.loads(
        json.dumps(
            plain(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _overlap(left: str, right: str) -> bool:
    return _within(left, right) or _within(right, left)


def _finding(
    code: PromptPlanAdmissionCode | str,
    domain: str,
    path: str,
    message: str,
    **values: Any,
) -> PromptPlanAdmissionFinding:
    return PromptPlanAdmissionFinding(
        code=code.value if isinstance(code, PromptPlanAdmissionCode) else str(code),
        domain=domain,
        path=path,
        message=message,
        **values,
    )


def _stable_topology(graph: PromptGoalGraph) -> tuple[str, ...]:
    dependencies = {
        task.task_cid: set(task.dependency_task_cids) for task in graph.tasks
    }
    dependents: dict[str, set[str]] = {key: set() for key in dependencies}
    for task_id, required in dependencies.items():
        for dependency in required:
            dependents[dependency].add(task_id)
    ready = sorted(key for key, value in dependencies.items() if not value)
    result: list[str] = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for dependent in sorted(dependents[current]):
            dependencies[dependent].discard(current)
            if (
                not dependencies[dependent]
                and dependent not in result
                and dependent not in ready
            ):
                ready.append(dependent)
                ready.sort()
    if len(result) != len(dependencies):
        raise ValueError("task graph contains a cycle")
    return tuple(result)


def _transitive_dependencies(graph: PromptGoalGraph) -> dict[str, set[str]]:
    direct = {
        task.task_cid: set(task.dependency_task_cids) for task in graph.tasks
    }
    closure: dict[str, set[str]] = {}

    def visit(task_id: str) -> set[str]:
        if task_id in closure:
            return closure[task_id]
        result = set(direct[task_id])
        for dependency in tuple(result):
            result.update(visit(dependency))
        closure[task_id] = result
        return result

    for key in sorted(direct):
        visit(key)
    return closure


def _goal_descends_from(
    candidate_goal_id: str,
    ancestor_goal_id: str,
    graph: PromptGoalGraph,
) -> bool:
    goals = {item.goal_cid: item for item in graph.goals}
    cursor = goals[candidate_goal_id]
    while True:
        if cursor.goal_cid == ancestor_goal_id:
            return True
        if not cursor.parent_goal_cid:
            return False
        cursor = goals[cursor.parent_goal_cid]


def _graph_findings(
    graph: PromptGoalGraph,
    *,
    tree_id: str,
    workflow: PromptWorkflowRequest | None,
    scan: DirectoryScanReceipt | None,
    policy: PromptPlanAdmissionPolicy,
) -> list[PromptPlanAdmissionFinding]:
    findings: list[PromptPlanAdmissionFinding] = []
    evidence = {item.evidence_cid: item for item in graph.evidence}
    validation_keys = {
        validation.validation_key
        for task in graph.tasks
        for validation in task.validations
    }

    if graph.unresolved_questions:
        findings.append(
            _finding(
                PromptPlanAdmissionCode.UNKNOWN_MANDATORY_STATE,
                "quality",
                "$.unresolved_questions",
                "mandatory planning questions remain unresolved",
                counterexample={
                    "unresolved_questions": list(graph.unresolved_questions)
                },
            )
        )

    if workflow is not None:
        expected_roots = {
            workflow.policy_root,
            workflow.intent_ir_root,
            workflow.legal_ir_root,
            workflow.security_ir_root,
        }
        root_checks = (
            ("request_cid", workflow.request_cid, graph.request_cid),
            ("program_root", workflow.program_root, graph.program_root),
        )
        for kind, expected, observed in root_checks:
            if expected != observed:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.STALE_ROOT,
                        "root",
                        f"$.{kind}",
                        f"{kind} is stale",
                        counterexample={
                            "expected": expected,
                            "observed": observed,
                        },
                    )
                )
        if set(graph.policy_roots) != expected_roots:
            findings.append(
                _finding(
                    PromptPlanAdmissionCode.STALE_ROOT,
                    "root",
                    "$.policy_roots",
                    "graph policy roots do not exactly match the pinned workflow roots",
                    counterexample={
                        "expected": sorted(expected_roots),
                        "observed": list(graph.policy_roots),
                    },
                )
            )
    if scan is not None:
        root_checks = (
            ("scan_cid", scan.scan_cid, graph.scan_cid),
            ("scan.request_cid", scan.request_cid, graph.request_cid),
            ("scan.program_root", scan.program_root, graph.program_root),
            ("repository_tree_id", scan.dirty_worktree_root, tree_id),
        )
        for kind, expected, observed in root_checks:
            if expected != observed:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.STALE_ROOT,
                        "root",
                        f"$.{kind}",
                        f"{kind} is stale",
                        counterexample={
                            "expected": expected,
                            "observed": observed,
                        },
                    )
                )

    scan_scope = ""
    if workflow is not None:
        root = PurePosixPath(workflow.repository_root)
        directory = PurePosixPath(workflow.directory)
        relative = directory.relative_to(root).as_posix()
        scan_scope = "" if relative == "." else relative

    for goal_index, goal in enumerate(graph.goals):
        if scan_scope:
            for scope_path in goal.scope_paths:
                if not _within(scope_path, scan_scope):
                    findings.append(
                        _finding(
                            PromptPlanAdmissionCode.UNBOUND_PATH,
                            "output",
                            f"$.goals[{goal_index}].scope_paths",
                            "goal scope is outside the pinned scan directory",
                            counterexample={
                                "path": scope_path,
                                "scan_scope": scan_scope,
                            },
                        )
                    )
        descendant_goal_ids = {
            candidate.goal_cid
            for candidate in graph.goals
            if _goal_descends_from(candidate.goal_cid, goal.goal_cid, graph)
        }
        covering_tasks = [
            task for task in graph.tasks if task.goal_cid in descendant_goal_ids
        ]
        for criterion_index, criterion in enumerate(goal.acceptance):
            missing = set(criterion.validation_keys) - validation_keys
            if not criterion.validation_keys or missing:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.ACCEPTANCE_UNCOVERED,
                        "quality",
                        f"$.goals[{goal_index}].acceptance[{criterion_index}]",
                        "goal acceptance must be covered by declared validation",
                        counterexample={
                            "criterion_key": criterion.criterion_key,
                            "missing_validation_keys": sorted(missing),
                        },
                    )
                )
            covered = any(
                task_criterion.criterion_key == criterion.criterion_key
                and set(criterion.validation_keys).issubset(
                    task_criterion.validation_keys
                )
                and set(criterion.evidence_cids).issubset(
                    task_criterion.evidence_cids
                )
                for task in covering_tasks
                for task_criterion in task.acceptance
            )
            if not covered:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.ACCEPTANCE_UNCOVERED,
                        "quality",
                        f"$.goals[{goal_index}].acceptance[{criterion_index}]",
                        "goal acceptance has no task-level coverage in its goal subtree",
                        counterexample={
                            "criterion_key": criterion.criterion_key,
                            "covering_task_cids": [
                                task.task_cid for task in covering_tasks
                            ],
                        },
                    )
                )
            if (
                not criterion.evidence_cids
                or not set(criterion.evidence_cids).issubset(goal.evidence_cids)
            ):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.EVIDENCE_UNTRACED,
                        "evidence",
                        f"$.goals[{goal_index}].acceptance[{criterion_index}]",
                        "goal acceptance evidence is not bound to the goal",
                        source_ids=criterion.evidence_cids,
                    )
                )

    for task_index, task in enumerate(graph.tasks):
        path = f"$.tasks[{task_index}]"
        breadth = {
            "acceptance": (
                len(task.acceptance),
                policy.max_acceptance_per_task,
            ),
            "outputs": (len(task.outputs), policy.max_outputs_per_task),
            "scope_paths": (
                len(task.scope_paths),
                policy.max_scope_paths_per_task,
            ),
            "dependencies": (
                len(task.dependency_task_cids),
                policy.max_dependencies_per_task,
            ),
        }
        for kind, (observed, maximum) in breadth.items():
            if observed > maximum:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.TASK_TOO_BROAD,
                        "quality",
                        f"{path}.{kind}",
                        f"task {kind} breadth {observed} exceeds {maximum}",
                        action_id=task.task_cid,
                        counterexample={"maximum": maximum, "observed": observed},
                    )
                )
        if task.resource_class not in policy.allowed_resource_classes:
            findings.append(
                _finding(
                    PromptPlanAdmissionCode.RESOURCE_INFEASIBLE,
                    "resource",
                    f"{path}.resource_class",
                    "task resource class is not in the closed feasible set",
                    action_id=task.task_cid,
                    counterexample={
                        "allowed": list(policy.allowed_resource_classes),
                        "observed": task.resource_class,
                    },
                )
            )
        for scope_path in task.scope_paths:
            if scan_scope and not _within(scope_path, scan_scope):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.UNBOUND_PATH,
                        "output",
                        f"{path}.scope_paths",
                        "task scope is outside the pinned scan directory",
                        action_id=task.task_cid,
                        counterexample={
                            "path": scope_path,
                            "scan_scope": scan_scope,
                        },
                    )
                )
        for output_index, output in enumerate(task.outputs):
            output_path = f"{path}.outputs[{output_index}]"
            if output.effect not in policy.allowed_output_effects:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.OUTPUT_FORBIDDEN,
                        "output",
                        f"{output_path}.effect",
                        "output effect is not allowed by admission policy",
                        action_id=task.task_cid,
                        counterexample={
                            "allowed": list(policy.allowed_output_effects),
                            "observed": output.effect,
                        },
                    )
                )
            if output.media_type not in policy.allowed_media_types:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.OUTPUT_FORBIDDEN,
                        "output",
                        f"{output_path}.media_type",
                        "output media type is outside the closed output policy",
                        action_id=task.task_cid,
                        counterexample={
                            "allowed": list(policy.allowed_media_types),
                            "observed": output.media_type,
                        },
                    )
                )
            if any(
                _within(output.path, protected)
                for protected in policy.protected_paths
            ):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.OUTPUT_FORBIDDEN,
                        "output",
                        f"{output_path}.path",
                        "output targets an operator-protected path",
                        action_id=task.task_cid,
                        counterexample={"path": output.path},
                    )
                )
            if policy.allowed_path_roots and not any(
                _within(output.path, root) for root in policy.allowed_path_roots
            ):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.OUTPUT_FORBIDDEN,
                        "output",
                        f"{output_path}.path",
                        "output is outside the allowed repository paths",
                        action_id=task.task_cid,
                        counterexample={
                            "allowed_path_roots": list(policy.allowed_path_roots),
                            "path": output.path,
                        },
                    )
                )
            if scan_scope and not _within(output.path, scan_scope):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.UNBOUND_PATH,
                        "output",
                        f"{output_path}.path",
                        "output is outside the pinned scan directory",
                        action_id=task.task_cid,
                        counterexample={
                            "path": output.path,
                            "scan_scope": scan_scope,
                        },
                    )
                )
            if not any(_within(output.path, scope) for scope in task.scope_paths):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.UNBOUND_PATH,
                        "output",
                        f"{output_path}.path",
                        "output is not bound by a declared task scope",
                        action_id=task.task_cid,
                        counterexample={
                            "path": output.path,
                            "scope_paths": list(task.scope_paths),
                        },
                    )
                )
            if output.path not in task.predicted_files:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.UNBOUND_PATH,
                        "output",
                        f"{output_path}.path",
                        "output is absent from predicted_files",
                        action_id=task.task_cid,
                        counterexample={"path": output.path},
                    )
                )
            referenced_evidence = [
                evidence[cid]
                for cid in task.evidence_cids
                if cid in evidence
            ]
            bound = any(
                _overlap(output.path, evidence_path)
                for item in referenced_evidence
                for evidence_path in item.repository_paths
            )
            if policy.require_evidence_path_binding and not bound:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.EVIDENCE_UNTRACED,
                        "evidence",
                        f"{output_path}.path",
                        "output has no path-overlapping evidence trace",
                        action_id=task.task_cid,
                        source_ids=task.evidence_cids,
                        counterexample={"path": output.path},
                    )
                )
        unbound_predicted = set(task.predicted_files) - {
            output.path for output in task.outputs
        }
        if unbound_predicted:
            findings.append(
                _finding(
                    PromptPlanAdmissionCode.UNBOUND_PATH,
                    "output",
                    f"{path}.predicted_files",
                    "predicted files are not bound to declared output effects",
                    action_id=task.task_cid,
                    counterexample={
                        "unbound_paths": sorted(
                            unbound_predicted
                        )
                    },
                )
            )
        task_validation_keys = {
            validation.validation_key for validation in task.validations
        }
        for criterion_index, criterion in enumerate(task.acceptance):
            missing = set(criterion.validation_keys) - task_validation_keys
            if not criterion.validation_keys or missing:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.ACCEPTANCE_UNCOVERED,
                        "quality",
                        f"{path}.acceptance[{criterion_index}]",
                        "task acceptance must be covered by its own validation",
                        action_id=task.task_cid,
                        counterexample={
                            "criterion_key": criterion.criterion_key,
                            "missing_validation_keys": sorted(missing),
                        },
                    )
                )
            if (
                not criterion.evidence_cids
                or not set(criterion.evidence_cids).issubset(task.evidence_cids)
            ):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.EVIDENCE_UNTRACED,
                        "evidence",
                        f"{path}.acceptance[{criterion_index}]",
                        "task acceptance evidence is not bound to the task",
                        action_id=task.task_cid,
                        source_ids=criterion.evidence_cids,
                    )
                )
        for validation_index, validation in enumerate(task.validations):
            argv = validation.argv
            shell_bearing = any(
                token.casefold() in _SHELL_TOKENS
                or bool(_SHELL_META_RE.search(token))
                or "\n" in token
                or "\r" in token
                or "\x00" in token
                for token in argv
            )
            allowed_prefix = any(
                argv[: len(prefix)] == prefix
                for prefix in policy.allowed_validation_prefixes
            )
            exact = (
                not policy.exact_validation_argv
                or argv in policy.exact_validation_argv
            )
            if shell_bearing or not allowed_prefix or not exact:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.SHELL_VALIDATION,
                        "validation",
                        f"{path}.validations[{validation_index}].argv",
                        "validation is shell-bearing or outside the closed argv policy",
                        action_id=task.task_cid,
                        counterexample={
                            "argv": list(argv),
                            "allowed_prefixes": [
                                list(item)
                                for item in policy.allowed_validation_prefixes
                            ],
                        },
                    )
                )
            if validation.expected_exit_codes != (0,):
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.VALIDATION_FORBIDDEN,
                        "validation",
                        (
                            f"{path}.validations[{validation_index}]"
                            ".expected_exit_codes"
                        ),
                        "admission validation must require successful exit code zero",
                        action_id=task.task_cid,
                        counterexample={
                            "observed": list(validation.expected_exit_codes)
                        },
                    )
                )
            invalid_path_tokens = [
                token
                for token in argv
                if (
                    "\\" in token
                    or PurePosixPath(token).is_absolute()
                    or ".."
                    in PurePosixPath(token.split("::", 1)[0]).parts
                )
            ]
            if invalid_path_tokens:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.UNBOUND_PATH,
                        "validation",
                        f"{path}.validations[{validation_index}].argv",
                        "validation argv contains an absolute or escaping path",
                        action_id=task.task_cid,
                        counterexample={"tokens": invalid_path_tokens},
                    )
                )
            if validation.cwd != ".":
                if scan_scope and not _within(validation.cwd, scan_scope):
                    findings.append(
                        _finding(
                            PromptPlanAdmissionCode.UNBOUND_PATH,
                            "validation",
                            f"{path}.validations[{validation_index}].cwd",
                            "validation working directory is outside the pinned scan scope",
                            action_id=task.task_cid,
                            counterexample={
                                "cwd": validation.cwd,
                                "scan_scope": scan_scope,
                            },
                        )
                    )
                if policy.allowed_path_roots and not any(
                    _within(validation.cwd, root)
                    for root in policy.allowed_path_roots
                ):
                    findings.append(
                        _finding(
                            PromptPlanAdmissionCode.VALIDATION_FORBIDDEN,
                            "validation",
                            f"{path}.validations[{validation_index}].cwd",
                            "validation working directory is outside allowed paths",
                            action_id=task.task_cid,
                            counterexample={"cwd": validation.cwd},
                        )
                    )

    closure = _transitive_dependencies(graph)
    tasks = sorted(graph.tasks, key=lambda item: item.task_cid)
    for index, left in enumerate(tasks):
        for right in tasks[index + 1 :]:
            conflicts = sorted(
                {
                    left_output.path
                    for left_output in left.outputs
                    for right_output in right.outputs
                    if _overlap(left_output.path, right_output.path)
                }
            )
            ordered = (
                left.task_cid in closure[right.task_cid]
                or right.task_cid in closure[left.task_cid]
            )
            if conflicts and not ordered:
                findings.append(
                    _finding(
                        PromptPlanAdmissionCode.CONFLICT_UNORDERED,
                        "conflict",
                        "$.tasks",
                        "overlapping output effects are not dependency-ordered",
                        counterexample={
                            "left_action_id": left.task_cid,
                            "right_action_id": right.task_cid,
                            "paths": conflicts,
                        },
                    )
                )
    return findings


def _formal_findings(
    result: PlanCompilationResult,
) -> list[PromptPlanAdmissionFinding]:
    if result.status is CompilationStatus.COMPILED:
        return []
    return [
        _finding(
            PromptPlanAdmissionCode.FORMAL_REJECTED,
            "formal",
            issue.path,
            issue.message,
            source_ids=(issue.source_id,) if issue.source_id else (),
            counterexample={
                "code": issue.code.value,
                "field_name": issue.field_name,
                "value": issue.value,
            },
        )
        for issue in result.issues
    ] or [
        _finding(
            PromptPlanAdmissionCode.FORMAL_REJECTED,
            "formal",
            "$",
            f"formal compilation ended with {result.status.value}",
        )
    ]


def _ir_binding_findings(
    graph: PromptGoalGraph,
    formal: PlanCompilationResult,
    request: PlanAdmissionRequest,
    policy: PromptPlanAdmissionPolicy,
    workflow: PromptWorkflowRequest | None,
) -> list[PromptPlanAdmissionFinding]:
    projection = formal.admission_projection
    if projection is None:
        return []
    findings: list[PromptPlanAdmissionFinding] = []

    def mismatch(path: str, message: str, expected: Any, observed: Any) -> None:
        findings.append(
            _finding(
                PromptPlanAdmissionCode.IR_BINDING_MISMATCH,
                "ir_binding",
                path,
                message,
                counterexample={"expected": expected, "observed": observed},
            )
        )

    if request.candidate_graph_id != _projection_graph_id(projection.to_dict()):
        mismatch(
            "$.ir_request.candidate_plan",
            "IR request is not bound to the exact formal action/effect graph",
            _projection_graph_id(projection.to_dict()),
            request.candidate_graph_id,
        )
    if request.candidate_plan_id != projection.plan_id:
        mismatch(
            "$.ir_request.candidate_plan.plan_id",
            "IR candidate plan identity differs from the formal plan",
            projection.plan_id,
            request.candidate_plan_id,
        )
    if request.repository_tree_id != projection.repository_tree_id:
        mismatch(
            "$.ir_request.repository_tree_id",
            "IR repository tree differs from the formal plan tree",
            projection.repository_tree_id,
            request.repository_tree_id,
        )
    if workflow is not None:
        expected_roots = {
            "intent": workflow.intent_ir_root,
            "legal": workflow.legal_ir_root,
            "security": workflow.security_ir_root,
            "program": projection.repository_tree_id,
        }
        observed_roots = dict(request.semantic_roots)
        for kind, expected in sorted(expected_roots.items()):
            observed = observed_roots.get(kind, "")
            if observed != expected:
                mismatch(
                    f"$.ir_request.root_bindings[{kind}]",
                    f"{kind} IR root is absent or differs from the pinned workflow root",
                    expected,
                    observed,
                )
    if request.generated_formula_ids != projection.generated_formula_ids:
        mismatch(
            "$.ir_request.generated_formula_ids",
            "generated formula IDs differ from the compiler output",
            list(projection.generated_formula_ids),
            list(request.generated_formula_ids),
        )
    action_ids = set(projection.action_ids)
    binding_ids = {item.action_id for item in request.action_bindings}
    if binding_ids != action_ids:
        mismatch(
            "$.ir_request.action_bindings",
            "every exact action requires one LegalIR/SecurityIR binding",
            sorted(action_ids),
            sorted(binding_ids),
        )
    security_request_by_id = {
        item.content_id: item for item in request.security_requests
    }
    binding_by_action = {
        item.action_id: item for item in request.action_bindings
    }
    effects_by_action: dict[str, list[str]] = {
        action_id: [] for action_id in action_ids
    }
    for effect in projection.effects:
        action_id = str(effect.get("action_id") or "")
        effects_by_action.setdefault(action_id, []).append(
            json.dumps(
                _ir_effect_projection(effect),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    for action_id in sorted(action_ids):
        binding = binding_by_action.get(action_id)
        observed_effects = [
            json.dumps(
                _canonical_value(
                    security_request_by_id[request_id].expected_effect
                ),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for request_id in (
                binding.security_request_ids if binding is not None else ()
            )
            if request_id in security_request_by_id
        ]
        expected_effects = effects_by_action.get(action_id, [])
        if sorted(observed_effects) != sorted(expected_effects):
            mismatch(
                f"$.ir_request.action_bindings[{action_id}]",
                "SecurityIR requests must cover every exact formal effect once",
                sorted(expected_effects),
                sorted(observed_effects),
            )
    dependency_action_ids = {
        item.action_id for item in request.program_dependencies
    }
    if dependency_action_ids != action_ids:
        mismatch(
            "$.ir_request.program_dependencies",
            "every exact action requires a declared program dependency state",
            sorted(action_ids),
            sorted(dependency_action_ids),
        )
    dependency_by_action = {
        item.action_id: set(item.depends_on_action_ids)
        for item in request.program_dependencies
    }
    for action in projection.actions:
        action_id = str(action["action_id"])
        expected_dependencies = set(action.get("depends_on", ()))
        observed_dependencies = dependency_by_action.get(action_id, set())
        if expected_dependencies != observed_dependencies:
            mismatch(
                f"$.ir_request.program_dependencies[{action_id}]",
                "program dependency edges differ from the formal topology",
                sorted(expected_dependencies),
                sorted(observed_dependencies),
            )
    declared_assumption_ids = {
        item
        for action in projection.actions
        for item in action.get("assumption_ids", ())
    }
    request_assumption_ids = {
        item.assumption_id for item in request.assumptions if item.required
    }
    if declared_assumption_ids != request_assumption_ids:
        mismatch(
            "$.ir_request.assumptions",
            "IR mandatory assumptions differ from formal action preconditions",
            sorted(declared_assumption_ids),
            sorted(request_assumption_ids),
        )
    if policy.require_mandatory_closure and request.mandatory_closure is None:
        findings.append(
            _finding(
                PromptPlanAdmissionCode.UNKNOWN_MANDATORY_STATE,
                "ir_binding",
                "$.ir_request.mandatory_closure",
                "mandatory semantic closure is unknown",
            )
        )

    validation_by_action: dict[str, set[str]] = {
        action_id: set() for action_id in action_ids
    }
    for requirement in request.validation_requirements:
        for action_id in requirement.action_ids:
            if action_id in validation_by_action and requirement.required:
                validation_by_action[action_id].add(requirement.requirement_id)
    declared_validation_ids = {
        item
        for action in projection.actions
        for item in action.get("validation_requirement_ids", ())
    }
    request_validation_ids = {
        item.requirement_id
        for item in request.validation_requirements
        if item.required
    }
    if declared_validation_ids != request_validation_ids:
        mismatch(
            "$.ir_request.validation_requirements",
            "IR validation requirements differ from formal action requirements",
            sorted(declared_validation_ids),
            sorted(request_validation_ids),
        )
    validation_results_ids = {
        item.requirement_id for item in request.validation_results
    }
    if validation_results_ids != request_validation_ids:
        mismatch(
            "$.ir_request.validation_results",
            "validation results must cover exactly the required validations",
            sorted(request_validation_ids),
            sorted(validation_results_ids),
        )
    request_requirements = {
        item.requirement_id: item
        for item in request.validation_requirements
        if item.required
    }
    formal_requirements = (
        {
            item.requirement_id: item
            for item in formal.plan.evidence_requirements
        }
        if formal.plan is not None
        else {}
    )
    for action in projection.actions:
        action_id = str(action["action_id"])
        expected_ids = set(action.get("validation_requirement_ids", ()))
        observed_ids = {
            requirement.requirement_id
            for requirement in request.validation_requirements
            if requirement.required and action_id in requirement.action_ids
        }
        if expected_ids != observed_ids:
            mismatch(
                f"$.ir_request.validation_requirements[{action_id}]",
                "per-action validation bindings differ from the formal plan",
                sorted(expected_ids),
                sorted(observed_ids),
            )
        for requirement_id in sorted(expected_ids & set(request_requirements)):
            request_requirement = request_requirements[requirement_id]
            formal_requirement = formal_requirements.get(requirement_id)
            allowed_commands = (
                set(formal_requirement.fallback_check_ids)
                if formal_requirement is not None
                else set()
            )
            if (
                not request_requirement.command
                or request_requirement.command not in allowed_commands
            ):
                mismatch(
                    (
                        "$.ir_request.validation_requirements"
                        f"[{requirement_id}].command"
                    ),
                    "validation command is not one of the compiler-bound structured checks",
                    sorted(allowed_commands),
                    request_requirement.command,
                )
    for action_id, requirement_ids in sorted(validation_by_action.items()):
        if not requirement_ids:
            findings.append(
                _finding(
                    PromptPlanAdmissionCode.MISSING_VALIDATION,
                    "validation",
                    "$.ir_request.validation_requirements",
                    "every exact action/effect requires validation",
                    action_id=action_id,
                )
            )

    declared_proof_ids = {
        item
        for action in projection.actions
        for item in action.get("proof_obligation_ids", ())
    }
    proof_by_id: dict[str, list[Any]] = {}
    for proof in request.proof_results:
        proof_by_id.setdefault(proof.obligation_id, []).append(proof)
    external_proof_ids = {
        item.obligation_id
        for item in request.intent_request.constraint_set.proof_obligations
    } | {
        item.obligation_id
        for legal in request.legal_results
        for item in legal.proof_obligations
        if item.required
    }
    expected_proof_ids = declared_proof_ids | external_proof_ids
    if set(proof_by_id) != expected_proof_ids:
        mismatch(
            "$.ir_request.proof_results",
            "proof receipts must cover exactly the formal, IntentIR, and LegalIR obligations",
            sorted(expected_proof_ids),
            sorted(proof_by_id),
        )
    missing_proofs = declared_proof_ids - set(proof_by_id)
    for obligation_id in sorted(missing_proofs):
        findings.append(
            _finding(
                PromptPlanAdmissionCode.MISSING_PROOF,
                "proof",
                "$.ir_request.proof_results",
                "formal proof/evidence obligation has no independent proof receipt",
                source_ids=(obligation_id,),
                counterexample={"obligation_id": obligation_id},
            )
        )
    for obligation_id in sorted(declared_proof_ids & set(proof_by_id)):
        proofs = proof_by_id[obligation_id]
        valid = [
            proof
            for proof in proofs
            if (
                proof.plan_id == projection.plan_id
                and proof.repository_tree_id == projection.repository_tree_id
                and proof.freshness is EvidenceFreshness.CURRENT
                and proof.authoritative_verdict is ProofVerdict.PROVED
                and proof.satisfies(AssuranceLevel.KERNEL_VERIFIED)
            )
        ]
        if not valid:
            findings.append(
                _finding(
                    PromptPlanAdmissionCode.MISSING_PROOF,
                    "proof",
                    "$.ir_request.proof_results",
                    "formal proof/evidence receipt is stale, mismatched, or not kernel verified",
                    source_ids=(
                        obligation_id,
                        *(proof.receipt_id for proof in proofs),
                    ),
                    counterexample={
                        "expected_plan_id": projection.plan_id,
                        "expected_repository_tree_id": projection.repository_tree_id,
                        "observed_plan_ids": sorted(
                            {proof.plan_id for proof in proofs}
                        ),
                        "observed_repository_tree_ids": sorted(
                            {proof.repository_tree_id for proof in proofs}
                        ),
                    },
                )
            )

    declared_output_effects = {
        (
            task.task_cid,
            output.path,
            output.effect,
        )
        for task in graph.tasks
        for output in task.outputs
    }
    formal_output_effects = set()
    for effect in projection.effects:
        metadata = effect.get("metadata")
        source_effect = (
            metadata.get("source_effect")
            if isinstance(metadata, Mapping)
            else None
        )
        if isinstance(source_effect, Mapping):
            formal_output_effects.add(
                (
                    str(effect.get("action_id") or ""),
                    str(source_effect.get("path") or ""),
                    str(effect.get("value") or ""),
                )
            )
    if declared_output_effects != formal_output_effects:
        findings.append(
            _finding(
                PromptPlanAdmissionCode.UNDECLARED_EFFECT,
                "program",
                "$.formal.effects",
                "declared prompt outputs and formal program effects differ",
                counterexample={
                    "declared": sorted(declared_output_effects),
                    "formal": sorted(formal_output_effects),
                },
            )
        )
    return findings


def _projection_graph_id(candidate: Mapping[str, Any]) -> str:
    """Use the authoritative IR normalizer without duplicating its semantics."""

    # Constructing a full request solely for an ID would manufacture semantic
    # state.  These private helpers are the canonical graph identity functions
    # used by PlanAdmissionRequest itself.
    from ..proof.ir_constraint_compiler import _graph_id

    return _graph_id(candidate)


def _ir_effect_projection(effect: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _canonical_value(value)
        for key, value in effect.items()
        if key not in {"effect_id", "action_id", "task_id", "metadata"}
    }


def _ir_findings(receipt: PlanAdmissionReceipt) -> list[PromptPlanAdmissionFinding]:
    findings = []
    for rejection in receipt.rejection_reasons:
        findings.append(
            _finding(
                f"ir.{rejection.code.value}",
                f"ir.{rejection.domain.value}",
                "$.ir_request",
                rejection.message,
                action_id=rejection.action_id,
                effect_id=rejection.effect_id,
                source_ids=rejection.source_ids,
                counterexample={
                    "dependency_id": rejection.dependency_id,
                    "obligation_id": rejection.obligation_id,
                    "details": dict(rejection.details),
                },
            )
        )
    if not receipt.admitted and not findings:
        findings.append(
            _finding(
                PromptPlanAdmissionCode.IR_REJECTED,
                "ir",
                "$.ir_request",
                "IR constraint compiler rejected without a domain finding",
            )
        )
    return findings


def admit_prompt_plan(
    request_or_graph: PromptPlanAdmissionRequest | PromptGoalGraph | Mapping[str, Any],
    *,
    repository_tree_id: str = "",
    ir_request: PlanAdmissionRequest | None = None,
    workflow_request: PromptWorkflowRequest | None = None,
    scan_receipt: DirectoryScanReceipt | None = None,
    policy: PromptPlanAdmissionPolicy | None = None,
    compiler: FormalPlanCompiler | None = None,
    irrelevant_corpus: Sequence[Any] = (),
) -> PromptPlanAdmissionResult:
    """Admit a canonical prompt graph through quality, formal, and IR gates.

    ``irrelevant_corpus`` is accepted only as an explicit non-input witness for
    determinism tests.  Admission never reads or hashes it; relevant evidence
    must already be present in the canonical graph.
    """

    selected_policy = policy or PromptPlanAdmissionPolicy()
    selected_compiler = compiler or FormalPlanCompiler()
    if isinstance(irrelevant_corpus, (str, bytes)) or not isinstance(
        irrelevant_corpus, Sequence
    ):
        raise TypeError("irrelevant_corpus must be a sequence")
    if isinstance(request_or_graph, PromptPlanAdmissionRequest):
        request = request_or_graph
        graph: PromptGoalGraph | None = request.graph
        tree_id = request.repository_tree_id
        selected_ir = request.ir_request
        workflow = request.workflow_request
        scan = request.scan_receipt
        selected_policy = request.policy
    else:
        tree_id = str(repository_tree_id or "").strip()
        selected_ir = ir_request
        workflow = workflow_request
        scan = scan_receipt
        try:
            graph = (
                request_or_graph
                if isinstance(request_or_graph, PromptGoalGraph)
                else PromptGoalGraph.from_dict(request_or_graph)
            )
        except (TypeError, ValueError) as exc:
            graph = None
            message = str(exc)
            if "cycle" in message:
                code = PromptPlanAdmissionCode.CYCLIC_GRAPH
            elif "disconnected" in message or "unknown goal" in message:
                code = PromptPlanAdmissionCode.DISCONNECTED_GRAPH
            else:
                code = PromptPlanAdmissionCode.MALFORMED_GRAPH
            candidate_id = content_identity(
                {
                    "namespace": "malformed-prompt-plan",
                    "candidate": _safe_counterexample(request_or_graph),
                }
            )
            finding = _finding(
                code,
                "graph",
                "$",
                str(exc),
                counterexample={"exception": type(exc).__name__},
            )
            receipt = PromptPlanAdmissionReceipt(
                candidate_plan_cid=candidate_id,
                repository_tree_id=tree_id,
                policy_id=selected_policy.policy_id,
                verdict=PromptPlanAdmissionVerdict.REJECTED,
                findings=(finding,),
                invariants={"canonical_graph": False},
            )
            return PromptPlanAdmissionResult(receipt=receipt)
    if graph is None:
        raise AssertionError("graph resolution failed without a receipt")

    findings: list[PromptPlanAdmissionFinding] = []
    if not tree_id:
        findings.append(
            _finding(
                PromptPlanAdmissionCode.STALE_ROOT,
                "root",
                "$.repository_tree_id",
                "repository tree root is missing",
            )
        )
    topology: tuple[str, ...] = ()
    try:
        topology = _stable_topology(graph)
    except ValueError as exc:
        findings.append(
            _finding(
                PromptPlanAdmissionCode.CYCLIC_GRAPH,
                "graph",
                "$.tasks",
                str(exc),
            )
        )
    topology_id = content_identity(
        {
            "namespace": "prompt-plan-topology",
            "task_cids": list(topology),
            "edges": sorted(
                (task.task_cid, dependency)
                for task in graph.tasks
                for dependency in task.dependency_task_cids
            ),
        }
    )
    findings.extend(
        _graph_findings(
            graph,
            tree_id=tree_id,
            workflow=workflow,
            scan=scan,
            policy=selected_policy,
        )
    )

    formal = selected_compiler.compile_prompt_graph(
        graph,
        repository_tree_id=tree_id or "missing:repository-tree",
    )
    findings.extend(_formal_findings(formal))

    ir_receipt: PlanAdmissionReceipt | None = None
    if formal.status is CompilationStatus.COMPILED:
        if selected_ir is None:
            # Factory absence (or any missing independent request) is an
            # unknown mandatory IR state, not a binding mismatch.  Binding
            # mismatch is reserved for a present request that disagrees with
            # the formal projection.  PlanAdmissionService constructs its own
            # exact PlanAdmissionRequest when domain materials are supplied
            # through the independent admission path.
            findings.append(
                _finding(
                    PromptPlanAdmissionCode.UNKNOWN_MANDATORY_STATE,
                    "ir_binding",
                    "$.ir_request",
                    (
                        "independent PlanAdmissionRequest materials are required "
                        "for hard-domain admission; absence of an "
                        "admission_request_factory is not an IR binding mismatch"
                    ),
                )
            )
        else:
            findings.extend(
                _ir_binding_findings(
                    graph,
                    formal,
                    selected_ir,
                    selected_policy,
                    workflow,
                )
            )
            # Run the hard-domain compiler even when an adapter-level mismatch
            # exists, retaining its independent rejection evidence.
            ir_receipt = selected_compiler.compile_admission(selected_ir)
            findings.extend(_ir_findings(ir_receipt))

    findings = list(
        {
            item.finding_id: item for item in findings
        }.values()
    )
    findings.sort(key=lambda item: item.finding_id)
    admitted = (
        not findings
        and formal.status is CompilationStatus.COMPILED
        and ir_receipt is not None
        and ir_receipt.admitted
    )
    task_cids = tuple(sorted(task.task_cid for task in graph.tasks))
    final_plan_cid = ""
    final_task_cids: tuple[str, ...] = ()
    if admitted:
        final_task_cids = task_cids
        final_plan_cid = prompt_workflow_cid(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/admitted-prompt-plan@1",
                "candidate_plan_cid": graph.plan_root_cid,
                "formal_plan_id": formal.plan_id,
                "ir_receipt_id": ir_receipt.receipt_id,
                "policy_id": selected_policy.policy_id,
                "repository_tree_id": tree_id,
                "task_cids": list(task_cids),
                "topology_id": topology_id,
            }
        )

    invariants = {
        "acceptance_coverage": not any(
            item.code == PromptPlanAdmissionCode.ACCEPTANCE_UNCOVERED.value
            for item in findings
        ),
        "acyclic": bool(topology),
        "canonical_graph": True,
        "conflict_resource_feasible": not any(
            item.code
            in {
                PromptPlanAdmissionCode.CONFLICT_UNORDERED.value,
                PromptPlanAdmissionCode.RESOURCE_INFEASIBLE.value,
            }
            for item in findings
        ),
        "connected": not any(
            item.code == PromptPlanAdmissionCode.DISCONNECTED_GRAPH.value
            for item in findings
        ),
        "evidence_traceable": not any(
            item.code == PromptPlanAdmissionCode.EVIDENCE_UNTRACED.value
            for item in findings
        ),
        "formal_compiled": formal.status is CompilationStatus.COMPILED,
        "effects_declared": not any(
            item.code
            in {
                PromptPlanAdmissionCode.UNDECLARED_EFFECT.value,
                "ir.undeclared_effect",
            }
            for item in findings
        ),
        "hard_domains_admitted": bool(ir_receipt and ir_receipt.admitted)
        and not any(
            item.code == PromptPlanAdmissionCode.IR_BINDING_MISMATCH.value
            for item in findings
        ),
        "ir_admitted": bool(ir_receipt and ir_receipt.admitted),
        "output_validation_policy": not any(
            item.code
            in {
                PromptPlanAdmissionCode.OUTPUT_FORBIDDEN.value,
                PromptPlanAdmissionCode.UNBOUND_PATH.value,
                PromptPlanAdmissionCode.SHELL_VALIDATION.value,
                PromptPlanAdmissionCode.VALIDATION_FORBIDDEN.value,
            }
            for item in findings
        ),
        "proof_complete": not any(
            item.code
            in {
                PromptPlanAdmissionCode.MISSING_PROOF.value,
                "ir.missing_proof",
                "ir.invalid_proof",
            }
            for item in findings
        ),
        "roots_current": not any(
            item.code
            in {
                PromptPlanAdmissionCode.STALE_ROOT.value,
                PromptPlanAdmissionCode.UNKNOWN_MANDATORY_STATE.value,
                "ir.stale_root",
            }
            for item in findings
        ),
        "stable_topology": bool(topology),
        "task_granularity": not any(
            item.code == PromptPlanAdmissionCode.TASK_TOO_BROAD.value
            for item in findings
        ),
        "validation_complete": not any(
            item.code
            in {
                PromptPlanAdmissionCode.MISSING_VALIDATION.value,
                PromptPlanAdmissionCode.VALIDATION_FORBIDDEN.value,
                "ir.validation_missing",
                "ir.validation_failed",
            }
            for item in findings
        ),
    }
    receipt = PromptPlanAdmissionReceipt(
        candidate_plan_cid=graph.plan_root_cid,
        repository_tree_id=tree_id,
        policy_id=selected_policy.policy_id,
        verdict=(
            PromptPlanAdmissionVerdict.ADMITTED
            if admitted
            else PromptPlanAdmissionVerdict.REJECTED
        ),
        candidate_task_cids=task_cids,
        topological_task_cids=topology,
        topology_id=topology_id,
        formal_plan_id=formal.plan_id,
        formal_source_identity=formal.source_identity,
        ir_request_id=selected_ir.request_id if selected_ir is not None else "",
        ir_receipt_id=ir_receipt.receipt_id if ir_receipt is not None else "",
        final_plan_cid=final_plan_cid,
        final_task_cids=final_task_cids,
        findings=tuple(findings),
        invariants=invariants,
    )
    return PromptPlanAdmissionResult(
        receipt=receipt,
        formal_compilation=formal,
        ir_receipt=ir_receipt,
        admitted_graph=graph if admitted else None,
    )


def _safe_counterexample(value: Any) -> Any:
    try:
        return _canonical_value(value)
    except (TypeError, ValueError):
        return {"type": type(value).__name__}


compile_prompt_plan_admission = admit_prompt_plan
admit_prompt_generated_plan = admit_prompt_plan
PromptPlanAdmission = PromptPlanAdmissionReceipt


__all__ = [
    "PROMPT_PLAN_ADMISSION_POLICY_SCHEMA",
    "PROMPT_PLAN_ADMISSION_RECEIPT_SCHEMA",
    "PROMPT_PLAN_ADMISSION_VERSION",
    "PromptPlanAdmission",
    "PromptPlanAdmissionCode",
    "PromptPlanAdmissionFinding",
    "PromptPlanAdmissionPolicy",
    "PromptPlanAdmissionReceipt",
    "PromptPlanAdmissionRequest",
    "PromptPlanAdmissionResult",
    "PromptPlanAdmissionVerdict",
    "admit_prompt_generated_plan",
    "admit_prompt_plan",
    "compile_prompt_plan_admission",
]
