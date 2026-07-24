"""Objective-graph scanner and bundle planner for autonomous agent todos.

This module ports the objective-driven backlog generation that was previously
implemented as repository scripts into ``ipfs_accelerate_py``.  It is designed
for acceleration-oriented agent systems: one objective heap can generate many
bundle-local todo shards, and those shards can be submitted to the existing P2P
task queue so multiple Codex workers can drain independent lanes.
"""

from __future__ import annotations

import ast
import heapq
import json
import math
import os
import re
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha1, sha256
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset_store import DatasetArtifact, ObjectiveDatasetStore
from .scan_receipts import (
    RefillScanResult,
    ScanTerminalReason,
    build_scan_result,
    canonical_revision,
    scan_identity,
)
from .task_identity import TaskIdentity, canonical_bundle_identity, canonical_task_identity
from .taskboard_store import (
    locked_taskboard,
    replace_locked_taskboard,
    task_ids_from_artifact_names,
)


DEFAULT_EMBEDDING_DIMENSIONS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_EMBEDDING_DIMENSIONS", "64"))
DEFAULT_EMBEDDING_MIN_SCORE = float(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_EMBEDDING_MIN_SCORE", "0.62"))
DEFAULT_BUNDLE_CLUSTER_MIN_SCORE = float(os.environ.get("IPFS_ACCELERATE_AGENT_BUNDLE_CLUSTER_MIN_SCORE", "0.42"))
DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX = os.environ.get(
    "IPFS_ACCELERATE_AGENT_OBJECTIVE_TASK_SUMMARY_PREFIX",
    "Close objective gap",
)


def parse_python_ast_quietly(text: str) -> ast.AST:
    """Parse Python source without surfacing scanner-only syntax warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        return ast.parse(text)


DEFAULT_DISCOVERY_OUTPUT_PATH = os.environ.get(
    "IPFS_ACCELERATE_AGENT_DISCOVERY_OUTPUT_PATH",
    "data/agent_supervisor/discovery",
)
DEFAULT_SURPLUS_FINDINGS_PER_GOAL = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", "3")
)
DEFAULT_SURPLUS_MIN_TERMS_PER_TODO = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", "3")
)
DEFAULT_SCAN_OVERSAMPLE_MULTIPLIER = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_OVERSAMPLE_MULTIPLIER", "2")
)
DEFAULT_TASK_PREFIX = "AUTO-"
OBJECTIVE_SCAN_ANALYZER_VERSION = "objective-gap-analyzer/v1"
DEFAULT_AST_DATASET_MAX_CHARS = int(os.environ.get("IPFS_ACCELERATE_AGENT_AST_DATASET_MAX_CHARS", "1000000"))
AST_DATASET_RECORD_SCHEMA_VERSION = 2
LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND = (
    "(test ! -f swissknife/package.json || npm --prefix swissknife run test:e2e:meta-glasses) && "
    "(test ! -f hallucinate_app/package.json || "
    "npm --prefix hallucinate_app run test:e2e -- multimodal-control-surface.spec.ts)"
)
LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS = (
    "test:e2e:meta-glasses",
    "meta-glasses-virtual-os.spec.ts",
    "multimodal-control-surface.spec.ts",
)
LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE = "launch Playwright validation gate"
SCAN_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "playwright-report",
    "test-results",
}

OPAQUE_EVIDENCE_REQUIREMENT_PATTERN = re.compile(r"^[0-9]{20,}$")
EVIDENCE_SOURCE_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/evidence-source-policy@1"
)


class EvidenceRequirementKind(str, Enum):
    """Authority class of an objective evidence requirement."""

    OPAQUE_RECEIPT = "opaque_receipt"
    PATH = "path"
    CODE = "code"
    TEST = "test"
    PROOF = "proof"
    BENCHMARK = "benchmark"
    RUNTIME = "runtime"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class EvidenceSourceTier(str, Enum):
    """Trust tier of a discovered evidence source."""

    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    PROOF = "proof"
    BENCHMARK = "benchmark"
    RUNTIME = "runtime"
    RECEIPT = "receipt"
    DOCUMENTATION = "documentation"
    PROPOSAL = "proposal"
    UNKNOWN = "unknown"


class EvidenceMatchKind(str, Enum):
    """How a source was associated with a requirement."""

    PATH = "path"
    EXACT_TEXT = "exact_text"
    EXACT_AST = "exact_ast"
    SEMANTIC = "semantic"
    RETRIEVAL = "retrieval"
    TYPED_RECEIPT = "typed_receipt"


@dataclass(frozen=True)
class EvidenceSourceDecision:
    """Fail-closed decision separating discovery from completion authority."""

    requirement: str
    requirement_kind: EvidenceRequirementKind
    source_tier: EvidenceSourceTier
    match_kind: EvidenceMatchKind
    source_path: str = ""
    reference: str = ""
    satisfies: bool = False
    nominated: bool = True
    reason_codes: tuple[str, ...] = ()

    @property
    def qualifies(self) -> bool:
        return self.satisfies

    @property
    def nomination_only(self) -> bool:
        return self.nominated and not self.satisfies

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SOURCE_POLICY_SCHEMA,
            "requirement": self.requirement,
            "requirement_kind": self.requirement_kind.value,
            "source_tier": self.source_tier.value,
            "match_kind": self.match_kind.value,
            "source_path": self.source_path,
            "reference": self.reference,
            "satisfies": self.satisfies,
            "qualifies": self.satisfies,
            "nominated": self.nominated,
            "nomination_only": self.nomination_only,
            "reason_codes": list(self.reason_codes),
        }


def _evidence_enum(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)).strip().lower())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _receipt_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(
            dict.fromkeys(str(item).strip() for item in value if str(item).strip())
        )
    return ()


@dataclass(frozen=True)
class EvidenceSourcePolicy:
    """Policy for objective discovery and completion-evidence admission.

    Text and semantic matches remain useful nominations.  They are never
    promoted into opaque receipt authority, and proposal-tier prose never
    satisfies implementation, validation, proof, benchmark, or runtime
    requirements.
    """

    policy_id: str = "objective-evidence-source-policy@1"
    max_nominations_per_requirement: int = 8
    max_nomination_bytes: int = 16_384

    def __post_init__(self) -> None:
        if not str(self.policy_id).strip():
            raise ValueError("policy_id is required")
        for name in ("max_nominations_per_requirement", "max_nomination_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def requirement_kind(
        requirement: str,
        declared: EvidenceRequirementKind | str | None = None,
    ) -> EvidenceRequirementKind:
        if declared is not None:
            return _evidence_enum(
                declared, EvidenceRequirementKind, "requirement kind"
            )
        normalized = str(requirement or "").strip()
        if OPAQUE_EVIDENCE_REQUIREMENT_PATTERN.fullmatch(normalized):
            return EvidenceRequirementKind.OPAQUE_RECEIPT
        if repo_relative_path_safe(normalized):
            suffix = Path(normalized).suffix.lower()
            if suffix or "/" in normalized:
                return EvidenceRequirementKind.PATH
        words = set(re.findall(r"[a-z0-9_]+", normalized.casefold()))
        if words & {"benchmark", "benchmarks", "performance", "throughput"}:
            return EvidenceRequirementKind.BENCHMARK
        if words & {"proof", "proofs", "theorem", "formal"}:
            return EvidenceRequirementKind.PROOF
        if words & {"runtime", "telemetry", "production", "operational"}:
            return EvidenceRequirementKind.RUNTIME
        if words & {"test", "tests", "testing", "validation"}:
            return EvidenceRequirementKind.TEST
        if words & {"code", "implementation", "source", "symbol", "ast"}:
            return EvidenceRequirementKind.CODE
        return EvidenceRequirementKind.OTHER

    @staticmethod
    def classify_path(path: str | os.PathLike[str] | None) -> EvidenceSourceTier:
        normalized = str(path or "").strip().replace("\\", "/").lower()
        if not normalized:
            return EvidenceSourceTier.UNKNOWN
        parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
        name = parts[-1] if parts else ""
        proposal_markers = {
            "discovery",
            "generated-discovery",
            "generated_discovery",
            "goal-packet",
            "objective_bundles",
            "objective-bundles",
            "execution_packet",
            "execution_packets",
            "execution-packet",
            "execution-packets",
            "goal_packet",
            "goal_packets",
            "planning",
            "plans",
            "proposals",
            "task-board",
            "task-boards",
            "task_board",
            "task_boards",
            "taskboard",
            "taskboards",
        }
        if (
            any(part in proposal_markers for part in parts)
            or ".todo." in name
            or name.endswith(".todo")
            or "-objective-gap-" in name
            or "objective" in name
            or name.endswith(("_plan.md", "-plan.md", ".plan.md"))
            or name
            in {
                "plan.md",
                "plan.json",
                "todo.md",
                "todo.json",
                "taskboard.md",
                "taskboard.json",
                "task-board.md",
                "task-board.json",
            }
        ):
            return EvidenceSourceTier.PROPOSAL
        if name.endswith(
            (".receipt.json", ".receipt.jsonl", ".receipt.cbor", ".receipt")
        ):
            return EvidenceSourceTier.RECEIPT
        if (
            "test" in parts
            or "tests" in parts
            or name.startswith("test_")
            or name.endswith((".spec.ts", ".spec.js", ".test.ts", ".test.js"))
        ):
            return EvidenceSourceTier.VALIDATION
        if "benchmarks" in parts or "benchmark" in name or "bench" in parts:
            return EvidenceSourceTier.BENCHMARK
        if "proof" in parts or "proofs" in parts or name.endswith((".lean", ".smt2")):
            return EvidenceSourceTier.PROOF
        if "runtime" in parts or "telemetry" in parts:
            return EvidenceSourceTier.RUNTIME
        if name.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".c", ".h", ".sh")):
            return EvidenceSourceTier.IMPLEMENTATION
        if name.endswith((".md", ".rst", ".txt")) or "docs" in parts:
            return EvidenceSourceTier.DOCUMENTATION
        return EvidenceSourceTier.UNKNOWN

    @staticmethod
    def _receipt_tier(receipt: Mapping[str, Any]) -> EvidenceSourceTier:
        raw = str(
            receipt.get("source_tier")
            or receipt.get("receipt_kind")
            or receipt.get("producer_kind")
            or receipt.get("kind")
            or ""
        ).strip().lower()
        aliases = {
            "task": EvidenceSourceTier.VALIDATION,
            "scan": EvidenceSourceTier.VALIDATION,
            "code": EvidenceSourceTier.IMPLEMENTATION,
            "implementation": EvidenceSourceTier.IMPLEMENTATION,
            "test": EvidenceSourceTier.VALIDATION,
            "validation": EvidenceSourceTier.VALIDATION,
            "proof": EvidenceSourceTier.PROOF,
            "benchmark": EvidenceSourceTier.BENCHMARK,
            "runtime": EvidenceSourceTier.RUNTIME,
            "receipt": EvidenceSourceTier.RECEIPT,
            "proposal": EvidenceSourceTier.PROPOSAL,
            "objective": EvidenceSourceTier.PROPOSAL,
            "plan": EvidenceSourceTier.PROPOSAL,
            "task_board": EvidenceSourceTier.PROPOSAL,
            "generated_discovery": EvidenceSourceTier.PROPOSAL,
        }
        if raw in aliases:
            return aliases[raw]
        if raw:
            return EvidenceSourceTier.UNKNOWN
        schema = str(
            receipt.get("schema") or receipt.get("schema_version") or ""
        ).casefold()
        schema_tiers = (
            (("benchmark", "bench"), EvidenceSourceTier.BENCHMARK),
            (("runtime", "telemetry"), EvidenceSourceTier.RUNTIME),
            (("proof", "formal"), EvidenceSourceTier.PROOF),
            (("validation", "test"), EvidenceSourceTier.VALIDATION),
            (("implementation", "code", "ast"), EvidenceSourceTier.IMPLEMENTATION),
            (("receipt",), EvidenceSourceTier.RECEIPT),
        )
        for markers, tier in schema_tiers:
            if any(marker in schema for marker in markers):
                return tier
        return EvidenceSourceTier.UNKNOWN

    @staticmethod
    def _receipt_tier_allowed(
        requirement_kind: EvidenceRequirementKind,
        source_tier: EvidenceSourceTier,
    ) -> bool:
        """Return whether a typed receipt's producer can prove this claim kind."""

        allowed = {
            EvidenceRequirementKind.CODE: {
                EvidenceSourceTier.IMPLEMENTATION,
                EvidenceSourceTier.VALIDATION,
                EvidenceSourceTier.RECEIPT,
            },
            EvidenceRequirementKind.TEST: {
                EvidenceSourceTier.VALIDATION,
                EvidenceSourceTier.RECEIPT,
            },
            EvidenceRequirementKind.PROOF: {
                EvidenceSourceTier.PROOF,
                EvidenceSourceTier.VALIDATION,
                EvidenceSourceTier.RECEIPT,
            },
            EvidenceRequirementKind.BENCHMARK: {
                EvidenceSourceTier.BENCHMARK,
                EvidenceSourceTier.VALIDATION,
                EvidenceSourceTier.RECEIPT,
            },
            EvidenceRequirementKind.RUNTIME: {
                EvidenceSourceTier.RUNTIME,
                EvidenceSourceTier.VALIDATION,
                EvidenceSourceTier.RECEIPT,
            },
            EvidenceRequirementKind.OPAQUE_RECEIPT: {
                EvidenceSourceTier.IMPLEMENTATION,
                EvidenceSourceTier.VALIDATION,
                EvidenceSourceTier.PROOF,
                EvidenceSourceTier.BENCHMARK,
                EvidenceSourceTier.RUNTIME,
                EvidenceSourceTier.RECEIPT,
            },
        }
        expected = allowed.get(requirement_kind)
        if expected is None:
            return source_tier not in {
                EvidenceSourceTier.PROPOSAL,
                EvidenceSourceTier.UNKNOWN,
            }
        return source_tier in expected

    @staticmethod
    def _direct_source_tier_allowed(
        requirement_kind: EvidenceRequirementKind,
        source_tier: EvidenceSourceTier,
    ) -> bool:
        allowed = {
            EvidenceRequirementKind.CODE: {
                EvidenceSourceTier.IMPLEMENTATION,
                EvidenceSourceTier.VALIDATION,
            },
            EvidenceRequirementKind.TEST: {EvidenceSourceTier.VALIDATION},
            EvidenceRequirementKind.PROOF: {EvidenceSourceTier.PROOF},
            EvidenceRequirementKind.BENCHMARK: {
                EvidenceSourceTier.BENCHMARK
            },
            EvidenceRequirementKind.RUNTIME: {EvidenceSourceTier.RUNTIME},
        }
        expected = allowed.get(requirement_kind)
        return expected is not None and source_tier in expected

    @staticmethod
    def _receipt_requirement_ids(receipt: Mapping[str, Any]) -> tuple[str, ...]:
        ids: list[str] = []
        for name in (
            "requirement_id",
            "requirement_ids",
            "evidence_claim_references",
            "authoritative_evidence_claim_references",
            "completion_evidence_receipt_ids",
        ):
            ids.extend(_receipt_strings(receipt.get(name)))
        return tuple(dict.fromkeys(ids))

    @staticmethod
    def _receipt_terminal_reasons(receipt: Mapping[str, Any]) -> tuple[str, ...]:
        reasons: list[str] = []
        status = str(
            receipt.get("status")
            or receipt.get("outcome")
            or receipt.get("terminal_reason")
            or ""
        ).strip().lower().replace("-", "_")
        bad = {
            "failed",
            "failure",
            "error",
            "partial",
            "timed_out",
            "timeout",
            "cancelled",
            "canceled",
            "stale",
            "expired",
            "inconclusive",
            "unsupported",
            "skipped",
        }
        if not status:
            reasons.append("receipt_terminal_status_missing")
        elif status in bad:
            reasons.append(f"receipt_terminal_status_{status}")
        elif status not in {
            "passed",
            "pass",
            "success",
            "successful",
            "succeeded",
            "completed",
            "complete",
            "conclusive",
            "satisfied",
            "verified",
            "fresh",
            "current",
        }:
            reasons.append("receipt_terminal_status_unsupported")
        freshness = str(receipt.get("freshness") or "").strip().lower()
        if freshness in {"stale", "expired", "unknown"}:
            reasons.append("receipt_stale")
        if receipt.get("validation_passed") is False or receipt.get("passed") is False:
            reasons.append("receipt_failed")
        if receipt.get("safe_for_completion_reasoning") is False:
            reasons.append("receipt_not_completion_safe")
        if receipt.get("complete") is False or receipt.get("coverage_complete") is False:
            reasons.append("receipt_partial")
        if receipt.get("truncated") is True:
            reasons.append("receipt_truncated")
        return tuple(dict.fromkeys(reasons))

    def evaluate(
        self,
        requirement: str,
        *,
        match_kind: EvidenceMatchKind | str,
        source_path: str = "",
        source_tier: EvidenceSourceTier | str | None = None,
        requirement_kind: EvidenceRequirementKind | str | None = None,
        typed_receipt: Mapping[str, Any] | Any | None = None,
        repository_tree: str = "",
        policy_id: str = "",
        reference: str = "",
    ) -> EvidenceSourceDecision:
        normalized = " ".join(str(requirement or "").strip().split())
        if not normalized:
            raise ValueError("requirement is required")
        kind = self.requirement_kind(normalized, requirement_kind)
        match = _evidence_enum(match_kind, EvidenceMatchKind, "match kind")
        tier = (
            _evidence_enum(source_tier, EvidenceSourceTier, "source tier")
            if source_tier is not None
            else self.classify_path(source_path)
        )
        receipt: Mapping[str, Any] | None = None
        if typed_receipt is not None:
            if isinstance(typed_receipt, Mapping):
                receipt = typed_receipt
            else:
                converter = getattr(typed_receipt, "to_dict", None)
                projected = converter() if callable(converter) else None
                if isinstance(projected, Mapping):
                    receipt = projected
            if receipt is None:
                return EvidenceSourceDecision(
                    normalized, kind, tier, match, source_path, reference,
                    reason_codes=("typed_receipt_required",),
                )
            tier = self._receipt_tier(receipt)
            if self.classify_path(source_path) is EvidenceSourceTier.PROPOSAL:
                tier = EvidenceSourceTier.PROPOSAL

        reasons: list[str] = []
        satisfies = False
        authoritative_kinds = {
            EvidenceRequirementKind.CODE,
            EvidenceRequirementKind.TEST,
            EvidenceRequirementKind.PROOF,
            EvidenceRequirementKind.BENCHMARK,
            EvidenceRequirementKind.RUNTIME,
            EvidenceRequirementKind.OPAQUE_RECEIPT,
        }
        if tier is EvidenceSourceTier.PROPOSAL:
            reasons.append("proposal_source_forbidden")
        if match in {EvidenceMatchKind.SEMANTIC, EvidenceMatchKind.RETRIEVAL}:
            reasons.append("semantic_match_nomination_only")
        if kind is EvidenceRequirementKind.OPAQUE_RECEIPT and match is not EvidenceMatchKind.TYPED_RECEIPT:
            reasons.append("opaque_requirement_requires_typed_receipt")
        if receipt is not None:
            if match is not EvidenceMatchKind.TYPED_RECEIPT:
                reasons.append("typed_receipt_match_kind_required")
            if not receipt.get("schema") and not receipt.get("schema_version"):
                reasons.append("receipt_schema_missing")
            if not any(
                receipt.get(name)
                for name in ("receipt_id", "evidence_id", "witness_id", "provenance_cid")
            ):
                reasons.append("receipt_identity_missing")
            if normalized not in self._receipt_requirement_ids(receipt):
                reasons.append("receipt_requirement_id_mismatch")
            receipt_tree = str(
                receipt.get("repository_tree")
                or receipt.get("repository_tree_id")
                or receipt.get("tree_id")
                or ""
            ).strip()
            if not receipt_tree:
                reasons.append("receipt_tree_missing")
            elif repository_tree and receipt_tree != str(repository_tree):
                reasons.append("receipt_tree_mismatch")
            expected_policy = str(policy_id or "").strip()
            receipt_policy = str(
                receipt.get("policy_id") or receipt.get("policy_digest") or ""
            ).strip()
            if expected_policy and receipt_policy != expected_policy:
                reasons.append("receipt_policy_mismatch")
            reasons.extend(self._receipt_terminal_reasons(receipt))
            if not self._receipt_tier_allowed(kind, tier):
                reasons.append("receipt_source_kind_not_allowed")
            if tier is EvidenceSourceTier.PROPOSAL:
                reasons.append("proposal_receipt_forbidden")
            satisfies = not reasons
        elif (
            kind in authoritative_kinds
            and kind is not EvidenceRequirementKind.OPAQUE_RECEIPT
            and match
            in {
                EvidenceMatchKind.PATH,
                EvidenceMatchKind.EXACT_TEXT,
                EvidenceMatchKind.EXACT_AST,
            }
            and self._direct_source_tier_allowed(kind, tier)
        ):
            satisfies = True
        elif (
            kind not in authoritative_kinds
            and tier is not EvidenceSourceTier.PROPOSAL
            and (
                match in {
                    EvidenceMatchKind.PATH,
                    EvidenceMatchKind.EXACT_TEXT,
                    EvidenceMatchKind.EXACT_AST,
                }
                or (
                    match in {
                        EvidenceMatchKind.SEMANTIC,
                        EvidenceMatchKind.RETRIEVAL,
                    }
                    and kind
                    in {
                        EvidenceRequirementKind.OTHER,
                        EvidenceRequirementKind.DOCUMENTATION,
                    }
                )
            )
        ):
            satisfies = True
            reasons = [
                reason
                for reason in reasons
                if reason != "semantic_match_nomination_only"
            ]
        elif not reasons:
            reasons.append("source_not_authoritative_for_requirement")
        return EvidenceSourceDecision(
            requirement=normalized,
            requirement_kind=kind,
            source_tier=tier,
            match_kind=match,
            source_path=str(source_path or ""),
            reference=str(reference or ""),
            satisfies=satisfies,
            nominated=True,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    def validate_completion_evidence(
        self,
        requirement: str,
        evidence: Mapping[str, Any] | Any,
        *,
        repository_tree: str = "",
        policy_id: str = "",
        requirement_kind: EvidenceRequirementKind | str | None = None,
    ) -> EvidenceSourceDecision:
        source_path = ""
        if isinstance(evidence, Mapping):
            metadata = evidence.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            source_path = str(
                evidence.get("source_path")
                or evidence.get("path")
                or metadata.get("source_path")
                or metadata.get("path")
                or ""
            )
        return self.evaluate(
            requirement,
            match_kind=EvidenceMatchKind.TYPED_RECEIPT,
            source_path=source_path,
            requirement_kind=requirement_kind,
            typed_receipt=evidence,
            repository_tree=repository_tree,
            policy_id=policy_id,
        )


# Discoverable aliases retained for callers using objective-specific wording.
ObjectiveEvidenceSourcePolicy = EvidenceSourcePolicy
ObjectiveEvidenceDecision = EvidenceSourceDecision


def completion_evidence_source_decision(
    evidence: Mapping[str, Any] | Any,
    *,
    requirement: str = "",
    repository_tree: str = "",
    policy: EvidenceSourcePolicy | None = None,
    policy_id: str = "",
) -> EvidenceSourceDecision:
    """Evaluate a goal-completion record as an exact typed source receipt."""

    if isinstance(evidence, Mapping):
        payload = dict(evidence)
    else:
        converter = getattr(evidence, "to_dict", None)
        projected = converter() if callable(converter) else None
        if not isinstance(projected, Mapping):
            raise TypeError("completion evidence must be a mapping or expose to_dict()")
        payload = dict(projected)
    criterion = str(
        requirement
        or payload.get("acceptance_criterion")
        or payload.get("criterion")
        or ""
    ).strip()
    payload.setdefault("requirement_id", criterion)
    payload.setdefault("receipt_id", payload.get("provenance_cid"))
    validation = payload.get("validation_receipt")
    validation = validation if isinstance(validation, Mapping) else {}
    if not payload.get("status"):
        passed = payload.get("validation_passed")
        if passed is None:
            passed = validation.get("passed")
        payload["status"] = "passed" if passed is True else (
            str(validation.get("status") or "failed")
        )
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    payload.setdefault(
        "source_tier",
        metadata.get("source_tier")
        or metadata.get("source_kind")
        or payload.get("producer_kind"),
    )
    source_path = str(
        metadata.get("source_path")
        or metadata.get("path")
        or payload.get("source_path")
        or ""
    )
    return (policy or EvidenceSourcePolicy()).evaluate(
        criterion,
        match_kind=EvidenceMatchKind.TYPED_RECEIPT,
        source_path=source_path,
        typed_receipt=payload,
        repository_tree=repository_tree,
        policy_id=policy_id,
        reference=str(payload.get("provenance_cid") or ""),
    )

_DERIVED_TASK_PLANNING_FIELDS = frozenset(
    {
        "ast_blob_records",
        "ast_records",
        "conflict_decisions",
        "conflict_edges",
        "conflict_graph",
        "conflict_planning_decisions",
        "conflict_surface",
        "coverage_inputs",
        "dependency_dag",
        "python_ast_records",
        "task_conflict_graph",
        "task_dependency_graph",
        "task_planning_graph",
        "todo_coverage_inputs",
        "todo_vector_summary",
    }
)


def _bounded_task_planning_metadata(task: Mapping[str, Any]) -> dict[str, Any]:
    """Retain task provenance without recursively embedding derived graphs."""

    return {
        str(key): value
        for key, value in task.items()
        if str(key) not in _DERIVED_TASK_PLANNING_FIELDS
    }


@dataclass(frozen=True)
class ObjectiveGoal:
    """One markdown objective-heap node."""

    goal_id: str
    title: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.fields.get("status") or "active").strip().lower()

    @property
    def lifecycle_state(self) -> Any:
        """Return this goal's canonical :class:`GoalState`.

        The import is intentionally local.  ``goal_completion`` consumes the
        objective parser defined in this module, so a module-level import would
        create a cycle.  Keeping normalization at this boundary lets older
        objective heaps continue to use labels such as ``open`` or
        ``completed`` while every graph consumer sees the six-state lifecycle.
        """

        from .goal_completion import normalize_goal_state

        return normalize_goal_state(self.status)

    @property
    def lifecycle_state_value(self) -> str:
        """Return the serializable value of :attr:`lifecycle_state`."""

        state = self.lifecycle_state
        return str(getattr(state, "value", state))

    @property
    def is_schedulable(self) -> bool:
        """Whether new implementation work may be generated for this goal."""

        from .goal_completion import is_schedulable_goal_state

        return is_schedulable_goal_state(self.lifecycle_state)

    @property
    def is_terminal(self) -> bool:
        """Whether the goal has reached the sole terminal lifecycle state."""

        from .goal_completion import is_terminal_goal_state

        return is_terminal_goal_state(self.lifecycle_state)

    @property
    def completion_evidence_metadata(self) -> dict[str, Any]:
        """Return completion-proof references recorded on the goal.

        Objective heaps predate the typed evidence ledger and consequently use
        a few field spellings in the wild.  This projection is deliberately
        lossless (values are not split or otherwise interpreted) and gives
        graph artifacts stable names for the proof dimensions required by the
        completion gate.
        """

        aliases = {
            "acceptance_criterion": (
                "acceptance_criterion",
                "acceptance_criteria",
                "acceptance",
            ),
            "producer": (
                "producing_task_or_scan",
                "producing_task",
                "producing_scan",
                "produced_by",
            ),
            "validation_receipt": (
                "validation_receipt",
                "validation_receipt_cid",
            ),
            "repository_tree": (
                "repository_tree",
                "repository_tree_cid",
                "repo_tree",
                "tree",
            ),
            "freshness": (
                "freshness",
                "evidence_freshness",
                "fresh_at",
            ),
            "provenance_cid": (
                "provenance_cid",
                "evidence_provenance_cid",
                "provenance",
            ),
        }
        metadata: dict[str, Any] = {}
        for canonical_name, field_names in aliases.items():
            for field_name in field_names:
                value = str(self.fields.get(field_name) or "").strip()
                if value:
                    metadata[canonical_name] = value
                    break
        records = self.completion_evidence_records
        if records:
            record = records[0]
            metadata.setdefault("acceptance_criterion", record.get("acceptance_criterion", ""))
            metadata.setdefault(
                "producer",
                record.get("producing_task_or_scan", record.get("producer_id", "")),
            )
            metadata.setdefault("validation_receipt", record.get("validation_receipt", ""))
            metadata.setdefault(
                "repository_tree", record.get("repository_tree", record.get("tree_id", ""))
            )
            metadata.setdefault("freshness", record.get("freshness", ""))
            metadata.setdefault(
                "provenance_cid", record.get("provenance_cid", record.get("receipt_cid", ""))
            )
        return metadata

    @property
    def completion_evidence_records(self) -> list[dict[str, Any]]:
        """Return typed completion records persisted by the tracker."""

        raw = str(
            self.fields.get("completion_evidence_records")
            or self.fields.get("completion_evidence_json")
            or ""
        ).strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list):
            return []
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    @property
    def priority(self) -> tuple[int, str]:
        try:
            fib_priority = int(str(self.fields.get("fib_priority") or "999999").strip())
        except ValueError:
            fib_priority = 999999
        return fib_priority, self.goal_id

    def _string_list_metadata(
        self,
        json_field: str,
        *legacy_fields: str,
    ) -> list[str]:
        raw_json = str(self.fields.get(json_field) or "").strip()
        if raw_json:
            try:
                payload = json.loads(raw_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
                return list(payload)
        for name in legacy_fields:
            value = str(self.fields.get(name) or "")
            if value.strip():
                return split_terms(value)
        return []

    @property
    def required_evidence(self) -> list[str]:
        return self._string_list_metadata(
            "evidence_requirements_json",
            "evidence",
            "required_evidence",
        )

    @property
    def parent_goal_ids(self) -> list[str]:
        return self._string_list_metadata(
            "parent_goal_ids_json",
            "parents",
            "parent",
        )

    @property
    def dependencies(self) -> list[str]:
        """Return durable planning dependencies attached at admission."""

        return self._string_list_metadata(
            "dependencies_json",
            "depends_on",
            "dependencies",
        )

    @property
    def predicted_files(self) -> list[str]:
        return self._string_list_metadata(
            "predicted_files_json",
            "outputs",
            "predicted_files",
        )

    @property
    def predicted_symbols(self) -> list[str]:
        return self._string_list_metadata(
            "predicted_symbols_json",
            "predicted_symbols",
            "ast_symbols",
        )

    @property
    def validation_commands(self) -> list[str]:
        return self._string_list_metadata(
            "validation_commands_json",
            "validation",
        )

    @property
    def semantic_key(self) -> str:
        """Return the admitted proposal's semantic deduplication identity."""

        return str(self.fields.get("semantic_key") or "").strip()

    @property
    def canonical_proposal_id(self) -> str:
        """Return the immutable proposal identity which owns this goal."""

        return str(
            self.fields.get("canonical_proposal_id")
            or self.fields.get("canonical_id")
            or ""
        ).strip()

    @property
    def lifecycle_owner(self) -> str:
        """Return the component responsible for lifecycle transitions."""

        return str(self.fields.get("lifecycle_owner") or "").strip()

    def bundle_key(self, missing_terms: Sequence[str]) -> str:
        explicit = str(self.fields.get("bundle") or "").strip()
        if explicit:
            return explicit.strip("/ ")
        track = str(self.fields.get("track") or "ops").strip().lower() or "ops"
        roots = split_terms(str(self.fields.get("outputs") or ""))
        root = "general"
        for candidate in roots:
            first = candidate.split("/", 1)[0].strip()
            if first and first not in {"data", "tests"}:
                root = first
                break
        fingerprint = sha1("|".join([self.goal_id, *missing_terms]).encode("utf-8")).hexdigest()[:8]
        return f"objective/{track}/{root}/{fingerprint}"


@dataclass(frozen=True)
class ObjectiveFinding:
    """A missing objective proof that can become a todo task."""

    fingerprint: str
    goal_id: str
    title: str
    summary: str
    priority: str
    track: str
    missing_evidence: list[str]
    present_evidence: dict[str, list[str]]
    evidence_methods: list[str]
    objective_path: str
    outputs: list[str]
    validation: str
    goal: str = ""
    refinement: str = ""
    gap_task: str = ""
    parent_goal_ids: list[str] = field(default_factory=list)
    graph_depth: int = 0
    bundle_key: str = "objective/general"
    parallel_lane: str = "objective/general"
    bundle_explicit: bool = False
    bundle_strategy: str = "semantic_ast"
    embedding_query: str = ""
    ast_query: str = ""
    conflict_policy: str = "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts"
    refinement_depth: str = "0"
    candidate_kind: str = "aggregate"
    surplus_group: str = ""
    merge_key: str = ""
    merge_family: str = ""
    merge_role: str = ""
    work_item_count: int = 0
    work_scope: str = ""
    todo_vector_key: str = ""
    goal_packet_key: str = ""
    goal_packet_role: str = ""
    goal_packet_goal_ids: list[str] = field(default_factory=list)
    goal_packet_task_count: int = 0
    goal_packet_work_item_count: int = 0
    predicted_files: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    ast_symbols: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)
    generated_artifacts: list[str] = field(default_factory=list)
    allow_concurrent_with: list[str] = field(default_factory=list)
    dedupe_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveTaskRecord:
    """A generated todo task and its bundle metadata."""

    task_id: str
    task_block: str
    finding: ObjectiveFinding
    discovery_path: Path


@dataclass(frozen=True)
class ObjectiveHeapRecord:
    """One scheduled objective-heap entry."""

    heap_index: int
    goal_id: str
    title: str
    status: str
    fib_priority: int
    priority: str
    priority_rank: int
    track: str
    graph_depth: int
    work_surface_score: int
    required_evidence_count: int
    output_count: int
    parents: list[str]
    sort_key: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEPENDENCY_EDGE_KINDS = frozenset(
    {"goal", "import", "interface", "output_input", "migration", "validation"}
)
SUCCESSFUL_MERGE_RECEIPT_STATUSES = frozenset(
    {"complete", "completed", "merged", "passed", "success", "succeeded"}
)


class CoverageSurfaceKind(str, Enum):
    """The implementation and proof surfaces addressable by goal coverage.

    These values deliberately describe nodes rather than relationships.  The
    coverage assembler is consequently free to use precise edge kinds such as
    ``maps_to_task`` or ``produced_receipt`` without expanding this stable
    interchange vocabulary.
    """

    ACCEPTANCE_CRITERION = "acceptance_criterion"
    TASK = "task"
    PREDICTED_FILE = "predicted_file"
    CHANGED_FILE = "changed_file"
    AST_SYMBOL = "ast_symbol"
    INTERFACE = "interface"
    VALIDATION_COMMAND = "validation_command"
    VALIDATION_RECEIPT = "validation_receipt"
    FINDING = "finding"


class CoverageStatus(str, Enum):
    """Mutually exclusive evidence assessments for one coverage edge."""

    UNCOVERED = "uncovered"
    WEAKLY_INFERRED = "weakly_inferred"
    STALE = "stale"
    CONTRADICTED = "contradicted"
    VERIFIED = "verified"


def _coverage_json_value(value: Any) -> Any:
    """Return a stable, JSON-safe value used by coverage IDs and artifacts."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _coverage_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        converted = [_coverage_json_value(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [_coverage_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _coverage_enum_value(value: Any, enum_type: type[Enum], *, field_name: str) -> str:
    normalized = str(getattr(value, "value", value) or "").strip().lower()
    allowed = {str(item.value) for item in enum_type}
    if normalized not in allowed:
        raise ValueError(
            f"unsupported coverage {field_name} {value!r}; expected one of {sorted(allowed)}"
        )
    return normalized


@dataclass(frozen=True)
class ObjectiveCoverageEdge:
    """One explainable relationship in a goal coverage graph.

    ``edge_id`` is derived from the full normalized assertion, including the
    evidence and provenance CID.  Equal assertions therefore deduplicate
    safely, while a refreshed or contradictory assertion remains visible as a
    separate edge for reconciliation.
    """

    source: str
    target: str
    kind: str
    status: CoverageStatus | str = CoverageStatus.UNCOVERED
    confidence: float = 0.0
    explanation: str = ""
    evidence: Sequence[Any] = field(default_factory=tuple)
    provenance_cid: str = ""

    def __post_init__(self) -> None:
        source = str(self.source or "").strip()
        target = str(self.target or "").strip()
        kind = str(self.kind or "").strip().lower()
        if not source or not target:
            raise ValueError("coverage edges require non-empty source and target node IDs")
        if not kind:
            raise ValueError("coverage edges require a non-empty relationship kind")
        status = _coverage_enum_value(self.status, CoverageStatus, field_name="status")
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("coverage edge confidence must be numeric") from exc
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("coverage edge confidence must be between 0.0 and 1.0")
        explanation = " ".join(str(self.explanation or "").split())
        raw_evidence: Iterable[Any]
        if isinstance(self.evidence, Mapping):
            raw_evidence = (self.evidence,)
        elif isinstance(self.evidence, (str, bytes)):
            raw_evidence = (self.evidence,)
        else:
            raw_evidence = self.evidence or ()
        normalized_evidence = [_coverage_json_value(item) for item in raw_evidence]
        normalized_evidence.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "status", CoverageStatus(status))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "explanation", explanation)
        object.__setattr__(self, "evidence", tuple(normalized_evidence))
        object.__setattr__(self, "provenance_cid", str(self.provenance_cid or "").strip())

    @property
    def edge_id(self) -> str:
        canonical = json.dumps(
            {
                "source": self.source,
                "target": self.target,
                "kind": self.kind,
                "status": self.status.value,
                "confidence": self.confidence,
                "explanation": self.explanation,
                "evidence": list(self.evidence),
                "provenance_cid": self.provenance_cid,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "coverage-edge:" + sha1(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": list(self.evidence),
            "provenance_cid": self.provenance_cid,
        }


@dataclass(frozen=True)
class ObjectiveCoverageGraph:
    """Deterministic node/edge projection for goal coverage consumers."""

    nodes: tuple[dict[str, Any], ...]
    edges: tuple[ObjectiveCoverageEdge, ...]

    @property
    def edges_by_status(self) -> dict[str, list[ObjectiveCoverageEdge]]:
        grouped = {status.value: [] for status in CoverageStatus}
        for edge in self.edges:
            grouped[edge.status.value].append(edge)
        return grouped

    @property
    def status_counts(self) -> dict[str, int]:
        return {status: len(edges) for status, edges in self.edges_by_status.items()}

    def edges_for_status(self, status: CoverageStatus | str) -> list[ObjectiveCoverageEdge]:
        normalized = _coverage_enum_value(status, CoverageStatus, field_name="status")
        return list(self.edges_by_status[normalized])

    @property
    def uncovered(self) -> list[ObjectiveCoverageEdge]:
        return self.edges_for_status(CoverageStatus.UNCOVERED)

    @property
    def weakly_inferred(self) -> list[ObjectiveCoverageEdge]:
        return self.edges_for_status(CoverageStatus.WEAKLY_INFERRED)

    @property
    def stale(self) -> list[ObjectiveCoverageEdge]:
        return self.edges_for_status(CoverageStatus.STALE)

    @property
    def contradicted(self) -> list[ObjectiveCoverageEdge]:
        return self.edges_for_status(CoverageStatus.CONTRADICTED)

    @property
    def verified(self) -> list[ObjectiveCoverageEdge]:
        return self.edges_for_status(CoverageStatus.VERIFIED)

    def to_dict(self) -> dict[str, Any]:
        projections = {
            status: [edge.to_dict() for edge in edges]
            for status, edges in self.edges_by_status.items()
        }
        return {
            "nodes": [dict(node) for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "status_counts": self.status_counts,
            "surfaces_by_status": projections,
            # Keep direct keys so diagnostics and JSON-query callers do not
            # have to know the projection container name.
            **projections,
        }


def _coverage_node_dict(node: Any) -> dict[str, Any]:
    if isinstance(node, Mapping):
        payload = dict(node)
    else:
        to_dict = getattr(node, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("coverage nodes must be mappings or provide to_dict()")
        payload = dict(to_dict())
    node_id = str(payload.get("node_id") or payload.get("id") or "").strip()
    if not node_id:
        raise ValueError("coverage nodes require a non-empty node_id or id")
    payload["node_id"] = node_id
    payload.setdefault("id", node_id)
    if payload.get("surface_kind"):
        payload["surface_kind"] = _coverage_enum_value(
            payload["surface_kind"], CoverageSurfaceKind, field_name="surface kind"
        )
    elif payload.get("kind"):
        # Grouping nodes (for example a goal or the explicit unmapped bucket)
        # may share this graph without pretending to be a coverage surface.
        # Validate known surface values while preserving those group kinds.
        kind = str(getattr(payload["kind"], "value", payload["kind"])).strip().lower()
        if kind in {item.value for item in CoverageSurfaceKind}:
            payload["kind"] = kind
    return _coverage_json_value(payload)


def _coverage_edge_from_value(edge: Any) -> ObjectiveCoverageEdge:
    if isinstance(edge, ObjectiveCoverageEdge):
        return edge
    if not isinstance(edge, Mapping):
        raise TypeError("coverage edges must be ObjectiveCoverageEdge instances or mappings")
    return ObjectiveCoverageEdge(
        source=str(edge.get("source") or edge.get("from") or edge.get("source_node_id") or ""),
        target=str(edge.get("target") or edge.get("to") or edge.get("target_node_id") or ""),
        kind=str(edge.get("kind") or edge.get("relationship") or ""),
        status=edge.get("status") or CoverageStatus.UNCOVERED,
        confidence=edge.get("confidence", 0.0),
        explanation=str(edge.get("explanation") or edge.get("reason") or ""),
        evidence=edge.get("evidence") or (),
        provenance_cid=str(edge.get("provenance_cid") or edge.get("receipt_cid") or ""),
    )


def materialize_objective_coverage_graph(
    nodes: Iterable[Any],
    edges: Iterable[ObjectiveCoverageEdge | Mapping[str, Any]],
) -> ObjectiveCoverageGraph:
    """Normalize, validate, deduplicate, and sort a coverage graph.

    Duplicate node IDs must contain the same normalized payload.  Treating a
    collision as an error, instead of allowing insertion order to win, is
    important because objective graph artifacts are used as completion proof.
    Edges must reference registered nodes and exact duplicate assertions are
    collapsed by their deterministic ID.
    """

    nodes_by_id: dict[str, dict[str, Any]] = {}
    for raw_node in nodes:
        node = _coverage_node_dict(raw_node)
        node_id = str(node["node_id"])
        previous = nodes_by_id.get(node_id)
        if previous is not None and previous != node:
            raise ValueError(f"conflicting coverage node registrations for {node_id!r}")
        nodes_by_id[node_id] = node

    edges_by_id: dict[str, ObjectiveCoverageEdge] = {}
    for raw_edge in edges:
        edge = _coverage_edge_from_value(raw_edge)
        missing = [node_id for node_id in (edge.source, edge.target) if node_id not in nodes_by_id]
        if missing:
            raise ValueError(
                f"coverage edge {edge.edge_id!r} references unregistered nodes: {sorted(set(missing))}"
            )
        edges_by_id[edge.edge_id] = edge

    normalized_nodes = tuple(nodes_by_id[key] for key in sorted(nodes_by_id))
    normalized_edges = tuple(edges_by_id[key] for key in sorted(edges_by_id))
    return ObjectiveCoverageGraph(nodes=normalized_nodes, edges=normalized_edges)


class ObjectiveWorkKind(str, Enum):
    """Kinds of work which may be introduced while refining an objective."""

    GOAL = "goal"
    SUBGOAL = "subgoal"
    TASK = "task"


@dataclass(frozen=True)
class ObjectiveGenerationLimits:
    """Hard limits applied to one autonomous objective-refinement cycle.

    The limits are deliberately independent.  In particular, a large token
    budget cannot bypass the hierarchy or open-board limits and retries do not
    gain a fresh breadth allocation.  This makes repeated daemon cycles safe
    even when their proposal source is nondeterministic.
    """

    max_depth: int = 3
    max_breadth_per_parent: int = 4
    max_new_work: int = 12
    max_open_work: int = 48
    token_budget: int = 8192
    max_retries: int = 2
    semantic_similarity_threshold: float = 0.82

    def __post_init__(self) -> None:
        for name in (
            "max_depth",
            "max_breadth_per_parent",
            "max_new_work",
            "max_open_work",
            "token_budget",
            "max_retries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        threshold = float(self.semantic_similarity_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("semantic_similarity_threshold must be between 0 and 1")
        object.__setattr__(self, "semantic_similarity_threshold", threshold)


def _objective_work_strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = re.split(r"[,;\n]+", value)
    elif isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Iterable):
        values = value
    else:
        values = (value,)
    normalized = {
        " ".join(str(item or "").split())
        for item in values
        if " ".join(str(item or "").split())
    }
    return tuple(sorted(normalized, key=lambda item: (item.casefold(), item)))


def _objective_work_value(payload: Mapping[str, Any], name: str, *aliases: str) -> Any:
    for key in (name, *aliases):
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return ""


def _objective_work_normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def semantic_objective_work_key(value: Any) -> str:
    """Return a stable semantic key which ignores display IDs and ordering."""

    payload = value.to_dict() if isinstance(value, ObjectiveWorkProposal) else _task_record_mapping(value)
    material = {
        "kind": str(_objective_work_value(payload, "kind", "work_kind") or "task").casefold(),
        "parent_goal_id": _objective_work_normalized_text(
            _objective_work_value(payload, "parent_goal_id", "parent_objective_id", "goal_id")
        ),
        "objective_terms": sorted(
            _objective_work_normalized_text(item)
            for item in _objective_work_strings(
                _objective_work_value(payload, "parent_objective_terms", "objective_terms")
            )
        ),
        "evidence_delta": sorted(
            _objective_work_normalized_text(item)
            for item in _objective_work_strings(
                _objective_work_value(
                    payload,
                    "expected_evidence_delta",
                    "missing_evidence",
                    "evidence_delta",
                )
            )
        ),
        "files": sorted(
            str(item).strip().replace("\\", "/").casefold()
            for item in _objective_work_strings(
                _objective_work_value(payload, "predicted_files", "outputs", "files")
            )
        ),
        "symbols": sorted(
            _objective_work_normalized_text(item)
            for item in _objective_work_strings(
                _objective_work_value(payload, "predicted_symbols", "ast_symbols", "symbols")
            )
        ),
        "validation": sorted(
            _objective_work_normalized_text(item)
            for item in _objective_work_strings(
                _objective_work_value(payload, "validation_commands", "validation")
            )
        ),
    }
    # A title is only identity material when no objective/evidence surface was
    # supplied.  This keeps harmless wording changes from regenerating work.
    if not any(material[key] for key in ("objective_terms", "evidence_delta", "files", "symbols")):
        material["title"] = _objective_work_normalized_text(
            _objective_work_value(payload, "title", "summary")
        )
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "objective-work/v1/" + sha1(canonical.encode("utf-8")).hexdigest()


def canonical_objective_work_identity(value: Any) -> str:
    """Return the canonical content identity used across refinement cycles."""

    payload = value.to_dict() if isinstance(value, ObjectiveWorkProposal) else _task_record_mapping(value)
    semantic_key = str(payload.get("semantic_key") or semantic_objective_work_key(payload))
    material = {
        "schema": "ipfs_accelerate_py/agent-supervisor/objective-work@1",
        "semantic_key": semantic_key,
        "title": _objective_work_normalized_text(
            _objective_work_value(payload, "title", "summary")
        ),
        "dependencies": sorted(
            _objective_work_normalized_text(item)
            for item in _objective_work_strings(
                _objective_work_value(payload, "dependencies", "depends_on")
            )
        ),
        "confidence": float(_objective_work_value(payload, "confidence") or 0.0),
        "estimated_cost": float(
            _objective_work_value(payload, "estimated_cost", "cost") or 0.0
        ),
        "novelty": float(_objective_work_value(payload, "novelty") or 0.0),
        "depth": int(_objective_work_value(payload, "depth", "graph_depth") or 0),
        "estimated_tokens": int(
            _objective_work_value(payload, "estimated_tokens", "token_cost") or 0
        ),
    }
    return "objective-work:" + sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ObjectiveWorkProposal:
    """A reviewable goal, subgoal, or task proposed from incomplete proof."""

    kind: ObjectiveWorkKind | str
    title: str
    parent_goal_id: str
    parent_objective_terms: tuple[str, ...]
    expected_evidence_delta: tuple[str, ...]
    dependencies: tuple[str, ...]
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    validation_commands: tuple[str, ...]
    confidence: float
    estimated_cost: float
    novelty: float
    depth: int
    estimated_tokens: int = 0
    retry_count: int = 0
    source: str = "deterministic"
    source_id: str = ""
    rationale: str = ""
    semantic_key: str = ""
    canonical_id: str = ""

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, ObjectiveWorkKind) else ObjectiveWorkKind(str(self.kind).lower())
        except ValueError as exc:
            raise ValueError("kind must be goal, subgoal, or task") from exc
        title = " ".join(str(self.title or "").split())
        parent = str(self.parent_goal_id or "").strip()
        if not title:
            raise ValueError("objective work title must be non-empty")
        if kind is not ObjectiveWorkKind.GOAL and not parent:
            raise ValueError("subgoal and task work require parent_goal_id")
        if isinstance(self.depth, bool) or int(self.depth) < 0:
            raise ValueError("depth must be a non-negative integer")
        if isinstance(self.estimated_tokens, bool) or int(self.estimated_tokens) < 0:
            raise ValueError("estimated_tokens must be a non-negative integer")
        if isinstance(self.retry_count, bool) or int(self.retry_count) < 0:
            raise ValueError("retry_count must be a non-negative integer")
        for name in ("confidence", "novelty"):
            number = float(getattr(self, name))
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, number)
        cost = float(self.estimated_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("estimated_cost must be finite and non-negative")
        object.__setattr__(self, "estimated_cost", cost)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "parent_goal_id", parent)
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "estimated_tokens", int(self.estimated_tokens))
        object.__setattr__(self, "retry_count", int(self.retry_count))
        object.__setattr__(
            self,
            "source",
            " ".join(str(self.source or "deterministic").split()),
        )
        object.__setattr__(
            self,
            "source_id",
            " ".join(str(self.source_id or "").split()),
        )
        object.__setattr__(self, "rationale", " ".join(str(self.rationale or "").split()))
        for name in (
            "parent_objective_terms",
            "expected_evidence_delta",
            "dependencies",
            "predicted_files",
            "predicted_symbols",
            "validation_commands",
        ):
            object.__setattr__(self, name, _objective_work_strings(getattr(self, name)))
        expected_semantic_key = semantic_objective_work_key(self)
        supplied_semantic_key = str(self.semantic_key or "").strip()
        if supplied_semantic_key and supplied_semantic_key != expected_semantic_key:
            raise ValueError("semantic_key does not match canonical objective work content")
        object.__setattr__(self, "semantic_key", expected_semantic_key)
        expected_canonical_id = canonical_objective_work_identity(self)
        supplied_canonical_id = str(self.canonical_id or "").strip()
        if supplied_canonical_id and supplied_canonical_id != expected_canonical_id:
            raise ValueError("canonical_id does not match canonical objective work content")
        object.__setattr__(self, "canonical_id", expected_canonical_id)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveWorkProposal":
        """Normalize deterministic and LLM-router spellings into one record."""

        if not isinstance(payload, Mapping):
            raise TypeError("objective work proposals must be mappings")
        kind = _objective_work_value(payload, "kind", "work_kind", "proposal_kind") or "task"
        terms = _objective_work_value(payload, "parent_objective_terms", "objective_terms", "terms")
        delta = _objective_work_value(
            payload, "expected_evidence_delta", "evidence_delta", "missing_evidence"
        )
        return cls(
            kind=kind,
            title=_objective_work_value(payload, "title", "summary"),
            parent_goal_id=_objective_work_value(
                payload, "parent_goal_id", "parent_objective_id", "goal_id"
            ),
            parent_objective_terms=_objective_work_strings(terms),
            expected_evidence_delta=_objective_work_strings(delta),
            dependencies=_objective_work_strings(
                _objective_work_value(payload, "dependencies", "depends_on")
            ),
            predicted_files=_objective_work_strings(
                _objective_work_value(payload, "predicted_files", "outputs", "files")
            ),
            predicted_symbols=_objective_work_strings(
                _objective_work_value(payload, "predicted_symbols", "ast_symbols", "symbols")
            ),
            validation_commands=_objective_work_strings(
                _objective_work_value(payload, "validation_commands", "validation")
            ),
            confidence=_objective_work_value(payload, "confidence") or 0.0,
            estimated_cost=_objective_work_value(payload, "estimated_cost", "cost") or 0.0,
            novelty=_objective_work_value(payload, "novelty") or 0.0,
            depth=_objective_work_value(payload, "depth", "graph_depth") or 0,
            estimated_tokens=_objective_work_value(payload, "estimated_tokens", "token_cost") or 0,
            retry_count=_objective_work_value(payload, "retry_count", "retries") or 0,
            source=_objective_work_value(payload, "source", "proposal_source") or "deterministic",
            source_id=_objective_work_value(payload, "source_id", "criterion_id", "finding_id", "receipt_id"),
            rationale=_objective_work_value(payload, "rationale", "explanation", "reason"),
            semantic_key=str(payload.get("semantic_key") or ""),
            canonical_id=str(payload.get("canonical_id") or payload.get("work_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        for name in (
            "parent_objective_terms",
            "expected_evidence_delta",
            "dependencies",
            "predicted_files",
            "predicted_symbols",
            "validation_commands",
        ):
            payload[name] = list(payload[name])
        # Task consumers already understand these canonical identity names.
        payload["canonical_task_key"] = self.semantic_key
        payload["dedupe_key"] = self.semantic_key
        payload["cost"] = self.estimated_cost
        payload["validation"] = list(self.validation_commands)
        payload["parent_objective_id"] = self.parent_goal_id
        return payload


@dataclass(frozen=True)
class ObjectiveGenerationRejection:
    """One proposal excluded by an explicit finite-refinement rule."""

    reason: str
    canonical_id: str
    source_id: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveGenerationResult:
    """Accepted work plus a complete deterministic accounting of rejections."""

    accepted: tuple[ObjectiveWorkProposal, ...]
    rejected: tuple[ObjectiveGenerationRejection, ...]
    limits: ObjectiveGenerationLimits
    initial_open_work: int
    consumed_tokens: int

    @property
    def exhausted(self) -> bool:
        return bool(self.rejected) and any(
            item.reason in {
                "depth_limit", "breadth_limit", "cycle_limit", "open_work_limit",
                "retry_limit", "token_budget",
            }
            for item in self.rejected
        )

    @property
    def generated_work(self) -> tuple[ObjectiveWorkProposal, ...]:
        return self.accepted

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: dict[str, int] = {}
        for item in self.rejected:
            rejection_counts[item.reason] = rejection_counts.get(item.reason, 0) + 1
        return {
            "accepted": [item.to_dict() for item in self.accepted],
            "generated_work": [item.to_dict() for item in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "limits": asdict(self.limits),
            "initial_open_work": self.initial_open_work,
            "final_open_work": self.initial_open_work + len(self.accepted),
            "consumed_tokens": self.consumed_tokens,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class ObjectiveGoalMaterializationPolicy:
    """Fail-closed policy for projecting proposals into the objective heap.

    This policy intentionally contains only graph invariants.  Admission mode,
    proof authority, and receipt validation belong to the daemon/admission
    boundary; a caller must satisfy those gates before applying a successful
    preview.  ``expected_*`` values fence the immutable input used by that
    boundary so a preview cannot be replayed against a different heap or root.
    """

    limits: ObjectiveGenerationLimits = field(default_factory=ObjectiveGenerationLimits)
    root_goal_id: str = ""
    expected_heap_content_id: str = ""
    expected_root_content_id: str = ""
    lifecycle_owner: str = "objective_daemon"
    require_schedulable_parent: bool = True
    atomic: bool = True
    root_parent_aliases: tuple[str, ...] = ("__unmapped__",)

    def __post_init__(self) -> None:
        limits = self.limits
        if isinstance(limits, Mapping):
            limits = ObjectiveGenerationLimits(**dict(limits))
        if not isinstance(limits, ObjectiveGenerationLimits):
            raise TypeError("limits must be ObjectiveGenerationLimits or a mapping")
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "root_goal_id", str(self.root_goal_id or "").strip())
        object.__setattr__(
            self,
            "expected_heap_content_id",
            str(self.expected_heap_content_id or "").strip(),
        )
        object.__setattr__(
            self,
            "expected_root_content_id",
            str(self.expected_root_content_id or "").strip(),
        )
        owner = " ".join(str(self.lifecycle_owner or "").split())
        if not owner:
            raise ValueError("lifecycle_owner must be non-empty")
        object.__setattr__(self, "lifecycle_owner", owner)
        object.__setattr__(
            self,
            "root_parent_aliases",
            _objective_work_strings(self.root_parent_aliases),
        )


@dataclass(frozen=True)
class ObjectiveGoalMaterializationRejection:
    """One goal/subgoal which cannot be safely projected into the heap."""

    reason: str
    canonical_id: str
    source_id: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class MaterializedObjectiveGoal:
    """A proposal's deterministic, reviewable objective-heap projection."""

    proposal: ObjectiveWorkProposal
    goal: ObjectiveGoal
    rendered_block: str
    graph_depth: int
    parent_goal_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "goal": {
                "goal_id": self.goal.goal_id,
                "title": self.goal.title,
                "fields": dict(self.goal.fields),
            },
            "rendered_block": self.rendered_block,
            "graph_depth": self.graph_depth,
            "parent_goal_ids": list(self.parent_goal_ids),
        }


@dataclass(frozen=True)
class ObjectiveGoalMaterializationPreview:
    """An immutable preview which a transactional writer may apply verbatim."""

    base_heap_content_id: str
    root_goal_id: str
    root_content_id: str
    materialized: tuple[MaterializedObjectiveGoal, ...]
    rejected: tuple[ObjectiveGoalMaterializationRejection, ...]
    fatal_reasons: tuple[str, ...]
    candidate_text: str
    policy: ObjectiveGoalMaterializationPolicy

    @property
    def ready(self) -> bool:
        """Whether the complete proposal set is safe to apply atomically."""

        if self.fatal_reasons or not self.materialized:
            return False
        return not (self.policy.atomic and self.rejected)

    @property
    def changed(self) -> bool:
        return self.ready and bool(self.materialized)

    @property
    def candidate_heap_content_id(self) -> str:
        return objective_heap_content_id(self.candidate_text)

    @property
    def admitted_proposal_ids(self) -> tuple[str, ...]:
        return tuple(item.proposal.canonical_id for item in self.materialized) if self.ready else ()

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: dict[str, int] = {}
        for item in self.rejected:
            rejection_counts[item.reason] = rejection_counts.get(item.reason, 0) + 1
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/objective-goal-materialization@1",
            "base_heap_content_id": self.base_heap_content_id,
            "candidate_heap_content_id": self.candidate_heap_content_id,
            "root_goal_id": self.root_goal_id,
            "root_content_id": self.root_content_id,
            "ready": self.ready,
            "changed": self.changed,
            "admitted_proposal_ids": list(self.admitted_proposal_ids),
            "materialized": [item.to_dict() for item in self.materialized],
            "rejected": [item.to_dict() for item in self.rejected],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "fatal_reasons": list(self.fatal_reasons),
            "policy": {
                "limits": asdict(self.policy.limits),
                "root_goal_id": self.policy.root_goal_id,
                "expected_heap_content_id": self.policy.expected_heap_content_id,
                "expected_root_content_id": self.policy.expected_root_content_id,
                "lifecycle_owner": self.policy.lifecycle_owner,
                "require_schedulable_parent": self.policy.require_schedulable_parent,
                "atomic": self.policy.atomic,
                "root_parent_aliases": list(self.policy.root_parent_aliases),
            },
        }


def _objective_work_tokens(value: ObjectiveWorkProposal) -> set[str]:
    fields: list[str] = [
        value.kind.value,
        value.parent_goal_id,
        *value.parent_objective_terms,
        *value.expected_evidence_delta,
        *value.predicted_files,
        *value.predicted_symbols,
    ]
    return set(re.findall(r"[a-z0-9]+", " ".join(fields).casefold()))


def _objective_work_similarity(left: ObjectiveWorkProposal, right: ObjectiveWorkProposal) -> float:
    if left.kind is not right.kind or left.parent_goal_id.casefold() != right.parent_goal_id.casefold():
        return 0.0
    left_tokens, right_tokens = _objective_work_tokens(left), _objective_work_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def materialize_bounded_objective_work(
    proposals: Iterable[ObjectiveWorkProposal | Mapping[str, Any]],
    *,
    existing_work: Iterable[ObjectiveWorkProposal | Mapping[str, Any]] = (),
    limits: ObjectiveGenerationLimits | Mapping[str, Any] | None = None,
    current_open_work: int | None = None,
) -> ObjectiveGenerationResult:
    """Validate, deduplicate, and bound autonomous objective refinement.

    Candidates are ordered by parent, depth, kind, semantic identity, rather
    than provider order.  Exact identities and high-overlap semantic work are
    checked against historical as well as newly accepted records, preventing
    equivalent work from reappearing on later daemon cycles.
    """

    if limits is None:
        policy = ObjectiveGenerationLimits()
    elif isinstance(limits, ObjectiveGenerationLimits):
        policy = limits
    elif isinstance(limits, Mapping):
        policy = ObjectiveGenerationLimits(**dict(limits))
    else:
        raise TypeError("limits must be ObjectiveGenerationLimits or a mapping")

    existing = [
        item if isinstance(item, ObjectiveWorkProposal) else ObjectiveWorkProposal.from_dict(item)
        for item in existing_work
    ]
    normalized = [
        item if isinstance(item, ObjectiveWorkProposal) else ObjectiveWorkProposal.from_dict(item)
        for item in proposals
    ]
    normalized.sort(
        key=lambda item: (
            item.parent_goal_id.casefold(),
            item.depth,
            {ObjectiveWorkKind.GOAL: 0, ObjectiveWorkKind.SUBGOAL: 1, ObjectiveWorkKind.TASK: 2}[item.kind],
            item.semantic_key,
            item.source_id,
        )
    )
    open_count = len(existing) if current_open_work is None else max(0, int(current_open_work))
    exact_ids = {item.canonical_id for item in existing}
    comparison = list(existing)
    accepted: list[ObjectiveWorkProposal] = []
    rejected: list[ObjectiveGenerationRejection] = []
    breadth: dict[str, int] = {}
    consumed_tokens = 0

    def reject(item: ObjectiveWorkProposal, reason: str, detail: str) -> None:
        rejected.append(ObjectiveGenerationRejection(reason, item.canonical_id, item.source_id, detail))

    for item in normalized:
        if item.canonical_id in exact_ids:
            reject(item, "canonical_duplicate", "canonical identity already exists")
            continue
        duplicate = next(
            (
                prior
                for prior in comparison
                if _objective_work_similarity(item, prior) >= policy.semantic_similarity_threshold
            ),
            None,
        )
        if duplicate is not None:
            reject(item, "semantic_duplicate", f"equivalent to {duplicate.canonical_id}")
            continue
        if item.depth > policy.max_depth:
            reject(item, "depth_limit", f"depth {item.depth} exceeds {policy.max_depth}")
            continue
        if item.retry_count > policy.max_retries:
            reject(item, "retry_limit", f"retry {item.retry_count} exceeds {policy.max_retries}")
            continue
        if len(accepted) >= policy.max_new_work:
            reject(item, "cycle_limit", f"cycle allows {policy.max_new_work} new records")
            continue
        if open_count + len(accepted) >= policy.max_open_work:
            reject(item, "open_work_limit", f"open work limit is {policy.max_open_work}")
            continue
        parent_key = item.parent_goal_id or "__root__"
        if breadth.get(parent_key, 0) >= policy.max_breadth_per_parent:
            reject(item, "breadth_limit", f"parent allows {policy.max_breadth_per_parent} open children")
            continue
        if consumed_tokens + item.estimated_tokens > policy.token_budget:
            reject(item, "token_budget", f"cycle token budget is {policy.token_budget}")
            continue
        accepted.append(item)
        exact_ids.add(item.canonical_id)
        comparison.append(item)
        breadth[parent_key] = breadth.get(parent_key, 0) + 1
        consumed_tokens += item.estimated_tokens

    return ObjectiveGenerationResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        limits=policy,
        initial_open_work=open_count,
        consumed_tokens=consumed_tokens,
    )


# Public aliases use both "goal generation" and "objective refinement"
# terminology because persisted callers and operator docs use both.
GoalGenerationLimits = ObjectiveGenerationLimits
GeneratedObjectiveWork = ObjectiveWorkProposal
ObjectiveGenerationPlan = ObjectiveGenerationResult
generate_bounded_objective_work = materialize_bounded_objective_work


class ProofObligationState(str, Enum):
    """Planning-relevant state of a proof obligation.

    Proof executors use a larger lifecycle vocabulary.  Objective planning
    only needs to distinguish unfinished work, trusted completion, and the
    three terminal conditions which require bounded repair work.
    """

    PENDING = "pending"
    SATISFIED = "satisfied"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    CONTRADICTED = "contradicted"

    @property
    def requires_repair(self) -> bool:
        return self in {
            ProofObligationState.UNSUPPORTED,
            ProofObligationState.FAILED,
            ProofObligationState.CONTRADICTED,
        }


class ProofRepairWorkKind(str, Enum):
    """Closed set of work that an unsuccessful proof may introduce."""

    TEMPLATE = "template"
    TEST = "test"
    PREMISE = "premise"
    MANUAL_REVIEW = "manual_review"


_PROOF_STATE_ALIASES: dict[str, ProofObligationState] = {
    "": ProofObligationState.PENDING,
    "active": ProofObligationState.PENDING,
    "blocked": ProofObligationState.FAILED,
    "cancelled": ProofObligationState.FAILED,
    "complete": ProofObligationState.SATISFIED,
    "completed": ProofObligationState.SATISFIED,
    "contradicted": ProofObligationState.CONTRADICTED,
    "contradiction": ProofObligationState.CONTRADICTED,
    "disproved": ProofObligationState.CONTRADICTED,
    "error": ProofObligationState.FAILED,
    "failed": ProofObligationState.FAILED,
    "failure": ProofObligationState.FAILED,
    "inconclusive": ProofObligationState.FAILED,
    "invalid": ProofObligationState.CONTRADICTED,
    "passed": ProofObligationState.SATISFIED,
    "pending": ProofObligationState.PENDING,
    "planned": ProofObligationState.PENDING,
    "proved": ProofObligationState.SATISFIED,
    "rejected": ProofObligationState.CONTRADICTED,
    "running": ProofObligationState.PENDING,
    "satisfied": ProofObligationState.SATISFIED,
    "stale": ProofObligationState.CONTRADICTED,
    "succeeded": ProofObligationState.SATISFIED,
    "success": ProofObligationState.SATISFIED,
    "timed_out": ProofObligationState.FAILED,
    "timeout": ProofObligationState.FAILED,
    "unavailable": ProofObligationState.UNSUPPORTED,
    "unsupported_proof": ProofObligationState.UNSUPPORTED,
    "unsupported_template": ProofObligationState.UNSUPPORTED,
    "unsupported": ProofObligationState.UNSUPPORTED,
    "verified": ProofObligationState.SATISFIED,
}


def _proof_obligation_state(value: Any) -> ProofObligationState:
    if isinstance(value, ProofObligationState):
        return value
    normalized = str(getattr(value, "value", value) or "").strip().lower().replace("-", "_")
    return _PROOF_STATE_ALIASES.get(normalized, ProofObligationState.PENDING)


def _proof_repair_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if hasattr(value, "__dict__"):
        return {
            str(key): item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError("proof obligations and outcomes must be mappings or provide to_dict()")


def _proof_repair_strings(value: Any, *, maximum: int | None = None) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = re.split(r"[,;\n]+", value)
    elif isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Iterable):
        values = value
    else:
        values = (value,)
    result = tuple(
        sorted(
            {
                " ".join(str(item or "").split())
                for item in values
                if " ".join(str(item or "").split())
            },
            key=lambda item: (item.casefold(), item),
        )
    )
    return result if maximum is None else result[: max(0, int(maximum))]


def _proof_repair_text(value: Any, *, maximum: int = 2048) -> str:
    result = " ".join(str(value or "").split())
    if len(result) <= maximum:
        return result
    return result[: max(0, maximum - 1)].rstrip() + "…"


@dataclass(frozen=True)
class ProofObligationInput:
    """Small, executor-independent obligation projection used by planning."""

    obligation_id: str
    state: ProofObligationState | str = ProofObligationState.PENDING
    statement: str = ""
    task_id: str = ""
    goal_id: str = ""
    template_id: str = ""
    template_version: str = ""
    code_shape: str = ""
    required_assurance: str = ""
    premise_ids: tuple[str, ...] = ()
    missing_premise_ids: tuple[str, ...] = ()
    fallback_checks: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    reason: str = ""
    source_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        obligation_id = str(self.obligation_id or "").strip()
        if not obligation_id:
            raise ValueError("proof repair planning requires obligation_id")
        object.__setattr__(self, "obligation_id", obligation_id)
        object.__setattr__(self, "state", _proof_obligation_state(self.state))
        for name in (
            "statement",
            "task_id",
            "goal_id",
            "template_id",
            "template_version",
            "code_shape",
            "required_assurance",
            "reason",
            "source_id",
        ):
            object.__setattr__(self, name, _proof_repair_text(getattr(self, name)))
        for name in (
            "premise_ids",
            "missing_premise_ids",
            "fallback_checks",
            "dependencies",
        ):
            object.__setattr__(self, name, _proof_repair_strings(getattr(self, name)))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("proof obligation metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            {
                str(key): _coverage_json_value(item)
                for key, item in sorted(self.metadata.items(), key=lambda pair: str(pair[0]))
            },
        )

    @property
    def status(self) -> ProofObligationState:
        """Compatibility spelling used by proof receipt consumers."""

        return self.state

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        outcome: Mapping[str, Any] | None = None,
    ) -> "ProofObligationInput":
        if not isinstance(payload, Mapping):
            raise TypeError("proof obligation input must be a mapping")
        values = dict(payload)
        metadata = values.get("metadata") if isinstance(values.get("metadata"), Mapping) else {}
        metadata = dict(metadata)
        outcome_payload = dict(outcome or {})
        outcome_metadata = (
            dict(outcome_payload.get("metadata") or {})
            if isinstance(outcome_payload.get("metadata"), Mapping)
            else {}
        )

        def first(name: str, *aliases: str, default: Any = "") -> Any:
            for source in (outcome_payload, outcome_metadata, values, metadata):
                for key in (name, *aliases):
                    if key in source and source[key] not in (None, ""):
                        return source[key]
            return default

        # A provider attempt may succeed while its semantic verdict disproves
        # the obligation.  Prefer verdict over transport/lifecycle status so a
        # successful execution can never hide a contradiction.
        declared_state: Any = ProofObligationState.PENDING.value
        for source in (outcome_payload, outcome_metadata, values, metadata):
            for key in (
                "verdict",
                "proof_verdict",
                "state",
                "proof_state",
                "status",
                "proof_status",
            ):
                if key in source and source[key] not in (None, ""):
                    declared_state = source[key]
                    break
            if declared_state != ProofObligationState.PENDING.value:
                break
        obligation_id = first(
            "obligation_id", "proof_obligation_id", "content_id", "id"
        )
        return cls(
            obligation_id=str(obligation_id or ""),
            state=declared_state,
            statement=first("statement", "canonical_statement", "obligation"),
            task_id=first("task_id"),
            goal_id=first("goal_id", "parent_goal_id"),
            template_id=first("template_id"),
            template_version=first("template_version", "version"),
            code_shape=first("code_shape", "reviewed_code_shape"),
            required_assurance=first("required_assurance", "assurance"),
            premise_ids=_proof_repair_strings(first("premise_ids", "premises", default=())),
            missing_premise_ids=_proof_repair_strings(
                first(
                    "missing_premise_ids",
                    "unsupported_premise_ids",
                    "failed_premise_ids",
                    default=(),
                )
            ),
            fallback_checks=_proof_repair_strings(
                first(
                    "fallback_checks",
                    "fallback_tests",
                    "validation_commands",
                    "validations",
                    default=(),
                )
            ),
            dependencies=_proof_repair_strings(
                first("dependencies", "depends_on", "dependency_ids", default=())
            ),
            reason=first(
                "reason",
                "failure_reason",
                "error",
                "summary",
                "diagnostic",
            ),
            source_id=first(
                "source_id", "receipt_id", "attempt_id", "selection_id"
            ),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "state": self.state.value,
            "status": self.state.value,
            "statement": self.statement,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "code_shape": self.code_shape,
            "required_assurance": self.required_assurance,
            "premise_ids": list(self.premise_ids),
            "missing_premise_ids": list(self.missing_premise_ids),
            "fallback_checks": list(self.fallback_checks),
            "dependencies": list(self.dependencies),
            "reason": self.reason,
            "source_id": self.source_id,
            "metadata": dict(self.metadata),
        }


# A longer spelling makes the boundary explicit in generated API docs while
# retaining the concise input name used by callers.
ProofObligationPlanningInput = ProofObligationInput


@dataclass(frozen=True)
class ProofRepairPolicy:
    """Independent hard bounds for proof-derived objective work."""

    max_obligations: int = 64
    max_total_work: int = 16
    max_work_per_obligation: int = 4
    max_existing_work: int = 256
    max_rejections: int = 128
    max_dependencies_per_work: int = 32
    max_validation_commands_per_work: int = 16
    semantic_similarity_threshold: float = 0.84
    # Concise constructor aliases used by objective planner configurations.
    max_work_items: int | None = None
    max_per_obligation: int | None = None

    def __post_init__(self) -> None:
        if self.max_work_items is not None:
            if isinstance(self.max_work_items, bool) or not isinstance(
                self.max_work_items, int
            ):
                raise ValueError("max_work_items must be a non-negative integer")
            object.__setattr__(self, "max_total_work", self.max_work_items)
        if self.max_per_obligation is not None:
            if isinstance(self.max_per_obligation, bool) or not isinstance(
                self.max_per_obligation, int
            ):
                raise ValueError("max_per_obligation must be a non-negative integer")
            object.__setattr__(
                self, "max_work_per_obligation", self.max_per_obligation
            )
        for name in (
            "max_obligations",
            "max_total_work",
            "max_work_per_obligation",
            "max_existing_work",
            "max_rejections",
            "max_dependencies_per_work",
            "max_validation_commands_per_work",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "max_work_items", self.max_total_work)
        object.__setattr__(self, "max_per_obligation", self.max_work_per_obligation)
        threshold = float(self.semantic_similarity_threshold)
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("semantic_similarity_threshold must be between 0 and 1")
        object.__setattr__(self, "semantic_similarity_threshold", threshold)


def _proof_repair_normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def semantic_proof_repair_key(value: Any) -> str:
    """Return wording-insensitive identity for one proof repair action."""

    payload = value.to_dict() if isinstance(value, ProofRepairWork) else _proof_repair_mapping(value)

    def values(name: str, *aliases: str) -> list[str]:
        raw: Any = ()
        for key in (name, *aliases):
            if key in payload and payload[key] not in (None, ""):
                raw = payload[key]
                break
        return sorted(_proof_repair_normalized_text(item) for item in _proof_repair_strings(raw))

    kind = str(
        getattr(
            payload.get("repair_kind", payload.get("work_kind", payload.get("kind", ""))),
            "value",
            payload.get("repair_kind", payload.get("work_kind", payload.get("kind", ""))),
        )
        or ""
    ).strip().lower()
    material: dict[str, Any] = {
        "repair_kind": kind,
        "template_id": _proof_repair_normalized_text(payload.get("template_id", "")),
        "template_version": _proof_repair_normalized_text(payload.get("template_version", "")),
        "semantic_scope": _proof_repair_normalized_text(
            payload.get(
                "semantic_scope",
                payload.get("statement", payload.get("code_shape", "")),
            )
        ),
        "required_assurance": _proof_repair_normalized_text(
            payload.get("required_assurance", "")
        ),
        "premise_ids": values("premise_ids", "missing_premise_ids"),
        "validation": values(
            "validation_commands", "fallback_checks", "validation"
        ),
        "evidence_delta": values(
            "expected_evidence_delta", "evidence_delta", "missing_evidence"
        ),
    }
    if not any(
        material[key]
        for key in (
            "template_id",
            "semantic_scope",
            "premise_ids",
            "validation",
            "evidence_delta",
        )
    ):
        material["title"] = _proof_repair_normalized_text(
            payload.get("title", payload.get("summary", ""))
        )
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "proof-repair/v1/" + sha1(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProofRepairWork:
    """One finite task produced from an unsupported or failed obligation."""

    repair_kind: ProofRepairWorkKind | str
    obligation_ids: tuple[str, ...]
    title: str
    rationale: str
    expected_evidence_delta: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    template_id: str = ""
    template_version: str = ""
    semantic_scope: str = ""
    required_assurance: str = ""
    source_state: ProofObligationState | str = ProofObligationState.FAILED
    source_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    semantic_key: str = ""
    canonical_id: str = ""

    def __post_init__(self) -> None:
        try:
            kind = (
                self.repair_kind
                if isinstance(self.repair_kind, ProofRepairWorkKind)
                else ProofRepairWorkKind(str(self.repair_kind).strip().lower())
            )
        except ValueError as exc:
            raise ValueError(
                "repair_kind must be template, test, premise, or manual_review"
            ) from exc
        obligation_ids = _proof_repair_strings(self.obligation_ids)
        if not obligation_ids:
            raise ValueError("proof repair work requires at least one obligation_id")
        title = _proof_repair_text(self.title)
        rationale = _proof_repair_text(self.rationale)
        if not title or not rationale:
            raise ValueError("proof repair work requires title and rationale")
        evidence_delta = _proof_repair_strings(self.expected_evidence_delta)
        if not evidence_delta:
            raise ValueError("proof repair work requires expected_evidence_delta")
        object.__setattr__(self, "repair_kind", kind)
        object.__setattr__(self, "obligation_ids", obligation_ids)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "expected_evidence_delta", evidence_delta)
        for name in (
            "dependencies",
            "validation_commands",
            "source_ids",
            "premise_ids",
        ):
            object.__setattr__(self, name, _proof_repair_strings(getattr(self, name)))
        for name in (
            "template_id",
            "template_version",
            "semantic_scope",
            "required_assurance",
        ):
            object.__setattr__(self, name, _proof_repair_text(getattr(self, name)))
        object.__setattr__(self, "source_state", _proof_obligation_state(self.source_state))
        expected_key = semantic_proof_repair_key(self)
        if self.semantic_key and str(self.semantic_key) != expected_key:
            raise ValueError("semantic_key does not match proof repair content")
        object.__setattr__(self, "semantic_key", expected_key)
        identity_payload = {
            "schema": "ipfs_accelerate_py/agent-supervisor/proof-repair-work@1",
            "semantic_key": expected_key,
            "obligation_ids": obligation_ids,
            "source_state": self.source_state.value,
        }
        expected_id = "proof-repair:" + sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.canonical_id and str(self.canonical_id) != expected_id:
            raise ValueError("canonical_id does not match proof repair content")
        object.__setattr__(self, "canonical_id", expected_id)

    @property
    def obligation_id(self) -> str:
        """Return the sole/first obligation for compatibility with task rows."""

        return self.obligation_ids[0]

    @property
    def kind(self) -> ProofRepairWorkKind:
        return self.repair_kind

    @property
    def dedupe_key(self) -> str:
        return self.semantic_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "work_id": self.canonical_id,
            "repair_kind": self.repair_kind.value,
            "work_kind": self.repair_kind.value,
            "kind": self.repair_kind.value,
            "obligation_id": self.obligation_id,
            "obligation_ids": list(self.obligation_ids),
            "title": self.title,
            "rationale": self.rationale,
            "expected_evidence_delta": list(self.expected_evidence_delta),
            "dependencies": list(self.dependencies),
            "validation_commands": list(self.validation_commands),
            "validation": list(self.validation_commands),
            "template_id": self.template_id,
            "template_version": self.template_version,
            "semantic_scope": self.semantic_scope,
            "required_assurance": self.required_assurance,
            "source_state": self.source_state.value,
            "source_ids": list(self.source_ids),
            "premise_ids": list(self.premise_ids),
            "semantic_key": self.semantic_key,
            "dedupe_key": self.semantic_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofRepairWork":
        if not isinstance(payload, Mapping):
            raise TypeError("proof repair work must be a mapping")
        obligation_ids = payload.get("obligation_ids")
        if not obligation_ids:
            obligation_ids = (payload.get("obligation_id"),)
        return cls(
            repair_kind=payload.get(
                "repair_kind", payload.get("work_kind", payload.get("kind", ""))
            ),
            obligation_ids=_proof_repair_strings(obligation_ids),
            title=str(payload.get("title") or payload.get("summary") or ""),
            rationale=str(
                payload.get("rationale")
                or payload.get("reason")
                or payload.get("explanation")
                or ""
            ),
            expected_evidence_delta=_proof_repair_strings(
                payload.get(
                    "expected_evidence_delta",
                    payload.get("evidence_delta", payload.get("missing_evidence", ())),
                )
            ),
            dependencies=_proof_repair_strings(
                payload.get("dependencies", payload.get("depends_on", ()))
            ),
            validation_commands=_proof_repair_strings(
                payload.get(
                    "validation_commands",
                    payload.get("validation", payload.get("fallback_checks", ())),
                )
            ),
            template_id=str(payload.get("template_id") or ""),
            template_version=str(payload.get("template_version") or ""),
            semantic_scope=str(
                payload.get(
                    "semantic_scope",
                    payload.get("statement", payload.get("code_shape", "")),
                )
                or ""
            ),
            required_assurance=str(payload.get("required_assurance") or ""),
            source_state=payload.get(
                "source_state", payload.get("state", payload.get("status", "failed"))
            ),
            source_ids=_proof_repair_strings(
                payload.get("source_ids", (payload.get("source_id"),))
            ),
            premise_ids=_proof_repair_strings(
                payload.get("premise_ids", payload.get("missing_premise_ids", ()))
            ),
            semantic_key=str(payload.get("semantic_key") or payload.get("dedupe_key") or ""),
            canonical_id=str(payload.get("canonical_id") or payload.get("work_id") or ""),
        )


@dataclass(frozen=True)
class ProofRepairRejection:
    """Bounded rationale for repair work that was not emitted."""

    reason: str
    obligation_id: str
    repair_kind: str = ""
    semantic_key: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ProofRepairResult:
    """Finite proof-repair tasks and a bounded rejection accounting."""

    work: tuple[ProofRepairWork, ...]
    rejected: tuple[ProofRepairRejection, ...]
    policy: ProofRepairPolicy
    considered_obligations: int
    ignored_obligations: int = 0
    input_truncated: bool = False
    rejection_count: int = 0
    rejections_truncated: bool = False

    @property
    def generated_work(self) -> tuple[ProofRepairWork, ...]:
        return self.work

    @property
    def accepted(self) -> tuple[ProofRepairWork, ...]:
        return self.work

    @property
    def truncated(self) -> bool:
        return (
            self.input_truncated
            or self.rejections_truncated
            or any(
                item.reason in {"per_obligation_limit", "total_work_limit"}
                for item in self.rejected
            )
        )

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: dict[str, int] = {}
        for item in self.rejected:
            rejection_counts[item.reason] = rejection_counts.get(item.reason, 0) + 1
        return {
            "work": [item.to_dict() for item in self.work],
            "generated_work": [item.to_dict() for item in self.work],
            "accepted": [item.to_dict() for item in self.work],
            "rejected": [item.to_dict() for item in self.rejected],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "rejection_count": self.rejection_count,
            "policy": asdict(self.policy),
            "considered_obligations": self.considered_obligations,
            "ignored_obligations": self.ignored_obligations,
            "input_truncated": self.input_truncated,
            "rejections_truncated": self.rejections_truncated,
            "truncated": self.truncated,
        }


def _proof_repair_tokens(value: ProofRepairWork) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                (
                    value.repair_kind.value,
                    value.template_id,
                    value.semantic_scope,
                    value.required_assurance,
                    *value.premise_ids,
                    *value.validation_commands,
                    *value.expected_evidence_delta,
                )
            ).casefold(),
        )
    )


def proof_repair_semantic_similarity(
    left: ProofRepairWork,
    right: ProofRepairWork,
) -> float:
    """Return Jaccard similarity for same-kind repair semantics."""

    if left.repair_kind is not right.repair_kind:
        return 0.0
    left_tokens = _proof_repair_tokens(left)
    right_tokens = _proof_repair_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _proof_repair_outcomes(
    outcomes: Iterable[Any] | Mapping[str, Any],
    *,
    maximum: int,
) -> dict[str, dict[str, Any]]:
    if isinstance(outcomes, Mapping):
        if any(
            key in outcomes
            for key in ("obligation_id", "proof_obligation_id", "status", "verdict")
        ):
            raw: Iterable[Any] = (outcomes,)
        else:
            raw = (
                dict(
                    (
                        _proof_repair_mapping(value)
                        if isinstance(value, Mapping)
                        or callable(getattr(value, "to_dict", None))
                        or hasattr(value, "__dict__")
                        else {"status": value}
                    ),
                    obligation_id=str(key),
                )
                for key, value in outcomes.items()
            )
    else:
        raw = outcomes
    indexed: dict[str, dict[str, Any]] = {}
    for value in islice(raw, max(0, maximum)):
        payload = _proof_repair_mapping(value)
        obligation_id = str(
            payload.get("obligation_id")
            or payload.get("proof_obligation_id")
            or ""
        ).strip()
        if not obligation_id:
            continue
        previous = indexed.get(obligation_id)
        if previous is None:
            indexed[obligation_id] = payload
            continue
        # A repair-requiring terminal result must not be hidden by an
        # insertion-order-dependent pending row.
        previous_state = _proof_obligation_state(
            previous.get("state", previous.get("status", previous.get("verdict", "")))
        )
        candidate_state = _proof_obligation_state(
            payload.get("state", payload.get("status", payload.get("verdict", "")))
        )
        rank = {
            ProofObligationState.PENDING: 0,
            ProofObligationState.SATISFIED: 1,
            ProofObligationState.FAILED: 2,
            ProofObligationState.UNSUPPORTED: 3,
            ProofObligationState.CONTRADICTED: 4,
        }
        if rank[candidate_state] > rank[previous_state]:
            indexed[obligation_id] = payload
    return indexed


def _proof_repair_kinds(item: ProofObligationInput) -> tuple[ProofRepairWorkKind, ...]:
    explicit = item.metadata.get(
        "repair_kinds",
        item.metadata.get("suggested_repair_kinds", item.metadata.get("repair_kind", ())),
    )
    if explicit:
        result: set[ProofRepairWorkKind] = set()
        for value in _proof_repair_strings(explicit):
            try:
                result.add(ProofRepairWorkKind(value.strip().lower()))
            except ValueError:
                # Unreviewed work kinds may not cross the planning boundary.
                continue
        if result:
            return tuple(sorted(result, key=lambda value: value.value))

    reason = " ".join((item.reason, item.code_shape)).casefold()
    kinds: set[ProofRepairWorkKind] = set()
    if item.missing_premise_ids or (
        item.premise_ids
        and (
            item.state in {
                ProofObligationState.FAILED,
                ProofObligationState.CONTRADICTED,
            }
            or any(term in reason for term in ("premise", "assumption"))
        )
    ):
        kinds.add(ProofRepairWorkKind.PREMISE)
    template_problem = (
        not item.template_id
        or any(
            term in reason
            for term in (
                "ambiguous template",
                "missing template",
                "no reviewed template",
                "template unsupported",
                "unknown template",
                "unsupported code shape",
            )
        )
        or item.metadata.get("template_supported") is False
    )
    if item.state is ProofObligationState.UNSUPPORTED and template_problem:
        kinds.add(ProofRepairWorkKind.TEMPLATE)
    executable_fallbacks = tuple(
        check
        for check in item.fallback_checks
        if not check.casefold().startswith(("manual:", "review:"))
    )
    if executable_fallbacks:
        kinds.add(ProofRepairWorkKind.TEST)
    if item.state is ProofObligationState.CONTRADICTED:
        kinds.add(ProofRepairWorkKind.MANUAL_REVIEW)
    if "ambiguous" in reason or "conflict" in reason or "manual review" in reason:
        kinds.add(ProofRepairWorkKind.MANUAL_REVIEW)
    if not kinds:
        kinds.add(ProofRepairWorkKind.MANUAL_REVIEW)
    order = {
        ProofRepairWorkKind.PREMISE: 0,
        ProofRepairWorkKind.TEMPLATE: 1,
        ProofRepairWorkKind.TEST: 2,
        ProofRepairWorkKind.MANUAL_REVIEW: 3,
    }
    return tuple(sorted(kinds, key=lambda value: order[value]))


def _proof_repair_candidate(
    item: ProofObligationInput,
    kind: ProofRepairWorkKind,
    policy: ProofRepairPolicy,
) -> ProofRepairWork:
    scope = item.statement or item.code_shape or item.template_id or item.obligation_id
    short_scope = _proof_repair_text(scope, maximum=160)
    rationale = item.reason or (
        f"Obligation {item.obligation_id} ended in {item.state.value} state."
    )
    missing_premises = item.missing_premise_ids or (
        item.premise_ids
        if kind is ProofRepairWorkKind.PREMISE
        else ()
    )
    if kind is ProofRepairWorkKind.TEMPLATE:
        title = f"Add reviewed proof template for {short_scope}"
        delta = (
            "reviewed versioned template with executable reference semantics",
            "template mutation cases and exact code-shape support evidence",
        )
        validations: tuple[str, ...] = ()
    elif kind is ProofRepairWorkKind.TEST:
        title = f"Add fallback regression test for {short_scope}"
        delta = (
            "current fallback validation receipt bound to the obligation scope",
        )
        validations = tuple(
            check
            for check in item.fallback_checks
            if not check.casefold().startswith(("manual:", "review:"))
        )[: policy.max_validation_commands_per_work]
    elif kind is ProofRepairWorkKind.PREMISE:
        title = f"Establish proof premises for {short_scope}"
        delta = (
            "trusted current evidence for each missing proof premise",
        )
        validations = ()
    else:
        title = f"Manually review proof obligation for {short_scope}"
        delta = (
            "recorded manual-review decision with rationale and provenance",
        )
        validations = ()
    return ProofRepairWork(
        repair_kind=kind,
        obligation_ids=(item.obligation_id,),
        title=title,
        rationale=rationale,
        expected_evidence_delta=delta,
        dependencies=item.dependencies[: policy.max_dependencies_per_work],
        validation_commands=validations,
        template_id=item.template_id,
        template_version=item.template_version,
        semantic_scope=scope,
        required_assurance=item.required_assurance,
        source_state=item.state,
        source_ids=(item.source_id,) if item.source_id else (),
        premise_ids=missing_premises,
    )


def _merge_proof_repair_work(
    left: ProofRepairWork,
    right: ProofRepairWork,
    policy: ProofRepairPolicy,
) -> ProofRepairWork:
    return ProofRepairWork(
        repair_kind=left.repair_kind,
        obligation_ids=_proof_repair_strings(
            (*left.obligation_ids, *right.obligation_ids)
        ),
        title=min((left.title, right.title), key=lambda value: (value.casefold(), value)),
        rationale=min(
            (left.rationale, right.rationale),
            key=lambda value: (value.casefold(), value),
        ),
        expected_evidence_delta=_proof_repair_strings(
            (*left.expected_evidence_delta, *right.expected_evidence_delta)
        ),
        dependencies=_proof_repair_strings(
            (*left.dependencies, *right.dependencies),
            maximum=policy.max_dependencies_per_work,
        ),
        validation_commands=_proof_repair_strings(
            (*left.validation_commands, *right.validation_commands),
            maximum=policy.max_validation_commands_per_work,
        ),
        template_id=left.template_id or right.template_id,
        template_version=left.template_version or right.template_version,
        semantic_scope=left.semantic_scope or right.semantic_scope,
        required_assurance=left.required_assurance or right.required_assurance,
        source_state=(
            ProofObligationState.CONTRADICTED
            if ProofObligationState.CONTRADICTED
            in {left.source_state, right.source_state}
            else left.source_state
        ),
        source_ids=_proof_repair_strings((*left.source_ids, *right.source_ids)),
        premise_ids=_proof_repair_strings((*left.premise_ids, *right.premise_ids)),
    )


def generate_proof_repair_work(
    obligations: Iterable[ProofObligationInput | Mapping[str, Any] | Any],
    *,
    outcomes: Iterable[Any] | Mapping[str, Any] = (),
    existing_work: Iterable[ProofRepairWork | Mapping[str, Any]] = (),
    policy: ProofRepairPolicy | Mapping[str, Any] | None = None,
    limits: ProofRepairPolicy | Mapping[str, Any] | None = None,
    max_work_items: int | None = None,
    max_work_per_obligation: int | None = None,
) -> ProofRepairResult:
    """Generate finite repair tasks for unsuccessful proof obligations.

    Inputs are consumed under ``max_obligations`` and existing work under
    ``max_existing_work``.  Consequently an accidentally unbounded iterator,
    a retry storm, or a large receipt ledger cannot expand one planning cycle
    without limit.  Unsupported work kinds are never model-invented: the
    output vocabulary is the closed :class:`ProofRepairWorkKind` enum.
    """

    if policy is not None and limits is not None:
        raise ValueError("provide either policy or limits, not both")
    raw_policy = policy if policy is not None else limits
    if raw_policy is None:
        selected_policy = ProofRepairPolicy()
    elif isinstance(raw_policy, ProofRepairPolicy):
        selected_policy = raw_policy
    elif isinstance(raw_policy, Mapping):
        selected_policy = ProofRepairPolicy(**dict(raw_policy))
    else:
        raise TypeError("policy must be ProofRepairPolicy or a mapping")
    overrides: dict[str, Any] = {}
    if max_work_items is not None:
        overrides["max_total_work"] = int(max_work_items)
    if max_work_per_obligation is not None:
        overrides["max_work_per_obligation"] = int(max_work_per_obligation)
    if overrides:
        selected_policy = replace(selected_policy, **overrides)

    outcome_index = _proof_repair_outcomes(
        outcomes,
        maximum=max(1, selected_policy.max_obligations * 2),
    )
    obligation_rows = list(
        islice(obligations, selected_policy.max_obligations + 1)
    )
    input_truncated = len(obligation_rows) > selected_policy.max_obligations
    obligation_rows = obligation_rows[: selected_policy.max_obligations]
    normalized: list[ProofObligationInput] = []
    for raw in obligation_rows:
        if isinstance(raw, ProofObligationInput):
            base = raw
            outcome = outcome_index.get(base.obligation_id)
            normalized.append(
                ProofObligationInput.from_dict(base.to_dict(), outcome=outcome)
                if outcome
                else base
            )
        else:
            payload = _proof_repair_mapping(raw)
            obligation_id = str(
                payload.get("obligation_id")
                or payload.get("proof_obligation_id")
                or payload.get("content_id")
                or payload.get("id")
                or ""
            ).strip()
            normalized.append(
                ProofObligationInput.from_dict(
                    payload,
                    outcome=outcome_index.get(obligation_id),
                )
            )
    normalized.sort(key=lambda item: (item.obligation_id, item.state.value))

    existing: list[ProofRepairWork] = []
    existing_keys: set[str] = set()
    for raw in islice(existing_work, selected_policy.max_existing_work):
        if isinstance(raw, ProofRepairWork):
            prior = raw
        else:
            payload = _proof_repair_mapping(raw)
            try:
                prior = ProofRepairWork.from_dict(payload)
            except (TypeError, ValueError):
                supplied_key = str(
                    payload.get("semantic_key") or payload.get("dedupe_key") or ""
                )
                if supplied_key.startswith("proof-repair/v1/"):
                    existing_keys.add(supplied_key)
                continue
        existing.append(prior)
        existing_keys.add(prior.semantic_key)

    accepted: list[ProofRepairWork] = []
    accepted_by_key: dict[str, int] = {}
    rejected: list[ProofRepairRejection] = []
    rejection_count = 0
    ignored = 0

    def reject(
        item: ProofObligationInput,
        reason: str,
        *,
        candidate: ProofRepairWork | None = None,
        detail: str = "",
    ) -> None:
        nonlocal rejection_count
        rejection_count += 1
        if len(rejected) >= selected_policy.max_rejections:
            return
        rejected.append(
            ProofRepairRejection(
                reason=reason,
                obligation_id=item.obligation_id,
                repair_kind=(
                    candidate.repair_kind.value if candidate is not None else ""
                ),
                semantic_key=(
                    candidate.semantic_key if candidate is not None else ""
                ),
                detail=detail,
            )
        )

    for item in normalized:
        if not item.state.requires_repair:
            ignored += 1
            continue
        kinds = _proof_repair_kinds(item)
        if len(kinds) > selected_policy.max_work_per_obligation:
            for kind in kinds[selected_policy.max_work_per_obligation :]:
                reject(
                    item,
                    "per_obligation_limit",
                    detail=f"repair kind {kind.value} exceeds per-obligation bound",
                )
            kinds = kinds[: selected_policy.max_work_per_obligation]
        for kind in kinds:
            candidate = _proof_repair_candidate(item, kind, selected_policy)
            if candidate.semantic_key in existing_keys:
                reject(
                    item,
                    "semantic_duplicate",
                    candidate=candidate,
                    detail="equivalent persisted proof repair work already exists",
                )
                continue
            existing_duplicate = next(
                (
                    prior
                    for prior in existing
                    if proof_repair_semantic_similarity(candidate, prior)
                    >= selected_policy.semantic_similarity_threshold
                ),
                None,
            )
            if existing_duplicate is not None:
                reject(
                    item,
                    "semantic_duplicate",
                    candidate=candidate,
                    detail=f"equivalent to {existing_duplicate.canonical_id}",
                )
                continue
            merge_index = accepted_by_key.get(candidate.semantic_key)
            if merge_index is None:
                merge_index = next(
                    (
                        index
                        for index, prior in enumerate(accepted)
                        if (
                            item.obligation_id in prior.obligation_ids
                            and prior.repair_kind is candidate.repair_kind
                        )
                        or proof_repair_semantic_similarity(candidate, prior)
                        >= selected_policy.semantic_similarity_threshold
                    ),
                    None,
                )
            if merge_index is not None:
                merged = _merge_proof_repair_work(
                    accepted[merge_index], candidate, selected_policy
                )
                old_key = accepted[merge_index].semantic_key
                accepted[merge_index] = merged
                accepted_by_key.pop(old_key, None)
                accepted_by_key[merged.semantic_key] = merge_index
                reject(
                    item,
                    "semantic_duplicate",
                    candidate=candidate,
                    detail=f"coalesced into {merged.canonical_id}",
                )
                continue
            if len(accepted) >= selected_policy.max_total_work:
                reject(
                    item,
                    "total_work_limit",
                    candidate=candidate,
                    detail=f"cycle allows {selected_policy.max_total_work} proof repair records",
                )
                continue
            accepted_by_key[candidate.semantic_key] = len(accepted)
            accepted.append(candidate)

    accepted.sort(
        key=lambda item: (
            item.repair_kind.value,
            item.semantic_key,
            item.obligation_ids,
        )
    )
    return ProofRepairResult(
        work=tuple(accepted),
        rejected=tuple(rejected),
        policy=selected_policy,
        considered_obligations=len(normalized),
        ignored_obligations=ignored,
        input_truncated=input_truncated,
        rejection_count=rejection_count,
        rejections_truncated=rejection_count > len(rejected),
    )


# Compatibility spellings used by planner and objective-refinement callers.
ProofRepairGenerationPolicy = ProofRepairPolicy
ProofRepairGenerationResult = ProofRepairResult
materialize_proof_repair_work = generate_proof_repair_work
generate_obligation_repair_work = generate_proof_repair_work


@dataclass(frozen=True)
class DependencyEdge:
    """A prerequisite relationship directed from producer to consumer.

    ``provenance`` intentionally remains structured instead of being collapsed
    into a reason string.  Persisted DAGs can consequently explain which task
    field and value produced an edge, even after task aliases are reconciled.
    """

    source_task_cid: str
    target_task_cid: str
    kind: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskDependencyNode:
    """One canonical task in a materialized dependency graph."""

    task_cid: str
    task_id: str
    goal_id: str
    status: str
    objective_priority: int
    created_at_ms: int
    estimated_duration: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DependencyRepairEvidence:
    """Bounded, actionable evidence for malformed dependency metadata."""

    kind: str
    task_cid: str
    task_id: str
    reference: str
    message: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskScheduleRecord:
    """Critical-path scheduling metrics for one canonical task."""

    task_cid: str
    task_id: str
    claimable: bool
    blocking_task_cids: list[str]
    critical_path_length: int
    slack: int
    downstream_unlock_value: int
    age_seconds: int
    objective_priority: int
    score: int
    sort_key: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskDependencyGraph:
    """A task DAG, its schedule, and finite repair evidence.

    The type retains cyclic or incomplete nodes for diagnosis.  Such nodes are
    simply made unclaimable; they never prevent independent acyclic components
    from being scheduled.
    """

    nodes: dict[str, TaskDependencyNode]
    edges: list[DependencyEdge]
    schedule: list[TaskScheduleRecord] = field(default_factory=list)
    repair_evidence: list[DependencyRepairEvidence] = field(default_factory=list)
    invalid_task_cids: list[str] = field(default_factory=list)

    @property
    def claimable_task_cids(self) -> list[str]:
        return [record.task_cid for record in self.schedule if record.claimable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {key: value.to_dict() for key, value in sorted(self.nodes.items())},
            "edges": [edge.to_dict() for edge in self.edges],
            "schedule": [record.to_dict() for record in self.schedule],
            "repair_evidence": [item.to_dict() for item in self.repair_evidence],
            "invalid_task_cids": sorted(self.invalid_task_cids),
            "claimable_task_cids": self.claimable_task_cids,
        }


# A concise alias for callers that use the backlog task's DAG terminology.
TaskDependencyDAG = TaskDependencyGraph


@dataclass(frozen=True)
class TaskPlanningGraph:
    """Combined dependency schedule and conflict-colored execution plan.

    Keeping the two graph types together prevents callers from accidentally
    treating a dependency-ready task as concurrency-safe.  The conflict graph
    remains the owner of lane assignments and their explanations, while the
    dependency graph remains the owner of prerequisite claimability.
    """

    dependency_graph: TaskDependencyGraph
    conflict_graph: Any

    @property
    def claimable_task_cids(self) -> list[str]:
        return self.dependency_graph.claimable_task_cids

    def to_dict(self) -> dict[str, Any]:
        dependency = self.dependency_graph.to_dict()
        conflict = self.conflict_graph.to_dict()
        return {
            "task_dependency_graph": dependency,
            "dependency_dag": dependency,
            "task_conflict_graph": conflict,
            "conflict_graph": conflict,
            "claimable_task_cids": self.claimable_task_cids,
            "lanes": conflict.get("lanes", {}),
            "lane_assignments": conflict.get("assignments", []),
            "planning_decisions": conflict.get("decisions", []),
        }


@dataclass(frozen=True)
class BundleWriteResult:
    """Files produced when bundle shards and the index are written."""

    generated_paths: list[Path]
    index_path: Path
    bundle_paths: dict[str, Path]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_terms(value: str) -> list[str]:
    terms: list[str] = []
    for raw in re.split(r"[,;]", value):
        term = " ".join(raw.strip().split())
        if term:
            terms.append(term)
    return terms


def objective_tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 1]


def text_embedding(value: str, *, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> list[float]:
    vector = [0.0] * max(1, int(dimensions))
    for token in objective_tokens(value):
        digest = sha1(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1.0
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        return vector
    return [item / norm for item in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def normalize_field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_goal_heap(text: str) -> list[ObjectiveGoal]:
    """Parse flat markdown objective records.

    Records use ``## GOAL-ID Title`` headers and ``- Field: value`` rows.  The
    goal id is intentionally not tied to a project prefix so other packages can
    provide their own objective heaps.
    """

    goals: list[ObjectiveGoal] = []
    current_id = ""
    current_title = ""
    current_fields: dict[str, str] = {}
    header_pattern = re.compile(r"^##\s+(\S+)\s+(.+?)\s*$")

    def flush() -> None:
        if current_id and current_fields:
            goals.append(ObjectiveGoal(goal_id=current_id, title=current_title.strip(), fields=dict(current_fields)))

    for line in text.splitlines():
        header = header_pattern.match(line)
        if header:
            flush()
            current_id = header.group(1)
            current_title = header.group(2)
            current_fields = {}
            continue
        if not current_id or not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        current_fields[normalize_field_key(key)] = value.strip()
    flush()
    return goals


def safe_bundle_key(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip("/ ").lower()).strip("-")
    return safe or "objective-general"


def repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def repo_relative_path_safe(relative: str) -> bool:
    if not relative or relative.startswith("/") or "\0" in relative:
        return False
    return ".." not in Path(relative).parts


def symbol_terms(path: Path, text: str) -> set[str]:
    """Extract AST/schema-ish terms from code and structured files."""

    suffix = path.suffix.lower()
    symbols: set[str] = set()
    if suffix == ".py":
        try:
            tree = parse_python_ast_quietly(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            class_stack: list[str] = []

            class Visitor(ast.NodeVisitor):
                def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                    symbols.add(node.name)
                    class_stack.append(node.name)
                    self.generic_visit(node)
                    class_stack.pop()

                def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                    symbols.add(node.name)
                    if class_stack:
                        symbols.add(f"{class_stack[-1]}.{node.name}")
                    self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                    self.visit_FunctionDef(node)  # type: ignore[arg-type]

                def visit_Import(self, node: ast.Import) -> Any:
                    for alias in node.names:
                        symbols.add(alias.name)

                def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                    if node.module:
                        symbols.add(node.module)
                        for alias in node.names:
                            symbols.add(f"{node.module}.{alias.name}")

            Visitor().visit(tree)
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        for match in re.finditer(
            r"\b(?:class|function|interface|type|const|let|var)\s+([A-Za-z_$][\w$]*)",
            text,
        ):
            symbols.add(match.group(1))
        for match in re.finditer(r"\bexport\s+\{([^}]+)\}", text):
            for raw in match.group(1).split(","):
                symbol = raw.strip().split(" as ", 1)[0].strip()
                if symbol:
                    symbols.add(symbol)
    elif suffix in {".md", ".rst"}:
        for line in text.splitlines():
            stripped = line.strip("#= ")
            if line.startswith("#") and stripped:
                symbols.add(stripped)
    elif suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    symbols.add(str(key))
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(payload)

    expanded: set[str] = set()
    for symbol in symbols:
        expanded.add(symbol)
        expanded.add(" ".join(objective_tokens(symbol)))
    return {item.lower() for item in expanded if item.strip()}


def ast_dataset_payload(path: Path, text: str, *, max_chars: int = DEFAULT_AST_DATASET_MAX_CHARS) -> dict[str, Any]:
    """Return a serializable AST/symbol payload suitable for dataset storage."""

    suffix = path.suffix.lower()
    ast_kind = "symbols"
    ast_text = ""
    parse_error = ""
    if suffix == ".py":
        ast_kind = "python_ast"
        try:
            tree = parse_python_ast_quietly(text)
            ast_text = ast.dump(tree, include_attributes=True)
        except SyntaxError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    elif suffix == ".json":
        ast_kind = "json_keys"
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        ast_kind = "js_ts_symbols"
    elif suffix in {".md", ".rst"}:
        ast_kind = "markdown_headings"

    truncated = False
    if len(ast_text) > max_chars:
        ast_text = ast_text[:max_chars]
        truncated = True
    return {
        "ast_kind": ast_kind,
        "ast_text": ast_text,
        "ast_truncated": truncated,
        "parse_error": parse_error,
    }


def collect_ast_dataset_records(
    repo_root: Path,
    *,
    objective_path: Path,
    max_ast_chars: int = DEFAULT_AST_DATASET_MAX_CHARS,
    previous_records: Sequence[Mapping[str, Any]] = (),
    scan_stats: dict[str, Any] | None = None,
    excluded_roots: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    """Collect a complete snapshot while reusing unchanged source blobs."""

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    prior_rows = [dict(row) for row in previous_records if isinstance(row, Mapping)]
    prior_by_blob: dict[str, list[dict[str, Any]]] = {}
    prior_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in prior_rows:
        if int(row.get("record_schema_version") or 0) != AST_DATASET_RECORD_SCHEMA_VERSION:
            continue
        blob_hash = str(row.get("blob_hash") or "")
        source_hash = str(row.get("source_sha1") or "")
        if blob_hash:
            prior_by_blob.setdefault(blob_hash, []).append(row)
        if source_hash:
            prior_by_source.setdefault(source_hash, []).append(row)
    for candidates in [*prior_by_blob.values(), *prior_by_source.values()]:
        candidates.sort(key=lambda item: str(item.get("root_relative_path") or ""))

    blob_hashes = tracked_blob_hashes(repo_root)
    current_paths: set[str] = set()
    current_path_blobs: dict[str, str] = {}
    parsed_count = 0
    reused_count = 0
    parse_elapsed = 0.0
    saved_parse_seconds = 0.0
    excluded = tuple(root.resolve() for root in excluded_roots)
    for path in objective_candidate_files(repo_root, objective_path=objective_path):
        resolved_path = path.resolve()
        if any(resolved_path == root or root in resolved_path.parents for root in excluded):
            continue
        root_relative = repo_relative_path(repo_root, path)
        current_paths.add(root_relative)
        blob_hash = blob_hashes.get(resolved_path, "")
        prior = prior_by_blob.get(blob_hash, [None])[0] if blob_hash else None
        source_bytes: bytes | None = None
        text: str | None = None
        source_hash = ""
        if prior is None:
            try:
                source_bytes = path.read_bytes()
            except OSError:
                continue
            text = source_bytes.decode("utf-8", errors="replace")
            source_hash = sha1(source_bytes).hexdigest()
            prior = prior_by_source.get(source_hash, [None])[0]

        if prior is not None:
            row = dict(prior)
            if str(row.get("root_relative_path") or "") != root_relative:
                row.update(
                    _ast_evidence_fields(
                        root_relative,
                        str(row.get("evidence_text") or ""),
                        _record_symbols(row),
                    )
                )
            row.update(
                {
                    "root_relative_path": root_relative,
                    "suffix": path.suffix.lower(),
                    "blob_hash": blob_hash or str(row.get("blob_hash") or source_hash),
                }
            )
            rows.append(row)
            reused_count += 1
            saved_parse_seconds += _nonnegative_seconds(row.get("parse_elapsed_seconds"))
            current_path_blobs[root_relative] = str(row.get("blob_hash") or row.get("source_sha1") or "")
            continue

        assert text is not None and source_bytes is not None
        parse_started = time.monotonic()
        symbols = sorted(symbol_terms(path, text))
        payload = ast_dataset_payload(path, text, max_chars=max_ast_chars)
        row_parse_elapsed = max(0.0, time.monotonic() - parse_started)
        parsed_count += 1
        parse_elapsed += row_parse_elapsed
        effective_blob_hash = blob_hash or source_hash
        rows.append(
            {
                "record_schema_version": AST_DATASET_RECORD_SCHEMA_VERSION,
                "root_relative_path": root_relative,
                "suffix": path.suffix.lower(),
                "blob_hash": effective_blob_hash,
                "source_sha1": source_hash,
                "source_bytes": len(source_bytes),
                "symbols_json": json.dumps(symbols, sort_keys=True),
                "token_count": len(objective_tokens(text)),
                "parse_elapsed_seconds": row_parse_elapsed,
                **_ast_evidence_fields(root_relative, text, symbols),
                **payload,
            }
        )
        current_path_blobs[root_relative] = effective_blob_hash

    rows.sort(key=lambda row: str(row.get("root_relative_path") or ""))
    prior_paths = {
        str(row.get("root_relative_path") or "")
        for row in prior_rows
        if str(row.get("root_relative_path") or "")
    }
    deleted_paths = sorted(prior_paths - current_paths)
    prior_blob_by_path = {
        str(row.get("root_relative_path") or ""): str(row.get("blob_hash") or row.get("source_sha1") or "")
        for row in prior_rows
    }
    deleted_blob_counts: dict[str, int] = {}
    added_blob_counts: dict[str, int] = {}
    for path in deleted_paths:
        blob = prior_blob_by_path.get(path, "")
        if blob:
            deleted_blob_counts[blob] = deleted_blob_counts.get(blob, 0) + 1
    for path in current_paths - prior_paths:
        blob = current_path_blobs.get(path, "")
        if blob:
            added_blob_counts[blob] = added_blob_counts.get(blob, 0) + 1
    renamed_count = sum(
        min(count, added_blob_counts.get(blob, 0))
        for blob, count in deleted_blob_counts.items()
    )
    if scan_stats is not None:
        scan_stats.clear()
        scan_stats.update(
            {
                "scanned_record_count": len(rows),
                "parsed_record_count": parsed_count,
                "reused_record_count": reused_count,
                "deleted_record_count": len(deleted_paths),
                "renamed_record_count": renamed_count,
                "invalidated_record_count": len(deleted_paths),
                "scan_elapsed_seconds": max(0.0, time.monotonic() - started),
                "parse_elapsed_seconds": parse_elapsed,
                "saved_parse_seconds": saved_parse_seconds,
                "deleted_paths": deleted_paths,
            }
        )
    return rows


def _record_symbols(row: Mapping[str, Any]) -> list[str]:
    try:
        value = json.loads(str(row.get("symbols_json") or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip().lower() for item in value if str(item).strip()})


def _ast_evidence_fields(root_relative: str, text: str, symbols: Sequence[str]) -> dict[str, Any]:
    document_text = f"{root_relative}\n{' '.join(sorted(symbols))}\n{text[:12000]}"
    return {
        "evidence_text": text,
        "document_tokens_json": json.dumps(sorted(set(objective_tokens(document_text)))),
        "document_embedding_json": json.dumps(text_embedding(document_text)),
    }


def _nonnegative_seconds(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def persist_objective_ast_dataset(
    *,
    repo_root: Path,
    objective_path: Path,
    dataset_dir: Path,
    dataset_id: str = "objective-ast",
) -> DatasetArtifact:
    """Persist scan AST/symbol records with the optional ipfs_datasets backend."""

    store = ObjectiveDatasetStore(dataset_dir)
    stats: dict[str, Any] = {}
    rows = collect_ast_dataset_records(
        repo_root,
        objective_path=objective_path,
        previous_records=store.load_records(dataset_id),
        scan_stats=stats,
        excluded_roots=(dataset_dir,),
    )
    return store.persist_records(
        dataset_id=dataset_id,
        records=rows,
        scan_stats=stats,
    )


def discover_git_worktrees(repo_root: Path) -> list[Path]:
    """Return repo_root plus nested tracked git worktrees."""

    roots = [repo_root]
    result = subprocess.run(
        ["git", "submodule", "foreach", "--quiet", "printf '%s\\n' \"$sm_path\""],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                roots.append(repo_root / line)
    return list(dict.fromkeys(path.resolve() for path in roots if path.exists()))


def tracked_blob_hashes(repo_root: Path) -> dict[Path, str]:
    """Map clean tracked paths to their Git blob ids.

    Unstaged paths are omitted because the index blob does not describe their
    current source.  Those files fall back to a content read/hash in the
    incremental collector, while unchanged files need no source read at all.
    """

    hashes: dict[Path, str] = {}
    for git_root in discover_git_worktrees(repo_root):
        listed = subprocess.run(
            ["git", "ls-files", "-s", "-z"],
            cwd=git_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if listed.returncode != 0:
            continue
        dirty_result = subprocess.run(
            ["git", "diff-files", "--name-only", "-z"],
            cwd=git_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        dirty = (
            {
                raw.decode("utf-8", errors="surrogateescape")
                for raw in dirty_result.stdout.split(b"\0")
                if raw
            }
            if dirty_result.returncode == 0
            else set()
        )
        for entry in listed.stdout.split(b"\0"):
            if not entry or b"\t" not in entry:
                continue
            metadata, raw_relative = entry.split(b"\t", 1)
            fields = metadata.split()
            if len(fields) < 3 or fields[2] != b"0":
                continue
            relative = raw_relative.decode("utf-8", errors="surrogateescape")
            if relative in dirty or not repo_relative_path_safe(relative):
                continue
            hashes[(git_root / relative).resolve()] = fields[1].decode("ascii", errors="ignore")
    return hashes


def tracked_files(git_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=git_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return [path for path in git_root.rglob("*") if path.is_file()]
    files: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if repo_relative_path_safe(relative):
            path = git_root / relative
            if path.is_file():
                files.append(path)
    return files


def scan_candidate(path: Path, *, repo_root: Path, objective_path: Path) -> bool:
    if path.resolve() == objective_path.resolve():
        return False
    root_relative = repo_relative_path(repo_root, path)
    parts = set(Path(root_relative).parts)
    if parts & SKIP_DIRS:
        return False
    if path.suffix.lower() not in SCAN_SUFFIXES:
        return False
    if "-objective-gap-" in path.name or path.name == "index.json":
        return False
    if {"discovery", "objective_bundles"} & parts:
        return False
    if root_relative.startswith("data/"):
        return False
    try:
        return path.stat().st_size <= 262144
    except OSError:
        return False


def objective_candidate_files(repo_root: Path, *, objective_path: Path) -> list[Path]:
    files: list[Path] = []
    for git_root in discover_git_worktrees(repo_root):
        for path in tracked_files(git_root):
            if scan_candidate(path, repo_root=repo_root, objective_path=objective_path):
                files.append(path)
    return sorted(dict.fromkeys(files), key=lambda path: repo_relative_path(repo_root, path))


def evidence_methods(present_evidence: Mapping[str, Any]) -> list[str]:
    methods: set[str] = set()
    for paths in present_evidence.values():
        values = paths if isinstance(paths, list) else [paths]
        for value in values:
            match = re.search(r"\((path|exact|ast|embedding)(?::[^)]*)?\)\s*$", str(value))
            if match:
                methods.add(match.group(1))
    return sorted(methods)


@dataclass(frozen=True)
class ObjectiveEvidenceIndex:
    """Bounded objective evidence with authority and nominations separated."""

    qualifying: Mapping[str, tuple[str, ...]]
    nominations: Mapping[str, tuple[EvidenceSourceDecision, ...]]
    total_nominations: int
    returned_nominations: int
    omitted_nominations: int
    truncated: bool
    max_nominations_per_requirement: int
    max_nomination_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SOURCE_POLICY_SCHEMA,
            "qualifying": {
                key: list(self.qualifying[key]) for key in sorted(self.qualifying)
            },
            "nominations": {
                key: [item.to_dict() for item in self.nominations[key]]
                for key in sorted(self.nominations)
            },
            "truncation": {
                "truncated": self.truncated,
                "total_nominations": self.total_nominations,
                "returned_nominations": self.returned_nominations,
                "omitted_nominations": self.omitted_nominations,
                "max_nominations_per_requirement": self.max_nominations_per_requirement,
                "max_nomination_bytes": self.max_nomination_bytes,
            },
        }


def evidence_index(
    repo_root: Path,
    *,
    objective_path: Path,
    terms: Sequence[str],
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
    records: Sequence[Mapping[str, Any]] | None = None,
    typed_receipts: Sequence[Mapping[str, Any] | Any] = (),
    source_policy: EvidenceSourcePolicy | None = None,
    repository_tree: str = "",
    policy_id: str = "",
    return_metadata: bool = False,
) -> dict[str, list[str]] | ObjectiveEvidenceIndex:
    normalized_terms = [term for term in dict.fromkeys(str(term).strip() for term in terms) if term]
    evidence = {term: [] for term in normalized_terms}
    selected_policy = source_policy or EvidenceSourcePolicy()
    effective_repository_tree = str(repository_tree or "").strip()
    if typed_receipts and not effective_repository_tree:
        effective_repository_tree = scan_identity(repo_root).tree_id
    nominations: dict[str, list[EvidenceSourceDecision]] = {
        term: [] for term in normalized_terms
    }
    priority = {
        EvidenceMatchKind.TYPED_RECEIPT: 0,
        EvidenceMatchKind.PATH: 1,
        EvidenceMatchKind.EXACT_AST: 2,
        EvidenceMatchKind.EXACT_TEXT: 3,
        EvidenceMatchKind.RETRIEVAL: 4,
        EvidenceMatchKind.SEMANTIC: 5,
    }
    nomination_count = 0
    candidate_cap = max(
        8, selected_policy.max_nominations_per_requirement * 4
    )
    if not normalized_terms:
        if not return_metadata:
            return evidence
        return ObjectiveEvidenceIndex(
            qualifying={},
            nominations={},
            total_nominations=0,
            returned_nominations=0,
            omitted_nominations=0,
            truncated=False,
            max_nominations_per_requirement=selected_policy.max_nominations_per_requirement,
            max_nomination_bytes=selected_policy.max_nomination_bytes,
        )

    def consider(
        term: str,
        *,
        match_kind: EvidenceMatchKind,
        source_path: str = "",
        reference: str,
        typed_receipt: Mapping[str, Any] | Any | None = None,
    ) -> None:
        nonlocal nomination_count
        decision = selected_policy.evaluate(
            term,
            match_kind=match_kind,
            source_path=source_path,
            typed_receipt=typed_receipt,
            repository_tree=effective_repository_tree,
            policy_id=policy_id,
            reference=reference,
        )
        nomination_count += 1
        nominations[term].append(decision)
        if len(nominations[term]) > candidate_cap:
            nominations[term].sort(
                key=lambda item: (
                    0 if item.satisfies else 1,
                    priority[item.match_kind],
                    item.source_path,
                    item.reference,
                )
            )
            del nominations[term][candidate_cap:]
        if (
            decision.satisfies
            and len(evidence[term]) < 3
            and reference not in evidence[term]
        ):
            evidence[term].append(reference)

    for receipt in typed_receipts:
        if isinstance(receipt, Mapping):
            projected = receipt
        else:
            converter = getattr(receipt, "to_dict", None)
            projected = converter() if callable(converter) else {}
        if not isinstance(projected, Mapping):
            continue
        receipt_id = str(
            projected.get("receipt_id")
            or projected.get("evidence_id")
            or projected.get("witness_id")
            or projected.get("provenance_cid")
            or "unknown"
        )
        source_path = str(projected.get("source_path") or projected.get("path") or "")
        reference = f"{receipt_id} (typed_receipt)"
        for term in normalized_terms:
            consider(
                term,
                match_kind=EvidenceMatchKind.TYPED_RECEIPT,
                source_path=source_path,
                reference=reference,
                typed_receipt=projected,
            )

    term_tokens = {term: set(objective_tokens(term)) for term in normalized_terms}
    term_embeddings = {term: text_embedding(term) for term in normalized_terms}

    for term in normalized_terms:
        if not repo_relative_path_safe(term):
            continue
        candidate = repo_root / term
        if candidate.exists():
            reference = f"{Path(term).as_posix()} (path)"
            consider(
                term,
                match_kind=EvidenceMatchKind.PATH,
                source_path=Path(term).as_posix(),
                reference=reference,
            )

    lowered_terms = {term: term.lower() for term in normalized_terms}
    cached_records = list(records) if records is not None else None
    candidates: list[tuple[str, str, set[str], set[str], list[float]]] = []
    if cached_records is not None:
        for row in sorted(cached_records, key=lambda item: str(item.get("root_relative_path") or "")):
            root_relative = str(row.get("root_relative_path") or "")
            if not root_relative:
                continue
            text = str(row.get("evidence_text") or "")
            symbols = set(_record_symbols(row))
            try:
                raw_tokens = json.loads(str(row.get("document_tokens_json") or "[]"))
            except (TypeError, ValueError):
                raw_tokens = []
            document_tokens = {str(item) for item in raw_tokens} if isinstance(raw_tokens, list) else set()
            try:
                raw_embedding = json.loads(str(row.get("document_embedding_json") or "[]"))
                document_embedding = [float(item) for item in raw_embedding] if isinstance(raw_embedding, list) else []
            except (TypeError, ValueError):
                document_embedding = []
            # Legacy/incomplete rows are never silently treated as negative
            # evidence.  Rebuild their cheap derived fields from cached text.
            if not document_embedding or not document_tokens:
                document_text = f"{root_relative}\n{' '.join(sorted(symbols))}\n{text[:12000]}"
                document_embedding = text_embedding(document_text)
                document_tokens = set(objective_tokens(document_text))
            candidates.append((root_relative, text, symbols, document_tokens, document_embedding))
    else:
        for path in objective_candidate_files(repo_root, objective_path=objective_path):
            root_relative = repo_relative_path(repo_root, path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            symbols = symbol_terms(path, text)
            document_text = f"{root_relative}\n{' '.join(sorted(symbols))}\n{text[:12000]}"
            candidates.append(
                (
                    root_relative,
                    text,
                    symbols,
                    set(objective_tokens(document_text)),
                    text_embedding(document_text),
                )
            )

    for root_relative, text, symbols, document_tokens, document_embedding in candidates:
        haystack = f"{root_relative}\n{text}".lower()

        for term, lowered in lowered_terms.items():
            if len(evidence[term]) >= 3:
                continue
            if lowered in haystack:
                consider(
                    term,
                    match_kind=EvidenceMatchKind.EXACT_TEXT,
                    source_path=root_relative,
                    reference=f"{root_relative} (exact)",
                )
                continue
            normalized_symbol = " ".join(objective_tokens(term))
            normalized_symbols = {
                " ".join(objective_tokens(symbol))
                for symbol in symbols
                if " ".join(objective_tokens(symbol))
            }
            if normalized_symbol and normalized_symbol in normalized_symbols:
                consider(
                    term,
                    match_kind=EvidenceMatchKind.EXACT_AST,
                    source_path=root_relative,
                    reference=f"{root_relative} (ast)",
                )
                continue
            overlap = term_tokens[term] & document_tokens
            required_overlap = max(1, min(2, len(term_tokens[term])))
            score = cosine(term_embeddings[term], document_embedding)
            overlap_ratio = len(overlap) / max(1, len(term_tokens[term]))
            threshold = embedding_min_score
            if overlap_ratio >= 0.75:
                threshold = min(threshold, 0.30)
            if len(overlap) >= required_overlap and score >= threshold:
                consider(
                    term,
                    match_kind=EvidenceMatchKind.SEMANTIC,
                    source_path=root_relative,
                    reference=f"{root_relative} (embedding:{score:.2f})",
                )
    if not return_metadata:
        return evidence

    total = nomination_count
    selected: dict[str, tuple[EvidenceSourceDecision, ...]] = {}
    returned = 0
    used_bytes = 0
    for term in normalized_terms:
        ordered = sorted(
            nominations[term],
            key=lambda item: (
                0 if item.satisfies else 1,
                priority[item.match_kind],
                item.source_path,
                item.reference,
            ),
        )
        accepted: list[EvidenceSourceDecision] = []
        for decision in ordered[: selected_policy.max_nominations_per_requirement]:
            encoded = len(
                json.dumps(
                    decision.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if used_bytes + encoded > selected_policy.max_nomination_bytes:
                break
            accepted.append(decision)
            used_bytes += encoded
        selected[term] = tuple(accepted)
        returned += len(accepted)
    return ObjectiveEvidenceIndex(
        qualifying={
            term: tuple(evidence[term]) for term in normalized_terms
        },
        nominations=selected,
        total_nominations=total,
        returned_nominations=returned,
        omitted_nominations=max(0, total - returned),
        truncated=returned < total,
        max_nominations_per_requirement=selected_policy.max_nominations_per_requirement,
        max_nomination_bytes=selected_policy.max_nomination_bytes,
    )


def goal_graph(goals: Sequence[ObjectiveGoal]) -> dict[str, Any]:
    """Materialize objective hierarchy, lifecycle, and proof requirements.

    ``nodes``, ``edges``, ``children``, ``roots``, and ``depths`` retain their
    historical shapes.  The additional projections make the graph safe for
    lifecycle-aware callers: no consumer needs to infer completion from a lack
    of scheduled tasks, and proof references remain attached to their goal.
    """

    nodes = {goal.goal_id: goal for goal in goals if goal.goal_id}
    edges: list[dict[str, str]] = []
    children: dict[str, list[str]] = {goal_id: [] for goal_id in nodes}
    roots: list[str] = []
    for goal_id, goal in nodes.items():
        parents = [parent for parent in goal.parent_goal_ids if parent]
        if not parents:
            roots.append(goal_id)
        for parent in parents:
            edges.append({"from": parent, "to": goal_id, "kind": "refines"})
            children.setdefault(parent, []).append(goal_id)

    depths: dict[str, int] = {}

    def depth_for(goal_id: str, seen: set[str] | None = None) -> int:
        if goal_id in depths:
            return depths[goal_id]
        seen = set(seen or set())
        if goal_id in seen:
            depths[goal_id] = 0
            return 0
        seen.add(goal_id)
        parents = nodes.get(goal_id, ObjectiveGoal(goal_id, "", {})).parent_goal_ids
        if not parents:
            depths[goal_id] = 0
            return 0
        known_parents = [parent for parent in parents if parent in nodes]
        if not known_parents:
            depths[goal_id] = 1
            return 1
        depths[goal_id] = 1 + max(depth_for(parent, seen) for parent in known_parents)
        return depths[goal_id]

    for goal_id in nodes:
        depth_for(goal_id)

    node_details: dict[str, dict[str, Any]] = {}
    state_counts: dict[str, int] = {}
    schedulable_goal_ids: list[str] = []
    terminal_goal_ids: list[str] = []
    evidence_nodes: list[dict[str, Any]] = []
    evidence_edges: list[dict[str, str]] = []
    for goal_id, goal in sorted(nodes.items()):
        state = goal.lifecycle_state_value
        state_counts[state] = state_counts.get(state, 0) + 1
        if goal.is_schedulable:
            schedulable_goal_ids.append(goal_id)
        if goal.is_terminal:
            terminal_goal_ids.append(goal_id)
        required_evidence = goal.required_evidence
        node_details[goal_id] = {
            "goal_id": goal_id,
            "title": goal.title,
            "status": goal.status,
            "lifecycle_state": state,
            "schedulable": goal.is_schedulable,
            "terminal": goal.is_terminal,
            "parents": goal.parent_goal_ids,
            "required_evidence": required_evidence,
            "completion_evidence": goal.completion_evidence_metadata,
        }
        for term in required_evidence:
            evidence_id = "evidence:" + sha1(
                f"{goal_id}\0{term}".encode("utf-8")
            ).hexdigest()[:16]
            evidence_nodes.append(
                {
                    "id": evidence_id,
                    "goal_id": goal_id,
                    "acceptance_criterion": term,
                }
            )
            evidence_edges.append(
                {
                    "from": goal_id,
                    "to": evidence_id,
                    "kind": "requires_evidence",
                }
            )
        for record in goal.completion_evidence_records:
            criterion = str(record.get("acceptance_criterion") or "").strip()
            provenance_cid = str(
                record.get("provenance_cid") or record.get("receipt_cid") or ""
            ).strip()
            proof_id = "completion-evidence:" + sha1(
                f"{goal_id}\0{criterion}\0{provenance_cid}".encode("utf-8")
            ).hexdigest()[:16]
            evidence_nodes.append(
                {
                    "id": proof_id,
                    "goal_id": goal_id,
                    "acceptance_criterion": criterion,
                    "kind": "completion_evidence",
                    "producing_task_or_scan": str(
                        record.get("producing_task_or_scan")
                        or record.get("producer_id")
                        or ""
                    ),
                    "validation_receipt": record.get("validation_receipt"),
                    "repository_tree": str(
                        record.get("repository_tree") or record.get("tree_id") or ""
                    ),
                    "freshness": record.get("freshness"),
                    "provenance_cid": provenance_cid,
                }
            )
            evidence_edges.append(
                {"from": goal_id, "to": proof_id, "kind": "supported_by"}
            )

    lifecycle = {
        "state_counts": dict(sorted(state_counts.items())),
        "schedulable_goal_ids": schedulable_goal_ids,
        "terminal_goal_ids": terminal_goal_ids,
        "nonterminal_goal_ids": sorted(set(nodes) - set(terminal_goal_ids)),
    }

    return {
        "nodes": sorted(nodes),
        "node_details": node_details,
        "edges": edges,
        "children": {key: sorted(value) for key, value in children.items() if value},
        "roots": sorted(roots),
        "depths": depths,
        "lifecycle": lifecycle,
        "state_counts": lifecycle["state_counts"],
        "schedulable_goal_ids": lifecycle["schedulable_goal_ids"],
        "terminal_goal_ids": lifecycle["terminal_goal_ids"],
        "evidence_nodes": evidence_nodes,
        "evidence_edges": evidence_edges,
    }


def objective_heap_content_id(text: str) -> str:
    """Return the exact content identity used to fence heap preview writes."""

    if not isinstance(text, str):
        raise TypeError("objective heap text must be a string")
    return "objective-heap:sha256:" + sha256(text.encode("utf-8")).hexdigest()


def objective_goal_content_id(goal: ObjectiveGoal | Mapping[str, Any]) -> str:
    """Return a stable semantic identity for one parsed heap goal.

    Unlike :func:`objective_heap_content_id`, this identity ignores markdown
    layout.  It is used to freeze the root while allowing unrelated metadata
    or child records to be appended transactionally.
    """

    if isinstance(goal, ObjectiveGoal):
        goal_id, title, fields = goal.goal_id, goal.title, goal.fields
    elif isinstance(goal, Mapping):
        goal_id = str(goal.get("goal_id") or goal.get("id") or "")
        title = str(goal.get("title") or "")
        raw_fields = goal.get("fields") or {}
        if not isinstance(raw_fields, Mapping):
            raise TypeError("objective goal fields must be a mapping")
        fields = {str(key): str(value) for key, value in raw_fields.items()}
    else:
        raise TypeError("goal must be ObjectiveGoal or a mapping")
    material = {
        "goal_id": str(goal_id or "").strip(),
        "title": " ".join(str(title or "").split()),
        "fields": {
            normalize_field_key(str(key)): " ".join(str(value or "").split())
            for key, value in sorted(fields.items(), key=lambda pair: normalize_field_key(str(pair[0])))
        },
    }
    return "objective-goal:sha256:" + sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_goal_hierarchy_errors(goals: Sequence[ObjectiveGoal]) -> list[str]:
    """Return structural errors which make a heap unsafe to extend."""

    errors: list[str] = []
    ids = [goal.goal_id for goal in goals]
    duplicate_ids = sorted(
        goal_id for goal_id in set(ids) if ids.count(goal_id) > 1
    )
    if duplicate_ids:
        errors.append("duplicate goal ids: " + ", ".join(duplicate_ids))
    nodes = {goal.goal_id: goal for goal in goals}
    missing_edges = sorted(
        {
            f"{goal.goal_id}->{parent}"
            for goal in goals
            for parent in goal.parent_goal_ids
            if parent and parent not in nodes
        }
    )
    if missing_edges:
        errors.append("unresolved parent edges: " + ", ".join(missing_edges))

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(goal_id: str) -> None:
        current = state.get(goal_id, 0)
        if current == 2:
            return
        if current == 1:
            try:
                start = stack.index(goal_id)
            except ValueError:
                start = 0
            cycle = stack[start:] + [goal_id]
            errors.append("parent cycle: " + " -> ".join(cycle))
            return
        state[goal_id] = 1
        stack.append(goal_id)
        for parent in nodes[goal_id].parent_goal_ids:
            if parent in nodes:
                visit(parent)
        stack.pop()
        state[goal_id] = 2

    for goal_id in sorted(nodes):
        visit(goal_id)
    return sorted(set(errors))


def _materialized_goal_fields(
    proposal: ObjectiveWorkProposal,
    *,
    parent_goal_ids: Sequence[str],
    graph_depth: int,
    lifecycle_owner: str,
) -> dict[str, str]:
    """Project all durable proposal semantics into objective-heap fields."""

    fields: dict[str, str] = {
        "Status": "active",
        "Goal": "; ".join(proposal.parent_objective_terms) or proposal.title,
        "Evidence": ", ".join(proposal.expected_evidence_delta),
        "Evidence requirements JSON": json.dumps(
            list(proposal.expected_evidence_delta),
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "Parents": ", ".join(parent_goal_ids),
        "Parent goal IDs JSON": json.dumps(
            list(parent_goal_ids),
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "Depends on": ", ".join(proposal.dependencies),
        "Dependencies JSON": json.dumps(
            list(proposal.dependencies),
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "Outputs": ", ".join(proposal.predicted_files),
        "Predicted files JSON": json.dumps(
            list(proposal.predicted_files),
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "Predicted symbols": ", ".join(proposal.predicted_symbols),
        "Predicted symbols JSON": json.dumps(
            list(proposal.predicted_symbols),
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "Validation": "; ".join(proposal.validation_commands),
        "Validation commands JSON": json.dumps(
            list(proposal.validation_commands),
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "Graph depth": str(graph_depth),
        "Canonical proposal ID": proposal.canonical_id,
        "Semantic key": proposal.semantic_key,
        "Proposal kind": proposal.kind.value,
        "Proposal source": proposal.source,
        "Proposal source ID": proposal.source_id,
        "Lifecycle owner": lifecycle_owner,
    }
    if proposal.rationale:
        fields["Refinement"] = proposal.rationale
    # Keep rows present even when their value is empty.  This gives reviewers
    # an explicit accounting of dependencies/evidence/validation instead of
    # making absence indistinguishable from a lossy projection.
    return fields


def render_objective_work_goal_block(
    proposal: ObjectiveWorkProposal | Mapping[str, Any],
    *,
    parent_goal_ids: Sequence[str],
    graph_depth: int,
    lifecycle_owner: str = "objective_daemon",
) -> str:
    """Render one proposal as a canonical objective-heap markdown block."""

    item = (
        proposal
        if isinstance(proposal, ObjectiveWorkProposal)
        else ObjectiveWorkProposal.from_dict(proposal)
    )
    if item.kind not in {ObjectiveWorkKind.GOAL, ObjectiveWorkKind.SUBGOAL}:
        raise ValueError("only goal and subgoal proposals can become objective goals")
    if isinstance(graph_depth, bool) or int(graph_depth) < 0:
        raise ValueError("graph_depth must be a non-negative integer")
    owner = " ".join(str(lifecycle_owner or "").split())
    if not owner:
        raise ValueError("lifecycle_owner must be non-empty")
    parents = _objective_work_strings(parent_goal_ids)
    rows = [f"## {item.canonical_id} {item.title}", ""]
    for key, value in _materialized_goal_fields(
        item,
        parent_goal_ids=parents,
        graph_depth=int(graph_depth),
        lifecycle_owner=owner,
    ).items():
        rows.append(f"- {key}: {value}")
    return "\n".join(rows).rstrip() + "\n"


def preview_objective_goal_materialization(
    objective_text: str,
    proposals: Iterable[ObjectiveWorkProposal | Mapping[str, Any]],
    *,
    policy: ObjectiveGoalMaterializationPolicy | Mapping[str, Any] | None = None,
    limits: ObjectiveGenerationLimits | Mapping[str, Any] | None = None,
    root_goal_id: str = "",
    expected_heap_content_id: str = "",
    expected_root_content_id: str = "",
    lifecycle_owner: str = "objective_daemon",
) -> ObjectiveGoalMaterializationPreview:
    """Preview an all-or-nothing GOAL/SUBGOAL objective-heap append.

    The function is deliberately pure: it never opens or writes a path.  A
    tracker can persist ``candidate_text`` only after rechecking
    ``base_heap_content_id`` under its lease.  The preview independently
    revalidates canonical identities, hierarchy, lifecycle, finite-generation
    limits, semantic uniqueness, root immutability, and every rendered block.

    ``policy`` is the preferred interface.  The keyword arguments are a
    compatibility convenience for callers which already hold generation
    limits and frozen-root values separately.
    """

    if not isinstance(objective_text, str):
        raise TypeError("objective_text must be a string")
    if policy is None:
        selected_limits: ObjectiveGenerationLimits
        if limits is None:
            selected_limits = ObjectiveGenerationLimits()
        elif isinstance(limits, ObjectiveGenerationLimits):
            selected_limits = limits
        elif isinstance(limits, Mapping):
            selected_limits = ObjectiveGenerationLimits(**dict(limits))
        else:
            raise TypeError("limits must be ObjectiveGenerationLimits or a mapping")
        selected_policy = ObjectiveGoalMaterializationPolicy(
            limits=selected_limits,
            root_goal_id=root_goal_id,
            expected_heap_content_id=expected_heap_content_id,
            expected_root_content_id=expected_root_content_id,
            lifecycle_owner=lifecycle_owner,
        )
    elif isinstance(policy, ObjectiveGoalMaterializationPolicy):
        selected_policy = policy
    elif isinstance(policy, Mapping):
        selected_policy = ObjectiveGoalMaterializationPolicy(**dict(policy))
    else:
        raise TypeError("policy must be ObjectiveGoalMaterializationPolicy or a mapping")
    if limits is not None and policy is not None:
        raise ValueError("supply limits through policy or limits, not both")

    base_content_id = objective_heap_content_id(objective_text)
    goals = parse_goal_heap(objective_text)
    graph = goal_graph(goals)
    goals_by_id = {goal.goal_id: goal for goal in goals}
    fatal_reasons = _strict_goal_hierarchy_errors(goals)
    if (
        selected_policy.expected_heap_content_id
        and selected_policy.expected_heap_content_id != base_content_id
    ):
        fatal_reasons.append("stale objective heap content")

    roots = list(graph["roots"])
    selected_root_id = selected_policy.root_goal_id
    if selected_root_id:
        if selected_root_id not in goals_by_id:
            fatal_reasons.append(f"frozen root {selected_root_id!r} is not present")
        elif selected_root_id not in roots:
            fatal_reasons.append(f"frozen root {selected_root_id!r} is not a heap root")
    elif len(roots) == 1:
        selected_root_id = roots[0]
    elif roots:
        fatal_reasons.append("root_goal_id is required for a multi-root objective heap")

    root_content_id = (
        objective_goal_content_id(goals_by_id[selected_root_id])
        if selected_root_id in goals_by_id
        else ""
    )
    if (
        selected_policy.expected_root_content_id
        and selected_policy.expected_root_content_id != root_content_id
    ):
        fatal_reasons.append("frozen objective root content changed")

    normalized: list[ObjectiveWorkProposal] = []
    rejected: list[ObjectiveGoalMaterializationRejection] = []
    for raw in proposals:
        try:
            item = (
                raw
                if isinstance(raw, ObjectiveWorkProposal)
                else ObjectiveWorkProposal.from_dict(raw)
            )
        except (TypeError, ValueError) as exc:
            rejected.append(
                ObjectiveGoalMaterializationRejection(
                    reason="invalid_proposal",
                    canonical_id="",
                    detail=str(exc),
                )
            )
            continue
        if item.kind not in {ObjectiveWorkKind.GOAL, ObjectiveWorkKind.SUBGOAL}:
            rejected.append(
                ObjectiveGoalMaterializationRejection(
                    reason="unsupported_kind",
                    canonical_id=item.canonical_id,
                    source_id=item.source_id,
                    detail="only goal and subgoal proposals alter the objective heap",
                )
            )
            continue
        normalized.append(item)

    def proposal_sort_key(item: ObjectiveWorkProposal) -> tuple[Any, ...]:
        return (
            item.depth,
            item.parent_goal_id.casefold(),
            0 if item.kind is ObjectiveWorkKind.GOAL else 1,
            item.semantic_key,
            item.canonical_id,
        )

    normalized.sort(key=proposal_sort_key)
    candidate_ids = {item.canonical_id for item in normalized}
    groups: dict[str, list[ObjectiveWorkProposal]] = {}
    for item in normalized:
        groups.setdefault(item.canonical_id, []).append(item)
    representatives = {key: values[0] for key, values in groups.items()}
    prerequisites: dict[str, set[str]] = {}
    dependents: dict[str, set[str]] = {key: set() for key in representatives}
    for canonical_id, item in representatives.items():
        refs = {
            ref
            for ref in (item.parent_goal_id, *item.dependencies)
            if ref in candidate_ids
        }
        prerequisites[canonical_id] = refs
        for ref in refs:
            dependents.setdefault(ref, set()).add(canonical_id)
    ready_ids: list[tuple[tuple[Any, ...], str]] = [
        (proposal_sort_key(item), canonical_id)
        for canonical_id, item in representatives.items()
        if not prerequisites[canonical_id]
    ]
    heapq.heapify(ready_ids)
    ordered_ids: list[str] = []
    while ready_ids:
        _sort_key, canonical_id = heapq.heappop(ready_ids)
        ordered_ids.append(canonical_id)
        for dependent_id in sorted(dependents.get(canonical_id, ())):
            prerequisites[dependent_id].discard(canonical_id)
            if not prerequisites[dependent_id]:
                heapq.heappush(
                    ready_ids,
                    (proposal_sort_key(representatives[dependent_id]), dependent_id),
                )
    dependency_cycle_ids = set(representatives) - set(ordered_ids)
    ordered_ids.extend(
        sorted(
            dependency_cycle_ids,
            key=lambda canonical_id: proposal_sort_key(representatives[canonical_id]),
        )
    )
    normalized = [
        item
        for canonical_id in ordered_ids
        for item in groups[canonical_id]
    ]
    limits_policy = selected_policy.limits
    current_depths = {str(key): int(value) for key, value in graph["depths"].items()}
    child_counts = {
        # Breadth is a structural bound, not an open-work bound.  Terminal
        # children remain part of the hierarchy and cannot be ignored to grow
        # an unbounded branch over repeated admission cycles.
        str(parent): sum(1 for child_id in children if child_id in goals_by_id)
        for parent, children in graph["children"].items()
    }
    current_open = sum(1 for goal in goals if goal.is_schedulable)
    existing_semantic_keys = {
        str(goal.fields.get("semantic_key") or "").strip()
        for goal in goals
        if str(goal.fields.get("semantic_key") or "").strip()
    }
    exact_ids = set(goals_by_id)
    accepted_proposals: list[ObjectiveWorkProposal] = []
    materialized: list[MaterializedObjectiveGoal] = []
    consumed_tokens = 0

    def reject(item: ObjectiveWorkProposal, reason: str, detail: str) -> None:
        rejected.append(
            ObjectiveGoalMaterializationRejection(
                reason=reason,
                canonical_id=item.canonical_id,
                source_id=item.source_id,
                detail=detail,
            )
        )

    if not fatal_reasons:
        for item in normalized:
            if item.canonical_id in dependency_cycle_ids:
                reject(
                    item,
                    "dependency_cycle",
                    "proposal parent/dependency references form an intra-batch cycle",
                )
                continue
            if item.canonical_id in exact_ids:
                reject(item, "canonical_duplicate", "goal identity already exists in the heap")
                continue
            if item.semantic_key in existing_semantic_keys:
                reject(item, "semantic_duplicate", "semantic work already exists in the heap")
                continue
            equivalent = next(
                (
                    prior
                    for prior in accepted_proposals
                    if _objective_work_similarity(item, prior)
                    >= limits_policy.semantic_similarity_threshold
                ),
                None,
            )
            if equivalent is not None:
                reject(
                    item,
                    "semantic_duplicate",
                    f"equivalent to {equivalent.canonical_id}",
                )
                continue
            unresolved_dependencies = sorted(
                dependency
                for dependency in item.dependencies
                if dependency in candidate_ids and dependency not in exact_ids
            )
            if unresolved_dependencies:
                reject(
                    item,
                    "unresolved_dependency",
                    "intra-batch dependencies were not admitted: "
                    + ", ".join(unresolved_dependencies),
                )
                continue
            parent_id = item.parent_goal_id
            if item.kind is ObjectiveWorkKind.GOAL and (
                not parent_id or parent_id in selected_policy.root_parent_aliases
            ):
                parent_id = selected_root_id
            if not parent_id:
                # Creating the first root is supported, but a populated heap is
                # never allowed to gain an accidental unowned root.
                if goals_by_id or materialized:
                    reject(item, "unresolved_parent", "proposal has no materializable parent")
                    continue
                actual_depth = 0
                parent_ids: tuple[str, ...] = ()
            else:
                parent = goals_by_id.get(parent_id)
                if parent is None:
                    detail = (
                        "parent proposal has not been admitted"
                        if parent_id in candidate_ids
                        else f"parent {parent_id!r} is not in the objective heap"
                    )
                    reject(item, "unresolved_parent", detail)
                    continue
                if selected_policy.require_schedulable_parent and not parent.is_schedulable:
                    reject(
                        item,
                        "parent_lifecycle",
                        f"parent {parent_id!r} is {parent.lifecycle_state_value}",
                    )
                    continue
                parent_ids = (parent_id,)
                actual_depth = current_depths[parent_id] + 1
            if item.depth != actual_depth:
                reject(
                    item,
                    "depth_mismatch",
                    f"proposal depth {item.depth} does not match hierarchy depth {actual_depth}",
                )
                continue
            if actual_depth > limits_policy.max_depth:
                reject(
                    item,
                    "depth_limit",
                    f"depth {actual_depth} exceeds {limits_policy.max_depth}",
                )
                continue
            if item.retry_count > limits_policy.max_retries:
                reject(
                    item,
                    "retry_limit",
                    f"retry {item.retry_count} exceeds {limits_policy.max_retries}",
                )
                continue
            if len(materialized) >= limits_policy.max_new_work:
                reject(
                    item,
                    "cycle_limit",
                    f"cycle allows {limits_policy.max_new_work} new records",
                )
                continue
            if current_open + len(materialized) >= limits_policy.max_open_work:
                reject(
                    item,
                    "open_work_limit",
                    f"open work limit is {limits_policy.max_open_work}",
                )
                continue
            parent_key = parent_id or "__root__"
            if child_counts.get(parent_key, 0) >= limits_policy.max_breadth_per_parent:
                reject(
                    item,
                    "breadth_limit",
                    f"parent allows {limits_policy.max_breadth_per_parent} children",
                )
                continue
            if consumed_tokens + item.estimated_tokens > limits_policy.token_budget:
                reject(
                    item,
                    "token_budget",
                    f"cycle token budget is {limits_policy.token_budget}",
                )
                continue

            block = render_objective_work_goal_block(
                item,
                parent_goal_ids=parent_ids,
                graph_depth=actual_depth,
                lifecycle_owner=selected_policy.lifecycle_owner,
            )
            parsed = parse_goal_heap(block)
            if len(parsed) != 1 or parsed[0].goal_id != item.canonical_id:
                reject(item, "render_validation", "rendered goal did not round-trip")
                continue
            projected_goal = parsed[0]
            projected_parents = tuple(projected_goal.parent_goal_ids)
            if projected_parents != parent_ids:
                reject(item, "render_validation", "rendered parent hierarchy changed")
                continue
            round_trip_values: tuple[tuple[str, Any, Any], ...] = (
                ("dependencies", tuple(projected_goal.dependencies), item.dependencies),
                (
                    "evidence requirements",
                    tuple(projected_goal.required_evidence),
                    item.expected_evidence_delta,
                ),
                (
                    "predicted files",
                    tuple(projected_goal.predicted_files),
                    item.predicted_files,
                ),
                (
                    "predicted symbols",
                    tuple(projected_goal.predicted_symbols),
                    item.predicted_symbols,
                ),
                (
                    "validation commands",
                    tuple(projected_goal.validation_commands),
                    item.validation_commands,
                ),
                (
                    "canonical proposal identity",
                    projected_goal.canonical_proposal_id,
                    item.canonical_id,
                ),
                ("semantic identity", projected_goal.semantic_key, item.semantic_key),
                (
                    "lifecycle owner",
                    projected_goal.lifecycle_owner,
                    selected_policy.lifecycle_owner,
                ),
                (
                    "proposal kind",
                    str(projected_goal.fields.get("proposal_kind") or ""),
                    item.kind.value,
                ),
                (
                    "graph depth",
                    str(projected_goal.fields.get("graph_depth") or ""),
                    str(actual_depth),
                ),
            )
            changed_fields = [
                name
                for name, actual, expected in round_trip_values
                if actual != expected
            ]
            if changed_fields:
                reject(
                    item,
                    "render_validation",
                    "rendered proposal metadata changed: " + ", ".join(changed_fields),
                )
                continue
            materialized.append(
                MaterializedObjectiveGoal(
                    proposal=item,
                    goal=projected_goal,
                    rendered_block=block,
                    graph_depth=actual_depth,
                    parent_goal_ids=parent_ids,
                )
            )
            accepted_proposals.append(item)
            exact_ids.add(item.canonical_id)
            existing_semantic_keys.add(item.semantic_key)
            goals_by_id[item.canonical_id] = projected_goal
            current_depths[item.canonical_id] = actual_depth
            child_counts[parent_key] = child_counts.get(parent_key, 0) + 1
            consumed_tokens += item.estimated_tokens

    proposed_text = objective_text
    if materialized:
        separator = "\n\n" if objective_text.rstrip() else ""
        proposed_text = (
            objective_text.rstrip()
            + separator
            + "\n\n".join(item.rendered_block.strip() for item in materialized)
            + "\n"
        )
        projected_goals = parse_goal_heap(proposed_text)
        projected_errors = _strict_goal_hierarchy_errors(projected_goals)
        if projected_errors:
            fatal_reasons.extend(f"projected heap: {item}" for item in projected_errors)
        elif selected_root_id:
            projected_root = next(
                (goal for goal in projected_goals if goal.goal_id == selected_root_id),
                None,
            )
            if (
                projected_root is None
                or objective_goal_content_id(projected_root) != root_content_id
            ):
                fatal_reasons.append("projected heap changed the frozen root")

    fatal_reasons = sorted(set(fatal_reasons))
    if fatal_reasons or (selected_policy.atomic and rejected):
        candidate_text = objective_text
    else:
        candidate_text = proposed_text
    return ObjectiveGoalMaterializationPreview(
        base_heap_content_id=base_content_id,
        root_goal_id=selected_root_id,
        root_content_id=root_content_id,
        materialized=tuple(materialized),
        rejected=tuple(
            sorted(
                rejected,
                key=lambda item: (
                    item.canonical_id,
                    item.reason,
                    item.source_id,
                    item.detail,
                ),
            )
        ),
        fatal_reasons=tuple(fatal_reasons),
        candidate_text=candidate_text,
        policy=selected_policy,
    )


# Compatibility phrasing used by generation-ledger and tracker callers.
preview_objective_work_materialization = preview_objective_goal_materialization


def priority_rank(value: str) -> int:
    normalized = str(value or "").strip().upper()
    if normalized.startswith("P"):
        try:
            return max(0, int(normalized[1:]))
        except ValueError:
            pass
    return 9


def _task_record_mapping(task: Any) -> dict[str, Any]:
    if isinstance(task, Mapping):
        return dict(task)
    if hasattr(task, "to_dict"):
        value = task.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if hasattr(task, "__dataclass_fields__"):
        return asdict(task)
    return {
        name: getattr(task, name)
        for name in dir(task)
        if not name.startswith("_") and not callable(getattr(task, name, None))
    }


def _dependency_values(task: Mapping[str, Any], *names: str) -> list[str]:
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_"): value
        for key, value in task.items()
    }
    values: list[str] = []
    for name in names:
        value = normalized.get(name)
        if value in (None, "", [], ()):
            continue
        if isinstance(value, str):
            candidates = re.split(r"[,;]", value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            text = " ".join(str(candidate).strip().split())
            if text and text.lower() not in {"none", "n/a"} and text not in values:
                values.append(text)
    return values


def _task_created_at_ms(task: Mapping[str, Any]) -> int:
    for name in ("created_at_ms", "queued_at_ms", "submitted_at_ms"):
        value = task.get(name)
        try:
            if value not in (None, ""):
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    for name in ("created_at", "queued_at", "submitted_at"):
        value = str(task.get(name) or "").strip()
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0, int(parsed.timestamp() * 1000))
        except ValueError:
            continue
    return 0


def _task_duration(task: Mapping[str, Any]) -> int:
    for name in ("estimated_duration", "duration", "work_item_count", "weight"):
        try:
            value = int(task.get(name) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 1


def _successful_merge_receipt_cids(
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> set[str]:
    succeeded: set[str] = set()
    if isinstance(merge_receipts, Mapping):
        receipt_items: list[tuple[str, Any]] = list(merge_receipts.items())
    else:
        receipt_items = [("", receipt) for receipt in merge_receipts]
    for key, raw_receipt in receipt_items:
        receipt = dict(raw_receipt) if isinstance(raw_receipt, Mapping) else {"status": raw_receipt}
        reference = str(
            receipt.get("canonical_task_cid")
            or receipt.get("task_cid")
            or receipt.get("task_id")
            or key
            or ""
        ).strip()
        status = str(
            receipt.get("merge_status")
            or receipt.get("status")
            or receipt.get("outcome")
            or receipt.get("result")
            or ""
        ).strip().lower()
        successful = receipt.get("succeeded") is True or receipt.get("success") is True
        if successful or status in SUCCESSFUL_MERGE_RECEIPT_STATUSES:
            cid = aliases.get(reference, reference)
            if cid:
                succeeded.add(cid)
    return succeeded


def _cycle_components(nodes: Iterable[str], adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    """Return cyclic strongly-connected components in deterministic order."""

    node_list = sorted(set(nodes))
    visited: set[str] = set()
    finish_order: list[str] = []
    # Iterative Kosaraju traversal avoids recursion failures on malformed,
    # machine-generated graphs with thousands of nodes.
    for root in node_list:
        if root in visited:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in sorted(adjacency.get(node, set()), reverse=True):
                if child not in visited:
                    stack.append((child, False))

    reverse: dict[str, set[str]] = {node: set() for node in node_list}
    for source in node_list:
        for target in adjacency.get(source, set()):
            if target in reverse:
                reverse[target].add(source)
    assigned: set[str] = set()
    components: list[list[str]] = []
    for root in reversed(finish_order):
        if root in assigned:
            continue
        assigned.add(root)
        component: list[str] = []
        stack = [(root, False)]
        while stack:
            node, _unused = stack.pop()
            component.append(node)
            for parent in sorted(reverse[node], reverse=True):
                if parent not in assigned:
                    assigned.add(parent)
                    stack.append((parent, False))
        component.sort()
        if len(component) > 1 or (component and component[0] in adjacency.get(component[0], set())):
            components.append(component)
    return sorted(components, key=lambda item: tuple(item))


def critical_path_schedule(
    graph: TaskDependencyGraph,
    *,
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] = (),
    now: datetime | int | None = None,
) -> list[TaskScheduleRecord]:
    """Schedule all tasks by critical path, slack, unlock value, age, and priority.

    Prerequisite completion is deliberately defined by a successful merge
    receipt, not by mutable todo status.  Invalid components remain visible at
    the end of the schedule but cannot be claimed.
    """

    aliases: dict[str, str] = {}
    for cid, node in graph.nodes.items():
        aliases[cid] = cid
        if node.task_id:
            aliases[node.task_id] = cid
    succeeded = _successful_merge_receipt_cids(merge_receipts, aliases)
    incoming: dict[str, set[str]] = {cid: set() for cid in graph.nodes}
    outgoing: dict[str, set[str]] = {cid: set() for cid in graph.nodes}
    for edge in graph.edges:
        if edge.source_task_cid in graph.nodes and edge.target_task_cid in graph.nodes:
            incoming[edge.target_task_cid].add(edge.source_task_cid)
            outgoing[edge.source_task_cid].add(edge.target_task_cid)

    invalid = set(graph.invalid_task_cids)
    invalid.update(item.task_cid for item in graph.repair_evidence if item.task_cid in graph.nodes)
    unresolved_nodes = set(graph.nodes) - succeeded
    unresolved_adjacency = {
        cid: {child for child in outgoing[cid] if child in unresolved_nodes}
        for cid in unresolved_nodes
    }
    cycle_nodes = {
        cid
        for group in _cycle_components(unresolved_nodes, unresolved_adjacency)
        for cid in group
    }
    invalid.update(cycle_nodes)

    indegree = {
        cid: len([parent for parent in incoming[cid] if parent not in invalid])
        for cid in graph.nodes
        if cid not in invalid
    }
    ready = sorted(cid for cid, count in indegree.items() if count == 0)
    topo: list[str] = []
    while ready:
        cid = ready.pop(0)
        topo.append(cid)
        for child in sorted(outgoing[cid]):
            if child not in indegree:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    longest_to_finish: dict[str, int] = {}
    descendants: dict[str, set[str]] = {}
    for cid in reversed(topo):
        valid_children = [child for child in outgoing[cid] if child not in invalid]
        longest_to_finish[cid] = graph.nodes[cid].estimated_duration + max(
            (longest_to_finish.get(child, 0) for child in valid_children), default=0
        )
        unlocked: set[str] = set(valid_children)
        for child in valid_children:
            unlocked.update(descendants.get(child, set()))
        descendants[cid] = unlocked

    earliest_start: dict[str, int] = {}
    for cid in topo:
        earliest_start[cid] = max(
            (
                earliest_start.get(parent, 0) + graph.nodes[parent].estimated_duration
                for parent in incoming[cid]
                if parent not in invalid
            ),
            default=0,
        )
    project_duration = max(
        (earliest_start.get(cid, 0) + graph.nodes[cid].estimated_duration for cid in topo),
        default=0,
    )
    if isinstance(now, datetime):
        current_ms = int(now.timestamp() * 1000)
    elif now is None:
        current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    else:
        current_ms = int(now)

    records: list[TaskScheduleRecord] = []
    for cid, node in graph.nodes.items():
        blockers = sorted(parent for parent in incoming[cid] if parent not in succeeded)
        claimable = cid not in invalid and node.status not in SUCCESSFUL_MERGE_RECEIPT_STATUSES and not blockers
        critical_length = longest_to_finish.get(cid, 0)
        slack = max(0, project_duration - earliest_start.get(cid, 0) - critical_length)
        unlock_value = len(descendants.get(cid, set()))
        age_seconds = max(0, (current_ms - node.created_at_ms) // 1000) if node.created_at_ms else 0
        # The score is informational and stable.  Ordering uses the full tuple
        # below so no weighting can hide a critical-path distinction.
        score = (
            critical_length * 1_000_000
            + unlock_value * 10_000
            + min(age_seconds, 999_999)
            + node.objective_priority * 100
            - slack * 1_000
        )
        sort_key: list[Any] = [
            0 if claimable else 1,
            -critical_length,
            slack,
            -unlock_value,
            -age_seconds,
            -node.objective_priority,
            node.task_id,
            cid,
        ]
        records.append(
            TaskScheduleRecord(
                task_cid=cid,
                task_id=node.task_id,
                claimable=claimable,
                blocking_task_cids=blockers,
                critical_path_length=critical_length,
                slack=slack,
                downstream_unlock_value=unlock_value,
                age_seconds=age_seconds,
                objective_priority=node.objective_priority,
                score=score,
                sort_key=sort_key,
            )
        )
    records.sort(key=lambda item: tuple(item.sort_key))
    return records


def materialize_task_dependency_dag(
    tasks: Sequence[Any],
    *,
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] = (),
    now: datetime | int | None = None,
    max_repair_evidence: int = 64,
) -> TaskDependencyGraph:
    """Materialize typed task prerequisites and their critical-path schedule.

    Supported metadata is intentionally permissive because records arrive from
    markdown, vector indexes, and Profile G adapters.  Direct task ids/CIDs and
    producer/consumer artifact declarations are both accepted.
    """

    limit = max(0, int(max_repair_evidence))
    raw_tasks = [_task_record_mapping(task) for task in tasks]
    nodes: dict[str, TaskDependencyNode] = {}
    records_by_cid: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    repair: list[DependencyRepairEvidence] = []
    invalid_task_cids: set[str] = set()

    def add_repair(kind: str, cid: str, reference: str, message: str, provenance: Mapping[str, Any]) -> None:
        if cid:
            invalid_task_cids.add(cid)
        if len(repair) >= limit:
            return
        node = nodes.get(cid)
        repair.append(
            DependencyRepairEvidence(
                kind=kind,
                task_cid=cid,
                task_id=node.task_id if node else "",
                reference=reference,
                message=message,
                provenance=dict(provenance),
            )
        )

    for index, task in enumerate(raw_tasks):
        task_id = str(task.get("task_id") or task.get("id") or "").strip()
        task_status = str(task.get("status") or "todo").strip().lower()
        provided_cid = str(task.get("canonical_task_cid") or task.get("task_cid") or "").strip()
        if provided_cid:
            cid = provided_cid
        else:
            identity_input = dict(task)
            if not any(
                identity_input.get(name)
                for name in (
                    "canonical_task_key",
                    "dedupe_key",
                    "title",
                    "summary",
                    "outputs",
                    "acceptance",
                    "missing_evidence",
                    "goal_id",
                    "semantic_key",
                    "bundle_key",
                    "work_scope",
                    "fingerprint",
                )
            ):
                # Legacy bundle indexes sometimes contain only a display id.
                # Preserve support without allowing aliases from different
                # boards to collapse onto one execution identity.
                identity_input["dedupe_key"] = ":".join(
                    [
                        "legacy-task",
                        str(task.get("board_namespace") or "objective-graph"),
                        task_id or str(index),
                    ]
                )
            identity = canonical_task_identity(
                identity_input,
                board_namespace=str(task.get("board_namespace") or "objective-graph"),
                source_path=str(task.get("source_path") or ""),
            )
            cid = identity.canonical_task_cid
        if cid in nodes:
            aliases[task_id] = cid
            existing = nodes[cid]
            aliases_metadata = dict(existing.metadata)
            aliases_metadata["task_id_aliases"] = sorted(
                {
                    existing.task_id,
                    task_id,
                    *[str(value) for value in aliases_metadata.get("task_id_aliases", [])],
                }
            )
            nodes[cid] = replace(existing, metadata=aliases_metadata)
            if (
                existing.status in SUCCESSFUL_MERGE_RECEIPT_STATUSES
                or task_status in SUCCESSFUL_MERGE_RECEIPT_STATUSES
            ):
                continue
            add_repair(
                "duplicate_task",
                cid,
                task_id,
                f"multiple task records resolve to canonical task CID {cid}",
                {"record_index": index},
            )
            continue
        rank = priority_rank(str(task.get("priority") or task.get("objective_priority") or "P2"))
        configured_priority = max(0, 9 - rank)
        try:
            explicit_priority = int(task.get("objective_priority"))
            configured_priority = max(0, explicit_priority)
        except (TypeError, ValueError):
            pass
        node = TaskDependencyNode(
            task_cid=cid,
            task_id=task_id,
            goal_id=str(task.get("goal_id") or "").strip(),
            status=task_status,
            objective_priority=configured_priority,
            created_at_ms=_task_created_at_ms(task),
            estimated_duration=_task_duration(task),
            metadata=_bounded_task_planning_metadata(task),
        )
        nodes[cid] = node
        records_by_cid[cid] = task
        aliases[cid] = cid
        if task_id:
            if task_id in aliases and aliases[task_id] != cid:
                add_repair(
                    "duplicate_alias",
                    cid,
                    task_id,
                    f"task alias {task_id} resolves to more than one canonical task",
                    {"field": "task_id"},
                )
            else:
                aliases[task_id] = cid
        canonical_key = str(task.get("canonical_task_key") or "").strip()
        if canonical_key:
            aliases[canonical_key] = cid

    goals: dict[str, list[str]] = {}
    for cid, node in nodes.items():
        if node.goal_id:
            goals.setdefault(node.goal_id, []).append(cid)

    provider_fields = {
        "import": ("provides_imports", "provided_imports", "exports", "modules"),
        "interface": ("provides_interfaces", "provided_interfaces", "interfaces"),
        "migration": ("provides_migrations", "provided_migrations", "migrations"),
        "validation": ("provides_validations", "validation_outputs", "validation_receipts"),
        "output_input": ("outputs", "output_paths", "produces"),
    }
    requirement_fields = {
        "import": ("import_dependencies", "required_imports", "imports"),
        "interface": ("interface_dependencies", "required_interfaces"),
        "migration": ("migration_dependencies", "required_migrations", "migrations_after"),
        "validation": ("validation_dependencies", "validation_prerequisites"),
        "output_input": ("input_dependencies", "inputs", "input_paths", "consumes"),
    }
    providers: dict[str, dict[str, set[str]]] = {kind: {} for kind in provider_fields}
    for cid, task in records_by_cid.items():
        for kind, fields_for_kind in provider_fields.items():
            for value in _dependency_values(task, *fields_for_kind):
                normalized = value.strip().replace("\\", "/")
                providers[kind].setdefault(normalized, set()).add(cid)

    edges: list[DependencyEdge] = []
    edge_keys: set[tuple[str, str, str, str, str]] = set()

    def add_edge(source: str, target: str, kind: str, *, field_name: str, value: str, resolution: str) -> None:
        provenance = {
            "field": field_name,
            "value": value,
            "resolution": resolution,
            "source_task_id": nodes[source].task_id,
            "target_task_id": nodes[target].task_id,
        }
        key = (source, target, kind, field_name, value)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append(DependencyEdge(source, target, kind, provenance))

    for target, task in records_by_cid.items():
        parent_goals = _dependency_values(task, "parent_goal_ids", "graph_parents", "goal_parents")
        for parent_goal in parent_goals:
            sources = goals.get(parent_goal, [])
            # Parent goals primarily describe objective hierarchy. They become
            # executable prerequisites only when this planning snapshot also
            # materializes a task for that goal. Explicit dependency fields
            # below remain strict and still produce repair evidence.
            for source in sources:
                add_edge(source, target, "goal", field_name="parent_goal_ids", value=parent_goal, resolution="goal_id")

        direct_fields = ("dependency_task_cids", "depends_on", "dependency_task_ids", "prerequisite_task_cids")
        dependency_kinds = task.get("dependency_kinds") if isinstance(task.get("dependency_kinds"), Mapping) else {}
        for field_name in direct_fields:
            for reference in _dependency_values(task, field_name):
                source = aliases.get(reference)
                configured_kind = str(dependency_kinds.get(reference) or "goal").strip().lower().replace("-", "_")
                kind = configured_kind if configured_kind in DEPENDENCY_EDGE_KINDS else "goal"
                if source is None:
                    add_repair(
                        "missing_dependency",
                        target,
                        reference,
                        f"{field_name} references an unknown prerequisite task",
                        {"edge_kind": kind, "field": field_name},
                    )
                    continue
                add_edge(source, target, kind, field_name=field_name, value=reference, resolution="task_alias")

        for kind, fields_for_kind in requirement_fields.items():
            for field_name in fields_for_kind:
                for requirement in _dependency_values(task, field_name):
                    direct_source = aliases.get(requirement)
                    matched_sources = set(providers[kind].get(requirement.replace("\\", "/"), set()))
                    if direct_source:
                        matched_sources.add(direct_source)
                    for source in sorted(matched_sources):
                        if source != target:
                            add_edge(
                                source,
                                target,
                                kind,
                                field_name=field_name,
                                value=requirement,
                                resolution="task_alias" if source == direct_source else "producer_consumer_match",
                            )
                    # Explicit dependency/prerequisite fields promise an
                    # in-graph producer. Plain imports/inputs may be external.
                    if not matched_sources and ("dependencies" in field_name or "prerequisites" in field_name):
                        add_repair(
                            "missing_dependency",
                            target,
                            requirement,
                            f"{kind} prerequisite {requirement} has no materialized producer",
                            {"edge_kind": kind, "field": field_name},
                        )

    edges.sort(key=lambda edge: (edge.source_task_cid, edge.target_task_cid, edge.kind, json.dumps(edge.provenance, sort_keys=True)))
    adjacency: dict[str, set[str]] = {cid: set() for cid in nodes}
    for edge in edges:
        adjacency[edge.source_task_cid].add(edge.target_task_cid)
    succeeded_task_cids = _successful_merge_receipt_cids(merge_receipts, aliases)
    unresolved_nodes = set(nodes) - succeeded_task_cids
    unresolved_adjacency = {
        cid: {target for target in adjacency[cid] if target in unresolved_nodes}
        for cid in unresolved_nodes
    }
    for component in _cycle_components(unresolved_nodes, unresolved_adjacency):
        cycle = " -> ".join([*(nodes[cid].task_id or cid for cid in component), nodes[component[0]].task_id or component[0]])
        for cid in component:
            add_repair(
                "dependency_cycle",
                cid,
                cycle,
                f"task participates in dependency cycle: {cycle}",
                {"component_task_cids": component},
            )

    graph = TaskDependencyGraph(
        nodes=nodes,
        edges=edges,
        repair_evidence=repair[:limit],
        invalid_task_cids=sorted(invalid_task_cids),
    )
    schedule = critical_path_schedule(graph, merge_receipts=merge_receipts, now=now)
    return replace(graph, schedule=schedule)


# Verbose aliases make the API discoverable to callers that say graph rather
# than DAG, without creating separate implementations.
materialize_task_dependency_graph = materialize_task_dependency_dag
schedule_critical_path = critical_path_schedule


def materialize_task_planning_graph(
    tasks: Sequence[Any],
    *,
    repo_root: Path | None = None,
    merge_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] = (),
    branch_diffs: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    conflict_receipts: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    concurrency_overrides: Mapping[str, Any] | Iterable[Any] | None = None,
    conflict_history: Any = None,
    max_lanes: int | None = None,
    now: datetime | int | None = None,
    max_repair_evidence: int = 64,
) -> TaskPlanningGraph:
    """Materialize dependency readiness and conflict-colored lanes together.

    ``tasks`` is copied before either graph consumes it, which is important for
    callers that supply generators.  Historical branch diffs and merge-conflict
    receipts are deliberately routed only to the conflict model; successful
    merge receipts independently control dependency claimability.
    """

    from .conflict_graph import materialize_task_conflict_graph

    task_records = list(tasks)
    dependency_graph = materialize_task_dependency_dag(
        task_records,
        merge_receipts=merge_receipts,
        now=now,
        max_repair_evidence=max_repair_evidence,
    )
    conflict_graph = materialize_task_conflict_graph(
        task_records,
        repo_root=repo_root,
        branch_diffs=branch_diffs,
        conflict_receipts=conflict_receipts,
        concurrency_overrides=concurrency_overrides,
        history=conflict_history,
        max_lanes=max_lanes,
    )
    return TaskPlanningGraph(
        dependency_graph=dependency_graph,
        conflict_graph=conflict_graph,
    )


# Discoverable aliases for scheduler and planner callers.
materialize_task_execution_graph = materialize_task_planning_graph
plan_task_lanes = materialize_task_planning_graph


def objective_goal_work_surface(goal: ObjectiveGoal) -> int:
    """Estimate how much coherent work one goal can support."""

    evidence_count = len(goal.required_evidence)
    output_count = len(split_terms(str(goal.fields.get("outputs") or "")))
    ast_count = len(split_terms(str(goal.fields.get("ast_query") or "")))
    interface_count = len(split_terms(str(goal.fields.get("interfaces") or goal.fields.get("interface_contracts") or "")))
    submodule_count = len(split_terms(str(goal.fields.get("submodules") or goal.fields.get("interoperability_pair") or "")))
    return evidence_count * 4 + output_count * 2 + ast_count + interface_count * 3 + submodule_count * 3


def objective_goal_requires_launch_playwright_validation(goal: ObjectiveGoal) -> bool:
    """Return whether a goal represents the launch slice that needs browser proof."""

    fields = goal.fields
    track = str(fields.get("track") or "").strip().lower()
    bundle = str(fields.get("bundle") or "").strip().lower()
    haystack = " ".join([goal.goal_id, goal.title, *fields.values()]).lower()
    if track == "launch" or bundle.startswith("objective/launch/"):
        return True
    return all(term in haystack for term in ("phone", "desktop", "swissknife", "meta glasses"))


def objective_goal_validation(goal: ObjectiveGoal, fallback_validation: str) -> str:
    """Return validation for generated work from an objective goal."""

    validation = str(goal.fields.get("validation") or fallback_validation).strip()
    if not objective_goal_requires_launch_playwright_validation(goal):
        return validation
    lowered = validation.lower()
    if any(marker in lowered for marker in LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS):
        return validation
    if not validation:
        return LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND
    return f"{validation} && {LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND}"


def objective_goal_validation_gap_terms(goal: ObjectiveGoal) -> list[str]:
    """Return synthetic evidence terms for forced validation-gate work."""

    if not objective_goal_requires_launch_playwright_validation(goal):
        validation = str(goal.fields.get("validation") or "").strip()
        if validation:
            return ["objective validation repair"]
        return []
    return [LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE]


def canonical_interoperability_component(value: str) -> str:
    """Return a stable component key for interoperability dedupe."""

    normalized = " ".join(str(value or "").strip().replace("\\", "/").split()).lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    leaf = normalized.rsplit("/", 1)[-1]
    key = re.sub(r"[^a-z0-9]+", "_", leaf).strip("_")
    aliases = {
        "ipfs_accelerate_py": "ipfs_accelerate",
        "ipfs_datasets_py": "ipfs_datasets",
        "ipfs_kit_py": "ipfs_kit",
        "mcpplusplus": "mcp_plus_plus",
    }
    return aliases.get(key, key)


def interoperability_pair_schedule_key(goal: ObjectiveGoal) -> str:
    """Return a stable key for duplicate interoperability pair goals."""

    pair_value = str(goal.fields.get("interoperability_pair") or "").strip()
    if not pair_value:
        return ""
    pair_terms = sorted(
        key
        for term in split_terms(pair_value)
        for key in [canonical_interoperability_component(term)]
        if key
    )
    if not pair_terms:
        return ""
    return "\0".join(pair_terms)


def objective_heap_schedule(goals: Sequence[ObjectiveGoal]) -> list[ObjectiveHeapRecord]:
    """Return schedulable goals in Fibonacci-priority heap order.

    The persisted objective heap stores Fibonacci priority buckets in markdown.
    This function materializes those buckets as a deterministic heap schedule and
    adds work-surface tie breakers so larger integration goals win within the
    same Fibonacci band.
    """

    graph = goal_graph(goals)
    heap: list[tuple[tuple[Any, ...], ObjectiveGoal]] = []
    seen_interoperability_pairs: set[str] = set()
    for goal in goals:
        if not goal.is_schedulable:
            continue
        interoperability_key = interoperability_pair_schedule_key(goal)
        if interoperability_key:
            if interoperability_key in seen_interoperability_pairs:
                continue
            seen_interoperability_pairs.add(interoperability_key)
        fib_priority = goal.priority[0]
        rank = priority_rank(str(goal.fields.get("priority") or "P2"))
        work_surface = objective_goal_work_surface(goal)
        graph_depth = int(graph["depths"].get(goal.goal_id, 0))
        sort_key = (
            fib_priority,
            rank,
            -work_surface,
            graph_depth,
            goal.goal_id,
        )
        heapq.heappush(heap, (sort_key, goal))

    records: list[ObjectiveHeapRecord] = []
    while heap:
        sort_key, goal = heapq.heappop(heap)
        records.append(
            ObjectiveHeapRecord(
                heap_index=len(records),
                goal_id=goal.goal_id,
                title=goal.title,
                status=goal.status,
                fib_priority=goal.priority[0],
                priority=str(goal.fields.get("priority") or "P2"),
                priority_rank=priority_rank(str(goal.fields.get("priority") or "P2")),
                track=str(goal.fields.get("track") or "ops"),
                graph_depth=int(graph["depths"].get(goal.goal_id, 0)),
                work_surface_score=objective_goal_work_surface(goal),
                required_evidence_count=len(goal.required_evidence),
                output_count=len(split_terms(str(goal.fields.get("outputs") or ""))),
                parents=goal.parent_goal_ids,
                sort_key=list(sort_key),
            )
        )
    return records


def objective_fingerprint(goal: ObjectiveGoal, missing_terms: Sequence[str]) -> str:
    payload = "\0".join(
        [
            "objective_goal_gap",
            goal.goal_id,
            goal.title,
            *[" ".join(str(term).lower().split()) for term in missing_terms],
        ]
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def normalize_objective_evidence_requirement(value: str) -> str:
    """Return the canonical comparison form for one evidence requirement."""

    return " ".join(str(value or "").strip().casefold().split())


def objective_evidence_lineage_ids(
    goal_id: str,
    requirements: Sequence[str],
    graph: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the earliest ancestors that introduced an evidence obligation.

    Refinement may move the same criterion from a parent into a more focused
    child.  Those two records need one stable task identity.  Unrelated
    workstreams that merely use the same generic term must retain independent
    identities, so ancestry is followed only while every requested criterion
    is still present.
    """

    details = graph.get("node_details")
    details = details if isinstance(details, Mapping) else {}
    required = {
        normalized
        for term in requirements
        for normalized in [normalize_objective_evidence_requirement(term)]
        if normalized
    }
    if not required:
        return (str(goal_id),)

    memo: dict[str, tuple[str, ...]] = {}

    def resolve(current: str, seen: frozenset[str] = frozenset()) -> tuple[str, ...]:
        if current in memo:
            return memo[current]
        if not current or current in seen:
            return (current,) if current else ()
        detail = details.get(current)
        detail = detail if isinstance(detail, Mapping) else {}
        matching_parents: list[str] = []
        for raw_parent in detail.get("parents", ()):
            parent = str(raw_parent)
            parent_detail = details.get(parent)
            parent_detail = (
                parent_detail if isinstance(parent_detail, Mapping) else {}
            )
            parent_requirements = {
                normalized
                for term in parent_detail.get("required_evidence", ())
                for normalized in [
                    normalize_objective_evidence_requirement(term)
                ]
                if normalized
            }
            if required.issubset(parent_requirements):
                matching_parents.append(parent)
        result = tuple(
            sorted(
                {
                    ancestor
                    for parent in matching_parents
                    for ancestor in resolve(parent, seen | {current})
                    if ancestor
                }
            )
        )
        if not result:
            result = (current,)
        memo[current] = result
        return result

    return resolve(str(goal_id))


def objective_evidence_obligation_key(
    goal: ObjectiveGoal,
    missing_terms: Sequence[str],
    *,
    graph: Mapping[str, Any],
    candidate_kind: str = "aggregate",
) -> str:
    """Return a goal-refinement-stable identity for missing evidence work."""

    normalized_kind = (
        "validation_gate"
        if candidate_kind == "validation_gate"
        else "evidence_gap"
    )
    material = {
        "schema": "ipfs_accelerate_py/agent-supervisor/evidence-obligation@1",
        "kind": normalized_kind,
        "goal_lineage_ids": list(
            objective_evidence_lineage_ids(
                goal.goal_id,
                missing_terms,
                graph,
            )
        ),
        "requirements": sorted(
            {
                normalized
                for term in missing_terms
                for normalized in [normalize_objective_evidence_requirement(term)]
                if normalized
            }
        ),
    }
    digest = sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"objective-evidence-obligation/v1/{digest}"


def objective_evidence_owner_by_requirement(
    goals: Sequence[ObjectiveGoal],
    graph: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Assign shared requirements to every leaf-most refinement lineage."""

    candidates: dict[str, list[ObjectiveGoal]] = {}
    for goal in goals:
        if not goal.is_schedulable:
            continue
        for term in goal.required_evidence:
            key = normalize_objective_evidence_requirement(term)
            if key:
                candidates.setdefault(key, []).append(goal)

    children = graph.get("children")
    children = children if isinstance(children, Mapping) else {}
    descendant_cache: dict[str, frozenset[str]] = {}

    def descendants(goal_id: str, seen: frozenset[str] = frozenset()) -> frozenset[str]:
        if goal_id in descendant_cache:
            return descendant_cache[goal_id]
        if not goal_id or goal_id in seen:
            return frozenset()
        direct = {
            str(item)
            for item in children.get(goal_id, ())
            if str(item) and str(item) != goal_id
        }
        result = frozenset(
            {
                *direct,
                *(
                    descendant
                    for child in direct
                    for descendant in descendants(child, seen | {goal_id})
                ),
            }
        )
        descendant_cache[goal_id] = result
        return result

    owners: dict[str, tuple[str, ...]] = {}
    for key, requirement_goals in candidates.items():
        candidate_ids = {goal.goal_id for goal in requirement_goals}
        owners[key] = tuple(
            sorted(
                goal.goal_id
                for goal in requirement_goals
                if not (descendants(goal.goal_id) & candidate_ids)
            )
        )
    return owners


def objective_merge_key(goal: ObjectiveGoal, missing_terms: Sequence[str], *, candidate_kind: str = "aggregate") -> str:
    payload = {
        "goal_id": goal.goal_id,
        "candidate_kind": candidate_kind,
        "missing_terms": sorted(" ".join(str(term).split()).lower() for term in missing_terms),
        "outputs": sorted(split_terms(str(goal.fields.get("outputs") or ""))),
        "ast_query": str(goal.fields.get("ast_query") or ""),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def objective_surplus_group(goal: ObjectiveGoal) -> str:
    return f"objective/{goal.goal_id}"


def objective_todo_vector_key(goal: ObjectiveGoal, missing_terms: Sequence[str], *, candidate_kind: str) -> str:
    payload = "\0".join([goal.goal_id, candidate_kind, *sorted(str(term) for term in missing_terms)])
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def objective_goal_packet_aggregate_fingerprint(
    packet_key: str,
    findings: Sequence[ObjectiveFinding],
    missing_terms: Sequence[str],
) -> str:
    payload = {
        "kind": "objective_goal_packet_aggregate",
        "packet_key": packet_key,
        "finding_fingerprints": sorted(finding.fingerprint for finding in findings),
        "missing_terms": sorted(" ".join(str(term).lower().split()) for term in missing_terms),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def objective_goal_packet_aggregate_key(packet_key: str, missing_terms: Sequence[str]) -> str:
    payload = {
        "packet_key": packet_key,
        "missing_terms": sorted(" ".join(str(term).lower().split()) for term in missing_terms),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def objective_goal_packet_key(findings: Sequence[ObjectiveFinding]) -> str:
    """Return a stable packet key for related goal/subgoal findings."""

    goals = sorted({finding.goal_id for finding in findings if finding.goal_id})
    parents = sorted({parent for finding in findings for parent in finding.parent_goal_ids})
    tracks = sorted({finding.track for finding in findings if finding.track})
    roots = sorted({finding_conflict_root(finding) for finding in findings})
    ast_queries = sorted({finding.ast_query for finding in findings if finding.ast_query})
    seed = {
        "goals": goals,
        "parents": parents or goals,
        "tracks": tracks,
        "roots": roots,
        "ast_queries": ast_queries,
    }
    digest = sha1(json.dumps(seed, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    track = safe_bundle_key(tracks[0] if tracks else "ops").replace("-", "_") or "ops"
    root = safe_bundle_key(roots[0] if roots else "general").replace("-", "_") or "general"
    return f"goal_packet/{track}/{root}/{digest}"


def _goal_packet_group_key(finding: ObjectiveFinding) -> tuple[str, tuple[str, ...], str]:
    family = tuple(sorted(finding.parent_goal_ids or [finding.goal_id]))
    return (finding.track, family, finding_conflict_root(finding))


def assign_goal_subgoal_packets(findings: Sequence[ObjectiveFinding]) -> list[ObjectiveFinding]:
    """Annotate findings with packet metadata for sibling goal/subgoal work."""

    groups: dict[tuple[str, tuple[str, ...], str], list[ObjectiveFinding]] = {}
    for finding in findings:
        groups.setdefault(_goal_packet_group_key(finding), []).append(finding)

    packet_by_fingerprint: dict[str, ObjectiveFinding] = {}
    for group_findings in groups.values():
        if len(group_findings) < 2:
            continue
        packet_key = objective_goal_packet_key(group_findings)
        goal_ids = sorted({finding.goal_id for finding in group_findings if finding.goal_id})
        task_count = len(group_findings)
        packet_work_items = sum(finding.work_item_count or len(finding.missing_evidence) for finding in group_findings)
        multi_goal_packet = len(goal_ids) > 1
        for index, finding in enumerate(
            sorted(group_findings, key=lambda item: (item.priority, item.goal_id, item.candidate_kind, item.fingerprint))
        ):
            role = "packet_anchor" if index == 0 else "packet_member"
            merge_family = packet_key if multi_goal_packet else finding.merge_family
            work_scope = finding.work_scope or "goal_subgoal_multi_evidence_batch"
            if "goal_subgoal_packet" not in work_scope:
                work_scope = f"{work_scope}; goal_subgoal_packet"
            packet_by_fingerprint[finding.fingerprint] = replace(
                finding,
                merge_family=merge_family,
                goal_packet_key=packet_key,
                goal_packet_role=role,
                goal_packet_goal_ids=goal_ids,
                goal_packet_task_count=task_count,
                goal_packet_work_item_count=packet_work_items,
                work_scope=work_scope,
            )

    return [packet_by_fingerprint.get(finding.fingerprint, finding) for finding in findings]


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _goal_conflict_terms(goal: ObjectiveGoal, *field_names: str) -> list[str]:
    """Collect normalized conflict-surface declarations from goal fields."""

    return _unique_strings(
        term
        for field_name in field_names
        for term in split_terms(str(goal.fields.get(field_name) or ""))
    )


def _merge_present_evidence(findings: Sequence[ObjectiveFinding]) -> dict[str, list[str]]:
    present: dict[str, list[str]] = {}
    for finding in findings:
        for term, paths in finding.present_evidence.items():
            bucket = present.setdefault(str(term), [])
            for path in paths:
                path_text = str(path)
                if path_text and path_text not in bucket:
                    bucket.append(path_text)
    return present


def _first_non_empty(values: Iterable[str], default: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def add_goal_packet_aggregate_findings(
    findings: Sequence[ObjectiveFinding],
    *,
    max_findings: int,
    seen_fingerprints: Iterable[str] = (),
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
) -> list[ObjectiveFinding]:
    """Add larger packet-level todos for related goal/subgoal findings when capacity allows."""

    planned = list(findings)
    if len(planned) >= max_findings:
        return planned[:max_findings]

    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    seen.update(finding.fingerprint for finding in planned)
    groups: dict[str, list[ObjectiveFinding]] = {}
    for finding in planned:
        if finding.goal_packet_key:
            groups.setdefault(finding.goal_packet_key, []).append(finding)

    for packet_key, group_findings in sorted(groups.items()):
        if len(planned) >= max_findings:
            break
        if len(group_findings) < 2:
            continue
        sorted_group = sorted(
            group_findings,
            key=lambda item: (item.priority, item.goal_id, item.candidate_kind, item.fingerprint),
        )
        goal_ids = _unique_strings(finding.goal_id for finding in sorted_group)
        if len(goal_ids) < 2:
            continue
        missing_terms = _unique_strings(term for finding in sorted_group for term in finding.missing_evidence)
        if len(missing_terms) < 2:
            continue
        fingerprint = objective_goal_packet_aggregate_fingerprint(packet_key, sorted_group, missing_terms)
        if fingerprint in seen:
            continue

        anchor = sorted_group[0]
        parent_goal_ids = _unique_strings(parent for finding in sorted_group for parent in finding.parent_goal_ids)
        outputs = _unique_strings(output for finding in sorted_group for output in finding.outputs)
        evidence_methods = _unique_strings(method for finding in sorted_group for method in finding.evidence_methods)
        goal_lines = [
            f"{finding.goal_id}: {finding.goal or finding.title}"
            for finding in sorted_group
            if finding.goal_id or finding.goal or finding.title
        ]
        merge_key = objective_goal_packet_aggregate_key(packet_key, missing_terms)
        title = f"Goal packet aggregate for {', '.join(goal_ids)}"
        summary = f"{summary_prefix} packet: {', '.join(goal_ids)}"
        ast_terms = _unique_strings(term for finding in sorted_group for term in split_terms(finding.ast_query))
        if not ast_terms:
            ast_terms = missing_terms
        aggregate = ObjectiveFinding(
            fingerprint=fingerprint,
            goal_id=anchor.goal_id,
            title=title,
            summary=summary,
            priority=anchor.priority,
            track=anchor.track,
            missing_evidence=missing_terms,
            present_evidence=_merge_present_evidence(sorted_group),
            evidence_methods=evidence_methods,
            objective_path=anchor.objective_path,
            outputs=outputs,
            validation=_first_non_empty((finding.validation for finding in sorted_group), anchor.validation),
            goal="Close packet goals:\n" + "\n".join(f"- {line}" for line in goal_lines),
            refinement=_first_non_empty((finding.refinement for finding in sorted_group)),
            gap_task=(
                "Close the packet-level missing evidence across these related goals/subgoals in one cohesive "
                "change when the output paths overlap."
            ),
            parent_goal_ids=parent_goal_ids,
            graph_depth=min(finding.graph_depth for finding in sorted_group),
            bundle_key=anchor.bundle_key,
            parallel_lane=anchor.parallel_lane,
            bundle_explicit=anchor.bundle_explicit,
            bundle_strategy=anchor.bundle_strategy,
            embedding_query="; ".join(
                _unique_strings(
                    [
                        f"goal packet {packet_key}",
                        *(finding.embedding_query for finding in sorted_group),
                        *(finding.title for finding in sorted_group),
                    ]
                )
            ),
            ast_query=", ".join(ast_terms),
            conflict_policy=anchor.conflict_policy,
            refinement_depth=str(min(_parse_int(finding.refinement_depth, 0) for finding in sorted_group)),
            candidate_kind="goal_packet_aggregate",
            surplus_group=packet_key,
            merge_key=merge_key,
            merge_family=packet_key,
            merge_role="packet_aggregate",
            work_item_count=len(missing_terms),
            work_scope="goal_subgoal_packet_aggregate; vector_ast_bundle",
            todo_vector_key=sha1(f"{packet_key}\0goal_packet_aggregate\0{merge_key}".encode("utf-8")).hexdigest()[:16],
            goal_packet_key=packet_key,
            goal_packet_role="packet_aggregate",
            goal_packet_goal_ids=goal_ids,
            goal_packet_task_count=len(sorted_group) + 1,
            goal_packet_work_item_count=sum(
                finding.work_item_count or len(finding.missing_evidence) for finding in sorted_group
            ),
            predicted_files=_unique_strings(
                path for finding in sorted_group for path in (finding.predicted_files or finding.outputs)
            ),
            changed_paths=_unique_strings(
                path for finding in sorted_group for path in finding.changed_paths
            ),
            ast_symbols=_unique_strings(symbol for finding in sorted_group for symbol in finding.ast_symbols),
            interfaces=_unique_strings(interface for finding in sorted_group for interface in finding.interfaces),
            submodules=_unique_strings(submodule for finding in sorted_group for submodule in finding.submodules),
            generated_artifacts=_unique_strings(
                artifact for finding in sorted_group for artifact in finding.generated_artifacts
            ),
            allow_concurrent_with=_unique_strings(
                task for finding in sorted_group for task in finding.allow_concurrent_with
            ),
            dedupe_key=(
                "objective-evidence-packet/v1/"
                + sha256(
                    json.dumps(
                        {
                            "packet_key": packet_key,
                            "requirements": sorted(
                                normalize_objective_evidence_requirement(term)
                                for term in missing_terms
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            ),
        )
        planned.append(aggregate)
        seen.add(fingerprint)
    return planned[:max_findings]


def objective_scan_candidate_limit(*, max_findings: int, surplus_findings_per_goal: int) -> int:
    """Return the internal candidate pool size used before final todo selection."""

    try:
        surplus_count = max(1, int(surplus_findings_per_goal))
    except (TypeError, ValueError):
        surplus_count = DEFAULT_SURPLUS_FINDINGS_PER_GOAL
    try:
        oversample_multiplier = max(1, int(DEFAULT_SCAN_OVERSAMPLE_MULTIPLIER))
    except (TypeError, ValueError):
        oversample_multiplier = 2
    return max_findings * max(2, surplus_count, oversample_multiplier)


def prioritize_larger_work_surface_findings(
    findings: Sequence[ObjectiveFinding],
    *,
    max_findings: int,
) -> list[ObjectiveFinding]:
    """Select findings that give each Codex invocation a larger coherent work surface."""

    if max_findings <= 0:
        return []

    def candidate_rank(finding: ObjectiveFinding) -> int:
        if finding.candidate_kind == "goal_packet_aggregate":
            return 0
        if finding.candidate_kind == "aggregate":
            return 1
        if finding.candidate_kind == "evidence_cluster":
            return 2
        return 3

    def sort_key(finding: ObjectiveFinding) -> tuple[Any, ...]:
        packet_goal_count = len(finding.goal_packet_goal_ids)
        packet_work_items = finding.goal_packet_work_item_count or 0
        work_items = finding.work_item_count or len(finding.missing_evidence)
        return (
            candidate_rank(finding),
            -packet_goal_count,
            -packet_work_items,
            -work_items,
            finding.priority,
            finding.graph_depth,
            finding.goal_id,
            finding.candidate_kind,
            finding.fingerprint,
        )

    return sorted(findings, key=sort_key)[:max_findings]


def surplus_missing_term_groups(
    missing_terms: Sequence[str],
    *,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
) -> list[tuple[str, list[str]]]:
    """Return mergeable objective-gap candidate groups for one goal.

    The first group preserves the historical behavior: one aggregate task that
    covers all missing terms.  Additional groups are multi-evidence batches by
    default so each generated todo has enough implementation work to justify a
    Codex invocation while still carrying merge-family metadata for bundling.
    """

    terms = [term for term in dict.fromkeys(str(item).strip() for item in missing_terms) if term]
    if not terms:
        return []
    try:
        surplus_count = max(1, int(surplus_findings_per_goal))
    except (TypeError, ValueError):
        surplus_count = 1
    groups: list[tuple[str, list[str]]] = [("aggregate", terms)]
    if surplus_count <= 1 or len(terms) <= 1:
        return groups

    try:
        minimum_terms = max(1, int(min_terms_per_todo))
    except (TypeError, ValueError):
        minimum_terms = 2
    minimum_terms = min(max(1, minimum_terms), len(terms))
    extra_count = surplus_count - 1
    group_size = min(len(terms), max(minimum_terms, math.ceil(len(terms) / max(1, extra_count))))
    seen_term_sets = {tuple(terms)}

    def append_group(candidate_terms: Sequence[str]) -> None:
        normalized = [term for term in dict.fromkeys(str(item).strip() for item in candidate_terms) if term]
        if len(normalized) < minimum_terms:
            return
        key = tuple(normalized)
        if key in seen_term_sets:
            return
        seen_term_sets.add(key)
        groups.append(("evidence_cluster" if len(normalized) > 1 else "evidence_term", normalized))

    for start in range(0, len(terms), group_size):
        if len(groups) >= surplus_count:
            break
        append_group(terms[start : start + group_size])
    start = 1
    while len(groups) < surplus_count and start < len(terms):
        if start + group_size <= len(terms):
            append_group(terms[start : start + group_size])
        else:
            append_group([*terms[start:], *terms[: max(0, group_size - (len(terms) - start))]])
        start += 1
    return groups


def _path_root(value: str) -> str:
    value = str(value or "").strip()
    if not value or not repo_relative_path_safe(value):
        return ""
    first = Path(value).parts[0] if Path(value).parts else ""
    if first in {"data", "tests", "docs", "."}:
        return ""
    return first


def finding_conflict_root(finding: ObjectiveFinding) -> str:
    """Return the path-root conflict domain used for implicit bundle planning."""

    for output in finding.outputs:
        root = _path_root(output)
        if root:
            return root
    for paths in finding.present_evidence.values():
        for raw_path in paths:
            root = _path_root(str(raw_path).split(" (", 1)[0])
            if root:
                return root
    return "general"


def finding_semantic_bundle_text(finding: ObjectiveFinding) -> str:
    present_paths: list[str] = []
    for paths in finding.present_evidence.values():
        present_paths.extend(str(path) for path in paths)
    return "\n".join(
        [
            finding.title,
            finding.goal,
            finding.embedding_query,
            finding.ast_query,
            " ".join(finding.missing_evidence),
            " ".join(present_paths),
        ]
    )


def _bundle_cluster_key(*, finding: ObjectiveFinding, root: str, cluster_text: str) -> str:
    digest = sha1(cluster_text.encode("utf-8")).hexdigest()[:8]
    track = safe_bundle_key(finding.track).replace("-", "_") or "ops"
    safe_root = safe_bundle_key(root).replace("-", "_") or "general"
    return f"objective/{track}/{safe_root}/semantic-{digest}"


def plan_semantic_ast_bundles(
    findings: Sequence[ObjectiveFinding],
    *,
    min_score: float = DEFAULT_BUNDLE_CLUSTER_MIN_SCORE,
    preserve_explicit: bool = True,
) -> list[ObjectiveFinding]:
    """Assign implicit objective findings to AST/embedding-aware bundle lanes.

    Explicit ``Bundle:`` fields are left unchanged.  Remaining findings are
    clustered inside a conflict-domain path root using deterministic token
    embeddings built from the goal text, missing evidence, AST query, and present
    evidence paths.  This keeps likely-overlapping work in the same lane while
    letting unrelated roots drain in parallel.
    """

    planned: list[ObjectiveFinding] = []
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for finding in findings:
        if preserve_explicit and finding.bundle_explicit:
            planned.append(finding)
            continue
        root = finding_conflict_root(finding)
        group_key = (finding.track, root)
        text = finding_semantic_bundle_text(finding)
        vector = text_embedding(text)
        selected: dict[str, Any] | None = None
        best_score = -1.0
        for cluster in clusters.get(group_key, []):
            score = cosine(vector, cluster["vector"])
            if score > best_score:
                best_score = score
                selected = cluster
        if selected is None or best_score < min_score:
            bundle_key = _bundle_cluster_key(finding=finding, root=root, cluster_text=text)
            selected = {"bundle_key": bundle_key, "vector": vector, "texts": [text]}
            clusters.setdefault(group_key, []).append(selected)
        else:
            selected["texts"].append(text)
            vectors = [text_embedding(item) for item in selected["texts"]]
            averaged = [sum(values) / len(vectors) for values in zip(*vectors)]
            norm = math.sqrt(sum(value * value for value in averaged))
            selected["vector"] = [value / norm for value in averaged] if norm else averaged
        bundle_key = str(selected["bundle_key"])
        planned.append(
            replace(
                finding,
                bundle_key=bundle_key,
                parallel_lane=bundle_key,
                bundle_strategy="semantic_ast",
            )
        )
    return planned


def scan_objective_gaps(
    repo_root: Path,
    *,
    objective_path: Path,
    max_findings: int = 10,
    seen_fingerprints: Iterable[str] = (),
    force_goal_ids: Iterable[str] = (),
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    dataset_dir: Path | None = None,
    dataset_id: str = "objective-ast",
    scan_stats: dict[str, Any] | None = None,
    analysis_pipeline: Any = None,
    analysis_pipeline_request: Any = None,
    typed_evidence_receipts: Sequence[Mapping[str, Any] | Any] = (),
    evidence_source_policy: EvidenceSourcePolicy | None = None,
    evidence_repository_tree: str = "",
    evidence_policy_id: str = "",
) -> list[ObjectiveFinding]:
    if max_findings <= 0 or not objective_path.exists():
        return []
    all_goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    goals = [goal for goal in all_goals if goal.is_schedulable]
    if not goals:
        return []
    graph = goal_graph(all_goals)
    required_terms: list[str] = []
    for goal in goals:
        required_terms.extend(goal.required_evidence)
    persisted_receipts = [
        record
        for goal in all_goals
        for record in goal.completion_evidence_records
    ]
    receipt_values: list[Mapping[str, Any] | Any] = []
    receipt_keys: set[str] = set()
    for receipt in [*typed_evidence_receipts, *persisted_receipts]:
        if isinstance(receipt, Mapping):
            projected: Any = dict(receipt)
        else:
            converter = getattr(receipt, "to_dict", None)
            projected = converter() if callable(converter) else receipt
        try:
            key = json.dumps(
                projected,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            key = repr(projected)
        if key in receipt_keys:
            continue
        receipt_keys.add(key)
        receipt_values.append(receipt)
    cached_records: list[dict[str, Any]] | None = None
    if dataset_dir is not None:
        artifact = persist_objective_ast_dataset(
            repo_root=repo_root,
            objective_path=objective_path,
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
        )
        cached_records = ObjectiveDatasetStore(dataset_dir).load_records(dataset_id)
        if scan_stats is not None:
            scan_stats.clear()
            scan_stats.update(artifact.to_dict())
    pipeline_diagnostics: dict[str, Any] = {}
    if dataset_dir is not None or analysis_pipeline is not None:
        try:
            from .analysis_cache import AnalysisCache
            from .analysis_pipeline import (
                AnalysisPipeline,
                AnalysisPipelineRequest,
                make_analysis_stage_receipt,
            )
            from .conflict_graph import build_python_ast_blob_record

            if analysis_pipeline is None:
                def objective_scan_analyzer(context: Any) -> Any:
                    return make_analysis_stage_receipt(
                        context.request,
                        successful=True,
                        reason_code="bounded_objective_scan_complete",
                    )

                analysis_pipeline = AnalysisPipeline(
                    AnalysisCache(Path(dataset_dir) / "analysis_cache"),
                    objective_scan_analyzer,
                )
            if analysis_pipeline_request is None:
                identity = scan_identity(repo_root)
                ast_records: list[tuple[str, Any]] = []
                retrieval_records: list[dict[str, Any]] = []
                for row in cached_records or ():
                    path = str(row.get("root_relative_path") or "")
                    compact = {
                        str(key): value
                        for key, value in row.items()
                        if key not in {"evidence_text", "ast_text"}
                    }
                    compact.setdefault("record_id", path)
                    compact.setdefault("path", path)
                    retrieval_records.append(compact)
                    if str(row.get("suffix") or "").lower() != ".py":
                        continue
                    source = row.get("evidence_text")
                    if isinstance(source, str):
                        ast_records.append(
                            (
                                path,
                                build_python_ast_blob_record(
                                    source,
                                    blob_identity=str(
                                        row.get("blob_hash")
                                        or row.get("source_sha1")
                                        or ""
                                    ),
                                ),
                            )
                        )
                analysis_pipeline_request = AnalysisPipelineRequest(
                    repository_id=identity.repository_id,
                    tree_id=canonical_revision(
                        {
                            "repository_id": identity.repository_id,
                            "objective_revision": objective_heap_content_id(
                                objective_path.read_text(
                                    encoding="utf-8", errors="replace"
                                )
                            ),
                            "records": [
                                {
                                    "path": str(
                                        row.get("root_relative_path") or ""
                                    ),
                                    "blob": str(
                                        row.get("blob_hash")
                                        or row.get("source_sha1")
                                        or ""
                                    ),
                                }
                                for row in cached_records or ()
                            ],
                        },
                        namespace="objective-analysis-tree",
                    ),
                    objective_revision=objective_heap_content_id(
                        objective_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    ),
                    query={
                        "text": " ".join(required_terms),
                        "objective_terms": required_terms,
                    },
                    analyzer_id="objective.gap_scan",
                    analyzer_version=OBJECTIVE_SCAN_ANALYZER_VERSION,
                    configuration={
                        "dataset_id": dataset_id,
                        "embedding_min_score": embedding_min_score,
                    },
                    policy={"purpose": "objective_evidence_nomination"},
                    ast_records=tuple(ast_records),
                    retrieval_inputs={"records": tuple(retrieval_records)},
                )
            pipeline_value = analysis_pipeline.analyze(
                analysis_pipeline_request
            )
            pipeline_projection = pipeline_value.to_dict()
            pipeline_diagnostics = {
                "result_id": pipeline_projection.get("result_id", ""),
                "cache_status": pipeline_projection.get("cache_status", ""),
                "cache_lookup_status": pipeline_projection.get(
                    "cache_lookup_status", ""
                ),
                "cache_reason_codes": list(
                    pipeline_projection.get("cache_reason_codes") or ()
                ),
                "retrieval_response_id": pipeline_projection.get(
                    "retrieval_response_id", ""
                ),
                "ranked_evidence_references": list(
                    pipeline_projection.get("ranked_evidence_references") or ()
                ),
                "retrieval_backend_health": dict(
                    pipeline_projection.get("retrieval_backend_health") or {}
                ),
                "retrieval_truncation": dict(
                    pipeline_projection.get("retrieval_truncation") or {}
                ),
                "nomination_only": True,
                "safe_for_completion_reasoning": False,
            }
        except Exception as exc:
            pipeline_diagnostics = {
                "status": "failed",
                "reason_code": "integrated_analysis_pipeline_failed",
                "error_type": type(exc).__name__,
                "detail": str(exc)[:500],
                "nomination_only": True,
                "safe_for_completion_reasoning": False,
            }
        if scan_stats is not None:
            scan_stats["analysis_pipeline"] = pipeline_diagnostics
    evidence = evidence_index(
        repo_root,
        objective_path=objective_path,
        terms=required_terms,
        embedding_min_score=embedding_min_score,
        records=cached_records,
        typed_receipts=receipt_values,
        source_policy=evidence_source_policy,
        repository_tree=evidence_repository_tree,
        policy_id=evidence_policy_id,
    )
    seen = {str(item) for item in seen_fingerprints if str(item).strip()}
    forced_goal_ids = {str(item) for item in force_goal_ids if str(item).strip()}
    findings: list[ObjectiveFinding] = []
    candidate_limit = objective_scan_candidate_limit(
        max_findings=max_findings,
        surplus_findings_per_goal=surplus_findings_per_goal,
    )

    goals_by_id = {goal.goal_id: goal for goal in goals}
    scheduled_goals = [goals_by_id[record.goal_id] for record in objective_heap_schedule(goals) if record.goal_id in goals_by_id]
    evidence_owners = objective_evidence_owner_by_requirement(goals, graph)
    for goal in scheduled_goals:
        if goal.lifecycle_state_value == "provisionally_complete":
            # A provisional goal has left the implementation stage.  Missing
            # completion proof belongs to the typed completion gate, not an
            # ordinary implementation refill.
            continue
        terms = goal.required_evidence
        all_missing_terms = [term for term in terms if not evidence.get(term)]
        missing_terms = [
            term
            for term in all_missing_terms
            if goal.goal_id
            in evidence_owners.get(
                normalize_objective_evidence_requirement(term),
                (),
            )
        ]
        forced_goal = goal.goal_id in forced_goal_ids
        validation_gap = False
        if not missing_terms:
            if all_missing_terms:
                # A more specific child owns the unresolved obligation.
                continue
            if not forced_goal:
                continue
            missing_terms = objective_goal_validation_gap_terms(goal)
            if not missing_terms:
                continue
            validation_gap = True
        launch_validation_gap = validation_gap and objective_goal_requires_launch_playwright_validation(goal)
        fields = goal.fields
        present = {term: evidence.get(term, []) for term in terms if evidence.get(term)}
        explicit_bundle = bool(str(fields.get("bundle") or "").strip())
        for candidate_kind, candidate_missing_terms in surplus_missing_term_groups(
            missing_terms,
            surplus_findings_per_goal=surplus_findings_per_goal,
            min_terms_per_todo=surplus_min_terms_per_todo,
        ):
            if validation_gap:
                candidate_kind = "validation_gate"
            fingerprint = objective_fingerprint(goal, candidate_missing_terms)
            if fingerprint in seen and not forced_goal:
                continue
            bundle_key = goal.bundle_key(candidate_missing_terms)
            finding = ObjectiveFinding(
                fingerprint=fingerprint,
                goal_id=goal.goal_id,
                title=goal.title,
                summary=f"{summary_prefix}: {goal.title}",
                priority=str(fields.get("priority") or "P2"),
                track=str(fields.get("track") or "ops"),
                missing_evidence=candidate_missing_terms,
                present_evidence=present,
                evidence_methods=evidence_methods(present),
                objective_path=repo_relative_path(repo_root, objective_path),
                outputs=split_terms(str(fields.get("outputs") or "")),
                validation=objective_goal_validation(
                    goal,
                    f"test -f {repo_relative_path(repo_root, objective_path)}",
                ),
                goal=str(fields.get("goal") or ""),
                refinement=str(fields.get("refinement") or ""),
                gap_task=(
                    "Run and repair the launch readiness validation gate until the phone, desktop, "
                    "Swissknife, Hallucinate App, and Meta glasses Playwright checks pass."
                    if launch_validation_gap
                    else "Run and repair the objective validation command until it passes, then record the evidence."
                    if validation_gap
                    else str(fields.get("gap_task") or "")
                ),
                parent_goal_ids=goal.parent_goal_ids,
                graph_depth=int(graph["depths"].get(goal.goal_id, 0)),
                bundle_key=bundle_key,
                parallel_lane=str(fields.get("parallel_lane") or bundle_key),
                bundle_explicit=explicit_bundle,
                bundle_strategy="explicit" if explicit_bundle else "semantic_ast",
                embedding_query=str(fields.get("embedding_query") or fields.get("goal") or goal.title),
                ast_query=str(fields.get("ast_query") or ", ".join(terms)),
                conflict_policy=str(
                    fields.get("conflict_policy")
                    or "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts"
                ),
                refinement_depth=str(fields.get("refinement_depth") or graph["depths"].get(goal.goal_id, 0)),
                candidate_kind=candidate_kind,
                surplus_group=objective_surplus_group(goal),
                merge_key=objective_merge_key(goal, candidate_missing_terms, candidate_kind=candidate_kind),
                merge_family=objective_surplus_group(goal),
                merge_role=candidate_kind,
                work_item_count=len(candidate_missing_terms),
                work_scope=(
                    "launch_validation_gate"
                    if launch_validation_gap
                    else "objective_validation_repair"
                    if validation_gap
                    else "goal_subgoal_multi_evidence_batch"
                ),
                todo_vector_key=objective_todo_vector_key(
                    goal,
                    candidate_missing_terms,
                    candidate_kind=candidate_kind,
                ),
                predicted_files=_unique_strings(
                    [
                        *split_terms(str(fields.get("outputs") or "")),
                        *_goal_conflict_terms(goal, "predicted_files", "files"),
                    ]
                ),
                changed_paths=_goal_conflict_terms(
                    goal,
                    "changed_paths",
                    "actual_changed_paths",
                    "branch_diff_paths",
                ),
                ast_symbols=_unique_strings(
                    [
                        *_goal_conflict_terms(goal, "ast_symbols"),
                        *split_terms(str(fields.get("ast_query") or ", ".join(terms))),
                    ]
                ),
                interfaces=_goal_conflict_terms(
                    goal,
                    "interfaces",
                    "interface_contracts",
                    "provides_interfaces",
                    "requires_interfaces",
                    "required_interfaces",
                    "interface_dependencies",
                    "public_interfaces",
                ),
                submodules=_goal_conflict_terms(
                    goal,
                    "submodules",
                    "submodule_paths",
                    "interoperability_pair",
                    "gitlinks",
                ),
                generated_artifacts=_goal_conflict_terms(
                    goal,
                    "generated_artifacts",
                    "generated_outputs",
                    "generated_paths",
                    "artifacts",
                ),
                allow_concurrent_with=_goal_conflict_terms(
                    goal,
                    "allow_concurrent_with",
                    "concurrency_overrides",
                ),
                dedupe_key=objective_evidence_obligation_key(
                    goal,
                    candidate_missing_terms,
                    graph=graph,
                    candidate_kind=candidate_kind,
                ),
            )
            findings.append(finding)
            if not forced_goal:
                seen.add(fingerprint)
            if len(findings) >= candidate_limit:
                break
        if len(findings) >= candidate_limit:
            break
    packeted_findings = assign_goal_subgoal_packets(plan_semantic_ast_bundles(findings))
    expanded_findings = add_goal_packet_aggregate_findings(
        packeted_findings,
        max_findings=candidate_limit,
        seen_fingerprints=seen_fingerprints,
        summary_prefix=summary_prefix,
    )
    return prioritize_larger_work_surface_findings(expanded_findings, max_findings=max_findings)


def task_ids_from_todo(todo_text: str, *, task_prefix: str = DEFAULT_TASK_PREFIX) -> list[str]:
    ids: list[str] = []
    header_prefix = f"## {task_prefix}"
    for line in todo_text.splitlines():
        if line.startswith(header_prefix):
            parts = line[3:].strip().split(" ", 1)
            if parts:
                ids.append(parts[0])
    return ids


def canonical_task_cids_from_todo(todo_text: str) -> set[str]:
    """Return canonical task identities already materialized on a board."""

    prefix = "- Canonical task CID:"
    return {
        line[len(prefix) :].strip()
        for line in todo_text.splitlines()
        if line.startswith(prefix) and line[len(prefix) :].strip()
    }


def objective_evidence_obligation_keys_from_todo(
    todo_text: str,
    *,
    goals: Sequence[ObjectiveGoal],
    graph: Mapping[str, Any],
) -> set[str]:
    """Recover stable evidence identities from current and legacy task blocks.

    Boards created before evidence-obligation IDs existed retain useful work.
    A refinement child may also have been removed while its generated task
    still names an active graph parent. Reconstructing the key from the exact
    parent criterion prevents a one-time duplicate refill during migration.
    """

    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in todo_text.splitlines():
        if re.match(r"^##\s+[A-Z][A-Z0-9]*-\d+\b", line):
            if current is not None:
                blocks.append(current)
            current = {}
            continue
        if current is None or not line.startswith("- ") or ":" not in line:
            continue
        name, value = line[2:].split(":", 1)
        current[name.strip().casefold()] = value.strip()
    if current is not None:
        blocks.append(current)

    goals_by_id = {goal.goal_id: goal for goal in goals}
    keys: set[str] = set()
    for fields in blocks:
        explicit = fields.get("evidence obligation key", "").strip()
        if explicit:
            keys.add(explicit)
            continue
        missing_text = normalize_objective_evidence_requirement(
            fields.get("missing evidence", "")
        )
        if not missing_text:
            continue
        raw_goal_ids = " ".join(
            fields.get(name, "")
            for name in (
                "goal id",
                "graph parents",
                "goal packet goals",
            )
        )
        candidate_goal_ids = re.findall(
            r"\b[A-Z][A-Z0-9]*-G\d+\b",
            raw_goal_ids,
        )
        for goal_id in dict.fromkeys(candidate_goal_ids):
            goal = goals_by_id.get(goal_id)
            if goal is None:
                continue
            matching_terms = [
                term
                for term in goal.required_evidence
                if normalize_objective_evidence_requirement(term)
                in missing_text
            ]
            if not matching_terms:
                continue
            keys.add(
                objective_evidence_obligation_key(
                    goal,
                    matching_terms,
                    graph=graph,
                    candidate_kind=fields.get("candidate kind", "aggregate"),
                )
            )
            break
    return keys


def next_task_id(
    todo_text: str,
    *,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    reserved_task_ids: Iterable[str] = (),
) -> str:
    highest = 0
    normalized = task_prefix.rstrip("-")
    task_ids = [
        *task_ids_from_todo(todo_text, task_prefix=f"{normalized}-"),
        *(str(item) for item in reserved_task_ids),
    ]
    for task_id in task_ids:
        try:
            prefix, number = task_id.rsplit("-", 1)
            if prefix != normalized:
                continue
            highest = max(highest, int(number))
        except (IndexError, ValueError):
            continue
    return f"{normalized}-{highest + 1:03d}"


def bundle_path(bundle_dir: Path, bundle_key: str) -> Path:
    return bundle_dir / f"{safe_bundle_key(bundle_key)}.todo.md"


def write_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    finding: ObjectiveFinding,
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    path = discovery_dir / f"{date}-{task_id.lower()}-objective-gap-{finding.fingerprint[:12]}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    missing = "\n".join(f"- {term}" for term in finding.missing_evidence) or "- none"
    present_items: list[str] = []
    for term, paths in finding.present_evidence.items():
        present_items.append(f"- {term}: {', '.join(str(path) for path in paths)}")
    present = "\n".join(present_items) if present_items else "- none found for this goal"
    parents = ", ".join(finding.parent_goal_ids) or "none"
    packet_goals = ", ".join(finding.goal_packet_goal_ids) or "none"
    content = f"""# {task_id} Objective Goal Gap

Date: {date}
Fingerprint: {finding.fingerprint}
Goal id: {finding.goal_id}
Goal title: {finding.title}
Objective heap: {finding.objective_path}
Priority: {finding.priority}
Track: {finding.track}
Parent goals: {parents}
Graph depth: {finding.graph_depth}
Bundle: {finding.bundle_key}
Parallel lane: {finding.parallel_lane}
Bundle strategy: {finding.bundle_strategy}
Goal packet: {finding.goal_packet_key or "none"}
Goal packet role: {finding.goal_packet_role or "none"}
Goal packet goals: {packet_goals}
Goal packet task count: {finding.goal_packet_task_count}
Goal packet work item count: {finding.goal_packet_work_item_count}
Evidence methods: {", ".join(finding.evidence_methods) or "none"}
Embedding query: {finding.embedding_query}
AST query: {finding.ast_query}
Conflict policy: {finding.conflict_policy}
Predicted files: {", ".join(finding.predicted_files or finding.outputs) or "none"}
AST symbols: {", ".join(finding.ast_symbols) or "none"}
Interfaces: {", ".join(finding.interfaces) or "none"}
Submodules: {", ".join(finding.submodules) or "none"}
Generated artifacts: {", ".join(finding.generated_artifacts) or "none"}
Allow concurrent with: {", ".join(finding.allow_concurrent_with) or "none"}

## Goal

{finding.goal or finding.title}

## Missing Evidence

{missing}

## Present Evidence

{present}

## Suggested Handling

{finding.gap_task or "Close the missing evidence with focused code, tests, or documentation."}
"""
    path.write_text(content, encoding="utf-8")
    return path


def objective_finding_task_identity(task_id: str, finding: ObjectiveFinding) -> TaskIdentity:
    """Return the stable work identity for an objective finding."""

    return canonical_task_identity(
        {
            "task_id": task_id,
            "dedupe_key": (
                finding.dedupe_key
                or f"objective-finding:{finding.fingerprint}"
            ),
        },
        board_namespace="objective-graph",
        source_path=finding.objective_path,
    )


def objective_finding_conflict_record(task_id: str, finding: ObjectiveFinding) -> dict[str, Any]:
    """Return the canonical conflict-surface fields for a generated finding."""

    identity = objective_finding_task_identity(task_id, finding)
    predicted_files = _unique_strings([*(finding.predicted_files or finding.outputs), *finding.outputs])
    return {
        "task_id": task_id,
        "canonical_task_cid": identity.canonical_task_cid,
        "task_cid": identity.canonical_task_cid,
        "predicted_files": predicted_files,
        "files": predicted_files,
        "changed_paths": _unique_strings(finding.changed_paths),
        "outputs": _unique_strings(finding.outputs),
        "ast_symbols": _unique_strings(finding.ast_symbols or split_terms(finding.ast_query)),
        "interfaces": _unique_strings(finding.interfaces),
        "submodules": _unique_strings(finding.submodules),
        "generated_artifacts": _unique_strings(finding.generated_artifacts),
        "allow_concurrent_with": _unique_strings(finding.allow_concurrent_with),
        "conflict_policy": finding.conflict_policy,
    }


def render_task_block(
    *,
    task_id: str,
    finding: ObjectiveFinding,
    discovery_path: Path,
    depends_on: Sequence[str] = (),
    bundle_shard: str = "",
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = [discovery_output_path, finding.objective_path]
    outputs.extend(str(item) for item in finding.outputs if str(item).strip())
    unique_outputs = list(dict.fromkeys(outputs))
    missing = ", ".join(finding.missing_evidence)
    refinement = finding.refinement or "Refine the objective heap if the gap needs smaller child goals."
    parents = ", ".join(finding.parent_goal_ids) or "none"
    packet_goals = ", ".join(finding.goal_packet_goal_ids)
    packet_acceptance = (
        f"This task is part of {finding.goal_packet_key}; implement a complete, cohesive change that fully advances "
        f"the packet goals ({packet_goals}) and covers all the shared packet evidence in one comprehensive pass."
        if finding.goal_packet_key and packet_goals
        else ""
    )
    identity = objective_finding_task_identity(task_id, finding)
    bundle_shard = bundle_shard or f"data/agent_supervisor/objective_bundles/{safe_bundle_key(finding.bundle_key)}.todo.md"
    return f"""## {task_id} {finding.summary}

- Status: todo
- Completion: manual
- Priority: {finding.priority}
- Track: {finding.track}
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(unique_outputs)}
- Validation: {finding.validation}
- Bundle: {finding.bundle_key}
- Bundle shard: {bundle_shard}
- Bundle strategy: {finding.bundle_strategy}
- Graph parents: {parents}
- Graph depth: {finding.graph_depth}
- Parallel lane: {finding.parallel_lane}
- Conflict policy: {finding.conflict_policy}
- Predicted files: {", ".join(finding.predicted_files or finding.outputs)}
- Changed paths: {", ".join(finding.changed_paths)}
- AST symbols: {", ".join(finding.ast_symbols)}
- Interfaces: {", ".join(finding.interfaces)}
- Submodules: {", ".join(finding.submodules)}
- Generated artifacts: {", ".join(finding.generated_artifacts)}
- Allow concurrent with: {", ".join(finding.allow_concurrent_with)}
- Goal id: {finding.goal_id}
- Canonical task key: {identity.canonical_task_key}
- Canonical task CID: {identity.canonical_task_cid}
- Evidence obligation key: {finding.dedupe_key}
- Missing evidence: {missing}
- Embedding query: {finding.embedding_query}
- AST query: {finding.ast_query}
- Surplus group: {finding.surplus_group}
- Merge key: {finding.merge_key}
- Merge family: {finding.merge_family or finding.surplus_group}
- Merge role: {finding.merge_role or finding.candidate_kind}
- Work item count: {finding.work_item_count or len(finding.missing_evidence)}
- Work scope: {finding.work_scope or "goal_subgoal_multi_evidence_batch"}
- Goal packet: {finding.goal_packet_key}
- Goal packet role: {finding.goal_packet_role}
- Goal packet goals: {packet_goals}
- Goal packet task count: {finding.goal_packet_task_count}
- Goal packet work item count: {finding.goal_packet_work_item_count}
- Candidate kind: {finding.candidate_kind}
- Todo vector key: {finding.todo_vector_key}
- Acceptance: Objective scan filed this gap for {finding.goal_id}. Use evidence in {discovery_path}, add code/tests/docs or child goals that prove the missing evidence terms are covered ({missing}), and keep the supervisor-fed backlog aligned with the objective heap. {packet_acceptance} {refinement}
"""


def write_bundle_shards(
    *,
    bundle_dir: Path,
    repo_root: Path,
    todo_path: Path,
    records: Sequence[ObjectiveTaskRecord],
) -> BundleWriteResult:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    bundle_paths: dict[str, Path] = {}
    source_todo = repo_relative_path(repo_root, todo_path)
    groups: dict[str, list[ObjectiveTaskRecord]] = {}
    for record in records:
        groups.setdefault(record.finding.bundle_key, []).append(record)

    generated_planning_graph = materialize_task_planning_graph(
        [
            {
                **objective_finding_conflict_record(record.task_id, record.finding),
                "task_id": record.task_id,
                "canonical_task_cid": objective_finding_task_identity(record.task_id, record.finding).canonical_task_cid,
                "goal_id": record.finding.goal_id,
                "parent_goal_ids": record.finding.parent_goal_ids,
                "priority": record.finding.priority,
                "outputs": record.finding.outputs,
                "work_item_count": record.finding.work_item_count or len(record.finding.missing_evidence),
                "status": "todo",
            }
            for record in records
        ]
    )
    generated_graph = generated_planning_graph.dependency_graph
    generated_incoming: dict[str, set[str]] = {cid: set() for cid in generated_graph.nodes}
    for edge in generated_graph.edges:
        generated_incoming.setdefault(edge.target_task_cid, set()).add(edge.source_task_cid)
    generated_schedule = {item.task_cid: item for item in generated_graph.schedule}

    for key, bundle_records in sorted(groups.items()):
        shard_path = bundle_path(bundle_dir, key)
        bundle_paths[key] = shard_path
        if shard_path.exists():
            shard_text = shard_path.read_text(encoding="utf-8")
        else:
            shard_text = (
                f"# Objective Bundle: {key}\n\n"
                f"Source todo: {source_todo}\n"
                "Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.\n"
                "Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.\n"
            )

        changed = False
        for record in bundle_records:
            if f"## {record.task_id} " in shard_text:
                continue
            shard_text = shard_text.rstrip() + "\n\n" + record.task_block.strip() + "\n"
            changed = True
        if changed or not shard_path.exists():
            shard_path.write_text(shard_text, encoding="utf-8")
            generated_paths.append(shard_path)

    index_path = bundle_dir / "index.json"
    if index_path.exists():
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index_payload = {}
    else:
        index_payload = {}
    if not isinstance(index_payload, dict):
        index_payload = {}
    bundles = index_payload.get("bundles")
    if not isinstance(bundles, dict):
        bundles = {}

    for key, bundle_records in sorted(groups.items()):
        task_map: dict[str, dict[str, Any]] = {}
        existing = bundles.get(key, {})
        if isinstance(existing, Mapping):
            for item in existing.get("tasks", []) if isinstance(existing.get("tasks"), list) else []:
                if isinstance(item, Mapping) and str(item.get("task_id") or ""):
                    task_map[str(item["task_id"])] = dict(item)
        for record in bundle_records:
            identity = objective_finding_task_identity(record.task_id, record.finding)
            schedule_record = generated_schedule.get(identity.canonical_task_cid)
            existing_task = task_map.get(record.task_id, {})
            task_payload = {
                **objective_finding_conflict_record(record.task_id, record.finding),
                "task_id": record.task_id,
                "canonical_task_key": identity.canonical_task_key,
                "canonical_task_cid": identity.canonical_task_cid,
                "goal_id": record.finding.goal_id,
                "graph_depth": record.finding.graph_depth,
                "parent_goal_ids": record.finding.parent_goal_ids,
                "missing_evidence": record.finding.missing_evidence,
                "discovery_path": repo_relative_path(repo_root, record.discovery_path),
                "bundle_strategy": record.finding.bundle_strategy,
                "surplus_group": record.finding.surplus_group,
                "merge_key": record.finding.merge_key,
                "merge_family": record.finding.merge_family or record.finding.surplus_group,
                "merge_role": record.finding.merge_role or record.finding.candidate_kind,
                "work_item_count": record.finding.work_item_count or len(record.finding.missing_evidence),
                "work_scope": record.finding.work_scope or "goal_subgoal_multi_evidence_batch",
                "goal_packet_key": record.finding.goal_packet_key,
                "goal_packet_role": record.finding.goal_packet_role,
                "goal_packet_goal_ids": record.finding.goal_packet_goal_ids,
                "goal_packet_task_count": record.finding.goal_packet_task_count,
                "goal_packet_work_item_count": record.finding.goal_packet_work_item_count,
                "candidate_kind": record.finding.candidate_kind,
                "todo_vector_key": record.finding.todo_vector_key,
                "dependency_task_cids": sorted(generated_incoming.get(identity.canonical_task_cid, set())),
                "critical_path_length": schedule_record.critical_path_length if schedule_record else 1,
                "slack": schedule_record.slack if schedule_record else 0,
                "downstream_unlock_value": schedule_record.downstream_unlock_value if schedule_record else 0,
                "objective_priority": schedule_record.objective_priority if schedule_record else 0,
            }
            existing_status = str(existing_task.get("status") or "").strip()
            if existing_status:
                task_payload["status"] = existing_status
            task_map[record.task_id] = task_payload
        bundles[key] = {
            "bundle_key": key,
            "shard_path": repo_relative_path(repo_root, bundle_path(bundle_dir, key)),
            "parallel_lane": bundle_records[0].finding.parallel_lane,
            "bundle_strategy": bundle_records[0].finding.bundle_strategy,
            "conflict_policy": bundle_records[0].finding.conflict_policy,
            "tasks": [task_map[task_id] for task_id in sorted(task_map)],
        }

    all_index_tasks = [
        dict(item)
        for info in bundles.values()
        if isinstance(info, Mapping)
        for item in (info.get("tasks") or [])
        if isinstance(item, Mapping)
    ]
    index_planning_graph = materialize_task_planning_graph(all_index_tasks, repo_root=repo_root)
    index_graph = index_planning_graph.dependency_graph
    index_incoming: dict[str, set[str]] = {cid: set() for cid in index_graph.nodes}
    for edge in index_graph.edges:
        index_incoming.setdefault(edge.target_task_cid, set()).add(edge.source_task_cid)
    index_schedule = {item.task_cid: item for item in index_graph.schedule}
    for info in bundles.values():
        if not isinstance(info, dict):
            continue
        annotated: list[dict[str, Any]] = []
        for raw_task in info.get("tasks") or []:
            if not isinstance(raw_task, Mapping):
                continue
            item = dict(raw_task)
            cid = str(item.get("canonical_task_cid") or item.get("task_cid") or "")
            scheduled = index_schedule.get(cid)
            item["dependency_task_cids"] = sorted(index_incoming.get(cid, set()))
            if scheduled:
                item.update(
                    {
                        "critical_path_length": scheduled.critical_path_length,
                        "slack": scheduled.slack,
                        "downstream_unlock_value": scheduled.downstream_unlock_value,
                        "age_seconds": scheduled.age_seconds,
                        "objective_priority": scheduled.objective_priority,
                        "schedule_score": scheduled.score,
                    }
                )
            annotated.append(item)
        info["tasks"] = annotated

    index_payload["generated_at"] = utc_now()
    index_payload["source_todo"] = source_todo
    index_payload["bundles"] = bundles
    index_payload["task_dependency_graph"] = index_graph.to_dict()
    index_payload["dependency_dag"] = index_graph.to_dict()
    index_payload["task_conflict_graph"] = index_planning_graph.conflict_graph.to_dict()
    index_payload["conflict_graph"] = index_planning_graph.conflict_graph.to_dict()
    index_payload["task_planning_graph"] = index_planning_graph.to_dict()
    from .artifact_store import write_bundle_index_artifact

    write_bundle_index_artifact(index_path, index_payload)
    generated_paths.append(index_path)
    return BundleWriteResult(generated_paths=generated_paths, index_path=index_path, bundle_paths=bundle_paths)


def generate_objective_todos(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    discovery_dir: Path,
    bundle_dir: Path,
    dataset_dir: Path | None = None,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    depends_on: Sequence[str] = (),
    max_findings: int = 10,
    seen_fingerprints: Iterable[str] = (),
    force_goal_ids: Iterable[str] = (),
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    todo_vector_index_path: Path | None = None,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    precomputed_findings: Sequence[ObjectiveFinding] | None = None,
    typed_evidence_receipts: Sequence[Mapping[str, Any] | Any] = (),
    evidence_source_policy: EvidenceSourcePolicy | None = None,
    evidence_repository_tree: str = "",
    evidence_policy_id: str = "",
) -> list[ObjectiveTaskRecord]:
    """Append generated objective gap tasks and write bundle shards."""

    records: list[ObjectiveTaskRecord] = []
    if precomputed_findings is None:
        findings = scan_objective_gaps(
            repo_root,
            objective_path=objective_path,
            max_findings=max_findings,
            seen_fingerprints=seen_fingerprints,
            force_goal_ids=force_goal_ids,
            summary_prefix=summary_prefix,
            surplus_findings_per_goal=surplus_findings_per_goal,
            surplus_min_terms_per_todo=surplus_min_terms_per_todo,
            dataset_dir=(dataset_dir or bundle_dir.parent / "objective_datasets") if persist_ast_dataset else None,
            dataset_id=f"{task_prefix.rstrip('-').lower()}-objective-ast",
            typed_evidence_receipts=typed_evidence_receipts,
            evidence_source_policy=evidence_source_policy,
            evidence_repository_tree=evidence_repository_tree,
            evidence_policy_id=evidence_policy_id,
        )
    else:
        findings = list(precomputed_findings)
    with locked_taskboard(todo_path) as taskboard:
        todo_text = taskboard.read() or "# Objective Todo\n"
        existing_canonical_task_cids = canonical_task_cids_from_todo(todo_text)
        objective_goals = parse_goal_heap(
            objective_path.read_text(encoding="utf-8")
        )
        objective_goal_graph = goal_graph(objective_goals)
        existing_obligation_keys = objective_evidence_obligation_keys_from_todo(
            todo_text,
            goals=objective_goals,
            graph=objective_goal_graph,
        )
        reserved_task_ids = task_ids_from_artifact_names(
            discovery_dir,
            task_prefix=task_prefix,
        )
        for finding in findings:
            task_id = next_task_id(
                todo_text,
                task_prefix=task_prefix,
                reserved_task_ids=reserved_task_ids,
            )
            identity = objective_finding_task_identity(task_id, finding)
            if (
                identity.canonical_task_cid in existing_canonical_task_cids
                or (
                    finding.dedupe_key
                    and finding.dedupe_key in existing_obligation_keys
                )
            ):
                continue
            reserved_task_ids.add(task_id)
            existing_canonical_task_cids.add(identity.canonical_task_cid)
            if finding.dedupe_key:
                existing_obligation_keys.add(finding.dedupe_key)
            shard_relative = repo_relative_path(
                repo_root, bundle_path(bundle_dir, finding.bundle_key)
            )
            discovery_path = write_discovery(
                discovery_dir=discovery_dir,
                task_id=task_id,
                finding=finding,
            )
            task_block = render_task_block(
                task_id=task_id,
                finding=finding,
                discovery_path=discovery_path,
                depends_on=depends_on,
                bundle_shard=shard_relative,
                discovery_output_path=discovery_output_path,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            records.append(
                ObjectiveTaskRecord(
                    task_id=task_id,
                    task_block=task_block,
                    finding=finding,
                    discovery_path=discovery_path,
                )
            )

        if records:
            replace_locked_taskboard(taskboard, todo_text)

    if not records:
        return []
    bundle_result = write_bundle_shards(bundle_dir=bundle_dir, repo_root=repo_root, todo_path=todo_path, records=records)
    if write_todo_vector_index:
        from .todo_vector_index import write_todo_vector_index as write_index

        task_header_prefix = task_prefix.strip()
        if not task_header_prefix.startswith("## "):
            task_header_prefix = f"## {task_header_prefix.rstrip('-')}-"
        write_index(
            repo_root=repo_root,
            todo_path=todo_path,
            index_path=todo_vector_index_path or bundle_dir / "todo_vector_index.json",
            task_header_prefix=task_header_prefix,
            objective_path=objective_path,
            bundle_index_path=bundle_result.index_path,
            dataset_dir=(dataset_dir or bundle_dir.parent / "objective_datasets") if persist_ast_dataset else None,
            dataset_id=f"{task_prefix.rstrip('-').lower()}-todo-vector-index",
            persist_dataset=persist_ast_dataset,
        )
    return records


def generate_objective_todos_result(
    *,
    scan_mode: str = "direct",
    **kwargs: Any,
) -> RefillScanResult[ObjectiveTaskRecord]:
    """Generate objective todos and describe the terminal scan outcome.

    ``generate_objective_todos`` remains the list-returning primitive for
    callers that are explicitly operating on task records.  Refill and goal
    completion code should use this typed boundary so an empty collection can
    never be mistaken for successful exhaustion without a terminal reason.
    """

    started_at = datetime.now(timezone.utc)
    repo_root = Path(kwargs.get("repo_root") or ".").resolve()
    try:
        max_findings = int(kwargs.get("max_findings", 10))
    except (TypeError, ValueError):
        max_findings = 0
    if max_findings <= 0:
        return build_scan_result(
            ScanTerminalReason.DISABLED,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={"cause": "non_positive_max_findings"},
        )

    objective_path = Path(kwargs.get("objective_path") or "")
    todo_path = Path(kwargs.get("todo_path") or "")
    missing_inputs = [
        name
        for name, path in (("objective_path", objective_path), ("todo_path", todo_path))
        if not path.exists()
    ]
    if missing_inputs:
        return build_scan_result(
            ScanTerminalReason.FAILED,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=f"missing required scan input: {', '.join(missing_inputs)}",
            metadata={"missing_inputs": missing_inputs},
        )

    try:
        records = generate_objective_todos(**kwargs)
    except TimeoutError as exc:
        return build_scan_result(
            ScanTerminalReason.TIMED_OUT,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=str(exc) or type(exc).__name__,
        )
    except Exception as exc:
        return build_scan_result(
            ScanTerminalReason.FAILED,
            scan_mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=f"{type(exc).__name__}: {exc}",
        )

    terminal_reason = ScanTerminalReason.GENERATED
    duplicate_candidate_count = 0
    if not records:
        terminal_reason = ScanTerminalReason.EXHAUSTED
        seen_fingerprints = tuple(kwargs.get("seen_fingerprints") or ())
        if seen_fingerprints:
            try:
                duplicate_candidates = scan_objective_gaps(
                    repo_root,
                    objective_path=objective_path,
                    max_findings=max_findings,
                    seen_fingerprints=(),
                    force_goal_ids=kwargs.get("force_goal_ids") or (),
                    summary_prefix=str(
                        kwargs.get("summary_prefix") or DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX
                    ),
                    surplus_findings_per_goal=int(
                        kwargs.get("surplus_findings_per_goal", DEFAULT_SURPLUS_FINDINGS_PER_GOAL)
                    ),
                    surplus_min_terms_per_todo=int(
                        kwargs.get("surplus_min_terms_per_todo", DEFAULT_SURPLUS_MIN_TERMS_PER_TODO)
                    ),
                    typed_evidence_receipts=tuple(
                        kwargs.get("typed_evidence_receipts") or ()
                    ),
                    evidence_source_policy=kwargs.get("evidence_source_policy"),
                    evidence_repository_tree=str(
                        kwargs.get("evidence_repository_tree") or ""
                    ),
                    evidence_policy_id=str(
                        kwargs.get("evidence_policy_id") or ""
                    ),
                )
            except TimeoutError as exc:
                return build_scan_result(
                    ScanTerminalReason.TIMED_OUT,
                    scan_mode,
                    OBJECTIVE_SCAN_ANALYZER_VERSION,
                    repo_root,
                    started_at,
                    error=str(exc) or type(exc).__name__,
                )
            except Exception as exc:
                return build_scan_result(
                    ScanTerminalReason.FAILED,
                    scan_mode,
                    OBJECTIVE_SCAN_ANALYZER_VERSION,
                    repo_root,
                    started_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
            duplicate_candidate_count = len(duplicate_candidates)
            if duplicate_candidates:
                terminal_reason = ScanTerminalReason.DUPLICATE_ONLY
    return build_scan_result(
        terminal_reason,
        scan_mode,
        OBJECTIVE_SCAN_ANALYZER_VERSION,
        repo_root,
        started_at,
        records,
        safe_for_completion_reasoning=(
            terminal_reason is ScanTerminalReason.EXHAUSTED
            and str(scan_mode).endswith("exhaustive")
        ),
        metadata={
            "candidate_count": len(records),
            "duplicate_candidate_count": duplicate_candidate_count,
        },
    )


def _profile_g_safe_planning_value(value: Any) -> Any:
    """Encode graph weights without violating Profile G's no-float codec."""

    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return format(value, ".12g")
    if isinstance(value, Mapping):
        return {str(key): _profile_g_safe_planning_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_profile_g_safe_planning_value(item) for item in value]
    return value


def _project_task_conflict_graph(
    graph: Mapping[str, Any], member_task_cids: set[str]
) -> dict[str, Any]:
    """Return the incident conflict evidence needed by one bundle payload.

    The complete O(n²) decision matrix belongs once in the bundle index.  Queue
    payloads carry only their member surfaces, assignments, and incident
    decisions/edges; otherwise serializing and hashing n full graphs creates
    cubic planning work for large objective boards.
    """

    def incident(item: Any) -> bool:
        if not isinstance(item, Mapping):
            return False
        left = str(item.get("left_task_cid") or item.get("left") or "")
        right = str(item.get("right_task_cid") or item.get("right") or "")
        return bool({left, right} & member_task_cids)

    surfaces = graph.get("surfaces") if isinstance(graph.get("surfaces"), Mapping) else {}
    assignments = graph.get("assignments") if isinstance(graph.get("assignments"), list) else []
    lanes = graph.get("lanes") if isinstance(graph.get("lanes"), Mapping) else {}
    projected_lanes: dict[str, list[str]] = {}
    for color, raw_task_cids in lanes.items():
        if not isinstance(raw_task_cids, list):
            continue
        selected = [str(cid) for cid in raw_task_cids if str(cid) in member_task_cids]
        if selected:
            projected_lanes[str(color)] = selected
    return {
        "schema": graph.get("schema", "ipfs_accelerate_py.agent_supervisor.conflict_graph@1"),
        "projection": "bundle_incident",
        "surfaces": {
            cid: dict(surface)
            for cid, surface in surfaces.items()
            if str(cid) in member_task_cids and isinstance(surface, Mapping)
        },
        "edges": [dict(item) for item in graph.get("edges", []) if incident(item)],
        "assignments": [
            dict(item)
            for item in assignments
            if isinstance(item, Mapping) and str(item.get("task_cid") or "") in member_task_cids
        ],
        "decisions": [dict(item) for item in graph.get("decisions", []) if incident(item)],
        "lanes": projected_lanes,
    }


def build_bundle_task_payloads(bundle_index_path: Path) -> list[dict[str, Any]]:
    """Build dependency-aware task-queue payloads and Profile G adapters."""

    # Local import avoids making objective scanning depend on coordination
    # initialization while ensuring queue consumers receive immutable links.
    from .artifact_store import read_bundle_index_planning_projection
    from .lease_coordination import adapt_goal_bundle

    payload = read_bundle_index_planning_projection(
        bundle_index_path,
        field_names=("source_todo", "generated_at"),
    )
    bundles = payload.get("bundles") if isinstance(payload, Mapping) else {}
    if not isinstance(bundles, Mapping):
        return []
    profile_created_at_ms = _task_created_at_ms(
        {"created_at": payload.get("generated_at")}
    )
    if profile_created_at_ms <= 0:
        try:
            profile_created_at_ms = bundle_index_path.stat().st_mtime_ns // 1_000_000
        except OSError:
            profile_created_at_ms = 0
    task_payloads: list[dict[str, Any]] = []
    for key, info in sorted(bundles.items()):
        if not isinstance(info, Mapping):
            continue
        task_payload = {
            "bundle_key": str(key),
            "todo_path": info.get("shard_path", ""),
            "parallel_lane": info.get("parallel_lane", key),
            "conflict_policy": info.get("conflict_policy", ""),
            "tasks": info.get("tasks", []),
            "source_todo": payload.get("source_todo", ""),
            "objective_bundle_index": str(bundle_index_path),
        }
        task_payloads.append(task_payload)

    flat_tasks = [
        {**dict(item), "bundle_key": str(bundle_payload.get("bundle_key") or "objective/general")}
        for bundle_payload in task_payloads
        for item in bundle_payload.get("tasks", [])
        if isinstance(item, Mapping)
    ]
    terminal_receipts = {
        str(task.get("canonical_task_cid") or task.get("task_cid") or task.get("task_id") or ""): {
            "status": "succeeded"
        }
        for task in flat_tasks
        if str(task.get("status") or "").strip().lower() in SUCCESSFUL_MERGE_RECEIPT_STATUSES
        and str(task.get("canonical_task_cid") or task.get("task_cid") or task.get("task_id") or "")
    }
    graph = materialize_task_dependency_dag(
        flat_tasks,
        merge_receipts=terminal_receipts,
    )
    invalid_task_cids = set(graph.invalid_task_cids)
    schedule_by_cid = {item.task_cid: item for item in graph.schedule}
    member_cids_by_bundle_and_id = {
        (str(node.metadata.get("bundle_key") or ""), node.task_id): cid
        for cid, node in graph.nodes.items()
    }
    incoming: dict[str, set[str]] = {cid: set() for cid in graph.nodes}
    for edge in graph.edges:
        incoming.setdefault(edge.target_task_cid, set()).add(edge.source_task_cid)
    unresolved_incoming = {
        cid: (
            set()
            if node.status in SUCCESSFUL_MERGE_RECEIPT_STATUSES
            else set(schedule_by_cid.get(cid).blocking_task_cids if cid in schedule_by_cid else incoming.get(cid, set()))
        )
        for cid, node in graph.nodes.items()
    }

    member_bundle: dict[str, str] = {}
    bundle_identity_cids: dict[str, str] = {}
    for bundle_payload in task_payloads:
        bundle_key = str(bundle_payload["bundle_key"])
        bundle_identity_cids[bundle_key] = canonical_bundle_identity(bundle_payload).canonical_task_cid
        annotated_tasks: list[dict[str, Any]] = []
        for raw_task in bundle_payload.get("tasks", []):
            if not isinstance(raw_task, Mapping):
                continue
            item = dict(raw_task)
            cid = str(item.get("canonical_task_cid") or item.get("task_cid") or "")
            if not cid:
                cid = member_cids_by_bundle_and_id.get((bundle_key, str(item.get("task_id") or "")), "")
            if not cid:
                cid = canonical_task_identity(
                    {
                        **item,
                        "semantic_key": f"{bundle_key}:{item.get('task_id') or len(annotated_tasks)}",
                    }
                ).canonical_task_cid
            item["canonical_task_cid"] = cid
            member_bundle[cid] = bundle_key
            item["dependency_task_cids"] = sorted(incoming.get(cid, set()))
            scheduled = schedule_by_cid.get(cid)
            if scheduled:
                item.update(
                    {
                        "claimable": scheduled.claimable,
                        "blocking_task_cids": scheduled.blocking_task_cids,
                        "critical_path_length": scheduled.critical_path_length,
                        "slack": scheduled.slack,
                        "downstream_unlock_value": scheduled.downstream_unlock_value,
                        "age_seconds": scheduled.age_seconds,
                        "objective_priority": scheduled.objective_priority,
                        "schedule_score": scheduled.score,
                    }
                )
            annotated_tasks.append(item)
        bundle_payload["tasks"] = annotated_tasks

    for bundle_payload in task_payloads:
        bundle_key = str(bundle_payload["bundle_key"])
        member_cids = {
            str(item.get("canonical_task_cid") or item.get("task_cid") or "")
            for item in bundle_payload.get("tasks", [])
            if isinstance(item, Mapping)
        }
        completed_member_cids = {
            cid
            for cid in member_cids
            if cid in graph.nodes
            and graph.nodes[cid].status in SUCCESSFUL_MERGE_RECEIPT_STATUSES
        }
        active_member_cids = {
            cid
            for cid in member_cids
            if cid in graph.nodes
            and graph.nodes[cid].status
            in {"active", "implementing", "in_progress", "running"}
        }
        blocked_member_cids = {
            cid
            for cid in member_cids
            if cid in graph.nodes
            and graph.nodes[cid].status in {"blocked", "on_hold"}
        }
        unfinished_member_cids = member_cids - completed_member_cids
        ready_member_cids = {
            cid
            for cid in unfinished_member_cids - active_member_cids - blocked_member_cids
            if cid in schedule_by_cid and schedule_by_cid[cid].claimable
        }
        # Until member leases are shared across serial and bundle schedulers,
        # an externally active member fences the whole bundle from a duplicate
        # launch. Otherwise, lease only the dependency-closed ready slice.
        execution_member_cids = (
            set()
            if active_member_cids
            else (ready_member_cids or unfinished_member_cids)
        )
        deferred_member_cids = unfinished_member_cids - ready_member_cids
        schedule_order = {
            item.task_cid: index for index, item in enumerate(graph.schedule)
        }

        def ordered(cids: set[str]) -> list[str]:
            return sorted(cids, key=lambda cid: (schedule_order.get(cid, len(schedule_order)), cid))

        def task_ids(cids: set[str]) -> list[str]:
            return [
                graph.nodes[cid].task_id
                for cid in ordered(cids)
                if cid in graph.nodes and graph.nodes[cid].task_id
            ]

        dependency_bundle_keys = {
            member_bundle[source]
            for target in execution_member_cids
            for source in unresolved_incoming.get(target, set())
            if source in member_bundle and member_bundle[source] != bundle_key
        }
        dependency_task_cids = sorted(bundle_identity_cids[key] for key in dependency_bundle_keys)
        member_schedule = [
            schedule_by_cid[cid]
            for cid in ordered(execution_member_cids)
            if cid in schedule_by_cid
        ]
        repair_evidence: list[dict[str, Any]] = []
        bundle_resolved_cycle_cids: set[str] = set()
        for item in graph.repair_evidence:
            if item.task_cid not in execution_member_cids:
                continue
            component = {
                str(cid)
                for cid in item.provenance.get("component_task_cids", [])
                if str(cid)
            }
            if item.kind == "dependency_cycle" and component and component <= member_cids:
                # A lane owns every member of this strongly connected
                # component, so the cycle is an internal implementation order
                # concern rather than an impossible cross-lane prerequisite.
                bundle_resolved_cycle_cids.update(component)
                continue
            repair_evidence.append(item.to_dict())
        blocking_repair_cids = {
            str(item.get("task_cid") or "")
            for item in repair_evidence
            if str(item.get("task_cid") or "")
        }
        invalid_member_cids = sorted(
            blocking_repair_cids
            | ((execution_member_cids & invalid_task_cids) - bundle_resolved_cycle_cids)
        )
        if invalid_member_cids and not repair_evidence:
            # The graph keeps the complete invalid-CID set even when detailed
            # repair evidence reaches its global bound. Preserve one compact
            # bundle-local marker so truncation can never make invalid work
            # appear claimable at the lease boundary.
            repair_evidence.append(
                {
                    "kind": "missing_dependency",
                    "task_cid": invalid_member_cids[0],
                    "task_id": "",
                    "reference": "bounded_dependency_repair_evidence",
                    "message": "dependency repair details were truncated; regenerate or repair the task DAG",
                    "provenance": {
                        "invalid_task_cids": invalid_member_cids[:16],
                        "evidence_truncated": True,
                    },
                }
            )
        external_blockers = sorted(
            {
                bundle_identity_cids[member_bundle[source]]
                for target in execution_member_cids
                for source in unresolved_incoming.get(target, set())
                if source in member_bundle and member_bundle[source] != bundle_key
            }
        )
        projected_dependency_graph = {
            "schema": "ipfs_accelerate_py.agent_supervisor.bundle_dependency_projection@1",
            "projection": "bundle_incident",
            "task_cids": ordered(member_cids),
            "claimable_task_cids": ordered(ready_member_cids),
            "edges": [
                edge.to_dict()
                for edge in graph.edges
                if edge.source_task_cid in member_cids
                or edge.target_task_cid in member_cids
            ],
        }
        claimable = (
            not active_member_cids
            and (
                not unfinished_member_cids
                or bool(ready_member_cids)
                or (
                    bool(bundle_resolved_cycle_cids)
                    and not external_blockers
                    and not invalid_member_cids
                )
            )
        )
        bundle_payload.update(
            {
                "canonical_task_cid": bundle_identity_cids[bundle_key],
                "dependency_task_cids": dependency_task_cids,
                "blocking_task_cids": external_blockers,
                "claimable": claimable,
                "ready_member_task_cids": ordered(ready_member_cids),
                "ready_member_task_ids": task_ids(ready_member_cids),
                "deferred_member_task_cids": ordered(deferred_member_cids),
                "deferred_member_task_ids": task_ids(deferred_member_cids),
                "completed_member_task_cids": ordered(completed_member_cids),
                "completed_member_task_ids": task_ids(completed_member_cids),
                "active_member_task_cids": ordered(active_member_cids),
                "active_member_task_ids": task_ids(active_member_cids),
                "blocked_member_task_cids": ordered(blocked_member_cids),
                "blocked_member_task_ids": task_ids(blocked_member_cids),
                "execution_slice_task_cids": ordered(execution_member_cids),
                "execution_slice_task_ids": task_ids(execution_member_cids),
                "critical_path_length": max((item.critical_path_length for item in member_schedule), default=1),
                "slack": min((item.slack for item in member_schedule), default=0),
                "downstream_unlock_value": max(
                    (item.downstream_unlock_value for item in member_schedule), default=0
                ),
                "age_seconds": max((item.age_seconds for item in member_schedule), default=0),
                "objective_priority": max((item.objective_priority for item in member_schedule), default=0),
                "schedule_score": max((item.score for item in member_schedule), default=0),
                "dependency_repair_evidence": repair_evidence,
                "task_dependency_graph": projected_dependency_graph,
                "dependency_dag": projected_dependency_graph,
                "planning_evidence_ref": {
                    "schema": "ipfs_accelerate_py.agent_supervisor.planning_evidence_ref@1",
                    "bundle_index": str(bundle_index_path),
                    "bundle_index_duckdb": str(bundle_index_path.with_suffix(".duckdb")),
                    "bundle_key": bundle_key,
                    "bundle_table": "bundles",
                    "task_table": "bundle_tasks",
                    "dependency_table": "bundle_task_dependencies",
                    "dependency_edge_table": "dependency_edges",
                    "conflict_edge_table": "conflict_edges",
                    "planning_decision_table": "planning_decisions",
                },
            }
        )

    task_payloads.sort(
        key=lambda item: (
            0 if item["claimable"] else 1,
            -int(item["critical_path_length"]),
            int(item["slack"]),
            -int(item["downstream_unlock_value"]),
            -int(item["age_seconds"]),
            -int(item["objective_priority"]),
            str(item["bundle_key"]),
        )
    )
    for rank, task_payload in enumerate(task_payloads):
        task_payload["schedule_rank"] = rank
        task_payload["profile_g"] = adapt_goal_bundle(
            _profile_g_safe_planning_value(task_payload),
            created_at_ms=profile_created_at_ms,
        )
    return task_payloads


def submit_bundle_tasks(
    bundle_index_path: Path,
    *,
    queue: Any = None,
    queue_path: str | None = None,
    task_type: str = "codex.todo_bundle",
    model_name: str = "codex",
) -> list[str]:
    """Submit bundle shards to the ipfs_accelerate task queue.

    The queue parameter is injectable for tests.  When omitted, the local
    ``TaskQueue`` is used without importing ipfs_datasets_py.
    """

    if queue is None:
        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

        queue = TaskQueue(path=queue_path)
    task_ids: list[str] = []
    for payload in build_bundle_task_payloads(bundle_index_path):
        task_ids.append(
            queue.submit(
                task_type=task_type,
                model_name=model_name,
                payload=payload,
            )
        )
    return task_ids
