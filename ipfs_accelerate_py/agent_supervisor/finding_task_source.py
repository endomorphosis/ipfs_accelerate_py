"""Stable repair task source from admitted contract findings (VFS-031 / VFS-050).

Materializes the second repair taskboard (objective VFS-G101, evidence
``vfs/finding-taskboard@1``) from fresh admitted finding CIDs without letting
a report authorize edits.

Convert only *fresh admitted* findings into goal-backed repair tasks that bind:

* one root-cause family;
* exact output files, symbols, and effects;
* dependencies and a conflict domain;
* a validation/proof plan;
* goal lineage (objective heap ancestry) and stable identity;
* finding and provenance CIDs;
* risk, resource class, and a context ceiling.

Ambiguous, over-broad, or out-of-root findings produce **non-executable**
review records.  Stable findings replay as a no-op; changed evidence
supersedes the prior task rather than duplicating it; related tiny tasks
coalesce only when they share validation commands and merge fate.

Markdown, DuckDB-row, JSON, and SARIF-linked projections are diagnostic only
and never authorize edits or completion (no authority drift).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

from .contract_findings import (
    FINDING_LEDGER_EVIDENCE,
    FINDING_LEDGER_G100_EVIDENCE_TERMS,
    VULNERABILITY_EVIDENCE_POLICY,
    ContractFindingLedger,
    ContractFindingRecord,
    EvidenceFreshness,
    FindingStatus,
    build_contract_finding,
    finding_ledger_evidence_terms,
)
from .planning.task_quality import (
    RESOURCE_CLASSES,
    TOKEN_CLASSES,
    TOKEN_CLASS_LIMITS,
    TaskCandidate,
    TaskQualityPolicy,
    is_over_broad,
    is_tiny,
)
from .proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)
from .task_sources.task_identity import (
    TaskIdentity,
    canonical_task_identity,
    normalize_identity_path,
    normalize_identity_text,
)


# ---------------------------------------------------------------------------
# Version / bounds / authority flags
# ---------------------------------------------------------------------------

FINDING_TASK_SOURCE_VERSION: Final[int] = 1
LEDGER_VERSION: Final[str] = "finding-task-source@1"
# VFS-G101 objective evidence identity for the finding repair taskboard.
FINDING_TASKBOARD_EVIDENCE: Final[str] = "vfs/finding-taskboard@1"
FINDING_TASKBOARD_G101_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    FINDING_TASKBOARD_EVIDENCE,
)
# Parent VFS-G100 ledger evidence (admitted findings only feed this board).
# String literals are intentional so discovery scanners observe both
# taskboard and ledger coverage on this surface.
PARENT_FINDING_LEDGER_EVIDENCE: Final[str] = "vfs/finding-ledger@1"
PARENT_VULNERABILITY_EVIDENCE_POLICY: Final[str] = (
    "vfs/vulnerability-evidence-policy@1"
)
PARENT_FINDING_LEDGER_G100_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    "vfs/finding-ledger@1",
    "vfs/vulnerability-evidence-policy@1",
)
assert PARENT_FINDING_LEDGER_EVIDENCE == FINDING_LEDGER_EVIDENCE
assert PARENT_VULNERABILITY_EVIDENCE_POLICY == VULNERABILITY_EVIDENCE_POLICY
assert PARENT_FINDING_LEDGER_G100_EVIDENCE_TERMS == FINDING_LEDGER_G100_EVIDENCE_TERMS
DEFAULT_BOARD_NAMESPACE: Final[str] = "ipfs-kit-vfs-symbolic-assurance-v1"
DEFAULT_GOAL_ID: Final[str] = "VFS-G101"
# Objective-heap ancestry for the second repair taskboard (VFS-G000 → G100 → G101).
DEFAULT_PARENT_GOAL_ID: Final[str] = "VFS-G100"
DEFAULT_ROOT_GOAL_ID: Final[str] = "VFS-G000"
DEFAULT_GOAL_LINEAGE: Final[tuple[str, ...]] = (
    DEFAULT_ROOT_GOAL_ID,
    DEFAULT_PARENT_GOAL_ID,
    DEFAULT_GOAL_ID,
)
DEFAULT_RESOURCE_CLASS: Final[str] = "cpu-small"
DEFAULT_TOKEN_CLASS: Final[str] = "small"
DEFAULT_CONTEXT_CEILING_BYTES: Final[int] = 12_288
DEFAULT_CONTEXT_CEILING_TOKENS: Final[int] = 3_072
DEFAULT_TASK_PREFIX: Final[str] = "VFS-R"

MAX_TEXT_BYTES: Final[int] = 8_192
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_OUTPUT_PATHS: Final[int] = 8
MAX_SYMBOLS: Final[int] = 24
MAX_EFFECTS: Final[int] = 12
MAX_BOARD_TASKS: Final[int] = 4_096
MAX_MARKDOWN_BYTES: Final[int] = 1_000_000
MAX_JSON_BYTES: Final[int] = 2_000_000
MAX_DUCKDB_ROWS: Final[int] = 8_192
MAX_PROVENANCE_CIDS: Final[int] = 64

FINDING_TASK_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/finding-task/repair-record@1"
)
REVIEW_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/finding-task/review-record@1"
)
BOARD_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/finding-task/board-snapshot@1"
)
MATERIALIZATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/finding-task/materialization-receipt@1"
)
POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/finding-task/policy@1"
)
PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/finding-task/projection@1"
)

# Diagnostic projections never authorize repair or completion.
PROJECTION_AUTHORIZES_REPAIR: Final[bool] = False
PROJECTION_IS_COMPLETION_EVIDENCE: Final[bool] = False

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FindingTaskSourceError(ContractValidationError):
    """Base error for malformed or unsafe finding-task operations."""


class FindingTaskBoundsError(FindingTaskSourceError):
    """A task, board, or projection exceeded an explicit bound."""


class FindingTaskIntegrityError(FindingTaskSourceError):
    """Identity collision, forged authority, or corrupt durable state."""


class FindingTaskAuthorityError(FindingTaskSourceError):
    """A projection or input attempted to claim repair/completion authority."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskDisposition(str, Enum):
    """Disposition of one finding after materialization."""

    EXECUTABLE = "executable"
    REVIEW = "review"
    NO_OP = "no_op"
    SUPERSEDED = "superseded"
    SKIPPED = "skipped"


class ReviewReason(str, Enum):
    """Why a finding produced a non-executable review record."""

    AMBIGUOUS = "ambiguous"
    BROAD = "broad"
    OUT_OF_ROOT = "out_of_root"
    NOT_ADMITTED = "not_admitted"
    STALE = "stale"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    SUSPECTED = "suspected"
    MISSING_SCOPE = "missing_scope"
    MISSING_ROOT_CAUSE = "missing_root_cause"
    CONFLICT = "conflict"
    REJECTED = "rejected"


class MaterializationOutcome(str, Enum):
    """Outcome of one materialization pass."""

    CREATED = "created"
    UPDATED = "updated"
    NO_OP = "no_op"
    SUPERSEDED = "superseded"
    REVIEW_ONLY = "review_only"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    if "\x00" in text:
        raise FindingTaskSourceError(f"{field_name} must not contain NUL")
    text = text.strip()
    if required and not text:
        raise FindingTaskSourceError(f"{field_name} is required")
    encoded = text.encode("utf-8")
    if len(encoded) > maximum:
        raise FindingTaskBoundsError(
            f"{field_name} exceeds {maximum} bytes"
        )
    return text


def _optional_text(value: Any, *, field_name: str) -> str:
    return _text(value or "", field_name=field_name, required=False)


def _strings(
    value: Any,
    *,
    field_name: str,
    maximum: int = MAX_COLLECTION_ITEMS,
    unique: bool = True,
    sort: bool = True,
    paths: bool = False,
) -> tuple[str, ...]:
    if value in (None, ""):
        items: list[str] = []
    elif isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        items = [str(item) for item in value if item not in (None, "")]
    else:
        raise FindingTaskSourceError(f"{field_name} must be a sequence")
    cleaned: list[str] = []
    for item in items:
        if paths:
            path = normalize_identity_path(item)
            if path:
                cleaned.append(path)
        else:
            text = _text(item, field_name=field_name, required=False)
            if text:
                cleaned.append(text)
    if unique:
        cleaned = list(dict.fromkeys(cleaned))
    if sort:
        cleaned = sorted(cleaned)
    if len(cleaned) > maximum:
        raise FindingTaskBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    return tuple(cleaned)


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FindingTaskSourceError(f"{field_name} must be an int")
    if value < minimum:
        raise FindingTaskSourceError(
            f"{field_name} must be >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise FindingTaskBoundsError(
            f"{field_name} must be <= {maximum}"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise FindingTaskSourceError(
            f"{field_name} is not a valid {enum_type.__name__}"
        ) from exc


def _normalize_path(value: Any) -> str:
    return normalize_identity_path(value)


def _is_escape_path(path: str) -> bool:
    if not path:
        return True
    lowered = path.replace("\\", "/")
    if lowered.startswith("/") or lowered.startswith("~"):
        return True
    parts = [part for part in lowered.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return True
    if any(marker in lowered for marker in ("\x00", "://")):
        return True
    return False


def _path_within_roots(path: str, roots: Sequence[str]) -> bool:
    """Return whether ``path`` is under one of the relative write roots.

    Empty ``roots`` means "any non-escaping relative path is in-root".
    """

    normalized = _normalize_path(path)
    if _is_escape_path(normalized):
        return False
    if not roots:
        return True
    for root in roots:
        root_norm = _normalize_path(root)
        if not root_norm:
            continue
        if normalized == root_norm or normalized.startswith(root_norm + "/"):
            return True
    return False


def _severity_risk_millionths(severity: Any) -> int:
    ranking = {
        "info": 50_000,
        "low": 150_000,
        "medium": 400_000,
        "high": 700_000,
        "critical": 950_000,
    }
    key = str(getattr(severity, "value", severity) or "medium").casefold()
    return ranking.get(key, 400_000)


def _finding_as_record(value: Any) -> ContractFindingRecord:
    if isinstance(value, ContractFindingRecord):
        return value
    if isinstance(value, Mapping):
        if "schema" in value and "claim_level" in value:
            return ContractFindingRecord.from_dict(value)
        return build_contract_finding(**dict(value))
    raise FindingTaskSourceError(
        "finding must be a ContractFindingRecord or mapping"
    )


def _provenance_cids(record: ContractFindingRecord) -> tuple[str, ...]:
    evidence = record.evidence
    cids: list[str] = []
    for group in (
        evidence.counterexample_cids,
        evidence.proof_cids,
        evidence.runtime_cids,
        evidence.zk_cids,
        evidence.artifact_cids,
    ):
        cids.extend(group)
    if record.expected_contract_cid:
        cids.append(record.expected_contract_cid)
    if record.observed_contract_cid:
        cids.append(record.observed_contract_cid)
    if record.finding_cid:
        cids.append(record.finding_cid)
    return _strings(
        cids,
        field_name="provenance_cids",
        maximum=MAX_PROVENANCE_CIDS,
        unique=True,
        sort=True,
    )


def _output_paths_from_finding(record: ContractFindingRecord) -> tuple[str, ...]:
    """Collect candidate output paths (unbounded here; policy enforces breadth)."""

    paths: list[str] = []
    for item in record.remediation_scope:
        path = _normalize_path(item)
        if path and (
            "/" in path
            or path.endswith((".py", ".ts", ".js", ".tsx", ".json", ".md"))
        ):
            paths.append(path)
    for step in record.call_slice.steps:
        path = _normalize_path(step.path)
        if path:
            paths.append(path)
    # Bound only by the global collection cap; policy max_output_paths decides
    # whether the finding is broad (review) versus executable.
    return _strings(
        paths,
        field_name="output_paths",
        paths=True,
        maximum=MAX_COLLECTION_ITEMS,
    )


def _symbols_from_finding(record: ContractFindingRecord) -> tuple[str, ...]:
    """Collect candidate symbols (unbounded here; policy enforces breadth)."""

    if record.symbols:
        symbols = list(record.symbols)
    else:
        symbols = [
            step.symbol for step in record.call_slice.steps if step.symbol
        ]
    return _strings(
        symbols,
        field_name="symbols",
        maximum=MAX_COLLECTION_ITEMS,
    )


def _effects_from_finding(record: ContractFindingRecord) -> tuple[str, ...]:
    effects = [
        f"restore:{record.root_cause_family or 'contract'}",
        f"align_observed:{record.observed_contract_cid or 'unknown'}",
        f"satisfy_expected:{record.expected_contract_cid or 'unknown'}",
    ]
    if record.interfaces:
        effects.append(f"preserve_interfaces:{','.join(record.interfaces)}")
    return _strings(effects, field_name="effects", maximum=MAX_EFFECTS)


def _estimate_task_tokens(
    outputs: Sequence[str],
    symbols: Sequence[str],
    policy: FindingTaskSourcePolicy,
) -> int:
    """Heuristic token estimate scaled by exact scope, capped by the ceiling."""

    base = 256 + 128 * len(outputs) + 64 * len(symbols)
    return min(policy.context_ceiling_tokens, max(256, base))


def _default_validation_plan(
    outputs: Sequence[str],
    symbols: Sequence[str],
) -> tuple[str, ...]:
    test_paths = [
        path
        for path in outputs
        if "/test" in f"/{path}" or path.startswith("test/") or path.endswith("_test.py")
    ]
    if test_paths:
        joined = " ".join(sorted(test_paths)[:4])
        return (f"python -m pytest {joined} -q",)
    # Fall back to a deterministic module-level contract re-check command.
    if symbols:
        return (
            "python -m pytest test/api/test_agent_supervisor_contract_findings.py -q",
        )
    return (
        "python -m pytest test/api/test_agent_supervisor_finding_task_source.py -q",
    )


def _default_proof_plan(record: ContractFindingRecord) -> tuple[str, ...]:
    """Stable proof steps shared across findings with the same repair contract.

    Evidence CIDs live in ``provenance_cids`` / ``finding_cids`` rather than
    the plan step strings so related tiny tasks can share validation and merge
    fate without proof-plan drift blocking coalescing.
    """

    plan: list[str] = [
        "recompute_contract_finding_from_evidence",
        "verify_finding_cid_binds_expected_and_observed",
    ]
    if record.evidence.proof_cids:
        plan.append("replay_bound_proof_receipts")
    if record.evidence.counterexample_cids:
        plan.append("confirm_bound_counterexamples")
    if record.evidence.runtime_cids:
        plan.append("confirm_bound_runtime_witnesses")
    if record.evidence.zk_cids:
        plan.append("confirm_bound_zk_traces")
    return _strings(plan, field_name="proof_plan", maximum=16)


def _conflict_domain(
    *,
    root_cause_family: str,
    merge_fate: str,
    outputs: Sequence[str],
    symbols: Sequence[str],
) -> str:
    material = {
        "root_cause_family": normalize_identity_text(root_cause_family),
        "merge_fate": normalize_identity_text(merge_fate),
        "outputs": sorted(_normalize_path(p) for p in outputs if p),
        "symbols": sorted(normalize_identity_text(s) for s in symbols if s),
    }
    return content_identity(material)


def _semantic_task_key(
    *,
    root_cause_family: str,
    merge_fate: str,
    outputs: Sequence[str],
    symbols: Sequence[str],
    validation_plan: Sequence[str],
    goal_id: str,
) -> str:
    material = {
        "schema": "finding-task-semantic@1",
        "root_cause_family": normalize_identity_text(root_cause_family),
        "merge_fate": normalize_identity_text(merge_fate),
        "outputs": sorted(_normalize_path(p) for p in outputs if p),
        "symbols": sorted(normalize_identity_text(s) for s in symbols if s),
        "validation_plan": sorted(
            normalize_identity_text(v) for v in validation_plan if v
        ),
        "goal_id": normalize_identity_text(goal_id),
    }
    return content_identity(material)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _normalize_goal_lineage(
    lineage: Sequence[str] | None,
    *,
    goal_id: str,
    parent_goal_id: str = "",
) -> tuple[str, ...]:
    """Return a closed goal-lineage path ending at ``goal_id``.

    When *lineage* is empty, synthesize ``(parent_goal_id, goal_id)`` or
    ``(goal_id,)`` so every executable task still binds heap ancestry.
    """

    if lineage:
        normalized = _strings(lineage, field_name="goal_lineage")
    else:
        parts: list[str] = []
        if parent_goal_id:
            parts.append(_text(parent_goal_id, field_name="parent_goal_id"))
        parts.append(_text(goal_id, field_name="goal_id"))
        normalized = tuple(parts)
    if not normalized:
        raise FindingTaskSourceError("goal_lineage must not be empty")
    goal = _text(goal_id, field_name="goal_id")
    if normalize_identity_text(normalized[-1]) != normalize_identity_text(goal):
        raise FindingTaskSourceError(
            "goal_lineage must end with the task goal_id"
        )
    return normalized


@dataclass(frozen=True)
class FindingTaskSourcePolicy:
    """Bounds and roots that govern admission into executable repair tasks."""

    SCHEMA: ClassVar[str] = POLICY_SCHEMA

    board_namespace: str = DEFAULT_BOARD_NAMESPACE
    goal_id: str = DEFAULT_GOAL_ID
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID
    goal_lineage: tuple[str, ...] = DEFAULT_GOAL_LINEAGE
    resource_class: str = DEFAULT_RESOURCE_CLASS
    token_class: str = DEFAULT_TOKEN_CLASS
    context_ceiling_bytes: int = DEFAULT_CONTEXT_CEILING_BYTES
    context_ceiling_tokens: int = DEFAULT_CONTEXT_CEILING_TOKENS
    max_output_paths: int = MAX_OUTPUT_PATHS
    max_symbols: int = MAX_SYMBOLS
    max_effects: int = MAX_EFFECTS
    max_board_tasks: int = MAX_BOARD_TASKS
    task_prefix: str = DEFAULT_TASK_PREFIX
    write_roots: tuple[str, ...] = ()
    coalesce_tiny: bool = True
    default_track: str = "finding-generation"
    quality: TaskQualityPolicy = field(default_factory=TaskQualityPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "board_namespace",
            _text(self.board_namespace, field_name="board_namespace"),
        )
        object.__setattr__(
            self, "goal_id", _text(self.goal_id, field_name="goal_id")
        )
        object.__setattr__(
            self,
            "parent_goal_id",
            _optional_text(self.parent_goal_id, field_name="parent_goal_id"),
        )
        object.__setattr__(
            self,
            "goal_lineage",
            _normalize_goal_lineage(
                self.goal_lineage,
                goal_id=self.goal_id,
                parent_goal_id=self.parent_goal_id,
            ),
        )
        # Keep parent_goal_id aligned with lineage when present.
        if len(self.goal_lineage) >= 2 and not self.parent_goal_id:
            object.__setattr__(
                self, "parent_goal_id", self.goal_lineage[-2]
            )
        resource = _text(self.resource_class, field_name="resource_class").casefold()
        if resource not in RESOURCE_CLASSES:
            raise FindingTaskSourceError(
                f"unsupported resource_class {resource!r}"
            )
        object.__setattr__(self, "resource_class", resource)
        token = _text(self.token_class, field_name="token_class").casefold()
        if token not in TOKEN_CLASSES:
            raise FindingTaskSourceError(
                f"unsupported token_class {token!r}"
            )
        object.__setattr__(self, "token_class", token)
        object.__setattr__(
            self,
            "context_ceiling_bytes",
            _integer(
                self.context_ceiling_bytes,
                field_name="context_ceiling_bytes",
                minimum=256,
                maximum=1_000_000,
            ),
        )
        object.__setattr__(
            self,
            "context_ceiling_tokens",
            _integer(
                self.context_ceiling_tokens,
                field_name="context_ceiling_tokens",
                minimum=64,
                maximum=TOKEN_CLASS_LIMITS["xlarge"],
            ),
        )
        for name, default_max in (
            ("max_output_paths", MAX_OUTPUT_PATHS),
            ("max_symbols", MAX_SYMBOLS),
            ("max_effects", MAX_EFFECTS),
            ("max_board_tasks", MAX_BOARD_TASKS),
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    field_name=name,
                    minimum=1,
                    maximum=default_max * 4 if name != "max_board_tasks" else MAX_BOARD_TASKS,
                ),
            )
        object.__setattr__(
            self,
            "task_prefix",
            _text(self.task_prefix, field_name="task_prefix"),
        )
        object.__setattr__(
            self,
            "write_roots",
            _strings(self.write_roots, field_name="write_roots", paths=True),
        )
        object.__setattr__(
            self,
            "default_track",
            _text(self.default_track, field_name="default_track"),
        )
        if not isinstance(self.quality, TaskQualityPolicy):
            raise FindingTaskSourceError("quality must be a TaskQualityPolicy")

    @property
    def policy_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "board_namespace": self.board_namespace,
            "goal_id": self.goal_id,
            "parent_goal_id": self.parent_goal_id,
            "goal_lineage": list(self.goal_lineage),
            "resource_class": self.resource_class,
            "token_class": self.token_class,
            "context_ceiling_bytes": self.context_ceiling_bytes,
            "context_ceiling_tokens": self.context_ceiling_tokens,
            "max_output_paths": self.max_output_paths,
            "max_symbols": self.max_symbols,
            "max_effects": self.max_effects,
            "max_board_tasks": self.max_board_tasks,
            "task_prefix": self.task_prefix,
            "write_roots": self.write_roots,
            "coalesce_tiny": bool(self.coalesce_tiny),
            "default_track": self.default_track,
            "quality_policy_id": self.quality.policy_id,
            "evidence": FINDING_TASKBOARD_EVIDENCE,
        }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairTaskRecord:
    """One goal-backed executable repair task derived from admitted findings."""

    SCHEMA: ClassVar[str] = FINDING_TASK_RECORD_SCHEMA

    task_id: str
    title: str
    goal_id: str
    root_cause_family: str
    outputs: tuple[str, ...]
    symbols: tuple[str, ...]
    effects: tuple[str, ...]
    dependencies: tuple[str, ...]
    conflict_domain: str
    validation_plan: tuple[str, ...]
    proof_plan: tuple[str, ...]
    finding_cids: tuple[str, ...]
    provenance_cids: tuple[str, ...]
    risk_millionths: int
    resource_class: str
    token_class: str
    context_ceiling_bytes: int
    context_ceiling_tokens: int
    merge_fate: str
    semantic_key: str
    track: str = "finding-generation"
    interfaces: tuple[str, ...] = ()
    acceptance: tuple[str, ...] = ()
    board_namespace: str = DEFAULT_BOARD_NAMESPACE
    tree_id: str = ""
    policy_revision: str = ""
    parent_goal_id: str = DEFAULT_PARENT_GOAL_ID
    goal_lineage: tuple[str, ...] = DEFAULT_GOAL_LINEAGE
    supersedes_task_ids: tuple[str, ...] = ()
    executable: bool = True
    disposition: TaskDisposition = TaskDisposition.EXECUTABLE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id", _text(self.task_id, field_name="task_id")
        )
        object.__setattr__(
            self, "title", _text(self.title, field_name="title")
        )
        object.__setattr__(
            self, "goal_id", _text(self.goal_id, field_name="goal_id")
        )
        object.__setattr__(
            self,
            "parent_goal_id",
            _optional_text(self.parent_goal_id, field_name="parent_goal_id"),
        )
        object.__setattr__(
            self,
            "goal_lineage",
            _normalize_goal_lineage(
                self.goal_lineage,
                goal_id=self.goal_id,
                parent_goal_id=self.parent_goal_id,
            ),
        )
        if len(self.goal_lineage) >= 2 and not self.parent_goal_id:
            object.__setattr__(
                self, "parent_goal_id", self.goal_lineage[-2]
            )
        object.__setattr__(
            self,
            "root_cause_family",
            _text(self.root_cause_family, field_name="root_cause_family"),
        )
        object.__setattr__(
            self,
            "outputs",
            _strings(self.outputs, field_name="outputs", paths=True),
        )
        object.__setattr__(
            self, "symbols", _strings(self.symbols, field_name="symbols")
        )
        object.__setattr__(
            self, "effects", _strings(self.effects, field_name="effects")
        )
        object.__setattr__(
            self,
            "dependencies",
            _strings(self.dependencies, field_name="dependencies"),
        )
        object.__setattr__(
            self,
            "conflict_domain",
            _text(self.conflict_domain, field_name="conflict_domain"),
        )
        object.__setattr__(
            self,
            "validation_plan",
            _strings(self.validation_plan, field_name="validation_plan"),
        )
        object.__setattr__(
            self,
            "proof_plan",
            _strings(self.proof_plan, field_name="proof_plan"),
        )
        object.__setattr__(
            self,
            "finding_cids",
            _strings(self.finding_cids, field_name="finding_cids"),
        )
        object.__setattr__(
            self,
            "provenance_cids",
            _strings(
                self.provenance_cids,
                field_name="provenance_cids",
                maximum=MAX_PROVENANCE_CIDS,
            ),
        )
        object.__setattr__(
            self,
            "risk_millionths",
            _integer(
                self.risk_millionths,
                field_name="risk_millionths",
                minimum=0,
                maximum=1_000_000,
            ),
        )
        resource = _text(
            self.resource_class, field_name="resource_class"
        ).casefold()
        if resource not in RESOURCE_CLASSES:
            raise FindingTaskSourceError(
                f"unsupported resource_class {resource!r}"
            )
        object.__setattr__(self, "resource_class", resource)
        token = _text(self.token_class, field_name="token_class").casefold()
        if token not in TOKEN_CLASSES:
            raise FindingTaskSourceError(
                f"unsupported token_class {token!r}"
            )
        object.__setattr__(self, "token_class", token)
        object.__setattr__(
            self,
            "context_ceiling_bytes",
            _integer(
                self.context_ceiling_bytes,
                field_name="context_ceiling_bytes",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "context_ceiling_tokens",
            _integer(
                self.context_ceiling_tokens,
                field_name="context_ceiling_tokens",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "merge_fate",
            _text(self.merge_fate, field_name="merge_fate"),
        )
        object.__setattr__(
            self,
            "semantic_key",
            _text(self.semantic_key, field_name="semantic_key"),
        )
        object.__setattr__(
            self, "track", _optional_text(self.track, field_name="track") or "finding-generation"
        )
        object.__setattr__(
            self,
            "interfaces",
            _strings(self.interfaces, field_name="interfaces"),
        )
        object.__setattr__(
            self,
            "acceptance",
            _strings(self.acceptance, field_name="acceptance"),
        )
        object.__setattr__(
            self,
            "board_namespace",
            _optional_text(self.board_namespace, field_name="board_namespace")
            or DEFAULT_BOARD_NAMESPACE,
        )
        object.__setattr__(
            self,
            "tree_id",
            _optional_text(self.tree_id, field_name="tree_id"),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _optional_text(self.policy_revision, field_name="policy_revision"),
        )
        object.__setattr__(
            self,
            "supersedes_task_ids",
            _strings(
                self.supersedes_task_ids, field_name="supersedes_task_ids"
            ),
        )
        object.__setattr__(self, "executable", bool(self.executable))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, TaskDisposition, field_name="disposition"),
        )
        if not self.executable:
            raise FindingTaskSourceError(
                "RepairTaskRecord must be executable; use ReviewRecord"
            )
        if not self.outputs:
            raise FindingTaskSourceError(
                "executable repair tasks require exact output paths"
            )
        if not self.symbols:
            raise FindingTaskSourceError(
                "executable repair tasks require exact symbols"
            )
        if not self.finding_cids:
            raise FindingTaskSourceError(
                "executable repair tasks require finding CIDs"
            )
        if not self.validation_plan:
            raise FindingTaskSourceError(
                "executable repair tasks require a validation plan"
            )
        if not self.proof_plan:
            raise FindingTaskSourceError(
                "executable repair tasks require a proof plan"
            )

    @property
    def task_cid(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def identity(self) -> TaskIdentity:
        return canonical_task_identity(
            {
                "task_id": self.task_id,
                "title": self.title,
                "outputs": self.outputs,
                "acceptance": self.acceptance,
                "goal_id": self.goal_id,
                "dedupe_key": self.semantic_key,
                "metadata": {
                    "board_namespace": self.board_namespace,
                    "canonical_task_key": f"finding-task/v1/{self.semantic_key}",
                    "canonical_task_cid": self.task_cid,
                },
            },
            board_namespace=self.board_namespace,
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_id": self.task_id,
            "title": self.title,
            "goal_id": self.goal_id,
            "parent_goal_id": self.parent_goal_id,
            "goal_lineage": list(self.goal_lineage),
            "root_cause_family": self.root_cause_family,
            "outputs": self.outputs,
            "symbols": self.symbols,
            "effects": self.effects,
            "dependencies": self.dependencies,
            "conflict_domain": self.conflict_domain,
            "validation_plan": self.validation_plan,
            "proof_plan": self.proof_plan,
            "finding_cids": self.finding_cids,
            "provenance_cids": self.provenance_cids,
            "risk_millionths": self.risk_millionths,
            "resource_class": self.resource_class,
            "token_class": self.token_class,
            "context_ceiling_bytes": self.context_ceiling_bytes,
            "context_ceiling_tokens": self.context_ceiling_tokens,
            "merge_fate": self.merge_fate,
            "semantic_key": self.semantic_key,
            "track": self.track,
            "interfaces": self.interfaces,
            "acceptance": self.acceptance,
            "board_namespace": self.board_namespace,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "supersedes_task_ids": self.supersedes_task_ids,
            "executable": True,
            "disposition": TaskDisposition.EXECUTABLE.value,
            "evidence": FINDING_TASKBOARD_EVIDENCE,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["task_cid"] = self.task_cid
        payload["canonical_task_key"] = self.identity.canonical_task_key
        payload["canonical_task_cid"] = self.identity.canonical_task_cid
        return payload

    def to_task_candidate(self) -> TaskCandidate:
        estimated_tokens = min(
            self.context_ceiling_tokens,
            max(256, 256 + 128 * len(self.outputs) + 64 * len(self.symbols)),
        )
        return TaskCandidate(
            title=self.title,
            goal_id=self.goal_id,
            acceptance=self.acceptance,
            effects=self.effects,
            evidence_subset=self.finding_cids,
            outputs=self.outputs,
            validation_commands=self.validation_plan,
            context_paths=self.outputs,
            predicted_paths=self.outputs,
            predicted_symbols=self.symbols,
            predicted_interfaces=self.interfaces,
            proof_obligations=self.proof_plan,
            proof_commands=self.proof_plan,
            dependencies=self.dependencies,
            conflicts=(self.conflict_domain,),
            resource_class=self.resource_class,
            token_class=self.token_class,
            merge_fate=self.merge_fate,
            track=self.track,
            estimated_context_tokens=estimated_tokens,
            estimated_tokens=estimated_tokens,
            estimated_merge_risk_millionths=self.risk_millionths,
            source_id=self.task_id,
            metadata={
                "finding_cids": self.finding_cids,
                "provenance_cids": self.provenance_cids,
                "semantic_key": self.semantic_key,
                "root_cause_family": self.root_cause_family,
                "parent_goal_id": self.parent_goal_id,
                "goal_lineage": list(self.goal_lineage),
                "evidence": FINDING_TASKBOARD_EVIDENCE,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairTaskRecord":
        if not isinstance(payload, Mapping):
            raise FindingTaskSourceError("repair task payload must be a mapping")
        goal_id = str(payload.get("goal_id") or DEFAULT_GOAL_ID)
        parent_goal_id = str(
            payload.get("parent_goal_id") or DEFAULT_PARENT_GOAL_ID
        )
        raw_lineage = payload.get("goal_lineage")
        if raw_lineage is None:
            # Back-compat: older board snapshots only carried goal_id.
            goal_lineage = _normalize_goal_lineage(
                (),
                goal_id=goal_id,
                parent_goal_id=parent_goal_id,
            )
        else:
            goal_lineage = tuple(raw_lineage or ())
        result = cls(
            task_id=payload.get("task_id", ""),
            title=payload.get("title", ""),
            goal_id=goal_id,
            root_cause_family=payload.get("root_cause_family", ""),
            outputs=tuple(payload.get("outputs") or ()),
            symbols=tuple(payload.get("symbols") or ()),
            effects=tuple(payload.get("effects") or ()),
            dependencies=tuple(payload.get("dependencies") or ()),
            conflict_domain=payload.get("conflict_domain", ""),
            validation_plan=tuple(payload.get("validation_plan") or ()),
            proof_plan=tuple(payload.get("proof_plan") or ()),
            finding_cids=tuple(payload.get("finding_cids") or ()),
            provenance_cids=tuple(payload.get("provenance_cids") or ()),
            risk_millionths=int(payload.get("risk_millionths") or 0),
            resource_class=payload.get("resource_class", DEFAULT_RESOURCE_CLASS),
            token_class=payload.get("token_class", DEFAULT_TOKEN_CLASS),
            context_ceiling_bytes=int(
                payload.get("context_ceiling_bytes")
                or DEFAULT_CONTEXT_CEILING_BYTES
            ),
            context_ceiling_tokens=int(
                payload.get("context_ceiling_tokens")
                or DEFAULT_CONTEXT_CEILING_TOKENS
            ),
            merge_fate=payload.get("merge_fate", ""),
            semantic_key=payload.get("semantic_key", ""),
            track=payload.get("track", "finding-generation"),
            interfaces=tuple(payload.get("interfaces") or ()),
            acceptance=tuple(payload.get("acceptance") or ()),
            board_namespace=payload.get(
                "board_namespace", DEFAULT_BOARD_NAMESPACE
            ),
            tree_id=payload.get("tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            parent_goal_id=parent_goal_id,
            goal_lineage=goal_lineage,
            supersedes_task_ids=tuple(
                payload.get("supersedes_task_ids") or ()
            ),
            executable=bool(payload.get("executable", True)),
            disposition=payload.get(
                "disposition", TaskDisposition.EXECUTABLE
            ),
        )
        if "task_cid" in payload and payload["task_cid"] != result.task_cid:
            # Older board snapshots predate goal_lineage / evidence bindings.
            # Only hard-fail when the payload claimed those fields (or neither
            # migration marker is present) so forged modern CIDs stay rejected.
            claimed_lineage = raw_lineage is not None
            claimed_evidence = "evidence" in payload
            if claimed_lineage or claimed_evidence:
                raise FindingTaskIntegrityError(
                    "forged task_cid does not match derived identity"
                )
        return result


@dataclass(frozen=True)
class ReviewRecord:
    """Non-executable review record for non-actionable findings."""

    SCHEMA: ClassVar[str] = REVIEW_RECORD_SCHEMA

    review_id: str
    finding_cids: tuple[str, ...]
    reasons: tuple[str, ...]
    summary: str
    root_cause_family: str = ""
    outputs: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    goal_id: str = DEFAULT_GOAL_ID
    executable: bool = False
    disposition: TaskDisposition = TaskDisposition.REVIEW
    tree_id: str = ""
    provenance_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "review_id", _text(self.review_id, field_name="review_id")
        )
        object.__setattr__(
            self,
            "finding_cids",
            _strings(self.finding_cids, field_name="finding_cids"),
        )
        object.__setattr__(
            self, "reasons", _strings(self.reasons, field_name="reasons")
        )
        object.__setattr__(
            self, "summary", _text(self.summary, field_name="summary")
        )
        object.__setattr__(
            self,
            "root_cause_family",
            _optional_text(
                self.root_cause_family, field_name="root_cause_family"
            ),
        )
        object.__setattr__(
            self,
            "outputs",
            _strings(self.outputs, field_name="outputs", paths=True),
        )
        object.__setattr__(
            self, "symbols", _strings(self.symbols, field_name="symbols")
        )
        object.__setattr__(
            self,
            "goal_id",
            _optional_text(self.goal_id, field_name="goal_id") or DEFAULT_GOAL_ID,
        )
        object.__setattr__(self, "executable", False)
        object.__setattr__(
            self,
            "disposition",
            TaskDisposition.REVIEW,
        )
        object.__setattr__(
            self, "tree_id", _optional_text(self.tree_id, field_name="tree_id")
        )
        object.__setattr__(
            self,
            "provenance_cids",
            _strings(
                self.provenance_cids,
                field_name="provenance_cids",
                maximum=MAX_PROVENANCE_CIDS,
            ),
        )
        if not self.finding_cids:
            raise FindingTaskSourceError("review records require finding CIDs")
        if not self.reasons:
            raise FindingTaskSourceError("review records require reasons")

    @property
    def review_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "review_id": self.review_id,
            "finding_cids": self.finding_cids,
            "reasons": self.reasons,
            "summary": self.summary,
            "root_cause_family": self.root_cause_family,
            "outputs": self.outputs,
            "symbols": self.symbols,
            "goal_id": self.goal_id,
            "executable": False,
            "disposition": TaskDisposition.REVIEW.value,
            "tree_id": self.tree_id,
            "provenance_cids": self.provenance_cids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewRecord":
        if bool(payload.get("executable", False)):
            raise FindingTaskAuthorityError(
                "review records cannot claim executable authority"
            )
        return cls(
            review_id=payload.get("review_id", ""),
            finding_cids=tuple(payload.get("finding_cids") or ()),
            reasons=tuple(payload.get("reasons") or ()),
            summary=payload.get("summary", ""),
            root_cause_family=payload.get("root_cause_family", ""),
            outputs=tuple(payload.get("outputs") or ()),
            symbols=tuple(payload.get("symbols") or ()),
            goal_id=payload.get("goal_id", DEFAULT_GOAL_ID),
            tree_id=payload.get("tree_id", ""),
            provenance_cids=tuple(payload.get("provenance_cids") or ()),
        )


@dataclass(frozen=True)
class BoardSnapshot:
    """Immutable snapshot of the current finding-task board.

    Carries the closed ``vfs/finding-taskboard@1`` evidence term.  The board
    itself never authorizes repair or completion; only executable task records
    may drive work after independent admission.
    """

    SCHEMA: ClassVar[str] = BOARD_SNAPSHOT_SCHEMA

    tasks: tuple[RepairTaskRecord, ...] = ()
    reviews: tuple[ReviewRecord, ...] = ()
    board_namespace: str = DEFAULT_BOARD_NAMESPACE
    policy_id: str = ""
    tree_id: str = ""
    revision: int = 0
    goal_id: str = DEFAULT_GOAL_ID
    goal_lineage: tuple[str, ...] = DEFAULT_GOAL_LINEAGE
    evidence: str = FINDING_TASKBOARD_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "reviews", tuple(self.reviews))
        object.__setattr__(
            self,
            "board_namespace",
            _optional_text(self.board_namespace, field_name="board_namespace")
            or DEFAULT_BOARD_NAMESPACE,
        )
        object.__setattr__(
            self,
            "policy_id",
            _optional_text(self.policy_id, field_name="policy_id"),
        )
        object.__setattr__(
            self, "tree_id", _optional_text(self.tree_id, field_name="tree_id")
        )
        object.__setattr__(
            self,
            "revision",
            _integer(self.revision, field_name="revision", minimum=0),
        )
        object.__setattr__(
            self,
            "goal_id",
            _optional_text(self.goal_id, field_name="goal_id")
            or DEFAULT_GOAL_ID,
        )
        object.__setattr__(
            self,
            "goal_lineage",
            _normalize_goal_lineage(
                self.goal_lineage,
                goal_id=self.goal_id,
            ),
        )
        evidence = _text(self.evidence, field_name="evidence")
        if evidence != FINDING_TASKBOARD_EVIDENCE:
            raise FindingTaskIntegrityError(
                f"board evidence must be {FINDING_TASKBOARD_EVIDENCE!r}"
            )
        object.__setattr__(self, "evidence", evidence)
        if len(self.tasks) + len(self.reviews) > MAX_BOARD_TASKS:
            raise FindingTaskBoundsError(
                f"board exceeds {MAX_BOARD_TASKS} entries"
            )

    @property
    def board_cid(self) -> str:
        return content_identity(self.to_dict())

    @property
    def executable_tasks(self) -> tuple[RepairTaskRecord, ...]:
        return tuple(task for task in self.tasks if task.executable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "board_namespace": self.board_namespace,
            "policy_id": self.policy_id,
            "tree_id": self.tree_id,
            "revision": self.revision,
            "goal_id": self.goal_id,
            "goal_lineage": list(self.goal_lineage),
            "evidence": self.evidence,
            "evidence_terms": list(FINDING_TASKBOARD_G101_EVIDENCE_TERMS),
            "tasks": [task.to_dict() for task in self.tasks],
            "reviews": [review.to_dict() for review in self.reviews],
            "executable_count": len(self.executable_tasks),
            "review_count": len(self.reviews),
            "authorizes_repair": PROJECTION_AUTHORIZES_REPAIR,
            "is_completion_evidence": PROJECTION_IS_COMPLETION_EVIDENCE,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoardSnapshot":
        if payload.get("authorizes_repair") is True:
            raise FindingTaskAuthorityError(
                "board snapshot cannot authorize repair"
            )
        if payload.get("is_completion_evidence") is True:
            raise FindingTaskAuthorityError(
                "board snapshot cannot be completion evidence"
            )
        evidence = payload.get("evidence", FINDING_TASKBOARD_EVIDENCE)
        if evidence and evidence != FINDING_TASKBOARD_EVIDENCE:
            raise FindingTaskIntegrityError(
                f"board evidence must be {FINDING_TASKBOARD_EVIDENCE!r}"
            )
        goal_id = str(payload.get("goal_id") or DEFAULT_GOAL_ID)
        raw_lineage = payload.get("goal_lineage")
        if raw_lineage is None:
            goal_lineage = DEFAULT_GOAL_LINEAGE if goal_id == DEFAULT_GOAL_ID else (
                goal_id,
            )
        else:
            goal_lineage = tuple(raw_lineage or ())
        return cls(
            tasks=tuple(
                RepairTaskRecord.from_dict(item)
                for item in (payload.get("tasks") or ())
            ),
            reviews=tuple(
                ReviewRecord.from_dict(item)
                for item in (payload.get("reviews") or ())
            ),
            board_namespace=payload.get(
                "board_namespace", DEFAULT_BOARD_NAMESPACE
            ),
            policy_id=payload.get("policy_id", ""),
            tree_id=payload.get("tree_id", ""),
            revision=int(payload.get("revision") or 0),
            goal_id=goal_id,
            goal_lineage=goal_lineage,
            evidence=str(evidence or FINDING_TASKBOARD_EVIDENCE),
        )


@dataclass(frozen=True)
class MaterializationReceipt:
    """Receipt returned by one materialization pass.

    Binds ``vfs/finding-taskboard@1`` so scanners can treat a materialization
    as covering the VFS-G101 evidence obligation without granting edit
    authority (``authorizes_repair`` remains false).
    """

    SCHEMA: ClassVar[str] = MATERIALIZATION_RECEIPT_SCHEMA

    outcome: MaterializationOutcome
    created_task_ids: tuple[str, ...] = ()
    superseded_task_ids: tuple[str, ...] = ()
    no_op_finding_cids: tuple[str, ...] = ()
    review_ids: tuple[str, ...] = ()
    board_cid: str = ""
    revision: int = 0
    reasons: tuple[str, ...] = ()
    evidence: str = FINDING_TASKBOARD_EVIDENCE
    goal_id: str = DEFAULT_GOAL_ID
    goal_lineage: tuple[str, ...] = DEFAULT_GOAL_LINEAGE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            _enum(self.outcome, MaterializationOutcome, field_name="outcome"),
        )
        object.__setattr__(
            self,
            "created_task_ids",
            _strings(self.created_task_ids, field_name="created_task_ids"),
        )
        object.__setattr__(
            self,
            "superseded_task_ids",
            _strings(
                self.superseded_task_ids, field_name="superseded_task_ids"
            ),
        )
        object.__setattr__(
            self,
            "no_op_finding_cids",
            _strings(
                self.no_op_finding_cids, field_name="no_op_finding_cids"
            ),
        )
        object.__setattr__(
            self, "review_ids", _strings(self.review_ids, field_name="review_ids")
        )
        object.__setattr__(
            self,
            "board_cid",
            _optional_text(self.board_cid, field_name="board_cid"),
        )
        object.__setattr__(
            self,
            "revision",
            _integer(self.revision, field_name="revision", minimum=0),
        )
        object.__setattr__(
            self, "reasons", _strings(self.reasons, field_name="reasons")
        )
        evidence = _text(self.evidence, field_name="evidence")
        if evidence != FINDING_TASKBOARD_EVIDENCE:
            raise FindingTaskIntegrityError(
                f"receipt evidence must be {FINDING_TASKBOARD_EVIDENCE!r}"
            )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "goal_id",
            _optional_text(self.goal_id, field_name="goal_id")
            or DEFAULT_GOAL_ID,
        )
        object.__setattr__(
            self,
            "goal_lineage",
            _normalize_goal_lineage(
                self.goal_lineage,
                goal_id=self.goal_id,
            ),
        )

    @property
    def receipt_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "outcome": self.outcome.value,
            "created_task_ids": self.created_task_ids,
            "superseded_task_ids": self.superseded_task_ids,
            "no_op_finding_cids": self.no_op_finding_cids,
            "review_ids": self.review_ids,
            "board_cid": self.board_cid,
            "revision": self.revision,
            "reasons": self.reasons,
            "evidence": self.evidence,
            "evidence_terms": list(FINDING_TASKBOARD_G101_EVIDENCE_TERMS),
            "goal_id": self.goal_id,
            "goal_lineage": list(self.goal_lineage),
            "authorizes_repair": PROJECTION_AUTHORIZES_REPAIR,
            "is_completion_evidence": PROJECTION_IS_COMPLETION_EVIDENCE,
        }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_finding_for_task(
    finding: ContractFindingRecord | Mapping[str, Any],
    *,
    policy: FindingTaskSourcePolicy | None = None,
    admitted: bool | None = None,
) -> tuple[TaskDisposition, tuple[str, ...]]:
    """Classify a finding as executable, review, or skipped.

    Returns ``(disposition, reasons)``.  Only ``EXECUTABLE`` may become a
    repair task; everything else is review or skip.
    """

    selected = policy or FindingTaskSourcePolicy()
    record = _finding_as_record(finding)
    reasons: list[str] = []

    if admitted is False:
        return TaskDisposition.REVIEW, (ReviewReason.NOT_ADMITTED.value,)
    if record.rejection_reasons:
        return TaskDisposition.REVIEW, (ReviewReason.REJECTED.value,)
    if record.superseded_by_cid:
        return TaskDisposition.SKIPPED, (ReviewReason.STALE.value,)
    if record.freshness is EvidenceFreshness.STALE:
        return TaskDisposition.REVIEW, (ReviewReason.STALE.value,)
    if record.partial:
        return TaskDisposition.REVIEW, (ReviewReason.PARTIAL.value,)

    status = record.status
    if status is FindingStatus.AMBIGUOUS:
        return TaskDisposition.REVIEW, (ReviewReason.AMBIGUOUS.value,)
    if status is FindingStatus.UNSUPPORTED:
        return TaskDisposition.REVIEW, (ReviewReason.UNSUPPORTED.value,)
    if status is FindingStatus.INCONCLUSIVE:
        return TaskDisposition.REVIEW, (ReviewReason.INCONCLUSIVE.value,)
    if status is FindingStatus.SUSPECTED:
        return TaskDisposition.REVIEW, (ReviewReason.SUSPECTED.value,)
    if status is FindingStatus.STALE:
        return TaskDisposition.REVIEW, (ReviewReason.STALE.value,)
    if status is not FindingStatus.CONTRACT_BROKEN:
        return TaskDisposition.REVIEW, (ReviewReason.NOT_ADMITTED.value,)

    if not record.root_cause_family:
        return TaskDisposition.REVIEW, (ReviewReason.MISSING_ROOT_CAUSE.value,)

    outputs = _output_paths_from_finding(record)
    symbols = _symbols_from_finding(record)
    if not outputs or not symbols:
        return TaskDisposition.REVIEW, (ReviewReason.MISSING_SCOPE.value,)

    out_of_root = [
        path
        for path in outputs
        if not _path_within_roots(path, selected.write_roots)
    ]
    if out_of_root:
        return TaskDisposition.REVIEW, (ReviewReason.OUT_OF_ROOT.value,)

    if (
        len(outputs) > selected.max_output_paths
        or len(symbols) > selected.max_symbols
    ):
        return TaskDisposition.REVIEW, (ReviewReason.BROAD.value,)

    # Probe breadth through the shared task-quality policy.
    effects = _effects_from_finding(record)
    validation = _default_validation_plan(outputs, symbols)
    estimated_tokens = _estimate_task_tokens(outputs, symbols, selected)
    candidate = TaskCandidate(
        title=record.summary or record.root_cause_family,
        goal_id=selected.goal_id,
        acceptance=(
            f"Restore contract for {record.root_cause_family}",
            "Finding CID remains reproducible from evidence",
        ),
        effects=effects,
        evidence_subset=(record.finding_cid,),
        outputs=outputs,
        validation_commands=validation,
        context_paths=outputs,
        predicted_paths=outputs,
        predicted_symbols=symbols,
        predicted_interfaces=record.interfaces,
        resource_class=selected.resource_class,
        token_class=selected.token_class,
        merge_fate=record.merge_fate or record.root_cause_family,
        estimated_context_tokens=min(
            selected.context_ceiling_tokens, estimated_tokens
        ),
        estimated_tokens=estimated_tokens,
        estimated_merge_risk_millionths=_severity_risk_millionths(
            record.severity
        ),
    )
    if is_over_broad(candidate, selected.quality):
        return TaskDisposition.REVIEW, (ReviewReason.BROAD.value,)

    if admitted is None and not record.actionable:
        # Fail closed when admission is unknown and the record itself is not
        # actionable (e.g. missing counterexample binding).
        reasons.append(ReviewReason.NOT_ADMITTED.value)
        return TaskDisposition.REVIEW, tuple(reasons)

    return TaskDisposition.EXECUTABLE, ()


def build_review_record(
    finding: ContractFindingRecord | Mapping[str, Any],
    *,
    reasons: Sequence[str],
    policy: FindingTaskSourcePolicy | None = None,
    review_index: int = 1,
) -> ReviewRecord:
    selected = policy or FindingTaskSourcePolicy()
    record = _finding_as_record(finding)
    review_id = f"{selected.task_prefix}-REV-{review_index:04d}"
    return ReviewRecord(
        review_id=review_id,
        finding_cids=(record.finding_cid,),
        reasons=tuple(reasons),
        summary=record.summary or "Finding requires human review",
        root_cause_family=record.root_cause_family,
        outputs=_output_paths_from_finding(record),
        symbols=_symbols_from_finding(record),
        goal_id=selected.goal_id,
        tree_id=record.tree_id,
        provenance_cids=_provenance_cids(record),
    )


def build_repair_task(
    finding: ContractFindingRecord | Mapping[str, Any],
    *,
    policy: FindingTaskSourcePolicy | None = None,
    task_index: int = 1,
    dependencies: Sequence[str] = (),
    supersedes_task_ids: Sequence[str] = (),
    finding_cids: Sequence[str] | None = None,
) -> RepairTaskRecord:
    """Build one executable repair task from a classified admitted finding."""

    selected = policy or FindingTaskSourcePolicy()
    record = _finding_as_record(finding)
    disposition, reasons = classify_finding_for_task(
        record, policy=selected, admitted=True
    )
    if disposition is not TaskDisposition.EXECUTABLE:
        raise FindingTaskSourceError(
            "cannot build repair task for non-executable finding: "
            + ",".join(reasons)
        )

    outputs = _output_paths_from_finding(record)
    symbols = _symbols_from_finding(record)
    effects = _effects_from_finding(record)
    validation = _default_validation_plan(outputs, symbols)
    proof = _default_proof_plan(record)
    merge_fate = record.merge_fate or record.root_cause_family
    cids = tuple(finding_cids) if finding_cids is not None else (record.finding_cid,)
    semantic = _semantic_task_key(
        root_cause_family=record.root_cause_family,
        merge_fate=merge_fate,
        outputs=outputs,
        symbols=symbols,
        validation_plan=validation,
        goal_id=selected.goal_id,
    )
    task_id = f"{selected.task_prefix}-{task_index:04d}"
    acceptance = (
        f"Repair root-cause family {record.root_cause_family}",
        f"Exact outputs updated: {', '.join(outputs)}",
        "Validation and proof plans pass without expanding authority",
        f"Finding evidence remains bound to CIDs: {', '.join(cids)}",
    )
    return RepairTaskRecord(
        task_id=task_id,
        title=(
            f"Repair {record.root_cause_family}: "
            f"{record.summary[:120] if record.summary else symbols[0]}"
        ),
        goal_id=selected.goal_id,
        root_cause_family=record.root_cause_family,
        outputs=outputs,
        symbols=symbols,
        effects=effects,
        dependencies=tuple(dependencies),
        conflict_domain=_conflict_domain(
            root_cause_family=record.root_cause_family,
            merge_fate=merge_fate,
            outputs=outputs,
            symbols=symbols,
        ),
        validation_plan=validation,
        proof_plan=proof,
        finding_cids=cids,
        provenance_cids=_provenance_cids(record),
        risk_millionths=_severity_risk_millionths(record.severity),
        resource_class=selected.resource_class,
        token_class=selected.token_class,
        context_ceiling_bytes=selected.context_ceiling_bytes,
        context_ceiling_tokens=selected.context_ceiling_tokens,
        merge_fate=merge_fate,
        semantic_key=semantic,
        track=selected.default_track,
        interfaces=record.interfaces,
        acceptance=acceptance,
        board_namespace=selected.board_namespace,
        tree_id=record.tree_id,
        policy_revision=record.policy_revision,
        parent_goal_id=selected.parent_goal_id,
        goal_lineage=selected.goal_lineage,
        supersedes_task_ids=tuple(supersedes_task_ids),
    )


def _can_coalesce_repair_tasks(
    left: RepairTaskRecord,
    right: RepairTaskRecord,
    *,
    policy: FindingTaskSourcePolicy,
) -> bool:
    """Return whether two repair tasks may share one merge fate and validation.

    Acceptance requires shared validation and merge fate; risk uses the max
    of the pair (same conflict domain) rather than a sum that would block
    every high-severity family.
    """

    left_c = left.to_task_candidate()
    right_c = right.to_task_candidate()
    if left.semantic_key == right.semantic_key:
        return False
    if not is_tiny(left_c, policy.quality) or not is_tiny(right_c, policy.quality):
        return False
    if normalize_identity_text(left.merge_fate) != normalize_identity_text(
        right.merge_fate
    ):
        return False
    if normalize_identity_text(left.goal_id) != normalize_identity_text(
        right.goal_id
    ):
        return False
    if tuple(
        normalize_identity_text(item) for item in left.goal_lineage
    ) != tuple(normalize_identity_text(item) for item in right.goal_lineage):
        return False
    if normalize_identity_text(left.root_cause_family) != normalize_identity_text(
        right.root_cause_family
    ):
        return False
    if left.outputs != right.outputs:
        return False
    if left.validation_plan != right.validation_plan:
        return False
    if left.proof_plan != right.proof_plan:
        return False
    if left.resource_class != right.resource_class:
        return False
    if left.token_class != right.token_class:
        return False
    quality = policy.quality
    if (
        len(set(left.acceptance) | set(right.acceptance))
        > quality.max_acceptance_criteria
    ):
        return False
    if len(set(left.effects) | set(right.effects)) > quality.max_effects:
        return False
    if (
        len(set(left.symbols) | set(right.symbols))
        > quality.max_predicted_symbols
    ):
        return False
    if (
        left_c.estimated_tokens + right_c.estimated_tokens
        > quality.max_estimated_tokens
    ):
        return False
    if max(left.risk_millionths, right.risk_millionths) > quality.max_merge_risk_millionths:
        return False
    return True


def coalesce_repair_tasks(
    tasks: Sequence[RepairTaskRecord],
    *,
    policy: FindingTaskSourcePolicy | None = None,
) -> tuple[RepairTaskRecord, ...]:
    """Coalesce related *tiny* tasks that share validation and merge fate.

    Tasks that do not share validation commands, merge fate, resource class,
    and outputs remain independent.  Over-broad results of coalescing are
    rejected and the originals kept.
    """

    selected = policy or FindingTaskSourcePolicy()
    if not selected.coalesce_tiny or len(tasks) < 2:
        return tuple(tasks)

    # Group by shared validation + merge fate (and compatible scope/class).
    groups: dict[tuple[Any, ...], list[RepairTaskRecord]] = {}
    for task in tasks:
        key = (
            normalize_identity_text(task.merge_fate),
            tuple(sorted(normalize_identity_text(v) for v in task.validation_plan)),
            tuple(sorted(normalize_identity_text(v) for v in task.proof_plan)),
            task.resource_class,
            task.token_class,
            tuple(task.outputs),
            normalize_identity_text(task.goal_id),
            normalize_identity_text(task.root_cause_family),
        )
        groups.setdefault(key, []).append(task)

    result: list[RepairTaskRecord] = []
    for group in groups.values():
        group = sorted(group, key=lambda item: item.task_id)
        if len(group) == 1:
            result.append(group[0])
            continue
        anchor = group[0]
        if any(
            not _can_coalesce_repair_tasks(anchor, item, policy=selected)
            for item in group[1:]
        ):
            result.extend(group)
            continue

        finding_cids = _strings(
            [cid for task in group for cid in task.finding_cids],
            field_name="finding_cids",
        )
        provenance = _strings(
            [cid for task in group for cid in task.provenance_cids],
            field_name="provenance_cids",
            maximum=MAX_PROVENANCE_CIDS,
        )
        symbols = _strings(
            [symbol for task in group for symbol in task.symbols],
            field_name="symbols",
        )
        effects = _strings(
            [effect for task in group for effect in task.effects],
            field_name="effects",
        )
        acceptance = _strings(
            [item for task in group for item in task.acceptance],
            field_name="acceptance",
        )
        interfaces = _strings(
            [item for task in group for item in task.interfaces],
            field_name="interfaces",
        )
        dependencies = _strings(
            [item for task in group for item in task.dependencies],
            field_name="dependencies",
        )
        semantic = _semantic_task_key(
            root_cause_family=anchor.root_cause_family,
            merge_fate=anchor.merge_fate,
            outputs=anchor.outputs,
            symbols=symbols,
            validation_plan=anchor.validation_plan,
            goal_id=anchor.goal_id,
        )
        merged = RepairTaskRecord(
            task_id=anchor.task_id,
            title="; ".join(sorted({task.title for task in group if task.title})),
            goal_id=anchor.goal_id,
            root_cause_family=anchor.root_cause_family,
            outputs=anchor.outputs,
            symbols=symbols,
            effects=effects,
            dependencies=dependencies,
            conflict_domain=_conflict_domain(
                root_cause_family=anchor.root_cause_family,
                merge_fate=anchor.merge_fate,
                outputs=anchor.outputs,
                symbols=symbols,
            ),
            validation_plan=anchor.validation_plan,
            proof_plan=anchor.proof_plan,
            finding_cids=finding_cids,
            provenance_cids=provenance,
            risk_millionths=max(task.risk_millionths for task in group),
            resource_class=anchor.resource_class,
            token_class=anchor.token_class,
            context_ceiling_bytes=anchor.context_ceiling_bytes,
            context_ceiling_tokens=anchor.context_ceiling_tokens,
            merge_fate=anchor.merge_fate,
            semantic_key=semantic,
            track=anchor.track,
            interfaces=interfaces,
            acceptance=acceptance,
            board_namespace=anchor.board_namespace,
            tree_id=anchor.tree_id,
            policy_revision=anchor.policy_revision,
            parent_goal_id=anchor.parent_goal_id,
            goal_lineage=anchor.goal_lineage,
            supersedes_task_ids=tuple(
                sorted(
                    {
                        *anchor.supersedes_task_ids,
                        *(task.task_id for task in group[1:]),
                    }
                )
            ),
        )
        # Fail closed if the merge exceeds policy breadth.
        if (
            len(merged.outputs) > selected.max_output_paths
            or len(merged.symbols) > selected.max_symbols
            or len(merged.effects) > selected.max_effects
            or is_over_broad(merged.to_task_candidate(), selected.quality)
        ):
            result.extend(group)
            continue
        result.append(merged)
    result.sort(key=lambda task: (task.semantic_key, task.task_id))
    return tuple(result)


# ---------------------------------------------------------------------------
# Projections (diagnostic only)
# ---------------------------------------------------------------------------


def project_board_json(snapshot: BoardSnapshot) -> dict[str, Any]:
    """JSON projection.  Never authorizes repair or completion."""

    payload = snapshot.to_dict()
    payload["schema"] = PROJECTION_SCHEMA
    payload["projection_kind"] = "json"
    payload["evidence"] = FINDING_TASKBOARD_EVIDENCE
    payload["evidence_terms"] = list(FINDING_TASKBOARD_G101_EVIDENCE_TERMS)
    payload["authorizes_repair"] = PROJECTION_AUTHORIZES_REPAIR
    payload["is_completion_evidence"] = PROJECTION_IS_COMPLETION_EVIDENCE
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(raw) > MAX_JSON_BYTES:
        raise FindingTaskBoundsError(
            f"JSON projection exceeds {MAX_JSON_BYTES} bytes"
        )
    return payload


def project_board_markdown(
    snapshot: BoardSnapshot,
    *,
    max_bytes: int = MAX_MARKDOWN_BYTES,
) -> str:
    """Bounded Markdown projection of executable tasks and reviews."""

    lines: list[str] = [
        f"# Finding repair taskboard ({snapshot.board_namespace})",
        "",
        f"- revision: {snapshot.revision}",
        f"- policy_id: {snapshot.policy_id or 'unset'}",
        f"- tree_id: {snapshot.tree_id or 'unset'}",
        f"- goal_id: {snapshot.goal_id}",
        f"- goal_lineage: {', '.join(snapshot.goal_lineage)}",
        f"- evidence: {FINDING_TASKBOARD_EVIDENCE}",
        f"- executable_tasks: {len(snapshot.executable_tasks)}",
        f"- reviews: {len(snapshot.reviews)}",
        f"- authorizes_repair: {str(PROJECTION_AUTHORIZES_REPAIR).lower()}",
        f"- is_completion_evidence: {str(PROJECTION_IS_COMPLETION_EVIDENCE).lower()}",
        "",
    ]
    if snapshot.tasks:
        lines.append("## Executable repair tasks")
        lines.append("")
    for task in snapshot.tasks:
        lines.extend(
            [
                f"### {task.task_id} {task.title}",
                "",
                f"- goal_id: {task.goal_id}",
                f"- parent_goal_id: {task.parent_goal_id or 'unset'}",
                f"- goal_lineage: {', '.join(task.goal_lineage)}",
                f"- root_cause_family: {task.root_cause_family}",
                f"- merge_fate: {task.merge_fate}",
                f"- conflict_domain: {task.conflict_domain}",
                f"- resource_class: {task.resource_class}",
                f"- token_class: {task.token_class}",
                f"- risk_millionths: {task.risk_millionths}",
                f"- context_ceiling_bytes: {task.context_ceiling_bytes}",
                f"- context_ceiling_tokens: {task.context_ceiling_tokens}",
                f"- finding_cids: {', '.join(task.finding_cids)}",
                f"- provenance_cids: {', '.join(task.provenance_cids[:8])}",
                f"- outputs: {', '.join(task.outputs)}",
                f"- symbols: {', '.join(task.symbols)}",
                f"- effects: {', '.join(task.effects)}",
                f"- dependencies: {', '.join(task.dependencies) or 'none'}",
                f"- validation_plan: {'; '.join(task.validation_plan)}",
                f"- proof_plan: {'; '.join(task.proof_plan)}",
                f"- semantic_key: {task.semantic_key}",
                f"- executable: true",
                "",
            ]
        )
    if snapshot.reviews:
        lines.append("## Non-executable review records")
        lines.append("")
    for review in snapshot.reviews:
        lines.extend(
            [
                f"### {review.review_id}",
                "",
                f"- finding_cids: {', '.join(review.finding_cids)}",
                f"- reasons: {', '.join(review.reasons)}",
                f"- summary: {review.summary}",
                f"- root_cause_family: {review.root_cause_family or 'unset'}",
                f"- executable: false",
                "",
            ]
        )
    text = "\n".join(lines).rstrip() + "\n"
    if len(text.encode("utf-8")) > max_bytes:
        raise FindingTaskBoundsError(
            f"Markdown projection exceeds {max_bytes} bytes"
        )
    return text


def project_board_duckdb_rows(
    snapshot: BoardSnapshot,
    *,
    max_rows: int = MAX_DUCKDB_ROWS,
) -> tuple[dict[str, Any], ...]:
    """DuckDB-ready row projection (no DDL authority, no repair authority)."""

    rows: list[dict[str, Any]] = []
    for task in snapshot.tasks:
        rows.append(
            {
                "kind": "repair_task",
                "record_id": task.task_id,
                "record_cid": task.task_cid,
                "goal_id": task.goal_id,
                "parent_goal_id": task.parent_goal_id,
                "goal_lineage_json": json.dumps(
                    list(task.goal_lineage), sort_keys=True
                ),
                "root_cause_family": task.root_cause_family,
                "merge_fate": task.merge_fate,
                "conflict_domain": task.conflict_domain,
                "outputs_json": json.dumps(list(task.outputs), sort_keys=True),
                "symbols_json": json.dumps(list(task.symbols), sort_keys=True),
                "effects_json": json.dumps(list(task.effects), sort_keys=True),
                "dependencies_json": json.dumps(
                    list(task.dependencies), sort_keys=True
                ),
                "validation_plan_json": json.dumps(
                    list(task.validation_plan), sort_keys=True
                ),
                "proof_plan_json": json.dumps(
                    list(task.proof_plan), sort_keys=True
                ),
                "finding_cids_json": json.dumps(
                    list(task.finding_cids), sort_keys=True
                ),
                "provenance_cids_json": json.dumps(
                    list(task.provenance_cids), sort_keys=True
                ),
                "risk_millionths": task.risk_millionths,
                "resource_class": task.resource_class,
                "token_class": task.token_class,
                "context_ceiling_bytes": task.context_ceiling_bytes,
                "context_ceiling_tokens": task.context_ceiling_tokens,
                "semantic_key": task.semantic_key,
                "executable": True,
                "board_namespace": snapshot.board_namespace,
                "board_revision": snapshot.revision,
                "evidence": FINDING_TASKBOARD_EVIDENCE,
                "authorizes_repair": PROJECTION_AUTHORIZES_REPAIR,
                "is_completion_evidence": PROJECTION_IS_COMPLETION_EVIDENCE,
            }
        )
    for review in snapshot.reviews:
        rows.append(
            {
                "kind": "review",
                "record_id": review.review_id,
                "record_cid": review.review_cid,
                "goal_id": review.goal_id,
                "parent_goal_id": "",
                "goal_lineage_json": "[]",
                "root_cause_family": review.root_cause_family,
                "merge_fate": "",
                "conflict_domain": "",
                "outputs_json": json.dumps(list(review.outputs), sort_keys=True),
                "symbols_json": json.dumps(list(review.symbols), sort_keys=True),
                "effects_json": "[]",
                "dependencies_json": "[]",
                "validation_plan_json": "[]",
                "proof_plan_json": "[]",
                "finding_cids_json": json.dumps(
                    list(review.finding_cids), sort_keys=True
                ),
                "provenance_cids_json": json.dumps(
                    list(review.provenance_cids), sort_keys=True
                ),
                "risk_millionths": 0,
                "resource_class": "",
                "token_class": "",
                "context_ceiling_bytes": 0,
                "context_ceiling_tokens": 0,
                "semantic_key": "",
                "executable": False,
                "board_namespace": snapshot.board_namespace,
                "board_revision": snapshot.revision,
                "evidence": FINDING_TASKBOARD_EVIDENCE,
                "authorizes_repair": PROJECTION_AUTHORIZES_REPAIR,
                "is_completion_evidence": PROJECTION_IS_COMPLETION_EVIDENCE,
                "reasons_json": json.dumps(list(review.reasons), sort_keys=True),
                "summary": review.summary,
            }
        )
    if len(rows) > max_rows:
        raise FindingTaskBoundsError(
            f"DuckDB projection exceeds {max_rows} rows"
        )
    rows.sort(key=lambda row: (row["kind"], row["record_id"]))
    return tuple(rows)


def project_board_sarif_links(
    snapshot: BoardSnapshot,
) -> dict[str, Any]:
    """SARIF-linked projection: task/review → finding CIDs only.

    Does not embed source bodies or claim SARIF results as repair authority.
    """

    links: list[dict[str, Any]] = []
    for task in snapshot.tasks:
        links.append(
            {
                "kind": "repair_task",
                "record_id": task.task_id,
                "record_cid": task.task_cid,
                "finding_cids": list(task.finding_cids),
                "provenance_cids": list(task.provenance_cids),
                "root_cause_family": task.root_cause_family,
                "executable": True,
            }
        )
    for review in snapshot.reviews:
        links.append(
            {
                "kind": "review",
                "record_id": review.review_id,
                "record_cid": review.review_cid,
                "finding_cids": list(review.finding_cids),
                "provenance_cids": list(review.provenance_cids),
                "reasons": list(review.reasons),
                "executable": False,
            }
        )
    links.sort(key=lambda item: (item["kind"], item["record_id"]))
    return {
        "schema": PROJECTION_SCHEMA,
        "projection_kind": "sarif_links",
        "board_namespace": snapshot.board_namespace,
        "board_cid": snapshot.board_cid,
        "revision": snapshot.revision,
        "goal_id": snapshot.goal_id,
        "goal_lineage": list(snapshot.goal_lineage),
        "evidence": FINDING_TASKBOARD_EVIDENCE,
        "evidence_terms": list(FINDING_TASKBOARD_G101_EVIDENCE_TERMS),
        "links": links,
        "authorizes_repair": PROJECTION_AUTHORIZES_REPAIR,
        "is_completion_evidence": PROJECTION_IS_COMPLETION_EVIDENCE,
        "sarif_is_diagnostic_only": True,
    }


def finding_taskboard_evidence_terms() -> tuple[str, ...]:
    """Return the closed VFS-G101 evidence terms covered by this taskboard.

    Proves that the second repair taskboard exists as a first-class,
    content-addressed artifact (``vfs/finding-taskboard@1``) that binds
    exact outputs, goal lineage, and stable identity without granting
    report-driven edit authority.
    """

    return FINDING_TASKBOARD_G101_EVIDENCE_TERMS


# ---------------------------------------------------------------------------
# FindingTaskSource
# ---------------------------------------------------------------------------


@dataclass
class FindingTaskSource:
    """Materialize a stable repair taskboard from admitted findings.

    Durable optional root layout::

        board.json          current board snapshot
        state.json          finding_cid → task/review identity map
    """

    policy: FindingTaskSourcePolicy = field(
        default_factory=FindingTaskSourcePolicy
    )
    root: Path | None = None

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = Path(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks: dict[str, RepairTaskRecord] = {}
        self._reviews: dict[str, ReviewRecord] = {}
        # finding_cid → task_id or review_id
        self._finding_index: dict[str, str] = {}
        # semantic_key → task_id
        self._semantic_index: dict[str, str] = {}
        self._revision = 0
        self._tree_id = ""
        if self.root is not None:
            self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    @property
    def _board_path(self) -> Path | None:
        return None if self.root is None else self.root / "board.json"

    @property
    def _state_path(self) -> Path | None:
        return None if self.root is None else self.root / "state.json"

    def _load(self) -> None:
        board_path = self._board_path
        state_path = self._state_path
        if board_path is None or state_path is None:
            return
        if board_path.exists():
            payload = json.loads(board_path.read_text(encoding="utf-8"))
            snapshot = BoardSnapshot.from_dict(payload)
            self._tasks = {task.task_id: task for task in snapshot.tasks}
            self._reviews = {
                review.review_id: review for review in snapshot.reviews
            }
            self._revision = snapshot.revision
            self._tree_id = snapshot.tree_id
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self._finding_index = {
                str(k): str(v)
                for k, v in (state.get("finding_index") or {}).items()
            }
            self._semantic_index = {
                str(k): str(v)
                for k, v in (state.get("semantic_index") or {}).items()
            }

    def _persist(self, snapshot: BoardSnapshot) -> None:
        board_path = self._board_path
        state_path = self._state_path
        if board_path is None or state_path is None:
            return
        board_path.write_text(
            json.dumps(snapshot.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(
                {
                    "finding_index": self._finding_index,
                    "semantic_index": self._semantic_index,
                    "revision": self._revision,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def snapshot(self) -> BoardSnapshot:
        with self._lock:
            tasks = tuple(
                sorted(self._tasks.values(), key=lambda t: t.task_id)
            )
            reviews = tuple(
                sorted(self._reviews.values(), key=lambda r: r.review_id)
            )
            return BoardSnapshot(
                tasks=tasks,
                reviews=reviews,
                board_namespace=self.policy.board_namespace,
                policy_id=self.policy.policy_id,
                tree_id=self._tree_id,
                revision=self._revision,
                goal_id=self.policy.goal_id,
                goal_lineage=self.policy.goal_lineage,
                evidence=FINDING_TASKBOARD_EVIDENCE,
            )

    def materialize(
        self,
        findings: Sequence[ContractFindingRecord | Mapping[str, Any]] | None = None,
        *,
        ledger: ContractFindingLedger | None = None,
        admitted_only: bool = True,
        tree_id: str = "",
    ) -> MaterializationReceipt:
        """Materialize repair tasks / reviews from fresh admitted findings.

        Provide either ``findings`` or a ``ledger``.  When a ledger is given,
        only currently admitted findings are consumed (unless
        ``admitted_only`` is false, in which case current non-stale entries
        are still classified and may become reviews).
        """

        with self._lock:
            return self._materialize_locked(
                findings=findings,
                ledger=ledger,
                admitted_only=admitted_only,
                tree_id=tree_id,
            )

    def _materialize_locked(
        self,
        *,
        findings: Sequence[ContractFindingRecord | Mapping[str, Any]] | None,
        ledger: ContractFindingLedger | None,
        admitted_only: bool,
        tree_id: str,
    ) -> MaterializationReceipt:
        records: list[tuple[ContractFindingRecord, bool]] = []
        if ledger is not None:
            projection = ledger.projection()
            admitted_cids = {
                entry.finding_cid for entry in projection.admitted
            }
            if admitted_only:
                for record in ledger.current_findings(admitted_only=True):
                    records.append((record, True))
            else:
                for entry in projection.entries:
                    record = ledger.get(entry.finding_cid)
                    if record is None:
                        continue
                    records.append(
                        (record, entry.finding_cid in admitted_cids)
                    )
        if findings is not None:
            for item in findings:
                record = _finding_as_record(item)
                records.append((record, True if admitted_only else record.actionable))

        if not records and findings is None and ledger is None:
            raise FindingTaskSourceError(
                "materialize requires findings or a ledger"
            )

        # Deterministic order by finding CID.
        records.sort(key=lambda pair: pair[0].finding_cid)

        created: list[str] = []
        superseded: list[str] = []
        no_ops: list[str] = []
        review_ids: list[str] = []
        reasons: list[str] = []

        pending_tasks: list[RepairTaskRecord] = []
        next_task_index = self._next_task_index()
        next_review_index = self._next_review_index()

        for record, is_admitted in records:
            if tree_id:
                self._tree_id = tree_id
            elif record.tree_id and not self._tree_id:
                self._tree_id = record.tree_id

            # Stable finding replay → no-op when already bound to same body.
            existing_id = self._finding_index.get(record.finding_cid)
            if existing_id and existing_id in self._tasks:
                existing_task = self._tasks[existing_id]
                if record.finding_cid in existing_task.finding_cids:
                    # Same finding still on board: no-op.
                    no_ops.append(record.finding_cid)
                    continue
            if existing_id and existing_id in self._reviews:
                existing_review = self._reviews[existing_id]
                if (
                    record.finding_cid in existing_review.finding_cids
                    and set(existing_review.reasons)
                ):
                    # Re-classify; if still review with same reasons, no-op.
                    disposition, review_reasons = classify_finding_for_task(
                        record, policy=self.policy, admitted=is_admitted
                    )
                    if (
                        disposition is TaskDisposition.REVIEW
                        and set(review_reasons) == set(existing_review.reasons)
                    ):
                        no_ops.append(record.finding_cid)
                        continue

            disposition, review_reasons = classify_finding_for_task(
                record, policy=self.policy, admitted=is_admitted
            )
            if disposition is TaskDisposition.SKIPPED:
                no_ops.append(record.finding_cid)
                reasons.append("skipped_superseded_finding")
                continue
            if disposition is TaskDisposition.REVIEW:
                review = build_review_record(
                    record,
                    reasons=review_reasons,
                    policy=self.policy,
                    review_index=next_review_index,
                )
                next_review_index += 1
                self._reviews[review.review_id] = review
                self._finding_index[record.finding_cid] = review.review_id
                review_ids.append(review.review_id)
                reasons.extend(review_reasons)
                continue

            # Executable path.
            outputs = _output_paths_from_finding(record)
            symbols = _symbols_from_finding(record)
            validation = _default_validation_plan(outputs, symbols)
            merge_fate = record.merge_fate or record.root_cause_family
            semantic = _semantic_task_key(
                root_cause_family=record.root_cause_family,
                merge_fate=merge_fate,
                outputs=outputs,
                symbols=symbols,
                validation_plan=validation,
                goal_id=self.policy.goal_id,
            )

            prior_task_id = self._semantic_index.get(semantic)
            supersedes: list[str] = []
            if prior_task_id and prior_task_id in self._tasks:
                prior = self._tasks[prior_task_id]
                if set(prior.finding_cids) == {record.finding_cid}:
                    no_ops.append(record.finding_cid)
                    continue
                # Changed evidence under the same semantic key → supersede.
                supersedes.append(prior_task_id)
                for prior_cid in prior.finding_cids:
                    if self._finding_index.get(prior_cid) == prior_task_id:
                        self._finding_index.pop(prior_cid, None)
                del self._tasks[prior_task_id]
                self._semantic_index.pop(semantic, None)
                superseded.append(prior_task_id)
                reasons.append("superseded_changed_evidence")

            # Also supersede any prior task that listed this finding under a
            # different semantic key (evidence rewrite of the same CID).
            prior_binding = self._finding_index.get(record.finding_cid)
            if (
                prior_binding
                and prior_binding in self._tasks
                and prior_binding not in supersedes
            ):
                prior = self._tasks[prior_binding]
                supersedes.append(prior_binding)
                for prior_cid in prior.finding_cids:
                    if self._finding_index.get(prior_cid) == prior_binding:
                        self._finding_index.pop(prior_cid, None)
                self._semantic_index.pop(prior.semantic_key, None)
                del self._tasks[prior_binding]
                superseded.append(prior_binding)
                reasons.append("superseded_rebound_finding")

            task = build_repair_task(
                record,
                policy=self.policy,
                task_index=next_task_index,
                supersedes_task_ids=supersedes,
            )
            next_task_index += 1
            pending_tasks.append(task)

        # Coalesce only tiny related tasks with shared validation/merge fate.
        if pending_tasks:
            coalesced = coalesce_repair_tasks(
                pending_tasks, policy=self.policy
            )
            # Re-index after coalescing.
            for task in coalesced:
                # Drop any intermediate pending ids that were coalesced away.
                pass
            # Remove pending that are not in coalesced set.
            coalesced_ids = {task.task_id for task in coalesced}
            # Also include tasks that were merged into another (superseded by coalesce).
            for task in pending_tasks:
                if task.task_id not in coalesced_ids:
                    # Absorbed into a coalesced task.
                    for cid in task.finding_cids:
                        # Will be rebound by coalesced owner.
                        pass
            for task in coalesced:
                if len(self._tasks) + 1 > self.policy.max_board_tasks:
                    raise FindingTaskBoundsError(
                        f"board exceeds max_board_tasks={self.policy.max_board_tasks}"
                    )
                self._tasks[task.task_id] = task
                self._semantic_index[task.semantic_key] = task.task_id
                for cid in task.finding_cids:
                    self._finding_index[cid] = task.task_id
                created.append(task.task_id)

        if created or superseded or review_ids:
            self._revision += 1

        snapshot = self.snapshot()
        self._persist(snapshot)

        if created and not superseded and not review_ids and not no_ops:
            outcome = MaterializationOutcome.CREATED
        elif not created and not superseded and not review_ids and no_ops:
            outcome = MaterializationOutcome.NO_OP
        elif superseded and not created and not review_ids:
            outcome = MaterializationOutcome.SUPERSEDED
        elif review_ids and not created:
            outcome = MaterializationOutcome.REVIEW_ONLY
        elif created and superseded:
            outcome = MaterializationOutcome.UPDATED
        elif created or superseded or review_ids or no_ops:
            outcome = MaterializationOutcome.MIXED
        else:
            outcome = MaterializationOutcome.NO_OP

        return MaterializationReceipt(
            outcome=outcome,
            created_task_ids=tuple(sorted(set(created))),
            superseded_task_ids=tuple(sorted(set(superseded))),
            no_op_finding_cids=tuple(sorted(set(no_ops))),
            review_ids=tuple(sorted(set(review_ids))),
            board_cid=snapshot.board_cid,
            revision=self._revision,
            reasons=tuple(sorted(set(reasons))),
            evidence=FINDING_TASKBOARD_EVIDENCE,
            goal_id=self.policy.goal_id,
            goal_lineage=self.policy.goal_lineage,
        )

    def _next_task_index(self) -> int:
        indices = []
        prefix = self.policy.task_prefix + "-"
        for task_id in self._tasks:
            if task_id.startswith(prefix):
                suffix = task_id[len(prefix) :]
                if suffix.isdigit():
                    indices.append(int(suffix))
        return (max(indices) + 1) if indices else 1

    def _next_review_index(self) -> int:
        indices = []
        prefix = self.policy.task_prefix + "-REV-"
        for review_id in self._reviews:
            if review_id.startswith(prefix):
                suffix = review_id[len(prefix) :]
                if suffix.isdigit():
                    indices.append(int(suffix))
        return (max(indices) + 1) if indices else 1

    def project_json(self) -> dict[str, Any]:
        return project_board_json(self.snapshot())

    def project_markdown(self) -> str:
        return project_board_markdown(self.snapshot())

    def project_duckdb_rows(self) -> tuple[dict[str, Any], ...]:
        return project_board_duckdb_rows(self.snapshot())

    def project_sarif_links(self) -> dict[str, Any]:
        return project_board_sarif_links(self.snapshot())

    def get_task(self, task_id: str) -> RepairTaskRecord | None:
        return self._tasks.get(task_id)

    def get_review(self, review_id: str) -> ReviewRecord | None:
        return self._reviews.get(review_id)

    def task_for_finding(self, finding_cid: str) -> RepairTaskRecord | None:
        bound = self._finding_index.get(finding_cid)
        if bound and bound in self._tasks:
            return self._tasks[bound]
        return None


def materialize_finding_tasks(
    findings: Sequence[ContractFindingRecord | Mapping[str, Any]] | None = None,
    *,
    ledger: ContractFindingLedger | None = None,
    policy: FindingTaskSourcePolicy | None = None,
    root: Path | str | None = None,
    admitted_only: bool = True,
    tree_id: str = "",
) -> tuple[BoardSnapshot, MaterializationReceipt]:
    """Functional facade: materialize once and return snapshot + receipt."""

    source = FindingTaskSource(
        policy=policy or FindingTaskSourcePolicy(),
        root=Path(root) if root is not None else None,
    )
    receipt = source.materialize(
        findings,
        ledger=ledger,
        admitted_only=admitted_only,
        tree_id=tree_id,
    )
    return source.snapshot(), receipt


__all__ = [
    "BOARD_SNAPSHOT_SCHEMA",
    "DEFAULT_BOARD_NAMESPACE",
    "DEFAULT_CONTEXT_CEILING_BYTES",
    "DEFAULT_CONTEXT_CEILING_TOKENS",
    "DEFAULT_GOAL_ID",
    "DEFAULT_GOAL_LINEAGE",
    "DEFAULT_PARENT_GOAL_ID",
    "DEFAULT_ROOT_GOAL_ID",
    "DEFAULT_RESOURCE_CLASS",
    "DEFAULT_TASK_PREFIX",
    "DEFAULT_TOKEN_CLASS",
    "FINDING_TASKBOARD_EVIDENCE",
    "FINDING_TASKBOARD_G101_EVIDENCE_TERMS",
    "FINDING_TASK_RECORD_SCHEMA",
    "FINDING_TASK_SOURCE_VERSION",
    "LEDGER_VERSION",
    "MATERIALIZATION_RECEIPT_SCHEMA",
    "MAX_BOARD_TASKS",
    "MAX_OUTPUT_PATHS",
    "PARENT_FINDING_LEDGER_EVIDENCE",
    "PARENT_FINDING_LEDGER_G100_EVIDENCE_TERMS",
    "PARENT_VULNERABILITY_EVIDENCE_POLICY",
    "PROJECTION_AUTHORIZES_REPAIR",
    "PROJECTION_IS_COMPLETION_EVIDENCE",
    "PROJECTION_SCHEMA",
    "REVIEW_RECORD_SCHEMA",
    "BoardSnapshot",
    "FindingTaskAuthorityError",
    "FindingTaskBoundsError",
    "FindingTaskIntegrityError",
    "FindingTaskSource",
    "FindingTaskSourceError",
    "FindingTaskSourcePolicy",
    "MaterializationOutcome",
    "MaterializationReceipt",
    "RepairTaskRecord",
    "ReviewReason",
    "ReviewRecord",
    "TaskDisposition",
    "build_repair_task",
    "build_review_record",
    "finding_ledger_evidence_terms",
    "classify_finding_for_task",
    "coalesce_repair_tasks",
    "finding_taskboard_evidence_terms",
    "materialize_finding_tasks",
    "project_board_duckdb_rows",
    "project_board_json",
    "project_board_markdown",
    "project_board_sarif_links",
]
