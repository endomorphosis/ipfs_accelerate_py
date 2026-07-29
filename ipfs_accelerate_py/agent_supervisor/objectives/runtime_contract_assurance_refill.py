"""Supervisor-native continuous refill for runtime contract discovery.

``RuntimeContractAssuranceRefill`` (SCA-179) is the transaction boundary between
current runtime discovery (analyzer callback + reverse proof scope) and the
runtime repair board (``RuntimeContractMismatchRefinery`` / SCA-178).

Normative rules:

* no-op scans (empty change set, threshold, or cooldown) make zero provider,
  model, or LLM calls and do not re-invoke the analyzer when unnecessary;
* one-symbol / route / schema / policy changes invalidate **all and only**
  reverse-dependent obligations via :mod:`proof_scope_index`;
* admitted findings must carry current capability, healthy canaries, and a
  goal lineage that refills the correct runtime subgoal;
* task storms and cross-component duplicate repairs are bounded by open-work
  and finding limits plus deterministic finding-identity dedupe;
* crashes recover from checksummed durable state with exact cycle replay; and
* generated tasks never hold completion authority.

The analyzer is an injected typed Python callback.  This module does not walk
repository source or construct shell control routes.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from ..analysis.analyzer_health import ANALYZER_CANARY_SCHEMA, ANALYZER_HEALTH_SCHEMA
from ..proof.mcp_contract_edit_packet import McpContractEditPacket
from ..proof.proof_scope_index import (
    ProofScopeIndex,
    build_proof_scope_index,
    invalidate_proof_evidence,
)
from .contract_assurance_refill import (
    ContractAnalyzerCapability,
    ContractAssuranceAnalysis,
    ContractAssuranceFinding,
    ContractAssuranceGoalLineage,
)
from .contract_mismatch_refinery import ContractRepairTask, parse_contract_repair_board
from .runtime_contract_mismatch_refinery import (
    DEFAULT_RUNTIME_GOAL_ID,
    RuntimeContractMismatchRefinery,
    RuntimeContractMismatchRefineryDecision,
    RuntimeContractMismatchRefineryPolicy,
    RuntimeContractMismatchRefineryReason,
    RuntimeContractMismatchRefineryResult,
)
from .scan_receipts import (
    ExhaustionBinding,
    RefillScanResult,
    ScanMode,
    ScanTerminalReason,
    evaluate_exhaustion_quorum,
)


RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE: Final = (
    "RuntimeContractAssuranceRefill@1"
)
RUNTIME_CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-assurance-refill-state@1"
)
RUNTIME_CONTRACT_ASSURANCE_REFILL_METRICS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-assurance-refill-metrics@1"
)
RUNTIME_CONTRACT_ASSURANCE_REFILL_STATE_VERSION: Final = 1
DEFAULT_ANALYZER_ID: Final = "swissknife.runtime-contract-assurance"
DEFAULT_MIN_OPEN_TASKS: Final = 8
DEFAULT_MAX_OPEN_TASKS: Final = 48
DEFAULT_MAX_FINDINGS: Final = 8
DEFAULT_TIMEOUT_SECONDS: Final = 120.0
DEFAULT_COOLDOWN_SECONDS: Final = 900
DEFAULT_REQUIRED_EXHAUSTION_MEMBERS: Final = 2
DEFAULT_ROOT_GOAL_ID: Final = "SCA-G000"
DEFAULT_RUNTIME_PARENT_GOAL_ID: Final = DEFAULT_RUNTIME_GOAL_ID
MAX_REPLAY_RECORDS: Final = 128
MAX_EXHAUSTION_RECEIPTS: Final = 64
MAX_CHANGED_INPUTS: Final = 256
_GOAL_ID_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,255}$")

# Runtime component subgoals under the SCA-G170 catalog tree.  Findings for a
# component refill that subgoal (or SCA-G176 for cross-component drift).
RUNTIME_COMPONENT_SUBGOALS: Final[Mapping[str, str]] = {
    "model_server": "SCA-G171",
    "orchestrator": "SCA-G172",
    "scheduler": "SCA-G173",
    "supervisor": "SCA-G174",
    "cross_component": "SCA-G175",
    "runtime_drift": DEFAULT_RUNTIME_GOAL_ID,
}

# Semantic change kinds covered by continuous exact invalidation.
SUPPORTED_CHANGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "qualified_symbol",
        "symbol",
        "interface",
        "route",
        "schema",
        "policy",
        "toolchain",
        "file",
    }
)

_CHANGE_KIND_ALIASES: Final[Mapping[str, str]] = {
    "symbol": "qualified_symbol",
    "route": "interface",
    "schema": "interface",
}


class RuntimeContractAssuranceRefillError(ValueError):
    """A refill request, analyzer result, or durable record is malformed."""


class RuntimeContractAssuranceRefillReason(str, Enum):
    """Stable supervisor outcomes and fail-closed rejection reasons."""

    GENERATED = "generated"
    DUPLICATE_ONLY = "duplicate_only"
    EXHAUSTED = "exhausted"
    THRESHOLD_SATISFIED = "threshold_satisfied"
    COOLDOWN = "cooldown"
    NOOP = "noop"
    TIMED_OUT = "timed_out"
    ANALYZER_FAILED = "analyzer_failed"
    CAPABILITY_MISSING = "capability_missing"
    CAPABILITY_STALE = "capability_stale"
    ANALYZER_UNHEALTHY = "analyzer_unhealthy"
    CANARIES_FAILED = "canaries_failed"
    NO_GOAL_LINEAGE = "no_goal_lineage"
    STALE_FINDING = "stale_finding"
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    QUORUM_INCOMPLETE = "quorum_incomplete"
    FINDING_LIMIT = "finding_limit"
    OPEN_WORK_LIMIT = "open_work_limit"
    STATE_RECOVERED = "state_recovered"
    DEPENDENTS_ONLY = "dependents_only"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise RuntimeContractAssuranceRefillError(f"{name} must be a string")
    if (
        value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise RuntimeContractAssuranceRefillError(
            f"{name} must be normalized single-line text"
        )
    if required and not value:
        raise RuntimeContractAssuranceRefillError(f"{name} is required")
    if len(value.encode("utf-8")) > 16_384:
        raise RuntimeContractAssuranceRefillError(f"{name} exceeds its byte bound")
    return value


def _identifier(value: Any, name: str) -> str:
    selected = _text(value, name)
    if not _GOAL_ID_RE.fullmatch(selected):
        raise RuntimeContractAssuranceRefillError(f"{name} is malformed")
    return selected


def _bounded_integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeContractAssuranceRefillError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise RuntimeContractAssuranceRefillError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _finite_seconds(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise RuntimeContractAssuranceRefillError(f"{name} must be numeric")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractAssuranceRefillError(f"{name} must be numeric") from exc
    minimum = 0.0 if allow_zero else 0.001
    if not minimum <= selected <= 86_400.0:
        raise RuntimeContractAssuranceRefillError(
            f"{name} must be between {minimum} and 86400"
        )
    return selected


def _canonical_value(value: Any, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeContractAssuranceRefillError(
            f"{name} must be canonical JSON data"
        ) from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_at(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _digest(value: Mapping[str, Any] | Sequence[Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def normalize_changed_input(raw: Mapping[str, Any] | str) -> dict[str, str]:
    """Normalize one semantic change into ProofScopeIndex input form."""

    if isinstance(raw, str):
        return {"kind": "qualified_symbol", "value": _text(raw, "changed_input")}
    if not isinstance(raw, Mapping):
        raise RuntimeContractAssuranceRefillError("changed_input must be an object")
    kind_raw = raw.get("kind", raw.get("change_kind", "qualified_symbol"))
    kind = _text(kind_raw, "changed_input.kind").casefold()
    kind = _CHANGE_KIND_ALIASES.get(kind, kind)
    if kind not in SUPPORTED_CHANGE_KINDS and kind not in {
        "qualified_symbol",
        "interface",
        "policy",
        "toolchain",
        "file",
    }:
        raise RuntimeContractAssuranceRefillError(
            f"unsupported changed_input kind: {kind}"
        )
    # Map public acceptance nouns onto proof-scope kinds.
    proof_kind = {
        "qualified_symbol": "qualified_symbol",
        "interface": "interface",
        "policy": "policy",
        "toolchain": "toolchain",
        "file": "file",
    }.get(kind, kind)
    value = raw.get("value", raw.get("id", raw.get("name", "")))
    return {
        "kind": proof_kind,
        "value": _text(value, "changed_input.value"),
    }


def normalize_changed_inputs(
    values: Sequence[Mapping[str, Any] | str] | None,
) -> tuple[dict[str, str], ...]:
    if not values:
        return ()
    if len(values) > MAX_CHANGED_INPUTS:
        raise RuntimeContractAssuranceRefillError(
            f"changed_inputs exceeds bound of {MAX_CHANGED_INPUTS}"
        )
    normalized = tuple(
        sorted(
            (normalize_changed_input(item) for item in values),
            key=lambda item: (item["kind"], item["value"]),
        )
    )
    # Dedupe exact semantic inputs.
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in normalized:
        key = (item["kind"], item["value"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def resolve_runtime_subgoal(
    *,
    goal_id: str = "",
    component_id: str = "",
    default: str = DEFAULT_RUNTIME_GOAL_ID,
) -> str:
    """Map a finding's lineage / component onto its refill subgoal."""

    if goal_id:
        return _identifier(goal_id, "goal_id")
    if component_id:
        component = _text(component_id, "component_id").casefold()
        mapped = RUNTIME_COMPONENT_SUBGOALS.get(component)
        if mapped:
            return mapped
    return _identifier(default, "default_subgoal")


@dataclass(frozen=True, slots=True)
class RuntimeContractAssuranceRefillPolicy:
    """Bounds and current capability requirements for one runtime refill."""

    min_open_tasks: int = DEFAULT_MIN_OPEN_TASKS
    max_open_tasks: int = DEFAULT_MAX_OPEN_TASKS
    max_findings_per_run: int = DEFAULT_MAX_FINDINGS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    required_exhaustion_members: int = DEFAULT_REQUIRED_EXHAUSTION_MEMBERS
    analyzer_id: str = DEFAULT_ANALYZER_ID
    expected_analyzer_version: str = ""
    root_goal_id: str = DEFAULT_ROOT_GOAL_ID
    runtime_parent_goal_id: str = DEFAULT_RUNTIME_PARENT_GOAL_ID
    board_namespace: str = "swissknife-symbolic-contract-assurance-v1"
    default_goal_id: str = DEFAULT_RUNTIME_GOAL_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_open_tasks",
            _bounded_integer(
                self.min_open_tasks, "min_open_tasks", minimum=0, maximum=10_000
            ),
        )
        object.__setattr__(
            self,
            "max_open_tasks",
            _bounded_integer(
                self.max_open_tasks, "max_open_tasks", minimum=1, maximum=10_000
            ),
        )
        if self.max_open_tasks < self.min_open_tasks:
            raise RuntimeContractAssuranceRefillError(
                "max_open_tasks must not be below min_open_tasks"
            )
        object.__setattr__(
            self,
            "max_findings_per_run",
            _bounded_integer(
                self.max_findings_per_run,
                "max_findings_per_run",
                minimum=1,
                maximum=1_024,
            ),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _finite_seconds(self.timeout_seconds, "timeout_seconds"),
        )
        object.__setattr__(
            self,
            "cooldown_seconds",
            _bounded_integer(
                self.cooldown_seconds,
                "cooldown_seconds",
                minimum=0,
                maximum=31_536_000,
            ),
        )
        object.__setattr__(
            self,
            "required_exhaustion_members",
            _bounded_integer(
                self.required_exhaustion_members,
                "required_exhaustion_members",
                minimum=1,
                maximum=32,
            ),
        )
        object.__setattr__(
            self, "analyzer_id", _identifier(self.analyzer_id, "analyzer_id")
        )
        object.__setattr__(
            self,
            "expected_analyzer_version",
            _text(
                self.expected_analyzer_version,
                "expected_analyzer_version",
                required=False,
            ),
        )
        object.__setattr__(
            self, "root_goal_id", _identifier(self.root_goal_id, "root_goal_id")
        )
        object.__setattr__(
            self,
            "runtime_parent_goal_id",
            _identifier(self.runtime_parent_goal_id, "runtime_parent_goal_id"),
        )
        object.__setattr__(
            self,
            "board_namespace",
            _text(self.board_namespace, "board_namespace"),
        )
        object.__setattr__(
            self,
            "default_goal_id",
            _identifier(self.default_goal_id, "default_goal_id"),
        )

    @property
    def configuration_revision(self) -> str:
        return _digest(
            {
                "interface": RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE,
                "min_open_tasks": self.min_open_tasks,
                "max_open_tasks": self.max_open_tasks,
                "max_findings_per_run": self.max_findings_per_run,
                "timeout_seconds": self.timeout_seconds,
                "cooldown_seconds": self.cooldown_seconds,
                "required_exhaustion_members": self.required_exhaustion_members,
                "analyzer_id": self.analyzer_id,
                "expected_analyzer_version": self.expected_analyzer_version,
                "root_goal_id": self.root_goal_id,
                "runtime_parent_goal_id": self.runtime_parent_goal_id,
                "board_namespace": self.board_namespace,
                "default_goal_id": self.default_goal_id,
            }
        )


@dataclass(frozen=True, slots=True)
class RuntimeContractAssuranceRefillRequest:
    """Bounded request passed to the runtime analyzer callback."""

    cycle_id: str
    snapshot_id: str
    repository_id: str
    tree_id: str
    objective_revision: str
    configuration_revision: str
    current_open_tasks: int
    max_findings: int
    deadline_epoch: float
    changed_inputs: tuple[dict[str, str], ...] = ()
    affected_obligation_ids: tuple[str, ...] = ()
    affected_receipt_ids: tuple[str, ...] = ()
    change_digest: str = ""
    scan_mode: str = ScanMode.LOW_BACKLOG.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "configuration_revision": self.configuration_revision,
            "current_open_tasks": self.current_open_tasks,
            "max_findings": self.max_findings,
            "deadline_epoch": self.deadline_epoch,
            "changed_inputs": [dict(item) for item in self.changed_inputs],
            "affected_obligation_ids": list(self.affected_obligation_ids),
            "affected_receipt_ids": list(self.affected_receipt_ids),
            "change_digest": self.change_digest,
            "scan_mode": self.scan_mode,
            "provider_call_count": 0,
            "model_call_count": 0,
            "llm_call_count": 0,
        }


@dataclass(frozen=True, slots=True)
class RuntimeContractAssuranceRefillOutcome:
    """One complete runtime refill decision, including typed scan receipt."""

    reason: RuntimeContractAssuranceRefillReason
    scan_result: RefillScanResult[Mapping[str, Any]]
    tasks: tuple[ContractRepairTask, ...] = ()
    decisions: tuple[RuntimeContractMismatchRefineryDecision, ...] = ()
    board_markdown: str = ""
    quorum: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    replayed: bool = False
    recovered_state: bool = False
    completion_authoritative: bool = False
    affected_obligation_ids: tuple[str, ...] = ()
    affected_receipt_ids: tuple[str, ...] = ()
    changed_inputs: tuple[dict[str, str], ...] = ()
    provider_call_count: int = 0
    model_call_count: int = 0
    llm_call_count: int = 0
    analyzer_call_count: int = 0

    def __post_init__(self) -> None:
        reason = (
            self.reason
            if isinstance(self.reason, RuntimeContractAssuranceRefillReason)
            else RuntimeContractAssuranceRefillReason(str(self.reason))
        )
        object.__setattr__(self, "reason", reason)
        if self.completion_authoritative is not False:
            raise RuntimeContractAssuranceRefillError(
                "refill outcomes cannot grant completion authority"
            )
        for name in (
            "provider_call_count",
            "model_call_count",
            "llm_call_count",
            "analyzer_call_count",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                _bounded_integer(value, name, minimum=0, maximum=10_000),
            )

    @property
    def generated_count(self) -> int:
        return len(self.tasks)

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.scan_result.safe_for_completion_reasoning

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE,
            "reason": self.reason.value,
            "scan_result": self.scan_result.to_dict(),
            "tasks": [item.to_dict() for item in self.tasks],
            "decisions": [
                {
                    "finding_id": item.finding_id,
                    "task_id": item.task_id,
                    "reason_code": item.reason_code.value,
                    "detail": item.detail,
                }
                for item in self.decisions
            ],
            "board_markdown": self.board_markdown,
            "quorum": _canonical_value(self.quorum, "quorum"),
            "reason_codes": list(self.reason_codes),
            "replayed": self.replayed,
            "recovered_state": self.recovered_state,
            "completion_authoritative": False,
            "affected_obligation_ids": list(self.affected_obligation_ids),
            "affected_receipt_ids": list(self.affected_receipt_ids),
            "changed_inputs": [dict(item) for item in self.changed_inputs],
            "provider_call_count": self.provider_call_count,
            "model_call_count": self.model_call_count,
            "llm_call_count": self.llm_call_count,
            "analyzer_call_count": self.analyzer_call_count,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "RuntimeContractAssuranceRefillOutcome":
        scan = RefillScanResult.from_dict(value.get("scan_result") or {})
        decisions = tuple(
            RuntimeContractMismatchRefineryDecision(
                finding_id=item.get("finding_id", ""),
                task_id=item.get("task_id", ""),
                reason_code=RuntimeContractMismatchRefineryReason(
                    item.get("reason_code", "")
                ),
                detail=item.get("detail", ""),
            )
            for item in value.get("decisions", ())
        )
        return cls(
            reason=value.get("reason", ""),
            scan_result=scan,
            tasks=tuple(
                ContractRepairTask.from_dict(item) for item in value.get("tasks", ())
            ),
            decisions=decisions,
            board_markdown=value.get("board_markdown", ""),
            quorum=value.get("quorum") or {},
            reason_codes=tuple(value.get("reason_codes") or ()),
            replayed=value.get("replayed", False),
            recovered_state=value.get("recovered_state", False),
            completion_authoritative=value.get("completion_authoritative", False),
            affected_obligation_ids=tuple(
                value.get("affected_obligation_ids") or ()
            ),
            affected_receipt_ids=tuple(value.get("affected_receipt_ids") or ()),
            changed_inputs=tuple(value.get("changed_inputs") or ()),
            provider_call_count=value.get("provider_call_count", 0),
            model_call_count=value.get("model_call_count", 0),
            llm_call_count=value.get("llm_call_count", 0),
            analyzer_call_count=value.get("analyzer_call_count", 0),
        )


@dataclass(slots=True)
class _DurableState:
    board_markdown: str = ""
    last_refill_epoch: int = 0
    last_snapshot_id: str = ""
    last_change_digest: str = ""
    replay_records: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    exhaustion_receipts: list[Mapping[str, Any]] = field(default_factory=list)
    sequence: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA,
            "version": RUNTIME_CONTRACT_ASSURANCE_REFILL_STATE_VERSION,
            "sequence": self.sequence,
            "board_markdown": self.board_markdown,
            "last_refill_epoch": self.last_refill_epoch,
            "last_snapshot_id": self.last_snapshot_id,
            "last_change_digest": self.last_change_digest,
            "replay_records": dict(self.replay_records),
            "exhaustion_receipts": list(self.exhaustion_receipts),
        }

    def envelope(self) -> dict[str, Any]:
        payload = self.payload()
        return {**payload, "state_digest": _digest(payload)}

    @classmethod
    def from_envelope(cls, value: Mapping[str, Any]) -> "_DurableState":
        if value.get("schema") != RUNTIME_CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA:
            raise RuntimeContractAssuranceRefillError(
                "unsupported durable state schema"
            )
        if value.get("version") != RUNTIME_CONTRACT_ASSURANCE_REFILL_STATE_VERSION:
            raise RuntimeContractAssuranceRefillError(
                "unsupported durable state version"
            )
        payload = dict(value)
        declared = payload.pop("state_digest", "")
        if not declared or declared != _digest(payload):
            raise RuntimeContractAssuranceRefillError("durable state checksum mismatch")
        board_value = value.get("board_markdown", "")
        if not isinstance(board_value, str) or "\x00" in board_value:
            raise RuntimeContractAssuranceRefillError("board_markdown is malformed")
        board = board_value
        if board:
            namespace_match = re.search(
                r"(?m)^- Board namespace: (?P<namespace>.+)$", board
            )
            parse_contract_repair_board(
                board,
                board_namespace=(
                    namespace_match.group("namespace")
                    if namespace_match is not None
                    else "swissknife-symbolic-contract-assurance-v1"
                ),
            )
        replay = value.get("replay_records") or {}
        receipts = value.get("exhaustion_receipts") or ()
        if not isinstance(replay, Mapping) or not isinstance(receipts, Sequence):
            raise RuntimeContractAssuranceRefillError(
                "durable state collections are malformed"
            )
        normalized_replay: dict[str, Mapping[str, Any]] = {}
        for key, outcome in replay.items():
            selected_key = _text(key, "replay_key")
            if not isinstance(outcome, Mapping):
                raise RuntimeContractAssuranceRefillError("replay outcome is malformed")
            RuntimeContractAssuranceRefillOutcome.from_dict(outcome)
            normalized_replay[selected_key] = dict(outcome)
        normalized_receipts = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise RuntimeContractAssuranceRefillError(
                    "exhaustion receipt is malformed"
                )
            RefillScanResult.from_dict(receipt)
            normalized_receipts.append(dict(receipt))
        return cls(
            board_markdown=board,
            last_refill_epoch=_bounded_integer(
                value.get("last_refill_epoch", 0),
                "last_refill_epoch",
                minimum=0,
                maximum=2**63 - 1,
            ),
            last_snapshot_id=_text(
                value.get("last_snapshot_id", ""),
                "last_snapshot_id",
                required=False,
            ),
            last_change_digest=_text(
                value.get("last_change_digest", ""),
                "last_change_digest",
                required=False,
            ),
            replay_records=normalized_replay,
            exhaustion_receipts=normalized_receipts,
            sequence=_bounded_integer(
                value.get("sequence", 0),
                "sequence",
                minimum=0,
                maximum=2**63 - 1,
            ),
        )


AnalyzerCallback = Callable[
    [RuntimeContractAssuranceRefillRequest],
    ContractAssuranceAnalysis | Mapping[str, Any],
]


def _coerce_proof_scope_index(
    value: ProofScopeIndex | Mapping[str, Any] | None,
) -> ProofScopeIndex | None:
    if value is None:
        return None
    if isinstance(value, ProofScopeIndex):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeContractAssuranceRefillError(
            "proof_scope_index must be ProofScopeIndex or an object"
        )
    # Prefer full index payload; fall back to builder inputs.
    if "scope_blobs" in value or "obligations" in value:
        return build_proof_scope_index(
            scope_blobs=value.get("scope_blobs", ()),
            obligations=value.get("obligations", ()),
            receipts=value.get("receipts", ()),
            dependency_artifacts=value.get("artifacts", ()),
            root_id=str(value.get("root_id", "") or ""),
        )
    return ProofScopeIndex.from_dict(value)


def _packet_obligation_ids(packet: McpContractEditPacket) -> set[str]:
    ids = set(packet.obligation_ids or ())
    # Some packets nest obligations under the bounded slice.
    slice_payload = getattr(packet, "bounded_contract_slice", None) or {}
    if isinstance(slice_payload, Mapping):
        nested = slice_payload.get("obligation_ids") or ()
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            ids.update(str(item) for item in nested if item)
    return ids


def _finding_intersects_dependents(
    finding: ContractAssuranceFinding,
    affected_obligation_ids: set[str],
) -> bool:
    if not affected_obligation_ids:
        return True
    return bool(_packet_obligation_ids(finding.packet) & affected_obligation_ids)


class RuntimeContractAssuranceRefill:
    """Lease-serialized, restart-idempotent runtime contract refill handler."""

    interface: Final = RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE

    def __init__(
        self,
        analyzer: AnalyzerCallback,
        *,
        state_path: Path | str,
        policy: RuntimeContractAssuranceRefillPolicy | None = None,
        clock: Callable[[], float] = time.time,
        proof_scope_index: ProofScopeIndex | Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(analyzer):
            raise RuntimeContractAssuranceRefillError("analyzer must be callable")
        self.analyzer = analyzer
        self.state_path = Path(state_path)
        self.backup_path = self.state_path.with_name(self.state_path.name + ".bak")
        self.lock_path = self.state_path.with_name(self.state_path.name + ".lock")
        self.policy = policy or RuntimeContractAssuranceRefillPolicy()
        self.clock = clock
        self._default_proof_scope = proof_scope_index

    def _atomic_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    payload,
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _read_state_file(self, path: Path) -> _DurableState:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeContractAssuranceRefillError("durable state must be an object")
        return _DurableState.from_envelope(payload)

    def _quarantine(self, path: Path) -> None:
        if not path.exists():
            return
        digest = sha256(path.read_bytes()).hexdigest()[:12]
        quarantine = path.with_name(f"{path.name}.corrupt-{digest}")
        if quarantine.exists():
            quarantine = path.with_name(
                f"{path.name}.corrupt-{digest}-{time.time_ns()}"
            )
        os.replace(path, quarantine)

    def _load_state(self) -> tuple[_DurableState, bool]:
        if not self.state_path.exists():
            if self.backup_path.exists():
                try:
                    recovered = self._read_state_file(self.backup_path)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    self._quarantine(self.backup_path)
                    return _DurableState(), True
                self._atomic_write(self.state_path, recovered.envelope())
                return recovered, True
            return _DurableState(), False
        try:
            return self._read_state_file(self.state_path), False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._quarantine(self.state_path)
            try:
                recovered = self._read_state_file(self.backup_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self._quarantine(self.backup_path)
                return _DurableState(), True
            self._atomic_write(self.state_path, recovered.envelope())
            return recovered, True

    def _save_state(self, state: _DurableState) -> None:
        if self.state_path.exists():
            try:
                self._read_state_file(self.state_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self._quarantine(self.state_path)
            else:
                self.backup_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{self.backup_path.name}.",
                    suffix=".tmp",
                    dir=self.backup_path.parent,
                )
                os.close(descriptor)
                try:
                    shutil.copyfile(self.state_path, temporary_name)
                    with open(temporary_name, "rb") as stream:
                        os.fsync(stream.fileno())
                    os.replace(temporary_name, self.backup_path)
                except BaseException:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
                    raise
        state.sequence += 1
        self._atomic_write(self.state_path, state.envelope())
        if not self.backup_path.exists():
            self._atomic_write(self.backup_path, state.envelope())

    def _scan_result(
        self,
        reason: RuntimeContractAssuranceRefillReason,
        *,
        request: RuntimeContractAssuranceRefillRequest,
        started_at: datetime,
        finished_at: datetime,
        tasks: Sequence[ContractRepairTask] = (),
        metadata: Mapping[str, Any] | None = None,
        error: str | None = None,
        completion_safe: bool = False,
        analyzer_version: str = "",
        scan_mode: str | None = None,
    ) -> RefillScanResult[Mapping[str, Any]]:
        terminal = {
            RuntimeContractAssuranceRefillReason.GENERATED: (
                ScanTerminalReason.GENERATED
            ),
            RuntimeContractAssuranceRefillReason.DUPLICATE_ONLY: (
                ScanTerminalReason.DUPLICATE_ONLY
            ),
            RuntimeContractAssuranceRefillReason.EXHAUSTED: (
                ScanTerminalReason.EXHAUSTED
            ),
            RuntimeContractAssuranceRefillReason.THRESHOLD_SATISFIED: (
                ScanTerminalReason.THRESHOLD_SATISFIED
            ),
            RuntimeContractAssuranceRefillReason.COOLDOWN: ScanTerminalReason.COOLDOWN,
            RuntimeContractAssuranceRefillReason.NOOP: (
                ScanTerminalReason.THRESHOLD_SATISFIED
            ),
            RuntimeContractAssuranceRefillReason.TIMED_OUT: ScanTerminalReason.TIMED_OUT,
            RuntimeContractAssuranceRefillReason.ANALYZER_FAILED: (
                ScanTerminalReason.FAILED
            ),
        }.get(reason, ScanTerminalReason.PARTIAL)
        mode = scan_mode or request.scan_mode or ScanMode.LOW_BACKLOG.value
        return RefillScanResult(
            terminal_reason=terminal,
            scan_mode=mode,
            analyzer_version=analyzer_version
            or self.policy.expected_analyzer_version
            or self.policy.analyzer_id,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            started_at=started_at,
            finished_at=finished_at,
            items=tuple(item.to_dict() for item in tasks),
            safe_for_completion_reasoning=completion_safe,
            error=error,
            metadata={
                "interface": RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE,
                "reason_code": reason.value,
                "snapshot_id": request.snapshot_id,
                "objective_revision": request.objective_revision,
                "configuration_revision": request.configuration_revision,
                "completion_authoritative": False,
                "provider_call_count": 0,
                "model_call_count": 0,
                "llm_call_count": 0,
                "changed_inputs": [dict(item) for item in request.changed_inputs],
                "affected_obligation_ids": list(request.affected_obligation_ids),
                "affected_receipt_ids": list(request.affected_receipt_ids),
                **dict(metadata or {}),
            },
        )

    def _outcome(
        self,
        reason: RuntimeContractAssuranceRefillReason,
        *,
        request: RuntimeContractAssuranceRefillRequest,
        started_at: datetime,
        finished_at: datetime,
        state: _DurableState,
        tasks: Sequence[ContractRepairTask] = (),
        decisions: Sequence[RuntimeContractMismatchRefineryDecision] = (),
        quorum: Mapping[str, Any] | None = None,
        reason_codes: Iterable[str] = (),
        error: str | None = None,
        completion_safe: bool = False,
        recovered_state: bool = False,
        metadata: Mapping[str, Any] | None = None,
        analyzer_version: str = "",
        analyzer_call_count: int = 0,
        scan_mode: str | None = None,
    ) -> RuntimeContractAssuranceRefillOutcome:
        return RuntimeContractAssuranceRefillOutcome(
            reason=reason,
            scan_result=self._scan_result(
                reason,
                request=request,
                started_at=started_at,
                finished_at=finished_at,
                tasks=tasks,
                metadata=metadata,
                error=error,
                completion_safe=completion_safe,
                analyzer_version=analyzer_version,
                scan_mode=scan_mode,
            ),
            tasks=tuple(tasks),
            decisions=tuple(decisions),
            board_markdown=state.board_markdown,
            quorum=dict(quorum or {}),
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            recovered_state=recovered_state,
            completion_authoritative=False,
            affected_obligation_ids=request.affected_obligation_ids,
            affected_receipt_ids=request.affected_receipt_ids,
            changed_inputs=request.changed_inputs,
            provider_call_count=0,
            model_call_count=0,
            llm_call_count=0,
            analyzer_call_count=analyzer_call_count,
        )

    def _invoke_analyzer(
        self, request: RuntimeContractAssuranceRefillRequest
    ) -> ContractAssuranceAnalysis:
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="runtime-contract-assurance-refill"
        )
        future = executor.submit(self.analyzer, request)
        try:
            value = future.result(timeout=self.policy.timeout_seconds)
        except FutureTimeout:
            future.cancel()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return ContractAssuranceAnalysis.from_value(value)

    def _capability_reason(
        self,
        analysis: ContractAssuranceAnalysis,
        request: RuntimeContractAssuranceRefillRequest,
    ) -> RuntimeContractAssuranceRefillReason | None:
        capability = analysis.capability
        if not capability.available or not capability.supported_claim_families:
            return RuntimeContractAssuranceRefillReason.CAPABILITY_MISSING
        expected_version = self.policy.expected_analyzer_version
        if (
            analysis.snapshot_id != request.snapshot_id
            or analysis.repository_id != request.repository_id
            or analysis.tree_id != request.tree_id
            or capability.snapshot_id != request.snapshot_id
            or capability.repository_id != request.repository_id
            or capability.tree_id != request.tree_id
            or capability.analyzer_id != self.policy.analyzer_id
            or capability.analyzer_version != analysis.analyzer_version
            or (expected_version and analysis.analyzer_version != expected_version)
        ):
            return RuntimeContractAssuranceRefillReason.CAPABILITY_STALE
        return None

    @staticmethod
    def _health_reason(
        analysis: ContractAssuranceAnalysis,
    ) -> RuntimeContractAssuranceRefillReason | None:
        health = analysis.analyzer_health
        if (
            health.get("schema") != ANALYZER_HEALTH_SCHEMA
            or health.get("status") != "healthy"
            or health.get("healthy") is not True
            or health.get("safe_for_completion_reasoning") is not True
        ):
            return RuntimeContractAssuranceRefillReason.ANALYZER_UNHEALTHY
        canaries = analysis.canary_report
        fixture_count = canaries.get("fixture_count", 0)
        if isinstance(fixture_count, bool):
            fixture_count = 0
        try:
            fixture_count = int(fixture_count or 0)
        except (TypeError, ValueError, OverflowError):
            fixture_count = 0
        if (
            canaries.get("schema") != ANALYZER_CANARY_SCHEMA
            or canaries.get("passed") is not True
            or canaries.get("registry_present") is not True
            or canaries.get("registry_errors")
            or fixture_count < 1
            or canaries.get("analyzer_version") != analysis.analyzer_version
        ):
            return RuntimeContractAssuranceRefillReason.CANARIES_FAILED
        return None

    @staticmethod
    def _coverage_mapping_is_complete(
        coverage: Mapping[str, Any], *, declared_complete: bool
    ) -> bool:
        if not declared_complete:
            return False

        def count(*names: str) -> int:
            for name in names:
                if name in coverage:
                    value = coverage.get(name)
                    if isinstance(value, bool):
                        return -1
                    try:
                        return int(value)
                    except (TypeError, ValueError, OverflowError):
                        return -1
            return -1

        tracked = count("tracked_file_count", "tracked_files", "coverage_total")
        eligible = count("eligible_file_count", "eligible_files")
        excluded = count("excluded_file_count", "excluded_files")
        parsed = count("parsed_file_count", "parsed_files")
        cached = count("cache_hit_count", "cache_hits")
        failures = count("parser_failure_count", "parser_failures")
        disposed = count(
            "coverage_disposition_count", "disposed_file_count", "coverage_disposed"
        )
        if tracked <= 0 or failures != 0:
            return False
        if disposed >= 0:
            return disposed == tracked
        return (
            min(eligible, excluded, parsed, cached) >= 0
            and eligible + excluded == tracked
            and parsed + cached == eligible
        )

    @classmethod
    def _coverage_is_complete(cls, analysis: ContractAssuranceAnalysis) -> bool:
        return cls._coverage_mapping_is_complete(
            analysis.coverage,
            declared_complete=analysis.coverage_complete,
        )

    def _receipt_is_assured(
        self,
        receipt: RefillScanResult[Any] | Mapping[str, Any],
        *,
        request: RuntimeContractAssuranceRefillRequest,
        analyzer_version: str,
    ) -> bool:
        try:
            scan = (
                receipt
                if isinstance(receipt, RefillScanResult)
                else RefillScanResult.from_dict(receipt)
            )
        except (TypeError, ValueError):
            return False
        metadata = scan.metadata
        canaries = metadata.get("canary_report")
        capability = metadata.get("capability")
        coverage = metadata.get("coverage")
        if not isinstance(canaries, Mapping) or not isinstance(capability, Mapping):
            return False
        if not isinstance(coverage, Mapping):
            return False
        try:
            parsed_capability = ContractAnalyzerCapability.from_value(capability)
        except (TypeError, ValueError):
            return False
        fixture_count = canaries.get("fixture_count", 0)
        if isinstance(fixture_count, bool):
            fixture_count = 0
        try:
            fixture_count = int(fixture_count or 0)
        except (TypeError, ValueError, OverflowError):
            fixture_count = 0
        return (
            scan.terminal_reason is ScanTerminalReason.EXHAUSTED
            and scan.safe_for_completion_reasoning
            and scan.repository_id == request.repository_id
            and scan.tree_id == request.tree_id
            and scan.analyzer_version == analyzer_version
            and metadata.get("snapshot_id") == request.snapshot_id
            and metadata.get("configuration_revision")
            == request.configuration_revision
            and metadata.get("objective_revision") == request.objective_revision
            and canaries.get("schema") == ANALYZER_CANARY_SCHEMA
            and canaries.get("passed") is True
            and canaries.get("registry_present") is True
            and not canaries.get("registry_errors")
            and fixture_count > 0
            and canaries.get("analyzer_version") == analyzer_version
            and parsed_capability.available
            and parsed_capability.analyzer_id == self.policy.analyzer_id
            and parsed_capability.analyzer_version == analyzer_version
            and parsed_capability.repository_id == request.repository_id
            and parsed_capability.tree_id == request.tree_id
            and parsed_capability.snapshot_id == request.snapshot_id
            and self._coverage_mapping_is_complete(
                coverage,
                declared_complete=metadata.get("coverage_complete") is True,
            )
        )

    def _validate_lineages(
        self,
        findings: Sequence[ContractAssuranceFinding],
        request: RuntimeContractAssuranceRefillRequest,
    ) -> tuple[ContractAssuranceFinding, ...]:
        admitted: list[ContractAssuranceFinding] = []
        runtime_parent = self.policy.runtime_parent_goal_id
        allowed_subgoals = {
            runtime_parent,
            *RUNTIME_COMPONENT_SUBGOALS.values(),
        }
        for finding in sorted(findings, key=lambda item: item.finding_id):
            lineage = finding.goal_lineage
            if (
                lineage.root_goal_id != self.policy.root_goal_id
                or lineage.ancestor_goal_ids[0] != self.policy.root_goal_id
                or lineage.objective_revision != request.objective_revision
            ):
                raise RuntimeContractAssuranceRefillError(
                    f"finding {finding.finding_id} has no current goal lineage"
                )
            # Correct-subgoal gate: the finding goal must be a known runtime
            # subgoal, and the lineage must mention the runtime parent or be it.
            if lineage.goal_id not in allowed_subgoals:
                raise RuntimeContractAssuranceRefillError(
                    f"finding {finding.finding_id} targets non-runtime subgoal "
                    f"{lineage.goal_id}"
                )
            if (
                lineage.goal_id != runtime_parent
                and runtime_parent not in lineage.ancestor_goal_ids
                and lineage.goal_id not in RUNTIME_COMPONENT_SUBGOALS.values()
            ):
                raise RuntimeContractAssuranceRefillError(
                    f"finding {finding.finding_id} is not backed by "
                    f"{runtime_parent}"
                )
            # Component catalog goals (SCA-G171–G175) are valid refill targets
            # even when the runtime parent is not listed as a direct ancestor,
            # provided the root is SCA-G000.
            finding.packet.assert_current(request.snapshot_id)
            admitted.append(finding)
        return tuple(admitted)

    def _exhaustion_receipt(
        self,
        analysis: ContractAssuranceAnalysis,
        request: RuntimeContractAssuranceRefillRequest,
        started_at: datetime,
        finished_at: datetime,
    ) -> RefillScanResult[Mapping[str, Any]]:
        return RefillScanResult(
            terminal_reason=ScanTerminalReason.EXHAUSTED,
            scan_mode=ScanMode.EXHAUSTIVE.value,
            analyzer_version=analysis.analyzer_version,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            started_at=started_at,
            finished_at=finished_at,
            safe_for_completion_reasoning=True,
            metadata={
                "snapshot_id": request.snapshot_id,
                "exhaustive": True,
                "coverage_complete": True,
                "coverage": analysis.coverage,
                "health": "healthy",
                "analyzer_health": analysis.analyzer_health,
                "canary_report": analysis.canary_report,
                "capability": analysis.capability.to_dict(),
                "evidence_channel": analysis.evidence_channel,
                "configuration_revision": request.configuration_revision,
                "objective_revision": request.objective_revision,
                "provider_call_count": 0,
                "model_call_count": 0,
                "llm_call_count": 0,
            },
        )

    def _binding(
        self,
        analysis: ContractAssuranceAnalysis,
        request: RuntimeContractAssuranceRefillRequest,
    ) -> ExhaustionBinding:
        return ExhaustionBinding(
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            analyzer_version=analysis.analyzer_version,
            configuration_revision=request.configuration_revision,
            objective_revision=request.objective_revision,
        )

    @staticmethod
    def _trim_state(state: _DurableState) -> None:
        if len(state.replay_records) > MAX_REPLAY_RECORDS:
            overflow = len(state.replay_records) - MAX_REPLAY_RECORDS
            for key in tuple(state.replay_records)[:overflow]:
                state.replay_records.pop(key, None)
        if len(state.exhaustion_receipts) > MAX_EXHAUSTION_RECEIPTS:
            state.exhaustion_receipts[:] = state.exhaustion_receipts[
                -MAX_EXHAUSTION_RECEIPTS:
            ]

    def _invalidate_dependents(
        self,
        *,
        changed_inputs: Sequence[dict[str, str]],
        proof_scope_index: ProofScopeIndex | Mapping[str, Any] | None,
        source_tree: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], Mapping[str, Any] | None]:
        if not changed_inputs:
            return (), (), None
        index = _coerce_proof_scope_index(
            proof_scope_index if proof_scope_index is not None else self._default_proof_scope
        )
        if index is None:
            # Without a proof scope, change-driven scans still proceed but
            # cannot prove exact dependent closure.
            return (), (), None
        result = invalidate_proof_evidence(
            index,
            list(changed_inputs),
            source_tree=source_tree or "",
        )
        event = result.event
        event_payload = event.to_dict() if hasattr(event, "to_dict") else None
        return (
            tuple(event.affected_obligation_ids),
            tuple(event.affected_receipt_ids),
            event_payload if isinstance(event_payload, Mapping) else None,
        )

    def _refine_by_subgoal(
        self,
        findings: Sequence[ContractAssuranceFinding],
        *,
        snapshot: str,
        board_markdown: str,
        open_tasks: int,
        epoch: int,
        current_finding_record_ids: Mapping[str, str],
    ) -> RuntimeContractMismatchRefineryResult:
        """Project findings grouped by correct runtime subgoal."""

        groups: dict[str, list[McpContractEditPacket]] = {}
        for finding in findings:
            subgoal = resolve_runtime_subgoal(
                goal_id=finding.goal_lineage.goal_id,
                default=self.policy.default_goal_id,
            )
            groups.setdefault(subgoal, []).append(finding.packet)

        if not groups:
            return RuntimeContractMismatchRefinery(
                RuntimeContractMismatchRefineryPolicy(
                    max_open_work=self.policy.max_open_tasks,
                    max_findings_per_run=self.policy.max_findings_per_run,
                    cooldown_seconds=0,
                    board_namespace=self.policy.board_namespace,
                    goal_id=self.policy.default_goal_id,
                )
            ).refine(
                (),
                current_snapshot_id=snapshot,
                existing_board=board_markdown,
                current_open_work=open_tasks,
                now_epoch=epoch,
                current_finding_record_ids=current_finding_record_ids,
            )

        markdown = board_markdown
        all_decisions: list[RuntimeContractMismatchRefineryDecision] = []
        all_tasks: list[ContractRepairTask] = []
        initial_open = open_tasks
        final_open = open_tasks
        last_epoch = epoch
        remaining_slots = self.policy.max_findings_per_run

        for subgoal in sorted(groups):
            packets = groups[subgoal]
            if remaining_slots <= 0:
                for packet in packets:
                    all_decisions.append(
                        RuntimeContractMismatchRefineryDecision(
                            finding_id=packet.finding_id,
                            task_id="",
                            reason_code=RuntimeContractMismatchRefineryReason.FINDING_LIMIT,
                            detail="runtime refill finding limit reached",
                        )
                    )
                continue
            selected = packets[:remaining_slots]
            skipped = packets[remaining_slots:]
            refined = RuntimeContractMismatchRefinery(
                RuntimeContractMismatchRefineryPolicy(
                    max_open_work=self.policy.max_open_tasks,
                    max_findings_per_run=remaining_slots,
                    cooldown_seconds=0,
                    board_namespace=self.policy.board_namespace,
                    goal_id=subgoal,
                )
            ).refine(
                selected,
                current_snapshot_id=snapshot,
                existing_board=markdown,
                current_open_work=final_open,
                now_epoch=epoch,
                current_finding_record_ids=current_finding_record_ids,
            )
            markdown = refined.markdown
            all_decisions.extend(refined.decisions)
            all_tasks = list(refined.tasks)
            initial_open = max(initial_open, refined.initial_open_work)
            final_open = refined.final_open_work
            last_epoch = refined.last_refinery_epoch
            emitted = sum(
                1
                for item in refined.decisions
                if item.reason_code is RuntimeContractMismatchRefineryReason.EMITTED
            )
            remaining_slots = max(0, remaining_slots - max(emitted, len(selected)))
            for packet in skipped:
                all_decisions.append(
                    RuntimeContractMismatchRefineryDecision(
                        finding_id=packet.finding_id,
                        task_id="",
                        reason_code=RuntimeContractMismatchRefineryReason.FINDING_LIMIT,
                        detail="runtime refill finding limit reached",
                    )
                )

        return RuntimeContractMismatchRefineryResult(
            tasks=tuple(sorted(all_tasks, key=lambda item: item.task_id)),
            decisions=tuple(all_decisions),
            markdown=markdown,
            initial_open_work=initial_open,
            final_open_work=final_open,
            max_open_work=self.policy.max_open_tasks,
            last_refinery_epoch=last_epoch,
        )

    def refill(
        self,
        *,
        current_open_tasks: int,
        snapshot_id: str,
        repository_id: str,
        tree_id: str,
        objective_revision: str,
        idempotency_key: str = "",
        now_epoch: int | None = None,
        changed_inputs: Sequence[Mapping[str, Any] | str] | None = None,
        proof_scope_index: ProofScopeIndex | Mapping[str, Any] | None = None,
        force_scan: bool = False,
    ) -> RuntimeContractAssuranceRefillOutcome:
        """Run one serialized continuous runtime refill transaction."""

        open_tasks = _bounded_integer(
            current_open_tasks,
            "current_open_tasks",
            minimum=0,
            maximum=10_000,
        )
        snapshot = _text(snapshot_id, "snapshot_id")
        repository = _text(repository_id, "repository_id")
        tree = _text(tree_id, "tree_id")
        objective = _text(objective_revision, "objective_revision")
        replay_key = _text(idempotency_key, "idempotency_key", required=False)
        epoch = (
            int(self.clock())
            if now_epoch is None
            else _bounded_integer(
                now_epoch, "now_epoch", minimum=0, maximum=2**63 - 1
            )
        )
        changes = normalize_changed_inputs(changed_inputs)
        change_digest = _digest(list(changes)) if changes else ""
        affected_obligations, affected_receipts, invalidation_event = (
            self._invalidate_dependents(
                changed_inputs=changes,
                proof_scope_index=proof_scope_index,
                source_tree=tree,
            )
        )
        scan_mode = (
            ScanMode.INCREMENTAL.value if changes else ScanMode.LOW_BACKLOG.value
        )
        cycle_id = replay_key or _digest(
            {
                "snapshot_id": snapshot,
                "repository_id": repository,
                "tree_id": tree,
                "objective_revision": objective,
                "open_tasks": open_tasks,
                "epoch": epoch,
                "change_digest": change_digest,
            }
        )
        request = RuntimeContractAssuranceRefillRequest(
            cycle_id=cycle_id,
            snapshot_id=snapshot,
            repository_id=repository,
            tree_id=tree,
            objective_revision=objective,
            configuration_revision=self.policy.configuration_revision,
            current_open_tasks=open_tasks,
            max_findings=self.policy.max_findings_per_run,
            deadline_epoch=float(epoch) + self.policy.timeout_seconds,
            changed_inputs=changes,
            affected_obligation_ids=affected_obligations,
            affected_receipt_ids=affected_receipts,
            change_digest=change_digest,
            scan_mode=scan_mode,
        )
        replay_record_key = (
            _digest(
                {
                    "idempotency_key": replay_key,
                    "snapshot_id": snapshot,
                    "repository_id": repository,
                    "tree_id": tree,
                    "objective_revision": objective,
                    "configuration_revision": request.configuration_revision,
                    "current_open_tasks": open_tasks,
                    "change_digest": change_digest,
                }
            )
            if replay_key
            else ""
        )
        started = _iso_at(epoch)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state, recovered = self._load_state()
            if replay_record_key and replay_record_key in state.replay_records:
                replay = RuntimeContractAssuranceRefillOutcome.from_dict(
                    state.replay_records[replay_record_key]
                )
                return replace(
                    replay,
                    replayed=True,
                    recovered_state=recovered or replay.recovered_state,
                    reason_codes=tuple(
                        dict.fromkeys(
                            (
                                *replay.reason_codes,
                                *(
                                    (
                                        RuntimeContractAssuranceRefillReason.STATE_RECOVERED.value,
                                    )
                                    if recovered
                                    else ()
                                ),
                            )
                        )
                    ),
                )

            def finish(
                outcome: RuntimeContractAssuranceRefillOutcome,
                *,
                scanned: bool = False,
            ) -> RuntimeContractAssuranceRefillOutcome:
                final = replace(
                    outcome,
                    recovered_state=recovered,
                    reason_codes=tuple(
                        dict.fromkeys(
                            (
                                *outcome.reason_codes,
                                *(
                                    (
                                        RuntimeContractAssuranceRefillReason.STATE_RECOVERED.value,
                                    )
                                    if recovered
                                    else ()
                                ),
                            )
                        )
                    ),
                )
                if scanned:
                    state.last_refill_epoch = epoch
                    state.last_snapshot_id = snapshot
                    state.last_change_digest = change_digest
                if replay_record_key:
                    state.replay_records[replay_record_key] = final.to_dict()
                self._trim_state(state)
                self._save_state(state)
                return final

            # Reconcile persisted tasks against the requested snapshot even
            # when the backlog threshold suppresses a new analyzer call.
            if state.board_markdown:
                reconciled = RuntimeContractMismatchRefinery(
                    RuntimeContractMismatchRefineryPolicy(
                        max_open_work=self.policy.max_open_tasks,
                        max_findings_per_run=self.policy.max_findings_per_run,
                        cooldown_seconds=0,
                        board_namespace=self.policy.board_namespace,
                        goal_id=self.policy.default_goal_id,
                    )
                ).refine(
                    (),
                    current_snapshot_id=snapshot,
                    existing_board=state.board_markdown,
                    current_open_work=open_tasks,
                    now_epoch=epoch,
                )
                state.board_markdown = reconciled.markdown

            # --- No-op / threshold / cooldown (zero analyzer, zero providers)
            if not changes and not force_scan:
                if open_tasks >= self.policy.min_open_tasks:
                    outcome = self._outcome(
                        RuntimeContractAssuranceRefillReason.THRESHOLD_SATISFIED,
                        request=request,
                        started_at=started,
                        finished_at=started,
                        state=state,
                        recovered_state=recovered,
                        analyzer_call_count=0,
                    )
                    return finish(outcome)

                # True no-op: same snapshot, no semantic change, already scanned.
                if (
                    state.last_snapshot_id == snapshot
                    and state.last_refill_epoch > 0
                    and state.last_change_digest == ""
                ):
                    cooldown_active = (
                        self.policy.cooldown_seconds > 0
                        and (
                            epoch < state.last_refill_epoch
                            or epoch - state.last_refill_epoch
                            < self.policy.cooldown_seconds
                        )
                    )
                    if cooldown_active:
                        outcome = self._outcome(
                            RuntimeContractAssuranceRefillReason.COOLDOWN,
                            request=request,
                            started_at=started,
                            finished_at=started,
                            state=state,
                            reason_codes=(
                                RuntimeContractAssuranceRefillReason.COOLDOWN.value,
                            ),
                            analyzer_call_count=0,
                        )
                        return finish(outcome)
                    outcome = self._outcome(
                        RuntimeContractAssuranceRefillReason.NOOP,
                        request=request,
                        started_at=started,
                        finished_at=started,
                        state=state,
                        reason_codes=(
                            RuntimeContractAssuranceRefillReason.NOOP.value,
                        ),
                        analyzer_call_count=0,
                        scan_mode=ScanMode.INCREMENTAL.value,
                    )
                    return finish(outcome)

            # Change-driven cooldown: identical change digest on same snapshot.
            if changes and not force_scan:
                cooldown_active = (
                    self.policy.cooldown_seconds > 0
                    and state.last_snapshot_id == snapshot
                    and state.last_change_digest == change_digest
                    and state.last_refill_epoch > 0
                    and (
                        epoch < state.last_refill_epoch
                        or epoch - state.last_refill_epoch
                        < self.policy.cooldown_seconds
                    )
                )
                if cooldown_active:
                    outcome = self._outcome(
                        RuntimeContractAssuranceRefillReason.COOLDOWN,
                        request=request,
                        started_at=started,
                        finished_at=started,
                        state=state,
                        reason_codes=(
                            RuntimeContractAssuranceRefillReason.COOLDOWN.value,
                        ),
                        analyzer_call_count=0,
                        metadata={
                            "invalidation_event": invalidation_event,
                        },
                    )
                    return finish(outcome)

            # Low-backlog full scan still respects threshold when no changes.
            if (
                not changes
                and not force_scan
                and open_tasks >= self.policy.min_open_tasks
            ):
                outcome = self._outcome(
                    RuntimeContractAssuranceRefillReason.THRESHOLD_SATISFIED,
                    request=request,
                    started_at=started,
                    finished_at=started,
                    state=state,
                    analyzer_call_count=0,
                )
                return finish(outcome)

            if not changes and not force_scan:
                cooldown_active = (
                    self.policy.cooldown_seconds > 0
                    and state.last_snapshot_id == snapshot
                    and state.last_refill_epoch > 0
                    and (
                        epoch < state.last_refill_epoch
                        or epoch - state.last_refill_epoch
                        < self.policy.cooldown_seconds
                    )
                )
                if cooldown_active:
                    outcome = self._outcome(
                        RuntimeContractAssuranceRefillReason.COOLDOWN,
                        request=request,
                        started_at=started,
                        finished_at=started,
                        state=state,
                        reason_codes=(
                            RuntimeContractAssuranceRefillReason.COOLDOWN.value,
                        ),
                        analyzer_call_count=0,
                    )
                    return finish(outcome)

            try:
                analysis = self._invoke_analyzer(request)
                analyzer_calls = 1
            except FutureTimeout:
                finished = max(_utc_now(), started)
                outcome = self._outcome(
                    RuntimeContractAssuranceRefillReason.TIMED_OUT,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    error=(
                        "runtime contract analyzer exceeded "
                        f"{self.policy.timeout_seconds:g} seconds"
                    ),
                    reason_codes=(
                        RuntimeContractAssuranceRefillReason.TIMED_OUT.value,
                    ),
                    analyzer_call_count=1,
                )
                return finish(outcome, scanned=True)
            except Exception as exc:
                finished = max(_utc_now(), started)
                outcome = self._outcome(
                    RuntimeContractAssuranceRefillReason.ANALYZER_FAILED,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    error=f"{type(exc).__name__}: {exc}",
                    reason_codes=(
                        RuntimeContractAssuranceRefillReason.ANALYZER_FAILED.value,
                    ),
                    analyzer_call_count=1,
                )
                return finish(outcome, scanned=True)

            finished = max(_utc_now(), started)
            capability_reason = self._capability_reason(analysis, request)
            if capability_reason is not None:
                outcome = self._outcome(
                    capability_reason,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    reason_codes=(capability_reason.value,),
                    metadata={"capability": analysis.capability.to_dict()},
                    analyzer_version=analysis.analyzer_version,
                    analyzer_call_count=analyzer_calls,
                )
                return finish(outcome, scanned=True)
            health_reason = self._health_reason(analysis)
            if health_reason is not None:
                outcome = self._outcome(
                    health_reason,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    reason_codes=(health_reason.value,),
                    metadata={
                        "analyzer_health": analysis.analyzer_health,
                        "canary_report": analysis.canary_report,
                    },
                    analyzer_version=analysis.analyzer_version,
                    analyzer_call_count=analyzer_calls,
                )
                return finish(outcome, scanned=True)

            try:
                findings = self._validate_lineages(analysis.findings, request)
            except (ValueError, TypeError) as exc:
                outcome = self._outcome(
                    RuntimeContractAssuranceRefillReason.NO_GOAL_LINEAGE,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    reason_codes=(
                        RuntimeContractAssuranceRefillReason.NO_GOAL_LINEAGE.value,
                    ),
                    metadata={"lineage_error": str(exc)},
                    analyzer_version=analysis.analyzer_version,
                    analyzer_call_count=analyzer_calls,
                )
                return finish(outcome, scanned=True)

            affected_set = set(affected_obligations)
            if changes and affected_set:
                findings = tuple(
                    item
                    for item in findings
                    if _finding_intersects_dependents(item, affected_set)
                )
                dependents_filter = True
            else:
                dependents_filter = bool(changes)

            truncated = len(findings) > self.policy.max_findings_per_run
            selected = findings[: self.policy.max_findings_per_run]
            refined = self._refine_by_subgoal(
                selected,
                snapshot=snapshot,
                board_markdown=state.board_markdown,
                open_tasks=open_tasks,
                epoch=epoch,
                current_finding_record_ids=analysis.current_finding_record_ids,
            )
            state.board_markdown = refined.markdown
            emitted_ids = {
                decision.task_id
                for decision in refined.decisions
                if decision.reason_code is RuntimeContractMismatchRefineryReason.EMITTED
            }
            tasks = tuple(
                task for task in refined.tasks if task.task_id in emitted_ids
            )

            base_metadata: dict[str, Any] = {
                "capability": analysis.capability.to_dict(),
                "analyzer_health": analysis.analyzer_health,
                "canary_report": analysis.canary_report,
                "raw_finding_count": len(analysis.findings),
                "admitted_finding_count": len(selected),
                "invalidation_event": invalidation_event,
                "dependents_filter": dependents_filter,
                "provider_call_count": 0,
                "model_call_count": 0,
                "llm_call_count": 0,
            }

            if findings:
                reason_codes = [item.reason_code.value for item in refined.decisions]
                if truncated:
                    reason_codes.append(
                        RuntimeContractAssuranceRefillReason.FINDING_LIMIT.value
                    )
                if dependents_filter:
                    reason_codes.append(
                        RuntimeContractAssuranceRefillReason.DEPENDENTS_ONLY.value
                    )
                if refined.generated_count:
                    reason = RuntimeContractAssuranceRefillReason.GENERATED
                elif any(
                    item.reason_code
                    is RuntimeContractMismatchRefineryReason.OPEN_WORK_LIMIT
                    for item in refined.decisions
                ):
                    reason = RuntimeContractAssuranceRefillReason.OPEN_WORK_LIMIT
                else:
                    reason = RuntimeContractAssuranceRefillReason.DUPLICATE_ONLY
                outcome = self._outcome(
                    reason,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    tasks=(
                        tasks
                        if reason is RuntimeContractAssuranceRefillReason.GENERATED
                        else ()
                    ),
                    decisions=refined.decisions,
                    reason_codes=reason_codes,
                    metadata=base_metadata,
                    analyzer_version=analysis.analyzer_version,
                    analyzer_call_count=analyzer_calls,
                )
                return finish(outcome, scanned=True)

            if not analysis.exhaustive or not self._coverage_is_complete(analysis):
                outcome = self._outcome(
                    RuntimeContractAssuranceRefillReason.COVERAGE_INCOMPLETE,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    reason_codes=(
                        RuntimeContractAssuranceRefillReason.COVERAGE_INCOMPLETE.value,
                    ),
                    metadata={
                        **base_metadata,
                        "coverage": analysis.coverage,
                        "coverage_complete": analysis.coverage_complete,
                        "exhaustive": analysis.exhaustive,
                    },
                    analyzer_version=analysis.analyzer_version,
                    analyzer_call_count=analyzer_calls,
                )
                return finish(outcome, scanned=True)

            current_receipt = self._exhaustion_receipt(
                analysis, request, started, finished
            )
            binding = self._binding(analysis, request)
            raw_candidates: list[RefillScanResult[Any] | Mapping[str, Any]] = [
                *state.exhaustion_receipts,
                *analysis.exhaustion_receipts,
                current_receipt,
            ]
            candidates = [
                candidate
                for candidate in raw_candidates
                if self._receipt_is_assured(
                    candidate,
                    request=request,
                    analyzer_version=analysis.analyzer_version,
                )
            ]
            quorum = evaluate_exhaustion_quorum(
                candidates,
                binding=binding,
                required_members=self.policy.required_exhaustion_members,
            )
            eligible_cids = {item.receipt_cid for item in quorum.members}
            persisted: list[Mapping[str, Any]] = []
            for candidate in candidates:
                if isinstance(candidate, RefillScanResult):
                    mapping = candidate.to_dict()
                    cid = candidate.receipt_cid
                else:
                    mapping = dict(candidate)
                    try:
                        cid = RefillScanResult.from_dict(mapping).receipt_cid
                    except (TypeError, ValueError):
                        continue
                if cid in eligible_cids:
                    persisted.append(mapping)
            by_cid = {
                RefillScanResult.from_dict(item).receipt_cid: item
                for item in persisted
            }
            state.exhaustion_receipts = list(by_cid.values())
            quorum_record = quorum.to_dict()
            if not quorum.satisfied:
                outcome = self._outcome(
                    RuntimeContractAssuranceRefillReason.QUORUM_INCOMPLETE,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    quorum=quorum_record,
                    reason_codes=(
                        RuntimeContractAssuranceRefillReason.QUORUM_INCOMPLETE.value,
                    ),
                    metadata={
                        **base_metadata,
                        "coverage": analysis.coverage,
                        "coverage_complete": True,
                        "exhaustive": True,
                        "exhaustion_quorum": quorum_record,
                    },
                    analyzer_version=analysis.analyzer_version,
                    analyzer_call_count=analyzer_calls,
                )
                return finish(outcome, scanned=True)

            outcome = self._outcome(
                RuntimeContractAssuranceRefillReason.EXHAUSTED,
                request=request,
                started_at=started,
                finished_at=finished,
                state=state,
                quorum=quorum_record,
                reason_codes=(
                    RuntimeContractAssuranceRefillReason.EXHAUSTED.value,
                ),
                completion_safe=True,
                metadata={
                    **base_metadata,
                    "coverage": analysis.coverage,
                    "coverage_complete": True,
                    "exhaustive": True,
                    "exhaustion_quorum": quorum_record,
                },
                analyzer_version=analysis.analyzer_version,
                analyzer_call_count=analyzer_calls,
            )
            return finish(outcome, scanned=True)

    __call__ = refill
    run = refill


def run_runtime_contract_assurance_refill(
    analyzer: AnalyzerCallback,
    *,
    state_path: Path | str,
    policy: RuntimeContractAssuranceRefillPolicy | None = None,
    proof_scope_index: ProofScopeIndex | Mapping[str, Any] | None = None,
    **request: Any,
) -> RuntimeContractAssuranceRefillOutcome:
    """Functional supervisor handler entry point."""

    return RuntimeContractAssuranceRefill(
        analyzer,
        state_path=state_path,
        policy=policy,
        proof_scope_index=proof_scope_index,
    ).refill(**request)


def build_runtime_contract_assurance_refill_handler(
    analyzer: AnalyzerCallback,
    *,
    state_path: Path | str,
    policy: RuntimeContractAssuranceRefillPolicy | None = None,
    clock: Callable[[], float] = time.time,
    proof_scope_index: ProofScopeIndex | Mapping[str, Any] | None = None,
) -> RuntimeContractAssuranceRefill:
    """Return a callable suitable for objective/backlog refill registration."""

    return RuntimeContractAssuranceRefill(
        analyzer,
        state_path=state_path,
        policy=policy,
        clock=clock,
        proof_scope_index=proof_scope_index,
    )


def build_runtime_refill_metrics_report(
    metrics: Mapping[str, Any],
    *,
    task_id: str = "SCA-179",
    snapshot_id: str = "",
    source_tree: str = "",
    bounds: Mapping[str, Any] | None = None,
    evidence: str = "SCAEV176REFILL",
) -> dict[str, Any]:
    """Seal non-authoritative runtime refill metrics for durable publication."""

    payload: dict[str, Any] = {
        "schema": RUNTIME_CONTRACT_ASSURANCE_REFILL_METRICS_SCHEMA,
        "schema_version": 1,
        "task_id": _text(task_id, "task_id"),
        "interface": RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE,
        "evidence": _text(evidence, "evidence"),
        "source_tree": _text(source_tree, "source_tree", required=False),
        "snapshot_id": _text(snapshot_id, "snapshot_id", required=False),
        "bounds": _canonical_value(dict(bounds or {}), "bounds"),
        "metrics": _canonical_value(dict(metrics), "metrics"),
        "provider_call_count": int(metrics.get("provider_call_count", 0) or 0),
        "model_call_count": int(metrics.get("model_call_count", 0) or 0),
        "llm_call_count": int(metrics.get("llm_call_count", 0) or 0),
        "completion_authoritative": False,
    }
    payload["passed"] = bool(metrics.get("passed", False))
    payload["metrics_id"] = _digest(
        {k: v for k, v in payload.items() if k != "metrics_id"}
    )
    return payload


# Compatibility nouns used by supervisor registries.
RuntimeContractAssuranceRefillHandler = RuntimeContractAssuranceRefill
RuntimeContractRefill = RuntimeContractAssuranceRefill


__all__ = [
    "ANALYZER_CANARY_SCHEMA",
    "ANALYZER_HEALTH_SCHEMA",
    "DEFAULT_RUNTIME_GOAL_ID",
    "RUNTIME_COMPONENT_SUBGOALS",
    "RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE",
    "RUNTIME_CONTRACT_ASSURANCE_REFILL_METRICS_SCHEMA",
    "RUNTIME_CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA",
    "ContractAnalyzerCapability",
    "ContractAssuranceAnalysis",
    "ContractAssuranceFinding",
    "ContractAssuranceGoalLineage",
    "RuntimeContractAssuranceRefill",
    "RuntimeContractAssuranceRefillError",
    "RuntimeContractAssuranceRefillHandler",
    "RuntimeContractAssuranceRefillOutcome",
    "RuntimeContractAssuranceRefillPolicy",
    "RuntimeContractAssuranceRefillReason",
    "RuntimeContractAssuranceRefillRequest",
    "RuntimeContractRefill",
    "build_runtime_contract_assurance_refill_handler",
    "build_runtime_refill_metrics_report",
    "normalize_changed_input",
    "normalize_changed_inputs",
    "resolve_runtime_subgoal",
    "run_runtime_contract_assurance_refill",
]
