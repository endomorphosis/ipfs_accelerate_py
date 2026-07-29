"""Supervisor-native, bounded refill for symbolic contract findings.

The contract analyzer is intentionally injected as a typed Python callback.
This module does not construct shell commands and does not walk repository
source.  It is the transaction boundary between a current analyzer result and
the existing :mod:`contract_mismatch_refinery`:

* low backlog is the only condition which starts analysis;
* every admitted packet has current analyzer capability, healthy canaries, and
  an objective goal lineage;
* the repair refinery owns task identity, stale-evidence handling, and
  open-work/finding bounds;
* a checksummed state file and last-known-good sibling make restart replay
  exact and recover a torn or corrupt latest write; and
* an empty result is completion-safe only after complete coverage and a
  current, independent exhaustion quorum.

Generated tasks and scan receipts are evidence, never completion authority.
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
from .contract_mismatch_refinery import (
    ContractMismatchRefinery,
    ContractMismatchRefineryDecision,
    ContractMismatchRefineryPolicy,
    ContractMismatchRefineryReason,
    ContractRepairTask,
    parse_contract_repair_board,
)
from .scan_receipts import (
    ExhaustionBinding,
    RefillScanResult,
    ScanMode,
    ScanTerminalReason,
    evaluate_exhaustion_quorum,
)


CONTRACT_ASSURANCE_REFILL_INTERFACE: Final = "ContractAssuranceRefill@1"
CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-assurance-refill-state@1"
)
CONTRACT_ASSURANCE_ANALYSIS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-assurance-analysis@1"
)
CONTRACT_ASSURANCE_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-assurance-capability@1"
)
CONTRACT_ASSURANCE_LINEAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-assurance-goal-lineage@1"
)
CONTRACT_ASSURANCE_REFILL_STATE_VERSION: Final = 1
DEFAULT_ANALYZER_ID: Final = "swissknife.contract-assurance"
DEFAULT_MIN_OPEN_TASKS: Final = 8
DEFAULT_MAX_OPEN_TASKS: Final = 48
DEFAULT_MAX_FINDINGS: Final = 8
DEFAULT_TIMEOUT_SECONDS: Final = 120.0
DEFAULT_COOLDOWN_SECONDS: Final = 900
DEFAULT_REQUIRED_EXHAUSTION_MEMBERS: Final = 2
DEFAULT_ROOT_GOAL_ID: Final = "SCA-G000"
MAX_REPLAY_RECORDS: Final = 128
MAX_EXHAUSTION_RECEIPTS: Final = 64
_GOAL_ID_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,255}$")


class ContractAssuranceRefillError(ValueError):
    """A refill request, analyzer result, or durable record is malformed."""


class ContractAssuranceRefillReason(str, Enum):
    """Stable supervisor outcomes and fail-closed rejection reasons."""

    GENERATED = "generated"
    DUPLICATE_ONLY = "duplicate_only"
    EXHAUSTED = "exhausted"
    THRESHOLD_SATISFIED = "threshold_satisfied"
    COOLDOWN = "cooldown"
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


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ContractAssuranceRefillError(f"{name} must be a string")
    if (
        value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ContractAssuranceRefillError(
            f"{name} must be normalized single-line text"
        )
    if required and not value:
        raise ContractAssuranceRefillError(f"{name} is required")
    if len(value.encode("utf-8")) > 16_384:
        raise ContractAssuranceRefillError(f"{name} exceeds its byte bound")
    return value


def _identifier(value: Any, name: str) -> str:
    selected = _text(value, name)
    if not _GOAL_ID_RE.fullmatch(selected):
        raise ContractAssuranceRefillError(f"{name} is malformed")
    return selected


def _bounded_integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractAssuranceRefillError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ContractAssuranceRefillError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _finite_seconds(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ContractAssuranceRefillError(f"{name} must be numeric")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractAssuranceRefillError(f"{name} must be numeric") from exc
    minimum = 0.0 if allow_zero else 0.001
    if not minimum <= selected <= 86_400.0:
        raise ContractAssuranceRefillError(
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
        raise ContractAssuranceRefillError(
            f"{name} must be canonical JSON data"
        ) from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_at(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ContractAssuranceRefillPolicy:
    """Bounds and current capability requirements for one refill handler."""

    min_open_tasks: int = DEFAULT_MIN_OPEN_TASKS
    max_open_tasks: int = DEFAULT_MAX_OPEN_TASKS
    max_findings_per_run: int = DEFAULT_MAX_FINDINGS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    required_exhaustion_members: int = DEFAULT_REQUIRED_EXHAUSTION_MEMBERS
    analyzer_id: str = DEFAULT_ANALYZER_ID
    expected_analyzer_version: str = ""
    root_goal_id: str = DEFAULT_ROOT_GOAL_ID
    board_namespace: str = "swissknife-symbolic-contract-assurance-v1"

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
            raise ContractAssuranceRefillError(
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
        object.__setattr__(self, "analyzer_id", _identifier(self.analyzer_id, "analyzer_id"))
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
            "board_namespace",
            _text(self.board_namespace, "board_namespace"),
        )

    @property
    def configuration_revision(self) -> str:
        return _digest(
            {
                "interface": CONTRACT_ASSURANCE_REFILL_INTERFACE,
                "min_open_tasks": self.min_open_tasks,
                "max_open_tasks": self.max_open_tasks,
                "max_findings_per_run": self.max_findings_per_run,
                "timeout_seconds": self.timeout_seconds,
                "cooldown_seconds": self.cooldown_seconds,
                "required_exhaustion_members": self.required_exhaustion_members,
                "analyzer_id": self.analyzer_id,
                "expected_analyzer_version": self.expected_analyzer_version,
                "root_goal_id": self.root_goal_id,
                "board_namespace": self.board_namespace,
            }
        )


@dataclass(frozen=True, slots=True)
class ContractAssuranceGoalLineage:
    """Objective ancestry which authorizes one analyzer finding."""

    goal_id: str
    root_goal_id: str
    ancestor_goal_ids: tuple[str, ...]
    objective_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "root_goal_id", _identifier(self.root_goal_id, "root_goal_id")
        )
        ancestors = tuple(
            _identifier(item, "ancestor_goal_id") for item in self.ancestor_goal_ids
        )
        if not ancestors:
            raise ContractAssuranceRefillError(
                "goal lineage requires at least one ancestor"
            )
        if len(set(ancestors)) != len(ancestors) or self.goal_id in ancestors:
            raise ContractAssuranceRefillError("goal lineage is cyclic or duplicated")
        if ancestors[0] != self.root_goal_id:
            raise ContractAssuranceRefillError(
                "goal lineage must begin at root_goal_id"
            )
        object.__setattr__(self, "ancestor_goal_ids", ancestors)
        object.__setattr__(
            self,
            "objective_revision",
            _text(self.objective_revision, "objective_revision"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_ASSURANCE_LINEAGE_SCHEMA,
            "goal_id": self.goal_id,
            "root_goal_id": self.root_goal_id,
            "ancestor_goal_ids": list(self.ancestor_goal_ids),
            "objective_revision": self.objective_revision,
        }

    @classmethod
    def from_value(
        cls, value: "ContractAssuranceGoalLineage | Mapping[str, Any]"
    ) -> "ContractAssuranceGoalLineage":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractAssuranceRefillError("goal_lineage must be an object")
        return cls(
            goal_id=value.get("goal_id", ""),
            root_goal_id=value.get("root_goal_id", value.get("root_id", "")),
            ancestor_goal_ids=tuple(
                value.get(
                    "ancestor_goal_ids",
                    value.get("parent_goal_ids", value.get("ancestors", ())),
                )
                or ()
            ),
            objective_revision=value.get("objective_revision", ""),
        )


@dataclass(frozen=True, slots=True)
class ContractAnalyzerCapability:
    """Observed analyzer capability, bound to the exact source snapshot."""

    analyzer_id: str
    analyzer_version: str
    capability_id: str
    repository_id: str
    tree_id: str
    snapshot_id: str
    available: bool = True
    supported_claim_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "analyzer_id",
            "analyzer_version",
            "capability_id",
            "repository_id",
            "tree_id",
            "snapshot_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.available, bool):
            raise ContractAssuranceRefillError("available must be boolean")
        families = tuple(
            sorted(
                {
                    _text(item, "supported_claim_family")
                    for item in self.supported_claim_families
                }
            )
        )
        if not families:
            raise ContractAssuranceRefillError(
                "capability must declare supported claim families"
            )
        object.__setattr__(self, "supported_claim_families", families)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_ASSURANCE_CAPABILITY_SCHEMA,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "capability_id": self.capability_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "snapshot_id": self.snapshot_id,
            "available": self.available,
            "supported_claim_families": list(self.supported_claim_families),
        }

    @classmethod
    def from_value(
        cls, value: "ContractAnalyzerCapability | Mapping[str, Any]"
    ) -> "ContractAnalyzerCapability":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractAssuranceRefillError("capability must be an object")
        return cls(
            analyzer_id=value.get("analyzer_id", ""),
            analyzer_version=value.get("analyzer_version", value.get("version", "")),
            capability_id=value.get(
                "capability_id", value.get("capability_revision", "")
            ),
            repository_id=value.get(
                "repository_id", value.get("repository_identity", "")
            ),
            tree_id=value.get("tree_id", value.get("tree_identity", "")),
            snapshot_id=value.get("snapshot_id", ""),
            available=value.get("available", False),
            supported_claim_families=tuple(
                value.get("supported_claim_families", value.get("claim_families", ()))
                or ()
            ),
        )


@dataclass(frozen=True, slots=True)
class ContractAssuranceFinding:
    """One edit packet paired with its non-optional objective ancestry."""

    packet: McpContractEditPacket
    goal_lineage: ContractAssuranceGoalLineage

    def __post_init__(self) -> None:
        packet = self.packet
        if not isinstance(packet, McpContractEditPacket):
            if not isinstance(packet, Mapping):
                raise ContractAssuranceRefillError("finding packet is malformed")
            packet = McpContractEditPacket.from_dict(packet)
        object.__setattr__(self, "packet", packet)
        object.__setattr__(
            self,
            "goal_lineage",
            ContractAssuranceGoalLineage.from_value(self.goal_lineage),
        )

    @property
    def finding_id(self) -> str:
        return self.packet.finding_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet": self.packet.to_dict(),
            "goal_lineage": self.goal_lineage.to_dict(),
        }

    @classmethod
    def from_value(
        cls, value: "ContractAssuranceFinding | Mapping[str, Any]"
    ) -> "ContractAssuranceFinding":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractAssuranceRefillError("finding must be an object")
        packet = value.get("packet", value.get("edit_packet"))
        lineage = value.get("goal_lineage", value.get("lineage"))
        if packet is None or lineage is None:
            raise ContractAssuranceRefillError(
                "finding requires packet and goal_lineage"
            )
        return cls(packet=packet, goal_lineage=lineage)


@dataclass(frozen=True, slots=True)
class ContractAssuranceAnalysis:
    """Typed output returned by the injected current-snapshot analyzer."""

    snapshot_id: str
    repository_id: str
    tree_id: str
    analyzer_version: str
    capability: ContractAnalyzerCapability
    analyzer_health: Mapping[str, Any]
    canary_report: Mapping[str, Any]
    findings: tuple[ContractAssuranceFinding, ...] = ()
    coverage: Mapping[str, Any] = field(default_factory=dict)
    coverage_complete: bool = False
    exhaustive: bool = False
    evidence_channel: str = ""
    exhaustion_receipts: tuple[Mapping[str, Any] | RefillScanResult[Any], ...] = ()
    current_finding_record_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "snapshot_id",
            "repository_id",
            "tree_id",
            "analyzer_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "capability", ContractAnalyzerCapability.from_value(self.capability)
        )
        for name in ("analyzer_health", "canary_report", "coverage"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                to_dict = getattr(value, "to_dict", None)
                if not callable(to_dict):
                    raise ContractAssuranceRefillError(f"{name} must be an object")
                value = to_dict()
            object.__setattr__(self, name, _canonical_value(dict(value), name))
        if not isinstance(self.coverage_complete, bool):
            raise ContractAssuranceRefillError("coverage_complete must be boolean")
        if not isinstance(self.exhaustive, bool):
            raise ContractAssuranceRefillError("exhaustive must be boolean")
        channel = _text(
            self.evidence_channel, "evidence_channel", required=self.exhaustive
        )
        if self.exhaustive and not channel:
            raise ContractAssuranceRefillError(
                "exhaustive analysis requires an independent evidence_channel"
            )
        object.__setattr__(self, "evidence_channel", channel)
        object.__setattr__(
            self,
            "findings",
            tuple(ContractAssuranceFinding.from_value(item) for item in self.findings),
        )
        records: dict[str, str] = {}
        for finding_id, record_id in self.current_finding_record_ids.items():
            records[_text(finding_id, "finding_id")] = _text(
                record_id, "finding_record_id"
            )
        object.__setattr__(self, "current_finding_record_ids", records)
        object.__setattr__(
            self, "exhaustion_receipts", tuple(self.exhaustion_receipts)
        )

    @classmethod
    def from_value(
        cls, value: "ContractAssuranceAnalysis | Mapping[str, Any]"
    ) -> "ContractAssuranceAnalysis":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractAssuranceRefillError(
                "analyzer must return ContractAssuranceAnalysis or an object"
            )
        if value.get("schema") not in (None, CONTRACT_ASSURANCE_ANALYSIS_SCHEMA):
            raise ContractAssuranceRefillError(
                "unsupported contract assurance analysis schema"
            )
        return cls(
            snapshot_id=value.get("snapshot_id", ""),
            repository_id=value.get(
                "repository_id", value.get("repository_identity", "")
            ),
            tree_id=value.get("tree_id", value.get("tree_identity", "")),
            analyzer_version=value.get("analyzer_version", ""),
            capability=value.get("capability", value.get("capability_report")),
            analyzer_health=value.get(
                "analyzer_health", value.get("health_report", {})
            ),
            canary_report=value.get("canary_report", value.get("canaries", {})),
            findings=tuple(value.get("findings", value.get("items", ())) or ()),
            coverage=value.get("coverage", {}),
            coverage_complete=value.get("coverage_complete", False),
            exhaustive=value.get("exhaustive", False),
            evidence_channel=value.get(
                "evidence_channel", value.get("independence_key", "")
            ),
            exhaustion_receipts=tuple(value.get("exhaustion_receipts", ()) or ()),
            current_finding_record_ids=value.get(
                "current_finding_record_ids", {}
            ),
        )


@dataclass(frozen=True, slots=True)
class ContractAssuranceRefillRequest:
    """Bounded request passed to the analyzer callback."""

    cycle_id: str
    snapshot_id: str
    repository_id: str
    tree_id: str
    objective_revision: str
    configuration_revision: str
    current_open_tasks: int
    max_findings: int
    deadline_epoch: float

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
        }


@dataclass(frozen=True, slots=True)
class ContractAssuranceRefillOutcome:
    """One complete supervisor decision, including its typed scan receipt."""

    reason: ContractAssuranceRefillReason
    scan_result: RefillScanResult[Mapping[str, Any]]
    tasks: tuple[ContractRepairTask, ...] = ()
    decisions: tuple[ContractMismatchRefineryDecision, ...] = ()
    board_markdown: str = ""
    quorum: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    replayed: bool = False
    recovered_state: bool = False
    completion_authoritative: bool = False

    def __post_init__(self) -> None:
        reason = (
            self.reason
            if isinstance(self.reason, ContractAssuranceRefillReason)
            else ContractAssuranceRefillReason(str(self.reason))
        )
        object.__setattr__(self, "reason", reason)
        if self.completion_authoritative is not False:
            raise ContractAssuranceRefillError(
                "refill outcomes cannot grant completion authority"
            )

    @property
    def generated_count(self) -> int:
        return len(self.tasks)

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.scan_result.safe_for_completion_reasoning

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": CONTRACT_ASSURANCE_REFILL_INTERFACE,
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
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractAssuranceRefillOutcome":
        scan = RefillScanResult.from_dict(value.get("scan_result") or {})
        decisions = tuple(
            ContractMismatchRefineryDecision(
                finding_id=item.get("finding_id", ""),
                task_id=item.get("task_id", ""),
                reason_code=ContractMismatchRefineryReason(
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
        )


@dataclass(slots=True)
class _DurableState:
    board_markdown: str = ""
    last_refill_epoch: int = 0
    last_snapshot_id: str = ""
    replay_records: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    exhaustion_receipts: list[Mapping[str, Any]] = field(default_factory=list)
    sequence: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA,
            "version": CONTRACT_ASSURANCE_REFILL_STATE_VERSION,
            "sequence": self.sequence,
            "board_markdown": self.board_markdown,
            "last_refill_epoch": self.last_refill_epoch,
            "last_snapshot_id": self.last_snapshot_id,
            "replay_records": dict(self.replay_records),
            "exhaustion_receipts": list(self.exhaustion_receipts),
        }

    def envelope(self) -> dict[str, Any]:
        payload = self.payload()
        return {**payload, "state_digest": _digest(payload)}

    @classmethod
    def from_envelope(cls, value: Mapping[str, Any]) -> "_DurableState":
        if value.get("schema") != CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA:
            raise ContractAssuranceRefillError("unsupported durable state schema")
        if value.get("version") != CONTRACT_ASSURANCE_REFILL_STATE_VERSION:
            raise ContractAssuranceRefillError("unsupported durable state version")
        payload = dict(value)
        declared = payload.pop("state_digest", "")
        if not declared or declared != _digest(payload):
            raise ContractAssuranceRefillError("durable state checksum mismatch")
        board_value = value.get("board_markdown", "")
        if not isinstance(board_value, str) or "\x00" in board_value:
            raise ContractAssuranceRefillError("board_markdown is malformed")
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
            raise ContractAssuranceRefillError("durable state collections are malformed")
        # Validate replay records at load so a valid envelope cannot hide a
        # structurally corrupt result which would fail only after mutation.
        normalized_replay: dict[str, Mapping[str, Any]] = {}
        for key, outcome in replay.items():
            selected_key = _text(key, "replay_key")
            if not isinstance(outcome, Mapping):
                raise ContractAssuranceRefillError("replay outcome is malformed")
            ContractAssuranceRefillOutcome.from_dict(outcome)
            normalized_replay[selected_key] = dict(outcome)
        normalized_receipts = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise ContractAssuranceRefillError("exhaustion receipt is malformed")
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
    [ContractAssuranceRefillRequest],
    ContractAssuranceAnalysis | Mapping[str, Any],
]


class ContractAssuranceRefill:
    """Lease-serialized, restart-idempotent contract refill handler."""

    interface: Final = CONTRACT_ASSURANCE_REFILL_INTERFACE

    def __init__(
        self,
        analyzer: AnalyzerCallback,
        *,
        state_path: Path | str,
        policy: ContractAssuranceRefillPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(analyzer):
            raise ContractAssuranceRefillError("analyzer must be callable")
        self.analyzer = analyzer
        self.state_path = Path(state_path)
        self.backup_path = self.state_path.with_name(self.state_path.name + ".bak")
        self.lock_path = self.state_path.with_name(self.state_path.name + ".lock")
        self.policy = policy or ContractAssuranceRefillPolicy()
        self.clock = clock

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
            raise ContractAssuranceRefillError("durable state must be an object")
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
        # Seed recovery even for the first committed transaction.  On later
        # writes the backup intentionally remains the prior valid checkpoint.
        if not self.backup_path.exists():
            self._atomic_write(self.backup_path, state.envelope())

    def _scan_result(
        self,
        reason: ContractAssuranceRefillReason,
        *,
        request: ContractAssuranceRefillRequest,
        started_at: datetime,
        finished_at: datetime,
        tasks: Sequence[ContractRepairTask] = (),
        metadata: Mapping[str, Any] | None = None,
        error: str | None = None,
        completion_safe: bool = False,
        analyzer_version: str = "",
    ) -> RefillScanResult[Mapping[str, Any]]:
        terminal = {
            ContractAssuranceRefillReason.GENERATED: ScanTerminalReason.GENERATED,
            ContractAssuranceRefillReason.DUPLICATE_ONLY: ScanTerminalReason.DUPLICATE_ONLY,
            ContractAssuranceRefillReason.EXHAUSTED: ScanTerminalReason.EXHAUSTED,
            ContractAssuranceRefillReason.THRESHOLD_SATISFIED: (
                ScanTerminalReason.THRESHOLD_SATISFIED
            ),
            ContractAssuranceRefillReason.COOLDOWN: ScanTerminalReason.COOLDOWN,
            ContractAssuranceRefillReason.TIMED_OUT: ScanTerminalReason.TIMED_OUT,
            ContractAssuranceRefillReason.ANALYZER_FAILED: ScanTerminalReason.FAILED,
        }.get(reason, ScanTerminalReason.PARTIAL)
        return RefillScanResult(
            terminal_reason=terminal,
            scan_mode=ScanMode.LOW_BACKLOG.value,
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
                "interface": CONTRACT_ASSURANCE_REFILL_INTERFACE,
                "reason_code": reason.value,
                "snapshot_id": request.snapshot_id,
                "objective_revision": request.objective_revision,
                "configuration_revision": request.configuration_revision,
                "completion_authoritative": False,
                **dict(metadata or {}),
            },
        )

    def _outcome(
        self,
        reason: ContractAssuranceRefillReason,
        *,
        request: ContractAssuranceRefillRequest,
        started_at: datetime,
        finished_at: datetime,
        state: _DurableState,
        tasks: Sequence[ContractRepairTask] = (),
        decisions: Sequence[ContractMismatchRefineryDecision] = (),
        quorum: Mapping[str, Any] | None = None,
        reason_codes: Iterable[str] = (),
        error: str | None = None,
        completion_safe: bool = False,
        recovered_state: bool = False,
        metadata: Mapping[str, Any] | None = None,
        analyzer_version: str = "",
    ) -> ContractAssuranceRefillOutcome:
        return ContractAssuranceRefillOutcome(
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
            ),
            tasks=tuple(tasks),
            decisions=tuple(decisions),
            board_markdown=state.board_markdown,
            quorum=dict(quorum or {}),
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            recovered_state=recovered_state,
            completion_authoritative=False,
        )

    def _invoke_analyzer(
        self, request: ContractAssuranceRefillRequest
    ) -> ContractAssuranceAnalysis:
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="contract-assurance-refill"
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
        request: ContractAssuranceRefillRequest,
    ) -> ContractAssuranceRefillReason | None:
        capability = analysis.capability
        if not capability.available or not capability.supported_claim_families:
            return ContractAssuranceRefillReason.CAPABILITY_MISSING
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
            return ContractAssuranceRefillReason.CAPABILITY_STALE
        return None

    @staticmethod
    def _health_reason(
        analysis: ContractAssuranceAnalysis,
    ) -> ContractAssuranceRefillReason | None:
        health = analysis.analyzer_health
        if (
            health.get("schema") != ANALYZER_HEALTH_SCHEMA
            or health.get("status") != "healthy"
            or health.get("healthy") is not True
            or health.get("safe_for_completion_reasoning") is not True
        ):
            return ContractAssuranceRefillReason.ANALYZER_UNHEALTHY
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
            return ContractAssuranceRefillReason.CANARIES_FAILED
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
        request: ContractAssuranceRefillRequest,
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
        request: ContractAssuranceRefillRequest,
    ) -> tuple[ContractAssuranceFinding, ...]:
        admitted: list[ContractAssuranceFinding] = []
        for finding in sorted(findings, key=lambda item: item.finding_id):
            lineage = finding.goal_lineage
            if (
                lineage.root_goal_id != self.policy.root_goal_id
                or lineage.ancestor_goal_ids[0] != self.policy.root_goal_id
                or lineage.objective_revision != request.objective_revision
            ):
                raise ContractAssuranceRefillError(
                    f"finding {finding.finding_id} has no current goal lineage"
                )
            finding.packet.assert_current(request.snapshot_id)
            admitted.append(finding)
        return tuple(admitted)

    def _exhaustion_receipt(
        self,
        analysis: ContractAssuranceAnalysis,
        request: ContractAssuranceRefillRequest,
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
            },
        )

    def _binding(
        self,
        analysis: ContractAssuranceAnalysis,
        request: ContractAssuranceRefillRequest,
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
    ) -> ContractAssuranceRefillOutcome:
        """Run one serialized low-backlog refill transaction."""

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
        replay_key = _text(
            idempotency_key, "idempotency_key", required=False
        )
        epoch = (
            int(self.clock())
            if now_epoch is None
            else _bounded_integer(
                now_epoch, "now_epoch", minimum=0, maximum=2**63 - 1
            )
        )
        cycle_id = replay_key or _digest(
            {
                "snapshot_id": snapshot,
                "repository_id": repository,
                "tree_id": tree,
                "objective_revision": objective,
                "open_tasks": open_tasks,
                "epoch": epoch,
            }
        )
        request = ContractAssuranceRefillRequest(
            cycle_id=cycle_id,
            snapshot_id=snapshot,
            repository_id=repository,
            tree_id=tree,
            objective_revision=objective,
            configuration_revision=self.policy.configuration_revision,
            current_open_tasks=open_tasks,
            max_findings=self.policy.max_findings_per_run,
            deadline_epoch=float(epoch) + self.policy.timeout_seconds,
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
                replay = ContractAssuranceRefillOutcome.from_dict(
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
                                    (ContractAssuranceRefillReason.STATE_RECOVERED.value,)
                                    if recovered
                                    else ()
                                ),
                            )
                        )
                    ),
                )

            def finish(
                outcome: ContractAssuranceRefillOutcome,
                *,
                scanned: bool = False,
            ) -> ContractAssuranceRefillOutcome:
                final = replace(
                    outcome,
                    recovered_state=recovered,
                    reason_codes=tuple(
                        dict.fromkeys(
                            (
                                *outcome.reason_codes,
                                *(
                                    (ContractAssuranceRefillReason.STATE_RECOVERED.value,)
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
                if replay_record_key:
                    state.replay_records[replay_record_key] = final.to_dict()
                self._trim_state(state)
                self._save_state(state)
                return final

            # Reconcile persisted tasks against the requested snapshot even
            # when the backlog threshold suppresses a new analyzer call.
            if state.board_markdown:
                reconciled = ContractMismatchRefinery(
                    ContractMismatchRefineryPolicy(
                        max_open_work=self.policy.max_open_tasks,
                        max_findings_per_run=self.policy.max_findings_per_run,
                        cooldown_seconds=0,
                        board_namespace=self.policy.board_namespace,
                    )
                ).refine(
                    (),
                    current_snapshot_id=snapshot,
                    existing_board=state.board_markdown,
                    current_open_work=open_tasks,
                    now_epoch=epoch,
                )
                state.board_markdown = reconciled.markdown

            if open_tasks >= self.policy.min_open_tasks:
                outcome = self._outcome(
                    ContractAssuranceRefillReason.THRESHOLD_SATISFIED,
                    request=request,
                    started_at=started,
                    finished_at=started,
                    state=state,
                    recovered_state=recovered,
                )
                return finish(outcome)

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
                    ContractAssuranceRefillReason.COOLDOWN,
                    request=request,
                    started_at=started,
                    finished_at=started,
                    state=state,
                    reason_codes=(ContractAssuranceRefillReason.COOLDOWN.value,),
                )
                return finish(outcome)

            try:
                analysis = self._invoke_analyzer(request)
            except FutureTimeout:
                finished = max(_utc_now(), started)
                outcome = self._outcome(
                    ContractAssuranceRefillReason.TIMED_OUT,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    error=(
                        "contract analyzer exceeded "
                        f"{self.policy.timeout_seconds:g} seconds"
                    ),
                    reason_codes=(ContractAssuranceRefillReason.TIMED_OUT.value,),
                )
                return finish(outcome, scanned=True)
            except Exception as exc:
                finished = max(_utc_now(), started)
                outcome = self._outcome(
                    ContractAssuranceRefillReason.ANALYZER_FAILED,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    error=f"{type(exc).__name__}: {exc}",
                    reason_codes=(ContractAssuranceRefillReason.ANALYZER_FAILED.value,),
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
                )
                return finish(outcome, scanned=True)

            try:
                findings = self._validate_lineages(analysis.findings, request)
            except (ValueError, TypeError) as exc:
                outcome = self._outcome(
                    ContractAssuranceRefillReason.NO_GOAL_LINEAGE,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    error=None,
                    reason_codes=(
                        ContractAssuranceRefillReason.NO_GOAL_LINEAGE.value,
                    ),
                    metadata={"lineage_error": str(exc)},
                    analyzer_version=analysis.analyzer_version,
                )
                return finish(outcome, scanned=True)

            truncated = len(findings) > self.policy.max_findings_per_run
            selected = findings[: self.policy.max_findings_per_run]
            refinery = ContractMismatchRefinery(
                ContractMismatchRefineryPolicy(
                    max_open_work=self.policy.max_open_tasks,
                    max_findings_per_run=self.policy.max_findings_per_run,
                    cooldown_seconds=0,
                    board_namespace=self.policy.board_namespace,
                )
            )
            refined = refinery.refine(
                (item.packet for item in selected),
                current_snapshot_id=snapshot,
                existing_board=state.board_markdown,
                current_open_work=open_tasks,
                now_epoch=epoch,
                current_finding_record_ids=analysis.current_finding_record_ids,
            )
            state.board_markdown = refined.markdown
            emitted_ids = {
                decision.task_id
                for decision in refined.decisions
                if decision.reason_code is ContractMismatchRefineryReason.EMITTED
            }
            tasks = tuple(
                task for task in refined.tasks if task.task_id in emitted_ids
            )
            if findings:
                reason_codes = [item.reason_code.value for item in refined.decisions]
                if truncated:
                    reason_codes.append(
                        ContractAssuranceRefillReason.FINDING_LIMIT.value
                    )
                if refined.generated_count:
                    reason = ContractAssuranceRefillReason.GENERATED
                elif any(
                    item.reason_code is ContractMismatchRefineryReason.OPEN_WORK_LIMIT
                    for item in refined.decisions
                ):
                    reason = ContractAssuranceRefillReason.OPEN_WORK_LIMIT
                else:
                    reason = ContractAssuranceRefillReason.DUPLICATE_ONLY
                outcome = self._outcome(
                    reason,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    tasks=tasks if reason is ContractAssuranceRefillReason.GENERATED else (),
                    decisions=refined.decisions,
                    reason_codes=reason_codes,
                    metadata={
                        "capability": analysis.capability.to_dict(),
                        "analyzer_health": analysis.analyzer_health,
                        "canary_report": analysis.canary_report,
                        "raw_finding_count": len(findings),
                        "admitted_finding_count": len(selected),
                    },
                    analyzer_version=analysis.analyzer_version,
                )
                return finish(outcome, scanned=True)

            if not analysis.exhaustive or not self._coverage_is_complete(analysis):
                outcome = self._outcome(
                    ContractAssuranceRefillReason.COVERAGE_INCOMPLETE,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    reason_codes=(
                        ContractAssuranceRefillReason.COVERAGE_INCOMPLETE.value,
                    ),
                    metadata={
                        "coverage": analysis.coverage,
                        "coverage_complete": analysis.coverage_complete,
                        "exhaustive": analysis.exhaustive,
                        "analyzer_health": analysis.analyzer_health,
                        "canary_report": analysis.canary_report,
                    },
                    analyzer_version=analysis.analyzer_version,
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
            # Persist only validated members as compact, full receipts.  State
            # from another binding is discarded rather than counted or grown.
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
                    ContractAssuranceRefillReason.QUORUM_INCOMPLETE,
                    request=request,
                    started_at=started,
                    finished_at=finished,
                    state=state,
                    quorum=quorum_record,
                    reason_codes=(
                        ContractAssuranceRefillReason.QUORUM_INCOMPLETE.value,
                    ),
                    metadata={
                        "coverage": analysis.coverage,
                        "coverage_complete": True,
                        "exhaustive": True,
                        "analyzer_health": analysis.analyzer_health,
                        "canary_report": analysis.canary_report,
                        "exhaustion_quorum": quorum_record,
                    },
                    analyzer_version=analysis.analyzer_version,
                )
                return finish(outcome, scanned=True)

            outcome = self._outcome(
                ContractAssuranceRefillReason.EXHAUSTED,
                request=request,
                started_at=started,
                finished_at=finished,
                state=state,
                quorum=quorum_record,
                reason_codes=(ContractAssuranceRefillReason.EXHAUSTED.value,),
                completion_safe=True,
                metadata={
                    "coverage": analysis.coverage,
                    "coverage_complete": True,
                    "exhaustive": True,
                    "capability": analysis.capability.to_dict(),
                    "analyzer_health": analysis.analyzer_health,
                    "canary_report": analysis.canary_report,
                    "exhaustion_quorum": quorum_record,
                },
                analyzer_version=analysis.analyzer_version,
            )
            return finish(outcome, scanned=True)

    __call__ = refill
    run = refill


def run_contract_assurance_refill(
    analyzer: AnalyzerCallback,
    *,
    state_path: Path | str,
    policy: ContractAssuranceRefillPolicy | None = None,
    **request: Any,
) -> ContractAssuranceRefillOutcome:
    """Functional supervisor handler entry point."""

    return ContractAssuranceRefill(
        analyzer, state_path=state_path, policy=policy
    ).refill(**request)


def build_contract_assurance_refill_handler(
    analyzer: AnalyzerCallback,
    *,
    state_path: Path | str,
    policy: ContractAssuranceRefillPolicy | None = None,
    clock: Callable[[], float] = time.time,
) -> ContractAssuranceRefill:
    """Return a callable suitable for objective/backlog refill registration."""

    return ContractAssuranceRefill(
        analyzer,
        state_path=state_path,
        policy=policy,
        clock=clock,
    )


# Explicit compatibility nouns used by supervisor registries.
ContractAssuranceRefillHandler = ContractAssuranceRefill
ContractAnalysisRefill = ContractAssuranceRefill
GoalLineage = ContractAssuranceGoalLineage
AnalyzerCapability = ContractAnalyzerCapability
ContractRefillFinding = ContractAssuranceFinding
ContractRefillAnalysis = ContractAssuranceAnalysis


__all__ = [
    "ANALYZER_CANARY_SCHEMA",
    "ANALYZER_HEALTH_SCHEMA",
    "CONTRACT_ASSURANCE_ANALYSIS_SCHEMA",
    "CONTRACT_ASSURANCE_CAPABILITY_SCHEMA",
    "CONTRACT_ASSURANCE_LINEAGE_SCHEMA",
    "CONTRACT_ASSURANCE_REFILL_INTERFACE",
    "CONTRACT_ASSURANCE_REFILL_STATE_SCHEMA",
    "AnalyzerCapability",
    "ContractAnalysisRefill",
    "ContractAnalyzerCapability",
    "ContractAssuranceAnalysis",
    "ContractAssuranceFinding",
    "ContractAssuranceGoalLineage",
    "ContractAssuranceRefill",
    "ContractAssuranceRefillError",
    "ContractAssuranceRefillHandler",
    "ContractAssuranceRefillOutcome",
    "ContractAssuranceRefillPolicy",
    "ContractAssuranceRefillReason",
    "ContractAssuranceRefillRequest",
    "ContractRefillAnalysis",
    "ContractRefillFinding",
    "GoalLineage",
    "build_contract_assurance_refill_handler",
    "run_contract_assurance_refill",
]
