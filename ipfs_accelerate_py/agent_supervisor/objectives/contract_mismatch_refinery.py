"""Bounded projection of MCP contract edit packets into a repair task board.

``ContractMismatchRefinery`` is deliberately not a source scanner.  It accepts
the narrow, already-admitted ``CodeEditPacket@1`` produced by SCA-100,
after decision validation the proof-gated ``ContractRepairEditPacket@2``
path, and (when enabled) plan-bound ``ChangePropagationEditPacket@1`` work
projected through :class:`ChangePropagationTaskSource`.  All project
accelerator-owned packets into agent-supervisor Markdown tasks without
letting a provider expand write scope.

The projection is safe to run repeatedly:

* task IDs depend only on the board namespace and finding dedupe identity;
* evidence revisions update the same task instead of minting another task;
* stale packets block an existing task and never create new work;
* open-work, finding-count, and cooldown limits are explicit and typed;
* exact paths and dependency references are revalidated at this boundary; and
* generated tasks explicitly lack completion authority.

No method in this module walks or reads repository source files.  Artifact
bodies stay behind the packet's content-addressed expansion handles.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..analysis.contract_mismatch_analyzer import SourceOwner, route_source_owner
from ..analysis.contract_repair_contracts import (
    DecisionDisposition,
    RepairTargetDecision,
)
from ..proof.contract_repair_edit_packet import (
    CONTRACT_REPAIR_EDIT_PACKET_INTERFACE,
    CONTRACT_REPAIR_EDIT_PACKET_SCHEMA,
    ContractRepairEditPacket,
    ContractRepairEditPacketError,
)
from ..proof.formal_verification_contracts import canonical_json_bytes
from ..proof.mcp_contract_edit_packet import (
    WRITE_PATH_AUTHORITY_TARGET_DECISION,
    ContractEditPacketError,
    McpContractEditPacket,
)


CONTRACT_MISMATCH_REFINERY_INTERFACE: Final = "ContractMismatchRefinery@1"
CONTRACT_REPAIR_TASK_INTERFACE: Final = "ContractRepairTask@1"
CONTRACT_REPAIR_BOARD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-board@1"
)
CONTRACT_REPAIR_TASK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-task@1"
)
CONTRACT_MISMATCH_TRIAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-mismatch-triage@1"
)
DEFAULT_BOARD_NAMESPACE: Final = "swissknife-symbolic-contract-assurance-v1"
DEFAULT_GOAL_ID: Final = "SCA-G101"
DEFAULT_MAX_OPEN_WORK: Final = 48
DEFAULT_MAX_FINDINGS_PER_RUN: Final = 8
DEFAULT_COOLDOWN_SECONDS: Final = 21_600
HARD_MAX_OPEN_WORK: Final = 10_000
HARD_MAX_FINDINGS_PER_RUN: Final = 1_024
HARD_MAX_COOLDOWN_SECONDS: Final = 31_536_000
HARD_MAX_PATHS: Final = 1_024
HARD_MAX_DEPENDENCIES: Final = 1_024
MAX_EVIDENCE_REVISIONS: Final = 64
MAX_ONE_LINE_BYTES: Final = 16_384
TASK_ID_PREFIX: Final = "SCA-REPAIR-"
_TASK_RECORD_PREFIX: Final = "<!-- contract-repair-task-v1:"
_TASK_RECORD_SUFFIX: Final = " -->"
_DEPENDENCY_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_TASK_ID_RE: Final = re.compile(r"^SCA-REPAIR-[0-9A-F]{20}$")
_TASK_HEADER_RE: Final = re.compile(
    r"(?m)^## (?P<task_id>SCA-REPAIR-[0-9A-F]{20})(?: .*)?$"
)
_STATUS_RE: Final = re.compile(r"(?m)^- Status: (?P<status>[a-z][a-z0-9_-]*)$")
_LAST_EPOCH_RE: Final = re.compile(r"(?m)^- Last refinery epoch: (?P<epoch>[0-9]+)$")
_OPEN_STATUSES: Final = frozenset({"active", "in_progress", "ready", "todo"})
_SECURITY_TOKENS: Final = (
    "auth",
    "capability",
    "cwe",
    "injection",
    "owasp",
    "permission",
    "security",
    "traversal",
    "vulnerab",
)


class ContractMismatchRefineryReason(str, Enum):
    """Stable reasons returned by the fail-closed refinery."""

    EMITTED = "emitted"
    EVIDENCE_UPDATED = "evidence_updated"
    DUPLICATE = "duplicate"
    STALE_FINDING = "stale_finding"
    OPEN_WORK_LIMIT = "open_work_limit"
    FINDING_LIMIT = "finding_limit"
    COOLDOWN = "cooldown"
    UNSUPPORTED_FINDING = "unsupported_finding"
    MALFORMED_PACKET = "malformed_packet"
    MALFORMED_PATH = "malformed_path"
    OWNER_MISMATCH = "owner_mismatch"
    MALFORMED_DEPENDENCY = "malformed_dependency"
    SELF_DEPENDENCY = "self_dependency"
    MALFORMED_BOARD = "malformed_board"
    DECISION_INVALID = "decision_invalid"
    SCOPE_EXPANSION = "scope_expansion"


class ContractMismatchRefineryError(ValueError):
    """A packet or persisted projection failed deterministic admission."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: ContractMismatchRefineryReason | str,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


def _bounded_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractMismatchRefineryError(
            f"{name} must be an integer",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if not 0 <= value <= maximum:
        raise ContractMismatchRefineryError(
            f"{name} must be between 0 and {maximum}",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    return value


def _one_line(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ContractMismatchRefineryError(
            f"{name} must be a string",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if (
        value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ContractMismatchRefineryError(
            f"{name} must be bounded single-line text",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if required and not value:
        raise ContractMismatchRefineryError(
            f"{name} is required",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if len(value.encode("utf-8")) > MAX_ONE_LINE_BYTES:
        raise ContractMismatchRefineryError(
            f"{name} exceeds its byte bound",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    return value


def _strings(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = HARD_MAX_DEPENDENCIES,
) -> tuple[str, ...]:
    if isinstance(values, str):
        source: Sequence[Any] = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ContractMismatchRefineryError(
            f"{name} must be a sequence of strings",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    result = tuple(sorted({_one_line(item, name) for item in source}))
    if required and not result:
        raise ContractMismatchRefineryError(
            f"{name} must not be empty",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if len(result) > maximum:
        raise ContractMismatchRefineryError(
            f"{name} exceeds its item bound",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    return result


def _exact_path(value: Any, name: str) -> str:
    raw = _one_line(value, name)
    candidate = PurePosixPath(raw)
    if (
        "\\" in raw
        or candidate.is_absolute()
        or raw.startswith("./")
        or candidate.as_posix() != raw
        or candidate.as_posix() in {"", "."}
        or ".." in candidate.parts
        or any(character in raw for character in "*?[]{}")
    ):
        raise ContractMismatchRefineryError(
            f"{name} must contain exact normalized repository-relative paths",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PATH,
        )
    if route_source_owner(raw) is not SourceOwner.ACCELERATOR:
        raise ContractMismatchRefineryError(
            f"{name} path is not accelerator-owned: {raw}",
            reason_code=ContractMismatchRefineryReason.OWNER_MISMATCH,
        )
    return raw


def _paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        raise ContractMismatchRefineryError(
            f"{name} must be a sequence of exact paths",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PATH,
        )
    result = tuple(sorted({_exact_path(item, name) for item in values}))
    if required and not result:
        raise ContractMismatchRefineryError(
            f"{name} must not be empty",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PATH,
        )
    if len(result) > HARD_MAX_PATHS:
        raise ContractMismatchRefineryError(
            f"{name} exceeds its item bound",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PATH,
        )
    return result


def _dependencies(
    values: Any,
    *,
    generated_task_id: str,
) -> tuple[str, ...]:
    result = _strings(values, "dependency_ids")
    for dependency in result:
        if not _DEPENDENCY_RE.fullmatch(dependency):
            raise ContractMismatchRefineryError(
                f"malformed dependency reference: {dependency!r}",
                reason_code=ContractMismatchRefineryReason.MALFORMED_DEPENDENCY,
            )
        if dependency == generated_task_id:
            raise ContractMismatchRefineryError(
                "generated task cannot depend on itself",
                reason_code=ContractMismatchRefineryReason.SELF_DEPENDENCY,
            )
    return result


def deterministic_repair_task_id(
    finding_id: str,
    *,
    board_namespace: str = DEFAULT_BOARD_NAMESPACE,
) -> str:
    """Return the stable board alias for one finding dedupe identity."""

    selected_finding = _one_line(finding_id, "finding_id")
    selected_namespace = _one_line(board_namespace, "board_namespace")
    digest = sha256(
        canonical_json_bytes(
            {
                "board_namespace": selected_namespace,
                "finding_id": selected_finding,
                "schema": CONTRACT_REPAIR_TASK_SCHEMA,
            }
        )
    ).hexdigest()
    return TASK_ID_PREFIX + digest[:20].upper()


def _json_value(value: Any, name: str) -> Any:
    try:
        encoded = canonical_json_bytes(value)
        result = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractMismatchRefineryError(
            f"{name} must be canonical JSON data",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        ) from exc
    return result


def _untrusted_value(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get("instruction_authority") is False:
        return _json_value(value.get("value"), "labeled finding data")
    return _json_value(value, "finding data")


@dataclass(frozen=True, slots=True)
class ContractMismatchRefineryPolicy:
    """Hard limits for one deterministic board refill."""

    max_open_work: int = DEFAULT_MAX_OPEN_WORK
    max_findings_per_run: int = DEFAULT_MAX_FINDINGS_PER_RUN
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    board_namespace: str = DEFAULT_BOARD_NAMESPACE
    goal_id: str = DEFAULT_GOAL_ID
    # When true, accept ContractRepairEditPacket@2 after decision validation.
    # Default true so the integration cutover can project admitted @2 packets;
    # validation still fails closed without an admitted decision binding.
    accept_proof_gated_packets: bool = True
    # When true, accept ChangePropagationEditPacket@1 via the task source.
    # Default true so the RPR-044 cutover can project admitted propagation
    # packets; scope still equals packet write authority.
    accept_change_propagation_packets: bool = True
    # Opt-in: project logic-guided predictions into proof bundles / boards
    # (LPR-017).  Default off preserves legacy RPR board projection.
    accept_live_logic_repair: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_open_work",
            _bounded_int(
                self.max_open_work,
                "max_open_work",
                maximum=HARD_MAX_OPEN_WORK,
            ),
        )
        object.__setattr__(
            self,
            "max_findings_per_run",
            _bounded_int(
                self.max_findings_per_run,
                "max_findings_per_run",
                maximum=HARD_MAX_FINDINGS_PER_RUN,
            ),
        )
        object.__setattr__(
            self,
            "cooldown_seconds",
            _bounded_int(
                self.cooldown_seconds,
                "cooldown_seconds",
                maximum=HARD_MAX_COOLDOWN_SECONDS,
            ),
        )
        object.__setattr__(
            self,
            "board_namespace",
            _one_line(self.board_namespace, "board_namespace"),
        )
        object.__setattr__(self, "goal_id", _one_line(self.goal_id, "goal_id"))
        if not isinstance(self.accept_proof_gated_packets, bool):
            raise ContractMismatchRefineryError(
                "accept_proof_gated_packets must be a boolean",
                reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
            )
        if not isinstance(self.accept_change_propagation_packets, bool):
            raise ContractMismatchRefineryError(
                "accept_change_propagation_packets must be a boolean",
                reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
            )
        if not isinstance(self.accept_live_logic_repair, bool):
            raise ContractMismatchRefineryError(
                "accept_live_logic_repair must be a boolean",
                reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
            )


@dataclass(frozen=True, slots=True)
class ContractRepairTask:
    """One non-authoritative, exact-scope Markdown repair task."""

    task_id: str
    finding_id: str
    finding_record_id: str
    evidence_record_ids: tuple[str, ...]
    packet_id: str
    snapshot_id: str
    source_task_id: str
    title: str
    priority: str
    status: str
    blocked_reason: str
    contract_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    affected_symbols: tuple[str, ...]
    affected_paths: tuple[str, ...]
    read_paths: tuple[str, ...]
    write_paths: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    failed_premise_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    reproduction: tuple[str, ...]
    validation_commands: tuple[str, ...]
    reproof_commands: tuple[str, ...]
    expected_postcondition: Any
    bounded_contract_slice: Any
    counterexample_id: str
    counterexample: Any
    expansion_handles: tuple[Mapping[str, Any], ...]
    goal_id: str
    board_namespace: str
    last_observed_epoch: int = 0
    completion_authoritative: bool = False

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ContractMismatchRefineryError(
                "repair task ID is malformed",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        expected_id = deterministic_repair_task_id(
            self.finding_id, board_namespace=self.board_namespace
        )
        if self.task_id != expected_id:
            raise ContractMismatchRefineryError(
                "repair task ID does not match its finding identity",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        for name in (
            "finding_id",
            "finding_record_id",
            "packet_id",
            "snapshot_id",
            "source_task_id",
            "title",
            "priority",
            "status",
            "counterexample_id",
            "goal_id",
            "board_namespace",
        ):
            object.__setattr__(self, name, _one_line(getattr(self, name), name))
        object.__setattr__(
            self,
            "blocked_reason",
            _one_line(self.blocked_reason, "blocked_reason", required=False),
        )
        if self.status not in _OPEN_STATUSES | {"blocked", "completed"}:
            raise ContractMismatchRefineryError(
                f"unsupported repair task status: {self.status}",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        if self.status == "blocked" and not self.blocked_reason:
            raise ContractMismatchRefineryError(
                "blocked repair task requires a reason",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        for name in (
            "evidence_record_ids",
            "contract_ids",
            "obligation_ids",
            "affected_symbols",
            "failed_premise_ids",
            "reason_codes",
            "reproduction",
            "validation_commands",
            "reproof_commands",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name,
                    required=name
                    in {
                        "evidence_record_ids",
                        "contract_ids",
                        "obligation_ids",
                        "affected_symbols",
                        "reproduction",
                        "validation_commands",
                        "reproof_commands",
                    },
                    maximum=MAX_EVIDENCE_REVISIONS
                    if name == "evidence_record_ids"
                    else HARD_MAX_DEPENDENCIES,
                ),
            )
        object.__setattr__(
            self, "affected_paths", _paths(self.affected_paths, "affected_paths")
        )
        object.__setattr__(
            self, "read_paths", _paths(self.read_paths, "read_paths")
        )
        object.__setattr__(
            self, "write_paths", _paths(self.write_paths, "write_paths")
        )
        if self.affected_paths != self.write_paths:
            raise ContractMismatchRefineryError(
                "write paths must exactly match affected paths",
                reason_code=ContractMismatchRefineryReason.MALFORMED_PATH,
            )
        if not set(self.write_paths).issubset(self.read_paths):
            raise ContractMismatchRefineryError(
                "read paths must include every write path",
                reason_code=ContractMismatchRefineryReason.MALFORMED_PATH,
            )
        object.__setattr__(
            self,
            "dependency_ids",
            _dependencies(
                self.dependency_ids, generated_task_id=self.task_id
            ),
        )
        if self.finding_record_id not in self.evidence_record_ids:
            raise ContractMismatchRefineryError(
                "current finding record is missing from evidence history",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        object.__setattr__(
            self,
            "last_observed_epoch",
            _bounded_int(
                self.last_observed_epoch,
                "last_observed_epoch",
                maximum=2**63 - 1,
            ),
        )
        if self.completion_authoritative is not False:
            raise ContractMismatchRefineryError(
                "generated repair task cannot hold completion authority",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        object.__setattr__(
            self,
            "expected_postcondition",
            _json_value(self.expected_postcondition, "expected_postcondition"),
        )
        object.__setattr__(
            self,
            "bounded_contract_slice",
            _json_value(self.bounded_contract_slice, "bounded_contract_slice"),
        )
        object.__setattr__(
            self,
            "counterexample",
            _json_value(self.counterexample, "counterexample"),
        )
        handles = tuple(
            _json_value(item, "expansion_handle") for item in self.expansion_handles
        )
        if not all(isinstance(item, Mapping) for item in handles):
            raise ContractMismatchRefineryError(
                "expansion handles must be objects",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        if any(item.get("body_embedded") is not False for item in handles):
            raise ContractMismatchRefineryError(
                "expansion handle cannot embed an artifact body",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        object.__setattr__(self, "expansion_handles", handles)

    @property
    def is_open(self) -> bool:
        return self.status in _OPEN_STATUSES

    @property
    def can_certify_completion(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_TASK_SCHEMA,
            "interface": CONTRACT_REPAIR_TASK_INTERFACE,
            "task_id": self.task_id,
            "finding_id": self.finding_id,
            "finding_record_id": self.finding_record_id,
            "evidence_record_ids": list(self.evidence_record_ids),
            "packet_id": self.packet_id,
            "snapshot_id": self.snapshot_id,
            "source_task_id": self.source_task_id,
            "title": self.title,
            "priority": self.priority,
            "status": self.status,
            "blocked_reason": self.blocked_reason,
            "contract_ids": list(self.contract_ids),
            "obligation_ids": list(self.obligation_ids),
            "affected_symbols": list(self.affected_symbols),
            "affected_paths": list(self.affected_paths),
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "dependency_ids": list(self.dependency_ids),
            "failed_premise_ids": list(self.failed_premise_ids),
            "reason_codes": list(self.reason_codes),
            "reproduction": list(self.reproduction),
            "validation_commands": list(self.validation_commands),
            "reproof_commands": list(self.reproof_commands),
            "expected_postcondition": self.expected_postcondition,
            "bounded_contract_slice": self.bounded_contract_slice,
            "counterexample_id": self.counterexample_id,
            "counterexample": self.counterexample,
            "expansion_handles": list(self.expansion_handles),
            "goal_id": self.goal_id,
            "board_namespace": self.board_namespace,
            "last_observed_epoch": self.last_observed_epoch,
            "completion_authoritative": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractRepairTask":
        if value.get("schema") != CONTRACT_REPAIR_TASK_SCHEMA:
            raise ContractMismatchRefineryError(
                "unsupported persisted repair task schema",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        try:
            return cls(
                task_id=value.get("task_id", ""),
                finding_id=value.get("finding_id", ""),
                finding_record_id=value.get("finding_record_id", ""),
                evidence_record_ids=tuple(value.get("evidence_record_ids") or ()),
                packet_id=value.get("packet_id", ""),
                snapshot_id=value.get("snapshot_id", ""),
                source_task_id=value.get("source_task_id", ""),
                title=value.get("title", ""),
                priority=value.get("priority", ""),
                status=value.get("status", ""),
                blocked_reason=value.get("blocked_reason", ""),
                contract_ids=tuple(value.get("contract_ids") or ()),
                obligation_ids=tuple(value.get("obligation_ids") or ()),
                affected_symbols=tuple(value.get("affected_symbols") or ()),
                affected_paths=tuple(value.get("affected_paths") or ()),
                read_paths=tuple(value.get("read_paths") or ()),
                write_paths=tuple(value.get("write_paths") or ()),
                dependency_ids=tuple(value.get("dependency_ids") or ()),
                failed_premise_ids=tuple(value.get("failed_premise_ids") or ()),
                reason_codes=tuple(value.get("reason_codes") or ()),
                reproduction=tuple(value.get("reproduction") or ()),
                validation_commands=tuple(
                    value.get("validation_commands") or ()
                ),
                reproof_commands=tuple(value.get("reproof_commands") or ()),
                expected_postcondition=value.get("expected_postcondition"),
                bounded_contract_slice=value.get("bounded_contract_slice"),
                counterexample_id=value.get("counterexample_id", ""),
                counterexample=value.get("counterexample"),
                expansion_handles=tuple(value.get("expansion_handles") or ()),
                goal_id=value.get("goal_id", ""),
                board_namespace=value.get("board_namespace", ""),
                last_observed_epoch=value.get("last_observed_epoch", 0),
                completion_authoritative=value.get(
                    "completion_authoritative", False
                ),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ContractMismatchRefineryError):
                raise
            raise ContractMismatchRefineryError(
                f"persisted repair task is malformed: {exc}",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            ) from exc


@dataclass(frozen=True, slots=True)
class ContractMismatchRefineryDecision:
    finding_id: str
    task_id: str
    reason_code: ContractMismatchRefineryReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ContractMismatchRefineryResult:
    """Complete byte-stable projection plus typed accounting."""

    tasks: tuple[ContractRepairTask, ...]
    decisions: tuple[ContractMismatchRefineryDecision, ...]
    markdown: str
    initial_open_work: int
    final_open_work: int
    max_open_work: int
    last_refinery_epoch: int

    @property
    def generated_count(self) -> int:
        return sum(
            item.reason_code is ContractMismatchRefineryReason.EMITTED
            for item in self.decisions
        )

    @property
    def updated_count(self) -> int:
        return sum(
            item.reason_code
            in {
                ContractMismatchRefineryReason.EVIDENCE_UPDATED,
                ContractMismatchRefineryReason.STALE_FINDING,
            }
            for item in self.decisions
        )

    @property
    def blocked_count(self) -> int:
        return sum(item.status == "blocked" for item in self.tasks)

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    @property
    def can_certify_completion(self) -> bool:
        return False


def build_contract_mismatch_triage(
    result: ContractMismatchRefineryResult,
    *,
    current_snapshot_id: str,
    owner: str,
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return sealed, non-authoritative accounting for one refinery run."""

    reason_counts: dict[str, int] = {}
    decisions: list[dict[str, str]] = []
    for decision in result.decisions:
        reason = decision.reason_code.value
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        decisions.append(
            {
                "detail": decision.detail,
                "finding_id": decision.finding_id,
                "reason_code": reason,
                "task_id": decision.task_id,
            }
        )
    payload: dict[str, Any] = {
        "schema": CONTRACT_MISMATCH_TRIAGE_SCHEMA,
        "interface": CONTRACT_MISMATCH_REFINERY_INTERFACE,
        "snapshot_id": _one_line(
            current_snapshot_id, "current_snapshot_id"
        ),
        "owner": _one_line(owner, "owner"),
        "source_record_count": len(source_records),
        "source_records_id": "sha256:"
        + sha256(canonical_json_bytes(source_records)).hexdigest(),
        "generated_count": result.generated_count,
        "updated_count": result.updated_count,
        "initial_open_work": result.initial_open_work,
        "final_open_work": result.final_open_work,
        "max_open_work": result.max_open_work,
        "last_refinery_epoch": result.last_refinery_epoch,
        "reason_counts": dict(sorted(reason_counts.items())),
        "decisions": sorted(
            decisions,
            key=lambda item: (
                item["finding_id"],
                item["task_id"],
                item["reason_code"],
                item["detail"],
            ),
        ),
        "completion_authoritative": False,
        "provider_call_count": 0,
        "model_call_count": 0,
        "llm_call_count": 0,
    }
    payload["triage_id"] = "sha256:" + sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


@dataclass(frozen=True, slots=True)
class _ParsedBoard:
    tasks: tuple[ContractRepairTask, ...]
    last_refinery_epoch: int


def _record_text(task: ContractRepairTask) -> str:
    encoded = base64.urlsafe_b64encode(
        canonical_json_bytes(task.to_dict())
    ).decode("ascii").rstrip("=")
    return _TASK_RECORD_PREFIX + encoded + _TASK_RECORD_SUFFIX


def _decode_record(line: str) -> ContractRepairTask:
    if not (line.startswith(_TASK_RECORD_PREFIX) and line.endswith(_TASK_RECORD_SUFFIX)):
        raise ContractMismatchRefineryError(
            "repair task machine record is missing",
            reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    encoded = line[len(_TASK_RECORD_PREFIX) : -len(_TASK_RECORD_SUFFIX)]
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ContractMismatchRefineryError(
            "repair task machine record is malformed",
            reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
        ) from exc
    if not isinstance(payload, Mapping):
        raise ContractMismatchRefineryError(
            "repair task machine record must contain an object",
            reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    return ContractRepairTask.from_dict(payload)


def parse_contract_repair_board(
    markdown: str,
    *,
    board_namespace: str = DEFAULT_BOARD_NAMESPACE,
) -> _ParsedBoard:
    """Parse only refinery-owned records; reject tampered identities."""

    if not markdown.strip():
        return _ParsedBoard((), 0)
    selected_namespace = _one_line(board_namespace, "board_namespace")
    declared = re.search(r"(?m)^- Board namespace: (.+)$", markdown)
    if declared and declared.group(1) != selected_namespace:
        raise ContractMismatchRefineryError(
            "existing board namespace differs from refinery policy",
            reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    epoch_match = _LAST_EPOCH_RE.search(markdown)
    last_epoch = int(epoch_match.group("epoch")) if epoch_match else 0
    headers = list(_TASK_HEADER_RE.finditer(markdown))
    tasks: list[ContractRepairTask] = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(markdown)
        block = markdown[header.start() : end]
        records = [
            line
            for line in block.splitlines()
            if line.startswith(_TASK_RECORD_PREFIX)
        ]
        if len(records) != 1:
            raise ContractMismatchRefineryError(
                f"{header.group('task_id')} requires exactly one machine record",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        task = _decode_record(records[0])
        if task.task_id != header.group("task_id"):
            raise ContractMismatchRefineryError(
                "repair task heading and machine identity differ",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        status_match = _STATUS_RE.search(block)
        if status_match is None:
            raise ContractMismatchRefineryError(
                f"{task.task_id} has no status",
                reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
            )
        visible_status = status_match.group("status")
        blocked_reason_match = re.search(
            r"(?m)^- Blocked reason: (?P<reason>.*)$", block
        )
        blocked_reason = (
            blocked_reason_match.group("reason") if blocked_reason_match else ""
        )
        if visible_status != task.status or blocked_reason != task.blocked_reason:
            task = replace(
                task,
                status=visible_status,
                blocked_reason=blocked_reason,
            )
        tasks.append(task)
    by_id = {task.task_id: task for task in tasks}
    by_finding = {task.finding_id: task for task in tasks}
    if len(by_id) != len(tasks) or len(by_finding) != len(tasks):
        raise ContractMismatchRefineryError(
            "existing board contains duplicate task or finding identities",
            reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    return _ParsedBoard(tuple(sorted(tasks, key=lambda item: item.task_id)), last_epoch)


def _priority(reason_codes: Sequence[str]) -> str:
    text = " ".join(reason_codes).casefold()
    return "P0" if any(token in text for token in _SECURITY_TOKENS) else "P1"


def _is_contract_repair_edit_packet(
    raw: McpContractEditPacket | ContractRepairEditPacket | Mapping[str, Any],
) -> bool:
    if isinstance(raw, ContractRepairEditPacket):
        return True
    if isinstance(raw, McpContractEditPacket):
        return False
    if not isinstance(raw, Mapping):
        return False
    schema = str(raw.get("schema") or "")
    interface = str(raw.get("interface") or "")
    return (
        schema == CONTRACT_REPAIR_EDIT_PACKET_SCHEMA
        or interface == CONTRACT_REPAIR_EDIT_PACKET_INTERFACE
        or "contract-repair-edit-packet@2" in schema
    )


def _is_change_propagation_edit_packet(raw: Any) -> bool:
    """Detect ChangePropagationEditPacket@1 without a hard top-level import."""

    # Lazy type check: avoid importing the packet module on cold refinery load
    # unless a mapping schema/interface claims the propagation shape.
    if isinstance(raw, Mapping):
        schema = str(raw.get("schema") or "")
        interface = str(raw.get("interface") or "")
        return (
            "change-propagation-edit-packet@1" in schema
            or interface == "ChangePropagationEditPacket@1"
            or "change_propagation_edit_packet" in schema
        )
    type_name = type(raw).__name__
    return type_name == "ChangePropagationEditPacket"


def _coerce_mcp_or_repair_packet(
    raw: McpContractEditPacket | ContractRepairEditPacket | Mapping[str, Any],
) -> McpContractEditPacket | ContractRepairEditPacket:
    if isinstance(raw, (McpContractEditPacket, ContractRepairEditPacket)):
        return raw
    if not isinstance(raw, Mapping):
        raise ContractMismatchRefineryError(
            "packet must be a mapping or typed packet",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if _is_change_propagation_edit_packet(raw):
        # Propagation packets use a dedicated projection path; do not coerce
        # them into MCP / @2 repair packets.
        raise ContractMismatchRefineryError(
            "ChangePropagationEditPacket@1 must use project_change_propagation",
            reason_code=ContractMismatchRefineryReason.UNSUPPORTED_FINDING,
        )
    if _is_contract_repair_edit_packet(raw):
        return ContractRepairEditPacket.from_dict(raw)
    return McpContractEditPacket.from_dict(raw)


def _validate_decision_for_packet(
    packet: McpContractEditPacket | ContractRepairEditPacket,
    *,
    decision: RepairTargetDecision | None,
) -> None:
    """Require an admitted decision and forbid provider scope expansion."""

    if isinstance(packet, ContractRepairEditPacket):
        if not packet.decision_id:
            raise ContractMismatchRefineryError(
                "ContractRepairEditPacket@2 requires a decision identity",
                reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
            )
        write_paths = tuple(packet.write_paths)
        if decision is None:
            # Packet already embeds decision-bound write paths from
            # materialization; without a replay decision we still refuse empty
            # write authority.
            if not write_paths:
                raise ContractMismatchRefineryError(
                    "decision-bound packet has no write paths",
                    reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
                )
            return
        if decision.content_id != packet.decision_id:
            raise ContractMismatchRefineryError(
                "packet decision_id does not match the validated decision",
                reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
            )
        if decision.disposition is not DecisionDisposition.ADMITTED:
            raise ContractMismatchRefineryError(
                "refinery accepts @2 only for an admitted decision",
                reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
            )
        if write_paths != tuple(decision.permitted_write_paths):
            raise ContractMismatchRefineryError(
                "packet write paths expand beyond the validated decision",
                reason_code=ContractMismatchRefineryReason.SCOPE_EXPANSION,
            )
        if not set(packet.read_paths).issuperset(decision.permitted_read_paths):
            raise ContractMismatchRefineryError(
                "packet read paths drop decision read authority",
                reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
            )
        return

    # McpContractEditPacket @2 decision path embeds authority metadata.
    goal = packet.context_capsule.goal
    authority = packet.context_capsule.authority
    scope = packet.context_capsule.scope
    write_authority = str(
        scope.get("write_path_authority")
        or authority.get("write_path_authority")
        or ""
    )
    decision_id = str(
        goal.get("decision_id")
        or authority.get("decision_id")
        or scope.get("decision_id")
        or ""
    )
    packet_version = goal.get("packet_version") or authority.get("packet_version")
    decision_bound = (
        write_authority == WRITE_PATH_AUTHORITY_TARGET_DECISION
        or packet_version == 2
        or bool(decision_id)
    )
    if not decision_bound:
        return
    if not decision_id:
        raise ContractMismatchRefineryError(
            "decision-bound MCP packet is missing decision_id",
            reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
        )
    if decision is None:
        if not packet.write_paths:
            raise ContractMismatchRefineryError(
                "decision-bound packet has no write paths",
                reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
            )
        return
    if decision.content_id != decision_id:
        raise ContractMismatchRefineryError(
            "MCP packet decision_id does not match the validated decision",
            reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
        )
    if decision.disposition is not DecisionDisposition.ADMITTED:
        raise ContractMismatchRefineryError(
            "refinery accepts decision-bound packets only when admitted",
            reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
        )
    if tuple(packet.write_paths) != tuple(decision.permitted_write_paths):
        raise ContractMismatchRefineryError(
            "provider cannot expand write scope beyond the decision",
            reason_code=ContractMismatchRefineryReason.SCOPE_EXPANSION,
        )


def _packet_task(
    packet: McpContractEditPacket | ContractRepairEditPacket,
    *,
    policy: ContractMismatchRefineryPolicy,
    now_epoch: int,
    evidence_record_ids: Iterable[str] = (),
    decision: RepairTargetDecision | None = None,
) -> ContractRepairTask:
    if isinstance(packet, ContractRepairEditPacket):
        if not policy.accept_proof_gated_packets:
            raise ContractMismatchRefineryError(
                "proof-gated @2 packets are disabled by policy",
                reason_code=ContractMismatchRefineryReason.UNSUPPORTED_FINDING,
            )
        _validate_decision_for_packet(packet, decision=decision)
        finding_id = _one_line(packet.trace_id, "finding_id")
        task_id = deterministic_repair_task_id(
            finding_id, board_namespace=policy.board_namespace
        )
        write_paths = _paths(packet.write_paths, "write_paths")
        read_paths = _paths(packet.read_paths, "read_paths")
        # Task affected_paths track the admitted write allowlist for @2.
        affected_paths = write_paths
        contract_ids = _strings(
            (
                packet.sender_expected_contract_id,
                packet.receiver_expected_contract_id,
            ),
            "contract_ids",
            required=True,
        )
        obligations = _strings(
            packet.post_edit_obligation_ids, "obligation_ids", required=True
        )
        symbols = _strings(
            (packet.target_span.path, packet.strategy.value),
            "affected_symbols",
            required=True,
        )
        validation = _strings(
            packet.validation_commands, "validation_commands", required=True
        )
        reproof = _strings(
            packet.reproof_commands, "reproof_commands", required=True
        )
        reason_codes = _strings(
            (
                f"strategy:{packet.strategy.value}",
                f"decision:{packet.decision_id}",
            ),
            "reason_codes",
        )
        finding_record_id = _one_line(packet.decision_id, "finding_record_id")
        records = tuple(
            sorted(
                {
                    *(
                        _one_line(item, "evidence_record_id")
                        for item in evidence_record_ids
                    ),
                    finding_record_id,
                }
            )
        )
        if len(records) > MAX_EVIDENCE_REVISIONS:
            retained = tuple(
                item for item in records if item != finding_record_id
            )[-(MAX_EVIDENCE_REVISIONS - 1) :]
            records = tuple(sorted((*retained, finding_record_id)))
        title = _one_line(
            f"Repair contract {contract_ids[0]} at {packet.target_span.path}",
            "title",
        )
        return ContractRepairTask(
            task_id=task_id,
            finding_id=finding_id,
            finding_record_id=finding_record_id,
            evidence_record_ids=records,
            packet_id=packet.packet_id,
            snapshot_id=packet.roots.tree_id,
            source_task_id=packet.decision_id,
            title=title,
            priority=_priority(reason_codes),
            status="todo",
            blocked_reason="",
            contract_ids=contract_ids,
            obligation_ids=obligations,
            affected_symbols=symbols,
            affected_paths=affected_paths,
            read_paths=read_paths,
            write_paths=write_paths,
            dependency_ids=(),
            failed_premise_ids=(),
            reason_codes=reason_codes,
            reproduction=validation,
            validation_commands=validation,
            reproof_commands=reproof,
            expected_postcondition={
                "decision_id": packet.decision_id,
                "strategy": packet.strategy.value,
                "target_path": packet.target_span.path,
            },
            bounded_contract_slice={
                "decision_id": packet.decision_id,
                "write_paths": list(write_paths),
                "clauses": [item.to_dict() for item in packet.clauses],
            },
            counterexample_id=(
                packet.counterexample_refs[0].content_id
                if packet.counterexample_refs
                else packet.decision_id
            ),
            counterexample={
                "refs": [item.to_dict() for item in packet.counterexample_refs]
            },
            expansion_handles=tuple(
                item.to_dict() for item in packet.expansion_handles
            ),
            goal_id=policy.goal_id,
            board_namespace=policy.board_namespace,
            last_observed_epoch=now_epoch,
            completion_authoritative=False,
        )

    task_id = deterministic_repair_task_id(
        packet.finding_id, board_namespace=policy.board_namespace
    )
    goal = packet.context_capsule.goal
    scope = packet.context_capsule.scope
    acceptance = packet.context_capsule.acceptance
    _validate_decision_for_packet(packet, decision=decision)
    contract_ids = _strings(packet.contract_ids, "contract_ids", required=True)
    obligations = _strings(
        packet.obligation_ids, "obligation_ids", required=True
    )
    symbols = _strings(
        goal.get("affected_symbols", ()),
        "affected_symbols",
        required=True,
    )
    affected_paths = _paths(packet.write_paths, "affected_paths")
    read_paths = _paths(scope.get("read_paths", ()), "read_paths")
    write_paths = _paths(scope.get("write_paths", ()), "write_paths")
    if write_paths != affected_paths:
        raise ContractMismatchRefineryError(
            "packet write allowlist differs from affected paths",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PATH,
        )
    dependencies = _dependencies(
        packet.dependency_ids, generated_task_id=task_id
    )
    mandatory = _dependencies(
        packet.mandatory_dependency_ids, generated_task_id=task_id
    )
    if not set(mandatory).issubset(dependencies):
        raise ContractMismatchRefineryError(
            "mandatory packet dependency is absent",
            reason_code=ContractMismatchRefineryReason.MALFORMED_DEPENDENCY,
        )
    reason_codes = _strings(goal.get("reason_codes", ()), "reason_codes")
    failed_premises = _strings(
        goal.get("failed_premise_ids", ()), "failed_premise_ids"
    )
    validation = _strings(
        acceptance.get("validation_commands", ()),
        "validation_commands",
        required=True,
    )
    reproof = _strings(
        acceptance.get("reproof_commands", ()),
        "reproof_commands",
        required=True,
    )
    # CodeEditPacket@1 intentionally exposes exact executable validations, not
    # the source finding's potentially broader prose.  Reuse those commands as
    # the bounded reproduction entry and retain counterexample/obligation/
    # snapshot bindings in their typed fields below.
    reproduction = validation
    symbol_label = symbols[0]
    title = _one_line(
        f"Repair contract {contract_ids[0]} for {symbol_label}",
        "title",
    )
    records = tuple(
        sorted(
            {
                *(_one_line(item, "evidence_record_id") for item in evidence_record_ids),
                packet.finding_record_id,
            }
        )
    )
    if len(records) > MAX_EVIDENCE_REVISIONS:
        retained = tuple(
            item for item in records if item != packet.finding_record_id
        )[-(MAX_EVIDENCE_REVISIONS - 1) :]
        records = tuple(sorted((*retained, packet.finding_record_id)))
    return ContractRepairTask(
        task_id=task_id,
        finding_id=packet.finding_id,
        finding_record_id=packet.finding_record_id,
        evidence_record_ids=records,
        packet_id=packet.packet_id,
        snapshot_id=packet.snapshot_id,
        source_task_id=packet.task_id,
        title=title,
        priority=_priority(reason_codes),
        status="todo",
        blocked_reason="",
        contract_ids=contract_ids,
        obligation_ids=obligations,
        affected_symbols=symbols,
        affected_paths=affected_paths,
        read_paths=read_paths,
        write_paths=write_paths,
        dependency_ids=dependencies,
        failed_premise_ids=failed_premises,
        reason_codes=reason_codes,
        reproduction=reproduction,
        validation_commands=validation,
        reproof_commands=reproof,
        expected_postcondition=_untrusted_value(
            acceptance.get("expected_postcondition")
        ),
        bounded_contract_slice=_untrusted_value(
            goal.get("bounded_contract_slice")
        ),
        counterexample_id=packet.counterexample_id,
        counterexample=_untrusted_value(goal.get("counterexample")),
        expansion_handles=tuple(item.to_dict() for item in packet.expansion_handles),
        goal_id=policy.goal_id,
        board_namespace=policy.board_namespace,
        last_observed_epoch=now_epoch,
        completion_authoritative=False,
    )


def render_contract_repair_board(
    tasks: Sequence[ContractRepairTask],
    *,
    board_namespace: str = DEFAULT_BOARD_NAMESPACE,
    last_refinery_epoch: int = 0,
) -> str:
    """Render a stable agent-supervisor Markdown board."""

    namespace = _one_line(board_namespace, "board_namespace")
    epoch = _bounded_int(
        last_refinery_epoch, "last_refinery_epoch", maximum=2**63 - 1
    )
    ordered = tuple(sorted(tasks, key=lambda item: item.task_id))
    if len({item.task_id for item in ordered}) != len(ordered):
        raise ContractMismatchRefineryError(
            "cannot render duplicate repair task IDs",
            reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    if any(item.board_namespace != namespace for item in ordered):
        raise ContractMismatchRefineryError(
            "repair task belongs to another board namespace",
            reason_code=ContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    lines = [
        "# Generated ipfs_accelerate_py contract repairs",
        "",
        f"- Schema: {CONTRACT_REPAIR_BOARD_SCHEMA}",
        f"- Interface: {CONTRACT_MISMATCH_REFINERY_INTERFACE}",
        f"- Board namespace: {namespace}",
        "- Source: admitted CodeEditPacket@1 records only",
        "- Completion authority: external validation and re-proof only",
        "- Generated evidence authoritative: false",
        f"- Last refinery epoch: {epoch}",
        f"- Open task count: {sum(item.is_open for item in ordered)}",
        f"- Task count: {len(ordered)}",
    ]
    for task in ordered:
        postcondition = json.dumps(
            task.expected_postcondition,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        compact_slice = json.dumps(
            task.bounded_contract_slice,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        counterexample = json.dumps(
            task.counterexample,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        handles = json.dumps(
            list(task.expansion_handles),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        lines.extend(
            [
                "",
                f"## {task.task_id} {task.title}",
                "",
                f"- Status: {task.status}",
                f"- Blocked reason: {task.blocked_reason}",
                "- Completion: external-validation-and-reproof",
                "- Completion authoritative: false",
                f"- Priority: {task.priority}",
                "- Track: contract-repair",
                f"- Depends on: {', '.join(task.dependency_ids)}",
                f"- Goal id: {task.goal_id}",
                f"- Board namespace: {task.board_namespace}",
                f"- Source task id: {task.source_task_id}",
                f"- Finding ID: {task.finding_id}",
                f"- Finding record ID: {task.finding_record_id}",
                f"- Evidence record IDs: {', '.join(task.evidence_record_ids)}",
                f"- Packet ID: {task.packet_id}",
                f"- Snapshot ID: {task.snapshot_id}",
                f"- Contract IDs: {', '.join(task.contract_ids)}",
                f"- Obligation IDs: {', '.join(task.obligation_ids)}",
                f"- Affected symbols: {', '.join(task.affected_symbols)}",
                f"- Affected paths: {', '.join(task.affected_paths)}",
                f"- Read paths: {', '.join(task.read_paths)}",
                f"- Write paths: {', '.join(task.write_paths)}",
                f"- Outputs: {', '.join(task.write_paths)}",
                f"- Predicted files: {', '.join(task.write_paths)}",
                f"- Failed premise IDs: {', '.join(task.failed_premise_ids)}",
                f"- Reason codes: {', '.join(task.reason_codes)}",
                f"- Reproduction: {' ; '.join(task.reproduction)}",
                f"- Validation: {' ; '.join(task.validation_commands)}",
                f"- Re-proof: {' ; '.join(task.reproof_commands)}",
                f"- Acceptance: {postcondition}",
                f"- Bounded contract slice: {compact_slice}",
                f"- Counterexample ID: {task.counterexample_id}",
                f"- Counterexample data: {counterexample}",
                f"- Expansion handles: {handles}",
                "- Provider semantic authority: false",
                "- Generated completion evidence: none",
                f"- Last observed epoch: {task.last_observed_epoch}",
                _record_text(task),
            ]
        )
    return "\n".join(lines) + "\n"


class ContractMismatchRefinery:
    """Create or update a bounded repair-board projection."""

    interface: Final = CONTRACT_MISMATCH_REFINERY_INTERFACE

    def __init__(
        self, policy: ContractMismatchRefineryPolicy | None = None
    ) -> None:
        self.policy = policy or ContractMismatchRefineryPolicy()

    def project_change_propagation(
        self,
        packet: Any,
        *,
        current_roots: Any = None,
        current_tree_id: str | None = None,
        provider_outputs: Sequence[str] | None = None,
    ) -> Any:
        """Project one admitted ChangePropagationEditPacket@1 into tasks.

        Task and writer scopes are taken solely from the packet write
        allowlist (via ChangePropagationTaskSource).  Providers cannot expand
        paths.  Disabled by policy when ``accept_change_propagation_packets``
        is false.  Lazy-imports the task source so cold refinery loads stay
        free of the propagation stack.
        """

        if not self.policy.accept_change_propagation_packets:
            raise ContractMismatchRefineryError(
                "change-propagation packets are disabled by policy",
                reason_code=ContractMismatchRefineryReason.UNSUPPORTED_FINDING,
            )

        from .change_propagation_task_source import (
            ChangePropagationTaskProjectionReason,
            ChangePropagationTaskSource,
        )

        source = ChangePropagationTaskSource(
            current_roots=current_roots,
            current_tree_id=current_tree_id,
        )
        projection = source.project(
            packet,
            current_roots=current_roots,
            current_tree_id=current_tree_id,
            provider_outputs=provider_outputs,
        )
        # Scope invariant: projected write scope must equal packet admits.
        admitted = tuple(getattr(packet, "permitted_write_paths", ()) or ())
        if not admitted and isinstance(packet, Mapping):
            admitted = tuple(packet.get("permitted_write_paths") or ())
        if admitted and getattr(projection, "reason", None) is (
            ChangePropagationTaskProjectionReason.EMITTED
        ):
            projected_scope = tuple(
                getattr(projection, "write_scope", None)
                or getattr(projection, "predicted_files", ())
                or ()
            )
            if not projected_scope:
                # Fall back to union of step task write paths.
                paths: list[str] = []
                for task in (
                    getattr(projection, "step_tasks", None)
                    or getattr(projection, "tasks", None)
                    or ()
                ):
                    paths.extend(getattr(task, "write_paths", ()) or ())
                projected_scope = tuple(paths)
            if tuple(sorted(set(projected_scope))) != tuple(sorted(admitted)):
                raise ContractMismatchRefineryError(
                    "projected task scopes must equal admitted packet write paths",
                    reason_code=ContractMismatchRefineryReason.SCOPE_EXPANSION,
                )
        return projection

    def bridge_logic_predictions(
        self,
        *,
        candidate_id: str,
        repository_id: str,
        tree_id: str,
        prediction_decision: Any = None,
        prediction_receipts: Sequence[Any] = (),
        base_proof_bundle: Any = None,
    ) -> Any:
        """Bridge admitted logic predictions into a CandidateProofBundle.

        Feature-gated by ``accept_live_logic_repair``.  Lazy-imports the live
        controller so cold refinery loads stay free of the LPR edge stack.
        Predictions compose with rather than replace an existing proof bundle.
        """

        if not self.policy.accept_live_logic_repair:
            raise ContractMismatchRefineryError(
                "live logic-repair prediction bridge is disabled by policy",
                reason_code=ContractMismatchRefineryReason.UNSUPPORTED_FINDING,
            )

        from ..todo_daemon.live_logic_repair_controller import (
            bridge_predictions_into_proof_bundle,
        )

        return bridge_predictions_into_proof_bundle(
            candidate_id=candidate_id,
            repository_id=repository_id,
            tree_id=tree_id,
            prediction_decision=prediction_decision,
            prediction_receipts=prediction_receipts,
            base_proof_bundle=base_proof_bundle,
        )

    def refine(
        self,
        packets: Iterable[
            McpContractEditPacket
            | ContractRepairEditPacket
            | Mapping[str, Any]
        ],
        *,
        current_snapshot_id: str,
        existing_board: str = "",
        current_open_work: int = 0,
        now_epoch: int = 0,
        current_finding_record_ids: Mapping[str, str] | None = None,
        target_decisions: Mapping[str, RepairTargetDecision] | None = None,
    ) -> ContractMismatchRefineryResult:
        """Project packets without scanning source or asserting completion.

        ``target_decisions`` optionally maps decision_id (or finding/trace id)
        to a ``RepairTargetDecision``.  When present, @2 packets are admitted
        only after that decision validates and write paths match exactly.
        """

        snapshot = _one_line(current_snapshot_id, "current_snapshot_id")
        open_work = _bounded_int(
            current_open_work,
            "current_open_work",
            maximum=HARD_MAX_OPEN_WORK,
        )
        now = _bounded_int(now_epoch, "now_epoch", maximum=2**63 - 1)
        parsed = parse_contract_repair_board(
            existing_board, board_namespace=self.policy.board_namespace
        )
        by_finding = {task.finding_id: task for task in parsed.tasks}
        preblocked_findings: set[str] = set()
        stale_existing_decisions: list[ContractMismatchRefineryDecision] = []
        for finding_id, task in tuple(by_finding.items()):
            if (
                task.snapshot_id != snapshot
                and not (
                    task.status == "blocked"
                    and task.blocked_reason
                    == ContractMismatchRefineryReason.STALE_FINDING.value
                )
            ):
                by_finding[finding_id] = replace(
                    task,
                    status="blocked",
                    blocked_reason=ContractMismatchRefineryReason.STALE_FINDING.value,
                    last_observed_epoch=now,
                    completion_authoritative=False,
                )
                preblocked_findings.add(finding_id)
                stale_existing_decisions.append(
                    ContractMismatchRefineryDecision(
                        finding_id,
                        task.task_id,
                        ContractMismatchRefineryReason.STALE_FINDING,
                        "persisted task belongs to an older repository snapshot",
                    )
                )
        existing_open = sum(task.is_open for task in parsed.tasks)
        existing_open -= sum(
            parsed_task.is_open
            for parsed_task in parsed.tasks
            if parsed_task.finding_id in preblocked_findings
        )
        initial_open = max(open_work, existing_open)

        current_records: dict[str, str] = {}
        for raw_id, raw_record in (current_finding_record_ids or {}).items():
            current_records[_one_line(raw_id, "finding_id")] = _one_line(
                raw_record, "finding_record_id"
            )
        decision_index: dict[str, RepairTargetDecision] = {}
        for raw_key, raw_decision in (target_decisions or {}).items():
            if not isinstance(raw_decision, RepairTargetDecision):
                raise ContractMismatchRefineryError(
                    "target_decisions values must be RepairTargetDecision",
                    reason_code=ContractMismatchRefineryReason.DECISION_INVALID,
                )
            decision_index[_one_line(raw_key, "decision_key")] = raw_decision
            decision_index[raw_decision.content_id] = raw_decision

        grouped: dict[
            str, list[McpContractEditPacket | ContractRepairEditPacket]
        ] = {}
        malformed_decisions: list[ContractMismatchRefineryDecision] = []
        for raw in packets:
            try:
                if (
                    isinstance(raw, Mapping)
                    and raw.get("state") == "unsupported"
                ):
                    finding_id = _one_line(
                        raw.get("finding_id"), "finding_id"
                    )
                    finding_snapshot_id = _one_line(
                        raw.get("snapshot_id") or raw.get("snapshot_root"),
                        "finding snapshot identity",
                    )
                    reason_code = _one_line(
                        raw.get("reason_code"), "reason_code"
                    )
                    _one_line(raw.get("contract_id"), "contract_id")
                    _strings(
                        raw.get("affected_paths"),
                        "affected_paths",
                        required=True,
                        maximum=HARD_MAX_PATHS,
                    )
                    _json_value(raw.get("counterexample"), "counterexample")
                    malformed_decisions.append(
                        ContractMismatchRefineryDecision(
                            finding_id=finding_id,
                            task_id="",
                            reason_code=(
                                ContractMismatchRefineryReason.STALE_FINDING
                                if finding_snapshot_id != snapshot
                                else ContractMismatchRefineryReason.UNSUPPORTED_FINDING
                            ),
                            detail=(
                                "explicitly unsupported analyzer finding is "
                                f"not implementation-ready: {reason_code}"
                            ),
                        )
                    )
                    continue
                packet = _coerce_mcp_or_repair_packet(raw)
                if isinstance(packet, ContractRepairEditPacket):
                    if not self.policy.accept_proof_gated_packets:
                        malformed_decisions.append(
                            ContractMismatchRefineryDecision(
                                finding_id=packet.trace_id,
                                task_id="",
                                reason_code=(
                                    ContractMismatchRefineryReason.UNSUPPORTED_FINDING
                                ),
                                detail="proof-gated @2 packets are disabled",
                            )
                        )
                        continue
                    group_id = packet.trace_id
                else:
                    group_id = packet.finding_id
                grouped.setdefault(group_id, []).append(packet)
            except (
                ContractEditPacketError,
                ContractRepairEditPacketError,
                TypeError,
                ValueError,
            ) as exc:
                malformed_decisions.append(
                    ContractMismatchRefineryDecision(
                        finding_id="",
                        task_id="",
                        reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
                        detail=str(exc),
                    )
                )

        decisions = [*stale_existing_decisions, *malformed_decisions]
        generated = 0
        seen_count = 0
        cooldown_active = (
            self.policy.cooldown_seconds > 0
            and parsed.last_refinery_epoch > 0
            and now > 0
            and (
                now < parsed.last_refinery_epoch
                or now - parsed.last_refinery_epoch
                < self.policy.cooldown_seconds
            )
        )
        for finding_id in sorted(grouped):
            task_id = deterministic_repair_task_id(
                finding_id, board_namespace=self.policy.board_namespace
            )
            if seen_count >= self.policy.max_findings_per_run:
                decisions.append(
                    ContractMismatchRefineryDecision(
                        finding_id,
                        task_id,
                        ContractMismatchRefineryReason.FINDING_LIMIT,
                    )
                )
                continue
            seen_count += 1
            candidates = tuple(
                sorted(grouped[finding_id], key=lambda item: item.packet_id)
            )
            existing = by_finding.get(finding_id)
            expected_record = current_records.get(finding_id, "")

            def _record_id(
                item: McpContractEditPacket | ContractRepairEditPacket,
            ) -> str:
                if isinstance(item, ContractRepairEditPacket):
                    return item.decision_id
                return item.finding_record_id

            matching_candidates = tuple(
                item
                for item in candidates
                if not expected_record or _record_id(item) == expected_record
            )
            packet = (
                matching_candidates[-1] if matching_candidates else candidates[-1]
            )
            if isinstance(packet, ContractRepairEditPacket):
                packet_snapshot = packet.roots.tree_id
                packet_record = packet.decision_id
            else:
                packet_snapshot = packet.snapshot_id
                packet_record = packet.finding_record_id
            stale = packet_snapshot != snapshot or (
                expected_record and packet_record != expected_record
            )
            if stale:
                if existing is not None:
                    by_finding[finding_id] = replace(
                        existing,
                        status="blocked",
                        blocked_reason=ContractMismatchRefineryReason.STALE_FINDING.value,
                        last_observed_epoch=now,
                        completion_authoritative=False,
                    )
                if finding_id not in preblocked_findings:
                    decisions.append(
                        ContractMismatchRefineryDecision(
                            finding_id,
                            task_id,
                            ContractMismatchRefineryReason.STALE_FINDING,
                            "packet snapshot or finding record is no longer current",
                        )
                    )
                continue
            try:
                if isinstance(packet, McpContractEditPacket):
                    packet.assert_current(
                        snapshot,
                        finding_record_id=expected_record,
                    )
                elif packet.roots.tree_id != snapshot:
                    raise ContractMismatchRefineryError(
                        "repair packet tree is not current",
                        reason_code=ContractMismatchRefineryReason.STALE_FINDING,
                    )
                evidence_ids = {_record_id(item) for item in candidates}
                if existing is not None:
                    evidence_ids.update(existing.evidence_record_ids)
                validated_decision = None
                if isinstance(packet, ContractRepairEditPacket):
                    validated_decision = decision_index.get(packet.decision_id)
                else:
                    decision_id = str(
                        packet.context_capsule.goal.get("decision_id") or ""
                    )
                    if decision_id:
                        validated_decision = decision_index.get(decision_id)
                projected = _packet_task(
                    packet,
                    policy=self.policy,
                    now_epoch=now,
                    evidence_record_ids=evidence_ids,
                    decision=validated_decision,
                )
            except (
                ContractEditPacketError,
                ContractRepairEditPacketError,
                ContractMismatchRefineryError,
                TypeError,
                ValueError,
            ) as exc:
                reason = (
                    exc.reason_code
                    if isinstance(exc, ContractMismatchRefineryError)
                    else ContractMismatchRefineryReason.MALFORMED_PACKET.value
                )
                try:
                    reason_code = ContractMismatchRefineryReason(reason)
                except ValueError:
                    reason_code = ContractMismatchRefineryReason.MALFORMED_PACKET
                decisions.append(
                    ContractMismatchRefineryDecision(
                        finding_id, task_id, reason_code, str(exc)
                    )
                )
                continue

            if existing is not None:
                completed_invalidated = (
                    existing.status == "completed"
                    and existing.finding_record_id
                    != projected.finding_record_id
                )
                if existing.status == "completed":
                    if not completed_invalidated:
                        projected = replace(
                            projected,
                            status="completed",
                            completion_authoritative=False,
                        )
                    else:
                        projected = replace(
                            projected,
                            status="blocked",
                            blocked_reason=(
                                ContractMismatchRefineryReason.STALE_FINDING.value
                            ),
                            completion_authoritative=False,
                        )
                changed = projected.to_dict() != replace(
                    existing, last_observed_epoch=now
                ).to_dict()
                by_finding[finding_id] = projected
                decisions.append(
                    ContractMismatchRefineryDecision(
                        finding_id,
                        task_id,
                        (
                            ContractMismatchRefineryReason.STALE_FINDING
                            if completed_invalidated
                            else ContractMismatchRefineryReason.EVIDENCE_UPDATED
                            if changed
                            else ContractMismatchRefineryReason.DUPLICATE
                        ),
                    )
                )
                continue
            if cooldown_active:
                decisions.append(
                    ContractMismatchRefineryDecision(
                        finding_id,
                        task_id,
                        ContractMismatchRefineryReason.COOLDOWN,
                    )
                )
                continue
            if initial_open + generated >= self.policy.max_open_work:
                decisions.append(
                    ContractMismatchRefineryDecision(
                        finding_id,
                        task_id,
                        ContractMismatchRefineryReason.OPEN_WORK_LIMIT,
                    )
                )
                continue
            by_finding[finding_id] = projected
            generated += 1
            decisions.append(
                ContractMismatchRefineryDecision(
                    finding_id,
                    task_id,
                    ContractMismatchRefineryReason.EMITTED,
                )
            )

        tasks = tuple(sorted(by_finding.values(), key=lambda item: item.task_id))
        last_epoch = (
            now
            if grouped or preblocked_findings
            else parsed.last_refinery_epoch
        )
        markdown = render_contract_repair_board(
            tasks,
            board_namespace=self.policy.board_namespace,
            last_refinery_epoch=last_epoch,
        )
        final_open = max(open_work, sum(task.is_open for task in tasks))
        return ContractMismatchRefineryResult(
            tasks=tasks,
            decisions=tuple(decisions),
            markdown=markdown,
            initial_open_work=initial_open,
            final_open_work=final_open,
            max_open_work=self.policy.max_open_work,
            last_refinery_epoch=last_epoch,
        )


def refine_contract_mismatch_packets(
    packets: Iterable[
        McpContractEditPacket | ContractRepairEditPacket | Mapping[str, Any]
    ],
    *,
    current_snapshot_id: str,
    existing_board: str = "",
    current_open_work: int = 0,
    now_epoch: int = 0,
    policy: ContractMismatchRefineryPolicy | None = None,
    current_finding_record_ids: Mapping[str, str] | None = None,
    target_decisions: Mapping[str, RepairTargetDecision] | None = None,
) -> ContractMismatchRefineryResult:
    """Functional entry point for supervisor refill integration."""

    return ContractMismatchRefinery(policy).refine(
        packets,
        current_snapshot_id=current_snapshot_id,
        existing_board=existing_board,
        current_open_work=current_open_work,
        now_epoch=now_epoch,
        current_finding_record_ids=current_finding_record_ids,
        target_decisions=target_decisions,
    )


def write_contract_repair_board(path: str | Path, markdown: str) -> None:
    """Atomically replace one generated board with UTF-8 Markdown."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(markdown)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_packet_document(
    path: Path,
) -> tuple[tuple[Mapping[str, Any], ...], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractMismatchRefineryError(
            f"unable to load packet records: {exc}",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        ) from exc
    document_snapshot_ids: set[str] = set()
    if isinstance(payload, Mapping):
        for key in ("snapshot_id", "snapshot_root"):
            raw_snapshot_id = payload.get(key)
            if isinstance(raw_snapshot_id, str) and raw_snapshot_id:
                document_snapshot_ids.add(raw_snapshot_id)
        for key in ("packets", "edit_packets", "findings"):
            if key in payload:
                payload = payload[key]
                break
        else:
            payload = (payload,)
    if not isinstance(payload, Sequence) or isinstance(
        payload, (str, bytes, bytearray)
    ):
        raise ContractMismatchRefineryError(
            "packet input must be a record or sequence of records",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if not all(isinstance(item, Mapping) for item in payload):
        raise ContractMismatchRefineryError(
            "packet input contains a non-object record",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    records = tuple(payload)
    record_snapshot_ids = {
        str(item["snapshot_id"])
        for item in records
        if isinstance(item.get("snapshot_id"), str) and item.get("snapshot_id")
    }
    inferred_snapshot_ids = set(record_snapshot_ids)
    inferred_snapshot_ids.update(document_snapshot_ids)
    if len(inferred_snapshot_ids) > 1:
        raise ContractMismatchRefineryError(
            "packet input contains conflicting snapshot identity values",
            reason_code=ContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    inferred_snapshot_id = (
        next(iter(inferred_snapshot_ids)) if inferred_snapshot_ids else None
    )
    return records, inferred_snapshot_id


def _load_packet_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load packet records while retaining the original helper contract."""

    records, _ = _load_packet_document(path)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a bounded accelerator-owned contract repair board."
    )
    parser.add_argument(
        "--findings",
        "--packets",
        dest="packets_path",
        required=True,
        help="JSON CodeEditPacket@1 record(s); no repository source is scanned.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--triage-output",
        help=(
            "Optional path for sealed non-authoritative refinery accounting."
        ),
    )
    parser.add_argument(
        "--snapshot",
        help=(
            "Current repository snapshot identity. When omitted, exactly one "
            "snapshot_id or snapshot_root must be present in the packet "
            "document or records."
        ),
    )
    parser.add_argument(
        "--owner",
        default="external/ipfs_accelerate",
        choices=("external/ipfs_accelerate", "ipfs_accelerate_py"),
        help="Reviewed accelerator path root; other owners are never admitted.",
    )
    parser.add_argument("--current-open-work", type=int, default=0)
    parser.add_argument("--max-open-work", type=int, default=DEFAULT_MAX_OPEN_WORK)
    parser.add_argument(
        "--max-findings", type=int, default=DEFAULT_MAX_FINDINGS_PER_RUN
    )
    parser.add_argument(
        "--cooldown-seconds", type=int, default=DEFAULT_COOLDOWN_SECONDS
    )
    parser.add_argument("--now-epoch", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output)
    existing = output.read_text(encoding="utf-8") if output.exists() else ""
    try:
        packet_records, inferred_snapshot_id = _load_packet_document(
            Path(args.packets_path)
        )
    except ContractMismatchRefineryError as exc:
        parser.error(str(exc))
    current_snapshot_id = args.snapshot or inferred_snapshot_id
    if not current_snapshot_id:
        parser.error(
            "--snapshot is required when the packet document does not "
            "contain exactly one snapshot identity"
        )
    result = refine_contract_mismatch_packets(
        packet_records,
        current_snapshot_id=current_snapshot_id,
        existing_board=existing,
        current_open_work=args.current_open_work,
        now_epoch=int(time.time()) if args.now_epoch is None else args.now_epoch,
        policy=ContractMismatchRefineryPolicy(
            max_open_work=args.max_open_work,
            max_findings_per_run=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
        ),
    )
    structural_rejections = tuple(
        decision
        for decision in result.decisions
        if decision.reason_code
        in {
            ContractMismatchRefineryReason.MALFORMED_PACKET,
            ContractMismatchRefineryReason.MALFORMED_PATH,
            ContractMismatchRefineryReason.OWNER_MISMATCH,
            ContractMismatchRefineryReason.MALFORMED_DEPENDENCY,
            ContractMismatchRefineryReason.SELF_DEPENDENCY,
            ContractMismatchRefineryReason.MALFORMED_BOARD,
        }
    )
    if structural_rejections:
        first = structural_rejections[0]
        parser.error(
            "packet admission failed closed: "
            f"{first.reason_code.value}: {first.detail}"
        )
    write_contract_repair_board(output, result.markdown)
    if args.triage_output:
        triage = build_contract_mismatch_triage(
            result,
            current_snapshot_id=current_snapshot_id,
            owner=args.owner,
            source_records=packet_records,
        )
        write_contract_repair_board(
            args.triage_output,
            json.dumps(
                triage,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )
    return 0


# Names used by the SCA interface inventory and supervisor integration.
BacklogRefinery = ContractMismatchRefinery
MarkdownTaskSource = ContractRepairTask


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
