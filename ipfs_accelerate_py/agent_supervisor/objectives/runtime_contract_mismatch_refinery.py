"""Bounded projection of runtime contract edit packets into the repair board.

``RuntimeContractMismatchRefinery`` (SCA-178) sits beside SCA-101's static
``ContractMismatchRefinery``.  It accepts only already-admitted
``CodeEditPacket@1`` records produced from current runtime counterexamples
(SCA-100 / SCA-177) and **appends** accelerator-owned tasks onto the shared
repair board after baseline triage.

The projection is deliberately not a source scanner:

* one current counterexample impact cluster (finding dedupe identity) yields
  one task;
* task IDs depend only on the board namespace and finding identity;
* evidence revisions update the same task instead of minting another;
* stale packets block existing work and never create new tasks;
* unsupported, stale, and unknown-only findings are never implementation-ready;
* open-work, finding-count, and cooldown limits remain explicit;
* exact paths/symbols, expected postcondition, validation, and re-proof are
  required; repository corpus is never embedded; and
* generated tasks never hold completion authority.

Historical board tasks (including baseline SCA-101 work) are preserved when
this refinery runs.  Non-accelerator owners are rejected at the path boundary.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import canonical_json_bytes
from ..proof.mcp_contract_edit_packet import McpContractEditPacket
from .contract_mismatch_refinery import (
    CONTRACT_REPAIR_BOARD_SCHEMA,
    DEFAULT_BOARD_NAMESPACE,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_MAX_FINDINGS_PER_RUN,
    DEFAULT_MAX_OPEN_WORK,
    HARD_MAX_COOLDOWN_SECONDS,
    HARD_MAX_DEPENDENCIES,
    HARD_MAX_FINDINGS_PER_RUN,
    HARD_MAX_OPEN_WORK,
    HARD_MAX_PATHS,
    MAX_ONE_LINE_BYTES,
    ContractMismatchRefinery,
    ContractMismatchRefineryPolicy,
    ContractMismatchRefineryReason,
    ContractRepairTask,
    parse_contract_repair_board,
    write_contract_repair_board,
)


RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE: Final = (
    "RuntimeContractMismatchRefinery@1"
)
RUNTIME_CONTRACT_MISMATCH_TRIAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-mismatch-triage@1"
)
DEFAULT_RUNTIME_GOAL_ID: Final = "SCA-G176"
DEFAULT_RUNTIME_TRACK: Final = "runtime-repair"
BASELINE_TRACK: Final = "contract-repair"

# Raw finding states that may appear outside CodeEditPacket@1 and must never
# become implementation-ready work on their own.
_NOT_IMPLEMENTATION_READY_STATES: Final = frozenset(
    {
        "unsupported",
        "stale",
        "unknown",
        "open",
        "pending",
        "ambiguous",
        "inconclusive",
        "partial",
        "not_measured",
        "unmeasured",
        "skipped",
        "proved",
        "pass",
        "passed",
        "success",
        "current",
        "hit",
        "miss",
        "cache_miss",
        "not_cached",
    }
)
_STALE_STATES: Final = frozenset({"stale", "invalidated"})
_PACKET_SCHEMA_MARKERS: Final = frozenset(
    {
        "CodeEditPacket@1",
        "ipfs_accelerate_py/agent-supervisor/code-edit-packet@1",
        "ipfs_accelerate_py/agent-supervisor/mcp-contract-edit-packet@1",
    }
)
_TASK_RECORD_PREFIX: Final = "<!-- contract-repair-task-v1:"
_TASK_RECORD_SUFFIX: Final = " -->"


class RuntimeContractMismatchRefineryReason(str, Enum):
    """Stable reasons returned by the fail-closed runtime refinery."""

    EMITTED = "emitted"
    EVIDENCE_UPDATED = "evidence_updated"
    DUPLICATE = "duplicate"
    STALE_FINDING = "stale_finding"
    OPEN_WORK_LIMIT = "open_work_limit"
    FINDING_LIMIT = "finding_limit"
    COOLDOWN = "cooldown"
    UNSUPPORTED_FINDING = "unsupported_finding"
    UNKNOWN_ONLY = "unknown_only"
    MALFORMED_PACKET = "malformed_packet"
    MALFORMED_PATH = "malformed_path"
    OWNER_MISMATCH = "owner_mismatch"
    MALFORMED_DEPENDENCY = "malformed_dependency"
    SELF_DEPENDENCY = "self_dependency"
    MALFORMED_BOARD = "malformed_board"


class RuntimeContractMismatchRefineryError(ValueError):
    """A packet or persisted projection failed deterministic admission."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: RuntimeContractMismatchRefineryReason | str,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


def _one_line(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise RuntimeContractMismatchRefineryError(
            f"{name} must be a string",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if (
        value != value.strip()
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimeContractMismatchRefineryError(
            f"{name} must be bounded single-line text",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if required and not value:
        raise RuntimeContractMismatchRefineryError(
            f"{name} is required",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if len(value.encode("utf-8")) > MAX_ONE_LINE_BYTES:
        raise RuntimeContractMismatchRefineryError(
            f"{name} exceeds its byte bound",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    return value


def _bounded_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeContractMismatchRefineryError(
            f"{name} must be an integer",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if not 0 <= value <= maximum:
        raise RuntimeContractMismatchRefineryError(
            f"{name} must be between 0 and {maximum}",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
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
        raise RuntimeContractMismatchRefineryError(
            f"{name} must be a sequence of strings",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    result = tuple(sorted({_one_line(item, name) for item in source}))
    if required and not result:
        raise RuntimeContractMismatchRefineryError(
            f"{name} must not be empty",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if len(result) > maximum:
        raise RuntimeContractMismatchRefineryError(
            f"{name} exceeds its item bound",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    return result


def _json_value(value: Any, name: str) -> Any:
    try:
        encoded = canonical_json_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeContractMismatchRefineryError(
            f"{name} must be canonical JSON data",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        ) from exc


@dataclass(frozen=True, slots=True)
class RuntimeContractMismatchRefineryPolicy:
    """Hard limits for one deterministic runtime board refill."""

    max_open_work: int = DEFAULT_MAX_OPEN_WORK
    max_findings_per_run: int = DEFAULT_MAX_FINDINGS_PER_RUN
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    board_namespace: str = DEFAULT_BOARD_NAMESPACE
    goal_id: str = DEFAULT_RUNTIME_GOAL_ID

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

    def to_base_policy(self) -> ContractMismatchRefineryPolicy:
        return ContractMismatchRefineryPolicy(
            max_open_work=self.max_open_work,
            max_findings_per_run=self.max_findings_per_run,
            cooldown_seconds=self.cooldown_seconds,
            board_namespace=self.board_namespace,
            goal_id=self.goal_id,
        )


@dataclass(frozen=True, slots=True)
class RuntimeContractMismatchRefineryDecision:
    finding_id: str
    task_id: str
    reason_code: RuntimeContractMismatchRefineryReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RuntimeContractMismatchRefineryResult:
    """Complete byte-stable projection plus typed accounting."""

    tasks: tuple[ContractRepairTask, ...]
    decisions: tuple[RuntimeContractMismatchRefineryDecision, ...]
    markdown: str
    initial_open_work: int
    final_open_work: int
    max_open_work: int
    last_refinery_epoch: int

    @property
    def generated_count(self) -> int:
        return sum(
            item.reason_code is RuntimeContractMismatchRefineryReason.EMITTED
            for item in self.decisions
        )

    @property
    def updated_count(self) -> int:
        return sum(
            item.reason_code
            in {
                RuntimeContractMismatchRefineryReason.EVIDENCE_UPDATED,
                RuntimeContractMismatchRefineryReason.STALE_FINDING,
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


def _map_base_reason(
    reason: ContractMismatchRefineryReason | str,
) -> RuntimeContractMismatchRefineryReason:
    value = getattr(reason, "value", reason)
    try:
        return RuntimeContractMismatchRefineryReason(str(value))
    except ValueError:
        return RuntimeContractMismatchRefineryReason.MALFORMED_PACKET


def _looks_like_packet(raw: Mapping[str, Any]) -> bool:
    if "base_packet" in raw and "context_capsule" in raw:
        return True
    interface = raw.get("interface")
    schema = raw.get("schema")
    if interface in _PACKET_SCHEMA_MARKERS or schema in _PACKET_SCHEMA_MARKERS:
        return True
    # Compact serialized CodeEditPacket@1 often nests the interface under
    # base_packet; also accept records that already carry packet_id + finding_id
    # with acceptance scope (validation/reproof), which raw findings lack.
    if (
        isinstance(raw.get("packet_id"), str)
        and isinstance(raw.get("finding_id"), str)
        and isinstance(raw.get("finding_record_id"), str)
        and (
            "validation_commands" in raw
            or isinstance(raw.get("context_capsule"), Mapping)
        )
    ):
        return True
    return False


def _admit_non_packet(
    raw: Mapping[str, Any],
    *,
    current_snapshot_id: str,
) -> RuntimeContractMismatchRefineryDecision | None:
    """Return a rejection decision for non-packet findings, or None to continue.

    Unsupported, stale, and unknown-only records are explicitly not
    implementation-ready.  Actionable refuted non-packets are left for the
    base refinery, which fail-closes on missing CodeEditPacket structure.
    """

    state_raw = raw.get("state")
    if not isinstance(state_raw, str):
        return None
    state = state_raw.strip().casefold()
    if state not in _NOT_IMPLEMENTATION_READY_STATES and state not in _STALE_STATES:
        return None
    if _looks_like_packet(raw):
        return None

    finding_id = _one_line(raw.get("finding_id"), "finding_id")
    finding_snapshot = raw.get("snapshot_id") or raw.get("snapshot_root")
    if isinstance(finding_snapshot, str) and finding_snapshot:
        snapshot = _one_line(finding_snapshot, "finding snapshot identity")
    else:
        snapshot = current_snapshot_id

    reason_code_value = raw.get("reason_code") or state
    reason_code = _one_line(str(reason_code_value), "reason_code")
    if "contract_id" in raw:
        _one_line(raw.get("contract_id"), "contract_id")
    if "affected_paths" in raw:
        _strings(
            raw.get("affected_paths"),
            "affected_paths",
            required=True,
            maximum=HARD_MAX_PATHS,
        )
    if "counterexample" in raw:
        _json_value(raw.get("counterexample"), "counterexample")

    if snapshot != current_snapshot_id or state in _STALE_STATES:
        return RuntimeContractMismatchRefineryDecision(
            finding_id=finding_id,
            task_id="",
            reason_code=RuntimeContractMismatchRefineryReason.STALE_FINDING,
            detail=(
                "stale or snapshot-mismatched runtime finding is not "
                f"implementation-ready: {reason_code}"
            ),
        )
    if state in {"unknown", "open", "pending"} or state.startswith("unknown"):
        return RuntimeContractMismatchRefineryDecision(
            finding_id=finding_id,
            task_id="",
            reason_code=RuntimeContractMismatchRefineryReason.UNKNOWN_ONLY,
            detail=(
                "unknown-only runtime finding is not implementation-ready: "
                f"{reason_code}"
            ),
        )
    return RuntimeContractMismatchRefineryDecision(
        finding_id=finding_id,
        task_id="",
        reason_code=RuntimeContractMismatchRefineryReason.UNSUPPORTED_FINDING,
        detail=(
            "explicitly non-actionable runtime finding is not "
            f"implementation-ready: {reason_code}"
        ),
    )


def _task_track(task: ContractRepairTask) -> str:
    if task.goal_id == DEFAULT_RUNTIME_GOAL_ID:
        return DEFAULT_RUNTIME_TRACK
    return BASELINE_TRACK


def _record_text(task: ContractRepairTask) -> str:
    encoded = base64.urlsafe_b64encode(
        canonical_json_bytes(task.to_dict())
    ).decode("ascii").rstrip("=")
    return _TASK_RECORD_PREFIX + encoded + _TASK_RECORD_SUFFIX


def render_runtime_contract_repair_board(
    tasks: Sequence[ContractRepairTask],
    *,
    board_namespace: str = DEFAULT_BOARD_NAMESPACE,
    last_refinery_epoch: int = 0,
) -> str:
    """Render a stable board that preserves baseline and runtime task tracks."""

    namespace = _one_line(board_namespace, "board_namespace")
    epoch = _bounded_int(
        last_refinery_epoch, "last_refinery_epoch", maximum=2**63 - 1
    )
    ordered = tuple(sorted(tasks, key=lambda item: item.task_id))
    if len({item.task_id for item in ordered}) != len(ordered):
        raise RuntimeContractMismatchRefineryError(
            "cannot render duplicate repair task IDs",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    if any(item.board_namespace != namespace for item in ordered):
        raise RuntimeContractMismatchRefineryError(
            "repair task belongs to another board namespace",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_BOARD,
        )
    lines = [
        "# Generated ipfs_accelerate_py contract repairs",
        "",
        f"- Schema: {CONTRACT_REPAIR_BOARD_SCHEMA}",
        f"- Interface: {RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE}",
        f"- Board namespace: {namespace}",
        (
            "- Source: admitted CodeEditPacket@1 records only "
            "(runtime counterexample clusters; no repository corpus)"
        ),
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
                f"- Track: {_task_track(task)}",
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


def build_runtime_contract_mismatch_triage(
    result: RuntimeContractMismatchRefineryResult,
    *,
    current_snapshot_id: str,
    owner: str,
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return sealed, non-authoritative accounting for one runtime refill."""

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
        "schema": RUNTIME_CONTRACT_MISMATCH_TRIAGE_SCHEMA,
        "interface": RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE,
        "snapshot_id": _one_line(current_snapshot_id, "current_snapshot_id"),
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


class RuntimeContractMismatchRefinery:
    """Append runtime counterexample clusters onto the shared repair board."""

    interface: Final = RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE

    def __init__(
        self, policy: RuntimeContractMismatchRefineryPolicy | None = None
    ) -> None:
        self.policy = policy or RuntimeContractMismatchRefineryPolicy()
        self._base = ContractMismatchRefinery(self.policy.to_base_policy())

    def refine(
        self,
        packets: Iterable[McpContractEditPacket | Mapping[str, Any]],
        *,
        current_snapshot_id: str,
        existing_board: str = "",
        current_open_work: int = 0,
        now_epoch: int = 0,
        current_finding_record_ids: Mapping[str, str] | None = None,
    ) -> RuntimeContractMismatchRefineryResult:
        """Project runtime packets without scanning source or asserting completion."""

        snapshot = _one_line(current_snapshot_id, "current_snapshot_id")
        open_work = _bounded_int(
            current_open_work,
            "current_open_work",
            maximum=HARD_MAX_OPEN_WORK,
        )
        now = _bounded_int(now_epoch, "now_epoch", maximum=2**63 - 1)

        # Preserve historical board content before projection.
        parsed_existing = parse_contract_repair_board(
            existing_board, board_namespace=self.policy.board_namespace
        )

        pre_decisions: list[RuntimeContractMismatchRefineryDecision] = []
        admitted: list[McpContractEditPacket | Mapping[str, Any]] = []
        for raw in packets:
            if isinstance(raw, McpContractEditPacket):
                admitted.append(raw)
                continue
            if not isinstance(raw, Mapping):
                pre_decisions.append(
                    RuntimeContractMismatchRefineryDecision(
                        finding_id="",
                        task_id="",
                        reason_code=(
                            RuntimeContractMismatchRefineryReason.MALFORMED_PACKET
                        ),
                        detail="runtime packet input must be an object",
                    )
                )
                continue
            try:
                rejection = _admit_non_packet(
                    raw, current_snapshot_id=snapshot
                )
            except RuntimeContractMismatchRefineryError as exc:
                pre_decisions.append(
                    RuntimeContractMismatchRefineryDecision(
                        finding_id="",
                        task_id="",
                        reason_code=_map_base_reason(exc.reason_code),
                        detail=str(exc),
                    )
                )
                continue
            if rejection is not None:
                pre_decisions.append(rejection)
                continue
            admitted.append(raw)

        base_result = self._base.refine(
            admitted,
            current_snapshot_id=snapshot,
            existing_board=existing_board,
            current_open_work=open_work,
            now_epoch=now,
            current_finding_record_ids=current_finding_record_ids,
        )

        # Stamp the runtime goal on identities admitted in this run; preserve
        # historical baseline goal ids when appending to an existing board.
        admitted_findings: set[str] = set()
        for item in admitted:
            if isinstance(item, McpContractEditPacket):
                admitted_findings.add(item.finding_id)
            elif isinstance(item, Mapping) and isinstance(
                item.get("finding_id"), str
            ):
                if _looks_like_packet(item):
                    admitted_findings.add(item["finding_id"])
        rewritten: list[ContractRepairTask] = []
        for task in base_result.tasks:
            if task.finding_id in admitted_findings:
                rewritten.append(replace(task, goal_id=self.policy.goal_id))
            else:
                rewritten.append(task)
        tasks = tuple(sorted(rewritten, key=lambda item: item.task_id))

        last_epoch = base_result.last_refinery_epoch
        markdown = render_runtime_contract_repair_board(
            tasks,
            board_namespace=self.policy.board_namespace,
            last_refinery_epoch=last_epoch,
        )

        decisions = list(pre_decisions)
        for decision in base_result.decisions:
            decisions.append(
                RuntimeContractMismatchRefineryDecision(
                    finding_id=decision.finding_id,
                    task_id=decision.task_id,
                    reason_code=_map_base_reason(decision.reason_code),
                    detail=decision.detail,
                )
            )

        # Historical open work accounting must include preserved baseline tasks.
        final_open = max(open_work, sum(task.is_open for task in tasks))
        initial_open = max(
            open_work,
            sum(task.is_open for task in parsed_existing.tasks),
        )
        # Prefer base initial when it already observed pre-blocked stale work.
        if base_result.initial_open_work:
            initial_open = max(initial_open, base_result.initial_open_work)

        return RuntimeContractMismatchRefineryResult(
            tasks=tasks,
            decisions=tuple(decisions),
            markdown=markdown,
            initial_open_work=initial_open,
            final_open_work=final_open,
            max_open_work=self.policy.max_open_work,
            last_refinery_epoch=last_epoch,
        )


def refine_runtime_contract_mismatch_packets(
    packets: Iterable[McpContractEditPacket | Mapping[str, Any]],
    *,
    current_snapshot_id: str,
    existing_board: str = "",
    current_open_work: int = 0,
    now_epoch: int = 0,
    policy: RuntimeContractMismatchRefineryPolicy | None = None,
    current_finding_record_ids: Mapping[str, str] | None = None,
) -> RuntimeContractMismatchRefineryResult:
    """Functional entry point for runtime refill integration."""

    return RuntimeContractMismatchRefinery(policy).refine(
        packets,
        current_snapshot_id=current_snapshot_id,
        existing_board=existing_board,
        current_open_work=current_open_work,
        now_epoch=now_epoch,
        current_finding_record_ids=current_finding_record_ids,
    )


def _load_packet_document(
    path: Path,
) -> tuple[tuple[Mapping[str, Any], ...], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractMismatchRefineryError(
            f"unable to load packet records: {exc}",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
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
        raise RuntimeContractMismatchRefineryError(
            "packet input must be a record or sequence of records",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    if not all(isinstance(item, Mapping) for item in payload):
        raise RuntimeContractMismatchRefineryError(
            "packet input contains a non-object record",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
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
        raise RuntimeContractMismatchRefineryError(
            "packet input contains conflicting snapshot identity values",
            reason_code=RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
        )
    inferred_snapshot_id = (
        next(iter(inferred_snapshot_ids)) if inferred_snapshot_ids else None
    )
    return records, inferred_snapshot_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append runtime counterexample clusters onto the accelerator "
            "contract repair board."
        )
    )
    parser.add_argument(
        "--findings",
        "--packets",
        dest="packets_path",
        required=True,
        help=(
            "JSON CodeEditPacket@1 / runtime finding record(s); "
            "no repository source is scanned."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--triage-output",
        help=(
            "Optional path for sealed non-authoritative runtime triage "
            "accounting (runtime_triage.json)."
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
    parser.add_argument(
        "--goal-id",
        default=DEFAULT_RUNTIME_GOAL_ID,
        help="Goal identity stamped onto newly projected runtime tasks.",
    )
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
    except RuntimeContractMismatchRefineryError as exc:
        parser.error(str(exc))
    current_snapshot_id = args.snapshot or inferred_snapshot_id
    if not current_snapshot_id:
        parser.error(
            "--snapshot is required when the packet document does not "
            "contain exactly one snapshot identity"
        )
    result = refine_runtime_contract_mismatch_packets(
        packet_records,
        current_snapshot_id=current_snapshot_id,
        existing_board=existing,
        current_open_work=args.current_open_work,
        now_epoch=int(time.time()) if args.now_epoch is None else args.now_epoch,
        policy=RuntimeContractMismatchRefineryPolicy(
            max_open_work=args.max_open_work,
            max_findings_per_run=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
            goal_id=args.goal_id,
        ),
    )
    structural_rejections = tuple(
        decision
        for decision in result.decisions
        if decision.reason_code
        in {
            RuntimeContractMismatchRefineryReason.MALFORMED_PACKET,
            RuntimeContractMismatchRefineryReason.MALFORMED_PATH,
            RuntimeContractMismatchRefineryReason.OWNER_MISMATCH,
            RuntimeContractMismatchRefineryReason.MALFORMED_DEPENDENCY,
            RuntimeContractMismatchRefineryReason.SELF_DEPENDENCY,
            RuntimeContractMismatchRefineryReason.MALFORMED_BOARD,
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
        triage = build_runtime_contract_mismatch_triage(
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
BacklogRefinery = RuntimeContractMismatchRefinery
MarkdownTaskSource = ContractRepairTask
RuntimeContractRepairTask = ContractRepairTask


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
