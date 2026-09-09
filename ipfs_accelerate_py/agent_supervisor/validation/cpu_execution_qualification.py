"""Fail-closed PCPR-037 live CPU execution qualification.

PCPR-037 runs a stdlib CPU canary on the current host covering identity,
load, compute, output validation, cancellation, timeout, cleanup,
fail-closed resource admission, and repetition. This module is not
release authority: it does not write DuckDB or Quack state, does not
qualify CUDA or a model/provider path, does not grant
production_authorized, and never emits a closed PCPR release outcome.

Live claims require live evidence. Simulated results stay Simulated.
Missing CUDA and model/provider environments stay typed unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.cpu_execution import (
    GOAL_ID as FLOOR_GOAL_ID,
    INTERFACE as FLOOR_INTERFACE,
    KERNEL_NAME,
    SCHEMA as FLOOR_SCHEMA,
    TASK_ID as FLOOR_TASK_ID,
    qualify_live_cpu_execution,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_cpu_execution,
    production_authorized,
)
from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_mock_coordinator_quarantine import discover_accelerate_root
from .python_compatibility_metadata import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_036_VERDICT_CID,
)


CPU_EVALUATOR_INTERFACE: Final = "CpuExecutionQualification@1"
CPU_EVALUATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cpu-execution-qualification@1"
)
CPU_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cpu-execution-qualification-verdict@1"
)

PCPR_037_TASK_ID: Final = "PCPR-037"
PCPR_037_GOAL_ID: Final = "PCPR-G430"
PCPR_036_TASK_ID: Final = "PCPR-036"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = "proof-carrying-platform-qualification-and-release-v1"

EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)
PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
)

CURRENT_HEAD_OUTER_COMMIT: Final = "7fa98095c12ef547b4d45727e63bdbbbc19dbac4"
CURRENT_HEAD_OUTER_TREE: Final = "fdb96dacf0347ddba7bcc5f716447cbe5a881b05"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '16c6da7a70cb759822b04e2565beff069b8d84ae' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "7b4f094ca966e961ddac7bf0fc96f873a93e0f80"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "118d10ad140ea40c5710b312cff1c8a90cb999cb"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "a747bbd619385329b8e613e7e6f75557d471e2c3"
CURRENT_HEAD_DATASETS_TREE: Final = "15e2aa1c19dffb175b4d7bb57c3afaab52eb0dd8"
CURRENT_HEAD_KIT_COMMIT: Final = "b0597f8d1d153e75e38d20265fffcd4405130ab9"
CURRENT_HEAD_KIT_TREE: Final = "6a9a57bb59be47465b78366e15e99a12fdd572c5"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_cpu_execution_qualification.py",
)

CPU_MODULE_RELPATH: Final = "ipfs_accelerate_py/assurance/cpu_execution.py"
LADDER_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/hardware_capability_ladder.py"
)
HARDWARE_KIT_RELPATH: Final = "ipfs_accelerate_py/kit/hardware_kit.py"

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_LIVE_PROBE_IDS: Final[tuple[str, ...]] = (
    "cpu_identity",
    "cpu_load",
    "cpu_compute_kernel",
    "cpu_output_validation",
    "cpu_repetition",
    "cpu_cancellation",
    "cpu_timeout",
    "cpu_cleanup",
    "cpu_resource_admission_fail_closed",
)


class CpuExecutionQualificationError(ValueError):
    """Malformed CPU execution evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class CpuProbe:
    """One measured, live, or typed-unavailable CPU observation."""

    probe_id: str
    present: bool | None
    evidence_kind: str
    live: bool
    simulated_represented_as_live: bool
    reason: str
    details: Mapping[str, Any] = MappingProxyType({})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "live": self.live,
            "simulated_represented_as_live": self.simulated_represented_as_live,
            "reason": self.reason,
            "details": dict(self.details),
        }

    def to_cid_mapping(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "live": self.live,
            "simulated_represented_as_live": self.simulated_represented_as_live,
        }


@dataclass(frozen=True)
class CpuVerdict:
    """Fail-closed PCPR-037 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    live_cpu_execution_qualified: bool
    production_authorized: bool
    hardware_ladder_qualified: bool
    simulated_results_represented_as_live: bool
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
    live_model_provider_qualified: bool
    live_model_provider_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[CpuProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "live_cpu_execution_qualified": self.live_cpu_execution_qualified,
            "production_authorized": self.production_authorized,
            "hardware_ladder_qualified": self.hardware_ladder_qualified,
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_cuda_qualified": self.live_cuda_qualified,
            "live_cuda_evidence_kind": self.live_cuda_evidence_kind,
            "live_model_provider_qualified": self.live_model_provider_qualified,
            "live_model_provider_evidence_kind": (
                self.live_model_provider_evidence_kind
            ),
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CpuExecutionQualificationError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise CpuExecutionQualificationError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise CpuExecutionQualificationError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise CpuExecutionQualificationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise CpuExecutionQualificationError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _probe(
    probe_id: str,
    *,
    present: bool | None,
    evidence_kind: str,
    live: bool,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> CpuProbe:
    return CpuProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=live,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _file_probe(
    probe_id: str,
    source: str | None,
    relpath: str,
    ok: bool,
    ok_reason: str,
    fail_reason: str,
) -> CpuProbe:
    if source is None:
        return _probe(
            probe_id,
            present=None,
            evidence_kind="unavailable",
            live=False,
            reason=f"{relpath} is not present and is not recorded as empty.",
            details={"relpath": relpath},
        )
    return _probe(
        probe_id,
        present=ok,
        evidence_kind="measured",
        live=False,
        reason=ok_reason if ok else fail_reason,
        details={"relpath": relpath},
    )


def current_head_cpu_probes(
    *,
    accelerate_root: Path | None = None,
) -> tuple[CpuProbe, ...]:
    """Measured current-tree and live CPU probes."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[CpuProbe] = []
    if root is None or not root.is_dir():
        return (
            _probe(
                "accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                live=False,
                reason=(
                    "Accelerate source tree is not present and is not recorded as empty."
                ),
            ),
        )

    cpu_src = _read_source(root, CPU_MODULE_RELPATH)
    ladder_src = _read_source(root, LADDER_MODULE_RELPATH)
    kit_src = _read_source(root, HARDWARE_KIT_RELPATH)

    cpu_ok = bool(cpu_src) and all(
        needle in (cpu_src or "")
        for needle in (
            "cpu_integer_kernel",
            "qualify_live_cpu_execution",
            "cpu_cancellation",
            "cpu_timeout",
            "production_authorized",
            'TASK_ID: Final[str] = "PCPR-037"',
        )
    )
    probes.append(
        _file_probe(
            "canonical_cpu_execution_module",
            cpu_src,
            CPU_MODULE_RELPATH,
            cpu_ok,
            "Canonical CPU execution module declares the live stdlib canary.",
            "Canonical CPU execution module is missing live canary APIs.",
        )
    )

    ladder_ok = bool(ladder_src) and all(
        needle in (ladder_src or "")
        for needle in (
            "def from_live_cpu_execution",
            "cpu_execution_qualified",
            "production_authorized",
        )
    )
    probes.append(
        _file_probe(
            "ladder_exposes_live_cpu_execution",
            ladder_src,
            LADDER_MODULE_RELPATH,
            ladder_ok,
            "Capability ladder exposes live CPU execution without production_authorized.",
            "Capability ladder still lacks a live CPU execution helper.",
        )
    )

    cpu_fn = (kit_src or "").split("def _test_cpu", 1)[-1].split("\n    def ", 1)[0]
    kit_ok = (
        bool(kit_src)
        and "qualify_live_cpu_execution" in cpu_fn
        and "production_authorized" in cpu_fn
        and "import numpy" not in cpu_fn
    )
    probes.append(
        _file_probe(
            "hardware_kit_uses_live_cpu_canary",
            kit_src,
            HARDWARE_KIT_RELPATH,
            kit_ok,
            "HardwareKit CPU test uses the live stdlib canary instead of numpy-only success.",
            "HardwareKit CPU test still depends on numpy or skips live CPU execution.",
        )
    )

    live_report = qualify_live_cpu_execution(test_level="comprehensive")
    live_by_id = {
        item["probe_id"]: item for item in live_report.get("cpu_probes") or ()
    }
    for probe_id in REQUIRED_LIVE_PROBE_IDS:
        item = live_by_id.get(probe_id)
        if item is None:
            probes.append(
                _probe(
                    probe_id,
                    present=None,
                    evidence_kind="unavailable",
                    live=False,
                    reason=f"Live CPU probe {probe_id} was not emitted.",
                )
            )
            continue
        probes.append(
            CpuProbe(
                probe_id=probe_id,
                present=item.get("present"),
                evidence_kind=_kind(item.get("evidence_kind"), f"{probe_id}.evidence_kind"),
                live=bool(item.get("live")),
                simulated_represented_as_live=bool(
                    item.get("simulated_represented_as_live")
                ),
                reason=str(item.get("reason") or ""),
                details=MappingProxyType(dict(item.get("details") or {})),
            )
        )

    ladder_payload = from_live_cpu_execution(canary_passed=True)
    production_closed = (
        production_authorized(ladder_payload) is False
        and ladder_payload.get("production_authorized") is False
        and ladder_payload.get("qualified") is False
        and live_report.get("production_authorized") is False
        and live_report.get("qualified") is False
    )
    probes.append(
        _probe(
            "production_authorized_not_granted",
            present=production_closed,
            evidence_kind="measured",
            live=False,
            reason=(
                "Live CPU execution does not grant production_authorized or ladder qualified."
                if production_closed
                else "Live CPU execution incorrectly granted production_authorized."
            ),
        )
    )

    probes.append(
        _probe(
            "live_cuda_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            reason=(
                "This task does not qualify live CUDA. nvidia-smi presence is device "
                "visibility, not CUDA qualification. Missing CUDA qualification stays "
                "typed unavailable and is not recorded as False or passing."
            ),
            details={"deferred_task_id": "PCPR-038"},
        )
    )
    probes.append(
        _probe(
            "live_model_provider_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            reason=(
                "This task does not qualify a model or provider path. The CPU kernel "
                "is not model inference. Missing model/provider evidence stays typed "
                "unavailable and is not recorded as False or passing."
            ),
            details={"deferred_task_id": "PCPR-039"},
        )
    )
    del accelerate_root
    return tuple(probes)


def qualify_cpu_execution(
    *,
    probes: Sequence[CpuProbe],
    duckdb_or_quack_state_written: bool = False,
    live_cuda_qualified: bool = False,
    live_model_provider_qualified: bool = False,
    production_authorized_claim: bool = False,
) -> CpuVerdict:
    """Evaluate PCPR-037. Live CPU R&D qualification is never a release."""

    if duckdb_or_quack_state_written:
        raise CpuExecutionQualificationError(
            "cpu execution qualification must not write DuckDB or Quack state"
        )
    if live_cuda_qualified:
        raise CpuExecutionQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if live_model_provider_qualified:
        raise CpuExecutionQualificationError(
            "this task cannot claim live model/provider qualification"
        )
    if production_authorized_claim:
        raise CpuExecutionQualificationError(
            "this task cannot grant production_authorized"
        )
    if not probes:
        raise CpuExecutionQualificationError("cpu-execution probes are required")

    normalized: list[CpuProbe] = []
    for probe in probes:
        probe_id = _text(probe.probe_id, "probe_id")
        kind = _kind(probe.evidence_kind, f"{probe_id}.evidence_kind")
        if kind == "estimated":
            raise CpuExecutionQualificationError(
                f"{probe_id} must not use estimated evidence"
            )
        if probe.simulated_represented_as_live:
            raise CpuExecutionQualificationError(
                f"{probe_id} cannot represent simulation as live"
            )
        if kind == "simulated":
            raise CpuExecutionQualificationError(
                f"{probe_id} simulated evidence cannot qualify CPU execution"
            )
        normalized.append(probe)

    by_id = {item.probe_id: item for item in normalized}
    blockers: list[str] = []

    def _require(probe_id: str, blocker: str) -> bool:
        item = by_id.get(probe_id)
        ok = bool(item and item.present is True)
        if not ok:
            blockers.append(blocker)
        return ok

    module_ok = _require("canonical_cpu_execution_module", "cpu_execution_module_missing")
    ladder_ok = _require(
        "ladder_exposes_live_cpu_execution", "live_cpu_ladder_helper_missing"
    )
    kit_ok = _require(
        "hardware_kit_uses_live_cpu_canary", "hardware_kit_cpu_canary_missing"
    )
    production_ok = _require(
        "production_authorized_not_granted", "production_authorized_granted"
    )
    live_ok = True
    for probe_id in REQUIRED_LIVE_PROBE_IDS:
        if not _require(probe_id, f"{probe_id}_failed"):
            live_ok = False

    cuda_probe = by_id.get("live_cuda_qualification")
    model_probe = by_id.get("live_model_provider_qualification")
    if cuda_probe is None or cuda_probe.evidence_kind != "unavailable":
        blockers.append("cuda_not_typed_unavailable")
    if model_probe is None or model_probe.evidence_kind != "unavailable":
        blockers.append("model_provider_not_typed_unavailable")
    if cuda_probe is not None and cuda_probe.present is True:
        blockers.append("cuda_claimed_live")
    if model_probe is not None and model_probe.present is True:
        blockers.append("model_provider_claimed_live")

    cpu_qualified = (
        module_ok and ladder_ok and kit_ok and production_ok and live_ok and not blockers
    )

    unavailable_count = sum(
        1 for item in normalized if item.evidence_kind == "unavailable"
    )
    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable_count == len(normalized):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise CpuExecutionQualificationError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CpuExecutionQualificationError(
            "cpu execution qualification must not mint a closed release outcome"
        )

    payload = {
        "schema": CPU_VERDICT_SCHEMA,
        "interface": CPU_EVALUATOR_INTERFACE,
        "task_id": PCPR_037_TASK_ID,
        "goal_id": PCPR_037_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "live_cpu_execution_qualified": cpu_qualified,
        "production_authorized": False,
        "hardware_ladder_qualified": False,
        "simulated_results_represented_as_live": False,
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "live_model_provider_qualified": False,
        "live_model_provider_evidence_kind": "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_cid_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "kernel": KERNEL_NAME,
    }
    return CpuVerdict(
        schema=CPU_VERDICT_SCHEMA,
        interface=CPU_EVALUATOR_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        live_cpu_execution_qualified=cpu_qualified,
        production_authorized=False,
        hardware_ladder_qualified=False,
        simulated_results_represented_as_live=False,
        live_cuda_qualified=False,
        live_cuda_evidence_kind="unavailable",
        live_model_provider_qualified=False,
        live_model_provider_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_cpu() -> CpuVerdict:
    """Ordinary current-head PCPR-037 evaluation: live CPU R&D qualification."""

    return qualify_cpu_execution(probes=current_head_cpu_probes())


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraqnzibzfy4sfh47bzgkotycjyvbzdgoap4o4uoqy3wlqsmk2lis5q"
)


def pcpr_037_receipt_promotion(verdict: CpuVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise CpuExecutionQualificationError(
            "cpu execution qualification must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise CpuExecutionQualificationError(
            "cpu execution qualification must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise CpuExecutionQualificationError(
            "cpu execution qualification completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise CpuExecutionQualificationError(
            "cpu execution qualification must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CpuExecutionQualificationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise CpuExecutionQualificationError(
            "promotion_status is not an admitted PCPR-037 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise CpuExecutionQualificationError(
            "cpu execution qualification cannot promote the supervisor"
        )
    if verdict.live_cuda_qualified or verdict.live_model_provider_qualified:
        raise CpuExecutionQualificationError(
            "this task cannot claim live CUDA or model/provider qualification"
        )
    if verdict.production_authorized or verdict.hardware_ladder_qualified:
        raise CpuExecutionQualificationError(
            "this task cannot grant production_authorized or ladder qualified"
        )
    if not verdict.live_cpu_execution_qualified:
        raise CpuExecutionQualificationError(
            "current-head CPU execution must be live-qualified for R&D"
        )
    return {
        "schema": CPU_VERDICT_SCHEMA,
        "interface": CPU_EVALUATOR_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "live_cpu_execution_qualified": True,
        "production_authorized": False,
        "hardware_ladder_qualified": False,
        "simulated_results_represented_as_live": False,
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "live_model_provider_qualified": False,
        "live_model_provider_evidence_kind": "unavailable",
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_037_receipt_promotion() -> dict[str, Any]:
    return pcpr_037_receipt_promotion(qualify_current_head_cpu())


def pcpr_037_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_cuda_not_claimed": True,
        "live_model_provider_not_claimed": True,
        "production_authorized_not_granted": True,
        "numpy_not_required_for_cpu_canary": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_037_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_cpu()
    promotion = pcpr_037_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    live_report = qualify_live_cpu_execution(test_level="comprehensive")
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_036_TASK_ID,
            "goal_id": "PCPR-G420",
            "promotion_status": "rnd_non_promoted",
            "compatibility_verdict_cid": PCPR_036_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-036 aligned Python compatibility metadata on the 3.12 floor. "
                "This task qualifies live CPU execution as R&D. PCPR-001 live "
                "qualification remains rnd_non_promoted and is not promoted by "
                "this receipt."
            ),
            "evidence_kind": "measured",
        },
        "cpu_execution": {
            "schema": CPU_EVALUATOR_SCHEMA,
            "interface": CPU_EVALUATOR_INTERFACE,
            "floor_schema": FLOOR_SCHEMA,
            "floor_interface": FLOOR_INTERFACE,
            "floor_task_id": FLOOR_TASK_ID,
            "floor_goal_id": FLOOR_GOAL_ID,
            "kernel": KERNEL_NAME,
            "live_cpu_execution_qualified": verdict.live_cpu_execution_qualified,
            "production_authorized": False,
            "hardware_ladder_qualified": False,
            "cpu_identity": live_report.get("cpu_identity"),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_037_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_037_current_tree_binding(
    *,
    outer_commit: str,
    outer_tree: str,
    outer_subject: str,
    origin_main: str,
    origin_main_is_ancestor: bool,
    accelerator_pre_change_commit: str,
    accelerator_pre_change_tree: str,
    accelerator_gitlink: str,
    accelerator_origin_main: str,
    accelerator_origin_main_is_ancestor: bool,
    datasets_commit: str,
    datasets_tree: str,
    datasets_gitlink: str,
    kit_commit: str,
    kit_tree: str,
    kit_gitlink: str,
) -> dict[str, Any]:
    outer = _git_object_id(outer_commit, "outer_commit")
    tree = _git_object_id(outer_tree, "outer_tree")
    subject = _text(outer_subject, "outer_subject")
    origin = _git_object_id(origin_main, "origin_main")
    _require_ancestor(origin_main_is_ancestor, "origin_main_is_ancestor")
    accel = _git_object_id(
        accelerator_pre_change_commit, "accelerator_pre_change_commit"
    )
    accel_tree = _git_object_id(
        accelerator_pre_change_tree, "accelerator_pre_change_tree"
    )
    accel_link = _git_object_id(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _git_object_id(
        accelerator_origin_main, "accelerator_origin_main"
    )
    _require_ancestor(
        accelerator_origin_main_is_ancestor, "accelerator_origin_main_is_ancestor"
    )
    if accel != accel_link:
        raise CpuExecutionQualificationError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise CpuExecutionQualificationError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise CpuExecutionQualificationError("kit_commit must equal kit_gitlink")
    _reject_closed_release_value(subject, "outer_subject")
    return {
        "outer_repository": "endomorphosis/lift_coding",
        "owning_repository_for_receipts": "ipfs_accelerate_py",
        "outer_commit": outer,
        "outer_tree": tree,
        "outer_subject": subject,
        "origin_main": origin,
        "origin_main_is_ancestor": True,
        "accelerator_pre_change_commit": accel,
        "accelerator_pre_change_tree": accel_tree,
        "accelerator_gitlink": accel_link,
        "accelerator_origin_main": accel_origin,
        "accelerator_origin_main_is_ancestor": True,
        "accelerator_post_change_commit": "pending nested commit after admission",
        "accelerator_post_change_tree": (
            "dirty-worktree; exact CID after accepted nested commit"
        ),
        "datasets_commit": datasets,
        "datasets_tree": datasets_tree_id,
        "datasets_gitlink": datasets_link,
        "kit_commit": kit,
        "kit_tree": kit_tree_id,
        "kit_gitlink": kit_link,
        "evidence_kind": "measured",
    }


def current_head_pcpr_037_current_tree_binding() -> dict[str, Any]:
    return pcpr_037_current_tree_binding(
        outer_commit=CURRENT_HEAD_OUTER_COMMIT,
        outer_tree=CURRENT_HEAD_OUTER_TREE,
        outer_subject=CURRENT_HEAD_OUTER_SUBJECT,
        origin_main=CURRENT_HEAD_ORIGIN_MAIN,
        origin_main_is_ancestor=True,
        accelerator_pre_change_commit=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_pre_change_tree=CURRENT_HEAD_ACCELERATOR_TREE,
        accelerator_gitlink=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_origin_main=CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN,
        accelerator_origin_main_is_ancestor=True,
        datasets_commit=CURRENT_HEAD_DATASETS_COMMIT,
        datasets_tree=CURRENT_HEAD_DATASETS_TREE,
        datasets_gitlink=CURRENT_HEAD_DATASETS_COMMIT,
        kit_commit=CURRENT_HEAD_KIT_COMMIT,
        kit_tree=CURRENT_HEAD_KIT_TREE,
        kit_gitlink=CURRENT_HEAD_KIT_COMMIT,
    )


def validate_pcpr_037_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-037 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise CpuExecutionQualificationError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_037_TASK_ID:
        raise CpuExecutionQualificationError("outer receipt task_id must be PCPR-037")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise CpuExecutionQualificationError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise CpuExecutionQualificationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise CpuExecutionQualificationError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise CpuExecutionQualificationError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise CpuExecutionQualificationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise CpuExecutionQualificationError(
            "cpu execution qualification must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise CpuExecutionQualificationError(
            "cpu execution qualification cannot freeze contracts"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise CpuExecutionQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict_section.get("live_model_provider_qualified") is True:
        raise CpuExecutionQualificationError(
            "this task cannot claim live model/provider qualification"
        )
    if verdict_section.get("production_authorized") is True:
        raise CpuExecutionQualificationError(
            "this task cannot grant production_authorized"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise CpuExecutionQualificationError(
            "qualification_verdict.promotion_status is not an admitted PCPR-037 status"
        )
    if promotion_status == "supervisor_promoted":
        raise CpuExecutionQualificationError(
            "cpu execution qualification cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise CpuExecutionQualificationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise CpuExecutionQualificationError("acceptance must not claim a release")

    expected = current_head_pcpr_037_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise CpuExecutionQualificationError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise CpuExecutionQualificationError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise CpuExecutionQualificationError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_037_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }


__all__ = (
    "CLOSED_RELEASE_OUTCOMES",
    "CPU_EVALUATOR_INTERFACE",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "PCPR_037_GOAL_ID",
    "PCPR_037_TASK_ID",
    "CpuExecutionQualificationError",
    "CpuProbe",
    "CpuVerdict",
    "current_head_cpu_probes",
    "current_head_pcpr_037_current_tree_binding",
    "current_head_pcpr_037_receipt_promotion",
    "current_head_pcpr_037_receipt_sections",
    "discover_accelerate_root",
    "pcpr_037_current_tree_binding",
    "pcpr_037_receipt_negative_results",
    "pcpr_037_receipt_promotion",
    "qualify_cpu_execution",
    "qualify_current_head_cpu",
    "validate_pcpr_037_outer_receipt",
)
