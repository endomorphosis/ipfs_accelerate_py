"""Fail-closed PCPR-039 live model/provider execution qualification.

PCPR-039 runs a live local OpenAI-compatible canary model-server covering
identity, load, inference, output validation, cancellation, timeout,
cleanup, fail-closed resource admission, and repetition, and qualifies one
live llama.cpp local LLM path when present. Torch, transformers, and
HuggingFace Hub downloads are not qualification. This module is not
release authority: it does not write DuckDB or Quack state, does not grant
production_authorized, and never emits a closed PCPR release outcome.

Live claims require live evidence. Simulated results stay Simulated.
Missing llama.cpp stays typed unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.model_provider_execution import (
    CANARY_KERNEL,
    GOAL_ID as FLOOR_GOAL_ID,
    INTERFACE as FLOOR_INTERFACE,
    SCHEMA as FLOOR_SCHEMA,
    TASK_ID as FLOOR_TASK_ID,
    qualify_live_model_provider_execution,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_model_provider_execution,
    production_authorized,
)
from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .cuda_execution_qualification import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_038_VERDICT_CID,
)
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_mock_coordinator_quarantine import discover_accelerate_root


MODEL_EVALUATOR_INTERFACE: Final = "ModelProviderExecutionQualification@1"
MODEL_EVALUATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-provider-execution-qualification@1"
)
MODEL_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "model-provider-execution-qualification-verdict@1"
)

PCPR_039_TASK_ID: Final = "PCPR-039"
PCPR_039_GOAL_ID: Final = "PCPR-G430"
PCPR_038_TASK_ID: Final = "PCPR-038"
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

CURRENT_HEAD_OUTER_COMMIT: Final = "87363035a46d0423462ffb4d00ca19ead9cf7962"
CURRENT_HEAD_OUTER_TREE: Final = "66d5dffe32b76de7d0d775ea8068ed4261595594"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit 'c93e700842310b019bfe3152e63871254a96ae92' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "ef87c714c0bd8020ef24521a282ab3abaca8c3e6"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "368193ddf45f9a92c6210933bba8adedcfc5b29c"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "1635856f512c8f417577091ffd33f2834de0d62d"
CURRENT_HEAD_DATASETS_TREE: Final = "ebe30c3cce6d4f2c74bf6a3be8bbb07bb8c625dd"
CURRENT_HEAD_KIT_COMMIT: Final = "93ce8c123adb5ab85b6735618a08099c4d03db09"
CURRENT_HEAD_KIT_TREE: Final = "753ad198c8616051d5c214190a9fc51b0cc8216b"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_model_provider_execution_qualification.py",
)

MODEL_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/model_provider_execution.py"
)
LADDER_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/hardware_capability_ladder.py"
)
HARDWARE_KIT_RELPATH: Final = "ipfs_accelerate_py/kit/hardware_kit.py"

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_LIVE_PROBE_IDS: Final[tuple[str, ...]] = (
    "model_provider_identity",
    "model_load",
    "model_inference",
    "model_output_validation",
    "model_provider_chat_path",
    "model_repetition",
    "model_cancellation",
    "model_timeout",
    "model_cleanup",
    "model_resource_admission_fail_closed",
    "llamacpp_identity",
    "llamacpp_load",
    "llamacpp_inference",
    "llamacpp_output_validation",
    "llamacpp_repetition",
)


class ModelProviderExecutionQualificationError(ValueError):
    """Malformed model/provider evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class ModelProbe:
    """One measured, live, or typed-unavailable model/provider observation."""

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
class ModelVerdict:
    """Fail-closed PCPR-039 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    live_model_provider_execution_qualified: bool
    production_authorized: bool
    hardware_ladder_qualified: bool
    simulated_results_represented_as_live: bool
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
    live_model_provider_qualified: bool
    live_model_provider_evidence_kind: str
    live_llamacpp_qualified: bool
    live_llamacpp_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[ModelProbe, ...]
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
            "live_model_provider_execution_qualified": (
                self.live_model_provider_execution_qualified
            ),
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
            "live_llamacpp_qualified": self.live_llamacpp_qualified,
            "live_llamacpp_evidence_kind": self.live_llamacpp_evidence_kind,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelProviderExecutionQualificationError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise ModelProviderExecutionQualificationError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise ModelProviderExecutionQualificationError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise ModelProviderExecutionQualificationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise ModelProviderExecutionQualificationError(f"{name} must be true")


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
) -> ModelProbe:
    return ModelProbe(
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
) -> ModelProbe:
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


def current_head_model_provider_probes(
    *,
    accelerate_root: Path | None = None,
) -> tuple[ModelProbe, ...]:
    """Measured current-tree and live model/provider probes."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[ModelProbe] = []
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

    model_src = _read_source(root, MODEL_MODULE_RELPATH)
    ladder_src = _read_source(root, LADDER_MODULE_RELPATH)
    kit_src = _read_source(root, HARDWARE_KIT_RELPATH)

    model_ok = bool(model_src) and all(
        needle in (model_src or "")
        for needle in (
            "canary_embedding",
            "qualify_live_model_provider_execution",
            "model_cancellation",
            "model_timeout",
            "production_authorized",
            'TASK_ID: Final[str] = "PCPR-039"',
            "discover_llamacpp_server",
        )
    )
    probes.append(
        _file_probe(
            "canonical_model_provider_execution_module",
            model_src,
            MODEL_MODULE_RELPATH,
            model_ok,
            "Canonical model/provider execution module declares the live canary and llama.cpp path.",
            "Canonical model/provider execution module is missing live canary APIs.",
        )
    )

    ladder_ok = bool(ladder_src) and all(
        needle in (ladder_src or "")
        for needle in (
            "def from_live_model_provider_execution",
            "model_provider_execution_qualified",
            "production_authorized",
        )
    )
    probes.append(
        _file_probe(
            "ladder_exposes_live_model_provider_execution",
            ladder_src,
            LADDER_MODULE_RELPATH,
            ladder_ok,
            "Capability ladder exposes live model/provider execution without production_authorized.",
            "Capability ladder still lacks a live model/provider execution helper.",
        )
    )

    model_fn = (kit_src or "").split("def _test_model_provider", 1)
    model_fn_body = model_fn[-1].split("\n    def ", 1)[0] if len(model_fn) > 1 else ""
    kit_ok = (
        bool(kit_src)
        and "qualify_live_model_provider_execution" in model_fn_body
        and "production_authorized" in model_fn_body
        and "import torch" not in model_fn_body
        and "import transformers" not in model_fn_body
    )
    probes.append(
        _file_probe(
            "hardware_kit_uses_live_model_provider_canary",
            kit_src,
            HARDWARE_KIT_RELPATH,
            kit_ok,
            "HardwareKit model/provider test uses the live canary instead of torch success.",
            "HardwareKit model/provider test still depends on torch or skips live execution.",
        )
    )

    live_report = qualify_live_model_provider_execution(test_level="comprehensive")
    live_by_id = {
        item["probe_id"]: item
        for item in live_report.get("model_provider_probes") or ()
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
                    reason=f"Live model/provider probe {probe_id} was not emitted.",
                )
            )
            continue
        probes.append(
            ModelProbe(
                probe_id=probe_id,
                present=item.get("present"),
                evidence_kind=_kind(
                    item.get("evidence_kind"), f"{probe_id}.evidence_kind"
                ),
                live=bool(item.get("live")),
                simulated_represented_as_live=bool(
                    item.get("simulated_represented_as_live")
                ),
                reason=str(item.get("reason") or ""),
                details=MappingProxyType(dict(item.get("details") or {})),
            )
        )

    ladder_payload = from_live_model_provider_execution(canary_passed=True)
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
                "Live model/provider execution does not grant production_authorized or ladder qualified."
                if production_closed
                else "Live model/provider execution incorrectly granted production_authorized."
            ),
        )
    )
    probes.append(
        _probe(
            "torch_is_not_qualification",
            present=True,
            evidence_kind="measured",
            live=False,
            reason=(
                "Sealed-environment torch and transformers are not used as model qualification."
            ),
            details={
                "torch_not_used": True,
                "transformers_not_used": True,
                "huggingface_hub_not_used": True,
            },
        )
    )
    probes.append(
        _probe(
            "live_cuda_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            reason=(
                "This task does not re-qualify live CUDA. PCPR-038 owns CUDA "
                "execution. Missing CUDA re-qualification stays typed unavailable "
                "and is not recorded as False or passing."
            ),
            details={"deferred_task_id": "PCPR-038"},
        )
    )
    del accelerate_root
    return tuple(probes)


def qualify_model_provider_execution(
    *,
    probes: Sequence[ModelProbe],
    duckdb_or_quack_state_written: bool = False,
    live_cuda_qualified: bool = False,
    production_authorized_claim: bool = False,
) -> ModelVerdict:
    """Evaluate PCPR-039. Live model/provider R&D qualification is never a release."""

    if duckdb_or_quack_state_written:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification must not write DuckDB or Quack state"
        )
    if live_cuda_qualified:
        raise ModelProviderExecutionQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if production_authorized_claim:
        raise ModelProviderExecutionQualificationError(
            "this task cannot grant production_authorized"
        )
    if not probes:
        raise ModelProviderExecutionQualificationError(
            "model-provider probes are required"
        )

    normalized: list[ModelProbe] = []
    for probe in probes:
        probe_id = _text(probe.probe_id, "probe_id")
        kind = _kind(probe.evidence_kind, f"{probe_id}.evidence_kind")
        if kind == "estimated":
            raise ModelProviderExecutionQualificationError(
                f"{probe_id} must not use estimated evidence"
            )
        if probe.simulated_represented_as_live:
            raise ModelProviderExecutionQualificationError(
                f"{probe_id} cannot represent simulation as live"
            )
        if kind == "simulated":
            raise ModelProviderExecutionQualificationError(
                f"{probe_id} simulated evidence cannot qualify model/provider execution"
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

    module_ok = _require(
        "canonical_model_provider_execution_module",
        "model_provider_execution_module_missing",
    )
    ladder_ok = _require(
        "ladder_exposes_live_model_provider_execution",
        "live_model_provider_ladder_helper_missing",
    )
    kit_ok = _require(
        "hardware_kit_uses_live_model_provider_canary",
        "hardware_kit_model_provider_canary_missing",
    )
    production_ok = _require(
        "production_authorized_not_granted", "production_authorized_granted"
    )
    torch_ok = _require("torch_is_not_qualification", "torch_used_as_qualification")
    live_ok = True
    for probe_id in REQUIRED_LIVE_PROBE_IDS:
        if not _require(probe_id, f"{probe_id}_failed"):
            live_ok = False

    cuda_probe = by_id.get("live_cuda_qualification")
    if cuda_probe is None or cuda_probe.evidence_kind != "unavailable":
        blockers.append("cuda_not_typed_unavailable")
    if cuda_probe is not None and cuda_probe.present is True:
        blockers.append("cuda_claimed_live")

    model_qualified = (
        module_ok
        and ladder_ok
        and kit_ok
        and production_ok
        and torch_ok
        and live_ok
        and not blockers
    )
    llamacpp_ok = all(
        (by_id.get(probe_id) and by_id[probe_id].present is True)
        for probe_id in (
            "llamacpp_identity",
            "llamacpp_load",
            "llamacpp_inference",
            "llamacpp_output_validation",
            "llamacpp_repetition",
        )
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
        raise ModelProviderExecutionQualificationError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification must not mint a closed release outcome"
        )

    payload = {
        "schema": MODEL_VERDICT_SCHEMA,
        "interface": MODEL_EVALUATOR_INTERFACE,
        "task_id": PCPR_039_TASK_ID,
        "goal_id": PCPR_039_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "live_model_provider_execution_qualified": model_qualified,
        "production_authorized": False,
        "hardware_ladder_qualified": False,
        "simulated_results_represented_as_live": False,
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "live_model_provider_qualified": model_qualified,
        "live_model_provider_evidence_kind": "measured" if model_qualified else "unavailable",
        "live_llamacpp_qualified": llamacpp_ok,
        "live_llamacpp_evidence_kind": "measured" if llamacpp_ok else "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_cid_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "kernel": CANARY_KERNEL,
    }
    return ModelVerdict(
        schema=MODEL_VERDICT_SCHEMA,
        interface=MODEL_EVALUATOR_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        live_model_provider_execution_qualified=model_qualified,
        production_authorized=False,
        hardware_ladder_qualified=False,
        simulated_results_represented_as_live=False,
        live_cuda_qualified=False,
        live_cuda_evidence_kind="unavailable",
        live_model_provider_qualified=model_qualified,
        live_model_provider_evidence_kind=payload["live_model_provider_evidence_kind"],
        live_llamacpp_qualified=llamacpp_ok,
        live_llamacpp_evidence_kind=payload["live_llamacpp_evidence_kind"],
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_model_provider() -> ModelVerdict:
    """Ordinary current-head PCPR-039 evaluation: live model/provider R&D qualification."""

    return qualify_model_provider_execution(probes=current_head_model_provider_probes())


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraejqvwpevn5lawzy7pie7cz2atrp62ijbwj4dqx4fvqrxgbr32tla"
)


def pcpr_039_receipt_promotion(verdict: ModelVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise ModelProviderExecutionQualificationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise ModelProviderExecutionQualificationError(
            "promotion_status is not an admitted PCPR-039 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification cannot promote the supervisor"
        )
    if verdict.live_cuda_qualified:
        raise ModelProviderExecutionQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict.production_authorized or verdict.hardware_ladder_qualified:
        raise ModelProviderExecutionQualificationError(
            "this task cannot grant production_authorized or ladder qualified"
        )
    if (
        verdict.promotion_status == "rnd_non_promoted"
        and not verdict.live_model_provider_execution_qualified
    ):
        raise ModelProviderExecutionQualificationError(
            "rnd_non_promoted model/provider execution must be live-qualified for R&D"
        )
    return {
        "schema": MODEL_VERDICT_SCHEMA,
        "interface": MODEL_EVALUATOR_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "live_model_provider_execution_qualified": (
            verdict.live_model_provider_execution_qualified
        ),
        "production_authorized": False,
        "hardware_ladder_qualified": False,
        "simulated_results_represented_as_live": False,
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "live_model_provider_qualified": verdict.live_model_provider_qualified,
        "live_model_provider_evidence_kind": verdict.live_model_provider_evidence_kind,
        "live_llamacpp_qualified": verdict.live_llamacpp_qualified,
        "live_llamacpp_evidence_kind": verdict.live_llamacpp_evidence_kind,
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_039_receipt_promotion() -> dict[str, Any]:
    return pcpr_039_receipt_promotion(qualify_current_head_model_provider())


def pcpr_039_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "torch_is_not_model_qualification": True,
        "transformers_is_not_model_qualification": True,
        "huggingface_hub_download_not_used": True,
        "live_cuda_not_reclaimed": True,
        "production_authorized_not_granted": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_039_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_model_provider()
    promotion = pcpr_039_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    live_report = qualify_live_model_provider_execution(test_level="comprehensive")
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_038_TASK_ID,
            "goal_id": "PCPR-G430",
            "promotion_status": "rnd_non_promoted",
            "cuda_execution_verdict_cid": PCPR_038_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-038 qualified live CUDA execution as R&D. This task qualifies "
                "one live model/provider path as R&D. PCPR-001 live qualification "
                "remains rnd_non_promoted and is not promoted by this receipt."
            ),
            "evidence_kind": "measured",
        },
        "model_provider_execution": {
            "schema": MODEL_EVALUATOR_SCHEMA,
            "interface": MODEL_EVALUATOR_INTERFACE,
            "floor_schema": FLOOR_SCHEMA,
            "floor_interface": FLOOR_INTERFACE,
            "floor_task_id": FLOOR_TASK_ID,
            "floor_goal_id": FLOOR_GOAL_ID,
            "kernel": CANARY_KERNEL,
            "live_model_provider_execution_qualified": (
                verdict.live_model_provider_execution_qualified
            ),
            "live_llamacpp_qualified": verdict.live_llamacpp_qualified,
            "production_authorized": False,
            "hardware_ladder_qualified": False,
            "model_provider_identity": live_report.get("model_provider_identity"),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_039_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_039_current_tree_binding(
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
        raise ModelProviderExecutionQualificationError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise ModelProviderExecutionQualificationError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise ModelProviderExecutionQualificationError(
            "kit_commit must equal kit_gitlink"
        )
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


def current_head_pcpr_039_current_tree_binding() -> dict[str, Any]:
    return pcpr_039_current_tree_binding(
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


def validate_pcpr_039_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-039 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise ModelProviderExecutionQualificationError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_039_TASK_ID:
        raise ModelProviderExecutionQualificationError(
            "outer receipt task_id must be PCPR-039"
        )
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise ModelProviderExecutionQualificationError(
            "outer receipt must not claim a release"
        )
    if payload.get("completion_authoritative") is True:
        raise ModelProviderExecutionQualificationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise ModelProviderExecutionQualificationError(
            "qualification_verdict must be a mapping"
        )
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise ModelProviderExecutionQualificationError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise ModelProviderExecutionQualificationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification cannot freeze contracts"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise ModelProviderExecutionQualificationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict_section.get("production_authorized") is True:
        raise ModelProviderExecutionQualificationError(
            "this task cannot grant production_authorized"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise ModelProviderExecutionQualificationError(
            "qualification_verdict.promotion_status is not an admitted PCPR-039 status"
        )
    if promotion_status == "supervisor_promoted":
        raise ModelProviderExecutionQualificationError(
            "model/provider execution qualification cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise ModelProviderExecutionQualificationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise ModelProviderExecutionQualificationError(
                "acceptance must not claim a release"
            )

    expected = current_head_pcpr_039_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise ModelProviderExecutionQualificationError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise ModelProviderExecutionQualificationError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise ModelProviderExecutionQualificationError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_039_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }


__all__ = (
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "MODEL_EVALUATOR_INTERFACE",
    "PCPR_039_GOAL_ID",
    "PCPR_039_TASK_ID",
    "ModelProviderExecutionQualificationError",
    "ModelProbe",
    "ModelVerdict",
    "current_head_model_provider_probes",
    "current_head_pcpr_039_current_tree_binding",
    "current_head_pcpr_039_receipt_promotion",
    "current_head_pcpr_039_receipt_sections",
    "discover_accelerate_root",
    "pcpr_039_current_tree_binding",
    "pcpr_039_receipt_negative_results",
    "pcpr_039_receipt_promotion",
    "qualify_current_head_model_provider",
    "qualify_model_provider_execution",
    "validate_pcpr_039_outer_receipt",
)
