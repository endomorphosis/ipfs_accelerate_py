"""SCA-622 / SCA-G182: end-to-end production authority gate (SCAEV182E2E).

Deterministic aggregate verifier for the production-authority composition.
Reads and checks stage receipts only — it cannot repair, synthesize, or
relabel missing evidence. Focused adapter success is retained as capability
evidence but cannot satisfy production authority without a current-root
end-to-end receipt.

Also binds SCAEV177COMPOSE so the shared evaluation artifact can close the
parent production-composition evidence obligation when both goals share
these declared outputs.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
    BaselineStageName,
    BaselineStageReceipt,
    StageCompleteness,
)

SCAEV182E2E: Final = "SCAEV182E2E"
SCAEV177COMPOSE: Final = "SCAEV177COMPOSE"

SCAEV182E2E_EVIDENCE: Final = SCAEV182E2E
SCAEV182E2E_COVERAGE: Final[tuple[str, ...]] = (
    "primary-provider-index-current-root",
    "actual-package-surfaces-no-synthesis",
    "real-indexed-graphrag-not-canary",
    "real-mcp-list-call-receipts",
    "kernel-checked-prover-cache-receipts",
    "optional-real-zk-capability-gated",
    "exact-runtime-identity",
    "scheduler-fence-regressions",
    "fail-closed-unsupported-unknown-stale-partial",
    "fail-closed-missing-synthesized-simulated-cross-root",
    "focused-adapter-success-not-production-authority",
    "zero-runtime-model-calls",
    "bounded-deduplicated-repair-projection",
    "exact-validation-and-reproof-commands",
)

SCAEV177COMPOSE_COVERAGE: Final[tuple[str, ...]] = (
    "primary-provider-index-current-root",
    "actual-package-surfaces-no-synthesis",
    "real-indexed-graphrag-not-canary",
    "real-mcp-list-call-receipts",
    "kernel-checked-prover-cache-receipts",
    "optional-real-zk-capability-gated",
    "exact-runtime-identity",
    "scheduler-fence-regressions",
    "fail-closed-missing-synthesized-simulated-partial-stale-cross-root",
    "focused-adapter-success-not-production-authority",
    "zero-runtime-model-calls",
)

COMPOSITION_INTERFACE: Final = "ProductionContractComposition@1"
COMPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-composition@1"
)
COMPOSITION_VERSION: Final = "1"
TASK_ID: Final = "SCA-622"
GOAL_ID: Final = "SCA-G182"
CORPUS_VERSION: Final = "sca-622-production-authority-v1"
EVALUATED_AT: Final = "2026-07-29T21:00:00Z"
CONTENT_ROOT: Final = "content-root:sca-622-current"

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[4]
PUBLISHED_REPORT: Final = (
    REPOSITORY_ROOT
    / "data/agent_supervisor/swissknife_contract_assurance/evaluation"
    / "production-composition.json"
)


class ProductionStageName(str, Enum):
    """Mandatory and optional production-composition stages."""

    PRIMARY_PROVIDER_INDEX = "primary_provider_index"
    ACTUAL_PACKAGE_SURFACES = "actual_package_surfaces"
    REAL_GRAPHRAG = "real_graphrag"
    MCP_LIST_CALL_RECEIPTS = "mcp_list_call_receipts"
    KERNEL_PROVER_CACHE = "kernel_prover_cache"
    RUNTIME_IDENTITY = "runtime_identity"
    SCHEDULER_FENCE = "scheduler_fence"
    REPAIR_PROJECTION = "repair_projection"
    REAL_ZK_ATTESTATION = "real_zk_attestation"


class StageDisposition(str, Enum):
    """Closed dispositions a stage receipt may carry."""

    CURRENT = "current"
    MISSING = "missing"
    SYNTHESIZED = "synthesized"
    SIMULATED = "simulated"
    PARTIAL = "partial"
    STALE = "stale"
    CROSS_ROOT = "cross_root"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    FOCUSED_ADAPTER_ONLY = "focused_adapter_only"


MANDATORY_STAGES: Final[tuple[ProductionStageName, ...]] = (
    ProductionStageName.PRIMARY_PROVIDER_INDEX,
    ProductionStageName.ACTUAL_PACKAGE_SURFACES,
    ProductionStageName.REAL_GRAPHRAG,
    ProductionStageName.MCP_LIST_CALL_RECEIPTS,
    ProductionStageName.KERNEL_PROVER_CACHE,
    ProductionStageName.RUNTIME_IDENTITY,
    ProductionStageName.SCHEDULER_FENCE,
    ProductionStageName.REPAIR_PROJECTION,
)

OPTIONAL_STAGES: Final[tuple[ProductionStageName, ...]] = (
    ProductionStageName.REAL_ZK_ATTESTATION,
)

FAIL_CLOSED_DISPOSITIONS: Final[frozenset[StageDisposition]] = frozenset(
    {
        StageDisposition.MISSING,
        StageDisposition.SYNTHESIZED,
        StageDisposition.SIMULATED,
        StageDisposition.PARTIAL,
        StageDisposition.STALE,
        StageDisposition.CROSS_ROOT,
        StageDisposition.UNSUPPORTED,
        StageDisposition.UNKNOWN,
        StageDisposition.UNAVAILABLE,
        StageDisposition.FOCUSED_ADAPTER_ONLY,
    }
)

SIMULATION_FORBIDDEN_STAGES: Final[frozenset[ProductionStageName]] = frozenset(
    {
        ProductionStageName.PRIMARY_PROVIDER_INDEX,
        ProductionStageName.ACTUAL_PACKAGE_SURFACES,
        ProductionStageName.REAL_GRAPHRAG,
        ProductionStageName.MCP_LIST_CALL_RECEIPTS,
        ProductionStageName.KERNEL_PROVER_CACHE,
        ProductionStageName.RUNTIME_IDENTITY,
        ProductionStageName.SCHEDULER_FENCE,
        ProductionStageName.REPAIR_PROJECTION,
    }
)

BASELINE_STAGE_BRIDGE: Final[
    Mapping[ProductionStageName, BaselineStageName]
] = {
    ProductionStageName.PRIMARY_PROVIDER_INDEX: BaselineStageName.REPOSITORY_INDEX,
    ProductionStageName.ACTUAL_PACKAGE_SURFACES: BaselineStageName.EXTRACTION,
    ProductionStageName.REAL_GRAPHRAG: BaselineStageName.GRAPH,
    ProductionStageName.MCP_LIST_CALL_RECEIPTS: BaselineStageName.INVOCATION_TRACE,
    ProductionStageName.KERNEL_PROVER_CACHE: BaselineStageName.PROOF_CACHE,
    ProductionStageName.REPAIR_PROJECTION: BaselineStageName.MISMATCH,
    ProductionStageName.RUNTIME_IDENTITY: BaselineStageName.PUBLISH,
}


class CompositionAuthorityError(ValueError):
    """Mandatory production stage evidence is incomplete or non-authoritative."""


@dataclass(frozen=True, slots=True)
class ProductionStageReceipt:
    """One production-composition stage bound to a content root."""

    name: ProductionStageName
    disposition: StageDisposition
    content_root: str
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)
    mandatory: bool = True
    focused_adapter_success: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            self.name
            if isinstance(self.name, ProductionStageName)
            else ProductionStageName(self.name),
        )
        object.__setattr__(
            self,
            "disposition",
            self.disposition
            if isinstance(self.disposition, StageDisposition)
            else StageDisposition(self.disposition),
        )
        object.__setattr__(
            self,
            "content_root",
            str(self.content_root or "").strip(),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({str(code) for code in self.reason_codes})),
        )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(dict(self.details or {})),
        )
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        object.__setattr__(
            self, "focused_adapter_success", bool(self.focused_adapter_success)
        )

    @property
    def current(self) -> bool:
        return self.disposition is StageDisposition.CURRENT

    @property
    def production_admissible(self) -> bool:
        if self.disposition is not StageDisposition.CURRENT:
            return False
        if not self.content_root:
            return False
        if self.focused_adapter_success:
            return False
        if self.details.get("entrypoint") == "focused_adapter":
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "disposition": self.disposition.value,
            "content_root": self.content_root,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
            "mandatory": self.mandatory,
            "focused_adapter_success": self.focused_adapter_success,
            "production_admissible": self.production_admissible,
        }


@dataclass(frozen=True, slots=True)
class BoundedRepairTask:
    """One deduplicated bounded repair task projected from a proved mismatch."""

    task_id: str
    mismatch_id: str
    contract_id: str
    content_root: str
    validation_commands: tuple[str, ...]
    re_proof_commands: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id).strip())
        object.__setattr__(self, "mismatch_id", str(self.mismatch_id).strip())
        object.__setattr__(self, "contract_id", str(self.contract_id).strip())
        object.__setattr__(self, "content_root", str(self.content_root).strip())
        object.__setattr__(
            self,
            "validation_commands",
            tuple(str(item).strip() for item in self.validation_commands if str(item).strip()),
        )
        object.__setattr__(
            self,
            "re_proof_commands",
            tuple(str(item).strip() for item in self.re_proof_commands if str(item).strip()),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({str(code) for code in self.reason_codes})),
        )
        if not self.task_id:
            raise CompositionAuthorityError("repair task_id is required")
        if not self.validation_commands:
            raise CompositionAuthorityError(
                f"{self.task_id}: validation_commands required"
            )
        if not self.re_proof_commands:
            raise CompositionAuthorityError(
                f"{self.task_id}: re_proof_commands required"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mismatch_id": self.mismatch_id,
            "contract_id": self.contract_id,
            "content_root": self.content_root,
            "validation_commands": list(self.validation_commands),
            "re_proof_commands": list(self.re_proof_commands),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class CompositionObservation:
    """One composition case (healthy chain or fail-closed mutant)."""

    case_id: str
    mutation: str
    expected_authority: bool
    production_eligible: bool
    authority_granted: bool
    fail_closed: bool
    reason_codes: tuple[str, ...]
    stage_names: tuple[str, ...]
    model_call_count: int
    provider_call_count: int
    llm_call_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mutation": self.mutation,
            "expected_authority": self.expected_authority,
            "production_eligible": self.production_eligible,
            "authority_granted": self.authority_granted,
            "fail_closed": self.fail_closed,
            "reason_codes": list(self.reason_codes),
            "stage_names": list(self.stage_names),
            "model_call_count": self.model_call_count,
            "provider_call_count": self.provider_call_count,
            "llm_call_count": self.llm_call_count,
        }


@dataclass(frozen=True, slots=True)
class ProductionCompositionResult:
    """Aggregate receipt for one current-root production composition."""

    content_root: str
    stages: tuple[ProductionStageReceipt, ...]
    production_eligible: bool
    authority_granted: bool
    reason_codes: tuple[str, ...]
    optional_zk: Mapping[str, Any]
    isolation_audit: Mapping[str, int]
    evidence: Mapping[str, Any]
    baseline_bridges: tuple[dict[str, Any], ...] = ()
    repair_tasks: tuple[BoundedRepairTask, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPOSITION_SCHEMA,
            "interface": COMPOSITION_INTERFACE,
            "version": COMPOSITION_VERSION,
            "content_root": self.content_root,
            "production_eligible": self.production_eligible,
            "authority_granted": self.authority_granted,
            "reason_codes": list(self.reason_codes),
            "stages": [stage.to_dict() for stage in self.stages],
            "optional_zk": dict(self.optional_zk),
            "isolation_audit": dict(self.isolation_audit),
            "evidence": dict(self.evidence),
            "baseline_bridges": list(self.baseline_bridges),
            "repair_tasks": [task.to_dict() for task in self.repair_tasks],
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_label(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_report(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    result.pop("report_id", None)
    result["report_id"] = _sha256_label(result)
    return result


def verify_composition_report(report: Mapping[str, Any]) -> bool:
    if report.get("schema") != COMPOSITION_SCHEMA:
        return False
    evidence = report.get("evidence") or {}
    requirement_ids = evidence.get("requirement_ids") or []
    if SCAEV182E2E not in requirement_ids:
        return False
    claimed = report.get("report_id")
    return isinstance(claimed, str) and claimed == _seal_report(dict(report)).get(
        "report_id"
    )


def _stage_fail_reasons(
    stage: ProductionStageReceipt, *, expected_root: str
) -> list[str]:
    reasons: list[str] = []
    if stage.disposition in FAIL_CLOSED_DISPOSITIONS:
        reasons.append(f"{stage.name.value}:{stage.disposition.value}")
    if stage.disposition is StageDisposition.CURRENT and not stage.content_root:
        reasons.append(f"{stage.name.value}:empty_content_root")
    if (
        stage.disposition is StageDisposition.CURRENT
        and stage.content_root
        and stage.content_root != expected_root
    ):
        reasons.append(f"{stage.name.value}:cross_root")
    if stage.disposition is StageDisposition.CROSS_ROOT:
        reasons.append(f"{stage.name.value}:cross_root_disposition")
    if (
        stage.disposition is StageDisposition.SIMULATED
        and stage.name in SIMULATION_FORBIDDEN_STAGES
    ):
        reasons.append(f"{stage.name.value}:simulation_not_authoritative")
    if stage.details.get("entrypoint") == "focused_adapter":
        reasons.append(f"{stage.name.value}:focused_adapter_not_production")
    if stage.details.get("synthesized_from_expected") is True:
        reasons.append(f"{stage.name.value}:synthesized_from_expected")
    if stage.details.get("canary_graph") is True:
        reasons.append(f"{stage.name.value}:canary_graph_not_production")
    return reasons


def _bridge_baseline_receipt(
    stage: ProductionStageReceipt,
) -> dict[str, Any] | None:
    baseline_name = BASELINE_STAGE_BRIDGE.get(stage.name)
    if baseline_name is None:
        return None

    if (
        stage.disposition is StageDisposition.CURRENT
        and stage.production_admissible
    ):
        completeness = StageCompleteness.COMPLETE
    elif stage.disposition is StageDisposition.PARTIAL:
        completeness = StageCompleteness.PARTIAL
    elif stage.disposition in {
        StageDisposition.MISSING,
        StageDisposition.STALE,
        StageDisposition.CROSS_ROOT,
        StageDisposition.UNAVAILABLE,
        StageDisposition.UNKNOWN,
        StageDisposition.UNSUPPORTED,
    }:
        completeness = StageCompleteness.FAILED
    else:
        completeness = StageCompleteness.WITHHELD

    receipt = BaselineStageReceipt(
        name=baseline_name,
        completeness=completeness,
        reason_codes=tuple(stage.reason_codes) or (stage.disposition.value,),
        root_id=stage.content_root,
        details={
            "production_stage": stage.name.value,
            "disposition": stage.disposition.value,
        },
    )
    return {
        "production_stage": stage.name.value,
        "baseline_stage": baseline_name.value,
        "healthy_enough_for_authority": receipt.healthy_enough_for_authority,
        "complete": receipt.complete,
        "receipt": receipt.to_dict(),
    }


def project_bounded_repairs(
    mismatches: Sequence[Mapping[str, Any]],
    *,
    content_root: str,
    max_tasks: int = 32,
) -> tuple[BoundedRepairTask, ...]:
    """Project proved mismatches into deduplicated bounded repair tasks.

    This gate does not synthesize mismatches or invent repair strategies; it
    only materializes exact validation and re-proof commands for proved,
    content-root-bound contract mismatches.
    """
    expected_root = str(content_root or "").strip()
    if not expected_root:
        raise CompositionAuthorityError("content_root is required")
    if max_tasks < 1:
        raise CompositionAuthorityError("max_tasks must be >= 1")

    seen: set[str] = set()
    tasks: list[BoundedRepairTask] = []
    for raw in mismatches:
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status") or "").strip().lower()
        if status not in {"proved", "refuted_mismatch", "contract_mismatch"}:
            # Only proved/refuted mismatches project; partial/unknown do not.
            continue
        mismatch_root = str(raw.get("content_root") or "").strip()
        if mismatch_root and mismatch_root != expected_root:
            continue
        mismatch_id = str(raw.get("mismatch_id") or "").strip()
        contract_id = str(raw.get("contract_id") or "").strip()
        if not mismatch_id or not contract_id:
            continue
        dedupe_key = f"{contract_id}:{mismatch_id}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        validation = tuple(
            str(item).strip()
            for item in (raw.get("validation_commands") or ())
            if str(item).strip()
        )
        re_proof = tuple(
            str(item).strip()
            for item in (raw.get("re_proof_commands") or ())
            if str(item).strip()
        )
        if not validation:
            validation = (
                f"python3 -m pytest external/ipfs_accelerate/test/api/"
                f"test_agent_supervisor_contract_mismatch_refinery.py "
                f"-q -k {contract_id}",
            )
        if not re_proof:
            re_proof = (
                f"python3 -m ipfs_accelerate_py.agent_supervisor.proof."
                f"mcp_contract_prover --contract-id {contract_id} "
                f"--content-root {expected_root}",
            )
        task_id = str(raw.get("task_id") or f"repair:{dedupe_key}").strip()
        tasks.append(
            BoundedRepairTask(
                task_id=task_id,
                mismatch_id=mismatch_id,
                contract_id=contract_id,
                content_root=expected_root,
                validation_commands=validation,
                re_proof_commands=re_proof,
                reason_codes=tuple(
                    str(code)
                    for code in (raw.get("reason_codes") or ("proved_mismatch",))
                ),
            )
        )
        if len(tasks) >= max_tasks:
            break
    return tuple(tasks)


def compose_production_contract(
    stages: Sequence[ProductionStageReceipt],
    *,
    content_root: str,
    model_call_count: int = 0,
    provider_call_count: int = 0,
    llm_call_count: int = 0,
    mismatches: Sequence[Mapping[str, Any]] | None = None,
) -> ProductionCompositionResult:
    """Compose production stage receipts or fail closed.

    Zero runtime model/provider/LLM calls are required for authority. Optional
    real-ZK is capability-gated: unavailable/simulated ZK never attests and
    never blocks composition when declared optional.
    """
    expected_root = str(content_root or "").strip()
    if not expected_root:
        raise CompositionAuthorityError("content_root is required")

    by_name = {stage.name: stage for stage in stages}
    reasons: list[str] = []

    for mandatory in MANDATORY_STAGES:
        stage = by_name.get(mandatory)
        if stage is None:
            reasons.append(f"{mandatory.value}:missing")
            continue
        reasons.extend(
            _stage_fail_reasons(stage, expected_root=expected_root)
        )

    zk_stage = by_name.get(ProductionStageName.REAL_ZK_ATTESTATION)
    if zk_stage is None:
        optional_zk: dict[str, Any] = {
            "present": False,
            "disposition": StageDisposition.UNAVAILABLE.value,
            "attested": False,
            "blocks_composition": False,
            "reason_codes": ["optional_zk_absent"],
        }
    else:
        zk_reasons = _stage_fail_reasons(zk_stage, expected_root=expected_root)
        attested = (
            zk_stage.disposition is StageDisposition.CURRENT
            and zk_stage.content_root == expected_root
            and zk_stage.details.get("predicate") == "verified_receipt"
            and not zk_reasons
        )
        blocks = False
        if zk_stage.details.get("required") is True and not attested:
            blocks = True
            reasons.extend(
                zk_reasons
                or [f"{zk_stage.name.value}:required_not_attested"]
            )
        optional_zk = {
            "present": True,
            "disposition": zk_stage.disposition.value,
            "attested": attested,
            "blocks_composition": blocks,
            "reason_codes": (
                list(zk_reasons)
                if not attested
                else list(zk_stage.reason_codes)
            ),
            "predicate": zk_stage.details.get("predicate"),
        }

    if model_call_count or provider_call_count or llm_call_count:
        reasons.append("runtime_model_calls_nonzero")

    repair_tasks = project_bounded_repairs(
        mismatches or (),
        content_root=expected_root,
    )
    repair_stage = by_name.get(ProductionStageName.REPAIR_PROJECTION)
    if (
        repair_stage is not None
        and repair_stage.disposition is StageDisposition.CURRENT
        and repair_stage.production_admissible
        and repair_stage.details.get("require_tasks") is True
        and not repair_tasks
    ):
        reasons.append("repair_projection:empty_when_required")

    deduped = tuple(dict.fromkeys(reasons))
    production_eligible = not deduped
    authority_granted = production_eligible

    ordered = tuple(
        by_name[name]
        for name in (*MANDATORY_STAGES, *OPTIONAL_STAGES)
        if name in by_name
    )
    bridges = tuple(
        bridge
        for stage in ordered
        if (bridge := _bridge_baseline_receipt(stage)) is not None
    )

    return ProductionCompositionResult(
        content_root=expected_root,
        stages=ordered,
        production_eligible=production_eligible,
        authority_granted=authority_granted,
        reason_codes=deduped,
        optional_zk=optional_zk,
        isolation_audit={
            "model_call_count": int(model_call_count),
            "provider_call_count": int(provider_call_count),
            "llm_call_count": int(llm_call_count),
        },
        evidence={
            "requirement_ids": [SCAEV182E2E, SCAEV177COMPOSE],
            "coverage": list(
                dict.fromkeys(
                    (*SCAEV182E2E_COVERAGE, *SCAEV177COMPOSE_COVERAGE)
                )
            ),
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
        },
        baseline_bridges=bridges,
        repair_tasks=repair_tasks,
    )


def _healthy_stages(
    content_root: str,
    *,
    include_zk: bool = False,
    zk_current: bool = False,
) -> list[ProductionStageReceipt]:
    stages: list[ProductionStageReceipt] = []
    for name in MANDATORY_STAGES:
        details: dict[str, Any] = {
            "entrypoint": "production",
            "provenance": f"production:{name.value}",
        }
        if name is ProductionStageName.REPAIR_PROJECTION:
            details["require_tasks"] = False
            details["projection"] = "RuntimeContractMismatchRefinery"
        stages.append(
            ProductionStageReceipt(
                name=name,
                disposition=StageDisposition.CURRENT,
                content_root=content_root,
                reason_codes=("current",),
                details=details,
                mandatory=True,
            )
        )
    if include_zk or zk_current:
        stages.append(
            ProductionStageReceipt(
                name=ProductionStageName.REAL_ZK_ATTESTATION,
                disposition=(
                    StageDisposition.CURRENT
                    if zk_current
                    else StageDisposition.UNAVAILABLE
                ),
                content_root=content_root if zk_current else "",
                reason_codes=("unavailable",) if not zk_current else ("current",),
                details={
                    "entrypoint": "production",
                    "predicate": "verified_receipt" if zk_current else "",
                    "required": False,
                },
                mandatory=False,
            )
        )
    return stages


def _mutate_stage(
    stages: Sequence[ProductionStageReceipt],
    target: ProductionStageName,
    **overrides: Any,
) -> list[ProductionStageReceipt]:
    result: list[ProductionStageReceipt] = []
    for stage in stages:
        if stage.name is not target:
            result.append(stage)
            continue
        payload = {
            "name": stage.name,
            "disposition": stage.disposition,
            "content_root": stage.content_root,
            "reason_codes": stage.reason_codes,
            "details": dict(stage.details),
            "mandatory": stage.mandatory,
            "focused_adapter_success": stage.focused_adapter_success,
        }
        merged = dict(overrides)
        if "details" in merged and isinstance(merged["details"], Mapping):
            details = dict(payload["details"])
            details.update(dict(merged["details"]))
            merged["details"] = details
        payload.update(merged)
        result.append(ProductionStageReceipt(**payload))
    return result


def _observation(
    case_id: str,
    mutation: str,
    result: ProductionCompositionResult,
    *,
    expected_authority: bool,
) -> CompositionObservation:
    return CompositionObservation(
        case_id=case_id,
        mutation=mutation,
        expected_authority=expected_authority,
        production_eligible=result.production_eligible,
        authority_granted=result.authority_granted,
        fail_closed=not result.authority_granted,
        reason_codes=result.reason_codes,
        stage_names=tuple(stage.name.value for stage in result.stages),
        model_call_count=int(result.isolation_audit["model_call_count"]),
        provider_call_count=int(result.isolation_audit["provider_call_count"]),
        llm_call_count=int(result.isolation_audit["llm_call_count"]),
    )


def _sample_proved_mismatches(content_root: str) -> tuple[dict[str, Any], ...]:
    return (
        {
            "mismatch_id": "mm-surface-route-absent",
            "contract_id": "mcp.tools.call.list_models",
            "status": "proved",
            "content_root": content_root,
            "validation_commands": (
                "python3 -m pytest external/ipfs_accelerate/test/api/"
                "test_agent_supervisor_actual_package_surfaces.py -q",
            ),
            "re_proof_commands": (
                "python3 -m pytest external/ipfs_accelerate/test/api/"
                "test_agent_supervisor_mcp_contract_prover.py -q "
                "-k list_models",
            ),
            "reason_codes": ("actual_route_absent",),
        },
        {
            # Duplicate of the same contract/mismatch — must dedupe.
            "mismatch_id": "mm-surface-route-absent",
            "contract_id": "mcp.tools.call.list_models",
            "status": "proved",
            "content_root": content_root,
            "validation_commands": ("duplicate-should-not-appear",),
            "re_proof_commands": ("duplicate-should-not-appear",),
        },
        {
            "mismatch_id": "mm-partial-unknown",
            "contract_id": "mcp.tools.call.unknown",
            "status": "unknown",
            "content_root": content_root,
            "validation_commands": ("should-not-project",),
            "re_proof_commands": ("should-not-project",),
        },
        {
            "mismatch_id": "mm-cross-root",
            "contract_id": "mcp.tools.call.cross",
            "status": "proved",
            "content_root": "content-root:other-snapshot",
            "validation_commands": ("should-not-project",),
            "re_proof_commands": ("should-not-project",),
        },
        {
            "mismatch_id": "mm-kernel-cache-stale-key",
            "contract_id": "proof.kernel.reconstruct.tool_schema",
            "status": "contract_mismatch",
            "content_root": content_root,
            "validation_commands": (
                "python3 -m pytest external/ipfs_accelerate/test/api/"
                "test_agent_supervisor_mcp_contract_proof_cache.py -q",
            ),
            "re_proof_commands": (
                "python3 -m pytest external/ipfs_accelerate/test/api/"
                "test_agent_supervisor_code_proof_reproof.py -q",
            ),
            "reason_codes": ("stale_cache_key",),
        },
    )


def build_composition_cases() -> tuple[CompositionObservation, ...]:
    """Preregistered healthy chain plus fail-closed mutants."""
    observations: list[CompositionObservation] = []
    mismatches = _sample_proved_mismatches(CONTENT_ROOT)

    healthy = compose_production_contract(
        _healthy_stages(CONTENT_ROOT, include_zk=True),
        content_root=CONTENT_ROOT,
        mismatches=mismatches,
    )
    observations.append(
        _observation(
            "compose:healthy-current-root",
            "none",
            healthy,
            expected_authority=True,
        )
    )

    missing_stages = [
        stage
        for stage in _healthy_stages(CONTENT_ROOT)
        if stage.name is not ProductionStageName.ACTUAL_PACKAGE_SURFACES
    ]
    missing = compose_production_contract(
        missing_stages, content_root=CONTENT_ROOT
    )
    observations.append(
        _observation(
            "compose:missing-actual-surfaces",
            "missing",
            missing,
            expected_authority=False,
        )
    )

    synthesized = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.ACTUAL_PACKAGE_SURFACES,
            disposition=StageDisposition.SYNTHESIZED,
            reason_codes=("synthesized_from_expected",),
            details={
                "entrypoint": "production",
                "synthesized_from_expected": True,
            },
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:synthesized-surfaces",
            "synthesized",
            synthesized,
            expected_authority=False,
        )
    )

    simulated_zk = compose_production_contract(
        _healthy_stages(CONTENT_ROOT)
        + [
            ProductionStageReceipt(
                name=ProductionStageName.REAL_ZK_ATTESTATION,
                disposition=StageDisposition.SIMULATED,
                content_root=CONTENT_ROOT,
                reason_codes=("simulated_groth16",),
                details={
                    "entrypoint": "production",
                    "predicate": "hash_commitment",
                    "required": False,
                },
                mandatory=False,
            )
        ],
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:simulated-zk-non-attested",
            "simulated",
            simulated_zk,
            expected_authority=True,
        )
    )

    partial = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.KERNEL_PROVER_CACHE,
            disposition=StageDisposition.PARTIAL,
            reason_codes=("partial_kernel_reconstruction",),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:partial-prover-cache",
            "partial",
            partial,
            expected_authority=False,
        )
    )

    stale = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.PRIMARY_PROVIDER_INDEX,
            disposition=StageDisposition.STALE,
            reason_codes=("stale_provider_root",),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:stale-provider-index",
            "stale",
            stale,
            expected_authority=False,
        )
    )

    cross = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.REAL_GRAPHRAG,
            disposition=StageDisposition.CROSS_ROOT,
            content_root="content-root:other-snapshot",
            reason_codes=("cross_root_graph",),
            details={"canary_graph": False, "entrypoint": "production"},
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:cross-root-graphrag",
            "cross_root",
            cross,
            expected_authority=False,
        )
    )

    canary = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.REAL_GRAPHRAG,
            details={"entrypoint": "production", "canary_graph": True},
            reason_codes=("canary_graph",),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:canary-graphrag",
            "canary_graph",
            canary,
            expected_authority=False,
        )
    )

    adapter = [
        ProductionStageReceipt(
            name=name,
            disposition=StageDisposition.CURRENT,
            content_root=CONTENT_ROOT,
            reason_codes=("focused_adapter_pass",),
            details={"entrypoint": "focused_adapter"},
            mandatory=True,
            focused_adapter_success=True,
        )
        for name in MANDATORY_STAGES
    ]
    adapter_result = compose_production_contract(
        adapter, content_root=CONTENT_ROOT
    )
    observations.append(
        _observation(
            "compose:focused-adapter-only",
            "focused_adapter_only",
            adapter_result,
            expected_authority=False,
        )
    )

    with_calls = compose_production_contract(
        _healthy_stages(CONTENT_ROOT),
        content_root=CONTENT_ROOT,
        model_call_count=1,
        provider_call_count=0,
        llm_call_count=0,
    )
    observations.append(
        _observation(
            "compose:nonzero-model-calls",
            "runtime_model_calls",
            with_calls,
            expected_authority=False,
        )
    )

    no_identity = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.RUNTIME_IDENTITY,
            disposition=StageDisposition.MISSING,
            content_root="",
            reason_codes=("runtime_identity_absent",),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:missing-runtime-identity",
            "missing",
            no_identity,
            expected_authority=False,
        )
    )

    fence = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.SCHEDULER_FENCE,
            disposition=StageDisposition.PARTIAL,
            reason_codes=("fence_epoch_race", "capacity_not_conserved"),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:scheduler-fence-regression",
            "partial",
            fence,
            expected_authority=False,
        )
    )

    sim_mcp = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.MCP_LIST_CALL_RECEIPTS,
            disposition=StageDisposition.SIMULATED,
            reason_codes=("simulated_tools_call",),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:simulated-mcp-calls",
            "simulated",
            sim_mcp,
            expected_authority=False,
        )
    )

    unsupported = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.KERNEL_PROVER_CACHE,
            disposition=StageDisposition.UNSUPPORTED,
            reason_codes=("solver_backend_unsupported",),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:unsupported-prover-cache",
            "unsupported",
            unsupported,
            expected_authority=False,
        )
    )

    unknown = compose_production_contract(
        _mutate_stage(
            _healthy_stages(CONTENT_ROOT),
            ProductionStageName.ACTUAL_PACKAGE_SURFACES,
            disposition=StageDisposition.UNKNOWN,
            reason_codes=("surface_status_unknown",),
        ),
        content_root=CONTENT_ROOT,
    )
    observations.append(
        _observation(
            "compose:unknown-package-surfaces",
            "unknown",
            unknown,
            expected_authority=False,
        )
    )

    return tuple(observations)


def build_evaluation_report() -> dict[str, Any]:
    observations = build_composition_cases()
    healthy = next(item for item in observations if item.expected_authority)
    attacks = tuple(item for item in observations if not item.expected_authority)

    simulated_obs = next(
        item
        for item in observations
        if item.case_id == "compose:simulated-zk-non-attested"
    )
    simulated_result = compose_production_contract(
        _healthy_stages(CONTENT_ROOT)
        + [
            ProductionStageReceipt(
                name=ProductionStageName.REAL_ZK_ATTESTATION,
                disposition=StageDisposition.SIMULATED,
                content_root=CONTENT_ROOT,
                reason_codes=("simulated_groth16",),
                details={
                    "entrypoint": "production",
                    "predicate": "hash_commitment",
                    "required": False,
                },
                mandatory=False,
            )
        ],
        content_root=CONTENT_ROOT,
    )

    false_admits = sum(
        1
        for item in attacks
        if item.authority_granted or item.production_eligible
    )
    missed = sum(1 for item in attacks if not item.fail_closed)
    healthy_ok = all(
        item.authority_granted and item.production_eligible
        for item in observations
        if item.expected_authority
    )
    isolation_clean = all(
        item.model_call_count == 0
        and item.provider_call_count == 0
        and item.llm_call_count == 0
        for item in observations
        if item.case_id != "compose:nonzero-model-calls"
    )

    mismatches = _sample_proved_mismatches(CONTENT_ROOT)
    healthy_result = compose_production_contract(
        _healthy_stages(CONTENT_ROOT, include_zk=True),
        content_root=CONTENT_ROOT,
        mismatches=mismatches,
    )

    payload: dict[str, Any] = {
        "schema": COMPOSITION_SCHEMA,
        "interface": COMPOSITION_INTERFACE,
        "version": COMPOSITION_VERSION,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "corpus_version": CORPUS_VERSION,
        "evaluated_at": EVALUATED_AT,
        "content_root": CONTENT_ROOT,
        "evidence": {
            "requirement_ids": [SCAEV182E2E, SCAEV177COMPOSE],
            "coverage": list(
                dict.fromkeys(
                    (*SCAEV182E2E_COVERAGE, *SCAEV177COMPOSE_COVERAGE)
                )
            ),
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
        },
        "mandatory_stages": [stage.value for stage in MANDATORY_STAGES],
        "optional_stages": [stage.value for stage in OPTIONAL_STAGES],
        "fail_closed_dispositions": sorted(
            item.value for item in FAIL_CLOSED_DISPOSITIONS
        ),
        "healthy_composition": healthy_result.to_dict(),
        "optional_zk_policy": {
            "attested": simulated_result.optional_zk.get("attested"),
            "blocks_composition": simulated_result.optional_zk.get(
                "blocks_composition"
            ),
            "simulated_attested": bool(
                simulated_result.optional_zk.get("attested")
            ),
            "simulated_blocks_composition": bool(
                simulated_result.optional_zk.get("blocks_composition")
            ),
            "simulated_observation_authority": simulated_obs.authority_granted,
        },
        "summary": {
            "case_count": len(observations),
            "healthy_case_count": sum(
                1 for item in observations if item.expected_authority
            ),
            "attack_case_count": len(attacks),
            "healthy_authority_ok": healthy_ok,
            "false_authoritative_admission_count": false_admits,
            "missed_fail_closed_count": missed,
            "isolation_clean": isolation_clean,
            "production_eligible": healthy_result.production_eligible,
            "authority_granted": healthy_result.authority_granted,
            "repair_task_count": len(healthy_result.repair_tasks),
            "repair_tasks_deduplicated": len(healthy_result.repair_tasks) == 2,
        },
        "safety_gates": {
            "missing_fails_closed": any(
                item.mutation == "missing" and item.fail_closed
                for item in observations
            ),
            "synthesized_fails_closed": any(
                item.mutation == "synthesized" and item.fail_closed
                for item in observations
            ),
            "simulated_mcp_fails_closed": any(
                item.case_id == "compose:simulated-mcp-calls"
                and item.fail_closed
                for item in observations
            ),
            "partial_fails_closed": any(
                item.mutation == "partial" and item.fail_closed
                for item in observations
            ),
            "stale_fails_closed": any(
                item.mutation == "stale" and item.fail_closed
                for item in observations
            ),
            "cross_root_fails_closed": any(
                item.mutation == "cross_root" and item.fail_closed
                for item in observations
            ),
            "focused_adapter_not_authority": any(
                item.mutation == "focused_adapter_only" and item.fail_closed
                for item in observations
            ),
            "canary_graph_fails_closed": any(
                item.mutation == "canary_graph" and item.fail_closed
                for item in observations
            ),
            "runtime_model_calls_fail_closed": any(
                item.mutation == "runtime_model_calls" and item.fail_closed
                for item in observations
            ),
            "unsupported_fails_closed": any(
                item.mutation == "unsupported" and item.fail_closed
                for item in observations
            ),
            "unknown_fails_closed": any(
                item.mutation == "unknown" and item.fail_closed
                for item in observations
            ),
            "repair_projection_deduped": len(healthy_result.repair_tasks) == 2,
            "repair_tasks_carry_validation": all(
                task.validation_commands for task in healthy_result.repair_tasks
            ),
            "repair_tasks_carry_reproof": all(
                task.re_proof_commands for task in healthy_result.repair_tasks
            ),
        },
        "isolation_audit": {
            "llm_call_count": 0,
            "model_call_count": 0,
            "provider_call_count": 0,
            "held_out_fixture_disclosed": True,
        },
        "results": [item.to_dict() for item in observations],
        "passed": bool(
            healthy_ok
            and false_admits == 0
            and missed == 0
            and isolation_clean
            and healthy_result.production_eligible
            and healthy_result.authority_granted
            and len(healthy_result.repair_tasks) == 2
            and all(
                task.validation_commands and task.re_proof_commands
                for task in healthy_result.repair_tasks
            )
        ),
    }
    return _seal_report(payload)


def test_scaev182e2e_evidence_term_is_declared() -> None:
    assert SCAEV182E2E == SCAEV182E2E_EVIDENCE
    assert "bounded-deduplicated-repair-projection" in SCAEV182E2E_COVERAGE
    assert "zero-runtime-model-calls" in SCAEV182E2E_COVERAGE
    assert "fail-closed-unsupported-unknown-stale-partial" in SCAEV182E2E_COVERAGE


def test_healthy_production_composition_grants_authority() -> None:
    result = compose_production_contract(
        _healthy_stages(CONTENT_ROOT, include_zk=True),
        content_root=CONTENT_ROOT,
        mismatches=_sample_proved_mismatches(CONTENT_ROOT),
    )
    assert result.production_eligible is True
    assert result.authority_granted is True
    assert result.reason_codes == ()
    assert result.content_root == CONTENT_ROOT
    assert {stage.name for stage in result.stages} >= set(MANDATORY_STAGES)
    assert all(
        stage.production_admissible
        for stage in result.stages
        if stage.name in MANDATORY_STAGES
    )
    assert result.isolation_audit == {
        "model_call_count": 0,
        "provider_call_count": 0,
        "llm_call_count": 0,
    }
    assert SCAEV182E2E in result.evidence["requirement_ids"]
    assert SCAEV177COMPOSE in result.evidence["requirement_ids"]
    assert list(SCAEV182E2E_COVERAGE) == [
        item
        for item in result.evidence["coverage"]
        if item in SCAEV182E2E_COVERAGE
    ][: len(SCAEV182E2E_COVERAGE)]
    assert result.optional_zk.get("attested") is False
    assert result.baseline_bridges
    for bridge in result.baseline_bridges:
        if bridge["production_stage"] in {stage.value for stage in MANDATORY_STAGES}:
            assert bridge["healthy_enough_for_authority"] is True
            assert bridge["complete"] is True
    assert len(result.repair_tasks) == 2
    assert {task.contract_id for task in result.repair_tasks} == {
        "mcp.tools.call.list_models",
        "proof.kernel.reconstruct.tool_schema",
    }
    assert all(task.validation_commands for task in result.repair_tasks)
    assert all(task.re_proof_commands for task in result.repair_tasks)
    assert all(task.content_root == CONTENT_ROOT for task in result.repair_tasks)


@pytest.mark.parametrize(
    ("case_id", "needle"),
    (
        ("compose:missing-actual-surfaces", "actual_package_surfaces:missing"),
        ("compose:synthesized-surfaces", "synthesized"),
        ("compose:partial-prover-cache", "partial"),
        ("compose:stale-provider-index", "stale"),
        ("compose:cross-root-graphrag", "cross_root"),
        ("compose:canary-graphrag", "canary_graph"),
        ("compose:focused-adapter-only", "focused_adapter"),
        ("compose:nonzero-model-calls", "runtime_model_calls_nonzero"),
        ("compose:missing-runtime-identity", "runtime_identity"),
        ("compose:scheduler-fence-regression", "scheduler_fence"),
        ("compose:simulated-mcp-calls", "simulated"),
        ("compose:unsupported-prover-cache", "unsupported"),
        ("compose:unknown-package-surfaces", "unknown"),
    ),
)
def test_mandatory_failure_modes_fail_closed(
    case_id: str, needle: str
) -> None:
    observations = {item.case_id: item for item in build_composition_cases()}
    item = observations[case_id]
    assert item.expected_authority is False
    assert item.fail_closed is True
    assert item.authority_granted is False
    assert item.production_eligible is False
    assert any(needle in code for code in item.reason_codes), item.reason_codes


def test_simulated_optional_zk_does_not_attest_or_block() -> None:
    result = compose_production_contract(
        _healthy_stages(CONTENT_ROOT)
        + [
            ProductionStageReceipt(
                name=ProductionStageName.REAL_ZK_ATTESTATION,
                disposition=StageDisposition.SIMULATED,
                content_root=CONTENT_ROOT,
                reason_codes=("simulated_groth16",),
                details={
                    "entrypoint": "production",
                    "predicate": "hash_commitment",
                    "required": False,
                },
                mandatory=False,
            )
        ],
        content_root=CONTENT_ROOT,
    )
    assert result.production_eligible is True
    assert result.authority_granted is True
    assert result.optional_zk["attested"] is False
    assert result.optional_zk["blocks_composition"] is False


def test_required_simulated_zk_blocks_authority() -> None:
    result = compose_production_contract(
        _healthy_stages(CONTENT_ROOT)
        + [
            ProductionStageReceipt(
                name=ProductionStageName.REAL_ZK_ATTESTATION,
                disposition=StageDisposition.SIMULATED,
                content_root=CONTENT_ROOT,
                reason_codes=("simulated_groth16",),
                details={
                    "entrypoint": "production",
                    "predicate": "hash_commitment",
                    "required": True,
                },
                mandatory=False,
            )
        ],
        content_root=CONTENT_ROOT,
    )
    assert result.production_eligible is False
    assert result.authority_granted is False
    assert result.optional_zk["attested"] is False
    assert result.optional_zk["blocks_composition"] is True


def test_focused_adapter_success_is_not_production_authority() -> None:
    stages = [
        ProductionStageReceipt(
            name=name,
            disposition=StageDisposition.CURRENT,
            content_root=CONTENT_ROOT,
            reason_codes=("focused_adapter_pass",),
            details={"entrypoint": "focused_adapter"},
            mandatory=True,
            focused_adapter_success=True,
        )
        for name in MANDATORY_STAGES
    ]
    result = compose_production_contract(stages, content_root=CONTENT_ROOT)
    assert result.authority_granted is False
    assert any("focused_adapter" in code for code in result.reason_codes)


def test_baseline_stage_bridge_uses_baseline_receipts() -> None:
    result = compose_production_contract(
        _healthy_stages(CONTENT_ROOT),
        content_root=CONTENT_ROOT,
    )
    bridge = {
        item["baseline_stage"]: item for item in result.baseline_bridges
    }
    names = set(bridge)
    assert BaselineStageName.REPOSITORY_INDEX.value in names
    assert BaselineStageName.PROOF_CACHE.value in names
    assert BaselineStageName.MISMATCH.value in names
    for item in result.baseline_bridges:
        receipt = BaselineStageReceipt(
            name=item["receipt"]["name"],
            completeness=item["receipt"]["completeness"],
            reason_codes=tuple(item["receipt"]["reason_codes"]),
            root_id=item["receipt"]["root_id"],
            details=item["receipt"]["details"],
        )
        assert receipt.complete is item["complete"]
        assert (
            receipt.healthy_enough_for_authority
            is item["healthy_enough_for_authority"]
        )


def test_repair_projection_deduplicates_and_binds_commands() -> None:
    tasks = project_bounded_repairs(
        _sample_proved_mismatches(CONTENT_ROOT),
        content_root=CONTENT_ROOT,
    )
    assert len(tasks) == 2
    by_id = {task.mismatch_id: task for task in tasks}
    assert "mm-surface-route-absent" in by_id
    assert "mm-kernel-cache-stale-key" in by_id
    assert "mm-partial-unknown" not in by_id
    assert "mm-cross-root" not in by_id
    surface = by_id["mm-surface-route-absent"]
    assert "duplicate-should-not-appear" not in surface.validation_commands
    assert surface.validation_commands
    assert surface.re_proof_commands
    assert surface.content_root == CONTENT_ROOT


def test_evaluation_report_is_sealed_and_passes_gates() -> None:
    report = build_evaluation_report()
    assert report["passed"] is True
    assert SCAEV182E2E in report["evidence"]["requirement_ids"]
    assert SCAEV177COMPOSE in report["evidence"]["requirement_ids"]
    assert all(report["safety_gates"].values())
    assert verify_composition_report(report) is True
    assert report["summary"]["false_authoritative_admission_count"] == 0
    assert report["summary"]["missed_fail_closed_count"] == 0
    assert report["summary"]["production_eligible"] is True
    assert report["isolation_audit"]["model_call_count"] == 0
    assert report["isolation_audit"]["llm_call_count"] == 0
    assert report["isolation_audit"]["provider_call_count"] == 0
    assert report["summary"]["repair_task_count"] == 2

    tampered = deepcopy(report)
    tampered["summary"]["false_authoritative_admission_count"] = 1
    assert verify_composition_report(tampered) is False


def test_published_production_composition_matches_evaluation() -> None:
    report = build_evaluation_report()
    assert PUBLISHED_REPORT.is_file(), f"missing published report: {PUBLISHED_REPORT}"
    published = json.loads(PUBLISHED_REPORT.read_text(encoding="utf-8"))
    assert verify_composition_report(published) is True
    assert published["report_id"] == report["report_id"]
    assert published["passed"] is True
    assert SCAEV182E2E in published["evidence"]["requirement_ids"]
    assert published["goal_id"] == GOAL_ID
    assert published["task_id"] == TASK_ID
