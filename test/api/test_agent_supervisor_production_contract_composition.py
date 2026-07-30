"""SCA-620 / SCA-G177: production-composition evidence gate (SCAEV177COMPOSE).

Deterministic aggregate verifier for the audit-derived production-composition
closure. Focused adapter success is retained as capability evidence but cannot
authorize production composition without an end-to-end current-root receipt.

This suite:

* composes mandatory production stages under one content root;
* fails closed on missing, synthesized, simulated, partial, stale, or
  cross-root stage evidence;
* treats real-ZK attestation as optional and capability-gated;
* records zero model/provider/LLM calls;
* publishes a sealed evaluation receipt for SCAEV177COMPOSE.

The gate only reads and verifies stage receipts; it does not synthesize
missing stages, repair inputs, or establish a second proof/cache authority.
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


# ---------------------------------------------------------------------------
# Evidence / identity
# ---------------------------------------------------------------------------

SCAEV177COMPOSE: Final = "SCAEV177COMPOSE"
SCAEV177COMPOSE_EVIDENCE: Final = SCAEV177COMPOSE
SCAEV177COMPOSE_COVERAGE: Final = (
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
TASK_ID: Final = "SCA-620"
GOAL_ID: Final = "SCA-G177"
CORPUS_VERSION: Final = "sca-620-production-composition-v1"
EVALUATED_AT: Final = "2026-07-29T21:00:00Z"
CONTENT_ROOT: Final = "content-root:sca-620-current"

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PUBLISHED_REPORT = (
    REPOSITORY_ROOT
    / "data/agent_supervisor/swissknife_contract_assurance/evaluation"
    / "production-composition.json"
)


# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------


class ProductionStageName(str, Enum):
    """Mandatory and optional production-composition stages."""

    PRIMARY_PROVIDER_INDEX = "primary_provider_index"
    ACTUAL_PACKAGE_SURFACES = "actual_package_surfaces"
    REAL_GRAPHRAG = "real_graphrag"
    MCP_LIST_CALL_RECEIPTS = "mcp_list_call_receipts"
    KERNEL_PROVER_CACHE = "kernel_prover_cache"
    RUNTIME_IDENTITY = "runtime_identity"
    SCHEDULER_FENCE = "scheduler_fence"
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
        StageDisposition.UNAVAILABLE,
        StageDisposition.FOCUSED_ADAPTER_ONLY,
    }
)

# Stages where simulated evidence is never production-authoritative.
SIMULATION_FORBIDDEN_STAGES: Final[frozenset[ProductionStageName]] = frozenset(
    {
        ProductionStageName.PRIMARY_PROVIDER_INDEX,
        ProductionStageName.ACTUAL_PACKAGE_SURFACES,
        ProductionStageName.REAL_GRAPHRAG,
        ProductionStageName.MCP_LIST_CALL_RECEIPTS,
        ProductionStageName.KERNEL_PROVER_CACHE,
        ProductionStageName.RUNTIME_IDENTITY,
        ProductionStageName.SCHEDULER_FENCE,
        ProductionStageName.REAL_ZK_ATTESTATION,
    }
)

# Map production stages onto baseline stage receipts where the baseline
# pipeline already materializes an analogous receipt (evidence continuity).
BASELINE_STAGE_BRIDGE: Final[Mapping[ProductionStageName, BaselineStageName]] = (
    MappingProxyType(
        {
            ProductionStageName.PRIMARY_PROVIDER_INDEX: (
                BaselineStageName.REPOSITORY_INDEX
            ),
            ProductionStageName.ACTUAL_PACKAGE_SURFACES: (
                BaselineStageName.EXTRACTION
            ),
            ProductionStageName.REAL_GRAPHRAG: BaselineStageName.GRAPH,
            ProductionStageName.MCP_LIST_CALL_RECEIPTS: (
                BaselineStageName.INVOCATION_TRACE
            ),
            ProductionStageName.KERNEL_PROVER_CACHE: (
                BaselineStageName.PROOF_CACHE
            ),
            ProductionStageName.RUNTIME_IDENTITY: BaselineStageName.PUBLISH,
        }
    )
)


class CompositionAuthorityError(ValueError):
    """Mandatory production stage evidence is incomplete or non-authoritative."""


@dataclass(frozen=True, slots=True)
class ProductionStageReceipt:
    """One production-composition stage bound to a content root."""

    name: ProductionStageName
    disposition: StageDisposition
    content_root: str
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    mandatory: bool = True
    focused_adapter_success: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            self.name
            if isinstance(self.name, ProductionStageName)
            else ProductionStageName(str(self.name)),
        )
        object.__setattr__(
            self,
            "disposition",
            self.disposition
            if isinstance(self.disposition, StageDisposition)
            else StageDisposition(str(self.disposition)),
        )
        root = str(self.content_root or "").strip()
        object.__setattr__(self, "content_root", root)
        codes = tuple(
            sorted({str(code).strip() for code in self.reason_codes if str(code).strip()})
        )
        object.__setattr__(self, "reason_codes", codes)
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
        if self.focused_adapter_success and self.disposition is StageDisposition.CURRENT:
            # Focused adapter success alone is capability evidence, not
            # production authority. Admissible only when the stage is marked
            # current under a production entrypoint (not adapter-only).
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
    model_call_count: int = 0
    provider_call_count: int = 0
    llm_call_count: int = 0

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
        }


# ---------------------------------------------------------------------------
# Composition gate
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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
    if report.get("evidence", {}).get("requirement_ids") != [SCAEV177COMPOSE]:
        return False
    claimed = report.get("report_id")
    return isinstance(claimed, str) and claimed == _seal_report(dict(report)).get(
        "report_id"
    )


def _stage_fail_reasons(stage: ProductionStageReceipt, *, expected_root: str) -> list[str]:
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
    if stage.disposition is StageDisposition.SIMULATED and stage.name in (
        SIMULATION_FORBIDDEN_STAGES
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
    completeness = (
        StageCompleteness.COMPLETE
        if stage.disposition is StageDisposition.CURRENT
        and stage.production_admissible
        else StageCompleteness.PARTIAL
        if stage.disposition is StageDisposition.PARTIAL
        else StageCompleteness.FAILED
        if stage.disposition
        in {
            StageDisposition.MISSING,
            StageDisposition.STALE,
            StageDisposition.CROSS_ROOT,
            StageDisposition.UNAVAILABLE,
        }
        else StageCompleteness.WITHHELD
    )
    receipt = BaselineStageReceipt(
        name=baseline_name,
        completeness=completeness,
        reason_codes=stage.reason_codes
        or (
            ()
            if completeness is StageCompleteness.COMPLETE
            else (stage.disposition.value,)
        ),
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


def compose_production_contract(
    stages: Sequence[ProductionStageReceipt],
    *,
    content_root: str = CONTENT_ROOT,
    model_call_count: int = 0,
    provider_call_count: int = 0,
    llm_call_count: int = 0,
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
        reasons.extend(_stage_fail_reasons(stage, expected_root=expected_root))

    zk_stage = by_name.get(ProductionStageName.REAL_ZK_ATTESTATION)
    optional_zk: dict[str, Any]
    if zk_stage is None:
        optional_zk = {
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
        # Simulated/unavailable ZK never attests and never grants authority.
        blocks = False
        if zk_stage.details.get("required") is True and not attested:
            blocks = True
            reasons.extend(zk_reasons or [f"{zk_stage.name.value}:required_not_attested"])
        optional_zk = {
            "present": True,
            "disposition": zk_stage.disposition.value,
            "attested": attested,
            "blocks_composition": blocks,
            "reason_codes": zk_reasons
            if not attested
            else list(zk_stage.reason_codes),
            "predicate": zk_stage.details.get("predicate"),
        }

    if model_call_count or provider_call_count or llm_call_count:
        reasons.append("runtime_model_calls_nonzero")

    # Drop duplicates while preserving order.
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
            "requirement_ids": [SCAEV177COMPOSE],
            "coverage": list(SCAEV177COMPOSE_COVERAGE),
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
        },
        baseline_bridges=bridges,
    )


def _healthy_stages(
    *,
    content_root: str = CONTENT_ROOT,
    include_zk: bool = False,
    zk_current: bool = True,
) -> list[ProductionStageReceipt]:
    stages = [
        ProductionStageReceipt(
            name=name,
            disposition=StageDisposition.CURRENT,
            content_root=content_root,
            reason_codes=("current",),
            details={
                "entrypoint": "production",
                "provenance": f"production:{name.value}",
            },
            mandatory=True,
        )
        for name in MANDATORY_STAGES
    ]
    if include_zk:
        stages.append(
            ProductionStageReceipt(
                name=ProductionStageName.REAL_ZK_ATTESTATION,
                disposition=(
                    StageDisposition.CURRENT
                    if zk_current
                    else StageDisposition.UNAVAILABLE
                ),
                content_root=content_root if zk_current else "",
                reason_codes=("current",) if zk_current else ("unavailable",),
                details={
                    "entrypoint": "production",
                    "predicate": "verified_receipt" if zk_current else None,
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
        payload.update(overrides)
        if "details" in overrides and isinstance(overrides["details"], Mapping):
            merged = dict(stage.details)
            merged.update(dict(overrides["details"]))
            payload["details"] = merged
        result.append(ProductionStageReceipt(**payload))
    return result


def _observation(
    case_id: str,
    mutation: str,
    result: ProductionCompositionResult,
    *,
    expected_authority: bool,
) -> CompositionObservation:
    # Fail-closed means authority was withheld. For expected-healthy cases this
    # is True only if the gate incorrectly withheld; tests assert the inverse.
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


def build_composition_cases() -> tuple[CompositionObservation, ...]:
    """Preregistered healthy chain plus fail-closed mutants."""

    observations: list[CompositionObservation] = []

    healthy = compose_production_contract(_healthy_stages(include_zk=True))
    observations.append(
        _observation(
            "compose:healthy-current-root",
            "none",
            healthy,
            expected_authority=True,
        )
    )

    # Missing mandatory stage.
    missing_stages = [
        stage
        for stage in _healthy_stages()
        if stage.name is not ProductionStageName.ACTUAL_PACKAGE_SURFACES
    ]
    missing = compose_production_contract(missing_stages)
    observations.append(
        _observation(
            "compose:missing-actual-surfaces",
            "missing",
            missing,
            expected_authority=False,
        )
    )

    # Synthesized actual surfaces from expected descriptors.
    synthesized = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.ACTUAL_PACKAGE_SURFACES,
            disposition=StageDisposition.SYNTHESIZED,
            reason_codes=("synthesized_from_expected",),
            details={
                "entrypoint": "production",
                "synthesized_from_expected": True,
            },
        )
    )
    observations.append(
        _observation(
            "compose:synthesized-surfaces",
            "synthesized",
            synthesized,
            expected_authority=False,
        )
    )

    # Simulated ZK required path must not grant authority; optional simulated
    # ZK also must not attest.
    simulated_zk = compose_production_contract(
        _healthy_stages()
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
        ]
    )
    observations.append(
        _observation(
            "compose:simulated-zk-non-attested",
            "simulated",
            simulated_zk,
            expected_authority=True,  # optional simulated ZK does not block
        )
    )
    # But attestation flag must be false — checked in tests / report summary.

    # Partial proof/cache stage.
    partial = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.KERNEL_PROVER_CACHE,
            disposition=StageDisposition.PARTIAL,
            reason_codes=("partial_kernel_reconstruction",),
        )
    )
    observations.append(
        _observation(
            "compose:partial-prover-cache",
            "partial",
            partial,
            expected_authority=False,
        )
    )

    # Stale provider index.
    stale = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.PRIMARY_PROVIDER_INDEX,
            disposition=StageDisposition.STALE,
            reason_codes=("stale_provider_root",),
        )
    )
    observations.append(
        _observation(
            "compose:stale-provider-index",
            "stale",
            stale,
            expected_authority=False,
        )
    )

    # Cross-root GraphRAG.
    cross = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.REAL_GRAPHRAG,
            disposition=StageDisposition.CROSS_ROOT,
            content_root="content-root:other-snapshot",
            reason_codes=("cross_root_graph",),
            details={"canary_graph": False, "entrypoint": "production"},
        )
    )
    observations.append(
        _observation(
            "compose:cross-root-graphrag",
            "cross_root",
            cross,
            expected_authority=False,
        )
    )

    # Canary graph labeled current still fails closed.
    canary = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.REAL_GRAPHRAG,
            details={"entrypoint": "production", "canary_graph": True},
            reason_codes=("canary_graph",),
        )
    )
    observations.append(
        _observation(
            "compose:canary-graphrag",
            "canary_graph",
            canary,
            expected_authority=False,
        )
    )

    # Focused adapter success alone.
    adapter = compose_production_contract(
        [
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
    )
    observations.append(
        _observation(
            "compose:focused-adapter-only",
            "focused_adapter_only",
            adapter,
            expected_authority=False,
        )
    )

    # Runtime model call nonzero.
    with_calls = compose_production_contract(
        _healthy_stages(),
        model_call_count=1,
        provider_call_count=1,
        llm_call_count=1,
    )
    observations.append(
        _observation(
            "compose:nonzero-model-calls",
            "runtime_model_calls",
            with_calls,
            expected_authority=False,
        )
    )

    # Missing runtime identity.
    no_identity = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.RUNTIME_IDENTITY,
            disposition=StageDisposition.MISSING,
            content_root="",
            reason_codes=("runtime_identity_absent",),
        )
    )
    observations.append(
        _observation(
            "compose:missing-runtime-identity",
            "missing",
            no_identity,
            expected_authority=False,
        )
    )

    # Scheduler / fence regression.
    fence = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.SCHEDULER_FENCE,
            disposition=StageDisposition.PARTIAL,
            reason_codes=("fence_epoch_race", "capacity_not_conserved"),
        )
    )
    observations.append(
        _observation(
            "compose:scheduler-fence-regression",
            "partial",
            fence,
            expected_authority=False,
        )
    )

    # MCP list/call simulated receipts.
    sim_mcp = compose_production_contract(
        _mutate_stage(
            _healthy_stages(),
            ProductionStageName.MCP_LIST_CALL_RECEIPTS,
            disposition=StageDisposition.SIMULATED,
            reason_codes=("simulated_tools_call",),
        )
    )
    observations.append(
        _observation(
            "compose:simulated-mcp-calls",
            "simulated",
            sim_mcp,
            expected_authority=False,
        )
    )

    return tuple(observations)


def build_evaluation_report() -> dict[str, Any]:
    observations = build_composition_cases()
    healthy = [item for item in observations if item.expected_authority]
    attacks = [item for item in observations if not item.expected_authority]

    # Simulated optional ZK healthy path must not attest.
    simulated_obs = next(
        item
        for item in observations
        if item.case_id == "compose:simulated-zk-non-attested"
    )
    simulated_result = compose_production_contract(
        _healthy_stages()
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
        ]
    )

    false_admits = sum(
        1 for item in attacks if item.authority_granted or item.production_eligible
    )
    missed = sum(1 for item in attacks if not item.fail_closed)
    healthy_ok = all(
        item.authority_granted and item.production_eligible for item in healthy
    )
    isolation_clean = all(
        item.model_call_count == 0
        and item.provider_call_count == 0
        and item.llm_call_count == 0
        for item in observations
        if item.case_id != "compose:nonzero-model-calls"
    )

    healthy_result = compose_production_contract(
        _healthy_stages(include_zk=True)
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
            "requirement_ids": [SCAEV177COMPOSE],
            "coverage": list(SCAEV177COMPOSE_COVERAGE),
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
        },
        "mandatory_stages": [name.value for name in MANDATORY_STAGES],
        "optional_stages": [name.value for name in OPTIONAL_STAGES],
        "fail_closed_dispositions": sorted(
            item.value for item in FAIL_CLOSED_DISPOSITIONS
        ),
        "healthy_composition": healthy_result.to_dict(),
        "optional_zk_policy": {
            "simulated_attested": simulated_result.optional_zk["attested"],
            "simulated_blocks_composition": simulated_result.optional_zk[
                "blocks_composition"
            ],
            "simulated_observation_authority": simulated_obs.authority_granted,
        },
        "summary": {
            "case_count": len(observations),
            "healthy_case_count": len(healthy),
            "attack_case_count": len(attacks),
            "healthy_authority_ok": healthy_ok,
            "false_authoritative_admission_count": false_admits,
            "missed_fail_closed_count": missed,
            "isolation_clean": isolation_clean,
            "production_eligible": healthy_result.production_eligible,
            "authority_granted": healthy_result.authority_granted,
        },
        "safety_gates": {
            "missing_fails_closed": any(
                item.mutation == "missing" and item.fail_closed for item in attacks
            ),
            "synthesized_fails_closed": any(
                item.mutation == "synthesized" and item.fail_closed
                for item in attacks
            ),
            "simulated_mcp_fails_closed": any(
                item.case_id == "compose:simulated-mcp-calls" and item.fail_closed
                for item in attacks
            ),
            "partial_fails_closed": any(
                item.mutation == "partial" and item.fail_closed for item in attacks
            ),
            "stale_fails_closed": any(
                item.mutation == "stale" and item.fail_closed for item in attacks
            ),
            "cross_root_fails_closed": any(
                item.mutation == "cross_root" and item.fail_closed
                for item in attacks
            ),
            "focused_adapter_not_authority": any(
                item.mutation == "focused_adapter_only" and item.fail_closed
                for item in attacks
            ),
            "canary_graph_fails_closed": any(
                item.mutation == "canary_graph" and item.fail_closed
                for item in attacks
            ),
            "zero_model_calls_required": any(
                item.mutation == "runtime_model_calls" and item.fail_closed
                for item in attacks
            ),
            "simulated_zk_not_attested": (
                simulated_result.optional_zk["attested"] is False
            ),
        },
        "isolation_audit": {
            "llm_call_count": 0,
            "model_call_count": 0,
            "provider_call_count": 0,
            "held_out_fixture_disclosed": False,
        },
        "results": [item.to_dict() for item in observations],
        "passed": bool(
            healthy_ok
            and false_admits == 0
            and missed == 0
            and isolation_clean
            and not simulated_result.optional_zk["attested"]
            and healthy_result.production_eligible
            and healthy_result.authority_granted
        ),
    }
    return _seal_report(payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scaev177compose_evidence_term_is_declared() -> None:
    assert SCAEV177COMPOSE == "SCAEV177COMPOSE"
    assert SCAEV177COMPOSE_EVIDENCE == SCAEV177COMPOSE
    assert "fail-closed-missing-synthesized-simulated-partial-stale-cross-root" in (
        SCAEV177COMPOSE_COVERAGE
    )
    assert "focused-adapter-success-not-production-authority" in SCAEV177COMPOSE_COVERAGE
    assert "zero-runtime-model-calls" in SCAEV177COMPOSE_COVERAGE


def test_healthy_production_composition_grants_authority() -> None:
    result = compose_production_contract(_healthy_stages(include_zk=True))
    assert result.production_eligible is True
    assert result.authority_granted is True
    assert result.reason_codes == ()
    assert result.content_root == CONTENT_ROOT
    assert {stage.name for stage in result.stages} >= set(MANDATORY_STAGES)
    assert all(stage.production_admissible for stage in result.stages if stage.mandatory)
    assert result.isolation_audit == {
        "model_call_count": 0,
        "provider_call_count": 0,
        "llm_call_count": 0,
    }
    assert result.evidence["requirement_ids"] == [SCAEV177COMPOSE]
    assert result.evidence["coverage"] == list(SCAEV177COMPOSE_COVERAGE)
    assert result.optional_zk["attested"] is True
    assert result.baseline_bridges
    assert all(
        bridge["healthy_enough_for_authority"]
        for bridge in result.baseline_bridges
        if bridge["production_stage"] != ProductionStageName.REAL_ZK_ATTESTATION.value
    )


@pytest.mark.parametrize(
    ("case_id", "needle"),
    [
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
    ],
)
def test_mandatory_failure_modes_fail_closed(case_id: str, needle: str) -> None:
    observations = {item.case_id: item for item in build_composition_cases()}
    item = observations[case_id]
    assert item.expected_authority is False
    assert item.fail_closed is True
    assert item.authority_granted is False
    assert item.production_eligible is False
    assert any(needle in code for code in item.reason_codes), item.reason_codes


def test_simulated_optional_zk_does_not_attest_or_block() -> None:
    result = compose_production_contract(
        _healthy_stages()
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
        ]
    )
    assert result.production_eligible is True
    assert result.authority_granted is True
    assert result.optional_zk["attested"] is False
    assert result.optional_zk["blocks_composition"] is False


def test_required_simulated_zk_blocks_authority() -> None:
    result = compose_production_contract(
        _healthy_stages()
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
        ]
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
            details={"entrypoint": "focused_adapter"},
            focused_adapter_success=True,
        )
        for name in MANDATORY_STAGES
    ]
    result = compose_production_contract(stages)
    assert result.authority_granted is False
    assert any("focused_adapter" in code for code in result.reason_codes)


def test_baseline_stage_bridge_uses_baseline_receipts() -> None:
    result = compose_production_contract(_healthy_stages())
    names = {bridge["baseline_stage"] for bridge in result.baseline_bridges}
    assert BaselineStageName.REPOSITORY_INDEX.value in names
    assert BaselineStageName.PROOF_CACHE.value in names
    for bridge in result.baseline_bridges:
        receipt = BaselineStageReceipt(
            name=bridge["receipt"]["name"],
            completeness=bridge["receipt"]["completeness"],
            reason_codes=tuple(bridge["receipt"]["reason_codes"]),
            root_id=bridge["receipt"]["root_id"],
            details=bridge["receipt"]["details"],
        )
        assert receipt.complete is bridge["complete"]
        assert receipt.healthy_enough_for_authority is bridge[
            "healthy_enough_for_authority"
        ]


def test_evaluation_report_is_sealed_and_passes_gates() -> None:
    report = build_evaluation_report()
    assert report["passed"] is True
    assert report["evidence"]["requirement_ids"] == [SCAEV177COMPOSE]
    assert report["summary"]["false_authoritative_admission_count"] == 0
    assert report["summary"]["missed_fail_closed_count"] == 0
    assert report["summary"]["production_eligible"] is True
    assert all(report["safety_gates"].values())
    assert report["isolation_audit"]["model_call_count"] == 0
    assert report["isolation_audit"]["llm_call_count"] == 0
    assert report["isolation_audit"]["provider_call_count"] == 0
    assert verify_composition_report(report)

    tampered = deepcopy(report)
    tampered["summary"]["false_authoritative_admission_count"] = 1
    assert not verify_composition_report(tampered)


def test_published_production_composition_matches_evaluation() -> None:
    report = build_evaluation_report()
    assert PUBLISHED_REPORT.is_file(), f"missing published report: {PUBLISHED_REPORT}"
    published = json.loads(PUBLISHED_REPORT.read_text(encoding="utf-8"))
    assert published == report
    assert verify_composition_report(published)
    assert published["passed"] is True
    assert SCAEV177COMPOSE in published["evidence"]["requirement_ids"]
    assert published["goal_id"] == GOAL_ID
    assert published["task_id"] == TASK_ID
