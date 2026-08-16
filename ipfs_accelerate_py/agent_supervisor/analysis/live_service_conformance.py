"""DCR-091 typed, pending-only three-service conformance assessment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from ..proof.formal_verification_contracts import content_identity
from .hermetic_conformance import HermeticConformanceReport
from .mcp_live_observer import McpObservationEpoch, is_current_mcp_observation_epoch
from .runtime_service_identity import RuntimeServiceIdentity, ServiceIdentityStatus

DCR091_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/live-service-conformance@1"


@dataclass(frozen=True)
class IndependentLogicComparisonReceipt:
    """Content-addressed comparison result; no raw expected/remote mapping."""

    local_result_cid: str
    local_proof_cid: str
    observed_result_cid: str
    observed_proof_cid: str

    @property
    def receipt_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/dcr-091-logic-comparison@1",
            "local_result_cid": self.local_result_cid,
            "local_proof_cid": self.local_proof_cid,
            "observed_result_cid": self.observed_result_cid,
            "observed_proof_cid": self.observed_proof_cid,
        }


@dataclass(frozen=True)
class LiveConformanceResult:
    disposition: str
    reason_codes: tuple[str, ...]
    conformance_cid: str = ""
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def assess_live_services(
    conformance: HermeticConformanceReport | None,
    identities: tuple[RuntimeServiceIdentity, ...],
    *,
    observation_epoch: McpObservationEpoch | None,
    graph_cid: str,
    semantic_roots: Mapping[str, str],
    snapshot_roots: Mapping[str, str],
    comparison: IndependentLogicComparisonReceipt | None,
) -> LiveConformanceResult:
    """Revalidate typed DCR-090/022/023 inputs without opening a transport."""

    if not isinstance(conformance, HermeticConformanceReport):
        return LiveConformanceResult("integration_pending", ("typed_dcr090_report_required",))
    body = conformance.to_dict()
    if (
        conformance.report_cid != content_identity(body)
        or body.get("structural_fixture") is not True
        or body.get("live_conformance") is not False
    ):
        return LiveConformanceResult(
            "integration_pending", ("dcr090_report_forged_or_not_structural",)
        )
    if (
        not isinstance(identities, tuple)
        or len(identities) != 3
        or not all(isinstance(item, RuntimeServiceIdentity) for item in identities)
    ):
        return LiveConformanceResult(
            "integration_pending", ("three_typed_dcr022_identities_required",)
        )
    roles = {item.observation.role for item in identities}
    if roles != {"accelerate", "datasets", "kit"}:
        return LiveConformanceResult(
            "integration_pending", ("exact_accelerate_datasets_kit_roles_required",)
        )
    if any(item.status is not ServiceIdentityStatus.VALID or item.reasons for item in identities):
        return LiveConformanceResult(
            "integration_pending", ("current_dcr022_process_witnesses_required",)
        )
    if not isinstance(
        observation_epoch, McpObservationEpoch
    ) or not is_current_mcp_observation_epoch(
        observation_epoch,
        graph_cid=graph_cid,
        semantic_roots=semantic_roots,
        snapshot_roots=snapshot_roots,
    ):
        return LiveConformanceResult(
            "integration_pending", ("current_dcr023_epoch_with_exact_roots_required",)
        )
    if not isinstance(comparison, IndependentLogicComparisonReceipt) or not all(
        comparison.to_dict().values()
    ):
        return LiveConformanceResult(
            "integration_pending", ("typed_independent_logic_comparison_required",)
        )
    if (
        comparison.local_result_cid != comparison.observed_result_cid
        or comparison.local_proof_cid != comparison.observed_proof_cid
    ):
        return LiveConformanceResult(
            "integration_pending", ("independent_logic_equivalence_not_proved",)
        )
    # DCR-090 is deliberately structural, so no collection of current-shaped
    # values can mint a live conformance success until a non-fixture receipt is
    # defined and observed.
    return LiveConformanceResult(
        "integration_pending",
        ("dcr090_structural_fixture_cannot_authorize_live_conformance",),
        content_identity(
            {
                "schema": DCR091_SCHEMA,
                "dcr090": conformance.report_cid,
                "epoch": observation_epoch.epoch_cid,
                "comparison": comparison.receipt_cid,
            }
        ),
    )


__all__ = [
    "DCR091_SCHEMA",
    "IndependentLogicComparisonReceipt",
    "LiveConformanceResult",
    "assess_live_services",
]
