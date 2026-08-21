"""Accelerator consumer of datasets/kit assurance authorities (PCCE-015).

Campaign execution stays in the inventoried AssuranceCampaignApi. This bridge
exposes bounded outcomes to the v0.1 lifecycle and forbids self-approval or
simulated promotion.
"""

from __future__ import annotations

from typing import Any, Mapping

from ipfs_accelerate_py.proof_context.dependencies import DependencyUnavailable

DATASETS_SPEC = "ipfs_datasets_py.proof_context.assurance_specification"
KIT_STORE = "ipfs_kit_py.adversarial_assurance_store"
RUNTIME = "ipfs_accelerate_py.agent_supervisor.adversarial_assurance"

TYPED_OUTCOMES = (
    "omission",
    "vacuity",
    "critical_survivor",
    "context_expansion",
    "timeout",
    "unavailable",
    "infrastructure_failure",
    "human_review_required",
    "succeeded",
    "rejected",
)


class AssuranceBridgeError(RuntimeError):
    reason = "invalid"


def open_campaign_api(**dependencies: Any) -> Any:
    try:
        from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api import (
            create_assurance_campaign_api,
        )
    except ImportError as exc:
        raise DependencyUnavailable(
            "assurance campaign API is unavailable; this is not success"
        ) from exc
    return create_assurance_campaign_api(**dependencies)


def admit_campaign_outcome(
    status: str,
    *,
    critical_survivor: bool = False,
    provenance: str = "live",
    self_approved: bool = False,
    hidden_benchmark_exposed: bool = False,
) -> dict[str, Any]:
    if self_approved:
        raise AssuranceBridgeError("assurance engine cannot self-approve outcomes")
    if hidden_benchmark_exposed:
        raise AssuranceBridgeError("hidden benchmark answers must not be exposed")
    if provenance == "simulated":
        raise AssuranceBridgeError("simulated assurance cannot be promoted to live")
    if critical_survivor:
        return {
            "accepted": False,
            "status": "critical_survivor",
            "authority": RUNTIME,
        }
    if status not in TYPED_OUTCOMES:
        raise AssuranceBridgeError(f"unknown assurance outcome {status!r}")
    accepted = status == "succeeded"
    if status in {"unavailable", "timeout", "infrastructure_failure"}:
        accepted = False
    return {
        "accepted": accepted,
        "status": status,
        "authority": RUNTIME,
        "datasets_spec": DATASETS_SPEC,
        "kit_store": KIT_STORE,
    }
