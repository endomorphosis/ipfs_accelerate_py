"""Host-capability gates for Legal Corpora Reindex (LCR) R&D.

The LCR board contains production-acceptance tasks that this development host
cannot finish: official live fetches need a residential proxy, and current-bundle
Hub publication is a sealed authorization boundary. Retrying those tasks forever
stalls the supervisor. Snapshot/research GraphRAG and fail-closed local work
remain feasible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final


CAMPAIGN_HOST_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/campaign-host-capability@1"
)
RESIDENTIAL_PROXY_BLOCK_REASON: Final = "residential_proxy_not_used"
HUB_CURRENT_BUNDLE_BLOCK_REASON: Final = "current_bundle_not_authorized"
SNAPSHOT_RESEARCH_LANE: Final = "snapshot_research"

_LIVE_OFFICIAL_TASK_IDS: Final[frozenset[str]] = frozenset({"LCR-084"})
_LIVE_OFFICIAL_MARKERS: Final[tuple[str, ...]] = (
    "require-live-official",
    "require_live_official",
    "residential proxy",
    "residential_proxy",
    "fresh exhaustive official evidence for exactly 51",
    "reobserves all official state",
    "closed_frontier_required",
    "official_source_receipt_required",
)
_PUBLICATION_MARKERS: Final[tuple[str, ...]] = (
    "current-bundle",
    "current_bundle",
    "authorizing_for_publication",
    "justicedao/ipfs_state_laws",
    "justicedao/ipfs_federal_register",
)
_SNAPSHOT_MARKERS: Final[tuple[str, ...]] = (
    "snapshot_research",
    "snapshot-graphrag",
    "skillcenter",
    "tmp_laws",
)


class HostCapabilityClass(str, Enum):
    RD_FEASIBLE = "rd_feasible"
    HOST_BLOCKED = "host_blocked"
    PUBLICATION_GATED = "publication_gated"


@dataclass(frozen=True)
class HostCapabilityPolicy:
    """What this machine is allowed to attempt."""

    residential_proxy_available: bool = False
    hub_current_bundle_authorized: bool = False
    snapshot_research_allowed: bool = True
    snapshot_hub_publish_allowed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "hub_current_bundle_authorized": self.hub_current_bundle_authorized,
            "residential_proxy_available": self.residential_proxy_available,
            "schema": CAMPAIGN_HOST_CAPABILITY_SCHEMA,
            "snapshot_hub_publish_allowed": self.snapshot_hub_publish_allowed,
            "snapshot_research_allowed": self.snapshot_research_allowed,
        }


@dataclass(frozen=True)
class HostCapabilityDecision:
    classification: HostCapabilityClass
    reason: str
    retry_implementation: bool
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "reason": self.reason,
            "retry_implementation": self.retry_implementation,
            "schema": CAMPAIGN_HOST_CAPABILITY_SCHEMA,
            "task_id": self.task_id,
        }


def _haystack(task_id: str, text: str) -> str:
    return f"{task_id}\n{text}".lower()


def classify_lcr_demand(
    task_id: str,
    *,
    text: str = "",
    policy: HostCapabilityPolicy | None = None,
) -> HostCapabilityDecision:
    """Classify one LCR task against this host's capabilities."""

    pin = policy or HostCapabilityPolicy()
    ident = str(task_id or "").strip().upper()
    blob = _haystack(ident, text)
    if ident in _LIVE_OFFICIAL_TASK_IDS or any(
        marker in blob for marker in _LIVE_OFFICIAL_MARKERS
    ):
        if pin.residential_proxy_available:
            return HostCapabilityDecision(
                classification=HostCapabilityClass.RD_FEASIBLE,
                reason="residential_proxy_available",
                retry_implementation=True,
                task_id=ident,
            )
        return HostCapabilityDecision(
            classification=HostCapabilityClass.HOST_BLOCKED,
            reason=RESIDENTIAL_PROXY_BLOCK_REASON,
            retry_implementation=False,
            task_id=ident,
        )
    if any(marker in blob for marker in _PUBLICATION_MARKERS) and not any(
        marker in blob for marker in _SNAPSHOT_MARKERS
    ):
        if pin.hub_current_bundle_authorized:
            return HostCapabilityDecision(
                classification=HostCapabilityClass.RD_FEASIBLE,
                reason="hub_current_bundle_authorized",
                retry_implementation=True,
                task_id=ident,
            )
        return HostCapabilityDecision(
            classification=HostCapabilityClass.PUBLICATION_GATED,
            reason=HUB_CURRENT_BUNDLE_BLOCK_REASON,
            retry_implementation=False,
            task_id=ident,
        )
    return HostCapabilityDecision(
        classification=HostCapabilityClass.RD_FEASIBLE,
        reason="rd_development_host",
        retry_implementation=True,
        task_id=ident,
    )


def should_retry_stalled_task(
    task_id: str,
    *,
    stall_reason: str = "",
    text: str = "",
    policy: HostCapabilityPolicy | None = None,
) -> HostCapabilityDecision:
    """Fail closed: do not retry host-impossible LCR work as if it were a hang."""

    decision = classify_lcr_demand(task_id, text=text, policy=policy)
    combined = f"{stall_reason} {text}".lower()
    if decision.retry_implementation and (
        RESIDENTIAL_PROXY_BLOCK_REASON in combined
        or "live_official_fetch_blocked" in combined
    ):
        return HostCapabilityDecision(
            classification=HostCapabilityClass.HOST_BLOCKED,
            reason=RESIDENTIAL_PROXY_BLOCK_REASON,
            retry_implementation=False,
            task_id=decision.task_id,
        )
    return decision


def snapshot_builder_cmdline_markers() -> tuple[str, ...]:
    """Process markers that count as live GraphRAG progress even without log lines."""

    return (
        "build_state_laws_snapshot_sparse_graphrag.py",
        "snapshot-graphrag-v2026.08.31-all51-cuda",
    )


def host_has_snapshot_builder(processes: list[Mapping[str, Any]]) -> bool:
    markers = snapshot_builder_cmdline_markers()
    for proc in processes:
        argv = proc.get("argv") or proc.get("cmdline") or ()
        if isinstance(argv, str):
            line = argv
        else:
            line = " ".join(str(item) for item in argv)
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in markers):
            return True
    return False
