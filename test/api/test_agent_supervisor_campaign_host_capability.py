from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.campaign_host_capability import (
    HUB_CURRENT_BUNDLE_BLOCK_REASON,
    RESIDENTIAL_PROXY_BLOCK_REASON,
    HostCapabilityClass,
    HostCapabilityPolicy,
    classify_lcr_demand,
    host_has_snapshot_builder,
    should_retry_stalled_task,
    snapshot_builder_cmdline_markers,
)


def test_lcr_084_is_host_blocked_without_residential_proxy() -> None:
    decision = classify_lcr_demand(
        "LCR-084",
        text="require-live-official exhaustive official evidence for exactly 51",
    )
    assert decision.classification is HostCapabilityClass.HOST_BLOCKED
    assert decision.retry_implementation is False
    assert decision.reason == RESIDENTIAL_PROXY_BLOCK_REASON


def test_lcr_084_retries_only_when_proxy_is_available() -> None:
    decision = classify_lcr_demand(
        "LCR-084",
        policy=HostCapabilityPolicy(residential_proxy_available=True),
    )
    assert decision.retry_implementation is True


def test_snapshot_research_is_feasible() -> None:
    decision = classify_lcr_demand(
        "LCR-090",
        text="snapshot_research GraphRAG over tmp_laws and local JSON-LD",
    )
    assert decision.classification is HostCapabilityClass.RD_FEASIBLE
    assert decision.retry_implementation is True
    policy = HostCapabilityPolicy()
    assert policy.snapshot_hub_publish_allowed is True
    assert policy.hub_current_bundle_authorized is False


def test_current_bundle_publication_is_gated() -> None:
    decision = classify_lcr_demand(
        "LCR-039",
        text="authorizing_for_publication current-bundle justicedao/ipfs_state_laws",
    )
    assert decision.classification is HostCapabilityClass.PUBLICATION_GATED
    assert decision.reason == HUB_CURRENT_BUNDLE_BLOCK_REASON
    assert decision.retry_implementation is False


def test_stall_reason_with_live_fetch_block_does_not_retry() -> None:
    decision = should_retry_stalled_task(
        "LCR-023",
        stall_reason="live_official_fetch_blocked: residential_proxy_not_used",
    )
    assert decision.retry_implementation is False
    assert decision.classification is HostCapabilityClass.HOST_BLOCKED


def test_snapshot_builder_cmdline_is_detected() -> None:
    assert "build_state_laws_snapshot_sparse_graphrag.py" in snapshot_builder_cmdline_markers()
    assert host_has_snapshot_builder(
        [
            {
                "argv": [
                    "python",
                    "scripts/ops/legal_data/build_state_laws_snapshot_sparse_graphrag.py",
                    "--prefer-real-embeddings",
                ]
            }
        ]
    )
    assert not host_has_snapshot_builder([{"argv": ["python", "pytest"]}])
