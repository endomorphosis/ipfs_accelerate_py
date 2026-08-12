"""WPD-021: ImplementationDaemon pre-implementation provider gate tests.

Acceptance:

* Monkeypatched provider is not called for ``closed_deterministic``.
* Residual path requires a residual packet CID.
* Events include kernel receipt identity.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
    ImplementationForestRoots,
    implementation_disposition_cid,
    provider_invocation_authorized,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_kernel import (
    AnalyticalRepairCandidate,
    PreImplementationKernel,
    REASON_ANALYTICAL_UNIQUE_MAPPING,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_provider_gate import (
    EVENT_PRE_IMPLEMENTATION_KERNEL,
    PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE,
    ProviderGateDecision,
    assert_provider_dispatch_allowed,
    evaluate_provider_gate,
)


def _cid(name: str) -> str:
    return implementation_disposition_cid({"fixture": name})


@pytest.fixture
def forest() -> ImplementationForestRoots:
    return ImplementationForestRoots(
        repository_id="repository:sha256:test",
        repository_forest_cid=_cid("forest"),
        git_tree_id=_cid("tree"),
        policy_root=_cid("policy"),
    )


def test_interface_identity() -> None:
    assert (
        PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE
        == "ImplementationDaemon@pre_implementation_kernel"
    )


def test_closed_deterministic_blocks_provider(forest: ImplementationForestRoots) -> None:
    provider_calls = {"count": 0}

    def fake_provider() -> None:
        provider_calls["count"] += 1

    decision = evaluate_provider_gate(
        task_cid=_cid("task"),
        forest_roots=forest,
        analytical_candidates=(
            AnalyticalRepairCandidate(
                candidate_id="only",
                reason_code=REASON_ANALYTICAL_UNIQUE_MAPPING,
            ),
        ),
        allow_legacy_residual=False,
    )
    assert decision.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC
    assert decision.skip_provider is True
    assert decision.provider_authorized is False
    assert decision.provider_hook_count == 0
    assert decision.receipt_cid

    with pytest.raises(PermissionError, match="provider dispatch blocked"):
        assert_provider_dispatch_allowed(decision)
        fake_provider()
    assert provider_calls["count"] == 0


def test_residual_path_requires_packet(forest: ImplementationForestRoots) -> None:
    # Explicit residual without packet and no legacy fallback → not authorized.
    decision = evaluate_provider_gate(
        task_cid=_cid("task"),
        forest_roots=forest,
        residual_packet_cid="",
        analytical_candidates=(),
        allow_legacy_residual=False,
    )
    # No candidates → abstain; provider blocked.
    assert decision.provider_authorized is False
    assert decision.skip_provider is True
    with pytest.raises(PermissionError):
        assert_provider_dispatch_allowed(decision)

    packet = _cid("residual-packet")
    authorized = evaluate_provider_gate(
        task_cid=_cid("task"),
        forest_roots=forest,
        residual_packet_cid=packet,
        allow_legacy_residual=False,
    )
    assert authorized.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    assert authorized.provider_authorized is True
    assert authorized.skip_provider is False
    assert authorized.residual_packet_cid == packet
    assert_provider_dispatch_allowed(authorized)


def test_event_payload_includes_receipt_identity(
    forest: ImplementationForestRoots,
) -> None:
    decision = evaluate_provider_gate(
        task_cid=_cid("task-event"),
        forest_roots=forest,
        analytical_candidates=(
            AnalyticalRepairCandidate(candidate_id="only"),
        ),
        allow_legacy_residual=False,
    )
    event = decision.to_event_payload(task_id="WPD-021", attempt=1)
    assert event["event"] == EVENT_PRE_IMPLEMENTATION_KERNEL
    assert event["task_id"] == "WPD-021"
    assert event["attempt"] == 1
    assert event["receipt_cid"] == decision.receipt_cid
    assert event["disposition"] == "closed_deterministic"
    assert event["kernel_receipt"]["task_cid"] == _cid("task-event")
    assert "content_id" in dir(decision.receipt)  # content-addressed receipt
    assert decision.receipt_cid


def test_injectable_kernel_controls_disposition(
    forest: ImplementationForestRoots,
) -> None:
    kernel = PreImplementationKernel(planner_available=False)
    decision = evaluate_provider_gate(
        task_cid=_cid("task"),
        forest_roots=forest,
        kernel=kernel,
        allow_legacy_residual=False,
    )
    assert decision.disposition is ImplementationDisposition.DEFER_CAPABILITY
    assert decision.skip_provider is True
    assert not provider_invocation_authorized(decision.disposition)


def test_legacy_residual_keeps_model_path_reachable(
    forest: ImplementationForestRoots,
) -> None:
    decision = evaluate_provider_gate(
        task_cid=_cid("task"),
        forest_roots=forest,
        allow_legacy_residual=True,
    )
    assert decision.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    assert decision.provider_authorized is True
    assert decision.residual_packet_cid
    assert_provider_dispatch_allowed(decision)


def test_daemon_method_exists_and_shapes_gate() -> None:
    """Smoke: PortalImplementationDaemon exposes the WPD-021 gate method."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    assert hasattr(PortalImplementationDaemon, "_evaluate_pre_implementation_provider_gate")
    assert hasattr(PortalImplementationDaemon, "_run_implementation_in_ephemeral_worktree")
    source = inspect_source_contains_gate_hook()
    assert "pre_implementation_kernel_evaluated" in source
    assert "skip_provider" in source


def inspect_source_contains_gate_hook() -> str:
    import inspect
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as mod

    return inspect.getsource(mod.PortalImplementationDaemon._run_implementation_in_ephemeral_worktree)
