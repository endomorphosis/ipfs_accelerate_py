"""DCR-080 live daemon hook tests.

The deterministic worker receives its Planner/Doctor decision through the
receipt-bound composition root.  It is never a model-provider authorization
hook.
"""

from __future__ import annotations

import inspect

import pytest

from ipfs_accelerate_py.agent_supervisor.control.pre_implementation_provider_gate import (
    EVENT_PRE_IMPLEMENTATION_KERNEL,
    PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE,
    assert_provider_dispatch_allowed,
    evaluate_provider_gate,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


def test_control_gate_has_the_deterministic_composition_identity() -> None:
    assert (
        PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE
        == "DeterministicRepairCompositionRoot@pre_implementation_kernel"
    )


def test_gate_records_receipt_but_never_authorizes_provider() -> None:
    decision = evaluate_provider_gate(
        task_id="DCR-080",
        service_receipt_ids=("doctor-receipt", "planner-receipt"),
    )

    event = decision.to_event_payload(task_id="DCR-080", attempt=1)
    assert event["event"] == EVENT_PRE_IMPLEMENTATION_KERNEL
    assert event["receipt_cid"] == decision.receipt_cid
    assert event["provider_authorized"] is False
    assert event["provider_hook_count"] == 0
    with pytest.raises(PermissionError, match="forbidden"):
        assert_provider_dispatch_allowed(decision)


def test_missing_service_receipt_abstains() -> None:
    decision = evaluate_provider_gate(task_id="DCR-080")

    assert decision.disposition == "abstain"
    assert decision.kernel.receipt.reason_codes == ("missing_service_receipt",)
    assert decision.skip_provider is True


def test_daemon_wires_the_receipt_bound_composition() -> None:
    assert hasattr(PortalImplementationDaemon, "_run_dcr080_deterministic_repair_composition")
    source = inspect.getsource(
        PortalImplementationDaemon._run_dcr080_deterministic_repair_composition
    )
    assert "run_deterministic_repair" in source
    assert "checkout_root=worktree_path or self.repo_root" in source
