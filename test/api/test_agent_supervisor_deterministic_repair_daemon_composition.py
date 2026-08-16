"""Focused DCR-080 daemon composition safety tests."""

from __future__ import annotations

import inspect

from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_composition import (
    DCR080_COMPOSITION_SCHEMA,
    DeterministicRepairCompositionDisposition,
    DeterministicRepairCompositionRoot,
    run_deterministic_repair,
)


def test_missing_doctor_binding_defers_before_legacy_empty_success() -> None:
    result = run_deterministic_repair(task_id="DCR-080")

    assert result.disposition is DeterministicRepairCompositionDisposition.DEFER_CAPABILITY
    assert result.transitions == ("dcr050_doctor_reinspection",)
    assert "dcr050_not_current_live" in result.reason_codes
    payload = result.to_dict()
    assert payload["schema"] == DCR080_COMPOSITION_SCHEMA
    assert payload["execution_authorized"] is False
    assert payload["mutation_authorized"] is False
    assert payload["completion_authorized"] is False
    assert payload["model_call_count"] == 0
    assert payload["provider_call_count"] == 0
    assert payload["network_call_count"] == 0
    assert result.receipt_cid


def test_callable_or_provider_like_route_is_rejected_before_doctor() -> None:
    result = DeterministicRepairCompositionRoot().run(
        task_id="DCR-080-callable",
        doctor_binding={"forged": lambda: None},
    )

    assert result.disposition is DeterministicRepairCompositionDisposition.REJECTED
    assert result.reason_codes == ("callable_or_dynamic_route_rejected",)
    assert result.doctor_composition_cid == ""


def test_daemon_has_separate_dcr080_hook_and_never_uses_provider_gate() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    assert hasattr(PortalImplementationDaemon, "_run_dcr080_deterministic_repair_composition")
    source = inspect.getsource(PortalImplementationDaemon._run_implementation_in_ephemeral_worktree)
    assert "dcr080_deterministic_repair_composition_evaluated" in source
    assert "dcr080_deterministic_repair_deferred" in source
    assert "_evaluate_pre_implementation_provider_gate" in source
    dcr080_source = inspect.getsource(
        PortalImplementationDaemon._run_dcr080_deterministic_repair_composition
    )
    assert "run_deterministic_repair" in dcr080_source
    assert "evaluate_provider_gate" not in dcr080_source
