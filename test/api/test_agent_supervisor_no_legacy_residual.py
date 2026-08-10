"""DCR-080 has no model-provider authority on its live deterministic route."""

from __future__ import annotations

import inspect

from ipfs_accelerate_py.agent_supervisor.control import pre_implementation_provider_gate
from ipfs_accelerate_py.agent_supervisor.todo_daemon import deterministic_repair_composition
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon


def test_dcr080_control_gate_cannot_authorize_a_provider() -> None:
    gate = pre_implementation_provider_gate.evaluate_provider_gate(task_id="DCR-080")
    assert gate.provider_authorized is False
    assert gate.skip_provider is True
    assert gate.provider_hook_count == 0


def test_dcr080_live_route_has_no_provider_invocation_surface() -> None:
    source = inspect.getsource(deterministic_repair_composition)
    assert "assert_provider_dispatch_allowed" not in source
    assert "allow_legacy_residual" not in source
    assert "residual_packet" not in source


def test_daemon_selects_composition_before_any_provider_route() -> None:
    source = inspect.getsource(implementation_daemon.PortalImplementationDaemon._run_implementation)
    assert "or self._task_requires_dcr080_deterministic_repair(task)" in source


def test_daemon_provider_gate_is_a_terminal_denial() -> None:
    source = inspect.getsource(
        implementation_daemon.PortalImplementationDaemon._assert_current_provider_gate
    )
    assert "raise PermissionError" in source
    assert "authoring_provider_invocation_authorized" not in source
