"""P1 runtime audit barrier tests for DCR-001 deterministic repair."""

from __future__ import annotations

import importlib
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.no_llm_policy import (
    DeterministicRepairAuthorityPolicy,
    NoLlmExecutionDenied,
    NoLlmExecutionGuard,
    RepairExecutionRoute,
)


@pytest.fixture(autouse=True)
def isolated_process_guard() -> None:
    NoLlmExecutionGuard.reset_audit_for_testing()
    yield
    NoLlmExecutionGuard.reset_audit_for_testing()


@pytest.fixture
def policy() -> DeterministicRepairAuthorityPolicy:
    return DeterministicRepairAuthorityPolicy(
        local_logic_pins=frozenset({"logic:runtime-barrier@1"}),
        prover_subprocess_pins=frozenset({"prover:lean@4.19"}),
        loopback_mcp_pins=frozenset({"mcp:repair-local@1"}),
    )


def test_provider_import_is_denied_before_temp_module_effect(
    policy: DeterministicRepairAuthorityPolicy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "provider-imported"
    (tmp_path / "openai.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unexpected')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    prior = sys.modules.pop("openai", None)
    try:
        with pytest.raises(NoLlmExecutionDenied, match="forbidden_model_provider_import"):
            policy.invoke(
                RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC,
                lambda: importlib.import_module("openai"),
                pin="logic:runtime-barrier@1",
            )
        assert not marker.exists()
    finally:
        sys.modules.pop("openai", None)
        if prior is not None:
            sys.modules["openai"] = prior


def test_subprocess_and_remote_socket_are_denied_before_effects(
    policy: DeterministicRepairAuthorityPolicy,
) -> None:
    with pytest.raises(NoLlmExecutionDenied, match="subprocess_forbidden_for_route"):
        policy.invoke(
            RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC,
            lambda: subprocess.run([sys.executable, "-c", "raise SystemExit(9)"], check=True),
            pin="logic:runtime-barrier@1",
        )
    with pytest.raises(NoLlmExecutionDenied, match="dynamic_exec_forbidden_for_route"):
        policy.invoke(
            RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC,
            lambda: exec("raise AssertionError('unexpected dynamic execution')"),
            pin="logic:runtime-barrier@1",
        )

    def remote_connect() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.connect(("203.0.113.1", 443))

    with pytest.raises(NoLlmExecutionDenied, match="non_admitted_loopback_network"):
        policy.invoke(
            RepairExecutionRoute.LOOPBACK_MCP,
            remote_connect,
            pin="mcp:repair-local@1",
            endpoint="http://127.0.0.1:8765/mcp",
        )
    with pytest.raises(NoLlmExecutionDenied, match="non_admitted_loopback_network"):
        policy.invoke(
            RepairExecutionRoute.LOOPBACK_MCP,
            lambda: socket.getaddrinfo("example.invalid", 8765),
            pin="mcp:repair-local@1",
            endpoint="http://127.0.0.1:8765/mcp",
        )


def test_prover_shell_commands_and_child_threads_fail_closed(
    policy: DeterministicRepairAuthorityPolicy,
) -> None:
    with pytest.raises(NoLlmExecutionDenied, match="prover_executable_binding_unavailable_defer"):
        policy.invoke(
            RepairExecutionRoute.PROVER_SUBPROCESS,
            lambda: subprocess.run(["lean", "--version"], check=True),
            pin="prover:lean@4.19",
        )

    child_error: list[BaseException] = []

    def child() -> None:
        try:
            subprocess.run([sys.executable, "-c", "raise SystemExit(8)"], check=True)
        except BaseException as exc:  # Audit exceptions should not escape the test thread.
            child_error.append(exc)

    def spawn_child() -> None:
        worker = threading.Thread(target=child)
        worker.start()
        worker.join()

    policy.invoke(
        RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC,
        spawn_child,
        pin="logic:runtime-barrier@1",
    )
    assert len(child_error) == 1
    assert isinstance(child_error[0], NoLlmExecutionDenied)
    assert child_error[0].reason == "unguarded_thread_audit_event"


def test_nested_loopback_context_cannot_leak_into_local_logic(
    policy: DeterministicRepairAuthorityPolicy,
) -> None:
    def local_callback() -> None:
        policy.invoke(
            RepairExecutionRoute.LOOPBACK_MCP,
            lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).close(),
            pin="mcp:repair-local@1",
            endpoint="http://127.0.0.1:8765/mcp",
        )
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()

    with pytest.raises(NoLlmExecutionDenied, match="network_forbidden_for_route"):
        policy.invoke(
            RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC,
            local_callback,
            pin="logic:runtime-barrier@1",
        )
