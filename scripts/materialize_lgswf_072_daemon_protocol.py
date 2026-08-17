#!/usr/bin/env python3
"""LGSWF-072: add integration artifacts without rewriting the daemon framework."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from lgswf_emit_copy_writer import copy_pairs, emit

PAIRS = (
    (
        "scripts/lgswf_payloads/ipfs_datasets_quack_security.py",
        "ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_quack_security.py",
    ),
    (
        "scripts/lgswf_payloads/test_agent_supervisor_daemon_protocol_integration.py",
        "test/api/test_agent_supervisor_daemon_protocol_integration.py",
    ),
    (
        "scripts/lgswf_payloads/test_agent_supervisor_ipfs_datasets_quack_security.py",
        "test/api/test_agent_supervisor_ipfs_datasets_quack_security.py",
    ),
)

EXPECTED = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_transactions.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/runtime_cas.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
    "scripts/ops/agent_supervisor/quack_state_server.py",
    "ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_quack_security.py",
    "test/api/test_agent_supervisor_daemon_protocol_integration.py",
    "test/api/test_agent_supervisor_control_plane_repository.py",
    "test/api/test_agent_supervisor_control_plane_contracts.py",
    "test/api/test_agent_supervisor_quack_state_client.py",
    "test/api/test_agent_supervisor_quack_state_server.py",
    "test/api/test_agent_supervisor_quack_capabilities.py",
    "test/api/test_agent_supervisor_intent_repository.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_agent_supervisor_ipfs_datasets_quack_security.py",
)


if __name__ == "__main__":
    dest = Path.cwd()
    result = copy_pairs(PAIRS, dest=dest)
    present = [path for path in EXPECTED if (dest / path).exists()]
    staged = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *present],
        cwd=dest,
        text=True,
        capture_output=True,
        check=False,
    )
    result["expected_staged"] = present
    result["expected_stage_returncode"] = staged.returncode
    emit(result)
