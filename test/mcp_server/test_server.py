from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import (
    HierarchicalToolManager,
)
from ipfs_accelerate_py.mcp_server.server import (
    configure_agent_supervisor_tools,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    agent_supervisor_service_resolution_count,
)


def test_server_registers_agent_supervisor_as_lazy_category() -> None:
    manager = HierarchicalToolManager()
    resolutions_before = agent_supervisor_service_resolution_count()

    configure_agent_supervisor_tools(manager)

    assert "agent_supervisor" in manager.list_categories()
    assert agent_supervisor_service_resolution_count() == resolutions_before
    assert {item["name"] for item in manager.list_tools("agent_supervisor")} == {
        operation.value for operation in Operation
    }
    schema = manager.get_tool_schema("agent_supervisor", "pause")
    assert (
        schema["input_schema"]["properties"]["request"]["properties"][
            "operation"
        ]["const"]
        == "pause"
    )
    assert agent_supervisor_service_resolution_count() == resolutions_before


def test_cold_server_discovery_does_not_load_supervisor_providers_or_service(
    tmp_path: Path,
) -> None:
    script = tmp_path / "cold_discovery.py"
    script.write_text(
        """
import json
import subprocess
import sys

from ipfs_accelerate_py.mcp_server.hierarchical_tool_manager import (
    HierarchicalToolManager,
)
from ipfs_accelerate_py.mcp_server.server import (
    configure_agent_supervisor_tools,
)

manager = HierarchicalToolManager()
configure_agent_supervisor_tools(manager)
categories = manager.list_categories()
before_tools = sorted(
    name for name in sys.modules
    if name.startswith("ipfs_accelerate_py.agent_supervisor")
)
process_starts = 0
original_popen = subprocess.Popen
def forbidden_popen(*args, **kwargs):
    global process_starts
    process_starts += 1
    raise AssertionError("agent-supervisor discovery started a process")
subprocess.Popen = forbidden_popen
tools = manager.list_tools("agent_supervisor")
subprocess.Popen = original_popen
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    agent_supervisor_service_resolution_count,
)
after_tools = sorted(
    name for name in sys.modules
    if name.startswith("ipfs_accelerate_py.agent_supervisor")
)
providers = [
    name for name in after_tools
    if name.endswith(
        ("formal_verification_provider", "leanstral_proof_provider")
    )
]
print(json.dumps({
    "category": "agent_supervisor" in categories,
    "before_tools": before_tools,
    "provider_modules": providers,
    "process_starts": process_starts,
    "service_resolutions": agent_supervisor_service_resolution_count(),
    "tool_count": len(tools),
}))
""".strip(),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-c", script.read_text(encoding="utf-8")],
        check=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        timeout=30,
    )
    observation = json.loads(completed.stdout.splitlines()[-1])

    assert observation == {
        "category": True,
        "before_tools": [],
        "provider_modules": [],
        "process_starts": 0,
        "service_resolutions": 0,
        "tool_count": len(Operation),
    }
