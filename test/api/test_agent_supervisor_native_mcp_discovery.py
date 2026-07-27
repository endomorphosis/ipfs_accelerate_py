"""ASI-153: native MCP discovery is provider-free, process-free, and covers prompt workflow/rescue ops."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    Operation,
    PROMPT_CONTROL_OPERATIONS,
    get_operation_catalog,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_MCP_CATEGORY,
    AGENT_SUPERVISOR_MCP_DISPATCH_MODE,
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    agent_supervisor_discovery_manifest,
    agent_supervisor_service_resolution_count,
    mcp_control_surface_publication,
    register_native_agent_supervisor_tools,
    validate_agent_supervisor_mcp_catalog,
)


PROMPT_WORKFLOW_RESCUE_OPS = (
    Operation.WORKFLOW_PREVIEW,
    Operation.WORKFLOW_MATERIALIZE,
    Operation.RESTART,
    Operation.RESCUE_PREVIEW,
    Operation.RESCUE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _RecordingManager:
    def __init__(self) -> None:
        self.definitions: list[dict[str, Any]] = []

    def register_tool(self, **definition: Any) -> None:
        self.definitions.append(definition)


def test_native_mcp_discovery_covers_prompt_workflow_and_rescue_ops() -> None:
    """Discovery manifest must include all prompt workflow and rescue operations."""

    assert PROMPT_CONTROL_OPERATIONS == frozenset(PROMPT_WORKFLOW_RESCUE_OPS)
    manifest = agent_supervisor_discovery_manifest()
    catalog = get_operation_catalog()

    for op in PROMPT_WORKFLOW_RESCUE_OPS:
        assert op.value in manifest.operations
        assert op in catalog.operations

    for op in catalog.operations:
        assert op in AGENT_SUPERVISOR_OPERATION_TOOLS
        tool = AGENT_SUPERVISOR_OPERATION_TOOLS[op]
        assert callable(tool)
        assert tool.__name__ == f"agent_supervisor_{op.value}"
        assert getattr(tool, "__agent_supervisor_operation__", None) is op
        assert (
            getattr(tool, "__agent_supervisor_dispatch_mode__", None)
            == AGENT_SUPERVISOR_MCP_DISPATCH_MODE
        )
        # Module-level and package exports keep the lazy agent_supervisor_ prefix.
        assert hasattr(
            sys.modules[
                "ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools"
            ],
            tool.__name__,
        )

    publication = mcp_control_surface_publication()
    assert publication.provider_free is True
    assert publication.process_free is True
    assert publication.dispatch_mode == AGENT_SUPERVISOR_MCP_DISPATCH_MODE
    for op in PROMPT_WORKFLOW_RESCUE_OPS:
        assert (
            publication.dispatcher_ids[op]
            == DIRECT_CONTROL_SERVICE_DISPATCHER_ID
        )

    validated = validate_agent_supervisor_mcp_catalog()
    assert validated.operations == manifest.operations
    assert set(manifest.operations) == {item.value for item in Operation}


def test_native_mcp_discovery_is_provider_and_process_free(tmp_path: Path) -> None:
    """Import and discovery must not start providers, DuckDB, models, processes, or supervisor."""

    probe = tmp_path / "native_mcp_discovery_probe.py"
    probe.write_text(
        """
import json
import sys

provider_prefixes = (
    "ipfs_datasets_py",
    "ipfs_accelerate_py.agent_supervisor.ipfs_datasets_",
    "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
    "ipfs_accelerate_py.agent_supervisor.formal_verification_provider",
)
started = []

def audit(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
        started.append(event)
        raise RuntimeError("import or discovery started a process")

sys.addaudithook(audit)
before = set(sys.modules)

from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    agent_supervisor_discovery_manifest,
    agent_supervisor_service_resolution_count,
    mcp_control_surface_publication,
    register_native_agent_supervisor_tools,
)

class Manager:
    def __init__(self):
        self.definitions = []
    def register_tool(self, **definition):
        self.definitions.append(definition)

manager = Manager()
resolutions_before = agent_supervisor_service_resolution_count()
first = agent_supervisor_discovery_manifest()
publication = mcp_control_surface_publication()
register_native_agent_supervisor_tools(manager)
second = agent_supervisor_discovery_manifest()
resolutions_after = agent_supervisor_service_resolution_count()

loaded = sorted(
    name
    for name in set(sys.modules).difference(before)
    if name.startswith(provider_prefixes)
)

prompt_ops = [
    "workflow_preview",
    "workflow_materialize",
    "restart",
    "rescue_preview",
    "rescue",
]
present = [op for op in prompt_ops if op in first.operations]
export_names = [
    f"agent_supervisor_{name}"
    for name in prompt_ops
    if hasattr(
        sys.modules[
            "ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools"
        ],
        f"agent_supervisor_{name}",
    )
]

print(json.dumps({
    "loaded": loaded,
    "processes": started,
    "resolutions_before": resolutions_before,
    "resolutions_after": resolutions_after,
    "operation_count": len(first.operations),
    "tool_count": len(manager.definitions),
    "repeatable": first.to_record() == second.to_record(),
    "provider_free": publication.provider_free,
    "process_free": publication.process_free,
    "prompt_ops_present": present,
    "prompt_ops_expected": prompt_ops,
    "export_names": export_names,
}))
""".strip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("IPFS_ACCEL_SKIP_CORE", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
        if value
    )
    completed = subprocess.run(
        [sys.executable, str(probe)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"probe failed: {completed.stderr}\n{completed.stdout}"
    )
    observation = json.loads(completed.stdout)
    assert observation["loaded"] == [], (
        f"unexpected provider loads: {observation['loaded']}"
    )
    assert observation["processes"] == [], (
        f"unexpected process starts: {observation['processes']}"
    )
    assert observation["resolutions_before"] == 0
    assert observation["resolutions_after"] == 0
    assert observation["provider_free"] is True
    assert observation["process_free"] is True
    assert observation["repeatable"] is True
    assert observation["prompt_ops_present"] == observation["prompt_ops_expected"]
    assert observation["export_names"] == [
        f"agent_supervisor_{name}" for name in observation["prompt_ops_expected"]
    ]
    assert observation["operation_count"] == len(AGENT_SUPERVISOR_OPERATION_TOOLS)
    assert observation["tool_count"] == len(AGENT_SUPERVISOR_OPERATION_TOOLS)


def test_prompt_mcp_tools_have_exact_catalog_schemas_and_no_path_widening() -> None:
    """Prompt tools expose exact catalog schemas; descriptions cannot widen paths."""

    catalog = get_operation_catalog()
    publication = mcp_control_surface_publication()
    manager = _RecordingManager()
    before = agent_supervisor_service_resolution_count()
    register_native_agent_supervisor_tools(manager)
    after = agent_supervisor_service_resolution_count()
    assert after == before

    for op in PROMPT_WORKFLOW_RESCUE_OPS:
        descriptor = catalog.operation(op)
        definition = next(
            (item for item in manager.definitions if item["name"] == op.value),
            None,
        )
        assert definition is not None, f"missing tool definition for {op.value}"
        assert definition["category"] == AGENT_SUPERVISOR_MCP_CATEGORY
        assert definition["runtime"] == "fastapi"
        assert definition["func"].__name__ == f"agent_supervisor_{op.value}"

        input_schema = definition["input_schema"]
        assert "request" in input_schema["properties"]
        assert "x-output-schema" in input_schema
        assert "x-agent-supervisor-contract" in input_schema
        assert input_schema.get("additionalProperties") is False

        contract = input_schema["x-agent-supervisor-contract"]
        assert contract["operation"] == op.value
        assert contract["surface"] == "mcp"
        assert contract["request_schema_id"] == descriptor.request_schema_id
        assert contract["result_schema_id"] == descriptor.result_schema_id
        assert publication.request_schema_ids[op] == descriptor.request_schema_id
        assert publication.result_schema_ids[op] == descriptor.result_schema_id

        request_schema = input_schema["properties"]["request"]
        assert request_schema["properties"]["operation"]["const"] == op.value
        result_schema = input_schema["x-output-schema"]
        assert result_schema["properties"]["operation"]["const"] == op.value

        description = definition["description"].lower()
        assert "arbitrary" not in description
        assert "bypass" not in description
        assert any(
            token in description
            for token in ("shared", "canonical", "control")
        )
        assert "allowlist" in description
        assert "never widen" in description

        tags = set(definition["tags"])
        assert {
            "native",
            "agent-supervisor",
            "policy-controlled",
            "bounded",
            "redacted",
            "prompt-control",
            op.authority.value,
        }.issubset(tags)
        if op.mutating:
            assert {
                "authorization-required",
                "audit-receipt",
                "dry-run",
                "idempotent",
                "lease-fenced",
            }.issubset(tags)


def test_mcp_tool_descriptions_cannot_widen_policy_or_completion_authority() -> None:
    """Tool descriptions and schemas must not become authorization or completion authority."""

    manager = _RecordingManager()
    register_native_agent_supervisor_tools(manager)

    # Phrases that would claim or grant expanded rights. Descriptions may
    # mention that they *cannot* mark work complete, but must not advertise a
    # "completion authority" or path-widening capability.
    forbidden_phrases = (
        "allow any path",
        "any directory",
        "bypass authorization",
        "skip authorization",
        "completion authority",
        "widen policy",
        "override policy",
    )

    for definition in manager.definitions:
        desc = definition["description"].lower()
        for phrase in forbidden_phrases:
            assert phrase not in desc, (
                f"tool {definition['name']} description contains forbidden "
                f"phrase: {phrase}"
            )

        input_schema = definition["input_schema"]
        assert input_schema.get("additionalProperties") is False
        request_schema = input_schema["properties"]["request"]
        assert "operation" in request_schema["properties"]
        # Caller path fields stay request parameters; they never appear as
        # top-level schema defaults that would widen server allowlists.
        assert "repository_allowlist" not in request_schema.get(
            "properties", {}
        )
        assert "state_allowlist" not in request_schema.get("properties", {})


def test_discovery_manifest_is_repeatable_and_side_effect_free() -> None:
    """Discovery must be repeatable and not change runtime state."""

    before_count = agent_supervisor_service_resolution_count()
    first = agent_supervisor_discovery_manifest()
    second = agent_supervisor_discovery_manifest()
    after_count = agent_supervisor_service_resolution_count()

    assert first == second
    assert first.to_record() == second.to_record()
    assert after_count == before_count

    pub_first = mcp_control_surface_publication()
    pub_second = mcp_control_surface_publication()
    assert pub_first.catalog_id == pub_second.catalog_id
    assert pub_first.operations == pub_second.operations
    assert pub_first.request_schema_ids == pub_second.request_schema_ids
    assert pub_first.result_schema_ids == pub_second.result_schema_ids
    assert pub_first.provider_free is True
    assert pub_first.process_free is True
