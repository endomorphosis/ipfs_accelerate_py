from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


PACKAGE = "ipfs_accelerate_py.agent_supervisor"
V2_OWNER_MODULES = frozenset(
    {
        PACKAGE,
        f"{PACKAGE}.control_contracts",
        f"{PACKAGE}.control_plane",
        f"{PACKAGE}.supervisor_v2_contracts",
        f"{PACKAGE}.self_improvement_v2",
        f"{PACKAGE}.self_improvement_v2_rollout",
    }
)
V2_COLD_LAZY_MODULES = frozenset(
    {
        f"{PACKAGE}.self_improvement_v2",
        f"{PACKAGE}.supervisor_v2_benchmark",
        f"{PACKAGE}.self_improvement_v2_rollout",
    }
)


def _qualified_owner(owner: str) -> str:
    return (
        owner
        if owner == PACKAGE or owner.startswith(f"{PACKAGE}.")
        else f"{PACKAGE}.{owner}"
    )


def test_v2_stable_manifest_is_exact_immutable_and_canonical() -> None:
    api = importlib.import_module(PACKAGE)

    stable_exports = api.AGENT_SUPERVISOR_V2_STABLE_EXPORTS
    export_modules = api.AGENT_SUPERVISOR_V2_EXPORT_MODULES

    assert api.AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION == 2
    assert isinstance(stable_exports, tuple)
    assert stable_exports
    assert len(stable_exports) == len(set(stable_exports))
    assert api.V2_STABLE_EXPORTS is stable_exports
    assert isinstance(export_modules, MappingProxyType)
    assert tuple(export_modules) == stable_exports
    assert set(export_modules) == set(stable_exports)
    assert {_qualified_owner(owner) for owner in export_modules.values()} == (
        V2_OWNER_MODULES
    )
    assert all(name in api.__all__ for name in stable_exports)
    assert {
        "AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION",
        "AGENT_SUPERVISOR_V2_STABLE_EXPORTS",
        "AGENT_SUPERVISOR_V2_EXPORT_MODULES",
        "V2_STABLE_EXPORTS",
        "V2_LAZY_PUBLIC_API_REQUIREMENT_ID",
        "agent_supervisor_v2_discovery_manifest",
        "agent_supervisor_v2_control_surface_publication",
    }.issubset(api.__all__)

    try:
        export_modules["not_reviewed"] = "optional_provider"
    except TypeError:
        pass
    else:  # pragma: no cover - makes the immutability failure explicit
        raise AssertionError("the v2 export owner manifest is mutable")

    owner_modules = {
        module_name: importlib.import_module(module_name)
        for module_name in V2_OWNER_MODULES
    }
    for name in stable_exports:
        owner_name = _qualified_owner(export_modules[name])
        owner = owner_modules[owner_name]
        assert name in owner.__all__
        assert getattr(api, name) is getattr(owner, name)


def test_v1_rollout_surface_remains_compatible_after_v2_resolution() -> None:
    api = importlib.import_module(PACKAGE)
    v1_owner = importlib.import_module(f"{PACKAGE}.self_improvement_rollout")

    assert isinstance(api.PAIRED_ROLLOUT_STABLE_EXPORTS, tuple)
    assert len(api.PAIRED_ROLLOUT_STABLE_EXPORTS) == len(v1_owner.__all__)
    assert set(api.PAIRED_ROLLOUT_STABLE_EXPORTS) == set(v1_owner.__all__)
    assert all(
        getattr(api, name) is getattr(v1_owner, name)
        for name in api.PAIRED_ROLLOUT_STABLE_EXPORTS
    )
    assert not set(api.AGENT_SUPERVISOR_V2_STABLE_EXPORTS).intersection(
        api.PAIRED_ROLLOUT_STABLE_EXPORTS
    )


def test_python_cli_and_mcp_v2_aliases_preserve_catalog_identities() -> None:
    api = importlib.import_module(PACKAGE)
    control_cli = importlib.import_module(f"{PACKAGE}.control_cli")
    control_plane = importlib.import_module(f"{PACKAGE}.control_plane")
    contracts = importlib.import_module(f"{PACKAGE}.control_contracts")
    native_tools = importlib.import_module(
        "ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools."
        "native_agent_supervisor_tools"
    )

    assert api.agent_supervisor_v2_control_surface_publication is (
        control_plane.control_service_publication
    )
    assert control_cli.agent_cli_v2_discovery_manifest is (
        control_cli.agent_cli_discovery_manifest
    )
    assert control_cli.v2_cli_control_surface_publication is (
        control_cli.cli_control_surface_publication
    )
    assert native_tools.agent_supervisor_v2_discovery_manifest is (
        native_tools.agent_supervisor_discovery_manifest
    )
    assert native_tools.mcp_v2_control_surface_publication is (
        native_tools.mcp_control_surface_publication
    )

    manifests = (
        api.agent_supervisor_v2_discovery_manifest(),
        control_cli.agent_cli_v2_discovery_manifest(),
        native_tools.agent_supervisor_v2_discovery_manifest(),
    )
    publications = (
        api.agent_supervisor_v2_control_surface_publication(),
        control_cli.v2_cli_control_surface_publication(),
        native_tools.mcp_v2_control_surface_publication(),
    )

    assert {manifest.surface for manifest in manifests} == set(
        contracts.ControlSurface
    )
    assert {publication.surface for publication in publications} == set(
        contracts.ControlSurface
    )
    assert len({manifest.schema_population_id for manifest in manifests}) == 1
    assert len({publication.catalog_id for publication in publications}) == 1

    catalog = contracts.get_operation_catalog()
    canonical_operations = catalog.operations
    for manifest in manifests:
        assert len(manifest.operations) == len(canonical_operations)
        assert all(
            discovered is canonical
            for discovered, canonical in zip(
                manifest.operations, canonical_operations, strict=True
            )
        )
        assert manifest.request_schema_ids == manifests[0].request_schema_ids
        assert manifest.result_schema_ids == manifests[0].result_schema_ids
    for publication in publications:
        assert len(publication.operations) == len(canonical_operations)
        assert all(
            discovered is canonical
            for discovered, canonical in zip(
                publication.operations, canonical_operations, strict=True
            )
        )
        assert all(
            catalog.operation(operation).operation is operation
            for operation in publication.operations
        )

    assert set(control_cli.COMMAND_OPERATIONS.values()) == set(
        canonical_operations
    )
    assert set(native_tools.AGENT_SUPERVISOR_OPERATION_TOOLS) == set(
        canonical_operations
    )
    assert all(
        operation is contracts.Operation(operation.value)
        for operation in canonical_operations
    )


def test_fresh_v2_import_discovery_and_resolution_are_side_effect_free() -> None:
    program = f"""
import importlib
import json
import sys

PACKAGE = {PACKAGE!r}
OWNER_MODULES = {sorted(V2_OWNER_MODULES)!r}
COLD_LAZY_MODULES = {sorted(V2_COLD_LAZY_MODULES)!r}
OPTIONAL_PREFIXES = (
    "ipfs_datasets_py",
    f"{{PACKAGE}}.ipfs_datasets_",
    f"{{PACKAGE}}.leanstral_proof_provider",
    f"{{PACKAGE}}.leanstral_goal_development",
    f"{{PACKAGE}}.leanstral_goal_lifecycle",
    f"{{PACKAGE}}.formal_verification_provider",
    f"{{PACKAGE}}.todo_daemon.llm",
)
PROCESS_EVENTS = (
    "subprocess.Popen",
    "os.system",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.fork",
    "os.forkpty",
)
started = []

def audit(event, args):
    if event in PROCESS_EVENTS:
        started.append(event)
        raise RuntimeError(f"public API discovery started a process: {{event}}")

sys.addaudithook(audit)
initial_modules = set(sys.modules)
api = importlib.import_module(PACKAGE)
stable_exports = api.AGENT_SUPERVISOR_V2_STABLE_EXPORTS
export_modules = api.AGENT_SUPERVISOR_V2_EXPORT_MODULES

from ipfs_accelerate_py.agent_supervisor.control import control_cli
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    native_agent_supervisor_tools as native_tools,
)

resolution_before = native_tools.agent_supervisor_service_resolution_count()
manifests = (
    api.agent_supervisor_v2_discovery_manifest(),
    control_cli.agent_cli_v2_discovery_manifest(),
    native_tools.agent_supervisor_v2_discovery_manifest(),
)
publications = (
    api.agent_supervisor_v2_control_surface_publication(),
    control_cli.v2_cli_control_surface_publication(),
    native_tools.mcp_v2_control_surface_publication(),
)
resolution_after = native_tools.agent_supervisor_service_resolution_count()

after_discovery_modules = set(sys.modules)
loaded_optional_after_discovery = sorted(
    name
    for name in after_discovery_modules.difference(initial_modules)
    if name.startswith(OPTIONAL_PREFIXES)
)
lazy_modules_after_discovery = sorted(
    name for name in COLD_LAZY_MODULES if name in sys.modules
)

root_values = {{name: getattr(api, name) for name in stable_exports}}
canonical_exports = True
for name, value in root_values.items():
    owner_name = export_modules[name]
    if owner_name != PACKAGE and not owner_name.startswith(PACKAGE + "."):
        owner_name = f"{{PACKAGE}}.{{owner_name}}"
    canonical_exports = (
        canonical_exports
        and value is getattr(importlib.import_module(owner_name), name)
    )

loaded_optional_after_resolution = sorted(
    name
    for name in set(sys.modules).difference(initial_modules)
    if name.startswith(OPTIONAL_PREFIXES)
)
print(json.dumps({{
    "version": api.AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION,
    "alias_identity": api.V2_STABLE_EXPORTS is stable_exports,
    "manifest_unique": len(stable_exports) == len(set(stable_exports)),
    "manifest_mapping_exact": tuple(export_modules) == stable_exports,
    "lazy_modules_after_discovery": lazy_modules_after_discovery,
    "loaded_optional_after_discovery": loaded_optional_after_discovery,
    "loaded_optional_after_resolution": loaded_optional_after_resolution,
    "process_events": started,
    "service_resolutions": [resolution_before, resolution_after],
    "schema_population_count": len({{
        manifest.schema_population_id for manifest in manifests
    }}),
    "catalog_count": len({{
        publication.catalog_id for publication in publications
    }}),
    "operation_identity": all(
        all(
            item is manifests[0].operations[index]
            for index, item in enumerate(manifest.operations)
        )
        for manifest in manifests
    ),
    "canonical_exports": canonical_exports,
    "resolved_owner_modules": sorted(
        name
        for name in set(OWNER_MODULES).union(COLD_LAZY_MODULES)
        if name in sys.modules
    ),
}}, sort_keys=True))
"""
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("IPFS_ACCEL_SKIP_CORE", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (
            str(repository_root),
            environment.get("PYTHONPATH", ""),
        )
        if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    observation: Mapping[str, object] = json.loads(completed.stdout)
    assert observation == {
        "version": 2,
        "alias_identity": True,
        "manifest_unique": True,
        "manifest_mapping_exact": True,
        "lazy_modules_after_discovery": [],
        "loaded_optional_after_discovery": [],
        "loaded_optional_after_resolution": [],
        "process_events": [],
        "service_resolutions": [0, 0],
        "schema_population_count": 1,
        "catalog_count": 1,
        "operation_identity": True,
        "canonical_exports": True,
        "resolved_owner_modules": sorted(
            V2_OWNER_MODULES.union(V2_COLD_LAZY_MODULES)
        ),
    }
