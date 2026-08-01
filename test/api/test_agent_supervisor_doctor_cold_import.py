"""Fresh-process cold import gates for Deterministic Doctor surfaces (PDR-015).

These tests prove that service/contracts/discovery/help remain provider-free:
no network clients, model SDKs, optional storage engines, or datasets providers
load; no network/process/database/storage initialization occurs; and optional
capability probes report unavailable instead of failing the package root import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_ROOTS = (
    "aiohttp",
    "anthropic",
    "duckdb",
    "httpx",
    "ipfs_datasets_embedding",
    "ipfs_datasets_py",
    "llm_router",
    "neo4j",
    "openai",
    "requests",
    "sentence_transformers",
    "torch",
    "transformers",
    "urllib3",
)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["IPFS_ACCEL_SKIP_CORE"] = "1"
    env["IPFS_ACCEL_IMPORT_EAGER"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # Fresh-process probes must not inherit pytest's in-process markers.
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_VERSION", None)
    return env


def _run_probe(script: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, "probe produced no stdout"
    return json.loads(lines[-1])


def test_doctor_package_discovery_and_help_are_cold() -> None:
    script = f"""
import json
import sys
import time

def _vm_hwm_kb():
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

forbidden = {FORBIDDEN_ROOTS!r}
before = set(sys.modules)
t0 = time.perf_counter()
import ipfs_accelerate_py  # parent package must stay network-client free
import ipfs_accelerate_py.agent_supervisor as package
discovery = package.agent_supervisor_cold_discovery()
help_payload = package.agent_supervisor_cold_help()
from ipfs_accelerate_py.agent_supervisor.control import control_contracts
from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service import (
    DeterministicDoctorService,
    optional_providers_loaded,
)
service_discovery = DeterministicDoctorService.discovery()
# Construct the service (must not open storage/network or load providers).
service = DeterministicDoctorService()
elapsed_ms = (time.perf_counter() - t0) * 1000.0
after = set(sys.modules) - before
added_forbidden = sorted(
    {{
        name.split(".", 1)[0]
        for name in after
        if name.split(".", 1)[0] in forbidden
    }}
)
package_modules = sorted(
    name for name in after if "agent_supervisor" in name
)
print(json.dumps({{
    "elapsed_ms": elapsed_ms,
    "module_count": len(after),
    "package_module_count": len(package_modules),
    "rss_kb": _vm_hwm_kb(),
    "rss_metric": discovery["budgets"].get("rss_metric"),
    "added_forbidden": added_forbidden,
    "optional_providers": list(optional_providers_loaded()),
    "hf_space_loaded": "ipfs_accelerate_py.hf_space_inference" in sys.modules,
    "requests_loaded": "requests" in sys.modules,
    "interface": discovery["interface"],
    "discovery_schema": discovery["schema"],
    "help_schema": help_payload["schema"],
    "service_interface": service_discovery["interface"],
    "service_ops": service_discovery["operations"],
    "service_llm": service_discovery["llm_router_enabled"],
    "service_network": service_discovery["network_access_allowed"],
    "service_processes": service_discovery["processes_started"],
    "service_database": service_discovery["database_opened"],
    "service_optional_flag": service_discovery["optional_providers_loaded"],
    "budgets": discovery["budgets"],
    "contracts_has_capability_report": hasattr(control_contracts, "CapabilityReport"),
    "backends_available": list(service.backends_available),
}}))
"""
    payload = _run_probe(script)
    budgets = payload["budgets"]
    assert payload["interface"] == "AgentSupervisorColdDiscovery@1"
    assert payload["discovery_schema"].endswith("cold-discovery@1")
    assert payload["help_schema"].endswith("cold-help@1")
    assert payload["service_interface"] == "DeterministicDoctorService@1"
    assert "inspect" in payload["service_ops"]
    assert "status" in payload["service_ops"]
    assert payload["service_llm"] is False
    assert payload["service_network"] is False
    assert payload["service_processes"] is False
    assert payload["service_database"] is False
    assert payload["service_optional_flag"] is False
    assert payload["added_forbidden"] == []
    assert payload["optional_providers"] == []
    assert payload["hf_space_loaded"] is False
    assert payload["requests_loaded"] is False
    assert payload["contracts_has_capability_report"] is True
    assert payload["backends_available"] == []
    assert payload["rss_metric"] == "proc_self_status_vm_hwm_kb"
    assert payload["elapsed_ms"] <= budgets["max_latency_ms"]
    assert payload["rss_kb"] <= budgets["max_rss_kb"]
    assert payload["module_count"] <= budgets["max_modules"]
    assert payload["package_module_count"] <= budgets["max_package_modules"]


def test_doctor_optional_capability_access_is_lazy_and_unavailable() -> None:
    script = f"""
import json
import sys

forbidden = {FORBIDDEN_ROOTS!r}
before = {{name for name in sys.modules if name.split(".", 1)[0] in forbidden}}
import ipfs_accelerate_py.agent_supervisor as package
report = package.agent_supervisor_optional_capability("torch")
requests_report = package.agent_supervisor_optional_capability("requests")
unknown = package.agent_supervisor_optional_capability("not_a_real_capability")
after = {{name for name in sys.modules if name.split(".", 1)[0] in forbidden}}
print(json.dumps({{
    "added_forbidden": sorted(after - before),
    "torch": report,
    "requests": requests_report,
    "unknown": unknown,
    "torch_in_modules": "torch" in sys.modules,
    "requests_in_modules": "requests" in sys.modules,
}}))
"""
    payload = _run_probe(script)
    assert payload["added_forbidden"] == []
    assert payload["torch_in_modules"] is False
    assert payload["requests_in_modules"] is False
    for key in ("torch", "requests"):
        report = payload[key]
        assert report["available"] is False
        assert report["import_attempted"] is False
        assert report["package_presence_is_capability"] is False
        assert report["status"] in {"unavailable", "loaded_not_certified"}
        assert "lazy_access_only" in report["reason_codes"] or (
            "optional_provider_not_loaded" in report["reason_codes"]
        )
    assert payload["unknown"]["available"] is False
    assert "unknown_optional_capability" in payload["unknown"]["reason_codes"]


def test_doctor_service_module_source_never_imports_forbidden_providers() -> None:
    import ast

    service_path = (
        REPO_ROOT
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "control"
        / "deterministic_doctor_service.py"
    )
    tree = ast.parse(service_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    lowered = "\n".join(imported).lower()
    for forbidden in (
        "llm_router",
        "openai",
        "anthropic",
        "torch",
        "transformers",
        "duckdb",
        "neo4j",
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "ipfs_datasets_py",
        "ipfs_datasets_embedding",
    ):
        assert forbidden not in lowered


def test_package_root_skip_core_does_not_load_hf_space_or_requests() -> None:
    script = f"""
import json
import sys

forbidden = {FORBIDDEN_ROOTS!r}
before = {{name for name in sys.modules if name.split(".", 1)[0] in forbidden}}
import ipfs_accelerate_py as root
after = {{name for name in sys.modules if name.split(".", 1)[0] in forbidden}}
print(json.dumps({{
    "added_forbidden": sorted(after - before),
    "hf_space_loaded": "ipfs_accelerate_py.hf_space_inference" in sys.modules,
    "hf_space_names_unresolved": all(
        name not in root.__dict__
        for name in (
            "HFSpaceClient",
            "BatchProcessor",
            "EndpointContract",
        )
    ),
    "export_has_worker_key": "worker" in root.export,
}}))
"""
    payload = _run_probe(script)
    assert payload == {
        "added_forbidden": [],
        "hf_space_loaded": False,
        "hf_space_names_unresolved": True,
        "export_has_worker_key": True,
    }
