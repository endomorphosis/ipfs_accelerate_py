"""Fresh-process cold import gates for Planner surfaces (PDR-015).

Planner package discovery/help and adaptive/proof-carrying planner modules must
import without network clients, model SDKs, DuckDB/Neo4j, or optional datasets
providers. Optional capability access stays lazy and reports unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_planner_package_discovery_and_help_are_cold() -> None:
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
import ipfs_accelerate_py
import ipfs_accelerate_py.agent_supervisor as package
discovery = package.agent_supervisor_cold_discovery()
help_payload = package.agent_supervisor_cold_help()
from ipfs_accelerate_py.agent_supervisor.control import control_contracts
# Planner modules load only when requested; still must stay provider-free.
import ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner as adaptive
import ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_planner as pcp
elapsed_ms = (time.perf_counter() - t0) * 1000.0
after = set(sys.modules) - before
added_forbidden = sorted(
    {{
        name.split(".", 1)[0]
        for name in after
        if name.split(".", 1)[0] in forbidden
    }}
)
package_modules = [name for name in after if "agent_supervisor" in name]
surface_ids = [item["id"] for item in discovery["surfaces"]]
print(json.dumps({{
    "elapsed_ms": elapsed_ms,
    "module_count": len(after),
    "package_module_count": len(package_modules),
    "rss_kb": _vm_hwm_kb(),
    "rss_metric": discovery["budgets"].get("rss_metric"),
    "added_forbidden": added_forbidden,
    "interface": discovery["interface"],
    "surface_ids": surface_ids,
    "help_summary": help_payload["summary"][:40],
    "has_adaptive_planner": hasattr(adaptive, "AdaptivePlanner"),
    "has_pcp": hasattr(pcp, "ProofCarryingPlanner"),
    "contracts_ok": hasattr(control_contracts, "Operation"),
    "hf_space_loaded": "ipfs_accelerate_py.hf_space_inference" in sys.modules,
    "budgets": discovery["budgets"],
    "llm_router_enabled": discovery["llm_router_enabled"],
    "network_access": discovery["network_access"],
    "storage_initialized": discovery["storage_initialized"],
    "processes_started": discovery["processes_started"],
    "database_opened": discovery["database_opened"],
}}))
"""
    payload = _run_probe(script)
    budgets = payload["budgets"]
    assert payload["interface"] == "AgentSupervisorColdDiscovery@1"
    assert "adaptive_planner" in payload["surface_ids"]
    assert "proof_carrying_planner" in payload["surface_ids"]
    assert "deterministic_doctor_service" in payload["surface_ids"]
    assert payload["has_adaptive_planner"] is True
    assert payload["has_pcp"] is True
    assert payload["contracts_ok"] is True
    assert payload["added_forbidden"] == []
    assert payload["hf_space_loaded"] is False
    assert payload["llm_router_enabled"] is False
    assert payload["network_access"] is False
    assert payload["storage_initialized"] is False
    assert payload["processes_started"] is False
    assert payload["database_opened"] is False
    assert payload["rss_metric"] == "proc_self_status_vm_hwm_kb"
    # Package discovery budgets cover the cold package path; planner module
    # import is allowed to grow the module count but must stay under a fixed
    # absolute ceiling and the RSS/latency budgets.
    assert payload["elapsed_ms"] <= budgets["max_latency_ms"]
    assert payload["rss_kb"] <= budgets["max_rss_kb"]
    assert payload["module_count"] <= max(budgets["max_modules"] * 3, 400)


def test_planner_root_lazy_export_does_not_load_optional_providers() -> None:
    script = f"""
import json
import sys

forbidden = {FORBIDDEN_ROOTS!r}
before = {{name for name in sys.modules if name.split(".", 1)[0] in forbidden}}
import ipfs_accelerate_py.agent_supervisor as package
# Touch the reviewed planner export without importing optional providers.
planner_cls = package.AdaptivePlanner
cap = package.agent_supervisor_optional_capability("duckdb")
llm = package.agent_supervisor_optional_capability("llm_router")
after = {{name for name in sys.modules if name.split(".", 1)[0] in forbidden}}
print(json.dumps({{
    "added_forbidden": sorted(after - before),
    "planner_module": planner_cls.__module__,
    "duckdb": cap,
    "llm_router": llm,
    "duckdb_loaded": "duckdb" in sys.modules,
    "llm_router_loaded": "llm_router" in sys.modules
        or any(name.startswith("llm_router.") for name in sys.modules),
}}))
"""
    payload = _run_probe(script)
    assert payload["added_forbidden"] == []
    assert payload["planner_module"].endswith("adaptive_planner")
    assert payload["duckdb_loaded"] is False
    assert payload["llm_router_loaded"] is False
    assert payload["duckdb"]["available"] is False
    assert payload["duckdb"]["import_attempted"] is False
    assert payload["llm_router"]["available"] is False
    assert payload["llm_router"]["import_attempted"] is False


def test_planner_cold_discovery_records_strict_budgets() -> None:
    script = """
import json
import ipfs_accelerate_py.agent_supervisor as package
discovery = package.agent_supervisor_cold_discovery()
print(json.dumps(discovery["budgets"]))
"""
    budgets = _run_probe(script)
    assert budgets["max_latency_ms"] == 2500
    assert budgets["max_rss_kb"] == 80_000
    assert budgets["max_modules"] == 120
    assert budgets["max_package_modules"] == 25
    assert budgets["rss_metric"] == "proc_self_status_vm_hwm_kb"
    assert all(
        isinstance(value, int) and value > 0
        for key, value in budgets.items()
        if key != "rss_metric"
    )
