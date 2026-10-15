"""Installed-runtime qualification must use actual stdio and remain content-free."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_probe as probe


@pytest.fixture(scope="module")
def observed():
    return probe.run_probe()


def test_actual_installed_runtime_probe_and_closed_receipt(observed):
    assert probe.validate_probe(observed) == observed
    assert observed["qualified"] and all(observed["checks"].values())
    assert observed["schema"] == "repository-source-runtime-probe@2"
    assert observed["usage"]["canonical_edits"] == 2
    assert observed["usage"]["retrieval_observation"]["generation_bindings_observed"] == 2
    assert observed["provider_calls"] == 0 and observed["target_code_executed"] is False
    encoded = json.dumps(observed)
    assert "def value" not in encoded and "authored cache" not in encoded


@pytest.mark.parametrize("mutation", ["missing_check", "false_check", "provider_call", "extra",
    "missing_usage", "false_initialization", "no_reads", "bool_count", "retained_state", "uncertain_edit",
    "cleanup_error", "extra_usage", "missing_retrieval", "false_retrieval", "retrieval_authority",
    "wrong_retrieval_population", "missing_symbol_check", "false_symbol_check"])
def test_probe_rejects_partial_or_contradictory_receipts(observed, mutation):
    value = copy.deepcopy(observed)
    if mutation == "missing_check": value["checks"].pop("native_scope_joined")
    elif mutation == "false_check": value["checks"]["source_current"] = False
    elif mutation == "provider_call": value["provider_calls"] = 1
    elif mutation == "extra": value["extra"] = True
    elif mutation == "missing_usage": value.pop("usage")
    elif mutation == "false_initialization": value["usage"]["initializations"] = 0
    elif mutation == "no_reads": value["usage"]["successful_reads"] = 0
    elif mutation == "bool_count": value["usage"]["canonical_edits"] = True
    elif mutation == "retained_state": value["usage"]["store_cleanup_complete"] = False
    elif mutation == "cleanup_error": value["usage"]["store_cleanup_error"] = "cleanup_unavailable"
    elif mutation == "extra_usage": value["usage"]["source"] = "retained source text"
    elif mutation == "missing_retrieval": value["usage"].pop("retrieval_observation")
    elif mutation == "false_retrieval": value["usage"]["retrieval_observation"]["complete"] = False
    elif mutation == "retrieval_authority": value["usage"]["retrieval_observation"]["proof_authority"] = True
    elif mutation == "wrong_retrieval_population": value["usage"]["retrieval_observation"]["capsules_returned"] = 2
    elif mutation == "missing_symbol_check": value["checks"].pop("indexed_symbol_read")
    elif mutation == "false_symbol_check": value["checks"]["indexed_symbol_edit"] = False
    else: value["usage"]["uncertain_edits"] = 1
    with pytest.raises(ValueError):
        probe.validate_probe(value)


@pytest.mark.parametrize("retrieval", [False, True])
def test_historical_probe_is_reviewable_but_requires_current_deployment_qualification(retrieval):
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_deployment import authored_probe_receipt
    value = authored_probe_receipt()
    value["schema"] = probe.LEGACY_SCHEMA
    for key in ("indexed_symbol_read", "indexed_symbol_edit", "indexed_symbol_stale_refused",
                "indexed_symbol_refresh", "retrieval_usage_bound"):
        value["checks"].pop(key)
    value["usage"].update(tool_calls=9, successful_reads=2, compact_reads=1, canonical_edits=1, refreshes=1, errors=1)
    if not retrieval:
        value["usage"].pop("retrieval_observation")
    with pytest.raises(ValueError, match="current requalification"):
        probe.validate_probe(value)
    assert probe.validate_probe(value, allow_legacy=True) == value
    value["checks"]["context_query_current"] = False
    with pytest.raises(ValueError):
        probe.validate_probe(value, allow_legacy=True)


def test_current_probe_requires_retrieval_even_during_historical_review():
    from benchmarks.agent_supervisor.container_coding.test_terminal_repository_source_deployment import authored_probe_receipt
    value = authored_probe_receipt()
    assert probe.validate_probe(value) == value
    value["usage"].pop("retrieval_observation")
    with pytest.raises(ValueError):
        probe.validate_probe(value, allow_legacy=True)


def test_clean_environment_forwards_deployed_dependency_paths(tmp_path):
    import duckdb
    import ipfs_datasets_py
    source = Path(probe.__file__).resolve().parents[3]
    datasets = Path(ipfs_datasets_py.__file__).resolve().parent.parent
    # This host installs dependencies in the user site, whereas deployment uses
    # a dedicated venv. Bind the installed dependency directory explicitly so
    # -I and the disposable HOME model that venv without ambient user imports.
    dependencies = Path(duckdb.__file__).resolve().parent.parent
    roots = [str(source), str(datasets), str(dependencies)]
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                   "HOME": str(tmp_path), "PYTHONPATH": os.pathsep.join(roots),
                   "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   # Isolate this authored test's accounting from other host
                   # clients with different capacity policies. The real host
                   # pressure sampler and default safety policy stay enabled.
                   "IPFS_DATASETS_RESOURCE_SCHEDULER_PATH": str(tmp_path / "resources.json"),
                   "IPFS_DATASETS_RESOURCE_CPU_SLOTS": "1"}
    # The isolated parent inserts deployed roots explicitly just like WORKER_ENTRY.
    # Its real child has to obtain those roots from the exported environment.
    script = "import sys; sys.path[:0]=" + repr(roots) + "; from " + probe.__name__ + " import main; raise SystemExit(main())"
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=75, check=False)
    assert result.returncode == 0, result.stderr
    assert probe.validate_probe(json.loads(result.stdout))["qualified"]
