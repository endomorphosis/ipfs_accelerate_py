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
    assert observed["usage"]["canonical_edits"] == 1
    assert observed["provider_calls"] == 0 and observed["target_code_executed"] is False
    encoded = json.dumps(observed)
    assert "def value" not in encoded and "authored cache" not in encoded


@pytest.mark.parametrize("mutation", ["missing_check", "false_check", "provider_call", "extra",
    "missing_usage", "false_initialization", "no_reads", "bool_count", "retained_state", "uncertain_edit",
    "cleanup_error", "extra_usage"])
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
    else: value["usage"]["uncertain_edits"] = 1
    with pytest.raises(ValueError):
        probe.validate_probe(value)


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
                   "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    # The isolated parent inserts deployed roots explicitly just like WORKER_ENTRY.
    # Its real child has to obtain those roots from the exported environment.
    script = "import sys; sys.path[:0]=" + repr(roots) + "; from " + probe.__name__ + " import main; raise SystemExit(main())"
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=75, check=False)
    assert result.returncode == 0, result.stderr
    assert probe.validate_probe(json.loads(result.stdout))["qualified"]
