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
    reads = observed["usage"]["read_observation"]
    assert reads["complete"] is True
    assert (reads["read_calls"], reads["successful_reads"], reads["failed_reads"]) == (7, 5, 2)
    assert (reads["full_file_reads"], reads["definition_reads"], reads["line_range_reads"]) == (3, 2, 0)
    assert (reads["original_reads"], reads["compact_reads"]) == (3, 2)
    assert reads["line_range_selected_lines"] == reads["line_range_total_lines"] == reads["line_range_omitted_lines"] == 0
    assert reads["provider_usage_measured"] is reads["token_savings_measured"] is False
    assert observed["provider_calls"] == 0 and observed["target_code_executed"] is False
    encoded = json.dumps(observed)
    assert "def value" not in encoded and "authored cache" not in encoded


def test_probe_read_metadata_matches_the_fixed_workload_without_false_coverage(observed):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_mcp import validate_read_observation

    mutations = {"missing": "invalid_read_observation", "null": "invalid_read_observation",
        "bool_count": "invalid_read_observation", "authority": "invalid_read_observation",
        "extra": "invalid_read_observation", "wrong_sum": "invalid_read_observation",
        "digest": "invalid_read_observation", "exposure": "invalid_read_observation",
        "scopes": "repository runtime probe read usage differs",
        "range": "repository runtime probe read usage differs",
        "legacy": "repository runtime probe read usage differs",
        "unbound": "repository runtime probe read usage differs"}
    for mutation, reason in mutations.items():
        value = copy.deepcopy(observed)
        usage = value["usage"]
        reads = usage["read_observation"]
        if mutation == "missing":
            reads.pop("event_chain_sha256")
        elif mutation == "null":
            usage["read_observation"] = None
        elif mutation == "bool_count":
            reads["successful_reads"] = True
        elif mutation == "authority":
            reads["token_savings_measured"] = True
        elif mutation == "extra":
            reads["source_text"] = "private source must not qualify"
        elif mutation == "wrong_sum":
            reads["read_calls"] += 1
        elif mutation == "digest":
            reads["event_chain_sha256"] = "foreign"
        elif mutation == "exposure":
            reads["exposed_content_utf8_bytes"] = 0
        elif mutation == "scopes":
            reads.update(full_file_reads=2, definition_reads=3)
        elif mutation == "range":
            reads.update(full_file_reads=2, line_range_reads=1,
                line_range_selected_lines=1, line_range_total_lines=1, line_range_omitted_lines=0)
        elif mutation == "legacy":
            reads.update(read_calls=6, successful_reads=4, full_file_reads=3, definition_reads=1,
                original_reads=2, generation_bindings_observed=4, legacy_read_calls=1,
                legacy_successful_reads=1, complete=False)
        else:
            reads.update(full_file_reads=2, original_reads=2, generation_bindings_observed=4,
                unbound_successes=1, complete=False)
        if mutation in {"scopes", "range", "legacy", "unbound"}:
            # These are otherwise coherent read populations. Only the fixed
            # installed-probe workload can rule out invented scope coverage.
            validate_read_observation(reads, **{key: usage[key] for key in
                ("tool_calls", "errors", "successful_reads", "compact_reads")})
        with pytest.raises(ValueError) as rejected:
            probe.validate_probe(value)
        assert str(rejected.value) == reason, mutation


def test_historical_current_probe_without_read_metadata_remains_reviewable(observed):
    value = copy.deepcopy(observed)
    value["usage"].pop("read_observation")
    assert probe.validate_probe(value) == value
    assert "read_observation" not in value["usage"]


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
    import multiformats
    source = Path(probe.__file__).resolve().parents[3]
    datasets = Path(ipfs_datasets_py.__file__).resolve().parent.parent
    # The native owner requires both DuckDB and multiformats for source CIDs.
    # They may occupy separate installed dependency directories. Bind those
    # actual package origins explicitly, without inheriting ambient source roots.
    dependencies = Path(duckdb.__file__).resolve().parent.parent
    cid_dependencies = Path(multiformats.__file__).resolve().parent.parent
    roots = list(dict.fromkeys(map(str, (source, datasets, dependencies, cid_dependencies))))
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
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert probe.validate_probe(json.loads(result.stdout))["qualified"]
