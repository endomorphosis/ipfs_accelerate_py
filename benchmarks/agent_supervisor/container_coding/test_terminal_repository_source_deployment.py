"""Mocked deployment routing and probe receipt validation, without containers.

The archive contains the selected real source files, but commands below are
recorded only. Real MCP subprocess qualification lives in test_repository_source_probe.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_repository_source_policy as policy
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_uv_layout import RecordingEnvironment
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_probe as probe

PROBE_COMMAND = (deployment.PYTHON
    + " -P -m ipfs_accelerate_py.agent_supervisor.runtime.repository_source_probe")


def authored_probe_receipt():
    """Synthetic result for testing deployment validation, never runtime evidence."""
    checks = {name: True for name in (
        "initialized", "complete_tool_inventory", "source_current", "native_scope_joined",
        "cache_coverage_explicit", "compact_read", "original_hash_bound", "canonical_edit",
        "stale_generation_refused", "refreshed_original_exact", "native_symbol_observed",
        "no_proof_authority", "private_state_removed", "actual_tool_usage_observed")}
    usage = dict(status="observed", initializations=1, tool_calls=7, successful_reads=2,
        compact_reads=1, canonical_edits=1, refreshes=1, errors=1, uncertain_edits=0,
        returned_bytes=4096, store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    return dict(schema=probe.SCHEMA, qualified=True, checks=checks, usage=usage,
        provider_calls=0, target_code_executed=False, source_messages_retained=False,
        live_model_adoption_qualified=False)


class ProbeEnvironment(RecordingEnvironment):
    def __init__(self, *, stdout=None, return_code=0, failure=None):
        super().__init__()
        self.probe_stdout = json.dumps(authored_probe_receipt()) if stdout is None else stdout
        self.probe_return_code = return_code
        self.probe_failure = failure

    async def exec(self, command, **kwargs):
        if command != PROBE_COMMAND:
            return await super().exec(command, **kwargs)
        self.commands.append((command, kwargs))
        if self.probe_failure is not None:
            raise self.probe_failure
        return SimpleNamespace(return_code=self.probe_return_code, stdout=self.probe_stdout, stderr="")


@pytest.fixture(params=policy.REPOSITORY_SOURCE_POLICIES[1:])
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def supported_archive(tmp_path_factory):
    directory = tmp_path_factory.mktemp("repository-deployment-archive")
    inputs = _inputs(directory)
    checkout = Path(policy.__file__).resolve().parents[3]
    datasets = Path(importlib.util.find_spec("ipfs_datasets_py").origin).resolve().parent.parent
    for owner, source, paths in (("source", checkout, policy.SOURCE_PATHS),
                                  ("datasets", datasets, policy.DATASETS_SOURCE_PATHS)):
        for relative in paths:
            target = inputs[owner] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / relative).read_bytes())
    archive = directory / "bundle"
    manifest = deployment.build_runtime_archive(output=archive, **inputs)
    policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY)
    return archive


def test_selected_policy_rejects_missing_archive_support_before_container_mutation(tmp_path, mode):
    archive = tmp_path / "bundle"
    deployment.build_runtime_archive(output=archive, **_inputs(tmp_path))
    environment = ProbeEnvironment()
    output = tmp_path / "deployment"
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        asyncio.run(deployment.deploy_supervisor(environment, archive_dir=archive, output=output,
            install_codex=False, repository_source_policy=mode))
    assert environment.commands == [] and environment.uploads == []
    assert environment.default_user == "root" and not output.exists()


@pytest.mark.parametrize("selected", [None, True, {}, "repository-source-foreign@1"])
def test_invalid_selection_refuses_before_archive_or_container_io(tmp_path, selected):
    environment = ProbeEnvironment()
    with pytest.raises(ValueError, match="supported repository source policy"):
        asyncio.run(deployment.deploy_supervisor(environment, archive_dir=tmp_path / "absent",
            output=tmp_path / "deployment", install_codex=False, repository_source_policy=selected))
    assert environment.commands == [] and environment.uploads == []
    assert not (tmp_path / "deployment").exists()


def test_selected_runtime_probe_is_bounded_validated_and_retained(tmp_path, supported_archive, mode):
    environment = ProbeEnvironment()
    output = tmp_path / "deployment"
    report = asyncio.run(deployment.deploy_supervisor(environment, archive_dir=supported_archive,
        output=output, install_codex=False, repository_source_policy=mode))
    calls = [row for row in environment.commands if row[0] == PROBE_COMMAND]
    assert len(calls) == 1
    assert calls[0][1] == {"cwd": "/app", "user": "supervisor", "env": deployment.runtime_environment(), "timeout_sec": 75}
    labels = [row["step"] for row in report["steps"]]
    assert labels.index("python-install") < labels.index("repository-source-runtime-probe")
    assert labels.index("native-imports") < labels.index("repository-source-runtime-probe") < labels.index("empty-native-start")
    assert report["repository_source_policy"] == mode
    assert report["repository_source_runtime_probe"] == authored_probe_receipt()
    assert json.loads((output / "deployment.json").read_text()) == report
    assert (output / "repository-source-runtime-probe.log").read_text() == environment.probe_stdout
    assert report["qualified"] is True and report["task_source_preserved"] is True
    assert report["provider_calls"] == 0 and report["official_verifier_executed"] is False
    assert report["repository_source_runtime_probe"]["live_model_adoption_qualified"] is False
    assert environment.default_user == "root"


def test_absent_and_explicit_legacy_keep_deployment_without_probe(tmp_path, supported_archive):
    environments, reports = [], []
    for name, options in (("absent", {}), ("legacy", {"repository_source_policy": "legacy"})):
        environment = ProbeEnvironment(failure=AssertionError("legacy attempted a repository probe"))
        environments.append(environment)
        report = asyncio.run(deployment.deploy_supervisor(environment, archive_dir=supported_archive,
            output=tmp_path / name, install_codex=False, **options))
        report.pop("seconds")
        reports.append(report)
        assert "repository_source_policy" not in report and "repository_source_runtime_probe" not in report
        assert not (tmp_path / name / "repository-source-runtime-probe.log").exists()
        assert not any(row["step"] == "repository-source-runtime-probe" for row in report["steps"])
    assert reports[0] == reports[1]
    assert environments[0].commands == environments[1].commands
    assert environments[0].uploads == environments[1].uploads


@pytest.mark.parametrize("mutation", ["invalid_json", "not_object", "false_check", "missing_usage",
    "retained_state", "bool_provider_count", "process_failure", "timeout"])
def test_failed_or_malformed_probe_stops_before_native_start_and_success_receipt(tmp_path, supported_archive, mutation):
    value = deepcopy(authored_probe_receipt())
    if mutation == "false_check": value["checks"]["native_scope_joined"] = False
    elif mutation == "missing_usage": value.pop("usage")
    elif mutation == "retained_state": value["usage"]["store_cleanup_complete"] = False
    elif mutation == "bool_provider_count": value["provider_calls"] = False
    stdout = "not JSON" if mutation == "invalid_json" else "[]" if mutation == "not_object" else json.dumps(value)
    environment = ProbeEnvironment(stdout=stdout, return_code=7 if mutation == "process_failure" else 0,
        failure=TimeoutError("authored deployment probe timeout") if mutation == "timeout" else None)
    output = tmp_path / "deployment"
    expected_error = RuntimeError if mutation == "process_failure" else TimeoutError if mutation == "timeout" else ValueError
    with pytest.raises(expected_error):
        asyncio.run(deployment.deploy_supervisor(environment, archive_dir=supported_archive, output=output,
            install_codex=False, repository_source_policy=policy.COMPACT_REPOSITORY_SOURCE_POLICY))
    assert environment.commands[-1][0] == PROBE_COMMAND
    assert not any("isolated_benchmark_runtime --directory" in command for command, _ in environment.commands)
    assert not (output / "deployment.json").exists()
    assert environment.default_user == "root"
    if mutation != "timeout":
        assert (output / "repository-source-runtime-probe.log").read_text() == stdout


def test_valid_inventory_cannot_hide_invalid_archive_bytes(tmp_path, supported_archive, mode):
    archive = tmp_path / "archive"
    archive.mkdir()
    manifest = json.loads((supported_archive / "manifest.json").read_text())
    body = b"AUTHORED-TAR-CONTENT-MISMATCH"
    (archive / "runtime.tar.gz").write_bytes(body)
    # The outer archive hash agrees, and every source inventory row remains
    # current. Qualification still requires those exact members inside the tar.
    manifest["archive_sha256"] = hashlib.sha256(body).hexdigest()
    (archive / "manifest.json").write_text(json.dumps(manifest))
    environment = ProbeEnvironment()
    with pytest.raises(ValueError):
        asyncio.run(deployment.deploy_supervisor(environment, archive_dir=archive,
            output=tmp_path / "deployment", install_codex=False, repository_source_policy=mode))
    assert environment.commands == [] and environment.uploads == []
    assert not (tmp_path / "deployment").exists()
