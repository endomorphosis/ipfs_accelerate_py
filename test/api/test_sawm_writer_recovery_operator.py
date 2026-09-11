"""Actual operator routing and M70 closure admission; disposable metadata only."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"


@pytest.fixture
def operator():
    spec = importlib.util.spec_from_file_location(
        "sawm_recovery_operator_fixture", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recovery_runtime_uses_exact_checkout_without_ambient_import_path(tmp_path):
    probe = """import runpy,sys,json
before=list(sys.path)
n=runpy.run_path(sys.argv[1],run_name="native_reader_fixture")
r=n["_writer_recovery_runtime"]()
print(json.dumps({"origin":r.__file__,"restored":sys.path==before}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", probe, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    value = json.loads(result.stdout)
    assert value["restored"] is True
    assert (
        Path(value["origin"]).resolve()
        == ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/sawm_writer_recovery.py"
    )


@pytest.mark.parametrize("malformed", [False, True])
def test_public_inspect_requires_actual_native_immutable_configuration(
    operator, tmp_path, monkeypatch, malformed
):
    config = operator._config(operator.CONFIG_PATH)
    if malformed:
        config[operator._M70_SUCCESSOR_KEY] = {"invalid": "not accepted authority"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    calls = []
    runtime = operator._writer_recovery_runtime()
    monkeypatch.setattr(
        runtime,
        "inspect",
        lambda *args: (
            calls.append(args) or {"schema": runtime.SCHEMA, "inspected": True}
        ),
    )
    emitted = []
    monkeypatch.setattr(operator, "_emit", lambda value: emitted.append(value) or 0)
    operator.main(["--config", str(config_path), "writer-recovery-inspect"])
    if malformed:
        assert calls == []
        assert emitted[0]["valid"] is False
        assert "authority" in emitted[0]["error"]
    else:
        assert len(calls) == 1
        assert calls[0][0] == ROOT
        assert emitted[0]["inspected"] is True


@pytest.mark.parametrize("digest_valid", [False, True])
def test_public_close_binds_manifest_and_durable_phase_before_mechanism(
    operator, tmp_path, monkeypatch, digest_valid
):
    runtime = operator._writer_recovery_runtime()
    manifest = tmp_path / "reviewed.json"
    raw = b'{"schema":"disposable-entry-fixture"}'
    manifest.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest() if digest_valid else "0" * 64
    calls = []
    monkeypatch.setattr(runtime, "startup_exclusion", lambda *_: lambda: None)

    def close(**kwargs):
        assert kwargs["expected"] == json.loads(raw)
        kwargs["record_phase"]("controller_suspend_prepared")
        calls.append(kwargs)
        return {
            "owner_exited": True,
            "callback_settlement_authority": False,
            "generation_changed": False,
        }

    monkeypatch.setattr(runtime, "close_reviewed", close)
    emitted = []
    monkeypatch.setattr(operator, "_emit", lambda value: emitted.append(value) or 0)
    journal = tmp_path / "journal.jsonl"
    operator.main(
        [
            "writer-recovery-close",
            "--expected-manifest",
            str(manifest),
            "--expected-sha256",
            digest,
            "--journal",
            str(journal),
        ]
    )
    if not digest_valid:
        assert calls == [] and not journal.exists()
        assert emitted[0]["valid"] is False
    else:
        assert len(calls) == 1
        row = json.loads(journal.read_text())
        assert (
            row["manifest_sha256"] == digest
            and row["phase"] == "controller_suspend_prepared"
        )
        assert emitted[0]["callback_settlement_authority"] is False


def test_existing_m70_gate_reuses_48_after_actual_process_exit_and_endpoint_close(
    operator, tmp_path, monkeypatch
):
    # Validate the real immutable source/configuration first. The rest of this
    # test supplies only private stopped process metadata to the unchanged gate.
    config = operator._config(operator.CONFIG_PATH)
    admitted = operator._active_source_repair_materialization(config)
    original = copy.deepcopy(config)
    monkeypatch.setattr(operator, "REPO_ROOT", tmp_path)

    def prior_scope(value):
        assert value == original
        return admitted

    monkeypatch.setattr(operator, "_active_source_repair_materialization", prior_scope)
    child = subprocess.Popen([sys.executable, "-I", "-S", "-c", "pass"])
    assert child.wait(timeout=5) == 0
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    monkeypatch.setattr(operator, "_M70_TARGET_QUACK_PORT", port)
    status = (
        tmp_path / config["quack_owner"]["state_dir"] / "quack-state-server.status.json"
    )
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "lifecycle": "stopped",
                "identity": {"generation": 48, "process_birth": {"pid": child.pid}},
            }
        )
    )
    assert not Path(f"/proc/{child.pid}").exists()
    result = operator._validate_offline_quack_start(config)
    assert result["store"]["action"] == "admitted_stopped_generation_48_reuse"
    assert (
        result["store"]["target_generation"]
        == result["store"]["prior_generation"]
        == 48
    )
    assert result["store"]["reuse_expected_generation"] is True
    assert result["store"]["prestart_authorization_consumed"] is False
    assert result["store"]["stale_ready_recovered"] is False
    # A currently live PID never qualifies that same generation-reuse branch.
    status.write_text(
        json.dumps(
            {
                "lifecycle": "stopped",
                "identity": {"generation": 48, "process_birth": {"pid": os.getpid()}},
            }
        )
    )
    assert operator._m70_published_owner_is_process_dead(config) is False
