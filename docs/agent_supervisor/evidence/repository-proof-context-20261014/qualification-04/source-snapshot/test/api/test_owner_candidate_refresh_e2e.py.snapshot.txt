"""Installed monitor -> real failed child -> native projections -> report join.

Container UID admission, cleanup executable and router receipt are authored.
The owner monitor, real child wait, broker retirement, native datasets grammar,
private checkpoint writer and benchmark collector execute their production code.
No provider, solver, target program or second coding child is invoked.
"""
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_MONITOR
from benchmarks.agent_supervisor.container_coding import terminal_owner_candidate_refresh_observation as collector
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_checkpoint as checkpoint
from test.api.test_owner_candidate_round_context import source_service, native, SOURCE, CHANGED  # noqa: F401


@pytest.mark.parametrize("router_status", ["failed", "provider_returned"])
def test_installed_owner_refresh_is_observed_from_real_native_artifacts(
        source_service, monkeypatch, capsys, router_status):
    service = source_service
    release = service.workspace / "authored-worker-release"
    original_run = subprocess.run
    original_register = service.note_spawned
    registered, cleanup_calls = [], []

    def register(process):
        original_register(process)
        registered.append(process)
        release.write_text("go")

    def cleanup(command, **kwargs):
        if command == ['/usr/bin/sudo', '-n', '-u', 'benchmarkworker', '--',
                '/opt/ipfs-supervisor/bin/worker-entry', '--cleanup']:
            assert len(registered) == 1 and registered[0].poll() == 7
            cleanup_calls.append(command)
            return original_run([sys.executable, "-c", "pass"], **kwargs)
        return original_run(command, **kwargs)

    monkeypatch.setattr(service, "note_spawned", register)
    monkeypatch.setattr(subprocess, "run", cleanup)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    # The fixture authors container path/UID admission; disk modes, bytes,
    # identities, hashes and source-session bindings remain real below.
    monkeypatch.setattr(collector, "_protected_directory", lambda *args, **kwargs: None)
    worker = ("import sys,time; from pathlib import Path\n"
        "release=Path(sys.argv[1]); deadline=time.monotonic()+10\n"
        "while not release.exists():\n"
        " if time.monotonic()>deadline: raise SystemExit(8)\n"
        " time.sleep(.01)\n"
        "Path(sys.argv[2]).write_bytes(bytes.fromhex(sys.argv[3]))\n"
        "raise SystemExit(7)\n")
    namespace = dict(command=[sys.executable, "-c", worker, str(release),
        str(service.workspace / "mod.py"), CHANGED.hex()], worker_source_broker=service,
        numerical_obligation_broker=None, owner_candidate_round_deadline=time.monotonic() + 60)
    with pytest.raises(SystemExit) as finished:
        exec(compile(OWNER_MONITOR, "<installed-owner-monitor>", "exec"), namespace)
    assert finished.value.code == 7
    assert len(registered) == len(cleanup_calls) == 1
    assert registered[0].returncode == 7
    assert not service.socket_path.exists() and not service.descriptor_path.exists()
    audit = json.loads(service.audit_path.read_bytes())
    assert audit["service_stopped"] is audit["accounting_complete"] is True
    assert (service.repository / "mod.py").read_bytes() == SOURCE
    assert (service.workspace / "mod.py").read_bytes() == CHANGED
    with pytest.raises(ValueError, match="stale"):
        service.views.admission()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(checkpoint.PREFIX)]
    assert len(lines) == 1
    observed = json.loads(lines[0][len(checkpoint.PREFIX):])
    assert observed["status"] == "prepared"
    parent = Path(service.boundary).parent / "worker-observations"
    manifest_path = parent / observed["checkpoint_manifest"]["path"]
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    formal = json.loads((parent / manifest["artifacts"]["formal"]["path"]).read_bytes())
    lean = formal["reports"]["mod.py"]["projections"]["lean"]
    definitions = {binding["definition"] for binding in lean["bindings"]}
    assert "SelectionMin" in definitions and "SelectionMax" not in definitions
    assert lean["status"] == "prepared_not_built"
    capsules = json.loads((parent / manifest["artifacts"]["capsules"]["path"]).read_bytes())
    assert any("offset" in json.dumps(row["signature"]) for row in capsules["capsules"])
    assert manifest["context"]["source_hashes"]["mod.py"] == hashlib.sha256(CHANGED).hexdigest()
    assert manifest["context"]["pending_obligation_count"] > 0
    assert manifest["context"]["next_round_admitted"] is False

    # Provider receipt is authored; its status alone cannot admit a retry or
    # suppress a real checkpoint after a later worker-wrapper failure.
    receipt = dict(schema="worker-source-dispatch-observation@1",
        session_id=service.descriptor["session_id"], descriptor_sha256=service.descriptor_sha256,
        **service._binding, proof_authority=False, execution_authority=False, completion_authority=False)
    row = dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        invocation_id="authored-native-owner-refresh", status=router_status, source_view_retrieval=receipt)
    result = collector.collect(parent=parent, scope=dict(task_alias=service._binding["task_id"]),
        report=dict(provider_invocations=[row], worker_cleanup_returncode=0))
    assert result["status"] == "observed", result
    assert result["counts"]["sessions"] == 1 and result["counts"]["artifacts"] == 3
    assert result["counts"]["artifact_bytes"] == observed["artifact_bytes"] > 0
    assert result["sessions"][0]["checkpoint_manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["sessions"][0]["artifacts"] == manifest["context"]["artifacts"]
    assert all(result[key] == value for key, value in checkpoint.FLAGS.items())
    assert result["provider_tokens"] is None and result["total_token_savings_measured"] is False
    assert result["accepted_index_published"] is result["artifact_delivery_observed"] is False
    exported = json.dumps(result)
    assert "mod.py" not in exported and "argmin" not in exported and service.prompt not in exported
