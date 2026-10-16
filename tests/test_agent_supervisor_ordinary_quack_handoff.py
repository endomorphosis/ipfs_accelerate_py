"""Actual native control-process handoff regression tests.

Hermetic transport fixtures use a test-only sealed memfd, real subprocesses,
native PID/argv/executable verification, and temporary lifecycle roots. They
never contact Quack, run a model, or grant task/DB completion authority.
Execute only within the root's admitted native validation job.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import process_security as security


@pytest.fixture
def transport(tmp_path, monkeypatch):
    for name in (*security.STATE_AUTHORITY_CREDENTIAL_NAMES,
                 *security.STATE_AUTHORITY_HANDOFF_ENV_NAMES,
                 security.STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV,
                 "IPFS_ACCELERATE_AGENT_STATE_HANDOFF_EXECUTION_BINDING"):
        monkeypatch.delenv(name, raising=False)
    security.clear_captured_state_authority_credentials()
    descriptor = os.memfd_create("hermetic-supervisor-handoff", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    os.write(descriptor, b"fixture-only-broker-key-never-used-for-native-commands".ljust(64, b"."))
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD", str(descriptor))
    monkeypatch.setenv(security.STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV, "fixture-unused-owner-socket")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "fixture-only-token-never-used")
    source_root = Path(runner.__file__).resolve().parents[3]
    monkeypatch.setenv("PYTHONPATH", str(source_root))
    program = runner.DatabaseProgramConfig(authority_mode="quack", task_source_kind="duckdb",
        endpoint_secret_handle="env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        quack_endpoint="quack://127.0.0.1:65534", store_id="fixture-only-store",
        store_generation="1", schema_revision="1", failover_policy="fail_closed")
    yield {"root": tmp_path, "fd": descriptor, "program": program}
    try:
        os.close(descriptor)
    except OSError:
        pass
    security.clear_captured_state_authority_credentials()


def _track(transport, script, *, extra_args=(), program=True):
    root = transport["root"]
    return runner.SupervisorTrack(name="fixture-control-lane", script_path=script,
        log_path=root / "lane/supervisor.log", supervisor_pid_path=root / "lane/supervisor.pid",
        daemon_pid_path=root / "lane/daemon.pid", extra_args=tuple(map(str, extra_args)),
        database_program=transport["program"] if program else None)


def _start(transport, track, **kwargs):
    return runner.start_track(track, repo_root=transport["root"], common_args=(),
        python_executable=sys.executable, output=lambda _: None, **kwargs)


def _write_script(path, body):
    path.write_text(body)
    path.chmod(0o600)
    return path


def test_native_master_handoff_reaches_managed_daemon_and_scrubs_provider(transport):
    root = transport["root"]
    worker_receipt, supervisor_receipt = root / "worker-result.json", root / "supervisor-result.json"
    worker = _write_script(root / "trusted_worker.py", '''
import json,os,signal,subprocess,sys
from pathlib import Path
signal.alarm(20)
raw_descriptor_at_exec = bool(os.environ.get("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"))
from ipfs_accelerate_py.agent_supervisor.runtime.process_security import harden_state_authority_process,state_authority_pass_fds
harden_state_authority_process()
fds = state_authority_pass_fds(os.environ)
assert len(fds) == 1
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import provider_subprocess_environment
safe = provider_subprocess_environment(os.environ)
provider = subprocess.run([sys.executable,"-c", "import os; assert not any('STATE_GRANT' in k or k.endswith('QUACK_TOKEN') for k in os.environ); print('provider-isolated')"], env=safe, close_fds=True, capture_output=True, text=True, timeout=10)
assert provider.returncode == 0 and provider.stdout.strip() == "provider-isolated"
Path(sys.argv[1]).write_text(json.dumps({"raw_descriptor_at_exec":raw_descriptor_at_exec,"received_sealed_descriptor":len(fds)==1,"provider_isolated":True}))
''')
    supervisor = _write_script(root / "trusted_supervisor.py", '''
import json,os,signal,sys
from pathlib import Path
signal.alarm(30)
raw_descriptor_at_exec = bool(os.environ.get("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"))
from ipfs_accelerate_py.agent_supervisor.runtime.process_security import harden_state_authority_process,state_authority_pass_fds
harden_state_authority_process()
fds = state_authority_pass_fds(os.environ)
assert len(fds) == 1
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import SupervisedChildSpec,launch_supervised_child,wait_for_child_exit
root=Path(sys.argv[1])
child=launch_supervised_child(SupervisedChildSpec(repo_root=root,command=(sys.executable,sys.argv[2],sys.argv[3]),log_path=root/'worker.log',child_pid_path=root/'worker.pid',worker_credential_handoff=True))
assert wait_for_child_exit(child,poll_interval_seconds=.01) == 0
Path(sys.argv[4]).write_text(json.dumps({"raw_descriptor_at_exec":raw_descriptor_at_exec,"received_sealed_descriptor":len(fds)==1,"managed_daemon_completed":True}))
''')
    process = _start(transport, _track(transport, supervisor,
        extra_args=(root, worker, worker_receipt, supervisor_receipt)))
    try:
        assert process.wait(timeout=40) == 0
        observed = json.loads(supervisor_receipt.read_text())
        worker_observed = json.loads(worker_receipt.read_text())
        assert observed == {"raw_descriptor_at_exec": False, "received_sealed_descriptor": True,
                            "managed_daemon_completed": True}
        assert worker_observed == {"raw_descriptor_at_exec": False, "received_sealed_descriptor": True,
                                   "provider_isolated": True}
        assert os.fstat(transport["fd"]).st_size == 64
    finally:
        if process.poll() is None:
            fenced, _ = runner._terminate_managed_process(process, grace_seconds=2)
            assert fenced
            process.wait(timeout=5)


def test_unavailable_broker_descriptor_refuses_before_supervisor_birth(transport, monkeypatch):
    marker = transport["root"] / "must-not-run.json"
    script = _write_script(transport["root"] / "must_not_run.py",
                           "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('unexpected')\n")
    # A closed low descriptor can be reused by the native lifecycle-directory
    # guard before this check. Use a positively unavailable in-range slot while
    # retaining the fixture memfd for its normal teardown.
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD", "1048575")
    with pytest.raises(security.StateAuthorityProcessIsolationError, match="descriptor is unavailable"):
        _start(transport, _track(transport, script))
    assert not marker.exists()
    assert not (transport["root"] / "lane/supervisor.pid").exists()


def test_unacknowledged_native_handoff_fences_exact_born_tree(transport):
    script = _write_script(transport["root"] / "nonredeeming_supervisor.py",
                           "import time\ntime.sleep(60)\n")
    with pytest.raises(runner.SupervisorTrackStartError) as caught:
        _start(transport, _track(transport, script), birth_deadline_monotonic_seconds=time.monotonic()+1)
    error = caught.value
    assert error.cause_type == "StateAuthorityProcessIsolationError"
    assert error.all_trees_fenced is True
    assert not Path(f"/proc/{error.pid}").exists()
    assert not (transport["root"] / "lane/supervisor.pid").exists()


def test_legacy_track_without_authority_keeps_ordinary_launch(transport, monkeypatch):
    for name in (*security.STATE_AUTHORITY_CREDENTIAL_NAMES, security.STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV):
        monkeypatch.delenv(name, raising=False)
    marker = transport["root"] / "legacy-result.json"
    script = _write_script(transport["root"] / "legacy_track.py",
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('legacy-ok')\n")
    process = _start(transport, _track(transport, script, program=False))
    assert process.wait(timeout=10) == 0 and marker.read_text() == "legacy-ok"


def test_incomplete_broker_binding_fails_closed_before_birth(transport, monkeypatch):
    monkeypatch.delenv(security.STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV)
    script = _write_script(transport["root"] / "must_not_run.py", "raise SystemExit(99)\n")
    with pytest.raises(security.StateAuthorityProcessIsolationError, match="binding is incomplete"):
        _start(transport, _track(transport, script))
    assert not (transport["root"] / "lane/supervisor.pid").exists()


def test_real_cli_tracks_with_shared_quack_flags_redeem_without_program_metadata(transport):
    root=transport['root']
    receipt=root/'cli-handoff-result.json'
    script=_write_script(root/'trusted_cli_supervisor.py', '''
import argparse,json,os,signal
from pathlib import Path
signal.alarm(15)
parser=argparse.ArgumentParser()
parser.add_argument('--fixture-result',required=True)
args,_unknown=parser.parse_known_args()
raw_descriptor_at_exec=bool(os.environ.get('IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD'))
from ipfs_accelerate_py.agent_supervisor.runtime.process_security import harden_state_authority_process,state_authority_pass_fds
harden_state_authority_process()
fds=state_authority_pass_fds(os.environ)
assert len(fds)==1
Path(args.fixture_result).write_text(json.dumps({'raw_descriptor_at_exec':raw_descriptor_at_exec,'received_sealed_descriptor':True}))
''')
    flag='--implementation-track'
    spec=runner.implementation_supervisor_compact_track_spec(name='actual-cli-fixture',
        script_path=script,state_dir=root/'cli-lane',state_prefix='fixture_cli')
    common=[*transport['program'].cli_args(),'--fixture-result',str(receipt)]
    parsed=runner.build_arg_parser().parse_args(['--repo-root',str(root),flag,spec,
                                                *['--common-arg='+arg for arg in common]])
    tracks=runner.tracks_from_parsed_args(parsed)
    actual_common=runner.common_args_from_parsed_args(parsed)
    assert len(tracks)==1 and tracks[0].database_program is None
    assert actual_common[actual_common.index('--authority-mode')+1]=='quack'
    process=runner.start_track(tracks[0],repo_root=root,common_args=actual_common,
        python_executable=sys.executable,output=lambda _:None)
    try:
        assert process.wait(timeout=20)==0
        assert json.loads(receipt.read_text())=={'raw_descriptor_at_exec':False,
                                                'received_sealed_descriptor':True}
    finally:
        if process.poll() is None:
            fenced,_=runner._terminate_managed_process(process,grace_seconds=2)
            assert fenced
            process.wait(timeout=5)
