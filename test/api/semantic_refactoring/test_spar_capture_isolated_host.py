"""Exact -I controller, real user unit/sentinel/locks and disposable DB copies.

Native task/source and repair-idle observations are explicit public fixtures.
No production process, source, database, credential or stop marker is used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time

import pytest

from scripts.ops.agent_supervisor import spar_capture_runtime as runtime
from scripts.ops.agent_supervisor import spar_legacy_capture as capture
from scripts.ops.agent_supervisor import spar_retained_capture_driver as driver
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import read_process_birth
from test.api.semantic_refactoring.test_spar_legacy_capture import armed as armed_fixture
from test.api.semantic_refactoring.test_spar_retained_capture_driver import fleet_fixture
from test.api.semantic_refactoring.test_spar_retained_capture_driver_host import run

armed = armed_fixture
ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(os.environ.get("SPAR_RUN_HOST_CAPTURE_TEST") != "1",
                              reason="explicit disposable user-systemd qualification")

BOOTSTRAP = '''import sys,json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
from scripts.ops.agent_supervisor import spar_retained_capture_driver as d
from scripts.ops.agent_supervisor import spar_legacy_capture as c
fixture=json.loads(Path(sys.argv[2]).read_text());root=Path(fixture['root'])
board=SimpleNamespace(**fixture['board']);board.repo_root=root;board.path=lambda value:Path(value)
source=fixture['source']
operator=SimpleNamespace(
 _load_config=lambda path:(board,{}),
 _runtime_paths=lambda board:{'owner':root/'owner','bootstrap_receipt':root/'bootstrap.json'},
 _identity=d.role._cid,source_binding=lambda config:source,
 authoritative_status=lambda config:{'authoritative_task_observation':True,'board_namespace':'spar',
  'owner_identity':json.loads((root/'owner/quack-state-server.status.json').read_text())['identity'],
  'leases':[],'closeout_snapshot':{'closeout_facts':{'truncated':False,'all_relations_available':True}}})
c.UNIT=fixture['unit'];c._unit=lambda:d.unit_snapshot(fixture['unit'])
c._native_operator=lambda root:operator;c._repository_id=lambda root:'repo:spar'
c.configured_queue_root=lambda board:Path(fixture['queue'])
d._job_unit=lambda:fixture['repair_idle']
if fixture['failure']=='capture':
 original=c.producer.produce_offline_import_plan;calls=[]
 def injected(**kwargs):
  result=original(**kwargs);calls.append(1)
  if len(calls)==1:raise ModuleNotFoundError('not exported private example',name='duckdb')
  return result
 c.producer.produce_offline_import_plan=injected
elif fixture['failure']=='prepare':
 original=d.role.prepare_offline_clone;calls=[]
 def injected(**kwargs):
  result=original(**kwargs);calls.append(1)
  if len(calls)==1:raise OSError('not exported private example')
  return result
 d.role.prepare_offline_clone=injected
assert sys.flags.isolated==1
assert not any('/.local/' in p and 'site-packages' in p for p in sys.path)
raise SystemExit(d.main(fixture['argv']))
'''


def _reply(process):
    readable, _, _ = select.select([process.stdout], [], [], 40)
    assert readable, "isolated controller did not respond within its observed operation budget"
    raw = process.stdout.readline()
    assert raw, "isolated controller exited without a response"
    return json.loads(raw)


def _stage(process, action, **fields):
    process.stdin.write(json.dumps({"action": action, **fields}) + "\n")
    process.stdin.flush()
    return _reply(process)


def _queue_fds(pid, queue):
    result = []
    for entry in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        if target.startswith(str(queue) + "/"):
            locks = [line for line in Path(f"/proc/{pid}/fdinfo/{entry.name}").read_text().splitlines()
                     if line.startswith("lock:")]
            if locks:
                result.append((int(entry.name), target, locks))
    return sorted(result)


@pytest.mark.parametrize("failure", ["capture", "prepare"])
def test_isolated_controller_retains_custody_and_retries_before_install(armed, monkeypatch, failure):
    root = armed.tmp
    unit = f"spar-driver-isolated-{os.getpid()}-{time.time_ns()}.service"
    drops = Path(f"/run/user/{os.geteuid()}/systemd/user") / (unit + ".d")
    drops.mkdir(mode=0o700, parents=True)
    prior = drops / "10-preserved-test.conf"
    prior.write_bytes(b"[Service]\nCPUWeight=40\n")
    script = root / "scripts/materialize_semantic_preserving_remodularization_program.py"
    script.parent.mkdir()
    script.write_text("""import json,time
from pathlib import Path
root=Path.cwd();deadline=time.monotonic()+240
while not (root/'HOLD').exists() and time.monotonic()<deadline:time.sleep(.02)
p=root/'owner/quack-state-server.status.json';body=json.loads(p.read_text())
body['lifecycle']='stopped';p.write_text(json.dumps(body))
""")
    armed.session.close()
    armed.process.terminate()
    armed.process.wait(timeout=10)
    output = root.parent / (root.name + "-isolated-output")
    process = None
    owner_fd = sentinel_fd = None
    stderr_path = root / "isolated-controller.stderr"
    try:
        run(["systemd-run", "--user", "--unit=" + unit[:-8], "--property=Type=exec",
             "--property=WorkingDirectory=" + str(root), "--property=Restart=on-failure",
             "--property=SendSIGKILL=no", "/usr/bin/python3",
             "scripts/materialize_semantic_preserving_remodularization_program.py", "supervise", "--implement"])
        owner_pid = int(driver.unit_snapshot(unit)["MainPID"])
        owner_fd = os.pidfd_open(owner_pid, 0)
        identity = json.loads(armed.owner_path.read_text())["identity"]
        identity["process_birth"] = read_process_birth(owner_pid).to_dict()
        armed.owner_path.write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
        broker = json.loads(armed.broker_path.read_text());broker["controller_pid"] = owner_pid
        armed.broker_path.write_text(json.dumps(broker))
        monkeypatch.setattr(capture, "UNIT", unit)
        fleet = fleet_fixture(root, monkeypatch, root, root / "HOLD")
        manifest = root / "reviewed-runtime.json"
        manifest.write_text(json.dumps(runtime.observed_runtime(), sort_keys=True))
        manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        fixture = {
            "root": str(root), "unit": unit, "queue": str(armed.queue), "failure": failure,
            "repair_idle": fleet.unit,
            "source": {**armed.source, "forest": {"head": armed.source["head"]},
                       "config_sha256": driver.role._cid({}).removeprefix("sha256:")},
            "board": {"board_namespace": "spar", "task_prefix": "SPAR-", "max_lanes": 1,
                      "runtime_paths": {"root": str(root)}, "payload": {
                          "merge_target_branch": "main", "runtime_paths": {
                              "root": str(root), "state": str(root / "state"),
                              "merge_queue": str(armed.queue)}}},
            "argv": ["--repository-root", str(root), "--config-path", str(root / "config.json"),
                     "--fleet-config", str(fleet.config), "--operation-root", str(output),
                     "--expected-source-commit", armed.source["head"],
                     "--expected-source-tree", armed.source["tree"], "--session",
                     "--runtime-manifest", str(manifest), "--runtime-manifest-sha256", manifest_digest],
        }
        fixture_path = root / "public-fixture.json";fixture_path.write_text(json.dumps(fixture))
        bootstrap = root / "isolated-bootstrap.py";bootstrap.write_text(BOOTSTRAP)
        with stderr_path.open("w") as stderr:
            process = subprocess.Popen(["/usr/bin/python3", "-I", str(bootstrap), str(ROOT), str(fixture_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True,
                env={"PATH": os.defpath, "LANG": "C.UTF-8",
                     "XDG_RUNTIME_DIR": f"/run/user/{os.geteuid()}"})
        inspected = _reply(process)
        assert inspected["stage"] == "inspected", (inspected, stderr_path.read_text())
        assert inspected["runtime"]["isolated"] is True
        assert not output.exists() and not (root / "HOLD").exists()
        armed_result = _stage(process, "arm", inspection_cid=inspected["inspection_cid"])
        assert armed_result["stage"] == "armed", armed_result
        sentinel_pid = armed_result["sentinel"]["birth"]["pid"]
        sentinel_fd = os.pidfd_open(sentinel_pid, 0)
        assert _stage(process, "request_closure")["stage"] == "close-requested"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            closed = _stage(process, "poll")
            if closed["stage"] == "closed":break
            time.sleep(.02)
        assert closed["stage"] == "closed" and capture._exited(owner_fd)
        assert not capture._exited(sentinel_fd)
        captured = _stage(process, "capture")
        if failure == "capture":
            assert captured["stage"] == "capture-failed", captured
            assert captured["diagnostic"]["missing_module"] == "duckdb"
            before = _queue_fds(process.pid, armed.queue)
            assert len(before) == 4
            assert _stage(process, "status")["retry_available"] == "capture"
            captured = _stage(process, "retry_capture")
            assert captured["stage"] == "captured", captured
            assert _queue_fds(process.pid, armed.queue) == before
            assert captured["capture_path"].endswith("raw-capture-002")
            assert (output / "raw-capture/merge_queue.duckdb").is_file()
        else:
            assert captured["stage"] == "captured", captured
        prepared = _stage(process, "prepare")
        if failure == "prepare":
            assert prepared["stage"] == "prepare-failed", prepared
            before = _queue_fds(process.pid, armed.queue)
            assert len(before) == 4
            assert _stage(process, "status")["retry_available"] == "prepare"
            prepared = _stage(process, "retry_prepare")
            assert prepared["stage"] == "prepared", prepared
            assert _queue_fds(process.pid, armed.queue) == before
            assert (output / "prepared-clone/merge_queue.duckdb").is_file()
            assert (output / "prepared-clone-002/merge_queue.duckdb").is_file()
        assert prepared["stage"] == "prepared" and not capture._exited(sentinel_fd)
        installed = _stage(process, "install")
        assert installed["stage"] == "installed", installed
        finished = _stage(process, "finish")
        assert finished["stage"] == "finished", finished
        assert capture._exited(sentinel_fd) and finished["sentinel_close"]["returncode"] == 0
        assert process.wait(timeout=10) == 0
        assert (root / "HOLD").exists() and (drops / driver.DROPIN).exists()
        assert all(finished[key] is False for key in (
            "callback_settled", "completion_authority", "source_admitted", "successor_started"))
        (output / "isolated-host-qualification.json").write_text(json.dumps({
            "argv": ["/usr/bin/python3", "-I", str(bootstrap)], "failure": failure,
            "inspection": inspected, "retained_queue_fds": before, "finished": finished,
            "public_native_observations": "explicit isolated fixture only",
        }, indent=2))
        print("ISOLATED_HOST_CAPTURE_REPORT=" + str(output / "isolated-host-qualification.json"))
    finally:
        (root / "HOLD").touch(exist_ok=True)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        for descriptor in (owner_fd, sentinel_fd):
            if descriptor is not None:
                deadline = time.monotonic() + 15
                while not capture._exited(descriptor) and time.monotonic() < deadline:time.sleep(.02)
                assert capture._exited(descriptor)
                os.close(descriptor)
        for path in (drops / driver.DROPIN, prior):path.unlink(missing_ok=True)
        drops.rmdir()
        run(["systemctl", "--user", "daemon-reload"])
        state = driver._scalars(unit, ("Id", "LoadState", "ActiveState", "SubState", "MainPID"))
        if state["LoadState"] != "not-found":run(["systemctl", "--user", "stop", unit])
        final = driver._scalars(unit, ("Id", "LoadState", "ActiveState", "SubState", "MainPID"))
        assert final["LoadState"] == "not-found" and final["MainPID"] == "0"
        assert not drops.exists()
