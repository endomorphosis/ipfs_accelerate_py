"""Actual disposable user unit, drop-in, pidfds, sentinel and DuckDB capture.

Native task/source observations use the existing isolated public fixture. This
qualifies host unit/cgroup mechanics and driver ordering, not SPAR admission.
No production unit, source, queue, stop marker or credentials are used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from scripts.ops.agent_supervisor import spar_retained_capture_driver as driver
from scripts.ops.agent_supervisor import spar_legacy_capture as capture
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import read_process_birth
from test.api.semantic_refactoring.test_spar_legacy_capture import armed as armed_fixture
from test.api.semantic_refactoring.test_spar_retained_capture_driver import fleet_fixture

armed = armed_fixture

pytestmark = pytest.mark.skipif(os.environ.get("SPAR_RUN_HOST_CAPTURE_TEST") != "1",
                              reason="explicit disposable user-systemd qualification")


def run(argv):
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=15)


@pytest.mark.parametrize("scenario", ["capture-install", "changed-existing-dropin", "replaced-prepared-database"])
def test_actual_host_driver_retains_sentinel_through_capture_and_install(armed, monkeypatch, scenario):
    root = armed.tmp
    unit = f"spar-driver-qualification-{os.getpid()}-{time.time_ns()}.service"
    drops = Path(f"/run/user/{os.geteuid()}/systemd/user") / (unit + ".d")
    drops.mkdir(mode=0o700, parents=True, exist_ok=False)
    prior = drops / "10-preserved-test.conf"
    prior.write_bytes(b"[Service]\nCPUWeight=40\n")
    prior_bytes = prior.read_bytes()
    script = root / "scripts/materialize_semantic_preserving_remodularization_program.py"
    script.parent.mkdir()
    script.write_text("""import json,time
from pathlib import Path
root=Path.cwd()
deadline=time.monotonic()+240
while not (root/'HOLD').exists() and time.monotonic()<deadline:
    time.sleep(.02)
path=root/'owner/quack-state-server.status.json'
body=json.loads(path.read_text())
body['lifecycle']='stopped'
path.write_text(json.dumps(body))
""")
    armed.session.close()
    armed.process.terminate()
    armed.process.wait(timeout=10)
    value = None
    owner_fd = None
    output = root.parent / (root.name + "-host-output")
    try:
        run(["systemd-run", "--user", "--unit=" + unit[:-8], "--property=Type=exec",
             "--property=WorkingDirectory=" + str(root), "--property=Restart=on-failure",
             "--property=SendSIGKILL=no", "/usr/bin/python3",
             "scripts/materialize_semantic_preserving_remodularization_program.py", "supervise", "--implement"])
        before = driver.unit_snapshot(unit)
        pid = int(before["MainPID"])
        owner_fd = os.pidfd_open(pid, 0)
        birth = read_process_birth(pid).to_dict()
        native = capture._native_operator(root)
        identity = native.authoritative_status(root / "config.json")["owner_identity"]
        identity["process_birth"] = birth
        armed.owner_path.write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
        broker = json.loads(armed.broker_path.read_text())
        broker["controller_pid"] = pid
        armed.broker_path.write_text(json.dumps(broker))
        monkeypatch.setattr(capture, "UNIT", unit)
        monkeypatch.setattr(capture, "_unit", lambda: driver.unit_snapshot(unit))
        fleet = fleet_fixture(root, monkeypatch, root, root / "HOLD")
        value = driver.RetainedCaptureDriver(repository_root=root, config_path=root / "config.json",
            fleet_config=fleet.config, operation_root=output,
            expected_source_commit=armed.source["head"], expected_source_tree=armed.source["tree"])
        inspected = value.inspect()
        assert not (root / "HOLD").exists() and not output.exists()
        if scenario == "changed-existing-dropin":
            prior.write_bytes(b"[Service]\nCPUWeight=50\n")
            with pytest.raises(driver.CaptureDriverDenied):
                value.arm(inspected["inspection_cid"])
            assert not (root / "HOLD").exists() and not (drops / driver.DROPIN).exists()
            assert not capture._exited(owner_fd) and not value.session._closed
            return
        result = value.arm(inspected["inspection_cid"])
        assert result["stage"] == "armed" and not (root / "HOLD").exists()
        assert not capture._exited(owner_fd)
        assert prior.read_bytes() == prior_bytes
        sentinel_pid = value.session._sentinel.pid
        sentinel_fd = os.pidfd_open(sentinel_pid, 0)
        try:
            value.request_closure()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                status = value.poll()
                if status["closure"] is not None:
                    break
                time.sleep(.02)
            assert value.stage == "closed"
            closure = status["closure"]
            assert capture._exited(owner_fd) and not capture._exited(sentinel_fd)
            assert closure["native_actors_closed"] is True
            assert closure["cgroup_empty_observed"] is False
            assert closure["workflow_sentinel"]["observed_cgroup_members"] == [sentinel_pid]
            assert driver.unit_snapshot(unit)["MainPID"] == "0"
            value.capture()
            assert not capture._exited(sentinel_fd)
            value.prepare()
            if scenario == "replaced-prepared-database":
                original = (value.session.queue_root / "merge_queue.duckdb").read_bytes()
                value.prepared.database_path.write_bytes(b"unvalidated replacement")
                with pytest.raises(Exception):
                    value.install()
                assert value.stage == "installing" and not value.session._closed
                assert not capture._exited(sentinel_fd)
                assert (value.session.queue_root / "merge_queue.duckdb").read_bytes() == original
                assert not (value.session.queue_root / driver.origin.REQUIRED_MARKER).exists()
                assert (root / "HOLD").read_bytes() == value.hold_bytes
                assert (drops / driver.DROPIN).read_bytes() == value.inhibitor.raw
                with pytest.raises(driver.CaptureDriverDenied):
                    value.finish()
                return
            installed = value.install()
            assert installed["installed"]["installed"] is True
            assert not capture._exited(sentinel_fd)
            done = value.finish()
            assert done["stage"] == "finished" and capture._exited(sentinel_fd)
            assert done["sentinel_close"]["returncode"] == 0
            assert done["unit_inhibition_retained"] is True and done["hold_retained"] is True
            assert (drops / driver.DROPIN).read_bytes() == value.inhibitor.raw
            assert (root / "HOLD").read_bytes() == value.hold_bytes
            assert prior.read_bytes() == prior_bytes
            assert all(done[key] is False for key in ("callback_settled", "signing_authority",
                "source_admitted", "successor_started", "completion_authority"))
            report = {"unit": unit, "before": before, "inspection_cid": inspected["inspection_cid"],
                      "closure": closure, "finished": done,
                      "prior_dropin_sha256": hashlib.sha256(prior_bytes).hexdigest(),
                      "native_source_and_task_admission": "isolated public fixture only"}
            (output / "host-qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True))
            print("HOST_CAPTURE_DRIVER_REPORT=" + str(output / "host-qualification.json"))
        finally:
            os.close(sentinel_fd)
    finally:
        # Only this uniquely named test unit and its own exact helper are closed.
        # The fixture actor exits normally on its own test HOLD.
        (root / "HOLD").touch(exist_ok=True)
        if value is not None and value.session is not None and not value.session._closed:
            value.session.close()
        if value is not None and value.exclusion is not None:
            value.exclusion.close()
        if owner_fd is not None:
            deadline = time.monotonic() + 15
            while not capture._exited(owner_fd) and time.monotonic() < deadline:
                time.sleep(.02)
            assert capture._exited(owner_fd), "disposable native actor failed normal exit"
            os.close(owner_fd)
        for path in (drops / driver.DROPIN, prior):
            path.unlink(missing_ok=True)
        drops.rmdir()
        run(["systemctl", "--user", "daemon-reload"])
        state = driver._scalars(unit, ("Id", "LoadState", "ActiveState", "SubState", "MainPID"))
        if state["LoadState"] != "not-found":
            run(["systemctl", "--user", "stop", unit])
        final = driver._scalars(unit, ("Id", "LoadState", "ActiveState", "SubState", "MainPID"))
        assert final["LoadState"] == "not-found" and final["MainPID"] == "0"
        assert not drops.exists()
