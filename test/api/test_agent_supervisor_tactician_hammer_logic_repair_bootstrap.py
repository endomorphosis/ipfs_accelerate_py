from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts/tactician_hammer_logic_repair_supervisor.sh"
SCHEDULER = REPO_ROOT / "config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json"
BOARD_VALIDATOR = REPO_ROOT / "scripts/validate_tactician_hammer_logic_repair_board.py"


def _write_fake_runner(module_root: Path) -> None:
    module_root.joinpath("fake_lpr_runner.py").write_text(
        """\
from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


argv = sys.argv[1:]
repo_root = Path(value(argv, "--repo-root")).resolve()
pid_path = Path(value(argv, "--master-pid-path"))
log_path = Path(value(argv, "--master-log"))
if "--detach" in argv:
    child_argv = [item for item in argv if item != "--detach"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        child = subprocess.Popen(
            [sys.executable, "-m", "fake_lpr_runner", *child_argv],
            cwd=repo_root,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{child.pid}\\n", encoding="ascii")
    raise SystemExit(0)

pid_path.parent.mkdir(parents=True, exist_ok=True)
pid_path.write_text(f"{os.getpid()}\\n", encoding="ascii")
track = value(argv, "--implementation-track").split("|")
state_root = Path(track[2])
lane_count = int(value(argv, "--implementation-supervisor-lanes-per-track"))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
for lane in range(lane_count):
    lane_root = state_root / f"lane-{lane}"
    lane_root.mkdir(parents=True, exist_ok=True)
    prefix = f"lpr_lane_{lane}"
    task_path = lane_root / f"{prefix}_task_state.json"
    task_path.write_text(json.dumps({
        "active_task_id": "",
        "blocked_count": 0,
        "eligible_ready_count": 4,
        "implementation_in_progress": False,
        "selection_idle_reason": "",
    }), encoding="utf-8")
    (lane_root / f"{prefix}_supervisor_status.json").write_text(json.dumps({
        "current_status_path": str(task_path),
        "restart_count": 0,
        "status": "running",
        "supervisor_pid": os.getpid(),
        "updated_at": now,
    }), encoding="utf-8")


def stop(*_args: object) -> None:
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
print("fake LPR runner ready", flush=True)
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )


def _run(command: str, *, env: dict[str, str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(LAUNCHER), command],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == expected, (result.stdout, result.stderr)
    return result


def test_lpr_launcher_fake_process_lifecycle_is_idempotent_and_identity_safe(tmp_path):
    module_root = tmp_path / "modules"
    module_root.mkdir()
    _write_fake_runner(module_root)
    state_root = tmp_path / "isolated" / "lpr-state"
    canary = "LPR-SECRET-CANARY-MUST-NOT-BE-LOGGED"
    env = os.environ.copy()
    env.update(
        {
            "LPR_DURATION_SECONDS": "300",
            "LPR_RUNNER_MODULE": "fake_lpr_runner",
            "LPR_STATE_ROOT": str(state_root),
            "LPR_TEST_MODE": "1",
            "LPR_TEST_SECRET_CANARY": canary,
            "PYTHONPATH": os.pathsep.join(
                [str(module_root), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }
    )

    _run("stop", env=env)
    first = _run("start", env=env)
    assert "master: running" in first.stdout
    pid_path = state_root / "runtime" / "master.pid"
    first_pid = int(pid_path.read_text(encoding="ascii"))
    second = _run("start", env=env)
    assert int(pid_path.read_text(encoding="ascii")) == first_pid
    assert "already running" in second.stdout
    _run("status", env=env)

    argv = Path(f"/proc/{first_pid}/cmdline").read_bytes()
    log = (state_root / "runtime" / "master.log").read_text(encoding="utf-8")
    assert canary.encode() not in argv
    assert canary not in log

    _run("restart", env=env)
    restarted_pid = int(pid_path.read_text(encoding="ascii"))
    assert restarted_pid != first_pid
    _run("stop", env=env)
    _run("stop", env=env)

    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(f"{unrelated.pid}\n", encoding="ascii")
        (state_root / "runtime" / "master.identity.json").unlink(missing_ok=True)
        refused = _run("stop", env=env, expected=2)
        assert "unowned live PID" in refused.stderr
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
        pid_path.unlink(missing_ok=True)

    receipt = json.loads(
        (state_root / "runtime" / "launch-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["accelerator_branch"] == "agent/proof-gated-contract-repair"


def test_lpr_datasets_gitlink_contains_the_reviewed_tactician_contract():
    source = json.loads(SCHEDULER.read_text(encoding="utf-8"))["source_binding"]
    datasets_root = REPO_ROOT / source["datasets_submodule_path"]
    gitlink = subprocess.check_output(
        ["git", "rev-parse", "HEAD:ipfs_datasets_py"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    nested_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=datasets_root,
        text=True,
    ).strip()

    assert nested_head == gitlink
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            source["datasets_required_ancestor"],
            gitlink,
        ],
        cwd=datasets_root,
        check=True,
    )
    for required_path in source["datasets_required_paths"]:
        subprocess.run(
            ["git", "cat-file", "-e", f"{gitlink}:{required_path}"],
            cwd=datasets_root,
            check=True,
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(datasets_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ipfs_datasets_py.logic.tactician as tactician; "
                "print(tactician.TACTICIAN_INTERFACE)"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    assert imported.stdout.strip() == source["datasets_required_interface"]


def test_lpr_board_validator_keeps_retry_repairs_outside_the_sealed_dag():
    spec = importlib.util.spec_from_file_location(
        "lpr_board_validator_test",
        BOARD_VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    source = PortalTask(
        task_id="LPR-028",
        title="Canonical source",
        status="todo",
        completion="auto",
        priority="P0",
        track="release",
        depends_on=["LPR-027"],
        outputs=["src/cutover.py"],
        validation=["python -m pytest -q test_cutover.py"],
        acceptance="Complete the cutover.",
    )
    discovery_root = (
        "/tmp/agent-supervisor-state/tactician_hammer_logic_repair/"
        "state/discovery"
    )
    discovery_path = (
        f"{discovery_root}/2026-07-31-lpr-043-lpr-028-retry-budget.md"
    )
    repair = PortalTask(
        task_id="LPR-043",
        title="Resolve validation retry-budget failure for LPR-028",
        status="completed",
        completion="manual",
        priority="P1",
        track="ops",
        depends_on=["LPR-027"],
        outputs=["src/cutover.py", discovery_root],
        validation=[f"test -f {discovery_path}"],
        acceptance=(
            f"Use evidence in {discovery_path} to fix the blocker, then "
            "release LPR-028 from strategy blocked_tasks."
        ),
    )

    validator._validate_operational_repair_tasks(
        [repair],
        canonical_by_id={"LPR-028": source},
    )

    forged = PortalTask(
        **{
            **repair.__dict__,
            "title": "Unreviewed task outside the sealed DAG",
        }
    )
    with pytest.raises(
        validator.BoardValidationError,
        match="unrecognized operational retry-repair",
    ):
        validator._validate_operational_repair_tasks(
            [forged],
            canonical_by_id={"LPR-028": source},
        )
