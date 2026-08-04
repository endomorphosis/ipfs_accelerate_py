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


def test_lpr_launcher_exports_derived_state_root_to_child_validators():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    derive = 'PROGRAM_ROOT="${LPR_STATE_ROOT:-'
    propagate = 'export LPR_STATE_ROOT="${PROGRAM_ROOT}"'
    launch = "launch_master() {"

    assert derive in launcher
    assert propagate in launcher
    assert launcher.index(derive) < launcher.index(propagate) < launcher.index(launch)


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
    delay = float(os.environ.get("FAKE_LPR_DETACH_DELAY_SECONDS", "0"))
    if delay > 0:
        time.sleep(delay)
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


def test_lpr_launcher_serializes_concurrent_starts(tmp_path):
    module_root = tmp_path / "modules"
    module_root.mkdir()
    _write_fake_runner(module_root)
    state_root = tmp_path / "isolated" / "lpr-state"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_LPR_DETACH_DELAY_SECONDS": "0.5",
            "LPR_DURATION_SECONDS": "300",
            "LPR_RUNNER_MODULE": "fake_lpr_runner",
            "LPR_STATE_ROOT": str(state_root),
            "LPR_TEST_MODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(module_root), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }
    )

    first = subprocess.Popen(
        [str(LAUNCHER), "start"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    second = subprocess.Popen(
        [str(LAUNCHER), "start"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    first_stdout, first_stderr = first.communicate(timeout=45)
    second_stdout, second_stderr = second.communicate(timeout=45)
    try:
        assert first.returncode == 0, (first_stdout, first_stderr)
        assert second.returncode == 0, (second_stdout, second_stderr)
        assert "already running" in first_stdout + second_stdout
        log = (state_root / "runtime" / "master.log").read_text(encoding="utf-8")
        assert log.count("fake LPR runner ready") == 1
        identity = json.loads(
            (state_root / "runtime" / "master.identity.json").read_text(
                encoding="utf-8"
            )
        )
        assert identity["pid"] == int(
            (state_root / "runtime" / "master.pid").read_text(encoding="ascii")
        )
    finally:
        _run("stop", env=env)


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


def test_lpr_board_validator_accepts_only_provenanced_reconciliation_appendices(
    tmp_path,
    monkeypatch,
):
    spec = importlib.util.spec_from_file_location(
        "lpr_board_validator_reconciliation_test",
        BOARD_VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    fingerprint = "50d47eee193a7305a41ab449609cadbbd96264e7"
    discovery_root = str(tmp_path / "state" / "discovery")
    validator.EXPECTED_RECONCILIATION_DISCOVERY_ROOT = Path(
        discovery_root
    ).resolve()
    configured_state_root = tmp_path / "configured-program"
    monkeypatch.setenv("LPR_STATE_ROOT", str(configured_state_root))
    assert (
        configured_state_root / "state" / "discovery"
    ).resolve() in validator._expected_reconciliation_discovery_roots()
    discovery_path = (
        f"{discovery_root}/"
        "2026-07-31-lpr-045-reconciliation-50d47eee193a.md"
    )
    receipt = {
        "schema": validator.RECONCILIATION_RESOLUTION_SCHEMA,
        "task_id": "LPR-045",
        "reconciliation_fingerprint": fingerprint,
        "kind": "dirty_backlogged_worktree",
        "reason": "unsupported_status",
        "resolved": True,
        "resolved_at": "2026-07-31T22:30:00+00:00",
        "resolution_method": "test_fixture_cleanup",
        "postconditions": {
            "candidate_count_before": 1,
            "candidate_count_after": 0,
            "active_blocker_present_after": False,
            "dirty_worktree_group_count_after": 0,
            "cleanup_skip_count_after": 0,
        },
        "evidence": {"fixture": "completed reconciliation"},
    }
    receipt["receipt_digest"] = validator._resolution_receipt_digest(receipt)
    discovery_file = Path(discovery_path)
    discovery_file.parent.mkdir(parents=True)
    discovery_file.write_text(
        "## Resolution Receipt\n\n```json\n"
        + json.dumps(receipt, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    task = PortalTask(
        task_id="LPR-045",
        title=(
            "Resolve 1 dirty backlogged worktrees blocked by "
            "unsupported_status"
        ),
        status="completed",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=[
            discovery_root,
            "docs/architecture/"
            "agent_supervisor_tactician_hammer_logic_repair.todo.md",
        ],
        validation=[f"test -f {discovery_path}"],
        acceptance=(
            "Reconciliation guardrail filed this because 1 branch or "
            "worktree cleanup candidates are blocked by unsupported_status. "
            f"Use evidence in {discovery_path} to reconcile it."
        ),
        metadata={
            "generated by": (
                "ipfs_accelerate_py.agent_supervisor."
                "reconciliation-guardrail@1"
            ),
            "reconciliation kind": "dirty_backlogged_worktree",
            "reconciliation reason": "unsupported_status",
            "reconciliation fingerprint": fingerprint,
            "reconciliation discovery": discovery_path,
            "resolution receipt digest": receipt["receipt_digest"],
            "canonical board task": "false",
            "fingerprint": fingerprint,
            "dedupe key": (
                "reconciliation_guardrail:dirty_backlogged_worktree:"
                "unsupported_status"
            ),
            "is schedulable": "false",
            "review only": "true",
            "blocked reason": "operator_reconciliation_required",
        },
    )

    board_tasks = {
        item.task_id: item
        for item in validator.parse_task_file(
            validator.REPO_ROOT / validator.TODO_PATH,
            task_header_prefix=validator.TASK_PREFIX,
        )
    }
    canonical_by_id = {
        task_id: board_tasks[task_id]
        for task_id in validator.EXPECTED_TASK_IDS
    }
    legacy_repairs = [board_tasks["LPR-043"], board_tasks["LPR-044"]]

    validator._validate_operational_repair_tasks(
        [*legacy_repairs, task],
        canonical_by_id=canonical_by_id,
    )

    for legacy_task_id in sorted(validator.LEGACY_OPERATIONAL_REPAIR_TASK_IDS):
        retyped_legacy = PortalTask(
            **{**task.__dict__, "task_id": legacy_task_id}
        )
        with pytest.raises(
            validator.BoardValidationError,
            match="reserved for its historical retry-repair contract",
        ):
            validator._validate_operational_repair_tasks(
                (
                    [retyped_legacy]
                    if legacy_task_id == "LPR-043"
                    else [legacy_repairs[0], retyped_legacy]
                ),
                canonical_by_id=canonical_by_id,
            )

    forged = PortalTask(
        **{
            **task.__dict__,
            "metadata": {
                **task.metadata,
                "reconciliation reason": "arbitrary_reason",
            },
        }
    )
    with pytest.raises(
        validator.BoardValidationError,
        match="reconciliation reason is unsupported",
    ):
        validator._validate_operational_repair_tasks(
            [*legacy_repairs, forged],
            canonical_by_id=canonical_by_id,
        )

    duplicate_discovery_path = discovery_path.replace("lpr-045", "lpr-046")
    duplicate = PortalTask(
        **{
            **task.__dict__,
            "task_id": "LPR-046",
            "status": "blocked",
            "validation": [f"test -f {duplicate_discovery_path}"],
            "acceptance": task.acceptance.replace(
                discovery_path,
                duplicate_discovery_path,
            ),
            "metadata": {
                **{
                    key: value
                    for key, value in task.metadata.items()
                    if key != "resolution receipt digest"
                },
                "reconciliation discovery": duplicate_discovery_path,
            },
        }
    )
    validator._validate_operational_repair_tasks(
        [*legacy_repairs, task, duplicate],
        canonical_by_id=canonical_by_id,
    )

    anchored_blocked = PortalTask(**{**task.__dict__, "status": "blocked"})
    with pytest.raises(
        validator.BoardValidationError,
        match="blocked reconciliation has a stale receipt anchor",
    ):
        validator._validate_operational_repair_tasks(
            [*legacy_repairs, anchored_blocked],
            canonical_by_id=canonical_by_id,
        )

    blocked = PortalTask(
        **{
            **task.__dict__,
            "status": "blocked",
            "metadata": {
                key: value
                for key, value in task.metadata.items()
                if key != "resolution receipt digest"
            },
        }
    )
    with pytest.raises(
        validator.BoardValidationError,
        match="concurrent duplicate operational reconciliation task",
    ):
        validator._validate_operational_repair_tasks(
            [*legacy_repairs, blocked, duplicate],
            canonical_by_id=canonical_by_id,
        )

    unanchored = PortalTask(
        **{
            **task.__dict__,
            "metadata": {
                key: value
                for key, value in task.metadata.items()
                if key != "resolution receipt digest"
            },
        }
    )
    with pytest.raises(
        validator.BoardValidationError,
        match="resolution receipt anchor mismatch",
    ):
        validator._validate_operational_repair_tasks(
            [*legacy_repairs, unanchored],
            canonical_by_id=canonical_by_id,
        )

    receipt["evidence"] = {"fixture": "tampered and internally rehashed"}
    receipt["receipt_digest"] = validator._resolution_receipt_digest(receipt)
    discovery_file.write_text(
        "## Resolution Receipt\n\n```json\n"
        + json.dumps(receipt, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(
        validator.BoardValidationError,
        match="resolution receipt anchor mismatch",
    ):
        validator._validate_operational_repair_tasks(
            [*legacy_repairs, task],
            canonical_by_id=canonical_by_id,
        )

    receipt["evidence"] = {"fixture": "tampered after hashing"}
    discovery_file.write_text(
        "## Resolution Receipt\n\n```json\n"
        + json.dumps(receipt, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(
        validator.BoardValidationError,
        match="resolution receipt digest mismatch",
    ):
        validator._validate_operational_repair_tasks(
            [*legacy_repairs, task],
            canonical_by_id=canonical_by_id,
        )
