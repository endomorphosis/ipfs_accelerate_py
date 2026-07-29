"""Harden/validate the two-provider Grok Build/Codex VFS assurance control.

Behavioral coverage uses temporary git repositories, isolated state roots, and
a fake supervisor process. Real providers are never started.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "ops"
    / "agent_supervisor"
    / "ipfs_kit_vfs_symbolic_assurance_control.sh"
)

PROTECTED_PATHS = (
    "docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md",
    "docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md",
    "docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md",
    "scripts/ops/agent_supervisor/validate_ipfs_kit_vfs_symbolic_assurance.py",
)

SUBMODULE_PATHS = (
    "ipfs_accelerate_py/mcplusplus",
    "ipfs_datasets_py",
    "ipfs_kit_py",
)

TARGET_BRANCH = "agent/swissknife-contract-audit"
SECRET_DENYLIST = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
    "CODEX_API_KEY",
    "API_KEY",
    "AUTHORIZATION",
    "PASSWORD",
    "TOKEN",
    "SECRET",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        text=True,
        capture_output=True,
    )


def _init_temp_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "vfs-control@example.invalid")
    _git(root, "config", "user.name", "VFS Control Test")
    for rel in PROTECTED_PATHS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture {rel}\n", encoding="utf-8")
    control_copy = root / "scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance_control.sh"
    control_copy.parent.mkdir(parents=True, exist_ok=True)
    control_copy.write_text(CONTROL_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    control_copy.chmod(control_copy.stat().st_mode | stat.S_IXUSR)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    _git(root, "branch", "-M", TARGET_BRANCH)
    return root


def _write_fake_supervisor(bin_path: Path) -> Path:
    """Write a long-lived fake supervisor that honors ownership argv contracts."""
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import signal
            import sys
            import time
            from pathlib import Path

            args = sys.argv[1:]

            def _flag(name: str) -> str | None:
                if name in args:
                    idx = args.index(name)
                    if idx + 1 < len(args):
                        return args[idx + 1]
                return None

            state_dir = _flag("--state-dir")
            state_prefix = _flag("--state-prefix")
            todo_path = _flag("--todo-path")
            shard_index = _flag("--task-shard-index")
            shard_count = _flag("--task-shard-count")
            worktree_root = _flag("--worktree-root")
            merge_queue = _flag("--merge-queue-dir")
            if not state_dir or not state_prefix or not todo_path:
                print("fake supervisor missing required argv", file=sys.stderr)
                raise SystemExit(2)

            # Refuse secret-like argv tokens (mirrors control contract).
            secret_needles = (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "XAI_API_KEY",
                "GROK_API_KEY",
                "CODEX_API_KEY",
                "API_KEY",
                "AUTHORIZATION",
                "PASSWORD",
                "TOKEN",
                "SECRET",
            )
            joined = " ".join(args)
            for needle in secret_needles:
                if needle in joined:
                    print(f"secret-like argv rejected: {needle}", file=sys.stderr)
                    raise SystemExit(3)

            state = Path(state_dir)
            state.mkdir(parents=True, exist_ok=True)
            if worktree_root:
                Path(worktree_root).mkdir(parents=True, exist_ok=True)
            if merge_queue:
                Path(merge_queue).mkdir(parents=True, exist_ok=True)

            stop = {"value": False}

            def _handle(signum, frame):  # noqa: ARG001
                stop["value"] = True

            signal.signal(signal.SIGTERM, _handle)
            signal.signal(signal.SIGINT, _handle)

            if "--reconciliation-only" in args:
                receipt = {
                    "schema": "fake-supervisor-reconciliation@1",
                    "state_prefix": state_prefix,
                    "task_shard_index": shard_index,
                    "task_shard_count": shard_count,
                    "pid": os.getpid(),
                }
                (state / f"{state_prefix}_reconciliation.json").write_text(
                    json.dumps(receipt, sort_keys=True),
                    encoding="utf-8",
                )
                raise SystemExit(0)

            status_path = state / f"{state_prefix}_supervisor_status.json"
            task_state_path = state / f"{state_prefix}_task_state.json"
            payload = {
                "status": "running",
                "daemon_pid": os.getpid(),
                "active_task_id": None,
                "lane": state_prefix,
                "task_shard_index": int(shard_index or 0),
                "task_shard_count": int(shard_count or 1),
                "todo_path": todo_path,
                "argv": args,
                "provider": os.environ.get(
                    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER", ""
                ),
            }
            status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            task_state_path.write_text(
                json.dumps(
                    {
                        "heartbeat_at": time.time(),
                        "ready_count": 0,
                        "waiting_count": 0,
                        "blocked_count": 0,
                        "completed_count": 0,
                        "active_task_id": None,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            # Capture launch argv for tests (no secrets).
            (state / f"{state_prefix}_launch_argv.json").write_text(
                json.dumps(
                    {
                        "argv": args,
                        "env_provider": os.environ.get(
                            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER", ""
                        ),
                        "has_objective_refill": "--objective-refill-scan" in args,
                        "has_codebase_refill": "--codebase-refill-scan" in args,
                        "protected_paths": [
                            args[i + 1]
                            for i, token in enumerate(args)
                            if token == "--implementation-protected-path"
                            and i + 1 < len(args)
                        ],
                        "submodule_paths": [
                            args[i + 1]
                            for i, token in enumerate(args)
                            if token == "--worktree-submodule-path"
                            and i + 1 < len(args)
                        ],
                        "task_shard_count": shard_count,
                        "task_shard_index": shard_index,
                        "merge_queue_dir": merge_queue,
                        "worktree_root": worktree_root,
                        "state_dir": state_dir,
                        "state_prefix": state_prefix,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            while not stop["value"]:
                time.sleep(0.05)
            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _python_with_duckdb() -> str:
    """Prefer the active interpreter; fall back to python3 on PATH."""
    import shutil
    import sys

    candidates = [sys.executable]
    which = shutil.which("python3")
    if which:
        candidates.append(which)
    which_py = shutil.which("python")
    if which_py:
        candidates.append(which_py)
    for candidate in candidates:
        try:
            probe = subprocess.run(
                [candidate, "-c", "import duckdb"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    return sys.executable


def _control_env(
    *,
    repo: Path,
    state_root: Path,
    fake_bin: Path,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    # Never leak operator secrets into child argv via accidental expansion.
    for key in list(env):
        upper = key.upper()
        if any(token in upper for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            env.pop(key, None)
    env.update(
        {
            # Keep the real HOME so user-site DuckDB remains importable; isolate
            # runtime exclusively via STATE_ROOT / REPO_ROOT overrides.
            "IPFS_KIT_VFS_ASSURANCE_REPO_ROOT": str(repo),
            "IPFS_KIT_VFS_ASSURANCE_STATE_ROOT": str(state_root),
            "IPFS_KIT_VFS_ASSURANCE_BRANCH": TARGET_BRANCH,
            "IPFS_KIT_VFS_ASSURANCE_ALLOW_DIRTY_CHECKOUT": "1",
            "IPFS_KIT_VFS_ASSURANCE_SKIP_PROVIDER_PREFLIGHT": "1",
            "IPFS_KIT_VFS_ASSURANCE_SKIP_OBJECTIVE_PROJECT": "1",
            "IPFS_KIT_VFS_ASSURANCE_SKIP_RECONCILIATION": "1",
            "IPFS_KIT_VFS_ASSURANCE_SUPERVISOR_BIN": str(fake_bin),
            "IPFS_KIT_VFS_ASSURANCE_VERIFY_SECONDS": "15",
            "IPFS_KIT_VFS_ASSURANCE_STOP_SECONDS": "10",
            "IPFS_KIT_VFS_ASSURANCE_VERIFY_SUPERVISOR_ONLY": "1",
            # Always pin a DuckDB-capable interpreter; ambient agent env may point
            # at a python that lacks the extension.
            "IPFS_ACCELERATE_AGENT_PYTHON": _python_with_duckdb(),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if extra:
        env.update(extra)
    return env


def _run_control(
    command: str,
    *,
    env: Mapping[str, str],
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(CONTROL_SCRIPT), command],
        text=True,
        capture_output=True,
        env=dict(env),
        timeout=timeout,
        check=False,
    )


def _load_json_from_stdout(result: subprocess.CompletedProcess[str]) -> dict:
    text = result.stdout.strip()
    # status/config may be preceded by human lines; parse the first JSON object.
    start = text.find("{")
    if start < 0:
        raise AssertionError(f"no JSON in stdout:\n{result.stdout}\n{result.stderr}")
    decoder = json.JSONDecoder()
    payload, _end = decoder.raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise AssertionError(f"expected JSON object, got {type(payload)!r}")
    return payload


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")


def sys_executable() -> str:
    import sys

    return sys.executable


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(state_root: Path, lane: str) -> int | None:
    path = state_root / "runtime" / f"{lane}_supervisor.pid"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return int(text)


def _lane_argv(state_root: Path, lane: str) -> dict:
    path = state_root / "state" / lane / f"{lane}_launch_argv.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def control_harness(tmp_path: Path):
    repo = _init_temp_repo(tmp_path / "repo")
    state_root = tmp_path / "state-root"
    state_root.mkdir()
    fake_bin = _write_fake_supervisor(tmp_path / "bin" / "fake_supervisor")
    env = _control_env(repo=repo, state_root=state_root, fake_bin=fake_bin)
    yield {
        "repo": repo,
        "state_root": state_root,
        "fake_bin": fake_bin,
        "env": env,
    }
    # Best-effort cleanup of any leftover fake supervisors from this state root.
    for lane in ("vfs_grok", "vfs_codex"):
        pid = _read_pid(state_root, lane)
        if pid and _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Static contract (script source) — no processes required
# ---------------------------------------------------------------------------


def test_control_script_is_executable_and_present() -> None:
    assert CONTROL_SCRIPT.is_file()
    assert CONTROL_SCRIPT.stat().st_mode & stat.S_IXUSR


def test_static_contract_deterministic_shards_and_one_refill_owner() -> None:
    text = CONTROL_SCRIPT.read_text(encoding="utf-8")
    assert 'readonly TASK_SHARD_COUNT=2' in text
    assert 'readonly GROK_SHARD_INDEX=0' in text
    assert 'readonly CODEX_SHARD_INDEX=1' in text
    assert 'readonly REFILL_OWNER_LANE="${GROK_LANE}"' in text
    assert text.count('"--objective-refill-scan"') == 1
    assert text.count('"--codebase-refill-scan"') == 1
    # Codex branch must never gain refill authority.
    provider_block = text.split('if [[ "${provider}" == "grok-build" ]]; then', 1)[1]
    grok_block, codex_block = provider_block.split("\n  else\n", 1)
    assert "--objective-refill-scan" in grok_block
    assert "--codebase-refill-scan" in grok_block
    assert "--objective-refill-scan" not in codex_block
    assert "--codebase-refill-scan" not in codex_block
    assert "--auto-commit-generated-dirty" not in codex_block


def test_static_contract_protected_paths_submodules_timeouts() -> None:
    text = CONTROL_SCRIPT.read_text(encoding="utf-8")
    for path in PROTECTED_PATHS:
        assert path in text
    for sub in SUBMODULE_PATHS:
        assert f'"--worktree-submodule-path" "{sub}"' in text
    for token in (
        '"--max-task-attempts" "3"',
        '"--implementation-retry-budget" "3"',
        '"--validation-retry-budget" "3"',
        '"--merge-retry-budget" "3"',
        '"--implementation-timeout" "3600"',
        '"--implementation-max-timeout" "7200"',
        '"--stale-seconds" "1800"',
        '"--merge-queue-dir"',
    ):
        assert token in text


def test_static_contract_no_secret_argv_patterns() -> None:
    text = CONTROL_SCRIPT.read_text(encoding="utf-8")
    # Provider env may name binary paths/models, never raw secret values.
    assert "OPENAI_API_KEY=" not in text
    assert "XAI_API_KEY=" not in text
    assert "Authorization:" not in text
    for name in SECRET_DENYLIST:
        assert f'"{name}=' not in text
    assert "assert_no_secrets_in_argv" in text
    assert "recover_stale_pid" in text
    assert "lane_process_is_owned" in text


def test_static_contract_provider_loss_does_not_expand_authority() -> None:
    text = CONTROL_SCRIPT.read_text(encoding="utf-8")
    assert "codex will not inherit refill or shard 0" in text
    assert "grok will not inherit shard 1" in text
    assert "no authority expansion" in text
    assert "provider_preflight 1" in text


# ---------------------------------------------------------------------------
# Behavioral tests with fake processes + temporary repos
# ---------------------------------------------------------------------------


def test_config_reports_isolated_state_shared_merge_and_exact_root(control_harness) -> None:
    result = _run_control("config", env=control_harness["env"])
    assert result.returncode == 0, result.stderr
    config = _load_json_from_stdout(result)
    assert config["schema"] == "ipfs_accelerate_py/vfs-symbolic-assurance-control-config@1"
    assert config["repo_root"] == str(control_harness["repo"])
    assert config["state_root"] == str(control_harness["state_root"])
    assert config["merge_queue_dir"] == str(control_harness["state_root"] / "merge-queue")
    assert config["task_shard_count"] == 2
    assert config["refill_owner_lane"] == "vfs_grok"
    assert config["protected_paths"] == list(PROTECTED_PATHS)
    assert config["submodule_paths"] == list(SUBMODULE_PATHS)
    grok = config["lanes"]["vfs_grok"]
    codex = config["lanes"]["vfs_codex"]
    assert grok["task_shard_index"] == 0
    assert codex["task_shard_index"] == 1
    assert grok["refill_owner"] is True
    assert codex["refill_owner"] is False
    assert grok["state_dir"] != codex["state_dir"]
    assert grok["worktree_root"] != codex["worktree_root"]
    assert grok["state_dir"].endswith("/state/vfs_grok")
    assert codex["worktree_root"].endswith("/worktrees/vfs_codex")
    assert config["bounded_timeouts"]["max_task_attempts"] == 3
    assert config["bounded_timeouts"]["implementation_timeout"] == 3600


def test_idempotent_start_status_stop(control_harness) -> None:
    env = control_harness["env"]
    state_root = control_harness["state_root"]

    first = _run_control("start", env=env)
    assert first.returncode == 0, first.stderr + first.stdout
    status1 = _load_json_from_stdout(first)
    assert status1["mode"] in {"running", "degraded"}
    assert status1["lanes"]["vfs_grok"]["supervisor_alive"] is True
    assert status1["lanes"]["vfs_codex"]["supervisor_alive"] is True
    grok_pid = status1["lanes"]["vfs_grok"]["supervisor_pid"]
    codex_pid = status1["lanes"]["vfs_codex"]["supervisor_pid"]
    assert grok_pid and codex_pid and grok_pid != codex_pid

    # Idempotent start: same owned PIDs, no duplicate launch.
    second = _run_control("start", env=env)
    assert second.returncode == 0, second.stderr + second.stdout
    assert "already running" in second.stdout
    status2 = _load_json_from_stdout(second)
    assert status2["lanes"]["vfs_grok"]["supervisor_pid"] == grok_pid
    assert status2["lanes"]["vfs_codex"]["supervisor_pid"] == codex_pid

    # Status is read-only and stable.
    status_only = _run_control("status", env=env)
    assert status_only.returncode == 0, status_only.stderr
    status3 = _load_json_from_stdout(status_only)
    assert status3["lanes"]["vfs_grok"]["supervisor_pid"] == grok_pid
    assert status3["repo_root"] == str(control_harness["repo"])
    assert status3["merge_queue_dir"] == str(state_root / "merge-queue")
    assert status3["task_shard_count"] == 2
    assert status3["refill_owner_lane"] == "vfs_grok"

    stop1 = _run_control("stop", env=env)
    assert stop1.returncode == 0, stop1.stderr + stop1.stdout
    _wait_until(lambda: not _pid_alive(grok_pid) and not _pid_alive(codex_pid))
    assert _read_pid(state_root, "vfs_grok") is None
    assert _read_pid(state_root, "vfs_codex") is None

    # Idempotent stop with no recorded PIDs.
    stop2 = _run_control("stop", env=env)
    assert stop2.returncode == 0, stop2.stderr + stop2.stdout
    assert "no recorded PID" in stop2.stdout


def test_shards_do_not_duplicate_tasks_and_one_refill_owner(control_harness) -> None:
    env = control_harness["env"]
    state_root = control_harness["state_root"]
    result = _run_control("start", env=env)
    assert result.returncode == 0, result.stderr + result.stdout
    try:
        grok = _lane_argv(state_root, "vfs_grok")
        codex = _lane_argv(state_root, "vfs_codex")
        assert grok["task_shard_count"] == "2"
        assert codex["task_shard_count"] == "2"
        assert grok["task_shard_index"] == "0"
        assert codex["task_shard_index"] == "1"
        assert grok["has_objective_refill"] is True
        assert grok["has_codebase_refill"] is True
        assert codex["has_objective_refill"] is False
        assert codex["has_codebase_refill"] is False
        assert grok["env_provider"] == "grok-build"
        assert codex["env_provider"] == "codex"
        assert grok["merge_queue_dir"] == codex["merge_queue_dir"]
        assert grok["merge_queue_dir"] == str(state_root / "merge-queue")
        assert grok["state_dir"] != codex["state_dir"]
        assert grok["worktree_root"] != codex["worktree_root"]
        assert set(grok["protected_paths"]) == set(PROTECTED_PATHS)
        assert set(codex["protected_paths"]) == set(PROTECTED_PATHS)
        assert set(grok["submodule_paths"]) == set(SUBMODULE_PATHS)
        assert set(codex["submodule_paths"]) == set(SUBMODULE_PATHS)
        # Exact todo path binds ownership to this temporary repository root.
        assert str(control_harness["repo"] / PROTECTED_PATHS[2]) in grok["argv"]
        assert str(control_harness["repo"] / PROTECTED_PATHS[2]) in codex["argv"]
        for payload in (grok, codex):
            joined = " ".join(payload["argv"])
            for secret in SECRET_DENYLIST:
                assert secret not in joined
    finally:
        _run_control("stop", env=env)


def test_stale_pid_recovery_and_pid_ownership(control_harness) -> None:
    env = control_harness["env"]
    state_root = control_harness["state_root"]
    runtime = state_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    # Foreign live process that must never be signaled.
    foreign = subprocess.Popen(
        [sys_executable(), "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        (runtime / "vfs_grok_supervisor.pid").write_text(
            f"{foreign.pid}\n", encoding="utf-8"
        )
        (runtime / "vfs_codex_supervisor.pid").write_text(
            "999999\n", encoding="utf-8"
        )  # dead pid

        # Stop must clear stale records without killing the foreign process.
        stop = _run_control("stop", env=env)
        assert stop.returncode == 0, stop.stderr + stop.stdout
        assert "not a live owned supervisor" in stop.stdout
        assert _pid_alive(foreign.pid)
        assert _read_pid(state_root, "vfs_grok") is None
        assert _read_pid(state_root, "vfs_codex") is None

        # Start recovers stale records and launches owned fakes.
        start = _run_control("start", env=env)
        assert start.returncode == 0, start.stderr + start.stdout
        status = _load_json_from_stdout(start)
        grok_pid = status["lanes"]["vfs_grok"]["supervisor_pid"]
        codex_pid = status["lanes"]["vfs_codex"]["supervisor_pid"]
        assert grok_pid != foreign.pid
        assert codex_pid != foreign.pid
        assert _pid_alive(foreign.pid)
        assert _pid_alive(grok_pid)
        assert _pid_alive(codex_pid)
    finally:
        _run_control("stop", env=env)
        if _pid_alive(foreign.pid):
            foreign.send_signal(signal.SIGTERM)
            foreign.wait(timeout=5)


def test_stop_does_not_kill_unrelated_process(control_harness) -> None:
    env = control_harness["env"]
    state_root = control_harness["state_root"]
    runtime = state_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    victim = subprocess.Popen(
        [sys_executable(), "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        (runtime / "vfs_grok_supervisor.pid").write_text(f"{victim.pid}\n", encoding="utf-8")
        result = _run_control("stop", env=env)
        assert result.returncode == 0, result.stderr
        assert _pid_alive(victim.pid)
        assert "leaving it untouched" in result.stdout or "clearing stale record" in result.stdout
    finally:
        if _pid_alive(victim.pid):
            victim.send_signal(signal.SIGTERM)
            victim.wait(timeout=5)


def test_provider_loss_degrades_without_expanding_authority(control_harness) -> None:
    """When one lane dies, restart recovers it without expanding peer authority."""
    env = control_harness["env"]
    start = _run_control("start", env=env)
    assert start.returncode == 0, start.stderr + start.stdout
    state_root = control_harness["state_root"]
    try:
        grok = _lane_argv(state_root, "vfs_grok")
        codex = _lane_argv(state_root, "vfs_codex")
        assert grok["task_shard_index"] == "0"
        assert codex["task_shard_index"] == "1"
        grok_pid_before = _read_pid(state_root, "vfs_grok")
        # Simulate provider loss: stop codex only by signaling its owned process.
        codex_pid = _read_pid(state_root, "vfs_codex")
        assert codex_pid and _pid_alive(codex_pid)
        os.kill(codex_pid, signal.SIGTERM)
        _wait_until(lambda: not _pid_alive(codex_pid))
        # Restart should recover stale codex and leave grok shard/refill intact.
        restart = _run_control("start", env=env)
        assert restart.returncode == 0, restart.stderr + restart.stdout
        grok2 = _lane_argv(state_root, "vfs_grok")
        codex2 = _lane_argv(state_root, "vfs_codex")
        assert grok2["task_shard_index"] == "0"
        assert codex2["task_shard_index"] == "1"
        assert grok2["has_objective_refill"] is True
        assert codex2["has_objective_refill"] is False
        assert grok2["has_codebase_refill"] is True
        assert codex2["has_codebase_refill"] is False
        # Grok identity remains; codex gets a new owned process.
        grok_pid_after = _read_pid(state_root, "vfs_grok")
        codex_pid_after = _read_pid(state_root, "vfs_codex")
        assert grok_pid_after == grok_pid_before
        assert codex_pid_after and codex_pid_after != codex_pid
        assert _pid_alive(codex_pid_after)
    finally:
        _run_control("stop", env=env)


def test_authenticated_provider_probe_fail_closed_when_both_missing(
    control_harness,
) -> None:
    env = dict(control_harness["env"])
    env["IPFS_KIT_VFS_ASSURANCE_SKIP_PROVIDER_PREFLIGHT"] = "0"
    # Keep core utilities on PATH but hide provider CLIs.
    empty_bin = control_harness["state_root"] / "empty-bin"
    empty_bin.mkdir(exist_ok=True)
    env["PATH"] = f"/usr/bin:/bin:{empty_bin}"
    env.pop("IPFS_ACCELERATE_AGENT_GROK_BIN", None)
    # Point Grok bin at a non-executable path.
    env["IPFS_ACCELERATE_AGENT_GROK_BIN"] = str(
        control_harness["state_root"] / "missing-grok"
    )
    result = _run_control("start", env=env)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "No authenticated providers" in combined or "no authenticated providers" in combined.lower()


def test_provider_probe_writes_receipt_without_secrets(control_harness) -> None:
    env = dict(control_harness["env"])
    # Skip path still writes a probe receipt.
    result = _run_control("start", env=env)
    assert result.returncode == 0, result.stderr
    try:
        probe_path = control_harness["state_root"] / "projection" / "provider_probe.json"
        assert probe_path.is_file()
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        dumped = json.dumps(probe)
        for secret in SECRET_DENYLIST:
            assert secret not in dumped
        assert "grok" in probe and "codex" in probe
    finally:
        _run_control("stop", env=env)


def test_exact_repository_root_required(tmp_path: Path, control_harness) -> None:
    env = dict(control_harness["env"])
    missing = tmp_path / "not-a-repo"
    missing.mkdir()
    env["IPFS_KIT_VFS_ASSURANCE_REPO_ROOT"] = str(missing)
    result = _run_control("config", env=env)
    assert result.returncode != 0
    assert "repository root" in (result.stderr + result.stdout).lower()


def test_require_protected_paths_at_repo_root(control_harness) -> None:
    repo = control_harness["repo"]
    # Remove a protected path and ensure start fails closed.
    target = repo / PROTECTED_PATHS[0]
    target.unlink()
    result = _run_control("start", env=control_harness["env"])
    assert result.returncode != 0
    assert "missing required path" in (result.stderr + result.stdout).lower()


def test_bounded_timeout_flags_present_on_both_lanes(control_harness) -> None:
    env = control_harness["env"]
    state_root = control_harness["state_root"]
    result = _run_control("start", env=env)
    assert result.returncode == 0, result.stderr
    try:
        for lane in ("vfs_grok", "vfs_codex"):
            argv = _lane_argv(state_root, lane)["argv"]
            as_map = {}
            i = 0
            while i < len(argv):
                if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    as_map[argv[i]] = argv[i + 1]
                    i += 2
                else:
                    i += 1
            assert as_map["--max-task-attempts"] == "3"
            assert as_map["--implementation-retry-budget"] == "3"
            assert as_map["--validation-retry-budget"] == "3"
            assert as_map["--merge-retry-budget"] == "3"
            assert as_map["--implementation-timeout"] == "3600"
            assert as_map["--implementation-max-timeout"] == "7200"
            assert as_map["--task-shard-count"] == "2"
            assert as_map["--stale-seconds"] == "1800"
    finally:
        _run_control("stop", env=env)


def test_status_schema_includes_lane_isolation_metadata(control_harness) -> None:
    env = control_harness["env"]
    start = _run_control("start", env=env)
    assert start.returncode == 0, start.stderr
    try:
        status = _load_json_from_stdout(_run_control("status", env=env))
        assert status["schema"] == "ipfs_accelerate_py/vfs-symbolic-assurance-control-status@1"
        assert status["lanes"]["vfs_grok"]["task_shard_index"] == 0
        assert status["lanes"]["vfs_codex"]["task_shard_index"] == 1
        assert status["lanes"]["vfs_grok"]["refill_owner"] is True
        assert status["lanes"]["vfs_codex"]["refill_owner"] is False
        assert status["lanes"]["vfs_grok"]["worktree_root"] != status["lanes"]["vfs_codex"]["worktree_root"]
        assert status["protected_paths"] == list(PROTECTED_PATHS)
    finally:
        _run_control("stop", env=env)


def test_logs_do_not_store_secrets(control_harness) -> None:
    env = dict(control_harness["env"])
    # Plant secret-like env vars; control must not echo them into logs/argv.
    env["OPENAI_API_KEY"] = "sk-test-should-never-appear"
    env["XAI_API_KEY"] = "xai-test-should-never-appear"
    result = _run_control("start", env=env)
    assert result.returncode == 0, result.stderr
    state_root = control_harness["state_root"]
    try:
        for lane in ("vfs_grok", "vfs_codex"):
            log_path = state_root / "logs" / f"{lane}_supervisor.log"
            if log_path.is_file():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                assert "sk-test-should-never-appear" not in text
                assert "xai-test-should-never-appear" not in text
            argv = " ".join(_lane_argv(state_root, lane)["argv"])
            assert "sk-test-should-never-appear" not in argv
            assert "xai-test-should-never-appear" not in argv
        combined = result.stdout + result.stderr
        assert "sk-test-should-never-appear" not in combined
        assert "xai-test-should-never-appear" not in combined
    finally:
        _run_control("stop", env=env)


def test_usage_message_lists_supported_commands() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL_SCRIPT), "not-a-command"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "start" in result.stderr
    assert "status" in result.stderr
    assert "stop" in result.stderr
    assert "preflight" in result.stderr
    assert "config" in result.stderr


def test_existing_generated_dirty_guard_for_grok_only() -> None:
    """Preserve VFS board-repair guard from test_ipfs_kit_vfs_supervisor_control."""
    text = CONTROL_SCRIPT.read_text(encoding="utf-8")
    provider_block = text.split(
        'if [[ "${provider}" == "grok-build" ]]; then',
        1,
    )[1]
    grok_block, codex_block = provider_block.split("\n  else\n", 1)
    assert text.count('"--auto-commit-generated-dirty"') == 1
    assert grok_block.count('"--generated-dirty-path"') == 2
    assert '"--generated-dirty-path" "${OBJECTIVE_ABS}"' in grok_block
    assert '"--generated-dirty-path" "${TODO_ABS}"' in grok_block
    assert '"--generated-dirty-max-paths" "2"' in grok_block
    assert "--auto-commit-generated-dirty" not in codex_block
