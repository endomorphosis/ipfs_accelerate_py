from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_residual_intelligence.py"
CONFIG_PATH = ROOT / "config/agent_supervisor_residual_intelligence_scheduler.json"


def _operator():
    spec = importlib.util.spec_from_file_location("vrif_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RestartBoard:
    def __init__(self, root: Path) -> None:
        self.repo_root = root
        self.config_path = root / "config/restart.json"
        self.taskboard_path = "docs/taskboard.md"
        self.objectives_path = "docs/objectives.md"
        self.plan_path = "docs/plan.md"
        self.validator_path = "scripts/validator.py"

    def path(self, value: str) -> Path:
        return self.repo_root / value


def _run_fixture_git(root: Path, *arguments: str, input_text: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _write_fixture_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _owner_restart_fixture(
    operator: object,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    descendant: bool,
) -> SimpleNamespace:
    monkeypatch.setattr(operator, "ROOT", root)
    _run_fixture_git(root, "init", "-b", "restart-test")
    _run_fixture_git(root, "config", "user.name", "VRIF Test")
    _run_fixture_git(root, "config", "user.email", "vrif@example.invalid")
    (root / ".gitignore").write_text("runtime/\n", encoding="utf-8")
    (root / "planning.txt").write_text("bootstrap planning\n", encoding="utf-8")
    _run_fixture_git(root, "add", ".gitignore", "planning.txt")
    _run_fixture_git(root, "commit", "-m", "planning baseline")
    bootstrap_planning = _run_fixture_git(root, "rev-parse", "HEAD")
    bootstrap_planning_tree = _run_fixture_git(root, "rev-parse", "HEAD^{tree}")

    board = _RestartBoard(root)
    control_files = {
        board.path(board.taskboard_path): "## VRIF-000 Test\n",
        board.path(board.objectives_path): "## VRIF-G000 Test\n",
        board.path(board.plan_path): "# Test plan\n",
        board.path(board.validator_path): "raise SystemExit(0)\n",
    }
    for path, text in control_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    config = {
        "board_namespace": "agent-supervisor-verified-residual-intelligence-foundry-v1",
        "merge_target_branch": "restart-test",
        "source_binding": {
            "accelerator_required_ancestor": bootstrap_planning,
            "accelerator_planning_revision": bootstrap_planning,
            "accelerator_planning_tree": bootstrap_planning_tree,
            "accelerator_required_branch": "restart-test",
            "bootstrap_task_source": "duckdb",
        },
        "database_program": {
            "authority_mode": "quack",
            "task_source_kind": "duckdb",
            "store_id": "runtime/control.duckdb",
            "store_generation": "restart-test-v1",
            "schema_revision": "1",
            "quack_endpoint": "quack:127.0.0.1:41327",
            "endpoint_secret_handle": "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
            "failover_policy": "fail_closed",
        },
        "runtime_paths": {
            "root": "runtime",
            "evidence": "runtime/evidence",
            "quack_owner": "runtime/quack-owner",
        },
        "taskboard_path": board.taskboard_path,
        "objectives_path": board.objectives_path,
        "plan_path": board.plan_path,
        "validator_path": board.validator_path,
    }
    _write_fixture_json(board.config_path, config)
    _run_fixture_git(root, "add", "config", "docs", "scripts")
    _run_fixture_git(root, "commit", "-m", "seal bootstrap")
    bootstrap_head = _run_fixture_git(root, "rev-parse", "HEAD")
    bootstrap_tree = _run_fixture_git(root, "rev-parse", "HEAD^{tree}")
    source_paths = {
        "config": board.config_path,
        "taskboard": board.path(board.taskboard_path),
        "objectives": board.path(board.objectives_path),
        "plan": board.path(board.plan_path),
        "validator": board.path(board.validator_path),
    }
    bootstrap = {
        "schema": operator.BOOTSTRAP_SCHEMA,
        "source_head": bootstrap_head,
        "repository_tree_id": bootstrap_tree,
        "plan_root_cid": "plan:restart-test",
        "source_identities": {
            name: operator._identity(path.read_bytes())
            for name, path in sorted(source_paths.items())
        },
        "database_task_source_receipt": {
            "schema": "ipfs_accelerate_py/agent-supervisor/database-task-source@1",
            "repository_tree_id": bootstrap_tree,
            "plan_root_cid": "plan:restart-test",
            "projection_cid": "projection:restart-test",
            "task_cids": ["task:vrif-000"],
            "task_count": 1,
            "goal_count": 1,
            "plan_count": 1,
        },
    }
    bootstrap["bootstrap_receipt_id"] = operator._identity(bootstrap)
    bootstrap_path = root / "runtime/evidence/bootstrap/bootstrap-materialization.json"
    _write_fixture_json(bootstrap_path, bootstrap)

    if descendant:
        (root / "implementation.txt").write_text(
            "accepted implementation\n", encoding="utf-8"
        )
        _run_fixture_git(root, "add", "implementation.txt")
        _run_fixture_git(root, "commit", "-m", "accepted implementation")
        current_planning = _run_fixture_git(root, "rev-parse", "HEAD")
        current_planning_tree = _run_fixture_git(root, "rev-parse", "HEAD^{tree}")
        config["source_binding"].update(
            {
                "accelerator_required_ancestor": current_planning,
                "accelerator_planning_revision": current_planning,
                "accelerator_planning_tree": current_planning_tree,
            }
        )
        _write_fixture_json(board.config_path, config)
        _run_fixture_git(root, "add", "config/restart.json")
        _run_fixture_git(root, "commit", "-m", "advance source binding")

    return SimpleNamespace(
        board=board,
        config=json.loads(board.config_path.read_text(encoding="utf-8")),
        paths={"bootstrap_receipt": bootstrap_path},
        bootstrap=bootstrap,
        bootstrap_path=bootstrap_path,
    )


def _absent_prior_owner() -> dict[str, object]:
    return {
        "state": "absent",
        "status_identity": "",
        "server_id": "",
        "database_uuid": "",
        "store_id": "",
        "schema_revision": 0,
        "schema_fingerprint": "",
        "generation": 0,
        "fence_epoch": 0,
        "process_birth_id": "",
    }


def _database_verification_for(operator: object, admission: dict[str, object]) -> dict[str, object]:
    authority = admission["database_authority"]
    verification = {
        "schema": operator.OWNER_DATABASE_VERIFICATION_SCHEMA,
        "bootstrap_database_receipt_identity": authority["receipt_identity"],
        "repository_tree_id": admission["bootstrap_source_tree"],
        "source_head": admission["bootstrap_source_head"],
        "plan_root_cid": admission["plan_root_cid"],
        "task_cids": list(authority["task_cids"]),
        "task_count": authority["task_count"],
        "goal_count": authority["goal_count"],
        "plan_count": authority["plan_count"],
    }
    verification["verification_id"] = operator._identity(verification)
    return verification


def _owner_status_payload(*, lifecycle: str = "ready") -> dict[str, object]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/quack-state-server@1",
        "interface": "QuackStateServer@1",
        "lifecycle": lifecycle,
        "identity": {
            "server_id": "server:prior-owner",
            "store_id": "runtime/control.duckdb",
            "database_uuid": "00000000-0000-0000-0000-000000000001",
            "schema_revision": 1,
            "schema_fingerprint": "sha256:" + ("1" * 64),
            "generation": 1,
            "fence_epoch": 1,
            "process_birth_id": "birth:prior-owner",
            "process_birth": {
                "boot_id": "boot:test",
                "parent_pid": 1,
                "pid": 3672880,
                "start_time_ticks": 1,
            },
        },
    }


def test_bold_and_plain_board_metadata_are_equivalent() -> None:
    operator = _operator()
    assert operator._metadata_value("completed") == "completed"
    assert operator._metadata_value("** completed") == "completed"
    goals = operator._goal_blocks("## VRIF-G000 Root\n- **Status:** active\n- **Parent:**\n")
    assert goals == [("VRIF-G000", "Root", {"status": "active", "parent": ""})]


def test_runtime_config_closes_authority_and_training_fallbacks() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    projection = config["initial_projection"]
    assert projection["task_count"] == 33
    assert projection["task_dependency_count"] == 111
    assert projection["goal_count"] == 9
    assert projection["ready_task_ids"] == [
        "VRIF-009",
        "VRIF-010",
        "VRIF-011",
        "VRIF-012",
    ]
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["task_source_kind"] == "duckdb"
    assert config["database_program"]["failover_policy"] == "fail_closed"
    assert config["ducklake_projection_program"]["authority"] is False
    assert config["training_policy"]["training_enabled_at_bootstrap"] is False
    assert config["reconciliation_guardrail_enabled"] is False


def test_owner_command_vocabulary_has_no_raw_sql_surface() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnectionPolicyError,
        validate_quack_owner_command,
    )

    assert validate_quack_owner_command(
        "record_queue_retry", {"task_cid": "task-cid"}
    ) == {"task_cid": "task-cid"}
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(
            "execute_sql", {"sql": "UPDATE tasks SET status='completed'"}
        )
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(
            "record_queue_retry",
            {"task_cid": "task-cid", "sql": "DELETE FROM tasks"},
        )


def test_owner_command_inbox_requires_signed_store_bound_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
        quack_owner_command_signature,
    )

    operator = _operator()
    request_id = "a" * 32
    token = "vrif_test_token_0123456789"
    payload = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000),
        "writer_identity": "supervisor-process:1234",
        "store_id": "data/agent_supervisor/residual_intelligence_foundry/control.duckdb",
        "store_generation": "vrif-v1",
        "command": "record_queue_retry",
        "payload": {"task_cid": "task-cid"},
    }
    payload["signature"] = quack_owner_command_signature(payload, token=token)
    request_path = tmp_path / f"{request_id}.request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    calls: list[tuple[object, str, object, object]] = []

    def execute(
        repository: object,
        command: str,
        command_payload: object,
        **bindings: object,
    ) -> dict[str, object]:
        calls.append((repository, command, command_payload, bindings))
        return {"schema": "test-result@1", "changed": True}

    repository = object()
    monkeypatch.setattr(database_task_source, "execute_quack_owner_command", execute)
    operator._process_owner_commands(
        repository,
        tmp_path,
        token=token,
        expected_store_id=payload["store_id"],
        expected_store_generation=payload["store_generation"],
    )
    assert calls == [
        (
            repository,
            "record_queue_retry",
            {"task_cid": "task-cid"},
            {
                "request_id": request_id,
                "store_id": payload["store_id"],
                "store_generation": "vrif-v1",
            },
        )
    ]
    result = json.loads((tmp_path / f"{request_id}.done.json").read_text())
    assert result["schema"] == QUACK_OWNER_COMMAND_RESPONSE_SCHEMA
    assert result["ok"] is True
    assert result["result"] == {"schema": "test-result@1", "changed": True}


def test_owner_command_inbox_rejects_raw_sql_envelopes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source

    operator = _operator()
    request_id = "b" * 32
    request_path = tmp_path / f"{request_id}.request.json"
    request_path.write_text(
        json.dumps(
            {
                "sql": "UPDATE tasks SET status = 'completed'",
                "parameters": None,
            }
        ),
        encoding="utf-8",
    )

    def execute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("raw SQL must never reach the typed dispatcher")

    monkeypatch.setattr(database_task_source, "execute_quack_owner_command", execute)
    operator._process_owner_commands(
        object(),
        tmp_path,
        token="vrif_test_token_0123456789",
        expected_store_id="store",
        expected_store_generation="generation",
    )
    result = json.loads((tmp_path / f"{request_id}.done.json").read_text())
    assert result["ok"] is False
    assert result["error_code"] == "owner_error"
    assert result["error_message"] == "typed owner command rejected"


def test_state_credential_process_denies_same_uid_environment_probe() -> None:
    script = """
import subprocess
import sys
from ipfs_accelerate_py.agent_supervisor.runtime.process_security import harden_state_authority_process
assert harden_state_authority_process() is True
probe = subprocess.run(
    [sys.executable, '-c', "import os; open(f'/proc/{os.getppid()}/environ', 'rb').read()"],
    capture_output=True,
    text=True,
)
print(probe.returncode)
print('PermissionError' in probe.stderr)
"""
    environment = dict(os.environ)
    environment["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = "isolated_test_token_123456"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == ["1", "True"]


def test_real_launch_token_vault_unlink_is_fail_closed(tmp_path: Path) -> None:
    operator = _operator()
    token_path = tmp_path / "owner.quack-token"
    token_path.write_text("opaque-test-token", encoding="utf-8")
    token_path.chmod(0o600)
    operator._unlink_token_vault(token_path)
    assert not token_path.exists()

    target = tmp_path / "unsafe-target"
    target.write_text("must-remain", encoding="utf-8")
    link = tmp_path / "unsafe.quack-token"
    link.symlink_to(target)
    with pytest.raises(operator.OperatorError, match="unsafe Quack token vault"):
        operator._unlink_token_vault(link)
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "must-remain"


def test_owner_restart_admits_exact_bootstrap_and_binds_new_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    fixture = _owner_restart_fixture(
        operator,
        monkeypatch,
        tmp_path,
        descendant=False,
    )
    admission = operator._owner_restart_admission(
        fixture.board,
        fixture.config,
        fixture.paths,
    )
    assert admission["schema"] == operator.OWNER_RESTART_ADMISSION_SCHEMA
    assert admission["mode"] == "exact_bootstrap"
    assert admission["bootstrap_source_head"] == admission["current_source_head"]
    assert admission["bootstrap_source_tree"] == admission["current_source_tree"]

    identity = SimpleNamespace(
        server_id="server:restart-test",
        store_id=fixture.config["database_program"]["store_id"],
        database_uuid="00000000-0000-0000-0000-000000000001",
        schema_revision=1,
        schema_fingerprint="sha256:" + ("1" * 64),
        generation=2,
        fence_epoch=2,
        process_birth_id="birth:restart-test",
    )
    receipt = operator._owner_restart_receipt(
        admission,
        identity,
        expected_store_id=identity.store_id,
        prior_owner=_absent_prior_owner(),
        database_verification=_database_verification_for(operator, admission),
    )
    assert receipt["schema"] == operator.OWNER_RESTART_RECEIPT_SCHEMA
    assert receipt["state_owner"]["database_uuid"] == identity.database_uuid
    assert receipt["state_owner"]["generation"] == 2
    assert receipt["state_owner"]["fence_epoch"] == 2
    body = dict(receipt)
    receipt_id = body.pop("receipt_id")
    assert receipt_id == operator._identity(body)
    assert "token" not in json.dumps(receipt).lower()


def test_owner_restart_prior_status_admits_dead_ready_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    status_path = tmp_path / "quack-state-server.status.json"
    _write_fixture_json(status_path, _owner_status_payload())
    status_path.chmod(0o600)
    monkeypatch.setattr(operator, "_owner_liveness", lambda _payload: "dead")
    prior = operator._owner_restart_prior_status(status_path)
    assert prior["state"] == "dead"
    assert prior["lifecycle"] == "ready"
    assert prior["generation"] == 1
    assert prior["server_id"] == "server:prior-owner"


def test_owner_restart_prior_status_rejects_live_ready_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    status_path = tmp_path / "quack-state-server.status.json"
    _write_fixture_json(status_path, _owner_status_payload())
    status_path.chmod(0o600)
    monkeypatch.setattr(operator, "_owner_liveness", lambda _payload: "alive")
    with pytest.raises(operator.OperatorError, match="process-birth liveness"):
        operator._owner_restart_prior_status(status_path)


def test_owner_restart_receipt_advances_dead_prior_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    fixture = _owner_restart_fixture(
        operator,
        monkeypatch,
        tmp_path,
        descendant=True,
    )
    admission = operator._owner_restart_admission(
        fixture.board,
        fixture.config,
        fixture.paths,
    )
    prior = {
        **_absent_prior_owner(),
        "state": "dead",
        "lifecycle": "ready",
        "liveness": "dead",
        "server_id": "server:prior-owner",
        "database_uuid": "00000000-0000-0000-0000-000000000001",
        "store_id": "runtime/control.duckdb",
        "schema_revision": 1,
        "schema_fingerprint": "sha256:" + ("1" * 64),
        "generation": 1,
        "fence_epoch": 1,
        "process_birth_id": "birth:prior-owner",
    }
    identity = SimpleNamespace(
        server_id="server:restart-test",
        store_id="runtime/control.duckdb",
        database_uuid="00000000-0000-0000-0000-000000000001",
        schema_revision=1,
        schema_fingerprint="sha256:" + ("1" * 64),
        generation=2,
        fence_epoch=2,
        process_birth_id="birth:restart-test",
    )
    receipt = operator._owner_restart_receipt(
        admission,
        identity,
        expected_store_id=identity.store_id,
        prior_owner=prior,
        database_verification=_database_verification_for(operator, admission),
    )
    assert receipt["mode"] == "verified_descendant"
    assert receipt["state_owner"]["generation"] == 2
    assert receipt["prior_state_owner"]["server_id"] == "server:prior-owner"
    reused = SimpleNamespace(**{**identity.__dict__, "server_id": prior["server_id"]})
    with pytest.raises(operator.OperatorError, match="server identity was reused"):
        operator._owner_restart_receipt(
            admission,
            reused,
            expected_store_id=identity.store_id,
            prior_owner=prior,
            database_verification=_database_verification_for(operator, admission),
        )


def test_owner_restart_admits_only_monotonic_source_binding_on_descendant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    fixture = _owner_restart_fixture(
        operator,
        monkeypatch,
        tmp_path,
        descendant=True,
    )
    admission = operator._owner_restart_admission(
        fixture.board,
        fixture.config,
        fixture.paths,
    )
    assert admission["mode"] == "verified_descendant"
    assert admission["bootstrap_source_head"] != admission["current_source_head"]
    assert admission["planning_lineage"]["bootstrap_revision"] != (
        admission["planning_lineage"]["current_revision"]
    )


def test_owner_restart_rejects_tampered_bootstrap_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    fixture = _owner_restart_fixture(
        operator,
        monkeypatch,
        tmp_path,
        descendant=True,
    )
    tampered = dict(fixture.bootstrap)
    tampered["plan_root_cid"] = "plan:tampered"
    _write_fixture_json(fixture.bootstrap_path, tampered)
    with pytest.raises(operator.OperatorError, match="receipt identity is invalid"):
        operator._owner_restart_admission(
            fixture.board,
            fixture.config,
            fixture.paths,
        )


def test_owner_restart_rejects_non_descendant_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    fixture = _owner_restart_fixture(
        operator,
        monkeypatch,
        tmp_path,
        descendant=True,
    )
    current_tree = _run_fixture_git(tmp_path, "rev-parse", "HEAD^{tree}")
    unrelated = _run_fixture_git(
        tmp_path,
        "commit-tree",
        current_tree,
        input_text="unrelated root\n",
    )
    forged = dict(fixture.bootstrap)
    forged["source_head"] = unrelated
    forged["repository_tree_id"] = current_tree
    forged["database_task_source_receipt"] = {
        **forged["database_task_source_receipt"],
        "repository_tree_id": current_tree,
    }
    forged.pop("bootstrap_receipt_id", None)
    forged["bootstrap_receipt_id"] = operator._identity(forged)
    _write_fixture_json(fixture.bootstrap_path, forged)
    with pytest.raises(operator.OperatorError, match="source ancestry is not monotonic"):
        operator._owner_restart_admission(
            fixture.board,
            fixture.config,
            fixture.paths,
        )


def test_owner_restart_rejects_database_or_runtime_authority_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    fixture = _owner_restart_fixture(
        operator,
        monkeypatch,
        tmp_path,
        descendant=True,
    )
    changed = dict(fixture.config)
    changed["database_program"] = {
        **changed["database_program"],
        "store_generation": "unauthorized-generation",
    }
    _write_fixture_json(fixture.board.config_path, changed)
    _run_fixture_git(tmp_path, "add", "config/restart.json")
    _run_fixture_git(tmp_path, "commit", "-m", "unauthorized authority delta")
    with pytest.raises(operator.OperatorError, match="outside the admitted"):
        operator._owner_restart_admission(
            fixture.board,
            changed,
            fixture.paths,
        )


@pytest.mark.parametrize("failure_mode", ["nonzero", "exception"])
def test_failed_supervisor_launch_preserves_one_use_token_vault(
    failure_mode: str,
    tmp_path: Path,
) -> None:
    operator = _operator()
    token_path = tmp_path / "owner.quack-token"
    token_path.write_text("opaque-test-token", encoding="utf-8")
    token_path.chmod(0o600)

    def fail(_arguments: object) -> int:
        if failure_mode == "exception":
            raise RuntimeError("launch failed")
        return 7

    if failure_mode == "exception":
        with pytest.raises(RuntimeError, match="launch failed"):
            operator._launch_with_one_use_owner_token(
                fail,
                ["launch"],
                token_path=token_path,
            )
    else:
        assert (
            operator._launch_with_one_use_owner_token(
                fail,
                ["launch"],
                token_path=token_path,
            )
            == 7
        )
    assert token_path.read_text(encoding="utf-8") == "opaque-test-token"


def test_successful_supervisor_launch_unlinks_one_use_token_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = _operator()
    token_path = tmp_path / "owner.quack-token"
    token_path.write_text("opaque-test-token", encoding="utf-8")
    token_path.chmod(0o600)
    observed: list[Path] = []
    original_unlink = operator._unlink_token_vault

    def unlink(path: Path) -> None:
        observed.append(path)
        original_unlink(path)

    monkeypatch.setattr(operator, "_unlink_token_vault", unlink)
    assert (
        operator._launch_with_one_use_owner_token(
            lambda arguments: 0 if arguments == ["launch"] else 9,
            ["launch"],
            token_path=token_path,
        )
        == 0
    )
    assert observed == [token_path]
    assert not token_path.exists()


def test_provider_environment_removes_state_authority_credentials() -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        provider_subprocess_environment,
    )

    cleaned = provider_subprocess_environment(
        {
            "PATH": "/usr/bin",
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "not-for-provider",
            "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN": "not-for-provider",
        }
    )
    assert cleaned == {"PATH": "/usr/bin"}


def test_implementation_provider_receives_scrubbed_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
        launch_process_child,
    )

    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "not-for-implementation-provider"
    )
    monkeypatch.setenv("VRIF_TEST_PROVIDER_API_KEY", "preserved-provider-key")

    class _Daemon:
        @staticmethod
        def _canonical_ref(_task: object) -> str:
            return "task-cid:vrif-test"

    environment = PortalImplementationDaemon._implementation_process_environment(
        _Daemon(),
        SimpleNamespace(task_id="VRIF-TEST"),
        attempt=2,
        checkpoint_dir=tmp_path,
    )
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment
    assert environment["VRIF_TEST_PROVIDER_API_KEY"] == "preserved-provider-key"
    assert environment["IPFS_ACCELERATE_AGENT_TASK_ID"] == "VRIF-TEST"
    child = launch_process_child(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_TOKEN', 'absent'))",
        ],
        cwd=ROOT,
        env=environment,
        replace_env=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = child.communicate(timeout=5)
    assert child.returncode == 0, stderr
    assert stdout.strip() == "absent"


def test_auto_rescue_worktree_command_receives_no_state_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "not-for-auto-rescue-command"
    )

    class _Daemon:
        implementation_timeout = 5

        @staticmethod
        def _record_event(_event: str, _payload: object) -> None:
            return None

    command = (
        f'{sys.executable} -c "import os; '
        "print(os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_TOKEN', 'absent'))\""
    )
    results = PortalImplementationDaemon._run_auto_rescue_materialize_commands(
        _Daemon(),
        workspace_path=tmp_path,
        log_path=tmp_path / "auto-rescue.log",
        commands=(command,),
        task=SimpleNamespace(task_id="VRIF-TEST"),
    )
    assert results[0]["ok"] is True
    assert results[0]["output_tail"].strip() == "absent"
