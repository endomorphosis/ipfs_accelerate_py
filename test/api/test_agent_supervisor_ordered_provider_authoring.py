"""Fail-closed ordered-provider authoring authority tests."""

from __future__ import annotations

import builtins
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_bootstrap_seal import (
    BOOTSTRAP_SEAL_SCHEMA,
    canonical_json_bytes,
    content_id,
)
from ipfs_accelerate_py.agent_supervisor.runtime.ordered_provider_authoring import (
    AUTHORING_LAUNCH_SCHEMA,
    BOOTSTRAP_AUTHORING_BOARD_ID_ENV,
    BOOTSTRAP_BASELINE_ID_ENV,
    BOOTSTRAP_FOREST_ID_ENV,
    BOOTSTRAP_INVENTORY_ID_ENV,
    BOOTSTRAP_SEAL_ID_ENV,
    BOOTSTRAP_SEAL_PATH_ENV,
    CODEX_MODEL_ENV,
    CODEX_REASONING_EFFORT_ENV,
    CONFIGURED_BOARD_CONFIG_PATH_ENV,
    CONFIGURED_BOARD_LAUNCH_HEAD_ENV,
    CONFIGURED_BOARD_LAUNCH_ID_ENV,
    CONFIGURED_BOARD_LAUNCH_TREE_ENV,
    CONFIGURED_BOARD_NAMESPACE_ENV,
    FALLBACK_PROVIDER_ENV,
    FALLBACK_TRIGGER_ENV,
    GROK_MODEL_ENV,
    PROVIDER_ENV,
    OrderedProviderAuthoringError,
    authoring_provider_invocation_authorized,
    build_authoring_board_projection,
    evaluate_ordered_provider_authoring,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_content_cid,
    canonical_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    parse_task_text,
)

NAMESPACE = "deterministic-contract-repair-test"
TASK_PREFIX = "## DCR-"
TASK_ID = "DCR-010"
TITLE = "Reconcile current deterministic evidence"


def _metadata(*, extra: tuple[tuple[str, str], ...] = ()) -> list[tuple[str, str]]:
    values = [
        ("Status", "todo"),
        ("Completion", "manual"),
        ("Priority", "P0"),
        ("Track", "evidence"),
        ("Depends on", "DCR-004"),
        ("Goal id", "DCR-G020"),
        ("Outputs", "src/current_state.py, test/test_current_state.py"),
        ("Validation", "python3 -m pytest -q test/test_current_state.py"),
        ("Board namespace", NAMESPACE),
        ("Implementation mode", "ordered_provider"),
        ("Runtime model calls", "0"),
        ("Symbolic first", "true"),
        ("LLM context budget bytes", "262144"),
        ("Context budget tokens", "16384"),
        (
            "Provider role",
            "grok-primary-implement, codex-fallback-implement",
        ),
        ("Predicted files", "src/current_state.py, test/test_current_state.py"),
        ("Acceptance", "Current evidence is content-addressed."),
    ]
    return [*values, *extra]


def _write_board(
    path: Path,
    *,
    metadata: list[tuple[str, str]] | None = None,
) -> None:
    lines = [f"## {TASK_ID} {TITLE}", ""]
    for key, value in metadata or _metadata():
        lines.append(f"- {key}: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _seal(repo: Path, board: Path) -> tuple[dict[str, object], dict[str, str]]:
    projection = build_authoring_board_projection(
        taskboard_path=board,
        task_header_prefix=TASK_PREFIX,
        board_namespace=NAMESPACE,
    )
    root_body = {
        "path": "external/ipfs_accelerate",
        "tree": "1" * 40,
    }
    root_record = {**root_body, "root_id": content_id(root_body)}
    forest_body: dict[str, object] = {"roots": [root_record]}
    forest = {**forest_body, "forest_id": content_id(forest_body)}
    config_relative = "config/scheduler.json"
    seal_relative = "config/bootstrap-seal.json"
    board_relative = board.resolve().relative_to(repo.resolve()).as_posix()
    scheduler_config = {
        "board_namespace": NAMESPACE,
        "taskboard_path": board_relative,
        "task_prefix": "DCR-",
        "merge_target_branch": "main",
        "source_binding": {
            "bootstrap_seal_path": seal_relative,
            "accelerator_required_branch": "main",
        },
        "protected_paths": [config_relative, seal_relative, board_relative],
        "provider": {
            "primary_provider_id": "grok_cli",
            "primary_model_id": "grok-4.5",
            "fallback_provider_id": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_trigger": "primary_unavailable_or_quota_exhausted",
            "fallback_reasoning_effort": "high",
            "provider_fallback_for_other_failures": False,
        },
        "execution_policy": {
            "implementation_authoring_mode": "ordered_provider",
            "implementation_provider_role": (
                "grok-primary-implement, codex-fallback-implement"
            ),
            "repair_runtime_mode": "deterministic_only",
            "repair_runtime_model_calls": 0,
            "repair_runtime_llm_calls": 0,
            "implementation_llm_context_budget_bytes": 262144,
            "implementation_context_budget_tokens": 16384,
            "provider_fallback_allowed_only_for_primary_unavailability_or_quota_exhaustion": True,
        },
        "runtime_paths": {"worktrees": "worktree"},
        "worktree_submodule_paths": ["dependency"],
    }
    config_path = repo / config_relative
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_bytes = canonical_json_bytes(scheduler_config) + b"\n"
    config_path.write_bytes(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    inventory_body = {
        "forest_id": forest["forest_id"],
        "controls": [
            {
                "path": config_relative,
                "sha256": config_sha256,
            }
        ],
    }
    inventory = {**inventory_body, "inventory_id": content_id(inventory_body)}
    baseline_body = {
        "forest_id": forest["forest_id"],
        "inventory_id": inventory["inventory_id"],
        "validator_report_id": content_id({"valid": True}),
        "valid": True,
    }
    baseline = {**baseline_body, "baseline_id": content_id(baseline_body)}
    seal_body = {
        "schema": BOOTSTRAP_SEAL_SCHEMA,
        "board_namespace": NAMESPACE,
        "forest": forest,
        "inventory": inventory,
        "baseline": baseline,
        "authoring_board": projection,
    }
    payload = {**seal_body, "seal_id": content_id(seal_body)}
    seal_path = repo / seal_relative
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    launch_body = {
        "schema": AUTHORING_LAUNCH_SCHEMA,
        "board_namespace": NAMESPACE,
        "scheduler_config_path": config_relative,
        "scheduler_config_sha256": config_sha256,
        "seal_id": str(payload["seal_id"]),
        "forest_id": str(forest["forest_id"]),
        "inventory_id": str(inventory["inventory_id"]),
        "baseline_id": str(baseline["baseline_id"]),
        "authoring_board_id": str(projection["authoring_board_id"]),
        "launch_head": "3" * 40,
        "launch_tree": "4" * 40,
    }
    environment = {
        CONFIGURED_BOARD_NAMESPACE_ENV: NAMESPACE,
        CONFIGURED_BOARD_CONFIG_PATH_ENV: config_relative,
        BOOTSTRAP_SEAL_PATH_ENV: seal_relative,
        BOOTSTRAP_SEAL_ID_ENV: str(payload["seal_id"]),
        BOOTSTRAP_FOREST_ID_ENV: str(forest["forest_id"]),
        BOOTSTRAP_INVENTORY_ID_ENV: str(inventory["inventory_id"]),
        BOOTSTRAP_BASELINE_ID_ENV: str(baseline["baseline_id"]),
        BOOTSTRAP_AUTHORING_BOARD_ID_ENV: str(projection["authoring_board_id"]),
        CONFIGURED_BOARD_LAUNCH_HEAD_ENV: launch_body["launch_head"],
        CONFIGURED_BOARD_LAUNCH_TREE_ENV: launch_body["launch_tree"],
        CONFIGURED_BOARD_LAUNCH_ID_ENV: content_id(launch_body),
        PROVIDER_ENV: "grok_cli",
        GROK_MODEL_ENV: "grok-4.5",
        FALLBACK_PROVIDER_ENV: "codex",
        CODEX_MODEL_ENV: "gpt-5.6-terra",
        FALLBACK_TRIGGER_ENV: "primary_unavailable_or_quota_exhausted",
        CODEX_REASONING_EFFORT_ENV: "high",
    }
    return payload, environment


def _evaluate(
    repo: Path,
    board: Path,
    environment: dict[str, str],
    *,
    metadata: dict[str, str] | None = None,
    attempt: int = 1,
    current_git_tree_id: str = "1" * 40,
    primary_unavailability_reason: str = "",
) -> dict[str, object]:
    task_metadata = metadata or {key.lower(): value for key, value in _metadata()}
    workspace = repo / "worktree"
    workspace.mkdir(exist_ok=True)
    bin_dir = repo / "bin"
    bin_dir.mkdir(exist_ok=True)
    grok = bin_dir / "grok"
    codex = bin_dir / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    outputs = [
        item.strip()
        for item in task_metadata.get("outputs", "").split(",")
        if item.strip()
    ]
    task_cid = canonical_task_identity(
        {
            "task_id": TASK_ID,
            "title": TITLE,
            "outputs": outputs,
            "acceptance": task_metadata.get("acceptance", ""),
            "metadata": task_metadata,
        },
        board_namespace=NAMESPACE,
    ).canonical_task_cid
    fallback = [
        str(codex),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        str(workspace.resolve()),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'web_search="disabled"',
        "-",
    ]
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--workspace",
        str(workspace.resolve()),
        "--model",
        "grok-4.5",
        "--max-turns",
        "100000",
        "--mode",
        "agent",
        "--codex-fallback-reasoning-effort",
        "high",
        "--codex-fallback-command-json",
        json.dumps(fallback, separators=(",", ":")),
    ]
    if primary_unavailability_reason:
        command.extend(
            [
                "--primary-unavailability-reason",
                primary_unavailability_reason,
                "--codex-fallback-on-primary-unavailable",
            ]
        )
    else:
        command.extend(
            [
                "--grok-bin",
                str(grok),
                "--grok-failure-receipt-nonce",
                "a" * 64,
            ]
        )
    predicted = [
        item.strip()
        for item in task_metadata.get("predicted files", "").split(",")
        if item.strip()
    ]
    allowed = [
        item.strip()
        for item in task_metadata.get("allowed paths", "").split(",")
        if item.strip()
    ]
    return evaluate_ordered_provider_authoring(
        repo_root=repo,
        taskboard_path=board,
        task_header_prefix=TASK_PREFIX,
        task_id=TASK_ID,
        title=TITLE,
        metadata=task_metadata,
        canonical_task_cid=task_cid,
        current_forest_id="sha256:" + "2" * 64,
        current_git_tree_id=current_git_tree_id,
        workspace_path=workspace,
        provider_command=command,
        runtime_write_scope=tuple(dict.fromkeys([*outputs, *predicted, *allowed])),
        isolated_worktree=True,
        attempt=attempt,
        environment=environment,
    )


def _bind_launch_to_repo(repo: Path, environment: dict[str, str]) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    config_relative = environment[CONFIGURED_BOARD_CONFIG_PATH_ENV]
    config_sha256 = hashlib.sha256((repo / config_relative).read_bytes()).hexdigest()
    launch_body = {
        "schema": AUTHORING_LAUNCH_SCHEMA,
        "board_namespace": NAMESPACE,
        "scheduler_config_path": config_relative,
        "scheduler_config_sha256": config_sha256,
        "seal_id": environment[BOOTSTRAP_SEAL_ID_ENV],
        "forest_id": environment[BOOTSTRAP_FOREST_ID_ENV],
        "inventory_id": environment[BOOTSTRAP_INVENTORY_ID_ENV],
        "baseline_id": environment[BOOTSTRAP_BASELINE_ID_ENV],
        "authoring_board_id": environment[BOOTSTRAP_AUTHORING_BOARD_ID_ENV],
        "launch_head": head,
        "launch_tree": tree,
    }
    environment[CONFIGURED_BOARD_LAUNCH_HEAD_ENV] = head
    environment[CONFIGURED_BOARD_LAUNCH_TREE_ENV] = tree
    environment[CONFIGURED_BOARD_LAUNCH_ID_ENV] = content_id(launch_body)


def test_sealed_authoring_authorizes_proposal_without_repair_authority(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_board(board)
    _payload, environment = _seal(tmp_path, board)

    receipt = _evaluate(tmp_path, board, environment)

    assert receipt["status"] == "authorized"
    assert receipt["provider_authorized"] is True
    assert receipt["provider_route"]["primary_model_id"] == "grok-4.5"
    assert receipt["provider_route"]["fallback_reasoning_effort"] == "high"
    assert authoring_provider_invocation_authorized(receipt)
    assert receipt["authority"] == {
        "proposal_only": True,
        "deterministic_repair": False,
        "runtime_repair": False,
        "planner": False,
        "doctor": False,
        "proof": False,
        "publication": False,
        "completion": False,
    }


def test_sealed_authoring_binds_closed_primary_unavailable_mode(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_board(board)
    _payload, environment = _seal(tmp_path, board)

    receipt = _evaluate(
        tmp_path,
        board,
        environment,
        primary_unavailability_reason="grok_auth_unavailable",
    )

    assert receipt["status"] == "authorized"
    assert receipt["primary_dispatch_mode"] == (
        "codex_fallback_primary_unavailable"
    )
    assert receipt["primary_unavailability_reason"] == "grok_auth_unavailable"
    assert authoring_provider_invocation_authorized(receipt)
    assert receipt["authority"] == {
        "proposal_only": True,
        "deterministic_repair": False,
        "runtime_repair": False,
        "planner": False,
        "doctor": False,
        "proof": False,
        "publication": False,
        "completion": False,
    }

    forged = json.loads(json.dumps(receipt))
    forged["primary_unavailability_reason"] = "grok_provider_unavailable"
    body = {key: value for key, value in forged.items() if key != "receipt_id"}
    forged["receipt_id"] = canonical_content_cid(body)
    assert not authoring_provider_invocation_authorized(forged)


def test_status_updates_preserve_authority_but_contract_drift_does_not(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_board(board)
    _payload, environment = _seal(tmp_path, board)
    board.write_text(
        board.read_text(encoding="utf-8").replace(
            "- Status: todo", "- Status: completed"
        ),
        encoding="utf-8",
    )
    assert _evaluate(tmp_path, board, environment)["status"] == "authorized"

    board.write_text(
        board.read_text(encoding="utf-8").replace(
            "grok-primary-implement, codex-fallback-implement", "codex-implement"
        ),
        encoding="utf-8",
    )
    rejected = _evaluate(tmp_path, board, environment)
    assert rejected["status"] == "rejected"
    assert rejected["reason_code"] == "authoring_board_drift"
    assert not authoring_provider_invocation_authorized(rejected)


@pytest.mark.parametrize(
    ("environment_key", "value"),
    (
        (FALLBACK_TRIGGER_ENV, "primary_failed"),
        (CODEX_REASONING_EFFORT_ENV, "medium"),
        (CODEX_REASONING_EFFORT_ENV, "low"),
        (BOOTSTRAP_FOREST_ID_ENV, "sha256:" + "0" * 64),
    ),
)
def test_route_and_seal_drift_fail_closed(
    tmp_path: Path,
    environment_key: str,
    value: str,
) -> None:
    board = tmp_path / "tasks.md"
    _write_board(board)
    _payload, environment = _seal(tmp_path, board)
    environment[environment_key] = value

    receipt = _evaluate(tmp_path, board, environment)

    assert receipt["status"] == "rejected"
    assert receipt["provider_authorized"] is False
    assert not any(receipt["authority"].values())


def test_alias_collision_and_explicit_identity_override_are_rejected(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    _write_board(
        board,
        metadata=_metadata(extra=(("Provider_role", "codex-implement"),)),
    )
    with pytest.raises(
        OrderedProviderAuthoringError,
        match="repeats metadata key",
    ):
        build_authoring_board_projection(
            taskboard_path=board,
            task_header_prefix=TASK_PREFIX,
            board_namespace=NAMESPACE,
        )

    _write_board(
        board,
        metadata=_metadata(
            extra=(
                ("Canonical task key", "task/v1/" + "0" * 64),
                ("Canonical task cid", canonical_content_cid({"forged": True})),
            )
        ),
    )
    _payload, environment = _seal(tmp_path, board)
    metadata = {key.lower(): value for key, value in _metadata()}
    metadata["canonical task key"] = "task/v1/" + "0" * 64
    metadata["canonical task cid"] = canonical_content_cid({"forged": True})
    receipt = _evaluate(tmp_path, board, environment, metadata=metadata)
    assert receipt["reason_code"] == "explicit_task_identity_override_forbidden"


def test_daemon_uses_authoring_gate_instead_of_synthetic_kernel_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_board(board)
    _payload, environment = _seal(tmp_path, board)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    dependency_source = tmp_path.parent / f"{tmp_path.name}-dependency-source"
    dependency_source.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=dependency_source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=dependency_source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=dependency_source,
        check=True,
    )
    (dependency_source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=dependency_source, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture dependency"],
        cwd=dependency_source,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(dependency_source),
            "dependency",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    _bind_launch_to_repo(tmp_path, environment)
    workspace = tmp_path / "worktree"
    subprocess.run(
        [
            "git",
            "worktree",
            "add",
            "-q",
            "-b",
            "fixture-worktree",
            str(workspace),
            "HEAD",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "dependency",
        ],
        cwd=workspace,
        check=True,
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    tasks = parse_task_text(
        board.read_text(encoding="utf-8"),
        path=board,
        task_header_prefix=TASK_PREFIX,
    )
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.todo_path = board
    daemon.task_header_prefix = TASK_PREFIX
    daemon.pre_implementation_kernel = None
    daemon.use_ephemeral_worktree = True

    current_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    fixture_receipt = _evaluate(
        tmp_path,
        board,
        environment,
        current_git_tree_id=current_tree,
    )
    command = [str(item) for item in fixture_receipt["provider_command"]]
    # This test exercises receipt/gate revalidation, not ambient CLI auth.
    # Keep its sealed Grok-first argv stable while the new direct route is
    # separately covered by focused unavailable-readiness tests.
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_readiness",
        lambda: ("ready", ""),
    )

    original_import = builtins.__import__

    def fail_repair_gate_imports(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.endswith(
            ("pre_implementation_provider_gate", "implementation_disposition")
        ):
            raise ImportError("repair gate deliberately unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_repair_gate_imports)

    decision = daemon._evaluate_pre_implementation_provider_gate(
        task=tasks[0],
        attempt=1,
        worktree_path=workspace,
        command=command,
    )

    assert decision["gate_kind"] == "ordered_provider_authoring"
    assert decision["provider_authorized"] is True
    assert decision["skip_provider"] is False
    assert decision["residual_packet_cid"] == ""
    assert "plan_cid" not in decision["authoring_receipt"]
    assert "doctor_cid" not in decision["authoring_receipt"]
    daemon._assert_current_provider_gate(
        provider_gate=decision,
        task=tasks[0],
        attempt=1,
        worktree_path=workspace,
        command=command,
    )

    dependency_file = workspace / "dependency/module.py"
    dependency_file.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(
        OrderedProviderAuthoringError,
        match="clean provider baseline",
    ):
        daemon._assert_current_provider_gate(
            provider_gate=decision,
            task=tasks[0],
            attempt=1,
            worktree_path=workspace,
            command=command,
        )
    dependency_file.write_text("VALUE = 1\n", encoding="utf-8")

    (tmp_path / "README.md").write_text("advanced merge target\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "advance merge target"], cwd=tmp_path, check=True
    )
    with pytest.raises(
        OrderedProviderAuthoringError,
        match="no longer matches the current merge target",
    ):
        daemon._assert_current_provider_gate(
            provider_gate=decision,
            task=tasks[0],
            attempt=1,
            worktree_path=workspace,
            command=command,
        )

    repair_task = replace(
        tasks[0],
        metadata={
            **tasks[0].metadata,
            "implementation mode": "deterministic_only",
        },
    )
    repair_decision = daemon._evaluate_pre_implementation_provider_gate(
        task=repair_task,
        attempt=1,
        worktree_path=workspace,
        command=command,
    )
    assert repair_decision["provider_authorized"] is False
    assert repair_decision["reason_code"] == "pre_implementation_gate_import_failed"

    forged_gate = json.loads(json.dumps(decision))
    forged_receipt = forged_gate["authoring_receipt"]
    forged_receipt["attempt"] = 2
    forged_body = {
        key: value for key, value in forged_receipt.items() if key != "receipt_id"
    }
    forged_receipt["receipt_id"] = canonical_content_cid(forged_body)
    assert authoring_provider_invocation_authorized(forged_receipt)
    with pytest.raises(RuntimeError, match="stale authoring authority"):
        daemon._assert_current_provider_gate(
            provider_gate=forged_gate,
            task=tasks[0],
            attempt=1,
            worktree_path=workspace,
            command=command,
        )

    with pytest.raises(RuntimeError, match="unknown gate kind"):
        daemon._assert_current_provider_gate(
            provider_gate={
                "gate_kind": "unknown",
                "provider_authorized": True,
                "skip_provider": False,
                "residual_packet_cid": "forged",
            },
            task=tasks[0],
            attempt=1,
            worktree_path=workspace,
            command=command,
        )

    import inspect

    non_ephemeral_source = inspect.getsource(
        PortalImplementationDaemon._run_implementation
    )
    assert "_assert_current_provider_gate" in non_ephemeral_source


def test_receipt_tamper_cannot_reach_provider(tmp_path: Path) -> None:
    board = tmp_path / "tasks.md"
    _write_board(board)
    _payload, environment = _seal(tmp_path, board)
    receipt = _evaluate(tmp_path, board, environment)
    forged = json.loads(json.dumps(receipt))
    forged["authority"]["completion"] = True
    assert not authoring_provider_invocation_authorized(forged)


def test_daemon_builder_is_ephemeral_and_canonical_grok_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = tmp_path / "tasks.md"
    _write_board(board)
    _payload, environment = _seal(tmp_path, board)
    receipt = _evaluate(tmp_path, board, environment)
    workspace = Path(str(receipt["workspace_path"]))
    command = [str(item) for item in receipt["provider_command"]]
    [task] = parse_task_text(
        board.read_text(encoding="utf-8"),
        path=board,
        task_header_prefix=TASK_PREFIX,
    )
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.worktree_root = tmp_path
    daemon.use_ephemeral_worktree = True
    daemon.manual_completion_authority_revalidation_only = False
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_command",
        lambda **_kwargs: list(command),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_readiness",
        lambda: ("ready", ""),
    )

    receipt_id = str(receipt["receipt_id"])
    with pytest.raises(RuntimeError, match="not dispatch-verified"):
        daemon._require_ordered_provider_proposal_authority(
            workspace,
            task,
            provider_authority_receipt_cid=receipt_id,
        )
    daemon._ordered_provider_proposal_authorities = {
        (task.task_id, str(workspace.resolve())): receipt_id
    }
    daemon._require_ordered_provider_proposal_authority(
        workspace,
        task,
        provider_authority_receipt_cid=receipt_id,
    )

    assert daemon._build_implementation_command(workspace, task=task) == command

    bounded_command = list(command)
    bounded_command[bounded_command.index("--max-turns") + 1] = "1"
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_command",
        lambda **_kwargs: list(bounded_command),
    )
    with pytest.raises(
        OrderedProviderAuthoringError,
        match="max-turns is invalid",
    ):
        daemon._build_implementation_command(workspace, task=task)

    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_command",
        lambda **_kwargs: ["/bin/true"],
    )
    with pytest.raises(
        OrderedProviderAuthoringError,
        match="canonical runner argv",
    ):
        daemon._build_implementation_command(workspace, task=task)

    daemon.use_ephemeral_worktree = False
    with pytest.raises(RuntimeError, match="requires an ephemeral worktree"):
        daemon._build_implementation_command(workspace, task=task)

    for field, value in (
        ("attempt", True),
        ("task_id", 123),
        ("current_git_tree_id", "not-a-tree"),
        ("current_git_tree_id", int("1" * 40)),
        ("provider_hook_count", False),
        ("authorized_write_paths", ["/etc/passwd"]),
        ("authorized_write_paths", ["./src/current_state.py"]),
        ("provider_command", ["/bin/true"]),
    ):
        forged = json.loads(json.dumps(receipt))
        forged[field] = value
        body = {key: item for key, item in forged.items() if key != "receipt_id"}
        forged["receipt_id"] = canonical_content_cid(body)
        assert not authoring_provider_invocation_authorized(forged)

    forged = json.loads(json.dumps(receipt))
    forged["authority"]["completion"] = 0
    body = {key: item for key, item in forged.items() if key != "receipt_id"}
    forged["receipt_id"] = canonical_content_cid(body)
    assert not authoring_provider_invocation_authorized(forged)

    forged = json.loads(json.dumps(receipt))
    forged["provider_route"]["fallback_for_other_failures"] = 0
    body = {key: item for key, item in forged.items() if key != "receipt_id"}
    forged["receipt_id"] = canonical_content_cid(body)
    assert not authoring_provider_invocation_authorized(forged)

    forged = json.loads(json.dumps(receipt))
    max_turns_index = forged["provider_command"].index("--max-turns") + 1
    forged["provider_command"][max_turns_index] = 100000
    body = {key: item for key, item in forged.items() if key != "receipt_id"}
    forged["receipt_id"] = canonical_content_cid(body)
    assert not authoring_provider_invocation_authorized(forged)

    forged = json.loads(json.dumps(receipt))
    forged["completion_authoritative"] = True
    body = {key: item for key, item in forged.items() if key != "receipt_id"}
    forged["receipt_id"] = canonical_content_cid(body)
    assert not authoring_provider_invocation_authorized(forged)
