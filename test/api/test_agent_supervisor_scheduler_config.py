from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    SupervisorSchedulerConfigError,
    expand_supervisor_scheduler_config_args,
    load_supervisor_scheduler_config,
    parse_args,
    supervisor_config_from_args,
)


def _write_profile(
    root: Path,
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "module-a").mkdir()
    (root / "docs" / "tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (root / "docs" / "objectives.md").write_text(
        "# Objectives\n",
        encoding="utf-8",
    )
    (root / "docs" / "plan.md").write_text("# Plan\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "test.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "task_prefix": "## TEST-",
        "board_namespace": "test-supervisor-v1",
        "merge_target_branch": "main",
        "max_lanes": 4,
        "poll_interval_seconds": 3,
        "daemon_interval_seconds": 30,
        "check_interval_seconds": 7,
        "stale_seconds": 600,
        "max_restarts": 2,
        "max_task_attempts": 3,
        "implementation_timeout_seconds": 900,
        "validation_max_workers": 2,
        "worktree_submodule_paths": ["module-a"],
        "protected_paths": [
            "docs/plan.md",
            "docs/tasks.md",
            "docs/objectives.md",
            "config/profile.json",
        ],
        "derived_refill": {"enabled_at_bootstrap": False},
        "doctor": {"mutation_authorized": False},
        "rollout": {"automatic_enabled": False},
    }
    if overrides:
        payload.update(overrides)
    path = root / "config" / "profile.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_manual_task(
    root: Path,
    *,
    status: str,
    completion: str = "manual",
    output: str = "docs/sealed-policy.json",
) -> None:
    (root / "docs" / "tasks.md").write_text(
        (
            "# Tasks\n\n"
            "## TEST-001 Seal reviewed policy artifacts\n\n"
            f"- Status: {status}\n"
            f"- Completion: {completion}\n"
            f"- Outputs: {output}\n"
        ),
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": __import__("os").environ["PATH"],
            "GIT_AUTHOR_NAME": "Scheduler Test",
            "GIT_AUTHOR_EMAIL": "scheduler@example.invalid",
            "GIT_COMMITTER_NAME": "Scheduler Test",
            "GIT_COMMITTER_EMAIL": "scheduler@example.invalid",
        },
    ).stdout.strip()


def _write_test_operator_seal(
    root: Path,
    *,
    policy_revision: str = "1",
) -> str:
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "reviewed base")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    artifact_path = "docs/sealed-policy.json"
    artifact = (root / artifact_path).read_bytes()
    receipt: dict[str, object] = {
        "schema": "example.test.operator_seal@1",
        "interface": "TestOperatorSeal@1",
        "receipt_version": "1",
        "task_id": "TEST-001",
        "board_namespace": "test-supervisor-v1",
        "decision": "sealed",
        "policy_revision": policy_revision,
        "reviewed_base": {
            "commit": commit,
            "tree": tree,
            "git_object_format": "sha1",
            "relation_to_activation_head": "equal_or_ancestor",
        },
        "artifacts": [
            {
                "role": "policy",
                "path": artifact_path,
                "sha256": "sha256:" + hashlib.sha256(artifact).hexdigest(),
                "size_bytes": len(artifact),
            },
        ],
        "operator": {
            "identity": "interactive_user",
            "authority_basis": "interactive_user_delegation",
            "candidate": False,
            "model": False,
            "automatic_controller": False,
        },
        "grant": {
            "type": "policy_activation",
            "allowed_actions": ["activate_policy_revision"],
            "board_namespace": "test-supervisor-v1",
            "policy_revision": policy_revision,
            "delegable": False,
            "mutation_authority": False,
            "completion_authority": False,
            "promotion_authority": False,
            "task_status_authority": False,
            "protected_anchor_write_authority": False,
        },
    }
    body = copy.deepcopy(receipt)
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    (root / "config" / "operator-seal.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(receipt["receipt_id"])


def test_scheduler_config_maps_safe_defaults_and_cli_scalars_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = _write_profile(tmp_path)
    monkeypatch.setattr(supervisor_module, "REPO_ROOT", tmp_path)

    parsed = parse_args(
        [
            "--scheduler-config",
            str(profile_path.relative_to(tmp_path)),
            "--check-interval",
            "9",
            "--state-prefix",
            "operator-override",
            "--once",
        ]
    )

    assert parsed.scheduler_config == profile_path.resolve()
    assert parsed.todo_path == tmp_path / "docs" / "tasks.md"
    assert parsed.objective_path == tmp_path / "docs" / "objectives.md"
    assert parsed.task_prefix == "## TEST-"
    assert parsed.state_prefix == "operator-override"
    assert parsed.check_interval == 9
    assert parsed.daemon_interval == 30
    assert parsed.stale_seconds == 600
    assert parsed.max_restarts == 2
    assert parsed.max_task_attempts == 3
    assert parsed.implementation_timeout == 900
    assert parsed.validation_max_workers == 2
    assert parsed.merge_target_branch == "main"
    assert parsed.worktree_submodule_path == ["module-a"]
    assert parsed.implementation_protected_path == [
        "docs/plan.md",
        "docs/tasks.md",
        "docs/objectives.md",
        "config/profile.json",
    ]
    assert parsed.objective_task_janitor_enabled is False
    assert parsed.objective_reconcile_goal_completion is False
    assert parsed.implement is False
    assert parsed.objective_refill_scan is False
    assert parsed.codebase_refill_scan is False
    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    command = PortalImplementationSupervisor(config)._build_daemon_command()
    assert command[command.index("--validation-max-workers") + 1] == "2"
    assert "--implement" not in command


def test_scheduler_config_activates_protection_only_after_manual_completion(
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(
        tmp_path,
        overrides={
            "protected_after_manual_completion": {
                "TEST-001": [
                    "docs/sealed-policy.json",
                    "config/operator-seal.json",
                ],
            },
            "manual_completion_seals": {
                "TEST-001": {
                    "receipt_path": "config/operator-seal.json",
                    "schema": "example.test.operator_seal@1",
                    "interface": "TestOperatorSeal@1",
                    "policy_revision": "1",
                    "artifact_paths": {
                        "policy": "docs/sealed-policy.json",
                    },
                    "grant_type": "policy_activation",
                    "grant_action": "activate_policy_revision",
                    "grant_claims": {},
                    "reviewed_base_claims": {},
                },
            },
        },
    )
    (tmp_path / "docs" / "sealed-policy.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _write_manual_task(
        tmp_path,
        status="pending",
        output="docs/sealed-policy.json, config/operator-seal.json",
    )
    receipt_id = _write_test_operator_seal(tmp_path)
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_payload["manual_completion_seals"]["TEST-001"][
        "expected_receipt_id"
    ] = receipt_id
    profile_path.write_text(
        json.dumps(profile_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pending = load_supervisor_scheduler_config(
        profile_path,
        repo_root=tmp_path,
    )

    assert pending["activated_protected_task_ids"] == ()
    assert "docs/sealed-policy.json" not in pending["protected_paths"]
    assert pending["verified_manual_completion_seals"] == {}

    _write_manual_task(
        tmp_path,
        status="completed",
        output="docs/sealed-policy.json, config/operator-seal.json",
    )
    completed = load_supervisor_scheduler_config(
        profile_path,
        repo_root=tmp_path,
    )

    assert completed["activated_protected_task_ids"] == ("TEST-001",)
    assert completed["protected_paths"][-2:] == (
        "docs/sealed-policy.json",
        "config/operator-seal.json",
    )
    assert completed["verified_manual_completion_seals"]["TEST-001"].startswith(
        "sha256:"
    )
    first_epoch_id = completed["manual_completion_authority_epoch_id"]

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        authority_epoch_seal_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    seals = completed["manual_completion_seals"]
    pin_a = dict(seals["TEST-001"])
    pin_b = dict(pin_a)
    pin_b["expected_receipt_id"] = "sha256:" + ("f" * 64)
    assert content_identity(
        authority_epoch_seal_projection({"TEST-001": pin_a})
    ) == content_identity(
        authority_epoch_seal_projection({"TEST-001": pin_b})
    )
    # Live profile reload with the same verified seal must keep the epoch.
    pin_reload = load_supervisor_scheduler_config(
        profile_path,
        repo_root=tmp_path,
    )
    assert pin_reload["manual_completion_authority_epoch_id"] == first_epoch_id

    replacement_receipt_id = _write_test_operator_seal(
        tmp_path,
        policy_revision="2",
    )
    resealed_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    resealed_profile["manual_completion_seals"]["TEST-001"][
        "policy_revision"
    ] = "2"
    resealed_profile["manual_completion_seals"]["TEST-001"][
        "expected_receipt_id"
    ] = replacement_receipt_id
    profile_path.write_text(
        json.dumps(resealed_profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resealed = load_supervisor_scheduler_config(
        profile_path,
        repo_root=tmp_path,
    )

    assert resealed["activated_protected_task_ids"] == (
        "TEST-001",
    )
    assert resealed["manual_completion_authority_required_task_ids"] == ()
    assert resealed["verified_manual_completion_seals"] == {
        "TEST-001": replacement_receipt_id
    }
    assert resealed["manual_completion_authority_epoch_id"] != first_epoch_id

    seal_path = tmp_path / "config" / "operator-seal.json"
    original = seal_path.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["grant"]["mutation_authority"] = True
    body = {key: value for key, value in tampered.items() if key != "receipt_id"}
    tampered["receipt_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    seal_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(
            SupervisorSchedulerConfigError,
            match="protected pinned identity",
        ):
            load_supervisor_scheduler_config(
                profile_path,
                repo_root=tmp_path,
            )
    finally:
        seal_path.write_text(original, encoding="utf-8")

    artifact_path = tmp_path / "docs" / "sealed-policy.json"
    original_artifact = artifact_path.read_text(encoding="utf-8")
    artifact_path.write_text('{"candidate_replacement":true}\n', encoding="utf-8")
    rehashed = json.loads(original)
    replacement = artifact_path.read_bytes()
    rehashed["artifacts"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(replacement).hexdigest()
    )
    rehashed["artifacts"][0]["size_bytes"] = len(replacement)
    body = {key: value for key, value in rehashed.items() if key != "receipt_id"}
    rehashed["receipt_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    seal_path.write_text(
        json.dumps(rehashed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(
            SupervisorSchedulerConfigError,
            match="protected pinned identity",
        ):
            load_supervisor_scheduler_config(
                profile_path,
                repo_root=tmp_path,
            )
    finally:
        artifact_path.write_text(original_artifact, encoding="utf-8")
        seal_path.write_text(original, encoding="utf-8")


def test_scheduler_config_rejects_invalid_staged_protection(
    tmp_path: Path,
) -> None:
    unknown_root = tmp_path / "unknown"
    unknown_profile = _write_profile(
        unknown_root,
        overrides={
            "protected_after_manual_completion": {
                "TEST-404": ["docs/sealed-policy.json"],
            },
        },
    )
    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="declared tasks",
    ):
        load_supervisor_scheduler_config(
            unknown_profile,
            repo_root=unknown_root,
        )

    automatic_root = tmp_path / "automatic"
    automatic_profile = _write_profile(
        automatic_root,
        overrides={
            "protected_after_manual_completion": {
                "TEST-001": ["docs/sealed-policy.json"],
            },
        },
    )
    _write_manual_task(automatic_root, status="pending", completion="auto")
    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="manual completion",
    ):
        load_supervisor_scheduler_config(
            automatic_profile,
            repo_root=automatic_root,
        )

    undeclared_root = tmp_path / "undeclared"
    undeclared_profile = _write_profile(
        undeclared_root,
        overrides={
            "protected_after_manual_completion": {
                "TEST-001": ["docs/not-declared.json"],
            },
        },
    )
    _write_manual_task(undeclared_root, status="pending")
    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="declared task outputs",
    ):
        load_supervisor_scheduler_config(
            undeclared_profile,
            repo_root=undeclared_root,
        )

    omitted_root = tmp_path / "omitted"
    omitted_profile = _write_profile(
        omitted_root,
        overrides={
            "protected_after_manual_completion": {
                "TEST-001": ["docs/sealed-policy.json"],
            },
        },
    )
    _write_manual_task(
        omitted_root,
        status="pending",
        output="docs/sealed-policy.json, docs/omitted.json",
    )
    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="protect every declared task output",
    ):
        load_supervisor_scheduler_config(
            omitted_profile,
            repo_root=omitted_root,
        )

    missing_root = tmp_path / "missing"
    missing_profile = _write_profile(
        missing_root,
        overrides={
            "protected_after_manual_completion": {
                "TEST-001": ["docs/missing-policy.json"],
            },
        },
    )
    _write_manual_task(
        missing_root,
        status="completed",
        output="docs/missing-policy.json",
    )
    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="no operator seal configuration",
    ):
        load_supervisor_scheduler_config(
            missing_profile,
            repo_root=missing_root,
        )


def test_scheduler_config_never_enables_effects_but_explicit_operator_can(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = _write_profile(tmp_path)
    monkeypatch.setattr(supervisor_module, "REPO_ROOT", tmp_path)

    safe = parse_args(["--scheduler-config", str(profile_path), "--once"])
    explicit = parse_args(
        ["--scheduler-config", str(profile_path), "--implement", "--once"]
    )

    assert safe.implement is False
    assert safe.objective_refill_scan is False
    assert safe.codebase_refill_scan is False
    assert explicit.implement is True


@pytest.mark.parametrize(
    ("section_name", "switch_name"),
    [
        ("derived_refill", "enabled_at_bootstrap"),
        ("doctor", "enabled_at_bootstrap"),
        ("doctor", "mutation_authorized"),
        ("doctor", "narrow_autonomous_mutation_enabled"),
        ("rollout", "automatic_enabled"),
    ],
)
def test_scheduler_config_rejects_implicit_authority_elevation(
    tmp_path: Path,
    section_name: str,
    switch_name: str,
) -> None:
    profile_path = _write_profile(
        tmp_path,
        overrides={section_name: {switch_name: True}},
    )

    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="cannot be enabled",
    ):
        load_supervisor_scheduler_config(profile_path, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    [
        ("taskboard_path", "../outside.md"),
        ("objectives_path", "/tmp/objectives.md"),
        ("task_prefix", "PDR-"),
        ("max_lanes", True),
        ("poll_interval_seconds", float("inf")),
        ("merge_target_branch", "../main"),
        ("protected_paths", ["docs/"]),
    ],
)
def test_scheduler_config_rejects_unsafe_or_malformed_values(
    tmp_path: Path,
    field_name: str,
    unsafe_value: object,
) -> None:
    profile_path = _write_profile(
        tmp_path,
        overrides={field_name: unsafe_value},
    )

    with pytest.raises(SupervisorSchedulerConfigError):
        load_supervisor_scheduler_config(profile_path, repo_root=tmp_path)


def test_scheduler_config_rejects_duplicate_selector_and_outside_profile(
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text(profile_path.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        with pytest.raises(
            SupervisorSchedulerConfigError,
            match="exactly once",
        ):
            expand_supervisor_scheduler_config_args(
                [
                    "--scheduler-config",
                    str(profile_path),
                    f"--scheduler-config={profile_path}",
                ],
                repo_root=tmp_path,
            )
        with pytest.raises(
            SupervisorSchedulerConfigError,
            match="inside the repository",
        ):
            load_supervisor_scheduler_config(outside, repo_root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)


def test_scheduler_config_rejects_symlink_profile(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    symlink = tmp_path / "config" / "profile-link.json"
    symlink.symlink_to(profile_path.name)

    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="non-symlink",
    ):
        load_supervisor_scheduler_config(symlink, repo_root=tmp_path)


def test_pdr_scheduler_profile_is_directly_consumable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "config"
        / "agent_supervisor_proof_directed_planner_doctor_scheduler.json"
    )

    profile = load_supervisor_scheduler_config(
        config_path,
        repo_root=repo_root,
    )
    expanded, selected = expand_supervisor_scheduler_config_args(
        ["--scheduler-config", str(config_path), "--once"],
        repo_root=repo_root,
    )

    assert selected == config_path.resolve()
    assert profile["task_prefix"] == "## PDR-"
    assert profile["max_lanes"] == 6
    assert "--todo-path" in expanded
    assert "--objective-path" in expanded
    assert "--no-objective-task-janitor" in expanded
    assert "--no-objective-goal-completion-reconcile" in expanded
    assert "--no-objective-goal-migration" in expanded
    assert expanded[-1] == "--once"
