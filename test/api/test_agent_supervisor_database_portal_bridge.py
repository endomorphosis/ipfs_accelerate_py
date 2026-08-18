"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    SEMANTIC_WRITER_POLICY_ENV,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    parse_args,
    parse_task_text,
    task_declared_output_paths,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    preflight_validation_project_dependencies,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    validation_command_repository_root,
)


def _attempt() -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id="attempt:001",
        claim_id="claim:001",
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        attempt_number=1,
        owner_session_id="session:bridge",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:001",
        committed_phase="claimed",
        status="running",
        started_at_ms=1,
    )


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        goal_cid="goal:inventory",
        plan_cid="plan:lgswf:1",
        revision=11,
        priority="P0",
        dependencies=("task:cid:003",),
        outputs=({"path": "inventory/result.json"},),
        validations=({"argv": ["python3", "-m", "pytest", "focused.py"]},),
        acceptance=({"criterion": "Focused validation passes"},),
        body={
            "objective": "Produce the current authority inventory",
            "completion": "auto",
            "track": "analysis",
            "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
            "write_scope": ["inventory/result.json"],
            "completion_contract": "Focused validation passes",
        },
    )


def _owned_record(owner: str) -> SimpleNamespace:
    record = _record()
    record.outputs = (
        {"path": "ipfs_datasets_py/logic/verification_api.py"},
        {
            "path": (
                "tests/unit/logic/"
                "test_compositional_verification_public_api.py"
            )
        },
    )
    record.validations = (
        {
            "argv": [
                "python -m pytest -q "
                "tests/unit/logic/test_compositional_verification_public_api.py"
            ]
        },
    )
    record.body = {
        **record.body,
        "owning_repository": owner,
        "markdown_metadata": {"owning_repository": owner},
    }
    return record


def test_datasets_authority_marker_reaches_provider_without_state_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        '{"credential":"must-not-propagate"}',
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "secret-token")
    portal = SimpleNamespace(_canonical_ref=lambda task: "task:cid:004")
    task = SimpleNamespace(task_id="LGSWF-004")

    environment = PortalImplementationDaemon._implementation_process_environment(
        portal,
        task,
        attempt=2,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert environment[SEMANTIC_TRUTH_AUTHORITY_ENV] == "ipfs_datasets_py"
    assert environment[SEMANTIC_WRITER_POLICY_ENV] == "reference_only"
    assert "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION" not in environment
    assert "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON" not in environment
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION", "schema-v1")
    ordinary_environment = (
        PortalImplementationDaemon._implementation_process_environment(
            portal,
            task,
            attempt=3,
            checkpoint_dir=tmp_path / "ordinary-checkpoint",
        )
    )
    assert SEMANTIC_TRUTH_AUTHORITY_ENV not in ordinary_environment
    assert SEMANTIC_WRITER_POLICY_ENV not in ordinary_environment


class _TaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return self.record if task_cid == "task:cid:004" else None


class _CompletingPortal:
    def __init__(self, paths: object, task_alias: str) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.closed = False

    def run_once(self) -> dict[str, object]:
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_commit": "a" * 40,
                    "last_merge_returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        self.paths.events.write_text(
            json.dumps(
                {
                    "type": "task_completed",
                    "task_id": self.task_alias,
                    "event_id": "event:complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "task_count": 1,
            "completed_count": 1,
            "active_task_id": self.task_alias,
            "implementation_result": {
                "task_id": self.task_alias,
                "returncode": 0,
                "implementation_commit": "a" * 40,
                # Raw model output must not enter the database receipt.
                "model_response": "private provider payload",
            },
            "merge_reconciliation": [
                {
                    "task_id": self.task_alias,
                    "returncode": 0,
                    "merge_commit": "b" * 40,
                    "provider_payload": "private",
                }
            ],
        }

    def close_event_runtime(self) -> None:
        self.closed = True


def test_bridge_uses_only_attempt_local_projection_and_seals_receipt(
    tmp_path: Path,
) -> None:
    canonical_board = tmp_path / "canonical-board.md"
    canonical_board.write_text(
        "# Canonical\n\n## LGSWF-004 Authority\n\n- Status: ready\n",
        encoding="utf-8",
    )
    original = canonical_board.read_bytes()
    portals: list[_CompletingPortal] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        portal = _CompletingPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )
    provider = bridge.run_provider(_attempt())
    effect = bridge.apply_effect(_attempt(), provider)
    validation = bridge.validate_effect(_attempt(), effect)

    assert provider["schema"] == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
    assert provider["accepted"] is True
    assert provider["provider"] == "PortalImplementationDaemon"
    assert provider["completion_authority"] == "DatabaseImplementationDaemon"
    assert provider["evidence_digest"].startswith("sha256:")
    assert "private provider payload" not in json.dumps(provider)
    assert "provider_payload" not in json.dumps(provider)
    assert effect["status"] == "applied"
    assert validation["outcome"] == "passed"
    assert validation["evidence_digest"] == provider["evidence_digest"]
    assert canonical_board.read_bytes() == original
    assert portals and portals[0].closed is True
    attempt_boards = list((tmp_path / "attempts").glob("*/task-projection.md"))
    assert len(attempt_boards) == 1
    assert "Projection authority: false" in attempt_boards[0].read_text(encoding="utf-8")


def test_bridge_rejects_projection_contract_tampering(tmp_path: Path) -> None:
    class TamperingPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            text = self.paths.task_projection.read_text(encoding="utf-8")
            self.paths.task_projection.write_text(
                text.replace(
                    "- Acceptance: Focused validation passes",
                    "- Acceptance: no validation required",
                ),
                encoding="utf-8",
            )
            return {"implementation_result": {"returncode": 0}}

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: TamperingPortal(paths, alias),
    )
    with pytest.raises(DatabasePortalBridgeError, match="outside its mutable status"):
        bridge.run_provider(_attempt())


def test_bridge_scopes_validation_to_checked_nested_repository(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    nested_repository = repository_root / "ipfs_datasets_py"
    nested_repository.mkdir(parents=True)
    (nested_repository / ".git").write_text(
        "gitdir: ../.git/modules/ipfs_datasets_py\n",
        encoding="utf-8",
    )

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_owned_record("ipfs_datasets_py")),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    provider = bridge.run_provider(_attempt())

    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    projection = projection_path.read_text(encoding="utf-8")
    assert provider["accepted"] is True
    assert (
        "- Validation: cd ipfs_datasets_py && python -m pytest -q "
        "tests/unit/logic/test_compositional_verification_public_api.py"
    ) in projection
    scoped_command = next(
        line.removeprefix("- Validation: ")
        for line in projection.splitlines()
        if line.startswith("- Validation: ")
    )
    assert validation_command_repository_root(scoped_command) == "ipfs_datasets_py"
    assert (
        "- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, "
        "ipfs_datasets_py/tests/unit/logic/"
        "test_compositional_verification_public_api.py"
    ) in projection
    assert "- Outputs: ipfs_datasets_py/logic/verification_api.py" not in projection
    assert "- Validation: 'python -m pytest" not in projection


def test_bridge_projection_preserves_database_identity_through_scoped_preflight(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    nested_repository = repository_root / "ipfs_datasets_py"
    target = "tests/unit/logic/test_compositional_verification_public_api.py"
    target_path = nested_repository / target
    target_path.parent.mkdir(parents=True)
    target_payload = b"def test_public_api():\n    assert True\n"
    target_path.write_bytes(target_payload)
    (nested_repository / ".git").write_text(
        "gitdir: ../.git/modules/ipfs_datasets_py\n",
        encoding="utf-8",
    )
    setup_payload = (
        b"from setuptools import setup\n"
        b"setup(extras_require={'lgcvf-validation': ['pytest']})\n"
    )
    (nested_repository / "setup.py").write_bytes(setup_payload)
    scoped_requirements = ["pytest"]
    scoped_requirements_sha256 = hashlib.sha256(
        json.dumps(
            scoped_requirements,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    projected_validation = f"cd ipfs_datasets_py && python -m pytest -q {target}"
    (nested_repository / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "bridge-identity-fixture"',
                'version = "0.0.0"',
                'requires-python = ">=3.12"',
                'dynamic = ["dependencies"]',
                "",
                "[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]",
                'schema = "ipfs_accelerate_py/agent-supervisor/'
                'scoped-project-dependency-preflight@3"',
                'requires-python = ">=3.12"',
                "authority = { file = \"setup.py\", sha256 = \""
                + hashlib.sha256(setup_payload).hexdigest()
                + "\", extra = \"lgcvf-validation\", "
                "extra-requirements-sha256 = \""
                + scoped_requirements_sha256
                + "\" }",
                "",
                "[[tool.ipfs-accelerate-agent-supervisor."
                "project-dependency-preflight.targets]]",
                f'target = "{target}"',
                'validation-command-sha256 = "'
                + hashlib.sha256(projected_validation.encode("utf-8")).hexdigest()
                + '"',
                'requirements = ["pytest"]',
                'task = { board-namespace = "bridge-authority-v1", '
                'canonical-task-cid = "task:cid:004", declared-output = "'
                f'ipfs_datasets_py/{target}" }}',
                'baseline = { state = "present", sha256 = "'
                + hashlib.sha256(target_payload).hexdigest()
                + '" }',
                "",
            )
        ),
        encoding="utf-8",
    )

    record = _owned_record("ipfs_datasets_py")
    record.body = {**record.body, "board_namespace": "bridge-authority-v1"}
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
        task_header_prefix="## LGSWF-",
    )

    bridge.run_provider(_attempt())
    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    parsed = parse_task_text(
        projection_path.read_text(encoding="utf-8"),
        path=projection_path,
        task_header_prefix="## LGSWF-",
    )

    assert len(parsed) == 1
    task = parsed[0]
    assert task.canonical_task_cid == _attempt().task_cid
    assert task.canonical_task_key.startswith("task/v1/")
    receipt = preflight_validation_project_dependencies(
        repository_root,
        task.validation,
        task_authority={
            "board_namespace": task.board_namespace,
            "canonical_task_cid": task.canonical_task_cid,
            "declared_outputs": list(task_declared_output_paths(task)),
        },
    )
    assert receipt["passed"] is True
    assert (
        receipt["reason"]
        == "approved_validation_environment_satisfies_project_dependencies"
    )


def test_bridge_rejects_task_body_cid_conflicting_with_database_authority(
    tmp_path: Path,
) -> None:
    record = _record()
    record.body = {**record.body, "canonical_task_cid": "task:cid:forged"}
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="task body conflicts with its canonical CID",
    ):
        bridge.run_provider(_attempt())
    assert factory_calls == []


def test_bridge_preserves_root_repository_output_paths(tmp_path: Path) -> None:
    record = _record()
    record.outputs = (
        {"path": "ipfs_accelerate_py/agent_supervisor/runtime.py"},
        {"path": "test/api/test_runtime.py"},
    )
    record.body = {
        **record.body,
        "owning_repository": "ipfs_accelerate_py",
        "markdown_metadata": {"owning_repository": "ipfs_accelerate_py"},
    }
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )

    bridge.run_provider(_attempt())

    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    projection = projection_path.read_text(encoding="utf-8")
    assert (
        "- Outputs: ipfs_accelerate_py/agent_supervisor/runtime.py, "
        "test/api/test_runtime.py"
    ) in projection
    assert "ipfs_accelerate_py/ipfs_accelerate_py" not in projection


@pytest.mark.parametrize(
    "output",
    (
        "/tmp/escape.py",
        "../escape.py",
        "pkg/../../escape.py",
        "./pkg/module.py",
        "pkg//module.py",
        "pkg/one.py,pkg/two.py",
        "pkg\\module.py",
        "C:/escape.py",
    ),
)
def test_bridge_rejects_output_paths_that_cannot_be_scoped_losslessly(
    tmp_path: Path,
    output: str,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    record = _owned_record("ipfs_datasets_py")
    record.outputs = ({"path": output},)
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="task output path identity is unsafe or ambiguous",
    ):
        bridge.run_provider(_attempt())
    assert factory_calls == []


def test_bridge_rejects_ambiguous_output_mapping(tmp_path: Path) -> None:
    record = _record()
    record.outputs = ({"path": "src/one.py", "output": "src/two.py"},)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="task output mapping has ambiguous path identities",
    ):
        bridge.run_provider(_attempt())


def test_bridge_nested_output_projection_binding_is_stable(tmp_path: Path) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_owned_record("ipfs_datasets_py")),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    first = bridge.run_provider(_attempt())
    binding_path = next(
        (tmp_path / "attempts").glob("*/database-attempt-binding.json")
    )
    first_binding = binding_path.read_bytes()
    second = bridge.run_provider(_attempt())

    assert second["binding_id"] == first["binding_id"]
    assert binding_path.read_bytes() == first_binding


def test_bridge_projects_multiple_validations_under_one_repository_transition(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    record = _owned_record("ipfs_datasets_py")
    record.validations = (
        {"argv": ["python -m pytest -q tests/unit/logic/test_public_api.py"]},
        {"argv": ["python -m pytest -q tests/unit/logic/test_differential.py"]},
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    bridge.run_provider(_attempt())

    projection_path = next((tmp_path / "attempts").glob("*/task-projection.md"))
    projection = projection_path.read_text(encoding="utf-8")
    command = next(
        line.removeprefix("- Validation: ")
        for line in projection.splitlines()
        if line.startswith("- Validation: ")
    )
    assert command.count("cd ipfs_datasets_py") == 1
    assert "test_public_api.py && python -m pytest" in command
    assert validation_command_repository_root(command) == "ipfs_datasets_py"


@pytest.mark.parametrize(
    "argv",
    (
        ["python -m pytest -q tests/unit/test_safe.py\n&& rm -rf target"],
        ["python -m pytest -q tests/unit/test_safe.py\x00"],
        [" python -m pytest -q tests/unit/test_safe.py"],
        ["python", 7, "-m", "pytest"],
        [],
    ),
)
def test_bridge_rejects_noncanonical_validation_argv_before_projection(
    tmp_path: Path,
    argv: list[object],
) -> None:
    record = _record()
    record.validations = ({"argv": argv},)
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(DatabasePortalBridgeError, match="validation argv"):
        bridge.run_provider(_attempt())
    assert factory_calls == []
    assert not list((tmp_path / "attempts").glob("*/task-projection.md"))


@pytest.mark.parametrize(
    ("owner", "message"),
    (
        ("../outside", "owning repository metadata is unsafe"),
        ("other_repository", "not a configured worktree submodule"),
    ),
)
def test_bridge_rejects_unsafe_or_unconfigured_owning_repository(
    tmp_path: Path,
    owner: str,
    message: str,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    # Even an initialized nested Git repository is not authority unless it is
    # in the supervisor's configured worktree-submodule allowlist.
    (repository_root / "other_repository" / ".git").mkdir(parents=True)
    factory_calls: list[str] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        factory_calls.append(alias)
        return _CompletingPortal(paths, alias)

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_owned_record(owner)),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    with pytest.raises(DatabasePortalBridgeError, match=message):
        bridge.run_provider(_attempt())
    assert factory_calls == []


def test_bridge_rejects_validation_root_conflicting_with_owner(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "checkout"
    (repository_root / "ipfs_datasets_py" / ".git").mkdir(parents=True)
    record = _owned_record("ipfs_datasets_py")
    record.validations = ({"argv": ["cd other_repository && python -m pytest -q"]},)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: _CompletingPortal(paths, alias),
        repository_root=repository_root,
        worktree_submodule_paths=("ipfs_datasets_py",),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="repository root conflicts with owning repository",
    ):
        bridge.run_provider(_attempt())


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_production_database_daemon_cannot_complete_with_default_noops(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:fail-closed",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:bridge",
                "tasks": [
                    {
                        "task_cid": "task:cid:004",
                        "task_id": "LGSWF-004",
                        "goal_cid": "goal:inventory",
                        "status": "ready",
                        "priority": "P0",
                        "ordinal": 4,
                        "title": "Inventory",
                    }
                ],
            }
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="no provider executor",
        ):
            daemon.run_once()
        task = daemon.task_source.get_task("task:cid:004")
        assert task is not None
        assert task.status != "completed"
        assert (
            daemon.provider_invocation_recorded(
                daemon.list_running_attempts()[0].attempt_id,
                idempotency_key=f"provider:{daemon.list_running_attempts()[0].attempt_id}",
            )
            is None
        )
    finally:
        daemon.close()


def test_quack_mode_refuses_direct_duckdb_execution(tmp_path: Path) -> None:
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="loopback quack:",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            authority_mode="quack",
            task_source_kind="duckdb",
        )


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_configured_production_runner_binds_real_portal_bridge(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--todo-path",
            str(tmp_path / "canonical-board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "lgswf",
            "--worktree-root",
            ".worktrees",
            "--implement",
            "--once",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.require_real_execution is True
        assert daemon.execution_callbacks_bound is True
        assert daemon.markdown_path is None
        assert daemon.markdown_status_write_count == 0
    finally:
        daemon.close()
