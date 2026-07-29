from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    AGENT_CLI_EXIT_FAILED,
    AGENT_CLI_EXIT_INVALID,
    COMMAND_OPERATIONS,
    agent_cli_discovery_manifest,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    ControlDiscoveryManifest,
    ControlDiscoveryObservation,
    ControlSurface,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationError,
    OperationRequest,
    OperationResult,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    InMemoryControlStateStore,
    SupervisorControlService,
    capture_control_discovery_runtime_state,
)


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:fixture",
        "tree_id": "tree:abc",
        "objective_id": "ASI-G103",
        "objective_revision": "objective:1",
        "policy_id": "policy:control",
        "policy_revision": "policy:1",
        "caller": "operator:test",
    }


def _request(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    parameters: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> OperationRequest:
    effects = ()
    if operation.mutating:
        effects = (
            ExpectedEffect(
                effect_id=f"{operation.value}:fixture",
                kind=EffectKind.WRITE_STATE,
                resource="supervisor:fixture",
                paths=("supervisor.json",),
                description="Fixture state transition",
            ),
        )
    return OperationRequest(
        operation=operation,
        **_binding(repo_root, state_root),
        parameters=parameters or {},
        expected_effects=effects,
        dry_run=dry_run,
        bounds=ControlBounds(max_items=16, max_paths=16, max_effects=16),
    )


def _service(repo_root: Path, state_root: Path) -> SupervisorControlService:
    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={
            Operation.STATUS: lambda _request: {
                "state": "healthy",
                "phase": "idle",
            },
            Operation.PLAN: lambda _request: {"steps": ["inspect", "validate"]},
        },
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )


def _authorized_mutation_request(
    repo_root: Path,
    state_root: Path,
) -> OperationRequest:
    binding = _binding(repo_root, state_root)
    effect = ExpectedEffect(
        effect_id="pause:fixture",
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource="supervisor:fixture",
        paths=("supervisor.json",),
        description="Pause the fixture supervisor",
    )
    return OperationRequest(
        operation=Operation.PAUSE,
        **binding,
        parameters={
            "target_id": "supervisor:fixture",
            "reason": "guard validation",
            "requested_state": "paused",
        },
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key="cli:pause:guard:1",
            operation=Operation.PAUSE,
            caller=binding["caller"],
            repository_id=binding["repository_id"],
            objective_id=binding["objective_id"],
        ),
        authorization=AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=Operation.PAUSE,
            granted_authority=OperationAuthority.MUTATION,
            **binding,
            lease_id="lease:cli-guard",
            fencing_epoch=7,
            authorized_effect_ids=(effect.effect_id,),
            grant_ids=("grant:cli-guard",),
            evaluated_at_ms=500,
            expires_at_ms=1_500,
        ),
        lease_id="lease:cli-guard",
        fencing_epoch=7,
    )


def _guard_rejection_payload(
    request: OperationRequest, scenario: str
) -> dict[str, Any]:
    payload = request.to_record()
    payload.pop("content_id")
    if scenario == "unauthorized":
        payload.pop("authorization")
    elif scenario == "unscoped_idempotency":
        idempotency = dict(payload["idempotency"])
        idempotency.pop("content_id")
        idempotency["objective_id"] = "objective:outside-request-scope"
        payload["idempotency"] = idempotency
    elif scenario == "unfenced":
        payload.pop("lease_id")
        payload.pop("fencing_epoch")
    elif scenario == "stale_binding":
        payload["tree_id"] = f"{request.tree_id}:stale-request-binding"
    elif scenario == "path_escape":
        parameters = dict(payload["parameters"])
        parameters["target_path"] = "../outside-repository"
        payload["parameters"] = parameters
    elif scenario == "undeclared_effect":
        payload["expected_effects"] = []
    else:
        raise AssertionError(f"unknown guard scenario {scenario}")
    return payload


def _invoke(
    capsys: pytest.CaptureFixture[str],
    service: SupervisorControlService,
    command: str,
    request: OperationRequest,
) -> tuple[int, dict[str, Any]]:
    code = cli.main(
        [
            "agent",
            command,
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    return int(code), json.loads(captured.out)


def test_agent_group_covers_the_closed_operation_vocabulary() -> None:
    assert set(COMMAND_OPERATIONS.values()) == set(Operation)
    assert len(COMMAND_OPERATIONS) == len(Operation)


def test_cli_discovery_binds_the_python_schema_population() -> None:
    python_manifest = ControlDiscoveryManifest(surface=ControlSurface.PYTHON)
    cli_manifest = agent_cli_discovery_manifest()

    assert cli_manifest.surface is ControlSurface.CLI
    assert cli_manifest.operations == python_manifest.operations
    assert dict(cli_manifest.request_schema_ids) == dict(
        python_manifest.request_schema_ids
    )
    assert dict(cli_manifest.result_schema_ids) == dict(
        python_manifest.result_schema_ids
    )
    assert cli_manifest.schema_population_id == python_manifest.schema_population_id


def test_cli_discovery_is_repeatable_and_initializes_no_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service_resolutions = 0
    process_starts = 0
    provider_loads = 0

    def forbidden_factory(_request: OperationRequest) -> SupervisorControlService:
        nonlocal service_resolutions
        service_resolutions += 1
        raise AssertionError("CLI discovery resolved a control service")

    def forbidden_runtime() -> None:
        raise AssertionError("CLI discovery initialized the product runtime")

    def forbidden_process(*_args: Any, **_kwargs: Any) -> None:
        nonlocal process_starts
        process_starts += 1
        raise AssertionError("CLI discovery started a process")

    monkeypatch.setattr(cli, "IPFSAccelerateCLI", forbidden_runtime)
    monkeypatch.setattr(cli.subprocess, "Popen", forbidden_process)
    monkeypatch.setattr(cli, "_load_heavy_imports", forbidden_runtime)
    before = capture_control_discovery_runtime_state(
        service_resolution_count=service_resolutions,
        optional_provider_load_count=provider_loads,
        process_start_count=process_starts,
    )

    first = agent_cli_discovery_manifest()
    assert cli.main(
        ["agent"], agent_service_factory=forbidden_factory
    ) == 1
    capsys.readouterr()
    second = agent_cli_discovery_manifest()
    after = capture_control_discovery_runtime_state(
        service_resolution_count=service_resolutions,
        optional_provider_load_count=provider_loads,
        process_start_count=process_starts,
    )
    observation = ControlDiscoveryObservation(
        surface=ControlSurface.CLI,
        first_manifest=first,
        second_manifest=second,
        before=before,
        after=after,
    )

    assert observation.side_effect_free is True
    assert first.canonical_bytes() == second.canonical_bytes()
    assert service_resolutions == process_starts == provider_loads == 0


def test_cli_read_result_is_exactly_the_python_service_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    request = _request(repo_root, state_root, Operation.STATUS)

    python_record = service.execute(request).to_record()
    code, cli_record = _invoke(capsys, service, "status", request)

    assert code == 0
    assert cli_record == python_record
    assert OperationResult.from_dict(cli_record).result_id == python_record["content_id"]


def test_every_cli_operation_has_exact_python_execution_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    command_by_operation = {
        operation: command for command, operation in COMMAND_OPERATIONS.items()
    }

    for operation in sorted(Operation, key=lambda item: item.value):
        if operation.mutating:
            effect = ExpectedEffect(
                effect_id=f"{operation.value}:parity",
                kind=EffectKind.LIFECYCLE_TRANSITION,
                resource="supervisor:fixture",
                paths=("supervisor.json",),
                description=f"Preview {operation.value}",
            )
            request = OperationRequest(
                operation=operation,
                **_binding(repo_root, state_root),
                parameters={
                    "target_id": "supervisor:fixture",
                    "reason": "exhaustive CLI parity validation",
                },
                expected_effects=(effect,),
                dry_run=True,
            )
        else:
            request = _request(repo_root, state_root, operation)

        python_record = service.execute(request).to_record()
        code, cli_record = _invoke(
            capsys,
            service,
            command_by_operation[operation],
            request,
        )

        expected_exit = (
            0
            if OperationResult.from_dict(python_record).succeeded
            else AGENT_CLI_EXIT_FAILED
        )
        assert code == expected_exit, operation.value
        assert cli_record == python_record, operation.value
        assert OperationResult.from_dict(cli_record).operation is operation


def test_cli_proposal_and_dry_run_use_the_same_result_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    plan = _request(repo_root, state_root, Operation.PLAN)
    pause = _request(
        repo_root,
        state_root,
        Operation.PAUSE,
        parameters={
            "target_id": "supervisor:fixture",
            "reason": "maintenance",
            "requested_state": "paused",
        },
        dry_run=True,
    )

    plan_expected = service.execute(plan).to_record()
    pause_expected = service.execute(pause).to_record()
    assert _invoke(capsys, service, "plan", plan) == (0, plan_expected)
    code, pause_record = _invoke(capsys, service, "pause", pause)

    assert code == 0
    assert pause_record == pause_expected
    assert pause_record["authority"] == "proposal"
    assert pause_record["preview"]["would_change"] is True
    assert pause_record["effects"] == []


def test_cli_direct_mutation_flags_preserve_python_parity_and_idempotency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _authorized_mutation_request(repo_root, state_root)
    direct_dispatches: list[str] = []
    cli_dispatches: list[str] = []

    def handler(dispatches: list[str]):
        def execute(operation_request: OperationRequest) -> dict[str, Any]:
            dispatches.append(operation_request.request_id)
            return {"state": "paused", "target_id": "supervisor:fixture"}

        return execute

    direct_service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={Operation.PAUSE: handler(direct_dispatches)},
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )
    cli_service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={Operation.PAUSE: handler(cli_dispatches)},
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_000,
    )
    expected = direct_service.execute(request).to_record()

    argv = ["agent", "pause"]
    for name, value in _binding(repo_root, state_root).items():
        argv.extend(["--" + name.replace("_", "-"), value])
    argv.extend(
        [
            "--parameters-json",
            json.dumps(dict(request.parameters)),
            "--expected-effects-json",
            json.dumps(
                [effect.to_record() for effect in request.expected_effects]
            ),
            "--idempotency-key",
            request.idempotency_key,
            "--authorization-json",
            request.authorization.to_json(),
            "--lease-id",
            request.lease_id,
            "--fencing-epoch",
            str(request.fencing_epoch),
            "--output-json",
        ]
    )

    assert cli.main(argv, agent_control_service=cli_service) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli.main(argv, agent_control_service=cli_service) == 0
    replay = json.loads(capsys.readouterr().out)

    assert first == replay == expected
    assert direct_dispatches == [request.request_id]
    assert cli_dispatches == [request.request_id]


def test_cli_rejects_ambiguous_roots_and_non_dry_run_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["agent", "status", "--output-json"])
    missing = capsys.readouterr()
    assert code == AGENT_CLI_EXIT_INVALID
    assert json.loads(missing.err)["status"] == "invalid_request"
    assert "explicit target bindings" in missing.err

    binding = _binding(tmp_path / "repo", tmp_path / "state")
    argv = ["agent", "pause"]
    for name, value in binding.items():
        argv.extend(["--" + name.replace("_", "-"), value])
    argv.extend(["--target-id", "supervisor:fixture", "--output-json"])
    code = cli.main(argv)
    unsafe = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_INVALID
    assert "real mutations require a complete" in unsafe.err


def test_cli_rejects_canonical_request_overlays_before_service_resolution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _request(repo_root, state_root, Operation.STATUS)
    factory_calls = 0

    def forbidden_factory(_request: OperationRequest) -> SupervisorControlService:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("ambiguous request resolved a service")

    code = cli.main(
        [
            "agent",
            "status",
            "--request-json",
            request.to_json(),
            "--target-id",
            "ignored-target",
            "--max-items",
            str(ControlBounds().max_items),
        ],
        agent_service_factory=forbidden_factory,
    )
    captured = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_INVALID
    assert captured.out == ""
    assert "cannot be combined" in captured.err
    assert "--target-id" in captured.err
    assert "--max-items" in captured.err
    assert factory_calls == 0


@pytest.mark.parametrize(
    ("scenario", "message_fragment"),
    (
        ("unauthorized", "authorization"),
        ("unscoped_idempotency", "scope does not match"),
        ("unfenced", "lease_id and fencing_epoch"),
        ("stale_binding", "binding does not match"),
        ("path_escape", "repository-relative"),
        ("undeclared_effect", "declare expected effects"),
    ),
)
def test_cli_rejects_every_unsafe_real_mutation_before_service_resolution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
    message_fragment: str,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _authorized_mutation_request(repo_root, state_root)
    payload = _guard_rejection_payload(request, scenario)
    factory_calls = 0

    def forbidden_factory(_request: OperationRequest) -> SupervisorControlService:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("malformed mutation resolved a control service")

    code = cli.main(
        [
            "agent",
            "pause",
            "--request-json",
            json.dumps(payload),
            "--output-json",
        ],
        agent_service_factory=forbidden_factory,
    )
    captured = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_INVALID
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["status"] == "invalid_request"
    assert message_fragment in error["message"]
    assert factory_calls == 0


def test_cli_watch_is_bounded_and_emits_one_canonical_record_per_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    request = _request(
        repo_root,
        state_root,
        Operation.HEALTH,
        parameters={"health_path": "missing-health.json"},
    )

    code = cli.main(
        [
            "agent",
            "health",
            "--request-json",
            request.to_json(),
            "--watch-count",
            "2",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]

    assert code == 4  # both bounded reads consistently report not_found
    assert len(records) == 2
    assert records[0] == records[1]
    assert all(record["schema"].endswith("operation-result@1") for record in records)


def test_global_output_json_is_not_overwritten_by_agent_subparser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    request = _request(repo_root, state_root, Operation.STATUS)

    code = cli.main(
        [
            "--output-json",
            "agent",
            "status",
            "--request-json",
            request.to_json(),
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()

    assert code == 0
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["status"] == "succeeded"


def test_cli_rejects_unbounded_watch_before_dispatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root)
    request = _request(repo_root, state_root, Operation.STATUS)

    code = cli.main(
        [
            "agent",
            "status",
            "--request-json",
            request.to_json(),
            "--watch-count",
            "101",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_INVALID
    assert captured.out == ""
    assert "watch-count" in captured.err


def test_cli_rejects_meaningless_watch_interval_before_service_resolution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _request(repo_root, state_root, Operation.STATUS)
    factory_calls = 0

    def forbidden_factory(_request: OperationRequest) -> SupervisorControlService:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("invalid watch resolved a service")

    code = cli.main(
        [
            "agent",
            "status",
            "--request-json",
            request.to_json(),
            "--watch-interval-ms",
            "1",
        ],
        agent_service_factory=forbidden_factory,
    )
    captured = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_INVALID
    assert captured.out == ""
    assert "requires --watch-count greater than 1" in captured.err
    assert factory_calls == 0


def test_cli_emits_stable_json_for_unexpected_service_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _request(repo_root, state_root, Operation.STATUS)

    class FailingService:
        def execute(self, _request: OperationRequest) -> OperationResult:
            raise RuntimeError("backend credential must not be exposed")

    code = cli.main(
        [
            "agent",
            "status",
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=FailingService(),
    )
    captured = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_FAILED
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error == {
        "message": "agent control operation failed",
        "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
        "status": "internal_error",
    }
    assert "credential" not in captured.err


def test_cli_rejects_a_canonical_result_bound_to_another_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    request = _request(repo_root, state_root, Operation.STATUS)

    class MismatchedService:
        def execute(self, operation_request: OperationRequest) -> OperationResult:
            return OperationResult(
                request_id=operation_request.request_id,
                operation=operation_request.operation,
                authority=operation_request.effective_authority,
                status=OperationStatus.FAILED,
                repository_id=operation_request.repository_id,
                tree_id="tree:another-checkout",
                objective_id=operation_request.objective_id,
                policy_id=operation_request.policy_id,
                caller=operation_request.caller,
                bounds=operation_request.bounds,
                error=OperationError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="result is not bound to this tree",
                ),
            )

    code = cli.main(
        [
            "agent",
            "status",
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=MismatchedService(),
    )
    captured = capsys.readouterr()

    assert code == AGENT_CLI_EXIT_FAILED
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "message": "agent control operation failed",
        "schema": "ipfs_accelerate_py/agent-supervisor/cli-error@1",
        "status": "internal_error",
    }
    assert "another-checkout" not in captured.err
