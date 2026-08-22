"""APMC-017: Python / CLI / MCP autonomy control-surface conformance."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.autonomy.cli import (
    AUTONOMY_CLI_COMMANDS,
    autonomy_cli_discovery_manifest,
    run_autonomy_cli,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    AUTONOMY_CONTROL_CONFIRMATION_OPERATIONS,
    AUTONOMY_CONTROL_MUTATION_OPERATIONS,
    AUTONOMY_CONTROL_READ_OPERATIONS,
    AutonomyControlOperation,
    SUPERVISOR_AUTONOMY_ADMIN_AUTHORITY,
    SUPERVISOR_AUTONOMY_CANCEL_AUTHORITY,
    SUPERVISOR_AUTONOMY_CONTROL_REQUIREMENT_ID,
    SUPERVISOR_AUTONOMY_ESCALATION_AUTHORITY,
    SUPERVISOR_AUTONOMY_LEVEL_AUTHORITY,
    SUPERVISOR_AUTONOMY_LIFECYCLE_AUTHORITY,
    SUPERVISOR_AUTONOMY_POLICY_AUTHORITY,
    SUPERVISOR_AUTONOMY_READ_AUTHORITY,
    SUPERVISOR_AUTONOMY_REPAIR_AUTHORITY,
    autonomy_control_operations,
    discover_autonomy_control_catalog,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    AutonomyControl,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    native_agent_supervisor_tools as native_tools,
)


class _Args:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _RecordingToolManager:
    def __init__(self) -> None:
        self.definitions: list[dict[str, Any]] = []

    def register_tool(self, **definition: Any) -> None:
        self.definitions.append(definition)


class _Stream:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, data: str) -> int:
        self.chunks.append(data)
        return len(data)

    def text(self) -> str:
        return "".join(self.chunks)


def _controller() -> AutonomyControl:
    return AutonomyControl(
        catalog_revision_provider=lambda: "catalog-rev-autonomy",
        supervisor_revision_provider=lambda: "supervisor-rev-autonomy",
        policy_revision_provider=lambda: "policy-rev-autonomy",
        questions=(
            {
                "question_id": "q-1",
                "disposition": "unresolved",
                "type": "whether_human_choice_is_irreducible",
            },
        ),
        graph={"nodes": [{"id": "q-1"}], "edges": []},
        budget={
            "max_total_model_calls": 4,
            "committed_total_model_calls": 1,
            "remaining_total_model_calls": 3,
            "status": "active",
        },
        experience={"episode_count": 2, "success_count": 1, "failure_count": 1},
        distillation_candidates=({"candidate_id": "distill-1", "status": "candidate"},),
        shadow_results=({"result_id": "shadow-1", "agreed": True},),
    )


def _read_auth() -> list[str]:
    return [SUPERVISOR_AUTONOMY_READ_AUTHORITY]


def _mutation_auth() -> list[str]:
    return [
        SUPERVISOR_AUTONOMY_READ_AUTHORITY,
        SUPERVISOR_AUTONOMY_LIFECYCLE_AUTHORITY,
        SUPERVISOR_AUTONOMY_LEVEL_AUTHORITY,
        SUPERVISOR_AUTONOMY_POLICY_AUTHORITY,
        SUPERVISOR_AUTONOMY_REPAIR_AUTHORITY,
        SUPERVISOR_AUTONOMY_CANCEL_AUTHORITY,
        SUPERVISOR_AUTONOMY_ESCALATION_AUTHORITY,
        SUPERVISOR_AUTONOMY_ADMIN_AUTHORITY,
    ]


def _seed_mutation_targets(controller: AutonomyControl) -> None:
    controller.seed_policy_candidate(
        {
            "candidate_id": "policy-1",
            "status": "candidate",
            "external_authorization_id": "ext-auth-1",
            "shadow_only": True,
        }
    )
    controller.seed_policy_candidate(
        {
            "candidate_id": "policy-0",
            "status": "active",
            "external_authorization_id": "ext-auth-0",
            "shadow_only": False,
        }
    )
    controller.seed_repair({"repair_id": "repair-1", "plan_id": "repair-1", "status": "proposed"})
    controller.seed_action({"action_id": "action-1", "status": "running"})
    controller.deliver_escalation(
        {
            "packet_id": "packet-1",
            "question": "Which bounded option should continue?",
            "options": ["keep_current_safest", "request_authorized_review"],
            "expires_at_ms": 9_999_999_999,
        }
    )


def _issue_confirmation(
    controller: AutonomyControl,
    operation: AutonomyControlOperation,
    target_id: str,
    *,
    option: str = "",
    suffix: str = "",
) -> str:
    cid = f"conf:{operation.value}:{target_id}{suffix}"
    controller.register_external_authorization(
        confirmation_cid=cid,
        operation=operation,
        target_id=target_id,
        option=option,
        actor="operator:test",
        source="operator",
    )
    return cid


def _mutation_params(
    controller: AutonomyControl,
    operation: AutonomyControlOperation,
    *,
    suffix: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    target = {
        AutonomyControlOperation.PAUSE: "autonomy",
        AutonomyControlOperation.RESUME: "autonomy",
        AutonomyControlOperation.SET_LEVEL: "autonomy",
        AutonomyControlOperation.APPROVE_POLICY_CANDIDATE: "policy-1",
        AutonomyControlOperation.REJECT_POLICY_CANDIDATE: "policy-1",
        AutonomyControlOperation.ROLLBACK_POLICY_CANDIDATE: "policy-1",
        AutonomyControlOperation.APPROVE_REPAIR: "repair-1",
        AutonomyControlOperation.CANCEL_ACTION: "action-1",
        AutonomyControlOperation.BIND_ESCALATION_ANSWER: "packet-1",
    }[operation]
    params: dict[str, Any] = {
        "target_id": target,
        "expected_revision": controller.autonomy_revision(),
        "idempotency_key": f"idem:{operation.value}{suffix}",
        "lease_id": "lease:autonomy",
        "fence": 1,
        "expected_effects": [operation.value],
        "actor": "operator:test",
        "source": "operator",
        "dry_run": dry_run,
    }
    if operation is AutonomyControlOperation.SET_LEVEL:
        params["level"] = "recommend"
    if operation is AutonomyControlOperation.APPROVE_POLICY_CANDIDATE:
        params["candidate_id"] = "policy-1"
    if operation is AutonomyControlOperation.REJECT_POLICY_CANDIDATE:
        params["candidate_id"] = "policy-1"
    if operation is AutonomyControlOperation.APPROVE_REPAIR:
        params["repair_id"] = "repair-1"
    if operation is AutonomyControlOperation.CANCEL_ACTION:
        params["action_id"] = "action-1"
    if operation is AutonomyControlOperation.BIND_ESCALATION_ANSWER:
        params["packet_id"] = "packet-1"
        params["option"] = "keep_current_safest"
    if operation in AUTONOMY_CONTROL_CONFIRMATION_OPERATIONS:
        option = str(params.get("option") or "")
        params["confirmation_cid"] = _issue_confirmation(
            controller, operation, target, option=option, suffix=suffix
        )
    return params


def _cli_result(controller: AutonomyControl, operation: str, params: dict[str, Any], auth: list[str]) -> dict[str, Any]:
    passthrough = {
        key: value
        for key, value in params.items()
        if key
        not in {
            "target_id",
            "limit",
            "cursor",
            "expected_revision",
            "idempotency_key",
            "lease_id",
            "fence",
            "expected_effects",
            "confirmation_cid",
            "level",
            "candidate_id",
            "repair_id",
            "action_id",
            "packet_id",
            "option",
            "dry_run",
        }
    }
    args = _Args(
        autonomy_operation=operation,
        authorities_json=json.dumps(auth),
        target_id=params.get("target_id"),
        limit=params.get("limit", 50),
        cursor=params.get("cursor"),
        parameters_json=json.dumps(passthrough) if passthrough else None,
        expected_revision=params.get("expected_revision"),
        idempotency_key=params.get("idempotency_key"),
        lease_id=params.get("lease_id"),
        fence=params.get("fence"),
        expected_effects_json=json.dumps(params.get("expected_effects") or []),
        confirmation_cid=params.get("confirmation_cid"),
        level=params.get("level"),
        candidate_id=params.get("candidate_id"),
        repair_id=params.get("repair_id"),
        action_id=params.get("action_id"),
        packet_id=params.get("packet_id"),
        option=params.get("option"),
        dry_run=bool(params.get("dry_run")),
        output_json=True,
    )
    out = _Stream()
    err = _Stream()
    code = run_autonomy_cli(args, autonomy_control=controller, stdout=out, stderr=err)
    payload = json.loads(out.text() or err.text() or "{}")
    payload["_exit_code"] = code
    return payload


def test_specified_operations_are_present_and_typed() -> None:
    catalog = discover_autonomy_control_catalog()
    ops = {item["operation"] for item in catalog["operations"]}
    expected_reads = {
        "autonomy_capabilities",
        "autonomy_status",
        "autonomy_metrics",
        "autonomy_graph",
        "autonomy_unresolved_questions",
        "autonomy_budget",
        "autonomy_experience_summary",
        "autonomy_route_policy",
        "autonomy_distillation_candidates",
        "autonomy_repair_history",
        "autonomy_escalations",
        "autonomy_shadow_results",
    }
    expected_mutations = {
        "autonomy_pause",
        "autonomy_resume",
        "autonomy_set_level",
        "autonomy_approve_policy_candidate",
        "autonomy_reject_policy_candidate",
        "autonomy_rollback_policy_candidate",
        "autonomy_approve_repair",
        "autonomy_cancel_action",
        "autonomy_bind_escalation_answer",
    }
    assert expected_reads <= ops
    assert expected_mutations <= ops
    assert ops == set(autonomy_control_operations())
    assert {item.value for item in AUTONOMY_CONTROL_READ_OPERATIONS} == expected_reads
    assert {item.value for item in AUTONOMY_CONTROL_MUTATION_OPERATIONS} == expected_mutations
    by_name = {item["operation"]: item for item in catalog["operations"]}
    for name in expected_reads:
        assert by_name[name]["mutating"] is False
        assert by_name[name]["side_effect_free"] is True
        assert by_name[name]["requires_confirmation"] is False
    for name in expected_mutations:
        assert by_name[name]["mutating"] is True
        assert by_name[name]["requires_idempotency"] is True
        assert by_name[name]["requires_lease"] is True
        assert by_name[name]["requires_fence"] is True
        assert by_name[name]["requires_expected_effects"] is True
        assert by_name[name]["supports_dry_run"] is True
    for operation in AUTONOMY_CONTROL_CONFIRMATION_OPERATIONS:
        assert by_name[operation.value]["requires_confirmation"] is True


def test_discovery_populations_agree_across_python_cli_mcp() -> None:
    python_catalog = discover_autonomy_control_catalog()
    cli_manifest = autonomy_cli_discovery_manifest()
    mcp_manifest = native_tools.autonomy_mcp_discovery_manifest()

    assert python_catalog["requirement_id"] == SUPERVISOR_AUTONOMY_CONTROL_REQUIREMENT_ID
    assert cli_manifest["requirement_id"] == SUPERVISOR_AUTONOMY_CONTROL_REQUIREMENT_ID
    assert mcp_manifest["requirement_id"] == SUPERVISOR_AUTONOMY_CONTROL_REQUIREMENT_ID
    py_ops = {item["operation"] for item in python_catalog["operations"]}
    assert py_ops == set(autonomy_control_operations())
    assert set(cli_manifest["operations"]) == py_ops
    assert set(mcp_manifest["operations"]) == py_ops
    assert set(AUTONOMY_CLI_COMMANDS.values()) >= py_ops | {"discover"}
    assert cli_manifest["shells_out"] is False
    assert mcp_manifest["shells_out"] is False
    assert cli_manifest["mints_permission"] is False
    assert mcp_manifest["mints_permission"] is False

    manager = _RecordingToolManager()
    native_tools.register_native_agent_supervisor_autonomy_tools(manager)
    names = {item["name"] for item in manager.definitions}
    assert "agent_supervisor_autonomy" in names
    tool = next(item for item in manager.definitions if item["name"] == "agent_supervisor_autonomy")
    enum_ops = set(tool["input_schema"]["properties"]["operation"]["enum"])
    assert py_ops.issubset(enum_ops)
    assert "discover" in enum_ops
    contract = tool["input_schema"]["x-agent-supervisor-autonomy-contract"]
    assert contract["shells_out"] is False
    assert contract["mints_permission"] is False
    assert contract["dispatch_mode"] == "direct_service"


def test_cold_discovery_does_not_resolve_service_or_start_process() -> None:
    before = native_tools.agent_supervisor_service_resolution_count()
    python_catalog = discover_autonomy_control_catalog()
    cli_manifest = autonomy_cli_discovery_manifest()
    mcp_manifest = native_tools.autonomy_mcp_discovery_manifest()
    manager = _RecordingToolManager()
    native_tools.register_native_agent_supervisor_autonomy_tools(manager)
    after = native_tools.agent_supervisor_service_resolution_count()
    assert after == before
    assert python_catalog["provider_free"] is True
    assert python_catalog["process_free"] is True
    assert cli_manifest["provider_free"] is True
    assert mcp_manifest["provider_free"] is True


def test_python_service_methods_are_canonical() -> None:
    controller = _controller()

    class _Backend:
        registered_operations = frozenset()

        def execute(self, request: Any) -> Any:
            raise NotImplementedError(request)

    service = SupervisorControlService(
        repository_allowlist=("/tmp/repo-autonomy",),
        state_allowlist=("/tmp/state-autonomy",),
        autonomy_control=controller,
        backend=_Backend(),
    )
    assert service.autonomy_control is controller
    discovered = service.autonomy_discover()
    assert discovered["requirement_id"] == SUPERVISOR_AUTONOMY_CONTROL_REQUIREMENT_ID
    for operation in AUTONOMY_CONTROL_READ_OPERATIONS:
        python_direct = controller.execute(operation, authorities=_read_auth())
        via_service = service.autonomy_execute(operation, authorities=_read_auth())
        named = getattr(service, operation.value)(authorities=_read_auth())
        assert python_direct["success"] is True
        assert via_service["operation"] == python_direct["operation"]
        assert named["operation"] == python_direct["operation"]
        assert python_direct["provider_started"] is False
        assert python_direct["database_mutated"] is False


def test_every_read_operation_is_schema_result_error_equivalent() -> None:
    controller = _controller()
    native_tools.set_autonomy_control_service(controller)
    try:
        before = controller.runtime_observation()
        for operation in sorted(AUTONOMY_CONTROL_READ_OPERATIONS, key=lambda item: item.value):
            python_result = controller.execute(operation, authorities=_read_auth())
            assert python_result["success"] is True, (operation, python_result)
            assert python_result["operation"] == operation.value
            assert python_result["catalog_revision"] == "catalog-rev-autonomy"
            assert python_result["completion_authoritative"] is False
            assert python_result["provider_started"] is False
            assert python_result["database_mutated"] is False

            cli_result = _cli_result(controller, operation.value, {}, _read_auth())
            assert cli_result["_exit_code"] == 0, (operation, cli_result)
            assert cli_result["success"] is True
            assert cli_result["operation"] == python_result["operation"]
            assert cli_result["catalog_revision"] == python_result["catalog_revision"]

            mcp_result = asyncio.run(
                native_tools.agent_supervisor_autonomy(
                    operation.value,
                    authorities=_read_auth(),
                )
            )
            assert mcp_result["success"] is True
            assert mcp_result["operation"] == python_result["operation"]
            assert mcp_result["catalog_revision"] == python_result["catalog_revision"]
        after = controller.runtime_observation()
        assert after["provider_start_count"] == before["provider_start_count"] == 0
        assert after["database_write_count"] == before["database_write_count"] == 0
        assert after["revision"] == before["revision"]
        assert after["mutation_count"] == before["mutation_count"]
    finally:
        native_tools.set_autonomy_control_service(None)


def test_reads_cannot_start_providers_or_mutate_state() -> None:
    controller = _controller()
    before = controller.runtime_observation()
    for operation in AUTONOMY_CONTROL_READ_OPERATIONS:
        result = controller.execute(operation, authorities=_read_auth())
        assert result["success"] is True
        assert result["provider_started"] is False
        assert result["database_mutated"] is False
    after = controller.runtime_observation()
    assert after["paused"] is False
    assert after["level"] == before["level"]
    assert after["revision"] == before["revision"]
    assert after["provider_start_count"] == 0
    assert after["database_write_count"] == 0


def test_unauthorized_mutations_fail_closed_on_every_surface() -> None:
    controller = _controller()
    _seed_mutation_targets(controller)
    native_tools.set_autonomy_control_service(controller)
    try:
        for operation in sorted(AUTONOMY_CONTROL_MUTATION_OPERATIONS, key=lambda item: item.value):
            params = _mutation_params(controller, operation, suffix=":denied")
            denied = controller.execute(operation, authorities=_read_auth(), **params)
            assert denied["success"] is False, (operation, denied)
            assert denied["error_code"] in {
                "lifecycle_authority_denied",
                "level_authority_denied",
                "policy_authority_denied",
                "repair_authority_denied",
                "cancel_authority_denied",
                "escalation_authority_denied",
                "admin_denied",
            }
            cli_denied = _cli_result(controller, operation.value, params, _read_auth())
            assert cli_denied["success"] is False
            mcp_denied = asyncio.run(
                native_tools.agent_supervisor_autonomy(
                    operation.value,
                    authorities=_read_auth(),
                    target_id=params.get("target_id"),
                    parameters={
                        k: v for k, v in params.items() if k != "target_id"
                    },
                )
            )
            assert mcp_denied["success"] is False
        after = controller.runtime_observation()
        assert after["mutation_count"] == 0
        assert after["paused"] is False
    finally:
        native_tools.set_autonomy_control_service(None)


def test_lease_fence_and_expected_effects_are_required() -> None:
    controller = _controller()
    _seed_mutation_targets(controller)
    operation = AutonomyControlOperation.PAUSE
    base = {
        "target_id": "autonomy",
        "expected_revision": controller.autonomy_revision(),
        "idempotency_key": "idem:lease",
        "expected_effects": [operation.value],
        "actor": "operator:test",
        "source": "operator",
    }
    missing_lease = controller.execute(
        operation, authorities=_mutation_auth(), fence=1, **base
    )
    assert missing_lease["success"] is False
    assert missing_lease["error_code"] == "lease_required"
    missing_fence = controller.execute(
        operation, authorities=_mutation_auth(), lease_id="lease:1", **base
    )
    assert missing_fence["success"] is False
    assert missing_fence["error_code"] == "fence_required"
    stale = dict(base)
    stale["idempotency_key"] = "idem:stale-fence"
    stale_fence = controller.execute(
        operation,
        authorities=_mutation_auth(),
        lease_id="lease:1",
        fence=0,
        **stale,
    )
    assert stale_fence["success"] is False
    assert stale_fence["error_code"] == "stale_fence"
    no_effects = dict(base)
    no_effects["idempotency_key"] = "idem:no-effects"
    no_effects.pop("expected_effects")
    missing_effects = controller.execute(
        operation,
        authorities=_mutation_auth(),
        lease_id="lease:1",
        fence=1,
        **no_effects,
    )
    assert missing_effects["success"] is False


def test_dry_run_does_not_apply_or_consume_confirmation() -> None:
    controller = _controller()
    _seed_mutation_targets(controller)
    before = controller.runtime_observation()
    params = _mutation_params(
        controller, AutonomyControlOperation.PAUSE, suffix=":dry", dry_run=True
    )
    preview = controller.pause(authorities=_mutation_auth(), **params)
    assert preview["success"] is True
    assert preview["dry_run"] is True
    assert preview["changed"] is False
    assert "dry_run_preview" in preview["reason_codes"]
    after = controller.runtime_observation()
    assert after["paused"] is False
    assert after["revision"] == before["revision"]
    assert after["mutation_count"] == before["mutation_count"]


def test_idempotent_replay_and_cancel_and_level_mutations() -> None:
    controller = _controller()
    _seed_mutation_targets(controller)
    pause_params = _mutation_params(
        controller, AutonomyControlOperation.PAUSE, suffix=":pause"
    )
    pause = controller.pause(authorities=_mutation_auth(), **pause_params)
    assert pause["success"] is True
    assert pause["paused"] is True
    replay = controller.pause(authorities=_mutation_auth(), **pause_params)
    assert replay["success"] is True
    assert "idempotency_replay" in replay["reason_codes"]
    conflict = controller.pause(
        authorities=_mutation_auth(),
        **{**pause_params, "reason": "different-body"},
    )
    assert conflict["success"] is False
    assert conflict["error_code"] == "idempotency_conflict"

    resume = controller.resume(
        authorities=_mutation_auth(),
        **_mutation_params(controller, AutonomyControlOperation.RESUME, suffix=":resume"),
    )
    assert resume["success"] is True
    assert resume["paused"] is False

    level = controller.set_level(
        authorities=_mutation_auth(),
        **_mutation_params(controller, AutonomyControlOperation.SET_LEVEL, suffix=":level"),
    )
    assert level["success"] is True
    assert level["level"] == "recommend"

    cancelled = controller.cancel_action(
        authorities=_mutation_auth(),
        **_mutation_params(controller, AutonomyControlOperation.CANCEL_ACTION, suffix=":cancel"),
    )
    assert cancelled["success"] is True
    assert cancelled["status"] == "cancelled"
    history = controller.status(authorities=_read_auth())
    assert history["running_action_count"] == 0


def test_policy_repair_and_escalation_confirmation_binding() -> None:
    controller = _controller()
    _seed_mutation_targets(controller)
    approved = controller.approve_policy_candidate(
        authorities=_mutation_auth(),
        **_mutation_params(
            controller,
            AutonomyControlOperation.APPROVE_POLICY_CANDIDATE,
            suffix=":approve",
        ),
    )
    assert approved["success"] is True, approved
    assert approved["status"] == "approved"
    assert approved["audit"]["confirmation_cid"]

    replay_params = _mutation_params(
        controller,
        AutonomyControlOperation.APPROVE_POLICY_CANDIDATE,
        suffix=":approve-replay",
    )
    replay_params["confirmation_cid"] = approved["audit"]["confirmation_cid"]
    replay_params["expected_revision"] = controller.autonomy_revision()
    replay = controller.approve_policy_candidate(
        authorities=_mutation_auth(), **replay_params
    )
    assert replay["success"] is False
    assert replay["error_code"] == "confirmation_replay"

    repair = controller.approve_repair(
        authorities=_mutation_auth(),
        **_mutation_params(
            controller, AutonomyControlOperation.APPROVE_REPAIR, suffix=":repair"
        ),
    )
    assert repair["success"] is True
    assert repair["authorizes_merge"] is False

    bound = controller.bind_escalation_answer(
        authorities=_mutation_auth(),
        **_mutation_params(
            controller,
            AutonomyControlOperation.BIND_ESCALATION_ANSWER,
            suffix=":bind",
        ),
    )
    assert bound["success"] is True, bound
    assert bound["answer_bound"] is True
    assert bound["option"] == "keep_current_safest"
    listed = controller.escalations(authorities=_read_auth())
    assert listed["items"][0]["answer_bound"] is True


def test_adapters_cannot_mint_permission_or_confirmation() -> None:
    controller = _controller()
    _seed_mutation_targets(controller)
    native_tools.set_autonomy_control_service(controller)
    try:
        minted = controller.execute(
            AutonomyControlOperation.PAUSE,
            authorities=_mutation_auth(),
            target_id="autonomy",
            expected_revision=controller.autonomy_revision(),
            idempotency_key="idem:adapter",
            lease_id="lease:1",
            fence=1,
            expected_effects=["autonomy_pause"],
            source="adapter",
        )
        assert minted["success"] is False
        assert minted["error_code"] == "self_authorization_denied"

        try:
            controller.register_external_authorization(
                confirmation_cid="conf:adapter",
                operation=AutonomyControlOperation.APPROVE_REPAIR,
                target_id="repair-1",
                source="adapter",
            )
        except Exception as exc:
            assert "self_authorization" in str(exc) or "mint" in str(exc).lower()
        else:
            raise AssertionError("adapter source must not mint confirmation")

        default_cli = _cli_result(
            controller,
            AutonomyControlOperation.PAUSE.value,
            {
                "target_id": "autonomy",
                "expected_revision": controller.autonomy_revision(),
                "idempotency_key": "idem:cli-default",
                "lease_id": "lease:1",
                "fence": 1,
                "expected_effects": ["autonomy_pause"],
            },
            _read_auth(),
        )
        assert default_cli["success"] is False
    finally:
        native_tools.set_autonomy_control_service(None)


def test_mcp_calls_the_service_directly_and_never_shells_out() -> None:
    source_path = Path(inspect.getsourcefile(native_tools) or "")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"system", "popen", "spawn", "check_output", "Popen", "subprocess"}
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            called.add(node.attr)
        if isinstance(node, ast.Name):
            called.add(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    assert not forbidden.intersection(called)
    assert "subprocess" not in source
    assert "os.system" not in source
    assert "shell=True" not in source
    autonomy_source = inspect.getsource(native_tools.agent_supervisor_autonomy)
    assert "controller.execute" in autonomy_source or ".execute(" in autonomy_source

    controller = _controller()
    native_tools.set_autonomy_control_service(controller)
    try:
        result = asyncio.run(
            native_tools.agent_supervisor_autonomy(
                AutonomyControlOperation.STATUS.value,
                authorities=_read_auth(),
            )
        )
        assert result["success"] is True
        assert result["operation"] == "autonomy_status"
    finally:
        native_tools.set_autonomy_control_service(None)

    cli_path = Path(inspect.getsourcefile(run_autonomy_cli) or "")
    assert cli_path.exists()
    cli_source = cli_path.read_text(encoding="utf-8")
    assert "subprocess" not in cli_source
    assert "os.system" not in cli_source


def test_lifecycle_mutations_share_guardrails_across_transports() -> None:
    controller = _controller()
    _seed_mutation_targets(controller)
    native_tools.set_autonomy_control_service(controller)
    try:
        for operation in (
            AutonomyControlOperation.PAUSE,
            AutonomyControlOperation.SET_LEVEL,
            AutonomyControlOperation.CANCEL_ACTION,
        ):
            params = _mutation_params(
                controller, operation, suffix=f":parity-{operation.value}"
            )
            python_result = controller.execute(
                operation, authorities=_mutation_auth(), **params
            )
            assert python_result["success"] is True, (operation, python_result)
            cli_result = _cli_result(
                controller, operation.value, params, _mutation_auth()
            )
            assert cli_result["success"] is True
            assert "idempotency_replay" in cli_result["reason_codes"]
            mcp_params = dict(params)
            mcp_params["idempotency_key"] = f"idem:{operation.value}:mcp"
            mcp_params["expected_revision"] = controller.autonomy_revision()
            mcp_result = asyncio.run(
                native_tools.agent_supervisor_autonomy(
                    operation.value,
                    authorities=_mutation_auth(),
                    target_id=params.get("target_id"),
                    lifecycle=True,
                    level=True,
                    cancel=True,
                    parameters={k: v for k, v in mcp_params.items() if k != "target_id"},
                )
            )
            assert mcp_result["success"] is True, (operation, mcp_result)
            assert mcp_result["audit"]["lease_id"] == "lease:autonomy"
    finally:
        native_tools.set_autonomy_control_service(None)


def test_python_cli_mcp_mutation_parity_for_pause() -> None:
    controller = _controller()
    native_tools.set_autonomy_control_service(controller)
    try:
        params = _mutation_params(
            controller, AutonomyControlOperation.PAUSE, suffix=":parity"
        )
        python_result = controller.pause(authorities=_mutation_auth(), **params)
        assert python_result["success"] is True
        replay_cli = _cli_result(
            controller,
            AutonomyControlOperation.PAUSE.value,
            params,
            _mutation_auth(),
        )
        assert replay_cli["success"] is True
        assert "idempotency_replay" in replay_cli["reason_codes"]

        mcp_params = dict(params)
        mcp_params["idempotency_key"] = "idem:autonomy_pause:mcp"
        mcp_params["expected_revision"] = controller.autonomy_revision()
        mcp_result = asyncio.run(
            native_tools.agent_supervisor_autonomy(
                AutonomyControlOperation.PAUSE.value,
                authorities=_mutation_auth(),
                target_id="autonomy",
                lifecycle=True,
                parameters={k: v for k, v in mcp_params.items() if k != "target_id"},
            )
        )
        assert mcp_result["success"] is True, mcp_result
        assert mcp_result["audit"]["operation"] == "autonomy_pause"
        assert mcp_result["audit"]["lease_id"] == "lease:autonomy"
    finally:
        native_tools.set_autonomy_control_service(None)
