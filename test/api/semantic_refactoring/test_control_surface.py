"""Independent contract tests for SPAR-044 Python/CLI/MCP control surface."""

from __future__ import annotations

import ast
from io import StringIO
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cli import (
    MCP_NEVER_SHELLS as CLI_MCP_NEVER_SHELLS,
    SemanticRefactoringCLI,
    SemanticRefactoringControlBackend,
    mcp_dispatch,
    register_operations as cli_register_operations,
    run_spar_cli,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.service import (
    ALL_OPERATIONS,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    CONTEXT_COMPILER_REMAINS_AUTHORITY,
    CONTROL_CONTRACT_VERSION,
    CONTROL_RECEIPT_INTERFACE,
    CONTROL_REQUEST_INTERFACE,
    CONTROL_RESULT_INTERFACE,
    DECLARED_MUTATION_OPERATIONS,
    DECLARED_OPERATIONS,
    DECLARED_READ_OPERATIONS,
    DECLARED_RECEIPT_FLOOR,
    DECLARED_TERMINAL_KINDS,
    DIAGNOSTIC_REPORT_INTERFACE,
    DIAGNOSTICS_ARE_AUTHORITATIVE,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXISTING_ADAPTER_AUTHORITIES,
    FORBIDDEN_CONTROL_NAMES,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    INDEPENDENT_VALIDATION_REQUIRED,
    MARKDOWN_IS_NOT_COMPLETION,
    MCP_NEVER_SHELLS,
    MISSING_CONTEXT_MESSAGE,
    MISSING_RECEIPT_FLOOR_MESSAGE,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    MUTATION_OPERATIONS,
    MUTATION_SCOPE,
    NEGATIVE_EVIDENCE_RETAINED,
    NETWORK_DENIED,
    NETWORK_DENY,
    PREDECESSOR_TASK_IDS,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    READ_OPERATIONS,
    REQUIRED_RECEIPT_FLOOR,
    REQUIRED_ROLLOUT_CONSUMES_CONTEXT,
    REQUIRED_ROLLOUT_EMITS_RECEIPT_FLOOR,
    SEMANTIC_REFACTORING_SERVICE_INTERFACE,
    SERVICE_CAN_AUTHORIZE_COMPLETION,
    SERVICE_CAN_AUTHORIZE_TRANSITION,
    SERVICE_CAN_CHANGE_MODE,
    SERVICE_CAN_CREATE_AUTHORITY,
    SERVICE_CAN_WEAKEN_VALIDATION,
    SERVICE_IS_NOMINATION_ONLY,
    SERVICE_WRITES_REPOSITORY,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TYPED_TERMINAL_INTERFACE,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    ControlAuthorization,
    ControlBudget,
    ControlReceipt,
    ControlRequest,
    ControlResult,
    ControlStateStore,
    ControlStatus,
    ControlSurfaceError,
    SemanticRefactoringService,
    TerminalKind,
    assert_not_competing_capsule_family,
    compile_control_receipt,
    control_surface_cid_profile,
    control_surface_descriptor,
    decode_canonical_receipt,
    dry_run_control,
    encode_canonical_receipt,
    execute_control,
    provider_free_exports,
    register_operations,
)


ROOT = Path(__file__).resolve().parents[3]
SERVICE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "service.py"
)
CLI_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "cli.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/service.py",
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/cli.py",
    "test/api/semantic_refactoring/test_control_surface.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
FORBIDDEN_IMPORTS = frozenset({"subprocess", "os.system", "pty", "pexpect"})


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _auth() -> ControlAuthorization:
    return ControlAuthorization(
        subject="operator:one",
        permitted=True,
        scopes=(MUTATION_SCOPE,),
    )


def _budget() -> ControlBudget:
    return ControlBudget(max_units=4, requested_units=1)


def _floor() -> dict[str, str]:
    return {name: _cid(name) for name in REQUIRED_RECEIPT_FLOOR}


def _nominate_params(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "write_paths": [
            "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/service.py"
        ],
        "validation_commands": [
            "python3 -m pytest -q test/api/semantic_refactoring/test_control_surface.py"
        ],
        "context_receipt_cid": _cid("context"),
        "evidence_cids": [_cid("evidence")],
    }
    fields.update(overrides)
    return fields


def _mutation(
    operation: str,
    *,
    key: str = "idempotency-1",
    fence: int = 1,
    dry_run: bool = False,
    parameters: dict[str, Any] | None = None,
) -> ControlRequest:
    if parameters is None:
        params = _nominate_params()
        if operation == "spar.request_merge":
            params = {**params, **_floor()}
    else:
        params = parameters
    return ControlRequest(
        operation=operation,
        target_id="spar:one",
        dry_run=dry_run,
        idempotency_key=key,
        authorization=_auth(),
        lease_id="lease:one",
        fencing_epoch=fence,
        budget=_budget(),
        parameters=params,
        tree_id=TREE_ID,
    )


def _service(store: ControlStateStore | None = None) -> SemanticRefactoringService:
    return SemanticRefactoringService(store=store)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".", 1)[0])
                imported.add(node.module)
    return imported


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
                if isinstance(func.value, ast.Name):
                    names.add(f"{func.value.id}.{func.attr}")
    return names


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-044"
    assert GOAL_ID == "SPAR-G071"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert SEMANTIC_REFACTORING_SERVICE_INTERFACE == "SemanticRefactoringService@1"
    assert CONTROL_REQUEST_INTERFACE == "SparControlRequest@1"
    assert CONTROL_RESULT_INTERFACE == "SparControlResult@1"
    assert CONTROL_RECEIPT_INTERFACE == "SparControlReceipt@1"
    assert DIAGNOSTIC_REPORT_INTERFACE == "SparDiagnosticReport@1"
    assert TYPED_TERMINAL_INTERFACE == "TypedTerminal@1"
    assert CONTROL_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("service@1")
    assert PREDECESSOR_TASK_IDS == ("SPAR-035", "SPAR-043")
    assert SERVICE_PATH.is_file()
    assert CLI_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()
    assert DECLARED_READ_OPERATIONS == set(READ_OPERATIONS)
    assert DECLARED_MUTATION_OPERATIONS == set(MUTATION_OPERATIONS)
    assert DECLARED_OPERATIONS == set(ALL_OPERATIONS)
    assert DECLARED_RECEIPT_FLOOR == set(REQUIRED_RECEIPT_FLOOR)
    assert DECLARED_TERMINAL_KINDS >= {
        TerminalKind.MISSING_RECEIPT_FLOOR.value,
        TerminalKind.MISSING_CONTEXT.value,
        TerminalKind.STALE_FENCE.value,
    }


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "operational refactoring authority"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert SERVICE_CAN_AUTHORIZE_COMPLETION is False
    assert SERVICE_CAN_AUTHORIZE_TRANSITION is False
    assert SERVICE_CAN_CREATE_AUTHORITY is False
    assert SERVICE_CAN_WEAKEN_VALIDATION is False
    assert SERVICE_CAN_CHANGE_MODE is False
    assert SERVICE_WRITES_REPOSITORY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert SERVICE_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert NEGATIVE_EVIDENCE_RETAINED is True
    assert MCP_NEVER_SHELLS is True
    assert CLI_MCP_NEVER_SHELLS is True
    assert DIAGNOSTICS_ARE_AUTHORITATIVE is False
    assert INDEPENDENT_VALIDATION_REQUIRED is True
    assert CONTEXT_COMPILER_REMAINS_AUTHORITY is True
    assert REQUIRED_ROLLOUT_CONSUMES_CONTEXT is True
    assert REQUIRED_ROLLOUT_EMITS_RECEIPT_FLOOR is True
    profile = control_surface_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(SERVICE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "SemanticRefactoringService" in names
    assert "ControlRequest" in names
    assert "ControlResult" in names
    assert "ControlReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "SemanticRefactoringService" in exports
    assert "execute_control" in exports
    assert "dry_run_control" in exports
    assert "register_operations" in exports
    cli_tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    cli_names = {node.name for node in cli_tree.body if isinstance(node, ast.ClassDef)}
    assert "SemanticRefactoringCLI" in cli_names
    assert "SemanticRefactoringControlBackend" in cli_names


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_implement_forbidden_control_shortcuts() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_CONTROL_NAMES)
    descriptor = control_surface_descriptor()
    assert descriptor["interface"] == SEMANTIC_REFACTORING_SERVICE_INTERFACE
    assert descriptor["nomination_only"] is True
    assert descriptor["mcp_never_shells"] is True
    assert descriptor["diagnostics_are_authoritative"] is False
    assert descriptor["can_weaken_validation"] is False
    assert descriptor["can_authorize_completion"] is False
    assert tuple(descriptor["predecessor_task_ids"]) == PREDECESSOR_TASK_IDS
    forbids = set(descriptor["forbids"])
    assert "authorize_completion" in forbids
    assert "shell_mcp" in forbids
    assert "subprocess_dispatch" in forbids
    assert "open_network" in forbids
    assert "change_rollout_mode" in forbids


def test_exact_catalog_and_direct_python_cli_mcp_parity() -> None:
    catalog = register_operations()
    assert catalog == {"read": READ_OPERATIONS, "mutation": MUTATION_OPERATIONS}
    assert cli_register_operations() == catalog
    assert ALL_OPERATIONS == READ_OPERATIONS + MUTATION_OPERATIONS
    service = _service()
    request = ControlRequest(operation="spar.capabilities")
    python = service.execute(request)
    mcp = mcp_dispatch(service, request)
    output = StringIO()
    status = run_spar_cli(
        ["agent", "spar", "capabilities", "--request-json", json.dumps(request.to_dict())],
        service,
        stdout=output,
    )
    cli = json.loads(output.getvalue())
    assert status == 0
    assert python.to_dict() == mcp.to_dict() == cli
    assert service.operation_catalog() == catalog
    thin = SemanticRefactoringCLI(service)
    stdout = StringIO()
    assert thin.run(["capabilities", "--request-json", json.dumps(request.to_dict())], stdout=stdout) == 0
    assert json.loads(stdout.getvalue()) == python.to_dict()


def test_mcp_never_shells_or_round_trips_cli_text() -> None:
    imported = _imported_modules(CLI_PATH)
    assert "subprocess" not in imported
    assert "os" not in imported
    assert not (imported & FORBIDDEN_IMPORTS)
    calls = _call_names(CLI_PATH)
    assert "system" not in calls
    assert "Popen" not in calls
    assert "check_output" not in calls
    assert "run" in calls or "execute" in calls
    source = CLI_PATH.read_text(encoding="utf-8")
    assert "never executes a shell" in source
    assert "os.system" not in source
    assert "subprocess" not in source
    service_imports = _imported_modules(SERVICE_PATH)
    assert "subprocess" not in service_imports


def test_diagnostics_are_non_authoritative() -> None:
    service = _service()
    diagnosed = service.execute(ControlRequest(operation="spar.diagnose"))
    assert diagnosed.ok is True
    assert diagnosed.status == ControlStatus.DIAGNOSED.value
    assert diagnosed.payload["authoritative"] is False
    assert diagnosed.payload["can_authorize_completion"] is False
    assert diagnosed.payload["can_authorize_transition"] is False
    assert diagnosed.payload["mcp_never_shells"] is True
    assert diagnosed.payload["network"] == NETWORK_DENY
    assert set(diagnosed.payload["catalog"]["read"]) == set(READ_OPERATIONS)
    status = service.execute(ControlRequest(operation="spar.status"))
    assert status.payload["nomination_only"] is True
    assert status.payload["diagnostics_are_authoritative"] is False
    context = service.execute(ControlRequest(operation="spar.context_status"))
    assert context.payload["context_compiler_remains_authority"] is True
    assert context.payload["authoritative"] is False
    rollout = service.execute(ControlRequest(operation="spar.rollout_status"))
    assert rollout.payload["worker_may_change_mode"] is False
    assert rollout.payload["authoritative"] is False
    assert rollout.payload["receipt_floor"] == list(REQUIRED_RECEIPT_FLOOR)
    metrics = service.execute(ControlRequest(operation="spar.metrics"))
    assert metrics.payload["authoritative"] is False
    receipt = compile_control_receipt(
        operation="spar.diagnose",
        status=diagnosed.status,
        payload=diagnosed.payload,
    )
    assert receipt.diagnostics_are_authoritative is False
    assert receipt.can_authorize_completion is False
    assert receipt.mcp_never_shells is True


def test_mutation_authorization_idempotency_fence_dry_run_budget_and_audit() -> None:
    store = ControlStateStore()
    backend = SemanticRefactoringControlBackend(_service(store))

    with pytest.raises(ControlSurfaceError, match="authorization"):
        ControlRequest(
            operation="spar.rollback",
            idempotency_key="key",
            lease_id="lease",
            fencing_epoch=1,
            budget=_budget(),
            parameters=_nominate_params(),
        )
    with pytest.raises(ControlSurfaceError, match="exceeds"):
        ControlBudget(max_units=0, requested_units=1)

    dry = backend.execute(_mutation("spar.rollback", dry_run=True))
    assert dry.status == ControlStatus.DRY_RUN.value
    assert dry.payload["applied"] is False
    assert dry.ok is True
    assert backend.execute(ControlRequest(operation="spar.status")).payload["nomination_only"] is True

    applied = backend.execute(_mutation("spar.rollback"))
    replay = backend.execute(_mutation("spar.rollback"))
    assert applied.ok and applied.audit_id
    assert applied.status == ControlStatus.NOMINATED.value
    assert replay.to_dict() == {**applied.to_dict(), "idempotent_replay": True}
    assert len(store.audits) >= 3

    stale = backend.execute(_mutation("spar.request_review", key="idempotency-2", fence=1))
    assert stale.status == ControlStatus.CONFLICT.value
    assert stale.payload["reason_code"] == "spar_stale_fence"
    assert stale.ok is False


def test_request_merge_requires_receipt_floor_and_does_not_merge() -> None:
    service = _service()
    missing = service.execute(
        _mutation(
            "spar.request_merge",
            parameters={"context_receipt_cid": _cid("context")},
        )
    )
    assert missing.status == ControlStatus.TYPED_TERMINAL.value
    assert missing.ok is False
    assert missing.payload["reason_code"] == TerminalKind.MISSING_RECEIPT_FLOOR.value
    assert MISSING_RECEIPT_FLOOR_MESSAGE.split(",")[0] in missing.payload["error"]

    merged = service.execute(_mutation("spar.request_merge", key="merge-1", fence=2))
    assert merged.ok is True
    assert merged.payload["merge_requested"] is True
    assert merged.payload["merge_authorized"] is False
    assert merged.payload["can_authorize_completion"] is False
    assert merged.payload["can_authorize_transition"] is False
    assert set(merged.payload["receipt_floor"]) == set(REQUIRED_RECEIPT_FLOOR)


def test_nominate_requires_write_scope_context_and_validation() -> None:
    service = _service()
    with pytest.raises(ControlSurfaceError, match="unrestricted scope"):
        service.execute(
            _mutation(
                "spar.nominate",
                parameters=_nominate_params(write_paths=[]),
            )
        )
    with pytest.raises(ControlSurfaceError, match="unrestricted scope"):
        service.execute(
            _mutation(
                "spar.nominate",
                parameters=_nominate_params(write_paths=["pkg/*.py"]),
            )
        )
    with pytest.raises(ControlSurfaceError, match="unrestricted scope"):
        service.execute(
            _mutation(
                "spar.nominate",
                parameters=_nominate_params(write_paths=["/tmp/pkg/mod.py"]),
            )
        )
    with pytest.raises(ControlSurfaceError, match="unrestricted scope"):
        service.execute(
            _mutation(
                "spar.nominate",
                parameters=_nominate_params(write_paths=["pkg/../secret.py"]),
            )
        )
    with pytest.raises(ControlSurfaceError, match="validation_commands"):
        service.execute(
            _mutation(
                "spar.nominate",
                parameters=_nominate_params(validation_commands=[]),
            )
        )
    missing_context = service.execute(
        _mutation(
            "spar.nominate",
            parameters=_nominate_params(context_receipt_cid=""),
        )
    )
    assert missing_context.status == ControlStatus.TYPED_TERMINAL.value
    assert missing_context.payload["reason_code"] == TerminalKind.MISSING_CONTEXT.value
    assert MISSING_CONTEXT_MESSAGE in missing_context.payload["error"]

    nominated = service.execute(_mutation("spar.nominate", key="nom-1", fence=3))
    assert nominated.status == ControlStatus.NOMINATED.value
    assert nominated.payload["nominated"] is True
    assert nominated.payload["can_authorize_completion"] is False
    looked_up = service.execute(
        ControlRequest(
            operation="spar.get_receipt",
            parameters={"receipt_cid": nominated.receipt_cid},
        )
    )
    assert looked_up.payload["found"] is True
    assert looked_up.payload["stored_receipt"]["receipt_cid"] == nominated.receipt_cid


def test_body_free_requests_reject_source_dumps() -> None:
    with pytest.raises(ControlSurfaceError, match="body-free"):
        ControlRequest(operation="spar.diagnose", parameters={"source": "def fn():\n    return 1\n"})
    with pytest.raises(ControlSurfaceError, match="body-free"):
        ControlRequest(operation="spar.status", parameters={"full_task_dump": "task prose"})
    with pytest.raises(ControlSurfaceError, match="body-free"):
        ControlRequest(operation="spar.capabilities", parameters={"prompt": "full prompt"})


def test_vector_evidence_cannot_admit() -> None:
    service = _service()
    with pytest.raises(ControlSurfaceError, match="cannot admit"):
        service.execute(
            _mutation(
                "spar.nominate",
                parameters=_nominate_params(source_authority="vector_candidate"),
            )
        )
    with pytest.raises(ControlSurfaceError, match="cannot admit"):
        service.execute(
            _mutation(
                "spar.request_review",
                key="vec-1",
                fence=4,
                parameters=_nominate_params(heuristic=True),
            )
        )


def test_identity_excludes_observational_fields() -> None:
    receipt = compile_control_receipt(
        operation="spar.status",
        status=ControlStatus.OK.value,
        payload={"nomination_only": True},
    )
    payload = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(ControlSurfaceError, match="observational"):
        ControlReceipt.from_dict(dirty)


def test_receipt_cannot_claim_authority_flags() -> None:
    receipt = compile_control_receipt(
        operation="spar.diagnose",
        status=ControlStatus.DIAGNOSED.value,
        payload={"authoritative": False},
    )
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ControlSurfaceError, match="can_authorize_completion"):
        ControlReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["adapter_is_nomination_only"] = False
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ControlSurfaceError, match="nomination_only"):
        ControlReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["diagnostics_are_authoritative"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ControlSurfaceError, match="diagnostics are not authoritative"):
        ControlReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["mcp_never_shells"] = False
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ControlSurfaceError, match="MCP never shells"):
        ControlReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["network"] = "allow"
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ControlSurfaceError, match="network is denied"):
        ControlReceipt.from_dict(payload)


def test_round_trip_and_receipt_are_deterministic() -> None:
    first = execute_control(ControlRequest(operation="spar.diagnose"))
    second = execute_control(ControlRequest(operation="spar.diagnose"))
    assert first.receipt_cid == second.receipt_cid
    receipt = compile_control_receipt(
        operation=first.operation, status=first.status, payload=first.payload
    )
    restored = decode_canonical_receipt(encode_canonical_receipt(receipt))
    assert restored == receipt
    assert restored.receipt_cid == receipt.receipt_cid
    assert ControlReceipt.from_dict(receipt.to_dict()).receipt_cid == receipt.receipt_cid


def test_adapter_dry_run_is_deterministic_and_does_not_mutate() -> None:
    service = SemanticRefactoringService()
    request = _mutation("spar.nominate", dry_run=True)
    first = service.dry_run(request)
    second = dry_run_control(request, service=service)
    assert first.receipt_cid == second.receipt_cid
    assert first.payload["applied"] is False
    assert first.status == ControlStatus.DRY_RUN.value
    assert service.store.applied == {}
    assert service.store.fences == {}
    applied = service.execute(_mutation("spar.nominate", dry_run=False, key="live-1", fence=5))
    assert applied.payload["applied"] is True
    assert applied.receipt_cid != first.receipt_cid


def test_read_operations_cannot_carry_mutation_bindings() -> None:
    with pytest.raises(ControlSurfaceError, match="read operations"):
        ControlRequest(operation="spar.status", idempotency_key="key")
    with pytest.raises(ControlSurfaceError, match="read operations"):
        ControlRequest(operation="spar.diagnose", lease_id="lease")
    with pytest.raises(ControlSurfaceError, match="unknown SPAR operation"):
        ControlRequest(operation="spar.invent")


def test_existing_adapter_authorities_are_declared_not_replaced() -> None:
    descriptor = control_surface_descriptor()
    assert EXISTING_ADAPTER_AUTHORITIES == (
        "datasets_semantic",
        "kit_storage",
        "kit_vfs",
        "accelerator_supervisor",
        "accelerator_runtime",
        "spar_narrow",
    )
    assert descriptor["context_compiler_remains_authority"] is True
    assert "replace_context_compiler" in set(descriptor["forbids"])


def test_cli_and_mcp_call_the_same_service_without_shelling() -> None:
    service = _service()
    request = ControlRequest(operation="spar.status")
    mcp = mcp_dispatch(SemanticRefactoringControlBackend(service), request)
    stdout = StringIO()
    assert SemanticRefactoringCLI(service).run(
        ["status", "--request-json", json.dumps(request.to_dict())],
        stdout=stdout,
    ) == 0
    assert json.loads(stdout.getvalue())["operation"] == "spar.status"
    assert mcp.payload["nomination_only"] is True
    assert mcp.to_dict() == json.loads(stdout.getvalue())
