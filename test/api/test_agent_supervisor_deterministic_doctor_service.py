"""Conformance tests for DeterministicDoctorService (LPR-039)."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DeterministicDoctorRunReceipt,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorEvidenceSnapshot,
    DoctorMode,
    DoctorOperation,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service import (
    ALL_SERVICE_OPERATIONS,
    DoctorOperationRequest,
    DoctorOperationResult,
    DoctorServiceCapabilityCode,
    DoctorServiceSafetyError,
    DoctorStageBackends,
    DeterministicDoctorService,
    InMemoryDoctorReceiptStore,
    assert_body_free,
    build_doctor_operation_request,
    create_deterministic_doctor_service,
    optional_providers_loaded,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import (
    DeterministicDoctorPolicy,
    PolicyVerdict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "control"
    / "deterministic_doctor_service.py"
)


def _roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _snapshot(roots: DoctorAuthorityRoots | None = None) -> DoctorEvidenceSnapshot:
    roots = roots or _roots()
    return DoctorEvidenceSnapshot(
        roots=roots,
        snapshot_id="snapshot:fixture",
        file_blob_cids=("blob:a", "blob:b"),
        completeness="complete",
        invalidation_refs=("tree:fixture",),
        clean_rebuild_equivalence_receipt_id="rebuild:eq:1",
    )


def _consumer(
    roots: DoctorAuthorityRoots,
    consumer_id: str = "consumer:one",
) -> DoctorConsumerDisposition:
    return DoctorConsumerDisposition(
        roots=roots,
        consumer_id=consumer_id,
        disposition=DoctorRepairDisposition.SUPPORTED,
        reason_codes=("ok",),
    )


def _admitted_plan(roots: DoctorAuthorityRoots | None = None) -> DeterministicDoctorPlan:
    roots = roots or _roots()
    site = DoctorEditSite(
        path="pkg/module.py",
        before_hash="sha256:before",
        span_start=0,
        span_end=10,
        artifact_id="blob:module",
    )
    step = DoctorPlanStep(
        step_id="step:1",
        kind="analytical",
        operator_id="operator:rename",
        consumer_ids=("consumer:one",),
        edit_site_refs=(site.content_id,),
        write_paths=("pkg/module.py",),
    )
    return DeterministicDoctorPlan(
        roots=roots,
        plan_id="plan:fixture",
        snapshot_id="snapshot:fixture",
        finding_ids=("finding:one",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=(_consumer(roots),),
        impact_closure_id="impact:fixture",
        steps=(step,),
        edit_sites=(site,),
        operator_ids=("operator:rename",),
        target_ref="symbol:target",
        value_source_ref="value:source",
        placement_ref="placement:site",
        selected_operator_id="operator:rename",
        permitted_read_paths=("pkg/module.py",),
        permitted_write_paths=("pkg/module.py",),
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        proof_refs=("proof:fixture",),
        invalidation_refs=("tree:fixture",),
    )


def _enabled_policy(**overrides) -> DeterministicDoctorPolicy:
    base = {
        "enabled": True,
        "default_mode": DoctorMode.NARROW_AUTO,
        "narrow_autonomous_mutation_enabled": True,
    }
    base.update(overrides)
    return DeterministicDoctorPolicy(**base)


def _repair_backend_success(
    request: DoctorOperationRequest,
    *,
    policy: DeterministicDoctorPolicy,
    policy_decision,
) -> DeterministicDoctorRunReceipt:
    roots = request.effective_roots() or _roots()
    return DeterministicDoctorRunReceipt(
        roots=roots,
        receipt_id="receipt:repair:backend",
        operation=DoctorOperation.REPAIR,
        mode=request.mode,
        disposition=DoctorRepairDisposition.SUPPORTED,
        snapshot_id=request.effective_snapshot_id() or "snapshot:fixture",
        incident_id=request.incident_cid(),
        plan_id=request.plan.plan_id if request.plan is not None else "plan:fixture",
        lease_id=request.effective_lease_id(),
        checkpoint_ref=request.effective_checkpoint_ref(),
        rollback_ref=request.effective_rollback_ref(),
        transaction_ref="txn:backend",
        candidate_tree_cid="tree:candidate",
        committed_tree_cid="tree:committed",
        invalidation_refs=("tree:fixture",),
        reason_codes=("repair_backend",),
        network_denied=True,
        secrets_inherited=False,
    )


# ---------------------------------------------------------------------------
# Cold import / surface hygiene
# ---------------------------------------------------------------------------


def test_cold_import_loads_no_optional_providers_or_network_stack() -> None:
    script = """
import json, sys, os
os.environ['IPFS_ACCEL_SKIP_CORE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
forbidden = (
    'torch', 'transformers', 'openai', 'anthropic', 'neo4j', 'duckdb',
    'httpx', 'aiohttp', 'requests', 'llm_router', 'psycopg2', 'sqlalchemy',
)
before = {name for name in sys.modules if name.split('.')[0] in forbidden}
import ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service as m
after = {name for name in sys.modules if name.split('.')[0] in forbidden}
disc = m.DeterministicDoctorService.discovery()
print(json.dumps({
    'added': sorted(after - before),
    'optional': list(m.optional_providers_loaded()),
    'discovery_ops': disc['operations'],
    'llm': disc['llm_router_enabled'],
    'fallback': disc['automatic_fallback'],
    'processes': disc['processes_started'],
    'database': disc['database_opened'],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONPATH": str(REPO_ROOT)
            + (
                ":" + __import__("os").environ["PYTHONPATH"]
                if __import__("os").environ.get("PYTHONPATH")
                else ""
            ),
        },
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert payload["optional"] == []
    assert "inspect" in payload["discovery_ops"]
    assert payload["llm"] is False
    assert payload["fallback"] is False
    assert payload["processes"] is False
    assert payload["database"] is False


def test_service_module_source_never_imports_llm_or_optional_providers() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    lowered = "\n".join(imported).lower()
    for forbidden in (
        "llm_router",
        "openai",
        "anthropic",
        "torch",
        "transformers",
        "duckdb",
        "neo4j",
        "ipfs_datasets_embedding",
        "ipfs_datasets_py",
    ):
        assert forbidden not in lowered
    assert "import llm_router" not in source
    assert "from llm_router" not in source


def test_discovery_is_static_and_closed() -> None:
    disc = DeterministicDoctorService.discovery()
    assert set(disc["operations"]) == set(ALL_SERVICE_OPERATIONS)
    assert disc["llm_router_enabled"] is False
    assert disc["automatic_fallback"] is False
    assert "repair" in disc["write_operations"]
    assert "inspect" in disc["read_only_operations"]
    assert "status" in disc["read_only_operations"]


# ---------------------------------------------------------------------------
# Read-only operations
# ---------------------------------------------------------------------------


def test_inspect_explain_plan_are_read_only() -> None:
    service = create_deterministic_doctor_service()
    roots = _roots()
    snapshot = _snapshot(roots)
    plan = _admitted_plan(roots)

    inspect = service.inspect(
        roots=roots.to_dict(),
        snapshot=snapshot.to_dict(),
        incident_id="incident:inspect",
    )
    assert inspect.read_only is True
    assert inspect.changed is False
    assert inspect.succeeded
    assert inspect.run_receipt is not None
    assert inspect.run_receipt.operation is DoctorOperation.INSPECT
    assert inspect.run_receipt.committed_tree_cid == ""

    explain = service.explain(
        roots=roots.to_dict(),
        plan=plan.to_dict(),
        incident_id="incident:explain",
    )
    assert explain.read_only is True
    assert explain.changed is False
    assert explain.run_receipt is not None
    assert explain.run_receipt.operation is DoctorOperation.EXPLAIN

    planned = service.plan(
        roots=roots.to_dict(),
        plan=plan.to_dict(),
        incident_id="incident:plan",
        mode=DoctorMode.PLAN.value,
    )
    assert planned.read_only is True
    assert planned.changed is False
    assert planned.run_receipt is not None
    assert planned.run_receipt.operation is DoctorOperation.PLAN
    assert planned.run_receipt.committed_tree_cid == ""


def test_inspect_without_snapshot_abstains_actionably() -> None:
    service = create_deterministic_doctor_service()
    result = service.inspect(incident_id="incident:empty")
    assert result.abstained
    assert DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value in result.reason_codes
    assert result.read_only is True
    assert result.changed is False


# ---------------------------------------------------------------------------
# Repair gates
# ---------------------------------------------------------------------------


def test_repair_requires_enabled_policy_explicit_op_and_prerequisites() -> None:
    default_service = create_deterministic_doctor_service()
    roots = _roots()
    plan = _admitted_plan(roots)
    snapshot = _snapshot(roots)

    denied = default_service.repair(
        mode=DoctorMode.NARROW_AUTO.value,
        plan=plan.to_dict(),
        snapshot=snapshot.to_dict(),
        exact_clean_target=True,
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        incident_id="incident:repair-denied",
    )
    assert not denied.succeeded
    assert denied.changed is False

    enabled = create_deterministic_doctor_service(policy=_enabled_policy())
    missing_plan = enabled.repair(
        mode=DoctorMode.NARROW_AUTO.value,
        snapshot=snapshot.to_dict(),
        exact_clean_target=True,
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        incident_id="incident:repair-no-plan",
    )
    assert not missing_plan.succeeded
    assert any("plan" in code or "policy" in code for code in missing_plan.reason_codes)

    no_target = enabled.repair(
        mode=DoctorMode.NARROW_AUTO.value,
        plan=plan.to_dict(),
        roots=roots.to_dict(),
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        incident_id="incident:repair-no-target",
    )
    assert no_target.abstained
    assert (
        DoctorServiceCapabilityCode.EXACT_CLEAN_TARGET_REQUIRED.value
        in no_target.reason_codes
    )

    # Policy + exact clean target + lease/plan ok, but no transaction backend.
    abstain = enabled.repair(
        mode=DoctorMode.NARROW_AUTO.value,
        plan=plan.to_dict(),
        snapshot=snapshot.to_dict(),
        exact_clean_target=True,
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        write_paths=("pkg/module.py",),
        incident_id="incident:repair-no-backend",
    )
    assert abstain.abstained
    assert (
        DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value in abstain.reason_codes
    )
    assert abstain.changed is False


def test_repair_with_transaction_backend_and_eligible_plan() -> None:
    service = DeterministicDoctorService(
        policy=_enabled_policy(),
        backends=DoctorStageBackends(transaction=_repair_backend_success),
    )
    roots = _roots()
    result = service.repair(
        mode=DoctorMode.NARROW_AUTO.value,
        plan=_admitted_plan(roots).to_dict(),
        snapshot=_snapshot(roots).to_dict(),
        exact_clean_target=True,
        lease_id="lease:fixture",
        checkpoint_ref="checkpoint:fixture",
        rollback_ref="rollback:fixture",
        write_paths=("pkg/module.py",),
        incident_id="incident:repair-ok",
    )
    assert result.succeeded
    assert result.run_receipt is not None
    assert result.run_receipt.operation is DoctorOperation.REPAIR
    assert result.run_receipt.disposition is DoctorRepairDisposition.SUPPORTED
    assert result.run_receipt.plan_id == "plan:fixture"
    assert result.run_receipt.lease_id == "lease:fixture"
    assert result.run_receipt.checkpoint_ref == "checkpoint:fixture"
    assert result.run_receipt.rollback_ref == "rollback:fixture"
    assert result.run_receipt.model_invocation_count == 0
    assert result.run_receipt.llm_router_invoked is False


# ---------------------------------------------------------------------------
# Replay / status / verify
# ---------------------------------------------------------------------------


def test_replay_is_identity_equivalent_and_incident_idempotent() -> None:
    store = InMemoryDoctorReceiptStore()
    service = DeterministicDoctorService(receipt_store=store)
    roots = _roots()
    first = service.inspect(
        roots=roots.to_dict(),
        snapshot=_snapshot(roots).to_dict(),
        incident_id="incident:replay-1",
    )
    assert first.succeeded
    assert first.run_receipt is not None
    receipt_cid = first.run_receipt.content_id

    replay_a = service.replay(incident_id="incident:replay-1")
    assert replay_a.replayed is True
    assert replay_a.read_only is True
    assert replay_a.run_receipt is not None
    assert replay_a.run_receipt.content_id == receipt_cid

    replay_b = service.replay(incident_id="incident:replay-1")
    assert replay_b.run_receipt is not None
    assert replay_b.run_receipt.content_id == receipt_cid
    assert replay_a.run_receipt.to_dict() == replay_b.run_receipt.to_dict()


def test_status_and_verify() -> None:
    service = create_deterministic_doctor_service()
    roots = _roots()
    run = service.inspect(
        roots=roots.to_dict(),
        snapshot=_snapshot(roots).to_dict(),
        incident_id="incident:status-1",
    )
    status = service.status(incident_id="incident:status-1")
    assert status.read_only is True
    assert status.status["incident_known"] is True
    assert status.status["llm_router_enabled"] is False
    assert status.status["automatic_fallback"] is False

    verified = service.verify(incident_id="incident:status-1")
    assert verified.succeeded
    assert verified.status["verified"] is True
    assert run.run_receipt is not None
    assert verified.run_receipt is not None
    assert verified.run_receipt.content_id == run.run_receipt.content_id

    missing = service.verify(incident_id="incident:missing")
    assert missing.abstained
    assert DoctorServiceCapabilityCode.INCIDENT_NOT_FOUND.value in missing.reason_codes


# ---------------------------------------------------------------------------
# Safety: LLM intercept, body/secret, no fallback
# ---------------------------------------------------------------------------


def test_intercepted_llm_router_call_fails_with_no_fallback() -> None:
    service = create_deterministic_doctor_service()
    with pytest.raises(DoctorServiceSafetyError, match="llm_invocation_forbidden"):
        service.execute(
            {
                "operation": "inspect",
                "roots": _roots().to_dict(),
                "snapshot": _snapshot().to_dict(),
                "llm_router_invoked": True,
            }
        )

    service.note_provider_invocation(llm_router=True, model_count=1)
    with pytest.raises(DoctorServiceSafetyError, match="llm_invocation_forbidden"):
        service.inspect(
            roots=_roots().to_dict(),
            snapshot=_snapshot().to_dict(),
            incident_id="incident:after-llm",
        )


def test_importing_fake_llm_router_does_not_create_fallback_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intercepted llm_router module must not be used as a fallback."""

    calls: list[str] = []

    fake = types.ModuleType("ipfs_accelerate_py.agent_supervisor.llm_router")

    def _route(*_a, **_k):
        calls.append("route")
        return {"text": "should-never-run"}

    fake.route = _route  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "ipfs_accelerate_py.agent_supervisor.llm_router", fake
    )

    service = create_deterministic_doctor_service()
    # Service must not call into the fake module.
    result = service.inspect(
        roots=_roots().to_dict(),
        snapshot=_snapshot().to_dict(),
        incident_id="incident:no-fallback",
    )
    assert result.succeeded
    assert calls == []
    assert result.run_receipt is not None
    assert result.run_receipt.llm_router_invoked is False
    assert result.status.get("automatic_fallback") is False


def test_body_and_secret_keys_rejected_on_request() -> None:
    with pytest.raises(Exception, match="bodies or secrets"):
        build_doctor_operation_request(
            "inspect",
            body="print('nope')",
        )
    with pytest.raises(Exception, match="bodies or secrets"):
        assert_body_free({"api_key": "x"}, "payload")
    with pytest.raises(Exception, match="bodies or secrets"):
        DoctorOperationRequest.from_dict(
            {
                "operation": "inspect",
                "token": "secret-value",
            }
        )


def test_unsupported_capability_abstains_without_unhealthy_startup() -> None:
    service = create_deterministic_doctor_service()
    # Construction itself must not raise when backends are missing.
    assert service.backends_available == ()
    plan_result = service.plan(incident_id="incident:no-plan-backend")
    assert plan_result.abstained
    assert any(
        code
        in {
            DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
            DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
            "plan_backend_or_plan_required",
            "plan_materialization_deferred",
        }
        for code in plan_result.reason_codes
    )
    # Service remains usable after abstention.
    status = service.status()
    assert status.read_only is True
    assert status.status["processes_started"] is False


def test_request_and_result_round_trip() -> None:
    req = build_doctor_operation_request(
        DoctorOperation.INSPECT,
        incident_id="incident:rt",
        roots=_roots().to_dict(),
        snapshot=_snapshot().to_dict(),
    )
    again = DoctorOperationRequest.from_dict(req.to_dict())
    assert again.operation == req.operation
    assert again.incident_id == req.incident_id

    service = create_deterministic_doctor_service()
    result = service.execute(req)
    restored = DoctorOperationResult.from_dict(result.to_dict())
    assert restored.disposition == result.disposition
    assert restored.read_only is True


def test_policy_decision_attached_on_doctor_operations() -> None:
    service = create_deterministic_doctor_service()
    result = service.inspect(
        roots=_roots().to_dict(),
        snapshot=_snapshot().to_dict(),
        incident_id="incident:policy",
    )
    assert result.policy_decision is not None
    assert result.policy_decision.verdict is PolicyVerdict.ALLOW
    assert result.policy_decision.read_only is True


def test_module_exports_required_ast_symbols() -> None:
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service"
    )
    for name in (
        "DeterministicDoctorService",
        "DoctorOperation",
        "DoctorOperationRequest",
        "DoctorOperationResult",
    ):
        assert hasattr(mod, name)
