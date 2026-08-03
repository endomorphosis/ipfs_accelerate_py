"""Production composition tests for the lazy deterministic-Doctor runtime."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    DeterministicDoctorService,
    DoctorOperationRequest,
    DoctorServiceCapabilityCode,
    DoctorServiceSafetyError,
    DoctorStageBackends,
)
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import (
    DeterministicDoctorBackendFactory,
    DeterministicDoctorRuntime,
    DeterministicDoctorRuntimeError,
    DeterministicDoctorRuntimeSafetyError,
    DoctorRuntimeStage,
    DoctorRuntimeStageUnavailable,
    create_deterministic_doctor_runtime,
)
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import (
    DeterministicDoctorPolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = (
    REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "deterministic_doctor.py"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "doctor@example.invalid")
    _git(path, "config", "user.name", "Deterministic Doctor")
    for relative, body in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "fixture")
    return path


def _roots() -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id="repository:runtime-control",
        forest_id="forest:runtime-control",
        tree_id="tree:runtime-control",
        overlay_id="overlay:runtime-control",
        file_root_id="files:runtime-control",
        ast_root_id="ast:runtime-control",
        graph_id="graph:runtime-control",
        corpus_id="corpus:runtime-control",
        index_id="index:runtime-control",
        model_id="model:runtime-control",
        cache_id="cache:runtime-control",
        operator_registry_id="operators:runtime-control",
        translator_id="translator:runtime-control",
        solver_id="solver:runtime-control",
        kernel_id="kernel:runtime-control",
        toolchain_id="toolchain:runtime-control",
        policy_id="policy:runtime-control",
        sandbox_id="sandbox:runtime-control",
        environment_id="environment:runtime-control",
        lease_id="lease:runtime-control",
    )


def _snapshot(roots: DoctorAuthorityRoots) -> DoctorEvidenceSnapshot:
    return DoctorEvidenceSnapshot(
        roots=roots,
        snapshot_id="snapshot:runtime-control",
        file_blob_cids=("blob:runtime-control",),
        completeness="complete",
        invalidation_refs=(roots.tree_id,),
        clean_rebuild_equivalence_receipt_id="rebuild:runtime-control",
    )


def _plan(roots: DoctorAuthorityRoots) -> DeterministicDoctorPlan:
    site = DoctorEditSite(
        path="pkg/module.py",
        before_hash="sha256:before",
        span_start=0,
        span_end=4,
        artifact_id="blob:module",
    )
    consumer = DoctorConsumerDisposition(
        roots=roots,
        consumer_id="consumer:runtime-control",
        disposition=DoctorRepairDisposition.SUPPORTED,
        reason_codes=("closed",),
    )
    step = DoctorPlanStep(
        step_id="step:runtime-control",
        kind="analytical",
        operator_id="operator:runtime-control",
        consumer_ids=(consumer.consumer_id,),
        edit_site_refs=(site.content_id,),
        write_paths=(site.path,),
    )
    return DeterministicDoctorPlan(
        roots=roots,
        plan_id="plan:runtime-control",
        snapshot_id="snapshot:runtime-control",
        finding_ids=("finding:runtime-control",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=(consumer,),
        impact_closure_id="impact:runtime-control",
        steps=(step,),
        edit_sites=(site,),
        operator_ids=(step.operator_id,),
        target_ref="symbol:runtime-control",
        value_source_ref="value:runtime-control",
        placement_ref="placement:runtime-control",
        selected_operator_id=step.operator_id,
        permitted_read_paths=(site.path,),
        permitted_write_paths=(site.path,),
        lease_id="lease:runtime-control",
        checkpoint_ref="checkpoint:runtime-control",
        rollback_ref="rollback:runtime-control",
        proof_refs=("proof:runtime-control",),
        invalidation_refs=(roots.tree_id,),
    )


def _transaction_receipt(
    request: DoctorOperationRequest,
    *,
    policy: DeterministicDoctorPolicy,
    policy_decision: object,
) -> DeterministicDoctorRunReceipt:
    del policy, policy_decision
    roots = request.effective_roots()
    assert roots is not None
    assert request.plan is not None
    return DeterministicDoctorRunReceipt(
        roots=roots,
        receipt_id="receipt:runtime-control",
        operation=DoctorOperation.REPAIR,
        mode=request.mode,
        disposition=DoctorRepairDisposition.SUPPORTED,
        snapshot_id=request.effective_snapshot_id(),
        incident_id=request.incident_cid(),
        plan_id=request.plan.plan_id,
        lease_id=request.effective_lease_id(),
        checkpoint_ref=request.effective_checkpoint_ref(),
        rollback_ref=request.effective_rollback_ref(),
        transaction_ref="transaction:runtime-control",
        candidate_tree_cid="tree:candidate",
        committed_tree_cid="tree:committed",
        invalidation_refs=(roots.tree_id,),
        reason_codes=("transaction_applied",),
        network_denied=True,
        secrets_inherited=False,
    )


def test_runtime_discovery_is_static_and_provider_free() -> None:
    script = """
import json, sys
forbidden = (
    'torch', 'transformers', 'openai', 'anthropic', 'neo4j', 'duckdb',
    'httpx', 'aiohttp', 'requests', 'llm_router', 'psycopg2', 'sqlalchemy',
)
before = {name for name in sys.modules if name.split('.')[0] in forbidden}
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import (
    DeterministicDoctorRuntime,
)
discovery = DeterministicDoctorRuntime.discovery()
after = {name for name in sys.modules if name.split('.')[0] in forbidden}
print(json.dumps({'added': sorted(after - before), 'discovery': discovery}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    discovery = payload["discovery"]
    assert len(discovery["stages"]) == len(DoctorRuntimeStage)
    assert discovery["providers_started"] is False
    assert discovery["processes_started"] is False
    assert discovery["database_opened"] is False
    assert discovery["model_routes_allowed"] is False
    assert discovery["network_routes_allowed"] is False


def test_backend_factory_is_lazy_actionable_and_hard_fails_unsafe_routes() -> None:
    calls: list[str] = []

    def evidence() -> object:
        calls.append("evidence")
        return object()

    def unavailable() -> object:
        calls.append("proof")
        raise ImportError("approved prover is absent")

    factory = DeterministicDoctorBackendFactory(
        stage_factories={
            DoctorRuntimeStage.EVIDENCE: evidence,
            DoctorRuntimeStage.PROOF: unavailable,
            DoctorRuntimeStage.TACTICIAN: lambda: SimpleNamespace(uses_model=True),
            DoctorRuntimeStage.IMPACT: lambda: SimpleNamespace(uses_network=True),
        }
    )
    factory.discovery()
    factory.capabilities()
    assert calls == []
    assert factory.loaded_stages == ()

    factory.get(DoctorRuntimeStage.EVIDENCE)
    assert calls == ["evidence"]
    assert factory.loaded_stages == ("evidence",)

    with pytest.raises(DoctorRuntimeStageUnavailable) as exc_info:
        factory.get(DoctorRuntimeStage.PROOF)
    assert exc_info.value.reason_code == "stage_dependency_unavailable"
    assert "approved digest-bound prover" in exc_info.value.remediation
    capability = {
        row.stage: row for row in factory.capabilities()
    }[DoctorRuntimeStage.PROOF]
    assert capability.available is False
    assert capability.remediation

    with pytest.raises(DeterministicDoctorRuntimeSafetyError) as model_error:
        factory.get(DoctorRuntimeStage.TACTICIAN)
    assert model_error.value.reason_code == "model_route_forbidden"
    with pytest.raises(DeterministicDoctorRuntimeSafetyError) as network_error:
        factory.get(DoctorRuntimeStage.IMPACT)
    assert network_error.value.reason_code == "network_route_forbidden"


def test_report_only_builds_exact_evidence_and_loads_no_optional_stage(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path / "repo",
        {
            "app.py": "def add(left, right):\n    return left + right\n",
            "config.json": '{"enabled": true}\n',
            "README.md": "# Fixture\n",
        },
    )
    runtime = create_deterministic_doctor_runtime(repo)
    assert runtime.backend_factory.loaded_stages == ()
    assert runtime.evidence is None

    report = runtime.inspect()

    assert report.result.succeeded
    assert report.result.read_only is True
    assert report.result.changed is False
    assert report.evidence is not None
    inventory = {item.path: item for item in report.evidence.source_inventory}
    assert set(inventory) == {"README.md", "app.py", "config.json"}
    expected = "sha256:" + hashlib.sha256(
        (repo / "app.py").read_bytes()
    ).hexdigest()
    assert inventory["app.py"].content_digest == expected
    assert inventory["app.py"].byte_count == (repo / "app.py").stat().st_size
    # The inventory is complete; only semantic/structured coverage is parsed.
    assert report.evidence.diagnostic_source_paths == ("app.py", "config.json")
    assert runtime.backend_factory.loaded_stages == ("evidence", "diagnose")
    assert report.stage_receipts["evidence"]["status"] == "completed"
    assert report.stage_receipts["diagnose"]["status"] == "completed"


def test_checkout_must_match_explicit_allowlist(tmp_path: Path) -> None:
    allowed = _repository(tmp_path / "allowed", {"a.py": "A = 1\n"})
    rejected = _repository(tmp_path / "rejected", {"b.py": "B = 2\n"})
    with pytest.raises(DeterministicDoctorRuntimeError) as exc_info:
        DeterministicDoctorRuntime(
            checkout_root=rejected,
            repository_allowlist=(allowed,),
        )
    assert exc_info.value.reason_code == "checkout_not_allowlisted"


def test_configured_materialized_submodule_sources_are_enumerated(
    tmp_path: Path,
) -> None:
    child = _repository(tmp_path / "child", {"child.py": "CHILD = 7\n"})
    parent = _repository(tmp_path / "parent", {"main.py": "PARENT = 11\n"})
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(child),
        "libs/child",
    )
    _git(parent, "commit", "-qam", "configured submodule")

    evidence = create_deterministic_doctor_runtime(parent).build_evidence()

    paths = {item.path for item in evidence.source_inventory}
    assert "main.py" in paths
    assert "libs/child/child.py" in paths
    assert evidence.submodule_closure == (
        {
            "path": "libs/child",
            "commit_id": _git(child, "rev-parse", "HEAD").stdout.strip(),
            "depth": 0,
            "available": True,
            "reason_code": "configured_submodule",
        },
    )


def test_plan_wires_analytical_stages_lazily_and_abstains_actionably(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path / "repo", {"app.py": "VALUE = 1\n"})
    runtime = create_deterministic_doctor_runtime(
        repo,
        policy={"enabled": True, "default_mode": DoctorMode.PLAN.value},
    )

    report = runtime.plan(mode=DoctorMode.PLAN.value)

    assert report.result.abstained
    assert "plan_inputs_deferred" in report.result.reason_codes
    assert runtime.backend_factory.loaded_stages == (
        "evidence",
        "diagnose",
        "retrieve",
        "tactician",
        "proof",
        "synthesis_preview",
        "impact",
    )
    assert "transaction" not in runtime.backend_factory.loaded_stages
    assert "fixed_point" not in runtime.backend_factory.loaded_stages
    for stage in ("retrieve", "tactician", "proof", "synthesis_preview", "impact"):
        receipt = report.stage_receipts[stage]
        assert receipt["status"] == "wired"
        assert receipt["reason_code"] == "awaiting_typed_stage_inputs"
        assert receipt["remediation"]


def test_deterministic_service_and_runtime_hard_fail_network_routes(
    tmp_path: Path,
) -> None:
    service = DeterministicDoctorService()
    with pytest.raises(DoctorServiceSafetyError):
        service.inspect(network_access=True)

    repo = _repository(tmp_path / "repo", {"app.py": "VALUE = 1\n"})
    runtime = create_deterministic_doctor_runtime(repo)
    with pytest.raises(DeterministicDoctorRuntimeSafetyError) as exc_info:
        runtime.inspect(network_access=True)
    assert exc_info.value.reason_code == "deterministic_route_forbidden"
    assert runtime.backend_factory.loaded_stages == ()


class _ControlDependency:
    def __init__(self, *, permit: bool = True) -> None:
        self.permit = permit
        self.requests: list[object] = []
        self.audits: list[tuple[str, ...]] = []

    def authorize_doctor_operation(self, request: object) -> dict[str, object]:
        self.requests.append(request)
        effects = tuple(request.expected_effect_ids)
        return {
            "permitted": self.permit,
            "permit_id": "permit:runtime-control",
            "authorized_effect_ids": effects,
        }

    def record_doctor_effects(
        self,
        request: object,
        *,
        permit: object,
        applied_effect_ids: tuple[str, ...],
        changed: bool,
    ) -> dict[str, object]:
        del request, permit
        assert changed is True
        self.audits.append(tuple(applied_effect_ids))
        return {
            "recorded": True,
            "audit_receipt_id": "audit:runtime-control",
            "applied_effect_ids": applied_effect_ids,
        }


def test_service_consumes_control_permit_and_records_exact_effects() -> None:
    roots = _roots()
    snapshot = _snapshot(roots)
    plan = _plan(roots)
    control = _ControlDependency()
    service = DeterministicDoctorService(
        policy=DeterministicDoctorPolicy(
            enabled=True,
            default_mode=DoctorMode.NARROW_AUTO,
            narrow_autonomous_mutation_enabled=True,
        ),
        backends=DoctorStageBackends(transaction=_transaction_receipt),
        control_service=control,
    )

    result = service.repair(
        mode=DoctorMode.NARROW_AUTO.value,
        plan=plan.to_dict(),
        snapshot=snapshot.to_dict(),
        exact_clean_target=True,
        lease_id=plan.lease_id,
        checkpoint_ref=plan.checkpoint_ref,
        rollback_ref=plan.rollback_ref,
        write_paths=plan.permitted_write_paths,
        incident_id="incident:runtime-control",
    )

    assert result.succeeded
    assert result.changed is True
    assert len(control.requests) == 1
    control_request = control.requests[0]
    assert control_request.write_paths == ("pkg/module.py",)
    assert control.audits == [control_request.expected_effect_ids]
    assert result.status["control_effects_verified"] is True
    assert result.stage_refs["control_permit_id"] == "permit:runtime-control"
    assert result.stage_refs["control_audit_receipt_id"] == "audit:runtime-control"


def test_control_rejection_prevents_transaction() -> None:
    roots = _roots()
    plan = _plan(roots)
    control = _ControlDependency(permit=False)
    transaction_calls: list[str] = []

    def transaction(*args: object, **kwargs: object) -> object:
        transaction_calls.append("called")
        return _transaction_receipt(*args, **kwargs)

    service = DeterministicDoctorService(
        policy=DeterministicDoctorPolicy(
            enabled=True,
            default_mode=DoctorMode.NARROW_AUTO,
            narrow_autonomous_mutation_enabled=True,
        ),
        backends=DoctorStageBackends(transaction=transaction),
        control_service=control,
    )
    result = service.repair(
        mode=DoctorMode.NARROW_AUTO.value,
        plan=plan.to_dict(),
        snapshot=_snapshot(roots).to_dict(),
        exact_clean_target=True,
        lease_id=plan.lease_id,
        checkpoint_ref=plan.checkpoint_ref,
        rollback_ref=plan.rollback_ref,
        write_paths=plan.permitted_write_paths,
    )
    assert result.abstained
    assert transaction_calls == []
    assert (
        DoctorServiceCapabilityCode.CONTROL_PERMIT_REJECTED.value
        in result.reason_codes
    )


def test_cli_consumes_checkout_root_and_emits_runtime_evidence(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path / "repo", {"app.py": "VALUE = 1\n"})
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--checkout-root",
            str(repo),
            "inspect",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONPATH": str(REPO_ROOT)
            + (
                os.pathsep + os.environ["PYTHONPATH"]
                if os.environ.get("PYTHONPATH")
                else ""
            ),
        },
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["disposition"] == DoctorRepairDisposition.SUPPORTED.value
    assert payload["runtime"]["evidence"]["checkout_root"] == str(repo.resolve())
    assert [
        item["path"]
        for item in payload["runtime"]["evidence"]["source_inventory"]
    ] == ["app.py"]
    assert payload["runtime"]["capability_graph"]["loaded_stages"] == [
        "evidence",
        "diagnose",
    ]
