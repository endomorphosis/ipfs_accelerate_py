"""Tests for DefaultPlannerFactory@1 / DefaultPlannerHandles@1 (WPD-011)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import (
    AdaptivePlanner,
)
from ipfs_accelerate_py.agent_supervisor.planning.default_planner_factory import (
    DEFER_CAPABILITY_DISPOSITION,
    DEFAULT_OPTIONAL_PROVERS,
    DEFAULT_PLANNER_FACTORY_EVIDENCE,
    DEFAULT_PLANNER_FACTORY_INTERFACE,
    DEFAULT_PLANNER_HANDLES_INTERFACE,
    DefaultPlannerCapabilityError,
    DefaultPlannerFactory,
    DefaultPlannerHandles,
    OptionalProverId,
    OptionalProverStatus,
    PlannerStackDisposition,
    build_default_planner_factory,
    build_default_planner_handles,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator import (
    FormalPlanValidator,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    FormalReplanner,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_planner import (
    ProofCarryingPlanner,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
)


def _absent_which(_name: str) -> str | None:
    return None


def _present_which(name: str) -> str | None:
    table = {
        "lean": "/usr/bin/lean",
        "z3": "/usr/bin/z3",
        "cvc5": "/usr/bin/cvc5",
        "coqc": "/usr/bin/coqc",
        "coqtop": "/usr/bin/coqtop",
        "hammer": "/usr/bin/hammer",
    }
    return table.get(name)


def _minimal_plan_source() -> dict[str, object]:
    """Minimal formal-plan source aligned with formal_plan_compiler fixtures."""

    return {
        "schema": "fixture/formal-plan-input@1",
        "repository_tree_id": "tree:default-planner-factory",
        "objectives": [
            {
                "goal_id": "G12.S1",
                "goal_cid": "goal:cid:g12-s1",
                "owner_actor_id": "owner:supervisor",
                "title": "Compile formal plans",
                "acceptance_criteria": ["Every task has retained evidence."],
            }
        ],
        "taskboard": [
            {
                "task_id": "WPD-011",
                "task_cid": "task:cid:wpd-011",
                "goal_id": "G12.S1",
                "actor_id": "agent:planner",
                "resource_needs": ["cpu"],
                "changed_ast_scopes": ["symbol:cid:default-planner-factory"],
                "acceptance_criteria": ["contract tests pass"],
                "validation_commands": ["pytest test_default_planner_factory.py"],
                "lease": {
                    "lease_cid": "lease:cid:wpd-011",
                    "holder_id": "agent:planner",
                    "fencing_token": 1,
                },
            }
        ],
        "ast_records": [
            {
                "symbol_cid": "symbol:cid:default-planner-factory",
                "tree_cid": "tree:default-planner-factory",
                "task_cid": "task:cid:wpd-011",
                "symbol": "DefaultPlannerFactory",
            }
        ],
        "proof_policy": {
            "policy_cid": "policy:cid:wpd-planner",
            "minimum_code_assurance": "candidate",
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:pytest"],
            "required_evidence": [
                {
                    "kind": "plan_check",
                    "subject_ids": ["WPD-011"],
                    "source_scope_ids": ["symbol:cid:default-planner-factory"],
                }
            ],
        },
        "evidence_records": [
            {
                "evidence_cid": "evidence:cid:factory-tests",
                "task_cid": "task:cid:wpd-011",
                "kind": "test",
            }
        ],
    }


def test_interfaces_and_evidence_key_are_stable() -> None:
    assert DEFAULT_PLANNER_FACTORY_INTERFACE == "DefaultPlannerFactory@1"
    assert DEFAULT_PLANNER_HANDLES_INTERFACE == "DefaultPlannerHandles@1"
    assert DEFAULT_PLANNER_FACTORY_EVIDENCE == "wpd/default-planner-factory@1"
    assert DEFER_CAPABILITY_DISPOSITION == "defer_capability"
    assert (
        DEFER_CAPABILITY_DISPOSITION
        == ImplementationDisposition.DEFER_CAPABILITY.value
    )
    assert {item.value for item in DEFAULT_OPTIONAL_PROVERS} == {
        "lean",
        "z3",
        "cvc5",
        "coq",
    }


def test_factory_builds_compiler_validator_replanner_and_adaptive_planner() -> None:
    handles = build_default_planner_handles(which=_absent_which)

    assert isinstance(handles, DefaultPlannerHandles)
    assert isinstance(handles.compiler, FormalPlanCompiler)
    assert isinstance(handles.validator, FormalPlanValidator)
    assert isinstance(handles.replanner, FormalReplanner)
    assert isinstance(handles.adaptive_planner, AdaptivePlanner)
    assert handles.core_ready is True
    # Shared stack: replanner must reuse the factory compiler/validator.
    assert handles.replanner.compiler is handles.compiler
    assert handles.replanner.validator is handles.validator
    assert handles.factory_interface == DEFAULT_PLANNER_FACTORY_INTERFACE
    assert handles.handles_interface == DEFAULT_PLANNER_HANDLES_INTERFACE


def test_bound_compiler_produces_usable_formal_plan() -> None:
    handles = build_default_planner_handles(which=_absent_which)
    result = handles.compiler.compile(_minimal_plan_source())

    assert result.status is CompilationStatus.COMPILED
    assert result.valid and result.supported and result.plan is not None
    assert result.plan.repository_tree_id == "tree:default-planner-factory"


def test_optional_provers_absent_yield_defer_capability_not_silent_success() -> None:
    factory = DefaultPlannerFactory(
        which=_absent_which,
        require_optional_provers=("lean", "z3"),
    )
    handles = factory.build()

    assert handles.disposition is PlannerStackDisposition.DEFER_CAPABILITY
    assert handles.disposition.value == DEFER_CAPABILITY_DISPOSITION
    assert handles.defers_capability is True
    assert handles.capability_complete is False
    assert handles.claims_success is False
    assert "lean" in handles.missing_required_optional_provers
    assert "z3" in handles.missing_required_optional_provers
    # Optional inventory marks them unavailable — never available/success.
    status = dict(handles.optional_prover_status)
    assert status["lean"] == OptionalProverStatus.UNAVAILABLE.value
    assert status["z3"] == OptionalProverStatus.UNAVAILABLE.value
    for record in handles.optional_prover_records:
        if record.required:
            assert record.available is False
            assert record.status is not OptionalProverStatus.AVAILABLE
    # Core stack still bound for compile/validate/replan.
    assert handles.core_ready is True
    assert isinstance(handles.compiler, FormalPlanCompiler)
    # Proof-carrying handle must not claim success.
    assert handles.proof_carrying_handle is not None
    assert handles.proof_carrying_handle.available is False
    assert handles.proof_carrying_handle.claims_success is False
    assert (
        handles.proof_carrying_handle.disposition
        is PlannerStackDisposition.DEFER_CAPABILITY
    )


def test_require_proof_carrying_without_backends_defers() -> None:
    handles = build_default_planner_handles(
        which=_absent_which,
        require_proof_carrying=True,
    )

    assert handles.disposition is PlannerStackDisposition.DEFER_CAPABILITY
    assert handles.claims_success is False
    assert handles.proof_carrying_handle is not None
    assert handles.proof_carrying_handle.available is False
    with pytest.raises(DefaultPlannerCapabilityError, match="defer"):
        handles.proof_carrying_handle.build(
            _minimal_plan_source(),
            artifact_path=Path("/tmp/unused-proof-carrying.json"),
        )


def test_absent_provers_are_not_invented_as_available() -> None:
    handles = build_default_planner_handles(which=_absent_which)

    # Core stack remains usable without optional provers.
    assert handles.core_ready is True
    assert handles.available_optional_provers == ()
    assert set(handles.missing_optional_provers) >= {"lean", "z3", "cvc5", "coq"}
    payload = handles.to_dict()
    for record in payload["optional_provers"]:
        assert record["available"] is False
        assert record["status"] != OptionalProverStatus.AVAILABLE.value
        assert "invented" not in record["reason_code"]
        assert "synthetic" not in record["reason_code"]
    for record in handles.optional_prover_records:
        assert record.available is False
    # Proof-carrying must not silently succeed when backends are absent.
    assert handles.proof_carrying_handle is not None
    assert handles.proof_carrying_handle.available is False
    assert handles.proof_carrying_handle.claims_success is False


def test_injected_available_prover_is_recorded() -> None:
    handles = build_default_planner_handles(
        which=_present_which,
        require_optional_provers=("lean",),
        prover_executor=lambda _ctx: {
            "status": "unavailable",
            "accepted": False,
            "reason": "test-double",
        },
    )

    assert "lean" in handles.available_optional_provers
    lean = next(
        item
        for item in handles.optional_prover_records
        if item.prover_id is OptionalProverId.LEAN
    )
    assert lean.available is True
    assert lean.status is OptionalProverStatus.AVAILABLE
    assert lean.executable == "/usr/bin/lean"
    assert handles.disposition is PlannerStackDisposition.READY
    assert handles.capability_complete is True
    assert handles.claims_success is True
    assert handles.proof_carrying_handle is not None
    assert handles.proof_carrying_handle.available is True


def test_custom_probe_unavailable_defers_when_required() -> None:
    handles = build_default_planner_handles(
        which=_present_which,
        require_optional_provers=("z3",),
        optional_prover_probes={
            "z3": lambda _name: {
                "status": "unavailable",
                "reason_code": "bridge_not_importable",
            }
        },
    )

    assert handles.disposition is PlannerStackDisposition.DEFER_CAPABILITY
    assert handles.claims_success is False
    assert "z3" in handles.missing_required_optional_provers
    z3 = next(
        item
        for item in handles.optional_prover_records
        if item.prover_id is OptionalProverId.Z3
    )
    assert z3.available is False
    assert z3.reason_code == "bridge_not_importable"


def test_proof_carrying_handle_builds_when_backends_present(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "workflow.json"
    handles = build_default_planner_handles(
        which=_present_which,
        prover_executor=lambda _ctx: {
            "status": "unavailable",
            "accepted": False,
            "reason": "lane not executed in unit test",
        },
    )
    assert handles.proof_carrying_handle is not None
    assert handles.proof_carrying_handle.available is True

    planner = handles.proof_carrying_handle.build(
        _minimal_plan_source(),
        artifact_path=artifact,
    )
    assert isinstance(planner, ProofCarryingPlanner)
    assert planner.compiler is handles.compiler
    assert planner.validator is handles.validator


def test_handles_projection_is_content_addressed_and_body_free() -> None:
    first = build_default_planner_handles(which=_absent_which)
    second = build_default_planner_handles(which=_absent_which)

    payload = first.to_dict()
    assert first.content_id == content_identity(payload)
    assert first.content_id == second.content_id
    blob = str(payload).casefold()
    assert "source_text" not in blob
    assert "api_key" not in blob
    assert "class service" not in blob
    assert payload["schema"].endswith("default-planner-handles@1")
    assert payload["defers_capability"] is False or payload["core_ready"] is True
    assert payload["shared_stack"]["replanner_uses_factory_compiler"] is True
    assert payload["shared_stack"]["replanner_uses_factory_validator"] is True


def test_factory_last_handles_and_builder_helpers() -> None:
    factory = build_default_planner_factory(which=_absent_which)
    assert factory.last_handles is None
    handles = factory.build()
    assert factory.last_handles is handles
    assert factory.INTERFACE == DEFAULT_PLANNER_FACTORY_INTERFACE


def test_replanner_marks_verifier_unavailable_when_provers_absent() -> None:
    handles = build_default_planner_handles(which=_absent_which)
    assert handles.replanner.verifier_available is False
    assert handles.replanner.verifier is None


def test_cold_import_does_not_load_network_clients() -> None:
    # The factory module itself must not pull network clients as direct imports.
    module = sys.modules[
        "ipfs_accelerate_py.agent_supervisor.planning.default_planner_factory"
    ]
    forbidden = ("requests", "httpx", "aiohttp", "urllib3")
    for name in forbidden:
        assert name not in getattr(module, "__dict__", {})
        assert not hasattr(module, name)
    build_default_planner_handles(which=_absent_which)
    for name in forbidden:
        assert name not in getattr(module, "__dict__", {})
