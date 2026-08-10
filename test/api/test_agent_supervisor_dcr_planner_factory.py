"""DCR-060: production Planner composition with Doctor and datasets logic.

Acceptance:
* Default handles exercise real services (compiler, validator, replanner,
  candidate portfolio, scheduler, Doctor, datasets logic, IR logic hooks,
  plan-admission receipt service).
* A missing mandatory component is unavailable and cannot mint planner-view
  evidence.
* Planner-view evidence never grants execution authority.
* Cold discovery does not load LLM / network client surfaces.
"""

from __future__ import annotations

import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service import (
    DETERMINISTIC_DOCTOR_SERVICE_INTERFACE,
    DeterministicDoctorService,
)
from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import (
    AdaptivePlanner,
)
from ipfs_accelerate_py.agent_supervisor.planning.default_planner_factory import (
    DCR_PLANNER_FACTORY_EVIDENCE,
    DEFAULT_PLANNER_FACTORY_INTERFACE,
    DEFAULT_PLANNER_HANDLES_INTERFACE,
    MANDATORY_PLANNER_COMPONENTS,
    PLANNER_COMPOSITION_ROOT_INTERFACE,
    PLANNER_NODE_SCHEDULER_INTERFACE,
    PLANNER_VIEW_EVIDENCE_SCHEMA,
    DefaultPlannerCapabilityError,
    DefaultPlannerFactory,
    DefaultPlannerHandles,
    PlannerComponentId,
    PlannerComponentStatus,
    PlannerCompositionRoot,
    PlannerNodeScheduler,
    build_default_planner_handles,
    build_planner_composition_root,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    FormalPlanCompiler,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator import (
    FormalPlanValidator,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    FormalReplanner,
)
from ipfs_accelerate_py.agent_supervisor.planning.ir_logic_hooks import (
    IR_LOGIC_HOOKS_INTERFACE,
    IRLogicHooks,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_admission_service import (
    PLAN_ADMISSION_SERVICE_INTERFACE,
    PlanAdmissionService,
)
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
    SYMBOLIC_CANDIDATE_PLANNER_INTERFACE,
    SymbolicCandidatePlanner,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_integration import (
    DATASETS_LOGIC_FACADE_INTERFACE,
    DatasetsLogicFacade,
)


def _absent_which(_name: str) -> str | None:
    return None


def test_interfaces_and_evidence_are_stable() -> None:
    assert DEFAULT_PLANNER_FACTORY_INTERFACE == "DefaultPlannerFactory@1"
    assert DEFAULT_PLANNER_HANDLES_INTERFACE == "DefaultPlannerHandles@1"
    assert PLANNER_COMPOSITION_ROOT_INTERFACE == "PlannerCompositionRoot@1"
    assert DCR_PLANNER_FACTORY_EVIDENCE == "dcr/planner-factory@1"
    assert DATASETS_LOGIC_FACADE_INTERFACE == "DatasetsLogicFacade@1"
    assert DETERMINISTIC_DOCTOR_SERVICE_INTERFACE == "DeterministicDoctorService@1"
    assert set(MANDATORY_PLANNER_COMPONENTS) == {
        "compiler",
        "validator",
        "replanner",
        "candidate_portfolio",
        "scheduler",
        "doctor",
        "logic",
        "proof",
        "receipt",
    }
    discovery = PlannerCompositionRoot.discovery()
    assert discovery["interface"] == PLANNER_COMPOSITION_ROOT_INTERFACE
    assert discovery["evidence"] == DCR_PLANNER_FACTORY_EVIDENCE
    assert discovery["llm_router_enabled"] is False
    assert discovery["grants_execution_authority"] is False
    assert discovery["model_calls"] == 0
    assert set(discovery["mandatory_components"]) == set(MANDATORY_PLANNER_COMPONENTS)


def test_default_handles_bind_real_doctor_and_datasets_logic() -> None:
    handles = build_default_planner_handles(which=_absent_which)

    assert isinstance(handles, DefaultPlannerHandles)
    assert isinstance(handles.compiler, FormalPlanCompiler)
    assert isinstance(handles.validator, FormalPlanValidator)
    assert isinstance(handles.replanner, FormalReplanner)
    assert isinstance(handles.adaptive_planner, AdaptivePlanner)
    assert isinstance(handles.candidate_portfolio, SymbolicCandidatePlanner)
    assert isinstance(handles.scheduler, PlannerNodeScheduler)
    assert isinstance(handles.doctor, DeterministicDoctorService)
    assert isinstance(handles.datasets_logic, DatasetsLogicFacade)
    assert isinstance(handles.ir_logic_hooks, IRLogicHooks)
    assert isinstance(handles.receipt_service, PlanAdmissionService)

    assert handles.core_ready is True
    assert handles.composition_ready is True
    assert handles.missing_mandatory_components == ()
    assert set(handles.available_mandatory_components) == set(
        MANDATORY_PLANNER_COMPONENTS
    )
    assert handles.can_mint_planner_view_evidence is True

    status = dict(handles.mandatory_component_status)
    for component in MANDATORY_PLANNER_COMPONENTS:
        assert status[component] == PlannerComponentStatus.AVAILABLE.value


def test_default_handles_exercise_real_services() -> None:
    handles = build_default_planner_handles(which=_absent_which)
    self_tests = handles.exercise_real_services()

    assert self_tests["ok"] is True
    assert self_tests["composition_ready"] is True
    assert self_tests["can_mint_planner_view_evidence"] is True
    assert self_tests["model_calls"] == 0
    assert self_tests["failures"] == []

    exercised = self_tests["exercised"]
    assert exercised["compiler"]["ok"] is True
    assert exercised["validator"]["ok"] is True
    assert exercised["replanner"]["ok"] is True
    assert exercised["replanner"]["shares_compiler"] is True
    assert exercised["replanner"]["shares_validator"] is True
    assert exercised["candidate_portfolio"]["ok"] is True
    assert exercised["candidate_portfolio"]["interface"] == (
        SYMBOLIC_CANDIDATE_PLANNER_INTERFACE
    )
    assert exercised["scheduler"]["ok"] is True
    assert exercised["scheduler"]["schedule_id"]
    assert exercised["doctor"]["ok"] is True
    assert exercised["doctor"]["interface"] == DETERMINISTIC_DOCTOR_SERVICE_INTERFACE
    assert exercised["doctor"]["llm_router_enabled"] is False
    assert exercised["logic"]["ok"] is True
    assert exercised["logic"]["interface"] == DATASETS_LOGIC_FACADE_INTERFACE
    assert exercised["logic"]["authoritative"] is False
    assert exercised["logic"]["completion_authorized"] is False
    assert exercised["proof"]["ok"] is True
    assert exercised["proof"]["interface"] == IR_LOGIC_HOOKS_INTERFACE
    assert exercised["receipt"]["ok"] is True
    assert exercised["receipt"]["interface"] == PLAN_ADMISSION_SERVICE_INTERFACE

    # Direct service calls remain real (not synthetic probes).
    doctor_discovery = handles.doctor.discovery()
    assert doctor_discovery["interface"] == DETERMINISTIC_DOCTOR_SERVICE_INTERFACE
    logic_receipt = handles.datasets_logic.capability_receipt()
    assert logic_receipt["interface"] == DATASETS_LOGIC_FACADE_INTERFACE
    assert logic_receipt["completion_authorized"] is False
    schedule = handles.scheduler.schedule(
        [{"node_id": "n1"}, {"node_id": "n2", "depends_on": ("n1",)}]
    )
    assert schedule["node_count"] == 2
    assert schedule["execution_authority"] is False


def test_composition_root_composes_same_production_stack() -> None:
    root = build_planner_composition_root(which=_absent_which)
    assert isinstance(root, PlannerCompositionRoot)
    assert root.INTERFACE == PLANNER_COMPOSITION_ROOT_INTERFACE
    assert root.EVIDENCE == DCR_PLANNER_FACTORY_EVIDENCE

    handles = root.compose()
    assert root.last_handles is handles
    assert handles.composition_ready is True
    assert isinstance(handles.doctor, DeterministicDoctorService)
    assert isinstance(handles.datasets_logic, DatasetsLogicFacade)

    again = PlannerCompositionRoot.from_factory(
        DefaultPlannerFactory(which=_absent_which)
    ).build()
    assert again.composition_ready is True


def test_missing_mandatory_component_is_unavailable_and_cannot_mint() -> None:
    handles = build_default_planner_handles(
        which=_absent_which,
        omit_mandatory=(PlannerComponentId.DOCTOR, "logic"),
    )

    assert handles.core_ready is True
    assert handles.composition_ready is False
    assert "doctor" in handles.missing_mandatory_components
    assert "logic" in handles.missing_mandatory_components
    assert handles.doctor is None
    assert handles.datasets_logic is None
    assert handles.can_mint_planner_view_evidence is False

    status = dict(handles.mandatory_component_status)
    assert status["doctor"] == PlannerComponentStatus.OMITTED.value
    assert status["logic"] == PlannerComponentStatus.OMITTED.value
    # Other mandatory slots remain real and available.
    assert status["compiler"] == PlannerComponentStatus.AVAILABLE.value
    assert status["scheduler"] == PlannerComponentStatus.AVAILABLE.value
    assert isinstance(handles.scheduler, PlannerNodeScheduler)
    assert isinstance(handles.candidate_portfolio, SymbolicCandidatePlanner)

    with pytest.raises(DefaultPlannerCapabilityError) as excinfo:
        handles.mint_planner_view_evidence()
    assert excinfo.value.reason_code == "planner_view_unavailable"
    assert "doctor" in excinfo.value.missing_components
    assert "logic" in excinfo.value.missing_components

    self_tests = handles.exercise_real_services()
    assert self_tests["ok"] is False
    assert "doctor:unavailable" in self_tests["failures"]
    assert "logic:unavailable" in self_tests["failures"]


def test_disabled_dcr_composition_cannot_mint_planner_view() -> None:
    handles = build_default_planner_handles(
        which=_absent_which,
        bind_dcr_composition=False,
    )
    assert handles.core_ready is True
    assert handles.composition_ready is False
    assert handles.can_mint_planner_view_evidence is False
    assert set(handles.missing_mandatory_components) >= {
        "candidate_portfolio",
        "scheduler",
        "doctor",
        "logic",
        "proof",
        "receipt",
    }
    with pytest.raises(DefaultPlannerCapabilityError, match="planner-view"):
        handles.mint_planner_view_evidence()


def test_complete_composition_mints_non_authoritative_planner_view() -> None:
    handles = build_default_planner_handles(which=_absent_which)
    evidence = handles.mint_planner_view_evidence()

    assert evidence["schema"] == PLANNER_VIEW_EVIDENCE_SCHEMA
    assert evidence["evidence"] == DCR_PLANNER_FACTORY_EVIDENCE
    assert evidence["view"] == "planner"
    assert evidence["authoritative"] is False
    assert evidence["completion_authorized"] is False
    assert evidence["execution_authority"] is False
    assert evidence["grants_proof_authority"] is False
    assert evidence["model_calls"] == 0
    assert evidence["content_id"]
    identities = evidence["component_identities"]
    assert identities["doctor"]["interface"] == "DeterministicDoctorService@1"
    assert identities["logic"]["interface"] == "DatasetsLogicFacade@1"
    assert identities["scheduler"]["interface"] == PLANNER_NODE_SCHEDULER_INTERFACE
    assert identities["proof"]["interface"] == "IRLogicHooks@1"
    assert identities["receipt"]["interface"] == "PlanAdmissionService@1"
    receipt = evidence["capability_receipt"]
    assert receipt["authoritative"] is False
    assert receipt["grants_execution_authority"] is False
    assert receipt["can_mint_planner_view_evidence"] is True


def test_capability_receipt_is_body_free_and_non_authoritative() -> None:
    handles = build_default_planner_handles(which=_absent_which)
    receipt = handles.capability_receipt()
    blob = str(receipt).casefold()
    assert "source_text" not in blob
    assert "api_key" not in blob
    assert "class service" not in blob
    assert receipt["evidence"] == DCR_PLANNER_FACTORY_EVIDENCE
    assert receipt["composition_ready"] is True
    assert receipt["model_calls"] == 0

    projection = handles.to_dict()
    assert projection["dcr_evidence"] == DCR_PLANNER_FACTORY_EVIDENCE
    assert projection["composition_ready"] is True
    assert projection["grants_execution_authority"] is False
    assert projection["components"]["doctor"] == "DeterministicDoctorService"
    assert projection["components"]["logic"] == "DatasetsLogicFacade"


def test_scheduler_assigns_deterministic_partial_order() -> None:
    scheduler = PlannerNodeScheduler(default_resource_class="cpu-medium")
    first = scheduler.schedule(
        (
            {"node_id": "a", "resource_class": "cpu-proof-solver"},
            {"node_id": "b", "depends_on": ("a",)},
        )
    )
    second = scheduler.schedule(
        (
            {"node_id": "a", "resource_class": "cpu-proof-solver"},
            {"node_id": "b", "depends_on": ("a",)},
        )
    )
    assert first["schedule_id"] == second["schedule_id"]
    assert first["nodes"][0]["resource_class"] == "cpu-proof-solver"
    assert first["nodes"][1]["depends_on"] == ["a"]
    assert first["execution_authority"] is False


def test_cold_module_does_not_import_network_clients() -> None:
    module = sys.modules[
        "ipfs_accelerate_py.agent_supervisor.planning.default_planner_factory"
    ]
    forbidden = ("requests", "httpx", "aiohttp", "urllib3", "openai", "anthropic")
    for name in forbidden:
        assert name not in getattr(module, "__dict__", {})
        assert not hasattr(module, name)
    # Composition still succeeds without those clients.
    handles = build_default_planner_handles(which=_absent_which)
    assert handles.composition_ready is True
    for name in forbidden:
        assert name not in getattr(module, "__dict__", {})
