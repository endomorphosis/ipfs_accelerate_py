"""Explicit independent planning adapters survive public service composition."""
import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory import (
    PlanningAnalysisAdmissionError, PlanningAnalysisFactory,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.service_factory import resolve_production_composition


def test_native_analysis_and_admission_factory_reach_prompt_service(tmp_path):
    factory = PlanningAnalysisFactory(repository_allowlist=(tmp_path,), index_root=tmp_path / "index")
    composition = resolve_production_composition(repository_root=tmp_path, state_root=tmp_path / "state",
                                                 require_activation=False)
    composition.extras.update(optional_analysis=factory.optional_analysis,
                             admission_request_factory=factory.admission_request_factory)
    service = composition.prompt_supervisor_service()
    assert service.optional_analysis is factory.optional_analysis
    assert service.admission_request_factory is factory.admission_request_factory
    assert composition.prompt_supervisor_service() is service
    # Wiring is not authority: the real independent IR builder remains required.
    with pytest.raises(PlanningAnalysisAdmissionError) as caught:
        service.admission_request_factory.build(None, None, None)
    assert caught.value.reason_code == "ir_request_builder_unset"
    assert composition.intent_factory is None
