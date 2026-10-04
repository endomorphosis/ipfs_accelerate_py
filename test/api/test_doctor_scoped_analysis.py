"""Partial Doctor analysis preserves the signed global source boundary."""
import json

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_doctor_task_workflow import (
    SOURCE, _prepare, _prepare_analysis_guard_fixture,
)
from ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory import PlanningAnalysisSecretError
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_scoped_analysis as scoped
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


def _build(scenario, inputs):
    return scoped.build_scoped_doctor_analysis(repository=scenario['repository'],
        admission=inputs['admission'], task_cid=inputs['task_cid'])


def test_scoped_native_diagnostics_do_not_relabel_refused_whole_checkout(scenario, tmp_path):
    inputs = _prepare_analysis_guard_fixture(scenario, tmp_path)
    with pytest.raises(PlanningAnalysisSecretError):
        inputs['runtime'].build_evidence(refresh=True)
    result = _build(scenario, inputs)
    report = result.report
    assert result.sources == {'answer.py': SOURCE.encode()}
    assert report['source_count'] == 1 and report['omitted_source_count'] == 1
    assert report['source_tree_id'] == local._tree(inputs['admission']['manifest']['payload']['sources'])
    assert report['diagnostic_snapshot_id'] == result.snapshot.snapshot_id
    assert report['finding_count'] == len(result.findings)
    assert result.diagnostic_snapshot.open_frontiers
    assert 'frontier:interprocedural_dataflow' in report['open_frontiers']
    assert report['scoped_inventory_complete'] is True
    for field in ('whole_repository_analysis', 'full_static_analysis', 'proof_created',
                  'mutation_authority', 'completion_authority'):
        assert report[field] is False
    assert report['provider_calls'] == 0 and inputs['runtime'].evidence is None
    serialized = json.dumps(report)
    assert SOURCE not in serialized and 'AUTHORED-DUMMY' not in serialized
    assert 'docs/example.rst' not in serialized
    assert report['analysis_cid'] == local.content_identity({
        key: value for key, value in report.items() if key != 'analysis_cid'})
    result.assert_current()
    with pytest.raises(TypeError):
        result.sources['answer.py'] = b'changed'
    report['source_count'] = 999
    assert result.report['source_count'] == 1


def test_all_signed_task_consumers_are_kept(scenario, tmp_path):
    inputs = _prepare(scenario, tmp_path, external_consumer=True)
    result = _build(scenario, inputs)
    assert set(result.sources) == {'answer.py', 'test_answer.py'}
    assert result.report['omitted_source_count'] == 0
    assert result.report['whole_repository_analysis'] is False


def test_selected_secret_refuses_before_native_diagnostics(scenario, tmp_path, monkeypatch):
    inputs = _prepare(scenario, tmp_path,
        text=SOURCE + '\npassword = "AUTHORED-DUMMY-NOT-A-CREDENTIAL"\n')
    monkeypatch.setattr(scoped, 'diagnose_repository', lambda *_args, **_kwargs:
                        pytest.fail('refused source must not enter native diagnostics'))
    with pytest.raises(PlanningAnalysisSecretError) as exc:
        _build(scenario, inputs)
    assert 'AUTHORED-DUMMY' not in str(exc.value) and 'answer.py' not in str(exc.value)


@pytest.mark.parametrize('name', ['answer.py', 'docs/example.rst'])
def test_current_check_binds_selected_and_omitted_source(scenario, tmp_path, name):
    inputs = _prepare_analysis_guard_fixture(scenario, tmp_path)
    result = _build(scenario, inputs)
    path = scenario['repository'] / name
    path.write_bytes(path.read_bytes() + b'\n# changed after capture\n')
    with pytest.raises(local.LocalPlanningError):
        result.assert_current()


def test_source_change_during_diagnostics_refuses_result(scenario, tmp_path, monkeypatch):
    inputs = _prepare(scenario, tmp_path)
    diagnose = scoped.diagnose_repository
    def mutate(*args, **kwargs):
        result = diagnose(*args, **kwargs)
        (scenario['repository'] / 'answer.py').write_text(SOURCE + '# concurrent change\n')
        return result
    monkeypatch.setattr(scoped, 'diagnose_repository', mutate)
    with pytest.raises(local.LocalPlanningError):
        _build(scenario, inputs)


@pytest.mark.parametrize('wrong_binding', ['task', 'repository'])
def test_foreign_binding_cannot_choose_source_scope(scenario, tmp_path, wrong_binding):
    inputs = _prepare(scenario, tmp_path)
    with pytest.raises(scoped.ScopedDoctorAnalysisError):
        scoped.build_scoped_doctor_analysis(
            repository=tmp_path if wrong_binding == 'repository' else scenario['repository'],
            admission=inputs['admission'],
            task_cid='task:foreign' if wrong_binding == 'task' else inputs['task_cid'])


def test_native_diagnostic_errors_are_not_reinterpreted_as_unavailability(scenario, tmp_path, monkeypatch):
    inputs = _prepare(scenario, tmp_path)
    def fail(*_args, **_kwargs):
        raise RuntimeError('authored native diagnostic failure')
    monkeypatch.setattr(scoped, 'diagnose_repository', fail)
    with pytest.raises(RuntimeError, match='authored native diagnostic failure'):
        _build(scenario, inputs)
