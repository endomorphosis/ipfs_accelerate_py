"""Current canonical route selection refuses incompatible treatments before I/O.

The provider-free cases use the existing authored supervisor fixture. These
tests establish option custody, not native worker, provider or token authority.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE
from test.api.test_terminal_prompt_tokenizer_forwarding import supervisor

NOMINATION = {"semantic_decision_nomination_artifact": Path('/owner/nomination.json'),
              "semantic_decision_nomination_sha256": 'a' * 64}


@pytest.mark.parametrize('selected', [
    {'semantic_native_input': True},
    {'semantic_slot_policy_revision': 'closed-typed-identifiers@2'},
    NOMINATION,
])
@pytest.mark.parametrize('renderer', [
    {'semantic_transport_schema': 'supervisor-semantic-router-input@2'},
    {'semantic_metadata_view': 'common-bindings@1'},
    {'source_view_profile': 'dual-python@1'},
])
def test_mixed_renderers_refuse_before_archive_source_or_owner_access(tmp_path, monkeypatch, selected, renderer):
    def forbidden(*args, **kwargs):
        pytest.fail('incompatible renderer reached I/O')
    unread = SimpleNamespace(resolve=forbidden)
    monkeypatch.setattr(driver.os, 'geteuid', forbidden)
    options = {**selected, **renderer}
    with pytest.raises(ValueError):
        benchmark.config_for(unread, tmp_path/'out', unread, 'full', **options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=unread, output=tmp_path/'out', archive=unread, arm='full', **options)
    with pytest.raises(ValueError):
        harbor.FullSupervisorAgent(logs_dir=tmp_path/'logs', runtime_archive=str(tmp_path/'missing'), arm='full', **options)
    with pytest.raises(ValueError):
        driver.run(instruction=tmp_path/'instruction', state=tmp_path/'state', arm='full', **options)
    assert not (tmp_path/'out').exists() and not (tmp_path/'logs').exists()


def test_decision_provider_mismatch_refuses_before_archive_access(tmp_path):
    with pytest.raises(ValueError, match='qualified Codex provider'):
        benchmark.config_for(tmp_path, tmp_path/'out', tmp_path/'archive', 'full',
                             provider_profile=GROK_PROFILE, **NOMINATION)


@pytest.mark.parametrize('selected', [
    {'max_input_tokens': 4096},
    {'semantic_native_input': True},
    {'semantic_slot_policy_revision': 'closed-typed-identifiers@2'},
    NOMINATION,
    {**NOMINATION, 'disable_semantic_typed_nomination': True},
])
def test_provider_free_route_retains_requested_options_without_claiming_application(supervisor, monkeypatch, selected):
    state, planned, signed = supervisor
    monkeypatch.setattr(doctor, 'prepare_terminal_doctor_dispatch', lambda **kwargs: {
        'route': 'doctor_candidate', 'status': 'candidate_ready', 'provider_calls': 0,
        'artifact': '/owner/candidate.json', 'sha256': 'b'*64, 'task_cid': 'task'})
    result = driver.run(instruction=Path('/instruction'), state=state, arm='full', **selected)
    treatment = result['coding_control_application']
    assert treatment['status'] == 'not_applicable_provider_free'
    assert treatment['requested']['prompt_comparison_options'] == benchmark._prompt_options('full', **selected)
    assert treatment['model_coding_dispatched'] is False and treatment['token_savings_established'] is False
    assert len(signed) == 1 and '--doctor-candidate-artifact' in signed[0]
    assert '--semantic-native-input' not in signed[0]
    assert not any(arg.startswith('--semantic-decision-') for arg in signed[0])
