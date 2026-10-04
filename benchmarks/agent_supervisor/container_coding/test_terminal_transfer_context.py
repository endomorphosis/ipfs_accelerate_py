"""Real forked weights reach planning and retain an immutable asset selection."""
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from test.api.test_codebase_autoencoder_transfer import teacher, fork  # noqa: F401


def _prepare(original, fork):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    result = prep.initial_context(state=state, train_autoencoder=True, weight_transfer=fork)
    return root, state, prepared, result


def test_exact_fork_is_used_by_index_planner_and_admitted_reuse(original, teacher, fork, monkeypatch):
    import torch
    monkeypatch.setattr(torch, 'randn', lambda *_a, **_k: pytest.fail('this profile must use the legal weight fork'))
    root, state, prepared, result = _prepare(original, fork)
    learned = result['codebase_autoencoder']
    assert learned['weight_transfer'] == fork
    assert learned['metrics']['weight_transfer']['random_initialization'] is False
    selected = json.loads((state / 'code-learning-assets.json').read_bytes())
    assert selected == {'weight_transfer': fork, 'canonical_cve_training': None}
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    summary = next(row for row in loaded['summaries'] if row['schema'] == 'terminal-code-autoencoder-planning-summary@1')
    assert summary['legal_ir_weights_forked'] is True and summary['legal_ir_mutable_state_shared'] is False
    assert summary['weight_transfer']['initializer_sha256'] == fork['initializer_sha256']
    calls = []
    def router(prompt, **_kwargs):
        calls.append(prompt)
        assert fork['initializer_sha256'] in prompt and fork['source_checkpoint_sha256'] in prompt
        return {'text': _proposal_json(prepared), 'observation': {}, 'execution_receipt': None}
    _version(monkeypatch)
    planned = prep.plan(state, provider_callable=router)
    assert planned['qualified'], planned
    assert len(calls) == 1
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder as ae
    monkeypatch.setattr(ae, 'train_codebase_autoencoder', lambda **_: pytest.fail('admission must reuse exact trained fork'))
    rebound = prep.context(state=state)
    assert rebound['codebase_autoencoder'] == learned
    assert rebound['new_autoencoder_training_steps'] == 0
    assert teacher[0].read_bytes() == teacher[2]


@pytest.mark.parametrize('drift', ['selection', 'missing_selection', 'initializer'])
def test_training_asset_drift_is_rejected_before_provider(original, fork, monkeypatch, drift):
    _, state, _, _ = _prepare(original, fork)
    marker = state / 'code-learning-assets.json'
    if drift == 'missing_selection':
        marker.unlink()
    elif drift == 'selection':
        value = json.loads(marker.read_bytes())
        value['weight_transfer']['initializer_sha256'] = '0' * 64
        marker.write_text(json.dumps(value))
    else:
        path = Path(fork['output']) / 'initializer.json'
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b' ')
    _version(monkeypatch)
    calls = []
    with pytest.raises(ValueError):
        prep.plan(state, provider_callable=lambda *args, **kwargs: calls.append(args))
    assert calls == [] and not (state / 'planner-invoked.json').exists()


def test_asset_selection_cannot_silently_disable_training(original, fork):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    with pytest.raises(ValueError, match='explicit autoencoder profile'):
        prep.initial_context(state=state, weight_transfer=fork)
    assert not (state / 'code-learning-assets.json').exists()


@pytest.mark.parametrize('tamper', [False, True])
def test_canonical_cve_head_reaches_planning_and_is_revalidated(original, fork, tmp_path, monkeypatch, tamper):
    from test.api.test_security_cve_canonical_export import _fixture_export
    export, receipt, _ = _fixture_export(tmp_path, monkeypatch)
    selected = {'output': str(export), 'manifest_sha256': receipt['manifest_sha256']}
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    result = prep.initial_context(state=state, train_autoencoder=True,
        weight_transfer=fork, canonical_cve_training=selected)
    learner = result['codebase_autoencoder']
    assert learner['canonical_cve_training'] == selected
    metrics = learner['metrics']['security_candidate_training']
    assert metrics['sample_count'] == 2
    assert metrics['after']['training_bce'] < metrics['before']['training_bce']
    assert metrics['after']['holdout_evaluated'] is False
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    summary = next(row for row in loaded['summaries'] if row['schema'] == 'terminal-code-autoencoder-planning-summary@1')
    assert summary['security_candidate_nominations'] == learner['security_candidate_nominations']['rows']
    assert summary['security_candidate_projection']['execution_authority'] is False
    calls = []
    def router(prompt, **kwargs):
        calls.append(prompt)
        assert selected['manifest_sha256'] in prompt
        assert 'scores_are_calibrated_probabilities' in prompt
        assert 'security_candidate_nominations' in prompt
        return {'text': _proposal_json(prepared), 'observation': {}, 'execution_receipt': None}
    _version(monkeypatch)
    if tamper:
        path = export / 'training-pairs.json'
        path.write_bytes(path.read_bytes() + b' ')
        with pytest.raises(ValueError):
            prep.plan(state, provider_callable=router)
        assert calls == []
        return
    planned = prep.plan(state, provider_callable=router)
    assert planned['qualified'], (planned.get('failure'), planned.get('provider_receipt'), len(calls))
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder as ae
    monkeypatch.setattr(ae, 'train_codebase_autoencoder', lambda **_: pytest.fail('retraining during reuse'))
    rebound = prep.context(state=state)
    assert rebound['codebase_autoencoder'] == learner
    assert rebound['new_autoencoder_training_steps'] == 0
