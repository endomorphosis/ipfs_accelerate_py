"""Actual code-only training reaches the planner and is reused after admission."""
import builtins
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder as autoencoder


def _bytes(path):
    return Path(path).read_bytes()


def _training(original):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    receipt = prep.initial_context(state=state, train_autoencoder=True)
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    return root, state, prepared, receipt, loaded


def test_real_training_checkpoint_reaches_planner_without_legal_state_and_reuses_after_admission(original, tmp_path, monkeypatch):
    legal = tmp_path / 'legal-ir'
    legal.mkdir()
    checkpoint = legal / 'checkpoint.json'
    checkpoint.write_text('{"domain":"authored-legal-fixture","weights":[1,2,3]}\n')
    original_legal = checkpoint.read_bytes()
    import_calls = []
    native_import = builtins.__import__
    before_modules = set(sys.modules)
    def observe(name, globals=None, locals=None, fromlist=(), level=0):
        import_calls.append((name, tuple(fromlist or ())))
        return native_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', observe)
    root, state, prepared, receipt, loaded = _training(original)
    learner = receipt['codebase_autoencoder']
    catalog = receipt['codebase_autoencoder_catalog']
    assert catalog['hydrated'] is True
    assert catalog['metadata_ducklake']['status'] == catalog['world_ducklake']['status'] == 'projected'
    assert catalog['hydration']['catalog_count'] == catalog['hydration']['linked_count'] == 4
    metrics = learner['metrics']
    assert learner['domain'] == 'security-code@1'
    assert learner['output'] == str(state / 'code-autoencoder')
    assert metrics['initial_weights_sha256'] != metrics['final_weights_sha256']
    assert metrics['native_kernel_calls'] > 0
    assert any(value > 0 for value in metrics['native_gradient_norms'])
    assert metrics['holdout_evaluated'] is False
    assert metrics['after_reconstruction_loss'] < metrics['before_reconstruction_loss']
    assert learner['epochs_completed'] == metrics['epochs']
    assert receipt['nonoverlapping_seconds']['codebase_autoencoder_training'] > 0
    assert set(learner['source_hashes']) == {'bottle.py', prep.SMOKE}
    summaries = [row for row in loaded['summaries'] if row['schema'] == 'terminal-code-autoencoder-planning-summary@1']
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary['checkpoint_sha256'] == learner['checkpoint_sha256']
    assert summary['receipt_sha256'] == learner['receipt_sha256']
    assert summary['ranked_candidates']
    assert summary['formal_translation_authority'] is False
    assert summary['legal_ir_state_reused'] is False
    assert summary['nomination_only'] is True
    assert summary['catalog_receipt_sha256'] == catalog['receipt_sha256']
    assert summary['world_record_cid'] == catalog['hydration']['world_record_cid']
    assert summary['semantic_equivalence_claimed'] is False
    parameters = json.loads((state / 'code-autoencoder/checkpoint.json').read_bytes())
    assert parameters['legal_ir_weights_loaded'] is parameters['legal_ir_views_loaded'] is False
    assert parameters['projection_family'] == 'code-ast-control-flow-contract-advisory@1'
    assert parameters['tla_projection']['status'] == 'unsupported'
    files = {path.name: path.read_bytes() for path in (state / 'code-autoencoder').iterdir() if path.is_file()}
    requests = []
    def router(prompt, **_kwargs):
        requests.append(prompt)
        assert learner['checkpoint_sha256'] in prompt and learner['receipt_sha256'] in prompt
        assert 'terminal-code-autoencoder-planning-summary@1' in prompt
        assert '"formal_translation_authority":false' in prompt.replace(' ', '')
        return {'text': _proposal_json(prepared), 'observation': {}, 'execution_receipt': None}
    _version(monkeypatch)
    planned = prep.plan(state, provider_callable=router)
    assert planned['qualified'], planned
    assert len(requests) == 1
    monkeypatch.setattr(autoencoder, 'train_codebase_autoencoder', lambda **_:
                        pytest.fail('post-admission context must not retrain'))
    result = prep.context(state=state)
    assert result['initial_indexes_reused'] is True and result['new_embedding_calls'] == 0
    assert result['codebase_autoencoder'] == learner
    assert result['new_autoencoder_training_steps'] == 0
    reused = json.loads((root / '.runtime/terminal-context/result.json').read_bytes())
    assert result['codebase_autoencoder_catalog'] == reused['codebase_autoencoder_catalog'] == catalog
    assert reused['codebase_autoencoder'] == learner
    assert reused['new_autoencoder_training_steps'] == 0
    assert files == {path.name: path.read_bytes() for path in (state / 'code-autoencoder').iterdir() if path.is_file()}
    assert checkpoint.read_bytes() == original_legal
    assert sorted(path.name for path in legal.iterdir()) == ['checkpoint.json']
    assert not any('.legal_' in name or '.legalir' in name or '.legal_ir' in name
                   for name in set(sys.modules) - before_modules)
    assert not any('.legal_' in name or any(item.startswith('legal_') for item in fromlist)
                   for name, fromlist in import_calls)


@pytest.mark.parametrize('mutation', ['source', 'checkpoint', 'projection'])
def test_autoencoder_source_or_checkpoint_drift_refuses_before_provider(original, monkeypatch, mutation):
    root, state, _, receipt, _ = _training(original)
    if mutation == 'source':
        (root / 'bottle.py').write_bytes((root / 'bottle.py').read_bytes() + b'\n# changed\n')
    else:
        path = state / 'code-autoencoder/checkpoint.json'
        path.chmod(0o600)
        if mutation == 'checkpoint':
            path.write_bytes(path.read_bytes() + b' ')
        else:
            data = json.loads(path.read_bytes())
            data['tla_projection'] = {'status': 'proved'}
            path.write_text(json.dumps(data))
    calls = []
    _version(monkeypatch)
    with pytest.raises(ValueError):
        prep.plan(state, provider_callable=lambda *args, **kwargs: calls.append(args))
    assert calls == []
    assert not (state / 'planner-invoked.json').exists()


def test_descriptor_cannot_retarget_advisory_to_legal_checkpoint_namespace(original, tmp_path, monkeypatch):
    root, state, _, receipt, loaded = _training(original)
    legal = tmp_path / 'legal-ir'
    legal.mkdir()
    legal_checkpoint = legal / 'checkpoint.json'
    legal_checkpoint.write_text('{"domain":"authored-legal-fixture"}\n')
    original_bytes = legal_checkpoint.read_bytes()
    # Even matching mutable descriptor/receipt digests cannot change the
    # independently selected code namespace checked before the provider call.
    descriptor = loaded['descriptor']
    descriptor['codebase_autoencoder']['output'] = str(legal)
    receipt['codebase_autoencoder']['output'] = str(legal)
    path = root / receipt['descriptor']['artifact']
    raw = json.dumps(descriptor, sort_keys=True).encode()
    path.write_bytes(raw)
    receipt['descriptor']['sha256'] = hashlib.sha256(raw).hexdigest()
    (state / 'initial-context-result.json').write_text(json.dumps(receipt))
    calls = []
    _version(monkeypatch)
    with pytest.raises(ValueError, match='isolated namespace'):
        prep.plan(state, provider_callable=lambda *args, **kwargs: calls.append(args))
    assert calls == [] and not (state / 'planner-invoked.json').exists()
    assert legal_checkpoint.read_bytes() == original_bytes


def test_old_profile_does_not_train_autoencoder_without_explicit_profile_option(original, monkeypatch):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    monkeypatch.setattr(autoencoder, 'train_codebase_autoencoder', lambda **_:
                        pytest.fail('legacy profile must not silently add training'))
    result = prep.initial_context(state=state)
    assert 'codebase_autoencoder' not in result
    assert not (state / 'code-autoencoder').exists()


def test_published_semantic_refresh_does_not_relabel_initial_autoencoder_as_current():
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    learner = {'checkpoint_sha256': '1' * 64, 'receipt_sha256': '2' * 64,
               'source_hashes': {'source.py': '3' * 64}}
    report = {'arm': 'full', 'task_state': {'status': 'completed'}, 'stop': {'status': 'succeeded'},
              'remaining_processes': 0, 'phases': {}, 'initial_context': {'codebase_autoencoder': learner}}
    runtime = SimpleNamespace(refresh_after_stop=lambda: pytest.fail('no rebuild budget'))
    driver._refresh_completed_context(runtime, report, deadline=time.monotonic() + 1)
    assert report['post_publication_context']['status'] == 'deferred'
    observation = report['post_publication_autoencoder']
    assert observation['status'] == 'historical'
    assert observation['checkpoint_sha256'] == learner['checkpoint_sha256']
    assert observation['initial_source_hashes'] == learner['source_hashes']
    assert observation['freshness_checked'] is False
    assert observation['current_source_reuse_authority'] is False
    assert observation['new_autoencoder_training_steps'] == 0
    assert observation['proof_authority'] is observation['completion_authority'] is False
    observation['initial_source_hashes']['source.py'] = '4' * 64
    assert learner['source_hashes']['source.py'] == '3' * 64


@pytest.mark.parametrize('changed', [{'task_state': {'status': 'in_progress'}},
                                    {'stop': {'status': 'failed'}}, {'remaining_processes': 1}])
def test_historical_autoencoder_observation_requires_completed_published_shutdown(changed):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    report = {'arm': 'full', 'task_state': {'status': 'completed'}, 'stop': {'status': 'succeeded'},
              'remaining_processes': 0, 'initial_context': {'codebase_autoencoder': {}}, **changed}
    driver._refresh_completed_context(SimpleNamespace(), report, deadline=time.monotonic())
    assert 'post_publication_autoencoder' not in report
