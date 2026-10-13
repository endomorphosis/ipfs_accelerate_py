"""Native numerical stores with authored publication bindings, no model calls.

Actual completed owner and STOP admission are exercised by lifecycle tests.
"""
from copy import deepcopy
import hashlib
import subprocess
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
from ipfs_accelerate_py.agent_supervisor.runtime import published_numerical as successor


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture(params=native.PROFILES)
def authored(tmp_path, request):
    root = tmp_path / 'repository'
    root.mkdir()
    (root / 'answer.py').write_text('def answer(value):\n    return value + 1\n')
    (root / 'data.txt').write_text('authored task data\n')
    git(root, 'init', '-q')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Authored', '-c', 'user.email=authored@example.invalid', 'commit', '-qm', 'baseline')
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ('answer.py', 'data.txt')}
    manifest = dict(baseline_commit=git(root, 'rev-parse', 'HEAD'),
        sources={name: {'sha256': sha} for name, sha in hashes.items()},
        tasks=[{'outputs': [{'path': 'answer.py'}, {'path': 'new.py'}]}])
    previous = native.prepare_numerical_context(repository=root,
        source_hashes={'answer.py': hashes['answer.py']}, output=tmp_path / 'previous', profile=request.param)
    output_root = tmp_path / 'successors'
    output_root.mkdir(mode=0o700)
    policy = successor.bind_policy(repository=root, receipt=previous, manifest=manifest, output_root=output_root)
    return dict(root=root, previous=previous, policy=policy, manifest=manifest, output=output_root / 'attempt-1')


def publish(case):
    root = case['root']
    (root / 'answer.py').write_text('def answer(value):\n    return value + 2\n')
    (root / 'new.py').write_text('def outside_initial_scope():\n    return 9\n')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Authored', '-c', 'user.email=authored@example.invalid', 'commit', '-qm', 'publication')
    return dict(task_cid='task:authored', task_id='authored-task', task_status='completed',
        published_commit=git(root, 'rev-parse', 'HEAD'), accepted_source_tree_id=git(root, 'rev-parse', 'HEAD^{tree}'),
        completion_receipt_cid='receipt:authored')


def prepare(case, binding, **kwargs):
    return successor.prepare_successor(repository=case['root'], predecessor_receipt=case['previous'],
        policy=case['policy'], output=case['output'], publication_binding=binding, **kwargs)


def test_regenerates_exact_python_subset_and_reloads_native_ducklake(authored):
    case = authored
    original_receipt = (case['previous']['output'] + '/receipt.json')
    from pathlib import Path
    previous_bytes = Path(original_receipt).read_bytes()
    binding = publish(case)
    successor.validate_predecessor(repository=case['root'], predecessor_receipt=case['previous'], policy=case['policy'])
    with pytest.raises(ValueError, match='source'):
        native.validate_numerical_context(repository=case['root'], expected_receipt=case['previous'])
    result = prepare(case, binding)
    receipt, observation = result['numerical_formalization'], result['numerical_refresh']
    assert receipt['profile'] == case['previous']['profile'] == case['policy']['profile']
    assert receipt['summary']['profile'] == receipt['profile']
    assert set(receipt['source_hashes']) == {'answer.py'}
    assert receipt['source_hashes'] != case['previous']['source_hashes']
    assert receipt['producers'] == case['previous']['producers']
    assert observation['declared_outputs_outside_index_scope'] == ['new.py']
    assert observation['source_obligations_discharged'] == 0
    assert observation['scope_expanded'] is observation['proof_authority'] is False
    assert observation['model_inference_calls'] == observation['provider_calls'] == 0
    assert observation['requires_independent_manifest'] is True
    assert Path(original_receipt).read_bytes() == previous_bytes
    assert successor.validate_successor(repository=case['root'], predecessor_receipt=case['previous'],
        policy=case['policy'], receipt=receipt, observation=observation, publication_binding=binding) == receipt
    with pytest.raises(ValueError, match='external numerical'):
        prepare(case, binding)  # Existing per-attempt output can never be overwritten.


@pytest.mark.parametrize('change', ['profile', 'sources', 'producer', 'receipt', 'helper', 'authority', 'output'])
def test_policy_drift_refuses_before_successor_writes(authored, change):
    binding = publish(authored)
    policy = authored['policy']
    if change == 'profile':
        policy['profile'] = 'another-profile'
    elif change == 'sources':
        policy['source_hashes']['new.py'] = 'a' * 64
    elif change == 'producer':
        policy['producers']['grammar'] = 'a' * 64
    elif change == 'receipt':
        policy['predecessor_receipt_sha256'] = 'a' * 64
    elif change == 'helper':
        policy['successor_helper_sha256'] = 'a' * 64
    elif change == 'authority':
        policy['provider_calls'] = False  # Typed counter equality, not Python 0 == False.
    else:
        policy['output_root'] = authored['previous']['output']
    with pytest.raises(ValueError):
        prepare(authored, binding)
    assert not authored['output'].exists()


@pytest.mark.parametrize('change', ['baseline', 'source_population', 'source_digest'])
def test_binding_requires_signed_initial_source_subset(authored, change):
    manifest = deepcopy(authored['manifest'])
    if change == 'baseline':
        manifest['baseline_commit'] = '0' * 40
    elif change == 'source_population':
        del manifest['sources']['answer.py']
    else:
        manifest['sources']['answer.py']['sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='signed baseline'):
        successor.bind_policy(repository=authored['root'], receipt=authored['previous'],
            manifest=manifest, output_root=authored['output'].parent)


@pytest.mark.parametrize('change', ['baseline', 'population', 'digest', 'excluded', 'manifest_cid'])
def test_historical_admission_join_cannot_be_forged_by_a_manifest_cid(authored, change):
    publish(authored)
    policy = deepcopy(authored['policy'])
    # The original signed declaration remains independently supplied after
    # publication; changing HEAD or its source must not rewrite that scope.
    successor.validate_manifest(policy=policy, manifest=authored['manifest'])
    if change == 'baseline':
        policy['baseline_commit'] = git(authored['root'], 'rev-parse', 'HEAD')
    elif change == 'population':
        policy['source_hashes']['new.py'] = 'a' * 64
    elif change == 'digest':
        policy['source_hashes']['answer.py'] = hashlib.sha256((authored['root'] / 'answer.py').read_bytes()).hexdigest()
    elif change == 'excluded':
        policy['excluded_new_output_paths'] = []
    else:
        policy['manifest_cid'] = 'foreign-manifest'
    with pytest.raises(ValueError, match='admi'):
        successor.validate_manifest(policy=policy, manifest=authored['manifest'])


@pytest.mark.parametrize('change', ['source', 'lineage', 'missing_family_path', 'expanded_family_path', 'owner', 'artifact'])
def test_reload_refuses_stale_or_tampered_successor(authored, change):
    binding = publish(authored)
    result = prepare(authored, binding)
    receipt, observation = result['numerical_formalization'], result['numerical_refresh']
    if change == 'source':
        (authored['root'] / 'answer.py').write_text('different source')
    elif change == 'lineage':
        observation['publication_binding_cid'] = 'foreign'
    elif change == 'missing_family_path':
        receipt['source_hashes'].clear()
    elif change == 'expanded_family_path':
        receipt['source_hashes']['new.py'] = hashlib.sha256((authored['root'] / 'new.py').read_bytes()).hexdigest()
    elif change == 'owner':
        binding['completion_receipt_cid'] = 'receipt:foreign'
    else:
        (authored['output'] / 'planning-summary.json').write_text('{}')
    with pytest.raises(ValueError):
        successor.validate_successor(repository=authored['root'], predecessor_receipt=authored['previous'],
            policy=authored['policy'], receipt=receipt, observation=observation, publication_binding=binding)


@pytest.mark.parametrize('change', ['not_completed', 'foreign_head', 'expired', 'nonfinite', 'source_symlink', 'foreign_output'])
def test_preparation_refuses_invalid_transition_or_resource_boundary(authored, change, tmp_path):
    binding = publish(authored)
    kwargs = {}
    if change == 'not_completed':
        binding['task_status'] = 'running'
    elif change == 'foreign_head':
        binding['published_commit'] = '0' * 40
    elif change == 'expired':
        kwargs['deadline_monotonic'] = time.monotonic() - 1
    elif change == 'nonfinite':
        kwargs['deadline_monotonic'] = float('nan')
    elif change == 'source_symlink':
        (authored['root'] / 'answer.py').unlink()
        (authored['root'] / 'answer.py').symlink_to(authored['root'] / 'new.py')
    else:
        authored['output'] = tmp_path / 'not-admitted'
    with pytest.raises((ValueError, TimeoutError)):
        prepare(authored, binding, **kwargs)
    assert not authored['output'].exists()
