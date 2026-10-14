"""Source helper dispatch, real local read/edit, retained audit, and refusal paths."""
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import source_view_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from benchmarks.agent_supervisor.container_coding.terminal_context_audit import _model_projection, _receipt
from test.api.test_semantic_router_translation import native
from test.api.test_semantic_router_integration import allocated, provider, _run


@pytest.mark.parametrize('selected', policy.POLICIES[1:])
def test_real_semantic_router_guidance_exact_audit_and_no_adoption(allocated, provider, selected):
    repository, workspace, prompt, _ = allocated
    _, seen = provider
    output, receipt = _run(repository, prompt, source_view_advisory_policy=selected)
    binding = receipt['source_view_advisory_policy']
    assert binding['requested_policy'] == selected
    assert binding['helper_adoption_guaranteed'] is False
    assert binding['token_savings_measured'] is False
    assert binding['usage']['helper_operations_observed'] is False
    assert binding['usage']['store_cleanup_complete'] is True
    assert not Path(binding['audit_store']).parent.exists()
    assert '--format ' + ('raw' if selected == policy.POLICIES[1] else 'compact') in seen[0][0]
    reconstructed, advisory, checks = _model_projection(rendered=prompt, receipt=receipt, repository=repository)
    assert reconstructed == seen[0][0] and all(checks.values())
    assert checks['source_view_helper_source'] is True
    _receipt({**receipt, 'phase': 'coding'}, workspace.parent)
    assert receipt['workspace_advisory_sha256'] == hashlib.sha256(advisory.encode()).hexdigest()
    assert output == 'literal summary'


def test_matched_guidance_differs_only_in_requested_representation(tmp_path):
    raw = policy.create_binding(workspace=tmp_path, requested_policy=policy.POLICIES[1])
    compact = {**raw, 'requested_policy': policy.POLICIES[2]}
    assert policy._instruction(raw).replace('--format raw', '--format compact') == policy._instruction(compact)


@pytest.mark.parametrize('selected', policy.POLICIES[1:])
def test_advertised_cli_reads_edits_refreshes_original_checkout(allocated, provider, selected):
    repository, workspace, prompt, source = allocated
    state, seen = provider
    observations = []
    def use_helper():
        lines = seen[-1][0].splitlines()
        advertised = next(line for line in lines if 'code_source_view.py' in line and ' read PATH ' in line)
        command = shlex.split(advertised)
        command[command.index('PATH')] = 'mod.py'
        command[command.index('NAME')] = 'lengthy_first_function'
        before = json.loads(subprocess.run(command, check=True, capture_output=True, timeout=10).stdout)
        replacement = workspace / 'replacement.py'
        replacement.write_text('def lengthy_first_function(value):\n    return value + 2\n')
        edit = shlex.split(next(line for line in lines if 'code_source_view.py' in line and ' replace PATH ' in line))
        edit[edit.index('PATH')] = 'mod.py'
        edit[edit.index('NAME')] = 'lengthy_first_function'
        edit[edit.index('HASH')] = before['source_sha256']
        edit[edit.index('PATH')] = 'replacement.py'
        changed = json.loads(subprocess.run(edit, check=True, capture_output=True, timeout=10).stdout)
        after = json.loads(subprocess.run(command, check=True, capture_output=True, timeout=10).stdout)
        assert changed['changed'] is True
        assert after['source_sha256'] != before['source_sha256']
        assert (repository / 'mod.py').read_text() == source
        assert 'return value + 2' in (workspace / 'mod.py').read_text()
        stale = subprocess.run(edit, capture_output=True, timeout=10)
        assert stale.returncode != 0
        observations.append((before, after))
    state['after_generate'] = use_helper
    _, receipt = _run(repository, prompt, source_view_advisory_policy=selected)
    assert len(observations) == 1
    usage = receipt['source_view_advisory_policy']['usage']
    assert usage['helper_operations_observed'] is True
    assert usage['store_cleanup_complete'] is True
    assert 'return value' not in json.dumps(receipt['source_view_advisory_policy'])
    assert receipt['completion_authority'] is False
    reconstructed, _, checks = _model_projection(rendered=prompt, receipt=receipt, repository=repository)
    assert reconstructed == seen[0][0] and all(checks.values())


@pytest.mark.parametrize('mutation', ['hash', 'instruction', 'workspace', 'audit', 'authority', 'extra'])
def test_binding_tampering_rejected_or_helper_mismatch(allocated, provider, mutation):
    repository, workspace, prompt, _ = allocated
    _, receipt = _run(repository, prompt, source_view_advisory_policy=policy.POLICIES[1])
    changed = copy.deepcopy(receipt)
    binding = changed['source_view_advisory_policy']
    if mutation == 'hash': binding['helper_sha256'] = '0' * 64
    elif mutation == 'instruction': binding['instruction_sha256'] = '0' * 64
    elif mutation == 'workspace': binding['workspace'] = '/foreign/work'
    elif mutation == 'audit': binding['audit_store'] = '/tmp/../foreign'
    elif mutation == 'authority': binding['completion_authority'] = True
    else: binding['extra'] = 'foreign'
    if mutation == 'hash':
        _, _, checks = _model_projection(rendered=prompt, receipt=changed, repository=repository)
        assert checks['source_view_helper_source'] is False
    else:
        with pytest.raises(ValueError):
            _model_projection(rendered=prompt, receipt=changed, repository=repository)


@pytest.mark.parametrize('selected', [None, False, {}, 'source-view-foreign@1'])
def test_unknown_modes_rejected_before_dispatch(allocated, provider, selected):
    repository, _, prompt, _ = allocated
    with pytest.raises(ValueError):
        _run(repository, prompt, source_view_advisory_policy=selected)
    assert provider[1] == []


@pytest.mark.parametrize('changes', [{'purpose': 'planning'}, {'census_only': True}, {'provider': 'grok_cli'}])
def test_helper_is_coding_only_and_needs_full_route(allocated, provider, changes):
    repository, _, prompt, _ = allocated
    args = dict(prompt=prompt, provider='codex_cli', model='pinned', timeout=1, max_output_tokens=128,
                semantic_repository=repository, source_view_advisory_policy=policy.POLICIES[1])
    args.update(changes)
    with pytest.raises(ValueError):
        runner.run(**args)
    assert provider[1] == []


def test_provider_failure_keeps_original_error_and_cleans_log(allocated, provider, capsys):
    repository, _, prompt, _ = allocated
    def fail():
        raise TimeoutError('authored failure')
    provider[0]['after_generate'] = fail
    with pytest.raises(TimeoutError, match='authored failure'):
        _run(repository, prompt, source_view_advisory_policy=policy.POLICIES[1])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
    assert receipt['status'] == 'failed'
    assert receipt['source_view_advisory_policy']['usage']['store_cleanup_complete'] is True


def test_legacy_semantic_dispatch_adds_no_binding(allocated, provider):
    repository, _, prompt, _ = allocated
    _, receipt = _run(repository, prompt)
    assert 'source_view_advisory_policy' not in receipt
    assert 'code_source_view.py' not in provider[1][0][0]


@pytest.mark.parametrize('kind', ['malformed', 'symlink', 'extra_file'])
def test_invalid_or_surplus_worker_diagnostics_are_not_trusted(tmp_path, kind):
    binding = policy.create_binding(workspace=tmp_path, requested_policy=policy.POLICIES[1])
    identity = policy.create_audit_store(binding)
    audit = Path(binding['audit_store'])
    foreign = tmp_path / 'foreign'
    foreign.write_text('preserve')
    if kind == 'malformed': audit.write_text('{"untrusted":true}\n')
    elif kind == 'symlink': audit.symlink_to(foreign)
    else: (audit.parent / 'extra').write_text('keep')
    try:
        result = policy.finish_audit_store(binding, workspace=tmp_path, expected_identity=identity)
        assert foreign.read_text() == 'preserve'
        if kind == 'extra_file':
            assert result['store_cleanup_complete'] is False
            assert (audit.parent / 'extra').read_text() == 'keep'
        else:
            assert result['helper_operations_observed'] is None
            assert result['operations'] is None
            assert result['store_cleanup_complete'] is True
    finally:
        if (audit.parent / 'extra').exists(): (audit.parent / 'extra').unlink()
        if audit.parent.exists(): audit.parent.rmdir()


def test_directory_swap_refuses_summary_and_cleanup(tmp_path):
    binding = policy.create_binding(workspace=tmp_path, requested_policy=policy.POLICIES[1])
    identity = policy.create_audit_store(binding)
    directory = Path(binding['audit_store']).parent
    moved = directory.with_name(directory.name + '-moved')
    directory.rename(moved)
    directory.mkdir(mode=0o700)
    protected = directory / 'operations.jsonl'
    protected.write_text('leave alternate directory alone')
    try:
        result = policy.finish_audit_store(binding, workspace=tmp_path, expected_identity=identity)
        assert result['helper_operations_observed'] is None
        assert result['store_cleanup_complete'] is False
        assert protected.read_text() == 'leave alternate directory alone'
    finally:
        protected.unlink()
        directory.rmdir()
        moved.rmdir()


def test_summary_exception_preserves_provider_result_and_cleans(allocated, provider, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import code_source_view
    repository, _, prompt, _ = allocated
    monkeypatch.setattr(code_source_view, 'summarize_audit_store',
        lambda *a, **kw: (_ for _ in ()).throw(ValueError('authored bad diagnostic')))
    output, receipt = _run(repository, prompt, source_view_advisory_policy=policy.POLICIES[1])
    assert output == 'literal summary'
    usage = receipt['source_view_advisory_policy']['usage']
    assert usage['helper_operations_observed'] is None
    assert usage['status'] == 'unavailable'
    assert usage['store_cleanup_complete'] is True


def test_audit_setup_failure_receipts_and_never_dispatches(allocated, provider, monkeypatch, capsys):
    repository, _, prompt, _ = allocated
    monkeypatch.setattr(policy, 'create_audit_store',
        lambda *a, **kw: (_ for _ in ()).throw(OSError('authored unavailable store')))
    with pytest.raises(OSError, match='authored unavailable store'):
        _run(repository, prompt, source_view_advisory_policy=policy.POLICIES[1])
    assert provider[1] == []
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = next(row for row in rows if row.get('schema') == 'router-implementation-invocation@1')
    assert receipt['status'] == 'failed'
    assert receipt['failure_phase'] == 'source_view_store'
    assert receipt['source_view_advisory_policy']['usage']['helper_operations_observed'] is None


@pytest.mark.parametrize("extra", [
    {"source_view_advisory_policy": "source-view-raw@1"},
    {"repository_source_policy": "repository-source-raw@1"},
    {"source_input_policy": "source-input-raw@1"},
    {"query_reference_policy": "required-instruction-ref@1"},
])
def test_signed_source_descriptor_cannot_mix_unqualified_advisory_projection(allocated, provider, extra):
    repository, workspace, prompt, _ = allocated
    with pytest.raises(ValueError, match="independently qualified projection policies"):
        _run(repository, prompt, source_view_profile="identity@1",
            source_view_descriptor=workspace / "owner-descriptor.json",
            source_view_descriptor_sha256="1" * 64,
            container_boundary=workspace / "boundary.json",
            container_boundary_sha256="2" * 64, **extra)
    assert not provider[1]
