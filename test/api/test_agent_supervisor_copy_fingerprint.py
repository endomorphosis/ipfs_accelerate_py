"""Real Git copy detection must not change the validated mutation identity."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry, DiffChangeKind, collect_git_candidate_diff,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()


def collect(root: Path, base: str):
    return collect_git_candidate_diff(root, base_revision=base)


def fingerprint(entries) -> str:
    return PortalImplementationDaemon._proposal_candidate_fingerprint(entries)


@pytest.fixture
def candidate(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'Inert test')
    git(tmp_path, 'config', 'user.email', 'inert@example.invalid')
    git(tmp_path, 'config', 'core.fileMode', 'true')
    source = tmp_path / 'source.txt'
    source.write_text(''.join(f'unchanged line {i:03d}\n' for i in range(100)))
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'base')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    source.write_text(source.read_text() + 'one modified source line\n')
    dest = tmp_path / 'snapshot.txt'
    dest.write_bytes(source.read_bytes())
    return tmp_path, base, source, dest


def test_real_copy_detection_untracked_staged_committed_identity(candidate):
    root, base, source, dest = candidate
    untracked = collect(root, base)
    assert next(e for e in untracked if e.new_path == 'snapshot.txt').change_kind == DiffChangeKind.ADD
    expected = fingerprint(untracked)
    git(root, 'add', '-A')
    staged = collect(root, base)
    copied = next(e for e in staged if e.new_path == 'snapshot.txt')
    assert copied.change_kind == DiffChangeKind.COPY
    assert copied.old_path == 'source.txt'
    before = copy.deepcopy(copied.to_dict(include_sources=True))
    assert fingerprint(staged) == expected
    assert copied.to_dict(include_sources=True) == before  # proposal provenance is untouched
    git(root, 'commit', '-qm', 'candidate')
    assert fingerprint(collect(root, base)) == expected
    assert git(root, 'status', '--porcelain') == ''


@pytest.mark.parametrize('mutation', ['destination_bytes', 'destination_mode', 'destination_path', 'source_bytes', 'source_mode'])
def test_real_mutation_changes_fingerprint(candidate, mutation):
    root, base, source, dest = candidate
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'candidate')
    expected = fingerprint(collect(root, base))
    if mutation == 'destination_bytes':
        dest.write_text(dest.read_text() + 'different output\n')
    elif mutation == 'destination_mode':
        dest.chmod(0o755)
    elif mutation == 'destination_path':
        dest.rename(root / 'different-path.txt')
    elif mutation == 'source_bytes':
        source.write_text(source.read_text() + 'different source\n')
    else:
        source.chmod(0o755)
    changed = collect(root, base)
    assert fingerprint(changed) != expected
    if mutation.endswith('mode'):
        target = 'snapshot.txt' if mutation.startswith('destination') else 'source.txt'
        assert next(e for e in changed if e.new_path == target).metadata['after_mode'] == '100755'


def test_copy_normalization_keeps_other_metadata_and_binary_identity():
    entry = CandidateDiffEntry(old_path='source', new_path='dest', change_kind=DiffChangeKind.COPY,
                               before_source='old', after_source='new', before_blob_id='before',
                               after_blob_id='after', metadata={'before_mode':'100644', 'after_mode':'100644',
                                                              'authority_binding':'exact'})
    expected = fingerprint([entry])
    from dataclasses import replace
    assert fingerprint([replace(entry, metadata={**entry.metadata, 'authority_binding':'changed'})]) != expected
    assert fingerprint([replace(entry, binary=True)]) != expected
    assert fingerprint([replace(entry, after_blob_id='changed')]) != expected
    assert fingerprint([replace(entry, generated=True)]) != expected


def test_rename_representation_is_not_normalized_to_add():
    rename = CandidateDiffEntry(old_path='old', new_path='new', change_kind=DiffChangeKind.RENAME,
                                before_source='same', after_source='same', before_blob_id='same', after_blob_id='same')
    added = CandidateDiffEntry(new_path='new', change_kind=DiffChangeKind.ADD, after_source='same', after_blob_id='same')
    assert fingerprint([rename]) != fingerprint([added])


def test_symlink_mode_is_bound_without_following_source_target(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'Inert test')
    git(tmp_path, 'config', 'user.email', 'inert@example.invalid')
    (tmp_path/'seed').write_text('seed')
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'base')
    (tmp_path/'link').symlink_to('seed')
    entries = collect_git_candidate_diff(tmp_path)
    assert next(e for e in entries if e.new_path == 'link').metadata['after_mode'] == '120000'


def test_actual_handoff_guard_accepts_copy_commit_but_rejects_committed_mode_mutation(candidate):
    from types import SimpleNamespace
    root, base, source, dest = candidate
    daemon = object.__new__(PortalImplementationDaemon)
    # This inert repository has no submodules, ignored outputs, or native state.
    # Exercise the real workspace identity, source collector, fingerprint and
    # pre/post guard; only event recording and those absent integrations are inert.
    daemon._stage_declared_ignored_outputs = lambda *args: None
    daemon._proposal_scope_paths = lambda task: ('source.txt', 'snapshot.txt')
    daemon._collect_proposal_candidate_diff = lambda workspace, **kw: (
        tuple(collect_git_candidate_diff(workspace, base_revision=kw['baseline_ref'])), ()
    )
    daemon._record_event = lambda *args: None
    task = SimpleNamespace(task_id='INERT-COPY')
    expected = fingerprint(collect(root, base))
    binding = {'candidate_binding': {'verified': True, 'expected_fingerprint': expected,
               'current_fingerprint': expected, 'validated_workspace': daemon._candidate_workspace_identity(root)}}
    branch = git(root, 'branch', '--show-current')
    arguments = dict(attempt=1, baseline_ref=base, expected_branch=branch, validation_result=binding)
    before = daemon._validated_candidate_handoff_guard(root, task, phase='pre_commit', **arguments)
    assert before['allowed'], before['reasons']
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'candidate')
    committed = git(root, 'rev-parse', 'HEAD')
    after = daemon._validated_candidate_handoff_guard(root, task, phase='post_commit', implementation_commit=committed, **arguments)
    assert after['allowed'], after['reasons']
    dest.chmod(0o755)
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'unvalidated executable bit')
    changed = git(root, 'rev-parse', 'HEAD')
    rejected = daemon._validated_candidate_handoff_guard(root, task, phase='post_commit', implementation_commit=changed, **arguments)
    assert rejected['allowed'] is False
    assert rejected['reasons'] == ['candidate_fingerprint_changed']


def test_git_mode_uses_literal_paths_in_tree_and_index(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import _git_entry_mode
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'Inert test')
    git(tmp_path, 'config', 'user.email', 'inert@example.invalid')
    literal = 'notes[1]*?.txt'
    lookalike = 'notes1ab.txt'
    (tmp_path/literal).write_text('literal file\n')
    (tmp_path/lookalike).write_text('different wildcard match\n')
    git(tmp_path, 'add', '-A')
    git(tmp_path, 'commit', '-qm', 'literal paths')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    (tmp_path/literal).write_text('modified literal file\n')
    entries = collect(tmp_path, base)
    assert [e.new_path for e in entries] == [literal]
    assert entries[0].metadata == {'before_mode':'100644', 'after_mode':'100644'}
    # Two uninitialized gitlinks, one of which matches the other's glob syntax.
    git(tmp_path, 'update-index', '--add', '--cacheinfo', f'160000,{base},module[1]*?')
    git(tmp_path, 'update-index', '--add', '--cacheinfo', f'160000,{base},module1ab')
    assert _git_entry_mode(tmp_path, None, 'module[1]*?') == '160000'
    assert _git_entry_mode(tmp_path, None, 'module1ab') == '160000'
