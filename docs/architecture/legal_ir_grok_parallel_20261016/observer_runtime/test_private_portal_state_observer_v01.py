"""Finite native DuckDB fixtures for the read-only private-state observer.

No actual campaign catalog is opened. Native source methods create only an
explicit temporary projection/binding; fixture claims grant no task authority.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import database_portal_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.todo_daemon import portal_task_state_control_plane as state_owner
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as connection_owner

META = Path(__file__).resolve().parent
C = Path(bridge.__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('reviewed_private_portal_observer', META/'observe_private_portal_state_v01.py')
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def _pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _db_create(path, ddl, rows=()):
    connection = connection_owner.connect_duckdb_with_policy(duckdb, path)
    try:
        connection.execute(ddl)
        for sql, parameters in rows:
            connection.execute(sql, parameters)
    finally:
        connection.close()
    path.chmod(0o600)


@pytest.fixture
def case(tmp_path, monkeypatch):
    attempt_root = tmp_path/'fixture-private-attempt'
    attempt_root.mkdir(mode=0o700)
    owner = bridge.DatabasePortalExecutionBridge
    paths = owner._direct_attempt_paths(attempt_root)
    attempt = SimpleNamespace(attempt_id='fixture-native-attempt', claim_id='fixture-claim',
        task_cid='fixture-canonical-task-cid', task_alias='LIR-9001', lease_id='fixture-lease',
        fencing_token=3, fence_epoch=2, attempt_number=1, owner_session_id='fixture-owner')
    record = SimpleNamespace(task_alias='LIR-9001', revision=4, goal_cid='', plan_cid='',
        priority='P2', body={'objective':'Fixture planning only',
        'board_namespace':'legal-ir-span-graph-lean-refinement-20261016'})
    text = owner._render_projection_seed(attempt, record,
        board_namespace='legal-ir-span-graph-lean-refinement-20261016')
    binding = owner._binding(attempt, record, text)
    paths.task_projection.write_text(text); paths.task_projection.chmod(0o600)
    paths.binding.write_text(json.dumps(binding, sort_keys=True)); paths.binding.chmod(0o600)
    owner._strict_binding(paths.binding)
    owner._verify_projection(paths, binding)
    pins = {relative: hashlib.sha256((C/relative).read_bytes()).hexdigest()
            for relative in observer.SOURCE_PATHS}
    basename = observer._native_target_basename((C/observer.SOURCE_PATHS[1]).read_bytes())
    database = attempt_root/basename
    lock = database.with_name('.'+database.name+'.lock')
    lock.touch(mode=0o600)
    body = {'completion_authority':False, 'active_task_id':'LIR-9001',
        'active_task_cid':'fixture-local-portal-task-cid', 'active_task_key':'fixture-local-key',
        'active_attempt':1, 'active_phase':'implementation'}
    _db_create(database, state_owner.DDL, [(
        'INSERT INTO portal_task_state VALUES (?, ?, ?, ?, ?)',
        [paths.state.stem, json.dumps(body, sort_keys=True), 1, False, '2026-10-16T22:00:00+00:00'])])
    canonical = tmp_path/'fixture-canonical-control.duckdb'
    _db_create(canonical, 'CREATE TABLE canonical_sentinel (label VARCHAR)',
               [('INSERT INTO canonical_sentinel VALUES (?)', ['must-remain-unchanged'])])
    # Both ambient selectors intentionally disagree with the private target.
    monkeypatch.setenv(state_owner.ENV_QUACK, 'quack://127.0.0.1:65534')
    monkeypatch.setenv(state_owner.ENV_DUCKDB, str(canonical))
    expected = {key:getattr(attempt,key) for key in ('task_cid','attempt_id','claim_id','lease_id',
                                                   'fencing_token','fence_epoch')}
    expected['task_revision'] = record.revision
    kwargs = {'source_root': C, 'projection_observation': {
        'projection_file':_pin(paths.task_projection), 'native_binding_id':binding['binding_id']},
        'expected_claim':expected, 'source_sha256':pins}
    return {'kwargs':kwargs, 'paths':paths, 'database':database, 'lock':lock,
            'canonical':canonical, 'body':body, 'root':tmp_path}


def _observe(case):
    return observer.observe_private_portal_state(**case['kwargs'])


def test_reads_genuine_private_row_while_ambient_canonical_target_is_untouched(case):
    private_before, canonical_before = _pin(case['database']), _pin(case['canonical'])
    entries = sorted(path.name for path in case['paths'].root.iterdir())
    raw, receipt = _observe(case)
    assert json.loads(raw) == case['body']
    assert receipt['database_path'] == str(case['database']) and receipt['row_revision'] == 1
    assert receipt['body_sha256'] == hashlib.sha256(raw).hexdigest()
    assert receipt['read_only'] is True and receipt['native_shared_lock_observed'] is True
    assert receipt['schema_installation_attempted'] is False and receipt['canonical_database_opened'] is False
    assert receipt['claim_authority'] is False and receipt['completion_authority'] is False
    assert _pin(case['database']) == private_before and _pin(case['canonical']) == canonical_before
    assert sorted(path.name for path in case['paths'].root.iterdir()) == entries


def test_busy_native_writer_lock_is_pending_and_does_not_write(case):
    descriptor = os.open(case['lock'], os.O_RDWR|os.O_NOFOLLOW)
    before = _pin(case['database'])
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(observer.PrivatePortalStatePending, match='writer is active'):
            _observe(case)
        assert _pin(case['database']) == before
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN); os.close(descriptor)


@pytest.mark.parametrize('variant', ['absent_database','absent_table','absent_row'])
def test_unpublished_native_state_is_pending_without_creating_schema(case, variant):
    case['database'].unlink()
    if variant == 'absent_table':
        _db_create(case['database'], 'CREATE TABLE unrelated_fixture (value INTEGER)')
    elif variant == 'absent_row':
        _db_create(case['database'], state_owner.DDL)
    before = _pin(case['database']) if case['database'].exists() else None
    with pytest.raises(observer.PrivatePortalStatePending): _observe(case)
    assert (_pin(case['database']) if case['database'].exists() else None) == before


@pytest.mark.parametrize('variant', ['malformed_columns','same_name_view','authority_row','authority_body',
    'database_symlink','lock_symlink','wrong_claim','wrong_binding','changed_projection','bad_source_pin','wrong_import_origin'])
def test_refuses_schema_custody_authority_or_claim_drift(case, variant):
    kwargs = case['kwargs']
    if variant in ('malformed_columns','same_name_view'):
        case['database'].unlink()
        ddl = ('CREATE TABLE portal_task_state (lane_key VARCHAR, body_json VARCHAR)' if variant=='malformed_columns'
               else 'CREATE VIEW portal_task_state AS SELECT 1 AS lane_key')
        _db_create(case['database'], ddl)
    elif variant in ('authority_row','authority_body'):
        connection=connection_owner.connect_duckdb_with_policy(duckdb,case['database'])
        try:
            if variant=='authority_row': connection.execute('UPDATE portal_task_state SET completion_authority=true')
            else: connection.execute('UPDATE portal_task_state SET body_json=?', [json.dumps({'completion_authority':True})])
        finally: connection.close()
    elif variant in ('database_symlink','lock_symlink'):
        path = case['database'] if variant=='database_symlink' else case['lock']
        preserved = path.with_suffix(path.suffix+'.preserved')
        path.rename(preserved); path.symlink_to(preserved)
    elif variant=='wrong_claim': kwargs['expected_claim']['task_cid']='different-fixture-task'
    elif variant=='wrong_binding': kwargs['projection_observation']['native_binding_id']='different-binding'
    elif variant=='changed_projection': case['paths'].task_projection.write_text('## LIR-9001: Changed fixture\n')
    elif variant=='bad_source_pin': kwargs['source_sha256'][observer.SOURCE_PATHS[0]]='0'*64
    elif variant=='wrong_import_origin':
        alternate = case['root']/'alternate-native-source'
        for relative in observer.SOURCE_PATHS:
            destination=alternate/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(C/relative,destination);destination.chmod(0o600)
        kwargs['source_root']=alternate
    canonical_before=_pin(case['canonical'])
    # A schema mismatch is a refusal, never publication-pending.
    try:
        with pytest.raises((ValueError, bridge.DatabasePortalBridgeError, duckdb.Error)) as caught:
            _observe(case)
        assert not isinstance(caught.value, observer.PrivatePortalStatePending)
    finally:
        assert _pin(case['canonical'])==canonical_before
