"""Private-filesystem denial controls; synthetic metadata grants no authority.

Run only under the root's genuine finite native envelope. Positive cases use
real private files and no stand-in daemon, connection, lifecycle or native owner.
"""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import stat
from types import ModuleType

import pytest

SOURCE = Path(__file__).parent.parent/'attempt-disposition-v01/proposed/ipfs_accelerate_py/agent_supervisor/todo_daemon/attempt_disposition.py'
SPEC = importlib.util.spec_from_file_location('private_FS_attempt_disposition_source',SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def denial(tmp_path,*,reason='provider_output_unresolved',cid='bafy-test-revision-one'):
    # These are inert test metadata; none is accepted as an actual source join.
    return M.make_denial(
        scope=dict(repo_root=str(tmp_path/'repo'),state_path=str(tmp_path/'state.duckdb'),
                   task_id='TEST-1',canonical_task_cid=cid,board_namespace='test-board',board_sha256='a'*64),
        attempt=dict(ordinal=3,task_count=3,cid_count=3,configured_cap=3),
        lifecycle=dict(record_id='test-record',lease_id='test-lease',fence=9,
                       workspace_path=str(tmp_path/'workspace'),branch='implementation/test',
                       owner_pid=os.getpid(),owner_birth=1,owner_boot_id='test-boot',snapshot_sha256='b'*64),
        kind='parked_failure',reason_code=reason,source_sha256={str(SOURCE):M.sha(SOURCE.read_bytes())})


def directory(tmp_path):
    path=tmp_path/'journals';path.mkdir(mode=0o700)
    return path


def write_journal(path,value,*,raw=None,name=None,mode=0o600):
    target=path/(name or M.revision_key(value['scope'])+'.json')
    target.write_bytes(M.wire(value) if raw is None else raw);target.chmod(mode)
    return target


def remove_exact_fixture_alias(path, captured):
    """Remove only the same link this test created, including on assertion failure."""
    observed=path.lstat()
    if ((observed.st_dev,observed.st_ino,observed.st_uid,stat.S_IFMT(observed.st_mode))
            != (captured.st_dev,captured.st_ino,captured.st_uid,stat.S_IFMT(captured.st_mode))
            or observed.st_uid!=os.geteuid()):
        raise RuntimeError('fixture_created_alias_identity_changed')
    path.unlink()


def test_real_private_immutable_publication_has_canonical_single_link_0600_file(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    with M.DispositionJournal(path) as store:
        assert store.publish(value)==value
        assert store.denial_for_revision(value['scope'],prospective_attempt=4)==value
    files=list(path.iterdir());assert len(files)==1
    meta=files[0].lstat()
    assert stat.S_IMODE(meta.st_mode)==0o600 and meta.st_nlink==1
    assert files[0].read_bytes()==M.wire(value)


def test_exact_publication_is_idempotent_without_file_replacement(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    with M.DispositionJournal(path) as store:
        first=store.publish(value);target=path/(M.revision_key(value['scope'])+'.json')
        before=target.stat()
        assert store.publish(copy.deepcopy(value))==first
        after=target.stat()
    assert (before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns)


def test_same_revision_conflict_preserves_original_bytes(tmp_path):
    first=denial(tmp_path);other=denial(tmp_path,reason='different_original_failure')
    path=directory(tmp_path)
    with M.DispositionJournal(path) as store:
        store.publish(first)
        with pytest.raises(M.AttemptDispositionRefusal,match='conflicts'):store.publish(other)
        assert store.denial_for_revision(first['scope'])==first
    assert all(not entry.name.startswith('.pending-') for entry in path.iterdir())


def test_revision_denial_is_not_filtered_by_later_attempt_ordinal(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    with M.DispositionJournal(path) as store:
        store.publish(value)
        for attempt in (3,4,999999):
            assert store.denial_for_revision(value['scope'],prospective_attempt=attempt)['parking_id']==value['parking_id']


def test_display_alias_cannot_change_key_or_turn_existing_denial_into_absence(tmp_path):
    value=denial(tmp_path);alias=copy.deepcopy(value['scope']);alias['task_id']='ALIAS-77'
    assert M.revision_key(alias)==M.revision_key(value['scope'])
    with M.DispositionJournal(directory(tmp_path)) as store:
        store.publish(value)
        with pytest.raises(M.AttemptDispositionRefusal,match='scope_changed'):store.denial_for_revision(alias)


def test_board_status_digest_change_cannot_bypass_existing_revision_denial(tmp_path):
    value=denial(tmp_path);changed=copy.deepcopy(value['scope']);changed['board_sha256']='c'*64
    assert M.revision_key(changed)==M.revision_key(value['scope'])
    with M.DispositionJournal(directory(tmp_path)) as store:
        store.publish(value)
        with pytest.raises(M.AttemptDispositionRefusal,match='scope_changed'):store.denial_for_revision(changed)


def test_distinct_revision_records_still_coarsely_deny_same_original_state(tmp_path):
    first=denial(tmp_path);second=denial(tmp_path,cid='bafy-distinct-revision')
    assert M.revision_key(first['scope'])!=M.revision_key(second['scope'])
    with M.DispositionJournal(directory(tmp_path)) as store:
        store.publish(first);store.publish(second)
        with pytest.raises(M.AttemptDispositionDenied) as caught:store.assert_no_denials_for_state(first['scope']['state_path'])
        assert set(caught.value.parking_ids)=={first['parking_id'],second['parking_id']}
        assert caught.value.retry_authorized is caught.value.acceptance_authority is False


def test_changed_content_identity_is_rejected_before_publication(tmp_path):
    value=denial(tmp_path);value['reason_code']='changed_without_new_digest'
    path=directory(tmp_path)
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal,match='content_identity'):store.publish(value)
    assert list(path.iterdir())==[]


def test_positive_authority_flag_is_never_a_valid_denial(tmp_path):
    value=denial(tmp_path);value['cleanup_allowed']=True
    with pytest.raises(M.AttemptDispositionRefusal,match='grant_no_authority'):M.validate_denial(value)


def test_missing_original_attempt_count_is_not_accepted(tmp_path):
    value=denial(tmp_path);value['attempt']['cid_count']=2
    with pytest.raises(M.AttemptDispositionRefusal,match='consumed_original_ordinal'):M.validate_denial(value)


def test_duplicate_JSON_keys_are_not_a_valid_existing_journal(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    raw=M.wire(value).replace(b'{',b'{"schema":"duplicate",',1)
    write_journal(path,value,raw=raw)
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal):store.assert_no_denials_for_state(value['scope']['state_path'])


def test_noncanonical_existing_JSON_is_refused_not_rewritten(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    raw=(json.dumps(value,indent=2)+'\n').encode();target=write_journal(path,value,raw=raw)
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal,match='canonical_revision'):store.publish(value)
    assert target.read_bytes()==raw


def test_wrong_revision_filename_is_refused(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    write_journal(path,value,name='0'*64+'.json')
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal):store.denials_for_state(value['scope']['state_path'])


def test_nonprivate_directory_is_refused(tmp_path):
    path=directory(tmp_path);path.chmod(0o755)
    with pytest.raises(M.AttemptDispositionRefusal,match='private_journal'):M.DispositionJournal(path)


def test_symlink_directory_is_refused(tmp_path):
    path=directory(tmp_path);alias=tmp_path/'journal-alias';alias.symlink_to(path,target_is_directory=True)
    captured=alias.lstat()
    try:
        with pytest.raises(M.AttemptDispositionRefusal,match='unaliased'):M.DispositionJournal(alias)
    finally:
        remove_exact_fixture_alias(alias,captured)


def test_symlink_record_is_refused_without_following(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    outside=tmp_path/'outside';outside.write_bytes(M.wire(value));outside.chmod(0o600)
    target=path/(M.revision_key(value['scope'])+'.json');target.symlink_to(outside)
    captured=target.lstat()
    try:
        with M.DispositionJournal(path) as store:
            with pytest.raises((OSError,M.AttemptDispositionRefusal)):store.denials_for_state(value['scope']['state_path'])
        assert outside.read_bytes()==M.wire(value)
    finally:
        remove_exact_fixture_alias(target,captured)


def test_hard_link_record_is_refused(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path);target=write_journal(path,value)
    extra=tmp_path/'other-link';os.link(target,extra);captured=extra.lstat()
    try:
        with M.DispositionJournal(path) as store:
            with pytest.raises(M.AttemptDispositionRefusal,match='single_link'):store.publish(value)
    finally:
        remove_exact_fixture_alias(extra,captured)


def test_incomplete_publication_marker_denies_even_when_final_record_missing(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path)
    pending=path/('.pending-'+'1'*32);pending.write_bytes(b'partial');pending.chmod(0o600)
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal,match='incomplete_journal'):store.denial_for_revision(value['scope'])
    assert pending.read_bytes()==b'partial'


def test_unknown_directory_population_is_not_treated_as_no_denial(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path);(path/'unexpected').write_bytes(b'?')
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal,match='unknown_journal'):store.assert_no_denials_for_state(value['scope']['state_path'])


def test_record_directory_entry_is_refused_without_recursing(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path);(path/(M.revision_key(value['scope'])+'.json')).mkdir(mode=0o700)
    with M.DispositionJournal(path) as store:
        with pytest.raises((OSError,M.AttemptDispositionRefusal)):store.denials_for_state(value['scope']['state_path'])


def test_record_mode_change_fails_closed(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path);write_journal(path,value,mode=0o644)
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal):store.denial_for_revision(value['scope'])


def test_original_open_directory_replacement_is_refused(tmp_path):
    path=directory(tmp_path)
    with M.DispositionJournal(path) as store:
        path.rename(tmp_path/'old-journals');path.mkdir(mode=0o700)
        with pytest.raises(M.AttemptDispositionRefusal,match='directory_changed'):store.denials_for_state(tmp_path/'state.duckdb')


def test_journal_missing_state_is_absence_of_denial_only(tmp_path):
    with M.DispositionJournal(directory(tmp_path)) as store:
        assert store.assert_no_denials_for_state(tmp_path/'unrelated-state.duckdb') is None
    # No returned True/token/lease may be confused with permission to execute.


def test_source_join_refuses_projected_or_uninstalled_modules():
    with pytest.raises(M.AttemptDispositionRefusal,match='actual_loaded_original_modules'):
        M.OriginalDispositionScope({},ModuleType(M.LIFECYCLE_MODULE),object(),source_sha256={})


def test_no_product_import_or_provider_native_connection_execution_in_helper_source():
    tree=ast.parse(SOURCE.read_text())
    names=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):names.update(row.name.split('.')[0] for row in node.names)
        elif isinstance(node,ast.ImportFrom):names.add((node.module or '').split('.')[0])
    assert not names.intersection({'ipfs_accelerate_py','ipfs_datasets_py','subprocess','socket','duckdb','sqlite3','docker'})
    forbidden={'quarantine_current_owner','mark_terminal','compare_and_delete','run_ordinary_with_success_handoff',
               '_mark_implementation_finished','_restore_task_attempt','_record_task_attempt','acquire_child'}
    assert not {node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute)}.intersection(forbidden)


def test_original_state_gate_checks_quarantine_after_optional_journal_read():
    tree=ast.parse(SOURCE.read_text());scope=next(node for node in tree.body if isinstance(node,ast.ClassDef)
                                               and node.name=='OriginalDispositionScope')
    method=next(node for node in scope.body if isinstance(node,ast.FunctionDef) and node.name=='assert_no_denials_for_state')
    optional=next(node for node in method.body if isinstance(node,ast.Try))
    inventories=[node for node in ast.walk(method) if isinstance(node,ast.Call)
                 and isinstance(node.func,ast.Attribute) and node.func.attr=='iter_quarantines']
    assert len(inventories)==1 and inventories[0].lineno>optional.end_lineno
    assert not any(isinstance(node,ast.Return) for node in ast.walk(optional))


def test_oversized_existing_journal_is_refused_without_rewrite(tmp_path):
    value=denial(tmp_path);path=directory(tmp_path);raw=b'x'*(M.MAX_DOCUMENT+1)
    target=write_journal(path,value,raw=raw)
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal):store.publish(value)
    assert target.read_bytes()==raw


def test_nonfinite_or_float_JSON_values_are_refused():
    for raw in (b'{"a":NaN}',b'{"a":Infinity}',b'{"a":1.5}'):
        with pytest.raises(M.AttemptDispositionRefusal):M.strict(raw)


def test_inventory_overflow_fails_closed(tmp_path):
    path=directory(tmp_path)
    for index in range(M.MAX_RECORDS+1):
        (path/(f'{index:064x}.json')).write_bytes(b'{}\n')
    with M.DispositionJournal(path) as store:
        with pytest.raises(M.AttemptDispositionRefusal,match='bounded_journal_inventory'):
            store.denials_for_state(tmp_path/'state.duckdb')


def test_actual_file_fsync_precedes_link_and_directory_fsync_follows(tmp_path,monkeypatch):
    value=denial(tmp_path);path=directory(tmp_path);events=[]
    real_fsync,real_link=os.fsync,os.link
    def fsync(fd):
        events.append('file_fsync' if stat.S_ISREG(os.fstat(fd).st_mode) else 'directory_fsync')
        return real_fsync(fd)
    def link(*args,**kwargs):
        events.append('link');return real_link(*args,**kwargs)
    monkeypatch.setattr(M.os,'fsync',fsync);monkeypatch.setattr(M.os,'link',link)
    with M.DispositionJournal(path) as store:store.publish(value)
    assert events.index('file_fsync')<events.index('link')<events.index('directory_fsync')


def test_directory_fsync_failure_preserves_published_bytes_for_exact_idempotent_retry(tmp_path,monkeypatch):
    value=denial(tmp_path);path=directory(tmp_path);real_fsync=os.fsync
    def interrupted_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError('injected durability uncertainty')
        return real_fsync(fd)
    with M.DispositionJournal(path) as store:
        with monkeypatch.context() as patch:
            patch.setattr(M.os,'fsync',interrupted_directory_fsync)
            with pytest.raises(OSError,match='durability uncertainty'):store.publish(value)
        target=path/(M.revision_key(value['scope'])+'.json');before=target.stat()
        assert target.read_bytes()==M.wire(value)
        assert store.publish(value)==value
        assert target.stat().st_ino==before.st_ino


def test_original_store_namespace_preflight_stops_at_lower_finite_bound(tmp_path):
    path=tmp_path/'native-store';path.mkdir(mode=0o700)
    for name in ('one','two','three'):(path/name).write_bytes(b'')
    with pytest.raises(M.AttemptDispositionRefusal,match='namespace_bound_exceeded'):
        M._bounded_namespace(path,maximum=2)
    assert len(list(path.iterdir()))==3


def test_original_strict_inventory_is_preceded_and_followed_by_finite_namespace_observation():
    tree=ast.parse(SOURCE.read_text());scope=next(node for node in tree.body if isinstance(node,ast.ClassDef)
                                               and node.name=='OriginalDispositionScope')
    method=next(node for node in scope.body if isinstance(node,ast.FunctionDef) and node.name=='assert_no_denials_for_state')
    scans=[node for node in ast.walk(method) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name) and node.func.id=='_bounded_namespace']
    inventories=[node for node in ast.walk(method) if isinstance(node,ast.Call)
                 and isinstance(node.func,ast.Attribute) and node.func.attr=='iter_quarantines']
    assert len(scans)==2 and len(inventories)==1
    assert min(node.lineno for node in scans)<inventories[0].lineno<max(node.lineno for node in scans)
