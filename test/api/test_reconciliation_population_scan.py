"""Bounded receipt scans retain exact bytes, authority, and publication custody."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import stat
import fcntl
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import database_portal_bridge as M


def prepared(tmp_path, count=3):
    bridge=M.DatabasePortalExecutionBridge(task_source=None,attempt_root=tmp_path/'attempts',portal_factory=lambda *_:None)
    attempt=SimpleNamespace(attempt_id='attempt:scan',claim_id='claim:scan',task_cid='task:scan',task_alias='PCTDD-008',
        attempt_number=1,owner_session_id='session:scan',fencing_token=1,fence_epoch=1)
    records=[]
    for i in range(count):
        r=bridge.persist_reconciliation_receipt(attempt,{'stage':'blocked','trigger':'scan-test','reconciled_at':'2026-09-12T00:00:00Z',
            'reconciled':False,'blocked':True,'reason':f'unresolved-{i}'})
        r.pop('receipt_path');records.append(r)
    return bridge,attempt,sorted(records,key=lambda r:r['receipt_id'])


def test_population_has_constant_fsyncs_and_preserves_all_authority(tmp_path,monkeypatch):
    bridge,attempt,expected=prepared(tmp_path,128)
    syncs=[];original=os.fsync
    def sync(fd):
        assert stat.S_ISDIR(os.fstat(fd).st_mode)
        syncs.append(fd);original(fd)
    monkeypatch.setattr(M.os,'fsync',sync)
    assert bridge._reconciliation_evidence_population(attempt)==expected
    assert len(syncs)==2


@pytest.mark.parametrize('damage',['payload','authority','symlink','hardlink','fifo','extra_name','noncanonical'])
def test_population_refuses_corrupt_or_ambiguous_files(tmp_path,monkeypatch,damage):
    bridge,attempt,expected=prepared(tmp_path)
    root=bridge._paths(attempt).reconciliation
    p=root/(expected[0]['receipt_id'][7:]+'.json')
    if damage=='payload':p.write_bytes(b'{}')
    elif damage=='authority':
        r=json.loads(p.read_text());r['claim_id']='foreign';r.pop('receipt_id');r['receipt_id']=M._sha256_bytes(M._canonical_json(r));p.unlink()
        (root/(r['receipt_id'][7:]+'.json')).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    elif damage=='symlink':p.rename(tmp_path/'foreign');p.symlink_to(tmp_path/'foreign')
    elif damage=='hardlink':os.link(p,tmp_path/'alias')
    elif damage=='fifo':p.unlink();os.mkfifo(p)
    elif damage=='extra_name':(root/'unknown').write_bytes(b'')
    else:p.write_text(json.dumps(json.loads(p.read_text())))
    with pytest.raises((M.DatabasePortalBridgeError,OSError)):
        bridge._reconciliation_evidence_population(attempt)


@pytest.mark.parametrize('change',['replace_file','append_file','replace_directory','sync_failure'])
def test_population_revalidates_after_durability_boundary(tmp_path,monkeypatch,change):
    bridge,attempt,expected=prepared(tmp_path)
    root=bridge._paths(attempt).reconciliation;p=root/(expected[0]['receipt_id'][7:]+'.json')
    original=os.fsync;changed=False
    def sync(fd):
        nonlocal changed
        if not changed:
            changed=True
            if change=='replace_file':
                replacement=tmp_path/'replacement';replacement.write_bytes(p.read_bytes());os.replace(replacement,p)
            elif change=='append_file':(root/'unexpected').write_bytes(b'')
            elif change=='replace_directory':root.rename(root.with_name('old'));root.mkdir()
            else:raise OSError('durability failure')
        original(fd)
    monkeypatch.setattr(M.os,'fsync',sync)
    with pytest.raises((M.DatabasePortalBridgeError,OSError)):
        bridge._reconciliation_evidence_population(attempt)


def test_population_holds_real_publication_lock_through_sync(tmp_path,monkeypatch):
    bridge,attempt,_=prepared(tmp_path)
    root=bridge._paths(attempt).reconciliation;original=os.fsync;checked=[]
    def sync(fd):
        other=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            checked.append(True)
        finally:os.close(other)
        original(fd)
    monkeypatch.setattr(M.os,'fsync',sync)
    bridge._reconciliation_evidence_population(attempt)
    assert len(checked)==2
    other=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:os.close(other)


@pytest.mark.parametrize('method',['_interrupted_validation_recovery_evidence','_interrupted_implementation_retry_evidence','_stale_dispatch_migration_retry_evidence'])
def test_historical_searches_do_not_repeat_durability_per_unrelated_receipt(tmp_path,monkeypatch,method):
    bridge,attempt,_=prepared(tmp_path,32)
    count=[];original=os.fsync
    def sync(fd):count.append(fd);original(fd)
    monkeypatch.setattr(M.os,'fsync',sync)
    assert getattr(bridge,method)(attempt,{'task_alias':attempt.task_alias,'binding_id':'binding:unresolved'}) is None
    assert len(count)==2


def test_population_refuses_replacement_between_lock_and_directory_open(tmp_path,monkeypatch):
    bridge,attempt,_=prepared(tmp_path)
    root=bridge._paths(attempt).reconciliation;original=os.open;changed=False
    def opened(path,*args,**kwargs):
        nonlocal changed
        if not changed and path==root.parent:
            changed=True;root.rename(root.with_name('old'));root.mkdir()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(M.os,'open',opened)
    with pytest.raises(M.DatabasePortalBridgeError,match='population changed'):
        bridge._reconciliation_evidence_population(attempt)



def test_preportal_reconciliation_uses_one_durable_receipt_population(tmp_path,monkeypatch):
    bridge,attempt,_=prepared(tmp_path,32)
    count=[];original=os.fsync
    def sync(fd):count.append(fd);original(fd)
    monkeypatch.setattr(M.os,'fsync',sync)
    result=bridge.reconcile_quiesced_attempt(attempt)
    assert result['reason']=='portal_attempt_artifacts_absent'
    assert result['terminal_provider_evidence'] is False
    assert len(count)==2


@pytest.mark.parametrize('prefix',['linked_final','final_absent','stage_only','invalid_ready'])
def test_preportal_population_recovers_only_actual_publication_prefixes(tmp_path,monkeypatch,prefix):
    bridge,attempt,expected=prepared(tmp_path)
    root=bridge._paths(attempt).reconciliation
    final=root/(expected[0]['receipt_id'][7:]+'.json')
    temporary=root/('.'+final.name+'.interrupted.tmp')
    if prefix=='linked_final':os.link(final,temporary)
    elif prefix=='final_absent':final.rename(temporary)
    elif prefix=='stage_only':
        final.unlink();temporary=root/('.'+final.name+'.interrupted.stage');temporary.write_bytes(b'incomplete unauthoritative stage')
    else:final.unlink();temporary.write_bytes(b'not a receipt')
    targets=[];original=M._recover_immutable_link_publication
    def recover(path,**kwargs):targets.append(path);return original(path,**kwargs)
    monkeypatch.setattr(M,'_recover_immutable_link_publication',recover)
    if prefix=='invalid_ready':
        with pytest.raises(M.DatabasePortalBridgeError):bridge.reconcile_quiesced_attempt(attempt)
        assert temporary.exists() and not final.exists()
    else:
        result=bridge.reconcile_quiesced_attempt(attempt)
        assert result['terminal_provider_evidence'] is False and not temporary.exists()
        assert final.exists() is (prefix!='stage_only')
        if final.exists():assert final.stat().st_nlink==1
    assert targets==[final]


def test_preportal_population_rejects_invalid_settled_final(tmp_path):
    bridge,attempt,expected=prepared(tmp_path)
    root=bridge._paths(attempt).reconciliation
    (root/(expected[0]['receipt_id'][7:]+'.json')).write_bytes(b'{}')
    with pytest.raises(M.DatabasePortalBridgeError):bridge.reconcile_quiesced_attempt(attempt)
