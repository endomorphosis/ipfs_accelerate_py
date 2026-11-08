"""Actual-file evidence joins; surrounding accepted original-read custody modeled.

These source controls exercise the historical event-window adapter only. They
create no original/native qualification receipt and supply no cycle162 credit.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import copy,hashlib,json
import pytest
from ipfs_accelerate_py.agent_supervisor.rescue import scientific_executor as science


@pytest.fixture
def external_progress(tmp_path,monkeypatch):
    def pin(path):
        raw=Path(path).read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def write(name,value):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));path.chmod(0o400);return pin(path)
    def bound(ref):
        science.need(pin(ref['path'])==ref,'modeled_evidence_descriptor_changed')
        return Path(ref['path']).read_bytes()
    h=SimpleNamespace(_document=lambda r:json.loads(bound(r)),_bound=bound,
        _integer=lambda v,lo=0,hi=2**63-1:type(v)is int and lo<=v<=hi,
        _number=lambda v:type(v)in (int,float),_text=lambda v:type(v)is str and bool(v),
        _keys=lambda v,expected,reason:science.need(type(v)is dict and set(v)==set(expected),reason))
    monkeypatch.setattr(science,'health_owner',lambda:h)
    manifest=write('manifest',dict(task_cid='one',operation='independent-metadata-review'))
    report=write('report',dict(task_cid='one',manifest=manifest))
    timestamp='2026-10-31T16:58:28Z';epoch=datetime.fromisoformat(timestamp.replace('Z','+00:00')).timestamp()
    flags=('attempt_budget_reset','attempts_rearmed','dispatch_authorized','launcher_invoked','native_admission',
        'new_training_launched','training_authorized','formalized','admitted','gold_or_teacher_admission_granted',
        'Lean_admitted','task_completion_authority')
    receipt=dict({key:False for key in flags},manifest=manifest,report=report)
    item=dict(evidence_id='evidence',task_cid='one',digest='digest',body=dict(manifest=manifest,report=report))
    completion=dict(task_cid='one',previous_status='blocked',status='completed',revision=3,
        evidence_digests=['digest'],receipt=receipt)
    events=[]
    for number,event,subject,body in ((270,'intent.evidence_recorded','evidence',item),
                                    (271,'intent.completion_recorded','one',completion)):
        events.append(dict(global_sequence=number,event_type=event,task_cid='one',recorded_at=timestamp,
            body=dict(schema='ipfs_accelerate_py/agent-supervisor/intent-event@1',event_type=event,
                owner_id='independent-owner',recorded_at=timestamp,subject_id=subject,body=body)))
    # Later observed events belong to the accepted snapshot, not the original
    # failed-cycle window, and do not expand its reconciliation authority.
    events.append(dict(global_sequence=272,event_type='intent.task_upserted',task_cid='later'))
    evidence=dict(schema='legal-ir-proposed-boundary-observation-evidence/v1',role='modeled-accepted-original-read')
    ref=write('evidence',evidence)
    bundle=dict(evidence=evidence,snapshot=dict(event_watermark=272,full_event_page=events,
        event_page_coverage=dict(after_global_sequence_exclusive=269,through_snapshot_watermark_inclusive=272,
            complete_to_snapshot_watermark=True,global_sequence_contiguous=True,row_count=3)))
    recognition=dict(from_cursor=269,to_cursor=271,task_changes=[dict(task_cid='one',
        before=dict(task_cid='one',status='blocked',revision=2),after=dict(task_cid='one',status='completed',revision=3))])
    proof=dict(started_at=epoch-5,finished_at=epoch+5)
    return dict(ref=ref,bundle=bundle,recognition=recognition,proof=proof,events=events,
        manifest=manifest,report=report,pin=pin)


def invoke(c):
    return science._catalogue_race_external_progress(c['ref'],c['recognition'],c['proof'],{},boundary_observation=c['bundle'])


def test_only_actual_historical_pair_and_independent_file_evidence_reconcile_without_credit(external_progress):
    c=external_progress;before=copy.deepcopy(c['bundle']);out=invoke(c)
    assert [row['global_sequence']for row in out['event_window']]==[270,271]
    assert out['task_changes']==c['recognition']['task_changes']and out['health_or_task_credit']is False
    assert c['bundle']==before and c['pin'](c['manifest']['path'])==c['manifest']
    assert c['pin'](c['report']['path'])==c['report']


@pytest.mark.parametrize('fault',['cursor_gap','changed_task_without_evidence','missing_completion','wrong_status',
    'wrong_revision','wrong_owner','wrong_evidence_digest','clock_outside_original','timezone_missing',
    'report_descriptor_crossed','authority','changed_report_bytes','unobserved_closing_cursor'])
def test_missing_or_crossed_or_unauthorized_external_progress_cannot_normalize_original_failure(external_progress,fault):
    c=external_progress;events=c['events'];recognition=c['recognition']
    if fault=='cursor_gap':events[1]['global_sequence']=273
    elif fault=='changed_task_without_evidence':recognition['task_changes'].append(dict(task_cid='other'))
    elif fault=='missing_completion':events[1]['event_type']='intent.evidence_recorded'
    elif fault=='wrong_status':recognition['task_changes'][0]['after']['status']='blocked'
    elif fault=='wrong_revision':recognition['task_changes'][0]['after']['revision']=4
    elif fault=='wrong_owner':events[1]['body']['owner_id']='other-owner'
    elif fault=='wrong_evidence_digest':events[1]['body']['body']['evidence_digests']=['other-digest']
    elif fault=='clock_outside_original':c['proof']['finished_at']=c['proof']['started_at']
    elif fault=='timezone_missing':
        events[0]['recorded_at']=events[0]['body']['recorded_at']='2026-10-31T16:58:28'
    elif fault=='report_descriptor_crossed':events[1]['body']['body']['receipt']['report']={'path':'unrelated'}
    elif fault=='authority':events[1]['body']['body']['receipt']['native_admission']=True
    elif fault=='changed_report_bytes':
        path=Path(c['report']['path']);path.chmod(0o600);path.write_text('{}')
    else:recognition['to_cursor']=272
    with pytest.raises((ValueError,OSError,KeyError)):invoke(c)
