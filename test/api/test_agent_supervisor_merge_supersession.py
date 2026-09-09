"""Only current independent acceptance can retire a quarantined candidate."""
from dataclasses import replace
import pytest
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueueFenceError,MergeQueueIntegrityError
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
from test.api.test_agent_supervisor_merge_queue import _bound_queue,_enqueue

def fixture(tmp_path):
 queue=_bound_queue(tmp_path/'queue');request=_enqueue(queue,1);queue.quarantine(request,'old candidate conflicts',metadata={'preserve':'negative evidence'});request=queue.get(request.request_id)
 review={'schema':'ipfs_accelerate_py/agent-supervisor/merge-quarantine-supersession-review@1','request_id':request.request_id,'candidate_commit':request.commit_sha,'task_cid':request.canonical_identity,'accepted_task_revision':10,'completion_receipt_cid':'receipt:current','current_target_commit':'f'*40,'validation_evidence_cid':'validation:current','reason':'independently reviewed newer accepted implementation'}
 review['review_id']=content_identity(review)
 return queue,request,review

def test_supersession_preserves_old_candidate_and_never_marks_it_merged(tmp_path):
 queue,request,review=fixture(tmp_path);calls=[]
 result=queue.supersede_quarantined(request,review=review,verify_current_acceptance=lambda req,binding:calls.append(req.request_id) is None)
 assert result.status=='cancelled' and result.commit_sha==request.commit_sha
 assert result.metadata['quarantine']=={'preserve':'negative evidence'}
 assert result.metadata['reviewed_supersession']==review
 assert result.metadata['supersession_preserved_quarantine']['failure_reason']=='old candidate conflicts'
 assert len(calls)==2 and not queue.completed_requests() and not queue.quarantined_requests()
 assert not (queue.quarantine_dir/f'{request.request_id}.json').exists()
 assert (queue.cancelled_dir/f'{request.request_id}.json').is_file()
 replay=queue.supersede_quarantined(request,review=review,verify_current_acceptance=lambda *_:True)
 assert replay.claim_generation==result.claim_generation

@pytest.mark.parametrize('failure',['denied','changed_before_commit','changed_generation','changed_task','changed_hash'])
def test_supersession_rejects_false_stale_or_drifted_acceptance_atomically(tmp_path,failure):
 queue,request,review=fixture(tmp_path);calls=[]
 def verify(*_):
  calls.append(1);return failure!='denied' and not (failure=='changed_before_commit' and len(calls)>1)
 if failure=='changed_generation':request=replace(request,claim_generation=request.claim_generation-1)
 if failure=='changed_task':review['task_cid']='foreign'
 if failure=='changed_hash':review['review_id']='forged'
 with pytest.raises((MergeQueueFenceError,MergeQueueIntegrityError)):
  queue.supersede_quarantined(request,review=review,verify_current_acceptance=verify)
 assert queue.get(request.request_id).status=='quarantined'
 assert (queue.quarantine_dir/f'{request.request_id}.json').exists()
 assert not (queue.cancelled_dir/f'{request.request_id}.json').exists()

def test_enqueue_cannot_inject_supersession_authority(tmp_path):
 queue,_,review=fixture(tmp_path)
 with pytest.raises(ValueError,match='queue-reserved'):
  queue.enqueue(branch_name='candidate',task_id='task',metadata={'reviewed_supersession':review})


def test_supersession_replay_repairs_stage_receipt_after_commit_crash(tmp_path, monkeypatch):
    queue, request, review = fixture(tmp_path)
    original = queue._write_stage_receipt
    def crash(_):
        raise OSError("simulated crash after durable database commit")
    monkeypatch.setattr(queue, "_write_stage_receipt", crash)
    with pytest.raises(OSError):
        queue.supersede_quarantined(request, review=review, verify_current_acceptance=lambda *_: True)
    assert queue.get(request.request_id).status == "cancelled"
    assert (queue.quarantine_dir / f"{request.request_id}.json").exists()
    monkeypatch.setattr(queue, "_write_stage_receipt", original)
    result = queue.supersede_quarantined(request, review=review, verify_current_acceptance=lambda *_: True)
    assert result.status == "cancelled"
    assert result.metadata["reviewed_supersession"] == review
    assert (queue.cancelled_dir / f"{request.request_id}.json").is_file()
    assert not (queue.quarantine_dir / f"{request.request_id}.json").exists()
