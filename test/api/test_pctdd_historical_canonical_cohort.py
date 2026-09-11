"""Canonical cohort replacement preserves history without accepting a splice."""
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
from test.api.test_agent_supervisor_retained_cross_lane_authority import (
    _historical_admission, _historical_population, _open_lane, _seed_predecessor,
)
from test.api.test_agent_supervisor_retained_occurrence_recovery_v2 import _arm_retrying_task


def test_current_canonical_cohort_can_supersede_old_nomination_without_peer_open(tmp_path, monkeypatch):
    old = [_historical_admission(monkeypatch, task_alias=alias)
           for alias in ('PCTDD-006', 'PCTDD-007', 'PCTDD-034')]
    current = []
    for occurrence, prior in old:
        admission = dict(prior, controller_quiescence_receipt_id='sha256:'+'1'*64,
                         owner_binding_cid='sha256:'+'2'*64,
                         owner_live_generation=prior['owner_live_generation']+1)
        admission.pop('admission_id')
        admission['admission_id'] = module._database_fenced_provider_retained_digest(admission)
        assert module.database_fenced_provider_historical_retained_admission_valid(admission)
        current.append((occurrence, admission))
    lane = _open_lane(tmp_path, lane=0, owner=current[0][0]['predecessor_owner_session_id'])
    try:
        lane.materialize_population(_historical_population(current))
        for occurrence, admission in current:
            if occurrence['task_alias'] != 'PCTDD-034':_seed_predecessor(lane, occurrence)
            _arm_retrying_task(lane, occurrence, admission)
        receipts = {pin['task_cid']:lane.task_source.get(pin['task_cid']).body['completion_receipt']
                    for pin, _ in current}
        # The population selector can be historical, but cannot grant that old
        # token current task authority. The actual canonical cohort is complete.
        assert lane._retained_recovery_historical_population_durable_current(old[0][1])
        assert lane._retained_recovery_historical_population_durable_current(current[0][1])
        task = lane.task_source.get(old[0][0]['task_cid'])
        assert not lane._retained_recovery_admission_is_current_for_task(task, old[0][1])
        assert lane._retained_recovery_admission_is_current_for_task(task, current[0][1])
        assert receipts == {pin['task_cid']:lane.task_source.get(pin['task_cid']).body['completion_receipt']
                            for pin, _ in current}
        attempt = lane.claim_next(exclude_task_cids={current[1][0]['task_cid']})
        assert attempt is not None and attempt.task_cid == current[0][0]['task_cid']
        # Consumed records remain population members without releasing or
        # reopening the independent peer lane's execution/coordinator stores.
        assert lane._retained_recovery_historical_population_durable_current(current[1][1])
        assert not (tmp_path/'lane-3').exists()
        assert lane._retained_recovery_attempt_authorities is None
    finally:
        lane.close()
