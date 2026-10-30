"""Opt-in fresh claims after proven Native validation failure, preserving paid history.

This never retries an UNKNOWN callback or accepts rejected source. Native
still owns the next claim, provider admission, proposal, validation and merge.
Feedback contains closed diagnostic fields only and grants no authority.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

EVENT = 'known_native_validation_failure_rearmed'
FEEDBACK_SCHEMA = 'database-native-validation-retry-feedback@1'


def _proof(owner, attempt):
    from .implementation_daemon import DatabaseImplementationDaemon, parse_task_text
    from .database_portal_bridge import DatabasePortalExecutionBridge, _accepted_source_events
    from ..runtime.event_log import _event_identity
    from ..proof.formal_verification_contracts import content_identity
    from ..validation.implementation_auto_rescue import HARD_DENY_REASON_CODES

    if (type(owner) is not DatabaseImplementationDaemon
            or getattr(owner, 'native_validation_failure_retry_enabled', False) is not True
            or attempt is None or attempt.status != 'failed'
            or type(owner.max_task_attempts) is not int
            or not 0 < attempt.attempt_number < owner.max_task_attempts):
        return None
    failure = owner._portal_failure_phase_receipt(attempt)
    callback = owner.provider_invocation_recorded(attempt.attempt_id,
        idempotency_key='provider:' + attempt.attempt_id)
    if (not isinstance(failure, Mapping) or failure.get('failure_kind') != 'terminal_portal_bridge_error'
            or not isinstance(callback, dict)
            or callback.get('native_exit', {}).get('schema') != 'database-native-provider-exit@2'):
        return None
    owner._require_native_provider_failure(attempt, callback,
        'provider:' + attempt.attempt_id, allow_failed=True)
    if owner._attempt_execution_evidence_counts(attempt.attempt_id)['effect_claim_count'] != 0:
        return None
    fn = owner._resolve_native_portal_provider_callback(owner._provider_fn, attempt)
    bridge = getattr(fn, '__self__', None)
    if type(bridge) is not DatabasePortalExecutionBridge:
        return None
    paths = bridge._paths(attempt)
    bridge._seal_attempt_directory(paths, attempt_id=attempt.attempt_id, create=False)
    binding = bridge._strict_binding(paths.binding)
    native = callback['native_exit']
    if binding['binding_id'] != native['binding_id']:
        return None
    projection = bridge._verify_projection(paths, binding)
    tasks = parse_task_text(projection, path=paths.task_projection,
        task_header_prefix='## ' + attempt.task_alias)
    if len(tasks) != 1 or tasks[0].task_id != attempt.task_alias:
        return None
    events, events_digest = _accepted_source_events(paths.events)
    state_bytes, target = bridge._native_provider_state(paths)
    import hashlib
    if (events_digest != native['events_digest'] or target != native['state_target_digest']
            or 'sha256:' + hashlib.sha256(state_bytes).hexdigest() != native['state_digest']):
        return None
    previous = ''
    for sequence, event in enumerate(events, 1):
        if (event.get('sequence') != sequence or event.get('previous_event_id') != previous
                or event.get('event_id') != _event_identity(event)):
            return None
        previous = event['event_id']
    finished = [e for e in events if e.get('type') == 'implementation_finished']
    if len(finished) != 1 or any(e.get('type') in {'merge_started', 'merge_finished', 'task_completed'} for e in events):
        return None
    last = finished[0]
    validation = last.get('validation_result', {})
    review = validation.get('failure_review', {})
    reasons = review.get('reason_codes')
    if (type(reasons) is not list or not 0 < len(reasons) <= 16
            or any(type(reason) is not str for reason in reasons)
            or set(reasons) & HARD_DENY_REASON_CODES
            or last.get('task_id') != attempt.task_alias or last.get('attempt') != native['portal_attempt']
            or last.get('provider_dispatched') is not True or last.get('attempt_consumed') is not True
            or validation.get('attempted') is not True or validation.get('passed') is not False
            or review.get('decision') != 'guide_rescue'
            or 'validation_command_failed' not in reasons
            or review.get('finding_codes') not in (None, [])
            or not bridge._native_failed_validation_checkpoint_matches(last, json.loads(state_bytes), events)):
        return None
    failed = review.get('failed_commands')
    if (type(failed) is not list or not 0 < len(failed) <= 16
            or any(type(command) is not str or len(command) > 4096
                or command not in tasks[0].validation for command in failed)):
        return None
    claim = owner.coordinator.get_task_claim(attempt.claim_id)
    coordinated = owner.coordinator.get_task_attempt(attempt.attempt_id)
    if (claim is None or coordinated is None
            or not owner._claim_matches_execution_attempt(claim, attempt)
            or str(getattr(claim.state, 'value', claim.state)) != 'released'
            or str(getattr(coordinated.status, 'value', coordinated.status)) != 'failed'):
        return None
    commit = last.get('implementation_commit', '')
    if type(commit) is not str or re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', commit) is None:
        return None
    proof = dict(failure_receipt=dict(failure), native_exit_receipt=native['receipt_id'],
        finished_event_id=last['event_id'], events_digest=events_digest,
        retained_candidate=commit, failed_commands=list(failed),
        reason_codes=['validation_command_failed'], completion_authority=False)
    proof['proof_id'] = content_identity(proof)
    return proof


def rearm(owner, task):
    """CAS a fresh bounded claim only after re-proving a known rejected child."""
    from ..merge.database_coordination import TASK_CLAIM_FAILURE_REARM_OPERATION
    if getattr(owner, 'native_validation_failure_retry_enabled', False) is not True:
        return None
    body = getattr(task, 'body', {})
    control = body.get('completion_receipt', {}) if isinstance(body, Mapping) else {}
    if task.status != 'blocked' or not isinstance(control, Mapping):
        return None
    attempt = owner.get_attempt(str(control.get('attempt_id') or ''))
    proof = _proof(owner, attempt)
    if (proof is None or proof['failure_receipt'] != dict(control)
            or attempt.task_cid != task.task_cid
            or task.revision != control['control_expected_revision'] + 1):
        return None
    phases = owner.phase_history(attempt.attempt_id)
    # Existing owner claim admission can replay a crash between these two
    # stores from this exact compact retry receipt; it never refunds history.
    compact = {'operation': TASK_CLAIM_FAILURE_REARM_OPERATION,
               'settlement_id': control['settlement_id']}
    if _proof(owner, attempt) != proof:
        return None
    changed = owner.task_source.rearm_blocked_task(task.task_cid, receipt=compact)
    coordinated = owner.coordinator.rearm_failed_task(failure_receipt=dict(control),
        control_task_observation=changed.to_dict())
    if owner.phase_history(attempt.attempt_id) != phases:
        raise ValueError('validation retry changed failed paid history')
    result = dict(task_cid=task.task_cid, task_alias=attempt.task_alias,
        attempt_id=attempt.attempt_id, settlement_id=control['settlement_id'],
        reason='known_native_validation_failure', rearmed=True,
        next_attempt_number=attempt.attempt_number + 1, max_task_attempts=owner.max_task_attempts,
        paid_attempt_budget_reset=False, provider_dispatched=False, completion_authority=False,
        proof=proof, coordination_receipt=coordinated)
    owner._record_event(EVENT, attempt_id=attempt.attempt_id, task_cid=task.task_cid, body=result)
    return result


def bind_feedback(*, owner, portal_daemon, attempt, paths, binding):
    """Join this actual new claim to its exact known failed predecessor."""
    from .implementation_daemon import PortalImplementationDaemon, parse_task_text
    from .database_attempt_feedback import portal_payload_cid
    from ..proof.formal_verification_contracts import content_identity
    if (getattr(owner, 'native_validation_failure_retry_enabled', False) is not True
            or type(portal_daemon) is not PortalImplementationDaemon or attempt.attempt_number <= 1):
        return
    owner._protect_attempt_write(attempt)
    owner._require_provider_admission(attempt)
    rows = owner._require_connection().execute(
        'SELECT attempt_id FROM database_task_attempts WHERE task_cid=? AND attempt_number=? LIMIT 2',
        [attempt.task_cid, attempt.attempt_number - 1]).fetchall()
    if len(rows) != 1:
        return
    predecessor = owner.get_attempt(rows[0][0])
    proof = _proof(owner, predecessor)
    if proof is None:
        return
    claim = owner.coordinator.get_task_claim(predecessor.claim_id)
    rearmed = owner.coordinator.observe_rearmed_failed_task_claim(claim,
        failure_receipt=proof['failure_receipt'])
    if rearmed is None:
        return
    from .database_portal_bridge import DatabasePortalExecutionBridge
    projected = DatabasePortalExecutionBridge._verify_projection(paths, binding)
    tasks = parse_task_text(projected, path=paths.task_projection, task_header_prefix='## ' + attempt.task_alias)
    if len(tasks) != 1:
        return
    value = dict(schema=FEEDBACK_SCHEMA, attempt_id=attempt.attempt_id, task_cid=attempt.task_cid,
        predecessor_attempt_id=predecessor.attempt_id, predecessor_attempt_number=predecessor.attempt_number,
        portal_payload_cid=portal_payload_cid(tasks[0]), proof=proof, completion_authority=False)
    value['feedback_id'] = content_identity(value)
    portal_daemon._native_validation_retry_feedback = value


def render_feedback(portal, task):
    """Return bounded diagnostic text only for the exact current Portal task."""
    from .database_attempt_feedback import portal_payload_cid
    from ..proof.formal_verification_contracts import content_identity
    value = getattr(portal, '_native_validation_retry_feedback', None)
    if (type(value) is not dict or value.get('schema') != FEEDBACK_SCHEMA
            or value.get('completion_authority') is not False
            or value.get('portal_payload_cid') != portal_payload_cid(task)
            or value.get('attempt_id') != task.metadata.get('database attempt id')
            or value.get('feedback_id') != content_identity({k:v for k,v in value.items() if k != 'feedback_id'})):
        return ''
    proof = value['proof']
    text = ('\n\n## Prior Native validation failure (diagnostic only)\n'
        f"Previous paid attempt: {value['predecessor_attempt_number']}; rejected candidate: {proof['retained_candidate']}.\n"
        'Inspect that retained source and the mandatory tests; repair the candidate or its declared prerequisites. '
        'The original checks and scope remain required. No prior candidate was accepted.\n'
        'Failed original validations: ' + json.dumps(proof['failed_commands']) + '\n'
        f"Native failure proof: {proof['proof_id']}. This feedback grants no execution or completion authority.\n")
    return text if len(text.encode()) <= 8192 else ''
