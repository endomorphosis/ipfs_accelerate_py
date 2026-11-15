"""Read closed public check observations without execution or owner authority."""
import hashlib
from diagnostic_json_contract import canonical_json
from diagnostic_owner_rounds_contract import _observation
from diagnostic_public_validation_contract import validate_public_validation_observation
from diagnostic_doctor_dispatch_contract import validate_doctor_dispatch_observation
from diagnostic_doctor_policy_contract import PUBLIC_VALIDATION_POLICY

FLAGS=dict(observation_only=True, completion_authority=False, proof_authority=False,
    execution_authority=False, continuation_authority=False, independent_owner_replay_performed=False,
    raw_sources_or_private_owner_artifacts_exported=False)


def _observations(report):
    if type(report) is not dict or report.get('doctor_coding_policy') != PUBLIC_VALIDATION_POLICY:
        raise ValueError('new_public_policy_required')
    markers=report.get('owner_candidate_rounds')
    if type(markers) is not list or len(markers)!=1:
        raise ValueError('unique_owner_observation_required')
    marker=markers[0]
    if type(marker) is not dict or marker.get('schema')!='owner-candidate-rounds-observation@2':
        raise ValueError('public_owner_observation_required')
    marker={key:value for key,value in marker.items()
        if key not in {'owner_lifecycle_retention','owner_lifecycle_retention_unavailable'}}
    _observation(marker, normalized=False)
    accounting=report.get('candidate_attempt_accounting')
    if type(accounting) is not dict or accounting.get('doctor_coding_policy')!=PUBLIC_VALIDATION_POLICY:
        raise ValueError('matching_native_accounting_required')
    observations=[validate_public_validation_observation(value) for value in marker['public_validation_observations']]
    if (canonical_json(observations)!=canonical_json(accounting.get('public_validation_observations'))
        or marker['owner_validation_outcome']!=accounting.get('owner_validation_outcome')
        or type(marker['owner_returncode']) is not type(accounting.get('owner_returncode'))
        or marker['owner_returncode']!=accounting.get('owner_returncode')):
        raise ValueError('public_observation_accounting_mismatch')
    return marker,observations


def public_validation_report(report):
    """Export statuses, closed reasons and timing measurements, never source rows."""
    if report.get('doctor_coding_policy') != PUBLIC_VALIDATION_POLICY:
        return None
    try:
        marker,observations=_observations(report)
        rounds=[]
        for value in observations:
            feedback=dict(status=value['status'],failed_obligations=value['failed_obligations'],
                checks=[dict(validation_key=row['validation_key'],status=row['status'],reason=row['reason'],
                    measurements=[] if row['result'] is None else row['result']['measurements'])
                    for row in value['checks']])
            rounds.append({key:value[key] for key in ('round_number','session_id','descriptor_sha256','actual_returncode',
                'contract_sha256','workspace_generation')} | dict(observation_sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
                status=feedback['status'],failed_obligations=feedback['failed_obligations'],checks=feedback['checks']))
        return dict(status='validated_retained_observation',schema=marker['schema'],
            owner_validation_outcome=marker['owner_validation_outcome'],owner_returncode=marker['owner_returncode'],
            owner_failure_reported=marker['failure_type'] is not None,rounds=rounds,**FLAGS)
    except (ValueError,TypeError,KeyError,RecursionError):
        return dict(status='invalid_or_unavailable',rounds=[],**FLAGS)


def public_continuation_matches(report, doctor):
    """Join the reported second receipt to exact first-check bytes, not retry authority."""
    try:
        _,observations=_observations(report)
        doctor=validate_doctor_dispatch_observation(doctor)
        if not observations or doctor['policy_id']!=PUBLIC_VALIDATION_POLICY:
            return False
        first=observations[0]
        return (first['round_number']==1
            and doctor['public_validation_contract_sha256']==first['contract_sha256']
            and doctor['public_validation_observation_sha256']==hashlib.sha256(canonical_json(first)).hexdigest()
            and doctor['actual_prior_returncode']==first['actual_returncode']
            and doctor['public_validation_status']==first['status'])
    except (ValueError,TypeError,KeyError,RecursionError):
        return False
