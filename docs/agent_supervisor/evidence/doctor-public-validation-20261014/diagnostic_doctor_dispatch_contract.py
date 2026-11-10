"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import hashlib
import json
from diagnostic_json_contract import canonical_json
SCHEMA = 'supervisor-doctor-candidate-dispatch-observation@1'
PUBLIC_SCHEMA = 'supervisor-doctor-candidate-dispatch-observation@2'
_OBSERVATION_FLAGS = dict(canonical_source_current=True, workspace_source_current=True, original_obligations_preserved=True, proposal_budget_changed=False, owner_round_number=2, native_token_cap_enforced=False, native_cost_cap_enforced=False, completion_authority=False, proof_authority=False)
_DIGEST_FIELDS = ('coding_policy_sha256', 'doctor_context_sha256', 'doctor_artifact_sha256', 'doctor_advisory_sha256', 'proposal_budget_sha256', 'original_requirements_sha256', 'source_manifest_sha256')

def _need(value, reason):
    if not value:
        raise ValueError(reason)

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def validate_doctor_dispatch_observation(value):
    """Check a receipt's closed wire shape; no source or admission is replayed."""
    from diagnostic_source_digest_contract import _digest
    from diagnostic_residual_budget_contract import LIMIT_FIELDS, validate_residual_budget
    from diagnostic_doctor_policy_contract import DOCTOR_CODING_POLICY, PUBLIC_VALIDATION_POLICY
    fields = {'schema', 'policy_id', 'proposal_budget', 'doctor_advisory_bytes', 'source_files', 'task_cid', 'context_cid', 'native_capsule_id', *_DIGEST_FIELDS, *_OBSERVATION_FLAGS}
    public = type(value) is dict and value.get('schema') == PUBLIC_SCHEMA
    if public:
        fields.update(('public_validation_contract_sha256', 'public_validation_observation_sha256', 'actual_prior_returncode', 'public_validation_status'))
    _need(type(value) is dict and set(value) == fields and (value['schema'] in (SCHEMA, PUBLIC_SCHEMA)) and (value['policy_id'] == (PUBLIC_VALIDATION_POLICY if public else DOCTOR_CODING_POLICY)) and all((_digest(value[key]) for key in _DIGEST_FIELDS)) and all((type(value[key]) is str and 1 <= len(value[key]) <= 1024 for key in ('task_cid', 'context_cid', 'native_capsule_id'))) and (type(value['doctor_advisory_bytes']) is int) and (1 <= value['doctor_advisory_bytes'] <= 32768) and (type(value['source_files']) is int) and (1 <= value['source_files'] <= 256) and all((type(value[key]) is type(expected) and value[key] == expected for key, expected in _OBSERVATION_FLAGS.items())), 'invalid Doctor candidate dispatch observation')
    if public:
        _need(all((_digest(value[key]) for key in ('public_validation_contract_sha256', 'public_validation_observation_sha256'))) and type(value['actual_prior_returncode']) is int and (0 <= value['actual_prior_returncode'] <= 255) and (value['public_validation_status'] in ('passed', 'failed', 'inconclusive')) and (value['actual_prior_returncode'] > 0 or value['public_validation_status'] == 'failed'), 'invalid public-validation candidate dispatch observation')
    budget = value['proposal_budget']
    _need(type(budget) is dict, 'closed unchanged Doctor proposal budget required')
    validate_residual_budget(dict(residual_budget=budget, token_budget={'residual_' + key: budget.get(key) for key in LIMIT_FIELDS}))
    _need(value['proposal_budget_sha256'] == _sha(canonical_json(budget)), 'Doctor proposal budget digest differs')
    return json.loads(canonical_json(value))
