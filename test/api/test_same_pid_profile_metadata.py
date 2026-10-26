"""Metadata comparisons only; no native/claim/admission constructor is run."""
from pathlib import Path
import pytest
import ipfs_accelerate_py.cli_runtime.grok_same_pid_owner_profile as profile


def main_binding_fixture():
    # Shape follows published main _binding_payload, not a made-up owner row.
    # In particular this serialized Portal binding has no owner_session_id.
    return {'schema': 'fixture', 'interface': 'fixture', 'attempt_id': 'attempt-a',
            'claim_id': 'claim-a', 'task_cid': 'task-a', 'task_alias': 'LIR-9001',
            'goal_cid': 'goal-a', 'plan_cid': 'plan-a', 'task_revision': 7,
            'fencing_token': 13, 'fence_epoch': 11, 'lease_id': 'lease-a',
            'task_body_digest': 'a'*64, 'projection_seed_digest': 'b'*64,
            'projection_immutable_digest': 'c'*64, 'authoritative_task_store': 'duckdb',
            'projection_authority': False}


def compare(binding, **overrides):
    expected = {key: main_binding_fixture()[key] for key in profile.MAIN_ATTEMPT_FIELDS}
    owners = dict(claim_owner_session_id='owner-a', attempt_owner_session_id='owner-a',
                  database_owner_session_id='owner-a')
    owners.update(overrides)
    profile._require_main_bridge_join(binding, expected, **owners)


def test_actual_main_shape_has_no_serialized_owner_and_compares_metadata():
    binding = main_binding_fixture()
    assert 'owner_session_id' not in binding
    compare(binding)


def test_large_unrelated_main_binding_metadata_keeps_required_joins():
    binding = main_binding_fixture()
    binding['unrelated_legitimate_metadata'] = 'x'*100_000
    compare(binding)
    binding['fencing_token'] += 1
    with pytest.raises(profile.SamePidProfileRefusal):
        compare(binding)


@pytest.mark.parametrize('field', profile.MAIN_ATTEMPT_FIELDS)
def test_every_main_attempt_join_rejects_drift(field):
    binding = main_binding_fixture()
    binding[field] = binding[field]+1 if type(binding[field]) is int else binding[field]+'-other'
    with pytest.raises(profile.SamePidProfileRefusal):
        compare(binding)


@pytest.mark.parametrize('field', ('fencing_token', 'fence_epoch'))
def test_boolean_fences_are_not_integers(field):
    binding = main_binding_fixture()
    binding[field] = True
    with pytest.raises(profile.SamePidProfileRefusal):
        compare(binding)


@pytest.mark.parametrize('field', ('claim_owner_session_id', 'attempt_owner_session_id', 'database_owner_session_id'))
def test_live_owner_session_join_rejects_drift(field):
    with pytest.raises(profile.SamePidProfileRefusal):
        compare(main_binding_fixture(), **{field: 'owner-other'})


def test_exact_normal_module_mapping_preserves_frozen_factory_contract():
    mapping = profile._expected_module_paths(Path('/workspace/accelerator'))
    assert len(mapping) == 6
    assert mapping['ipfs_accelerate_py.cli_runtime.grok_existing_phase_owner'] == Path(
        '/workspace/accelerator/ipfs_accelerate_py/cli_runtime/grok_existing_phase_owner.py')
    assert 'ipfs_accelerate_py.cli_runtime.existing_phase_owner' not in mapping
    assert mapping[profile.MODULE_NAME].name == 'grok_same_pid_owner_profile.py'


def portal_join_fixture():
    return {'database_attempt_id': 'attempt-a', 'database_task_cid': 'task-a',
            'database_claim_id': 'claim-a', 'database_attempt_number': 1,
            'database_binding_id': 'binding-a', 'task_id': 'LIR-9001'}


def test_guarded_projection_metadata_compares_without_granting_authority():
    profile._require_portal_projection_join(portal_join_fixture(), portal_join_fixture())


@pytest.mark.parametrize('field', tuple(portal_join_fixture()))
def test_portal_alias_attempt_and_binding_drift_rejected(field):
    changed = portal_join_fixture()
    changed[field] = changed[field]+1 if type(changed[field]) is int else changed[field]+'-other'
    with pytest.raises(profile.SamePidProfileRefusal):
        profile._require_portal_projection_join(changed, portal_join_fixture())


def test_portal_ordinal_bool_is_not_integer():
    changed = portal_join_fixture()
    changed['database_attempt_number'] = True
    with pytest.raises(profile.SamePidProfileRefusal):
        profile._require_portal_projection_join(changed, portal_join_fixture())
