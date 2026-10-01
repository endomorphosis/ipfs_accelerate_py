"""Accelerate catalog integration; reusable transfer tests belong to datasets."""
from copy import deepcopy
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder as ae
from tests.unit.logic.formalization.autoencoder.test_codebase_autoencoder_transfer import teacher, fork, joint_inputs  # noqa: F401


def test_task_security_nominations_are_bound_to_source_rows_targets_and_native_catalog(joint_inputs, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder_index as catalogs
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import ProgramWorldDatabase
    descriptor = ae.train_codebase_autoencoder(**joint_inputs)
    output = tmp_path / 'candidate-catalog'
    receipt = catalogs.build_codebase_autoencoder_index(repository=joint_inputs['repository'], learner=descriptor, output=output)
    assert catalogs.validate_codebase_autoencoder_index(repository=joint_inputs['repository'], learner=descriptor, expected_receipt=receipt)['status'] == 'verified'
    world = ProgramWorldDatabase(output / 'world.duckdb', output / 'world-lake')
    record = world.records_for_decision(task_id='code-index:' + descriptor['checkpoint_sha256'], operation=catalogs.OPERATION)['records'][0]['payload']
    assert record['security_candidate_nominations']['row_count'] == descriptor['sample_count']
    assert record['security_candidate_nominations']['nomination_bytes_sha256'] == ae._sha(ae._json(descriptor['security_candidate_nominations']))
    for kind in ('row', 'target'):
        altered = deepcopy(descriptor)
        block = altered['security_candidate_nominations']
        if kind == 'row':
            block['rows'][0]['row_id'] = '0' * 64
        else:
            block['target_vocabulary'][0] = 'effect:deny'
        with pytest.raises(ValueError, match='descriptor differs'):
            ae.validate_codebase_autoencoder(repository=joint_inputs['repository'], expected_receipt=altered)
