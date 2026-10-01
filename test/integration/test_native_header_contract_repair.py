"""Real native owner lifecycle for scoped symbolic header repair."""
import os
import shutil

import pytest


def test_native_header_repair_proof_publication_completion_and_refreshed_indexes(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
    if not probe_quack_capabilities().passes_health_check:
        pytest.skip('actual installed Quack transport required')
    if not shutil.which('z3') or not (os.environ.get('DOCTOR_COMPOSITION_LEAN') or shutil.which('elan')):
        pytest.skip('actual installed Lean and Z3 required')
    from benchmarks.agent_supervisor.container_coding.native_header_supervision import qualify
    result = qualify(tmp_path / 'native-header')
    assert result['qualified'], result
    assert result['initial_public_check_exit_code'] != 0
    assert result['final_public_check_exit_code'] == 0
    assert result['pre_start_task_status'] == 'ready'
    assert result['canonical_unchanged_before_start'] is True
    assert result['task']['status'] == 'completed'
    assert result['task']['revision'] >= 4
    assert result['start']['status'] == result['stop']['status'] == 'succeeded'
    assert result['remaining_processes'] == 0 and not result['bootstrap_errors']
    assert result['published_commit'] != result['baseline_commit']
    assert result['provider_calls'] == 0 and result['benchmark_result'] is False
    assert result['whole_program_proved'] is False
    assert result['security_ir']['status'] == 'compiled_local_declarations'
    assert result['security_ir']['claim_count'] == result['security_ir']['obligation_count'] == 4
    assert result['security_ir']['solver_executed'] is False
    assert result['security_ir']['provider_calls'] == 0
    assert result['security_ir']['declaration_cid'] and result['security_ir']['artifact_cid']
    assert result['doctor_proof']['status'] == 'proved_local_contract'
    assert result['doctor_proof']['proof']['kernel_replay']['executed'] is True
    assert result['doctor_proof']['proof']['kernel_replay']['kernel_verified'] is True
    assert result['contract_index']['hydrated'] is True
    assert result['contract_index']['active_receipt_ids']
    assert result['contract_index']['metadata']['stored_catalogs'] == 2
    assert result['scoped_analysis']['omitted_source_count'] == 1
    assert result['scoped_analysis']['whole_repository_analysis'] is False
    assert result['after_stop']['published_context']
    assert all(item['status'] == 'refreshed' and item['retrieval_status'] == 'current'
               for item in result['after_stop']['published_context'])
