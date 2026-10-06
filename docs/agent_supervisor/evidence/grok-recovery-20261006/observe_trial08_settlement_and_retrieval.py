"""Project retained trial08 metadata without task, provider or verifier bodies."""
import hashlib
import json
from pathlib import Path
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')
HEAD = '585cf2adccdbaff8a838a5f5438ca80ea75c8b0e'


def reference(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    indexed_path = A / 'indexed-path-observation-08.json'
    indexed = json.loads(indexed_path.read_text())
    report_path = A / indexed['retained_driver_report']['path']
    assert report_path.is_relative_to(A) and report_path.resolve() == report_path.absolute()
    assert reference(report_path) == {key: indexed['retained_driver_report'][key] for key in ('bytes', 'sha256')}
    report = json.loads(report_path.read_text())
    live = json.loads((A / 'live-trial-summary-08.json').read_text())['trials'][0]
    assert live['selection_binding']['source_revisions']['source'] == HEAD
    manifest_path = A / 'grok-bundle-08/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_names = (
        'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',
        'benchmarks/agent_supervisor/container_coding/terminal_program_population.py',
        'benchmarks/agent_supervisor/container_coding/terminal_failure_observation.py',
        'ipfs_accelerate_py/agent_supervisor/todo_daemon/bridge_failure_diagnostics.py',
        'benchmarks/agent_supervisor/container_coding/vector_index_preflight.py',
        'benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py',
        'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
        'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
    )
    inventory = {row['path']: row for row in manifest['files']}
    source_hashes = {}
    for name in source_names:
        raw = subprocess.check_output(['git', '-C', str(P), 'show', HEAD + ':' + name])
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == inventory['source/' + name]['sha256']
        source_hashes[name] = digest

    progress = report['native_progress']
    latest = progress['latest']['task']
    failure = latest['completion_receipt']
    expected = {'operation': 'database_task_claim_failure', 'failure_kind': 'terminal_portal_bridge_error',
        'attempt_number': 1, 'control_expected_revision': 3, 'provider_invocation_count': 1,
        'effect_claim_count': 0, 'automatic_retry_admitted': False}
    assert all(type(failure[key]) is type(value) and failure[key] == value for key, value in expected.items())
    assert latest['status'] == 'blocked' and latest['revision'] == 4
    assert progress['stop_reason'] == 'native_task_blocked'
    assert len(live['router_invocations']) == 1 and live['router_invocations'][0]['phase'] == 'planning'
    assert report['unreceipted_provider_attempt']['phase'] == 'coding'
    bridge = live['native_failure_observations']['bridge']
    assert bridge['status'] == 'observed' and bridge['scope'] == 'exact_admitted_task_and_attempt'
    diagnostic = bridge['diagnostic']
    assert diagnostic['reason_code'] == 'portal_provider_failed'
    assert diagnostic['callback']['state'] == 'failed_outcome_settled'
    assert diagnostic['callback']['native_exit']['returncode'] == 1
    assert diagnostic['child_reported_router_failure']['status'] == 'missing'
    settlement = dict(schema='grok-native-terminal-settlement-observation@1', trial_name='grok-tune-mjcf-08',
        source_head=HEAD, retained_report=reference(report_path), source_sha256=source_hashes,
        native_stop_reason='native_task_blocked', task_status='blocked', task_revision=4,
        settlement=expected, planning_router_receipts=1, coding_router_receipts=0,
        coding_provider_dispatch_observed=None, coding_token_usage=None,
        aggregate_token_usage=None, invocation_counter_is_native_router_dispatch_proof=False,
        exact_bridge_failure_reason=diagnostic['reason_code'], native_exception_types_observed=len(diagnostic['exceptions']),
        native_traceback_frames_observed=sum(len(row['frames']) for row in diagnostic['exceptions']),
        native_bridge_observation=bridge, child_router_diagnostic_status='missing',
        exact_pre_router_failure_cause_established_by_trial_receipt=False, automatic_retry_performed=False,
        native_start_status=live['lifecycle']['start_status'], native_stop_status=live['lifecycle']['stop_status'],
        runtime_close_succeeded=live['custody']['runtime_close_succeeded'],
        original_verifier_reward=live['trial']['reward'], completion_authority=False,
        failure_receipt_does_not_establish_task_success=True,
        raw_model_task_verifier_or_credential_data_exported=False)

    initial, context = report['initial_context'], report['context']
    assert initial['indexed_symbols'] == 4 and initial['learned_embeddings'] is False
    assert context['initial_indexes_reused'] is True and context['learned_embeddings'] is False
    assert manifest['model_snapshot_revision'] is None
    config = manifest['source384']['config']
    assert config['embedding_revision'] == '17e1f347d17fe144873b1201da91788898c639cd'
    assert config['checkpoint_sha256'] == '2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5'
    retrieval = dict(schema='grok-retrieval-source384-distinction@1', trial_name='grok-tune-mjcf-08',
        source_head=HEAD, retained_report=reference(report_path), archive_manifest=reference(manifest_path),
        source_sha256=source_hashes, indexed_symbols=4, full_capsules=11,
        initial_and_admitted_learned_embeddings=False, retrieval_model_revision=None,
        retrieval_backend='lexical-tfidf-symbols@1',
        retrieval_backend_basis='nonempty_completed_initial_context_and_absent_separate_retrieval_model_selection_in_frozen_source',
        vector_values='normalized_term_frequency_inverse_document_frequency_over_qualified_symbol_names',
        hash_vectors_claimed=False, vector_and_metadata_persistence='DuckDB_with_DuckLake_metadata_projection',
        persistence_basis='completed_source_bound_qualification_path_requires_reopen_and_metadata_projection',
        database_contents_independently_reopened_by_this_observer=False,
        same_initial_index_reused=indexed['admitted_context']['same_initial_index_reference'],
        admitted_context_new_embedding_calls=context['new_embedding_calls'],
        source384_embedding_model='thenlper/gte-small', source384_embedding_revision=config['embedding_revision'],
        source384_checkpoint_sha256=config['checkpoint_sha256'],
        source384_fresh_inference=indexed['source384']['neural_inference_replayed'] is False,
        source384_inference_python_files=indexed['source384']['counts']['inference_python_files'],
        source384_training_steps=0, source384_is_separate_from_retrieval_index=True,
        neural_retrieval_index_not_selected=True, semantic_equivalence_claimed=False,
        proof_authority=False, completion_authority=False, raw_bodies_or_credentials_exported=False)
    for name, value in [('terminal-settlement-observation-08.json', settlement),
                        ('retrieval-mode-observation-08.json', retrieval)]:
        with (A / name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'settlement': 'terminal_portal_bridge_error', 'task_status': 'blocked',
        'exact_bridge_failure_reason': 'portal_provider_failed', 'retrieval_backend': 'lexical-tfidf-symbols@1',
        'source384_fresh_inference': True, 'new_provider_calls': 0}))


if __name__ == '__main__':
    main()
