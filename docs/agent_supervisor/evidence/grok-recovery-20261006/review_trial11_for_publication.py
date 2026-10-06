"""Independently bind closed trial11 exports without exporting private bodies."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = A.parent.parent / '.worktrees/grok-recovery-20261006'
HEAD = '0e2d9a8c62eabdfbd42fcad505c370365474514a'
MERGED = '0a8a1b3e8448a6604899bedcbc5175dc08bf6825'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(name):
    return json.loads((A / name).read_bytes())


def ref(name):
    raw = (A / name).read_bytes()
    return {'bytes': len(raw), 'sha256': sha(raw)}


def check_refs(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys():
            path = Path(value['path'])
            assert not path.is_absolute() and '..' not in path.parts
            assert ref(path) == {k: value[k] for k in ('bytes', 'sha256')}
        for child in value.values():
            check_refs(child)
    elif isinstance(value, list):
        for child in value:
            check_refs(child)


def main():
    names = [
        'live-trial-summary-11.json', 'indexed-path-observation-11.json',
        'performance-observation-11.json', 'retrieval-mode-observation-11.json',
        'trial-closure-observation-11.json', 'timeout-settlement-observation-11.json',
        'grok-container/archive-review-11.json', 'grok-container/preparation-review-11.json',
        'grok-container/resource-observation-11.json',
        'grok-container/learned-vector-runtime-observation-11.json',
        'grok-container/observer-coordinator-11.json',
        'grok-container/native-tools-grok-tune-mjcf-11.json',
        'publication-stage/automatic-native-monitor-review-11.json',
        'qualification/trial11-preparation-independent-review-01.json',
        'qualification/observe_trial11_closure.py',
        'qualification/observe_trial11_retrieval.py', 'qualification/observe_trial11_timeout.py',
        'grok-container/recipes/review_shutdown_learned_preparation.py',
        'timeout-recovery/coordinate_trial11_observers.py',
        'publication-stage/qualification-summary-origin-codex-merge-final.json',
        'publication-stage/independent-origin-codex-merge-review-01.json',
    ]
    summary = read(names[0])['trials'][0]
    index, performance, retrieval, closure, timeout = map(read, names[1:6])
    for value in (summary, index, performance, retrieval, closure, timeout):
        check_refs(value)
    assert summary['selection_binding']['source_revisions']['source'] == HEAD
    result_path = 'grok-tune-mjcf-11/jobs/supervisor-full-tune-mjcf/tune-mjcf__SRGTQxt/result.json'
    assert sha((A / result_path).read_bytes()) == summary['trial']['native_result_sha256']
    assert read(result_path)['verifier_result']['rewards']['reward'] == 0
    assert summary['trial']['reward'] == closure['official_reward'] == timeout['official_reward'] == 0
    assert summary['task_state'] == {'status': 'in_progress', 'revision': 3}
    assert summary['lifecycle'] == {'start_status': 'succeeded', 'stop_status': 'succeeded'}
    assert summary['shutdown_failures'] is None
    assert closure['runtime_close_succeeded'] is True and closure['remaining_processes'] == 0
    assert closure['worker_cleanup_returncode'] == 0
    assert closure['exact_owned_container_present_after_cleanup'] is False
    assert closure['source_unchanged'] and closure['source_clean_after']
    assert closure['observer_children_exited_zero']
    assert timeout['callback_state'] == 'started_outcome_unknown'
    assert timeout['native_exit_receipt_present'] is False
    assert timeout['callback_settlement_observed'] is False
    assert timeout['STOP_after_success_path_exercised'] is False
    assert timeout['task_and_callback_completion_qualified'] is False
    assert timeout['closure_flag_full_lifecycle_qualified_scope'] == 'START_STOP_tracked_tree_absence_and_runtime_close_only'
    planning, coding = performance['provider_invocations']
    assert planning['phase'] == 'planning' and planning['native_tokens']['total_tokens'] == 22806
    assert coding['phase'] == 'coding' and coding['timeout_seconds'] == 600
    assert coding['seconds'] == timeout['coding_seconds'] == 600.2467448359821
    assert coding['native_tokens'] is None and performance['native_observed_total_tokens'] is None
    assert timeout['coding_error_type'] == 'TimeoutExpired'
    assert index['actual_index']['indexed_symbols'] == 4 and index['actual_index']['full_capsules'] == 11
    assert index['actual_index']['learned_embeddings'] is True
    assert index['source384']['neural_inference_replayed'] is False
    assert index['source384']['training_steps'] == 0
    assert index['admitted_context']['new_embedding_calls'] == 0
    assert index['source384']['checkpoint_sha256'] == retrieval['source384_checkpoint_sha256']
    learned = read('grok-container/learned-vector-runtime-observation-11.json')
    assert learned['status'] == 'qualified' and learned['configuration']['device'] == 'cpu'
    assert learned['dimensions'] == 384 and learned['local_model_calls'] == 3 and learned['local_model_texts'] == 8
    assert learned['canary']['disposition'] == 'passed' and learned['ducklake_status'] == 'projected'
    assert learned['embedding_input'] == 'qualified symbol names only' and not learned['semantic_authority']
    assert retrieval['retrieval_model_revision'] == learned['model_revision'] == '17e1f347d17fe144873b1201da91788898c639cd'
    for path, expected in retrieval['source_sha256'].items():
        raw = subprocess.check_output(['git', '-C', str(P), 'show', HEAD + ':' + path])
        assert sha(raw) == expected
    source_path = timeout['source_binding']['path']
    source = subprocess.check_output(['git', '-C', str(P), 'show', HEAD + ':' + source_path])
    assert sha(source) == timeout['source_binding']['sha256']
    report = read(closure['retained_driver_report']['path'])
    assert report['error'] == {'type': 'TimeoutError', 'message': 'the total benchmark agent budget is exhausted'}
    lines = sorted(n.lineno for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and n.value == report['error']['message'])
    assert lines == timeout['source_binding']['matching_literal_lines']
    prepare = read('qualification/trial11-preparation-independent-review-01.json')
    for path, expected in prepare['evidence'].items():
        assert ref(path) == expected
    monitor = read('publication-stage/automatic-native-monitor-review-11.json')
    assert monitor['source_commit'] == HEAD and monitor['coverage']['all_observers_exit_zero']
    assert monitor['observations']['bridge_failure']['native_exit_present'] is False
    assert monitor['observations']['memory_events_maxima'] == {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}
    assert sha((A / monitor['monitor_log']).read_bytes()) == monitor['monitor_log_sha256']
    for path, expected in monitor['source_review']['source_bindings'].items():
        assert sha(subprocess.check_output(['git', '-C', str(P), 'show', HEAD + ':' + path])) == expected
    previous = read('publication-stage/qualification-summary-shutdown-final.json')
    merged = read('publication-stage/qualification-summary-origin-codex-merge-final.json')
    assert len(merged['runs']) == 58 and merged['runs'][:-1] == previous['runs']
    spec = importlib.util.spec_from_file_location('closed_qualification_export', A / 'qualification/export_qualification.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    last = exporter.collect(A / 'qualification', ['origin-codex-schema-merge-final-01'])['runs'][0]
    merged_row = merged['runs'][-1]
    # The final exporter retains all recorded paths and per-case ID hashes;
    # the shared exporter filters paths and emits one aggregate ID hash.
    differing_representation = {'source_sha256_before', 'source_sha256', 'test_case_id_sha256'}
    assert {k: v for k, v in last.items() if k not in differing_representation} == {
        k: v for k, v in merged_row.items() if k not in differing_representation}
    command = read('qualification/origin-codex-schema-merge-final-01-command.json')
    exit_record = read('qualification/origin-codex-schema-merge-final-01-exit.json')
    assert merged_row['source_sha256_before'] == command['before']['source_sha256']
    assert merged_row['source_sha256'] == exit_record['after']['source_sha256']
    cases = ET.fromstring((A / 'qualification/origin-codex-schema-merge-final-01.xml').read_bytes()).findall('.//testcase')
    assert merged_row['test_case_id_sha256'] == [sha((c.get('classname', '') + '::' + c.get('name', '')).encode()) for c in cases]
    assert last['counts'] == {'passed': 213, 'failed': 0, 'errors': 0, 'skipped': 0}
    assert last['qualified'] and last['source_head_after'] == MERGED
    review = {
        'schema': 'terminal-trial-publication-independent-review@1',
        'attempt': '11', 'source_head': HEAD, 'official_reward': 0.0,
        'native_task_status': 'in_progress', 'native_task_revision': 3,
        'driver_task_completed': False, 'native_start_status': 'succeeded',
        'native_stop_status': 'succeeded', 'runtime_close_succeeded': True,
        'remaining_processes': 0, 'worker_cleanup_returncode': 0,
        'owned_container_absence_reported': True, 'all_observer_children_exited_zero': True,
        'shutdown_failures_reported': False, 'process_cleanup_qualified': True,
        'task_or_callback_completion_qualified': False, 'STOP_after_success_path_exercised': False,
        'callback_state': 'started_outcome_unknown', 'native_exit_receipt_present': False,
        'exact_native_custody_gate_denial_known': False,
        'planning_native_tokens': 22806, 'coding_seconds': coding['seconds'],
        'coding_native_tokens': None, 'total_native_tokens': None,
        'billing_verified': False, 'usage_complete_claimed': False,
        'learned_cpu_runtime_evidence_reviewed': True,
        'learned_result_independently_reopened_by_this_review': False,
        'source384_fresh_inference': True, 'source384_proof_authority': False,
        'original_verifier_result_hash_and_reward_rechecked': True,
        'work_budget_error_source_literal_revalidated': True,
        'referenced_evidence_hashes_rechecked': True, 'reviewed_closed_exports_consistent': True,
        'merged_qualification_source': MERGED, 'merged_qualification_passes': 213,
        'cumulative_rows': 58, 'prior_57_rows_unchanged': True,
        'overlapping_test_counts_not_summed': True, 'merged_source_live_benchmarked': False,
        'evidence': {name: ref(name) for name in names},
        'limitations': [
            'Trial10 task success with failed native closure and trial11 timeout with clean process cleanup remain separate.',
            'No live task pass followed by clean native shutdown is established.',
            'Missing native exit receipt leaves callback settlement unknown; no retry or settlement authority is inferred.',
            'Neural retrieval inputs were symbol names and Source384 remained nomination-only; no whole-task formal proof or parallel-agent advantage is established.',
            'No matched baseline, token savings, billing completeness or whole-suite result is claimed.',
            'Three unresolved collection/API contracts and historical test failures remain recorded.',
        ],
        'source_edits': 0, 'provider_calls': 0, 'container_mutations': 0, 'new_test_runs': 0,
        'original_completed_receipts_overwritten': False,
        'raw_logs_task_model_verifier_or_credentials_exported': False,
        'publication_ready': False,
        'publication_readiness_scope': 'final README and inventory reviewed separately',
    }
    out = A / 'qualification/trial11-independent-publication-review-01.json'
    with out.open('x') as stream:
        json.dump(review, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'review': str(out.relative_to(A)), 'sha256': sha(out.read_bytes()),
                      'evidence_count': len(names), 'cumulative_rows': 58}))


if __name__ == '__main__':
    main()
