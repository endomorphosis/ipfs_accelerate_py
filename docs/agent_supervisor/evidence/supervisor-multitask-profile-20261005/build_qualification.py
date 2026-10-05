"""Build a bounded report from completed, independently retained runs."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

OUT = Path(__file__).parent


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    final = read('final-02.qualification.json')
    before = read('final-02-before.json')
    after = read('final-02-after.json')
    baseline = read('baseline-01.qualification.json')
    historical = read('final-01.qualification.json')
    baseline_dispatch = read('baseline-doctor-dispatch-01.qualification.json')
    focused = read('fixture-retrieval-01.qualification.json')
    native = read('native-integration-final.qualification.json')
    review = read('publication-independent-review.json')
    audit = read('final-02.native-process-audit.json')
    assert before == after and final['pins_unchanged'] and final['returncode'] == 0
    assert all(status == 'passed' for status in final['case_outcomes'].values())
    assert set(baseline['case_outcomes']) <= set(final['case_outcomes'])
    assert set(historical['case_outcomes']) <= set(final['case_outcomes'])
    assert set(focused['case_outcomes']) <= set(final['case_outcomes'])
    native_nodes = {case.attrib['classname'] + '::' + case.attrib['name']
                    for case in ET.parse(OUT / 'native-integration-final.xml').iter('testcase')}
    assert native_nodes <= set(final['case_outcomes'])
    assert len(native_nodes) == native['counts']['passed']
    assert final['ast_sealing']['records'] == len(final['case_outcomes'])
    assert final['ast_sealing']['any_completion_authority'] is False
    assert not audit['matching_live_processes'] and not audit['unavailable_worker_observations']
    assert audit['signal_calls'] == 0
    original = read('baseline-01-before.json')
    assert before['retained_assets'] == original['retained_assets']
    assert before['datasets_commit'] == original['datasets_commit']
    assert before['datasets_tracked_status'] == ''
    for path, sha in review['source_bindings'].items():
        assert before['source_files'][path]['sha256'] == sha
    root = Path(read('implementation-scope.json')['worktree'])
    assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == pin['sha256']
               for path, pin in before['source_files'].items())
    failed = lambda record: sorted(key for key, status in record['case_outcomes'].items()
                                  if status == 'failure')
    assert failed(historical) == failed(baseline_dispatch) and len(failed(historical)) == 3
    profile_cases = {key for key in final['case_outcomes']
                     if key.startswith('benchmarks.agent_supervisor.container_coding.test_terminal_task_profile::')}
    legacy_profile = profile_cases & set(baseline['case_outcomes'])
    native_cases = {key for key in final['case_outcomes']
                    if key.startswith('benchmarks.agent_supervisor.container_coding.test_terminal_multitask_preparation::')}
    assert len(profile_cases) == 125 and len(legacy_profile) == 73 and len(native_cases) == 41
    report = {
        'schema': 'supervisor-multitask-profile-qualification@1',
        'implementation_baseline': read('implementation-scope.json')['implementation_baseline'],
        'owned_implementation_commit': read('before-qualification-integration.json')['owned_implementation_commit'],
        'qualified_source_base_commit': before['base_commit'],
        'source_binding_note': 'Final source base also has the explicitly pinned test-only Doctor fixture correction. All seven reviewed production/profile/native-test files match the independently reviewed committed implementation.',
        'phase': 'reviewed administrative planning, admission and native storage',
        'public_profile_schema': 'terminal-public-task-profile@3',
        'prepared_schema': 'terminal-indexed-public-preparation@2',
        'final_run': 'final-02.qualification.json',
        'counts': final['counts'],
        'distinct_passing_cases': len(final['case_outcomes']),
        'selection': 'final-selection-02.json',
        'ast_sealing': final['ast_sealing'],
        'current_source_pins': 'final-02-before.json',
        'manually_pinned_source_and_test_files': len(before['source_files']),
        'source_assets_and_datasets_pins_unchanged': True,
        'datasets_commit': before['datasets_commit'],
        'retained_assets': before['retained_assets'],
        'retained_assets_scope': 'These four previously used assets remain byte-identical. This does not inventory all models or claim training/inference from them in this control-path qualification.',
        'new_profile_and_native_controls': len(profile_cases - legacy_profile) + len(native_cases),
        'preserved_legacy_profile_controls': len(legacy_profile),
        'baseline': {'run': 'baseline-01.qualification.json', 'counts': baseline['counts'],
                     'all_baseline_nodes_retained': True},
        'focused_native': {'run': 'native-integration-final.qualification.json', 'counts': native['counts']},
        'focused_fixture_and_real_retrieval': {'run': 'fixture-retrieval-01.qualification.json',
                                             'counts': focused['counts']},
        'historical_diagnostics': 'intermediate-diagnostics.json',
        'count_scope': 'Only final-02 distinct nodes contribute to the final total. Baseline, focused and diagnostic selections overlap and are not added.',
        'independent_review': 'publication-independent-review.json',
        'original_binding_failure': {'record': 'owner-binding-red.json',
            'source_snapshot': 'local_planning_admission-owner-red.py',
            'scope': 'Intermediate new @3 profile/preparation against the original 7dda native owner, not an unchanged full baseline with @3 support.'},
        'native_process_audit': 'final-02.native-process-audit.json',
        'limitations': [
            'Requirement interpretations and operations are authored reviewed candidates; these controls do not measure decoder reconstruction or autoformalization quality.',
            'Two currentness race controls inject a source edit around the actual compiler/native insertion boundary; the rest of their production path and rollback are real.',
            'AST seals are local evidence with no completion authority; manual owner bindings separately include benchmark helpers.',
            'Per-task contexts and reviewed-profile execution remain refused. No parallel worker execution is qualified.',
            'Structural smoke bounds are per file/task, not a qualified aggregate multi-task completion bound; serialized admission ceilings also apply.',
        ],
        'remaining_plan': [
            'Per-task source/dependency-wave contexts with separate IR family/schema/version/decoder task and parallel 8D/384D/768D identities, token/span budgets, database and pinned ModelManager/Hugging Face assets.',
            'Verified predecessor source successors, unchanged embedding-cache reuse and actual context cold-restart qualification.',
            'Isolated native multi-task worktree leases, validation/review, merge currentness and durable completion before execution admission.',
        ],
        'paid_provider_calls': 0, 'training_or_downloads': False,
        'new_model_weights': False, 'new_benchmark_score': False,
        'completion_authority': False,
        'hosted_ci': 'Not qualified by these local runs.',
    }
    if (OUT / 'documentation-gates.json').is_file():
        gate = read('documentation-gates.json')
        assert gate['returncode'] == 0 and gate['source_pins_unchanged']
        report['test_only_fixture_commit'] = gate['source_commit']
        report['local_documentation_gates'] = {
            'record': 'documentation-gates.json', 'returncode': gate['returncode'],
            'allowlisted_documentation_files': gate['allowlisted_documentation_files'],
            'version': gate['version'],
        }
    (OUT / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('counts', 'distinct_passing_cases',
                     'new_profile_and_native_controls', 'manually_pinned_source_and_test_files')}))


if __name__ == '__main__':
    main()
