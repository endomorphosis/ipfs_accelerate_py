"""Publish bounded evidence only after a frozen, passing merged qualification."""
from pathlib import Path
import json
import subprocess
import sys

OUT = Path(__file__).parent
ROOT = Path('/home/barberb/lift_coding/.worktrees/supervisor-context-gaps-20261004')
DESTINATION = ROOT / 'docs/agent_supervisor/evidence/supervisor-ordinary-identity-20261005'
name = sys.argv[1]
run = json.loads((OUT / f'{name}.qualification.json').read_text())
before = json.loads((OUT / f'{name}-before.json').read_text())
after = json.loads((OUT / f'{name}-after.json').read_text())
baseline = json.loads((OUT / 'existing-owner-baseline-01-before.json').read_text())
assert run['returncode'] == 0 and run['pins_unchanged'] and before == after
assert run['counts']['passed'] > 0
assert all(run['counts'][kind] == 0 for kind in ('failure', 'error', 'skipped'))
assert run['ast_sealing']['records'] == run['counts']['passed']
assert run['ast_sealing']['any_completion_authority'] is False
assert before['datasets_tracked_status'] == ''
assert len(before['retained_assets']) == 4
assert before['retained_assets'] == baseline['retained_assets']
ordinary = {key: value for key, value in run['case_outcomes'].items()
            if 'test_ordinary_supervisor_' in key}
assert len(ordinary) == 55 and set(ordinary.values()) == {'passed'}
native = json.loads((OUT / f'{name}.native-process-audit.json').read_text())
assert native['matching_live_processes'] == []
assert native['unavailable_worker_observations'] == []
assert native['signal_calls'] == 0
subprocess.run([sys.executable, str(OUT / 'collect_public_evidence.py')], check=True)
summary = {
    'schema': 'ordinary-supervisor-qualification@1',
    'implementation_baseline': baseline['base_commit'],
    'owned_implementation_commit': '082fc4769',
    'corrected_source_commit': before['base_commit'],
    'integrated_upstream_revision': subprocess.check_output(
        ['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip(),
    'qualified_merged_source_commit': before['base_commit'],
    'final_run': f'{name}.qualification.json',
    'counts': run['counts'],
    'distinct_passing_cases': len(run['case_outcomes']),
    'ast_sealing': run['ast_sealing'],
    'source_assets_and_datasets_pins_unchanged': True,
    'current_source_pins': f'{name}-before.json',
    'datasets_commit': before['datasets_commit'],
    'retained_assets': before['retained_assets'],
    'selection': 'final-selection.json',
    'independent_review': 'independent-review.json',
    'ordinary_native_and_cli_controls': len(ordinary),
    'native_process_audit': f'{name}.native-process-audit.json',
    'original_cleanup_and_staged_diagnostics': 'cleanup/ordinary-native-custody-audit.json',
    'signal_finally_reconciliation': 'cleanup/ordinary-signal-reconciliation-audit.json',
    'joined_regression_diagnostic': {'evidence': 'final-01.qualification.json',
        'counts': {'passed': 629, 'failure': 16, 'error': 0, 'skipped': 0},
        'cause': 'Valid exact --implement was incorrectly classified as an abbreviation of --implementation-protected-path; the production predicate was corrected and all original nodes retained.'},
    'historical_baseline': {
        'evidence': 'existing-owner-baseline-01.qualification.json',
        'counts': {'passed': 29, 'failure': 29, 'error': 0, 'skipped': 0},
        'absent_api_failures': 19,
        'legacy_behavior_or_fixture_failures': 10,
        'classification': 'All historical failed nodes are recorded separately from the final selected passing nodes; no retired APIs are restored.'
    },
    'observer_fixture_adaptation': {
        'evidence': 'existing-owner-join-01.qualification.json',
        'diagnostic_counts': {'passed': 27, 'failure': 2, 'error': 0, 'skipped': 0},
        'change': 'Two synthetic adoption fixtures now explicitly model current birth, dedicated SID/PGID and liveness; unreadable argv still refuses and never migrates ownership.'
    },
    'decomposition_baseline': {
        'evidence': 'decomposition-survey.json', 'passing_checks': 127,
        'count_scope': 'Separate unchanged-base survey; not added to this qualification.'
    },
    'remaining_plan': [
        'Opt-in terminal-public-task-profile@2 with at most 16 explicitly reviewed task/operation bindings and exact instruction, requirement, output, dependency and validation identities.',
        'Per-task source/dependency-wave contexts preserving selected IR family, schema, decoder task, dimension and retained asset identity.',
        'Explicit native multi-task execution scope with isolated worktrees/leases, readiness, review/validation/merge currentness and cold restart qualification.'
    ],
    'limitations': [
        'Actual kernel PID reuse was not forced; durable birth and boot mismatch are tested against real local processes.',
        'The existing userspace fence does not provide cgroup containment for descendants that escaped ancestry and captured groups before the first census; live dedicated-session members still block quiescence.',
        'Generic reviewed multi-task execution remains planned; this change closes ordinary child ownership and marker/state retirement.',
        'The historical broader failures remain separate; this is not a claim that the entire archived suite passes.'
    ],
    'paid_provider_calls': 0,
    'training_or_downloads': False,
    'new_model_weights': False,
    'new_benchmark_score': False,
    'completion_authority': False,
    'hosted_ci': 'Not qualified by these local runs.'
}
(DESTINATION / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
doc = ROOT / 'docs/agent_supervisor/terminal_symbolic_capabilities.md'
text = doc.read_text()
needle = 'Native process tests use only their own benign sessions; no model providers,\ntraining or checkpoint changes are involved.'
replacement = needle + (
    f'\n\nThe merged final selection passes **{run["counts"]["passed"]} distinct tests**, with no failures,\n'
    'errors or skips. Every passing case has an AST seal in a fresh catalog;\n'
    'source, dataset and retained-asset hashes stayed unchanged during the run.\n'
    'The 55 new adoption, CLI and cleanup controls are included in that total.\n'
    'The 29 failures from the broader unchanged-base owner diagnostic remain\n'
    'separate: 19 reference absent APIs and 10 reflect legacy behavior or fixtures.\n'
    'The selection retains 29 previously passing owner controls; two synthetic\n'
    'adoption fixtures were updated to model the newly required native session\n'
    'observations. No retired APIs were restored. Actual kernel PID reuse was not\n'
    'forced, and the existing process fence does not provide cgroup containment\n'
    'for workers that escaped before its first census. No executable test-owned\n'
    'process was observed in the final audited fixture roots.\n'
)
assert text.count(needle) == 1
doc.write_text(text.replace(needle, replacement))
print(json.dumps({'counts': run['counts'], 'ordinary_controls': len(ordinary),
                  'assets_unchanged': True, 'source_base': before['base_commit']}))
