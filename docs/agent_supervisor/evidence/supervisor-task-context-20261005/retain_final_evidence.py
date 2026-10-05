"""Retain only completed, frozen qualification facts; no model or network actions."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/home/barberb/lift_coding/.worktrees/supervisor-task-context-20261005')
OUT = Path(__file__).parent
EVIDENCE = ROOT / 'docs/agent_supervisor/evidence/supervisor-task-context-20261005'


def read(name):
    return json.loads((OUT / name).read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def main():
    final = read('final-01.qualification.json')
    before = read('final-01-before.json')
    after = read('final-01-after.json')
    baseline = read('baseline-01.qualification.json')
    audit = read('final-01.native-process-audit.json')
    assert final['returncode'] == 0 and final['pins_unchanged'] and before == after
    assert final['counts'] == {'passed': 445, 'failure': 0, 'error': 0, 'skipped': 0}
    assert len(final['case_outcomes']) == 445
    assert final['ast_sealing']['records'] == 445 and not final['ast_sealing']['any_completion_authority']
    assert set(baseline['case_outcomes']) <= set(final['case_outcomes'])
    assert len(before['retained_assets']) == 13 and len(before['source_files']) == 65
    assert not audit['matching_live_processes'] and not audit['unavailable_worker_observations']
    assert all(digest(ROOT / name) == pin for name, pin in before['source_files'].items())
    new_native = [node for node in final['case_outcomes'] if node.startswith('benchmarks.agent_supervisor.container_coding.test_terminal_multitask_context::')]
    new_metadata = [node for node in final['case_outcomes'] if node.startswith('test.api.test_task_ir_selection::')]
    assert len(new_native) == 67 and len(new_metadata) == 40
    for suffix in ['-before.json', '-after.json', '.command.sh', '.log', '.qualification.json', '.xml', '.native-process-audit.json']:
        shutil.copy2(OUT / ('final-01' + suffix), EVIDENCE / ('final-01' + suffix))
    shutil.copy2(Path(__file__), EVIDENCE / Path(__file__).name)
    qualification = {
        'schema': 'supervisor-task-context-qualification@1',
        'implementation_baseline': before['base_commit'],
        'owned_implementation_commit': None,
        'qualified_source_base_commit': before['base_commit'],
        'phase': 'advisory independent ready-root context and exact persisted IR metadata nomination',
        'public_profile_schema': 'terminal-public-task-profile@3',
        'receipt_schema': 'terminal-reviewed-ready-task-contexts@1',
        'final_run': 'final-01.qualification.json',
        'counts': final['counts'],
        'distinct_passing_cases': len(final['case_outcomes']),
        'selection': 'final-selection-01.json',
        'ast_sealing': final['ast_sealing'],
        'current_source_pins': 'final-01-before.json',
        'manually_pinned_source_and_test_files': len(before['source_files']),
        'source_assets_and_datasets_pins_unchanged': True,
        'datasets_commit': before['datasets_commit'],
        'retained_assets': before['retained_assets'],
        'retained_assets_scope': 'Thirteen exact existing files: nine selected catalog checkpoint witnesses plus four preceding embedding/decoder assets. No claim to inventory every model.',
        'new_native_context_controls': len(new_native),
        'new_exact_catalog_controls': len(new_metadata),
        'baseline': {'run': 'baseline-01.qualification.json', 'counts': baseline['counts'], 'all_baseline_nodes_retained': True},
        'count_scope': 'Only final-01 distinct nodes contribute to the final total. Baseline and development selections overlap and are not added.',
        'independent_review': 'independent-review.json',
        'historical_diagnostics': 'development-diagnostics.json',
        'native_process_audit': 'final-01.native-process-audit.json',
        'catalog_survey': {'record': 'live-model-manager-catalog-survey.json', 'scope': 'one explicit existing ModelManager DuckDB metadata store', 'declarations': 668, 'complete_schema_task_profile_format_identities': 2, 'nomination_record': 'retained-ir-catalog-nominations.json', 'existing_exact_nominations': 9, 'missing_768_families_in_this_store': ['codebase_ir', 'security_ir', 'intent_ir'], 'checkpoint_witness_record': 'retained-checkpoint-file-witnesses.json'},
        'limitations': [
            'Authored reviewed inputs and lexical test vectors do not measure GTE decoder reconstruction or query semantic alignment.',
            'Existing source ASTs/capsules are rebuilt; supplied numerical retrieval vectors are reused.',
            'New positive native fixture covers two independent roots in a three-task graph; bounds do not qualify sixteen-root capacity.',
            'Controlled source/native mutations test cooperative currentness; observations are not atomic FS/native/catalog snapshots.',
            'Existing Source384 regression tests are authored transport controls with a validator stub, not inference qualification. New plain-route guards do not invoke that validator.',
            'Exact catalog metadata nominations do not authenticate checkpoint/Hub bytes or admit decoder runtimes; separate filesystem witnesses leave those flags false.',
            'Dependent contexts, reviewed-profile launch, parallel worker execution and completion remain unqualified.',
        ],
        'remaining_plan': [
            'Recover missing schema/task/profile/format identities from original validated checkpoint manifests, preserving separate format registrations and ablations.',
            'Join exact family/schema/task/dimension geometry to existing encoder/cache/checkpoint/Hub revisions, separate DuckDB/DuckLake inventories and verified token/span/runtime receipts.',
            'Bind dependent contexts to checked predecessor publications and the resulting source successor.',
            'Qualify parallel isolated worktree leases, validation/review, merge currentness and durable completion/restart before execution admission.',
            'Keep autoformalized candidates separate from checked proofs; reuse old teachers, embeddings, heads and held-out splits for qualified distillation.',
        ],
        'new_path_provider_calls': 0,
        'new_path_embedding_calls': 0,
        'new_path_model_loading_calls': 0,
        'new_path_training_steps': 0,
        'retained_model_assets_regenerated': False,
        'new_benchmark_score': False,
        'execution_authority': False,
        'completion_authority': False,
        'proof_authority': False,
        'hosted_ci': 'Not qualified by these local runs; publication readback will inspect the actual hosted check.',
    }
    (EVIDENCE / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    results = '\n## Frozen qualification\n\nThe [final regression](qualification.json) passed **445 distinct checks**, with\nzero failures, errors or skips. It includes all 125 unchanged baseline cases,\n67 new native context controls and 40 new exact catalog controls. All 445 have\nfresh AST seals without completion authority. The 65 manual source/test pins,\npinned datasets checkout and thirteen retained files stayed unchanged. The\n[exact selection](final-selection-01.json), raw JUnit/log/commands and before/after\nwitnesses are retained; overlapping development/baseline runs are not added.\nThe scoped final native-process audit observed no live matching fixture process\nor unavailable recorded worker and sent no signals.\n'
    path = EVIDENCE / 'README.md'
    assert '## Frozen qualification' not in path.read_text()
    path.write_text(path.read_text() + results)
    path = ROOT / 'docs/agent_supervisor/terminal_symbolic_capabilities.md'
    text = path.read_text()
    marker = '## Ready-root contexts and exact retained IR nominations (2026-10-05)\n\n'
    paragraph = 'The [frozen context qualification](evidence/supervisor-task-context-20261005/qualification.json)\npassed **445 checks** with zero failures, errors or skips, including all 125\nunchanged baseline nodes, 67 new native context controls and 40 exact catalog\ncontrols. All cases have fresh AST seals without completion authority. The 65\nmanual source/test pins, datasets checkout and thirteen existing retained files\nstayed unchanged. Baseline and development runs overlap and are not added.\n\n'
    assert marker in text and paragraph not in text
    path.write_text(text.replace(marker, marker + paragraph))
    print(json.dumps({'passed': 445, 'new_native': 67, 'new_metadata': 40, 'baseline_retained': 125, 'pins_unchanged': True}))


if __name__ == '__main__':
    main()
