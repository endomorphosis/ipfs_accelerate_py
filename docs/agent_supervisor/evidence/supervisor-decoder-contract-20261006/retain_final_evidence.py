"""Retain exact completed frozen regression facts and explicit scopes."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/home/barberb/lift_coding/.worktrees/supervisor-decoder-contract-20261006')
OUT = Path(__file__).parent
EVIDENCE = ROOT / 'docs/agent_supervisor/evidence/supervisor-decoder-contract-20261006'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def main():
    final = json.loads((OUT / 'final-02.qualification.json').read_text())
    before = json.loads((OUT / 'final-02-before.json').read_text())
    after = json.loads((OUT / 'final-02-after.json').read_text())
    prior = json.loads((ROOT / 'docs/agent_supervisor/evidence/supervisor-task-context-20261005/final-01.qualification.json').read_text())
    audit = json.loads((OUT / 'final-02.native-process-audit.json').read_text())
    assert final['returncode'] == 0 and final['pins_unchanged'] and before == after
    assert final['counts'] == {'passed': 512, 'failure': 0, 'error': 0, 'skipped': 0}
    assert len(final['case_outcomes']) == 512
    assert final['ast_sealing']['records'] == 512 and not final['ast_sealing']['any_completion_authority']
    assert set(prior['case_outcomes']) <= set(final['case_outcomes'])
    assert len(before['retained_assets']) == 16 and len(before['source_files']) == 68
    assert not audit['matching_live_processes'] and not audit['unavailable_worker_observations']
    assert all(digest(ROOT / name) == pin for name, pin in before['source_files'].items())
    new_native = [node for node in final['case_outcomes'] if node.startswith('benchmarks.agent_supervisor.container_coding.test_terminal_checkpoint_context::')]
    new_bytes = [node for node in final['case_outcomes'] if node.startswith('test.api.test_task_ir_checkpoint::')]
    assert len(new_native) == 26 and len(new_bytes) == 41
    for suffix in ['-before.json', '-after.json', '.command.sh', '.log', '.qualification.json', '.xml', '.native-process-audit.json']:
        shutil.copy2(OUT / ('final-02' + suffix), EVIDENCE / ('final-02' + suffix))
    for name in ['final-selection-01.json', 'audit_native_processes.py', 'run_qualification.py', Path(__file__).name]:
        shutil.copy2(OUT / name, EVIDENCE / name)
    qualification = {
        'schema': 'supervisor-decoder-contract-qualification@1',
        'implementation_baseline': before['base_commit'],
        'owned_implementation_commit': None,
        'qualified_source_base_commit': before['base_commit'],
        'phase': 'original checkpoint byte authentication for advisory independent ready-root contexts',
        'receipt_schema': 'terminal-reviewed-ready-task-contexts@2',
        'default_receipt_schema': 'terminal-reviewed-ready-task-contexts@1',
        'final_run': 'final-02.qualification.json',
        'counts': final['counts'],
        'distinct_passing_cases': len(final['case_outcomes']),
        'selection': 'final-selection-01.json',
        'ast_sealing': final['ast_sealing'],
        'current_source_pins': 'final-02-before.json',
        'manually_pinned_source_and_test_files': len(before['source_files']),
        'source_assets_and_datasets_pins_unchanged': True,
        'datasets_commit': before['datasets_commit'],
        'retained_assets': before['retained_assets'],
        'retained_files_scope': '16 files: preceding13 assets plus2 stronger contextual states and original profile inventory. This preserves bytes and does not perform model quality replay.',
        'new_checkpoint_byte_controls': len(new_bytes),
        'new_native_context_controls': len(new_native),
        'prior_ready_root_regression': {'record': '../supervisor-task-context-20261005/final-01.qualification.json', 'counts': prior['counts'], 'all 445_prior_nodes_retained': True},
        'count_scope': 'Only512 distinct final-02 nodes contribute. Focused/development/prior cases overlap and are not added; interrupted final-01 is incomplete.',
        'independent_review': 'independent-source-review.json',
        'historical_diagnostics': 'development-diagnostics.json',
        'native_process_audit': 'final-02.native-process-audit.json',
        'actual_retained_authentication': {'record': 'retained-checkpoint-authentication.json', 'authenticated_existing_files': 8, 'authenticated_existing_bytes': 105868568, 'original_nine_batch_refusal': 'Codebase384 original locator traverses an ancestor current alias; requires reviewed canonical binding'},
        'native_profile_custody': {'survey': 'native-profile-custody-survey.json', 'rebind': 'original-profile-custody-rebind-receipt.json', 'original_routes_refused': 2, 'rebound_routes_resolved': 2, 'lanes_preserved': 12, 'supported_format_scope': 'Original Intent/Security384 serialized fragments only', 'native_owner_commit': '73db2c8f3edb9fbdfc5decb0e8f12bb4e86e10f3', 'mixed_into_987_runtime': False},
        'stronger_retained_reconstruction': {'survey': 'retained-decoder-contract-survey.json', 'scope': 'Historical cached paragraph/clause input semantic replay, not new inference/generalization/sourceprose/8192 qualification', 'contextual384_exact_ir': '48/48', 'contextual768_exact_ir': '48/48', 'selected_store_exact_sha_matches': 0, 'fresh_numerical_replay_performed': False},
        'limitations': [
            'Byte authentication does not deserialize or verify tensor shape/ABI, family output schema, token/span/input contracts, inventory or remoteHub bytes.',
            'Native catalog flags remain unchanged; only separate local byte observations become true.',
            'Source/native/catalog/file fences are cooperative endpoint observations, not atomic snapshots or leases.',
            'Authored native byte fixtures do not qualify GTE model inference; inherited Source384 tests are transport controls with separatevalidator stubs.',
            'Contextual historical success includes cachedclausevectors and masks; no single-vector, independentholdout, originalprose or8192 decodingclaim.',
            'Original supported nativeprofile rebind does not fill7unsupported legacy namespaces or admitruntime.',
            'Defaultmetadata nomination, dependenttask refusal and reviewed-profile executionguards remain.',
        ],
        'native_profile_publication': 'native-profile-publication-survey.json',
        'remaining_plan': [
            'Recover and qualify the complete original native metadata-owner closure against datasets main, where its committed historical file is absent; preserve current API and exact codec/profile identity.',
            'Review append-only input/output format contracts and publish/import exact complete strongercontextual384/768 states with originalcachedparagraph/clausevectors, masks and codecs; preserve unknownhistoricalrecords.',
            'Bind Codebase384 to reviewedcanonical originalbytes with borrowedSecurity payload and incomplete-native status preserved.',
            'Separate semanticIR, source-text reconstruction and FOL/TDFOL decoder tasks; count source-form anchors/residuals and evaluate ablations.',
            'Qualify exact tensor/ABI/encoder/token/span/inventory/Hub runtime contracts before decoderadmission; reuse8/384 teacherheads for768 interface/distillation.',
            'Qualify dependentcontexts against predecessor publications and then parallelleases/review/mergecurrentness/durablecompletion.',
        ],
        'new_path_embeddings': 0, 'new_path_training_steps': 0, 'new_path_model_loads': 0, 'new_path_provider_calls': 0,
        'model_manager_writes': False, 'remote_hub_writes': False, 'retained_model_assets_regenerated': False,
        'new_benchmark_score': False, 'decoder_runtime_admitted': False,
        'execution_authority': False, 'completion_authority': False, 'proof_authority': False,
        'hosted_ci': 'Separate publication check; these are local qualifications.',
    }
    (EVIDENCE / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    p = EVIDENCE / 'README.md'
    assert '## Frozen qualification' not in p.read_text()
    p.write_text(p.read_text() + '\n## Frozen qualification\n\nThe [final regression](qualification.json) passed **512 distinct checks**,\nwith zero failures, errors or skips. It retains all 445 preceding ready-root\nregression nodes and adds 41 byte-authentication and 26 native context controls.\nAll512 cases have fresh AST seals without completion authority. The 68 manual\nsource/test pins, datasets checkout and16 retained files remained unchanged.\nExact selections, raw logs/XML/commands and before/after witnesses are retained.\nOverlapping and interrupted runs are not added. The scoped native-process audit\nobserved no live matching fixture process or unavailable recorded worker.\n')
    p = ROOT / 'docs/agent_supervisor/terminal_symbolic_capabilities.md'
    s = p.read_text();marker='## Retained checkpoint authentication and decoder recovery (2026-10-06)\n\n'
    paragraph='The [frozen qualification](evidence/supervisor-decoder-contract-20261006/qualification.json)\npassed **512 checks** with zero failures, errors or skips, including all 445\npreceding ready-root regression nodes, 41 new byte-authentication controls and 26\nnew native context controls. All cases have fresh AST seals without completion\nauthority. The 68 manual source/test pins, datasets checkout and16 retained files\nstayed unchanged. Focused/development runs overlap and are not added.\n\n'
    assert marker in s and paragraph not in s
    p.write_text(s.replace(marker,marker+paragraph))
    print(json.dumps({'passed':512,'new_byte_controls':41,'new_native_controls':26,'prior_nodes_retained':445,'retained_files':16}))


if __name__ == '__main__':
    main()
