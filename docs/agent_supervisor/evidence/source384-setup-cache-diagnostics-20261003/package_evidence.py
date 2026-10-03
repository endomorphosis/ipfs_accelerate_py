"""Stage explicit public metadata for completed cache diagnostics only."""
import hashlib
import json
from pathlib import Path
import shutil

WORK = Path('/home/barberb/lift_coding')
ART = WORK / 'artifacts'
HERE = Path(__file__).resolve().parent
OUT = HERE / 'public-evidence'
if OUT.exists():
    raise RuntimeError('closed staging destination must be fresh')
OUT.mkdir()
sources = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(source, relative):
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 200_000:
        raise ValueError('only explicit small regular evidence files permitted: ' + source.name)
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    sources.append({'artifact': str(source.relative_to(ART)), 'path': relative,
                    'sha256': sha(source), 'bytes': source.stat().st_size})


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


archive = ART / 'source384-docker-setup-cache-fixed-layout-20261003'
first = ART / 'source384-docker-native-codex-cache-20261003'
retry = first / 'retry-02'
common = ['cache_candidate.py', 'run_diagnostic.py', 'run-command.json', 'scope.json',
    'host-archive-provenance.json', 'before-pins.json', 'after-pins.json',
    'containers-before.json', 'containers-after.json', 'exit.json', 'failure.json',
    'cache-advice.json', 'cache-advice-status.json', 'independent-review.json', 'verdict.json',
    'docker-01/qualification.json', 'docker-01/cache-resources-before.json',
    'docker-01/cache-resources-after.json']
for label, root in [('archive-only', archive), ('binary-receipt-refusal', first), ('four-binary-child-refusal', retry)]:
    for name in common:
        put(root / name, label + '/' + name)
    if label == 'archive-only':
        for name in ('test_cache_candidate.py', 'controls.json', 'controls.log'):
            put(root / name, label + '/' + name)
    else:
        for name in ('codex_cache_candidate.py', 'test_codex_cache_candidate.py',
                     'codex-cache-advice-status.json', 'experiment-provenance.json', 'final-sources.json'):
            put(root / name, label + '/' + name)
        generation = '03' if label == 'binary-receipt-refusal' else '02'
        for suffix in ('json', 'log'):
            name = 'controls-' + generation + '.' + suffix
            put(root / name, label + '/' + name)
        for name in ('docker-01/codex-cache-advice.stdout', 'docker-01/codex-cache-advice.stderr',
                     'docker-01/worker-boundary/installation/native-codex-binary.log',
                     'docker-01/codex-cache-resources-before.json'):
            put(root / name, label + '/' + name)
    if label != 'binary-receipt-refusal':
        for name in ('docker-01/resources.json', 'docker-01/admission-estimate.json',
                     'docker-01/detailed-resources.json', 'docker-01/source384-result.json',
                     'docker-01/source384-probe-status.json', 'docker-01/source384-probe.stdout',
                     'docker-01/source384-probe.stderr'):
            put(root / name, label + '/' + name)
    if label == 'four-binary-child-refusal':
        for name in ('codex-cache-advice.json', 'docker-01/codex-cache-resources-after.json',
                     'source-free-outcome.json', 'source-free-outcome.md', 'independent-outcome-audit.json'):
            put(root / name, label + '/' + name)

put(first / 'preceding-archive-only-outcome.json', 'archive-only/source-free-outcome.json')
put(first / 'first-safe-refusal-outcome.json', 'binary-receipt-refusal/source-free-outcome.json')
put(first / 'collect_outcome.py', 'collect_outcome.py')

omitted = []
for label, root in [('archive-only', archive), ('binary-receipt-refusal', first), ('four-binary-child-refusal', retry)]:
    p = root / 'docker-01/deployment/deployment.json'
    d = json.loads(p.read_text())
    assert d['original_inputs'] == d['retained_inputs'] and len(d['original_inputs']['files']) == 218
    assert json.loads((root / 'containers-after.json').read_text())['rows'] == []
    assert json.loads((root / 'before-pins.json').read_text()) == json.loads((root / 'after-pins.json').read_text())
    omitted.append({'generation': label, 'artifact': str(p.relative_to(ART)),
        'bytes': p.stat().st_size, 'sha256': sha(p), 'reason': 'Duplicative deployment/source-hash inventory body; exact equality independently checked in source-free outcome.',
        'original_count': 218, 'original_and_retained_equal': True})
write('omitted-receipt-metadata.json', {'schema': 'omitted-original-receipt-pins@1', 'files': omitted})

for root, control in ((archive, 'controls.json'), (first, 'controls-03.json'), (retry, 'controls-02.json')):
    c = json.loads((root / control).read_text())
    assert c['returncode'] == 0
    for name, pin in c['source_sha256'].items():
        if root == first and name == 'run_diagnostic.py':
            # The independent review records the final descriptive-label-only delta.
            assert pin == '8de59aad2d680a98ef6866cab21bf7fc92f48965244ba9dc1a954cb30eb4624e'
        else:
            assert sha(root / name) == pin

scheduler = WORK / '.worktrees/ir-release-datasets-20261002/ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py'
expected_scheduler = json.loads((retry / 'before-pins.json').read_text())[str(scheduler)]
assert sha(scheduler) == expected_scheduler
write('scheduler-explanation.json', {
    'schema': 'code-and-sample-consistency-review@1',
    'source': 'ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py',
    'source_sha256': expected_scheduler, 'function': 'ResourceScheduler._can_grant',
    'inspected_lines': [1260, 1303],
    'behavior': 'The safety gate includes configured headroom plus all outstanding root envelopes. A child adds zero additional global reservation, but the active root reservation still remains in the check.',
    'single_observed_root_memory_mib': 6144, 'headroom_mib': 2458,
    'resulting_threshold_if_no_other_root_reservations': 8602,
    'failure_handling_available_mib': 8592,
    'interpretation': 'Conservative outstanding-envelope headroom is an explanation consistent with the later sample and frozen code. The sample is not the exact child admission measurement; no decision receipt or backlog state proves the precise rejection cause.',
    'child_double_counting_claimed': False, 'policy_change_proposed_or_applied': False,
})

uv_manifest = WORK / '.worktrees/ir-release-accelerate-20261002/docs/agent_supervisor/evidence/source384-uv-layout-fix-20261003/manifest.json'
assert sha(uv_manifest) == '07bf0fc0629bc12c13dad2881c55d8b96346e3720a961039710ec7b534539ac0'
write('qualification.json', {
    'schema': 'completed-setup-cache-diagnostics-evidence@1',
    'diagnostic_only': True, 'runtime_archive_sha256': 'a83b157f478dbcb869f134e71bfd02b17bdb7f82f3de8d281cc6202f66464d0b',
    'host_deployer_sha256': 'bb4e4f845f03026572530b81fde6a540a2cd0ab89b17e606587ddfbc33dfc285',
    'archived_deployer_sha256': 'c4e3dcc3aa2fae3d77fa538ec4855778d26a5cd70392ff54d4ae6b399653869d',
    'uv_fix_evidence_manifest_sha256': sha(uv_manifest),
    'generations': [
        {'directory': 'archive-only', 'outcome': 'root lease refused', 'archive_hints': 26905,
         'archive_advised_bytes': 3914232094, 'failure_available_mib': 8195,
         'minimum_root_mib': 8602, 'helper_controls': 12, 'controls_inherited_unchanged': True},
        {'directory': 'binary-receipt-refusal', 'outcome': 'public receipt protection refused',
         'archive_hints': 26905, 'native_binary_hints': 0, 'native_binary_body_reads': 0,
         'native_context_reached': False, 'helper_controls': 14,
         'metadata_limit': 'Uploaded source was0664; destination stat was not retained, so exact destination ownership is not retrospectively asserted.'},
        {'directory': 'four-binary-child-refusal', 'outcome': 'root admitted; first index child lease refused',
         'archive_hints': 26905, 'native_binary_hints': 4, 'native_binary_hashed_and_advised_bytes': 628736528,
         'helper_controls': 18, 'failure_available_mib': 8592,
         'index_snapshot_publication_or_model_inference_reached': False,
         'owner_store_initialization_may_have_occurred': True},
    ],
    'all_generations_qualified': False, 'actual_resource_profile': '5 CPU /12288 MiB',
    'original_inputs_preserved_across_each_deployment': 218, 'all_containers_removed': True,
    'provider_calls': 0, 'official_verifier_executed': False, 'model_loads': None, 'coverage': None,
    'advised_bytes_are_freed_bytes': False, 'admission_or_deadline_changes': False,
    'production_cache_policy_changed': False, 'seven_payload_experiment_included': False,
    'benchmark_result': False, 'advantage_claimed': False, 'proof_authority': False,
    'execution_authority': False, 'completion_authority': False,
    'excluded': ['private auth', 'task/instruction/source/diff bodies', 'hidden verifier inputs',
        'Source384 inference/report bodies', 'embeddings and checkpoints', 'executable or runtime archive payloads',
        'databases', 'pending seven-payload experiment'],
})

(OUT / 'README.md').write_text('''# Completed setup-cache diagnostics after the UV layout fix

These are three separate instrumented failures. They establish narrow cache-advice behavior and failure locations, not production qualification or a benchmark improvement. The earlier UV migration fix is separately sealed in `source384-uv-layout-fix-20261003` (manifest07bf0fc0…). Its host deployer uses the nested Python root; these experiments intentionally reuse the older immutable a83b157f archive. The archived deployer is not the installer used for setup.

| Generation | Successful advice | Native outcome |
|---|---|---|
| `archive-only/` |26905 archive files,3914232094 bytes | Root lease refused; failure sample8195 MiB, root minimum8602 MiB. |
| `binary-receipt-refusal/` | Same archive population | Public receipt protection refused before binary body reads/advice or native context. Source receipt mode0664 was retained; no destination stat was captured. |
| `four-binary-child-refusal/` | Same archive population plus4 independently hashed public executable copies,628736528 bytes | Root admitted; first `index.prepare_current` child lease refused before snapshot/publication or inference. Owner-store initialization may already have occurred. |

The final run observed5256→8194 MiB across archive advice, then8194→8812 MiB across four-binary advice. Actual cgroups were5 CPU/12288 MiB. Native initial context failed after37.209 seconds; failure-handling availability was8592 MiB. The native probe lasted48.697 seconds and the whole diagnostic298.584 seconds. The exact trace and pinned consumer/index-owner control flow establish the admitted-root/first-child location. They do not retain the exact child decision sample or scheduler state.

The frozen scheduler conservatively checks outstanding root envelopes plus headroom, while adding zero new global memory for a child.6144+2458=8602 MiB is therefore consistent with a headroom refusal given the later8592 MiB sample. This is an explanation consistent with code and observations, not proof of the precise rejection cause. No child double counting, policy weakening, deadline increase or production cache policy is claimed or applied.

The authored helper suites remain distinct:12 inherited archive controls,14 first-binary controls, and18 receipt-fix controls. Counts are per generation and are not added into a unique test total. Exact helper/test sources and command/log receipts are retained. The first binary controls pin wrapper8de59aad; its final cfdae554 generation changed descriptive labels only, as recorded by the independent review. Earlier development controls remain in the local artifact directories and are not counted here. The archive helper review predates the UV cause investigation; its unchanged byte pin is the reason that review and controls carry forward.

Every advice helper reported metadata unchanged and no advice errors in stages that completed. Advice bytes are not measured freed bytes. The first binary attempt intentionally refused an unprotected public transport receipt; the retry protected only that checked singleton receipt via descriptor ownership/mode changes, preserving strict executable and SHA checks. No credentials were read by these advice helpers. All218 original files remained hash-equal across each deployment, source pins stayed stable, and cleanup observations found no matching containers. Original deployment inventory bodies are retained locally and described in `omitted-receipt-metadata.json` rather than duplicated here.

Source-free outcomes retain null model-load/coverage counters: no published inference artifact exists for these runs. Provider calls and official verifier execution remain zero/false. The four-binary outcome has an independent trace/receipt audit. Raw source, private authentication, verifier inputs, model bodies/embeddings/checkpoints, executable payloads and databases are excluded. Public framework helper sources and failure stack traces are included; no benchmark source body is present.

This staging package excludes the next seven-payload experiment. Scripts and embedded receipt paths preserve their original host execution layout; package directories group generations, and `provenance.json` maps copies. A script is provenance, not an instruction to rerun the benchmark. No closed earlier package or maintained repository document is modified.
''')
put(Path(__file__), 'package_evidence.py')
write('provenance.json', {'schema': 'explicit-whitelist-provenance@1', 'copies': sources})
rows = [{'path': str(path.relative_to(OUT)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
        for path in sorted(OUT.rglob('*')) if path.is_file()]
write('manifest.json', {'schema': 'closed-completed-setup-cache-diagnostics-evidence@1', 'files': rows})
result = {'staging_package': str(OUT), 'manifest_sha256': sha(OUT / 'manifest.json'),
          'members': len(rows), 'member_bytes': sum(row['bytes'] for row in rows),
          'files': [row['path'] for row in rows] + ['manifest.json']}
(HERE / 'scoped-files.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'files'}, sort_keys=True))
