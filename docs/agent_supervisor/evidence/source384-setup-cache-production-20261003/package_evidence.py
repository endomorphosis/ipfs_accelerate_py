"""Close the explicitly selected, source-body-free cache implementation evidence."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
A = Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
PRIOR = B.parent / 'source384-wheel-cache-production-draft-20261003'
DRAFT = PRIOR / 'timeout-02'
OUT = A / 'docs/agent_supervisor/evidence/source384-setup-cache-production-20261003'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, v: p.write_text(json.dumps(v, indent=2, sort_keys=True) + '\n')
assert not OUT.exists(), 'closed evidence is immutable'
command = json.loads((B / 'actual-01-command.json').read_text())
exit_receipt = json.loads((B / 'actual-01-exit.json').read_text())
assert exit_receipt['returncode'] == 0 and exit_receipt['production_pins_unchanged']
assert exit_receipt['all_owners_loaded_from_actual_checkout']
cases = ET.parse(B / 'actual-01.xml').findall('.//testcase')
assert len(cases) == 204 and len({(c.attrib['classname'], c.attrib['name']) for c in cases}) == 204
assert all(c.find(kind) is None for c in cases for kind in ('failure', 'error', 'skipped'))
pins = json.loads((DRAFT / 'draft-pins.json').read_text())
for row in pins['files']:
    assert sha(A / row['path']) == row['proposed_sha256'] == command['producer_pins'][row['path']]
for name, pin in command['test_pins'].items():
    assert sha(A / name) == pin
prior_pins = json.loads((PRIOR / 'draft-pins.json').read_text())
assert sha(PRIOR / 'production-integration.patch') == prior_pins['patch_sha256']
assert all(sha(PRIOR / 'proposed' / row['path']) == row['proposed_sha256'] for row in prior_pins['files'])
selection = {}
def add(source, target):
    assert target not in selection
    assert source.is_file() and not source.is_symlink()
    selection[target] = source
for name in ('run_actual_controls.py', 'actual-01-command.json', 'actual-01-exit.json', 'actual-01.log', 'actual-01.xml'):
    add(B / name, 'actual/' + name)
for name in ('draft-pins.json', 'base-pins.json', 'source-provenance.json', 'prior-generation.json',
             'production-integration.patch', 'timeout-only.patch', 'overlay-validation.json',
             'independent-preparation-review.json', 'run_timeout_controls.py', 'run_overlay_neighbors.py'):
    add(DRAFT / name, 'draft/' + name)
for label in ('prefail-01', 'final-authored-02', 'final-neighbors-02'):
    for suffix in ('-command.json', '-exit.json', '.log', '.xml'):
        add(DRAFT / (label + suffix), 'draft/' + label + suffix)
for name in ('draft-pins.json', 'overlay-validation.json', 'independent-peer-review.json'):
    add(PRIOR / name, 'superseded/' + name)
for name in sorted(command['producer_pins'].keys() | command['test_pins'].keys()):
    add(A / name, 'sources/' + name)
add(Path(__file__), 'package_evidence.py')
OUT.mkdir(parents=True)
for target, source in selection.items():
    destination = OUT / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
qualification = dict(schema='source384-setup-cache-implementation-qualification@1',
    applied_to_actual_checkout=True, implementation_controls_passed=True,
    production_Docker_qualification=False, benchmark_trial=False,
    actual_source_suite=dict(tests=204, failures=0, errors=0, skips=0,
        pytest_seconds=40.35, runner_seconds=exit_receipt['seconds'], artifact='actual/actual-01.xml',
        runtime_source_mode='actual checkout; no overlay or method injection'),
    earlier_overlay_suites=dict(authored_tests=57, neighboring_tests=147,
        final_artifacts=['draft/final-authored-02.xml', 'draft/final-neighbors-02.xml'],
        additional_distinct_tests=0),
    regression=dict(before_tests=6, before_failures=6, artifact='draft/prefail-01.xml',
        cause='TimeoutError raised by the 60-second SIGALRM handler is a subclass of OSError; both advice loops swallowed it.',
        correction='Re-raise TimeoutError before ordinary best-effort OSError handling in archive and finite advice loops.',
        timer_scope='Registered real main handler invoked at injected syscall boundaries; no real 60-second delay.'),
    policy='source384-native-aarch64-dontneed@1', default_selection=None,
    selection=dict(archive='exact verified manifest population', codex_binaries=4,
        native_payloads=131, native_payload_bytes=565702269,
        original_source_pins_sha256='2d0625a867bc45a9a695ae2f750a57d75179c6aa28ef31250bd478f86ec7846d'),
    patch_sha256=pins['patch_sha256'], prior_patch_sha256=prior_pins['patch_sha256'],
    model_provider_Docker_calls=0, reclaimed_memory_claim=False,
    semantic_proof_repair_benchmark_success_claim=False,
    exclusions=['benchmark source bodies', 'checkpoints', 'embeddings', 'native inference reports',
        'runtime archives', 'executable or wheel payloads', 'credentials', 'private databases'],
    limits=['These controls use tiny authored public fixtures and mocked container transport.',
        'The earlier 131-payload Docker experiment failed Source384 replay admission and is not reclassified.',
        'Best-effort advice does not grant resource or proof authority; fresh ordinary qualification remains required.'])
write(OUT / 'qualification.json', qualification)
write(OUT / 'provenance.json', dict(schema='source384-cache-evidence-provenance@1',
    a_head_at_test=command['a_head'], d_head_at_test=command['d_head'],
    prior_review='superseded/independent-peer-review.json predates discovery of the swallowed alarm; use draft/independent-preparation-review.json for the corrected generation.',
    file_provenance=[dict(path=name, original_path=str(source), sha256=sha(source), bytes=source.stat().st_size)
                     for name, source in sorted(selection.items())]))
(OUT / 'README.md').write_text('''# Opt-in Source384 setup-cache implementation controls

The applied implementation adds the explicit `source384-native-aarch64-dontneed@1` setup policy. Defaults remain unchanged. The same selected path runs after deployment and the worker boundary in Harbor and the standalone qualifier, for the full Source384 arm under the common 5 CPU / 12 GiB profile.

The archive scope binds the verified runtime manifest and checks its complete bounded metadata population. Separate scopes bind four exact public Codex executable copies and 131 exact native payloads (three archive-pinned extensions plus 128 reviewed CPU-wheel members, 565,702,269 logical bytes). Finite populations retain no-follow/no-atime descriptors and verify all selected body hashes before the first hint. Fixed sizes, modes, owners, inode identity, per-file and total-byte bounds remain enforced. Advice is best effort; it neither measures reclaimed bytes nor grants admission or proof authority. No credential body or general installed-tree discovery is added.

The initial review missed a deadline bug: the 60-second alarm raises `TimeoutError`, which the two `OSError` handlers swallowed. This package preserves that review as **superseded**. Both loops now explicitly propagate the timeout. Six new controls invoke the actual main's registered handler at `fdatasync` and `posix_fadvise` in each of the archive, four-binary and 131-library paths. All six failed before the correction. They now verify immediate failure, identical exception, descriptor and timer cleanup, and no success output. A receipt-write control also preserves an existing primary timeout.

**All 204 distinct controls passed against the actual applied checkout**, with no overlay or injected implementation methods: 57 authored cache controls and 147 existing deployment, Source384 transport/qualification, Torch-wheel and Harbor controls. There were no failures or skips; pytest took 40.35 seconds. All seven runtime owners loaded from the checkout and all eight changed source/test pins matched before and after. Earlier draft-overlay runs cover the same cases and are not counted as additional tests. The corrected independent review predates application and binds the identical final bytes.

This is implementation evidence, not Docker qualification or a benchmark result. The tests use small public fixtures and mocked container transport; the timer regression injects the registered handler without waiting 60 seconds. The previous instrumented 131-payload run accepted the selected assets but failed later replay admission. A fresh ordinary archive and successful native qualification remain required. This package contains framework source, tests, patches and bounded receipts; it excludes task source bodies, model reports, weights, embeddings, binary/wheel payloads, credentials and mutable databases.
''')
inventory = [dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(OUT.rglob('*')) if p.is_file()]
write(OUT / 'manifest.json', dict(schema='closed-source384-setup-cache-implementation-evidence@1', files=inventory))
summary = dict(package=str(OUT), manifest_sha256=sha(OUT / 'manifest.json'),
    members=len(inventory), bytes=sum(row['bytes'] for row in inventory),
    scoped_added_files=[str(p.relative_to(A)) for p in sorted(OUT.rglob('*')) if p.is_file()])
write(B / 'package-ready.json', summary)
print(json.dumps({key: value for key, value in summary.items() if key != 'scoped_added_files'}, indent=2))
