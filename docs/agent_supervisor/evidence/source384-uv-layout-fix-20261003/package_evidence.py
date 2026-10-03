"""Package bounded, source-free causal receipts; excludes pending experiments."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

WORK = Path('/home/barberb/lift_coding')
ART = WORK / 'artifacts'
FIX = ART / 'source384-uv-layout-fix-20261003'
CACHE = ART / 'source384-docker-setup-cache-20261003'
TRANSITION = ART / 'source384-docker-layout-transition-20261003'
OUT = WORK / '.worktrees/ir-release-accelerate-20261002/docs/agent_supervisor/evidence/source384-uv-layout-fix-20261003'
if OUT.exists():
    raise RuntimeError('new evidence destination required; closed packages are immutable')
OUT.mkdir(parents=True)
copied = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    assert source.is_file() and not source.is_symlink()
    assert source.stat().st_size < 300_000, source
    destination = OUT / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied.append({'artifact': str(source.relative_to(ART)), 'package': target,
                   'bytes': source.stat().st_size, 'sha256': sha(source)})


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


fix_names = ['before-pins.json', 'final-sources.json', 'focused-02-command.json',
    'focused-02-exit.json', 'focused-02.log', 'focused-02.xml', 'native-uv.json',
    'uv-download.log', 'download-command.json', 'qualification.json', 'independent-review.json']
for name in fix_names:
    copy(FIX / name, 'fix/' + name)
for directory in ('before', 'final-sources'):
    for path in sorted((FIX / directory).rglob('*.py')):
        copy(path, 'fix/' + str(path.relative_to(FIX)))
copy(ART / 'source384-uv-migration-20261003/primary-source-review.json', 'primary-source-review.json')

common_cache = ['cache_candidate.py', 'test_cache_candidate.py', 'run_diagnostic.py',
    'run-command.json', 'scope.json', 'before-pins.json', 'after-pins.json',
    'containers-before.json', 'containers-after.json', 'exit.json', 'failure.json',
    'cache-advice-status.json', 'independent-review.json',
    'docker-01/cache-advice.stdout', 'docker-01/cache-advice.stderr',
    'docker-01/qualification.json', 'docker-01/cache-resources-before.json']
for name in common_cache + ['controls-02.json', 'controls-02.log',
        'wrapper-final-validation.json', 'verdict-initial.json']:
    copy(CACHE / name, 'refusal-01/' + name)
for name in common_cache + ['controls.json', 'controls.log', 'complete-tar-topology.json']:
    copy(CACHE / 'retry-02' / name, 'refusal-02/' + name)

transition_names = ['layout_observer.py', 'run_diagnostic.py', 'test_layout_observer.py',
    'controls.json', 'controls.log', 'independent-review.json', 'run-command.json',
    'scope.json', 'before-pins.json', 'after-pins.json', 'containers-before.json',
    'containers-after.json', 'container-identity.json', 'exit.json', 'failure.json',
    'layout-events.jsonl', 'layout-transition.json', 'verdict.json', 'docker-01/qualification.json']
for name in transition_names:
    copy(TRANSITION / name, 'transition/' + name)

owner = 'benchmarks/agent_supervisor/container_coding/terminal_deployment.py'
test = 'benchmarks/agent_supervisor/container_coding/test_terminal_uv_layout.py'
before = (FIX / 'before' / owner).read_text().splitlines(keepends=True)
after = (FIX / 'final-sources' / owner).read_text().splitlines(keepends=True)
patch = ''.join(difflib.unified_diff(before, after, fromfile='a/' + owner, tofile='b/' + owner))
patch += ''.join(difflib.unified_diff([], (FIX / 'final-sources' / test).read_text().splitlines(keepends=True),
                                    fromfile='/dev/null', tofile='b/' + test))
(OUT / 'fix/production.patch').write_text(patch)

suites = list(ET.parse(FIX / 'focused-02.xml').getroot().iter('testsuite'))
counts = {key: sum(int(row.attrib.get(key, '0')) for row in suites)
          for key in ('tests', 'failures', 'errors', 'skipped')}
assert counts == {'tests': 56, 'failures': 0, 'errors': 0, 'skipped': 0}
observed = json.loads((TRANSITION / 'layout-transition.json').read_text())
assert observed['prior_kind'] == 'directory'
assert observed['confirmation']['toolchains']['link_target'] == '/opt/ipfs-supervisor/python'
for root in (CACHE, CACHE / 'retry-02', TRANSITION):
    assert json.loads((root / 'containers-after.json').read_text())['rows'] == []

write('qualification.json', {
    'schema': 'source384-uv-layout-cause-and-fix-evidence@1',
    'archive_sha256': 'a83b157f478dbcb869f134e71bfd02b17bdb7f82f3de8d281cc6202f66464d0b',
    'archive_manifest_sha256': '4d13cfeb3ad2daf67a50c135babbf58c6a197a71f70376bc730051508ce8fb27',
    'cache_preflight_refusals': {
        'first': 'refusal-01/verdict-initial.json',
        'refined': 'refusal-02/docker-01/cache-advice.stdout',
        'advice_issued': False, 'native_context_started': False,
        'refined_row_index': 11707, 'refined_path': 'toolchains/lean/bin/cadical',
        'refined_ancestor_kind': 'symlink', 'containers_cleaned': True,
        'authored_controls_first': 10, 'authored_controls_refined': 12,
    },
    'layout_transition': {
        'receipt': 'transition/layout-transition.json', 'verdict': 'transition/verdict.json',
        'operation_number': 7, 'operation': 'python-runtime-install',
        'before': 'directory', 'after': 'symlink', 'target': '/opt/ipfs-supervisor/python',
        'authored_controls': 5, 'provider_calls': 0, 'official_verifier_executed': False,
        'native_context_started': False, 'cache_advice_issued': False, 'container_cleaned': True,
    },
    'fix_controls': {'receipt': 'fix/qualification.json', 'xml': 'fix/focused-02.xml',
        **counts, 'real_uv_version': '0.9.24', 'native_execution_offline': True,
        'native_interpreter_install_success_claimed': False,
        'new_managed_root': '/opt/ipfs-supervisor/python-runtime/python',
        'source_sha256': sha(FIX / 'final-sources' / owner),
        'test_sha256': sha(FIX / 'final-sources' / test)},
    'pending_fixed_layout_cache_experiment_included': False,
    'fixed_layout_docker_or_inference_success_claimed': False,
    'admission_or_budget_changes': False, 'cache_advice_added_to_production': False,
    'benchmark_advantage_claimed': False, 'proof_authority': False,
    'execution_authority': False, 'completion_authority': False,
    'reference_resolution': 'Original artifact scripts/receipts preserve host paths. Package-local paths are defined here and in provenance.json; omitted intermediate controls are not part of final test counts.',
    'excluded': ['credential contents', 'raw task/source/instruction/diff bodies', 'verifier inputs',
        'model/checkpoint reports and weights', 'databases', 'UV wheel/binary', 'runtime archive',
        'full archive inventory', 'pending fixed-layout cache/inference experiment'],
})
write('provenance.json', {'schema': 'scoped-evidence-copy-provenance@1', 'copies': copied,
    'omitted_intermediate_uv_controls': 'focused-01 passed56 before the final nested-directory assertion; final focused-02 is the only claimed UV suite.'})

(OUT / 'README.md').write_text('''# UV toolchain-directory collision: cause and bounded fix

Pinned `uv==0.9.24` renamed the shipped Lean directory because its selected Python installation directory did not exist and had a sibling named `toolchains`. The fix selects `/opt/ipfs-supervisor/python-runtime/python` for both `uv python install` and `uv venv`. The public venv executable, Lean path, versions, resource policy and timeouts stay unchanged.

The retained generations have distinct scopes:

- `refusal-01/`: manifest-only cache advice stopped at `ENOTDIR` during the complete first preflight. Its10 authored controls passed. No advice or native context ran.
- `refusal-02/`: additional bounded metadata identified a root-owned27-byte symlink at the first Lean row11707 (`toolchains/lean/bin/cadical`), while both held descriptors pointed to the expected runtime root. Its12 controls passed. No advice or native context ran. `complete-tar-topology.json` independently reports all26905 archive entries were regular, with no duplicate or unlisted names. This is a topology summary, not a new payload/hash audit.
- `transition/`: five observer controls passed. Actual fixed-path observations show a directory before operation7 and a symlink to `/opt/ipfs-supervisor/python` immediately after the ordinary UV install command. A second observation in the same container confirms the target. This diagnostic stops there; no provider, verifier, cache advice or Source384 context runs. All three containers were absent in retained cleanup observations.
- `fix/`: the minimal production patch and exact before/final source snapshots are retained. Final56 controls passed with zero failures/errors/skips (three new plus53 neighboring controls). One control runs an isolated, independently pinned real UV0.9.24 binary offline: the original layout migrates the authored Lean sentinel, whereas the nested layout preserves its inode/content and creates a separate real managed directory. Both intentionally refuse the missing Python download; this does not establish successful interpreter installation. No global UV installation was changed. Tool-download metadata is included; wheel and binary payloads are excluded.

The [pinned upstream initializer](https://github.com/astral-sh/uv/blob/0.9.24/crates/uv-python/src/managed.rs#L164-L177) confirms the sibling migration rule. `primary-source-review.json` records inspected source URLs, byte hashes and ranges. `uv python dir` does not initialize the directory and would not reproduce this behavior; the offline regression exercises `python install` before its expected download refusal.

This package excludes the pending fixed-layout cache/admission/inference experiment. It establishes the directory collision and local regression controls, not a Docker success, cache-reclamation benefit, benchmark score, or proof. Native admission and strict no-follow asset validation are unchanged. The cache helper remains off-tree; no cache advice is added to production by the UV patch.

Scripts and original receipts retain paths from their local execution layout and are provenance rather than a standalone recipe. `fix/qualification.json` mentions the omitted earlier successful56-control generation; only final `focused-02` is counted here. `fix/independent-review.json` points to the original primary-source artifact; its packaged counterpart is `primary-source-review.json`. First-cache controls pin the wrapper before a telemetry-label-only correction, explicitly recorded in `wrapper-final-validation.json`. The full archive manifest, private credentials, benchmark source/instruction/diff bodies, hidden verifier inputs, model payloads/reports, databases and tool binaries are excluded. Framework implementation/test snapshots are included. No prior closed evidence package is modified.
''')

copy(Path(__file__), 'package_evidence.py')
rows = [{'path': str(path.relative_to(OUT)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
        for path in sorted(OUT.rglob('*')) if path.is_file()]
write('manifest.json', {'schema': 'closed-source384-uv-layout-fix-evidence@1', 'files': rows})
scope = {'package': str(OUT), 'manifest_sha256': sha(OUT / 'manifest.json'),
         'members': len(rows), 'member_bytes': sum(row['bytes'] for row in rows),
         'files': [str(path.relative_to(OUT.parents[3])) for path in sorted(OUT.rglob('*')) if path.is_file()]}
(FIX / 'scoped-added-files.json').write_text(json.dumps(scope, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: value for key, value in scope.items() if key != 'files'}, sort_keys=True))
