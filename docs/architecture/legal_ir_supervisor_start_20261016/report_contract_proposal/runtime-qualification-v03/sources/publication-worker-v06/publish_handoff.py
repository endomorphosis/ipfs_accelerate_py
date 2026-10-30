"""Publish source-only Legal IR handoff mechanics and scoped actual evidence.

Root alone executes under the original finite charged owner. This creates one
new documentation generation; no production runtime, model, task, board,
counter, native grant or acceptance operation is changed.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

HERE = Path(__file__).absolute().parent
PHASE = HERE.parent
HANDOFF = Path('/home/barberb/lift_coding/artifacts/legal-ir-real-handoff-20261017-01')
ACCELERATOR = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
PREFIX = 'docs/architecture/legal_ir_supervisor_start_20261016/report_contract_proposal'
PREVIOUS = ACCELERATOR / PREFIX / 'runtime-qualification-v02'
TARGET = ACCELERATOR / PREFIX / 'runtime-qualification-v03'
RUNS = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs')
CAMPAIGN = RUNS / 'legal-ir-ordinary-handoff-native-20261017-r01'
CAMPAIGN_V02 = RUNS / 'legal-ir-ordinary-handoff-native-20261017-r02'
PUBLIC_CAMPAIGNS = {'r01': CAMPAIGN, 'r02': CAMPAIGN_V02}
CAPSULE = CAMPAIGN / 'preparation/capsule'
BASE_HEAD = '06eff5086b8653f0a58f5013cc781482693886c5'
BRANCH = 'agent/legal-ir-ordinary-worker-20261016'
REMOTE = 'https://github.com/endomorphosis/ipfs_accelerate_py.git'
PR_ENDPOINT = 'repos/endomorphosis/ipfs_accelerate_py/pulls/292'
HOOK_SHA = '59761cd0b0ea3395efb82ef27c6bb9e3fdbe33ce7a118afb8827bea76e2fa5b9'
CUSTODY_SHA = '36742d137ef138fd720ffc9db0521e0107bb7616ef9c95494f785b21ef238ceb'
CAPSULE_SOURCE_SHA = '13653fd01ea820e4d229b85bf387fb321754c600f99219da421ea79c8649abb1'
CUSTODY_CONTROLS_SHA = 'c1a54ab8fe1ce307e283e6c29f3293ff0b958e04332e6d78ea676bb0462e182f'
CAPSULE_MANIFEST_SHA = 'b2a78de35845425f533f7a361b3998f41628d472fafbbccb3aa346d7b50e0925'
BUNDLE_ROOTS = {
    'ordinary-success-hook-v01': HANDOFF / 'ordinary-success-hook-v01',
    'ordinary-custody-hold-v01': PHASE / 'ordinary-custody-hold-v01',
    'capsule-preparation-v01': PHASE / 'capsule-preparation-v01',
    'custody-controls-v01': PHASE / 'custody-controls-v01',
    'native-gate-v01': PHASE / 'native-gate-v01',
    'native-gate-v02': PHASE / 'native-gate-v02',
    'custody-unwind-v01': PHASE / 'custody-unwind-v01',
}
FIXED_MANIFESTS = {'ordinary-success-hook-v01': HOOK_SHA, 'ordinary-custody-hold-v01': CUSTODY_SHA,
    'capsule-preparation-v01': CAPSULE_SOURCE_SHA, 'custody-controls-v01': CUSTODY_CONTROLS_SHA,
    'native-gate-v01': '4f92449615d3df1e10d3dcd2ec6652877ee5db0f46cf94cefc87fbd9650e9634',
    'native-gate-v02': '688dfdffe9c318118d922d3adfa9c7a29f8b8bd7a5ec68e0b1884e510fb19fa7',
    'custody-unwind-v01': 'f33996f2f67101bde026200094a87df3489ac839bac420f8511b64468dcbc17b'}
REQUIRED_EVIDENCE = {'previous_root', 'previous_result', 'previous_request', 'previous_qualification', 'previous_inventory',
    'hook_root', 'hook_request', 'hook_result', 'custody_root', 'custody_request', 'custody_result',
    'capsule_root', 'capsule_request', 'capsule_result', 'capsule_manifest',
    'native_gate_root', 'native_gate_request', 'native_gate_terminal',
    'native_gate_prior_request_v01', 'native_gate_prior_request_v02',
    'custody_unwind_root', 'custody_unwind_request', 'custody_unwind_terminal'}
V02_EVIDENCE = {'native_gate_v02_root', 'native_gate_v02_request', 'native_gate_v02_terminal'}
CONFIG_FIELDS = {'schema', 'expected_accelerator_head', 'expected_branch', 'push', 'bundles', 'evidence',
    'public_artifacts', 'source_sha256'}
CASES = ('positive', 'wrong_marker', 'partial_eof')
MAXIMUM = 8 * 1024 * 1024
FALSE = dict(source_pipeline_dispatchable=False, packet_materialized=False, board_enabled=False,
    task_attempts_changed=False, provider_calls_performed=False, canonical_tasks_accepted=0,
    actual_real_provider_to_acceptance_handoff_qualified=False, actual_provider_execution_qualified=False,
    production_hook_deployed=False, acceptance_authority=False, completion_authority=False)


def need(value, reason):
    if not value:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def canonical(path):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and '..' not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), 'canonical unaliased absolute path required')
    return path

def read(path, maximum=MAXIMUM):
    path = canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 <= before.st_size <= maximum, 'bounded owned regular source required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                          s.st_mode, s.st_uid, s.st_gid, s.st_nlink)
    need(len(raw) == before.st_size and identity(before) == identity(after) == identity(path.lstat()),
         'source or engineering observation changed during read')
    return raw

def safe_relative(spelling):
    need(type(spelling) is str and re.fullmatch(r'[A-Za-z0-9_./-]+', spelling), 'bounded source relative path required')
    path = PurePosixPath(spelling)
    need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == spelling,
         'canonical source relative path required')
    return Path(spelling)

def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def write_new(path, raw, mode=0o644):
    path = canonical(path)
    need(path.parent.is_dir() and type(raw) is bytes and len(raw) <= MAXIMUM, 'existing charged publication parent required')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)

def environment():
    result = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    result.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    return result

def git(*args, raw=False):
    result = subprocess.run(['/usr/bin/git', '-c', 'core.hooksPath=/dev/null', '-c', 'credential.helper=',
        '-c', 'credential.https://github.com.helper=!/usr/bin/gh auth git-credential', *args],
        cwd=ACCELERATOR, env=environment(), stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
    need(result.returncode == 0 and len(result.stdout) <= 16 * MAXIMUM,
         'bounded owned Git operation failed: ' + ' '.join(args[:2]) + ' (exit ' + str(result.returncode) + ')')
    return result.stdout if raw else result.stdout.decode().strip()

def gh(*args):
    result = subprocess.run(['/usr/bin/gh', *args], cwd=ACCELERATOR, env=environment(),
        stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
    need(result.returncode == 0 and len(result.stdout) <= 1024 * 1024, 'bounded official GitHub API operation failed')
    return result.stdout

def read_pr():
    result = strict_json(gh('api', PR_ENDPOINT))
    need(type(result) is dict and result.get('number') == 292 and result.get('state') == 'open'
         and result.get('merged') is False and result.get('draft') is True
         and result.get('head', {}).get('ref') == BRANCH
         and result.get('head', {}).get('repo', {}).get('full_name') == 'endomorphosis/ipfs_accelerate_py'
         and result.get('base', {}).get('repo', {}).get('full_name') == 'endomorphosis/ipfs_accelerate_py'
         and type(result.get('body')) is str and len(result['body'].encode()) <= 65536,
         'exact open draft owned PR292 required')
    return result

def verify_bundle(root, digest):
    root = canonical(root)
    manifest_raw = read(root / 'SOURCE_MANIFEST.json', 1024 * 1024)
    need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) and sha(manifest_raw) == digest,
         'frozen source manifest differs')
    manifest = strict_json(manifest_raw)
    need(type(manifest) is dict and type(manifest.get('files')) is list and 1 <= len(manifest['files']) <= 64,
         'finite frozen source bundle required')
    files = {}
    for row in manifest['files']:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'closed frozen source row required')
        spelling = row['path']
        if type(spelling) is str and spelling.startswith('/'):
            absolute = canonical(spelling)
            need(root in absolute.parents, 'absolute source row must remain within its selected frozen bundle')
            relative = absolute.relative_to(root)
        else:
            relative = safe_relative(spelling)
        need(relative.name != 'SOURCE_MANIFEST.json' and relative.as_posix() not in files, 'unique frozen source row required')
        raw = read(root / relative)
        need(type(row['bytes']) is int and len(raw) == row['bytes'] and sha(raw) == row['sha256'],
             'frozen source bundle bytes differ')
        files[relative.as_posix()] = raw
    need(sum(len(raw) for raw in files.values()) <= 4 * MAXIMUM, 'bounded frozen source bundle required')
    files['SOURCE_MANIFEST.json'] = manifest_raw
    return files


def evidence_bytes(role, row):
    need(type(row) is dict and set(row) == {'absolute_path', 'bytes', 'sha256'}, 'closed exact evidence selector required')
    path = canonical(row['absolute_path'])
    if role == 'previous_qualification':
        need(path == PREVIOUS / 'QUALIFICATION.json', 'exact previous qualification context required')
    elif role == 'previous_inventory':
        need(path == PREVIOUS / 'PUBLICATION_INVENTORY.json', 'exact previous publication inventory required')
    else:
        need((PHASE in path.parents or HANDOFF in path.parents or RUNS in path.parents)
             and path.suffix in {'.json', '.jsonl'}, 'owned scoped public engineering evidence required')
    need(type(row['bytes']) is int and 0 < row['bytes'] <= 4 * MAXIMUM
         and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']),
         'bounded exact public evidence digest required')
    raw = read(path, 4 * MAXIMUM)
    need(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'original public evidence bytes changed')
    return raw


def native_terminal(raw):
    # Preserve raw JSONL separately; only the final actual nonempty record can
    # be a terminal. Earlier custody warnings cannot become terminal authority.
    lines = [line for line in raw.splitlines() if line.strip()]
    need(1 <= len(lines) <= 256, 'bounded actual native terminal trace required')
    records = [strict_json(line) for line in lines]
    need(all(type(row) is dict for row in records), 'actual native trace records required')
    return records[-1]


def load_configuration(path, expected_digest):
    raw = read(path, 1024 * 1024)
    need(type(expected_digest) is str and re.fullmatch('[0-9a-f]{64}', expected_digest)
         and sha(raw) == expected_digest, 'reviewed publication configuration changed')
    config = strict_json(raw)
    need(type(config) is dict and set(config) == CONFIG_FIELDS
         and config['schema'] == 'legal-ir-handoff-native-publication-config@2'
         and config['expected_accelerator_head'] == BASE_HEAD
         and config['expected_branch'] == BRANCH and config['push'] is True,
         'exact source-only successor publication configuration required')
    need(type(config['bundles']) is dict and set(config['bundles']) == set(BUNDLE_ROOTS),
         'exact frozen handoff, custody, preparation and qualifier source roles required')
    bundles = {}
    for role, row in config['bundles'].items():
        need(type(row) is dict and set(row) == {'absolute_root', 'manifest_sha256'}
             and canonical(row['absolute_root']) == BUNDLE_ROOTS[role], 'fixed frozen source bundle root required')
        if role in FIXED_MANIFESTS:
            need(row['manifest_sha256'] == FIXED_MANIFESTS[role], 'already frozen historical source must stay unchanged')
        bundles[role] = verify_bundle(BUNDLE_ROOTS[role], row['manifest_sha256'])
    need(type(config['evidence']) is dict and REQUIRED_EVIDENCE <= set(config['evidence'])
         and set(config['evidence']) <= REQUIRED_EVIDENCE | V02_EVIDENCE
         and (not V02_EVIDENCE.intersection(config['evidence']) or V02_EVIDENCE <= set(config['evidence'])),
         'closed actual observation, request, result and terminal role set required')
    evidence = {role: evidence_bytes(role, row) for role, row in config['evidence'].items()}
    need(type(config['public_artifacts']) is list and len(config['public_artifacts']) <= 18,
         'finite explicit nonmodel inspection and cleanup allowlist required')
    public = {}
    for row in config['public_artifacts']:
        raw_artifact = evidence_bytes('native_gate_public', row)
        path = Path(row['absolute_path'])
        matching = [(generation, campaign) for generation, campaign in PUBLIC_CAMPAIGNS.items()
                    if campaign / 'native' in path.parents]
        need(len(matching) == 1, 'public gate file must belong to one exact isolated campaign')
        generation, campaign = matching[0]
        relative = path.relative_to(campaign / 'native').as_posix()
        if generation == 'r02':
            need(V02_EVIDENCE <= set(config['evidence']), 'V02 source without selected execution evidence cannot supply runtime files')
        allowed = {case + '/primary/' + name for case in CASES for name in (
            'created-container.json', 'actual-exited-container.json', 'mechanical-case.json')}
        output = generation + '/' + relative
        need(relative in allowed and output not in public,
             'private native capability, raw ledger or unlisted runtime file cannot be published')
        public[output] = raw_artifact
    pins = config['source_sha256']
    native = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets/ipfs_datasets_py/optimizers/logic_theorem_optimizer')
    required = {str(Path(__file__).absolute()), str(HERE / 'SOURCE_MANIFEST.json'),
        '/usr/bin/git', '/usr/bin/gh', '/usr/bin/python3.12',
        str(ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py'),
        str(ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'),
        str(ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/runtime/ordinary_provider_cleanup.py'),
        '/home/barberb/lift_coding/external/ipfs_accelerate/scripts/run_legal_autoformalization_supervisor.py',
        '/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/run_image_owner_v05.py',
        str(native.parent.parent / '__init__.py'), str(native.parent / '__init__.py'), str(native / '__init__.py'),
        str(native.parent.parent / 'router_deps.py'),
        *(str(native / (name + '.py')) for name in ('parented_phase_resources', 'autoencoder_daemon_resources',
            'existing_resource_scheduler', 'resource_scheduler', 'autoencoder_resource_recovery',
            'proof_resource_safety', 'runtime_telemetry'))}
    need(type(pins) is dict and required <= set(pins) and len(pins) <= 256,
         'protected actual publisher, owner, runner, daemon and cleanup source closure required')
    for spelling, digest in pins.items():
        need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest)
             and sha(read(canonical(spelling), 64 * 1024 * 1024)) == digest, 'protected active source changed')
    return config, raw, bundles, evidence, public


def closed_owner_terminal(terminal, attempt):
    need(type(terminal) is dict and terminal.get('status') == 'completed'
         and terminal.get('native_finalized') is True and terminal.get('child_returncode') == 0
         and terminal.get('failure_type') is None and canonical(terminal.get('receipt')) == attempt / 'receipt.json',
         'actual original generic native owner closeout required')


def assess_controls(config, evidence, bundles, prefix, count, source_role, result_schema, filename):
    root, request, result = (strict_json(evidence[prefix + '_' + name]) for name in ('root', 'request', 'result'))
    attempt = canonical(request.get('attempt_directory'))
    need(RUNS in attempt.parents
         and canonical(config['evidence'][prefix + '_result']['absolute_path']) == attempt / filename,
         'actual control request and result must share the original private attempt')
    digest = config['bundles'][source_role]['manifest_sha256']
    if prefix == 'hook':
        need(root.get('schema') == 'legal-ir-ordinary-hook-controls-root-observation@1'
             and root.get('actual_exec_exit_code') == 0
             and root.get('qualified_source_manifest_sha256') == digest
             and root.get('source_request_sha256') == sha(evidence['hook_request'])
             and root.get('actual_frozen_owner_request') == request, 'actual 104-case hook root correspondence required')
        terminal = root.get('actual_native_terminal')
    else:
        need(root.get('schema') == 'legal-ir-root-native-control-observation@1'
             and root.get('source_scope') == source_role and root.get('actual_terminal_exit_code') == 0
             and root.get('actual_terminal_native_finalized') is True
             and root.get('request_sha256') == sha(evidence['custody_request'])
             and root.get('controlled_cases') == count
             and canonical(root.get('result_path')) == attempt / filename,
             'actual 35-case custody root correspondence required')
        terminal = strict_json(root.get('terminal_output'))
    closed_owner_terminal(terminal, attempt)
    need(root.get('result_sha256') == sha(evidence[prefix + '_result'])
         and request.get('schema') == 'legal-ir-native-orchestration-request@1'
         and request.get('purpose') == 'controller_preparation'
         and result.get('schema') == result_schema and result.get('passed') is True
         and result.get('returncode') == 0 and result.get('controlled_testcases') == result.get('passed_testcases') == count
         and all(result.get(key) == 0 for key in ('failures', 'errors', 'skipped', 'canonical_tasks_accepted'))
         and result.get('source_manifest_sha256') == digest
         and result.get('provider_calls_performed') is False
         and result.get('actual_provider_execution_qualified') is False
         and result.get('production_hook_deployed') is False,
         'exact genuinely observed passed scoped source controls required')
    manifest = strict_json(bundles[source_role]['SOURCE_MANIFEST.json'])
    need(manifest.get('controlled_testcases') == count, 'case count must join the frozen product source')
    pins = request.get('source_sha256')
    need(type(pins) is dict and pins.get(str(BUNDLE_ROOTS[source_role] / 'SOURCE_MANIFEST.json')) == digest,
         'actual native owner request must pin the source manifest')
    for relative, raw in bundles[source_role].items():
        need(pins.get(str(BUNDLE_ROOTS[source_role] / relative)) == sha(raw), 'actual control request source closure differs')
    for row in manifest.get('input_pins', []):
        need(pins.get(row['path']) == row['sha256'], 'actual control request must join all frozen input pins')
    if prefix == 'custody':
        need(result.get('actual_custody_binding_qualified') is False
             and result.get('external_test_aliases_created') is False
             and request.get('external_charges') == [], 'memory controls cannot qualify actual custody or external effects')
    return count


def assess_previous(evidence):
    root, result, request, qualification, inventory = (strict_json(evidence['previous_' + name])
        for name in ('root', 'result', 'request', 'qualification', 'inventory'))
    need(root.get('schema') == 'legal-ir-real-handoff-publication-root-observation@1'
         and root.get('actual_exec_exit_code') == 0 and root.get('actual_publication') == result
         and root.get('publication_result_sha256') == sha(evidence['previous_result'])
         and root.get('source_request_sha256') == sha(evidence['previous_request'])
         and result.get('accelerator_commit') == BASE_HEAD and result.get('pushed') is True
         and result.get('production_hook_deployed') is False and result.get('canonical_tasks_accepted') == 0,
         'exact previously published documentation commit and original root observation required')
    closed_owner_terminal(root.get('actual_native_terminal'), canonical(request.get('attempt_directory')))
    need(qualification.get('schema') == 'legal-ir-real-handoff-runtime-qualification@2'
         and qualification.get('actual_full_native_v05_fixture_passed') is True
         and qualification.get('actual_public_candidate_observer_fixture_passed') is True
         and qualification.get('controlled_reader_testcases') == 60
         and qualification.get('ordinary_success_hook_controlled_testcases') == 104
         and qualification.get('historical_v02_public_git_r01_failed') is True
         and qualification.get('historical_v02_public_observer_qualified') is False
         and all(qualification.get(key) == value for key, value in FALSE.items() if key in qualification),
         'published V02 context must retain scoped qualifications and historical failures')
    need(inventory.get('schema') == 'legal-ir-real-handoff-publication-inventory@1'
         and inventory.get('configuration_sha256') == qualification.get('publication_configuration_sha256')
             == result.get('configuration_sha256')
         and type(inventory.get('files')) is list and 1 <= len(inventory['files']) <= 160,
         'exact bounded prior immutable publication inventory required')
    seen = set()
    for row in inventory['files']:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'closed previous publication row required')
        relative = safe_relative(row['path'])
        path = ACCELERATOR / relative
        need(PREVIOUS in path.parents and row['path'] not in seen, 'prior inventory must stay in its own generation')
        raw = read(path)
        need(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'historical publication bytes changed')
        seen.add(row['path'])
    return qualification


def assess_capsule(config, evidence, bundles):
    root, request, result, manifest = (strict_json(evidence['capsule_' + name])
        for name in ('root', 'request', 'result', 'manifest'))
    attempt = CAMPAIGN / 'preparation'
    need(root.get('schema') == 'legal-ir-handoff-capsule-root-observation@1'
         and root.get('actual_exec_exit_code') == 0 and root.get('original_native_preparation_finalized') is True
         and root.get('source_manifest_sha256') == CAPSULE_SOURCE_SHA
         and root.get('source_request_sha256') == sha(evidence['capsule_request'])
         and root.get('preparation_result_sha256') == sha(evidence['capsule_result'])
         and root.get('capsule_manifest_sha256') == sha(evidence['capsule_manifest']) == CAPSULE_MANIFEST_SHA
         and root.get('two_owner_native_gate_qualified') is False,
         'actual separately finalized capsule preparation provenance required')
    closed_owner_terminal(root.get('actual_native_terminal'), attempt)
    need(request.get('attempt_directory') == str(attempt)
         and canonical(config['evidence']['capsule_result']['absolute_path']) == attempt / 'capsule-preparation.json'
         and canonical(config['evidence']['capsule_manifest']['absolute_path']) == CAPSULE / 'SOURCE_MANIFEST.json'
         and request.get('storage_bytes') == 512 * 1024 * 1024 and request.get('external_charges') == []
         and request.get('source_sha256', {}).get(str(PHASE / 'capsule-preparation-v01/SOURCE_MANIFEST.json')) == CAPSULE_SOURCE_SHA,
         'actual bounded private preparation owner request must match the original capsule')
    inventory = bundles['capsule-preparation-v01']['SOURCE_INVENTORY.json']
    source_inventory = strict_json(inventory)
    need(root.get('source_inventory_sha256') == manifest.get('source_inventory_sha256') == sha(inventory)
         and manifest.get('schema') == 'legal-ir-native-handoff-capsule@1'
         and manifest.get('capsule_root') == str(CAPSULE) and manifest.get('frozen_hook_manifest_sha256') == HOOK_SHA
         and manifest.get('payload_file_count') == source_inventory.get('payload_file_count') == 2806
         and manifest.get('payload_bytes') == source_inventory.get('payload_bytes') == 106710527
         and result.get('capsule_manifest_sha256') == CAPSULE_MANIFEST_SHA
         and result.get('original_native_preparation_closeout_pending') is True,
         'producer pending-closeout metadata and genuine later finalization must remain distinct')
    need(all(manifest.get(key) is False for key in ('provider_calls_performed', 'production_hook_deployed',
        'acceptance_authority', 'completion_authority')) and manifest.get('canonical_tasks_accepted') == 0,
        'source capsule has no provider or acceptance qualification')
    rows = manifest.get('files')
    need(type(rows) is list and len(rows) == 2806, 'exact complete metadata-only capsule inventory required')
    expected = {row['path']: {'path': row['path'], 'bytes': row['bytes'], 'sha256': row['sha256']}
                for row in source_inventory['files']}
    need(len(expected) == len(rows) and all(type(row) is dict and expected.get(row.get('path')) == row for row in rows)
         and {row['path'] for row in rows} == set(expected),
         'actual capsule rows must join exact source-selected Python files including empty files')
    return {'capsule_payload_file_count': 2806, 'capsule_payload_bytes': 106710527,
            'capsule_manifest_sha256': CAPSULE_MANIFEST_SHA, 'actual_capsule_preparation_finalized': True}


def assess_native_qualifier(config, evidence, bundles, prefix, source_role, terminal_schema, worker_name,
                            *, terminal_request_field='request_sha256', request_manifest_field='qualifier_manifest_sha256'):
    root, request = (strict_json(evidence[prefix + '_' + name]) for name in ('root', 'request'))
    terminal = native_terminal(evidence[prefix + '_terminal'])
    manifest_sha = config['bundles'][source_role]['manifest_sha256']
    need(root.get('schema') == 'legal-ir-native-mechanical-root-observation@1'
         and root.get('source_scope') == source_role
         and type(root.get('actual_exec_session')) is int and root['actual_exec_session'] > 0
         and type(root.get('actual_exec_exit_code')) is int and root['actual_exec_exit_code'] in {0, 1}
         and root.get('source_manifest_sha256') == manifest_sha
         and root.get('source_request_sha256') == sha(evidence[prefix + '_request'])
         and root.get('terminal_sha256') == sha(evidence[prefix + '_terminal'])
         and root.get('actual_native_terminal') == terminal,
         'actual source-bound native qualifier root and terminal correspondence required')
    need(terminal.get('schema') == terminal_schema and type(terminal.get('passed')) is bool
         and root['actual_exec_exit_code'] == (0 if terminal['passed'] else 1)
         and terminal.get(terminal_request_field) == sha(evidence[prefix + '_request'])
         and terminal.get('canonical_tasks_accepted') == 0,
         'actual native qualifier outcome must be archived truthfully without forced pass')
    pins = request.get('source_sha256')
    need(type(pins) is dict
         and pins.get(str(BUNDLE_ROOTS[source_role] / worker_name)) == sha(bundles[source_role][worker_name])
         and request.get(request_manifest_field) == manifest_sha,
         'actual direct native request must pin the exact frozen executed qualifier')
    for relative, raw in bundles[source_role].items():
        need(pins.get(str(BUNDLE_ROOTS[source_role] / relative)) == sha(raw),
             'actual direct native request must join every frozen qualifier source row')
    manifest = strict_json(bundles[source_role]['SOURCE_MANIFEST.json'])
    for row in manifest.get('input_pins', []):
        need(pins.get(row['path']) == row['sha256'], 'actual native request must join all frozen input pins')
    return root, request, terminal


def inspect_public_profile(public, bundles):
    hook_source = bundles['ordinary-success-hook-v01'][
        'proposed/ipfs_accelerate_py/agent_supervisor/runtime/ordinary_success_handoff.py']
    tree = ast.parse(hook_source)
    gates = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == '_START_GATE' for target in node.targets)]
    need(len(gates) == 1 and type(gates[0]) is str and len(gates[0]) <= 4096,
         'exact source-only nonmodel gate literal required')
    command = ['-I', '-c', gates[0], '/opt/ipfs-task-tools/bin/python',
               '-I', '-S', '-B', '/opt/legal-ir/fixture_payload.py', 'positive']
    environment = ['PATH=/usr/bin:/bin', 'HOME=/workspace', 'TMPDIR=/workspace', 'LANG=C.UTF-8',
                   'LC_ALL=C.UTF-8', 'PYTHONDONTWRITEBYTECODE=1']
    expected_environment = dict(value.split('=', 1) for value in environment)
    private_fields = {'lease_key', 'api_key', 'auth_token', 'authorization', 'password', 'secret',
                      'private_key', 'native_handle', 'capability_key', 'raw_ledger', 'credentials'}
    for relative, raw in public.items():
        value = strict_json(raw)
        pending, visited = [(value, 0)], 0
        while pending:
            child, depth = pending.pop()
            visited += 1
            need(visited <= 32768 and depth <= 32, 'bounded public runtime metadata required')
            if type(child) is dict:
                need(not private_fields.intersection(key.lower() for key in child), 'private authority or credential field is not publishable')
                pending.extend((item, depth + 1) for item in child.values())
            elif type(child) is list:
                pending.extend((item, depth + 1) for item in child)
        generation, relative = relative.split('/', 1)
        source_role = 'native-gate-v01' if generation == 'r01' else 'native-gate-v02'
        campaign = PUBLIC_CAMPAIGNS[generation]
        if relative.endswith('/mechanical-case.json'):
            continue
        case = relative.split('/')[0]
        config, host = value.get('Config', {}), value.get('HostConfig', {})
        labels = config.get('Labels', {})
        worker = ast.parse(bundles[source_role]['qualify_native_gate.py'])
        profiles = [ast.literal_eval(node.value) for node in worker.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'PROFILE' for target in node.targets)]
        need(len(profiles) == 1 and type(profiles[0]) is str and len(profiles[0]) <= 128,
             'exact frozen nonmodel generation profile required')
        actual_environment = config.get('Env')
        need(type(actual_environment) is list and len(actual_environment) == len(expected_environment),
             'exact finite environment key population required')
        joined = {}
        for spelling in actual_environment:
            need(type(spelling) is str and '=' in spelling, 'literal public environment assignment required')
            key, content = spelling.split('=', 1)
            need(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key) and key not in joined,
                 'duplicate or malformed public environment key')
            joined[key] = content
        need(value.get('Image') == 'sha256:a91a42c918514203bb6cfc7c38b320dfb68e9dfc0571bf01976c0db993e24e5e'
             and joined == expected_environment and config.get('Cmd') == command
             and config.get('Entrypoint') == ['/opt/ipfs-task-tools/bin/python']
             and config.get('User') == str(os.getuid()) + ':' + str(os.getgid())
             and config.get('WorkingDir') == '/workspace'
             and labels.get('org.justicedao.nonmodel-fixture') == profiles[0]
             and labels.get('org.justicedao.fixture-case') == case
             and labels.get('org.ipfs-accelerate.ordinary-worker') == '2026-10-16-ordinary-v1'
             and labels.get('ipfs_accelerate.grok_isolation') == 'true'
             and host.get('NetworkMode') == 'none' and host.get('ReadonlyRootfs') is True
             and host.get('Privileged') is False and host.get('CapDrop') == ['ALL'],
             'public inspection must contain the exact credential-free nonmodel command and profile')
        expected = sorted([('/usr', '/usr', False), ('/usr/bin/python3.12', '/opt/ipfs-task-tools/bin/python', False),
            (str(BUNDLE_ROOTS[source_role] / 'fixture_payload.py'), '/opt/legal-ir/fixture_payload.py', False),
            (str(campaign / ('native/temporary-' + case + '/workspace')), '/workspace', True)])
        mounts = value.get('Mounts')
        need(type(mounts) is list and len(mounts) == 4
             and sorted((row.get('Source'), row.get('Destination'), row.get('RW')) for row in mounts) == expected,
             'public nonmodel inspection cannot include auth, provider-home, socket or arbitrary mounts')


def assess_gate(config, evidence, bundles, public, *, prefix, source_role, generation):
    campaign = PUBLIC_CAMPAIGNS[generation]
    root, request, terminal = assess_native_qualifier(config, evidence, bundles, prefix, source_role,
        'legal-ir-ordinary-handoff-native-mechanics@1', 'qualify_native_gate.py')
    inspect_public_profile(public, bundles)
    need(request.get('schema') == 'legal-ir-native-handoff-mechanical-request@1'
         and request.get('capsule_root') == str(CAPSULE)
         and request.get('capsule_manifest_sha256') == terminal.get('capsule_manifest_sha256') == CAPSULE_MANIFEST_SHA
         and request.get('frozen_hook_manifest_sha256') == terminal.get('frozen_hook_manifest_sha256') == HOOK_SHA
         and request.get('cases') == list(CASES)
         and terminal.get('fixture_only') is True
         and terminal.get('scope') == 'synthetic_nonmodel_engineering_fixture_only'
         and terminal.get('root_and_phase_contexts_finalized_after_cases') is True
         and terminal.get('all_case_owner_contexts_exited_after_independent_cleanup') is True
         and terminal.get('models_called') == 0
         and all(type(terminal.get(key)) is dict and terminal[key].get('released') is True
                 for key in ('actual_root_closeout', 'actual_phase_closeout'))
         and all(terminal.get(key) is False for key in ('provider_calls_performed', 'public_wrapper_exercised',
             'finish_or_success_callback_called', 'actual_provider_execution_qualified', 'production_hook_deployed',
             'acceptance_authority', 'completion_authority')),
         'nonmodel actual gate mechanics must remain separate from provider success or acceptance')
    if generation == 'r01':
        assess_gate_v01_request_history(evidence, request)
    else:
        controls = terminal.get('environment_controls', {})
        need(terminal.get('qualifier_source_revision') == 'native-gate-v02'
             and terminal.get('environment_check') == 'exact_unique_key_value_mapping_ignoring_order_only'
             and terminal.get('reused_readonly_capsule_root') == str(CAPSULE)
             and controls.get('schema') == 'legal-ir-exact-fixture-environment-controls@1'
             and controls.get('controlled_testcases') == controls.get('passed_testcases') == 15
             and controls.get('passed') is True and controls.get('pure_controls_only') is True
             and controls.get('models_called') == 0 and controls.get('acceptance_authority') is False,
             'actual V02 environment controls must preserve exact authority while ignoring ordering only')
    cases = terminal.get('cases')
    need(type(cases) is list and [row.get('fixture_case') for row in cases] == list(CASES)
         and all(type(row.get('passed')) is bool for row in cases)
         and terminal['passed'] == all(row['passed'] for row in cases), 'actual finite case outcomes must match terminal summary')
    outcomes = []
    for summary in cases:
        case = summary['fixture_case']
        relative = case + '/primary/mechanical-case.json'
        public_path = generation + '/' + relative
        need(public_path in public, 'exact full per-case stopped-closeout observation must be selected')
        raw = public[public_path]
        observed = strict_json(raw)
        retained = summary.get('private_case_file', {})
        finalization = summary.get('actual_owner_finalization', {})
        need(retained.get('path') == str(campaign / 'native' / relative)
             and retained.get('bytes') == len(raw) and retained.get('sha256') == sha(raw)
             and retained.get('written_before_owner_finalization') is True
             and summary.get('both_native_owner_contexts_exited_after_actual_cleanup') is True
             and observed.get('fixture_case') == case and observed.get('passed') == summary['passed'],
             'actual compact terminal must bind the retained full case and genuine owner closeout')
        need(set(finalization) == {'primary', 'watchdog'} and all(
            type(value) is dict and value.get('status') == 'released'
            and value.get('native_lease_released') is True for value in finalization.values()),
            'actual owner finalization must follow stopped proof and the prior private case write')
        cleanup = observed.get('cleanup', {})
        need(cleanup.get('cleanup_known') is True and observed.get('canonical_tasks_accepted') == 0
             and observed.get('provider_calls_performed') is False,
             'even a failed case must establish actual known stopped cleanup before publication')
        before = observed.get('actual_pre_effect_capacity_hold', {})
        after = observed.get('actual_post_cleanup_capacity_hold', {})
        acknowledged = observed.get('actual_capacity_acknowledged_after_cleanup', {})
        need(before.get('capacity_records_retained') is True and after.get('capacity_records_retained') is True
             and before.get('consumer_objects_released') == after.get('consumer_objects_released') == [False, False]
             and acknowledged.get('hold_references_forgotten') is True
             and acknowledged.get('native_consumers_acknowledged_and_absent') is True
             and acknowledged.get('native_leases_released_by_this_helper') is False,
             'actual consumer barrier must remain charged until stopped proof and explicit public acknowledgement')
        if cleanup.get('actual_allocation_observed') is True:
            need(all(cleanup.get(key) is True for key in ('container_absent_by_id_and_name', 'kernel_scope_quiescent',
                'actual_watchdog_group_dead', 'actual_transport_group_dead', 'lease_home_prompt_absent')),
                'actual allocated fixture must independently close container, kernel, groups and private paths')
        if summary['passed']:
            need(observed.get('actual_returncode') == (0 if case == 'positive' else 125)
                 and cleanup.get('actual_kernel_scope_captured') is True,
                 'passed gate control requires the actual captured kernel scope and corresponding exit')
        outcomes.append({'fixture_case': case, 'passed': summary['passed'],
            'failure_type': observed.get('failure_type'), 'failure_reason': observed.get('failure_reason'),
            'actual_cleanup_known': True, 'full_case_sha256': sha(raw)})
    return {'passed': terminal['passed'], 'cases': outcomes, 'exec_session': root['actual_exec_session'],
            'actual_cleanup_finalized': True}


def assess_gate_v01_request_history(evidence, request):
    original = strict_json(evidence['native_gate_prior_request_v01'])
    middle = strict_json(evidence['native_gate_prior_request_v02'])
    need(sha(evidence['native_gate_prior_request_v01']) == '57660384b1ac2840cb8702fed8b2aa6b3077e1416e5022b482ddd3930054ae45'
         and sha(evidence['native_gate_prior_request_v02']) == '81d44cf61b4b57a0b1c5e19c0c00a8ddfd55e79c6fdf55cd622fd4c9176bc76b'
         and sha(evidence['native_gate_request']) == '34a7f2c1f3926f2af724c131e0ff2cec93207fec00f429c27836acb56e740703',
         'unexecuted earlier request revisions must remain exact historical source metadata')
    need(sha(evidence['native_gate_root']) == '704a3822774bca4acc937b92d44c50c93f010621a6d74af636723e8cb1e19a35'
         and sha(evidence['native_gate_terminal']) == '6660be7b18f7b550775d0ba06050e8717de104ba35406bb95b806806b1eab2f6',
         'actual original V01 mixed pass/failure terminal must remain byte-preserved')
    lineage = request.get('request_revision_lineage', {})
    need(lineage.get('original_r01_request_sha256') == sha(evidence['native_gate_prior_request_v01'])
         and lineage.get('prior_v02_request_sha256') == sha(evidence['native_gate_prior_request_v02'])
         and lineage.get('prior_requests_preserved') is True
         and lineage.get('native_execution_attempted_by_source_author') is False,
         'source-only request revision lineage cannot imply extra runtime attempts')
    descriptive = {'source_sha256', 'request_revision_lineage', 'admission_transitive_source_closure'}
    for prior in (original, middle):
        need(type(prior.get('source_sha256')) is dict
             and all(request['source_sha256'].get(path) == digest for path, digest in prior['source_sha256'].items())
             and {key: value for key, value in prior.items() if key not in descriptive} == {
                 key: value for key, value in request.items() if key not in descriptive},
             'actual reviewed request may add exact dependency source guards without changing the controlled job')


def assess_unwind(config, evidence, bundles):
    need(sha(evidence['custody_unwind_root']) == '9c521254a4670328c64e8bceaca55d24cdfb5c587f2a822c3fce732f805f3ba5'
         and sha(evidence['custody_unwind_terminal']) == '0517314f9e3dad67493cdca332e46547c38c694ae0370ae7ceb52f4dc6d6d188',
         'actual original unwind observation and terminal must remain byte-preserved')
    root, request, terminal = assess_native_qualifier(config, evidence, bundles, 'custody_unwind', 'custody-unwind-v01',
        'legal-ir-ordinary-custody-unwind-terminal@1', 'qualify_custody_unwind.py',
        terminal_request_field='source_request_sha256', request_manifest_field='source_manifest_sha256')
    need(request.get('schema') == 'legal-ir-ordinary-custody-unwind-request@1'
         and terminal.get('source_manifest_sha256') == config['bundles']['custody-unwind-v01']['manifest_sha256']
         and terminal.get('fixture_only') is True
         and terminal.get('scope') == 'genuine_native_no_external_process_unwind_fixture_only'
         and terminal.get('provider_calls_performed') is False
         and terminal.get('models_called') == 0 and terminal.get('Docker_started') is False
         and terminal.get('OS_process_children_created') is False
         and terminal.get('productive_authority_after_cancellation') is False
         and terminal.get('actual_consumer_rows_drained') is True
         and terminal.get('root_and_phase_rows_drained') is True
         and all(terminal.get(key) is False for key in ('acceptance_authority', 'completion_authority',
             'dispatch_authority', 'actual_provider_execution_qualified', 'production_hook_deployed')),
         'actual no-Docker unwind qualifier must close all consumers and roots without restoring authority')
    statuses = terminal.get('actual_disk_owner_statuses')
    need(type(statuses) is dict and len(statuses) <= 2 and all(
        type(value) is dict and value.get('status') == 'released'
        and value.get('native_lease_released') in (True, None) for value in statuses.values()),
        'actual unwind disk owners must have stopped finalization independent of experimental outcome')
    if terminal['passed']:
        need(all(terminal.get(key) is True for key in ('actual_owner_contexts_unwound',
            'actual_phase_release_requested', 'capacity_retained_after_unwind', 'capacity_retained_after_cancellation')),
            'passed unwind qualifier requires genuine retention and cancelled ancestry observations')
        need(len(statuses) == 2, 'passed unwind experiment requires both genuine owner closeouts')
    return {'actual_custody_unwind_qualifier_passed': terminal['passed'],
        'custody_unwind_exec_session': root['actual_exec_session'],
        'actual_custody_unwind_rows_drained': True, 'productive_authority_after_cancellation': False}


def assess_qualification(config, evidence, bundles, public):
    previous = assess_previous(evidence)
    hook_count = assess_controls(config, evidence, bundles, 'hook', 104, 'ordinary-success-hook-v01',
        'legal-ir-ordinary-success-hook-controls@1', 'ordinary-hook-controls.json')
    custody_count = assess_controls(config, evidence, bundles, 'custody', 35, 'ordinary-custody-hold-v01',
        'legal-ir-ordinary-custody-memory-controls@1', 'ordinary-custody-controls.json')
    qualification = {**FALSE, 'historical_failures_preserved': True,
        'previous_publication_commit': BASE_HEAD,
        'historical_native_v05_fixture_passed': previous['actual_full_native_v05_fixture_passed'],
        'historical_candidate_observer_v03_fixture_passed': previous['actual_public_candidate_observer_fixture_passed'],
        'ordinary_success_hook_controlled_testcases': hook_count, 'custody_memory_controlled_testcases': custody_count,
        'scope': 'Source proposals, pure controls, actual nonmodel mechanics and no-Docker native retention only.'}
    qualification.update(assess_capsule(config, evidence, bundles))
    original_gate = assess_gate(config, evidence, bundles, public, prefix='native_gate',
        source_role='native-gate-v01', generation='r01')
    qualification.update(native_gate_v01_passed=original_gate['passed'],
        native_gate_v01_cases=original_gate['cases'], native_gate_v01_exec_session=original_gate['exec_session'],
        native_gate_v01_cleanup_finalized=original_gate['actual_cleanup_finalized'])
    if V02_EVIDENCE <= set(evidence):
        current_gate = assess_gate(config, evidence, bundles, public, prefix='native_gate_v02',
            source_role='native-gate-v02', generation='r02')
        qualification.update(native_gate_v02_execution_observed=True,
            native_gate_v02_environment_controlled_testcases=15,
            actual_nonmodel_gate_qualifier_passed=current_gate['passed'], native_gate_cases=current_gate['cases'],
            native_gate_exec_session=current_gate['exec_session'],
            actual_nonmodel_gate_cleanup_finalized=current_gate['actual_cleanup_finalized'])
    else:
        qualification.update(native_gate_v02_execution_observed=False,
            native_gate_v02_environment_controlled_testcases=None,
            actual_nonmodel_gate_qualifier_passed=None, native_gate_cases=None,
            native_gate_exec_session=None, actual_nonmodel_gate_cleanup_finalized=None)
    qualification.update(assess_unwind(config, evidence, bundles))
    return qualification


def readiness(qualification):
    return {'schema': 'legal-ir-handoff-readiness@3', **FALSE,
        'observed_engineering_status': {
            'source_hook_controls': qualification['ordinary_success_hook_controlled_testcases'],
            'custody_memory_controls': qualification['custody_memory_controlled_testcases'],
            'historical_nonmodel_gate_v01_passed': qualification['native_gate_v01_passed'],
            'nonmodel_gate_v02_execution_observed': qualification['native_gate_v02_execution_observed'],
            'nonmodel_gate_v02_pure_environment_controls': qualification['native_gate_v02_environment_controlled_testcases'],
            'nonmodel_gate_passed': qualification['actual_nonmodel_gate_qualifier_passed'],
            'no_Docker_custody_unwind_passed': qualification['actual_custody_unwind_qualifier_passed']},
        'inert_tasks_not_dispatched': [
            {'id': 'LIR-HOST-001', 'goal': 'Complete trusted original-provider handoff',
             'tasks': ['Integrate and qualify the opt-in original primary-provider success callback under two genuine owners.',
                       'Retain actual native capacity and actual object references through uncertain stopped cleanup.',
                       'Implement and independently qualify the requested Codex backup route; V01 supports the primary route only.']},
            {'id': 'LIR-CANDIDATE-001', 'goal': 'Bind a real report-only candidate to its original successful provider attempt',
             'tasks': ['Reuse daemon candidate commit serialization and declared bounded JSON staging.',
                       'Preserve the two historical consumed attempts and require the separately reviewed total cap 3.',
                       'Join actual provider execution, original attempt, exact candidate Git blobs and frozen original task population.']},
            {'id': 'LIR-ACCEPT-001', 'goal': 'Complete real-context validation and original-board acceptance',
             'tasks': ['Run the strict Docker/native real-context checker on the actual frozen candidate.',
                       'Use the separate trusted host one-use acceptance API and original controller CAS.',
                       'Keep all 63 later tasks blocked until the first original task is genuinely accepted.']},
            {'id': 'LIR-EVAL-001', 'goal': 'Measure legal autoformalization quality',
             'tasks': ['Evaluate held-out real US Code source fidelity and span/entity/definition graph consistency.',
                       'Evaluate the actual 8D, 384D and 768D decoder checkpoints by logic family and semantic validity.',
                       'Build the unified Lean 4 corpus with lake build legal and separate syntax success from legal fidelity.']}],
        'cli_contract': '`--check-report` remains non-accepting at exit 3; native acceptance uses the separate trusted host API.',
        'failure_policy': 'Preserve genuinely observed failed controls and original historical failures without retrospective qualification.'}


def readme(qualification):
    original = 'passed' if qualification['native_gate_v01_passed'] else 'failed'
    gate = outcome_word(qualification['actual_nonmodel_gate_qualifier_passed'])
    unwind = 'passed' if qualification['actual_custody_unwind_qualifier_passed'] else 'failed'
    return ('''# Legal IR handoff qualification: NON-DISPATCHABLE

The ordinary primary success-hook source proposal passed 104 source, marker-I/O and actual-builder assembly controls. The custody helper passed 35 pure correspondence and early-refusal controls. A separately admitted preparation owner created and finalized the exact private Python capsule: 2,806 files and 106,710,527 bytes, with the frozen hook sources substituted. These source and preparation results establish no provider outcome.

The original V01 three-case non-model Docker gate qualifier ''' + original + '''. Every original case Boolean, failure reason and cleanup fact is preserved. The V02 successor ''' + gate + '''. Its strict environment check compares unique key/value assignments, rejects duplicates and extras, and preserves the original raw environment order in the archived inspection. The separate genuine no-Docker native custody-unwind qualifier ''' + unwind + '''. Exact actual requests, frozen sources, root observations and final terminals are archived for each genuinely executed qualifier. A source-only successor without an observed execution has a null outcome. A failed qualifier remains failed. Stopped cleanup and owner closeout are recorded separately from test outcomes. No model, auth route, real provider success callback or acceptance was exercised by these controls.

The previously published V02 generation remains byte-preserved. Its synthetic V05 Docker/native acceptance fixture and V03 public candidate-observer fixture remain historical scoped qualifications. Earlier failed generations retain their original values; the new observations do not qualify older source generations retrospectively.

The production pipeline remains NON-DISPATCHABLE. Remaining tasks are the original provider callback integration, a real report-only candidate checkpoint, real provider/candidate/attempt binding, the real-context native checker and original-board controller CAS. V01 supports the primary Grok route only; the requested Codex backup still needs its own implementation and qualification. Productive cancellation and retained charged capacity are different facts; a retained child record cannot restore cancelled ancestry or grant a new job.

No production source, native policy, original task, board or attempt counter changes here. The two historical consumed attempts remain preserved, total cap 3 requires its separately reviewed packet, and all 63 later tasks remain blocked. `--check-report` remains non-accepting at exit 3; native acceptance uses the separate trusted host API. The readiness task file is inert proposal metadata for later work.

Canonical accepted legal tasks remain 0/64. These engineering controls establish no decoder accuracy, statutory fidelity, span/entity graph accuracy, logic-family semantics or Lean build result. Those evaluations require the later original corpus tasks. The archived metadata contains no native capability key, raw scheduler ledger, credential or portable acceptance token.
''').encode()


def pr_body(qualification):
    original = 'passed' if qualification['native_gate_v01_passed'] else 'failed'
    gate = outcome_word(qualification['actual_nonmodel_gate_qualifier_passed'])
    unwind = 'passed' if qualification['actual_custody_unwind_qualifier_passed'] else 'failed'
    return ('''Fix Legal IR worker cleanup and preserve finite retry history. Archive source proposals and the exact scoped engineering observations in immutable documentation generations.

The primary success-hook proposal passed 104 source/gate/builder controls, and the custody helper passed 35 pure controls. A separate genuine preparation owner finalized a complete private source capsule. The original non-model Docker gate V01 ''' + original + ''' with all original case outcomes preserved. The V02 successor ''' + gate + '''; the genuine no-Docker custody-unwind qualifier ''' + unwind + '''. Their actual source/request/result joins and independent cleanup facts are archived without forcing a positive qualification or inferring execution from a source proposal.

The earlier V05 synthetic native acceptance fixture and V03 public candidate-observer fixture remain qualified only in their documented scopes. Historical failures remain unchanged. The real provider-to-acceptance pipeline remains non-dispatchable: primary callback integration, real candidate checkpoint, real-context validation, original-board acceptance and the requested Codex backup are pending. No production hook is deployed, no original attempt or board changes, and canonical accepted legal tasks remain 0/64. No model-accuracy, legal-fidelity or Lean-build claim is made.

This remains a draft PR stacked on the existing Docker bootstrap branch. Previously observed hosted CI could not start because of the GitHub account billing lock; the archived local qualifications are separate. Do not merge this documentation as an enabled production pipeline.
''')


def outcome_word(value):
    need(value is True or value is False or value is None, 'genuine observed Boolean or explicitly unobserved null outcome required')
    return 'passed' if value is True else ('failed' if value is False else 'has no selected execution evidence')


def main():
    need(len(sys.argv) == 3, 'usage: publish_handoff.py ABSOLUTE_FROZEN_CONFIG EXPECTED_CONFIG_SHA256')
    # Capture the genuine owner-provided private attempt before any library can
    # redirect TMPDIR. The worker itself neither redirects it nor claims a grant.
    attempt = canonical(os.environ['TMPDIR'])
    info = attempt.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
         'genuine charged private publisher attempt required')
    config_path = canonical(sys.argv[1])
    config, config_raw, bundles, evidence, docker = load_configuration(config_path, sys.argv[2])
    qualification = assess_qualification(config, evidence, bundles, docker)
    def guard():
        checked, raw, checked_bundles, checked_evidence, checked_docker = load_configuration(config_path, sys.argv[2])
        need(raw == config_raw and checked == config and checked_bundles == bundles
             and checked_evidence == evidence and checked_docker == docker, 'frozen publication closure changed')
    guard()
    need(git('rev-parse', 'HEAD') == BASE_HEAD and git('branch', '--show-current') == BRANCH
         and git('remote', 'get-url', 'origin') == REMOTE
         and git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + BRANCH).split()[0] == BASE_HEAD,
         'exact owned source publication branch/head/origin required')
    need(git('status', '--porcelain=v1', '--untracked-files=all') == ''
         and git('rev-parse', '--path-format=absolute', '--git-dir') == str(ACCELERATOR / '.git')
         and git('rev-parse', '--path-format=absolute', '--git-common-dir') == str(ACCELERATOR / '.git'),
         'clean precharged self-contained owned Git checkout required')
    need(not TARGET.exists(), 'new immutable qualification publication generation must be absent')
    original_pr = read_pr()
    write_new(attempt / 'pr292-before.json', encoded({'number': 292, 'body': original_pr['body'],
        'updated_at': original_pr.get('updated_at')}), mode=0o600)
    TARGET.mkdir(mode=0o700)
    fsync_directory(TARGET.parent)
    contents, immutable = {}, set()
    def install(relative, raw, *, frozen=True):
        path = TARGET / safe_relative(relative)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_new(path, raw)
        name = str(path.relative_to(ACCELERATOR))
        need(name not in contents, 'duplicate publication destination')
        contents[name] = raw
        if frozen:
            immutable.add(name)
    for role, files in bundles.items():
        for relative, raw in files.items():
            install('sources/' + role + '/' + relative, raw)
    for role, raw in evidence.items():
        install('evidence/' + role + ('.jsonl' if role.endswith('_terminal') else '.json'), raw)
    for name, raw in docker.items():
        install('evidence/native-mechanical-public/' + name, raw)
    self_bundle = verify_bundle(HERE, config['source_sha256'].get(str(HERE / 'SOURCE_MANIFEST.json'), ''))
    for relative, raw in self_bundle.items():
        install('sources/publication-worker-v06/' + relative, raw)
    qualification.update(schema='legal-ir-real-handoff-runtime-qualification@3', historical_failures_preserved=True,
        qualification_source_manifests={role: row['manifest_sha256'] for role, row in config['bundles'].items()},
        evidence_sha256={role: sha(raw) for role, raw in evidence.items()}, publication_configuration_sha256=sha(config_raw))
    body = pr_body(qualification)
    install('README.md', readme(qualification), frozen=False)
    install('QUALIFICATION.json', encoded(qualification), frozen=False)
    install('READINESS_TASKS.json', encoded(readiness(qualification)), frozen=False)
    install('publication-config.json', config_raw)
    install('PR292_BODY.md', body.encode(), frozen=False)
    inventory = {'schema': 'legal-ir-real-handoff-publication-inventory@1', 'base_head': BASE_HEAD,
        'configuration_sha256': sha(config_raw), 'files': [
            {'path': path, 'bytes': len(raw), 'sha256': sha(raw)} for path, raw in sorted(contents.items())], **FALSE}
    install('PUBLICATION_INVENTORY.json', encoded(inventory), frozen=False)
    names = sorted(contents)
    need(len(names) <= 160 and all(path.startswith(str(TARGET.relative_to(ACCELERATOR)) + '/') for path in names),
         'bounded new documentation-only publication closure required')
    guard()
    need(git('rev-parse', 'HEAD') == BASE_HEAD and git('diff', '--cached', '--name-only') == '',
         'base or index changed during publication')
    for path, raw in contents.items():
        need(read(ACCELERATOR / path) == raw, 'installed publication bytes changed')
    git('add', '--sparse', '-f', '--', *names)
    staged = {raw.decode() for raw in git('diff', '--cached', '--name-only', '-z', raw=True).split(b'\0') if raw}
    need(staged == set(names), 'staged set differs from exact frozen source/evidence/document closure')
    for path, raw in contents.items():
        need(git('cat-file', 'blob', ':' + path, raw=True) == raw, 'staged blob differs from exact source bytes')
    checked = sorted(set(names) - immutable)
    git('diff', '--cached', '--check', '--', *checked)
    write_new(attempt / 'publication-staged-paths.json', encoded({
        'schema': 'legal-ir-real-handoff-staged-paths@1', 'paths': names,
        'file_sha256': {path: sha(raw) for path, raw in contents.items()},
        'whitespace_checked_generated_paths': checked,
        'exact_frozen_source_or_observation_exemptions': sorted(immutable)}), mode=0o600)
    guard()
    git('-c', 'user.name=Legal IR Supervisor', '-c', 'user.email=legal-ir-supervisor@localhost',
        '-c', 'commit.gpgsign=false', 'commit', '-m', 'Archive Legal IR handoff mechanics and native custody qualifications')
    commit = git('rev-parse', 'HEAD')
    need(git('rev-parse', 'HEAD^') == BASE_HEAD and git('status', '--porcelain=v1', '--untracked-files=all') == '',
         'one clean documentation-only publication commit required')
    committed_paths = {raw.decode() for raw in git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z',
        BASE_HEAD, commit, raw=True).split(b'\0') if raw}
    need(committed_paths == set(names), 'actual publication commit changed paths differ from the complete approved set')
    for path, raw in contents.items():
        need(git('cat-file', 'blob', commit + ':' + path, raw=True) == raw, 'committed publication blob differs')
    guard()
    git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
    need(git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + BRANCH).split()[0] == commit,
         'published GitHub branch commit differs')
    latest_pr = read_pr()
    need(latest_pr['body'] == original_pr['body'], 'PR body changed concurrently; preserve pushed commit and inspect')
    # This is a fresh body comparison followed by PATCH, not an atomic GitHub
    # compare-and-swap. A later concurrent editor could still race this update.
    patch = attempt / 'pr292-body-patch.json'
    write_new(patch, encoded({'body': body}), mode=0o600)
    updated = strict_json(gh('api', '--method', 'PATCH', PR_ENDPOINT, '--input', str(patch)))
    need(updated.get('number') == 292 and updated.get('body') == body and updated.get('state') == 'open'
         and updated.get('draft') is True and updated.get('merged') is False,
         'actual open draft PR292 body update failed verification')
    guard()
    need(git('rev-parse', 'HEAD') == commit and git('status', '--porcelain=v1', '--untracked-files=all') == '',
         'published owned checkout changed during PR metadata update')
    result = {'schema': 'legal-ir-real-handoff-source-publication@3', 'accelerator_commit': commit,
        'pushed': True, 'branch': BRANCH, 'configuration_sha256': sha(config_raw),
        'staged_path_count': len(names), 'tests_rerun': False, 'historical_failures_preserved': True,
        'previous_publication_commit': BASE_HEAD,
        'ordinary_success_hook_controlled_testcases': qualification['ordinary_success_hook_controlled_testcases'],
        'custody_memory_controlled_testcases': qualification['custody_memory_controlled_testcases'],
        'native_gate_v01_passed': qualification['native_gate_v01_passed'],
        'native_gate_v02_execution_observed': qualification['native_gate_v02_execution_observed'],
        'actual_nonmodel_gate_qualifier_passed': qualification['actual_nonmodel_gate_qualifier_passed'],
        'actual_custody_unwind_qualifier_passed': qualification['actual_custody_unwind_qualifier_passed'],
        'pr_url': 'https://github.com/endomorphosis/ipfs_accelerate_py/pull/292', 'pr_body_updated': True,
        'pr_merged': False, 'official_github_credential_helper_used': True, 'credential_values_exported': False, **FALSE}
    write_new(attempt / 'publication.json', encoded(result), mode=0o600)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
