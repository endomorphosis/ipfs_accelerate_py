"""Publish reviewed attempt-preservation sources under the root's real owner.

Authoring/importing this module performs no runtime action. The root launches
the CLI only through its genuine finite charged publication request. Serialized
configuration and archived observations do not grant resource or task authority.
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
import tempfile
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).absolute().parent
PHASE = HERE.parent
ACCELERATOR = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
TARGET = ACCELERATOR / 'docs/architecture/legal_ir_supervisor_start_20261016/attempt_preservation'
BRANCH = 'agent/legal-ir-ordinary-worker-20261016'
REMOTE = 'https://github.com/endomorphosis/ipfs_accelerate_py.git'
BASE_HEAD = '24af767c2587d82be03623c0d8a1ce19e1d902aa'
PYTEST_ROOT = Path('/tmp/pytest-of-barberb/legal-ir-preserve-attempts-20261017-tests01')
DEPLOYED_TEST = Path('test/api/test_preserve_task_attempts.py')
FROZEN_BUNDLES = {
    'preserve-attempts-patch-v01': '05f0369a1d3c2bc6b876c4fc51ea83ef7ab00ba2d2e25ad30b0aac4e66f8f225',
    'preserve-attempts-controls-v02': '8fa90c38f399e36dc37e7490252ff12340424f19f102344a8f62da5d9d005b28',
}
PRODUCTION = (
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py',
    'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py',
    'ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py',
)
CONFIG_FIELDS = {'schema', 'expected_accelerator_head', 'expected_branch', 'push',
                 'worker_manifest_sha256', 'prior_memory_observation'}
MAX_SOURCE_BYTES = 8 * 1024 * 1024


def need(condition, reason):
    if not condition:
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
    need(path.is_absolute() and os.path.abspath(path) == str(path) and '..' not in path.parts,
         'canonical absolute path required')
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink path refused')
    return path


def read(path, maximum=MAX_SOURCE_BYTES):
    path = canonical(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= maximum,
             'bounded single-link regular source/evidence required')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after) == identity(path.lstat()) and len(raw) == before.st_size,
         'source/evidence changed while reading')
    return raw


def safe_relative(spelling):
    need(type(spelling) is str and re.fullmatch(r'[A-Za-z0-9_./-]+', spelling) is not None,
         'bounded source relative spelling required')
    path = PurePosixPath(spelling)
    need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == spelling,
         'canonical source relative path required')
    return Path(spelling)


def new_directory(path):
    path = canonical(path)
    need(not path.exists(), 'new publication path already exists')
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    canonical(path)
    fsync_directory(path)
    fsync_directory(path.parent)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, raw, mode=0o644):
    path = canonical(path)
    need(path.parent.is_dir(), 'existing publication parent required')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def replace_verified(path, original_sha256, raw):
    path = canonical(path)
    need(sha(read(path)) == original_sha256, 'production source changed before replacement')
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.attempt-preservation-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        need(sha(read(path)) == original_sha256, 'production source changed during replacement')
        os.replace(temporary, path)
        temporary = None
        fsync_directory(path.parent)
        need(read(path) == raw, 'production replacement differs')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def git(*args, raw=False):
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                       GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    completed = subprocess.run(['/usr/bin/git', '-c', 'core.hooksPath=/dev/null',
        '-c', 'credential.helper=', '-c', 'credential.https://github.com.helper=!/usr/bin/gh auth git-credential',
        *args], cwd=ACCELERATOR, env=environment, capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
    need(completed.returncode == 0, 'owned Git operation failed: ' + ' '.join(args[:2])
         + ' (exit ' + str(completed.returncode) + ')')
    return completed.stdout if raw else completed.stdout.decode().strip()


def verify_bundle(root, expected_manifest):
    manifest_raw = read(root / 'SOURCE_MANIFEST.json', 1024 * 1024)
    need(sha(manifest_raw) == expected_manifest, 'immutable source manifest differs')
    manifest = strict_json(manifest_raw)
    need(type(manifest) is dict and type(manifest.get('files')) is list and 1 <= len(manifest['files']) <= 32,
         'finite immutable source manifest required')
    files = {}
    total = 0
    for row in manifest['files']:
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'closed source file pin required')
        relative = safe_relative(row['path'])
        need(relative.as_posix() not in files and relative.name != 'SOURCE_MANIFEST.json', 'unique source file required')
        content = read(root / relative)
        need(type(row['bytes']) is int and len(content) == row['bytes'] and sha(content) == row['sha256'],
             'immutable source file differs')
        files[relative.as_posix()] = content
        total += len(content)
    need(total <= MAX_SOURCE_BYTES, 'bounded immutable bundle required')
    files['SOURCE_MANIFEST.json'] = manifest_raw
    return files


def load_configuration(path, expected_sha256):
    raw = read(path, 1024 * 1024)
    need(sha(raw) == expected_sha256, 'frozen publication configuration differs')
    config = strict_json(raw)
    need(type(config) is dict and set(config) == CONFIG_FIELDS
         and config['schema'] == 'legal-ir-attempt-preservation-publication-config@1', 'closed publication config required')
    need(type(config['expected_accelerator_head']) is str
         and re.fullmatch(r'[0-9a-f]{40}', config['expected_accelerator_head']) is not None
         and config['expected_branch'] == BRANCH and config['push'] is True, 'explicit owned source publication required')
    need(type(config['worker_manifest_sha256']) is str
         and re.fullmatch(r'[0-9a-f]{64}', config['worker_manifest_sha256']) is not None, 'frozen worker manifest required')
    observation = config['prior_memory_observation']
    need(type(observation) is dict and set(observation) == {'absolute_path', 'bytes', 'sha256'},
         'closed root-observed prior engineering evidence pin required')
    path = canonical(observation['absolute_path'])
    owned_runs = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs')
    need(PHASE in path.parents or owned_runs in path.parents, 'explicit owned engineering observation required')
    content = read(path, 1024 * 1024)
    need(type(observation['bytes']) is int and len(content) == observation['bytes']
         and sha(content) == observation['sha256'] and type(strict_json(content)) is dict,
         'actual root-observed engineering evidence differs')
    return config, raw, content


def assess_junit(raw):
    root = ET.fromstring(raw)
    suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
    cases = [case for suite in suites for case in suite.findall('testcase')]
    need(len(cases) == 51 and all(int(suite.get(key, '0')) == 0 for suite in suites for key in ('failures', 'errors', 'skipped'))
         and all(not any(case.find(key) is not None for key in ('failure', 'error', 'skipped')) for case in cases),
         'actual deployed51-case zero-failure policy qualification required')
    return {'controlled_policy_testcases': 51, 'failures': 0, 'errors': 0, 'skipped': 0,
            'qualification_scope': 'ordinary-in-memory-policy-functions-and-source-contracts-only',
            'actual_provider_execution_qualified': False, 'actual_native_end_to_end_dispatch_qualified': False,
            'actual_legal_autoformalization_qualified': False, 'canonical_tasks_accepted': 0}


def main():
    need(len(sys.argv) == 3, 'usage: publish_attempt_preservation.py FROZEN_CONFIG EXPECTED_CONFIG_SHA256')
    # Generic V05 owner supplies private attempt TMPDIR; capture it before
    # selecting the separately precharged external pytest directory.
    attempt = canonical(os.environ['TMPDIR'])
    info = attempt.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not stat.S_IMODE(info.st_mode) & 0o077,
         'actual private charged owner attempt required')
    config, config_raw, prior_observation = load_configuration(Path(sys.argv[1]), sys.argv[2])
    bundles = {name: verify_bundle(PHASE / name, digest) for name, digest in FROZEN_BUNDLES.items()}
    worker_bundle = verify_bundle(HERE, config['worker_manifest_sha256'])
    need(worker_bundle.get('publish_attempt_preservation.py') == read(Path(__file__).absolute()),
         'executing immutable worker source differs')
    need(git('rev-parse', 'HEAD') == config['expected_accelerator_head'] and git('branch', '--show-current') == BRANCH
         and git('remote', 'get-url', 'origin') == REMOTE
         and git('status', '--porcelain=v1', '--untracked-files=all') == '', 'exact clean exclusively owned checkout required')
    git('merge-base', '--is-ancestor', BASE_HEAD, config['expected_accelerator_head'])
    need(git('rev-parse', '--path-format=absolute', '--git-dir') == str(ACCELERATOR / '.git')
         and git('rev-parse', '--path-format=absolute', '--git-common-dir') == str(ACCELERATOR / '.git'),
         'precharged self-contained owned Git directory required')
    need(not TARGET.exists() and not (ACCELERATOR / DEPLOYED_TEST).exists() and not PYTEST_ROOT.exists(),
         'new publication and unique pytest paths must be absent')
    canonical(PYTEST_ROOT.parent)
    fixture_parent = PYTEST_ROOT.parent.lstat()
    need(stat.S_ISDIR(fixture_parent.st_mode) and fixture_parent.st_uid == os.geteuid()
         and not stat.S_IMODE(fixture_parent.st_mode) & 0o022, 'owned existing pytest parent required')
    provenance = strict_json(bundles['preserve-attempts-patch-v01']['BASE_PROVENANCE.json'])
    rows = provenance['files']
    need(type(rows) is list and len(rows) == 4 and {r['path'] for r in rows} == set(PRODUCTION),
         'exact four-module source proposal required')
    new_sources = {}
    for row in rows:
        installed = read(ACCELERATOR / row['path'])
        proposed = bundles['preserve-attempts-patch-v01']['proposed/' + row['path']]
        need(len(installed) == row['original_bytes'] and sha(installed) == row['original_sha256']
             and len(proposed) == row['proposed_bytes'] and sha(proposed) == row['proposed_sha256'],
             'exact production base/proposed pin differs')
        ast.parse(proposed, filename=row['path'])
        new_sources[row['path']] = proposed
    frozen_test = bundles['preserve-attempts-controls-v02']['test_preserve_task_attempts.py']
    old_default = b"str(Path(__file__).parent.parent / 'preserve-attempts-patch-v01' / 'proposed')"
    new_default = b'str(Path(__file__).resolve().parents[2])'
    need(frozen_test.count(old_default) == 1, 'exact deployment-only test adaptation required')
    deployed_test = frozen_test.replace(old_default, new_default, 1)
    ast.parse(deployed_test, filename=str(DEPLOYED_TEST))
    # All base, source, evidence and destination checks precede repository writes.
    new_directory(TARGET)
    expected_contents = {}
    for name, files in {**bundles, 'publication-worker-v01': worker_bundle}.items():
        directory = TARGET / name
        new_directory(directory)
        for relative, content in files.items():
            destination = directory / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            write_new(destination, content)
            expected_contents[str(destination.relative_to(ACCELERATOR))] = content
    for row in rows:
        replace_verified(ACCELERATOR / row['path'], row['original_sha256'], new_sources[row['path']])
        expected_contents[row['path']] = new_sources[row['path']]
    (ACCELERATOR / DEPLOYED_TEST).parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    write_new(ACCELERATOR / DEPLOYED_TEST, deployed_test)
    expected_contents[DEPLOYED_TEST.as_posix()] = deployed_test
    new_directory(PYTEST_ROOT)
    junit_path = attempt / 'deployed-policy-tests.xml'
    log_path = attempt / 'deployed-policy-tests.log'
    need(not junit_path.exists() and not log_path.exists(), 'fresh private qualification outputs required')
    environment = dict(os.environ)
    environment.update(TMPDIR=str(PYTEST_ROOT), TMP=str(PYTEST_ROOT), TEMP=str(PYTEST_ROOT),
        PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='',
        PRESERVE_ATTEMPTS_SOURCE_ROOT=str(ACCELERATOR))
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '--noconftest', '-c', '/dev/null',
        '--rootdir=' + str(ACCELERATOR), '--confcutdir=' + str((ACCELERATOR / DEPLOYED_TEST).parent),
        '-p', 'no:cacheprovider', '--basetemp=' + str(PYTEST_ROOT), '--junitxml=' + str(junit_path),
        str(ACCELERATOR / DEPLOYED_TEST)]
    started = time.time()
    result = subprocess.run(command, cwd=ACCELERATOR, env=environment, capture_output=True,
                            stdin=subprocess.DEVNULL, timeout=180)
    write_new(log_path, result.stdout + b'\n--- STDERR ---\n' + result.stderr, mode=0o600)
    need(result.returncode == 0, 'actual deployed policy controls failed; private evidence preserved')
    junit_raw = read(junit_path, 4 * 1024 * 1024)
    assessment = assess_junit(junit_raw)
    # Exact source rechecks bind tests and the eventual staged blobs to the same
    # applied source. These checks confer no resource/task completion authority.
    for relative, content in expected_contents.items():
        need(read(ACCELERATOR / relative) == content, 'published source changed during qualification')
    deployment = {'schema': 'legal-ir-attempt-preservation-deployment-provenance@1',
        'base_accelerator_head': config['expected_accelerator_head'], 'source_proposal_manifest_sha256': FROZEN_BUNDLES['preserve-attempts-patch-v01'],
        'corrected_controls_manifest_sha256': FROZEN_BUNDLES['preserve-attempts-controls-v02'],
        'worker_manifest_sha256': config['worker_manifest_sha256'],
        'deployed_test_source_sha256': sha(deployed_test), 'frozen_test_source_sha256': sha(frozen_test),
        'only_test_source_adaptation': {'from': old_default.decode(), 'to': new_default.decode()},
        'production_sources': rows, 'prior_engineering_observation_sha256': sha(prior_observation),
        'prior_observation_classification': 'root-supplied-factual-engineering-observation-not-authority',
        'actual_deployed_junit_sha256': sha(junit_raw), 'qualification_started_at': started,
        'qualification_observed_finished_at': time.time(), 'actual_test_command': command,
        'external_pytest_directory': str(PYTEST_ROOT), 'taskboard_or_taskstate_changed': False,
        'resource_policy_or_ledger_changed': False, 'provider_execution_performed': False}
    readme = ('''# Preserve finite task attempt history

The optional scheduler setting `preserve_task_attempts: true` carries `--preserve-task-attempts` through the supervisor and daemon. It disables automatic exhausted-task rearming, completed-repair resets, repaired cooldown resets and unknown vanished-worker rollback. A restore may refund a consumed ordinal only on an explicitly verified no-provider prelaunch path. Omitted settings keep the existing behavior.

The actual deployed qualification passed all 51 in-memory policy/source-contract cases with zero failures, errors or skips. The preserved root-supplied observation records the earlier stdout-only controls (51 retry-policy and 31 archival cases, execution session 81356); it is factual engineering history, not a native capability. Both the original failed 50/51 controls and corrected controls remain preserved in their source bundles. `QUALIFICATION.json`, `deployed-policy-tests.xml`, and `DEPLOYMENT_PROVENANCE.json` bind the deployed source and exact test bytes.

This publication changes four implementation modules and adds the focused test and documentation. It does not enable the setting for an existing board, change an attempt cap, read or modify live task state, reset counters, dispatch a provider, or complete a canonical task. The next real packet must preserve the original canonical task identity and both consumed attempts, explicitly select the new setting and finite cap, and pass actual native/provider/candidate acceptance. These policy tests do not establish end-to-end dispatch, legal fidelity, training accuracy, logic-family compliance or Lean compilation.

The complete `preserve-attempts-patch-v01` and `preserve-attempts-controls-v02` sibling bundles retain their historical bytes and manifests. Only the installed production test's default root expression is adapted to `Path(__file__).resolve().parents[2]`; all assertions remain unchanged. `publication-worker-v01` contains the exact source-only publication worker. Runtime tools retain explicit host paths and need new genuine ownership on another machine.
''').encode()
    documents = {'README.md': readme, 'QUALIFICATION.json': encoded({**assessment, 'actual_junit_sha256': sha(junit_raw),
                    'actual_deployed_test_source_sha256': sha(deployed_test), 'new_runtime_setting_enabled': False}),
        'DEPLOYMENT_PROVENANCE.json': encoded(deployment), 'publication-config.json': config_raw,
        'prior-memory-observation.json': prior_observation, 'deployed-policy-tests.xml': junit_raw}
    for relative, content in documents.items():
        write_new(TARGET / relative, content)
        expected_contents[str((TARGET / relative).relative_to(ACCELERATOR))] = content
    stage_names = sorted(expected_contents)
    expected_non_docs = set(PRODUCTION) | {DEPLOYED_TEST.as_posix()}
    docs_prefix = str(TARGET.relative_to(ACCELERATOR)) + '/'
    need({p for p in stage_names if not p.startswith(docs_prefix)} == expected_non_docs,
         'stage closure contains an unexpected non-documentation path')
    for relative, content in expected_contents.items():
        need(read(ACCELERATOR / relative) == content, 'source/doc bytes changed before staging')
    need(git('rev-parse', 'HEAD') == config['expected_accelerator_head'] and git('diff', '--cached', '--name-only') == '',
         'publication base/index changed during qualification')
    git('add', '--sparse', '-f', '--', *stage_names)
    staged = git('diff', '--cached', '--name-only', '-z', raw=True).split(b'\0')
    staged = {p.decode() for p in staged if p}
    need(staged == set(stage_names), 'staged paths must be exactly four production modules plus new test and explicit docs')
    for relative, content in expected_contents.items():
        need(git('cat-file', 'blob', ':' + relative, raw=True) == content, 'staged blob differs from reviewed source/evidence')
    write_new(attempt / 'publication-staged-paths.json', encoded({'schema': 'legal-ir-attempt-preservation-staged-paths@1',
        'paths': stage_names, 'file_sha256': {p: sha(c) for p, c in expected_contents.items()}}), mode=0o600)
    git('diff', '--cached', '--check')
    git('-c', 'user.name=Legal IR Supervisor', '-c', 'user.email=legal-ir-supervisor@localhost',
        '-c', 'commit.gpgsign=false', 'commit', '-m', 'Preserve finite task attempts across automatic retry recovery')
    commit = git('rev-parse', 'HEAD')
    need(git('rev-parse', 'HEAD^') == config['expected_accelerator_head']
         and git('status', '--porcelain=v1', '--untracked-files=all') == '', 'publication must finish at one clean child commit')
    git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
    remote = git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + BRANCH)
    need(remote.split()[0] == commit, 'actual remote branch commit differs')
    outcome = {'schema': 'legal-ir-attempt-preservation-publication@1', 'accelerator_commit': commit,
        'branch': BRANCH, 'pushed': True, **assessment, 'staged_path_count': len(stage_names),
        'configuration_sha256': sha(config_raw), 'worker_manifest_sha256': config['worker_manifest_sha256'],
        'production_source_sha256': {p: sha(c) for p, c in new_sources.items()},
        'deployed_test_source_sha256': sha(deployed_test), 'actual_deployed_junit_sha256': sha(junit_raw),
        'official_github_credential_helper_used': True, 'credential_values_exported': False,
        'new_runtime_setting_enabled': False, 'taskboard_or_taskstate_changed': False,
        'resource_policy_or_ledger_changed': False}
    write_new(attempt / 'publication.json', encoded(outcome), mode=0o600)
    print(json.dumps(outcome, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
