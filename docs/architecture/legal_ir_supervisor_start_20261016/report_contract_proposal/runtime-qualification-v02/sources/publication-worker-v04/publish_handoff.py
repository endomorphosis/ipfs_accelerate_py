"""Publish exact source proposals and genuinely observed fixture qualifications.

Root executes this worker under the existing finite charged owner. It does not
deploy a runtime hook, run tests, invoke providers, materialize a packet, change
attempt counters or issue/consume a native acceptance handle.
"""
from __future__ import annotations

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
OLD_PHASE = Path('/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01')
ACCELERATOR = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller')
PREFIX = 'docs/architecture/legal_ir_supervisor_start_20261016'
TARGET = ACCELERATOR / PREFIX / 'report_contract_proposal/runtime-qualification-v02'
RUNS = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs')
BASE_HEAD = '871ad14eeb757500ad7b087a5acaa048901d09d4'
BRANCH = 'agent/legal-ir-ordinary-worker-20261016'
REMOTE = 'https://github.com/endomorphosis/ipfs_accelerate_py.git'
PR_ENDPOINT = 'repos/endomorphosis/ipfs_accelerate_py/pulls/292'
NATIVE_V05_MANIFEST = 'd5049a12a0622c7f4b9462db7657c3920130ed4c59a17919a0f2e440bb372313'
OBSERVER_V03_MANIFEST = '829d35224fc21fa40bd63e50b8d9073e3e66e58dd332710fc85d29caa214be97'
HISTORICAL_V02_MANIFEST = '1699b077d97ba65edb354826123d1785d38631d35651140f919432bb61b4827e'
HISTORICAL_GIT_REQUEST = '9b00dfd75f99b40737f26feadc9c754f2c1dc94fcb548bb3c81b5920dfea3530'
PARSER_HELPERS = tuple(ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/validation' / (name + '.py')
                       for name in ('validation_commands', 'evidence_output_scope'))
V05_REQUEST = '4130d2adba4c5202353109b7c3be810f0b788ceae1b97b73d18cd10a6a9d55f0'
BUNDLE_ROOTS = {
    'real-candidate-binding-v03': PHASE / 'real-candidate-binding-v03',
    'historical-real-candidate-binding-v02': PHASE / 'real-candidate-binding-v02',
    'candidate-observer-qualification-v02': PHASE / 'candidate-observer-qualification-v02',
    'historical-candidate-observer-qualification-v01': PHASE / 'candidate-observer-qualification-v01',
    'candidate-freezer-v01': PHASE / 'candidate-freezer-v01',
    'provider-observer-revision-v02': PHASE / 'provider-observer-v01/revision-v02',
}
REQUIRED_EVIDENCE = {'native_docker_root', 'native_docker_terminal', 'native_docker_request',
                     'reader_controls_root', 'reader_controls_result', 'reader_controls_request',
                     'git_observer_root', 'git_observer_terminal', 'git_observer_request',
                     'git_observer_prior_failure_root', 'git_observer_prior_failure_terminal', 'git_observer_prior_failure_request'}
HOOK_EVIDENCE = {'hook_controls_root', 'hook_controls_result'}
PRIOR_READER_EVIDENCE = {'reader_prior_failure_result', 'reader_prior_failure_request'}
PRIOR_V02_READER_PASS = {'reader_v02_prior_pass_root', 'reader_v02_prior_pass_result', 'reader_v02_prior_pass_request'}
OPTIONAL_EVIDENCE = HOOK_EVIDENCE | PRIOR_READER_EVIDENCE | PRIOR_V02_READER_PASS
DOCKER_PUBLIC_FILES = {'actual-attached-docker-client.json', 'actual-container-confinement-before-checker.json',
                       'actual-container-exit.json', 'actual-docker-cleanup.json', 'actual-validation-admission.json',
                       'frozen-docker-runtime.json'}
CONFIG_FIELDS = {'schema', 'expected_accelerator_head', 'expected_branch', 'push', 'bundles',
                 'evidence', 'docker_public_artifacts', 'source_sha256'}
FALSE = dict(source_pipeline_dispatchable=False, packet_materialized=False, board_enabled=False,
             task_attempts_changed=False, provider_calls_performed=False, canonical_tasks_accepted=0,
             actual_real_provider_to_acceptance_handoff_qualified=False, production_hook_deployed=False,
             acceptance_authority=False, completion_authority=False)
MAXIMUM = 8 * 1024 * 1024


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


def pinned_evidence(row):
    need(type(row) is dict and set(row) == {'absolute_path', 'bytes', 'sha256'}, 'closed explicit engineering evidence pin required')
    path = canonical(row['absolute_path'])
    need((PHASE in path.parents or RUNS in path.parents) and path.suffix in {'.json', '.jsonl'},
         'explicit owned public engineering JSON evidence required')
    need(type(row['bytes']) is int and 0 < row['bytes'] <= 1024 * 1024 and type(row['sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'bounded engineering evidence digest required')
    raw = read(path, 1024 * 1024)
    need(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'actual engineering evidence bytes differ')
    return raw


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


def assess_qualification(config, evidence, bundles):
    parsed = {key: strict_json(raw) for key, raw in evidence.items()}
    native, terminal, request = (parsed[key] for key in ('native_docker_root', 'native_docker_terminal', 'native_docker_request'))
    need(native.get('schema') == 'legal-ir-native-v05-fixture-root-observation@1'
         and native.get('actual_exec_exit_code') == 0 and native.get('actual_exec_session') == 16091
         and native.get('actual_full_native_v05_fixture_passed') is True and native.get('synthetic_fixture_only') is True
         and native.get('canonical_tasks_accepted') == 0 and native.get('provider_calls_performed') is False
         and native.get('old_failures_rewritten') is False and native.get('policy_changed') is False
         and native.get('regression_manifest_sha256') == NATIVE_V05_MANIFEST
         and native.get('source_request_sha256') == V05_REQUEST
         and native.get('terminal_sha256') == sha(evidence['native_docker_terminal'])
         and native.get('actual_native_terminal') == terminal,
         'actual source-bound V05 native fixture qualification required')
    outcome = terminal.get('outcome', {})
    need(terminal.get('schema') == 'legal-ir-native-docker-fixture-entry-terminal@4'
         and terminal.get('scope') == 'synthetic-native-acceptance-regression-only'
         and terminal.get('request_sha256') == V05_REQUEST and terminal.get('actual_preparation_finalized') is True
         and terminal.get('provider_calls_performed') is False and terminal.get('real_LIR_task_accepted') is False
         and all(outcome.get(key) is True for key in ('actual_container_exit_and_cleanup_verified',
             'actual_handle_consumed_once', 'actual_native_validation_finalized', 'actual_separate_orchestration_finalized', 'fixture_board_published'))
         and outcome.get('provider_calls_performed') is False and outcome.get('real_LIR_task_accepted') is False,
         'actual V05 cleanup and one-use synthetic acceptance closeout required')
    need(sha(evidence['native_docker_request']) == V05_REQUEST
         and request.get('regression_manifest_sha256') == NATIVE_V05_MANIFEST
         and request.get('schema') == 'legal-ir-native-docker-fixture-entry@4', 'actual V05 request correspondence required')
    selected_manifest = ACCELERATOR / PREFIX / 'factual_acceptance/native-regression/v05/SOURCE_MANIFEST.json'
    need(sha(read(selected_manifest)) == NATIVE_V05_MANIFEST, 'already published actual V05 source generation changed')
    reader_root, reader = parsed['reader_controls_root'], parsed['reader_controls_result']
    reader_request = parsed['reader_controls_request']
    native_closeout = reader_root.get('actual_native_terminal', {})
    reader_attempt = canonical(reader_request.get('attempt_directory'))
    reader_result_path = canonical(config['evidence']['reader_controls_result']['absolute_path'])
    reader_receipt_path = canonical(native_closeout.get('receipt'))
    need(RUNS in reader_attempt.parents and reader_result_path == reader_attempt / 'candidate-reader-qualification.json'
         and reader_receipt_path == reader_attempt / 'receipt.json',
         'reader request, actual owner terminal receipt and result must share the original attempt')
    need(reader_root.get('schema') == 'legal-ir-candidate-readers-root-observation@1'
         and reader_root.get('result_sha256') == sha(evidence['reader_controls_result'])
         and reader_root.get('actual_exec_exit_code') == 0
         and reader_root.get('source_manifest_sha256') == OBSERVER_V03_MANIFEST
         and reader_root.get('source_request_sha256') == sha(evidence['reader_controls_request'])
         and native_closeout.get('status') == 'completed' and native_closeout.get('native_finalized') is True
         and native_closeout.get('child_returncode') == 0,
         'actual reader control owner completion required')
    need(reader.get('schema') == 'legal-ir-candidate-reader-qualification@1' and reader.get('passed') is True
         and reader.get('returncode') == 0 and reader.get('controlled_testcases') == reader.get('passed_testcases') == 60
         and all(reader.get(key) == 0 for key in ('failures', 'errors', 'skipped', 'canonical_tasks_accepted'))
         and reader.get('provider_calls_performed') is False and reader.get('actual_public_observer_qualified') is False
         and reader.get('source_manifest_sha256') == OBSERVER_V03_MANIFEST
         and config['bundles']['real-candidate-binding-v03']['manifest_sha256'] == OBSERVER_V03_MANIFEST,
         'actual60 passed reader controls bound to V03 required')
    current_manifest = strict_json(bundles['real-candidate-binding-v03']['SOURCE_MANIFEST.json'])
    need(current_manifest.get('controlled_testcases') == 60
         and current_manifest.get('controlled_testcase_counts') == {
             'test_binding_memory.py': 36, 'test_binding_readers.py': 23, 'test_public_parser_regression.py': 1},
         'current60 controls must match exact frozen V03 source inventory')
    need(reader_request.get('schema') == 'legal-ir-native-orchestration-request@1'
         and reader_request.get('source_sha256', {}).get(str(PHASE / 'real-candidate-binding-v03/SOURCE_MANIFEST.json')) == OBSERVER_V03_MANIFEST,
         'actual reader owner request must pin the published V03 source')
    git_root, git_terminal, git_request = (parsed[key] for key in ('git_observer_root', 'git_observer_terminal', 'git_observer_request'))
    need(git_root.get('actual_exec_exit_code') == 0 and git_root.get('actual_native_terminal') == git_terminal
         and git_root.get('source_request_sha256') == sha(evidence['git_observer_request']),
         'actual real-Git public observer root provenance required')
    need(git_terminal.get('schema') == 'legal-ir-native-candidate-observer-qualification-terminal@1'
         and git_terminal.get('passed') is True and git_terminal.get('synthetic_metadata_fixture') is True
         and git_terminal.get('actual_public_observer_and_git_qualified') is True
         and git_terminal.get('request_sha256') == sha(evidence['git_observer_request'])
         and all(git_terminal.get(key) is True for key in ('actual_preparation_finalized_before_observation',
             'actual_preparation_finalized', 'actual_observation_finalized', 'actual_root_released', 'actual_phase_released'))
         and git_terminal.get('actual_setup_git_children') == 7 and git_terminal.get('actual_observer_git_children') == 10
         and git_terminal.get('canonical_tasks_accepted') == 0
         and all(git_terminal.get(key) is False for key in ('provider_calls_performed', 'actual_provider_outcome_observed',
             'native_validation_qualified', 'real_LIR_task_accepted', 'acceptance_authority', 'completion_authority',
             'dispatch_authority', 'canonical_attempt_counters_changed')),
         'actual independent native/Git metadata fixture qualification required')
    need(git_request.get('schema') == 'legal-ir-native-candidate-observer-qualification-request@1'
         and git_request.get('scope') == 'synthetic_metadata_fixture_actual_native_git_observation_only'
         and git_request.get('observer_source_sha256') == sha(bundles['real-candidate-binding-v03']['real_candidate_binding.py']),
         'actual Git fixture request must pin the published V03 observer')
    source_request = bundles['candidate-observer-qualification-v02'].get('request-v01.json')
    original_request = strict_json(source_request)
    original_pins, actual_pins = original_request.get('source_sha256'), git_request.get('source_sha256')
    helper_paths = {str(path) for path in PARSER_HELPERS}
    need(type(original_pins) is dict and type(actual_pins) is dict
         and set(actual_pins) == set(original_pins) | helper_paths
         and all(actual_pins[path] == digest for path, digest in original_pins.items())
         and {key: value for key, value in git_request.items() if key != 'source_sha256'} == {
             key: value for key, value in original_request.items() if key != 'source_sha256'},
         'actual Git request may only add exact original-parser helper pins to the frozen V02 worker request')
    for path in PARSER_HELPERS:
        need(actual_pins[str(path)] == reader_request.get('source_sha256', {}).get(str(path))
             == sha(read(path)), 'actual parser helper source must match reader and Git request closures')
    qualifier_worker = PHASE / 'candidate-observer-qualification-v02/qualify_candidate_observer.py'
    need(actual_pins.get(str(qualifier_worker)) == sha(bundles['candidate-observer-qualification-v02']['qualify_candidate_observer.py']),
         'actual extended request must use the unchanged frozen V02 qualifier worker')
    failed_root, failed_terminal = parsed['git_observer_prior_failure_root'], parsed['git_observer_prior_failure_terminal']
    need(failed_root.get('schema') == 'legal-ir-candidate-observer-failure-root-observation@1'
         and failed_root.get('actual_exec_exit_code') == 1 and failed_root.get('actual_exec_session') == 70147
         and failed_root.get('source_request_sha256') == HISTORICAL_GIT_REQUEST
         and failed_root.get('actual_native_terminal') == failed_terminal
         and failed_root.get('terminal_sha256') == sha(evidence['git_observer_prior_failure_terminal'])
         and failed_root.get('error_reason') == 'exact_legacy_task_population_required'
         and failed_root.get('old_failure_rewritten') is False
         and failed_root.get('canonical_tasks_accepted') == 0 and failed_root.get('provider_calls_performed') is False
         and sha(evidence['git_observer_prior_failure_request']) == HISTORICAL_GIT_REQUEST
         and bundles['historical-candidate-observer-qualification-v01'].get('request-v01.json') == evidence['git_observer_prior_failure_request']
         and config['bundles']['historical-real-candidate-binding-v02']['manifest_sha256'] == HISTORICAL_V02_MANIFEST,
         'original actual V02 public Git r01 population failure must remain preserved')
    need(failed_terminal.get('schema') == 'legal-ir-native-candidate-observer-qualification-terminal@1'
         and failed_terminal.get('passed') is False and failed_terminal.get('error_type') == 'BindingRefusal'
         and failed_terminal.get('actual_public_observer_and_git_qualified') is False
         and failed_terminal.get('canonical_tasks_accepted') == 0
         and all(failed_terminal.get(key) is False for key in ('provider_calls_performed', 'actual_provider_outcome_observed',
             'native_validation_qualified', 'real_LIR_task_accepted', 'acceptance_authority', 'completion_authority',
             'dispatch_authority', 'canonical_attempt_counters_changed')),
         'failed public V02 observer must not be relabeled qualified')
    hook_cases = None
    if 'ordinary-success-hook-proposal' in config['bundles']:
        need({'hook_controls_root', 'hook_controls_result'} <= set(evidence), 'hook proposal requires actual root-observed passed controls')
        hook_root, hook = parsed['hook_controls_root'], parsed['hook_controls_result']
        digest = config['bundles']['ordinary-success-hook-proposal']['manifest_sha256']
        need(hook_root.get('actual_exec_exit_code') == 0 and hook_root.get('qualified_source_manifest_sha256') == digest
             and hook.get('passed') is True and hook.get('source_manifest_sha256') == digest
             and type(hook.get('controlled_testcases')) is int and 0 < hook['controlled_testcases'] <= 256
             and all(hook.get(key) == 0 for key in ('failures', 'errors', 'canonical_tasks_accepted'))
             and hook.get('provider_calls_performed') is False and hook.get('actual_provider_execution_qualified') is False,
             'actual passed source-only hook controls required without provider qualification')
        hook_cases = hook['controlled_testcases']
    else:
        need(not (HOOK_EVIDENCE & set(evidence)), 'unselected hook controls must not imply a deployed hook')
    if PRIOR_READER_EVIDENCE & set(evidence):
        need(PRIOR_READER_EVIDENCE <= set(evidence), 'failed reader harness request/result must be preserved together')
        prior = parsed['reader_prior_failure_result']
        need(prior.get('schema') == 'legal-ir-candidate-reader-qualification@1' and prior.get('passed') is False
             and prior.get('controlled_testcases') == 54 and prior.get('passed_testcases') == 58
             and all(prior.get(key) == 0 for key in ('failures', 'errors', 'skipped', 'canonical_tasks_accepted'))
             and prior.get('provider_calls_performed') is False and prior.get('source_manifest_sha256') == HISTORICAL_V02_MANIFEST,
             'preserved prior reader harness double-count refusal must not be relabeled passed')
    if PRIOR_V02_READER_PASS & set(evidence):
        need(PRIOR_V02_READER_PASS <= set(evidence), 'historical V02 isolated reader pass evidence must remain complete')
        old_root, old_result, old_request = (parsed[key] for key in (
            'reader_v02_prior_pass_root', 'reader_v02_prior_pass_result', 'reader_v02_prior_pass_request'))
        old_closeout = old_root.get('actual_native_terminal', {})
        old_attempt = canonical(old_request.get('attempt_directory'))
        need(old_root.get('actual_exec_exit_code') == 0 and old_root.get('result_sha256') == sha(evidence['reader_v02_prior_pass_result'])
             and old_root.get('source_request_sha256') == sha(evidence['reader_v02_prior_pass_request'])
             and old_root.get('source_manifest_sha256') == HISTORICAL_V02_MANIFEST
             and old_closeout.get('status') == 'completed' and old_closeout.get('native_finalized') is True
             and old_closeout.get('child_returncode') == 0 and canonical(old_closeout.get('receipt')) == old_attempt / 'receipt.json'
             and canonical(config['evidence']['reader_v02_prior_pass_result']['absolute_path']) == old_attempt / 'candidate-reader-qualification.json'
             and old_result.get('schema') == 'legal-ir-candidate-reader-qualification@1'
             and old_result.get('passed') is True and old_result.get('controlled_testcases') == old_result.get('passed_testcases') == 54
             and all(old_result.get(key) == 0 for key in ('failures', 'errors', 'skipped', 'canonical_tasks_accepted'))
             and old_result.get('source_manifest_sha256') == HISTORICAL_V02_MANIFEST
             and old_result.get('actual_public_observer_qualified') is False and old_result.get('provider_calls_performed') is False,
             'historical54 V02 isolated controls remain distinct from failed actual public observer')
    return {'actual_full_native_v05_fixture_passed': True, 'actual_native_v05_exec_session': 16091,
            'actual_public_candidate_observer_fixture_passed': True, 'controlled_reader_testcases': 60,
            'ordinary_success_hook_controlled_testcases': hook_cases,
            'qualified_native_v05_source_manifest_sha256': NATIVE_V05_MANIFEST,
            'qualified_observer_v03_source_manifest_sha256': OBSERVER_V03_MANIFEST,
            'historical_v02_public_git_r01_failed': True,
            'historical_v02_public_observer_qualified': False,
            'native_fixture_scope': 'synthetic-native-acceptance-regression-only',
            'candidate_observer_scope': 'synthetic_metadata_fixture_actual_native_git_observation_only', **FALSE}


def load_configuration(path, expected_digest):
    raw = read(path, 1024 * 1024)
    need(sha(raw) == expected_digest, 'frozen handoff publication config differs')
    config = strict_json(raw)
    need(type(config) is dict and set(config) == CONFIG_FIELDS
         and config['schema'] == 'legal-ir-handoff-source-publication-config@1'
         and config['expected_accelerator_head'] == BASE_HEAD and config['expected_branch'] == BRANCH
         and config['push'] is True, 'exact source-only publication configuration required')
    need(type(config['bundles']) is dict and set(BUNDLE_ROOTS) <= set(config['bundles'])
         and set(config['bundles']) <= set(BUNDLE_ROOTS) | {'ordinary-success-hook-proposal'},
         'exact reviewed required/optional source role set required')
    bundles = {}
    for role, row in config['bundles'].items():
        need(type(row) is dict and set(row) == {'absolute_root', 'manifest_sha256'}, 'closed frozen bundle selector required')
        root = canonical(row['absolute_root'])
        if role in BUNDLE_ROOTS:
            need(root == BUNDLE_ROOTS[role], 'fixed source bundle root differs')
        else:
            need(root.parent == PHASE and re.fullmatch('ordinary-success-hook-v[0-9]{2}', root.name), 'bounded optional source-only hook generation required')
        bundles[role] = verify_bundle(root, row['manifest_sha256'])
    need(type(config['evidence']) is dict and REQUIRED_EVIDENCE <= set(config['evidence'])
         and set(config['evidence']) <= REQUIRED_EVIDENCE | OPTIONAL_EVIDENCE, 'closed actual evidence role set required')
    evidence = {key: pinned_evidence(row) for key, row in config['evidence'].items()}
    # Archive the actual current reader harness from its exact native owner
    # request. A prior failed harness may be preserved independently as well.
    for role, request_role in (('reader-harness-current', 'reader_controls_request'),
                               ('reader-harness-prior-failure', 'reader_prior_failure_request'),
                               ('reader-harness-v02-prior-pass', 'reader_v02_prior_pass_request')):
        if request_role not in evidence:
            continue
        owner_request = strict_json(evidence[request_role])
        command = owner_request.get('command')
        need(type(command) is list and len(command) == 3 and command[:2] == ['/usr/bin/python3.12', '-B'],
             'actual bounded reader harness command required')
        harness = canonical(command[2])
        need(harness.parent == PHASE and re.fullmatch('qualify_candidate_readers_v[0-9]{2}\\.py', harness.name),
             'exact phase-owned reader harness required')
        raw_harness = read(harness)
        need(owner_request.get('source_sha256', {}).get(str(harness)) == sha(raw_harness),
             'actual reader harness source must match original owner request')
        bundles[role] = {harness.name: raw_harness, 'frozen-owner-request.json': evidence[request_role]}
    need(type(config['docker_public_artifacts']) is list and len(config['docker_public_artifacts']) <= len(DOCKER_PUBLIC_FILES),
         'finite explicit public Docker evidence allowlist required')
    docker = {}
    actual_control = RUNS / 'legal-ir-native-acceptance-fixture-20261017-r04/preparation/docker-factual-control'
    for row in config['docker_public_artifacts']:
        content = pinned_evidence(row)
        path = Path(row['absolute_path'])
        need(path.parent == actual_control and path.name in DOCKER_PUBLIC_FILES and path.name not in docker,
             'private handle/key/raw ledger or unlisted runtime artifact is not publishable')
        docker[path.name] = content
    pins = config['source_sha256']
    required = {str(Path(__file__).absolute()), str(HERE / 'SOURCE_MANIFEST.json'),
                '/usr/bin/git', '/usr/bin/gh', '/usr/bin/python3.12',
                str(ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py'),
                str(ACCELERATOR / 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'),
                *(str(path) for path in PARSER_HELPERS)}
    need(type(pins) is dict and required <= set(pins) and len(pins) <= 256, 'protected actual publisher/runner/daemon source pins required')
    for spelling, digest in pins.items():
        need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest)
             and sha(read(canonical(spelling), 64 * 1024 * 1024)) == digest, 'protected source changed')
    return config, raw, bundles, evidence, docker


def readme(qualification):
    hook = qualification['ordinary_success_hook_controlled_testcases']
    hook_text = ('An ordinary success-hook source proposal is archived with ' + str(hook) +
                 ' root-observed source controls. It is not deployed or qualified for actual provider execution.\n') if hook else (
                 'The original successful provider host hook remains a documented upstream integration task.\n')
    return ('''# Current runtime qualification: NON-DISPATCHABLE

The actual source-bound V05 Docker/native fixture passed in exec 16091 with native validation finalized, a separate genuine ORCHESTRATION coordinator finalized, one-use handle consumed once, fixture board publication and container/watchdog cleanup verified. This is a synthetic engineering fixture. Historical r01-r03 failures and their original source-generation facts remain byte-preserved; this new generation does not relabel those failures or make an older source generation qualified retrospectively.

The additive candidate observer V03 passed 60 reader/correspondence/public-parser controls and a separate actual native/Git metadata fixture. It permits stable exact-SHA0664 trusted implementation sources while keeping candidate board/report input permissions strict. It independently binds the exact ordered original 64 task IDs to the frozen board and authored inventory. Earlier V02 passed 54 isolated controls but failed actual public Git r01 on its invented contiguous task-ID population; that failed generation remains preserved and is not qualified retrospectively. The actual V03 fixture used separate genuine preparation and fresh read-only observation owners, 7 attached setup Git children and 10 attached observer Git children. Its candidate is explicitly synthetic metadata; it does not establish a real successful provider outcome or accept an original LIR task.

''' + hook_text + '''
The real pipeline remains NON-DISPATCHABLE. Remaining work is the original successful provider host hook, an opt-in report-only candidate checkpoint reusing daemon commit/serialization scopes, real provider/candidate/attempt binding, real-context native Docker integration and original-board controller CAS. The two historical consumed attempts remain preserved, total cap 3 is proposed for a future reviewed packet, and all 63 later tasks remain blocked. No production hook, packet, board or counters are changed by this publication. `--check-report` remains non-accepting at exit 3; native acceptance uses the separate trusted host API.

Root observations, exact frozen requests/source manifests and selected public runtime evidence are archived here. They are inert engineering metadata and cannot mint native ownership or one-use authority. This publication does not rerun tests, make provider calls, change storage/CPU policies, publish private capability keys or alter GitHub billing/workflows. Previously observed hosted CI was unable to start because of an account billing lock; local qualifications are reported separately.

Canonical accepted legal tasks remain 0/64. Statutory fidelity, span/entity graph accuracy, decoder training accuracy, logic-family semantics and lake build legal require the later evaluation tasks and are not established by infrastructure qualification.
''').encode()


def pr_body(qualification):
    hook = qualification['ordinary_success_hook_controlled_testcases']
    optional = (' The archived success-hook source proposal passed ' + str(hook) +
                ' controls and remains undeployed.') if hook else ''
    return ('''Fix Legal IR worker cleanup races, preserve finite retry history and recover storage-ledger blockers. Publish the exact factual acceptance, packet proposal and candidate-observer sources with their independently scoped qualification evidence.

The current V05 full Docker/native synthetic fixture passed in exec 16091: actual validation and separate ORCHESTRATION closeout, one-use handle consumption, fixture publication and container/watchdog cleanup verified. Candidate observer V03 passed 60 reader/correspondence/public-parser controls and a separate genuine native/Git metadata fixture. Earlier 21 packet controls, 65 handoff source controls, 51 deployed retry controls and 65 SDK recovery/archive controls remain documented in their original generations.''' + optional + '''

The real provider-to-acceptance pipeline remains non-dispatchable. The successful original provider host hook, report-only candidate checkpoint, real-context validation and original-board acceptance integration are pending. No production hook is deployed here; two consumed attempts and all 63 future blocked tasks remain intact. No canonical legal task has been accepted, and no model-accuracy, legal-fidelity or Lean-build claim is made. Historical failed qualifications are preserved.

Hosted CI was observed unable to start because of the GitHub account billing lock. Local qualification results are separate. This is a draft PR stacked on the existing Docker bootstrap branch; do not merge as an enabled production pipeline.
''')


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
    qualification = assess_qualification(config, evidence, bundles)
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
        install('evidence/native-v05-public/' + name, raw)
    self_bundle = verify_bundle(HERE, config['source_sha256'].get(str(HERE / 'SOURCE_MANIFEST.json'), ''))
    for relative, raw in self_bundle.items():
        install('sources/publication-worker-v04/' + relative, raw)
    qualification.update(schema='legal-ir-real-handoff-runtime-qualification@2', historical_failures_preserved=True,
        qualification_source_manifests={role: row['manifest_sha256'] for role, row in config['bundles'].items()},
        evidence_sha256={role: sha(raw) for role, raw in evidence.items()}, publication_configuration_sha256=sha(config_raw))
    body = pr_body(qualification)
    install('README.md', readme(qualification), frozen=False)
    install('QUALIFICATION.json', encoded(qualification), frozen=False)
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
        '-c', 'commit.gpgsign=false', 'commit', '-m', 'Qualify Legal IR native fixtures and real Git candidate observer')
    commit = git('rev-parse', 'HEAD')
    need(git('rev-parse', 'HEAD^') == BASE_HEAD and git('status', '--porcelain=v1', '--untracked-files=all') == '',
         'one clean documentation-only publication commit required')
    for path, raw in contents.items():
        need(git('cat-file', 'blob', commit + ':' + path, raw=True) == raw, 'committed publication blob differs')
    guard()
    git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
    need(git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + BRANCH).split()[0] == commit,
         'published GitHub branch commit differs')
    latest_pr = read_pr()
    need(latest_pr['body'] == original_pr['body'], 'PR body changed concurrently; preserve pushed commit and inspect')
    patch = attempt / 'pr292-body-patch.json'
    write_new(patch, encoded({'body': body}), mode=0o600)
    updated = strict_json(gh('api', '--method', 'PATCH', PR_ENDPOINT, '--input', str(patch)))
    need(updated.get('number') == 292 and updated.get('body') == body and updated.get('state') == 'open'
         and updated.get('draft') is True and updated.get('merged') is False,
         'actual open draft PR292 body update failed verification')
    guard()
    need(git('rev-parse', 'HEAD') == commit and git('status', '--porcelain=v1', '--untracked-files=all') == '',
         'published owned checkout changed during PR metadata update')
    result = {'schema': 'legal-ir-real-handoff-source-publication@1', 'accelerator_commit': commit,
        'pushed': True, 'branch': BRANCH, 'configuration_sha256': sha(config_raw),
        'staged_path_count': len(names), 'tests_rerun': False, 'historical_failures_preserved': True,
        'actual_full_native_v05_fixture_passed': True, 'actual_public_candidate_observer_fixture_passed': True,
        'controlled_reader_testcases': 60, 'ordinary_success_hook_controlled_testcases': qualification['ordinary_success_hook_controlled_testcases'],
        'pr_url': 'https://github.com/endomorphosis/ipfs_accelerate_py/pull/292', 'pr_body_updated': True,
        'pr_merged': False, 'official_github_credential_helper_used': True, 'credential_values_exported': False, **FALSE}
    write_new(attempt / 'publication.json', encoded(result), mode=0o600)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
