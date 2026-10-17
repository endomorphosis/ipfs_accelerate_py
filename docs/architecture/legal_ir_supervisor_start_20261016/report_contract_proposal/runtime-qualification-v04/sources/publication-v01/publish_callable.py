"""Publish one reviewed documentation generation to owned draft PR292.

This never applies the proposed runtime patches or enables the legal board.
The exact closed configuration is reviewed before the genuine owner launches.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from publication_helpers import (
    ACCELERATOR, BRANCH, PR_ENDPOINT, canonical, encoded, fsync_directory,
    gh, git, need, read, read_pr, safe_relative, sha, strict_json,
    verify_bundle, write_new,
)

HERE = Path(__file__).absolute().parent
PHASE = HERE.parent
RUNS = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs')
BASE_HEAD = '003a25b3814611edc2136c020d1ea689dee033fd'
PREFIX = 'docs/architecture/legal_ir_supervisor_start_20261016/report_contract_proposal'
PREVIOUS = ACCELERATOR / PREFIX / 'runtime-qualification-v03'
TARGET = ACCELERATOR / PREFIX / 'runtime-qualification-v04'
REMOTE = 'https://github.com/endomorphosis/ipfs_accelerate_py.git'
DOCS = {'README.md', 'QUALIFICATION.json', 'READINESS_TASKS.json', 'PR292_BODY.md'}
FIELDS = {'schema', 'base_head', 'branch', 'push', 'bundles', 'evidence', 'documents', 'source_sha256'}
DENIED_KEYS = {'consumer_key', 'native_consumer_key', 'private_key', 'api_key', 'access_token',
               'refresh_token', 'password', 'authorization', 'authentication_credentials'}

def no_private_keys(value):
    if type(value) is dict:
        normalized = {re.sub(r'[^a-z0-9]', '', str(key).casefold()) for key in value}
        denied = {re.sub(r'[^a-z0-9]', '', key) for key in DENIED_KEYS}
        need(not normalized & denied,
             'private credential or native capability field cannot be published')
        for item in value.values():
            no_private_keys(item)
    elif type(value) is list:
        for item in value:
            no_private_keys(item)

def pinned_document(row):
    need(type(row) is dict and set(row) == {'absolute_path', 'bytes', 'sha256'}, 'closed public file row required')
    path = canonical(row['absolute_path'])
    need(PHASE in path.parents or RUNS in path.parents, 'scoped actual source or public engineering file required')
    need(path.suffix in {'.json', '.jsonl', '.md', '.txt'}, 'explicit public document type required')
    raw = read(path)
    need(type(row['bytes']) is int and len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'reviewed public file changed')
    if path.suffix == '.json':
        no_private_keys(strict_json(raw))
    elif path.suffix == '.jsonl':
        records = [line for line in raw.splitlines() if line.strip()]
        need(0 < len(records) <= 4096 and all(len(line) <= 1024 * 1024 for line in records),
             'bounded selected public JSONL records required')
        for line in records:
            value = strict_json(line)
            need(type(value) is dict, 'public JSONL engineering record object required')
            no_private_keys(value)
    return raw

CHECK_FIELDS = {'name', 'producer', 'source_bundle', 'request_role', 'result_role', 'root_role',
                'receipt_role', 'passed', 'scope'}
PRODUCERS = {'generic_memory_controls', 'generic_capsule_preparation', 'generic_freezer_controls',
             'direct_public_wrapper', 'direct_dual_pipe', 'direct_freezer_git'}


def selected_path(config, role):
    need(type(role) is str and role in config['evidence'], 'selected exact evidence role required')
    return canonical(config['evidence'][role]['absolute_path'])


def selected_json(evidence, role):
    need(type(role) is str and role in evidence, 'selected JSON evidence role required')
    value = strict_json(evidence[role])
    need(type(value) is dict, 'engineering evidence object required')
    return value


def evidence_at(config, evidence, path):
    path = canonical(path)
    roles = [role for role in evidence if selected_path(config, role) == path]
    need(len(roles) == 1, 'exact unique selected public evidence path required')
    return selected_json(evidence, roles[0])


def join_request_source(config, bundles, request, role):
    need(type(role) is str and role in bundles, 'executed frozen source bundle required')
    root = canonical(config['bundles'][role]['absolute_path'])
    pins = request.get('source_sha256')
    need(type(pins) is dict, 'actual executed request source closure required')
    for relative, raw in bundles[role].items():
        need(pins.get(str(root / relative)) == sha(raw), 'actual request must pin all executed frozen source rows')
    manifest = strict_json(bundles[role]['SOURCE_MANIFEST.json'])
    for field in ('input_pins', 'input_sources'):
        rows = manifest.get(field, [])
        if type(rows) is dict:
            pairs = rows.items()
        else:
            need(type(rows) is list, 'closed frozen input pin collection required')
            pairs = [(row['path'], row['sha256']) for row in rows]
        for path, digest in pairs:
            need(pins.get(path) == digest, 'actual request must join every frozen executed input pin')
    return manifest, config['bundles'][role]['manifest_sha256']


def root_execution(root, passed):
    need(type(passed) is bool and type(root.get('actual_exec_session_id')) is int
         and root['actual_exec_session_id'] > 0 and type(root.get('actual_exec_exit_code')) is int
         and root['actual_exec_exit_code'] == (0 if passed else 1), 'actual observed execution outcome required')
    need(root.get('canonical_tasks_accepted') == 0 and root.get('acceptance_authority') is False,
         'engineering root observation cannot carry task acceptance')


def generic_closeout(config, evidence, check, root, request):
    need(type(check['receipt_role']) is str, 'actual generic receipt role required')
    receipt = selected_json(evidence, check['receipt_role'])
    receipt_path = selected_path(config, check['receipt_role'])
    attempt = canonical(request.get('attempt_directory'))
    terminal = root.get('actual_final_terminal', root.get('actual_native_terminal'))
    status = 'completed' if check['passed'] else 'failed'
    need(request.get('schema') == 'legal-ir-native-orchestration-request@1'
         and request.get('purpose') == 'controller_preparation' and RUNS in attempt.parents
         and receipt_path == attempt / 'receipt.json' and type(terminal) is dict
         and terminal.get('native_finalized') is True and terminal.get('status') == status
         and terminal.get('child_returncode') == root['actual_exec_exit_code']
         and canonical(terminal.get('receipt')) == receipt_path,
         'actual generic owner terminal and original attempt required')
    need(receipt.get('schema') == 'legal-ir-native-orchestration-receipt@1'
         and receipt.get('request_sha256') == sha(evidence[check['request_role']])
         and receipt.get('purpose') == request['purpose'] and receipt.get('status') == status
         and receipt.get('child_returncode') == terminal['child_returncode']
         and receipt.get('owned_group_quiescent') is True and receipt.get('completion_authority') is False,
         'exact actual receipt/request/quiescence join required')
    for name in ('disk_reservation_id', 'phase_lease_id', 'root_lease_id'):
        need(type(receipt.get(name)) is str and bool(receipt[name]), 'actual generic receipt identity required')
    return attempt


def released_owner(row):
    need(type(row) is dict and row.get('status') == 'released'
         and type(row.get('reservation_id')) is str and bool(row['reservation_id']), 'actual released disk owner required')
    if type(row.get('resource_lease')) is dict:
        native = row['resource_lease']
        need(native.get('released') is True and type(native.get('lease_id')) is str and bool(native['lease_id']),
             'actual released native owner lease required')
    else:
        need(row.get('native_lease_released') is True and type(row.get('native_lease_id')) is str
             and bool(row['native_lease_id']), 'actual compact native owner closeout required')


def released_root_phase(result):
    ids = []
    for name in ('actual_root_closeout', 'actual_phase_closeout'):
        row = result.get(name)
        need(type(row) is dict and row.get('released') is True and type(row.get('lease_id')) is str
             and bool(row['lease_id']), 'actual root and phase release required')
        ids.append(row['lease_id'])
    need(ids[0] != ids[1], 'distinct actual root and phase required')


def assess_check(config, evidence, bundles, check):
    need(type(check) is dict and set(check) == CHECK_FIELDS and check['producer'] in PRODUCERS
         and type(check['name']) is str and bool(check['name']) and type(check['scope']) is str and bool(check['scope'])
         and type(check['passed']) is bool, 'closed scoped actual qualification check required')
    result, root, request = (selected_json(evidence, check[field]) for field in ('result_role', 'root_role', 'request_role'))
    root_execution(root, check['passed'])
    need(result.get('passed') is check['passed'] and result.get('canonical_tasks_accepted') == 0
         and result.get('acceptance_authority') is False, 'actual inert result outcome required')
    manifest, digest = join_request_source(config, bundles, request, check['source_bundle'])
    producer = check['producer']
    if producer in {'generic_memory_controls', 'generic_freezer_controls'}:
        need(root.get('schema') == 'legal-ir-callable-generic-root-observation@1'
             and root.get('result_sha256') == sha(evidence[check['result_role']])
             and canonical(root.get('result_path')) == selected_path(config, check['result_role'])
             and root.get('request_sha256') == sha(evidence[check['request_role']])
             and canonical(root.get('request_path')) == selected_path(config, check['request_role'])
             and root.get('receipt_sha256') == sha(evidence[check['receipt_role']])
             and canonical(root.get('receipt_path')) == selected_path(config, check['receipt_role'])
             and root.get('native_finalized_observed') is True and root.get('observed_passed') is check['passed'],
             'exact generic root result/request/receipt correspondence required')
        attempt = generic_closeout(config, evidence, check, root, request)
        need(selected_path(config, check['result_role']).parent == attempt, 'generic result must remain in original attempt')
        if producer == 'generic_memory_controls':
            count = manifest.get('controlled_testcases')
            need(result.get('schema') == 'legal-ir-callable-memory-controls@1'
                 and type(count) is int and count > 0 and result.get('expected_tests') == result.get('tests_run') == count
                 and type(result.get('failures')) is list and type(result.get('errors')) is list
                 and type(result.get('skipped')) is int and result['skipped'] >= 0
                 and (not result['failures'] and not result['errors'] and result['skipped'] == 0) is check['passed']
                 and result.get('provider_calls') == 0 and result.get('provider_execution_qualified') is False
                 and result.get('production_hook_deployed') is False, 'actual scoped memory-control counts required')
        else:
            count = manifest.get('authored_control_cases')
            need(result.get('schema') in {'legal-ir-report-freezer-source-controls@1', 'legal-ir-report-freezer-source-controls@2'}
                 and type(count) is int and count > 0 and result.get('expected_controlled_testcases') == result.get('controlled_testcases') == count
                 and type(result.get('passed_testcases')) is int and 0 <= result['passed_testcases'] <= count
                 and all(type(result.get(k)) is int and result[k] >= 0 for k in ('failures', 'errors', 'skipped'))
                 and (result['passed_testcases'] == count and result['failures'] == result['errors'] == result['skipped'] == 0) is check['passed']
                 and result.get('returncode') == root['actual_exec_exit_code']
                 and result.get('provider_calls_performed') is False and result.get('public_report_freezer_qualified') is False,
                 'actual scoped freezer-control counts required')
            freezer = config['bundles'].get('report-freezer-v02')
            need(type(freezer) is dict and result.get('source_manifest_sha256') == freezer.get('manifest_sha256')
                 and request['source_sha256'].get(str(canonical(freezer['absolute_path']) / 'SOURCE_MANIFEST.json')) == freezer['manifest_sha256'],
                 'actual controls must join the unchanged executed freezer source generation')
            if result['schema'].endswith('@2'):
                need(result.get('corrected_control_source_manifest_sha256') == digest, 'actual corrected control manifest required')
    elif producer == 'generic_capsule_preparation':
        need(root.get('schema') == 'legal-ir-callable-capsule-root-observation@1'
             and root.get('preparation_result_sha256') == sha(evidence[check['result_role']])
             and root.get('source_request_sha256') == sha(evidence[check['request_role']])
             and root.get('source_manifest_sha256') == digest and root.get('original_native_preparation_finalized') is True,
             'exact actual capsule preparation joins required')
        attempt = generic_closeout(config, evidence, check, root, request)
        need(selected_path(config, check['result_role']) == attempt / 'capsule-preparation.json'
             and result.get('schema') == 'legal-ir-callable-capsule-preparation@1' and result.get('passed') is True
             and result.get('all_files_fsynced_readonly') is True and result.get('actual_public_wrapper_qualified') is False
             and result.get('provider_calls_performed') is False, 'actual readonly capsule preparation outcome required')
        capsule_root = canonical(result.get('capsule_root'))
        need(capsule_root == attempt / 'capsule', 'exact original prepared capsule root required')
        capsule = evidence_at(config, evidence, capsule_root / 'SOURCE_MANIFEST.json')
        roles = [role for role in evidence if selected_path(config, role) == capsule_root / 'SOURCE_MANIFEST.json']
        need(root.get('capsule_manifest_sha256') == result.get('capsule_manifest_sha256') == sha(evidence[roles[0]])
             and capsule.get('schema') == 'legal-ir-callable-handoff-capsule@1'
             and capsule.get('capsule_root') == str(capsule_root)
             and capsule.get('payload_file_count') == result.get('payload_file_count') == 2807
             and capsule.get('payload_bytes') == result.get('payload_bytes')
             and capsule.get('source_inventory_sha256') == result.get('source_inventory_sha256'),
             'actual capsule manifest/population/result correspondence required')
    else:
        need(check['receipt_role'] is None and root.get('schema') == 'legal-ir-callable-direct-root-observation@1'
             and root.get('source_scope') == check['source_bundle'] and root.get('source_manifest_sha256') == digest
             and root.get('source_request_sha256') == sha(evidence[check['request_role']])
             and canonical(root.get('source_request_path')) == selected_path(config, check['request_role'])
             and root.get('terminal_sha256') == sha(evidence[check['result_role']])
             and canonical(root.get('terminal_path')) == selected_path(config, check['result_role'])
             and root.get('actual_final_terminal') == result, 'exact direct source/request/final-terminal joins required')
        if producer == 'direct_public_wrapper':
            need(request.get('schema') == 'legal-ir-callable-public-wrapper-prestart-request@1'
                 and request.get('source_manifest_sha256') == result.get('source_manifest_sha256') == digest
                 and result.get('source_request_sha256') == sha(evidence[check['request_role']])
                 and result.get('schema') == 'legal-ir-callable-public-wrapper-prestart-terminal@1'
                 and result.get('root_and_phase_rows_drained') is True
                 and result.get('provider_calls_performed') is False and result.get('G_run_dispatch_performed') is False,
                 'actual known-unstarted public-wrapper scope required')
            owners = result.get('actual_owner_finalization')
            need(type(owners) is dict and set(owners) == {'primary', 'watchdog'}, 'actual wrapper owner closeouts required')
            for owner in owners.values(): released_owner(owner)
            cases = result.get('cases')
            need(type(cases) is list and len(cases) == manifest.get('controlled_cases') == result.get('controlled_cases') == 5
                 and all(type(row.get('passed')) is bool and row.get('actual_original_consumer_rows_drained') is True for row in cases)
                 and all(row['passed'] for row in cases) is check['passed'], 'actual wrapper case/consumer outcomes required')
        elif producer == 'direct_freezer_git':
            need(request.get('schema') == 'legal-ir-report-freezer-git-component-request@1'
                 and request.get('qualifier_source_manifest_sha256') == digest
                 and result.get('schema') == 'legal-ir-report-freezer-git-component@1'
                 and result.get('source_request_sha256') == sha(evidence[check['request_role']])
                 and result.get('freezer_source_manifest_sha256') == request.get('freezer_source_manifest_sha256')
                 and result.get('public_report_freezer_qualified') is False and result.get('provider_calls_performed') is False,
                 'actual synthetic Git component scope required')
            released_owner(result.get('actual_owner_finalization')); released_root_phase(result)
            cases = result.get('cases')
            if cases is None:
                need(check['passed'] is False and type(result.get('failure_reason')) is str and result['failure_reason'],
                     'failed component may not invent successful cases')
            else:
                need(type(cases) is list and len(cases) == result.get('controlled_cases') == manifest.get('controlled_component_cases') == 11
                     and all(type(row.get('passed')) is bool for row in cases)
                     and all(row['passed'] for row in cases) is check['passed'], 'actual component case outcomes required')
        else:
            need(request.get('schema') == 'legal-ir-callable-dual-pipe-mechanical-request@1'
                 and request.get('qualifier_manifest_sha256') == result.get('qualifier_manifest_sha256') == digest
                 and result.get('request_sha256') == sha(evidence[check['request_role']])
                 and result.get('schema') == 'legal-ir-callable-dual-pipe-native-mechanics@1'
                 and result.get('provider_calls_performed') is False and result.get('actual_provider_execution_qualified') is False
                 and result.get('finish_or_success_callback_called') is False
                 and result.get('all_case_owner_contexts_exited_after_independent_cleanup') is True,
                 'actual nonmodel dual-pipe scope required')
            released_root_phase(result)
            cases = result.get('cases')
            need(type(cases) is list and len(cases) == manifest.get('controlled_mechanical_cases') == 2
                 and [row.get('fixture_case') for row in cases] == request.get('cases')
                 and all(type(row.get('passed')) is bool for row in cases)
                 and all(row['passed'] for row in cases) is check['passed'], 'actual dual-pipe case outcomes required')
            for case in cases:
                need(type(case.get('cleanup')) is dict and case['cleanup'].get('cleanup_known') is True,
                     'actual independent cleanup known before release required')
                owners = case.get('actual_owner_finalization')
                need(type(owners) is dict and set(owners) == {'primary', 'watchdog'}, 'actual dual-pipe owner closeouts required')
                for owner in owners.values(): released_owner(owner)


def assess_preflight(config, evidence, bundles, row):
    need(type(row) is dict and set(row) == {'name', 'request_role', 'root_role', 'passed', 'scope', 'source_bundle', 'native_admission_reached'}
         and row['passed'] is False and row['native_admission_reached'] is False, 'closed failed pre-native history required')
    root, request = selected_json(evidence, row['root_role']), selected_json(evidence, row['request_role'])
    _manifest, digest = join_request_source(config, bundles, request, row['source_bundle'])
    need(root.get('schema') == 'legal-ir-callable-preflight-root-observation@1'
         and root.get('actual_exec_exit_code') == 1 and root.get('source_request_sha256') == sha(evidence[row['request_role']])
         and canonical(root.get('source_request_path')) == selected_path(config, row['request_role'])
         and root.get('source_guard_failed_before_native_import') is True and root.get('native_admission_reached') is False
         and root.get('provider_calls') == 0 and root.get('canonical_tasks_accepted') == 0 and root.get('acceptance_authority') is False
         and request.get('qualifier_source_manifest_sha256') == digest,
         'actual failed preflight must not be reclassified as native qualification')
    tool = root.get('actual_tool_result')
    need(type(tool) is dict and tool.get('exit_code') == 1 and type(tool.get('output')) is str
         and 0 < len(tool['output'].encode()) <= 65536, 'actual bounded preflight tool failure required')


def publication_plan(raw_config, bundles, evidence, documents, previous):
    contents, frozen = {}, set()
    def add(relative, raw, immutable=True):
        relative = safe_relative(relative)
        name = (TARGET / relative).relative_to(ACCELERATOR).as_posix()
        need(name not in contents and type(raw) is bytes and len(raw) <= 8 * 1024 * 1024,
             'unique bounded planned publication file required')
        contents[name] = raw
        if immutable: frozen.add(name)
    for role, files in bundles.items():
        for relative, raw in files.items(): add('sources/' + role + '/' + relative, raw)
    for name, raw in evidence.items(): add('evidence/' + name, raw)
    for name, raw in documents.items(): add(name, raw, False)
    add('publication-config.json', raw_config)
    inventory = {'schema': 'legal-ir-callable-publication-inventory@1', 'base_head': BASE_HEAD,
        'configuration_sha256': sha(raw_config), 'previous_inventory_sha256': previous,
        'files': [{'path': name, 'bytes': len(raw), 'sha256': sha(raw)} for name, raw in sorted(contents.items())],
        'production_hook_deployed': False, 'canonical_tasks_accepted': 0, 'acceptance_authority': False}
    add('PUBLICATION_INVENTORY.json', encoded(inventory), False)
    names = sorted(contents)
    need(len(names) <= 256 and sum(len(raw) for raw in contents.values()) <= 64 * 1024 * 1024,
         'bounded complete publication required before first checkout write')
    paths = {Path(name) for name in names}
    need(not any(parent in paths for path in paths for parent in path.parents), 'planned file/directory destination collision')
    return contents, frozen

def load_configuration(path, expected):
    raw = read(path, 1024 * 1024)
    need(sha(raw) == expected, 'reviewed configuration differs')
    config = strict_json(raw)
    need(type(config) is dict and set(config) == FIELDS
         and config['schema'] == 'legal-ir-callable-publication-config@1'
         and config['base_head'] == BASE_HEAD and config['branch'] == BRANCH and config['push'] is True,
         'exact owned additive documentation publication required')
    pins = config['source_sha256']
    need(type(pins) is dict and 1 <= len(pins) <= 256, 'bounded actual protected source closure required')
    required = {str(HERE / name) for name in ('publish_callable.py', 'publication_helpers.py', 'SOURCE_MANIFEST.json')}
    required.update(('/usr/bin/python3.12', '/usr/bin/git', '/usr/bin/gh'))
    need(required <= set(pins), 'publication worker and actual executable pins required')
    for spelling, digest in pins.items():
        need(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'exact source pin required')
        need(sha(read(canonical(spelling), 256 * 1024 * 1024)) == digest, 'actual publication source changed')
    bundles = config['bundles']
    need(type(bundles) is dict and 1 <= len(bundles) <= 20, 'finite reviewed source bundles required')
    sources = {}
    for role, row in bundles.items():
        safe_relative(role)
        need('/' not in role and type(row) is dict and set(row) == {'absolute_path', 'manifest_sha256'}, 'closed source bundle row required')
        root = canonical(row['absolute_path'])
        need(root.parent == PHASE and root.name == role, 'exact campaign source bundle required')
        sources[role] = verify_bundle(root, row['manifest_sha256'])
    need(bundles.get('publication-v01', {}).get('manifest_sha256') == pins[str(HERE / 'SOURCE_MANIFEST.json')],
         'publisher self-source must be archived with exact reviewed manifest')
    rows = config['evidence']
    need(type(rows) is dict and 1 <= len(rows) <= 64, 'finite selected public evidence required')
    evidence = {}
    for role, row in rows.items():
        safe_relative(role)
        need('/' not in role and role.endswith(('.json', '.jsonl')), 'explicit evidence filename required')
        evidence[role] = pinned_document(row)
    docs = config['documents']
    need(type(docs) is dict and set(docs) == DOCS, 'exact reviewed generated documentation set required')
    documents = {name: pinned_document(row) for name, row in docs.items()}
    qualification = strict_json(documents['QUALIFICATION.json'])
    need(qualification.get('schema') == 'legal-ir-callable-runtime-qualification@1'
         and qualification.get('canonical_tasks_accepted') == 0 and qualification.get('provider_calls') == 0
         and qualification.get('production_hook_deployed') is False
         and qualification.get('real_provider_path_qualified') is False
         and qualification.get('acceptance_authority') is False,
         'engineering-only scoped qualification required')
    checks = qualification.get('checks')
    need(type(checks) is list and 1 <= len(checks) <= 16, 'finite actual qualification checks required')
    names = [check.get('name') for check in checks if type(check) is dict]
    need(len(names) == len(checks) == len(set(names)), 'unique actual qualification check names required')
    for check in checks:
        assess_check(config, evidence, sources, check)
    preflights = qualification.get('historical_preflight_failures', [])
    need(type(preflights) is list and len(preflights) <= 8, 'finite explicit pre-native failure history required')
    for row in preflights:
        assess_preflight(config, evidence, sources, row)
    return config, raw, sources, evidence, documents

def previous_guard():
    inventory_raw = read(PREVIOUS / 'PUBLICATION_INVENTORY.json')
    inventory = strict_json(inventory_raw)
    need(inventory.get('base_head') == '06eff5086b8653f0a58f5013cc781482693886c5', 'exact previous publication generation required')
    for row in inventory['files']:
        path = ACCELERATOR / safe_relative(row['path'])
        need(PREVIOUS in path.parents and sha(read(path)) == row['sha256'] and path.stat().st_size == row['bytes'],
             'previous frozen publication changed')
    return sha(inventory_raw)

def main():
    need(len(sys.argv) == 3 and re.fullmatch('[0-9a-f]{64}', sys.argv[2]), 'reviewed exact config and digest required')
    attempt = canonical(os.environ['TMPDIR'])
    info = attempt.lstat()
    need(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0 and attempt.is_dir(), 'genuine private publisher attempt required')
    config_path = canonical(sys.argv[1])
    config, raw_config, bundles, evidence, documents = load_configuration(config_path, sys.argv[2])
    previous = previous_guard()
    contents, frozen = publication_plan(raw_config, bundles, evidence, documents, previous)
    def guard():
        need(load_configuration(config_path, sys.argv[2]) == (config, raw_config, bundles, evidence, documents)
             and previous_guard() == previous, 'reviewed publication closure or prior generation changed')
    guard()
    need(git('rev-parse', 'HEAD') == BASE_HEAD and git('branch', '--show-current') == BRANCH
         and git('remote', 'get-url', 'origin') == REMOTE
         and git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + BRANCH).split()[0] == BASE_HEAD,
         'exact owned local and remote branch required')
    need(git('status', '--porcelain=v1', '--untracked-files=all') == ''
         and git('rev-parse', '--path-format=absolute', '--git-dir') == str(ACCELERATOR / '.git')
         and git('rev-parse', '--path-format=absolute', '--git-common-dir') == str(ACCELERATOR / '.git'),
         'clean self-contained precharged owned checkout required')
    need(not TARGET.exists(), 'fresh immutable publication generation required')
    original_pr = read_pr()
    write_new(attempt / 'pr292-before.json', encoded({'number': 292, 'body': original_pr['body'], 'updated_at': original_pr.get('updated_at')}), 0o600)
    TARGET.mkdir(mode=0o700)
    for name, raw in contents.items():
        path = ACCELERATOR / safe_relative(name)
        need(TARGET in path.parents, 'exact additive publication destination required')
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_new(path, raw)
    names = sorted(contents)
    guard()
    need(git('rev-parse', 'HEAD') == BASE_HEAD and git('diff', '--cached', '--name-only') == '', 'base or index changed')
    for name, raw in contents.items():
        need(read(ACCELERATOR / name) == raw, 'installed publication bytes differ')
    git('add', '--sparse', '-f', '--', *names)
    need({p.decode() for p in git('diff', '--cached', '--name-only', '-z', raw=True).split(b'\0') if p} == set(names), 'exact staged set required')
    for name, raw in contents.items():
        need(git('cat-file', 'blob', ':' + name, raw=True) == raw, 'exact staged bytes required')
    git('diff', '--cached', '--check', '--', *sorted(set(names) - frozen))
    guard()
    git('-c', 'user.name=Legal IR Supervisor', '-c', 'user.email=legal-ir-supervisor@localhost', '-c', 'commit.gpgsign=false',
        'commit', '-m', 'Archive callable Legal IR runner and report freezer qualifications')
    commit = git('rev-parse', 'HEAD')
    need(git('rev-parse', 'HEAD^') == BASE_HEAD and git('status', '--porcelain=v1', '--untracked-files=all') == '', 'one clean original-parent commit required')
    need({p.decode() for p in git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', BASE_HEAD, commit, raw=True).split(b'\0') if p} == set(names), 'exact actual committed paths required')
    for name, raw in contents.items():
        need(git('cat-file', 'blob', commit + ':' + name, raw=True) == raw, 'exact committed bytes required')
    guard()
    git('push', 'origin', 'HEAD:refs/heads/' + BRANCH)
    need(git('ls-remote', '--exit-code', 'origin', 'refs/heads/' + BRANCH).split()[0] == commit, 'actual published head differs')
    need(read_pr()['body'] == original_pr['body'], 'PR body changed concurrently; preserve pushed commit')
    # Fresh comparison followed by PATCH is not atomic compare-and-swap.
    body = documents['PR292_BODY.md'].decode()
    patch = attempt / 'pr292-body-patch.json'
    write_new(patch, encoded({'body': body}), 0o600)
    updated = strict_json(gh('api', '--method', 'PATCH', PR_ENDPOINT, '--input', str(patch)))
    need(updated.get('body') == body and updated.get('draft') is True and updated.get('state') == 'open' and updated.get('merged') is False,
         'actual draft PR body update required')
    guard()
    need(git('rev-parse', 'HEAD') == commit and git('status', '--porcelain=v1', '--untracked-files=all') == '', 'owned publication changed')
    result = {'schema': 'legal-ir-callable-publication-result@1', 'passed': True, 'accelerator_commit': commit,
              'previous_publication_commit': BASE_HEAD, 'pushed': True, 'staged_path_count': len(names),
              'configuration_sha256': sha(raw_config), 'historical_failures_preserved': True,
              'pr_url': 'https://github.com/endomorphosis/ipfs_accelerate_py/pull/292', 'pr_body_updated': True,
              'pr_merged': False, 'production_hook_deployed': False, 'provider_calls': 0,
              'canonical_tasks_accepted': 0, 'acceptance_authority': False}
    write_new(attempt / 'publication.json', encoded(result), 0o600)
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
