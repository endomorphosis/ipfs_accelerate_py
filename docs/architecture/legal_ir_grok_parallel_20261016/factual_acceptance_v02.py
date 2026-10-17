#!/usr/bin/env python3
"""Independent factual checks with distinct database and Portal identities.

Standard library only. No product import, provider call, subprocess, database,
signature or native acceptance write occurs here. The existing native owner
freezes the candidate, supplies exact attempt/source bindings, runs this in its
separate validation job, and may record the checked descriptor afterward.
The descriptor is factual evidence only, never legal/gold/training/Lean authority.

The protected controller prefix contains acceptance-contract.json, inputs.json
and the three pinned source metadata files used by planning_worker.py.
--check-all checks that frozen population/projection, not a live task database.
--check-report additionally requires the LEGAL_IR_FACTUAL_* native environment
bindings named in NATIVE_FIELDS below, plus a root-owned --receipt-root.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

TASK = 'LIR-9001'
NAMESPACE = 'legal-ir-span-graph-lean-refinement-20261016'
EVIDENCE_KIND = 'legal_ir_handoff_factual_acceptance_v1'
ROLES = ('task_manifest', 'source_bindings', 'retained_source_audit')
AUTHORITY = {key: False for key in (
    'completion_authority', 'execution_authority', 'proof_authority',
    'publication_authority', 'training_authority', 'gold_admission_authority',
    'teacher_admission_authority', 'legal_meaning_review_authority',
    'task_status_mutation_authority', 'automatic_retry_authority')}
FALSE_EFFECTS = (
    'fresh_task_database_or_other_agent_state_observed',
    'corpus_payload_or_model_weights_read', 'statutory_semantic_review_performed',
    'training_performed', 'Lean_executed', 'task_completed')
FINDINGS = {
    'reuse_retained_source_closure': ('retained_source_audit', '/retained_source'),
    'preserve_embedding_dispositions': ('retained_source_audit', '/retained_source/original_embedding_dispositions'),
    'preserve_frozen_splits': ('retained_source_audit', '/retained_source/original_split_counts'),
    'preserve_three_exposed_companions': ('retained_source_audit', '/selected_companions'),
    'keep_semantic_authority_unadmitted': ('retained_source_audit', '/authority'),
    'preserve_authored_task_dag': ('task_manifest', '/tasks'),
    'reconcile_source_snapshot_limits': ('source_bindings', '/limitations'),
}
GAPS = {
    'source_span_correspondence_open', 'statutory_review_open',
    'training_eligibility_open', 'lean_admission_open',
    'current_task_authority_reconciliation_open', 'representation_compatibility_open',
}
COLLISIONS = {
    'current_task_owner_state_not_observed', 'parallel_agent_state_not_observed',
    'related_lac_scope_requires_reconciliation',
}
NATIVE_FIELDS = {
    'task_cid': 'LEGAL_IR_FACTUAL_TASK_CID',
    'portal_task_cid': 'LEGAL_IR_FACTUAL_PORTAL_TASK_CID',
    'database_attempt_number': 'LEGAL_IR_FACTUAL_DATABASE_ATTEMPT_NUMBER',
    'task_revision': 'LEGAL_IR_FACTUAL_TASK_REVISION',
    'attempt': 'LEGAL_IR_FACTUAL_ATTEMPT',
    'native_attempt_id': 'LEGAL_IR_FACTUAL_NATIVE_ATTEMPT_ID',
    'report_sha256': 'LEGAL_IR_FACTUAL_REPORT_SHA256',
    'invocation_sha256': 'LEGAL_IR_FACTUAL_INVOCATION_SHA256',
    'native_prompt_sha256': 'LEGAL_IR_FACTUAL_NATIVE_PROMPT_SHA256',
}
REPORT_FIELDS = {
    'schema', 'task_id', 'task_cid', 'attempt', 'status', 'candidate',
    'provider', 'model', 'provider_purpose', 'native_prompt_sha256',
    'input_commitments', 'invocation_receipt', 'native_prompt_artifact', 'preserved_task_edges',
    'legacy_parent_crosswalk', 'owner_annex', 'model_proposal',
    'gold_mask', 'training_mask', *FALSE_EFFECTS, *AUTHORITY,
}
INVOCATION_FIELDS = {
    'schema', 'status', 'task_id', 'task_cid', 'attempt', 'adapter_invocation_id',
    'native_prompt_sha256', 'native_prompt_bytes', 'model_response_sha256',
    'model_response_bytes', 'controller_manifest_sha256', 'input_commitments',
    'router_invocation', 'input_custody_rechecked', 'native_prompt_artifact', *AUTHORITY,
}


class FactualRefusal(ValueError):
    pass


def need(value, reason):
    if not value:
        raise FactualRefusal(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def bad_constant(value):
        raise FactualRefusal('nonfinite_json_number')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                           parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FactualRefusal('invalid_json') from error
    # Bound recursion/containers after the byte bound; Python's parser itself
    # also refuses excessive recursion. Booleans remain distinct from numbers.
    def depth(item, level=0):
        need(level <= 32, 'json_depth_exceeded')
        if type(item) is dict:
            need(len(item) <= 2048, 'json_object_bound_exceeded')
            for child in item.values():
                depth(child, level + 1)
        elif type(item) is list:
            need(len(item) <= 2048, 'json_array_bound_exceeded')
            for child in item:
                depth(child, level + 1)
    depth(value)
    return value


def _witness(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_checked(path, *, maximum=1048576, pin=None, allow_owner_group_write=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path,
         'exact_regular_input_path_required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        source_mode_ok = (not before.st_mode & 0o022 or (
            allow_owner_group_write and before.st_uid == os.geteuid()
            and before.st_gid == os.getegid() and not before.st_mode & 0o002))
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_uid in (0, os.geteuid()) and source_mode_ok,
             'input_custody_differs')
        need(0 < before.st_size <= maximum, 'input_size_bound_exceeded')
        raw = bytearray()
        while len(raw) <= maximum:
            block = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        need(len(raw) == before.st_size and len(raw) <= maximum,
             'input_size_changed')
        need(_witness(before) == _witness(os.fstat(fd)) == _witness(path.lstat()),
             'input_changed_during_read')
        result = bytes(raw)
        actual = {'bytes': len(result), 'sha256': sha(result)}
        if pin is not None:
            need(type(pin) is dict and set(pin) >= {'bytes', 'sha256'},
                 'file_pin_shape_differs')
            need(type(pin['bytes']) is int and actual['bytes'] == pin['bytes']
                 and actual['sha256'] == pin['sha256'], 'frozen_file_pin_differs')
        return result, actual
    finally:
        os.close(fd)


def _name(name):
    need(type(name) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,96}\.json', name),
         'unsafe_metadata_name')
    return name


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def owner_annex(authoring):
    tasks, goals, parents = (authoring.get(key) for key in
                            ('tasks', 'goal_hierarchy', 'legacy_parent_crosswalk'))
    need(type(tasks) is list and len(tasks) == 64, 'task_population_differs')
    need(type(goals) is list and len(goals) == 34, 'goal_population_differs')
    need(type(parents) is list and len(parents) == 24, 'parent_population_differs')
    task_ids = [row['task_id'] for row in tasks]
    goal_ids = [row['goal_id'] for row in goals]
    need(len(set(task_ids)) == 64 and len(set(goal_ids)) == 34,
         'duplicate_task_or_goal')
    by_id = {row['task_id']: row for row in tasks}
    edges = []
    for row in tasks:
        deps = row['depends_on']
        need(type(deps) is list and len(set(deps)) == len(deps)
             and all(value in by_id and value != row['task_id'] for value in deps),
             'dependency_population_differs')
        need(row['goal_id'] in goal_ids, 'task_goal_missing')
        edges.append({'task_id': row['task_id'], 'depends_on': list(deps)})
    need(sum(len(row['depends_on']) for row in tasks) == 115, 'dependency_count_differs')
    visited, visiting = set(), set()
    def visit(alias):
        need(alias not in visiting, 'dependency_cycle')
        if alias in visited:
            return
        visiting.add(alias)
        for dependency in by_id[alias]['depends_on']:
            visit(dependency)
        visiting.remove(alias)
        visited.add(alias)
    for alias in by_id:
        visit(alias)
    need(by_id[TASK]['depends_on'] == [], 'bootstrap_dependency_changed')
    parent_ids = [row['id'] for row in parents]
    need(set(parent_ids) == {f'LIR-{index:02}' for index in range(1, 25)},
         'original_parent_ids_changed')
    for parent in parents:
        need(parent['related_lac_equivalence'] == 'unreviewed_related_scope_only',
             'lac_scope_promoted_to_equivalence')
        need(all(child in by_id and by_id[child]['legacy_parent_task'] == parent['id']
                 for child in parent['subtasks']), 'parent_crosswalk_changed')
    return {'attribution': 'owner_derived_from_frozen_task_manifest',
            'task_ids': task_ids, 'goal_ids': goal_ids,
            'legacy_parent_crosswalk': copy.deepcopy(parents),
            'dependency_edges': edges, 'task_count': 64, 'goal_count': 34,
            'legacy_parent_count': 24, 'dependency_count': 115}


def _verify_source_bindings(document):
    need(type(document) is dict, 'source_bindings_shape_differs')
    limitations = document.get('limitations')
    need(type(limitations) is list and 1 <= len(limitations) <= 64
         and all(type(item) is str and 0 < len(item.encode()) <= 8192 for item in limitations),
         'source_snapshot_limits_missing')
    # Repository metadata remains attributed snapshots. Selected source bytes
    # can be independently rehashed without importing or executing any product.
    verified = []
    repositories = document.get('repositories', [])
    need(type(repositories) is list and len(repositories) <= 4,
         'repository_metadata_population_exceeded')
    for repository in repositories:
        need(type(repository) is dict, 'repository_metadata_shape_differs')
        if repository.get('status') == 'unavailable':
            need(repository.get('head') is None and repository.get('tree') is None,
                 'unavailable_source_identity_invented')
            continue
        need(repository.get('status') == 'observed_metadata', 'unsupported_repository_status')
        need(all(type(repository.get(key)) is str
                 and re.fullmatch(r'[0-9a-f]{40}', repository[key])
                 for key in ('head', 'tree')), 'repository_identity_shape_differs')
        root = Path(repository['checkout_path'])
        need(root.is_absolute() and root.resolve(strict=True) == root,
             'repository_source_root_differs')
        files = repository.get('source_files', [])
        need(type(files) is list and 1 <= len(files) <= 256,
             'selected_source_population_differs')
        for pin in files:
            need(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'},
                 'selected_source_pin_shape_differs')
            name = Path(pin['path'])
            need(not name.is_absolute() and '..' not in name.parts
                 and name.suffix in {'.py', '.md', '.json'}, 'source_only_path_required')
            path = root / name
            # Existing shared owner source files may be group-writable. This
            # narrowly disclosed source-byte read permits only our own UID/GID,
            # never world write; original and closing inode/hash checks remain.
            raw, actual = read_checked(path, maximum=8388608, pin=pin,
                                       allow_owner_group_write=True)
            verified.append({'path': str(path), **actual,
                             'allow_owner_group_write': True})
    return verified


def _audit_predicates(audit):
    need(audit.get('schema') == 'legal-source-corpus-reuse-independent-native-audit/v1'
         and audit.get('attempt') == '04' and audit.get('audit_status') == 'pass'
         and audit.get('qualification_passed') is True,
         'successful_retained_source_audit_required')
    source = audit['retained_source']
    need(source['physical_rows'] == 62931 and source['verified_package_files'] == 103
         and source['verified_package_bytes'] == 263671066
         and source['declared_corpus_shards'] == 16
         and source['ordered_row_digest'] == '9ba5939bc2eab013dfbcb115b66c351e3d03aa3935ccf43346a7b4d0cab33db3',
         'retained_source_population_differs')
    need(source['original_embedding_dispositions'] == {
        'embedded': 51, 'token_limit_exceeded': 13, 'unattempted': 62867},
        'retained_embedding_dispositions_changed')
    need(source['original_split_counts'] == {
        'train': 50282, 'validation': 6358, 'canary': 3147, 'holdout': 3144},
        'retained_split_dispositions_changed')
    need(source['source_export_rebuilt'] is False
         and source['source_inventory_rebuilt'] is False
         and source['database_materialized'] is False
         and source['not_relabelled_as_current_hub_head'] is True,
         'retained_source_provenance_changed')
    need(type(audit['authority']) is dict and audit['authority']
         and all(value is False for value in audit['authority'].values()),
         'semantic_or_training_authority_promoted')
    companions = audit['selected_companions']
    need(companions['physical_rows'] == 3 and companions['exposure_status'] == 'exposed'
         and type(companions['gold_mask']) is int and companions['gold_mask'] == 0
         and type(companions['training_mask']) is int and companions['training_mask'] == 0
         and companions['normative_target_selected'] is False
         and companions['official_HTML_or_USLM_correspondence_claimed'] is False,
         'exposed_companions_promoted')


def load_capsule(prefix):
    prefix = Path(prefix).resolve(strict=True)
    contract_raw, contract_pin = read_checked(prefix / 'acceptance-contract.json', maximum=65536)
    contract = strict_json(contract_raw)
    need(type(contract) is dict and set(contract) == {
        'schema', 'task_id', 'board_namespace', 'inputs_manifest', 'selected_task_cid',
        'execution_packet_revision', 'evidence_kind', 'expected_route', 'initial_task_statuses'},
        'acceptance_contract_shape_differs')
    need(contract['schema'] == 'legal-ir-source-factual-acceptance-contract/v1'
         and contract['task_id'] == TASK and contract['board_namespace'] == NAMESPACE,
         'acceptance_contract_identity_differs')
    need(contract['evidence_kind'] == EVIDENCE_KIND, 'acceptance_kind_differs')
    need(type(contract['selected_task_cid']) is str
         and 0 < len(contract['selected_task_cid']) <= 512, 'task_cid_missing')
    need(type(contract['execution_packet_revision']) is int
         and contract['execution_packet_revision'] >= 2, 'actual_execution_revision_required')
    need(contract['expected_route'] == {
        'provider': 'grok_cli', 'model': 'grok-4.7', 'purpose': 'planning'}, 'route_differs')
    input_pin = contract['inputs_manifest']
    need(type(input_pin) is dict and set(input_pin) == {'name', 'bytes', 'sha256'}
         and input_pin['name'] == 'inputs.json', 'inputs_manifest_pin_differs')
    input_raw, actual_input_pin = read_checked(prefix / 'inputs.json', maximum=16384, pin=input_pin)
    manifest = strict_json(input_raw)
    need(type(manifest) is dict and set(manifest) == {'schema', 'task_id', 'files', 'router_runtime'}
         and manifest['schema'] == 'legal-ir-grok-planning-inputs/v1'
         and manifest['task_id'] == TASK and set(manifest['files']) == set(ROLES),
         'planning_inputs_schema_differs')
    docs, files, inode_ids = {}, [], set()
    for role in ROLES:
        pin = manifest['files'][role]
        need(type(pin) is dict and set(pin) == {'name', 'bytes', 'sha256'}, 'role_pin_shape_differs')
        path = prefix / _name(pin['name'])
        need(path.name not in {'inputs.json', 'acceptance-contract.json'}, 'metadata_role_alias')
        raw, actual = read_checked(path, pin=pin)
        info = path.lstat()
        need((info.st_dev, info.st_ino) not in inode_ids, 'metadata_roles_alias')
        inode_ids.add((info.st_dev, info.st_ino))
        docs[role] = strict_json(raw)
        files.append({'path': str(path), **actual})
    runtime = manifest['router_runtime']
    need(type(runtime) is dict and set(runtime) == {'path', 'bytes', 'sha256'},
         'router_runtime_pin_shape_differs')
    runtime_raw, runtime_actual = read_checked(Path(runtime['path']), maximum=2097152,
                                               pin=runtime, allow_owner_group_write=True)
    need(Path(runtime['path']).name == 'router_implementation_runner.py', 'router_owner_differs')
    files.append({'path': runtime['path'], **runtime_actual,
                  'allow_owner_group_write': True})
    authoring = docs['task_manifest']
    need(authoring.get('schema') == 'legal-ir-supervisor-refinement/v1'
         and authoring.get('board_namespace') == NAMESPACE, 'authoring_identity_differs')
    annex = owner_annex(authoring)
    statuses = contract['initial_task_statuses']
    need(type(statuses) is dict and set(statuses) == set(annex['task_ids']),
         'frozen_execution_projection_population_differs')
    need(statuses[TASK] == 'todo'
         and all(status == 'blocked' for alias, status in statuses.items() if alias != TASK),
         'frozen_execution_frontier_broadened')
    _audit_predicates(docs['retained_source_audit'])
    files.extend(_verify_source_bindings(docs['source_bindings']))
    files.extend([{'path': str(prefix / 'acceptance-contract.json'), **contract_pin},
                  {'path': str(prefix / 'inputs.json'), **actual_input_pin}])
    return {'prefix': str(prefix), 'contract': contract, 'contract_pin': contract_pin,
            'manifest': manifest, 'manifest_pin': actual_input_pin, 'documents': docs,
            'annex': annex, 'files': files}


def recheck_capsule(capsule):
    for pin in capsule['files']:
        read_checked(Path(pin['path']), maximum=8388608, pin=pin,
                     allow_owner_group_write=pin.get('allow_owner_group_write', False))


def native_bindings(environment):
    result = {key: environment.get(name) for key, name in NATIVE_FIELDS.items()}
    for key in ('task_cid', 'portal_task_cid', 'native_attempt_id'):
        need(type(result[key]) is str and 0 < len(result[key]) <= 512
             and not any(ord(c) < 32 for c in result[key]), 'native_identity_unavailable')
    for key in ('task_revision', 'attempt', 'database_attempt_number'):
        need(type(result[key]) is str and re.fullmatch(r'[1-9][0-9]{0,8}', result[key]),
             'native_revision_or_attempt_unavailable')
        result[key] = int(result[key])
    need(result['portal_task_cid'] != result['task_cid'], 'database_and_portal_identities_aliased')
    need(all(_sha(result[key]) for key in (
        'report_sha256', 'invocation_sha256', 'native_prompt_sha256')),
        'actual_native_freeze_bindings_unavailable')
    return result


def _pointer(document, pointer):
    need(type(pointer) is str and 0 < len(pointer) <= 1024 and pointer.startswith('/'),
         'evidence_pointer_missing')
    value = document
    for raw in pointer[1:].split('/'):
        need(re.search(r'~(?![01])', raw) is None, 'evidence_pointer_escape_invalid')
        part = raw.replace('~1', '/').replace('~0', '~')
        if type(value) is dict:
            value = value[part]
        elif type(value) is list and re.fullmatch(r'0|[1-9][0-9]*', part):
            value = value[int(part)]
        else:
            raise FactualRefusal('evidence_pointer_does_not_resolve')
    return value


def validate_report_data(report, capsule, native, invocation):
    """Pure structural/factual part, also used by declared negative controls."""
    need(type(report) is dict and set(report) == REPORT_FIELDS, 'candidate_report_fields_differ')
    need(report['schema'] == 'legal-ir-source-handoff-reconciliation-candidate/v2'
         and report['task_id'] == TASK and report['task_cid'] == native['portal_task_cid']
         and type(report['attempt']) is int and report['attempt'] == native['attempt'],
         'candidate_native_identity_differs')
    need(native['task_cid'] == capsule['contract']['selected_task_cid']
         and native['task_revision'] == capsule['contract']['execution_packet_revision'],
         'native_execution_packet_binding_differs')
    need(native['portal_task_cid'] != native['task_cid'], 'database_and_portal_identities_aliased')
    need(report['status'] == 'candidate_pending_independent_acceptance'
         and report['candidate'] is True, 'candidate_claims_accepted_status')
    need(report['provider'] == 'grok_cli' and report['model'] == 'grok-4.7'
         and report['provider_purpose'] == 'planning', 'candidate_route_differs')
    need(report['native_prompt_sha256'] == native['native_prompt_sha256'],
         'native_prompt_binding_differs')
    need(all(report[key] is False for key in AUTHORITY)
         and all(report[key] is False for key in FALSE_EFFECTS), 'candidate_promotes_authority_or_effects')
    need(type(report['gold_mask']) is int and report['gold_mask'] == 0
         and type(report['training_mask']) is int and report['training_mask'] == 0,
         'candidate_promotes_training_masks')
    commitments = {role: {key: capsule['manifest']['files'][role][key]
                         for key in ('bytes', 'sha256')} for role in ROLES}
    need(report['input_commitments'] == commitments, 'candidate_source_pins_differ')
    expected = owner_annex(capsule['documents']['task_manifest'])
    need(report['owner_annex'] == expected, 'owner_annex_population_or_attribution_differs')
    tasks = capsule['documents']['task_manifest']['tasks']
    preserved = [{'task_id': row['task_id'], 'depends_on': row['depends_on'],
                  'status_snapshot': row['status'], 'legacy_parent_task': row['legacy_parent_task']}
                 for row in tasks]
    need(report['preserved_task_edges'] == preserved
         and report['legacy_parent_crosswalk'] == expected['legacy_parent_crosswalk'],
         'candidate_dependency_or_parent_crosswalk_changed')
    _audit_predicates(capsule['documents']['retained_source_audit'])
    proposal = report['model_proposal']
    need(type(proposal) is dict and set(proposal) == {
        'task_id', 'summary', 'adoptions', 'collision_candidates', 'next_actions'},
        'model_proposal_shape_differs')
    need(proposal['task_id'] == TASK
         and proposal['summary'] == 'bounded source-only reconciliation proposal',
         'unsupported_model_summary')
    adoptions = proposal['adoptions']
    need(type(adoptions) is list and 1 <= len(adoptions) <= 24, 'finding_population_differs')
    seen = set()
    for row in adoptions:
        need(type(row) is dict and set(row) == {
            'evidence_role', 'json_pointer', 'finding_code', 'gap_code'}, 'finding_shape_differs')
        code = row['finding_code']
        need(code in FINDINGS and (row['evidence_role'], row['json_pointer']) == FINDINGS[code]
             and row['gap_code'] in GAPS, 'unsupported_finding_or_reference')
        need(code not in seen, 'duplicate_finding_code')
        seen.add(code)
        _pointer(capsule['documents'][row['evidence_role']], row['json_pointer'])
    collisions = proposal['collision_candidates']
    need(type(collisions) is list and len(collisions) <= 3
         and len(set(collisions)) == len(collisions)
         and all(value in COLLISIONS for value in collisions), 'unsupported_collision_claim')
    need(proposal['next_actions'] == [{'task_id': 'LIR-9002',
        'reason_code': 'bind_capabilities_after_factual_acceptance', 'prerequisites': [TASK]}],
        'next_phase_dependency_or_activation_changed')
    need(type(invocation) is dict and set(invocation) == INVOCATION_FIELDS,
         'invocation_receipt_fields_differ')
    need(invocation['schema'] == 'legal-ir-grok-source-planning-invocation/v2'
         and invocation['status'] == 'proposal_validated'
         and invocation['task_id'] == TASK and invocation['task_cid'] == native['portal_task_cid']
         and type(invocation['attempt']) is int and invocation['attempt'] == native['attempt']
         and invocation['native_prompt_sha256'] == native['native_prompt_sha256']
         and invocation['controller_manifest_sha256'] == capsule['manifest_pin']['sha256']
         and invocation['input_commitments'] == commitments
         and invocation['input_custody_rechecked'] is True,
         'invocation_source_or_native_binding_differs')
    need(report['native_prompt_artifact'] == invocation['native_prompt_artifact'],
         'original_native_prompt_artifact_binding_differs')
    need(all(invocation[key] is False for key in AUTHORITY), 'invocation_promotes_authority')
    need(type(invocation['adapter_invocation_id']) is str
         and re.fullmatch(r'[0-9a-f]{32}', invocation['adapter_invocation_id']),
         'adapter_invocation_identity_missing')
    need(type(invocation['native_prompt_bytes']) is int
         and 0 < invocation['native_prompt_bytes'] <= 128000
         and type(invocation['model_response_bytes']) is int
         and 0 < invocation['model_response_bytes'] <= 65536
         and _sha(invocation['model_response_sha256']), 'invocation_population_bounds_differ')
    route = invocation['router_invocation']
    need(type(route) is dict and route.get('provider') == 'grok_cli'
         and route.get('model') == 'grok-4.7' and route.get('purpose') == 'planning'
         and type(route.get('router_calls')) is int and route['router_calls'] == 1,
         'actual_router_invocation_route_differs')
    return {'task_count': 64, 'goal_count': 34, 'legacy_parent_count': 24,
            'dependency_count': 115, 'closed_model_findings': len(adoptions),
            'source_snapshot_limits_preserved': True,
            'retained_audit_scope_preserved': True, 'statutory_gold_created': False}


def check_report(path, *, capsule, native, receipt_root):
    raw, report_pin = read_checked(Path(path))
    need(report_pin['sha256'] == native['report_sha256'], 'frozen_report_hash_changed')
    report = strict_json(raw)
    receipt_pin = report.get('invocation_receipt') if type(report) is dict else None
    need(type(receipt_pin) is dict and set(receipt_pin) == {'name', 'bytes', 'sha256'},
         'invocation_pin_shape_differs')
    name = _name(receipt_pin['name'])
    need(re.fullmatch(r'LIR-9001-[0-9a-f]{32}\.json', name), 'invocation_receipt_name_differs')
    root = Path(receipt_root)
    need(root.is_absolute() and root.resolve(strict=True) == root,
         'native_receipt_root_required')
    invocation_raw, actual_receipt_pin = read_checked(root / name, pin=receipt_pin)
    need(actual_receipt_pin['sha256'] == native['invocation_sha256'],
         'native_invocation_receipt_hash_differs')
    invocation = strict_json(invocation_raw)
    prompt_pin = invocation.get('native_prompt_artifact') if type(invocation) is dict else None
    need(type(prompt_pin) is dict and set(prompt_pin) == {'name', 'bytes', 'sha256'},
         'original_native_prompt_pin_required')
    expected_prompt_name = 'LIR-9001-' + invocation['adapter_invocation_id'] + '-native-prompt.txt'
    need(prompt_pin['name'] == expected_prompt_name, 'original_native_prompt_name_differs')
    prompt_raw, actual_prompt_pin = read_checked(root / expected_prompt_name, maximum=128000, pin=prompt_pin)
    need(actual_prompt_pin['sha256'] == native['native_prompt_sha256']
         and actual_prompt_pin['bytes'] == invocation['native_prompt_bytes'],
         'actual_original_native_prompt_bytes_differ')
    facts = validate_report_data(report, capsule, native, invocation)
    recheck_capsule(capsule)
    read_checked(Path(path), pin=report_pin)
    read_checked(root / name, pin=actual_receipt_pin)
    read_checked(root / expected_prompt_name, maximum=128000, pin=actual_prompt_pin)
    return {'schema': 'legal-ir-source-factual-check-result/v2', 'valid': True,
            'evidence_kind': EVIDENCE_KIND, 'task_id': TASK,
            'task_cid': native['task_cid'], 'portal_task_cid': native['portal_task_cid'],
            'database_attempt_number': native['database_attempt_number'],
            'execution_packet_revision': native['task_revision'],
            'native_attempt_id': native['native_attempt_id'], 'attempt': native['attempt'],
            'report': report_pin, 'invocation_receipt': actual_receipt_pin,
            'native_prompt_sha256': native['native_prompt_sha256'],
            'original_native_prompt': actual_prompt_pin,
            'controller_manifest': capsule['manifest_pin'],
            'acceptance_contract': capsule['contract_pin'], 'checked_facts': facts,
            'scope': 'frozen metadata reconciliation and closed candidate findings only',
            'repository_head_currency_independently_checked': False,
            'database_opened': False, 'product_imported': False,
            'native_evidence_written': False, 'signature_created': False,
            'next_phase_activated': False, **AUTHORITY}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-all', action='store_true')
    mode.add_argument('--check-report', type=Path)
    parser.add_argument('--controller-prefix', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--receipt-root', type=Path)
    args = parser.parse_args(argv)
    try:
        capsule = load_capsule(args.controller_prefix)
        if args.check_all:
            recheck_capsule(capsule)
            result = {'schema': 'legal-ir-source-factual-preflight/v1', 'valid': True,
                      'tasks': 64, 'goals': 34, 'legacy_parents': 24, 'dependencies': 115,
                      'frozen_owner_projection_ready': [TASK], 'phase_blocked_tasks': 63,
                      'actual_live_frontier_rechecked': False,
                      'source_files_rehashed': len(capsule['files']),
                      'product_imported': False, 'database_opened': False,
                      'completion_authority': False, 'native_admission_supplied': False}
        else:
            need(args.receipt_root is not None, 'native_receipt_root_required')
            result = check_report(args.check_report, capsule=capsule,
                                  native=native_bindings(os.environ), receipt_root=args.receipt_root)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        # No arbitrary exception values, raw candidate text or source bodies.
        reason = str(error) if isinstance(error, FactualRefusal) else type(error).__name__
        print(json.dumps({'schema': 'legal-ir-source-factual-refusal/v1', 'valid': False,
                          'reason': reason, 'native_evidence_written': False,
                          'completion_authority': False}, sort_keys=True))
        return 1


if __name__ == '__main__':
    sys.exit(main())
