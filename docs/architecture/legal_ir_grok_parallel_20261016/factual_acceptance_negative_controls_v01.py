#!/usr/bin/env python3
"""Declared authored controls for the protected LIR-9001 factual validator.

Run only in a separately admitted native validation attempt. These are synthetic
protocol fixtures, not a Grok output, actual invocation or accepted native task.
The script imports only the supplied, pinned standard-library validator. It
retains fixture files in an exclusively created child of --workdir and emits
control counts; it never writes acceptance evidence to the task catalog.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile


def save_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError('short fixture write')
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validator', type=Path, required=True)
    parser.add_argument('--validator-sha256', required=True)
    parser.add_argument('--controller-prefix', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    # This is deliberately not run during source preparation/static compile.
    raw = args.validator.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.validator_sha256:
        raise ValueError('validator source pin differs')
    compile(raw, str(args.validator), 'exec')
    spec = importlib.util.spec_from_file_location('legal_ir_factual_controls_target', args.validator)
    module = importlib.util.module_from_spec(spec)
    # Execute the same already checked buffer; avoid a second unchecked load.
    exec(compile(raw, str(args.validator), 'exec'), module.__dict__)
    v = module
    capsule = v.load_capsule(args.controller_prefix)
    workdir = args.workdir.resolve(strict=True)
    info = workdir.lstat()
    v.need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
           and not info.st_mode & 0o022, 'owned_fixture_directory_required')
    fixture_root = Path(tempfile.mkdtemp(prefix='lir-factual-authored-controls-', dir=workdir))
    receipts = fixture_root / 'receipts'
    receipts.mkdir(mode=0o700)
    commitments = {role: {key: capsule['manifest']['files'][role][key]
                         for key in ('bytes', 'sha256')} for role in v.ROLES}
    native = {'task_cid': capsule['contract']['selected_task_cid'],
              'task_revision': capsule['contract']['execution_packet_revision'],
              'attempt': 1, 'native_attempt_id': 'declared-authored-fixture:not-a-native-attempt',
              'native_prompt_sha256': 'a' * 64}
    invocation = {
        'schema': 'legal-ir-grok-source-planning-invocation/v1', 'status': 'proposal_validated',
        'task_id': v.TASK, 'task_cid': native['task_cid'], 'attempt': native['attempt'],
        'adapter_invocation_id': 'c' * 32, 'native_prompt_sha256': native['native_prompt_sha256'],
        'native_prompt_bytes': 128, 'model_response_sha256': 'b' * 64,
        'model_response_bytes': 1024, 'controller_manifest_sha256': capsule['manifest_pin']['sha256'],
        'input_commitments': commitments,
        'router_invocation': {'provider': 'grok_cli', 'model': 'grok-4.7',
                              'purpose': 'planning', 'router_calls': 1},
        'input_custody_rechecked': True, **v.AUTHORITY,
    }
    receipt_raw = v.canonical(invocation)
    receipt_name = 'LIR-9001-' + 'c' * 32 + '.json'
    save_new(receipts / receipt_name, receipt_raw)
    receipt_pin = {'name': receipt_name, 'bytes': len(receipt_raw), 'sha256': v.sha(receipt_raw)}
    native['invocation_sha256'] = receipt_pin['sha256']
    tasks = capsule['documents']['task_manifest']['tasks']
    report = {
        'schema': 'legal-ir-source-handoff-reconciliation-candidate/v1',
        'task_id': v.TASK, 'task_cid': native['task_cid'], 'attempt': native['attempt'],
        'status': 'candidate_pending_independent_acceptance', 'candidate': True,
        'provider': 'grok_cli', 'model': 'grok-4.7', 'provider_purpose': 'planning',
        'native_prompt_sha256': native['native_prompt_sha256'], 'input_commitments': commitments,
        'invocation_receipt': receipt_pin,
        'preserved_task_edges': [{'task_id': row['task_id'], 'depends_on': row['depends_on'],
            'status_snapshot': row['status'], 'legacy_parent_task': row['legacy_parent_task']} for row in tasks],
        'legacy_parent_crosswalk': capsule['annex']['legacy_parent_crosswalk'],
        'owner_annex': capsule['annex'],
        'model_proposal': {'task_id': v.TASK, 'summary': 'bounded source-only reconciliation proposal',
            'adoptions': [{'evidence_role': 'retained_source_audit', 'json_pointer': '/retained_source',
                'finding_code': 'reuse_retained_source_closure', 'gap_code': 'source_span_correspondence_open'}],
            'collision_candidates': ['current_task_owner_state_not_observed'],
            'next_actions': [{'task_id': 'LIR-9002', 'reason_code': 'bind_capabilities_after_factual_acceptance',
                              'prerequisites': [v.TASK]}]},
        'gold_mask': 0, 'training_mask': 0,
        **{key: False for key in v.FALSE_EFFECTS}, **v.AUTHORITY,
    }
    positive_raw = v.canonical(report)
    native['report_sha256'] = v.sha(positive_raw)
    positive = fixture_root / 'positive-authored-fixture.json'
    save_new(positive, positive_raw)
    result = v.check_report(positive, capsule=capsule, native=native, receipt_root=receipts)
    v.need(result['valid'] is True and result['native_evidence_written'] is False,
           'declared_positive_control_failed')
    outcomes = []

    def refused(name, *, alter=None, native_change=None, capsule_change=None,
                raw_change=None, keep_frozen_hash=False):
        candidate = copy.deepcopy(report)
        context = copy.deepcopy(capsule)
        binding = dict(native)
        if alter:
            alter(candidate)
        if native_change:
            native_change(binding)
        if capsule_change:
            capsule_change(context)
        payload = v.canonical(candidate)
        if raw_change:
            payload = raw_change(payload)
        if not keep_frozen_hash:
            binding['report_sha256'] = v.sha(payload)
        path = fixture_root / (name + '.json')
        save_new(path, payload)
        try:
            v.check_report(path, capsule=context, native=binding, receipt_root=receipts)
        except (v.FactualRefusal, OSError, ValueError, TypeError, KeyError, IndexError):
            outcomes.append({'control': name, 'disposition': 'refused'})
            return
        raise AssertionError('negative control accepted: ' + name)

    def remove_edge(candidate):
        next(row for row in candidate['owner_annex']['dependency_edges']
             if row['depends_on'])['depends_on'].pop()
    refused('missing-dependency-edge', alter=remove_edge)
    refused('missing-legacy-parent', alter=lambda c: c['legacy_parent_crosswalk'].pop())
    refused('factual-owner-annex-relabelled-model', alter=lambda c: c['owner_annex'].__setitem__(
        'attribution', 'model_generated_and_accepted'))
    refused('fabricated-completion', alter=lambda c: c.__setitem__('task_completed', True))
    refused('semantic-authority-promotion', alter=lambda c: c.__setitem__('gold_admission_authority', True))
    refused('gold-mask-promotion', alter=lambda c: c.__setitem__('gold_mask', 1))
    refused('boolean-mask-substitution', alter=lambda c: c.__setitem__('training_mask', False))
    refused('unsupported-finding', alter=lambda c: c['model_proposal']['adoptions'][0].__setitem__(
        'finding_code', 'laws_are_formalized'))
    refused('false-evidence-reference', alter=lambda c: c['model_proposal']['adoptions'][0].__setitem__(
        'json_pointer', '/authority'))
    refused('next-phase-activation', alter=lambda c: c['model_proposal']['next_actions'][0].__setitem__(
        'task_id', 'LIR-9003'))
    refused('input-commitment-substitution', alter=lambda c: c['input_commitments']['task_manifest'].__setitem__(
        'sha256', '0' * 64))
    refused('changed-frozen-report', alter=lambda c: c.__setitem__('candidate', False), keep_frozen_hash=True)
    refused('stale-task-revision', native_change=lambda b: b.__setitem__('task_revision', b['task_revision'] + 1))
    refused('wrong-native-attempt', native_change=lambda b: b.__setitem__('attempt', b['attempt'] + 1))
    refused('counterfeit-invocation-pin', native_change=lambda b: b.__setitem__('invocation_sha256', '0' * 64))
    refused('changed-source-pin', capsule_change=lambda c: c['files'][0].__setitem__('sha256', '0' * 64))
    refused('duplicate-json-key', raw_change=lambda raw: b'{"task_id":"LIR-9001",' + raw[1:])
    refused('nonfinite-json-value', raw_change=lambda raw: raw.replace(b'"gold_mask":0', b'"gold_mask":NaN', 1))
    v.recheck_capsule(capsule)
    output = {'schema': 'legal-ir-source-factual-authored-controls/v1',
              'valid': True, 'positive_authored_fixtures': 1,
              'negative_controls_refused': len(outcomes), 'controls': outcomes,
              'fixture_root': str(fixture_root), 'fixtures_retained': True,
              'actual_grok_invocations': 0, 'actual_native_attempts_accepted': 0,
              'product_imported': False, 'database_opened': False,
              'native_evidence_written': False, 'completion_authority': False}
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
