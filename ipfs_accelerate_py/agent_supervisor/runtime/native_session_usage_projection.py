"""Diagnostic reconciliation of supplied retained native usage observations.

Codex cumulative sessions and Grok final envelopes remain separate scopes.
Reported Grok session IDs can be requested fallbacks; no identity, dispatch
population, complete task cost, billing, token cap or authority is established.
The two projection functions are pure. Only the explicit CLI reads/writes files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

from .cumulative_usage_reconciliation import (
    MAX_COUNTER, MAX_RECEIPTS, reconcile_router_cumulative_usage,
)

SCHEMA = 'retained-native-usage-observation@1'
GROK_SCHEMA = 'retained-grok-final-envelope-observations@1'
ROUTER_SCHEMA = 'router-implementation-invocation@1'
SEMANTIC_SCHEMA = 'router-semantic-decision-provider@1'
GROK_NATIVE_SCHEMA = 'native-grok-final-usage@1'
MAX_ISSUES = MAX_RECEIPTS
MAX_JSON_BYTES = 32 * 1024**2
MAX_JSON_DEPTH = 32
COUNTERS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
            'output_tokens', 'total_tokens', 'reasoning_output_tokens')
REQUIRED_COUNTERS = COUNTERS[:5]
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z')
_MODEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _safe_id(value):
    return type(value) is str and _ID.fullmatch(value) is not None


def _count(value):
    return type(value) is int and 0 <= value <= MAX_COUNTER


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _identity_hash(value):
    return _sha(value) if _safe_id(value) else None


def _workspace_hash(value):
    if type(value) is not str or not 1 <= len(value) <= 4096 or '\x00' in value:
        return None
    path = PurePosixPath(value)
    try:
        valid = path.is_absolute() and str(path) == value and '..' not in path.parts
        return _sha(value) if valid and len(value.encode('utf-8')) <= 4096 else None
    except UnicodeError:
        return None


def _sum(values):
    """Sum known categories without turning no observations into zero."""
    known = [value for value in values if value is not None]
    if not known:
        return None
    result = sum(known)
    return result if result <= MAX_COUNTER else None


def _scalar(value):
    # Signature values never include arbitrary provider/model/tool text.
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value if -MAX_COUNTER <= value <= MAX_COUNTER else {'invalid_type':'int'}
    if type(value) is str and len(value) <= 4096:
        return value
    return {'invalid_type':type(value).__name__}


def _pick(value, keys):
    if type(value) is not dict:
        return {'invalid_mapping':type(value).__name__}
    return {key:_scalar(value[key]) for key in keys if key in value}


def _native_signature(value):
    if type(value) is not dict:
        return {'invalid_native':type(value).__name__}
    result = _pick(value, ('schema', 'thread_id', 'rollout_sha256', 'usage_observation_sha256',
        'usage_available', 'observed_token_count_records', 'observed_final_usage_records',
        'task_complete_observed', 'usage_complete_observed', 'totals_scope',
        'cache_included_in_input', 'completion_basis', 'lifecycle_malformed_records',
        'usage_lifecycle_complete'))
    result['usage'] = _pick(value.get('usage'), COUNTERS)
    return result


def _wrapper_signature(receipt):
    result = _pick(receipt, ('schema', 'invocation_id', 'call_id', 'provider', 'model',
        'workspace', 'model_prompt_sha256', 'prompt_sha256', 'native_prompt_sha256',
        'router_prompt_sha256', 'workspace_advisory_sha256', 'provider_called',
        'status', 'purpose', 'generation', 'session_id', 'binding_sha256', 'router_calls'))
    # phase is an incidental field inferred by the existing log collector.
    result['usage'] = _pick(receipt.get('usage'), ('session_id', 'thread_id', 'model_id'))
    result['native_rollout_usage'] = _native_signature(receipt.get('native_rollout_usage'))
    if receipt.get('schema') == SEMANTIC_SCHEMA and 'native_invocation' in receipt:
        nested = receipt['native_invocation']
        result['native_invocation'] = _wrapper_signature(nested) if (
            type(nested) is dict and nested.get('schema') == ROUTER_SCHEMA) else {'invalid_nested':True}
    result['native_provider_outcome'] = _pick(receipt.get('native_provider_outcome'),
        ('schema', 'envelope_observed', 'process_outcome', 'stop_reason', 'error_subtype',
         'error_observed', 'reason_code', 'classification_source'))
    return json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _grok_row(receipt, ordinal):
    schema = receipt.get('schema')
    wrapper = receipt.get('invocation_id') if schema == ROUTER_SCHEMA else receipt.get('call_id')
    row = {'observation_ordinal':ordinal, 'wrapper_schema':schema,
        'wrapper_identity_sha256':_identity_hash(wrapper), 'reported_session_identity_sha256':None,
        'status':'partial', 'usage':dict.fromkeys(COUNTERS), 'normalized_input_tokens':None,
        'normalized_total_tokens':None, 'task_complete_observed':None,
        'usage_complete_observed':None, 'observed_native_scope_complete':None,
        'usage_observation_sha256':None, 'reasons':[]}
    reasons = row['reasons']
    fatal = False
    if not _safe_id(wrapper):
        reasons.append('missing_or_invalid_wrapper_identity')
    model = receipt.get('model')
    if type(model) is not str or _MODEL.fullmatch(model) is None or _workspace_hash(receipt.get('workspace')) is None:
        reasons.append('invalid_wrapper_binding')
        fatal = True
    usage = receipt.get('usage')
    usage = usage if type(usage) is dict else {}
    identities = [usage[key] for key in ('session_id', 'thread_id') if key in usage]
    if not identities:
        reasons.append('reported_session_identity_unavailable')
    elif any(not _safe_id(value) for value in identities):
        reasons.append('invalid_reported_session_identity')
        fatal = True
    elif len(set(identities)) != 1:
        reasons.append('reported_session_identity_alias_conflict')
        fatal = True
    else:
        row['reported_session_identity_sha256'] = _identity_hash(identities[0])
    if 'model_id' in usage and usage['model_id'] != model:
        reasons.append('reported_model_binding_conflict')
        fatal = True
    native = receipt.get('native_rollout_usage')
    if type(native) is not dict or native.get('schema') != GROK_NATIVE_SCHEMA:
        reasons.append('native_final_usage_unavailable')
        row['status'] = 'conflict' if fatal else 'partial'
        return row
    if native.get('totals_scope') != 'native_final_envelope' or native.get('cache_included_in_input') is not False:
        reasons.append('unsupported_native_scope_or_cache_convention')
        fatal = True
    raw_usage = native.get('usage')
    if type(raw_usage) is not dict:
        reasons.append('invalid_native_counter_mapping')
        raw_usage = {}
        fatal = True
    for key in COUNTERS:
        if key not in raw_usage:
            continue
        if _count(raw_usage[key]):
            row['usage'][key] = raw_usage[key]
        else:
            reasons.append('invalid_native_counter')
            fatal = True
    if native.get('usage_available') is not True:
        reasons.append('native_usage_availability_conflict')
        fatal = True
    if 'observed_final_usage_records' in native and (
            type(native['observed_final_usage_records']) is not int or native['observed_final_usage_records'] != 1):
        reasons.append('invalid_final_usage_record_count')
        fatal = True
    for key in ('task_complete_observed', 'usage_complete_observed'):
        if type(native.get(key)) is bool:
            row[key] = native[key]
        elif key in native and native[key] is not None:
            reasons.append('invalid_completeness_observation')
    flags = [row['task_complete_observed'], row['usage_complete_observed']]
    declared_scope_complete = (
        False if any(value is False for value in flags)
        else True if all(value is True for value in flags) else None)
    value = native.get('usage_observation_sha256')
    if type(value) is str and _SHA.fullmatch(value):
        row['usage_observation_sha256'] = value
    elif value is not None:
        reasons.append('invalid_usage_observation_digest')
    counters = row['usage']
    if (counters['reasoning_output_tokens'] is not None and counters['output_tokens'] is not None
            and counters['reasoning_output_tokens'] > counters['output_tokens']):
        reasons.append('reasoning_subset_exceeds_output')
        fatal = True
    required = all(counters[key] is not None for key in REQUIRED_COUNTERS)
    if required:
        inputs = sum(counters[key] for key in REQUIRED_COUNTERS[:3])
        total = inputs + counters['output_tokens']
        if inputs > MAX_COUNTER or total > MAX_COUNTER:
            reasons.append('derived_counter_overflow')
            fatal = True
        elif total != counters['total_tokens']:
            reasons.append('reported_total_conflict')
            fatal = True
        elif not fatal:
            row['normalized_input_tokens'] = inputs
            row['normalized_total_tokens'] = total
    else:
        reasons.append('native_counter_categories_incomplete')
    if fatal:
        row['status'] = 'conflict'
    elif not required:
        row['status'] = 'partial'
    elif row['wrapper_identity_sha256'] is None or row['reported_session_identity_sha256'] is None:
        row['status'] = 'identity_unavailable'
    else:
        row['status'] = 'observed'
    if row['normalized_total_tokens'] is not None:
        row['observed_native_scope_complete'] = declared_scope_complete
    row['reasons'] = sorted(set(reasons))
    return row


def _quarantine_codex(result, threads, issue):
    """Guard only this new result; the established reconciler stays unchanged."""
    changed = False
    for session in result['sessions']:
        if session['thread_id'] in threads:
            session['status'] = 'conflict'
            session['known_provider_token_subtotal'] = None
            session['provider_total_tokens'] = None
            session['strict_usage_complete'] = None
            session['reasons'] = sorted(set(session['reasons']) | {'wrapper_accounting_conflict'})
            changed = True
        amount = session['known_provider_token_subtotal']
        if amount is not None and amount > MAX_COUNTER:
            session['known_provider_token_subtotal'] = session['provider_total_tokens'] = None
            session['strict_usage_complete'] = None
            session['status'] = 'conflict'
            session['reasons'] = sorted(set(session['reasons']) | {'derived_counter_overflow'})
            issue(None, 'codex_derived_counter_overflow')
            changed = True
    for key, value in list(result['known_counter_subtotals'].items()):
        if value is not None and value > MAX_COUNTER:
            result['known_counter_subtotals'][key] = None
            issue(None, 'codex_aggregate_counter_overflow')
            changed = True
    for key in ('known_provider_token_subtotal', 'provider_total_tokens'):
        value = result[key]
        if value is not None and value > MAX_COUNTER:
            result[key] = None
            issue(None, 'codex_aggregate_counter_overflow')
            changed = True
    if changed:
        result['provider_total_tokens'] = None
        result['supplied_receipt_usage_complete'] = False
        if threads:
            result['known_provider_token_subtotal'] = None
            result['known_counter_subtotals'] = dict.fromkeys(result['known_counter_subtotals'])
    return result


def reconcile_native_usage_observations(receipts):
    """Project a bounded ordered population, without inferring its coverage."""
    if type(receipts) not in (list, tuple) or len(receipts) > MAX_RECEIPTS:
        raise ValueError('bounded ordered retained receipt population required')
    issues, omitted = [], 0
    def issue(ordinal, reason):
        nonlocal omitted
        if len(issues) < MAX_ISSUES:
            issues.append({'observation_ordinal':ordinal, 'reason':reason})
        else:
            omitted += 1
    recognized = duplicates = 0
    seen, grok_rows, codex_rows, quarantine_threads = {}, [], [], set()
    for ordinal, receipt in enumerate(receipts):
        if type(receipt) is not dict or receipt.get('schema') not in (ROUTER_SCHEMA, SEMANTIC_SCHEMA):
            issue(ordinal, 'unsupported_wrapper_schema')
            continue
        recognized += 1
        schema, provider = receipt['schema'], receipt.get('provider')
        identity = receipt.get('invocation_id') if schema == ROUTER_SCHEMA else receipt.get('call_id')
        key = (schema, identity) if _safe_id(identity) else None
        signature = _wrapper_signature(receipt)
        prior = seen.setdefault(key, []) if key is not None else []
        repeated = any(item['signature'] == signature for item in prior)
        if repeated:
            duplicates += 1
        entry = {'signature':signature, 'grok':None, 'thread':None}
        if provider == 'codex_cli':
            native = receipt if schema == ROUTER_SCHEMA else receipt.get('native_invocation')
            if type(native) is not dict or native.get('schema') != ROUTER_SCHEMA:
                issue(ordinal, 'codex_native_invocation_unavailable')
            else:
                child = dict(native)
                child.pop('phase', None)
                workspace = child.get('workspace')
                if type(workspace) is str:
                    try:
                        workspace.encode('utf-8')
                    except UnicodeError:
                        # Keep the original caller row untouched. The existing
                        # reconciler then emits its invalid binding conflict.
                        child['workspace'] = None
                        issue(ordinal, 'invalid_codex_workspace_encoding')
                codex_rows.append(child)
                usage = child.get('native_rollout_usage')
                thread = usage.get('thread_id') if type(usage) is dict else None
                entry['thread'] = thread if _safe_id(thread) else None
                if schema == SEMANTIC_SCHEMA:
                    mismatch = any(name in receipt and receipt[name] != child.get(name)
                        for name in ('provider', 'model', 'workspace', 'model_prompt_sha256'))
                    if ('native_rollout_usage' in receipt and
                            _native_signature(receipt['native_rollout_usage']) != _native_signature(usage)):
                        mismatch = True
                    if mismatch:
                        issue(ordinal, 'codex_semantic_wrapper_binding_conflict')
                        if entry['thread'] is not None:
                            quarantine_threads.add(entry['thread'])
        elif provider == 'grok_cli':
            if not repeated:
                row = _grok_row(receipt, ordinal)
                entry['grok'] = row
                grok_rows.append(row)
                for reason in row['reasons']:
                    issue(ordinal, reason)
        else:
            issue(ordinal, 'unsupported_wrapper_provider')
        if prior and not repeated:
            issue(ordinal, 'wrapper_accounting_alias_conflict')
            for old in [*prior, entry]:
                if old['thread'] is not None:
                    quarantine_threads.add(old['thread'])
                if old['grok'] is not None:
                    old['grok']['status'] = 'conflict'
                    old['grok']['normalized_input_tokens'] = old['grok']['normalized_total_tokens'] = None
                    old['grok']['observed_native_scope_complete'] = None
                    old['grok']['reasons'] = sorted(set(old['grok']['reasons']) | {'wrapper_accounting_alias_conflict'})
        if key is not None and not repeated:
            prior.append(entry)
    # A reported/fallback Grok session identity is not a cumulative charge ID.
    groups = {}
    for row in grok_rows:
        identity = row['reported_session_identity_sha256']
        if identity is not None:
            groups.setdefault(identity, []).append(row)
    ambiguous = 0
    for group in groups.values():
        if len(group) <= 1:
            continue
        ambiguous += 1
        for row in group:
            row['status'] = 'conflict'
            row['normalized_input_tokens'] = row['normalized_total_tokens'] = None
            row['observed_native_scope_complete'] = None
            row['reasons'] = sorted(set(row['reasons']) | {'ambiguous_reported_session_identity'})
            issue(row['observation_ordinal'], 'ambiguous_reported_session_identity')
    available = bool(grok_rows) and all(row['status'] == 'observed' for row in grok_rows)
    subtotals = dict.fromkeys((*COUNTERS, 'normalized_input_tokens', 'normalized_total_tokens'))
    if available:
        for key in subtotals:
            values = [(row['usage'][key] if key in COUNTERS else row[key]) for row in grok_rows]
            if any(value is None for value in values):
                continue
            amount = _sum(values)
            subtotals[key] = amount
            if amount is None:
                issue(None, 'grok_aggregate_counter_overflow')
                if key in ('normalized_input_tokens', 'normalized_total_tokens'):
                    available = False
        if not available:
            subtotals['normalized_input_tokens'] = subtotals['normalized_total_tokens'] = None
    codex = _quarantine_codex(reconcile_router_cumulative_usage(codex_rows), quarantine_threads, issue)
    from ipfs_accelerate_py.cli_runtime.grok_prompt_accounting import grok_prompt_accounting_coverage
    return {'schema':SCHEMA, 'source_scope':'supplied retained receipts only',
        'observations_received':len(receipts), 'recognized_observations':recognized,
        'duplicate_wrapper_observations':duplicates, 'codex_cumulative':codex,
        'grok_final_envelopes':{'schema':GROK_SCHEMA, 'source_scope':'native_final_envelope',
            'coverage':grok_prompt_accounting_coverage(),
            'cache_included_in_input':False, 'rows':grok_rows, 'reported_identity_groups':len(groups),
            'ambiguous_reported_identity_groups':ambiguous,
            'missing_reported_identity_count':sum(row['reported_session_identity_sha256'] is None for row in grok_rows),
            'diagnostic_subtotal_available':available, 'diagnostic_subtotals':subtotals},
        'issues':issues, 'omitted_issue_count':omitted, 'task_attempt_total':None,
        'campaign_usage_complete':None, 'billing_total_verified':False,
        'native_cumulative_cap_enforced':False, 'completion_authority':False, 'execution_authority':False}


def project_retained_native_usage(report):
    """Read only declared retained observation paths, never producer summaries."""
    if type(report) is not dict:
        raise ValueError('retained report mapping required')
    supplied = report.get('provider_invocations', [])
    dialogues = report.get('semantic_decision_dialogues', [])
    if type(supplied) is not list or type(dialogues) is not list or (
            len(supplied) > MAX_RECEIPTS or len(dialogues) > MAX_RECEIPTS):
        raise ValueError('bounded retained report populations required')
    receipts, attempts_received = list(supplied), 0
    for dialogue in dialogues:
        if type(dialogue) is not dict:
            receipts.append(None)
        else:
            attempts = dialogue.get('attempts', [])
            if type(attempts) is not list or len(attempts) > MAX_RECEIPTS:
                raise ValueError('bounded dialogue attempt population required')
            attempts_received += len(attempts)
            if attempts_received > MAX_RECEIPTS or len(receipts) + len(attempts) > MAX_RECEIPTS:
                raise ValueError('aggregate retained observation population exceeds bound')
            receipts.extend(attempt.get('provider_receipt') if type(attempt) is dict else None for attempt in attempts)
        if len(receipts) > MAX_RECEIPTS:
            raise ValueError('aggregate retained observation population exceeds bound')
    result = reconcile_native_usage_observations(receipts)
    result['report_observation'] = {'provider_invocations_received':len(supplied),
        'dialogues_received':len(dialogues), 'attempts_received':attempts_received,
        'unreceipted_provider_attempt_reported':'unreceipted_provider_attempt' in report}
    if 'unreceipted_provider_attempt' in report:
        if len(result['issues']) < MAX_ISSUES:
            result['issues'].append({'observation_ordinal':None, 'reason':'unreceipted_provider_attempt_reported'})
        else:
            result['omitted_issue_count'] += 1
    return result


def _read_report(path):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError('exact regular report file required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_JSON_BYTES:
            raise ValueError('bounded regular single-link report file required')
        raw = stream.read(MAX_JSON_BYTES + 1)
        after = os.fstat(stream.fileno())
    def identity(info):
        return tuple(getattr(info, key) for key in ('st_dev', 'st_ino', 'st_mode', 'st_nlink',
            'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    if len(raw) != before.st_size or identity(before) != identity(after) or identity(before) != identity(path.lstat()):
        raise ValueError('report changed while reading')
    return raw


def _decode_report(raw):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError('duplicate report key')
            result[name] = value
        return result
    def nonfinite(_value):
        raise ValueError('nonfinite report number')
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite report number')
        return number
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                           parse_constant=nonfinite, parse_float=finite_float)
    except (UnicodeError, ValueError, RecursionError):
        raise ValueError('invalid bounded report JSON') from None
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > MAX_JSON_DEPTH or visited > 1_000_000:
            raise ValueError('bounded report JSON structure required')
        if type(item) is dict:
            pending.extend((child, depth+1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth+1) for child in item)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        if output.parent.resolve(strict=True) != output.parent or os.path.lexists(output):
            raise ValueError('fresh exact output file required')
        raw = _read_report(args.report)
        projection = project_retained_native_usage(_decode_report(raw))
        payload = {'schema':'retained-native-usage-report@1',
            'input_report_sha256':hashlib.sha256(raw).hexdigest(), 'input_report_bytes':len(raw),
            'projection':projection}
        encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
        print(json.dumps({'schema':'retained-native-usage-cli-summary@1',
            'input_report_sha256':payload['input_report_sha256'], 'input_report_bytes':len(raw),
            'observations_received':projection['observations_received']}, sort_keys=True))
    except (ValueError, OSError, RecursionError):
        print('retained native usage projection refused', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
