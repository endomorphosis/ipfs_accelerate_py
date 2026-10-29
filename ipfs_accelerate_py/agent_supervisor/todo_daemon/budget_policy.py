"""Opt-in provider freeze with read-only custody of the retained campaign budget.

This is deliberately not a quota allocator. A configured, valid policy ALWAYS
denies a new call. Broken custody also denies; no counter is written or reset.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

SCHEMA = 'ipfs_accelerate_py/agent-supervisor/provider-budget-freeze@1'


class BudgetPolicyDenied(RuntimeError):
    """No router child may start for this configured invocation."""


def _require(value, message):
    if not value:
        raise ValueError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, 'duplicate JSON field')
        result[key] = value
    return result


def _read(path, *, policy=False):
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
             and 0 < info.st_size <= 1024*1024, 'owned bounded ordinary budget file required')
    if policy:
        _require(info.st_nlink == 1 and info.st_mode & 0o022 == 0,
                 'budget policy must not be shared or writable by another identity')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        _require((opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino), 'budget file replaced')
        raw = stream.read(1024*1024+1)
        after = os.fstat(stream.fileno())
    _require(len(raw) == info.st_size and
             (info.st_size, info.st_mtime_ns, info.st_ctime_ns) ==
             (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'budget file changed while reading')
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    _require(type(value) is dict, 'budget object required')
    return value, hashlib.sha256(raw).hexdigest()


def _anchored(root, entry, keys):
    _require(type(entry) is dict and set(entry) == keys, 'closed budget anchor required')
    name = entry['path']
    _require(type(name) is str and name and not Path(name).is_absolute()
             and '..' not in Path(name).parts, 'relative budget anchor required')
    path = root/name
    _require(path.resolve().is_relative_to(root), 'budget anchor escapes campaign')
    _require(type(entry['sha256']) is str and re.fullmatch('[0-9a-f]{64}', entry['sha256']),
             'budget SHA256 required')
    value, digest = _read(path)
    _require(digest == entry['sha256'], 'retained budget anchor changed')
    return value


def inspect_budget_freeze(path, *, env_prefix, repo_root):
    """Read-only verification for operator review; never authorizes a call."""
    path = Path(path)
    _require(path.is_absolute(), 'absolute provider budget policy required')
    policy, policy_sha = _read(path, policy=True)
    _require(set(policy) == {'schema', 'env_prefix', 'campaign_root', 'new_calls_allowed',
                             'original_goals', 'ledgers', 'aggregate_reserved', 'reason'}
             and policy['schema'] == SCHEMA and policy['new_calls_allowed'] is False
             and policy['reason'] == 'operator_provider_freeze', 'closed denying policy required')
    _require(type(env_prefix) is str and policy['env_prefix'] == env_prefix
             and re.fullmatch('[A-Z][A-Z0-9_]{0,63}', env_prefix), 'budget invocation prefix differs')
    root = Path(policy['campaign_root'])
    _require(root.is_absolute() and root.is_dir() and root == root.resolve(), 'canonical campaign root required')
    _require(Path(repo_root).resolve(strict=True).is_relative_to(root), 'invocation is outside frozen campaign')
    goals = _anchored(root, policy['original_goals'], {'path', 'sha256', 'router_calls_total'})
    cap = policy['original_goals']['router_calls_total']
    _require(type(cap) is int and cap >= 1 and
             type(goals['budgets']['router_calls_total']) is int and
             goals['budgets']['router_calls_total'] == cap, 'original campaign cap differs')
    entries = policy['ledgers']
    _require(type(entries) is list and len(entries) == 4
             and len({e['path'] for e in entries}) == len(entries), 'four unique retained ledgers required')
    used = 0
    for entry in entries:
        ledger = _anchored(root, entry, {'path', 'sha256', 'schema', 'calls'})
        _require(type(entry['calls']) is int and entry['calls'] >= 0
                 and type(ledger.get('calls')) is int and ledger['calls'] == entry['calls'],
                 'retained call count differs')
        if entry['schema'] is None:
            _require(set(ledger) == {'calls', 'contracts'} and type(ledger['contracts']) is dict
                     and all(type(v) is int and v >= 0 for v in ledger['contracts'].values())
                     and sum(ledger['contracts'].values()) == ledger['calls'], 'bootstrap reservations differ')
        else:
            _require(type(entry['schema']) is str and ledger.get('schema') == entry['schema']
                     and type(ledger.get('max_calls')) is int
                     and 0 <= ledger['calls'] <= ledger['max_calls']
                     and ledger.get('status') in {'RESERVED', 'FAILED', 'RESPONSE_RETAINED'},
                     'scientific review reservation differs')
        used += ledger['calls']
    _require(type(policy['aggregate_reserved']) is int and policy['aggregate_reserved'] == used,
             'aggregate reserved-call accounting differs')
    return {'policy_sha256': policy_sha, 'new_calls_allowed': False,
            'aggregate_reserved': used, 'original_cap': cap, 'completion_authority': False}


def enforce_budget_freeze(config):
    key = str(config.env_prefix)+'_BUDGET_POLICY'
    if key not in os.environ:
        return
    try:
        audit = inspect_budget_freeze(os.environ[key], env_prefix=config.env_prefix, repo_root=config.repo_root)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise BudgetPolicyDenied('Provider budget policy is missing or invalid; no router child was launched.') from None
    raise BudgetPolicyDenied('New provider calls are frozen by campaign policy; '
                             f"retained aggregate reservations {audit['aggregate_reserved']}, "
                             f"original cap {audit['original_cap']}. No router child was launched.")
