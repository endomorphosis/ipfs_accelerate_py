"""Seal the explicit public evidence allowlist after final tests and review.

Preparation does not publish anything. Executing this script creates one NEW
repository evidence directory, refusing an existing destination or incomplete
qualification. It never walks/copies private artifact directories.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET

R = Path(__file__).resolve().parent
P = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
DEST = P / 'docs/agent_supervisor/evidence/numerical-planning-selection-20261013'

# Closed relative filenames only. No recursive glob, directory copy, fallback
# discovery, raw broker audit, router observation, credentials or descriptors.
ALLOWLIST = (
    'baseline.json', 'baseline-summary-v1.json', 'baseline-summary-v2.json',
    'baseline-summary-v3.json', 'baseline-pages-v1.json', 'baseline-pages-v2.json',
    'baseline-pages-v3.json', 'capture_baseline.py',
    'qualified-summary-v3.json', 'public-replay.json', 'verify_baseline.py',
    'environment.json', 'run_checks.py', 'comparison-design.json', 'public_query_comparison.py',
    'datasets-pager-replay.json', 'replay_dataset_pager.py',
    'selection-contract-review.json', 'review_selection_contracts.py',
    'consumer-frozen.json', 'fixture-integration-audit.json', 'regression-summary.json',
    'docker-results-independent-audit.json', 'audit_docker_results.py',
    'documentation-independent-review.json',
    'datasets-tests/command.json', 'datasets-tests/stdout', 'datasets-tests/stderr',
    'datasets-tests/exit.json', 'datasets-tests/junit.xml',
    'accelerator-regression/command.json', 'accelerator-regression/stdout',
    'accelerator-regression/stderr', 'accelerator-regression/exit.json',
    'accelerator-regression/junit.xml',
    'consumer-focused-01/command.json', 'consumer-focused-01/stdout',
    'consumer-focused-01/stderr', 'consumer-focused-01/exit.json',
    'consumer-focused-01/junit.xml',
    'docker-selected-02/qualification.json', 'docker-selected-02/command.json',
    'docker-selected-02/container-controls.json', 'docker-selected-02/owner-command.json',
    'docker-selected-02/root-setup.py',
    'docker-selected-02/results/probe-qualification.json',
    'docker-selected-02/results/session-observation.json',
    'docker-selected-02/results/retrieval-comparison.json',
    'docker-selected-02/runner-snapshot/owner_probe.py',
    'docker-selected-02/runner-snapshot/worker_probe.py',
    'docker-selected-02/runner-snapshot/public_query_comparison.py',
    'docker-selected-02/runner-snapshot/run_docker_qualification.py',
    'docker-full-01/qualification.json', 'docker-full-01/command.json',
    'docker-full-01/container-controls.json', 'docker-full-01/owner-command.json',
    'docker-full-01/root-setup.py',
    'docker-full-01/results/probe-qualification.json',
    'docker-full-01/results/session-observation.json',
    'docker-full-01/runner-snapshot/owner_probe.py',
    'docker-full-01/runner-snapshot/worker_probe.py',
    'docker-full-01/runner-snapshot/public_query_comparison.py',
    'docker-full-01/runner-snapshot/run_docker_qualification.py',
    'docker-selected-01/qualification.json',
    'docker-selected-01/results/probe-qualification.json',
    'docker-release-02.json', 'seal_evidence.py',
)
DENIED_COMPONENTS = frozenset({
    'private-state', 'private-fixture-ipc', 'private-worker-observations',
    'private-fixture', 'worker-observations', 'broker-audit.json',
    'router-observation.json', 'owner.stdout', 'owner.stderr',
    'root-setup.stdout', 'root-setup.stderr', 'seal.duckdb',
})


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def read_regular(path):
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError('regular evidence file required: '+str(path))
    data = path.read_bytes()
    if path.is_symlink() or not path.is_file():
        raise ValueError('evidence path changed: '+str(path))
    return data


def load(snapshot, name):
    return json.loads(snapshot[name])


def junit(snapshot, directory, repository):
    if load(snapshot, directory+'/exit.json')['returncode'] != 0:
        raise ValueError('final regression did not pass: '+directory)
    root = ET.fromstring(snapshot[directory+'/junit.xml'])
    statuses = {}
    for case in root.iter('testcase'):
        key = repository+'::'+case.get('classname', '')+'::'+case.get('name', '')
        state = ('error' if case.find('error') is not None else
                 'failed' if case.find('failure') is not None else
                 'skipped' if case.find('skipped') is not None else 'passed')
        if key in statuses:
            raise ValueError('duplicate test identity in final JUnit: '+key)
        statuses[key] = state
    if not statuses or any(s in {'failed', 'error'} for s in statuses.values()):
        raise ValueError('final JUnit contains failure/error or no test cases: '+directory)
    return statuses


def qualification(snapshot):
    final_d = junit(snapshot, 'datasets-tests', 'datasets')
    final_p = junit(snapshot, 'accelerator-regression', 'accelerator')
    focused = junit(snapshot, 'consumer-focused-01', 'accelerator')
    if Counter(final_d.values()) != {'passed': 77}:
        raise ValueError('expected the final 77-test datasets pager qualification')
    primary = final_d | final_p
    for key in primary.keys() & focused.keys():
        if primary[key] != focused[key]:
            raise ValueError('overlapping test status differs: '+key)
    combined = primary | focused
    tests = dict(schema='numerical-selection-test-accounting@1',
        final_primary_counts=dict(Counter(primary.values())),
        datasets_final_counts=dict(Counter(final_d.values())),
        accelerator_final_counts=dict(Counter(final_p.values())),
        consumer_focused_counts=dict(Counter(focused.values())),
        consumer_focused_overlap_with_primary=len(primary.keys() & focused.keys()),
        consumer_focused_additional_unique=len(focused.keys() - primary.keys()),
        combined_unique_counts=dict(Counter(combined.values())),
        combined_test_statuses=dict(sorted(combined.items())),
        interpretation='Repeated runs and overlapping consumer checks are not added to unique pass counts.')
    regression = load(snapshot, 'regression-summary.json')
    if (regression['complete'] is not True
            or regression['distinct_passed'] != tests['combined_unique_counts'].get('passed', 0)
            or regression['focused_tests_are_subset'] is not True
            or tests['consumer_focused_additional_unique'] != 0):
        raise ValueError('independent JUnit counts differ from final regression summary')
    public = load(snapshot, 'public-replay.json')
    if not public['complete'] or not public['grammar_and_checkpoint_pins_equal']:
        raise ValueError('public source replay incomplete')
    for directory in ('docker-selected-02', 'docker-full-01'):
        result = load(snapshot, directory+'/qualification.json')
        if not all(result.get(k) is True for k in ('complete','container_removed','runner_unchanged','source_unchanged')):
            raise ValueError('successful Docker qualification incomplete: '+directory)
        if result['provider_calls'] != 0 or result['model_calls'] != 0:
            raise ValueError('offline evidence unexpectedly contains model/provider execution')
        command = load(snapshot, directory+'/command.json')
        if command['release_sha256'] != sha(snapshot['docker-release-02.json']):
            raise ValueError('Docker qualification is not bound to final release02')
    failed = load(snapshot, 'docker-selected-01/qualification.json')
    if failed['complete'] is not False or failed['owner_returncode'] == 0:
        raise ValueError('failed attempt label no longer matches its outcome')
    review = load(snapshot, 'selection-contract-review.json')
    if not review['passed']:
        raise ValueError('independent contract review failed')
    # The final Docker audit is required, with its explicit positive result.
    audit = load(snapshot, 'docker-results-independent-audit.json')
    if not (audit.get('passed') is True or audit.get('status') == 'passed'):
        raise ValueError('final Docker evidence audit not passed')
    return tests


def readme(snapshot, tests):
    comparison = load(snapshot, 'docker-selected-02/results/retrieval-comparison.json')
    old, exact = comparison['existing_filter'], comparison['exact_id']
    full = load(snapshot, 'docker-full-01/qualification.json')['probe']
    baseline = load(snapshot, 'baseline.json')
    before = next(r for r in baseline['rows'] if r['version'] == 3)
    summary = load(snapshot, 'qualified-summary-v3.json')
    current_bytes = len(json.dumps(summary, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())
    passes = tests['combined_unique_counts'].get('passed', 0)
    skips = tests['combined_unique_counts'].get('skipped', 0)
    return f'''# Numerical planning hints and exact obligation retrieval

This evidence qualifies deterministic compact hints and exact-ID retrieval over the two public `largest-eigenval` source files. It preserves complete source-replayed obligation rows and their guarded, CFG, loop and proof-phase scopes. No pending obligation was discharged.

The v3 summary changed from {before['summary_bytes']} to {current_bytes} canonical UTF-8 bytes and exposes {len(summary['pending_obligations'])} obligation hints. Both retrieval arms receive the same new hints. For five authored inspection scenarios, existing function/kind filtering used {old['requests']} requests and {old['canonical_request_response_bytes']} canonical request-plus-response bytes; exact-ID retrieval used {exact['requests']} requests and {exact['canonical_request_response_bytes']} bytes. Including one common summary delivery, totals are {old['summary_plus_canonical_request_response_bytes']} and {exact['summary_plus_canonical_request_response_bytes']} bytes. Every selected case is retained; the fixture does not discard less favorable cases. Canonical, framed IPC, and CLI stdout measures represent separate layers and must not be added together as independent costs.

The separate exhaustive session retrieved all {full['exact_retrieved_obligations']} rows across {full['page_count']} pages. It is an inventory reference, not the baseline for the selective retrieval comparison. Both successful sessions used real owner/worker UID separation, broker/client transport, native DuckLake hydration, and the production coding router with an explicitly authored model hook. Planner request compilation was exercised after authored preflight admission; this does not establish live planner selection or full-supervisor START. Intent autoencoding and learned Source384 inference were disabled for this qualification.

Final test evidence contains {passes} distinct passing tests and {skips} skipped tests. `validation-counts.json` records per-suite results and exact identities. The consumer-focused run overlaps {tests['consumer_focused_overlap_with_primary']} primary test identities and contributes {tests['consumer_focused_additional_unique']} additional unique identities; repeated passes are not counted twice.

`docker-selected-01` is retained only as a failed fixture attempt: `AttributeError` while inspecting an allowed string in a mixed planner-summary list, before the owner-worker launch. Its aggregate qualification and probe are not successful-run or measurement evidence. The fixed owner helper is bound by release02 and the successful selected02/full01 snapshots. The earlier static fixture review records its own pins and remains a design review; final Docker evidence audit establishes the successful-run evidence scope.

No provider/model calls, hidden verifier reads, benchmark-source execution, training, source proof discharge, official Terminal-Bench reward, or token/billing savings are claimed. The historical-summary discovery cost, actual model choices, semantic correctness, and end-to-end task performance remain unmeasured. All v1/v2 summary bytes, unfiltered/None-ID pager bytes, grammar/logic report bytes and checkpoint implementation pins retain their recorded compatibility guarantees.

Only the filenames in `seal_evidence.py` are copied. Private databases, source/report dumps, full prompts, raw broker audit, router observations, session descriptors, and private worker/fixture logs remain outside this directory. Allowed baseline pages contain a small explicitly captured public-source sample, not the private session dumps. Root setup and runner snapshots are authored fixture code. The base container remained writable for its existing sudo worker transition; the receipts do not establish a general hostile-worker containment proof.

`manifest.json` binds every other file by SHA-256 and length; its file count excludes itself and states the total including itself. The sealing script refuses an existing destination and incomplete final results. This directory is independent of older evidence directories, which are not modified.
'''


def main():
    if DEST.exists() or DEST.is_symlink():
        raise FileExistsError('refusing to overwrite evidence destination: '+str(DEST))
    if len(ALLOWLIST) != len(set(ALLOWLIST)):
        raise ValueError('duplicate allowlist path')
    snapshot = {}
    for name in ALLOWLIST:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or DENIED_COMPONENTS.intersection(path.parts):
            raise ValueError('disallowed evidence filename: '+name)
        selected = R / path
        # Do not follow a substituted private or external directory.
        if selected.resolve(strict=True) != selected.absolute():
            raise ValueError('evidence symlink/alias refused: '+name)
        snapshot[name] = read_regular(selected)
    tests = qualification(snapshot)
    snapshot['validation-counts.json'] = encoded(tests)
    snapshot['docker-selected-01/attempt-status.json'] = encoded(dict(
        schema='retained-failed-fixture-attempt@1', status='failed',
        error_type='AttributeError', error="'str' object has no attribute 'get'",
        cause='Authored owner helper called .get on an allowed string in a mixed planner-summary list.',
        phase='post-preflight authored planner-summary inspection before owner-worker launch',
        broker_requests=0, provider_calls=0, model_calls=0,
        qualifies_success=False, included_in_successful_measurement_totals=False,
        superseded_by='docker-selected-02', final_release='docker-release-02.json'))
    snapshot['README.md'] = readme(snapshot, tests).encode()
    snapshot['.gitattributes'] = b'**/stdout -whitespace\n**/pytest.log -whitespace\n'
    snapshot['manifest.json'] = encoded(dict(schema='numerical-planning-selection-evidence-manifest@1',
        files=[dict(path=name, bytes=len(data), sha256=sha(data)) for name,data in sorted(snapshot.items())],
        file_count_excluding_manifest=len(snapshot), total_file_count=len(snapshot)+1,
        copied_allowlist_count=len(ALLOWLIST), generated_file_count_excluding_manifest=4,
        scope='Explicit public allowlist; private state, prompts, raw broker/router observations and session descriptors excluded.'))
    # Recheck every source before any repository mutation. Only the immutable
    # bytes above are copied; no reread of changing source occurs while writing.
    for name in ALLOWLIST:
        if read_regular(R/name) != snapshot[name]:
            raise ValueError('evidence changed during sealing: '+name)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.numerical-planning-selection-',dir=DEST.parent))
    try:
        for name,data in sorted(snapshot.items()):
            target=temporary/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(data)
            target.chmod(0o644)
        if DEST.exists() or DEST.is_symlink():
            raise FileExistsError('destination appeared during sealing')
        os.rename(temporary,DEST)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    print(json.dumps(dict(destination=str(DEST),files=len(snapshot),manifest_sha256=sha(snapshot['manifest.json']),
        combined_unique_counts=tests['combined_unique_counts']),sort_keys=True))


if __name__ == '__main__':
    main()
