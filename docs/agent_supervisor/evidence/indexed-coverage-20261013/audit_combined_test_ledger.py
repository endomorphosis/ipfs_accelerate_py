"""Recompute the final distinct-test ledger from its three retained inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path('/home/barberb/lift_coding/artifacts/supervisor-indexed-coverage-20261013')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit_ids(path):
    rows = ET.parse(path).getroot().findall('.//testcase')
    assert rows and all(not any(row.find(tag) is not None for tag in
        ('failure', 'error', 'skipped')) for row in rows)
    ids = [row.attrib['classname'].replace('.', '/') + '.py::' + row.attrib['name'] for row in rows]
    assert len(ids) == len(set(ids))
    return set(ids)


def main():
    observed_path = ROOT / 'output-observation/combined-test-inventory.json'
    readiness_path = ROOT / 'readiness/run-02/pytest.xml'
    intent_path = ROOT / 'intent384/test-node-results.json'
    ledger_path = ROOT / 'combined-test-results.json'
    observed = json.loads(observed_path.read_bytes())
    observer = set(observed['node_ids'])
    assert len(observer) == len(observed['node_ids']) == observed['unique_passed'] == 295
    replayed_observer = set()
    for row in observed['runs'].values():
        path = Path(row['junit_path'])
        assert sha(path) == row['junit_sha256']
        ids = junit_ids(path)
        assert ids == set(row['node_ids']) and len(ids) == row['passed']
        replayed_observer |= ids
    assert replayed_observer == observer
    readiness = junit_ids(readiness_path)
    assert len(readiness) == 291
    intent = json.loads(intent_path.read_bytes())
    assert intent['failed'] == intent['skipped'] == 0 and intent['passed'] == 182
    assert all(row['status'] == 'passed' for row in intent['tests'])
    intent_nodes = {(row['repository'], row['node_id']) for row in intent['tests']}
    assert len(intent_nodes) == len(intent['tests']) == 182
    for record in intent['evidence']:
        path = Path(record['path'])
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    pools = dict(observer={('accelerate', node) for node in observer},
        readiness={('accelerate', node) for node in readiness}, intent=intent_nodes)
    combined = set().union(*pools.values())
    rows = [dict(repository=repo, node_id=node,
        passed_in=[key for key, values in pools.items() if (repo, node) in values])
        for repo, node in sorted(combined)]
    ledger = json.loads(ledger_path.read_bytes())
    expected_counts = {key: len(value) for key, value in pools.items()}
    expected_repos = dict(sorted(Counter(repo for repo, _ in combined).items()))
    repeated_memberships = sum(expected_counts.values()) - len(combined)
    overlapping = sum(len(row['passed_in']) > 1 for row in rows)
    assert repeated_memberships == overlapping == 26
    assert ledger['unique_passed'] == len(combined) == 742
    assert ledger['counts_by_suite'] == expected_counts
    assert ledger['counts_by_repository'] == expected_repos == dict(accelerate=681, datasets=61)
    assert ledger['overlapping_nodes_between_suites'] == overlapping
    assert ledger['unresolved_failures'] == ledger['remaining_skips'] == 0
    actual_rows = sorted(ledger['tests'], key=lambda row: (row['repository'], row['node_id']))
    assert actual_rows == rows
    report = dict(schema='independent-combined-test-ledger-audit@1', status='passed',
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            (observed_path, readiness_path, intent_path, ledger_path)},
        helper_sha256=sha(Path(__file__)), unique_passed=len(combined),
        counts_by_suite=expected_counts, counts_by_repository=expected_repos,
        overlapping_nodes=overlapping, repeated_memberships=repeated_memberships,
        pairwise_overlap={left + ':' + right: len(pools[left] & pools[right])
            for left, right in (('observer', 'readiness'), ('observer', 'intent'), ('readiness', 'intent'))},
        overlapping_files=dict(sorted(Counter(row['node_id'].split('::')[0]
            for row in rows if len(row['passed_in']) > 1).items())),
        all_combined_rows_match_exactly=True, observer_inventories_replayed_from_junit=True,
        readiness_replayed_from_junit=True, intent_inventory_evidence_digests_match=True,
        tests_executed_by_auditor=0,
        scope='Independent set-union and exact-row audit of retained successful test inventories; no new tests, benchmark, model inference or source proof.',
        limitations=['The Intent compact stdout is a count summary; its retained node inventory supplies per-node identities. The real-checkpoint run remains separate historical evidence.'])
    output = ROOT / 'combined-test-ledger-independent-audit.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(path=str(output), sha256=sha(output), unique_passed=len(combined), overlaps=overlapping)))


if __name__ == '__main__':
    main()
