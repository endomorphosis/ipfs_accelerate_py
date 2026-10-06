"""Export test counts and source bindings, excluding captured test output."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value, length=64):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is None:
        raise ValueError('bounded hexadecimal identity required')
    return value


def source_hashes(snapshot, key='source_sha256'):
    result = {}
    for name, value in snapshot[key].items():
        path = Path(name)
        if (not isinstance(name, str) or len(name) > 512 or path.is_absolute()
                or '..' in path.parts or path.as_posix() != name or path.suffix != '.py'):
            raise ValueError('canonical Python source identity required')
        if key == 'source_sha256' and not name.startswith(('ipfs_accelerate_py/', 'benchmarks/', 'test/')):
            continue
        result[name] = digest(value) if value is not None else None
    return result


def collect(root, labels):
    if (not labels or len(labels) != len(set(labels)) or any(not isinstance(label, str)
            or re.fullmatch('[A-Za-z0-9_-]{1,128}', label) is None for label in labels)):
        raise ValueError('distinct canonical qualification labels required')
    rows = []
    for label in labels:
        command = json.loads((root / (label + '-command.json')).read_text())
        result = json.loads((root / (label + '-exit.json')).read_text())
        raw = (root / (label + '.xml')).read_bytes()
        if sha(raw) != result['xml_sha256']:
            raise ValueError('qualification XML hash changed')
        if len(raw) > 16 * 1024 * 1024 or b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
            raise ValueError('bounded ordinary qualification XML required')
        tree = ET.fromstring(raw)
        cases = tree.findall('.//testcase')
        counts = dict(passed=0, failed=0, errors=0, skipped=0)
        ids = []
        for case in cases:
            ids.append(case.get('classname', '') + '::' + case.get('name', ''))
            kind = ('failed' if case.find('failure') is not None else
                    'errors' if case.find('error') is not None else
                    'skipped' if case.find('skipped') is not None else 'passed')
            counts[kind] += 1
        before = command['before']
        after = result['after']
        if type(result['source_unchanged']) is not bool or result['source_unchanged'] != (before == after):
            raise ValueError('source_unchanged differs from recorded snapshots')
        if (type(result['exit_code']) is not int or not 0 <= result['exit_code'] <= 255
                or type(result['seconds']) not in (int, float)
                or not math.isfinite(result['seconds']) or result['seconds'] < 0):
            raise ValueError('bounded process result required')
        suites = tree.findall('.//testsuite') if tree.tag != 'testsuite' else [tree]
        if not suites or any(list(suite.iter('testsuite')) != [suite] for suite in suites):
            raise ValueError('flat qualification suites required')
        declared = {name: sum(int(suite.get(key, '0')) for suite in suites) for name, key in
                    [('failed', 'failures'), ('errors', 'errors'), ('skipped', 'skipped')]}
        declared_tests = sum(int(suite.get('tests', '0')) for suite in suites)
        if declared_tests != len(cases) or any(declared[name] != counts[name] for name in declared):
            raise ValueError('qualification suite counts disagree with testcase results')
        collection_errors = sum(case.find('error') is not None and not case.get('classname') for case in cases)
        outcome = ('collection_error' if collection_errors else
                   'test_failure' if counts['failed'] or counts['errors'] else
                   'process_failure' if result['exit_code'] != 0 else
                   'zero_completed_cases' if not counts['passed'] else 'passed')
        rows.append(dict(label=label, exit_code=result['exit_code'], seconds=result['seconds'],
            outcome=outcome, collection_errors=collection_errors,
            source_unchanged=result['source_unchanged'],
            source_head_before=digest(before['accelerate_head'], 40), source_head_after=digest(after['accelerate_head'], 40),
            datasets_head_before=digest(before['datasets_head'], 40), datasets_head=digest(after['datasets_head'], 40),
            datasets_clean=not after['datasets_status'], accelerate_clean=not after['accelerate_status'],
            source_sha256_before=source_hashes(before), source_sha256=source_hashes(after),
            datasets_source_sha256_before=source_hashes(before, 'datasets_source_sha256'),
            datasets_source_sha256=source_hashes(after, 'datasets_source_sha256'),
            counts=counts, test_case_id_sha256=sha(json.dumps(sorted(ids)).encode()),
            xml_sha256=digest(result['xml_sha256']), log_sha256=digest(result['log_sha256']),
            qualified=outcome == 'passed' and result['source_unchanged'] and not after['datasets_status'] and not after['accelerate_status'],
            captured_test_output_exported=False))
    return dict(schema='grok-recovery-targeted-qualification@1', runs=rows,
        repeated_cases_not_summed=True, whole_repository_suite_claim=False,
        provider_calls=0, raw_test_bodies_or_credentials_exported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('labels', nargs='+')
    args = parser.parse_args()
    if any(not label or any(not (c.isascii() and (c.isalnum() or c in '-_')) for c in label)
           for label in args.labels):
        raise ValueError('canonical qualification labels required')
    result = collect(args.root, args.labels)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps([dict(label=row['label'], **row['counts']) for row in result['runs']]))


if __name__ == '__main__':
    main()
