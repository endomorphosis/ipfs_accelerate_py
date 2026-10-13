"""Replay admitted public source and actually compile guarded Lean artifacts.

Only authored probe code is executed. Public Python inputs are parsed as data.
The generated declarations retain unresolved interpretation parameters; a Lake
success is syntax/type evidence, never a proof of the benchmark program.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from ipfs_datasets_py.logic.software_verification import numerical_source, numerical_logic
from ipfs_datasets_py.logic.software_verification import numerical_source_v2, numerical_logic_v2
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384
from ipfs_datasets_py.logic.formalization.autoencoder.native_family_lake_v4 import _execute

ROOT = Path('/home/barberb/lift_coding')
HERE = Path(__file__).resolve().parent
PUBLIC = ROOT / '.benchmarks/terminal-bench-2/largest-eigenval/environment/src'


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


AUTHORED = '''def choose_and_assert(flag, left, right, width):
    selected = left() if flag else right()
    assert selected, f"selected={selected!r:{width()}}"
    return selected
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lake', required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    assert output.parent == HERE and not output.exists()
    output.mkdir()
    started = time.monotonic()
    baseline_path = HERE / 'baseline-v1.json'
    assert sha(baseline_path.read_bytes()) == '893ce8c68b3a706e4b65cb3199c802b888728a50a8cdf286346d2029cd859412'
    baseline = json.loads(baseline_path.read_bytes())
    assert baseline['producer_sha256'] == {
        module.__name__: sha(Path(module.__file__).read_bytes())
        for module in (numerical_source, numerical_logic)}
    assert structured_source_384._implementation() == baseline['structured_checkpoint_implementation']
    source_bindings = {module.__name__: sha(Path(module.__file__).read_bytes()) for module in
                       (numerical_source, numerical_logic, numerical_source_v2, numerical_logic_v2)}
    files = []
    for expected in baseline['files']:
        name = expected['path']
        content = (PUBLIC / name).read_bytes()
        assert sha(content) == expected['source_sha256']
        previous = numerical_logic.formalize_numerical_source(content.decode(), path=name)
        assert sha(raw(previous)) == expected['canonical_report_sha256']
        report = numerical_logic_v2.formalize_numerical_source(content.decode(), path=name)
        assert report == numerical_logic_v2.validate_numerical_logic_report(report, content.decode(), path=name)
        assert report['coverage']['source_obligations_discharged'] == 0
        assert all(row['status'] == 'pending' and row['function_ids']
                   and row['evaluation_contexts'] and not row['evaluation_guard_verified']
                   for row in report['pending_obligations'])
        write(output / (name + '.report.json'), report)
        lean = report['projections']['lean']
        assert sha(lean['source'].encode()) == lean['source_sha256']
        (output / (name + '.lean')).write_text(lean['source'])
        build = _execute(lean['source'], 'NumericalObligations', args.lake, 60)
        write(output / (name + '.lake.json'), build)
        assert build['status'] == 'passed' and build['backend_executed']
        altered = deepcopy(report)
        altered['pending_obligations'][0]['evaluation_contexts'] = []
        try:
            numerical_logic_v2.validate_numerical_logic_report(altered, content.decode(), path=name)
        except numerical_logic_v2.NumericalLogicError:
            pass
        else:
            raise AssertionError('altered guard context accepted')
        assert sha((PUBLIC / name).read_bytes()) == expected['source_sha256']
        files.append(dict(path=name, source_sha256=expected['source_sha256'],
            previous_per_function=expected['per_function'], per_function=report['summary']['per_function'],
            previous_pending_obligations=expected['pending_obligations'],
            pending_obligations=len(report['pending_obligations']),
            obligation_kinds=dict(Counter(row['kind'] for row in report['pending_obligations'])),
            guarded_region_count=len(report['structural']['guarded_regions']),
            compound_contract_count=len(report['compound_contracts']),
            assertion_requirement_count=len(report['assertion_requirements']),
            report_sha256=report['report_sha256'], canonical_report_sha256=sha(raw(report)),
            lean_source_sha256=lean['source_sha256'], real_lake_status=build['status'],
            modified_guard_refused=True, v1_exact_report_unchanged=True,
            source_unchanged=True, source_obligations_discharged=0))
    report = numerical_logic_v2.formalize_numerical_source(AUTHORED, path='authored_guard.py')
    assert report['summary']['structurally_complete_functions'] == 1
    write(output / 'authored-guard.report.json', report)
    lean = report['projections']['lean']['source']
    build = _execute(lean, 'NumericalObligations', args.lake, 60)
    write(output / 'authored-guard.lake.json', build)
    assert build['status'] == 'passed'
    false_source = lean + '\ntheorem unsupported_source_truth : False := by trivial\n'
    negative = _execute(false_source, 'NumericalObligations', args.lake, 60)
    write(output / 'false-source-truth.lake.json', negative)
    assert negative['status'] == 'failed' and negative['backend_executed']
    assert not negative['timed_out'] and not negative['output_truncated']
    assert source_bindings == {module.__name__: sha(Path(module.__file__).read_bytes()) for module in
                              (numerical_source, numerical_logic, numerical_source_v2, numerical_logic_v2)}
    result = dict(schema='numerical-guarded-public-source-qualification@1', complete=True,
        profile=numerical_logic_v2.PROFILE, files=files, baseline_sha256=sha(baseline_path.read_bytes()),
        producer_sha256=source_bindings, structured_checkpoint_pins_unchanged=True,
        v1_exact_reports_unchanged=True, authored_guard_build_passed=True,
        false_source_truth_build_rejected=True, real_lake_builds=4,
        original_public_source_executed=False, hidden_verifier_read=False,
        official_terminal_bench_score=None, native_provider_tokens=None,
        provider_calls=0, model_inference_calls=0, source_obligations_discharged=0,
        source_semantics_verified=False, seconds=time.monotonic()-started)
    write(output / 'qualification.json', result)
    print(json.dumps(dict(complete=True, public_files=len(files),
        real_lake_builds=4, pending_obligations=sum(row['pending_obligations'] for row in files),
        source_obligations_discharged=0)))


if __name__ == '__main__':
    main()
