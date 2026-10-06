"""Synthetic checks for the closed indexed-path exporter; no provider calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'export_indexed_path.py'
spec = importlib.util.spec_from_file_location('indexed_path_export', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='indexed-path-export-safety-review-02.json')
    args = parser.parse_args()
    if Path(args.output).name != args.output or not args.output.endswith('.json'):
        parser.error('simple new output filename required')
    marker = 'PRIVATE_BODY_NOT_FOR_EXPORT'
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    value = exporter.project({
        'initial_context': {'indexed_symbols': True, 'index_id': None, 'body': marker},
        'context': {'worker_capsules': marker},
        'doctor_dispatch': {
            'reason_codes': ['doctor_task_data_contract_unavailable', marker, {'body': marker}],
            'symbolic_capabilities': {
                'gap_codes': ['local_contract_proof_not_reported', marker],
                'proof': {'local_contract_proof_reported': True, 'scope': marker},
                'body': marker}}})
    check('raw bodies omitted', marker not in json.dumps(value))
    check('boolean count rejected', value['actual_index']['indexed_symbols'] is None)
    check('missing reference equality not inferred', value['admitted_context']['same_initial_index_reference'] is None)
    check('known reasons retained', value['doctor']['known_reason_codes'] == ['doctor_task_data_contract_unavailable'])
    check('unknown reasons counted only', value['doctor']['unexported_reason_count'] == 2)
    check('known gaps retained', value['doctor']['symbolic_capabilities']['gap_codes'] == ['local_contract_proof_not_reported'])
    check('unknown gaps counted only', value['doctor']['symbolic_capabilities']['unexported_gap_count'] == 1)
    proof = value['doctor']['symbolic_capabilities']['proof']
    check('proof report not upgraded to whole proof', proof['local_contract_proof_reported'] is True and proof['whole_program_verified'] is None)
    check('missing numeric fields unknown', value['source384']['counts']['provider_calls'] is None)
    check('nonfinite seconds rejected', exporter.pick({'seconds': float('nan')}, 'seconds')['seconds'] is None)
    check('digest format enforced', exporter.digest(marker) is None)
    result = dict(schema='indexed-path-export-safety-review@1', passed=True,
        checks=checks, check_count=len(checks), script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        provider_calls=0, containers_launched=0, raw_private_data_exported=False)
    with (ROOT / args.output).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'provider_calls': 0}))


if __name__ == '__main__':
    main()
