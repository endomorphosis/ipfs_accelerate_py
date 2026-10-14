"""Execute exactly one explicitly selected label from the frozen capsule pair.

No provider retries or fallback. A new prelaunch disk observation is recorded for
this invocation; the frozen recipe determines whether it is advisory or required.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from campaign_common import (A, Q, LABELS, BUDGETS, binding, check_binding, need, now,
    read, recipe, record_admission_gate, reporting_freeze_binding, safe_environment, sha, write_new)


def execute(label):
    lock_path = A / 'pair-lock.json'; lock = read(lock_path)
    need(lock.get('schema') == 'capsule-budget-live-pair-lock@1' and lock.get('frozen_before_launch') is True,
         'prelaunch capsule-budget pair lock required')
    for row in lock['launch_inputs'].values(): check_binding(row)
    for row in lock['runtime'].values(): check_binding(row)
    for key in ('verification', 'readiness', 'control_audit'): check_binding(lock[key])
    check_binding(lock['reporting_inputs'])
    reporting_freeze_binding(lock['reporting_inputs']['path'])
    for path, digest in lock['host_source_sha256'].items():
        need(sha(path) == digest, 'frozen benchmark source changed')
    for row in lock['attempts']:
        for key in ('preparation', 'configuration'): check_binding(row[key])
    selected = recipe(); attempt = next(row for row in lock['attempts'] if row['label'] == label)
    need(attempt['capsule_budget'] == BUDGETS[label], 'locked budget differs')
    trial = Path(attempt['directory'])
    need(str(trial) == selected['arms'][label]['output'], 'locked output differs')
    need(not (trial / 'launch-start.json').exists() and not (trial / 'invocation.json').exists(),
         'immutable attempt already launched; no automatic retries')
    command = selected['arms'][label]['execute_argv']
    need(command[-3:] == ['execute', '--output', str(trial)], 'declared execution command required')
    # This host observation precedes all generation. Production setup/preflight
    # independently determines whether the actual container may proceed.
    admission = record_admission_gate(trial / 'admission-gates', selected['host_disk_prerequisite'])
    start = dict(schema='capsule-budget-live-launch@1', label=label, capsule_budget=BUDGETS[label],
        arm='full', argv=command, cwd=str(Q), pair_lock_sha256=sha(lock_path),
        runner_sha256=sha(Path(__file__)), recipe_sha256=sha(A / 'recipe.json'),
        environment_manifest_sha256=sha(A / 'live-environment.json'),
        preparation_sha256=attempt['preparation']['sha256'], prelaunch_disk_observation=admission,
        started_at_utc=now(), inherited_environment_values_retained=False)
    write_new(trial / 'launch-start.json', start)
    print(json.dumps(dict(label=label,status='launching',pair_lock_sha256=start['pair_lock_sha256'])),flush=True)
    before = time.monotonic()
    with (trial / 'execute.stdout').open('xb') as out, (trial / 'execute.stderr').open('xb') as err:
        completed = subprocess.run(command, cwd=Q, env=safe_environment(), stdin=subprocess.DEVNULL,
            stdout=out, stderr=err)
    receipt_path = trial / 'receipt.json'; receipt = read(receipt_path) if receipt_path.is_file() else {}
    result = dict(schema='capsule-budget-live-launch-result@1', label=label, capsule_budget=BUDGETS[label],
        arm='full', exit_code=completed.returncode, seconds=time.monotonic()-before,
        preparation_unchanged=sha(attempt['preparation']['path']) == attempt['preparation']['sha256'],
        configuration_unchanged=sha(attempt['configuration']['path']) == attempt['configuration']['sha256'],
        host_source_unchanged=all(sha(path) == digest for path,digest in lock['host_source_sha256'].items()),
        stdout_sha256=sha(trial / 'execute.stdout'), stderr_sha256=sha(trial / 'execute.stderr'),
        receipt_sha256=sha(receipt_path) if receipt else None,
        complete_single_trial_receipt=receipt.get('complete_single_trial_receipt'), trial_count=receipt.get('trial_count'),
        results=[dict(reward=row.get('reward'), exception_type=row.get('exception_type'))
            for row in receipt.get('trials',[])])
    write_new(trial / 'launch-result.json', result)
    print(json.dumps(result,sort_keys=True),flush=True)
    return completed.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', choices=LABELS)
    return execute(parser.parse_args().label)


if __name__ == '__main__':
    raise SystemExit(main())
