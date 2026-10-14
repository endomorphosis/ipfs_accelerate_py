"""Closed native plan/turn metadata supplement; never reads model transcripts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from ipfs_accelerate_py.cli_runtime.grok_stream_capture import validate_grok_stream_progress

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('readout', HERE / 'collect_live_readout.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def boolean(value):
    return value if type(value) is bool else None


def exit_code(value):
    return value if type(value) is int and -(1 << 31) <= value < (1 << 31) else None


def project(report, expected_invocations):
    calls = report.get('provider_invocations', [])
    r.need(type(calls) is list and len(calls) <= 256, 'bounded native invocation metadata required')
    r.need([item.get('invocation_id') for item in calls] == expected_invocations, 'native invocation population differs from readout')
    planned = r.obj(report.get('planning'))
    rows = []
    for call in calls:
        usage = r.obj(call.get('usage'))
        progress = call.get('native_grok_progress')
        if progress is not None:
            progress = validate_grok_stream_progress(progress)
        rows.append(dict(invocation_id=call['invocation_id'], phase=r.tag(call.get('phase')),
            num_turns=r.count(usage.get('num_turns')), exit_code=exit_code(usage.get('exit_code')),
            timed_out=boolean(usage.get('timed_out')), model_prompt_bytes=r.count(call.get('model_prompt_bytes')),
            native_stream=progress, ordinary_source_read_or_edit_count=None,
            ordinary_source_count_scope='Tool-name counters do not identify source paths or distinguish ordinary edits from mapped-helper shell commands.'))
    marker = report.get('unreceipted_provider_attempt')
    return dict(planning=dict(qualified=boolean(planned.get('qualified')),
            strategy=planned.get('planning_strategy') if planned.get('planning_strategy') in {'direct', 'declared_contract', 'intent_symbolic', 'intent_coverage'} else None,
            goals=r.count(planned.get('goals')), tasks=r.count(planned.get('tasks')),
            subgoals=r.count(planned.get('subgoals')), provider_calls=r.count(planned.get('provider_calls')),
            elapsed_seconds=r.number(planned.get('elapsed_seconds')),
            count_scope='Counts explicitly retained in the admitted planner result; absent subgoal counts remain unknown.'),
        selected_task_status=r.obj(report.get('task_state')).get('status') if r.obj(report.get('task_state')).get('status') in {'queued','running','completed','failed','blocked','cancelled','ready','pending'} else None,
        native_dispatch_count=None, native_dispatch_count_scope='No dispatch count is inferred from model invocation or planned-task counts.',
        invocations=rows, unreceipted_attempt_present=marker is not None and marker is not False)


def collect(readout_path):
    readout, binding = r.read(readout_path)
    r.need(readout.get('schema') == 'compact-source-live-readout@1', 'native readout schema required')
    lock, lb = r.read(readout['pair_lock']['path'])
    r.need(lb['sha256'] == readout['pair_lock']['sha256'], 'readout pair lock changed')
    registered = {row['directory']: row for row in lock['attempts']}
    rows = []
    for attempt in readout['attempts']:
        r.need(attempt['directory'] in registered and attempt['label'] == registered[attempt['directory']]['label'], 'attempt not registered by lock')
        entry = dict(label=attempt['label'], status=attempt['status'], trials=[])
        if attempt['receipt'].get('availability') != 'missing':
            receipt, rb = r.read(Path(attempt['directory']) / 'receipt.json')
            r.need(rb == attempt['receipt'], 'retained receipt changed since readout')
            r.need(len(receipt['trials']) == len(attempt['trials']), 'trial population changed')
            for raw, summarized in zip(receipt['trials'], attempt['trials']):
                r.need(raw.get('trial') == summarized['trial'], 'trial identity differs')
                entry['trials'].append(dict(trial=summarized['trial'], receipt=rb,
                    **project(r.obj(raw.get('supervisor')), [row['invocation_id'] for row in summarized['usage']['invocations']])))
        rows.append(entry)
    return dict(schema='compact-source-native-shape-supplement@1', readout=binding, pair_lock=lb, attempts=rows,
        implementation_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__).resolve(), Path(r.__file__).resolve(),
            Path(__import__(validate_grok_stream_progress.__module__, fromlist=['']).__file__).resolve())},
        token_totals_come_from_bound_readout=True, raw_transcripts_read=False, verifier_or_solution_contents_read=False,
        provider_calls_by_supplement=0, causal_advantage_claimed=False,
        limitations=['Graph sizes, native turns, tool-name events and router invocations describe different scopes and are not additive.',
            'Reported native stream events are not authenticated tool execution records.',
            'Stream tool-name counts cannot prove whether the model read original source or used mapped helpers; use separately joined owner observations for mapped-helper delivery.',
            'No absent native count is treated as zero; no new token total is calculated here.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.readout)
    with args.output.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(dict(path=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))
