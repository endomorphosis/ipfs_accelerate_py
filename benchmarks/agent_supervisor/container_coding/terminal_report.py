"""Summarize Harbor rewards and native supervisor timings without hiding errors."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from terminal_worker import raw_usage


def elapsed(record):
    if not record or not record.get('started_at') or not record.get('finished_at'):
        return None
    return (datetime.fromisoformat(record['finished_at'].replace('Z', '+00:00')) -
            datetime.fromisoformat(record['started_at'].replace('Z', '+00:00'))).total_seconds()


def audit_tools(log_dir):
    counts = Counter()
    allowed = {'read_file', 'search_replace', 'grep', 'list_dir', 'todo_write'}
    for path in log_dir.glob('*.log'):
        for line in path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get('type') == 'tool_call':
                counts[event.get('toolName', event.get('title', 'unknown'))] += 1
    return {'observed': dict(counts), 'outside_file_profile': sorted(set(counts) - allowed)}


def summarize(job):
    rows = []
    config = json.loads((job / 'config.json').read_text())
    concurrency = config.get('n_concurrent_trials', 1)
    invalid_path = job / 'INVALID_FOR_COMPARISON.json'
    invalid = json.loads(invalid_path.read_text()) if invalid_path.exists() else None
    for path in sorted(job.glob('*/result.json')):
        trial = json.loads(path.read_text())
        worker_path = path.parent / 'agent/supervisor/worker-result.json'
        worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}
        rewards = (trial.get('verifier_result') or {}).get('rewards') or {}
        error = trial.get('exception_info') or {}
        tool_audit = audit_tools(path.parent / 'agent/supervisor/state/implementation_logs')
        rows.append({'task': trial['task_name'], 'trial': trial['trial_name'],
            'reward': rewards.get('reward'), 'error_type': error.get('exception_type'),
            'tool_audit': tool_audit,
            'environment_seconds': elapsed(trial.get('environment_setup')),
            'agent_seconds': elapsed(trial.get('agent_execution')),
            'verifier_seconds': elapsed(trial.get('verifier')), 'total_seconds': elapsed(trial),
            'native_setup_seconds': worker.get('setup_seconds'),
            'native_worker_seconds': worker.get('total_worker_seconds'),
            'merged_output_exists': worker.get('merged_output_exists'),
            'implementation_returncode': worker.get('implementation_returncode'),
            'usage': worker.get('usage') or raw_usage(path.parent / 'agent/supervisor/state/implementation_logs'),
            'final_commit': worker.get('final_commit')})
    job_result = json.loads((job / 'result.json').read_text())
    return {'job': job.name, 'valid_for_performance_comparison': invalid is None and not any(row['tool_audit']['outside_file_profile'] for row in rows),
        'integrity_notice': invalid, 'concurrent_trials': concurrency, 'trials': rows, 'passed': sum(row['reward'] == 1 for row in rows),
        'attempted': len(rows), 'errors': sum(bool(row['error_type']) for row in rows),
        'job_seconds': elapsed(job_result), 'limitations': [
            'Two selected tasks and one attempt each; not a general Terminal-Bench score.',
            'Host supervisor and native worker use a workspace bridge into Harbor task containers.',
            'Native public smoke validation is distinct from official verifier reward.',
            'Raw input/cache/output counters retained; no assumed inclusive total or dollar cost.',
            f'{concurrency} concurrent Harbor trials; no matched-baseline claim of token savings or parallel speedup.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    report = summarize(args.job)
    (args.job / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    def number(value):
        return '—' if value is None else f'{value:,.1f}'
    lines = [f'# Supervisor Terminal-Bench pilot: {report["job"]}', '',
        f'Official verifier passes: **{report["passed"]}/{report["attempted"]}**; infrastructure errors: {report["errors"]}.', '',
        '| Task | Reward | Agent seconds | Setup seconds | Verifier seconds | Error |',
        '|---|---:|---:|---:|---:|---|']
    for row in report['trials']:
        lines.append(f'| {row["task"]} | {row["reward"]} | {number(row["agent_seconds"])} | {number(row["environment_seconds"])} | {number(row["verifier_seconds"])} | {row["error_type"] or "—"} |')
    if report['integrity_notice']:
        lines += ['', '**Invalid for performance comparison:** ' + report['integrity_notice']['reason']]
    lines += ['', '## Native usage counters', '',
        '| Task | Usage events | Input | Cache read | Output | Reasoning |',
        '|---|---:|---:|---:|---:|---:|']
    for row in report['trials']:
        usage = row['usage'] or {}
        raw = usage.get('raw_token_counters', {})
        values = [usage.get('events'), *(raw.get(k) for k in ['input_tokens', 'cache_read_input_tokens', 'output_tokens', 'reasoning_tokens'])]
        lines.append('| ' + row['task'] + ' | ' + ' | '.join('—' if v is None else str(v) for v in values) + ' |')
    lines += ['', '## Interpretation', '', *['- ' + s for s in report['limitations']], '']
    (args.job / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
