"""Run the two-task native-supervisor Harbor pilot and retain provenance.

Harbor must be installed in a separate environment; this interpreter needs the
supervisor's dependencies. Dataset tasks and official verifiers stay unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from terminal_report import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harbor', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--jobs-dir', type=Path, required=True)
    parser.add_argument('--job-name', required=True)
    parser.add_argument('--concurrency', type=int, choices=[1, 2], default=1)
    parser.add_argument('--semantic-context', action='store_true', help='Use the semantic-context integration arm, not the full indexed arm')
    args = parser.parse_args()
    harness = Path(__file__).resolve().parent
    source = harness.parents[2]
    dataset, jobs = args.dataset.resolve(), args.jobs_dir.resolve()
    job = jobs / args.job_name
    if job.exists():
        parser.error('job already exists; choose a new name to preserve prior attempts')
    jobs.mkdir(parents=True, exist_ok=True)
    selected = ['cancel-async-tasks', 'polyglot-c-py']
    for task in selected:
        if not (dataset / task / 'task.toml').exists():
            parser.error(f'missing task: {task}')
    def git(root, *argv):
        return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    source_patch = git(source, 'diff')
    harness_bytes = {p.name: p.read_bytes() for p in harness.glob('*.py')}
    provenance = {
        'dataset_commit': git(dataset, 'rev-parse', 'HEAD'),
        'dataset_dirty': bool(git(dataset, 'status', '--porcelain')),
        'dataset_url': 'https://github.com/harbor-framework/terminal-bench-2',
        'supervisor_commit': git(source, 'rev-parse', 'HEAD'),
        'supervisor_dirty': bool(git(source, 'status', '--porcelain')),
        'arm': 'legacy-daemon-semantic-context' if args.semantic_context else 'legacy-daemon',
        'full_indexed_arm': False,
        'provider': 'grok', 'model': 'grok-4.7', 'concurrent_trials': args.concurrency,
        'tool_profile': 'files', 'trial_identity': 'distinct goal/task prefix per trial',
        'attempts_per_task': 1, 'implementation_timeout_seconds': 300,
        'host_architecture': os.uname().machine,
        'environment_build': 'unchanged task Dockerfiles, native architecture (--force-build)',
        'task_sha256': {str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for task in selected for p in (dataset / task).rglob('*')
                        if p.is_file() and 'solution' not in p.parts},
        'harness_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in harness.glob('*.py')},
    }
    command = [str(args.harbor.resolve()), 'run', '-p', str(dataset),
        '-i', selected[0], '-i', selected[1], '-a', 'harbor_agent:SupervisorWorkspaceAgent',
        '-m', 'grok-4.7', '-n', str(args.concurrency), '--n-attempts', '1', '--max-retries', '0',
        '--force-build', '--jobs-dir', str(jobs), '--job-name', args.job_name]
    env = {**os.environ, 'PYTHONPATH': str(harness), 'SUPERVISOR_BENCH_PYTHON': sys.executable,
           'SUPERVISOR_BENCH_SEMANTIC_CONTEXT': '1' if args.semantic_context else '0'}
    provenance['command'] = command
    (jobs / f'{args.job_name}.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    result = subprocess.run(command, env=env)
    if job.exists():
        (job / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        (job / 'supervisor-source.patch').write_text(source_patch + '\n')
        snapshot = job / 'harness_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name, content in harness_bytes.items():
            (snapshot / name).write_bytes(content)
    if (job / 'result.json').exists():
        subprocess.run([sys.executable, str(harness / 'terminal_report.py'), str(job)], check=True)
        report = summarize(job)
        # Harbor may exit zero even when all trials raised infrastructure errors.
        if not report['valid_for_performance_comparison'] or report['errors'] or report['attempted'] != len(selected):
            raise SystemExit(2)
        if report['passed'] != len(selected):
            raise SystemExit(1)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
