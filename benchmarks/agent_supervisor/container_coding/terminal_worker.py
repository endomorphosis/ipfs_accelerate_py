"""Native supervisor subprocess for the two-task Harbor workspace pilot.

Receives only the public instruction. Official tests and solutions never enter
this repository. Local validation is a public-contract smoke check only.
"""
import argparse
import ast
import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
OUTPUTS = {"cancel-async-tasks": "run.py", "polyglot-c-py": "polyglot/main.py.c"}
SMOKES = {
    "cancel-async-tasks": '''import asyncio
from run import run_tasks
async def check():
    active = peak = completed = 0
    async def job():
        nonlocal active, peak, completed
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            completed += 1
        finally:
            active -= 1
    await run_tasks([job] * 5, 2)
    assert completed == 5 and peak <= 2 and active == 0
asyncio.run(check())
''',
    "polyglot-c-py": '''import subprocess
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    binary = str(Path(directory) / 'cmain')
    subprocess.run(['gcc', 'polyglot/main.py.c', '-o', binary], check=True)
    for n, expected in [(0, '0'), (1, '1'), (10, '55')]:
        for cmd in [['python3', 'polyglot/main.py.c'], [binary]]:
            result = subprocess.check_output(cmd + [str(n)], text=True).strip()
            assert result == expected, (cmd, n, result)
''',
}


def raw_usage(log_dir):
    """Sum only native usage records; retain field semantics without guessing cost."""
    totals = {}
    events = 0
    for path in log_dir.glob('*.log'):
        for line in path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict) or event.get('type') != 'usage':
                continue
            usage = event.get('usage', event)
            numeric = {k: v for k, v in usage.items()
                       if 'token' in k and isinstance(v, (int, float)) and not isinstance(v, bool)}
            if numeric:
                events += 1
                for key, value in numeric.items():
                    totals[key] = totals.get(key, 0) + value
    return {'events': events, 'raw_token_counters': totals,
            'cost_usd': None, 'normalized_total_tokens': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=OUTPUTS, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--instruction', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--semantic-context', action='store_true', help='Prepare and require source-bound semantic context; this is not the full indexed benchmark arm')
    parser.add_argument('--datasets-source', type=Path, default=SOURCE.parent / 'ipfs_datasets')
    parser.add_argument('--kit-source', type=Path, default=SOURCE.parent / 'ipfs_kit')
    args = parser.parse_args()
    started = time.monotonic()
    root = args.root.resolve()
    repo, state = root / 'repo', root / 'state'
    # Native retry/receipt storage can outlive a Git checkout. Independent
    # benchmark trials must never share semantic goal and task identities.
    task_prefix = 'TB' + hashlib.sha256(str(root).encode()).hexdigest()[:12].upper()
    goal_prefix = task_prefix + '-G'
    repo.mkdir(parents=True, exist_ok=False)
    dependency_roots = [SOURCE]
    if args.semantic_context:
        for dependency in (args.datasets_source, args.kit_source):
            dependency = dependency.resolve(strict=True)
            dependency_roots.append(dependency)
            sys.path.append(str(dependency))
    os.environ['PYTHONPATH'] = os.pathsep.join(map(str, dependency_roots))
    os.environ['IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER'] = 'grok'
    os.environ['IPFS_ACCELERATE_AGENT_GROK_MODEL'] = 'grok-4.7'
    os.environ['IPFS_ACCELERATE_AGENT_GROK_TASK_TOOL_PROFILE'] = 'files'
    output = OUTPUTS[args.task]
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(repo), *argv], text=True).strip()
    (repo / 'README.md').write_text(args.instruction.read_text() + '\n\n'
        'Workspace adapter: this Git repository corresponds to /app in the task container. '
        'Use repository-relative paths here. Implement the complete public instruction above. '
        f'The only deliverable is {output}. Do not change README.md or public_smoke.py. '
        'Local smoke tests are incomplete; the external benchmark verifier decides success. '
        'Use only the provided instruction and local workspace files. Do not retrieve upstream '
        'benchmark tests, solutions, or other agents outputs. Shell and web tools are disabled; '
        'the supervisor runs validation after you edit the deliverable.\n')
    (repo / 'public_smoke.py').write_text(SMOKES[args.task])
    (repo / '.gitignore').write_text('__pycache__/\n.runtime/\n')
    git('init', '-b', 'main')
    git('config', 'user.name', 'Supervisor Benchmark')
    git('config', 'user.email', 'benchmark@localhost')
    git('add', '.')
    git('commit', '-m', 'Seed public Terminal-Bench instruction and smoke check')
    baseline = git('rev-parse', 'HEAD')
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import ensure_objective_tracking_document
    objective = repo / 'objectives.md'
    ensure_objective_tracking_document(objective,
        ultimate_goal=f'Implement {output} to satisfy every requirement in README.md. Read README.md before editing.',
        root_evidence=[f'{args.task} implementation'], goal_prefix=goal_prefix,
        root_goal_title=f'Complete {args.task}')
    objective.write_text(objective.read_text().replace(
        'Outputs: ipfs_accelerate_py/agent_supervisor, docs', f'Outputs: {output}').replace(
        'Validation: test -f ' + str(objective), 'Validation: python3 public_smoke.py'))
    command = [sys.executable, '-m', 'ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon',
        '--repo-root', str(repo), '--objective-path', 'objectives.md', '--todo-path', 'tasks.todo.md',
        '--discovery-dir', '.runtime/discovery', '--bundle-dir', '.runtime/bundles',
        '--dataset-dir', '.runtime/dataset', '--graph-path', '.runtime/graph.json',
        '--objective-generation-path', '.runtime/generation.json', '--task-prefix', task_prefix,
        '--goal-prefix', goal_prefix, '--refine-objective-heap', '--max-refinement-children', '1',
        '--max-refinement-depth', '1', '--max-findings', '1', '--surplus-findings-per-goal', '1',
        '--no-persist-ast-dataset', '--no-todo-vector-index', '--no-reconcile-goal-completion',
        '--objective-generation-max-new-work', '1', '--objective-generation-max-open-work', '1']
    generated = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=120)
    (root / 'generation.json').write_text(generated.stdout)
    (root / 'generation.log').write_text(generated.stderr)
    generated.check_returncode()
    from portable_context import localize_context
    localized_context_files = localize_context(repo)
    semantic_results = []
    semantic_artifacts = []
    if args.semantic_context:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file
        board = repo / 'tasks.todo.md'
        text = board.read_text()
        for task in parse_task_file(board, task_header_prefix='## ' + task_prefix + '-'):
            prepared = root / 'semantic' / task.task_id
            result = prepare_semantic_context(repository=repo,
                paths=['README.md', 'public_smoke.py'],
                required_raw_paths=['README.md', 'public_smoke.py'],
                objective=task.title, task_id=task.task_id, output=prepared)
            relative = '.runtime/semantic/' + task.task_id + '.json'
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prepared / 'worker-context.json', destination)
            pattern = r'(?m)^(## ' + re.escape(task.task_id) + r'(?=\s|$)[^\n]*)$'
            text, count = re.subn(pattern, lambda m: m.group(0) +
                '\n- Semantic context artifact: ' + relative +
                '\n- Semantic context sha256: ' + result['worker_payload_sha256'], text)
            if count != 1:
                raise RuntimeError('semantic task binding is absent or ambiguous')
            semantic_artifacts.append(relative)
            semantic_results.append(result)
        if not semantic_results:
            raise RuntimeError('semantic context requested but no generated tasks found')
        board.write_text(text)
        git('add', '-f', *semantic_artifacts)
        (root / 'semantic-preparation.json').write_text(json.dumps(semantic_results, indent=2) + '\n')
    git('add', 'objectives.md', 'tasks.todo.md', 'data')
    context = sorted((repo / '.runtime/discovery').glob('*.md')) + sorted((repo / '.runtime/bundles').glob('*.md'))
    if context:
        git('add', '-f', *(str(p.relative_to(repo)) for p in context))
    git('commit', '-m', 'Generate native objective and task packet')
    seeded = git('rev-parse', 'HEAD')
    setup_seconds = time.monotonic() - started
    if args.prepare_only:
        print(json.dumps({'repository': str(repo), 'task_prefix': task_prefix,
                          'goals': json.loads(generated.stdout)['objective_goal_count'], 'semantic_contexts': len(semantic_results)}))
        return
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import run_configured_portal_implementation_daemon
    argv = ['--todo-path', str(repo / 'tasks.todo.md'), '--task-source-kind', 'legacy-markdown',
        '--explicit-legacy-task-source', '--state-dir', str(state), '--state-prefix', 'terminal',
        '--task-prefix', '## ' + task_prefix + '-', '--board-namespace', f'terminal-{args.task}-{root.name}',
        '--implement', '--implementation-timeout', '300', '--max-task-attempts', '1',
        '--worktree-root', str(state / 'worktrees'), '--merge-target-branch', 'main',
        '--retain-worktree-artifacts', '--merged-worktree-cleanup-max', '0',
        '--implementation-protected-path', 'README.md', '--implementation-protected-path', 'public_smoke.py',
        '--objective-path', str(objective), '--objective-bundle-dir', str(repo / '.runtime/bundles'),
        '--objective-scan-max-findings', '0', '--codebase-scan-max-findings', '0',
        '--once', '--log-level', 'INFO']
    for artifact in semantic_artifacts:
        argv.extend(['--implementation-protected-path', artifact])
    os.chdir(repo)
    run_configured_portal_implementation_daemon(argv, repo_root=repo, logger=logging.getLogger('terminal.supervisor'))
    native_pass = {}
    for line in (root / 'daemon.log').read_text(errors='replace').splitlines():
        if 'pass complete: ' in line:
            native_pass = ast.literal_eval(line.split('pass complete: ', 1)[1])
    (root / 'native-pass.json').write_text(json.dumps(native_pass, indent=2, default=str) + '\n')
    implementation = native_pass.get('implementation_result') or {}
    result = {'task': args.task, 'provider': 'grok', 'model': 'grok-4.7',
        'implementation_returncode': implementation.get('returncode'),
        'selection_idle_reason': native_pass.get('selection_idle_reason'),
        'provider_dispatched': implementation.get('provider_dispatched'),
        'tool_profile': 'files', 'task_prefix': task_prefix,
        'baseline_commit': baseline, 'seeded_commit': seeded, 'final_commit': git('rev-parse', 'HEAD'),
        'semantic_contexts': semantic_results, 'full_indexed_arm': False,
        'localized_context_files': localized_context_files, 'setup_seconds': setup_seconds, 'total_worker_seconds': time.monotonic() - started,
        'output': output, 'merged_output_exists': (repo / output).is_file(),
        'usage': raw_usage(state / 'implementation_logs'),
        'goal_generation': json.loads(generated.stdout),
        'validation_scope': 'public smoke check; Harbor verifier is independent benchmark authority'}
    (root / 'worker-result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
