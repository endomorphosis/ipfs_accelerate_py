"""Exercise admitted START and a real isolated worker without any model calls.

The input is an already verified public-task admission. A fresh native database
keeps this diagnostic attempt separate from any benchmark attempt or custody.
The worker writes an undeclared scratch file; task completion is not expected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _failure_diagnostics, _native_diagnostics
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import (
    materialize_local_benchmark_plan,
    verify_local_benchmark_admission,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE

ROOT = Path('/opt/ipfs-supervisor')


def _workers() -> list[dict]:
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)
            if int(fields['Uid'].split()[0]) != 1001 or fields['State'].strip().startswith('Z'):
                continue
            argv = (path / 'cmdline').read_bytes().split(b'\0')
            entry = str(ROOT / 'bin/worker-entry').encode()
            coding_preflight = (entry in argv and argv[argv.index(entry) + 1:][:3]
                                == [b'--preflight', b'--preflight-sleep', b'20'])
            result.append({'pid': int(path.name), 'uid': 1001, 'parent_pid': int(fields['PPid']),
                           'state': fields['State'].strip().split()[0], 'coding_preflight': coding_preflight})
        except (OSError, ValueError, KeyError, ProcessLookupError):
            continue
    return sorted(result, key=lambda item: item['pid'])


def _preflight_receipt(state: Path) -> dict | None:
    for path in (state / 'run').glob('admitted_database_portal_attempts/*/implementation-logs/*-attempt-*.log'):
        if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        for line in path.read_text(errors='replace').splitlines():
            if not line.startswith('{'):
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get('schema') == 'container-worker-preflight@1':
                return value
    return None


def qualify(*, prepared_state: Path, output: Path, full_context: bool = False) -> dict:
    if os.getuid() != 1000 or os.geteuid() != 1000:
        raise ValueError('the actual deployed supervisor owner identity is required')
    output = output.absolute()
    if output.exists() or not output.is_relative_to(ROOT / 'state'):
        raise ValueError('a fresh private container state directory is required')
    if _workers():
        raise ValueError('the isolated single-worker container must be idle')
    output.mkdir(mode=0o700)
    admission = json.loads((prepared_state / 'admission.json').read_text())
    verified = verify_local_benchmark_admission(admission, initial=True)
    if len(verified['graph'].tasks) != 1:
        raise ValueError('exactly one admitted task is required')
    task = verified['graph'].tasks[0]
    database = output / 'intent.duckdb'
    with IntentRepository(database) as intent:
        materialized = materialize_local_benchmark_plan(admission=admission, intent=intent)
    report = {'schema': 'terminal-native-start-preflight@1', 'qualified': False,
              'text_generation_calls': 0, 'official_verifier_executed': False,
              'benchmark_success': None, 'task_completion_expected': False,
              'materialized': materialized, 'observations': []}

    def save():
        (output / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')

    def record_failure(error, phase):
        report['qualified'] = False
        diagnostic = _failure_diagnostics(error, phase=phase)
        detail = {'type': type(error).__name__, 'message': str(error)[:1024]}
        if 'error' not in report:
            report['error'] = detail
            report.update(diagnostic)
        else:
            report.setdefault('cleanup_errors', []).append({**detail, **diagnostic})

    started = time.monotonic()
    phase = 'owner_open'
    try:
        with open_existing_native_owner(database=database, checkout=Path(verified['manifest']['repository']),
                state_dir=output / 'owner', repository_id=verified['manifest']['repository_cid'],
                execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE}) as owner:
            rebound = None
            if full_context:
                phase = 'context_rebind'
                from .terminal_context_rebind import rebind_full_context
                rebound = rebind_full_context(prepared_state=prepared_state, admission=admission,
                    server=owner.server, output=Path(verified['manifest']['repository']) / '.runtime' / output.name)
                report['full_context'] = rebound
            phase = 'runtime_create'
            runtime = AdmittedBenchmarkRuntime.create(output / 'launch', admission=admission,
                server=owner.server, source=owner.source, implement=True,
                implementation_command=str(ROOT / 'bin/router-worker') + ' --preflight --preflight-sleep 20',
                timeout_ms=20_000, lifetime_seconds=180, max_task_attempts=1,
                worker_worktree_root=ROOT / 'worktrees',
                context_bundle=rebound['context_bundle'] if rebound else None,
                candidate_runner_argv=(str(ROOT / 'bin/validation-worker'),))
            try:
                phase = 'runtime_git'
                git = subprocess.run(['git', '-C', str(ROOT / 'source'), 'rev-parse', '--verify', 'HEAD'],
                    env={**os.environ, **dict(runtime.manifest['environment'])},
                    text=True, capture_output=True, timeout=10)
                report['installed_runtime_git'] = {'returncode': git.returncode,
                    'revision': git.stdout.strip() if git.returncode == 0 else None,
                    'git_dubious_ownership': 'detected dubious ownership' in git.stderr}
                if git.returncode:
                    raise RuntimeError('signed isolated Git environment cannot observe installed runtime')
                phase = 'start'
                report['start'] = runtime.start().to_dict()
                report['bootstrap_receipts'] = runtime.bootstrap_receipts
                report['bootstrap_errors'] = runtime.bootstrap_errors
                save()
                if not report['start']['status'] == 'succeeded':
                    raise RuntimeError('native START did not prove sustained health')
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    phase = 'observe'
                    observation = runtime.observe()
                    phase = 'worker_observation'
                    workers = _workers()
                    receipt = _preflight_receipt(runtime.state)
                    coding_workers = [worker for worker in workers if worker['coding_preflight']]
                    if (coding_workers and receipt and receipt.get('prompt_bytes', 0) > 0
                            and any(worker['pid'] == receipt.get('pid') for worker in coding_workers)):
                        report['worker_preflight_receipt'] = receipt
                        path = Path(receipt['workspace']) / 'worker-preflight-prompt.txt'
                        if (not path.is_relative_to(ROOT / 'worktrees') or path.resolve() != path
                                or path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024):
                            raise ValueError('worker prompt observation escaped its allocation')
                        prompt = path.read_bytes()
                        if len(prompt) != receipt['prompt_bytes'] or hashlib.sha256(prompt).hexdigest() != receipt['prompt_sha256']:
                            raise ValueError('worker stdin observation differs from its receipt')
                        if rebound is not None:
                            from .terminal_context_rebind import verify_worker_context_prompt
                            report['worker_context_observation'] = verify_worker_context_prompt(
                                prompt=prompt.decode(), rebound=rebound)
                        report['observations'].append({'healthy': observation['healthy'],
                            'native_heartbeat': observation['native_heartbeat'], 'workers': workers,
                            'process_tree': observation['process_tree']})
                        save()
                        if (len(report['observations']) >= 2 and observation['healthy']
                                and report['observations'][0]['healthy']
                                and observation['native_heartbeat']['owner_read_sequence']
                                    > report['observations'][0]['native_heartbeat']['owner_read_sequence']):
                            report['observations'] = [report['observations'][0], report['observations'][-1]]
                            break
                    time.sleep(.75)
                phase = 'task_state'
                state = owner.source.get_task(task.task_cid)
                report['task_state'] = {'status': state.status, 'revision': state.revision}
            except Exception as error:
                # Preserve the request's admission sample before cleanup changes
                # the ledger; post-unwind samples remain labelled separately.
                record_failure(error, phase)
                raise
            finally:
                try:
                    phase = 'stop'
                    report['stop'] = runtime.stop().to_dict()
                    report['remaining_native_processes'] = len(runtime.process.snapshot(runtime.profile).members)
                    deadline = time.monotonic() + 3
                    while _workers() and time.monotonic() < deadline:
                        time.sleep(.1)
                    report['remaining_worker_processes'] = _workers()
                    report['native_diagnostics'] = _native_diagnostics(runtime.state)
                    save()
                except Exception as error:
                    had_primary_failure = 'error' in report
                    record_failure(error, phase)
                    if not had_primary_failure:
                        raise
                finally:
                    try:
                        runtime.close()
                    except Exception as error:
                        had_primary_failure = 'error' in report
                        record_failure(error, 'close')
                        if not had_primary_failure:
                            raise
        observations = report['observations']
        report['qualified'] = bool('error' not in report and len(observations) == 2
            and all(row['healthy'] and row['workers'] for row in observations)
            and observations[1]['native_heartbeat']['owner_read_sequence'] > observations[0]['native_heartbeat']['owner_read_sequence']
            and report['stop']['status'] == 'succeeded'
            and report['remaining_native_processes'] == 0 and not report['remaining_worker_processes']
            and (not full_context or bool(report.get('worker_context_observation')))
            and report['task_state']['status'] != 'completed')
    except Exception as error:
        if 'error' not in report:
            record_failure(error, phase)
    finally:
        # Record fallback separately; it cannot turn a failed STOP into success.
        try:
            if _workers():
                report['qualified'] = False
                cleanup = subprocess.run(['sudo', '-n', '-u', 'benchmarkworker', '--',
                    str(ROOT / 'bin/worker-entry'), '--cleanup'], cwd='/', capture_output=True, timeout=15)
                report['fallback_cleanup_returncode'] = cleanup.returncode
        except Exception as error:
            record_failure(error, 'fallback_cleanup')
        try:
            report['workers_after_finally'] = _workers()
            if report['workers_after_finally']:
                report['qualified'] = False
        except Exception as error:
            record_failure(error, 'final_worker_observation')
        report['seconds'] = time.monotonic() - started
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-context', action='store_true')
    args = parser.parse_args()
    result = qualify(prepared_state=args.prepared_state, output=args.output, full_context=args.full_context)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
