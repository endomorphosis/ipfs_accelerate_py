"""Offline authored fixture: numerical preparation, then a fresh-process replay.

RSS observations are measurements, never a general memory bound. No provider,
benchmark input, verifier, solution, or model response is used or exported.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

P = Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')
D = Path('/home/barberb/lift_coding/.worktrees/terminal-bounded-header-datasets-20261005')
CONFIG = Path('/home/barberb/lift_coding/artifacts/terminal-suite-pilot-20261004/source384-config.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def phase(mode, directory):
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as owner
    started = time.monotonic()
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import get_global_resource_scheduler
    scheduler = get_global_resource_scheduler()
    profile = scheduler.config.persisted_dict()
    if mode == 'prepare':
        from test.api.test_terminal_planner_media_contract import _prepare_catalog_shape
        prepared = _prepare_catalog_shape(directory, 'tune-mjcf')
        sources = {name: item['sha256'] for name, item in prepared['manifest']['payload']['sources'].items()}
        receipt = owner.prepare_source384_context(repository=Path(prepared['repository']),
            source_hashes=sources, manifest_envelope=prepared['manifest'],
            config_path=CONFIG, output=directory / 'source384-context', timeout_seconds=180., scheduler=scheduler)
        write(directory / 'expected-receipt.json', receipt)
        inference = json.loads((Path(receipt['output']) / 'inference.json').read_bytes())
        forward = {'native_worker_executed': inference['native_worker_executed'],
            'inference_executed': inference['inference_executed'],
            'model_loads': inference['report']['output']['model_loads']}
        qualified = forward == {'native_worker_executed': True, 'inference_executed': True, 'model_loads': 1}
    else:
        from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as units
        from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as decoder
        forward_calls = []
        def forbidden(*args, **kwargs):
            forward_calls.append(True)
            raise AssertionError('cold replay attempted neural execution')
        units._worker = forbidden
        decoder.Runtime.infer = forbidden
        receipt = json.loads((directory / 'expected-receipt.json').read_bytes())
        observed = owner.validate_source384_context(repository=Path(receipt['repository']),
            expected_receipt=receipt, scheduler=scheduler, timeout_seconds=90.)
        qualified = observed == receipt and not forward_calls
        forward = {'new_numerical_worker_calls': len(forward_calls),
            'new_neural_forward_calls': len(forward_calls), 'forward_guard_installed': True,
            'checkpoint_validation_and_array_allocations_still_run': True}
    snapshot = scheduler.snapshot()
    report = {'schema': 'source384-native-memory-phase@1', 'phase': mode,
        'qualified': qualified, 'seconds': time.monotonic() - started,
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'child_process_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'checkpoint_sha256': receipt['checkpoint_sha256'],
        'inference_artifact_bytes': (Path(receipt['output']) / 'inference.json').stat().st_size,
        'source_files': len(receipt['source_hashes']),
        'native_execution': forward,
        'scheduler_total_memory_mb': profile['total_memory_mb'],
        'scheduler_headroom_mb': profile['proof_memory_headroom_mb'],
        'active_leases_after': snapshot['active_lease_count'],
        'waiting_requests_after': snapshot['waiting_request_count'],
        'admission_memory_bounds_unchanged': True, 'source_currentness_verified': qualified,
        'provider_calls': 0, 'completion_authority': False}
    write(directory / (mode + '-phase.json'), report)
    if not qualified or snapshot['active_lease_count'] or snapshot['waiting_request_count']:
        raise RuntimeError('native memory qualification did not complete')


def sample_phase(mode, directory, env):
    import psutil
    command = [sys.executable, '-B', str(Path(__file__).resolve()), mode, str(directory)]
    peak_tree_rss = 0
    sample_count = 0
    timed_out = False
    started = time.monotonic()
    with (directory / (mode + '.private.log')).open('xb') as log:
        proc = subprocess.Popen(command, cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
                total = 0
                for process in processes:
                    try:
                        total += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak_tree_rss = max(peak_tree_rss, total)
                sample_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.monotonic() - started > (240 if mode == 'prepare' else 120):
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(.02)
        code = proc.wait(timeout=10)
    child = directory / (mode + '-phase.json')
    return {'exit_code': code, 'timed_out': timed_out,
        'wall_seconds': time.monotonic() - started,
        'sampled_process_tree_peak_rss_bytes': peak_tree_rss,
        'rss_samples': sample_count, 'sample_interval_seconds': .02,
        'phase_receipt': json.loads(child.read_text()) if child.exists() else None,
        'private_log_exported': False}


def run(directory):
    directory.mkdir(mode=0o700)
    env = {**os.environ, 'PYTHONPATH': str(P) + ':' + str(D), 'PYTHONDONTWRITEBYTECODE': '1',
        'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'IPFS_DATASETS_PROOF_RESOURCE_PROFILE': 'local-benchmark@1',
        'IPFS_DATASETS_RESOURCE_SCHEDULER_PATH': str(directory / 'resources.json')}
    head = subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip()
    source_paths = [P / 'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
        D / 'ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py',
        D / 'ipfs_datasets_py/logic/software_contracts/codebase_source_384.py']
    before = {str(path.relative_to(P if path.is_relative_to(P) else D)): sha(path) for path in source_paths}
    phases = {'prepare': sample_phase('prepare', directory, env)}
    if phases['prepare']['exit_code'] == 0:
        phases['replay'] = sample_phase('replay', directory, env)
    after = {str(path.relative_to(P if path.is_relative_to(P) else D)): sha(path) for path in source_paths}
    report = {'schema': 'source384-cold-replay-memory-observation@1', 'accelerate_head': head,
        'datasets_head': subprocess.check_output(['git', '-C', str(D), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': before, 'relevant_source_unchanged': before == after,
        'script_sha256': sha(__file__), 'config_sha256': sha(CONFIG), 'phases': phases,
        'qualified': before == after and len(phases) == 2 and all(row['exit_code'] == 0 and
            row['phase_receipt']['qualified'] is True for row in phases.values()),
        'fixture': 'authored tune-mjcf catalog shape with one Python function and minimal XML',
        'fresh_process_per_phase': True, 'os_file_cache_reset': False,
        'runs_in_benchmark_container': False, 'memory_guarantee_or_general_upper_bound': False,
        'limitations': ['RSS includes imports; process-tree RSS may double-count shared resident pages',
            '20ms samples may miss transients; Linux ru_maxrss records per-process maxima',
            'Replay retains checkpoint validation, NumPy allocation and immutable artifact reconstruction',
            'One small authored corpus cannot justify reducing generic allowed memory envelopes'],
        'initial_and_replay_parent_memory_mb': 6144, 'child_memory_mb': 4096,
        'provider_calls': 0, 'benchmark_task_completed': False, 'official_reward': None,
        'model_or_source_bodies_exported': False, 'completion_authority': False}
    write(directory / 'memory-observation.json', report)
    print(json.dumps({'qualified': report['qualified'], 'phases': {
        name: {key: value for key, value in row.items() if key != 'phase_receipt'} for name, row in phases.items()}}))
    if not report['qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    mode, destination = sys.argv[1:]
    if mode == 'run':
        run(Path(destination))
    else:
        phase(mode, Path(destination))
