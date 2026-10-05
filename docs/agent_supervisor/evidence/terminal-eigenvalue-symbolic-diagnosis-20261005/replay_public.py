"""Prespecified, zero-model, public-input eigenvalue diagnostic."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import sysconfig
import time

import numpy as np

sys.path.insert(0, '/app')
ROOT = Path('/diagnostic')
PLAN = json.loads((ROOT / 'plan.json').read_text())
OUT = ROOT / 'output'


def digest(path):
    path = Path(path)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha)


def timeout(*_):
    raise TimeoutError('prespecified 300-second diagnostic limit reached')


def optional_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def measure(function, matrix):
    before = time.perf_counter_ns()
    function(matrix)
    return time.perf_counter_ns() - before


def main():
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, PLAN['limits']['runtime_seconds'])
    started = time.perf_counter()
    assert platform.python_version_tuple()[:2] == ('3', '13')
    assert np.__version__ == PLAN['numpy_version']
    assert digest('/app/eigen.py')['sha256'] == PLAN['candidate_sha256']
    environment = {
        'python': sys.version,
        'numpy_version': np.__version__,
        'architecture': platform.machine(),
        'thread_environment': {key: os.environ.get(key) for key in (
            'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')},
        'cpu_max': optional_text('/sys/fs/cgroup/cpu.max'),
        'memory_max': optional_text('/sys/fs/cgroup/memory.max'),
        'memory_swap_max': optional_text('/sys/fs/cgroup/memory.swap.max'),
        'cpu_pressure_before': optional_text('/proc/pressure/cpu'),
        'gcc_version': subprocess.run(['gcc', '--version'], capture_output=True,
                                      text=True, check=True).stdout.splitlines()[0],
        'sysconfig_cc': sysconfig.get_config_var('CC'),
        'numpy_build_dependencies': np.__config__.CONFIG.get('Build Dependencies'),
    }
    compile_calls = []
    original_run = subprocess.run

    def observe_compile(command, **kwargs):
        before = time.perf_counter()
        result = original_run(command, **kwargs)
        compile_calls.append({'argv': list(command), 'returncode': result.returncode,
                              'seconds': time.perf_counter() - before,
                              'source_body_recorded': False})
        return result

    subprocess.run = observe_compile
    before = time.perf_counter()
    try:
        import eigen
    finally:
        subprocess.run = original_run
    import_seconds = time.perf_counter() - before
    solve = eigen.find_dominant_eigenvalue_and_eigenvector
    import eval as public_eval
    reference = public_eval.ref_solution
    symbols = ('scipy_dgeev_64_', 'dgeev_64_', 'scipy_dgeev_', 'dgeev_')
    symbol = next(name for name in symbols if hasattr(eigen._lapack_library, name))
    environment['lapack_symbol'] = symbol
    environment['integer_abi'] = 'ILP64' if '_64_' in symbol else 'LP64'
    environment['numpy_linalg_binary'] = digest(eigen._numpy_lapack.__file__)
    legacy_stream = io.StringIO()
    legacy_started = time.perf_counter()
    np.random.seed(173104)
    with contextlib.redirect_stdout(legacy_stream):
        for size in public_eval.MAT_SIZES:
            public_eval.test_eigen_pair(size)
            public_eval.test_speedup(size)
    legacy = {
        'sizes': public_eval.MAT_SIZES,
        'calls_per_function_size': public_eval.N,
        'elapsed_seconds': time.perf_counter() - legacy_started,
        'stdout': legacy_stream.getvalue(),
        'unchanged_public_producer': digest('/app/eval.py'),
        'unpaired_different_random_inputs': True,
        'speed_assertion_in_public_producer': False,
    }
    correctness = {'checked': 0, 'failures': [], 'max_scaled_residual': 0.0,
                   'family_counts': {}}

    def check(matrix, label):
        saved = matrix.copy()
        value, vector = solve(matrix)
        vals = np.linalg.eigvals(matrix)
        finite = bool(np.isfinite(value) and np.isfinite(vector).all())
        norm = float(np.linalg.norm(vector))
        norm_a = float(np.linalg.norm(matrix))
        residual = float(np.linalg.norm(matrix @ vector - value * vector))
        scale = (norm_a + abs(value)) * norm
        scaled = residual / max(scale, np.finfo(np.float64).tiny)
        tests = {
            'numpy_scalar': isinstance(value, np.generic),
            'shape': vector.shape == (matrix.shape[0],),
            'finite': finite,
            'nonzero': norm > 0,
            'unit_norm': bool(np.isclose(norm, 1, rtol=1e-10, atol=1e-12)),
            'public_allclose_residual': bool(np.allclose(matrix @ vector, value * vector)),
            'scaled_residual': residual <= 1e-10 * scale + 1e-12,
            'dominant_modulus_oracle_agreement': bool(np.isclose(
                abs(value), np.max(np.abs(vals)), rtol=1e-10, atol=1e-12)),
            'input_unchanged': bool(np.array_equal(matrix, saved)),
        }
        correctness['checked'] += 1
        family = label.split(':')[0]
        correctness['family_counts'][family] = correctness['family_counts'].get(family, 0) + 1
        correctness['max_scaled_residual'] = max(correctness['max_scaled_residual'], scaled)
        if not all(tests.values()):
            correctness['failures'].append({'label': label, 'size': matrix.shape[0],
                                            'failed_checks': [key for key, val in tests.items() if not val],
                                            'norm': norm, 'scaled_residual': scaled})

    groups = []
    for seed in PLAN['seeds']:
        for size in PLAN['sizes']:
            rng = np.random.default_rng(np.random.SeedSequence([seed, size]))
            matrices = [rng.normal(size=(size, size)).astype(np.float64)
                        for _ in range(PLAN['matrices_per_seed_size'])]
            for index, matrix in enumerate(matrices):
                check(matrix, f'random:{seed}:{size}:{index}')
            for matrix in matrices[:PLAN['warmup_pairs']]:
                solve(matrix)
                reference(matrix)
            order_rng = random.Random(seed * 100 + size)
            rounds = []
            all_candidate, all_reference = [], []
            for round_index in range(PLAN['rounds']):
                indices = list(range(len(matrices)))
                order_rng.shuffle(indices)
                candidate_ns, reference_ns = [], []
                for index in indices:
                    matrix = matrices[index]
                    if order_rng.getrandbits(1):
                        candidate_ns.append(measure(solve, matrix))
                        reference_ns.append(measure(reference, matrix))
                    else:
                        reference_ns.append(measure(reference, matrix))
                        candidate_ns.append(measure(solve, matrix))
                c = float(np.median(candidate_ns))
                r = float(np.median(reference_ns))
                rounds.append({'round': round_index, 'candidate_median_ns': c,
                               'reference_median_ns': r, 'candidate_divided_by_reference': c / r})
                all_candidate.extend(candidate_ns)
                all_reference.extend(reference_ns)
            c = float(np.median(all_candidate))
            r = float(np.median(all_reference))
            groups.append({'seed': seed, 'size': size, 'rounds': rounds,
                           'candidate_median_ns': c, 'reference_median_ns': r,
                           'candidate_divided_by_reference': c / r,
                           'candidate_p10_p90_ns': np.percentile(all_candidate, [10, 90]).tolist(),
                           'reference_p10_p90_ns': np.percentile(all_reference, [10, 90]).tolist(),
                           'candidate_faster_rounds': sum(x['candidate_divided_by_reference'] < 1 for x in rounds),
                           'matrix_bytes_sha256': hashlib.sha256(b''.join(x.tobytes() for x in matrices)).hexdigest()})

    for size in PLAN['sizes']:
        rng = np.random.default_rng(np.random.SeedSequence([76531, size]))
        diagonal = np.linspace(0.1, 0.9, size)
        diagonal[0] = -4
        structured = {
            'zero': np.zeros((size, size)),
            'identity': np.eye(size),
            'negative_dominant': np.diag(diagonal),
            'jordan': 2 * np.eye(size) + np.diag(np.ones(max(0, size - 1)), 1),
            'triangular': np.triu(rng.normal(size=(size, size))) + np.diag(diagonal),
            'rank_one': np.outer(np.arange(1, size + 1), np.ones(size)),
            'tiny_scaled': 1e-12 * rng.normal(size=(size, size)),
            'large_scaled': 1e12 * rng.normal(size=(size, size)),
        }
        if size >= 2:
            rotation = np.diag(np.linspace(0.1, 0.9, size))
            rotation[:2, :2] = [[1, -3], [3, 1]]
            structured['complex_dominant_pair'] = rotation
        for family, matrix in structured.items():
            matrix = np.asarray(matrix, dtype=np.float64)
            variants = {'C': matrix.copy(order='C'), 'F': matrix.copy(order='F'),
                        'transpose': matrix.T, 'negative_strides': matrix[::-1, ::-1]}
            readonly = matrix.copy()
            readonly.flags.writeable = False
            variants['readonly'] = readonly
            for layout, variant in variants.items():
                check(variant, f'{family}:{size}:{layout}')

    per_size = []
    for size in PLAN['sizes']:
        rows = [x for x in groups if x['size'] == size]
        ratios = [x['candidate_divided_by_reference'] for x in rows]
        per_size.append({'size': size, 'seed_count': len(rows),
                         'candidate_faster_seeds': sum(x < 1 for x in ratios),
                         'ratio_min_median_max': [min(ratios), float(np.median(ratios)), max(ratios)],
                         'candidate_faster_rounds': sum(x['candidate_faster_rounds'] for x in rows),
                         'round_count': PLAN['rounds'] * len(rows)})
    result = {'schema': 'public-eigenvalue-diagnostic-result@1',
              'environment': environment, 'plan': PLAN,
              'candidate_import_seconds': import_seconds,
              'candidate_compile_calls': compile_calls,
              'candidate': digest('/app/eigen.py'), 'instruction': digest('/diagnostic/instruction.md'),
              'producer': digest(__file__), 'legacy_public_even_only': legacy,
              'correctness': correctness, 'paired_timing_groups': groups,
              'paired_timing_per_size': per_size,
              'timer': time.get_clock_info('perf_counter')._asdict()
                       if hasattr(time.get_clock_info('perf_counter'), '_asdict')
                       else str(time.get_clock_info('perf_counter')),
              'cpu_pressure_after': optional_text('/proc/pressure/cpu'),
              'elapsed_seconds': time.perf_counter() - started,
              'interpretation': 'One warmed public diagnostic in a rebuilt runtime. No official score or universal causal claim.',
              'provider_calls': 0, 'official_verifier_executed': False,
              'hidden_inputs_read': False, 'candidate_optimized': False}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'correctness_checked': correctness['checked'],
                      'correctness_failures': len(correctness['failures']),
                      'paired_timing_per_size': per_size,
                      'elapsed_seconds': result['elapsed_seconds']}))
    return 0 if not correctness['failures'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BaseException as error:
        if not isinstance(error, SystemExit):
            (OUT / 'failure.json').write_text(json.dumps({'type': type(error).__name__,
                'message': str(error)[:600], 'provider_calls': 0}) + '\n')
        raise
