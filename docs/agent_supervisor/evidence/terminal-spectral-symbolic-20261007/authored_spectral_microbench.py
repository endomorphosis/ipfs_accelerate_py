"""Local authored microbenchmark; never reads Terminal Bench evaluation files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

from ipfs_accelerate_py.agent_supervisor.runtime import spectral_eigen_kernel as kernel


def public_numpy_reference(A):
    # Exact algorithm visible in public environment/src/eigen.py.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]


def matrices():
    rng = np.random.default_rng(20261007)
    result = []
    for n in range(1, 11):
        result.append((f"general_{n}", rng.normal(size=(n, n)).astype(np.float64)))
        result.append((f"triangular_{n}", np.triu(rng.normal(size=(n, n))).astype(np.float64)))
        if n >= 2:
            matrix = np.eye(n, dtype=np.float64)
            matrix[:2, :2] = [[0., -3.], [3., 0.]]
            result.append((f"complex_pair_{n}", matrix))
    result.extend([
        ("large_rotation_2", np.array([[0., -2.], [2., 0.]]) * 1e150),
        ("tiny_rotation_2", np.array([[0., -2.], [2., 0.]]) * 1e-150),
    ])
    return result


def main(output_path):
    setup_start = time.perf_counter_ns()
    source = kernel.candidate_source()
    namespace = {"__name__": "authored_microbench_candidate"}
    exec(compile(source, "authored_eigen.py", "exec"), namespace)
    setup_ns = time.perf_counter_ns() - setup_start
    candidate = namespace["find_dominant_eigenvalue_and_eigenvector"]
    import scipy
    repeats, calls_per_batch, warmups = 7, 400, 30
    records = []
    for case_id, matrix in matrices():
        snapshot = matrix.tobytes()
        candidate_check = kernel.validate_eigenpair(matrix, *candidate(matrix))
        baseline_check = kernel.validate_eigenpair(matrix, *public_numpy_reference(matrix))
        if not candidate_check.accepted or not baseline_check.accepted:
            raise RuntimeError("authored numerical qualification failed: " + case_id)
        for _ in range(warmups):
            candidate(matrix)
            public_numpy_reference(matrix)
        samples = {"candidate": [], "public_numpy_reference": []}
        for repeat in range(repeats):
            order = (("candidate", candidate), ("public_numpy_reference", public_numpy_reference))
            if repeat % 2:
                order = order[::-1]
            for name, function in order:
                started = time.perf_counter_ns()
                for _ in range(calls_per_batch):
                    function(matrix)
                samples[name].append((time.perf_counter_ns() - started) / calls_per_batch)
        if matrix.tobytes() != snapshot:
            raise RuntimeError("authored input mutation: " + case_id)
        candidate_ns = statistics.median(samples["candidate"])
        baseline_ns = statistics.median(samples["public_numpy_reference"])
        records.append({
            "case_id": case_id, "n": len(matrix), "candidate_median_ns": candidate_ns,
            "public_numpy_reference_median_ns": baseline_ns,
            "candidate_to_reference_ratio": candidate_ns / baseline_ns,
            "candidate_faster_in_this_local_case": candidate_ns < baseline_ns,
            "candidate_samples_ns": samples["candidate"],
            "public_numpy_reference_samples_ns": samples["public_numpy_reference"],
            "candidate_numeric_accepted": candidate_check.accepted,
            "reference_numeric_accepted": baseline_check.accepted,
            "input_unchanged": True,
        })
    report = {
        "schema": "authored-spectral-microbench@1", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "microbench_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "candidate_compile_import_workspace_setup_ns": setup_ns,
        "full_candidate_function_cost_included": True,
        "candidate_validation_copy_selection_reconstruction_included": True,
        "setup_cost_included_in_per_call_samples": False,
        "setup_cost_reported_separately": True,
        "repeats": repeats, "calls_per_batch": calls_per_batch, "warmups_per_arm_per_case": warmups,
        "case_count": len(records), "candidate_faster_case_count": sum(r["candidate_faster_in_this_local_case"] for r in records),
        "all_authored_cases_faster": all(r["candidate_faster_in_this_local_case"] for r in records),
        "terminal_bench_speed_qualified": False, "target_interpreter_qualified": False,
        "provider_calls": 0, "benchmark_launches": 0,
        "limitations": ["Authored local matrices only; no evaluation files read.",
                        "Timing is local Python3.12/NumPy1.26/SciPy1.17 and cannot certify target Python3.13/NumPy2.3.",
                        "Initial compilation, explicit SciPy import and workspace queries are reported setup costs.",
                        "Numerical cross-check uses independent public NumPy path, not an exact spectral certificate."],
        "cases": records,
    }
    Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("source_sha256", "candidate_compile_import_workspace_setup_ns", "case_count", "candidate_faster_case_count", "all_authored_cases_faster")}, sort_keys=True))
    for record in records:
        print(record["case_id"], round(record["candidate_median_ns"] / 1000, 2), round(record["public_numpy_reference_median_ns"] / 1000, 2), round(record["candidate_to_reference_ratio"], 3))


if __name__ == "__main__":
    main(sys.argv[1])
