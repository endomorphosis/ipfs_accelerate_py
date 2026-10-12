"""Explicit numerical preparation selection, bound to exact runtime producers."""
from __future__ import annotations

import hashlib
from pathlib import Path

PROFILE = "python-numerical-structure@1"
PROFILES = (PROFILE,)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_numerical_policy.py",
    "benchmarks/agent_supervisor/container_coding/terminal_numerical_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_output_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_native_preflight.py",
    "benchmarks/agent_supervisor/container_coding/terminal_native_start_preflight.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/native_usage_accounting.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_resource_profile.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_initial_context.py",
    "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/published_numerical.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py",
)
DATASET_PATHS = (
    "ipfs_datasets_py/logic/software_contracts/source_unit_numerical_links.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/source_function_units.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/source_screening.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/security/security_formula_corpus.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/security/security_formalization_evaluation.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/security/source_program_binding_384.py",
    "ipfs_datasets_py/logic/software_verification/numerical_source.py",
    "ipfs_datasets_py/logic/software_verification/numerical_logic.py",
    "ipfs_datasets_py/logic/software_verification/numerical_obligation_retrieval.py",
    "ipfs_datasets_py/logic/software_verification/__init__.py",
    "ipfs_datasets_py/logic/software_verification/program.py",
    "ipfs_datasets_py/logic/software_verification/source_adapters.py",
    "ipfs_datasets_py/logic/software_verification/syntax_bridge.py",
    "ipfs_datasets_py/logic/families/registry.py",
    "ipfs_datasets_py/logic/syntax_core/ast.py",
    "ipfs_datasets_py/logic/syntax_core/signatures.py",
    "ipfs_datasets_py/logic/backends/smt/compiler.py",
)


def validate_numerical_profile(value):
    if value is not None and (type(value) is not str or value not in PROFILES):
        raise ValueError("explicit supported numerical formalization profile required")
    return value


def numerical_selection(value):
    return {} if validate_numerical_profile(value) is None else {"numerical_formalization_profile": value}


def numerical_observation(value, arm):
    if arm not in {"full", "no-index"}:
        raise ValueError("explicit supervisor ablation required")
    return {} if validate_numerical_profile(value) is None else {
        "numerical_formalization_profile": value,
        "numerical_formalization_enabled": arm == "full",
        "numerical_formalization_disabled_reason": None if arm == "full" else "no_index_ablation",
        "numerical_formalization_proof_authority": False,
    }


def _producer_paths():
    # Import only for the selected new profile; legacy archives need no new
    # datasets module. The checked files must be those of the executing library.
    from ipfs_datasets_py.logic.software_verification import numerical_logic, numerical_source
    if numerical_logic.PROFILE != PROFILE or numerical_source.PROFILE != PROFILE:
        raise ValueError("numerical datasets profile differs from consumer selection")
    source = Path(__file__).resolve().parents[3]
    datasets = Path(numerical_source.__file__).resolve().parents[3]
    if Path(numerical_logic.__file__).resolve().parents[3] != datasets:
        raise ValueError("numerical datasets producer roots differ")
    return {**{"source/" + name: source / name for name in SOURCE_PATHS},
            **{"datasets/" + name: datasets / name for name in DATASET_PATHS}}


def require_numerical_archive(manifest, value):
    """Reject stale, duplicate or missing producer inventory before any effects."""
    if validate_numerical_profile(value) is None:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("numerical profile requires current runtime source inventory")
    try:
        producers = _producer_paths()
    except ImportError as exc:
        raise ValueError("numerical profile requires current datasets producers") from exc
    for name, path in producers.items():
        matches = [row for row in rows if type(row) is dict and row.get("path") == name]
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("numerical profile requires a newly qualified runtime archive")
