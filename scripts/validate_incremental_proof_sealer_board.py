"""Cheap, fail-closed validator for the IncrementalProofSealer board.

The validator deliberately uses only the Python standard library.  Ordinary
check modes are read-only and do not import project packages, run proof
backends, install dependencies, or mutate a repository.  Two explicit runner
modes execute fixed benchmark/release argv and publish only their declared
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_incremental_proof_sealer_scheduler.json"
)
PLAN_PATH = REPO_ROOT / "docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md"
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs/architecture/incremental_proof_sealer.objectives.md"
)
TASKBOARD_PATH = (
    REPO_ROOT / "docs/architecture/incremental_proof_sealer.todo.md"
)

BOARD_NAMESPACE = "incremental-proof-sealer-v1"
BRANCH = "agent/incremental-proof-sealer-v1"
ACCELERATE_REVISION = "8881344bb2162f3f8d82f22d8348bc0ac7536f95"
DATASETS_REVISION = "bd2ff6245ebe476fc744d45c7c66235c92b0e19c"
KIT_REVISION = "5a7a2df8181cfdc33bc19be09989df7ff83f2d4e"
PLANNING_TREES = {
    "accelerate": "25b334babbf93e9891178c04b9169ddd8fd89f3c",
    "datasets": "ec59c4527dd442e6318c24a79a9f4ad80b4548a9",
    "kit": "55e8dd7139658e23ba0f278e70a28b118f0aee3f",
}
REPOSITORY_PATHS = {
    "accelerate": Path("."),
    "datasets": Path("ipfs_datasets_py"),
    "kit": Path("ipfs_kit_py"),
}

# Sealed product DAG is IPS-000..IPS-056. Contiguous residual appendix tasks
# (retry-budget repairs, etc.) may appear as IPS-057+ without rewriting the
# sealed plan; they are validated separately as operational appendices.
SEALED_TASK_IDS = tuple(f"IPS-{index:03d}" for index in range(57))
TASK_IDS = SEALED_TASK_IDS
MAX_OPERATIONAL_RESIDUAL_TASKS = len(SEALED_TASK_IDS) * 3
RETRY_BUDGET_REPAIR_GENERATOR = (
    "ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1"
)
RECONCILIATION_GUARDRAIL_GENERATOR = (
    "ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1"
)
OPERATIONAL_RESIDUAL_GENERATORS = frozenset(
    {
        RETRY_BUDGET_REPAIR_GENERATOR,
        RECONCILIATION_GUARDRAIL_GENERATOR,
    }
)
RECONCILIATION_RESIDUAL_KINDS = frozenset(
    {
        "dirty_backlogged_worktree",
        "preflight_merge_conflict",
        "dirty_worktree",
        "content_not_in_target",
        "main_checkout_dirty",
    }
)
GOAL_IDS = ("IPS-G000",) + tuple(
    f"IPS-G{index:03d}" for index in range(10, 131, 10)
)
INITIAL_COMPLETED = frozenset({"IPS-000"})
INITIAL_READY = frozenset({"IPS-001", "IPS-002", "IPS-003"})
TERMINAL_TASK = "IPS-056"
ARTIFACT_CHECK_TASKS = (
    "IPS-001",
    "IPS-002",
    "IPS-003",
    "IPS-004",
    "IPS-053",
    "IPS-054",
    "IPS-055",
    "IPS-056",
)

BASELINE_RECEIPT_ROOT = (
    "artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts"
)
BASELINE_LOG_ROOT = f"{BASELINE_RECEIPT_ROOT}/logs"
BASELINE_RECEIPT_SCHEMA = "incremental-proof-sealer-baseline-receipt@4"
BASELINE_OPERATOR_ORIGIN = "operator_capture"
BASELINE_ENVIRONMENT_POLICY = "incremental-proof-sealer-controlled-offline-pytest@3"
BASELINE_IGNORED_INPUT_POLICY = "incremental-proof-sealer-clean-materialized-trees@1"
BASELINE_GIT_ENVIRONMENT_POLICY = "incremental-proof-sealer-fixed-git-environment@2"
BASELINE_MAX_RECEIPT_BYTES = 2 * 1024 * 1024
BASELINE_MAX_LOG_BYTES = 64 * 1024 * 1024
BASELINE_MAX_REGISTRY_BYTES = 256 * 1024
BASELINE_DEFAULT_OBSERVATION = (
    "Current controlled-offline pytest observation; historical counts are not "
    "reconstructed or claimed."
)
BASELINE_CORE_15_OBSERVATION = (
    "New current controlled-offline 15-path pytest observation; it does not "
    "reconstruct or claim the historical 257-result slice."
)
BASELINE_ENVIRONMENT_KEYS = frozenset(
    {
        "CARGO_NET_OFFLINE",
        "COLUMNS",
        "GIT_TERMINAL_PROMPT",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "HOME",
        "HYPOTHESIS_STORAGE_DIRECTORY",
        "IPFS_ACCEL_AUTO_INSTALL",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS",
        "IPFS_DATASETS_ENABLE_GROTH16",
        "IPFS_DATASETS_PY_AUTO_GROTH16_BUILD",
        "IPFS_DATASETS_RUN_GROTH16_EVM",
        "IPFS_DATASETS_RUN_PROVEKIT_TESTS",
        "IPFS_OFFLINE",
        "IPFS_PATH",
        "IPFS_TEST_PROOF_REUSE_MODE",
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL",
        "LANG",
        "LC_ALL",
        "NO_COLOR",
        "PATH",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PIP_NO_INDEX",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONPYCACHEPREFIX",
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "TERM",
        "TMPDIR",
        "TRANSFORMERS_OFFLINE",
        "TZ",
    }
)
BASELINE_OUTCOME_FIELDS = (
    "passed",
    "failed",
    "errors",
    "skipped",
    "deselected",
    "xfailed",
    "xpassed",
    "selected",
)
BASELINE_RECEIPT_SPECS: Mapping[str, Mapping[str, Any]] = {
    "IPS-001": {
        "repository": "accelerate",
        "revision": ACCELERATE_REVISION,
        "receipt": f"{BASELINE_RECEIPT_ROOT}/accelerate.json",
        "inventory": (
            "docs/architecture/incremental_proof_sealer_inventory/accelerate.json"
        ),
        "report": (
            "docs/architecture/incremental_proof_sealer_inventory/accelerate.md"
        ),
        "command_ids": (
            "accelerate-proof-focused-core-15",
            "accelerate-proof-focused-wide-36",
            "accelerate-proof-reuse-migration",
            "accelerate-proof-reuse-cross-repo",
        ),
        "cwd": ".",
        "timeouts": (300, 600, 600, 300),
    },
    "IPS-002": {
        "repository": "datasets",
        "revision": DATASETS_REVISION,
        "receipt": f"{BASELINE_RECEIPT_ROOT}/datasets.json",
        "inventory": (
            "ipfs_datasets_py/docs/architecture/"
            "incremental_proof_sealer_inventory.json"
        ),
        "report": (
            "ipfs_datasets_py/docs/architecture/"
            "INCREMENTAL_PROOF_SEALER_INVENTORY.md"
        ),
        "command_ids": (
            "datasets-zkp-focused-current",
            "datasets-zkp-unit-wide-current",
            "datasets-proof-cache-adapters",
            "datasets-zkp-broad-safe-current",
        ),
        "cwd": "ipfs_datasets_py",
        "timeouts": (600, 600, 300, 600),
    },
    "IPS-003": {
        "repository": "kit",
        "revision": KIT_REVISION,
        "receipt": f"{BASELINE_RECEIPT_ROOT}/kit.json",
        "inventory": (
            "ipfs_kit_py/docs/architecture/"
            "incremental_proof_sealer_inventory.json"
        ),
        "report": (
            "ipfs_kit_py/docs/architecture/"
            "INCREMENTAL_PROOF_SEALER_INVENTORY.md"
        ),
        "command_ids": (
            "kit-proof-certificate",
            "kit-reuse-capabilities",
            "kit-profile-d",
            "kit-coordination",
            "kit-modern-wal",
            "kit-proof-reuse-bootstrap",
            "kit-agent-receipts",
            "kit-iroh-release",
            "kit-release-receipt",
        ),
        "cwd": "ipfs_kit_py",
        "timeouts": (120, 120, 120, 120, 300, 300, 120, 120, 120),
    },
}

EXPECTED_TASK_GROUPS: Mapping[str, tuple[str, ...]] = {
    "IPS-G010": tuple(f"IPS-{index:03d}" for index in range(5)),
    "IPS-G020": tuple(f"IPS-{index:03d}" for index in range(5, 13)),
    "IPS-G030": tuple(f"IPS-{index:03d}" for index in range(13, 18)),
    "IPS-G040": tuple(f"IPS-{index:03d}" for index in range(18, 23)),
    "IPS-G050": tuple(f"IPS-{index:03d}" for index in range(23, 28)),
    "IPS-G060": tuple(f"IPS-{index:03d}" for index in range(28, 32)),
    "IPS-G070": tuple(f"IPS-{index:03d}" for index in range(32, 38)),
    "IPS-G080": tuple(f"IPS-{index:03d}" for index in range(38, 43)),
    "IPS-G090": tuple(f"IPS-{index:03d}" for index in range(43, 45)),
    "IPS-G100": tuple(f"IPS-{index:03d}" for index in range(45, 48)),
    "IPS-G110": tuple(f"IPS-{index:03d}" for index in range(48, 52)),
    "IPS-G120": tuple(f"IPS-{index:03d}" for index in range(52, 55)),
    "IPS-G130": tuple(f"IPS-{index:03d}" for index in range(55, 57)),
}
EXPECTED_TASK_TO_GOAL = {
    task_id: goal_id
    for goal_id, task_ids in EXPECTED_TASK_GROUPS.items()
    for task_id in task_ids
}

REQUIRED_TASK_FIELDS = (
    "status",
    "completion",
    "is schedulable",
    "review only",
    "priority",
    "track",
    "depends on",
    "goal id",
    "outputs",
    "validation",
    "board namespace",
    "bundle",
    "parallel lane",
    "resource class",
    "resource stage",
    "estimated tokens",
    "implementation timeout seconds",
    "predicted files",
    "submodules",
    "interfaces",
    "allow concurrent with",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "symbolic first",
    "acceptance",
    "embedding query",
)
REQUIRED_GOAL_FIELDS = (
    "status",
    "parent",
    "depends on",
    "fib priority",
    "priority",
    "track",
    "bundle",
    "parallel lane",
    "resource class",
    "goal",
    "evidence",
    "acceptance criteria",
    "outputs",
    "validation",
    "acceptance",
    "gap task",
    "refinement",
    "conflict policy",
)

CONTROL_PATHS = (
    ".gitignore",
    "config/agent_supervisor_incremental_proof_sealer_scheduler.json",
    "docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md",
    "docs/architecture/incremental_proof_sealer.objectives.md",
    "docs/architecture/incremental_proof_sealer.todo.md",
    "scripts/validate_incremental_proof_sealer_board.py",
)
BASELINE_CAPTURE_SCRIPT = "scripts/capture_incremental_proof_sealer_baselines.py"
BASELINE_SUITE_REGISTRY = "config/incremental_proof_sealer_baseline_suite_registry.json"
BASELINE_SUITE_REGISTRY_SCHEMA = "incremental-proof-sealer-reviewed-suite-registry@1"
BASELINE_SUITE_REGISTRY_DIGEST = (
    "sha256:4489fa11df5fd7c2e7f3aaf6201266eb74861ef6a36d94ae2bf3e6d543e55c3e"
)
BASELINE_SYNTHESIS_SCHEMA = "incremental-proof-sealer-trust-baseline@2"
BASELINE_SYNTHESIS_JSON = (
    "docs/architecture/incremental_proof_sealer_inventory/matrix.json"
)
BASELINE_SYNTHESIS_REPORT = (
    "docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_BASELINE.md"
)
BASE_PROTECTED_PATHS = CONTROL_PATHS + (
    BASELINE_CAPTURE_SCRIPT,
    BASELINE_SUITE_REGISTRY,
)

BENCHMARK_SCHEMA = "incremental-proof-sealer-benchmark-results@2"
BENCHMARK_ID = "incremental-proof-sealer-40-transition@1"
BENCHMARK_SEED = 20260811
BENCHMARK_TRANSITION_COUNT = 40
BENCHMARK_JSON = "artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json"
BENCHMARK_CSV = "artifacts/agent_supervisor/incremental_proof_sealer/benchmark.csv"
BENCHMARK_CLI = "benchmarks/agent_supervisor/incremental_proof_sealer.py"
BENCHMARK_MAX_ARTIFACT_BYTES = 1024 * 1024
BENCHMARK_VALIDATION_ARGV = (
    "python",
    "scripts/validate_incremental_proof_sealer_board.py",
    "--run-benchmark",
)
MATERIALIZATION_REQUEST_SCHEMA = "incremental-proof-sealer-materialization-request@1"
BENCHMARK_REQUEST_JSON = (
    b'{"schema_version":"incremental-proof-sealer-materialization-request@1",'
    b'"task_id":"IPS-053"}\n'
)
BENCHMARK_REQUEST_CSV = b"incremental-proof-sealer-materialization-request@1,IPS-053\n"
BENCHMARK_SCENARIOS = (
    "initial repository",
    "localized private source edit",
    "unrelated documentation",
    "one test-source edit",
    "one fixture edit",
    "unrelated module edit",
    "public-interface edit",
    "dependent module edit",
    "selected test addition",
    "authorized test deletion",
    "relevant configuration edit",
    "ordinary documentation",
    "dependency-lock class upgrade",
    "localized source edit",
    "two independent module edits",
    "branch A edit",
    "branch B edit from prior accepted parent",
    "merge A/B",
    "rollback of source bytes",
    "property-test edit",
    "periodic N-commit checkpoint",
    "documentation-only",
    "circuit version change",
    "localized source edit",
    "verification-key change",
    "test-selector change",
    "network-policy change",
    "environment trust-policy change",
    "integration fixture edit",
    "requirement policy change",
    "periodic checkpoint",
    "integration-test addition",
    "proof schema/canonicalization change",
    "checked-specification document edit",
    "ordinary documentation edit",
    "injected cache corruption detection",
    "two independent modules",
    "wrong-parent attempt then valid",
    "merge plus unaffected reuse",
    "release tag/compaction",
)
BENCHMARK_FULL_TRANSITIONS = frozenset({0, 12, 20, 22, 24, 27, 30, 32, 35, 39})
BENCHMARK_CONDITIONAL_FULL_TRANSITIONS = frozenset({17, 29, 38})
BENCHMARK_METRICS = (
    "leaf_proving_seconds",
    "aggregation_seconds",
    "prover_cpu_seconds",
    "prover_gpu_seconds",
    "peak_memory_bytes",
    "proof_size_bytes",
    "seal_size_bytes",
    "storage_growth_bytes",
    "seal_verification_seconds",
    "wall_clock_seconds",
    "full_proof_cost",
    "incremental_proof_cost",
)
BENCHMARK_CSV_FIELDS = (
    "index",
    "scenario",
    "seal_status",
    "measurement_provenance",
    "required_units",
    "reused_units",
    "invalidated_units",
    "added_units",
    "removed_units",
    "newly_proved_units",
    "cache_hit_rate",
    *BENCHMARK_METRICS,
    "compute_saved_percent",
    "chain_depth",
    "fallback_reason",
    "deterministic_roots_match",
    "simulated_required_units",
)
BENCHMARK_SUMMARY_SCHEMA = "incremental-proof-sealer-benchmark-summary@1"
BENCHMARK_SUMMARY_JSON = (
    "artifacts/agent_supervisor/incremental_proof_sealer/summary.json"
)
BENCHMARK_REPORT = "docs/architecture/INCREMENTAL_PROOF_SEALER_BENCHMARK.md"

TRUST_BASELINE_AUTHORITIES = {
    "proof_unit_manifest_identity": "ipfs_datasets_py",
    "proof_object_cache_forest_wal_cas": "ipfs_kit_py",
    "prover_scheduler_aggregation_planner_metrics": "ipfs_accelerate_py",
}
TRUST_BASELINE_PROOF_CLASS_DECISIONS = {
    "integrity_commitment": "integrity_only",
    "signed_execution_receipt": "trusted_signer_assertion_not_direct_execution",
    "receipt_aggregation_zk_proof": "receipt_completeness_not_test_execution",
    "direct_execution_proof": "declared_computation_only",
    "incremental_commit_seal": "parent_bound_verified_leaf_transition",
}
TRUST_BASELINE_AGGREGATION_DECISION = {
    "mode": "merkle_manifest_aggregation",
    "recursive_self_verification_supported": False,
    "child_proofs_individually_verified": True,
    "test_execution_directly_proven": False,
}
TRUST_BASELINE_BACKEND_DECISIONS = {
    "existing_recursive_backend": "unsupported",
    "groth16": "bounded_declared_computation_only",
    "provekit": "optional_capability_unavailable_is_typed",
    "simulated": "production_seal_forbidden",
    "unknown": "rejected",
}
TRUST_BASELINE_NONCLAIMS = (
    "entire_repository_proven_correct",
    "pytest_execution_cryptographically_proven",
    "semantically_correct_change",
    "recursive_proof_verification_available",
)

RELEASE_VALIDATION_SCHEMA = "incremental-proof-sealer-release-validation@2"
RELEASE_VALIDATION_JSON = (
    "artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json"
)
RELEASE_VALIDATION_LOG = (
    "artifacts/agent_supervisor/incremental_proof_sealer/release_validation.log"
)
RELEASE_WORK_ROOT = "artifacts/agent_supervisor/incremental_proof_sealer/release-work"
RELEASE_REPORT = "docs/architecture/INCREMENTAL_PROOF_SEALER_REPORT.md"
RELEASE_RUNNER_ID = "protected-board-release-validation-runner@1"
RELEASE_ENVIRONMENT_POLICY = "incremental-proof-sealer-isolated-tree-offline-pytest@2"
RELEASE_PUBLIC_LOG_POLICY = "public-full-log-secret-scan@1"
RELEASE_FIXED_EXECUTABLE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RELEASE_MAX_LOG_BYTES = 6 * 1024 * 1024
RELEASE_MAX_REPORT_BYTES = 1024 * 1024
RELEASE_PROCESS_MAX_OUTPUT_BYTES = 256 * 1024
RELEASE_VALIDATION_ARGV = (
    "python",
    "scripts/validate_incremental_proof_sealer_board.py",
    "--run-release-validation",
)
RELEASE_REQUEST_JSON = (
    b'{"schema_version":"incremental-proof-sealer-materialization-request@1",'
    b'"task_id":"IPS-056"}\n'
)
RELEASE_REQUEST_LOG = b"incremental-proof-sealer-materialization-request@1 IPS-056\n"
RELEASE_REPORT_REQUEST_MARKER = "<!-- IPS-056 RELEASE EVIDENCE: MATERIALIZE ONCE -->"
RUNNER_SOURCE_BINDING_POLICY = (
    "incremental-proof-sealer-pristine-index-worktree-content@1"
)
RUNNER_MATERIALIZATION_MAX_ENTRIES = 250_000
RUNNER_MATERIALIZATION_MAX_DEPTH = 64
RUNNER_MATERIALIZATION_MAX_LEAF_BYTES = 256 * 1024 * 1024
RUNNER_MATERIALIZATION_MAX_HASH_BYTES = 4 * 1024 * 1024 * 1024
RUNNER_MAX_DESCENDANT_PROCESSES = 4096
RUNNER_UNMATERIALIZED_GITLINKS: Mapping[str, Mapping[str, str]] = {
    "accelerate": {
        "docs/fastmcp": "1d932cc778a24cc0bf46fc4baad8306d4fed9c4b",
        "docs/mcp-python-sdk": "0da9a074d09267a927d72faa58c26d828f0f8edb",
        "ipfs_accelerate_py/mcplusplus": (
            "15c1816d6c63a2b11edd505704f6a04a9abc6167"
        ),
        "ipfs_model_manager_py": "f6151d2113f42e75ea7d83a1b2362fc97e55e44d",
        "ipfs_transformers_py": "b397988ed9e3e656475c1cf4417b84efdb95daf3",
        "test/doc-builder": "6108e850ae1cf2f71bb0815a600bcd50c39abfa7",
        "test/huggingface_doc_builder": (
            "6108e850ae1cf2f71bb0815a600bcd50c39abfa7"
        ),
        "test/huggingface_transformers": (
            "44752c8dd99f3fb0da23006dc4fde4a07d9c417f"
        ),
    },
    "datasets": {
        ".tools/ipfs_kit_py": "80afdad2fa6db5875f40e5f495f26b98b7f3c767",
        "ipfs_accelerate_py": "48f13ab632dec4c3575acaad6e309ef60420904b",
        "ipfs_datasets_py/logic/CEC/DCEC_Library": (
            "a4beb5b3280595be6b9221cac3c91dd019e6d371"
        ),
        "ipfs_datasets_py/logic/CEC/Eng-DCEC": (
            "df518c21ef81b8001e6db59f5fd70f10cc04ff6c"
        ),
        "ipfs_datasets_py/logic/CEC/ShadowProver": (
            "3060ede1ac1ec3f8ef9f9c9e41386aed1dbbe7f9"
        ),
        "ipfs_datasets_py/logic/CEC/Talos": (
            "e0b7650d3e3a403924773f8253e924c719748d36"
        ),
        "ipfs_datasets_py/multimedia/convert_to_txt_based_on_mime_type": (
            "d58933631a5362b1e2fdc45254ef620fa231223a"
        ),
        "ipfs_datasets_py/multimedia/omni_converter_mk2": (
            "c1d9b0d517cea022516aab5b5d8fa5e3bc9a65aa"
        ),
        "ipfs_datasets_py/processors/web_archiving/common_crawl_search_engine": (
            "5c7c2ab8a509073f39359b2a35446183855f460a"
        ),
        "ipfs_kit_py": "80afdad2fa6db5875f40e5f495f26b98b7f3c767",
    },
    "kit": {
        "docs/filesystem_spec": "fec09b04ad626df44a03bc605cb2e526b752b042",
        "docs/ipfs-docs": "4cf83720b59738d93db4068976f9c2a11f023e45",
        "docs/ipfs_cluster": "c7ca8b5f87b41fcc795297ca65b0bb41c10234bf",
        "docs/ipfsspec": "03f5199b9bf5a96c7ebf5e2e6f5dce8cf58b655f",
        "docs/lassie": "c6ba777810d03fed23aea11b5969b7d8a97f1edf",
        "docs/libp2p-universal-connectivity": (
            "e18a6de9c020c5e406d9f61b638f5d276054798d"
        ),
        "docs/libp2p_docs": "17cee4a438797313d1e878b103abc1dbefdf423e",
        "docs/lighthouse-python-sdk": (
            "6b2c86693090c770d2c9a4d82ba315000a77068b"
        ),
        "docs/mcp-python-sdk": "d3133ae6ce7333a501e38046aff4275c44326f90",
        "docs/storacha_specs": "3b6791869635735ddb1a54aed7450ad6ef687c06",
        "ipfs_accelerate_py": "48f13ab632dec4c3575acaad6e309ef60420904b",
    },
}
BENCHMARK_PROPOSAL_ENVELOPE = {
    "schema": "ipfs_accelerate_py/agent-supervisor/task-artifact-envelope@1",
    "paths": [BENCHMARK_JSON, BENCHMARK_CSV],
    "max_file_bytes": 2_000_000,
    "max_patch_bytes": 4_000_000,
    "max_output_bytes": 8_000_000,
}
RELEASE_PROPOSAL_ENVELOPE = {
    "schema": "ipfs_accelerate_py/agent-supervisor/task-artifact-envelope@1",
    "paths": [RELEASE_REPORT, RELEASE_VALIDATION_JSON, RELEASE_VALIDATION_LOG],
    "max_file_bytes": 7_000_000,
    "max_patch_bytes": 12_000_000,
    "max_output_bytes": 20_000_000,
}
RELEASE_NEW_SUITES = (
    (
        "accelerate-incremental-sealing",
        ".",
        "test/api/incremental_sealing",
        1200,
    ),
    (
        "datasets-incremental-sealing",
        "ipfs_datasets_py",
        "tests/unit/logic/zkp/incremental_sealing",
        1200,
    ),
    (
        "kit-proof-seal-store",
        "ipfs_kit_py",
        "tests/proof_seal_store",
        1200,
    ),
)

REQUIRED_PLAN_CONCEPTS = (
    "IntegrityCommitment",
    "SignedExecutionReceipt",
    "ReceiptAggregationZkProof",
    "DirectExecutionProof",
    "IncrementalCommitSeal",
    "ProofUnit@1",
    "VerificationRequirementManifest@1",
    "ProofCacheKey@1",
    "source_root_cid",
    "repository_state_cid",
    "source_depends_on",
    "schema_depends_on",
    "fixture_depends_on",
    "config_depends_on",
    "proof_depends_on",
    "aggregate_contains",
    "supersedes",
    "invalidates",
    "RepositoryProofForest",
    "source_integrity_root",
    "static_analysis_root",
    "type_check_root",
    "unit_test_root",
    "integration_test_root",
    "property_test_root",
    "formal_obligation_root",
    "direct_zk_root",
    "release_invariant_root",
    "FullCheckpointSeal",
    "DeltaSeal@1",
    "compare-and-swap",
    "WAL phases",
    "manifest aggregation",
    "bounded fan-in",
    "chain compaction",
    "render-pins",
    "--check-bootstrap",
    "incremental-proof-sealer-trust-baseline@2",
    "incremental-proof-sealer-benchmark-results@2",
    "--run-benchmark",
    "incremental-proof-sealer-release-validation@2",
    "--run-release-validation",
)
REQUIRED_CLI_TERMS = (
    "`full`",
    "`incremental`",
    "`verify`",
    "`plan`",
    "`explain-reuse`",
    "`explain-invalidation`",
    "`benchmark`",
    "`cache-status`",
    "`force-full`",
    "`compact`",
)
REQUIRED_INVALIDATION_TERMS = (
    "Source implementation change",
    "public interface",
    "Test source change",
    "Deleted tests",
    "Added selected tests",
    "Dependency-lock change",
    "Fixture or configuration change",
    "Circuit or proving/verification-key change",
    "Canonicalization or dependency-graph schema change",
    "Environment-policy change",
    "policy changes",
    "Documentation-only changes",
)
REQUIRED_NEGATIVE_TERMS = (
    "source root",
    "environment",
    "selector",
    "verification key",
    "circuit",
    "dependency closure",
    "public input",
    "invalid cryptography",
    "unsigned required receipt",
    "unauthorized test removal",
    "changed manifest with old aggregate",
    "wrong parent",
    "missing invalidated unit",
    "simulated production unit",
    "unknown/timeout",
    "lost unaffected leaf",
    "stale CAS writer",
)
REQUIRED_CRASH_TERMS = (
    "before proof execution",
    "after proof execution, before receipt persistence",
    "after receipt persistence, before forest update",
    "after forest update, before aggregate generation",
    "after aggregate generation, before seal persistence",
    "after seal persistence, before current-root CAS",
    "after CAS, before transaction cleanup",
)


class DuplicateJsonKey(ValueError):
    """Raised when a JSON object repeats a key."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKey(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        errors.append(f"refusing to load JSON outside repository: {path}")
        return {}
    retained = _secure_read_repo_file(
        relative,
        required_parent=str(Path(relative).parent),
        label=f"JSON input {relative}",
        maximum_bytes=BASELINE_MAX_RECEIPT_BYTES,
        bound_label="two-MiB",
        errors=errors,
    )
    if retained is None:
        return {}
    try:
        result = json.loads(
            retained[0].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"cannot load duplicate-free config {path}: {exc}")
        return {}
    if not isinstance(result, dict):
        errors.append("scheduler config must be one JSON object")
        return {}
    return result


def _read(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return ""


def _parse_markdown_records(
    text: str,
    heading_pattern: re.Pattern[str],
    label: str,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    matches = list(heading_pattern.finditer(text))
    records: dict[str, dict[str, Any]] = {}
    seen_titles: set[str] = set()
    for index, match in enumerate(matches):
        record_id = match.group(1)
        title = match.group(2).strip()
        if record_id in records:
            errors.append(f"duplicate {label} heading: {record_id}")
            continue
        full_title = f"{record_id} {title}".casefold()
        if full_title in seen_titles:
            errors.append(f"duplicate {label} title: {record_id} {title}")
        seen_titles.add(full_title)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        fields: dict[str, str] = {}
        for field_match in re.finditer(
            r"^- ([^:\n]+):[ \t]*(.*)$", body, flags=re.MULTILINE
        ):
            key = field_match.group(1).strip().casefold()
            if key in fields:
                errors.append(f"{record_id} repeats metadata field {key!r}")
            else:
                fields[key] = field_match.group(2).strip()
        records[record_id] = {"title": title, "fields": fields}
    return records


def _ids(value: str, pattern: str) -> tuple[str, ...]:
    return tuple(re.findall(pattern, value))


def _as_bool(value: str) -> bool | None:
    folded = value.strip().casefold()
    if folded == "true":
        return True
    if folded == "false":
        return False
    return None


def _as_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validation_argv(
    owner_id: str,
    value: str,
    errors: list[str],
) -> list[str]:
    """Parse one direct-exec validation command and reject shell syntax."""

    if re.search(r"[;&|<>`]", value) or "$(" in value:
        errors.append(f"{owner_id} validation contains a forbidden shell control operator")
    try:
        argv = shlex.split(value)
    except ValueError as exc:
        errors.append(f"{owner_id} validation command does not parse: {exc}")
        return []
    if not argv:
        return argv
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1]
    if executable in {
        "bash",
        "cmd",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "pwsh",
        "sh",
        "zsh",
    }:
        errors.append(f"{owner_id} validation uses a forbidden shell")
    if (
        len(argv) >= 2
        and executable in {"node", "perl", "python", "python3", "ruby"}
        and argv[1] in {"-c", "-e", "--eval"}
    ):
        errors.append(f"{owner_id} validation uses forbidden dynamic eval")
    return argv


def _cycle_nodes(graph: Mapping[str, Iterable[str]]) -> set[str]:
    """Return nodes participating in, or downstream from, a dependency cycle."""

    indegree = {node: 0 for node in graph}
    dependents: dict[str, set[str]] = defaultdict(set)
    for node, dependencies in graph.items():
        for dependency in dependencies:
            if dependency in indegree:
                indegree[node] += 1
                dependents[dependency].add(node)
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited: set[str] = set()
    while queue:
        node = queue.popleft()
        visited.add(node)
        for dependent in sorted(dependents[node]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    return set(graph) - visited


def _reachable_from(
    start: Iterable[str], dependencies: Mapping[str, set[str]]
) -> set[str]:
    dependents: dict[str, set[str]] = defaultdict(set)
    for node, prerequisites in dependencies.items():
        for prerequisite in prerequisites:
            dependents[prerequisite].add(node)
    reached = set(start)
    queue = deque(sorted(reached))
    while queue:
        for dependent in sorted(dependents[queue.popleft()]):
            if dependent not in reached:
                reached.add(dependent)
                queue.append(dependent)
    return reached


def _ancestors(node: str, dependencies: Mapping[str, set[str]]) -> set[str]:
    reached: set[str] = set()
    queue = deque(sorted(dependencies.get(node, set())))
    while queue:
        dependency = queue.popleft()
        if dependency in reached:
            continue
        reached.add(dependency)
        queue.extend(sorted(dependencies.get(dependency, set())))
    return reached


def _fixed_git_environment() -> dict[str, str]:
    # Revision, cleanliness, and diff decisions are trust boundaries.  Do not
    # let caller-supplied GIT_DIR/GIT_INDEX_FILE/object alternates or user Git
    # configuration redirect those decisions.
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git(
    *args: str, cwd: Path | None = None, timeout: float = 3.0
) -> subprocess.CompletedProcess[str]:
    if cwd is None:
        cwd = REPO_ROOT
    try:
        return subprocess.run(
            ("git", *args),
            cwd=cwd,
            env=_fixed_git_environment(),
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            args=("git", *args), returncode=124, stdout="", stderr=str(exc)
        )


def _git_stdout(
    cwd: Path,
    errors: list[str],
    label: str,
    *args: str,
) -> str:
    result = _git(*args, cwd=cwd)
    if result.returncode != 0:
        errors.append(f"{label} failed: {result.stderr.strip() or result.returncode}")
        return ""
    return result.stdout.strip()


def _git_bytes(
    cwd: Path,
    errors: list[str],
    label: str,
    *args: str,
) -> bytes | None:
    """Run trust-boundary Git plumbing without lossy path decoding."""

    status, exit_code, _duration, output = _run_observed_process(
        [
            "git",
            "-c",
            "core.filemode=true",
            "-c",
            "core.fsmonitor=false",
            *args,
        ],
        cwd=cwd,
        environment=_fixed_git_environment(),
        timeout_seconds=30,
        maximum_output_bytes=64 * 1024 * 1024,
    )
    if status != "completed" or exit_code != 0:
        detail = output.decode("utf-8", "replace").strip()
        errors.append(f"{label} failed: {detail or f'{status}/{exit_code}'}")
        return None
    return output


def _reject_git_replacement_state(
    root: Path,
    *,
    label: str,
    errors: list[str],
) -> None:
    """Reject replacement refs and legacy grafts even though Git ignores them."""

    replacements = _git_bytes(
        root,
        errors,
        f"inspect {label} replacement refs",
        "for-each-ref",
        "--format=%(refname)%00",
        "refs/replace",
    )
    if replacements is not None and replacements.strip(b"\0\r\n"):
        errors.append(f"{label} contains forbidden Git replacement refs")
    common_text = _git_stdout(
        root,
        errors,
        f"resolve {label} common Git directory",
        "rev-parse",
        "--git-common-dir",
    )
    if not common_text:
        return
    common = Path(common_text)
    if not common.is_absolute():
        common = root / common
    try:
        common = common.resolve(strict=True)
    except OSError as exc:
        errors.append(
            f"cannot resolve {label} common Git directory: {type(exc).__name__}"
        )
        return
    if os.path.lexists(common / "info" / "grafts"):
        errors.append(f"{label} contains a forbidden legacy Git grafts file")


NESTED_INVENTORY_OUTPUTS: Mapping[str, frozenset[str]] = {
    "ipfs_datasets_py": frozenset(
        {
            "docs/architecture/incremental_proof_sealer_inventory.json",
            "docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md",
        }
    ),
    "ipfs_kit_py": frozenset(
        {
            "docs/architecture/incremental_proof_sealer_inventory.json",
            "docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md",
        }
    ),
}
ACCELERATE_INVENTORY_OUTPUTS = frozenset(
    {
        "docs/architecture/incremental_proof_sealer_inventory/accelerate.json",
        "docs/architecture/incremental_proof_sealer_inventory/accelerate.md",
    }
)
POST_CAPTURE_PIN_CONFIG_PATH = (
    "config/agent_supervisor_incremental_proof_sealer_scheduler.json"
)
CURRENT_SENSITIVE_SCAN_MAX_ENTRIES = 200_000


def _is_explicit_irrelevant_ignored_root(repository: str, relative: str) -> bool:
    path = Path(relative)
    if path.name.casefold() in {
        ".hypothesis",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }:
        return True
    if repository == "accelerate" and relative in {
        "data/agent_supervisor",
        f"{BASELINE_RECEIPT_ROOT}/work",
        RELEASE_WORK_ROOT,
        ".agent_git",
    }:
        return True
    return repository == "datasets" and relative == "workspace/test-logs"


def _is_allowed_ignored_container(repository: str, relative: str) -> bool:
    """Allow fixed directory scaffolding while still scanning every descendant."""

    if repository == "accelerate" and relative in {
        "artifacts",
        "artifacts/agent_supervisor",
        "artifacts/agent_supervisor/incremental_proof_sealer",
        BASELINE_RECEIPT_ROOT,
        BASELINE_LOG_ROOT,
    }:
        return True
    return (repository, relative) in {
        ("datasets", ".benchmarks"),
        ("datasets", "bin"),
        ("datasets", "bin/.deps"),
        ("datasets", "bin/.deps/npm"),
        ("datasets", "bin/.deps/npm/bin"),
        ("kit", ".cache"),
        ("kit", ".cache/ipfs-repo"),
    }


def _git_ignored_paths(
    repository_root: Path,
    paths: Iterable[str],
    repository: str,
    errors: list[str],
) -> set[str]:
    ordered = sorted(set(paths))
    if not ordered:
        return set()
    payload = b"".join(os.fsencode(path) + b"\0" for path in ordered)
    try:
        result = subprocess.run(
            ("git", "check-ignore", "-z", "--stdin"),
            cwd=repository_root,
            env=_fixed_git_environment(),
            input=payload,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(
            f"cannot classify {repository} ignored paths: {type(exc).__name__}"
        )
        return set(ordered)
    if result.returncode not in {0, 1}:
        errors.append(
            f"cannot classify {repository} ignored paths: "
            f"{result.stderr.decode('utf-8', 'replace').strip() or result.returncode}"
        )
        return set(ordered)
    return {
        os.fsdecode(raw)
        for raw in result.stdout.split(b"\0")
        if raw
    }


def _materialized_opaque_gitlink_is_clean(
    *,
    repository: str,
    relative: str,
    target: Path,
    expected_oid: str,
    errors: list[str],
) -> bool:
    """Verify an initialized allowlisted gitlink before treating it as opaque."""

    initial_error_count = len(errors)
    label = f"{repository} current opaque gitlink {relative!r}"
    top = _git_stdout(
        target,
        errors,
        f"resolve {label} worktree root",
        "rev-parse",
        "--show-toplevel",
    )
    try:
        resolved_target = target.resolve(strict=True)
        resolved_top = Path(top).resolve(strict=True) if top else None
    except OSError as exc:
        errors.append(f"cannot resolve {label}: {type(exc).__name__}")
        return False
    if resolved_top != resolved_target:
        errors.append(f"{label} is materialized as a non-repository directory")
        return False
    _reject_git_replacement_state(target, label=label, errors=errors)
    head = _git_stdout(target, errors, f"resolve {label} HEAD", "rev-parse", "HEAD")
    if head != expected_oid:
        errors.append(f"{label} HEAD does not match its reviewed index OID")

    tree_raw = _git_bytes(
        target,
        errors,
        f"read {label} HEAD tree",
        "ls-tree",
        "-rz",
        "--full-tree",
        "HEAD",
    )
    index_raw = _git_bytes(
        target,
        errors,
        f"read {label} index",
        "ls-files",
        "--stage",
        "-z",
        "--",
    )
    tags_raw = _git_bytes(
        target,
        errors,
        f"read {label} index flags",
        "ls-files",
        "-v",
        "-z",
        "--",
    )
    status_raw = _git_bytes(
        target,
        errors,
        f"inspect {label} worktree",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--ignore-submodules=none",
    )
    if None not in (tree_raw, index_raw, tags_raw, status_raw):
        assert tree_raw is not None
        assert index_raw is not None
        assert tags_raw is not None
        assert status_raw is not None
        tree_entries = _parse_runner_tree(tree_raw, repository, errors)
        index_entries = _parse_runner_index(index_raw, repository, errors)
        expected_index = {
            path: [(mode, oid, 0)]
            for path, (mode, _kind, oid) in tree_entries.items()
        }
        if index_entries != expected_index:
            errors.append(f"{label} index differs from its exact HEAD tree")
        index_tags = _parse_runner_index_tags(tags_raw, repository, errors)
        if set(index_tags) != set(index_entries) or any(
            tag != "H" for tag in index_tags.values()
        ):
            errors.append(f"{label} index contains non-ordinary flags")
        if status_raw:
            errors.append(
                f"{label} worktree contains staged, unstaged, untracked, or ignored drift"
            )
    return len(errors) == initial_error_count


def _validated_current_opaque_gitlinks(
    repository: str,
    repository_root: Path,
    errors: list[str],
) -> set[str] | None:
    """Bind the closed opaque-gitlink set to index OIDs and disk state."""

    expected = dict(RUNNER_UNMATERIALIZED_GITLINKS.get(repository, {}))
    initial_error_count = len(errors)
    index_raw = _git_bytes(
        repository_root,
        errors,
        f"read {repository} current index for opaque gitlinks",
        "ls-files",
        "--stage",
        "-z",
        "--",
    )
    if index_raw is None:
        return None
    index_entries = _parse_runner_index(index_raw, repository, errors)
    delegated_repositories = (
        {
            relative.as_posix()
            for name, relative in REPOSITORY_PATHS.items()
            if repository == "accelerate"
            and name != "accelerate"
            and relative != Path(".")
        }
        if repository == "accelerate"
        else set()
    )
    observed_gitlinks = {
        relative
        for relative, entries in index_entries.items()
        if any(mode == "160000" for mode, _oid, _stage in entries)
    }
    unknown = sorted(observed_gitlinks - set(expected) - delegated_repositories)
    if unknown:
        errors.append(
            f"{repository} current index contains unknown opaque gitlinks: {unknown[:12]}"
        )

    for relative, expected_oid in sorted(expected.items()):
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or relative != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            errors.append(
                f"{repository} opaque gitlink allowlist has an unsafe path: {relative!r}"
            )
            continue
        entries = index_entries.get(relative)
        if entries != [("160000", expected_oid, 0)]:
            errors.append(
                f"{repository} current opaque gitlink {relative!r} index mode/OID drifted"
            )

    if len(errors) != initial_error_count:
        return None

    for relative, expected_oid in sorted(expected.items()):
        target = repository_root / PurePosixPath(relative)
        if not os.path.lexists(target):
            continue
        try:
            info = target.lstat()
        except OSError as exc:
            errors.append(
                f"cannot inspect {repository} current opaque gitlink {relative!r}: "
                f"{type(exc).__name__}"
            )
            continue
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            errors.append(
                f"{repository} current opaque gitlink {relative!r} is a symlink or special entry"
            )
            continue
        try:
            has_entries = next(target.iterdir(), None) is not None
        except OSError as exc:
            errors.append(
                f"cannot enumerate {repository} current opaque gitlink {relative!r}: "
                f"{type(exc).__name__}"
            )
            continue
        if has_entries:
            _materialized_opaque_gitlink_is_clean(
                repository=repository,
                relative=relative,
                target=target,
                expected_oid=expected_oid,
                errors=errors,
            )
    if len(errors) != initial_error_count:
        return None
    return set(expected)


def _validate_current_trust_sensitive_ignored_inputs(errors: list[str]) -> None:
    """Deny current ignored inputs outside explicitly irrelevant cache roots."""

    for repository, relative_root in REPOSITORY_PATHS.items():
        repository_root = (REPO_ROOT / relative_root).resolve()
        opaque_gitlinks = _validated_current_opaque_gitlinks(
            repository, repository_root, errors
        )
        if opaque_gitlinks is None:
            continue
        visited = 0
        candidates: list[str] = []
        try:
            for current_root, directory_names, file_names in os.walk(
                repository_root, topdown=True, followlinks=False
            ):
                current = Path(current_root)
                relative_current = current.relative_to(repository_root)
                pruned: list[str] = []
                for name in directory_names:
                    relative = (relative_current / name).as_posix()
                    if not relative_current.parts and name == ".git":
                        continue
                    if name == ".git":
                        errors.append(
                            f"{repository} current checkout contains unknown nested "
                            f"Git administration entry: {relative}"
                        )
                        continue
                    if repository == "accelerate" and relative in {
                        "ipfs_datasets_py",
                        "ipfs_kit_py",
                    }:
                        continue
                    if relative in opaque_gitlinks:
                        continue
                    child = repository_root / relative
                    try:
                        child_info = child.lstat()
                    except OSError as exc:
                        errors.append(
                            f"cannot inspect {repository} current path {relative}: "
                            f"{type(exc).__name__}"
                        )
                        continue
                    if _is_explicit_irrelevant_ignored_root(repository, relative):
                        if stat.S_ISDIR(child_info.st_mode) and not stat.S_ISLNK(
                            child_info.st_mode
                        ):
                            continue
                        candidates.append(relative)
                        continue
                    pruned.append(name)
                    visited += 1
                    if not _is_allowed_ignored_container(repository, relative):
                        candidates.append(relative)
                directory_names[:] = pruned
                for name in file_names:
                    relative = (relative_current / name).as_posix()
                    if not relative_current.parts and name == ".git":
                        continue
                    if name == ".git":
                        errors.append(
                            f"{repository} current checkout contains unknown nested "
                            f"Git administration entry: {relative}"
                        )
                        continue
                    visited += 1
                    candidates.append(relative)
                if visited > CURRENT_SENSITIVE_SCAN_MAX_ENTRIES:
                    errors.append(
                        f"{repository} current sensitive-input scan exceeded its bound"
                    )
                    break
        except (OSError, ValueError) as exc:
            errors.append(
                f"cannot scan {repository} current trust-sensitive inputs: "
                f"{type(exc).__name__}"
            )
            continue
        for relative in sorted(
            _git_ignored_paths(repository_root, candidates, repository, errors)
        ):
            candidate = repository_root / relative
            try:
                info = candidate.lstat()
            except OSError as exc:
                errors.append(
                    f"cannot inspect {repository} current sensitive path {relative}: "
                    f"{type(exc).__name__}"
                )
                continue
            kind = (
                "symlink"
                if stat.S_ISLNK(info.st_mode)
                else "regular"
                if stat.S_ISREG(info.st_mode)
                else "directory"
                if stat.S_ISDIR(info.st_mode)
                else "special"
            )
            errors.append(
                f"{repository} current checkout contains undeclared ignored "
                f"{kind} input: {relative}"
            )


def _git_json_at_revision(
    revision: str,
    relative: str,
    errors: list[str],
    label: str,
) -> dict[str, Any]:
    raw = _git_stdout(
        REPO_ROOT, errors, label, "show", f"{revision}:{relative}"
    )
    if not raw:
        return {}
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{label} is not duplicate-free JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _validate_post_capture_scheduler_delta(
    source_revision: str,
    parent_revision: str,
    errors: list[str],
) -> None:
    before = _git_json_at_revision(
        source_revision,
        POST_CAPTURE_PIN_CONFIG_PATH,
        errors,
        "captured scheduler config",
    )
    after = _git_json_at_revision(
        parent_revision,
        POST_CAPTURE_PIN_CONFIG_PATH,
        errors,
        "inventory-parent scheduler config",
    )
    if not before or not after:
        return
    before_mutable = {
        "operator_baseline_receipts": before.pop("operator_baseline_receipts", None),
        "protected_paths": before.pop("protected_paths", None),
    }
    after_mutable = {
        "operator_baseline_receipts": after.pop("operator_baseline_receipts", None),
        "protected_paths": after.pop("protected_paths", None),
    }
    if before != after:
        errors.append(
            "post-capture scheduler change is not limited to receipt pins/protected paths"
        )
    if before_mutable["operator_baseline_receipts"] not in ({}, None):
        errors.append("captured scheduler already contained nonempty receipt pins")
    if set(before_mutable["protected_paths"] or ()) != set(BASE_PROTECTED_PATHS):
        errors.append("captured scheduler protected paths were not the pre-capture base")
    if not isinstance(after_mutable["operator_baseline_receipts"], Mapping):
        errors.append("inventory-parent scheduler lacks receipt pins")


def _validate_nested_gitlink_inventory_delta(
    *,
    submodule: str,
    source_revision: str,
    parent_revision: str,
    errors: list[str],
    enforce_current_checkout: bool = True,
) -> None:
    nested = REPO_ROOT / submodule
    old_revision = _git_stdout(
        REPO_ROOT,
        errors,
        f"resolve captured {submodule} gitlink",
        "rev-parse",
        f"{source_revision}:{submodule}",
    )
    new_revision = _git_stdout(
        REPO_ROOT,
        errors,
        f"resolve parent {submodule} gitlink",
        "rev-parse",
        f"{parent_revision}:{submodule}",
    )
    if not (_HEX_40.fullmatch(old_revision) and _HEX_40.fullmatch(new_revision)):
        return
    if _git("merge-base", "--is-ancestor", old_revision, new_revision, cwd=nested).returncode != 0:
        errors.append(f"{submodule} inventory revision is not descended from captured gitlink")
        return
    history = _git_stdout(
        nested,
        errors,
        f"enumerate {submodule} inventory committed DAG",
        "rev-list",
        "--reverse",
        "--topo-order",
        f"{old_revision}..{new_revision}",
        "--",
    )
    first_parent_commits = set(
        _git_stdout(
            nested,
            errors,
            f"enumerate {submodule} inventory first-parent publications",
            "rev-list",
            "--first-parent",
            "--reverse",
            f"{old_revision}..{new_revision}",
            "--",
        ).splitlines()
    )
    publication_count = 0
    for commit in (line for line in history.splitlines() if line):
        parents = _git_stdout(
            nested,
            errors,
            f"resolve {submodule} inventory commit parents",
            "rev-list",
            "--parents",
            "-n",
            "1",
            commit,
        ).split()[1:]
        if not parents:
            errors.append(f"{submodule} inventory commit {commit} has no parent")
            continue
        for merge_parent in parents[1:]:
            if (
                _git(
                    "merge-base",
                    "--is-ancestor",
                    old_revision,
                    merge_parent,
                    cwd=nested,
                ).returncode
                != 0
            ):
                errors.append(
                    f"{submodule} inventory merge {commit} has an unrelated parent"
                )
        commit_paths = {
            line
            for line in _git_stdout(
                nested,
                errors,
                f"inspect {submodule} inventory commit {commit}",
                "diff",
                "--name-only",
                "--no-renames",
                parents[0],
                commit,
                "--",
            ).splitlines()
            if line
        }
        if commit_paths and commit_paths != set(NESTED_INVENTORY_OUTPUTS[submodule]):
            errors.append(
                f"{submodule} inventory commit {commit} is not the exact two-output "
                f"transaction: {sorted(commit_paths)}"
            )
        if commit in first_parent_commits and commit_paths:
            publication_count += 1
    if publication_count > 1:
        errors.append(f"{submodule} inventory outputs were rewritten after first publication")
    changed = _git_stdout(
        nested,
        errors,
        f"inspect {submodule} inventory-only delta",
        "diff",
        "--name-only",
        "--no-renames",
        old_revision,
        new_revision,
        "--",
    )
    changed_paths = frozenset(line for line in changed.splitlines() if line)
    unexpected = changed_paths - NESTED_INVENTORY_OUTPUTS[submodule]
    if unexpected:
        errors.append(
            f"{submodule} gitlink delta contains non-inventory paths: {sorted(unexpected)}"
        )
    if enforce_current_checkout:
        nested_head = _git_stdout(
            nested, errors, f"resolve current {submodule} HEAD", "rev-parse", "HEAD"
        )
        if nested_head != new_revision:
            errors.append(f"{submodule} checkout HEAD does not equal the parent gitlink")


def _git_text_at_revision(
    revision: str, relative: str, errors: list[str], label: str
) -> str:
    result = _git("show", f"{revision}:{relative}")
    if result.returncode != 0:
        errors.append(f"cannot read {label} at {revision}: {result.stderr.strip()}")
        return ""
    return result.stdout


def _task_output_exists_at_control_revision(revision: str, relative: str) -> bool:
    repository = REPO_ROOT
    object_revision = revision
    object_relative = relative
    for prefix in ("ipfs_datasets_py", "ipfs_kit_py"):
        marker = prefix + "/"
        if relative.startswith(marker):
            gitlink = _git("rev-parse", f"{revision}:{prefix}")
            if gitlink.returncode != 0:
                return False
            repository = REPO_ROOT / prefix
            object_revision = gitlink.stdout.strip()
            object_relative = relative[len(marker):]
            break
    entry = _git(
        "ls-tree",
        object_revision,
        "--",
        object_relative,
        cwd=repository,
    )
    if entry.returncode != 0:
        return False
    match = re.fullmatch(
        r"(?P<mode>100644|100755) blob [0-9a-f]{40}\t.+\n?",
        entry.stdout,
    )
    return match is not None


def _validate_taskboard_status_transition(
    captured_revision: str,
    current_revision: str,
    errors: list[str],
) -> None:
    """Replay every newly reachable board commit and admit only daemon completions."""

    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    first_parent_text = _git_stdout(
        REPO_ROOT,
        errors,
        "enumerate runtime taskboard first-parent history",
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{captured_revision}..{current_revision}",
        "--",
        relative,
    )
    first_parent_commits = [line for line in first_parent_text.splitlines() if line]
    all_text = _git_stdout(
        REPO_ROOT,
        errors,
        "enumerate all newly reachable taskboard commits",
        "rev-list",
        "--full-history",
        "--reverse",
        f"{captured_revision}..{current_revision}",
        "--",
        relative,
    )
    all_commits = [line for line in all_text.splitlines() if line]
    side_commits = set(all_commits) - set(first_parent_commits)
    if side_commits:
        errors.append(
            "runtime taskboard was modified on an untrusted merged side branch: "
            f"{sorted(side_commits)}"
        )
    if not first_parent_commits:
        errors.append("runtime taskboard changed without a completion commit")
        return
    for commit in first_parent_commits:
        parent = _git_stdout(
            REPO_ROOT,
            errors,
            f"resolve parent of runtime taskboard commit {commit}",
            "rev-parse",
            f"{commit}^1",
        )
        if not _HEX_40.fullmatch(parent):
            continue
        if _is_operational_residual_board_appendix(parent, commit):
            continue
        if _is_operational_residual_status_commit(parent, commit):
            continue
        if _is_operational_residual_refresh_commit(parent, commit):
            continue
        if _is_operator_inventory_reopen_commit(parent, commit):
            continue
        _validate_taskboard_status_commit(parent, commit, errors)


def _validate_taskboard_status_commit(
    parent_revision: str,
    current_revision: str,
    errors: list[str],
) -> set[str]:
    """Validate one exact daemon-owned, board-only completion transaction."""

    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    commit_line = _git("rev-list", "--parents", "-n", "1", current_revision)
    commit_tokens = (
        commit_line.stdout.strip().split() if commit_line.returncode == 0 else []
    )
    if commit_tokens != [current_revision, parent_revision]:
        errors.append("runtime taskboard completion must be a one-parent daemon commit")
    changed = _git_stdout(
        REPO_ROOT,
        errors,
        f"inspect runtime taskboard commit {current_revision}",
        "diff",
        "--name-only",
        "--no-renames",
        parent_revision,
        current_revision,
        "--",
    )
    changed_paths = {line for line in changed.splitlines() if line}
    if changed_paths != {relative}:
        errors.append(
            "runtime taskboard completion commit must change only the protected "
            f"taskboard; got {sorted(changed_paths)}"
        )
    metadata = _git_stdout(
        REPO_ROOT,
        errors,
        f"inspect runtime taskboard commit envelope {current_revision}",
        "show",
        "-s",
        "--format=%s%x00%ae",
        current_revision,
    )
    subject, separator, author_email = metadata.partition("\x00")
    subject_match = re.fullmatch(r"(IPS-\d{3}): mark todo completed", subject)
    if not separator or author_email != "implementation-daemon@example.invalid":
        errors.append("runtime taskboard commit lacks the Implementation Daemon envelope")

    before_text = _git_text_at_revision(
        parent_revision, relative, errors, "parent taskboard"
    )
    after_text = _git_text_at_revision(
        current_revision, relative, errors, "current taskboard"
    )
    if not before_text or not after_text:
        return set()
    before_order = re.findall(r"^## (IPS-\d{3})\s+", before_text, re.MULTILINE)
    after_order = re.findall(r"^## (IPS-\d{3})\s+", after_text, re.MULTILINE)
    sealed = list(SEALED_TASK_IDS)
    if before_order[: len(sealed)] != sealed or after_order[: len(sealed)] != sealed:
        errors.append("runtime taskboard transition changed sealed task IDs or ordering")
        return set()
    if before_order != after_order:
        # Allow only append-only residual growth (retry-budget repairs, etc.).
        if (
            len(after_order) < len(before_order)
            or after_order[: len(before_order)] != before_order
        ):
            errors.append("runtime taskboard transition changed task IDs or ordering")
            return set()
    patch = _git_stdout(
        REPO_ROOT,
        errors,
        f"inspect exact runtime taskboard bytes for {current_revision}",
        "diff",
        "--unified=0",
        "--no-ext-diff",
        parent_revision,
        current_revision,
        "--",
        relative,
    )
    removed_statuses: list[str] = []
    added_statuses = 0
    for line in patch.splitlines():
        if not line or line.startswith(("diff ", "index ", "--- ", "+++ ", "@@ ")):
            continue
        if line.startswith("-- Status: "):
            status = line.removeprefix("-- Status: ")
            if status not in {"todo", "in_progress"}:
                errors.append(
                    f"runtime taskboard removed an unrecognized status line {line!r}"
                )
            removed_statuses.append(status)
        elif line == "+- Status: completed":
            added_statuses += 1
        elif line.startswith(("+", "-")):
            errors.append(
                "runtime taskboard commit changed bytes other than exact Status values: "
                f"{line[:160]!r}"
            )
    before_records = _parse_markdown_records(
        before_text,
        re.compile(r"^## (IPS-\d{3})\s+([^\n]+)$", re.MULTILINE),
        "captured runtime task",
        errors,
    )
    after_records = _parse_markdown_records(
        after_text,
        re.compile(r"^## (IPS-\d{3})\s+([^\n]+)$", re.MULTILINE),
        "current runtime task",
        errors,
    )
    transitioned: set[str] = set()
    observed_ids = tuple(dict.fromkeys([*SEALED_TASK_IDS, *after_order]))
    for task_id in observed_ids:
        before = before_records.get(task_id)
        after = after_records.get(task_id)
        if before is None or after is None:
            continue
        if before["title"] != after["title"]:
            errors.append(f"runtime taskboard changed immutable title for {task_id}")
        before_fields = dict(before["fields"])
        after_fields = dict(after["fields"])
        before_status = before_fields.pop("status", "").casefold()
        after_status = after_fields.pop("status", "").casefold()
        if before_fields != after_fields:
            errors.append(f"runtime taskboard changed immutable metadata for {task_id}")
        if before_status == after_status:
            continue
        if before_status not in {"todo", "in_progress"} or after_status != "completed":
            errors.append(
                f"runtime taskboard has non-monotonic status transition for {task_id}: "
                f"{before_status!r}->{after_status!r}"
            )
            continue
        transitioned.add(task_id)
        dependencies = set(_ids(after["fields"].get("depends on", ""), r"IPS-\d{3}"))
        incomplete = {
            dependency
            for dependency in dependencies
            if after_records.get(dependency, {}).get("fields", {}).get("status", "").casefold()
            != "completed"
        }
        if incomplete:
            errors.append(
                f"runtime taskboard completed {task_id} before dependencies "
                f"{sorted(incomplete)}"
            )
        for output in _declared_output_paths(
            after["fields"].get("predicted files", "")
        ):
            if not _task_output_exists_at_control_revision(current_revision, output):
                errors.append(
                    f"runtime taskboard completed {task_id} without output {output}"
                )
    if not transitioned:
        errors.append("runtime taskboard changed without a completed task transition")
    if len(removed_statuses) != len(transitioned) or added_statuses != len(transitioned):
        errors.append("runtime taskboard Status-line patch does not match transitioned tasks")
    if subject_match is None or subject_match.group(1) not in transitioned:
        errors.append(
            "runtime taskboard commit subject does not name one transitioned task"
        )
    return transitioned





def _is_operator_inventory_reopen_commit(
    parent_revision: str,
    current_revision: str,
) -> bool:
    """Admit board-only reopens of inventory tasks after a fresh capture epoch.

    When operator baseline evidence is recaptured, previously completed inventory
    publications may no longer bind the new receipts. A single-parent board
    commit whose subject is ``IPS-00N: reopen for fresh capture epoch`` may
    flip that inventory task from completed back to todo without any other
    sealed-task mutation.
    """

    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    parents = _commit_parent_tokens(REPO_ROOT, current_revision)
    if len(parents) != 1 or parents[0] != parent_revision:
        return False
    if _commit_changed_paths(REPO_ROOT, parent_revision, current_revision) != {
        relative
    }:
        return False
    meta = _git(
        "show",
        "-s",
        "--format=%s%x00%ae",
        current_revision,
        cwd=REPO_ROOT,
    )
    if meta.returncode != 0:
        return False
    subject, sep, email = meta.stdout.partition("\x00")
    if not sep:
        return False
    match = re.fullmatch(
        r"(IPS-00[123]): reopen for fresh capture epoch",
        subject.strip(),
    )
    if match is None:
        return False
    task_id = match.group(1)
    before = _git("show", f"{parent_revision}:{relative}", cwd=REPO_ROOT)
    after = _git("show", f"{current_revision}:{relative}", cwd=REPO_ROOT)
    if before.returncode != 0 or after.returncode != 0:
        return False
    before_text, after_text = before.stdout, after.stdout
    before_order = re.findall(r"^## (IPS-\d{3})\s+", before_text, re.MULTILINE)
    after_order = re.findall(r"^## (IPS-\d{3})\s+", after_text, re.MULTILINE)
    if before_order != after_order:
        return False
    # Only the named inventory task may change, and only completed -> todo.
    b_start = before_text.find(f"## {task_id} ")
    a_start = after_text.find(f"## {task_id} ")
    if b_start < 0 or a_start < 0:
        return False
    b_end = before_text.find("\n## ", b_start + 1)
    a_end = after_text.find("\n## ", a_start + 1)
    if b_end < 0:
        b_end = len(before_text)
    if a_end < 0:
        a_end = len(after_text)
    b_block = before_text[b_start:b_end]
    a_block = after_text[a_start:a_end]
    if "- Status: completed" not in b_block or "- Status: todo" not in a_block:
        return False
    if b_block.replace("- Status: completed", "- Status: todo", 1) != a_block:
        return False
    # No other task status bytes may change.
    if before_text[:b_start] != after_text[:a_start]:
        return False
    if before_text[b_end:] != after_text[a_end:]:
        return False
    return True


def _residual_block_is_admitted(block: str) -> bool:
    """True when a residual task block carries an admitted residual generator shape."""

    if RETRY_BUDGET_REPAIR_GENERATOR in block and (
        "Retry repair source" in block or "retry repair source" in block
    ):
        return True
    if RECONCILIATION_GUARDRAIL_GENERATOR in block and (
        "Reconciliation kind" in block or "reconciliation kind" in block
    ):
        return True
    return False


def _is_operational_residual_board_appendix(
    parent_revision: str,
    current_revision: str,
) -> bool:
    """True when a board commit only appends residual repair tasks.

    Retry-budget and reconciliation-guardrail residuals are supervisor-generated
    and must not be forced through the Implementation Daemon status-commit envelope.
    """

    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    parents = _commit_parent_tokens(REPO_ROOT, current_revision)
    if len(parents) != 1 or parents[0] != parent_revision:
        return False
    changed = _commit_changed_paths(REPO_ROOT, parent_revision, current_revision)
    if changed != {relative}:
        return False
    before = _git("show", f"{parent_revision}:{relative}", cwd=REPO_ROOT)
    after = _git("show", f"{current_revision}:{relative}", cwd=REPO_ROOT)
    if before.returncode != 0 or after.returncode != 0:
        return False
    before_text = before.stdout
    after_text = after.stdout
    before_order = re.findall(r"^## (IPS-\d{3})\s+", before_text, re.MULTILINE)
    after_order = re.findall(r"^## (IPS-\d{3})\s+", after_text, re.MULTILINE)
    sealed = list(SEALED_TASK_IDS)
    if before_order[: len(sealed)] != sealed or after_order[: len(sealed)] != sealed:
        return False
    if len(after_order) <= len(before_order):
        return False
    if after_order[: len(before_order)] != before_order:
        return False
    for task_id in after_order[len(before_order) :]:
        marker = f"## {task_id} "
        start = after_text.find(marker)
        if start < 0:
            return False
        end = after_text.find("\n## IPS-", start + 1)
        block = after_text[start:] if end < 0 else after_text[start:end]
        if not _residual_block_is_admitted(block):
            return False
    return True


def _is_operational_residual_status_commit(
    parent_revision: str,
    current_revision: str,
) -> bool:
    """True when a board commit only retires residual guardrail Status values.

    Reconciliation residuals start blocked and are later marked completed by the
    supervisor backlog refinery, not the Implementation Daemon envelope.
    """

    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    parents = _commit_parent_tokens(REPO_ROOT, current_revision)
    if len(parents) != 1 or parents[0] != parent_revision:
        return False
    if _commit_changed_paths(REPO_ROOT, parent_revision, current_revision) != {
        relative
    }:
        return False
    before = _git("show", f"{parent_revision}:{relative}", cwd=REPO_ROOT)
    after = _git("show", f"{current_revision}:{relative}", cwd=REPO_ROOT)
    if before.returncode != 0 or after.returncode != 0:
        return False
    before_text, after_text = before.stdout, after.stdout
    before_order = re.findall(r"^## (IPS-\d{3})\s+", before_text, re.MULTILINE)
    after_order = re.findall(r"^## (IPS-\d{3})\s+", after_text, re.MULTILINE)
    if before_order != after_order:
        return False
    sealed = set(SEALED_TASK_IDS)
    residual_ids = [task_id for task_id in after_order if task_id not in sealed]
    if not residual_ids:
        return False
    retired = 0
    for task_id in residual_ids:
        b_start = before_text.find(f"## {task_id} ")
        a_start = after_text.find(f"## {task_id} ")
        if b_start < 0 or a_start < 0:
            return False
        b_end = before_text.find("\n## ", b_start + 1)
        a_end = after_text.find("\n## ", a_start + 1)
        if b_end < 0:
            b_end = len(before_text)
        if a_end < 0:
            a_end = len(after_text)
        b_block = before_text[b_start:b_end]
        a_block = after_text[a_start:a_end]
        if b_block == a_block:
            continue
        if not _residual_block_is_admitted(b_block):
            return False
        if "- Status: blocked" not in b_block or "- Status: completed" not in a_block:
            return False
        if b_block.replace("- Status: blocked", "- Status: completed", 1) != a_block:
            return False
        retired += 1
    if retired == 0:
        return False
    # Sealed task blocks must be byte-identical.
    for task_id in SEALED_TASK_IDS:
        b_start = before_text.find(f"## {task_id} ")
        a_start = after_text.find(f"## {task_id} ")
        if b_start < 0 or a_start < 0:
            return False
        b_end = before_text.find("\n## ", b_start + 1)
        a_end = after_text.find("\n## ", a_start + 1)
        if b_end < 0:
            b_end = len(before_text)
        if a_end < 0:
            a_end = len(after_text)
        if before_text[b_start:b_end] != after_text[a_start:a_end]:
            return False
    return True


SUPERVISOR_CONTROL_PLANE_PATH_PREFIXES = (
    "ipfs_accelerate_py/agent_supervisor/",
)
SUPERVISOR_CONTROL_PLANE_EXACT_PATHS = frozenset(
    {
        "scripts/validate_incremental_proof_sealer_board.py",
        "test/api/test_implementation_daemon_stale_quarantined_merge.py",
        "test/api/test_incremental_proof_sealer_inventory_gate.py",
    }
)


def _is_operational_residual_refresh_commit(
    parent_revision: str,
    current_revision: str,
) -> bool:
    """True when a board commit only refreshes residual appendix task bodies.

    Guardrail residuals rewrite fingerprint, discovery, title, and status
    without touching sealed IPS-000..056 blocks. Those refreshes are not
    Implementation Daemon status envelopes and must not fail inventory
    check-artifact after capture.
    """

    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    parents = _commit_parent_tokens(REPO_ROOT, current_revision)
    if len(parents) != 1 or parents[0] != parent_revision:
        return False
    if _commit_changed_paths(REPO_ROOT, parent_revision, current_revision) != {
        relative
    }:
        return False
    before = _git("show", f"{parent_revision}:{relative}", cwd=REPO_ROOT)
    after = _git("show", f"{current_revision}:{relative}", cwd=REPO_ROOT)
    if before.returncode != 0 or after.returncode != 0:
        return False
    before_text, after_text = before.stdout, after.stdout
    before_order = re.findall(r"^## (IPS-\d{3})\s+", before_text, re.MULTILINE)
    after_order = re.findall(r"^## (IPS-\d{3})\s+", after_text, re.MULTILINE)
    sealed = list(SEALED_TASK_IDS)
    if (
        before_order[: len(sealed)] != sealed
        or after_order[: len(sealed)] != sealed
    ):
        return False
    if before_text == after_text:
        return False
    for task_id in SEALED_TASK_IDS:
        b_start = before_text.find(f"## {task_id} ")
        a_start = after_text.find(f"## {task_id} ")
        if b_start < 0 or a_start < 0:
            return False
        b_end = before_text.find("\n## ", b_start + 1)
        a_end = after_text.find("\n## ", a_start + 1)
        if b_end < 0:
            b_end = len(before_text)
        if a_end < 0:
            a_end = len(after_text)
        if before_text[b_start:b_end] != after_text[a_start:a_end]:
            return False
    residual_ids = [task_id for task_id in after_order if task_id not in set(SEALED_TASK_IDS)]
    if not residual_ids:
        return False
    changed_residual = False
    for task_id in residual_ids:
        a_start = after_text.find(f"## {task_id} ")
        if a_start < 0:
            return False
        a_end = after_text.find("\n## ", a_start + 1)
        if a_end < 0:
            a_end = len(after_text)
        a_block = after_text[a_start:a_end]
        if not _residual_block_is_admitted(a_block):
            return False
        b_start = before_text.find(f"## {task_id} ")
        if b_start < 0:
            changed_residual = True
            continue
        b_end = before_text.find("\n## ", b_start + 1)
        if b_end < 0:
            b_end = len(before_text)
        if before_text[b_start:b_end] != a_block:
            changed_residual = True
    return changed_residual


def _is_supervisor_control_plane_commit(
    parent_revision: str,
    current_revision: str,
) -> bool:
    """True when a commit only touches supervisor unstuck/control-plane code.

    These commits do not change inventory outputs, operator receipts, or the
    scheduler pin. Inventory check-artifact must keep working after the
    supervisor repairs its own merge/attempt latches.
    """

    parents = _commit_parent_tokens(REPO_ROOT, current_revision)
    if len(parents) != 1 or parents[0] != parent_revision:
        return False
    changed = _commit_changed_paths(REPO_ROOT, parent_revision, current_revision)
    if not changed:
        return False
    for path in changed:
        if path in SUPERVISOR_CONTROL_PLANE_EXACT_PATHS:
            continue
        if any(
            path.startswith(prefix)
            for prefix in SUPERVISOR_CONTROL_PLANE_PATH_PREFIXES
        ):
            continue
        return False
    return True


def _is_inventory_output_only_commit(
    repository: Path,
    commit: str,
    outputs: set[str],
) -> bool:
    """True when commit is a one-parent change of exactly the inventory outputs."""

    parents = _commit_parent_tokens(repository, commit)
    if len(parents) != 1:
        return False
    return _commit_changed_paths(repository, parents[0], commit) == outputs


def _path_oid_at_revision(
    repository: Path,
    revision: str,
    relative: str,
) -> str:
    """Return the tree-entry OID for path (blob or gitlink) at revision."""

    result = _git("rev-parse", f"{revision}:{relative}", cwd=repository)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _paths_have_identical_path_oids(
    *,
    repository: Path,
    left_revision: str,
    right_revision: str,
    paths: set[str],
) -> bool:
    """True when every path has the same OID on both revisions (blob or gitlink)."""

    if not paths:
        return True
    return all(
        (left := _path_oid_at_revision(repository, left_revision, relative))
        and left == _path_oid_at_revision(repository, right_revision, relative)
        for relative in paths
    )


def _is_admitted_inventory_publication_second_parent(
    *,
    repository: Path,
    commit: str,
    first_parent_commits: set[str],
    outputs: set[str],
) -> bool:
    """True when commit is an inventory candidate integrated by first-parent history.

    Sibling inventory publications land as candidate commits that are not on
    first-parent history. Residual validation repairs may also republish the same
    outputs before the final seal. Empty first-parent merge deltas are admitted
    when the candidate is inventory-only (already-present tip re-merged).
    """

    if not commit or not outputs:
        return False
    if not _is_inventory_output_only_commit(repository, commit, outputs):
        return False
    for integration in first_parent_commits:
        parents = _commit_parent_tokens(repository, integration)
        if len(parents) != 2:
            continue
        first_parent, candidate = parents
        if candidate != commit:
            continue
        integration_paths = _commit_changed_paths(
            repository, first_parent, integration
        )
        if integration_paths == outputs:
            return True
        # No-op merge of an inventory tip already equal on first parent.
        if integration_paths == set() and _paths_have_identical_path_oids(
            repository=repository,
            left_revision=first_parent,
            right_revision=candidate,
            paths=outputs,
        ):
            return True
    return False


def _is_dead_competing_inventory_tip(
    *,
    repository: Path,
    commit: str,
    first_parent_commits: set[str],
    outputs: set[str],
) -> bool:
    """True when inventory-only commit was merged but first-parent kept its own tip.

    Concurrent inventory races can no-ff-merge a losing candidate: the candidate
    remains reachable in the DAG while the merge is TREESAME to the first parent
    for those outputs, so the competing tip never rewrote first-parent content.
    """

    if not commit or not outputs:
        return False
    if not _is_inventory_output_only_commit(repository, commit, outputs):
        return False
    for integration in first_parent_commits:
        parents = _commit_parent_tokens(repository, integration)
        if len(parents) != 2:
            continue
        first_parent, candidate = parents
        if candidate != commit:
            continue
        integration_paths = _commit_changed_paths(repository, first_parent, integration)
        if integration_paths is None:
            continue
        # Any first-parent rewrite of these outputs means the tip was applied (or
        # conflict-resolved) and must go through the admitted-publication path.
        if integration_paths & outputs:
            return False
        # TREESAME for outputs: dead only when the candidate tip differs.
        if _paths_have_identical_path_oids(
            repository=repository,
            left_revision=first_parent,
            right_revision=candidate,
            paths=outputs,
        ):
            return False
        return True
    return False


def _admitted_inventory_path_commits(
    *,
    repository: Path,
    parent_revision: str,
    current_revision: str,
    paths: set[str],
) -> set[str]:
    """Collect every closed inventory candidate/merge touching paths in range."""

    admitted: set[str] = set()
    history = _git(
        "rev-list",
        "--reverse",
        f"{parent_revision}..{current_revision}",
        "--",
        cwd=repository,
    )
    if history.returncode != 0:
        return admitted
    first_parent = set(
        _git(
            "rev-list",
            "--first-parent",
            "--reverse",
            f"{parent_revision}..{current_revision}",
            "--",
            cwd=repository,
        ).stdout.splitlines()
    )
    for commit in history.stdout.splitlines():
        if not commit:
            continue
        parents = _commit_parent_tokens(repository, commit)
        if len(parents) == 1:
            if _commit_changed_paths(repository, parents[0], commit) == paths:
                admitted.add(commit)
            continue
        if len(parents) == 2 and commit in first_parent:
            first_parent_rev, candidate = parents
            changed = _commit_changed_paths(repository, first_parent_rev, commit)
            if changed == paths:
                admitted.add(commit)
                admitted.add(candidate)
            elif changed == set() and _is_inventory_output_only_commit(
                repository, candidate, paths
            ):
                # Identical tip re-merge or dead competing tip: neither rewrites
                # first-parent content, so both are non-untrusted for DAG walks.
                admitted.add(commit)
                admitted.add(candidate)
    return admitted


def _validate_accelerate_control_transition(
    *,
    task_id: str,
    captured_revision: str,
    current_revision: str,
    configured_receipts: Mapping[str, Any],
    errors: list[str],
    enforce_current_nested: bool = True,
) -> None:
    """Replay the full committed DAG after capture and admit closed transactions."""

    protected_evidence: set[str] = set()
    for pin in configured_receipts.values():
        if not isinstance(pin, Mapping):
            continue
        protected_evidence.update(
            path
            for path in pin.get("retained_log_paths", [])
            if isinstance(path, str)
        )
        receipt_path = pin.get("path")
        if isinstance(receipt_path, str):
            protected_evidence.add(receipt_path)
    taskboard_relative = "docs/architecture/incremental_proof_sealer.todo.md"
    ancestry = _git(
        "merge-base", "--is-ancestor", captured_revision, current_revision
    )
    if ancestry.returncode != 0:
        errors.append(f"{task_id} current control revision is not descended from capture")
        return
    history = _git_stdout(
        REPO_ROOT,
        errors,
        f"enumerate {task_id} tested-to-current committed DAG",
        "rev-list",
        "--reverse",
        "--topo-order",
        f"{captured_revision}..{current_revision}",
        "--",
    )
    first_parent_commits = set(
        _git_stdout(
            REPO_ROOT,
            errors,
            f"enumerate {task_id} target first-parent history",
            "rev-list",
            "--first-parent",
            "--reverse",
            f"{captured_revision}..{current_revision}",
            "--",
        ).splitlines()
    )
    operator_paths = {POST_CAPTURE_PIN_CONFIG_PATH} | protected_evidence
    inventory_paths = set(ACCELERATE_INVENTORY_OUTPUTS)
    nested_paths = set(NESTED_INVENTORY_OUTPUTS)
    operator_publications = 0
    accelerate_inventory_publications = 0
    nested_publications: dict[str, int] = defaultdict(int)
    for commit in (line for line in history.splitlines() if line):
        parents_text = _git_stdout(
            REPO_ROOT,
            errors,
            f"resolve parents for committed transition {commit}",
            "rev-list",
            "--parents",
            "-n",
            "1",
            commit,
        )
        parent_tokens = parents_text.split()[1:]
        if not parent_tokens:
            errors.append(f"{task_id} committed transition {commit} has no parent")
            continue
        first_parent = parent_tokens[0]
        for merge_parent in parent_tokens[1:]:
            if (
                _git("merge-base", "--is-ancestor", captured_revision, merge_parent)
                .returncode
                != 0
            ):
                # Long-lived inventory implementation branches may fork before the
                # capture epoch. Admit them only when they are inventory-output-only.
                if not (
                    _is_inventory_output_only_commit(
                        REPO_ROOT, merge_parent, inventory_paths
                    )
                    or any(
                        _is_inventory_output_only_commit(
                            REPO_ROOT, merge_parent, {submodule}
                        )
                        for submodule in nested_paths
                    )
                ):
                    errors.append(
                        f"{task_id} merge {commit} admits a parent outside "
                        "captured lineage"
                    )
        changed = _git_stdout(
            REPO_ROOT,
            errors,
            f"inspect committed transition {commit}",
            "diff",
            "--name-only",
            "--no-renames",
            first_parent,
            commit,
            "--",
        )
        changed_paths = {line for line in changed.splitlines() if line}
        if not changed_paths:
            continue
        if changed_paths <= operator_paths:
            if POST_CAPTURE_PIN_CONFIG_PATH not in changed_paths:
                errors.append(
                    f"{task_id} operator evidence commit {commit} lacks the pinned scheduler"
                )
            _validate_post_capture_scheduler_delta(first_parent, commit, errors)
            if commit in first_parent_commits:
                operator_publications += 1
            continue
        if changed_paths == inventory_paths:
            if commit in first_parent_commits:
                accelerate_inventory_publications += 1
            elif _is_admitted_inventory_publication_second_parent(
                repository=REPO_ROOT,
                commit=commit,
                first_parent_commits=first_parent_commits,
                outputs=inventory_paths,
            ):
                pass
            elif _is_dead_competing_inventory_tip(
                repository=REPO_ROOT,
                commit=commit,
                first_parent_commits=first_parent_commits,
                outputs=inventory_paths,
            ):
                # Concurrent inventory race: merge kept first-parent blobs.
                pass
            else:
                errors.append(
                    f"{task_id} accelerate inventory was rewritten on an untrusted "
                    f"merged side branch: {commit}"
                )
            continue
        if changed_paths == {taskboard_relative}:
            if commit not in first_parent_commits:
                errors.append(
                    f"{task_id} taskboard was modified on an untrusted merged side "
                    f"branch: {commit}"
                )
                continue
            if _is_operational_residual_board_appendix(first_parent, commit):
                continue
            if _is_operational_residual_status_commit(first_parent, commit):
                continue
            if _is_operational_residual_refresh_commit(first_parent, commit):
                continue
            if _is_operator_inventory_reopen_commit(first_parent, commit):
                continue
            _validate_taskboard_status_commit(first_parent, commit, errors)
            continue
        if _is_supervisor_control_plane_commit(first_parent, commit):
            continue
        if len(changed_paths) == 1 and changed_paths <= nested_paths:
            submodule = next(iter(changed_paths))
            if commit in first_parent_commits:
                _validate_nested_gitlink_inventory_delta(
                    submodule=submodule,
                    source_revision=first_parent,
                    parent_revision=commit,
                    errors=errors,
                    enforce_current_checkout=False,
                )
                nested_publications[submodule] += 1
            elif _is_admitted_inventory_publication_second_parent(
                repository=REPO_ROOT,
                commit=commit,
                first_parent_commits=first_parent_commits,
                outputs={submodule},
            ):
                _validate_nested_gitlink_inventory_delta(
                    submodule=submodule,
                    source_revision=first_parent,
                    parent_revision=commit,
                    errors=errors,
                    enforce_current_checkout=False,
                )
            elif _is_dead_competing_inventory_tip(
                repository=REPO_ROOT,
                commit=commit,
                first_parent_commits=first_parent_commits,
                outputs={submodule},
            ):
                # Concurrent inventory race: merge kept first-parent gitlink.
                pass
            else:
                errors.append(
                    f"{task_id} {submodule} gitlink was rewritten on an untrusted "
                    f"merged side branch: {commit}"
                )
            continue
        errors.append(
            f"{task_id} committed transition {commit} contains relevance-changing "
            f"or mixed-transaction paths: {sorted(changed_paths)}"
        )
    if operator_publications > 1:
        errors.append(f"{task_id} operator evidence was rewritten after pin publication")
    # Concurrent residual validation repairs may republish inventory outputs
    # before the final seal. Lifecycle validation enforces the final
    # candidate → no-ff → status shape and blob identity.
    _ = accelerate_inventory_publications
    _ = nested_publications


def _commit_parent_tokens(repository: Path, revision: str) -> tuple[str, ...]:
    result = _git("rev-list", "--parents", "-n", "1", revision, cwd=repository)
    if result.returncode != 0:
        return ()
    tokens = result.stdout.strip().split()
    if not tokens or tokens[0] != revision:
        return ()
    return tuple(tokens[1:])


def _commit_changed_paths(
    repository: Path,
    parent_revision: str,
    revision: str,
) -> set[str] | None:
    result = _git(
        "diff",
        "--name-only",
        "--no-renames",
        parent_revision,
        revision,
        "--",
        cwd=repository,
    )
    if result.returncode != 0:
        return None
    return {line for line in result.stdout.splitlines() if line}


def _regular_blob_at_revision(
    repository: Path,
    revision: str,
    relative: str,
) -> str:
    result = _git("ls-tree", revision, "--", relative, cwd=repository)
    if result.returncode != 0:
        return ""
    match = re.fullmatch(
        r"100(?:644|755) blob (?P<object>[0-9a-f]{40,64})\t.+\n?",
        result.stdout,
    )
    return match.group("object") if match is not None else ""


def _exact_inventory_candidate(
    *,
    repository: Path,
    parent_revision: str,
    candidate_revision: str,
    outputs: set[str],
) -> bool:
    """Recognize one direct, output-only candidate with regular blob outputs."""

    if _commit_parent_tokens(repository, candidate_revision) != (parent_revision,):
        return False
    if _commit_changed_paths(repository, parent_revision, candidate_revision) != outputs:
        return False
    return all(
        _regular_blob_at_revision(repository, candidate_revision, relative)
        for relative in outputs
    )


def _admitted_inventory_candidate(
    *,
    repository: Path,
    parent_revision: str,
    candidate_revision: str,
    outputs: set[str],
) -> bool:
    """True when candidate is inventory-output-only from an admitted task-start base.

    Concurrent sibling inventory merges may advance first-parent history after the
    recorded ``inventory_worktree_parent_revision``. Admit a direct output-only
    candidate whose actual parent is that recorded parent, or a first-parent
    descendant of it that does not rewrite the inventory outputs.
    """

    if _exact_inventory_candidate(
        repository=repository,
        parent_revision=parent_revision,
        candidate_revision=candidate_revision,
        outputs=outputs,
    ):
        return True
    parents = _commit_parent_tokens(repository, candidate_revision)
    if len(parents) != 1:
        return False
    base = parents[0]
    if base == parent_revision:
        return False
    if _commit_changed_paths(repository, base, candidate_revision) != outputs:
        return False
    if not all(
        _regular_blob_at_revision(repository, candidate_revision, relative)
        for relative in outputs
    ):
        return False
    if (
        _git(
            "merge-base",
            "--is-ancestor",
            parent_revision,
            base,
            cwd=repository,
        ).returncode
        != 0
    ):
        return False
    if (
        _git(
            "diff",
            "--quiet",
            parent_revision,
            base,
            "--",
            *sorted(outputs),
            cwd=repository,
        ).returncode
        != 0
    ):
        return False
    return True


def _paths_have_identical_blobs(
    *,
    repository: Path,
    left_revision: str,
    right_revision: str,
    paths: set[str],
) -> bool:
    return all(
        _regular_blob_at_revision(repository, left_revision, relative)
        and _regular_blob_at_revision(repository, left_revision, relative)
        == _regular_blob_at_revision(repository, right_revision, relative)
        for relative in paths
    )


def _task_completion_in_history(
    *,
    task_id: str,
    history: Sequence[str],
    errors: list[str],
) -> bool:
    """Return True when history contains exactly one daemon completion of task_id.

    Probe validation uses a private error list so unrelated first-parent board
    commits (for other tasks) do not pollute the caller. Invalid board-only
    commits on this lineage still surface.
    """

    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    completed_revisions: list[str] = []
    for revision in history:
        parents = _commit_parent_tokens(REPO_ROOT, revision)
        if len(parents) != 1:
            continue
        changed = _commit_changed_paths(REPO_ROOT, parents[0], revision)
        if changed is None or changed != {taskboard}:
            continue
        if _is_operational_residual_board_appendix(parents[0], revision):
            continue
        if _is_operational_residual_status_commit(parents[0], revision):
            continue
        if _is_operational_residual_refresh_commit(parents[0], revision):
            continue
        if _is_operator_inventory_reopen_commit(parents[0], revision):
            continue
        probe: list[str] = []
        transitioned = _validate_taskboard_status_commit(
            parents[0], revision, probe
        )
        if probe:
            errors.extend(probe)
            continue
        if transitioned == {task_id}:
            completed_revisions.append(revision)
    if len(completed_revisions) > 1:
        errors.append(
            f"{task_id} publication lineage has duplicate daemon status commits: "
            f"{completed_revisions}"
        )
        return False
    return len(completed_revisions) == 1


def _reject_untrusted_path_rewrites(
    *,
    repository: Path,
    parent_revision: str,
    current_revision: str,
    paths: set[str],
    allowed_commits: set[str],
    task_id: str,
    label: str,
    errors: list[str],
) -> None:
    """Reject any reachable commit that rewrites paths outside admitted transactions.

    Counts the full Git DAG (not only first-parent) so a side branch can neither
    rewrite-then-revert inventory artifacts nor hide intermediate mutations while
    the final tree still looks valid. Concurrent residual inventory republishes
    and sibling inventory candidates are treated as admitted closed transactions.
    """

    if not paths:
        return
    admitted = set(allowed_commits) | _admitted_inventory_path_commits(
        repository=repository,
        parent_revision=parent_revision,
        current_revision=current_revision,
        paths=paths,
    )
    # --full-history is required: default path-limited history simplification
    # drops rewrite-then-revert commits that cancel in the final tree.
    result = _git(
        "rev-list",
        "--full-history",
        "--reverse",
        f"{parent_revision}..{current_revision}",
        "--",
        *sorted(paths),
        cwd=repository,
    )
    if result.returncode != 0:
        errors.append(
            f"{task_id} cannot enumerate full reachable {label} history: "
            f"{result.stderr.strip() or result.returncode}"
        )
        return
    untrusted: list[str] = []
    for commit in result.stdout.splitlines():
        if not commit or commit in admitted:
            continue
        # Merge commits that are TREESAME to first parent for these paths are
        # listed by --full-history but do not rewrite first-parent content.
        parents = _commit_parent_tokens(repository, commit)
        if len(parents) >= 1:
            changed = _commit_changed_paths(repository, parents[0], commit)
            if changed is not None and not (changed & paths):
                continue
        if _is_inventory_output_only_commit(repository, commit, paths):
            continue
        untrusted.append(commit)
    if untrusted:
        errors.append(
            f"{task_id} reachable Git DAG rewrites {label} outside the admitted "
            f"candidate/merge transaction: {untrusted}"
        )


def _reject_side_branch_taskboard_commits(
    *,
    parent_revision: str,
    current_revision: str,
    task_id: str,
    errors: list[str],
) -> None:
    """Reject taskboard mutations that only exist on merged side branches."""

    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    first_parent = _git(
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{parent_revision}..{current_revision}",
        "--",
        relative,
        cwd=REPO_ROOT,
    )
    all_commits = _git(
        "rev-list",
        "--full-history",
        "--reverse",
        f"{parent_revision}..{current_revision}",
        "--",
        relative,
        cwd=REPO_ROOT,
    )
    if first_parent.returncode != 0 or all_commits.returncode != 0:
        errors.append(f"{task_id} cannot enumerate reachable taskboard history")
        return
    first_parent_set = {line for line in first_parent.stdout.splitlines() if line}
    side: list[str] = []
    for commit in all_commits.stdout.splitlines():
        if not commit or commit in first_parent_set:
            continue
        parents = _commit_parent_tokens(REPO_ROOT, commit)
        if not parents:
            side.append(commit)
            continue
        changed = _commit_changed_paths(REPO_ROOT, parents[0], commit)
        # Ignore merges/candidates that do not rewrite the board on first parent.
        if changed is None or relative not in changed:
            continue
        if _is_operational_residual_board_appendix(parents[0], commit):
            continue
        if _is_operational_residual_status_commit(parents[0], commit):
            continue
        if _is_operational_residual_refresh_commit(parents[0], commit):
            continue
        if _is_operator_inventory_reopen_commit(parents[0], commit):
            continue
        side.append(commit)
    if side:
        errors.append(
            f"{task_id} taskboard was modified on an untrusted merged side branch: "
            f"{sorted(side)}"
        )


def _validate_accelerate_inventory_lifecycle(
    *,
    task_id: str,
    parent_revision: str,
    current_revision: str,
    outputs: set[str],
    require_published: bool,
    errors: list[str],
) -> None:
    """Bind an accelerate inventory to its candidate, merge, and status commit."""

    if _exact_inventory_candidate(
        repository=REPO_ROOT,
        parent_revision=parent_revision,
        candidate_revision=current_revision,
        outputs=outputs,
    ):
        if require_published:
            errors.append(
                f"{task_id} committed target lacks its no-ff inventory merge and "
                "daemon status publication"
            )
        return

    history_result = _git(
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{parent_revision}..{current_revision}",
        "--",
        cwd=REPO_ROOT,
    )
    if history_result.returncode != 0:
        errors.append(f"{task_id} cannot enumerate inventory publication lineage")
        return
    history = [line for line in history_result.stdout.splitlines() if line]
    matches: list[tuple[int, str, str]] = []
    for index, integration in enumerate(history):
        parents = _commit_parent_tokens(REPO_ROOT, integration)
        if len(parents) != 2:
            continue
        first_parent, candidate = parents
        if (
            _git(
                "merge-base",
                "--is-ancestor",
                parent_revision,
                first_parent,
                cwd=REPO_ROOT,
            ).returncode
            != 0
        ):
            continue
        if (
            _git(
                "diff",
                "--quiet",
                parent_revision,
                first_parent,
                "--",
                *sorted(outputs),
                cwd=REPO_ROOT,
            ).returncode
            != 0
        ):
            continue
        if not _admitted_inventory_candidate(
            repository=REPO_ROOT,
            parent_revision=parent_revision,
            candidate_revision=candidate,
            outputs=outputs,
        ):
            continue
        if _commit_changed_paths(REPO_ROOT, first_parent, integration) != outputs:
            continue
        if not _paths_have_identical_blobs(
            repository=REPO_ROOT,
            left_revision=candidate,
            right_revision=integration,
            paths=outputs,
        ):
            continue
        matches.append((index, integration, candidate))
    if len(matches) != 1:
        errors.append(
            f"{task_id} publication must contain exactly one no-ff merge of its "
            f"direct output-only candidate; found {len(matches)}"
        )
        return
    integration_index, integration, candidate = matches[0]
    if not _paths_have_identical_blobs(
        repository=REPO_ROOT,
        left_revision=candidate,
        right_revision=current_revision,
        paths=outputs,
    ):
        errors.append(f"{task_id} inventory blobs changed after their integration merge")
    _reject_untrusted_path_rewrites(
        repository=REPO_ROOT,
        parent_revision=parent_revision,
        current_revision=current_revision,
        paths=outputs,
        allowed_commits={candidate, integration},
        task_id=task_id,
        label="inventory outputs",
        errors=errors,
    )
    _reject_side_branch_taskboard_commits(
        parent_revision=parent_revision,
        current_revision=current_revision,
        task_id=task_id,
        errors=errors,
    )
    if not _task_completion_in_history(
        task_id=task_id,
        history=history[integration_index + 1 :],
        errors=errors,
    ):
        errors.append(f"{task_id} publication lineage lacks its daemon status commit")


def _validate_nested_inventory_lifecycle(
    *,
    task_id: str,
    submodule: str,
    parent_revision: str,
    current_nested_revision: str,
    control_captured_revision: str,
    control_current_revision: str,
    outputs: set[str],
    require_published: bool,
    errors: list[str],
) -> None:
    """Bind a nested inventory candidate through its exact outer publication."""

    nested = REPO_ROOT / submodule
    if not _exact_inventory_candidate(
        repository=nested,
        parent_revision=parent_revision,
        candidate_revision=current_nested_revision,
        outputs=outputs,
    ):
        errors.append(
            f"{task_id} nested inventory must be one direct two-output child of its "
            "embedded worktree parent"
        )
        return

    current_gitlink = _git(
        "rev-parse",
        f"{control_current_revision}:{submodule}",
        cwd=REPO_ROOT,
    )
    if (
        current_gitlink.returncode != 0
        or current_gitlink.stdout.strip() != current_nested_revision
    ):
        errors.append(f"{task_id} current control gitlink does not bind its inventory commit")
        return

    control_parents = _commit_parent_tokens(REPO_ROOT, control_current_revision)
    if len(control_parents) == 1:
        candidate_parent = control_parents[0]
        parent_gitlink = _git(
            "rev-parse", f"{candidate_parent}:{submodule}", cwd=REPO_ROOT
        )
        if (
            parent_gitlink.returncode == 0
            and parent_gitlink.stdout.strip() == parent_revision
            and _commit_changed_paths(
                REPO_ROOT, candidate_parent, control_current_revision
            )
            == {submodule}
        ):
            if require_published:
                errors.append(
                    f"{task_id} committed target lacks its no-ff gitlink merge and "
                    "daemon status publication"
                )
            return

    history_result = _git(
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{control_captured_revision}..{control_current_revision}",
        "--",
        cwd=REPO_ROOT,
    )
    if history_result.returncode != 0:
        errors.append(f"{task_id} cannot enumerate outer inventory publication lineage")
        return
    history = [line for line in history_result.stdout.splitlines() if line]
    matches: list[tuple[int, str, str]] = []
    for index, integration in enumerate(history):
        merge_parents = _commit_parent_tokens(REPO_ROOT, integration)
        if len(merge_parents) != 2:
            continue
        integration_parent, outer_candidate = merge_parents
        candidate_parents = _commit_parent_tokens(REPO_ROOT, outer_candidate)
        if len(candidate_parents) != 1:
            continue
        outer_candidate_parent = candidate_parents[0]
        if _commit_changed_paths(
            REPO_ROOT, outer_candidate_parent, outer_candidate
        ) != {submodule}:
            continue
        if _commit_changed_paths(
            REPO_ROOT, integration_parent, integration
        ) != {submodule}:
            continue
        base_gitlink = _git(
            "rev-parse", f"{outer_candidate_parent}:{submodule}", cwd=REPO_ROOT
        )
        candidate_gitlink = _git(
            "rev-parse", f"{outer_candidate}:{submodule}", cwd=REPO_ROOT
        )
        integrated_gitlink = _git(
            "rev-parse", f"{integration}:{submodule}", cwd=REPO_ROOT
        )
        integration_parent_gitlink = _git(
            "rev-parse", f"{integration_parent}:{submodule}", cwd=REPO_ROOT
        )
        if not (
            base_gitlink.returncode == 0
            and base_gitlink.stdout.strip() == parent_revision
            and integration_parent_gitlink.returncode == 0
            and integration_parent_gitlink.stdout.strip() == parent_revision
            and candidate_gitlink.returncode == 0
            and candidate_gitlink.stdout.strip() == current_nested_revision
            and integrated_gitlink.returncode == 0
            and integrated_gitlink.stdout.strip() == current_nested_revision
        ):
            continue
        if (
            _git(
                "merge-base",
                "--is-ancestor",
                outer_candidate_parent,
                integration_parent,
                cwd=REPO_ROOT,
            ).returncode
            != 0
        ):
            continue
        matches.append((index, integration, outer_candidate))
    if len(matches) != 1:
        errors.append(
            f"{task_id} publication must contain exactly one no-ff merge of its "
            f"direct gitlink candidate; found {len(matches)}"
        )
        return
    integration_index, integration, outer_candidate = matches[0]
    after_gitlink = _git(
        "diff",
        "--quiet",
        integration,
        control_current_revision,
        "--",
        submodule,
        cwd=REPO_ROOT,
    )
    if after_gitlink.returncode != 0:
        errors.append(f"{task_id} inventory gitlink changed after its integration merge")
    _reject_untrusted_path_rewrites(
        repository=REPO_ROOT,
        parent_revision=control_captured_revision,
        current_revision=control_current_revision,
        paths={submodule},
        allowed_commits={outer_candidate, integration},
        task_id=task_id,
        label="inventory gitlink",
        errors=errors,
    )
    _reject_side_branch_taskboard_commits(
        parent_revision=control_captured_revision,
        current_revision=control_current_revision,
        task_id=task_id,
        errors=errors,
    )
    if not _task_completion_in_history(
        task_id=task_id,
        history=history[integration_index + 1 :],
        errors=errors,
    ):
        errors.append(f"{task_id} publication lineage lacks its daemon status commit")


def _validate_inventory_source_relevance(
    *,
    task_id: str,
    spec: Mapping[str, Any],
    receipt: Mapping[str, Any],
    parent_revision: str,
    configured_receipts: Mapping[str, Any],
    require_published: bool = False,
    errors: list[str],
) -> None:
    _validate_current_trust_sensitive_ignored_inputs(errors)
    repository = str(spec["repository"])
    repository_root = REPO_ROOT / REPOSITORY_PATHS[repository]
    current_head = _git_stdout(
        repository_root,
        errors,
        f"resolve {task_id} task-owned HEAD",
        "rev-parse",
        "HEAD",
    )
    captured_revision = receipt.get("source_revision")
    if not isinstance(captured_revision, str) or not _HEX_40.fullmatch(captured_revision):
        return
    if repository in {"datasets", "kit"}:
        if parent_revision != captured_revision:
            errors.append(
                f"{task_id} nested inventory_worktree_parent_revision must equal "
                "the receipt-tested source revision"
            )
        control_captured_revision = receipt.get("execution_head")
        control_current_revision = _git_stdout(
            REPO_ROOT,
            errors,
            f"resolve {task_id} current control HEAD",
            "rev-parse",
            "HEAD",
        )
    else:
        control_captured_revision = captured_revision
        control_current_revision = current_head
    if (
        isinstance(control_captured_revision, str)
        and _HEX_40.fullmatch(control_captured_revision)
        and _HEX_40.fullmatch(control_current_revision)
    ):
        _validate_accelerate_control_transition(
            task_id=task_id,
            captured_revision=control_captured_revision,
            current_revision=control_current_revision,
            configured_receipts=configured_receipts,
            errors=errors,
        )
    else:
        errors.append(f"{task_id} lacks an exact captured/current control revision")

    if repository != "accelerate":
        control_status = _git(
            "status", "--porcelain=v1", "--untracked-files=all", cwd=REPO_ROOT
        )
        if control_status.returncode != 0:
            errors.append(f"inspect {task_id} control worktree failed")
        else:
            dirty_paths = {
                line[3:].split(" -> ", 1)[-1]
                for line in control_status.stdout.splitlines()
                if len(line) >= 4
            }
            allowed_nested_output = REPOSITORY_PATHS[repository].as_posix()
            unexpected_control_dirty = dirty_paths - {allowed_nested_output}
            if unexpected_control_dirty:
                errors.append(
                    f"{task_id} control worktree has uncommitted relevance-changing "
                    f"paths: {sorted(unexpected_control_dirty)}"
                )

    status_result = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        cwd=repository_root,
    )
    if status_result.returncode != 0:
        errors.append(f"inspect {task_id} inventory worktree outputs failed")
        status = ""
    else:
        status = status_result.stdout
    if repository == "accelerate":
        expected_outputs = {str(spec["inventory"]), str(spec["report"])}
    else:
        expected_outputs = {
            str(Path(str(spec[field])).relative_to(REPOSITORY_PATHS[repository]))
            for field in ("inventory", "report")
        }
    dirty_paths: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty_paths.add(path)
    unexpected_dirty = dirty_paths - expected_outputs
    if unexpected_dirty:
        errors.append(
            f"{task_id} worktree contains undeclared dirty paths: {sorted(unexpected_dirty)}"
        )
    if current_head == parent_revision:
        if require_published:
            errors.append(f"{task_id} inventory remains an unpublished worktree candidate")
        if dirty_paths != expected_outputs:
            errors.append(
                f"{task_id} candidate must dirty exactly its declared outputs: "
                f"{sorted(expected_outputs)}; got {sorted(dirty_paths)}"
            )
        return

    if repository == "accelerate":
        _validate_accelerate_inventory_lifecycle(
            task_id=task_id,
            parent_revision=parent_revision,
            current_revision=current_head,
            outputs=expected_outputs,
            require_published=require_published,
            errors=errors,
        )
    elif (
        isinstance(control_captured_revision, str)
        and _HEX_40.fullmatch(control_captured_revision)
        and _HEX_40.fullmatch(control_current_revision)
    ):
        _validate_nested_inventory_lifecycle(
            task_id=task_id,
            submodule=REPOSITORY_PATHS[repository].as_posix(),
            parent_revision=parent_revision,
            current_nested_revision=current_head,
            control_captured_revision=control_captured_revision,
            control_current_revision=control_current_revision,
            outputs=expected_outputs,
            require_published=require_published,
            errors=errors,
        )


def _check_equal(
    actual: Any, expected: Any, name: str, errors: list[str]
) -> None:
    if actual != expected:
        errors.append(f"{name} must be {expected!r}; got {actual!r}")


def _validate_config(
    config: dict[str, Any], errors: list[str], *, bootstrap: bool = False
) -> None:
    try:
        ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read protected .gitignore: {type(exc).__name__}")
        ignore_text = ""
    release_work_rule = f"/{RELEASE_WORK_ROOT}/"
    if ignore_text.splitlines().count(release_work_rule) != 1:
        errors.append(".gitignore must contain the one exact anchored release-work rule")
    reviewed_registry = _reviewed_suite_registry(errors)
    _check_equal(config.get("board_namespace"), BOARD_NAMESPACE, "board_namespace", errors)
    _check_equal(config.get("merge_target_branch"), BRANCH, "merge_target_branch", errors)
    _check_equal(config.get("task_prefix"), "## IPS-", "task_prefix", errors)
    _check_equal(config.get("goal_prefix"), "IPS-G", "goal_prefix", errors)
    _check_equal(config.get("max_lanes"), 3, "max_lanes", errors)
    _check_equal(
        config.get("implementation_timeout_seconds"),
        10800,
        "implementation_timeout_seconds",
        errors,
    )
    _check_equal(
        config.get("implementation_max_timeout_seconds"),
        14400,
        "implementation_max_timeout_seconds",
        errors,
    )
    _check_equal(config.get("strict_task_sharding"), True, "strict_task_sharding", errors)
    _check_equal(
        config.get("exit_when_all_tracks_terminal"),
        True,
        "exit_when_all_tracks_terminal",
        errors,
    )
    _check_equal(config.get("objective_refill_enabled"), False, "objective_refill_enabled", errors)
    _check_equal(config.get("codebase_refill_enabled"), False, "codebase_refill_enabled", errors)
    _check_equal(
        set(config.get("worktree_submodule_paths", ()))
        if isinstance(config.get("worktree_submodule_paths"), list)
        else config.get("worktree_submodule_paths"),
        {"ipfs_datasets_py", "ipfs_kit_py"},
        "worktree_submodule_paths",
        errors,
    )
    _check_equal(
        config.get("baseline_capture_script"),
        BASELINE_CAPTURE_SCRIPT,
        "baseline_capture_script",
        errors,
    )
    _check_equal(
        config.get("baseline_suite_registry"),
        BASELINE_SUITE_REGISTRY,
        "baseline_suite_registry",
        errors,
    )
    _check_equal(
        config.get("baseline_suite_registry_digest"),
        BASELINE_SUITE_REGISTRY_DIGEST,
        "baseline_suite_registry_digest",
        errors,
    )
    receipt_pins = config.get("operator_baseline_receipts")
    pinned_artifact_paths: set[str] = set()
    if not isinstance(receipt_pins, Mapping):
        errors.append("operator_baseline_receipts must be a protected task-id map")
    elif bootstrap:
        if receipt_pins != {}:
            errors.append("bootstrap operator_baseline_receipts must be exactly empty")
    else:
        _check_equal(
            set(receipt_pins),
            set(BASELINE_RECEIPT_SPECS),
            "operator_baseline_receipts task ids",
            errors,
        )
        for task_id, spec in BASELINE_RECEIPT_SPECS.items():
            pin = receipt_pins.get(task_id)
            if not isinstance(pin, Mapping):
                errors.append(f"operator_baseline_receipts.{task_id} must be an object")
                continue
            _closed_keys(
                pin,
                (
                    "path",
                    "receipt_digest",
                    "planning_revision",
                    "source_revision",
                    "source_tree",
                    "required_command_ids",
                    "suite_definition_digests",
                    "retained_log_paths",
                ),
                f"operator_baseline_receipts.{task_id}",
                errors,
            )
            _check_equal(
                pin.get("path"),
                spec["receipt"],
                f"operator_baseline_receipts.{task_id}.path",
                errors,
            )
            pinned_artifact_paths.add(str(spec["receipt"]))
            retained_log_paths = pin.get("retained_log_paths")
            if (
                not isinstance(retained_log_paths, list)
                or len(retained_log_paths) != len(spec["command_ids"])
                or any(not isinstance(path, str) for path in retained_log_paths)
                or len(set(retained_log_paths)) != len(retained_log_paths)
            ):
                errors.append(
                    f"operator_baseline_receipts.{task_id}.retained_log_paths must "
                    "be a unique ordered path for each command"
                )
            else:
                for command_id, path in zip(
                    spec["command_ids"], retained_log_paths, strict=True
                ):
                    if Path(path).parent.as_posix() != BASELINE_LOG_ROOT:
                        errors.append(
                            f"operator_baseline_receipts.{task_id}.retained_log_paths "
                            "contains a path outside the fixed log root"
                        )
                    if not Path(path).name.startswith(f"{command_id}-"):
                        errors.append(
                            f"operator_baseline_receipts.{task_id}.retained_log_paths "
                            f"does not bind {command_id}"
                        )
                    pinned_artifact_paths.add(path)
            _check_equal(
                pin.get("required_command_ids"),
                list(spec["command_ids"]),
                f"operator_baseline_receipts.{task_id}.required_command_ids",
                errors,
            )
            _check_equal(
                pin.get("planning_revision"),
                spec["revision"],
                f"operator_baseline_receipts.{task_id}.planning_revision",
                errors,
            )
            for field in ("source_revision", "source_tree"):
                value = pin.get(field)
                if not isinstance(value, str) or not _HEX_40.fullmatch(value):
                    errors.append(
                        f"operator_baseline_receipts.{task_id}.{field} must be a Git id"
                    )
            suite_digests = pin.get("suite_definition_digests")
            if not isinstance(suite_digests, Mapping) or set(suite_digests) != set(
                spec["command_ids"]
            ):
                errors.append(
                    f"operator_baseline_receipts.{task_id}.suite_definition_digests "
                    "must exactly cover required commands"
                )
            else:
                expected_suite_digests = {
                    command_id: reviewed_registry.get(command_id, {}).get(
                        "suite_definition_digest"
                    )
                    for command_id in spec["command_ids"]
                }
                if suite_digests != expected_suite_digests:
                    errors.append(
                        f"operator_baseline_receipts.{task_id}.suite_definition_digests "
                        "does not match the protected reviewed registry"
                    )
                for command_id in spec["command_ids"]:
                    _sha256_value(
                        suite_digests.get(command_id),
                        (
                            f"operator_baseline_receipts.{task_id}."
                            f"suite_definition_digests.{command_id}"
                        ),
                        errors,
                    )
            _sha256_value(
                pin.get("receipt_digest"),
                f"operator_baseline_receipts.{task_id}.receipt_digest",
                errors,
            )
    protected_paths = config.get("protected_paths")
    expected_protected_paths = set(BASE_PROTECTED_PATHS) | pinned_artifact_paths
    _check_equal(
        set(protected_paths) if isinstance(protected_paths, list) else protected_paths,
        expected_protected_paths,
        "protected_paths",
        errors,
    )
    if isinstance(protected_paths, list) and BASELINE_RECEIPT_ROOT in protected_paths:
        errors.append("protected_paths may not contain the launch-invalid receipt directory")
    if isinstance(protected_paths, list) and len(protected_paths) != len(
        set(protected_paths)
    ):
        errors.append("protected_paths must be a unique exact-file list")
    validation_workers = config.get("validation_max_workers")
    if not isinstance(validation_workers, int) or validation_workers <= 0:
        errors.append("validation_max_workers must be a positive integer")
    provider = config.get("provider")
    if not isinstance(provider, dict):
        errors.append("provider must be an object")
    else:
        expected_route = {
            "primary_provider_id": "grok_cli",
            "primary_model_id": "grok-4.5",
            "fallback_provider_id": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_trigger": "primary_quota_exhausted",
            "fallback_reasoning_effort": "medium",
            "max_concurrency": 3,
            "secrets_from_environment_only": True,
            "secrets_in_argv_prompts_logs_or_receipts": False,
        }
        for key, expected in expected_route.items():
            _check_equal(provider.get(key), expected, f"provider.{key}", errors)
    _check_equal(config.get("taskboard_path"), str(TASKBOARD_PATH.relative_to(REPO_ROOT)), "taskboard_path", errors)
    _check_equal(config.get("objectives_path"), str(OBJECTIVES_PATH.relative_to(REPO_ROOT)), "objectives_path", errors)
    _check_equal(config.get("plan_path"), str(PLAN_PATH.relative_to(REPO_ROOT)), "plan_path", errors)
    _check_equal(
        config.get("validator_path"),
        "scripts/validate_incremental_proof_sealer_board.py",
        "validator_path",
        errors,
    )

    projection = config.get("initial_projection")
    if not isinstance(projection, dict):
        errors.append("initial_projection must be an object")
    else:
        _check_equal(projection.get("task_count"), len(TASK_IDS), "initial task_count", errors)
        _check_equal(
            projection.get("completed_task_ids"),
            sorted(INITIAL_COMPLETED),
            "initial completed_task_ids",
            errors,
        )
        _check_equal(
            projection.get("ready_task_ids"),
            sorted(INITIAL_READY),
            "initial ready_task_ids",
            errors,
        )
        _check_equal(projection.get("blocked_task_ids"), [], "initial blocked_task_ids", errors)
        _check_equal(projection.get("terminal_task_id"), TERMINAL_TASK, "initial terminal_task_id", errors)
        _check_equal(projection.get("goal_count"), len(GOAL_IDS), "initial goal_count", errors)
        _check_equal(projection.get("root_goal_id"), "IPS-G000", "root_goal_id", errors)

    source = config.get("source_binding")
    if not isinstance(source, dict):
        errors.append("source_binding must be an object")
    else:
        expected_source = {
            "accelerator_required_ancestor": ACCELERATE_REVISION,
            "accelerator_required_branch": BRANCH,
            "accelerator_planning_revision": ACCELERATE_REVISION,
            "ipfs_datasets_submodule_path": "ipfs_datasets_py",
            "ipfs_datasets_planning_revision": DATASETS_REVISION,
            "ipfs_kit_submodule_path": "ipfs_kit_py",
            "ipfs_kit_planning_revision": KIT_REVISION,
            "require_initialized_gitlinks": True,
            "require_superproject_gitlink_equals_nested_head": True,
            "require_clean_nested_worktree_at_task_start": True,
            "changed_revision_requires_fresh_inventory_and_baseline": True,
            "planning_revision_is_runtime_completion_evidence": False,
        }
        for key, expected in expected_source.items():
            _check_equal(source.get(key), expected, f"source_binding.{key}", errors)

    actual_groups = config.get("task_groups")
    if not isinstance(actual_groups, dict):
        errors.append("task_groups must be an object")
    else:
        normalized_groups = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in actual_groups.items()
        }
        _check_equal(normalized_groups, dict(EXPECTED_TASK_GROUPS), "task_groups", errors)

    lanes = config.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 3:
        errors.append("lanes must contain exactly three entries")
    else:
        seen_indices: set[int] = set()
        lane_tasks: set[str] = set()
        for lane in lanes:
            if not isinstance(lane, dict):
                errors.append("each lane must be an object")
                continue
            index = lane.get("index")
            remainder = lane.get("strict_shard_remainder")
            if not isinstance(index, int) or index not in range(3):
                errors.append(f"invalid lane index: {index!r}")
                continue
            if index in seen_indices:
                errors.append(f"duplicate lane index: {index}")
            seen_indices.add(index)
            _check_equal(remainder, index, f"lane {index} shard remainder", errors)
            initial = lane.get("initial_task_ids")
            if not isinstance(initial, list) or len(initial) != 1:
                errors.append(f"lane {index} must have exactly one initial task")
                continue
            task_id = initial[0]
            if task_id in lane_tasks:
                errors.append(f"initial lane task repeated: {task_id}")
            lane_tasks.add(task_id)
            shard = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) % 3
            if shard != index:
                errors.append(
                    f"{task_id} hashes to strict shard {shard}, not configured lane {index}"
                )
        _check_equal(lane_tasks, set(INITIAL_READY), "lane initial task set", errors)



def _validate_operational_residual_tasks(
    records: Mapping[str, Any],
    residual_ids: Sequence[str],
    errors: list[str],
) -> None:
    """Admit bounded operational residual tasks outside the sealed IPS DAG.

    Supports retry-budget repair residuals and reconciliation-guardrail residuals
    that the supervisor may append after inventory/merge work.
    """

    if len(residual_ids) > MAX_OPERATIONAL_RESIDUAL_TASKS:
        errors.append(
            f"operational residual appendix exceeds bound {MAX_OPERATIONAL_RESIDUAL_TASKS}"
        )
    sealed_count = len(SEALED_TASK_IDS)
    previous_source_kind: dict[tuple[str, str], str] = {}
    previous_reconciliation_key: dict[str, str] = {}
    for offset, task_id in enumerate(residual_ids):
        expected_id = f"IPS-{sealed_count + offset:03d}"
        if task_id != expected_id:
            errors.append(
                f"operational residual ids must be contiguous: expected {expected_id}, "
                f"got {task_id}"
            )
        record = records.get(task_id)
        if record is None:
            continue
        fields = record["fields"]
        generated_by = fields.get("generated by", "")
        if generated_by not in OPERATIONAL_RESIDUAL_GENERATORS:
            errors.append(
                f"{task_id} residual must declare generated by one of "
                f"{sorted(OPERATIONAL_RESIDUAL_GENERATORS)!r}"
            )
            continue
        if generated_by == RETRY_BUDGET_REPAIR_GENERATOR:
            source = fields.get("retry repair source", "").strip()
            kind = fields.get("retry failure kind", "").strip()
            if source not in SEALED_TASK_IDS or kind not in {
                "validation",
                "implementation",
                "merge",
            }:
                errors.append(
                    f"{task_id} residual has unrecognized retry-repair provenance "
                    f"source={source!r} kind={kind!r}"
                )
                continue
            key = (source, kind)
            previous = previous_source_kind.get(key)
            if previous is not None:
                prev_status = (
                    records.get(previous, {})
                    .get("fields", {})
                    .get("status", "")
                    .casefold()
                )
                if prev_status != "completed":
                    errors.append(
                        f"concurrent duplicate operational residual for {key}: "
                        f"{previous} and {task_id}"
                    )
            previous_source_kind[key] = task_id
        else:
            recon_kind = fields.get("reconciliation kind", "").strip()
            recon_reason = fields.get("reconciliation reason", "").strip()
            fingerprint = fields.get("reconciliation fingerprint", "").strip()
            if not recon_kind or not recon_reason:
                errors.append(
                    f"{task_id} residual lacks reconciliation kind/reason provenance"
                )
                continue
            if (
                recon_kind not in RECONCILIATION_RESIDUAL_KINDS
                and recon_reason not in RECONCILIATION_RESIDUAL_KINDS
            ):
                # Allow forward-compatible kinds while still requiring structured fields.
                pass
            if not fingerprint:
                errors.append(
                    f"{task_id} residual lacks reconciliation fingerprint provenance"
                )
                continue
            previous = previous_reconciliation_key.get(fingerprint)
            if previous is not None:
                prev_status = (
                    records.get(previous, {})
                    .get("fields", {})
                    .get("status", "")
                    .casefold()
                )
                if prev_status not in {"completed", "cancelled"}:
                    errors.append(
                        f"concurrent duplicate reconciliation residual for "
                        f"{fingerprint}: {previous} and {task_id}"
                    )
            previous_reconciliation_key[fingerprint] = task_id
        deps = set(_ids(fields.get("depends on", ""), r"IPS-\d{3}"))
        bad = sorted(dep for dep in deps if dep not in SEALED_TASK_IDS)
        if bad:
            errors.append(f"{task_id} residual depends on non-sealed tasks: {bad}")


def _validate_tasks(text: str, config: dict[str, Any], errors: list[str]) -> dict[str, set[str]]:
    raw_headings = re.findall(r"^## (IPS-[^\s]+)(?:\s+.*)?$", text, re.MULTILINE)
    records = _parse_markdown_records(
        text,
        re.compile(r"^## (IPS-\d{3})\s+([^\n]+)$", re.MULTILINE),
        "task",
        errors,
    )
    actual_ids = set(records)
    expected_ids = set(SEALED_TASK_IDS)
    residual_ids = sorted(actual_ids - expected_ids)
    if len(raw_headings) != len(records):
        errors.append(
            "one or more IPS task headings is duplicated, malformed, or lacks a title"
        )
    for task_id in sorted(expected_ids - actual_ids):
        errors.append(f"missing task heading: {task_id}")
    _validate_operational_residual_tasks(records, residual_ids, errors)
    known_ids = expected_ids | set(residual_ids)

    dependencies: dict[str, set[str]] = {}
    concurrency: dict[str, set[str]] = {}
    writer_submodules: dict[str, set[str]] = {}
    # Sealed DAG is validated fully; residuals only contribute dependency edges.
    for task_id in (*SEALED_TASK_IDS, *residual_ids):
        record = records.get(task_id)
        if record is None:
            dependencies[task_id] = set()
            continue
        fields = record["fields"]
        is_residual = task_id in residual_ids
        if not is_residual:
            for field in REQUIRED_TASK_FIELDS:
                if field not in fields:
                    errors.append(f"{task_id} is missing metadata field {field!r}")
            for field in ("outputs", "validation", "conflict policy", "acceptance"):
                if not fields.get(field, "").strip():
                    errors.append(f"{task_id} metadata field {field!r} may not be empty")
        else:
            for field in ("outputs", "validation", "acceptance"):
                if not fields.get(field, "").strip():
                    errors.append(f"{task_id} residual metadata field {field!r} may not be empty")

        if task_id in BASELINE_RECEIPT_SPECS:
            spec = BASELINE_RECEIPT_SPECS[task_id]
            inputs = fields.get("inputs", "")
            effects = fields.get("effects", "")
            acceptance = fields.get("acceptance", "")
            conflict = fields.get("conflict policy", "")
            for term in (
                str(spec["receipt"]),
                "protected scheduler digest pin",
            ):
                if term not in inputs:
                    errors.append(f"{task_id} Inputs omits protected term {term!r}")
            for command_id in spec["command_ids"]:
                if command_id not in acceptance:
                    errors.append(
                        f"{task_id} Acceptance omits required command id {command_id!r}"
                    )
            for term in (
                "reference-only",
                "operator_capture",
                "process_observed_only",
                "pytest_execution_not_cryptographically_proven",
                "planning_revision",
                "inventory_worktree_parent_revision",
                "static",
            ):
                if term.casefold() not in acceptance.casefold():
                    errors.append(f"{task_id} Acceptance omits provenance term {term!r}")
            if "separately captured operator pytest receipt" not in effects:
                errors.append(f"{task_id} Effects must distinguish operator capture")
            for term in ("no shell authority", "must not run pytest", "two declared"):
                if term not in conflict:
                    errors.append(f"{task_id} Conflict policy omits {term!r}")
        if task_id in {"IPS-045", "IPS-046", "IPS-048"}:
            mutation_contract = " ".join(
                (fields.get("effects", ""), fields.get("acceptance", ""))
            ).casefold()
            for term in (
                "selector",
                "fixture",
                "configuration",
                "network policy",
                "policy",
                "lock",
                "tool",
                "schema",
                "canonicalization",
                "checked-spec",
            ):
                if term not in mutation_contract:
                    errors.append(f"{task_id} mutation contract omits {term!r}")

        validation = fields.get("validation", "")
        validation_argv = _validation_argv(task_id, validation, errors)
        if task_id == "IPS-004" and validation_argv != [
            "python",
            "scripts/validate_incremental_proof_sealer_board.py",
            "--check-artifact",
            "IPS-004",
        ]:
            errors.append("IPS-004 must use its candidate synthesis artifact gate")
        if task_id == "IPS-000" and validation_argv != [
            "python",
            "scripts/validate_incremental_proof_sealer_board.py",
            "--check-bootstrap",
        ]:
            errors.append("IPS-000 must use the pristine empty-pin bootstrap gate")
        if task_id == "IPS-053" and validation_argv != list(
            BENCHMARK_VALIDATION_ARGV
        ):
            errors.append("IPS-053 must use the protected convergent benchmark ensure gate")
        if task_id == "IPS-053":
            benchmark_contract = " ".join(
                (fields.get("effects", ""), fields.get("acceptance", ""))
            )
            for term in (
                MATERIALIZATION_REQUEST_SCHEMA,
                "stabilization",
                "read-only",
                "partial",
            ):
                if term not in benchmark_contract:
                    errors.append(f"IPS-053 convergence contract omits {term!r}")
        if task_id == TERMINAL_TASK and validation_argv != list(
            RELEASE_VALIDATION_ARGV
        ):
            errors.append("IPS-056 must use the protected convergent release ensure gate")
        if task_id == TERMINAL_TASK:
            release_contract = " ".join(
                (fields.get("effects", ""), fields.get("acceptance", ""))
            ).casefold()
            for term in (
                "baseline-compatible-or-improved",
                "baseline_compatible_non_green",
                "three new",
                "secret",
                "live `ipfs`",
                "process-tree termination",
                MATERIALIZATION_REQUEST_SCHEMA,
                RELEASE_REPORT_REQUEST_MARKER.casefold(),
                "stabilization",
                "read-only",
            ):
                if term not in release_contract:
                    errors.append(f"IPS-056 release contract omits {term!r}")

        predicted_files = fields.get("predicted files", "")
        # Sealed tasks always bind Outputs == Predicted files. Residuals may omit
        # Predicted files (reconciliation guardrails) or keep the retry-budget shape.
        if task_id != "IPS-000" and not is_residual and (
            fields.get("outputs", "") != predicted_files
        ):
            errors.append(f"{task_id} Outputs must exactly equal Predicted files")
        if is_residual and predicted_files and fields.get("outputs", "") != predicted_files:
            errors.append(f"{task_id} Outputs must exactly equal Predicted files")
        expected_envelope = {
            "IPS-053": BENCHMARK_PROPOSAL_ENVELOPE,
            "IPS-056": RELEASE_PROPOSAL_ENVELOPE,
        }.get(task_id)
        raw_envelope = fields.get("proposal artifact envelope", "")
        if expected_envelope is not None:
            try:
                parsed_envelope = json.loads(
                    raw_envelope,
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_nonfinite,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"{task_id} proposal artifact envelope is invalid: {exc}")
                parsed_envelope = None
            if parsed_envelope != expected_envelope:
                errors.append(f"{task_id} proposal artifact envelope is not exact")
            if raw_envelope != _canonical_json_bytes(expected_envelope).decode("utf-8"):
                errors.append(f"{task_id} proposal artifact envelope is not canonical")
        elif raw_envelope:
            errors.append(f"{task_id} has an unreviewed proposal artifact envelope")
        predicted_submodules: list[str] = []
        if re.search(
            r"(?:^|[,\s])ipfs_datasets_py(?:/|[,\s]|$)", predicted_files
        ):
            predicted_submodules.append("ipfs_datasets_py")
        if re.search(r"(?:^|[,\s])ipfs_kit_py(?:/|[,\s]|$)", predicted_files):
            predicted_submodules.append("ipfs_kit_py")
        expected_submodules = ", ".join(predicted_submodules) or "none"
        if not is_residual:
            _check_equal(
                fields.get("submodules"),
                expected_submodules,
                f"{task_id} submodules derived from Predicted files",
                errors,
            )

        status = fields.get("status", "").casefold()
        if status not in {"todo", "in_progress", "blocked", "completed"}:
            errors.append(f"{task_id} has invalid status {status!r}")
        schedulable = _as_bool(fields.get("is schedulable", ""))
        timeout = _as_int(fields.get("implementation timeout seconds", ""))
        if task_id == "IPS-000":
            if status != "completed":
                errors.append("IPS-000 must remain completed")
            if fields.get("completion", "").casefold() != "manual":
                errors.append("IPS-000 completion must be manual")
            # The daemon skips this already-completed bootstrap card, but its
            # parser still requires every card to carry a schedulable shape
            # and a positive timeout.
            if schedulable is not True:
                errors.append("IPS-000 must retain a schedulable card shape")
            if timeout is None or timeout <= 0:
                errors.append("IPS-000 must have a positive parser-safe timeout")
        elif not is_residual:
            if fields.get("completion", "").casefold() != "auto":
                errors.append(f"{task_id} completion must be auto")
            if schedulable is not True:
                errors.append(f"{task_id} must be schedulable")
            if timeout is None or timeout <= 0:
                errors.append(f"{task_id} must have a positive implementation timeout")

        if not is_residual:
            _check_equal(
                fields.get("board namespace"),
                BOARD_NAMESPACE,
                f"{task_id} board namespace",
                errors,
            )
            _check_equal(
                fields.get("goal id"),
                EXPECTED_TASK_TO_GOAL.get(task_id),
                f"{task_id} goal id",
                errors,
            )

        dependency_ids = set(_ids(fields.get("depends on", ""), r"IPS-\d{3}"))
        dependency_text = fields.get("depends on", "").strip()
        if dependency_text and not re.fullmatch(
            r"IPS-\d{3}(?:\s*,\s*IPS-\d{3})*", dependency_text
        ):
            errors.append(f"{task_id} has malformed dependency metadata")
        if task_id in dependency_ids:
            errors.append(f"{task_id} depends on itself")
        unknown_dependencies = dependency_ids - known_ids
        if unknown_dependencies:
            errors.append(
                f"{task_id} has unknown dependencies: {sorted(unknown_dependencies)}"
            )
        dependencies[task_id] = dependency_ids & known_ids

        concurrent = set(
            _ids(fields.get("allow concurrent with", ""), r"IPS-\d{3}")
        )
        concurrency[task_id] = concurrent & known_ids
        unknown_concurrent = concurrent - known_ids
        if unknown_concurrent:
            errors.append(
                f"{task_id} has unknown concurrency peers: {sorted(unknown_concurrent)}"
            )
        if task_id in concurrent:
            errors.append(f"{task_id} lists itself as a concurrency peer")
        writer_submodules[task_id] = set(predicted_submodules)

    cycles = _cycle_nodes(dependencies)
    if cycles:
        errors.append(f"task dependency graph is cyclic: {sorted(cycles)}")

    ancestors = {
        task_id: _ancestors(task_id, dependencies) for task_id in SEALED_TASK_IDS
    }
    for task_id in TASK_IDS:
        for peer in sorted(concurrency.get(task_id, set())):
            if task_id not in concurrency.get(peer, set()):
                errors.append(
                    f"{task_id} concurrency peer {peer} is not declared symmetrically"
                )
            if peer in ancestors[task_id] or task_id in ancestors[peer]:
                errors.append(
                    f"{task_id} concurrency peer {peer} is an ancestor or descendant"
                )
    for index, task_id in enumerate(SEALED_TASK_IDS):
        for peer in SEALED_TASK_IDS[index + 1 :]:
            shared = writer_submodules.get(task_id, set()) & writer_submodules.get(
                peer, set()
            )
            if shared and peer not in ancestors[task_id] and task_id not in ancestors[peer]:
                errors.append(
                    f"same-submodule writers {task_id} and {peer} are unordered for "
                    f"{sorted(shared)}"
                )

    initially_ready = {
        task_id
        for task_id, prerequisites in dependencies.items()
        if task_id in SEALED_TASK_IDS
        and task_id not in INITIAL_COMPLETED
        and prerequisites.issubset(INITIAL_COMPLETED)
    }
    _check_equal(initially_ready, set(INITIAL_READY), "DAG initial ready set", errors)

    reachable = _reachable_from(INITIAL_COMPLETED, dependencies)
    missing_sealed = set(SEALED_TASK_IDS) - reachable
    if missing_sealed:
        errors.append(
            "tasks unreachable from IPS-000: "
            f"{sorted(missing_sealed)}"
        )
    terminal_ancestors = _ancestors(TERMINAL_TASK, dependencies)
    expected_ancestors = set(SEALED_TASK_IDS) - {TERMINAL_TASK}
    if terminal_ancestors != expected_ancestors:
        errors.append(
            "IPS-056 is not a true terminal fan-in; missing ancestors: "
            f"{sorted(expected_ancestors - terminal_ancestors)}"
        )

    projection = config.get("initial_projection", {})
    if isinstance(projection, dict):
        _check_equal(
            projection.get("terminal_task_id"), TERMINAL_TASK, "terminal task", errors
        )
    return dependencies


def _validate_goals(text: str, errors: list[str]) -> None:
    raw_headings = re.findall(r"^## (IPS-G[^\s]+)(?:\s+.*)?$", text, re.MULTILINE)
    records = _parse_markdown_records(
        text,
        re.compile(r"^## (IPS-G\d{3})\s+([^\n]+)$", re.MULTILINE),
        "goal",
        errors,
    )
    actual_ids = set(records)
    expected_ids = set(GOAL_IDS)
    if len(raw_headings) != len(records):
        errors.append(
            "one or more IPS goal headings is duplicated, malformed, or lacks a title"
        )
    for goal_id in sorted(expected_ids - actual_ids):
        errors.append(f"missing goal heading: {goal_id}")
    for goal_id in sorted(actual_ids - expected_ids):
        errors.append(f"unexpected goal heading: {goal_id}")

    dependencies: dict[str, set[str]] = {}
    for goal_id in GOAL_IDS:
        record = records.get(goal_id)
        if record is None:
            dependencies[goal_id] = set()
            continue
        fields = record["fields"]
        for field in REQUIRED_GOAL_FIELDS:
            if field not in fields:
                errors.append(f"{goal_id} is missing metadata field {field!r}")
        for field in ("goal", "evidence", "outputs", "validation", "acceptance", "conflict policy"):
            if not fields.get(field, "").strip():
                errors.append(f"{goal_id} metadata field {field!r} may not be empty")
        validation_argv = _validation_argv(
            goal_id, fields.get("validation", ""), errors
        )
        if goal_id == "IPS-G130" and validation_argv != list(
            RELEASE_VALIDATION_ARGV
        ):
            errors.append("IPS-G130 Validation must use the read-only release artifact gate")
        if fields.get("status", "").casefold() not in {
            "active",
            "provisionally_complete",
            "verified_complete",
            "analysis_inconclusive",
            "blocked",
            "reopened",
        }:
            errors.append(f"{goal_id} has an invalid status")
        parent = fields.get("parent", "").strip()
        if goal_id == "IPS-G000":
            if parent:
                errors.append("IPS-G000 must not have a parent")
        elif parent != "IPS-G000":
            errors.append(f"{goal_id} parent must be IPS-G000")
        dependency_ids = set(_ids(fields.get("depends on", ""), r"IPS-G\d{3}"))
        dependency_text = fields.get("depends on", "").strip()
        if dependency_text and not re.fullmatch(
            r"IPS-G\d{3}(?:\s*,\s*IPS-G\d{3})*", dependency_text
        ):
            errors.append(f"{goal_id} has malformed dependency metadata")
        if goal_id in dependency_ids:
            errors.append(f"{goal_id} depends on itself")
        unknown = dependency_ids - expected_ids
        if unknown:
            errors.append(f"{goal_id} has unknown dependencies: {sorted(unknown)}")
        dependencies[goal_id] = dependency_ids & expected_ids

        gap_tasks = set(_ids(fields.get("gap task", ""), r"IPS-\d{3}"))
        if goal_id == "IPS-G000":
            expected_gap_tasks = {TERMINAL_TASK}
        else:
            expected_gap_tasks = set(EXPECTED_TASK_GROUPS.get(goal_id, ()))
            expected_gap_tasks.discard("IPS-000")
        if gap_tasks != expected_gap_tasks:
            errors.append(
                f"{goal_id} gap task set must be {sorted(expected_gap_tasks)}; "
                f"got {sorted(gap_tasks)}"
            )

    cycles = _cycle_nodes(dependencies)
    if cycles:
        errors.append(f"goal dependency graph is cyclic: {sorted(cycles)}")


def _declared_output_paths(value: str) -> set[str]:
    return {
        item.strip().rstrip("/")
        for item in value.split(",")
        if item.strip() and item.strip().casefold() != "none"
    }


def _validation_local_paths(
    record_id: str, validation: str, errors: list[str]
) -> set[str]:
    paths: set[str] = set()
    for token in _validation_argv(record_id, validation, errors):
        candidate = token.split("::", 1)[0]
        if candidate.startswith("-") or candidate in {"python", "pytest"}:
            continue
        if "/" not in candidate and not candidate.endswith(
            (".py", ".json", ".md", ".csv", ".log")
        ):
            continue
        path = Path(candidate)
        # Residual guardrails may emit absolute discovery paths under the repo.
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            except ValueError:
                errors.append(f"{record_id} validation path is not canonical: {token!r}")
                continue
            if any(part in {"", ".", ".."} for part in Path(relative).parts):
                errors.append(f"{record_id} validation path is not canonical: {token!r}")
                continue
            paths.add(relative.rstrip("/"))
            continue
        if "\\" in candidate or any(part in {"", ".", ".."} for part in path.parts):
            errors.append(f"{record_id} validation path is not canonical: {token!r}")
            continue
        paths.add(path.as_posix().rstrip("/"))
    return paths


def _validation_path_existed_at_bootstrap(relative: str) -> bool:
    if relative in BASE_PROTECTED_PATHS:
        return True
    revision = ACCELERATE_REVISION
    repository_relative = relative
    root = REPO_ROOT
    for prefix, candidate_revision in (
        ("ipfs_datasets_py/", DATASETS_REVISION),
        ("ipfs_kit_py/", KIT_REVISION),
    ):
        if relative.startswith(prefix):
            revision = candidate_revision
            repository_relative = relative[len(prefix) :]
            root = REPO_ROOT / prefix.rstrip("/")
            break
    if not repository_relative:
        return False
    result = _git(
        "cat-file",
        "-e",
        f"{revision}:{repository_relative}",
        cwd=root,
    )
    return result.returncode == 0


def _path_is_available(relative: str, produced: set[str]) -> bool:
    if _validation_path_existed_at_bootstrap(relative):
        return True
    return any(
        relative == output
        or relative.startswith(output + "/")
        or output.startswith(relative + "/")
        for output in produced
    )


def _validate_validation_path_closure(
    task_text: str,
    goal_text: str,
    task_dependencies: Mapping[str, set[str]],
    errors: list[str],
) -> None:
    """Reject validation argv that name paths no predecessor is required to create."""

    task_records = _parse_markdown_records(
        task_text,
        re.compile(r"^## (IPS-\d{3})\s+([^\n]+)$", re.MULTILINE),
        "task validation closure",
        errors,
    )
    goal_records = _parse_markdown_records(
        goal_text,
        re.compile(r"^## (IPS-G\d{3})\s+([^\n]+)$", re.MULTILINE),
        "goal validation closure",
        errors,
    )
    task_outputs = {
        task_id: _declared_output_paths(
            # Residuals may omit Predicted files and only declare Outputs.
            record["fields"].get("predicted files", "")
            or record["fields"].get("outputs", "")
        )
        for task_id, record in task_records.items()
    }
    task_ancestors = {
        task_id: _ancestors(task_id, dict(task_dependencies)) | {task_id}
        for task_id in task_records
    }
    for task_id, record in task_records.items():
        ancestors = task_ancestors.get(task_id, {task_id})
        produced = set().union(
            *(task_outputs.get(ancestor, set()) for ancestor in ancestors)
        )
        for relative in sorted(
            _validation_local_paths(
                task_id, record["fields"].get("validation", ""), errors
            )
        ):
            if not _path_is_available(relative, produced):
                errors.append(
                    f"{task_id} validation path is neither bootstrap-present nor "
                    f"dependency-produced: {relative}"
                )

    goal_dependencies = {
        goal_id: set(
            _ids(
                goal_records.get(goal_id, {}).get("fields", {}).get("depends on", ""),
                r"IPS-G\d{3}",
            )
        )
        for goal_id in GOAL_IDS
    }
    for goal_id, record in goal_records.items():
        if goal_id == "IPS-G000":
            closure_tasks = set(TASK_IDS)
        else:
            closure_goals = _ancestors(goal_id, goal_dependencies) | {goal_id}
            closure_tasks = set()
            for closure_goal in closure_goals:
                goal_record = goal_records.get(closure_goal, {})
                closure_tasks.update(
                    _ids(
                        goal_record.get("fields", {}).get("gap task", ""),
                        r"IPS-\d{3}",
                    )
                )
            closure_tasks = set().union(
                *(task_ancestors.get(task_id, {task_id}) for task_id in closure_tasks)
            )
        produced = set().union(
            *(task_outputs.get(task_id, set()) for task_id in closure_tasks)
        )
        for relative in sorted(
            _validation_local_paths(
                goal_id, record["fields"].get("validation", ""), errors
            )
        ):
            if not _path_is_available(relative, produced):
                errors.append(
                    f"{goal_id} validation path is neither bootstrap-present nor "
                    f"dependency-produced: {relative}"
                )


def _require_terms(
    text: str, terms: Iterable[str], category: str, errors: list[str]
) -> None:
    folded = text.casefold()
    whitespace_normalized = re.sub(r"\s+", " ", folded)
    missing = [
        term
        for term in terms
        if term.casefold() not in folded
        and re.sub(r"\s+", " ", term.casefold()) not in whitespace_normalized
    ]
    if missing:
        errors.append(f"plan is missing {category} terms: {missing}")


def _validate_plan(text: str, config: dict[str, Any], errors: list[str]) -> None:
    _require_terms(text, REQUIRED_PLAN_CONCEPTS, "architecture", errors)
    _require_terms(text, REQUIRED_CLI_TERMS, "CLI", errors)
    _require_terms(text, REQUIRED_INVALIDATION_TERMS, "invalidation", errors)
    _require_terms(text, REQUIRED_NEGATIVE_TERMS, "negative-test", errors)
    _require_terms(text, REQUIRED_CRASH_TERMS, "crash-recovery", errors)

    transition_rows = {
        int(match.group(1))
        for match in re.finditer(r"^\|\s*(\d{2})\s*\|", text, flags=re.MULTILINE)
    }
    expected_rows = set(range(40))
    if transition_rows != expected_rows:
        errors.append(
            "benchmark must define exactly transition rows 00..39; "
            f"missing={sorted(expected_rows - transition_rows)}, "
            f"extra={sorted(transition_rows - expected_rows)}"
        )
    benchmark = config.get("benchmark_policy")
    if not isinstance(benchmark, dict):
        errors.append("benchmark_policy must be an object")
    else:
        _check_equal(
            benchmark.get("sequential_commit_count"),
            40,
            "benchmark sequential_commit_count",
            errors,
        )
        _check_equal(
            benchmark.get("full_and_incremental_compared_per_commit"),
            True,
            "benchmark full/incremental comparison",
            errors,
        )
        _check_equal(
            benchmark.get("targets_are_not_reported_as_results"),
            True,
            "benchmark target honesty",
            errors,
        )


def _check_git_result(
    result: subprocess.CompletedProcess[str], description: str, errors: list[str]
) -> str:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        errors.append(f"{description} failed: {detail}")
        return ""
    return result.stdout.strip()


def _validate_git_state(config: dict[str, Any], errors: list[str]) -> None:
    _validate_current_trust_sensitive_ignored_inputs(errors)
    branch = _check_git_result(
        _git("branch", "--show-current"), "resolve control branch", errors
    )
    if branch and branch != BRANCH:
        errors.append(f"control branch must be {BRANCH!r}; got {branch!r}")
    ancestor = _git("merge-base", "--is-ancestor", ACCELERATE_REVISION, "HEAD")
    if ancestor.returncode != 0:
        errors.append(
            f"accelerate planning revision {ACCELERATE_REVISION} is not an ancestor of HEAD"
        )

    source = config.get("source_binding", {})
    if not isinstance(source, dict):
        return
    nested_specs = (
        ("ipfs_datasets_py", DATASETS_REVISION),
        ("ipfs_kit_py", KIT_REVISION),
    )
    for relative, planning_revision in nested_specs:
        nested = REPO_ROOT / relative
        if not nested.is_dir():
            errors.append(f"required initialized submodule is missing: {relative}")
            continue
        nested_head = _check_git_result(
            _git("rev-parse", "HEAD", cwd=nested),
            f"resolve {relative} HEAD",
            errors,
        )
        gitlink = _check_git_result(
            _git("rev-parse", f"HEAD:{relative}"),
            f"resolve superproject gitlink {relative}",
            errors,
        )
        if nested_head and gitlink and nested_head != gitlink:
            errors.append(
                f"{relative} nested HEAD {nested_head} does not equal gitlink {gitlink}"
            )
        nested_ancestor = _git(
            "merge-base", "--is-ancestor", planning_revision, "HEAD", cwd=nested
        )
        if nested_ancestor.returncode != 0:
            errors.append(
                f"{relative} planning revision {planning_revision} is not an ancestor of HEAD"
            )
        dirty = _check_git_result(
            _git("status", "--porcelain=v1", "--untracked-files=normal", cwd=nested),
            f"inspect {relative} worktree",
            errors,
        )
        if dirty:
            errors.append(f"{relative} nested worktree is dirty: {dirty.splitlines()[:8]}")

    protected_paths = config.get("protected_paths")
    if not isinstance(protected_paths, list):
        errors.append("cannot inspect protected Git state without protected_paths")
        protected_paths = list(BASE_PROTECTED_PATHS)
    for relative in protected_paths:
        tracked = _git("ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            errors.append(f"protected operator input is not tracked: {relative}")
    status = _check_git_result(
        _git("status", "--porcelain=v1", "--", *protected_paths),
        "inspect protected operator-input cleanliness",
        errors,
    )
    if status:
        errors.append(f"protected operator inputs are dirty: {status.splitlines()}")


def validate(*, check_all: bool, check_terminal: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    config = _load_json(CONFIG_PATH, errors)
    task_text = _read(TASKBOARD_PATH, errors)
    goal_text = _read(OBJECTIVES_PATH, errors)
    plan_text = _read(PLAN_PATH, errors)

    _validate_config(config, errors)
    dependencies = _validate_tasks(task_text, config, errors)
    _validate_goals(goal_text, errors)
    _validate_validation_path_closure(task_text, goal_text, dependencies, errors)
    _validate_plan(plan_text, config, errors)
    if check_all:
        _validate_no_capture_lock(errors)
        _validate_git_state(config, errors)
        configured = config.get("operator_baseline_receipts")
        if isinstance(configured, Mapping) and set(configured) == set(
            BASELINE_RECEIPT_SPECS
        ):
            receipts = _validate_operator_baseline_bundle(
                config, errors, enforce_current_sources=False
            )
            synthesis_valid = _validated_baseline_synthesis(config, receipts, errors)
            if check_terminal and not synthesis_valid:
                errors.append(
                    "terminal validation requires the committed, bound IPS-004 synthesis"
                )
            if not synthesis_valid:
                _validate_published_inventory_artifacts(errors)
                if not check_terminal:
                    _validate_current_baseline_sources(configured, receipts, errors)

    edge_count = sum(len(value) for value in dependencies.values())
    return {
        "valid": not errors,
        "check_all": check_all,
        "check_terminal": check_terminal,
        "errors": errors,
        "counts": {
            "tasks_expected": len(TASK_IDS),
            "task_dependency_edges": edge_count,
            "goals_expected": len(GOAL_IDS),
            "strict_lanes": 3,
            "benchmark_transitions": 40,
            "errors": len(errors),
        },
        "source_binding": {
            "accelerate": ACCELERATE_REVISION,
            "ipfs_datasets_py": DATASETS_REVISION,
            "ipfs_kit_py": KIT_REVISION,
        },
    }


def _validate_no_capture_lock(errors: list[str]) -> None:
    current = REPO_ROOT
    try:
        for part in Path(BASELINE_RECEIPT_ROOT).parts:
            current = current / part
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                errors.append("operator baseline receipt root has an unsafe path component")
                return
    except FileNotFoundError:
        return
    except OSError as exc:
        errors.append(f"cannot inspect operator capture lock: {type(exc).__name__}")
        return
    try:
        (current / ".capture.lock").lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        errors.append(f"cannot inspect operator capture lock: {type(exc).__name__}")
        return
    errors.append(
        "operator baseline evidence is ambiguous while stale .capture.lock exists"
    )


def _validate_bootstrap_receipt_root(errors: list[str]) -> None:
    root = REPO_ROOT / BASELINE_RECEIPT_ROOT
    if not os.path.lexists(root):
        return
    try:
        current = REPO_ROOT
        for part in Path(BASELINE_RECEIPT_ROOT).parts:
            current = current / part
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                errors.append("bootstrap baseline receipt root has an unsafe path component")
                return
        with os.scandir(root) as root_entries:
            entries = sorted(root_entries, key=lambda entry: entry.name)
        allowed = {"logs", "work"}
        for entry in entries:
            if entry.name not in allowed:
                errors.append(
                    f"bootstrap baseline receipt root contains unexpected entry {entry.name!r}"
                )
                continue
            if not entry.is_dir(follow_symlinks=False):
                errors.append(
                    f"bootstrap {BASELINE_RECEIPT_ROOT}/{entry.name} is not a safe empty directory"
                )
                continue
            with os.scandir(entry.path) as children:
                if next(children, None) is not None:
                    errors.append(
                        f"bootstrap {BASELINE_RECEIPT_ROOT}/{entry.name} must contain zero "
                        "pre-capture entries"
                    )
    except OSError as exc:
        errors.append(
            f"bootstrap cannot inspect baseline receipt root: {type(exc).__name__}"
        )


def validate_bootstrap() -> dict[str, Any]:
    """Validate committed pre-capture infrastructure with exact empty pins."""

    errors: list[str] = []
    config = _load_json(CONFIG_PATH, errors)
    task_text = _read(TASKBOARD_PATH, errors)
    goal_text = _read(OBJECTIVES_PATH, errors)
    plan_text = _read(PLAN_PATH, errors)
    _validate_config(config, errors, bootstrap=True)
    dependencies = _validate_tasks(task_text, config, errors)
    _validate_goals(goal_text, errors)
    _validate_validation_path_closure(task_text, goal_text, dependencies, errors)
    _validate_plan(plan_text, config, errors)
    _validate_git_state(config, errors)
    outer_status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if outer_status.returncode != 0:
        errors.append("bootstrap could not inspect the full control worktree")
    elif outer_status.stdout.strip():
        errors.append(
            f"bootstrap control worktree is not pristine: {outer_status.stdout.splitlines()[:12]}"
        )
    _validate_bootstrap_receipt_root(errors)
    return {
        "valid": not errors,
        "check_bootstrap": True,
        "errors": errors,
        "counts": {
            "tasks_expected": len(TASK_IDS),
            "task_dependency_edges": sum(len(value) for value in dependencies.values()),
            "goals_expected": len(GOAL_IDS),
            "errors": len(errors),
        },
    }


def _artifact_json(
    relative: str,
    errors: list[str],
    *,
    maximum_bytes: int = BASELINE_MAX_RECEIPT_BYTES,
    bound_label: str = "two-MiB",
) -> Mapping[str, Any]:
    """Load one task-owned JSON artifact without importing project code."""

    retained = _secure_read_repo_file(
        relative,
        required_parent=Path(relative).parent.as_posix(),
        label=f"artifact JSON {relative}",
        maximum_bytes=maximum_bytes,
        bound_label=bound_label,
        errors=errors,
    )
    if retained is None:
        return {}
    try:
        payload = json.loads(
            retained[0].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"artifact JSON is invalid: {relative}: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"artifact must contain a JSON object: {relative}")
        return {}
    return payload


def _require_nonempty_file(
    relative: str,
    errors: list[str],
    *,
    maximum_bytes: int = BASELINE_MAX_LOG_BYTES,
    bound_label: str = "64-MiB",
) -> str:
    retained = _secure_read_repo_file(
        relative,
        required_parent=str(Path(relative).parent),
        label=f"artifact {relative}",
        maximum_bytes=maximum_bytes,
        bound_label=bound_label,
        errors=errors,
    )
    if retained is None:
        return ""
    try:
        text = retained[0].decode("utf-8")
    except UnicodeError:
        errors.append(f"artifact is not UTF-8 text: {relative}")
        return ""
    if not text.strip():
        errors.append(f"artifact is empty: {relative}")
    return text


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_PYTEST_SUMMARY_COUNT = re.compile(
    r"(?<![\w])(?P<count>\d+)\s+"
    r"(?P<label>passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b"
)
_PYTEST_NONPASS = re.compile(
    r"^(?P<status>FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+"
    r"(?:(?:\[\d+\])\s+)?(?P<body>.+?)\s*$"
)
_PYTEST_ITEM_OUTCOME = re.compile(
    r"^(?P<node>\S.*?::\S.*?)\s+"
    r"(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)
_PYTEST_COLLECTION_ERROR = re.compile(
    r"^(?:[=_-]+\s+)?ERROR collecting (?P<node>\S.*?)(?:\s+[=_-]+)?$"
)
_PYTEST_COLLECTED_ERROR = re.compile(
    r"\bcollected\s+\d+\s+items?\b.*(?:/|,)\s*\d+\s+errors?\b",
    re.IGNORECASE,
)
_PYTEST_SKIPPED_SUMMARY = re.compile(
    r"^SKIPPED\s+(?:\[\d+\]\s+)?(?P<node>\S+?:\d+)(?::\s+.*)?$"
)
_PYTEST_SESSION_HEADER = re.compile(
    r"^platform\s+.+?\s+--\s+Python\s+(?P<python>[0-9.]+),\s+"
    r"pytest-(?P<pytest>[0-9.]+)",
    re.MULTILINE,
)
_PYTEST_SUMMARY_DURATION = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the one receipt-canonical JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_ignored_sensitive_inputs(
    value: Any,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label}.ignored_sensitive_inputs must be an object")
        return
    _closed_keys(
        value,
        ("policy_id", "repositories"),
        f"{label}.ignored_sensitive_inputs",
        errors,
    )
    if value.get("policy_id") != BASELINE_IGNORED_INPUT_POLICY:
        errors.append(f"{label}.ignored_sensitive_inputs policy is not reviewed")
    repositories = value.get("repositories")
    if not isinstance(repositories, Mapping):
        errors.append(f"{label}.ignored_sensitive_inputs.repositories must be an object")
        repositories = {}
    _closed_keys(
        repositories,
        REPOSITORY_PATHS,
        f"{label}.ignored_sensitive_inputs.repositories",
        errors,
    )
    for repository in REPOSITORY_PATHS:
        recorded = repositories.get(repository)
        if not isinstance(recorded, Mapping):
            errors.append(
                f"{label}.ignored_sensitive_inputs.repositories.{repository} must be an object"
            )
            recorded = {}
        _closed_keys(
            recorded,
            ("count", "digest"),
            f"{label}.ignored_sensitive_inputs.repositories.{repository}",
            errors,
        )
        count = recorded.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append(
                f"{label}.ignored_sensitive_inputs.repositories.{repository}.count "
                "must be a nonnegative integer"
            )
        _sha256_value(
            recorded.get("digest"),
            f"{label}.ignored_sensitive_inputs.repositories.{repository}.digest",
            errors,
        )
        expected_empty = {
            "count": 0,
            "digest": "sha256:"
            + hashlib.sha256(_canonical_json_bytes([])).hexdigest(),
        }
        if recorded != expected_empty:
            errors.append(
                f"{label}.ignored_sensitive_inputs.repositories.{repository} "
                "is not the captured zero-input binding"
            )


def _reviewed_suite_registry(errors: list[str]) -> dict[str, dict[str, Any]]:
    """Parse the closed protected registry strictly, without executing code."""

    retained = _secure_read_repo_file(
        BASELINE_SUITE_REGISTRY,
        required_parent="config",
        label="protected baseline suite registry",
        maximum_bytes=BASELINE_MAX_REGISTRY_BYTES,
        bound_label="256-KiB",
        errors=errors,
    )
    if retained is None:
        return {}
    raw, retained_digest = retained
    if f"sha256:{retained_digest}" != BASELINE_SUITE_REGISTRY_DIGEST:
        errors.append("protected baseline registry digest differs from the reviewed pin")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"protected baseline registry is not duplicate-free JSON: {exc}")
        return {}
    if raw != _canonical_json_bytes(payload) + b"\n":
        errors.append("protected baseline registry is not canonical JSON")
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "environment_policy_id",
        "repositories",
    }:
        errors.append("protected baseline registry has an unknown top-level schema")
        return {}
    if payload.get("schema_version") != BASELINE_SUITE_REGISTRY_SCHEMA:
        errors.append("protected baseline registry schema drifted")
    if payload.get("environment_policy_id") != BASELINE_ENVIRONMENT_POLICY:
        errors.append("protected baseline registry environment policy drifted")
    repositories = payload.get("repositories")
    if not isinstance(repositories, Mapping) or set(repositories) != {
        "accelerate",
        "datasets",
        "kit",
    }:
        errors.append("protected baseline registry repository set drifted")
        return {}
    reviewed: dict[str, dict[str, Any]] = {}
    for task_spec in BASELINE_RECEIPT_SPECS.values():
        repository = str(task_spec["repository"])
        suites = repositories.get(repository)
        if not isinstance(suites, list):
            errors.append(f"protected baseline registry {repository} suites must be an array")
            continue
        actual_ids = [
            suite.get("id") if isinstance(suite, Mapping) else None for suite in suites
        ]
        if actual_ids != list(task_spec["command_ids"]):
            errors.append(
                f"protected baseline registry {repository} ordered suite ids drifted"
            )
        for index, expected_id in enumerate(task_spec["command_ids"]):
            if index >= len(suites) or not isinstance(suites[index], Mapping):
                continue
            suite = dict(suites[index])
            _closed_keys(
                suite,
                (
                    "id",
                    "repository",
                    "cwd",
                    "argv_template",
                    "environment_policy_id",
                    "timeout_seconds",
                    "observation_note",
                ),
                f"protected suite {expected_id}",
                errors,
            )
            if suite.get("id") != expected_id:
                errors.append(f"protected suite index {index} does not bind {expected_id}")
            if suite.get("repository") != repository:
                errors.append(f"protected suite {expected_id} repository drifted")
            if suite.get("cwd") != task_spec["cwd"]:
                errors.append(f"protected suite {expected_id} cwd drifted")
            if suite.get("timeout_seconds") != task_spec["timeouts"][index]:
                errors.append(f"protected suite {expected_id} timeout drifted")
            if suite.get("environment_policy_id") != BASELINE_ENVIRONMENT_POLICY:
                errors.append(f"protected suite {expected_id} environment policy drifted")
            expected_note = (
                BASELINE_CORE_15_OBSERVATION
                if expected_id == "accelerate-proof-focused-core-15"
                else BASELINE_DEFAULT_OBSERVATION
            )
            if suite.get("observation_note") != expected_note:
                errors.append(f"protected suite {expected_id} observation nonclaim drifted")
            template = suite.get("argv_template")
            if (
                not isinstance(template, list)
                or len(template) < 12
                or any(not isinstance(item, str) or not item for item in template)
            ):
                errors.append(f"protected suite {expected_id} argv template is invalid")
                continue
            if template[:7] != [
                "{python}",
                "-m",
                "pytest",
                "-vv",
                "-ra",
                "--tb=line",
                "--color=no",
            ]:
                errors.append(f"protected suite {expected_id} pytest prefix drifted")
            for required in (
                "--trace-config",
                "-o",
                "cache_dir={cache_dir}",
                "--basetemp={basetemp}",
            ):
                if template.count(required) != 1:
                    errors.append(f"protected suite {expected_id} omits {required!r}")
            if any(
                token in {"-c", "--co", "--collect-only", "no:cacheprovider"}
                or "http://" in token
                or "https://" in token
                for token in template
            ):
                errors.append(f"protected suite {expected_id} contains unsafe argv")
            canonical_digest = "sha256:" + hashlib.sha256(
                _canonical_json_bytes(suite)
            ).hexdigest()
            suite["suite_definition_digest"] = canonical_digest
            reviewed[expected_id] = suite
    return reviewed


def _expected_controlled_environment(
    *,
    capture_root: PurePosixPath,
    workspace_relative: str,
    pytest_module_path: str,
) -> dict[str, str]:
    workspace = capture_root / PurePosixPath(workspace_relative)
    pytest_path = PurePosixPath(pytest_module_path)
    try:
        site_packages = pytest_path.parents[1]
    except IndexError:
        site_packages = pytest_path.parent
    workspace_parts = Path(workspace_relative).parts
    work_prefix = (*Path(BASELINE_RECEIPT_ROOT).parts, "work")
    capture_id = (
        workspace_parts[len(work_prefix)]
        if workspace_parts[: len(work_prefix)] == work_prefix
        and len(workspace_parts) >= len(work_prefix) + 2
        else "invalid-capture"
    )
    source_root = capture_root.joinpath(*work_prefix, capture_id, "source")
    python_path = os.pathsep.join(
        str(path)
        for path in (
            source_root,
            source_root / "ipfs_datasets_py",
            source_root / "ipfs_kit_py",
            site_packages,
        )
    )
    return {
        "CARGO_NET_OFFLINE": "true",
        "COLUMNS": "120",
        "GIT_TERMINAL_PROMPT": "0",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(workspace / "home"),
        "HYPOTHESIS_STORAGE_DIRECTORY": str(workspace / "hypothesis"),
        "IPFS_ACCEL_AUTO_INSTALL": "0",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
        "IPFS_DATASETS_ENABLE_GROTH16": "0",
        "IPFS_DATASETS_PY_AUTO_GROTH16_BUILD": "0",
        "IPFS_DATASETS_RUN_GROTH16_EVM": "0",
        "IPFS_DATASETS_RUN_PROVEKIT_TESTS": "0",
        "IPFS_OFFLINE": "1",
        "IPFS_PATH": str(workspace / "ipfs-repo"),
        "IPFS_TEST_PROOF_REUSE_MODE": "off",
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_COLOR": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": str(workspace / "pycache"),
        "PYTHONPATH": python_path,
        "PYTEST_ADDOPTS": (
            f"--benchmark-storage=file://{workspace / 'pytest-benchmark'}"
        ),
        "TERM": "dumb",
        "TMPDIR": str(workspace / "tmp"),
        "TRANSFORMERS_OFFLINE": "1",
        "TZ": "UTC",
    }


def _capture_root_from_absolute_path(
    value: Any,
    *,
    expected_suffix: str,
    label: str,
    errors: list[str],
) -> PurePosixPath | None:
    """Strip one exact capture-local suffix from a canonical absolute path."""

    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a canonical absolute capture path")
        return None
    path = PurePosixPath(value)
    suffix = PurePosixPath(expected_suffix)
    if (
        not path.is_absolute()
        or value.startswith("//")
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        errors.append(f"{label} must be a canonical absolute capture path")
        return None
    suffix_parts = suffix.parts
    if (
        suffix.is_absolute()
        or expected_suffix != suffix.as_posix()
        or not suffix_parts
        or any(part in {"", ".", ".."} for part in suffix_parts)
        or tuple(path.parts[-len(suffix_parts) :]) != suffix_parts
    ):
        errors.append(f"{label} does not have the exact capture-local suffix")
        return None
    root_parts = path.parts[: -len(suffix_parts)]
    if not root_parts:
        errors.append(f"{label} does not contain an absolute capture root")
        return None
    root = PurePosixPath(*root_parts)
    if not root.is_absolute() or root.as_posix().startswith("//"):
        errors.append(f"{label} does not contain an absolute capture root")
        return None
    return root


def _infer_historical_capture_root(
    commands: list[Any],
    *,
    expected_ids: list[str],
    capture_id: str,
    receipt_label: str,
    errors: list[str],
) -> PurePosixPath | None:
    """Infer one immutable historical checkout root from every command anchor."""

    roots: list[tuple[str, PurePosixPath]] = []

    def add_anchor(value: Any, suffix: str, anchor_label: str) -> None:
        root = _capture_root_from_absolute_path(
            value,
            expected_suffix=suffix,
            label=anchor_label,
            errors=errors,
        )
        if root is not None:
            roots.append((anchor_label, root))

    work_root = f"{BASELINE_RECEIPT_ROOT}/work/{capture_id}"
    for index, expected_id in enumerate(expected_ids):
        command = commands[index] if index < len(commands) else None
        if not isinstance(command, Mapping):
            continue
        label = f"{receipt_label}.commands[{expected_id}]"
        workspace_relative = f"{work_root}/{expected_id}"
        argv = command.get("argv")
        if isinstance(argv, list):
            for prefix, leaf in (
                ("cache_dir=", "pytest-cache"),
                ("--basetemp=", "pytest"),
            ):
                values = [
                    token[len(prefix) :]
                    for token in argv
                    if isinstance(token, str) and token.startswith(prefix)
                ]
                if len(values) != 1:
                    errors.append(f"{label}.argv must contain one exact {prefix} anchor")
                else:
                    add_anchor(
                        values[0],
                        f"{workspace_relative}/{leaf}",
                        f"{label}.argv {prefix}",
                    )
        environment = command.get("environment")
        variables = (
            environment.get("variables") if isinstance(environment, Mapping) else None
        )
        if not isinstance(variables, Mapping):
            continue
        for key, leaf in (
            ("HOME", "home"),
            ("HYPOTHESIS_STORAGE_DIRECTORY", "hypothesis"),
            ("IPFS_PATH", "ipfs-repo"),
            ("PYTHONPYCACHEPREFIX", "pycache"),
            ("TMPDIR", "tmp"),
        ):
            add_anchor(
                variables.get(key),
                f"{workspace_relative}/{leaf}",
                f"{label}.environment.{key}",
            )
        benchmark = variables.get("PYTEST_ADDOPTS")
        benchmark_prefix = "--benchmark-storage=file://"
        if not isinstance(benchmark, str) or not benchmark.startswith(
            benchmark_prefix
        ):
            errors.append(
                f"{label}.environment.PYTEST_ADDOPTS lacks its exact file anchor"
            )
        else:
            add_anchor(
                benchmark[len(benchmark_prefix) :],
                f"{workspace_relative}/pytest-benchmark",
                f"{label}.environment.PYTEST_ADDOPTS",
            )
        python_path = variables.get("PYTHONPATH")
        python_entries = (
            python_path.split(os.pathsep) if isinstance(python_path, str) else []
        )
        if len(python_entries) != 4:
            errors.append(
                f"{label}.environment.PYTHONPATH must contain four ordered paths"
            )
        else:
            source_relative = f"{work_root}/source"
            for entry_index, suffix in enumerate(
                (
                    source_relative,
                    f"{source_relative}/ipfs_datasets_py",
                    f"{source_relative}/ipfs_kit_py",
                )
            ):
                add_anchor(
                    python_entries[entry_index],
                    suffix,
                    f"{label}.environment.PYTHONPATH[{entry_index}]",
                )

    distinct = {root.as_posix() for _label, root in roots}
    if not roots:
        errors.append(f"{receipt_label} has no valid historical capture-root anchors")
        return None
    if len(distinct) != 1:
        errors.append(
            f"{receipt_label} command paths do not bind one canonical absolute "
            "historical capture root"
        )
        return None
    return roots[0][1]


def _looks_patterned_digest(hex_digest: str) -> bool:
    """Reject conspicuous placeholder digests even before content rehashing."""

    if len(set(hex_digest)) < 8:
        return True
    for period in range(1, 17):
        if len(hex_digest) % period == 0:
            fragment = hex_digest[:period]
            if fragment * (len(hex_digest) // period) == hex_digest:
                return True
    patterned = (
        "0123456789abcdef" * 4,
        "fedcba9876543210" * 4,
        "00112233445566778899aabbccddeeff" * 2,
        "deadbeef" * 8,
    )
    return hex_digest in patterned


def _sha256_value(
    value: Any,
    label: str,
    errors: list[str],
) -> str | None:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        errors.append(f"{label} must use the sha256:<64 lowercase hex> form")
        return None
    hex_digest = value[7:]
    if not _HEX_64.fullmatch(hex_digest):
        errors.append(f"{label} is not a canonical lowercase SHA-256 digest")
        return None
    if _looks_patterned_digest(hex_digest):
        errors.append(f"{label} is a conspicuously patterned placeholder digest")
        return None
    return hex_digest


def _secure_read_repo_file(
    relative: Any,
    *,
    required_parent: str,
    label: str,
    maximum_bytes: int,
    bound_label: str,
    errors: list[str],
) -> tuple[bytes, str] | None:
    """Open through no-follow dirfds, bound the read, and verify stable identity."""

    if not isinstance(relative, str) or not relative:
        errors.append(f"{label} must be a non-empty repository-relative path")
        return None
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or "\\" in relative
        or candidate.as_posix() != relative
        or ".." in candidate.parts
        or relative.startswith("./")
    ):
        errors.append(f"{label} is not a canonical repository-relative path")
        return None
    parent = Path(required_parent)
    if candidate.parent != parent:
        errors.append(f"{label} must be directly beneath {required_parent}")
        return None
    try:
        root = REPO_ROOT.resolve(strict=True)
    except OSError as exc:
        errors.append(f"cannot resolve repository root for {label}: {type(exc).__name__}")
        return None
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent_fd: int | None = None
    file_fd: int | None = None
    try:
        parent_fd = os.open(root, directory_flags)
        for part in candidate.parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        file_fd = os.open(candidate.parts[-1], file_flags, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            errors.append(f"{label} must name a retained regular non-symlink file")
            return None
        if before.st_size > maximum_bytes:
            errors.append(f"{label} exceeds the fixed {bound_label} bound")
            return None
        raw = bytearray()
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, maximum_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            digest.update(chunk)
            if len(raw) > maximum_bytes:
                errors.append(f"{label} exceeds the fixed {bound_label} bound")
                return None
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
        )
        if identity_before != identity_after or len(raw) != after.st_size:
            errors.append(f"{label} changed while it was read")
            return None
        path_after = os.stat(
            candidate.parts[-1], dir_fd=parent_fd, follow_symlinks=False
        )
        if identity_after != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_size,
        ):
            errors.append(f"{label} path identity changed while it was read")
            return None
        return bytes(raw), digest.hexdigest()
    except OSError as exc:
        errors.append(
            f"cannot safely read {label}; a symlink or unsafe component may be present: "
            f"{type(exc).__name__}"
        )
        return None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _parse_timestamp(value: Any, label: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty ISO-8601 timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{label} is not an ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        errors.append(f"{label} must include a UTC offset")
        return None
    return parsed


def _summary_counts(summary_line: str) -> dict[str, int]:
    counts = {
        field: 0
        for field in BASELINE_OUTCOME_FIELDS
        if field not in {"selected", "collected_count"}
    }
    for match in _PYTEST_SUMMARY_COUNT.finditer(summary_line):
        label = match.group("label")
        key = "errors" if label in {"error", "errors"} else label
        counts[key] += int(match.group("count"))
    counts["selected"] = sum(
        counts[field]
        for field in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
    )
    return counts


def _collection_count(log_text: str) -> int | None:
    matches = re.findall(
        r"(?i)\bcollected\s+(\d+)\s+items?\b",
        log_text,
    )
    return int(matches[-1]) if matches else None


def _collection_complete(log_text: str) -> bool:
    """Distinguish collected test items from module-level collection skips."""

    collected = _collection_count(log_text)
    if collected is None or any(
        _PYTEST_COLLECTION_ERROR.match(line.strip())
        for line in log_text.splitlines()
    ):
        return False
    if any(_PYTEST_COLLECTED_ERROR.search(line) for line in log_text.splitlines()):
        return False
    return not any(
        item["status"] == "skipped" and "::" not in item["node_id"]
        for item in _nonpass_nodes(log_text)
    )


def _nonpass_nodes(log_text: str) -> list[dict[str, str]]:
    """Reparse pytest's retained short-summary node records."""

    lines = log_text.splitlines()
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    collection_error_nodes = {
        match.group("node").strip()
        for raw_line in lines
        for match in (
            _PYTEST_COLLECTION_ERROR.match(
                _ANSI_ESCAPE.sub("", raw_line).strip()
            ),
        )
        if match is not None
    }
    collection_count_failed = any(
        _PYTEST_COLLECTED_ERROR.search(raw_line) for raw_line in lines
    )
    status_names = {
        "FAILED": "failed",
        "ERROR": "error",
        "SKIPPED": "skipped",
        "XFAIL": "xfailed",
        "XPASS": "xpassed",
    }
    for raw_line in lines:
        detail = _ANSI_ESCAPE.sub("", raw_line).strip()
        item_match = _PYTEST_ITEM_OUTCOME.match(detail)
        collection_match = _PYTEST_COLLECTION_ERROR.match(detail)
        summary_match = _PYTEST_NONPASS.fullmatch(detail)
        skipped_match = _PYTEST_SKIPPED_SUMMARY.fullmatch(detail)
        if item_match is not None:
            raw_status = item_match.group("status")
            if raw_status == "PASSED":
                continue
            node_id = item_match.group("node").strip()
        elif collection_match is not None:
            raw_status = "ERROR"
            node_id = f"collecting {collection_match.group('node')}"
        elif skipped_match is not None:
            raw_status = "SKIPPED"
            node_id = skipped_match.group("node")
            if any(
                existing["status"] == "skipped"
                and "::" in existing["node_id"]
                and node_id.startswith(existing["node_id"].split("::", 1)[0] + ":")
                for existing in result
            ):
                continue
        elif summary_match is not None and summary_match.group("status") != "SKIPPED":
            raw_status = summary_match.group("status")
            node_id = summary_match.group("body").split(" - ", 1)[0].strip()
        else:
            continue
        if (
            raw_status == "ERROR"
            and not node_id.startswith("collecting ")
            and (
                node_id in collection_error_nodes
                or (collection_count_failed and "::" not in node_id)
            )
        ):
            node_id = f"collecting {node_id}"
        item = {
            "status": status_names[raw_status],
            "node_id": node_id,
            "detail": detail,
        }
        identity = (item["status"], item["node_id"])
        if identity not in seen:
            seen.add(identity)
            result.append(item)
    return result


def _closed_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    label: str,
    errors: list[str],
) -> None:
    expected_keys = set(expected)
    actual_keys = set(value)
    for key in sorted(expected_keys - actual_keys):
        errors.append(f"{label} is missing field {key!r}")
    for key in sorted(actual_keys - expected_keys):
        errors.append(f"{label} has undeclared field {key!r}")


def _relative_directory(value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty relative directory")
        return None
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or "\\" in value
        or candidate.as_posix() != value
        or ".." in candidate.parts
        or value.startswith("./")
    ):
        errors.append(f"{label} is not a canonical repository-relative directory")
        return None
    path = REPO_ROOT / candidate
    try:
        if path.is_symlink() or not path.is_dir():
            errors.append(f"{label} does not name a retained repository directory")
            return None
        if not path.resolve().is_relative_to(REPO_ROOT.resolve()):
            errors.append(f"{label} resolves outside the repository")
            return None
    except OSError as exc:
        errors.append(f"cannot inspect {label}: {type(exc).__name__}")
        return None
    return path


def _validate_baseline_command(
    command: Mapping[str, Any],
    *,
    expected_id: str,
    expected_cwd: str,
    expected_timeout: int,
    expected_suite: Mapping[str, Any],
    capture_id: str,
    historical_capture_root: PurePosixPath | None,
    receipt_label: str,
    seen_logs: set[str],
    errors: list[str],
) -> None:
    label = f"{receipt_label}.commands[{expected_id}]"
    _closed_keys(
        command,
        (
            "id",
            "evidence_type",
            "suite_definition_digest",
            "command_digest",
            "argv",
            "cwd",
            "workspace_relative_path",
            "python",
            "pytest",
            "environment",
            "started_at",
            "finished_at",
            "duration_ns",
            "timeout_seconds",
            "exit_code",
            "capture_status",
            "collected_count",
            "collection_complete",
            "outcome_counts",
            "non_pass_nodes",
            "summary_line",
            "parse_error",
            "log",
            "assurance",
        ),
        label,
        errors,
    )
    if command.get("id") != expected_id:
        errors.append(f"{label}.id must equal {expected_id!r}")
    if command.get("evidence_type") != "pytest_execution_observation":
        errors.append(
            f"{label}.evidence_type must equal 'pytest_execution_observation'"
        )
    _sha256_value(command.get("suite_definition_digest"), f"{label}.suite_definition_digest", errors)
    if command.get("suite_definition_digest") != expected_suite.get(
        "suite_definition_digest"
    ):
        errors.append(f"{label}.suite_definition_digest differs from reviewed registry")
    declared_command_digest = _sha256_value(
        command.get("command_digest"), f"{label}.command_digest", errors
    )

    argv = command.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        errors.append(f"{label}.argv must be a non-empty string array")
        argv = []
    if argv:
        executable_name = Path(argv[0]).name.casefold()
        if executable_name in {
            "bash",
            "cmd",
            "dash",
            "fish",
            "ksh",
            "powershell",
            "pwsh",
            "sh",
            "zsh",
        }:
            errors.append(f"{label}.argv may not invoke a shell")
        if len(argv) < 3 or argv[1:3] != ["-m", "pytest"]:
            errors.append(f"{label}.argv must invoke the measured pytest module")
        if any(item in {"-c", "--collect-only", "--co"} for item in argv[1:]):
            errors.append(f"{label}.argv contains a non-execution or dynamic-eval flag")

    _relative_directory(command.get("cwd"), f"{label}.cwd", errors)
    if command.get("cwd") != expected_cwd:
        errors.append(f"{label}.cwd does not match the fixed suite repository")
    workspace_relative = command.get("workspace_relative_path")
    expected_workspace_prefix = f"{BASELINE_RECEIPT_ROOT}/work/{capture_id}/{expected_id}"
    if workspace_relative != expected_workspace_prefix:
        errors.append(f"{label}.workspace_relative_path is not the fixed capture workspace")
    elif historical_capture_root is not None:
        workspace = historical_capture_root / PurePosixPath(expected_workspace_prefix)
        expected_basetemp = f"--basetemp={workspace / 'pytest'}"
        expected_cache = f"cache_dir={workspace / 'pytest-cache'}"
        for token in (
            "-vv",
            "-ra",
            "--tb=line",
            "--color=no",
            "--trace-config",
            "-o",
            expected_cache,
            expected_basetemp,
        ):
            if token not in argv:
                errors.append(f"{label}.argv omits fixed hermetic token {token!r}")
        if "-p" in argv and "no:cacheprovider" in argv:
            errors.append(f"{label}.argv disables the required cacheprovider")
    python_info = command.get("python")
    if not isinstance(python_info, Mapping):
        errors.append(f"{label}.python must be an object")
        python_info = {}
    else:
        _closed_keys(
            python_info,
            ("executable", "implementation", "version"),
            f"{label}.python",
            errors,
        )
    python_executable = python_info.get("executable")
    if not isinstance(python_executable, str) or not Path(python_executable).is_absolute():
        errors.append(f"{label}.python.executable must be an absolute path")
    elif argv and argv[0] != python_executable:
        errors.append(f"{label}.argv[0] does not match python.executable")
    if python_info.get("implementation") != "CPython":
        errors.append(f"{label}.python.implementation must equal 'CPython'")
    if not isinstance(python_info.get("version"), str) or not python_info.get("version"):
        errors.append(f"{label}.python.version must be a concrete version")

    pytest_info = command.get("pytest")
    if not isinstance(pytest_info, Mapping):
        errors.append(f"{label}.pytest must be an object")
        pytest_info = {}
    else:
        _closed_keys(
            pytest_info,
            ("version", "module_path", "autoload_plugins"),
            f"{label}.pytest",
            errors,
        )
    if not isinstance(pytest_info.get("version"), str) or not pytest_info.get("version"):
        errors.append(f"{label}.pytest.version must be concrete")
    module_path = pytest_info.get("module_path")
    if not isinstance(module_path, str) or not Path(module_path).is_absolute():
        errors.append(f"{label}.pytest.module_path must be an absolute path")
    autoload_plugins = pytest_info.get("autoload_plugins")
    if not isinstance(autoload_plugins, list):
        errors.append(f"{label}.pytest.autoload_plugins must be an ordered array")
    else:
        seen_plugins: set[tuple[str, str]] = set()
        plugin_identities: list[tuple[str, str]] = []
        for index, plugin in enumerate(autoload_plugins):
            if not isinstance(plugin, Mapping):
                errors.append(f"{label}.pytest.autoload_plugins[{index}] must be an object")
                continue
            _closed_keys(
                plugin,
                ("name", "value", "distribution", "version"),
                f"{label}.pytest.autoload_plugins[{index}]",
                errors,
            )
            if any(
                not isinstance(plugin.get(field), str) or not plugin.get(field)
                for field in ("name", "value")
            ) or any(
                plugin.get(field) is not None
                and (not isinstance(plugin.get(field), str) or not plugin.get(field))
                for field in ("distribution", "version")
            ):
                errors.append(
                    f"{label}.pytest.autoload_plugins[{index}] has empty metadata"
                )
            identity = (str(plugin.get("name")), str(plugin.get("value")))
            plugin_identities.append(identity)
            if identity in seen_plugins:
                errors.append(f"{label}.pytest.autoload_plugins repeats {identity!r}")
            seen_plugins.add(identity)
        if plugin_identities != sorted(plugin_identities):
            errors.append(f"{label}.pytest.autoload_plugins is not deterministic")

    environment = command.get("environment")
    if not isinstance(environment, Mapping):
        errors.append(f"{label}.environment must be an object")
        environment = {}
    else:
        _closed_keys(
            environment,
            ("policy_id", "variables"),
            f"{label}.environment",
            errors,
        )
    if environment.get("policy_id") != BASELINE_ENVIRONMENT_POLICY:
        errors.append(
            f"{label}.environment.policy_id must equal {BASELINE_ENVIRONMENT_POLICY!r}"
        )
    variables = environment.get("variables")
    if not isinstance(variables, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in (variables.items() if isinstance(variables, Mapping) else ())
    ):
        errors.append(f"{label}.environment.variables must be a string map")
    elif set(variables) != BASELINE_ENVIRONMENT_KEYS:
        errors.append(f"{label}.environment.variables is not the closed hermetic set")
    elif (
        workspace_relative == expected_workspace_prefix
        and historical_capture_root is not None
    ):
        workspace = historical_capture_root / PurePosixPath(expected_workspace_prefix)
        if variables.get("HOME") != str(workspace / "home"):
            errors.append(f"{label}.environment HOME is not capture-local")
        if variables.get("TMPDIR") != str(workspace / "tmp"):
            errors.append(f"{label}.environment TMPDIR is not capture-local")
        if variables.get("PYTHONPYCACHEPREFIX") != str(workspace / "pycache"):
            errors.append(
                f"{label}.environment PYTHONPYCACHEPREFIX is not capture-local"
            )
        if variables.get("HYPOTHESIS_STORAGE_DIRECTORY") != str(
            workspace / "hypothesis"
        ):
            errors.append(
                f"{label}.environment HYPOTHESIS_STORAGE_DIRECTORY is not capture-local"
            )
        for offline_key in (
            "CARGO_NET_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "HF_HUB_OFFLINE",
            "IPFS_OFFLINE",
            "PIP_NO_INDEX",
            "TRANSFORMERS_OFFLINE",
        ):
            if variables.get(offline_key) not in {"1", "true"}:
                errors.append(f"{label}.environment {offline_key} is not offline")
    if (
        isinstance(workspace_relative, str)
        and isinstance(python_executable, str)
        and isinstance(module_path, str)
        and historical_capture_root is not None
    ):
        workspace = historical_capture_root / PurePosixPath(workspace_relative)
        template = expected_suite.get("argv_template")
        if isinstance(template, list):
            expected_argv = [
                token.replace("{python}", python_executable)
                .replace("{basetemp}", str(workspace / "pytest"))
                .replace("{cache_dir}", str(workspace / "pytest-cache"))
                for token in template
            ]
            if argv != expected_argv:
                errors.append(f"{label}.argv differs from the protected reviewed suite")
        expected_variables = _expected_controlled_environment(
            capture_root=historical_capture_root,
            workspace_relative=workspace_relative,
            pytest_module_path=module_path,
        )
        if variables != expected_variables:
            errors.append(f"{label}.environment differs from the controlled policy")
    if argv and isinstance(environment, Mapping):
        command_preimage = {
            "id": expected_id,
            "argv": argv,
            "cwd": command.get("cwd"),
            "environment": environment,
        }
        actual_command_digest = hashlib.sha256(
            _canonical_json_bytes(command_preimage)
        ).hexdigest()
        if declared_command_digest is not None and declared_command_digest != actual_command_digest:
            errors.append(f"{label}.command_digest does not match canonical command content")

    started = _parse_timestamp(command.get("started_at"), f"{label}.started_at", errors)
    finished = _parse_timestamp(command.get("finished_at"), f"{label}.finished_at", errors)
    duration_ns = command.get("duration_ns")
    if isinstance(duration_ns, bool) or not isinstance(duration_ns, int) or duration_ns <= 0:
        errors.append(f"{label}.duration_ns must be a positive measured integer")
    if started is not None and finished is not None and finished <= started:
        errors.append(f"{label}.finished_at must be later than started_at")
    if started is not None and finished is not None and isinstance(duration_ns, int):
        wall_ns = int((finished - started).total_seconds() * 1_000_000_000)
        if abs(wall_ns - duration_ns) > 5_000_000_000:
            errors.append(f"{label}.duration_ns disagrees with UTC timestamps")
    timeout_seconds = command.get("timeout_seconds")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        errors.append(f"{label}.timeout_seconds must be a positive bounded integer")
    elif timeout_seconds != expected_timeout:
        errors.append(f"{label}.timeout_seconds does not match the fixed suite")
    exit_code = command.get("exit_code")
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or exit_code not in range(5)
    ):
        errors.append(f"{label}.exit_code must be a closed pytest exit code")
    if command.get("capture_status") != "completed":
        errors.append(f"{label}.capture_status is not an admissible completed run")

    counts = command.get("outcome_counts")
    if not isinstance(counts, Mapping):
        errors.append(f"{label}.outcome_counts must be an object")
        counts = {}
    else:
        _closed_keys(counts, BASELINE_OUTCOME_FIELDS, f"{label}.outcome_counts", errors)
    for field in BASELINE_OUTCOME_FIELDS:
        value = counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{label}.outcome_counts.{field} must be a nonnegative integer")
    selected_formula = sum(
        counts.get(field, 0)
        for field in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
        if isinstance(counts.get(field, 0), int)
        and not isinstance(counts.get(field, 0), bool)
    )
    if counts.get("selected") != selected_formula:
        errors.append(f"{label}.outcome_counts.selected violates the closed formula")
    if selected_formula == 0:
        errors.append(f"{label} is zero-count evidence, not an executed pytest baseline")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        blocking = counts.get("failed", 0) + counts.get("errors", 0)
        if exit_code == 0 and blocking:
            errors.append(f"{label} reports blocking outcomes with a zero exit code")
        if exit_code == 1 and not blocking:
            errors.append(f"{label} has a nonzero exit code without failed/error outcomes")

    collected_count = command.get("collected_count")
    collection_complete = command.get("collection_complete")
    if (
        collected_count is not None
        and (isinstance(collected_count, bool) or not isinstance(collected_count, int) or collected_count < 0)
    ):
        errors.append(f"{label}.collected_count must be null or a nonnegative integer")
    if not isinstance(collection_complete, bool):
        errors.append(f"{label}.collection_complete must be boolean")
    deselected_count = counts.get("deselected")
    if (
        collection_complete is True
        and isinstance(collected_count, int)
        and not isinstance(collected_count, bool)
        and isinstance(deselected_count, int)
        and not isinstance(deselected_count, bool)
        and collected_count != selected_formula + deselected_count
    ):
        errors.append(f"{label}.collection and outcome counts disagree")
    if command.get("parse_error") is not None:
        errors.append(f"{label}.parse_error must be null for admissible evidence")

    expected_assurance = {
        "process_observed": True,
        "test_execution_cryptographically_proven": False,
        "cryptographic_proof": False,
        "signature": None,
        "network_isolation_enforced": False,
        "offline_controls_requested": True,
        "pytest_plugin_allowlist_enforced": False,
        "public_log_witness_policy": "public-full-log-secret-scan@1",
        "inherited_secrets_forwarded": False,
        "remaining_trust": [
            "Host socket access is not isolated; fixed offline controls are requested and trusted.",
            "Installed pytest11 plugins are recorded but not allowlisted.",
            "Selected tests and subprocesses are trusted not to bypass fixed controls.",
        ],
        "claim": "Observed output from the fixed pytest subprocess only.",
    }
    if command.get("assurance") != expected_assurance:
        errors.append(f"{label}.assurance overstates or changes the process-only claim")

    log = command.get("log")
    if not isinstance(log, Mapping):
        errors.append(f"{label}.log must be an object")
        log = {}
    else:
        _closed_keys(log, ("relative_path", "sha256", "bytes"), f"{label}.log", errors)
    log_relative = log.get("relative_path")
    retained_log = _secure_read_repo_file(
        log_relative,
        required_parent=BASELINE_LOG_ROOT,
        label=f"{label}.log.relative_path",
        maximum_bytes=BASELINE_MAX_LOG_BYTES,
        bound_label="64-MiB",
        errors=errors,
    )
    if isinstance(log_relative, str):
        if log_relative in seen_logs:
            errors.append(f"{label} repeats retained log path {log_relative!r}")
        seen_logs.add(log_relative)
        expected_log_relative = f"{BASELINE_LOG_ROOT}/{expected_id}-{capture_id}.log"
        if log_relative != expected_log_relative:
            errors.append(f"{label}.log path does not bind the command and capture ids")
    declared_log_digest = _sha256_value(log.get("sha256"), f"{label}.log.sha256", errors)
    log_bytes = log.get("bytes")
    if isinstance(log_bytes, bool) or not isinstance(log_bytes, int) or log_bytes <= 0:
        errors.append(f"{label}.log.bytes must be a positive integer")

    if retained_log is None:
        return
    raw_log, actual_digest = retained_log
    if isinstance(log_bytes, int) and not isinstance(log_bytes, bool) and len(raw_log) != log_bytes:
        errors.append(f"{label}.log.bytes does not match the full retained log")
    if declared_log_digest is not None and actual_digest != declared_log_digest:
        errors.append(f"{label}.log.sha256 does not match the full retained log")
    try:
        log_text = raw_log.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{label}.log is not canonical UTF-8")
        return
    stripped_log = _ANSI_ESCAPE.sub("", log_text)
    header = _PYTEST_SESSION_HEADER.search(stripped_log)
    if header is None or "test session starts" not in stripped_log:
        errors.append(f"{label}.log lacks a concrete pytest session header")
    else:
        python_version = python_info.get("version")
        pytest_version = pytest_info.get("version")
        if not isinstance(python_version, str) or not python_version.startswith(
            header.group("python")
        ):
            errors.append(f"{label}.python.version disagrees with the retained log")
        if pytest_version != header.group("pytest"):
            errors.append(f"{label}.pytest.version disagrees with the retained log")
    summary_line = command.get("summary_line")
    if not isinstance(summary_line, str) or not summary_line.strip():
        errors.append(f"{label}.summary_line must be a non-empty parsed line")
        return
    if summary_line != _ANSI_ESCAPE.sub("", summary_line).strip():
        errors.append(f"{label}.summary_line must be stripped and ANSI-free")
    matching_summary_lines = [
        line.strip()
        for line in stripped_log.splitlines()
        if line.strip() == summary_line
    ]
    if len(matching_summary_lines) != 1:
        errors.append(f"{label}.summary_line must occur exactly once in its retained log")
    retained_nonempty_lines = [
        line.strip() for line in stripped_log.splitlines() if line.strip()
    ]
    if not retained_nonempty_lines or retained_nonempty_lines[-1] != summary_line:
        errors.append(f"{label}.summary_line must be the final retained output line")
    if not _PYTEST_SUMMARY_DURATION.search(summary_line):
        errors.append(f"{label}.summary_line lacks a measured pytest duration")
    parsed_counts = _summary_counts(summary_line)
    if not any(parsed_counts.values()):
        errors.append(f"{label}.summary_line has no parseable pytest outcomes")
    for field, parsed_value in parsed_counts.items():
        if counts.get(field) != parsed_value:
            errors.append(
                f"{label}.outcome_counts.{field} does not match the retained summary"
            )
    parsed_collected = _collection_count(stripped_log)
    if parsed_collected != collected_count:
        errors.append(
            f"{label}.collected_count does not match the retained log"
        )
    parsed_collection_complete = _collection_complete(stripped_log)
    if collection_complete != parsed_collection_complete:
        errors.append(f"{label}.collection_complete does not match the retained log")
    declared_nonpass = command.get("non_pass_nodes")
    if not isinstance(declared_nonpass, list):
        errors.append(f"{label}.non_pass_nodes must be an array")
    else:
        for index, item in enumerate(declared_nonpass):
            if not isinstance(item, Mapping):
                errors.append(f"{label}.non_pass_nodes[{index}] must be an object")
                continue
            _closed_keys(
                item,
                ("status", "node_id", "detail"),
                f"{label}.non_pass_nodes[{index}]",
                errors,
            )
            if item.get("status") not in {"failed", "error", "skipped", "xfailed", "xpassed"}:
                errors.append(f"{label}.non_pass_nodes[{index}].status is invalid")
            if not isinstance(item.get("node_id"), str) or not item.get("node_id"):
                errors.append(f"{label}.non_pass_nodes[{index}].node_id is empty")
            if not isinstance(item.get("detail"), str) or not item.get("detail"):
                errors.append(f"{label}.non_pass_nodes[{index}].detail is empty")
        if declared_nonpass != _nonpass_nodes(stripped_log):
            errors.append(f"{label}.non_pass_nodes do not exactly reparse from the retained log")


def _validate_baseline_receipt(
    task_id: str,
    spec: Mapping[str, Any],
    errors: list[str],
    *,
    bundle_capture_roots: dict[str, PurePosixPath] | None = None,
) -> Mapping[str, Any]:
    receipt_relative = str(spec["receipt"])
    retained_receipt = _secure_read_repo_file(
        receipt_relative,
        required_parent=BASELINE_RECEIPT_ROOT,
        label=f"{task_id} operator baseline receipt path",
        maximum_bytes=BASELINE_MAX_RECEIPT_BYTES,
        bound_label="two-MiB",
        errors=errors,
    )
    label = f"{task_id} operator baseline receipt"
    receipt: Mapping[str, Any] = {}
    if retained_receipt is not None:
        raw_receipt, _ = retained_receipt
        try:
            decoded = json.loads(
                raw_receipt.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{label} is not canonical duplicate-free JSON: {exc}")
        else:
            if not isinstance(decoded, Mapping):
                errors.append(f"{label} must be one JSON object")
            else:
                receipt = decoded
        expected_receipt_bytes = _canonical_json_bytes(receipt) + b"\n"
        if raw_receipt != expected_receipt_bytes:
            errors.append(f"{label} file bytes are not the canonical JSON encoding")
    _closed_keys(
        receipt,
        (
            "schema_version",
            "operator_origin",
            "repository",
            "task_id",
            "capture_id",
            "captured_at",
            "required_command_ids",
            "planning_revision",
            "planning_tree",
            "source_revision",
            "source_tree",
            "execution_head",
            "execution_tree",
            "source_revisions",
            "source_trees",
            "source_clean_before",
            "source_clean_after",
            "ignored_sensitive_inputs",
            "git_environment_policy_id",
            "commands",
            "assurance",
            "receipt_digest",
        ),
        label,
        errors,
    )
    if receipt.get("schema_version") != BASELINE_RECEIPT_SCHEMA:
        errors.append(f"{label}.schema_version is not the reviewed schema")
    if receipt.get("repository") != spec["repository"]:
        errors.append(f"{label}.repository does not match its fixed receipt path")
    if receipt.get("task_id") != task_id:
        errors.append(f"{label}.task_id does not match the inventory task")
    capture_id = receipt.get("capture_id")
    if not isinstance(capture_id, str) or not re.fullmatch(
        r"\d{8}T\d{6}\.\d{6}Z-\d+", capture_id
    ):
        errors.append(f"{label}.capture_id is not the fixed capture identifier form")
        capture_id = "invalid-capture-id"
    if receipt.get("planning_revision") != spec["revision"]:
        errors.append(f"{label}.planning_revision does not match the exact bound revision")
    if receipt.get("planning_tree") != PLANNING_TREES[spec["repository"]]:
        errors.append(f"{label}.planning_tree does not match the reviewed source tree")

    planning_revisions = {
        "accelerate": ACCELERATE_REVISION,
        "datasets": DATASETS_REVISION,
        "kit": KIT_REVISION,
    }
    source_revisions = receipt.get("source_revisions")
    source_trees = receipt.get("source_trees")
    if not isinstance(source_revisions, Mapping) or set(source_revisions) != set(planning_revisions):
        errors.append(f"{label}.source_revisions must exactly bind all repositories")
        source_revisions = {}
    if not isinstance(source_trees, Mapping) or set(source_trees) != set(planning_revisions):
        errors.append(f"{label}.source_trees must exactly bind all repositories")
        source_trees = {}
    for repository, planning_revision in planning_revisions.items():
        tested_revision = source_revisions.get(repository)
        tested_tree = source_trees.get(repository)
        if not isinstance(tested_revision, str) or not _HEX_40.fullmatch(tested_revision):
            errors.append(f"{label}.source_revisions.{repository} is not a Git commit")
            continue
        if not isinstance(tested_tree, str) or not _HEX_40.fullmatch(tested_tree):
            errors.append(f"{label}.source_trees.{repository} is not a Git tree")
            continue
        repository_root = REPO_ROOT / REPOSITORY_PATHS[repository]
        ancestry = _git(
            "merge-base",
            "--is-ancestor",
            planning_revision,
            tested_revision,
            cwd=repository_root,
        )
        if ancestry.returncode != 0:
            errors.append(
                f"{label}.source_revisions.{repository} does not descend from planning"
            )
        resolved_tree = _git("rev-parse", f"{tested_revision}^{{tree}}", cwd=repository_root)
        if resolved_tree.returncode != 0 or resolved_tree.stdout.strip() != tested_tree:
            errors.append(f"{label}.source_trees.{repository} mismatches its commit")
    tested_control_revision = source_revisions.get("accelerate")
    if isinstance(tested_control_revision, str) and _HEX_40.fullmatch(
        tested_control_revision
    ):
        for nested_repository, nested_path in (
            ("datasets", "ipfs_datasets_py"),
            ("kit", "ipfs_kit_py"),
        ):
            captured_gitlink = _git(
                "rev-parse",
                f"{tested_control_revision}:{nested_path}",
                cwd=REPO_ROOT,
            )
            if (
                captured_gitlink.returncode != 0
                or captured_gitlink.stdout.strip()
                != source_revisions.get(nested_repository)
            ):
                errors.append(
                    f"{label}.source_revisions.{nested_repository} does not match "
                    "the captured control gitlink"
                )
    repository_name = str(spec["repository"])
    if receipt.get("source_revision") != source_revisions.get(repository_name):
        errors.append(f"{label}.source_revision does not match its tested repository map")
    if receipt.get("source_tree") != source_trees.get(repository_name):
        errors.append(f"{label}.source_tree does not match its tested repository map")
    if receipt.get("execution_head") != source_revisions.get("accelerate"):
        errors.append(f"{label}.execution_head does not match the tested control repository")
    if receipt.get("execution_tree") != source_trees.get("accelerate"):
        errors.append(f"{label}.execution_tree does not match the tested control repository")
    expected_clean = {repository: True for repository in planning_revisions}
    if receipt.get("source_clean_before") != expected_clean:
        errors.append(f"{label}.source_clean_before is not an exact all-clean map")
    if receipt.get("source_clean_after") != expected_clean:
        errors.append(f"{label}.source_clean_after is not an exact all-clean map")
    _validate_ignored_sensitive_inputs(
        receipt.get("ignored_sensitive_inputs"),
        label,
        errors,
    )
    if receipt.get("git_environment_policy_id") != BASELINE_GIT_ENVIRONMENT_POLICY:
        errors.append(f"{label}.git_environment_policy_id is not reviewed")
    _parse_timestamp(receipt.get("captured_at"), f"{label}.captured_at", errors)
    if receipt.get("operator_origin") != BASELINE_OPERATOR_ORIGIN:
        errors.append(f"{label}.operator_origin must identify the operator capture")

    expected_ids = list(spec["command_ids"])
    reviewed_registry = _reviewed_suite_registry(errors)
    if receipt.get("required_command_ids") != expected_ids:
        errors.append(f"{label}.required_command_ids is not the fixed ordered command set")
    commands = receipt.get("commands")
    if not isinstance(commands, list):
        errors.append(f"{label}.commands must be an ordered array")
        commands = []
    actual_ids = [
        command.get("id") if isinstance(command, Mapping) else None
        for command in commands
    ]
    if actual_ids != expected_ids:
        errors.append(f"{label}.commands does not exactly cover the fixed ordered command set")
    historical_capture_root = _infer_historical_capture_root(
        commands,
        expected_ids=expected_ids,
        capture_id=capture_id,
        receipt_label=label,
        errors=errors,
    )
    if historical_capture_root is not None and bundle_capture_roots is not None:
        bundle_capture_roots[task_id] = historical_capture_root
    seen_logs: set[str] = set()
    for index, expected_id in enumerate(expected_ids):
        command = commands[index] if index < len(commands) else None
        if not isinstance(command, Mapping):
            errors.append(f"{label}.commands[{index}] must be an object")
            continue
        _validate_baseline_command(
            command,
            expected_id=expected_id,
            expected_cwd=str(spec["cwd"]),
            expected_timeout=int(spec["timeouts"][index]),
            expected_suite=reviewed_registry.get(expected_id, {}),
            capture_id=capture_id,
            historical_capture_root=historical_capture_root,
            receipt_label=label,
            seen_logs=seen_logs,
            errors=errors,
        )

    expected_assurance = {
        "process_observed": True,
        "test_execution_cryptographically_proven": False,
        "cryptographic_proof": False,
        "signature": None,
        "network_isolation_enforced": False,
        "offline_controls_requested": True,
        "pytest_plugin_allowlist_enforced": False,
        "public_log_witness_policy": "public-full-log-secret-scan@1",
        "inherited_secrets_forwarded": False,
        "remaining_trust": [
            "Host socket access is not isolated; fixed offline controls are requested and trusted.",
            "Installed pytest11 plugins are recorded but not allowlisted.",
            "Selected tests and subprocesses are trusted not to bypass fixed controls.",
        ],
        "claim": "Integrity-protected observations of fixed pytest subprocesses only.",
    }
    if receipt.get("assurance") != expected_assurance:
        errors.append(f"{label}.assurance overstates or changes the process-only claim")

    receipt_digest = receipt.get("receipt_digest")
    declared_digest = _sha256_value(receipt_digest, f"{label}.receipt_digest", errors)
    digest_preimage = dict(receipt)
    digest_preimage.pop("receipt_digest", None)
    actual_digest = hashlib.sha256(_canonical_json_bytes(digest_preimage)).hexdigest()
    if declared_digest is not None and declared_digest != actual_digest:
        errors.append(f"{label}.receipt_digest does not match canonical receipt content")
    return receipt


def _expected_receipt_pin(
    spec: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    commands = receipt.get("commands")
    if not isinstance(commands, list):
        commands = []
    return {
        "path": spec["receipt"],
        "receipt_digest": receipt.get("receipt_digest"),
        "planning_revision": spec["revision"],
        "source_revision": receipt.get("source_revision"),
        "source_tree": receipt.get("source_tree"),
        "required_command_ids": list(spec["command_ids"]),
        "suite_definition_digests": {
            command.get("id"): command.get("suite_definition_digest")
            for command in commands
            if isinstance(command, Mapping) and isinstance(command.get("id"), str)
        },
        "retained_log_paths": [
            command.get("log", {}).get("relative_path")
            for command in commands
            if isinstance(command, Mapping)
            and isinstance(command.get("log"), Mapping)
        ],
    }


def _validate_operator_baseline_bundle(
    config: Mapping[str, Any],
    errors: list[str],
    *,
    enforce_current_sources: bool,
) -> dict[str, Mapping[str, Any]]:
    """Validate the complete protected operator bundle before supervisor work."""

    configured = config.get("operator_baseline_receipts")
    if not isinstance(configured, Mapping) or set(configured) != set(
        BASELINE_RECEIPT_SPECS
    ):
        return {}
    receipts: dict[str, Mapping[str, Any]] = {}
    capture_roots: dict[str, PurePosixPath] = {}
    for task_id, spec in BASELINE_RECEIPT_SPECS.items():
        receipt = _validate_baseline_receipt(
            task_id,
            spec,
            errors,
            bundle_capture_roots=capture_roots,
        )
        receipts[task_id] = receipt
        if configured.get(task_id) != _expected_receipt_pin(spec, receipt):
            errors.append(f"{task_id} protected operator receipt pin is not exact")
    source_maps = [receipt.get("source_revisions") for receipt in receipts.values()]
    if not source_maps or any(source_map != source_maps[0] for source_map in source_maps[1:]):
        errors.append("operator baseline receipts do not share one exact source revision map")
    ignored_bindings = [
        receipt.get("ignored_sensitive_inputs") for receipt in receipts.values()
    ]
    if not ignored_bindings or any(
        binding != ignored_bindings[0] for binding in ignored_bindings[1:]
    ):
        errors.append("operator baseline receipts do not share one ignored-input binding")
    capture_ids = [receipt.get("capture_id") for receipt in receipts.values()]
    if not capture_ids or any(capture_id != capture_ids[0] for capture_id in capture_ids[1:]):
        errors.append("operator baseline receipts do not share one exact capture id")
    if set(capture_roots) != set(BASELINE_RECEIPT_SPECS) or len(
        {root.as_posix() for root in capture_roots.values()}
    ) != 1:
        errors.append(
            "operator baseline receipts do not share one canonical absolute "
            "historical capture root"
        )
    toolchain_contexts: list[list[tuple[Any, Any]]] = []
    for receipt in receipts.values():
        commands = receipt.get("commands")
        if not isinstance(commands, list):
            toolchain_contexts.append([])
            continue
        toolchain_contexts.append(
            [
                (command.get("python"), command.get("pytest"))
                for command in commands
                if isinstance(command, Mapping)
            ]
        )
    flattened_contexts = [
        context
        for repository_context in toolchain_contexts
        for context in repository_context
    ]
    if flattened_contexts and any(
        context != flattened_contexts[0] for context in flattened_contexts[1:]
    ):
        errors.append("operator baseline receipts do not share one exact pytest toolchain")
    if enforce_current_sources and receipts:
        _validate_current_baseline_sources(configured, receipts, errors)
    return receipts


def _validate_current_baseline_sources(
    configured: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    first = receipts.get("IPS-001") or next(iter(receipts.values()), {})
    control_captured = first.get("execution_head")
    control_current = _git_stdout(
        REPO_ROOT, errors, "resolve preflight control HEAD", "rev-parse", "HEAD"
    )
    if isinstance(control_captured, str) and _HEX_40.fullmatch(control_captured):
        _validate_accelerate_control_transition(
            task_id="preflight",
            captured_revision=control_captured,
            current_revision=control_current,
            configured_receipts=configured,
            errors=errors,
        )
    else:
        errors.append("preflight receipts lack an exact captured control revision")
    control_status = _git(
        "status", "--porcelain=v1", "--untracked-files=all", cwd=REPO_ROOT
    )
    if control_status.returncode != 0:
        errors.append("cannot inspect preflight control worktree")
    else:
        dirty_paths: set[str] = set()
        for line in control_status.stdout.splitlines():
            if len(line) < 4:
                continue
            relative = line[3:].split(" -> ", 1)[-1]
            dirty_paths.add(relative)
        unexpected_dirty = dirty_paths - {
            BASELINE_SYNTHESIS_JSON,
            BASELINE_SYNTHESIS_REPORT,
        }
        if unexpected_dirty:
            errors.append(
                "preflight control worktree contains relevance-changing dirty paths: "
                f"{sorted(unexpected_dirty)}"
            )

    source_revisions = first.get("source_revisions")
    if not isinstance(source_revisions, Mapping):
        return
    for repository, gitlink_path in (
        ("datasets", "ipfs_datasets_py"),
        ("kit", "ipfs_kit_py"),
    ):
        captured = source_revisions.get(repository)
        repository_root = REPO_ROOT / REPOSITORY_PATHS[repository]
        current = _git_stdout(
            repository_root,
            errors,
            f"resolve preflight {repository} HEAD",
            "rev-parse",
            "HEAD",
        )
        if not isinstance(captured, str) or not _HEX_40.fullmatch(captured):
            errors.append(f"preflight receipt lacks captured {repository} revision")
            continue
        ancestry = _git(
            "merge-base", "--is-ancestor", captured, current, cwd=repository_root
        )
        if ancestry.returncode != 0:
            errors.append(f"preflight {repository} HEAD does not descend from capture")
        changed = _git_stdout(
            repository_root,
            errors,
            f"inspect preflight {repository} inventory delta",
            "diff",
            "--name-only",
            "--no-renames",
            captured,
            current,
            "--",
        )
        allowed = NESTED_INVENTORY_OUTPUTS[gitlink_path]
        unexpected = {line for line in changed.splitlines() if line} - allowed
        if unexpected:
            errors.append(
                f"preflight {repository} delta contains relevance-changing paths: "
                f"{sorted(unexpected)}"
            )
        gitlink = _git_stdout(
            REPO_ROOT,
            errors,
            f"resolve preflight {repository} gitlink",
            "rev-parse",
            f"HEAD:{gitlink_path}",
        )
        if gitlink != current:
            errors.append(f"preflight {repository} HEAD does not equal current gitlink")


def _validate_historical_synthesis_transition(
    config: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    parent: str,
    errors: list[str],
) -> None:
    """Revalidate capture-to-IPS-004 history without requiring current HEAD equality."""

    configured = config.get("operator_baseline_receipts")
    if not isinstance(configured, Mapping):
        errors.append("IPS-004 historical transition lacks protected receipt pins")
        return
    first = receipts.get("IPS-001") or next(iter(receipts.values()), {})
    captured = first.get("execution_head")
    if not isinstance(captured, str) or not _HEX_40.fullmatch(captured):
        errors.append("IPS-004 historical transition lacks a captured control revision")
        return
    if _git("merge-base", "--is-ancestor", captured, parent).returncode != 0:
        errors.append("IPS-004 synthesis parent does not descend from capture")
        return
    _validate_accelerate_control_transition(
        task_id="IPS-004 historical synthesis",
        captured_revision=captured,
        current_revision=parent,
        configured_receipts=configured,
        errors=errors,
        enforce_current_nested=False,
    )
    evidence_paths: set[str] = set()
    for pin in configured.values():
        if not isinstance(pin, Mapping):
            continue
        path = pin.get("path")
        if isinstance(path, str):
            evidence_paths.add(path)
        logs = pin.get("retained_log_paths")
        if isinstance(logs, list):
            evidence_paths.update(path for path in logs if isinstance(path, str))
    for relative in sorted(evidence_paths):
        parent_object = _git_stdout(
            REPO_ROOT,
            errors,
            f"resolve IPS-004 parent evidence {relative}",
            "rev-parse",
            f"{parent}:{relative}",
        )
        current_object = _git_stdout(
            REPO_ROOT,
            errors,
            f"resolve current protected evidence {relative}",
            "rev-parse",
            f"HEAD:{relative}",
        )
        if parent_object and current_object and parent_object != current_object:
            errors.append(
                f"IPS-004 synthesis parent evidence differs from current protected {relative}"
            )


def _validated_baseline_synthesis(
    config: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    errors: list[str],
    *,
    candidate: bool = False,
) -> bool:
    """Validate IPS-004 as a task candidate or a committed lineage milestone."""

    matrix_path = REPO_ROOT / BASELINE_SYNTHESIS_JSON
    report_path = REPO_ROOT / BASELINE_SYNTHESIS_REPORT
    present = (os.path.lexists(matrix_path), os.path.lexists(report_path))
    if present == (False, False):
        if candidate:
            errors.append("IPS-004 candidate synthesis artifacts are missing")
        return False
    local: list[str] = []
    if not all(present):
        errors.append("IPS-004 baseline synthesis artifacts are only partially present")
        return False
    if not candidate:
        tracked = tuple(
            _git("ls-files", "--error-unmatch", "--", relative).returncode == 0
            for relative in (BASELINE_SYNTHESIS_JSON, BASELINE_SYNTHESIS_REPORT)
        )
        if tracked == (False, False):
            return False
        if not all(tracked):
            local.append("IPS-004 baseline synthesis artifacts are only partially tracked")
            errors.extend(local)
            return False
        dirty = _git(
            "status",
            "--porcelain=v1",
            "--",
            BASELINE_SYNTHESIS_JSON,
            BASELINE_SYNTHESIS_REPORT,
        )
        if dirty.returncode != 0 or dirty.stdout.strip():
            local.append("IPS-004 synthesis artifacts are not clean committed evidence")
            errors.extend(local)
            return False
    retained_matrix = _secure_read_repo_file(
        BASELINE_SYNTHESIS_JSON,
        required_parent=str(Path(BASELINE_SYNTHESIS_JSON).parent),
        label="IPS-004 trust baseline matrix",
        maximum_bytes=BASELINE_MAX_RECEIPT_BYTES,
        bound_label="two-MiB",
        errors=local,
    )
    retained_report = _secure_read_repo_file(
        BASELINE_SYNTHESIS_REPORT,
        required_parent=str(Path(BASELINE_SYNTHESIS_REPORT).parent),
        label="IPS-004 trust baseline report",
        maximum_bytes=BASELINE_MAX_RECEIPT_BYTES,
        bound_label="two-MiB",
        errors=local,
    )
    if retained_matrix is None or retained_report is None:
        errors.extend(local)
        return False
    raw_matrix, _ = retained_matrix
    raw_report, _ = retained_report
    try:
        matrix = json.loads(
            raw_matrix.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
        report = raw_report.decode("utf-8")
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        local.append(f"IPS-004 baseline synthesis is not canonical UTF-8 JSON/text: {exc}")
        errors.extend(local)
        return False
    if not isinstance(matrix, Mapping):
        local.append("IPS-004 trust baseline matrix must be one object")
        errors.extend(local)
        return False
    if raw_matrix != _canonical_json_bytes(matrix) + b"\n":
        local.append("IPS-004 trust baseline matrix is not canonical JSON")
    _closed_keys(
        matrix,
        (
            "schema_version",
            "synthesis_worktree_parent_revision",
            "baseline_receipts",
            "inventory_artifacts",
            "repository_authorities",
            "proof_class_decisions",
            "aggregation_decision",
            "backend_decisions",
            "trust_nonclaims",
        ),
        "IPS-004 trust baseline matrix",
        local,
    )
    if matrix.get("schema_version") != BASELINE_SYNTHESIS_SCHEMA:
        local.append("IPS-004 trust baseline matrix schema is not reviewed")
    configured = config.get("operator_baseline_receipts")
    if matrix.get("baseline_receipts") != configured:
        local.append("IPS-004 trust baseline does not bind the exact operator receipt pins")
    if matrix.get("repository_authorities") != TRUST_BASELINE_AUTHORITIES:
        local.append("IPS-004 repository authority decisions are not exact")
    if matrix.get("proof_class_decisions") != TRUST_BASELINE_PROOF_CLASS_DECISIONS:
        local.append("IPS-004 proof-class decisions are not exact")
    if matrix.get("aggregation_decision") != TRUST_BASELINE_AGGREGATION_DECISION:
        local.append("IPS-004 aggregation decision is not exact")
    if matrix.get("backend_decisions") != TRUST_BASELINE_BACKEND_DECISIONS:
        local.append("IPS-004 backend decisions are not exact")
    if matrix.get("trust_nonclaims") != list(TRUST_BASELINE_NONCLAIMS):
        local.append("IPS-004 trust nonclaims are not exact and ordered")
    parent = matrix.get("synthesis_worktree_parent_revision")
    if not isinstance(parent, str) or not _HEX_40.fullmatch(parent):
        local.append("IPS-004 synthesis_worktree_parent_revision is not a commit")
        parent = ""
    if parent and not candidate:
        _validate_historical_synthesis_transition(config, receipts, parent, local)
    inventory_artifacts = matrix.get("inventory_artifacts")
    if not isinstance(inventory_artifacts, Mapping) or set(inventory_artifacts) != set(
        BASELINE_RECEIPT_SPECS
    ):
        local.append("IPS-004 inventory_artifacts must exactly bind IPS-001/002/003")
        inventory_artifacts = {}
    for task_id, spec in BASELINE_RECEIPT_SPECS.items():
        item = inventory_artifacts.get(task_id)
        if not isinstance(item, Mapping):
            local.append(f"IPS-004 inventory_artifacts.{task_id} must be an object")
            continue
        _closed_keys(
            item,
            ("inventory", "report", "completion_revision"),
            f"IPS-004 inventory_artifacts.{task_id}",
            local,
        )
        if item.get("inventory") != spec["inventory"] or item.get("report") != spec["report"]:
            local.append(f"IPS-004 inventory_artifacts.{task_id} paths drifted")
        completion = item.get("completion_revision")
        if not isinstance(completion, str) or not _HEX_40.fullmatch(completion):
            local.append(f"IPS-004 inventory_artifacts.{task_id} completion is not a commit")
            continue
        repository = str(spec["repository"])
        repository_root = REPO_ROOT / REPOSITORY_PATHS[repository]
        relative_outputs: list[str] = []
        for field in ("inventory", "report"):
            relative = Path(str(spec[field]))
            if repository != "accelerate":
                relative = relative.relative_to(REPOSITORY_PATHS[repository])
            relative_outputs.append(relative.as_posix())
            if _git("cat-file", "-e", f"{completion}:{relative.as_posix()}", cwd=repository_root).returncode != 0:
                local.append(
                    f"IPS-004 {task_id} completion does not contain {relative.as_posix()}"
                )
            if parent:
                parent_reference = (
                    f"{parent}:{relative.as_posix()}"
                    if repository == "accelerate"
                    else f"{completion}:{relative.as_posix()}"
                )
                completion_object = _git_stdout(
                    repository_root,
                    local,
                    f"resolve IPS-004 {task_id} completion object {relative}",
                    "rev-parse",
                    f"{completion}:{relative.as_posix()}",
                )
                parent_object = _git_stdout(
                    repository_root,
                    local,
                    f"resolve IPS-004 {task_id} consumed object {relative}",
                    "rev-parse",
                    parent_reference,
                )
                if completion_object and parent_object and completion_object != parent_object:
                    local.append(
                        f"IPS-004 {task_id} consumed inventory differs from its named completion"
                    )
        completion_parent = _git_stdout(
            repository_root,
            local,
            f"resolve IPS-004 {task_id} completion parent",
            "rev-parse",
            f"{completion}^1",
        )
        changed = _git_stdout(
            repository_root,
            local,
            f"inspect IPS-004 {task_id} completion paths",
            "diff",
            "--name-only",
            "--no-renames",
            completion_parent,
            completion,
            "--",
        )
        if {line for line in changed.splitlines() if line} != set(relative_outputs):
            local.append(f"IPS-004 {task_id} completion is not inventory-output-only")
        if parent:
            if repository == "accelerate":
                if _git(
                    "merge-base",
                    "--is-ancestor",
                    completion,
                    parent,
                    cwd=repository_root,
                ).returncode != 0:
                    local.append("IPS-004 accelerate inventory completion is not in its parent")
            else:
                gitlink = _git_stdout(
                    REPO_ROOT,
                    local,
                    f"resolve IPS-004 {repository} inventory gitlink",
                    "rev-parse",
                    f"{parent}:{REPOSITORY_PATHS[repository].as_posix()}",
                )
                if gitlink != completion:
                    local.append(f"IPS-004 {repository} completion does not match its gitlink")
    if candidate:
        current_parent = _git_stdout(
            REPO_ROOT,
            local,
            "resolve IPS-004 candidate parent",
            "rev-parse",
            "HEAD",
        )
        if parent and current_parent != parent:
            local.append("IPS-004 candidate does not bind its current task parent")
    else:
        commits = [
            _git_stdout(
                REPO_ROOT,
                local,
                f"resolve committed {relative}",
                "log",
                "-1",
                "--format=%H",
                "--",
                relative,
            )
            for relative in (BASELINE_SYNTHESIS_JSON, BASELINE_SYNTHESIS_REPORT)
        ]
        if not commits[0] or commits[0] != commits[1]:
            local.append("IPS-004 synthesis artifacts were not committed together")
        elif parent:
            actual_parent = _git_stdout(
                REPO_ROOT,
                local,
                "resolve IPS-004 synthesis commit parent",
                "rev-parse",
                f"{commits[0]}^",
            )
            if actual_parent != parent:
                local.append("IPS-004 synthesis commit does not bind its declared parent")
            if _git("merge-base", "--is-ancestor", commits[0], "HEAD").returncode != 0:
                local.append("IPS-004 synthesis commit is not in the current lineage")
    for task_id, receipt in receipts.items():
        for term in (
            task_id,
            str(receipt.get("receipt_digest", "")),
            str(BASELINE_RECEIPT_SPECS[task_id]["receipt"]),
            "pytest_execution_not_cryptographically_proven",
        ):
            if term not in report:
                local.append(f"IPS-004 trust baseline report omits {term!r}")
    report_decisions: list[tuple[str, Any]] = [
        *TRUST_BASELINE_AUTHORITIES.items(),
        *TRUST_BASELINE_PROOF_CLASS_DECISIONS.items(),
        *TRUST_BASELINE_AGGREGATION_DECISION.items(),
        *TRUST_BASELINE_BACKEND_DECISIONS.items(),
    ]
    for key, value in report_decisions:
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        if key not in report or rendered not in report:
            local.append(
                f"IPS-004 trust baseline report omits decision {key}={rendered}"
            )
    for nonclaim in TRUST_BASELINE_NONCLAIMS:
        if nonclaim not in report:
            local.append(f"IPS-004 trust baseline report omits nonclaim {nonclaim!r}")
    errors.extend(local)
    return not local


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _normalized_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


_COPIED_EVIDENCE_ALIASES = frozenset(
    {
        "argv",
        "baselinecommand",
        "baselinecommands",
        "baselinecounts",
        "baselinelog",
        "baselineresult",
        "baselineresults",
        "collectedcount",
        "commandargv",
        "commanddigest",
        "commandline",
        "durationns",
        "executedargv",
        "executedcommand",
        "exitcode",
        "failedcount",
        "failurecount",
        "logbytes",
        "logdigest",
        "logsha256",
        "nonpassnodes",
        "observedcounts",
        "outcomecounts",
        "passcount",
        "passedcount",
        "pytestcommand",
        "pytestcounts",
        "pytestresult",
        "pytestresults",
        "resultcounts",
        "retainedlog",
        "retainedtranscript",
        "selectedcount",
        "skipcount",
        "skippedcount",
        "stderr",
        "stdout",
        "summaryline",
        "testcounts",
        "testresult",
        "testresults",
        "transcript",
        "xfailedcount",
        "xpassedcount",
    }
)


def _validate_reference_only_inventory_namespace(
    task_id: str,
    payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    """Reject provider-owned shadows of protected operator execution evidence."""

    execution_sentence = re.compile(
        r"(?:\b\d+\s+(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b|"
        r"\b(?:pytest|tests?|test\s+cases|execution|suite|run)\b.{0,48}?\b"
        r"(?:passed|failed|succeeded|successful|verified|green|red|ok)\b|"
        r"\b(?:passed|failed|succeeded|successful|verified|green|red|ok)\b"
        r".{0,32}?\b(?:pytest|tests?|cases?|execution|suite|run)\b)",
        re.IGNORECASE,
    )
    claim_values = {
        "pass",
        "passed",
        "success",
        "successful",
        "succeeded",
        "green",
        "verified",
    }
    claim_field_terms = ("execution", "outcome", "result", "run", "status", "test", "verification")

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = _normalized_field_name(key)
                child_path = (*path, key)
                normalized_path = _normalized_field_name(".".join(child_path))
                if path == () and key == "baseline_evidence":
                    continue
                if normalized in _COPIED_EVIDENCE_ALIASES:
                    errors.append(
                        f"{task_id} inventory copies protected evidence through "
                        f"{'.'.join(child_path)!r}"
                    )
                if (
                    isinstance(child, (bool, int, float))
                    and any(
                        term in normalized
                        or term in normalized_path
                        for term in ("execution", "test", "pytest", "case", "suite", "run")
                    )
                    and any(
                        term in normalized
                        or term in normalized_path
                        for term in (
                            "fail",
                            "green",
                            "ok",
                            "outcome",
                            "pass",
                            "red",
                            "result",
                            "status",
                            "success",
                        )
                    )
                ):
                    errors.append(
                        f"{task_id} inventory asserts a provider-owned numeric execution "
                        f"outcome at {'.'.join(child_path)!r}"
                    )
                if (
                    isinstance(child, bool)
                    and child
                    and any(term in normalized for term in ("passed", "succeeded", "successful"))
                ):
                    errors.append(
                        f"{task_id} inventory asserts provider-owned execution success at "
                        f"{'.'.join(child_path)!r}"
                    )
                if isinstance(child, str):
                    folded = child.strip().casefold()
                    if execution_sentence.search(child):
                        errors.append(
                            f"{task_id} inventory asserts provider-owned execution outcomes at "
                            f"{'.'.join(child_path)!r}"
                        )
                    if folded in claim_values and any(
                        term in normalized_path for term in claim_field_terms
                    ):
                        errors.append(
                            f"{task_id} inventory asserts provider-owned success status at "
                            f"{'.'.join(child_path)!r}"
                        )
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
        elif isinstance(value, str) and execution_sentence.search(value):
            errors.append(
                f"{task_id} inventory contains provider-owned execution outcome prose at "
                f"{'.'.join(path)!r}"
            )

    visit(payload, ())


def _validate_inventory_provenance(
    task_id: str,
    payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    forbidden_tokens = (
        "pending-local-run",
        "plan-derived",
        "plan_derived",
        "copied-from-plan",
        "copied_from_plan",
        "placeholder-result",
        "placeholder_result",
        "synthetic-success",
        "synthetic_success",
        "declared-for-rerun",
        "declared_for_rerun",
    )
    helper_keys = {
        "helper_output",
        "helper_outputs",
        "baseline_helper_output",
        "baseline_helper_outputs",
        "generated_baseline_helper",
        "undeclared_outputs",
    }
    origin_keys = {
        "basis",
        "classification_method",
        "evidence_basis",
        "evidence_origin",
        "evidence_source",
        "measurement_provenance",
        "method",
        "inspection_mode",
        "origin",
        "provenance",
    }
    result_keys = {
        "execution_status",
        "outcome",
        "result",
        "status",
        "terminal_status",
        "verification_status",
    }
    for mapping in _walk_mappings(payload):
        normalized_keys = {str(key).casefold() for key in mapping}
        copied_outcomes = normalized_keys & set(BASELINE_OUTCOME_FIELDS)
        if copied_outcomes:
            errors.append(
                f"{task_id} inventory copies protected outcome fields: "
                f"{sorted(copied_outcomes)}"
            )
        undeclared = normalized_keys & helper_keys
        if undeclared:
            errors.append(f"{task_id} inventory declares forbidden helper outputs: {sorted(undeclared)}")
        for key, value in mapping.items():
            if not isinstance(value, str):
                continue
            lowered = value.casefold()
            if any(token in lowered for token in forbidden_tokens):
                errors.append(f"{task_id} inventory contains non-executable placeholder provenance")
                break
            key_name = str(key).casefold()
            if key_name.endswith(("sha256", "_hash", "_digest")):
                possible = value[7:] if value.startswith("sha256:") else value
                if _HEX_64.fullmatch(possible) and _looks_patterned_digest(possible):
                    errors.append(f"{task_id} inventory contains a patterned fake hash")
                if key_name in {
                    "canonical_sha256",
                    "log_digest",
                    "output_digest",
                    "receipt_digest",
                    "sha256",
                } and _sha256_value(value, f"{task_id} inventory {key_name}", errors) is None:
                    break
        origins = " ".join(
            str(mapping[key]).casefold()
            for key in mapping
            if str(key).casefold() in origin_keys
        )
        results = " ".join(
            str(mapping[key]).casefold()
            for key in mapping
            if str(key).casefold() in result_keys
        )
        all_text = " ".join(str(item).casefold() for item in mapping.values())
        outcome_claim = re.search(
            r"\b(?:\d+\s+)?(?:pass(?:ed)?|fail(?:ed)?|errors?|success|verified|valid)\b",
            f"{results} {all_text}",
        )
        if "static" in origins and outcome_claim:
            errors.append(f"{task_id} inventory treats static inspection as an execution result")
        if "plan" in origins and outcome_claim:
            errors.append(f"{task_id} inventory treats plan-derived prose as execution evidence")
        if outcome_claim and any(
            isinstance(item, str)
            and item.casefold() in {"plan", "planning_document", "planning_prose"}
            for item in mapping.values()
        ):
            errors.append(f"{task_id} inventory treats plan-derived prose as execution evidence")


def _nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close_number(actual: Any, expected: float, *, tolerance: float = 1e-9) -> bool:
    return _nonnegative_number(actual) and abs(float(actual) - expected) <= tolerance * max(
        1.0, abs(expected)
    )


def _close_signed_number(actual: Any, expected: float, *, tolerance: float = 1e-9) -> bool:
    return _finite_number(actual) and abs(float(actual) - expected) <= tolerance * max(
        1.0, abs(expected)
    )


def _runner_declared_paths(runner: str) -> frozenset[str]:
    if runner == "benchmark":
        return frozenset((BENCHMARK_JSON, BENCHMARK_CSV))
    if runner == "release":
        return frozenset(
            (RELEASE_REPORT, RELEASE_VALIDATION_JSON, RELEASE_VALIDATION_LOG)
        )
    raise ValueError(f"unknown protected runner {runner!r}")


def _runner_path_is_transient(relative: str) -> bool:
    return relative == RELEASE_WORK_ROOT or relative.startswith(
        f"{RELEASE_WORK_ROOT}/"
    )


def _runner_path_is_allowed(
    relative: str, declared_paths: frozenset[str]
) -> bool:
    return relative in declared_paths or _runner_path_is_transient(relative)


def _decode_git_path(raw: bytes, label: str, errors: list[str]) -> str | None:
    relative = os.fsdecode(raw)
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or relative != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        errors.append(f"{label} contains an unsafe repository path {relative!r}")
        return None
    return relative


def _parse_runner_tree(
    raw: bytes, repository: str, errors: list[str]
) -> dict[str, tuple[str, str, str]]:
    entries: dict[str, tuple[str, str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_kind, raw_oid = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii")
            kind = raw_kind.decode("ascii")
            oid = raw_oid.decode("ascii")
        except (UnicodeError, ValueError):
            errors.append(f"protected runner cannot parse {repository} HEAD tree")
            continue
        relative = _decode_git_path(
            raw_path, f"protected runner {repository} HEAD tree", errors
        )
        if relative is None:
            continue
        if relative in entries:
            errors.append(
                f"protected runner {repository} HEAD tree repeats {relative!r}"
            )
            continue
        if (mode, kind) not in {
            ("100644", "blob"),
            ("100755", "blob"),
            ("120000", "blob"),
            ("160000", "commit"),
        } or not re.fullmatch(r"[0-9a-f]{40,64}", oid):
            errors.append(
                f"protected runner {repository} HEAD tree has an unsafe entry "
                f"for {relative!r}"
            )
            continue
        entries[relative] = (mode, kind, oid)
    return entries


def _validate_unmaterialized_gitlinks(
    repository: str,
    entries: Mapping[str, tuple[str, str, str]],
    runner: str,
    errors: list[str],
) -> None:
    """Bind every deliberately absent gitlink to one reviewed path and OID."""

    expected = dict(RUNNER_UNMATERIALIZED_GITLINKS.get(repository, {}))
    materialized_paths = (
        {
            relative.as_posix()
            for name, relative in REPOSITORY_PATHS.items()
            if name != "accelerate"
        }
        if repository == "accelerate"
        else set()
    )
    observed = {
        relative: oid
        for relative, (mode, kind, oid) in entries.items()
        if (mode, kind) == ("160000", "commit")
        and relative not in materialized_paths
    }
    if observed == expected:
        return
    missing = sorted(set(expected) - set(observed))
    unknown = sorted(set(observed) - set(expected))
    drifted = sorted(
        relative
        for relative in set(expected) & set(observed)
        if expected[relative] != observed[relative]
    )
    errors.append(
        f"protected {runner} {repository} deliberately unmaterialized gitlink "
        "allowlist drifted: "
        f"missing={missing[:12]}, unknown={unknown[:12]}, oid_drift={drifted[:12]}"
    )


def _parse_runner_index(
    raw: bytes, repository: str, errors: list[str]
) -> dict[str, list[tuple[str, str, int]]]:
    entries: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_oid, raw_stage = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii")
            oid = raw_oid.decode("ascii")
            stage = int(raw_stage.decode("ascii"))
        except (UnicodeError, ValueError):
            errors.append(f"protected runner cannot parse {repository} index")
            continue
        relative = _decode_git_path(
            raw_path, f"protected runner {repository} index", errors
        )
        if relative is None:
            continue
        if mode not in {"100644", "100755", "120000", "160000"} or not re.fullmatch(
            r"[0-9a-f]{40,64}", oid
        ):
            errors.append(
                f"protected runner {repository} index has an unsafe entry "
                f"for {relative!r}"
            )
            continue
        entries[relative].append((mode, oid, stage))
    return dict(entries)


def _parse_runner_index_tags(
    raw: bytes, repository: str, errors: list[str]
) -> dict[str, str]:
    tags: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            errors.append(f"protected runner cannot parse {repository} index flags")
            continue
        try:
            tag = record[:1].decode("ascii")
        except UnicodeError:
            errors.append(f"protected runner cannot parse {repository} index flags")
            continue
        relative = _decode_git_path(
            record[2:], f"protected runner {repository} index flags", errors
        )
        if relative is None:
            continue
        if relative in tags:
            errors.append(
                f"protected runner {repository} index flags repeat {relative!r}"
            )
        tags[relative] = tag
    return tags


def _parse_runner_status(
    raw: bytes, repository: str, errors: list[str]
) -> list[tuple[str, str]]:
    records = raw.split(b"\0")
    parsed: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            errors.append(f"protected runner cannot parse {repository} worktree status")
            continue
        try:
            status_code = record[:2].decode("ascii")
        except UnicodeError:
            errors.append(f"protected runner cannot parse {repository} worktree status")
            continue
        relative = _decode_git_path(
            record[3:].rstrip(b"/"),
            f"protected runner {repository} worktree status",
            errors,
        )
        if relative is not None:
            parsed.append((status_code, relative))
        if any(flag in status_code for flag in "RC"):
            if index >= len(records) or not records[index]:
                errors.append(
                    f"protected runner cannot parse {repository} rename/copy status"
                )
            else:
                original = _decode_git_path(
                    records[index].rstrip(b"/"),
                    f"protected runner {repository} rename/copy source",
                    errors,
                )
                if original is not None:
                    parsed.append(("rename/copy-source", original))
                index += 1
    return parsed


def _validate_runner_mutable_paths(
    root: Path,
    runner: str,
    declared_paths: frozenset[str],
    errors: list[str],
) -> None:
    """Reject link/special-file substitution at every runner-writable boundary."""

    checked_parents: set[Path] = set()
    for relative in sorted((*declared_paths, RELEASE_WORK_ROOT)):
        path = root / relative
        current = root
        for part in Path(relative).parts[:-1]:
            current /= part
            if current in checked_parents or not os.path.lexists(current):
                continue
            checked_parents.add(current)
            try:
                info = current.lstat()
            except OSError as exc:
                errors.append(
                    f"protected {runner} runner cannot inspect mutable-path parent "
                    f"{current.relative_to(root).as_posix()!r}: {type(exc).__name__}"
                )
                continue
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                errors.append(
                    f"protected {runner} runner mutable-path parent is not a regular "
                    f"directory: {current.relative_to(root).as_posix()!r}"
                )
        if not os.path.lexists(path):
            continue
        try:
            info = path.lstat()
        except OSError as exc:
            errors.append(
                f"protected {runner} runner cannot inspect mutable path {relative!r}: "
                f"{type(exc).__name__}"
            )
            continue
        if relative == RELEASE_WORK_ROOT:
            safe = stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            expected = "regular non-symlink directory"
        else:
            safe = stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            expected = "regular non-symlink file"
        if not safe:
            errors.append(
                f"protected {runner} runner mutable path {relative!r} must be a "
                f"{expected}"
            )


def _runner_repository_snapshot(
    *,
    repository: str,
    root: Path,
    runner: str,
    declared_paths: frozenset[str],
    errors: list[str],
) -> dict[str, Any]:
    """Bind one repository's HEAD, index flags/entries, and physical worktree."""

    try:
        root_info = root.lstat()
    except OSError as exc:
        errors.append(
            f"protected {runner} runner cannot inspect {repository} repository root: "
            f"{type(exc).__name__}"
        )
        return {}
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        errors.append(
            f"protected {runner} runner {repository} repository root is not a "
            "regular non-symlink directory"
        )
        return {}
    _reject_git_replacement_state(
        root,
        label=f"protected {runner} {repository} repository",
        errors=errors,
    )
    revision_before = _git_stdout(
        root,
        errors,
        f"resolve protected {runner} {repository} HEAD before worktree binding",
        "rev-parse",
        "HEAD",
    )
    tree_before = _git_stdout(
        root,
        errors,
        f"resolve protected {runner} {repository} tree before worktree binding",
        "rev-parse",
        "HEAD^{tree}",
    )
    tree_raw = _git_bytes(
        root,
        errors,
        f"read protected {runner} {repository} HEAD tree",
        "ls-tree",
        "-rz",
        "--full-tree",
        "HEAD",
    )
    index_raw = _git_bytes(
        root,
        errors,
        f"read protected {runner} {repository} index",
        "ls-files",
        "--stage",
        "-z",
        "--",
    )
    tags_raw = _git_bytes(
        root,
        errors,
        f"read protected {runner} {repository} index flags",
        "ls-files",
        "-v",
        "-z",
        "--",
    )
    status_raw = _git_bytes(
        root,
        errors,
        f"inspect protected {runner} {repository} worktree",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--ignore-submodules=none",
    )
    if None in (tree_raw, index_raw, tags_raw, status_raw):
        return {}
    assert tree_raw is not None
    assert index_raw is not None
    assert tags_raw is not None
    assert status_raw is not None
    tree_entries = _parse_runner_tree(tree_raw, repository, errors)
    index_entries = _parse_runner_index(index_raw, repository, errors)
    index_tags = _parse_runner_index_tags(tags_raw, repository, errors)
    _validate_unmaterialized_gitlinks(
        repository, tree_entries, runner, errors
    )

    for relative in sorted(set(tree_entries) | set(index_entries)):
        if _runner_path_is_transient(relative):
            errors.append(
                f"protected {runner} runner fixed transient root contains tracked "
                f"or staged path: {relative!r}"
            )

    bound_tree = {
        relative: entry
        for relative, entry in tree_entries.items()
        if relative not in declared_paths
    }
    bound_index: dict[str, tuple[str, str, int]] = {}
    for relative, entries in index_entries.items():
        if relative in declared_paths:
            if len(entries) != 1 or entries[0][2] != 0 or entries[0][0] not in {
                "100644",
                "100755",
            }:
                errors.append(
                    f"protected {runner} runner declared path has an unsafe index "
                    f"entry: {relative!r}"
                )
            continue
        if len(entries) != 1 or entries[0][2] != 0:
            errors.append(
                f"protected {runner} runner {repository} index has unresolved "
                f"stages for {relative!r}"
            )
            continue
        bound_index[relative] = entries[0]
    projected_tree = {
        relative: (mode, oid, 0)
        for relative, (mode, _kind, oid) in bound_tree.items()
    }
    if projected_tree != bound_index:
        changed = sorted(set(projected_tree) ^ set(bound_index))
        changed.extend(
            relative
            for relative in sorted(set(projected_tree) & set(bound_index))
            if projected_tree[relative] != bound_index[relative]
        )
        errors.append(
            f"protected {runner} runner {repository} index differs from HEAD "
            f"outside declared outputs: {changed[:12]}"
        )
    for relative in sorted(bound_index):
        tag = index_tags.get(relative)
        if tag != "H":
            errors.append(
                f"protected {runner} runner {repository} index flag for "
                f"{relative!r} is {tag!r}, not ordinary tracked state"
            )

    unexpected_status: list[str] = []
    for status_code, relative in _parse_runner_status(
        status_raw, repository, errors
    ):
        allowed = repository == "accelerate" and _runner_path_is_allowed(
            relative, declared_paths
        )
        if not allowed:
            unexpected_status.append(f"{status_code} {relative}")
    if unexpected_status:
        errors.append(
            f"protected {runner} runner {repository} worktree has staged, unstaged, "
            "untracked, or ignored execution-relevant mutations: "
            f"{unexpected_status[:12]}"
        )

    revision_after = _git_stdout(
        root,
        errors,
        f"resolve protected {runner} {repository} HEAD after worktree binding",
        "rev-parse",
        "HEAD",
    )
    tree_after = _git_stdout(
        root,
        errors,
        f"resolve protected {runner} {repository} tree after worktree binding",
        "rev-parse",
        "HEAD^{tree}",
    )
    index_after = _git_bytes(
        root,
        errors,
        f"re-read protected {runner} {repository} index",
        "ls-files",
        "--stage",
        "-z",
        "--",
    )
    status_after = _git_bytes(
        root,
        errors,
        f"re-inspect protected {runner} {repository} worktree",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--ignore-submodules=none",
    )
    if revision_before != revision_after or tree_before != tree_after:
        errors.append(
            f"protected {runner} runner {repository} HEAD/tree changed during "
            "worktree binding"
        )
    if index_after is None or index_after != index_raw:
        errors.append(
            f"protected {runner} runner {repository} index changed during "
            "worktree binding"
        )
    if status_after is None:
        return {}
    unexpected_after = [
        (status_code, relative)
        for status_code, relative in _parse_runner_status(
            status_after, repository, errors
        )
        if not (
            repository == "accelerate"
            and _runner_path_is_allowed(relative, declared_paths)
        )
    ]
    if unexpected_after:
        errors.append(
            f"protected {runner} runner {repository} worktree changed during "
            f"content binding: {unexpected_after[:12]}"
        )

    content_digest = hashlib.sha256()
    content_digest.update(RUNNER_SOURCE_BINDING_POLICY.encode("ascii"))
    content_digest.update(b"\0")
    content_digest.update(repository.encode("utf-8"))
    content_digest.update(b"\0")
    content_digest.update(revision_before.encode("ascii", "replace"))
    content_digest.update(b"\0")
    content_digest.update(tree_before.encode("ascii", "replace"))
    for relative, (mode, _kind, oid) in sorted(bound_tree.items()):
        content_digest.update(b"\0")
        content_digest.update(os.fsencode(relative))
        content_digest.update(b"\0")
        content_digest.update(mode.encode("ascii"))
        content_digest.update(b"\0")
        content_digest.update(oid.encode("ascii"))
    return {
        "revision": revision_before,
        "tree": tree_before,
        "content_digest": "sha256:" + content_digest.hexdigest(),
        "tree_entries": tree_entries,
    }


def _capture_runner_source_binding(
    runner: str, errors: list[str]
) -> dict[str, dict[str, str]]:
    """Fail closed unless every executable checkout is the exact indexed HEAD tree."""

    initial_error_count = len(errors)
    try:
        declared_paths = _runner_declared_paths(runner)
    except ValueError as exc:
        errors.append(str(exc))
        return {}
    _validate_runner_mutable_paths(REPO_ROOT, runner, declared_paths, errors)
    internal: dict[str, dict[str, Any]] = {}
    for repository, relative_root in REPOSITORY_PATHS.items():
        internal[repository] = _runner_repository_snapshot(
            repository=repository,
            root=REPO_ROOT / relative_root,
            runner=runner,
            declared_paths=(declared_paths if repository == "accelerate" else frozenset()),
            errors=errors,
        )
    outer_entries = internal.get("accelerate", {}).get("tree_entries")
    if isinstance(outer_entries, Mapping):
        for repository, relative_root in REPOSITORY_PATHS.items():
            if repository == "accelerate":
                continue
            relative = relative_root.as_posix()
            expected = outer_entries.get(relative)
            nested_revision = internal.get(repository, {}).get("revision")
            if expected != ("160000", "commit", nested_revision):
                errors.append(
                    f"protected {runner} runner nested {repository} HEAD does not "
                    f"equal outer gitlink {relative!r}"
                )
    public: dict[str, dict[str, str]] = {}
    for repository, snapshot in internal.items():
        revision = snapshot.get("revision")
        tree = snapshot.get("tree")
        digest = snapshot.get("content_digest")
        if all(isinstance(value, str) and value for value in (revision, tree, digest)):
            public[repository] = {
                "revision": revision,
                "tree": tree,
                "content_digest": digest,
            }
    if set(public) != set(REPOSITORY_PATHS):
        errors.append(f"protected {runner} runner source binding is incomplete")
    if len(errors) != initial_error_count:
        return {}
    return public


def _verify_runner_source_binding(
    runner: str,
    expected: Mapping[str, Mapping[str, str]],
    errors: list[str],
) -> None:
    observed = _capture_runner_source_binding(runner, errors)
    if observed != expected:
        errors.append(
            f"protected {runner} runner source/index/worktree binding changed "
            "during execution"
        )


def _run_materialization_git(
    runner: str,
    cwd: Path,
    errors: list[str],
    *args: str,
) -> bool:
    status, exit_code, _duration, output = _run_observed_process(
        ["git", *args],
        cwd=cwd,
        environment=_fixed_git_environment(),
        timeout_seconds=600,
        maximum_output_bytes=256 * 1024,
    )
    if status != "completed" or exit_code != 0:
        detail = output.decode("utf-8", "replace").strip()
        errors.append(
            "protected "
            f"{runner} source materialization failed: "
            f"{(detail or f'{status}/{exit_code}')[:2000]}"
        )
        return False
    return True


def _remove_materialized_gitlink_placeholders(
    root: Path,
    entries: Mapping[str, tuple[str, str, str]],
    runner: str,
    errors: list[str],
) -> None:
    for relative, (mode, kind, _oid) in sorted(
        entries.items(), key=lambda item: len(Path(item[0]).parts), reverse=True
    ):
        if (mode, kind) != ("160000", "commit"):
            continue
        target = root / relative
        if not os.path.lexists(target):
            continue
        try:
            info = target.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                errors.append(
                    f"protected {runner} materialized gitlink is not an empty "
                    f"directory: {relative!r}"
                )
                continue
            target.rmdir()
        except OSError as exc:
            errors.append(
                f"protected {runner} materialized gitlink is not empty: "
                f"{relative!r} ({type(exc).__name__})"
            )
            continue
        current = target.parent
        while current != root:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def _materialized_repository_content(
    *,
    root: Path,
    repository: str,
    runner: str,
    expected_binding: Mapping[str, str],
    nested_bindings: Mapping[str, Mapping[str, str]],
    errors: list[str],
) -> tuple[str, dict[str, tuple[str, str, int, bool]]]:
    """Hash every physical tracked leaf and compare it to the exact Git blob."""

    _reject_git_replacement_state(
        root,
        label=f"protected {runner} materialized {repository} repository",
        errors=errors,
    )
    revision = _git_stdout(
        root, errors, f"resolve materialized {repository} revision", "rev-parse", "HEAD"
    )
    tree = _git_stdout(
        root, errors, f"resolve materialized {repository} tree", "rev-parse", "HEAD^{tree}"
    )
    if revision != expected_binding.get("revision") or tree != expected_binding.get(
        "tree"
    ):
        errors.append(
            f"protected {runner} materialized {repository} revision/tree drifted"
        )
    remotes = _git_stdout(
        root, errors, f"inspect materialized {repository} remotes", "remote"
    )
    if remotes:
        errors.append(
            f"protected {runner} materialized {repository} repository retains a remote"
        )
    git_directory_text = _git_stdout(
        root,
        errors,
        f"resolve materialized {repository} Git directory",
        "rev-parse",
        "--absolute-git-dir",
    )
    git_directory = Path(git_directory_text) if git_directory_text else Path("/")
    try:
        if not git_directory.resolve().is_relative_to(root.resolve()):
            errors.append(
                f"protected {runner} materialized {repository} Git directory is external"
            )
    except OSError:
        errors.append(
            f"protected {runner} cannot resolve materialized {repository} Git directory"
        )
    alternates = git_directory / "objects" / "info" / "alternates"
    if os.path.lexists(alternates):
        errors.append(
            f"protected {runner} materialized {repository} uses object alternates"
        )
    object_format = _git_stdout(
        root,
        errors,
        f"resolve materialized {repository} object format",
        "rev-parse",
        "--show-object-format",
    )
    if object_format not in {"sha1", "sha256"}:
        errors.append(
            f"protected {runner} materialized {repository} object format is unsupported"
        )
        object_format = "sha1"
    tree_raw = _git_bytes(
        root,
        errors,
        f"enumerate materialized {repository} HEAD tree",
        "ls-tree",
        "-rz",
        "--full-tree",
        "HEAD",
    )
    index_raw = _git_bytes(
        root,
        errors,
        f"enumerate materialized {repository} index",
        "ls-files",
        "--stage",
        "-z",
        "--",
    )
    if tree_raw is None or index_raw is None:
        return "", {}
    tree_entries = _parse_runner_tree(tree_raw, repository, errors)
    index_entries = _parse_runner_index(index_raw, repository, errors)
    _validate_unmaterialized_gitlinks(
        repository, tree_entries, runner, errors
    )
    projected_index = {
        path: [(mode, oid, 0)]
        for path, (mode, _kind, oid) in tree_entries.items()
    }
    if index_entries != projected_index:
        errors.append(
            f"protected {runner} materialized {repository} index differs from HEAD"
        )

    reviewed_nested_paths = {
        REPOSITORY_PATHS[name].as_posix(): binding.get("revision")
        for name, binding in nested_bindings.items()
    }
    expected_blobs: dict[str, tuple[str, str]] = {}
    for relative, (mode, kind, oid) in tree_entries.items():
        if (mode, kind) == ("160000", "commit"):
            reviewed_revision = reviewed_nested_paths.get(relative)
            if reviewed_revision is not None and oid != reviewed_revision:
                errors.append(
                    f"protected {runner} materialized nested gitlink {relative!r} "
                    "does not match its bound revision"
                )
            elif reviewed_revision is None and os.path.lexists(root / relative):
                errors.append(
                    f"protected {runner} unmaterialized gitlink exists on disk: "
                    f"{relative!r}"
                )
            continue
        expected_blobs[relative] = (mode, oid)

    nested_roots = set(reviewed_nested_paths)
    actual: dict[str, tuple[str, str, int, bool]] = {}
    actual_directories: set[str] = set()
    stack: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    visited = 0
    hashed_bytes = 0
    while stack:
        directory, parent_parts = stack.pop()
        if len(parent_parts) > RUNNER_MATERIALIZATION_MAX_DEPTH:
            errors.append(
                f"protected {runner} materialized {repository} scan exceeded depth"
            )
            break
        try:
            with os.scandir(directory) as stream:
                children = sorted(stream, key=lambda child: child.name)
        except OSError as exc:
            errors.append(
                f"protected {runner} cannot scan materialized {repository}: "
                f"{type(exc).__name__}"
            )
            break
        for child in children:
            visited += 1
            if visited > RUNNER_MATERIALIZATION_MAX_ENTRIES:
                errors.append(
                    f"protected {runner} materialized {repository} scan exceeded entries"
                )
                stack.clear()
                break
            parts = (*parent_parts, child.name)
            relative = PurePosixPath(*parts).as_posix()
            if child.name == ".git" and not parent_parts:
                continue
            if repository == "accelerate" and relative in nested_roots:
                continue
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append(
                    f"protected {runner} cannot inspect materialized leaf "
                    f"{relative!r}: {type(exc).__name__}"
                )
                continue
            if stat.S_ISDIR(info.st_mode):
                actual_directories.add(relative)
                stack.append((Path(child.path), parts))
                continue
            if stat.S_ISREG(info.st_mode):
                if info.st_size > RUNNER_MATERIALIZATION_MAX_LEAF_BYTES:
                    errors.append(
                        f"protected {runner} materialized leaf exceeds hash bound: "
                        f"{relative!r}"
                    )
                    continue
                hashed_bytes += info.st_size
                if hashed_bytes > RUNNER_MATERIALIZATION_MAX_HASH_BYTES:
                    errors.append(
                        f"protected {runner} materialized {repository} hash budget exceeded"
                    )
                    stack.clear()
                    break
                identity = (
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
                sha_digest = hashlib.sha256()
                git_digest = hashlib.new(object_format)
                git_digest.update(f"blob {info.st_size}\0".encode("ascii"))
                descriptor = -1
                try:
                    descriptor = os.open(
                        child.path,
                        os.O_RDONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                    )
                    opened = os.fstat(descriptor)
                    if identity != (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_mode,
                        opened.st_size,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    ):
                        raise OSError("leaf changed before hashing")
                    while True:
                        chunk = os.read(descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        sha_digest.update(chunk)
                        git_digest.update(chunk)
                    finished = os.fstat(descriptor)
                    if identity != (
                        finished.st_dev,
                        finished.st_ino,
                        finished.st_mode,
                        finished.st_size,
                        finished.st_mtime_ns,
                        finished.st_ctime_ns,
                    ):
                        raise OSError("leaf changed while hashing")
                except OSError as exc:
                    errors.append(
                        f"protected {runner} cannot stably hash materialized leaf "
                        f"{relative!r}: {type(exc).__name__}"
                    )
                    continue
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
                executable = bool(info.st_mode & 0o111)
                actual[relative] = (
                    "regular",
                    sha_digest.hexdigest(),
                    info.st_size,
                    executable,
                )
                expected = expected_blobs.get(relative)
                expected_mode = "100755" if executable else "100644"
                if expected != (expected_mode, git_digest.hexdigest()):
                    errors.append(
                        f"protected {runner} materialized leaf differs from exact "
                        f"Git blob: {relative!r}"
                    )
                continue
            if stat.S_ISLNK(info.st_mode):
                try:
                    target = os.readlink(child.path)
                    target_bytes = os.fsencode(target)
                except OSError as exc:
                    errors.append(
                        f"protected {runner} cannot read materialized symlink "
                        f"{relative!r}: {type(exc).__name__}"
                    )
                    continue
                git_digest = hashlib.new(object_format)
                git_digest.update(f"blob {len(target_bytes)}\0".encode("ascii"))
                git_digest.update(target_bytes)
                sha_digest = hashlib.sha256(target_bytes).hexdigest()
                actual[relative] = (
                    "symlink",
                    sha_digest,
                    len(target_bytes),
                    False,
                )
                if expected_blobs.get(relative) != (
                    "120000",
                    git_digest.hexdigest(),
                ):
                    errors.append(
                        f"protected {runner} materialized symlink differs from exact "
                        f"Git blob: {relative!r}"
                    )
                continue
            errors.append(
                f"protected {runner} materialized tree contains special file: "
                f"{relative!r}"
            )

    if set(actual) != set(expected_blobs):
        differences = sorted(set(actual) ^ set(expected_blobs))
        errors.append(
            f"protected {runner} materialized {repository} filesystem differs from "
            f"tracked leaves: {differences[:12]}"
        )
    expected_directories = {
        PurePosixPath(*path.parts[:depth]).as_posix()
        for value in (*expected_blobs, *reviewed_nested_paths)
        for path in (PurePosixPath(value),)
        for depth in range(1, len(path.parts))
    }
    if actual_directories != expected_directories:
        differences = sorted(actual_directories ^ expected_directories)
        errors.append(
            f"protected {runner} materialized {repository} directory projection "
            f"differs from tracked tree: {differences[:12]}"
        )
    digest = hashlib.sha256()
    digest.update(RUNNER_SOURCE_BINDING_POLICY.encode("ascii"))
    for relative, description in sorted(actual.items()):
        digest.update(b"\0")
        digest.update(os.fsencode(relative))
        digest.update(b"\0")
        digest.update(repr(description).encode("ascii"))
    return "sha256:" + digest.hexdigest(), actual


def _verify_materialized_source(
    source_root: Path,
    runner: str,
    source_binding: Mapping[str, Mapping[str, str]],
    errors: list[str],
    *,
    expected_digests: Mapping[str, str] | None = None,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    try:
        source_info = source_root.lstat()
        if not stat.S_ISDIR(source_info.st_mode) or stat.S_ISLNK(source_info.st_mode):
            errors.append(
                f"protected {runner} materialized source root is not a regular directory"
            )
            return digests
    except OSError as exc:
        errors.append(
            f"protected {runner} cannot inspect materialized source root: "
            f"{type(exc).__name__}"
        )
        return digests
    for repository, relative in REPOSITORY_PATHS.items():
        repository_root = source_root / relative
        try:
            repository_info = repository_root.lstat()
            if not stat.S_ISDIR(repository_info.st_mode) or stat.S_ISLNK(
                repository_info.st_mode
            ):
                errors.append(
                    f"protected {runner} materialized {repository} root is unsafe"
                )
                continue
        except OSError as exc:
            errors.append(
                f"protected {runner} cannot inspect materialized {repository} root: "
                f"{type(exc).__name__}"
            )
            continue
        nested = (
            {
                name: source_binding[name]
                for name in REPOSITORY_PATHS
                if name != "accelerate"
            }
            if repository == "accelerate"
            else {}
        )
        digest, _structure = _materialized_repository_content(
            root=repository_root,
            repository=repository,
            runner=runner,
            expected_binding=source_binding.get(repository, {}),
            nested_bindings=nested,
            errors=errors,
        )
        if digest:
            digests[repository] = digest
    if set(digests) != set(REPOSITORY_PATHS):
        errors.append(f"protected {runner} materialized source binding is incomplete")
    if expected_digests is not None and digests != expected_digests:
        errors.append(
            f"protected {runner} materialized source bytes changed during execution"
        )
    return digests


def _make_materialized_directory_read_only(
    descriptor: int,
    *,
    relative: PurePosixPath,
    depth: int,
    visited: list[int],
    errors: list[str],
) -> None:
    """Apply read-only modes through held descriptors without following links."""

    if depth > RUNNER_MATERIALIZATION_MAX_DEPTH:
        errors.append("protected runner read-only walk exceeded its depth bound")
        return
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        errors.append(
            f"protected runner cannot enumerate materialized directory {relative}: "
            f"{type(exc).__name__}"
        )
        return
    for name in names:
        visited[0] += 1
        child_relative = relative / name
        if visited[0] > RUNNER_MATERIALIZATION_MAX_ENTRIES:
            errors.append("protected runner read-only walk exceeded its entry bound")
            return
        child_descriptor = -1
        try:
            child_descriptor = os.open(
                name, directory_flags, dir_fd=descriptor
            )
        except OSError as exc:
            if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
                errors.append(
                    f"protected runner cannot inspect materialized entry "
                    f"{child_relative}: {type(exc).__name__}"
                )
                continue
            try:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode):
                    errors.append(
                        "protected runner materialized read-only tree contains "
                        f"special leaf: {child_relative}"
                    )
                    continue
                child_descriptor = os.open(
                    name, file_flags, dir_fd=descriptor
                )
                opened = os.fstat(child_descriptor)
                if opened.st_nlink != 1:
                    errors.append(
                        "protected runner materialized regular leaf is hardlinked: "
                        f"{child_relative}"
                    )
                    continue
                os.fchmod(
                    child_descriptor,
                    0o555 if opened.st_mode & 0o111 else 0o444,
                )
                path_after = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False
                )
                if (opened.st_dev, opened.st_ino) != (
                    path_after.st_dev,
                    path_after.st_ino,
                ):
                    errors.append(
                        "protected runner materialized leaf path changed while "
                        f"making it read-only: {child_relative}"
                    )
            except OSError as leaf_error:
                errors.append(
                    f"protected runner cannot make materialized leaf read-only "
                    f"{child_relative}: {type(leaf_error).__name__}"
                )
            finally:
                if child_descriptor >= 0:
                    os.close(child_descriptor)
            continue
        try:
            _make_materialized_directory_read_only(
                child_descriptor,
                relative=child_relative,
                depth=depth + 1,
                visited=visited,
                errors=errors,
            )
            os.fchmod(child_descriptor, 0o555)
        except OSError as directory_error:
            errors.append(
                f"protected runner cannot make materialized directory read-only "
                f"{child_relative}: {type(directory_error).__name__}"
            )
        finally:
            os.close(child_descriptor)


def _make_materialized_source_read_only(source_root: Path, errors: list[str]) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(source_root, directory_flags)
        _make_materialized_directory_read_only(
            descriptor,
            relative=PurePosixPath("."),
            depth=0,
            visited=[0],
            errors=errors,
        )
        os.fchmod(descriptor, 0o555)
    except OSError as exc:
        errors.append(
            f"protected runner cannot make materialized source read-only: "
            f"{type(exc).__name__}"
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _materialize_runner_source(
    runner: str,
    source_binding: Mapping[str, Mapping[str, str]],
    errors: list[str],
) -> tuple[Path, Path, dict[str, str]] | None:
    materialization_root = REPO_ROOT / RELEASE_WORK_ROOT / "materialized"
    source_root = materialization_root / "source"
    stage_root = materialization_root / "staged"
    try:
        materialization_root.mkdir(parents=True, exist_ok=False)
        stage_root.mkdir()
    except OSError as exc:
        errors.append(
            f"protected {runner} cannot create materialization root: "
            f"{type(exc).__name__}"
        )
        return None
    outer_source = REPO_ROOT / REPOSITORY_PATHS["accelerate"]
    if not _run_materialization_git(
        runner,
        materialization_root,
        errors,
        "clone",
        "--no-local",
        "--no-checkout",
        "--no-tags",
        str(outer_source),
        str(source_root),
    ):
        return None
    if not _run_materialization_git(
        runner,
        source_root,
        errors,
        "-c",
        "core.hooksPath=/dev/null",
        "checkout",
        "--detach",
        str(source_binding["accelerate"]["revision"]),
    ):
        return None
    if not _run_materialization_git(
        runner, source_root, errors, "remote", "remove", "origin"
    ):
        return None
    outer_tree_raw = _git_bytes(
        source_root,
        errors,
        f"enumerate protected {runner} outer materialized gitlinks",
        "ls-tree",
        "-rz",
        "--full-tree",
        "HEAD",
    )
    if outer_tree_raw is None:
        return None
    _remove_materialized_gitlink_placeholders(
        source_root,
        _parse_runner_tree(outer_tree_raw, "accelerate", errors),
        runner,
        errors,
    )
    for repository, relative in REPOSITORY_PATHS.items():
        if repository == "accelerate":
            continue
        source = REPO_ROOT / relative
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _run_materialization_git(
            runner,
            source_root,
            errors,
            "clone",
            "--no-local",
            "--no-checkout",
            "--no-tags",
            str(source),
            str(target),
        ):
            return None
        if not _run_materialization_git(
            runner,
            target,
            errors,
            "-c",
            "core.hooksPath=/dev/null",
            "checkout",
            "--detach",
            str(source_binding[repository]["revision"]),
        ):
            return None
        if not _run_materialization_git(
            runner, target, errors, "remote", "remove", "origin"
        ):
            return None
        nested_tree_raw = _git_bytes(
            target,
            errors,
            f"enumerate protected {runner} {repository} materialized gitlinks",
            "ls-tree",
            "-rz",
            "--full-tree",
            "HEAD",
        )
        if nested_tree_raw is None:
            return None
        _remove_materialized_gitlink_placeholders(
            target,
            _parse_runner_tree(nested_tree_raw, repository, errors),
            runner,
            errors,
        )
    if errors:
        return None
    try:
        for label, path in (
            ("materialization", materialization_root),
            ("source", source_root),
            ("stage", stage_root),
        ):
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                errors.append(
                    f"protected {runner} {label} root is not a regular directory"
                )
    except OSError as exc:
        errors.append(
            f"protected {runner} cannot inspect materialization roots: "
            f"{type(exc).__name__}"
        )
    if errors:
        return None
    digests = _verify_materialized_source(
        source_root, runner, source_binding, errors
    )
    if errors:
        return None
    _make_materialized_source_read_only(source_root, errors)
    if errors:
        return None
    return source_root, stage_root, digests


def _read_staged_runner_artifact(
    stage_root: Path,
    relative: str,
    maximum_bytes: int,
    errors: list[str],
) -> bytes | None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    held_directories: list[int] = []
    descriptor = -1
    try:
        held_directories.append(os.open(stage_root, directory_flags))
        for part in Path(relative).parts[:-1]:
            held_directories.append(
                os.open(part, directory_flags, dir_fd=held_directories[-1])
            )
        descriptor = os.open(
            Path(relative).name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=held_directories[-1],
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            errors.append(
                f"protected benchmark staged output is unsafe or exceeds its bound: "
                f"{relative}"
            )
            return None
        raw = bytearray()
        while len(raw) <= maximum_bytes:
            chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(raw) != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            errors.append(
                f"protected benchmark staged output changed while reading: {relative}"
            )
            return None
        return bytes(raw)
    except OSError as exc:
        errors.append(
            f"protected benchmark cannot read staged output {relative}: "
            f"{type(exc).__name__}"
        )
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for directory_descriptor in reversed(held_directories):
            os.close(directory_descriptor)


def _staged_output_projection(
    stage_root: Path, errors: list[str]
) -> tuple[set[str], set[str]]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    files: set[str] = set()
    directories: set[str] = set()
    root_descriptor = -1
    stack: list[tuple[int, tuple[str, ...]]] = []
    try:
        root_descriptor = os.open(stage_root, directory_flags)
        stack.append((os.dup(root_descriptor), ()))
        while stack:
            descriptor, parent_parts = stack.pop()
            try:
                names = sorted(os.listdir(descriptor))
                for name in names:
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    parts = (*parent_parts, name)
                    relative = PurePosixPath(*parts).as_posix()
                    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                        directories.add(relative)
                        child = os.open(name, directory_flags, dir_fd=descriptor)
                        stack.append((child, parts))
                    elif stat.S_ISREG(info.st_mode):
                        files.add(relative)
                    else:
                        errors.append(
                            "protected benchmark staged output tree contains a "
                            f"symlink or special file: {relative!r}"
                        )
            finally:
                os.close(descriptor)
    except OSError as exc:
        errors.append(
            f"protected benchmark cannot safely inspect staged output tree: "
            f"{type(exc).__name__}"
        )
    finally:
        if root_descriptor >= 0:
            os.close(root_descriptor)
        for descriptor, _parts in stack:
            os.close(descriptor)
    return files, directories


def _publish_staged_benchmark_outputs(stage_root: Path, errors: list[str]) -> None:
    expected = {BENCHMARK_JSON, BENCHMARK_CSV}
    observed_files, observed_directories = _staged_output_projection(stage_root, errors)
    expected_directories = {
        PurePosixPath(*path.parts[:depth]).as_posix()
        for relative in expected
        for path in (PurePosixPath(relative),)
        for depth in range(1, len(path.parts))
    }
    if observed_files != expected:
        errors.append(
            f"protected benchmark staged output set is not exact: "
            f"{sorted(observed_files)}"
        )
        return
    if observed_directories != expected_directories:
        errors.append(
            f"protected benchmark staged directory set is not exact: "
            f"{sorted(observed_directories)}"
        )
        return
    retained: dict[str, bytes] = {}
    for relative in sorted(expected):
        raw = _read_staged_runner_artifact(
            stage_root, relative, BENCHMARK_MAX_ARTIFACT_BYTES, errors
        )
        if raw is not None:
            retained[relative] = raw
    if errors:
        return
    for relative, raw in retained.items():
        _atomic_write_artifact(relative, raw)


def _current_repository_bindings(
    errors: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    revisions: dict[str, str] = {}
    trees: dict[str, str] = {}
    for repository, relative in REPOSITORY_PATHS.items():
        root = REPO_ROOT / relative
        revision = _git_stdout(
            root, errors, f"resolve current {repository} release revision", "rev-parse", "HEAD"
        )
        tree = _git_stdout(
            root, errors, f"resolve current {repository} release tree", "rev-parse", "HEAD^{tree}"
        )
        if _HEX_40.fullmatch(revision):
            revisions[repository] = revision
        if _HEX_40.fullmatch(tree):
            trees[repository] = tree
    return revisions, trees


def _benchmark_workload_argv() -> list[str]:
    return [
        sys.executable,
        BENCHMARK_CLI,
        "--seed",
        str(BENCHMARK_SEED),
        "--transitions",
        str(BENCHMARK_TRANSITION_COUNT),
        "--json-output",
        f"../staged/{BENCHMARK_JSON}",
        "--csv-output",
        f"../staged/{BENCHMARK_CSV}",
    ]


def _benchmark_expected_provenance(metric_provenance: Mapping[str, Any]) -> str:
    values = set(metric_provenance.values())
    if values == {"measured"}:
        return "measured"
    if values == {"estimated"}:
        return "estimated"
    return "mixed"


def _validate_benchmark_row(
    row: Any,
    index: int,
    errors: list[str],
) -> None:
    label = f"IPS-053 transition {index:02d}"
    if not isinstance(row, Mapping):
        errors.append(f"{label} must be an object")
        return
    expected_keys = (
        "index",
        "scenario",
        "repository_revision",
        "parent_seal",
        "seal_status",
        "required_units",
        "reused_units",
        "invalidated_units",
        "added_units",
        "removed_units",
        "newly_proved_units",
        "unit_count_provenance",
        "cache_hit_rate",
        *BENCHMARK_METRICS,
        "metric_provenance",
        "measurement_provenance",
        "compute_saved_percent",
        "chain_depth",
        "fallback_reason",
        "full_seal_root",
        "incremental_seal_root",
        "deterministic_roots_match",
        "simulated_required_units",
        "rejected_attempts",
    )
    _closed_keys(row, expected_keys, label, errors)
    if row.get("index") != index or row.get("scenario") != BENCHMARK_SCENARIOS[index]:
        errors.append(f"{label} does not match the fixed ordered workload")
    if not isinstance(row.get("repository_revision"), str) or not _HEX_40.fullmatch(
        str(row.get("repository_revision", ""))
    ):
        errors.append(f"{label}.repository_revision must be a concrete Git revision")
    if index == 0:
        if row.get("parent_seal") is not None:
            errors.append(f"{label}.parent_seal must be null for the initial checkpoint")
    elif _sha256_value(row.get("parent_seal"), f"{label}.parent_seal", errors) is None:
        pass
    expected_status = "sealed_full" if index in BENCHMARK_FULL_TRANSITIONS else "sealed_incremental"
    actual_status = row.get("seal_status")
    if index in BENCHMARK_CONDITIONAL_FULL_TRANSITIONS:
        if actual_status not in {"sealed_full", "sealed_incremental"}:
            errors.append(f"{label}.seal_status must be an honest full/incremental decision")
    elif actual_status != expected_status:
        errors.append(f"{label}.seal_status must be {expected_status}")
    counts: dict[str, int] = {}
    for field in (
        "required_units",
        "reused_units",
        "invalidated_units",
        "added_units",
        "removed_units",
        "newly_proved_units",
        "chain_depth",
        "simulated_required_units",
    ):
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{label}.{field} must be a nonnegative integer")
        else:
            counts[field] = value
    if row.get("unit_count_provenance") != "observed_planner_output":
        errors.append(f"{label}.unit_count_provenance is not reviewed")
    if counts.get("newly_proved_units") != (
        counts.get("invalidated_units", -1) + counts.get("added_units", -1)
    ):
        errors.append(f"{label} newly-proved arithmetic is inconsistent")
    if counts.get("required_units") != (
        counts.get("reused_units", -1) + counts.get("newly_proved_units", -1)
    ):
        errors.append(f"{label} required-unit arithmetic is inconsistent")
    required = counts.get("required_units")
    reused = counts.get("reused_units")
    expected_hit = 0.0 if required == 0 else float(reused or 0) / required if required else 0.0
    if not _close_number(row.get("cache_hit_rate"), expected_hit):
        errors.append(f"{label}.cache_hit_rate is not derived from unit counts")
    provenance = row.get("metric_provenance")
    if not isinstance(provenance, Mapping):
        errors.append(f"{label}.metric_provenance must be an object")
        provenance = {}
    _closed_keys(provenance, BENCHMARK_METRICS, f"{label}.metric_provenance", errors)
    for metric in BENCHMARK_METRICS:
        source = provenance.get(metric)
        value = row.get(metric)
        if source not in {"measured", "estimated", "unavailable"}:
            errors.append(f"{label}.{metric} provenance is invalid")
        if source == "unavailable":
            if value is not None:
                errors.append(f"{label}.{metric} must be null when unavailable")
        elif not _nonnegative_number(value):
            errors.append(f"{label}.{metric} must be a nonnegative number")
    if row.get("measurement_provenance") != _benchmark_expected_provenance(provenance):
        errors.append(f"{label}.measurement_provenance does not summarize metric provenance")
    full_cost = row.get("full_proof_cost")
    incremental_cost = row.get("incremental_proof_cost")
    savings = row.get("compute_saved_percent")
    if _nonnegative_number(full_cost) and _nonnegative_number(incremental_cost):
        expected_savings = (
            0.0
            if float(full_cost) == 0
            else (float(full_cost) - float(incremental_cost)) / float(full_cost) * 100.0
        )
        if not _close_signed_number(savings, expected_savings):
            errors.append(f"{label}.compute_saved_percent arithmetic is inconsistent")
    elif savings is not None:
        errors.append(f"{label}.compute_saved_percent must be null when costs are unavailable")
    fallback = row.get("fallback_reason")
    if actual_status == "sealed_full":
        if not isinstance(fallback, str) or not fallback.strip():
            errors.append(f"{label}.fallback_reason must explain the full checkpoint")
    elif fallback is not None:
        errors.append(f"{label}.fallback_reason must be null for an incremental seal")
    full_root = _sha256_value(row.get("full_seal_root"), f"{label}.full_seal_root", errors)
    incremental_root = _sha256_value(
        row.get("incremental_seal_root"), f"{label}.incremental_seal_root", errors
    )
    if row.get("deterministic_roots_match") is not True or (
        full_root is not None and incremental_root is not None and full_root != incremental_root
    ):
        errors.append(f"{label} full and incremental deterministic roots do not match")
    if counts.get("simulated_required_units", 0) != 0:
        errors.append(f"{label} treats simulated required evidence as production-sealed")
    attempts = row.get("rejected_attempts")
    expected_attempts = (
        [{"kind": "wrong_parent", "terminal_status": "stale_parent"}]
        if index == 37
        else []
    )
    if attempts != expected_attempts:
        errors.append(f"{label}.rejected_attempts does not match the fixed workload")


def _validate_benchmark_csv(
    text: str,
    transitions: list[Any],
    errors: list[str],
) -> None:
    try:
        reader = csv.DictReader(io.StringIO(text), strict=True)
        rows = list(reader)
    except (csv.Error, UnicodeError) as exc:
        errors.append(f"IPS-053 benchmark CSV cannot be parsed: {exc}")
        return
    if reader.fieldnames != list(BENCHMARK_CSV_FIELDS):
        errors.append("IPS-053 benchmark CSV header is not the exact reviewed projection")
        return
    if len(rows) != BENCHMARK_TRANSITION_COUNT:
        errors.append("IPS-053 benchmark CSV must contain exactly 40 data rows")
        return
    for index, (csv_row, json_row) in enumerate(zip(rows, transitions)):
        if not isinstance(json_row, Mapping):
            continue
        for field in BENCHMARK_CSV_FIELDS:
            actual = csv_row.get(field, "")
            expected = json_row.get(field)
            if expected is None:
                matches = actual == ""
            elif isinstance(expected, bool):
                matches = actual == str(expected).lower()
            elif isinstance(expected, int):
                matches = actual == str(expected)
            elif isinstance(expected, float):
                try:
                    matches = float(actual) == expected
                except ValueError:
                    matches = False
            else:
                matches = actual == str(expected)
            if not matches:
                errors.append(
                    f"IPS-053 benchmark CSV row {index:02d} field {field} drifts from JSON"
                )


def _validate_parent_bound_output_lifecycle(
    *,
    label: str,
    completion_task_id: str,
    parent_revision: Any,
    source_revisions: Any,
    source_trees: Any,
    completion_outputs: set[str],
    errors: list[str],
) -> None:
    if not isinstance(parent_revision, str) or not _HEX_40.fullmatch(parent_revision):
        errors.append(f"{label} worktree parent revision is not a Git commit")
        return
    if not isinstance(source_revisions, Mapping) or set(source_revisions) != set(
        REPOSITORY_PATHS
    ):
        errors.append(f"{label} source_revisions must bind all three repositories")
        return
    if not isinstance(source_trees, Mapping) or set(source_trees) != set(
        REPOSITORY_PATHS
    ):
        errors.append(f"{label} source_trees must bind all three repositories")
        return
    if source_revisions.get("accelerate") != parent_revision:
        errors.append(f"{label} source revision does not equal its worktree parent")
    for repository, relative in REPOSITORY_PATHS.items():
        revision = source_revisions.get(repository)
        tree = source_trees.get(repository)
        if not isinstance(revision, str) or not _HEX_40.fullmatch(revision):
            errors.append(f"{label} source_revisions.{repository} is not a Git commit")
            continue
        if not isinstance(tree, str) or not _HEX_40.fullmatch(tree):
            errors.append(f"{label} source_trees.{repository} is not a Git tree")
            continue
        resolved = _git("rev-parse", f"{revision}^{{tree}}", cwd=REPO_ROOT / relative)
        if resolved.returncode != 0 or resolved.stdout.strip() != tree:
            errors.append(f"{label} source_trees.{repository} mismatches its commit")

    current_revisions, current_trees = _current_repository_bindings(errors)
    for repository in ("datasets", "kit"):
        if (
            current_revisions.get(repository) != source_revisions.get(repository)
            or current_trees.get(repository) != source_trees.get(repository)
        ):
            errors.append(f"{label} nested {repository} source binding changed")
    current = current_revisions.get("accelerate")
    if current == parent_revision:
        if current_trees.get("accelerate") != source_trees.get("accelerate"):
            errors.append(f"{label} candidate source tree changed before validation")
        return
    if not isinstance(current, str) or not _HEX_40.fullmatch(current):
        errors.append(f"{label} current completion revision is unavailable")
        return
    current_line = _git("rev-list", "--parents", "-n", "1", current)
    current_tokens = (
        current_line.stdout.strip().split() if current_line.returncode == 0 else []
    )
    if current_tokens == [current, parent_revision]:
        candidate_changed = _git(
            "diff",
            "--name-only",
            "--no-renames",
            parent_revision,
            current,
            "--",
        )
        candidate_paths = {
            line for line in candidate_changed.stdout.splitlines() if line
        }
        if (
            candidate_changed.returncode != 0
            or candidate_paths != completion_outputs
        ):
            errors.append(
                f"{label} committed candidate must change exactly "
                f"{sorted(completion_outputs)}; got {sorted(candidate_paths)}"
            )
        dirty = _git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *sorted(completion_outputs),
        )
        if dirty.returncode != 0 or dirty.stdout.strip():
            errors.append(f"{label} committed candidate evidence has later mutation")
        return
    history = _git_stdout(
        REPO_ROOT,
        errors,
        f"enumerate {label} first-parent publication history",
        "rev-list",
        "--first-parent",
        "--reverse",
        f"{parent_revision}..{current}",
        "--",
    ).splitlines()
    if not history:
        errors.append(f"{label} has no publication commit after its worktree parent")
        return
    integration_index: int | None = None
    integration = ""
    integration_parent = ""
    candidate = ""
    for index, commit in enumerate(history):
        parent_line = _git("rev-list", "--parents", "-n", "1", commit)
        tokens = (
            parent_line.stdout.strip().split()
            if parent_line.returncode == 0
            else []
        )
        if len(tokens) != 3:
            continue
        possible_candidate = tokens[2]
        candidate_line = _git(
            "rev-list", "--parents", "-n", "1", possible_candidate
        )
        candidate_tokens = (
            candidate_line.stdout.strip().split()
            if candidate_line.returncode == 0
            else []
        )
        if candidate_tokens == [possible_candidate, parent_revision]:
            integration_index = index
            integration = commit
            integration_parent = tokens[1]
            candidate = possible_candidate
            break
    if integration_index is None:
        errors.append(
            f"{label} publication must be one no-ff merge of an exact candidate child"
        )
        return
    if (
        _git(
            "merge-base", "--is-ancestor", parent_revision, integration_parent
        ).returncode
        != 0
    ):
        errors.append(f"{label} integration first parent left its bound lineage")
    prefix_artifact_drift = _git(
        "diff",
        "--quiet",
        parent_revision,
        integration_parent,
        "--",
        *sorted(completion_outputs),
    )
    if prefix_artifact_drift.returncode != 0:
        errors.append(f"{label} evidence paths changed before their integration merge")
    candidate_changed = _git(
        "diff",
        "--name-only",
        "--no-renames",
        parent_revision,
        candidate,
        "--",
    )
    candidate_paths = {
        line for line in candidate_changed.stdout.splitlines() if line
    }
    if candidate_changed.returncode != 0 or candidate_paths != completion_outputs:
        errors.append(
            f"{label} candidate must change exactly {sorted(completion_outputs)}; "
            f"got {sorted(candidate_paths)}"
        )
    changed = _git(
        "diff",
        "--name-only",
        "--no-renames",
        integration_parent,
        integration,
        "--",
    )
    changed_paths = {line for line in changed.stdout.splitlines() if line}
    if changed.returncode != 0 or changed_paths != completion_outputs:
        errors.append(
            f"{label} integration must change exactly {sorted(completion_outputs)}; "
            f"got {sorted(changed_paths)}"
        )
    candidate_projection = _git(
        "diff", "--quiet", candidate, integration, "--", *sorted(completion_outputs)
    )
    if candidate_projection.returncode != 0:
        errors.append(f"{label} merged evidence differs from its reviewed candidate bytes")
    completion_seen = False
    for descendant in history[integration_index + 1 :]:
        descendant_parent = _git_stdout(
            REPO_ROOT,
            errors,
            f"resolve {label} descendant parent",
            "rev-parse",
            f"{descendant}^1",
        )
        if not _HEX_40.fullmatch(descendant_parent):
            continue
        descendant_paths = {
            line
            for line in _git_stdout(
                REPO_ROOT,
                errors,
                f"inspect {label} descendant {descendant}",
                "diff",
                "--name-only",
                "--no-renames",
                descendant_parent,
                descendant,
                "--",
            ).splitlines()
            if line
        }
        if "docs/architecture/incremental_proof_sealer.todo.md" not in descendant_paths:
            # Another authorized lane may integrate between this task's merge
            # and the serialized daemon status publication. The phase-aware
            # full board gate audits that commit; this lifecycle gate only
            # binds the retained evidence and its own completion transaction.
            continue
        transitioned = _validate_taskboard_status_commit(
            descendant_parent, descendant, errors
        )
        if completion_task_id in transitioned:
            # The task's integration is already the first-parent ancestor at
            # this point; the status commit may follow other admitted status
            # publications serialized by the daemon.
            completion_seen = True
    if not completion_seen:
        errors.append(
            f"{label} publication lineage lacks the daemon completion for "
            f"{completion_task_id}"
        )
    artifact_drift = _git(
        "diff", "--quiet", integration, current, "--", *sorted(completion_outputs)
    )
    if artifact_drift.returncode != 0:
        errors.append(f"{label} evidence changed after its integration merge")
    dirty = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *sorted(completion_outputs),
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        errors.append(f"{label} committed evidence has later staged/worktree mutation")


def _validate_benchmark_artifacts(
    errors: list[str],
) -> tuple[Mapping[str, Any], str]:
    payload = _artifact_json(
        BENCHMARK_JSON,
        errors,
        maximum_bytes=BENCHMARK_MAX_ARTIFACT_BYTES,
        bound_label="one-MiB",
    )
    _closed_keys(
        payload,
        (
            "schema_version",
            "benchmark_id",
            "seed",
            "transition_count",
            "benchmark_worktree_parent_revision",
            "source_revisions",
            "source_trees",
            "execution_context",
            "capabilities",
            "transitions",
        ),
        "IPS-053 benchmark",
        errors,
    )
    if payload.get("schema_version") != BENCHMARK_SCHEMA:
        errors.append("IPS-053 benchmark schema is not reviewed")
    if payload.get("benchmark_id") != BENCHMARK_ID:
        errors.append("IPS-053 benchmark_id is not reviewed")
    if payload.get("seed") != BENCHMARK_SEED or payload.get("transition_count") != 40:
        errors.append("IPS-053 benchmark seed/transition count is not fixed")
    _validate_parent_bound_output_lifecycle(
        label="IPS-053 benchmark",
        completion_task_id="IPS-053",
        parent_revision=payload.get("benchmark_worktree_parent_revision"),
        source_revisions=payload.get("source_revisions"),
        source_trees=payload.get("source_trees"),
        completion_outputs={BENCHMARK_JSON, BENCHMARK_CSV},
        errors=errors,
    )
    context = payload.get("execution_context")
    expected_context = {
        "runner_id": "protected-board-benchmark-runner@1",
        "argv": _benchmark_workload_argv(),
        "process_observed": True,
        "test_execution_cryptographically_proven": False,
        "claim": "benchmark_process_observed_metrics_retain_per_metric_provenance",
    }
    if context != expected_context:
        errors.append("IPS-053 execution_context is not the exact protected runner contract")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, Mapping):
        errors.append("IPS-053 capabilities must be an object")
    else:
        _closed_keys(
            capabilities,
            ("real_prover_available", "recursive_verification_available", "gpu_available", "notes"),
            "IPS-053 capabilities",
            errors,
        )
        for field in ("real_prover_available", "recursive_verification_available", "gpu_available"):
            if not isinstance(capabilities.get(field), bool):
                errors.append(f"IPS-053 capabilities.{field} must be boolean")
        if not isinstance(capabilities.get("notes"), str) or not capabilities.get("notes", "").strip():
            errors.append("IPS-053 capabilities.notes must state capability limits")
    transitions = payload.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != BENCHMARK_TRANSITION_COUNT:
        errors.append("IPS-053 transitions must contain exactly 40 rows")
        transitions = []
    for index, row in enumerate(transitions):
        _validate_benchmark_row(row, index, errors)
    benchmark_retained = _secure_read_repo_file(
        BENCHMARK_JSON,
        required_parent=str(Path(BENCHMARK_JSON).parent),
        label="IPS-053 benchmark JSON",
        maximum_bytes=BENCHMARK_MAX_ARTIFACT_BYTES,
        bound_label="one-MiB",
        errors=errors,
    )
    benchmark_digest = ""
    if benchmark_retained is not None:
        raw, digest = benchmark_retained
        benchmark_digest = f"sha256:{digest}"
        if raw != _canonical_json_bytes(payload) + b"\n":
            errors.append("IPS-053 benchmark JSON must be canonical with one final newline")
    csv_text = _require_nonempty_file(
        BENCHMARK_CSV,
        errors,
        maximum_bytes=BENCHMARK_MAX_ARTIFACT_BYTES,
        bound_label="one-MiB",
    )
    if transitions and csv_text:
        _validate_benchmark_csv(csv_text, transitions, errors)
    return payload, benchmark_digest


def _combined_provenance(values: Iterable[str]) -> str:
    sources = set(values)
    if sources == {"measured"}:
        return "measured"
    if sources == {"estimated"}:
        return "estimated"
    return "mixed"


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _derived_benchmark_summary(
    benchmark: Mapping[str, Any], benchmark_digest: str
) -> dict[str, Any]:
    transitions = [row for row in benchmark.get("transitions", []) if isinstance(row, Mapping)]
    provenance_values = [str(row.get("measurement_provenance")) for row in transitions]
    measurement_counts = {
        name: provenance_values.count(name) for name in ("measured", "estimated", "mixed")
    }
    reuse_values = [float(row["cache_hit_rate"]) * 100.0 for row in transitions]
    saving_rows = [
        row for row in transitions if _finite_number(row.get("compute_saved_percent"))
    ]
    saving_values = [float(row["compute_saved_percent"]) for row in saving_rows]
    average_provenance = _combined_provenance(provenance_values)
    best = (
        max(saving_rows, key=lambda row: float(row["compute_saved_percent"]))
        if saving_rows
        else None
    )
    worst = (
        min(saving_rows, key=lambda row: float(row["compute_saved_percent"]))
        if saving_rows
        else None
    )

    def target(indices: tuple[int, ...], threshold: float, field: str) -> dict[str, Any]:
        selected = [transitions[index] for index in indices]
        values = [
            float(row[field]) * (100.0 if field == "cache_hit_rate" else 1.0)
            for row in selected
            if (
                _finite_number(row.get(field))
                if field == "compute_saved_percent"
                else _nonnegative_number(row.get(field))
            )
        ]
        actual = _average(values)
        return {
            "target_percent": threshold,
            "actual_percent": actual,
            "met": actual is not None and actual >= threshold,
            "provenance": _combined_provenance(
                str(row.get("measurement_provenance")) for row in selected
            ),
            "transition_indices": list(indices),
        }

    metric_summary: dict[str, Any] = {}
    for metric in (
        "proof_size_bytes",
        "seal_size_bytes",
        "seal_verification_seconds",
        "storage_growth_bytes",
        "prover_cpu_seconds",
        "prover_gpu_seconds",
        "peak_memory_bytes",
    ):
        available = [row for row in transitions if _nonnegative_number(row.get(metric))]
        values = [float(row[metric]) for row in available]
        sources = [str(row["metric_provenance"].get(metric)) for row in transitions]
        metric_summary[metric] = {
            "available_rows": len(available),
            "measured_rows": sources.count("measured"),
            "estimated_rows": sources.count("estimated"),
            "unavailable_rows": sources.count("unavailable"),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": _average(values),
        }
    return {
        "schema_version": BENCHMARK_SUMMARY_SCHEMA,
        "benchmark_digest": benchmark_digest,
        "transition_count": BENCHMARK_TRANSITION_COUNT,
        "measurement_counts": measurement_counts,
        "average_reuse_rate": {
            "value_percent": _average(reuse_values),
            "weighting": "unweighted_transition_mean",
            "provenance": average_provenance,
        },
        "average_compute_reduction": {
            "value_percent": _average(saving_values),
            "weighting": "unweighted_available_transition_mean",
            "provenance": _combined_provenance(
                str(row.get("measurement_provenance")) for row in saving_rows
            ),
        },
        "best_case": (
            {
                "transition_index": best["index"],
                "scenario": best["scenario"],
                "compute_saved_percent": best["compute_saved_percent"],
                "provenance": best["measurement_provenance"],
            }
            if best is not None
            else {
                "transition_index": None,
                "scenario": None,
                "compute_saved_percent": None,
                "provenance": "unavailable",
            }
        ),
        "worst_case": (
            {
                "transition_index": worst["index"],
                "scenario": worst["scenario"],
                "compute_saved_percent": worst["compute_saved_percent"],
                "provenance": worst["measurement_provenance"],
            }
            if worst is not None
            else {
                "transition_index": None,
                "scenario": None,
                "compute_saved_percent": None,
                "provenance": "unavailable",
            }
        ),
        "fallback_transition_indices": [
            int(row["index"])
            for row in transitions
            if row.get("seal_status") == "sealed_full"
        ],
        "target_assessment": {
            "localized_reuse_70_percent": target((1, 5, 13, 23, 36), 70.0, "cache_hit_rate"),
            "mixed_compute_reduction_50_percent": target(
                tuple(range(BENCHMARK_TRANSITION_COUNT)), 50.0, "compute_saved_percent"
            ),
            "documentation_compute_reduction_80_percent": target(
                (2, 11, 21, 34), 80.0, "compute_saved_percent"
            ),
        },
        "metric_summary": metric_summary,
    }


def _validate_benchmark_summary(errors: list[str]) -> None:
    benchmark_errors: list[str] = []
    benchmark, benchmark_digest = _validate_benchmark_artifacts(benchmark_errors)
    errors.extend(f"IPS-054 prerequisite: {item}" for item in benchmark_errors)
    payload = _artifact_json(BENCHMARK_SUMMARY_JSON, errors)
    _closed_keys(
        payload,
        (
            "schema_version",
            "benchmark_digest",
            "transition_count",
            "measurement_counts",
            "average_reuse_rate",
            "average_compute_reduction",
            "best_case",
            "worst_case",
            "fallback_transition_indices",
            "target_assessment",
            "metric_summary",
            "limitations",
        ),
        "IPS-054 benchmark summary",
        errors,
    )
    if benchmark and len(benchmark.get("transitions", [])) == BENCHMARK_TRANSITION_COUNT:
        expected = _derived_benchmark_summary(benchmark, benchmark_digest)
        for field, value in expected.items():
            if payload.get(field) != value:
                errors.append(f"IPS-054 {field} does not derive exactly from benchmark.json")
    limitations = payload.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        errors.append("IPS-054 limitations must be a non-empty string list")
    report = _require_nonempty_file(BENCHMARK_REPORT, errors)
    required_terms = (
        benchmark_digest,
        BENCHMARK_SUMMARY_SCHEMA,
        "measured",
        "estimated",
        "unavailable",
        "targets are not facts",
        "receipt aggregation does not prove test execution",
        "simulated required units cannot satisfy a production seal",
        "localized_reuse_70_percent",
        "mixed_compute_reduction_50_percent",
        "documentation_compute_reduction_80_percent",
        "proof_size_bytes",
        "seal_verification_seconds",
        "storage_growth_bytes",
        "prover_cpu_seconds",
        "prover_gpu_seconds",
        "peak_memory_bytes",
    )
    folded = report.casefold()
    for term in required_terms:
        if term and term.casefold() not in folded:
            errors.append(f"IPS-054 benchmark report omits {term!r}")


def _validate_trust_and_migration_docs(errors: list[str]) -> None:
    trust = _require_nonempty_file(
        "docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_MODEL.md", errors
    )
    migration = _require_nonempty_file(
        "docs/architecture/INCREMENTAL_PROOF_SEALER_MIGRATION.md", errors
    )
    trust_terms = (
        "integrity commitment",
        "exact bytes",
        "does not establish correct execution",
        "signed execution receipt",
        "trusted signer assertion",
        "signature verification",
        "receipt-aggregation zk proof",
        "does not prove the underlying tests ran",
        "direct execution proof",
        "declared deterministic computation",
        "incremental or recursive commit seal",
        "accepted parent",
        "public inputs",
        "private inputs",
        "sensitive witness",
        "child signatures are not verified inside the circuit",
        "test execution is not directly proven",
        "manifest aggregation, not recursive proof verification",
        "trusted setup origin",
        "test-only keys",
        "content-addressed verification keys",
        "allowlist",
        "no proof key is silently generated",
        "unknown proof systems are rejected",
        "no arbitrary circuit or executable",
        "proofs are verified before cache admission",
        "canonicalization change requires a full checkpoint",
        "circuit change requires a full checkpoint",
        "verification-key change requires a full checkpoint",
        "cache corruption requires a full checkpoint",
        "compare-and-swap",
        "wal",
        "ambiguous external prover outcome is not success",
        "remaining work before production use",
    )
    migration_terms = (
        "accept",
        "adapt",
        "reverify",
        "reject",
        "simulated",
        "integrity-only",
        "signed receipt",
        "direct execution",
        "no assurance upgrade",
        "verification-key allowlist",
        "proof verification before cache admission",
        "schema change",
        "canonicalization change",
        "full checkpoint",
        "staged migration",
        "rollback",
        "unknown legacy proof system",
        "sensitive witness",
        "test-only key",
    )
    for term in trust_terms:
        if term not in trust.casefold():
            errors.append(f"IPS-055 trust model is missing {term!r}")
    for term in migration_terms:
        if term not in migration.casefold():
            errors.append(f"IPS-055 migration guide is missing {term!r}")
    disallowed = (
        "the entire repository was proven correct",
        "all pytest execution was proven in zero knowledge",
        "the code change is semantically correct",
    )
    for phrase in disallowed:
        if phrase in trust.casefold() or phrase in migration.casefold():
            errors.append(f"IPS-055 documentation contains disallowed claim {phrase!r}")


def _release_environment(workspace: Path, *, source_root: Path = REPO_ROOT) -> dict[str, str]:
    python_path = os.pathsep.join(
        str((source_root / relative).resolve())
        for relative in (Path("."), Path("ipfs_datasets_py"), Path("ipfs_kit_py"))
    )
    environment = {
        **_fixed_git_environment(),
        "CARGO_NET_OFFLINE": "true",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(workspace / "home"),
        "HYPOTHESIS_STORAGE_DIRECTORY": str(workspace / "hypothesis"),
        "IPS_PROTECTED_MATERIALIZED_SOURCE": "1",
        "IPFS_ACCEL_AUTO_INSTALL": "0",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
        "IPFS_DATASETS_ENABLE_GROTH16": "0",
        "IPFS_DATASETS_PY_AUTO_GROTH16_BUILD": "0",
        "IPFS_DATASETS_RUN_GROTH16_EVM": "0",
        "IPFS_DATASETS_RUN_PROVEKIT_TESTS": "0",
        "IPFS_OFFLINE": "1",
        "IPFS_PATH": str(workspace / "ipfs-repo"),
        "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL": "0",
        "IPFS_TEST_PROOF_REUSE_MODE": "off",
        "NO_COLOR": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": str(workspace / "pycache"),
        "PYTHONPATH": python_path,
        "PYTEST_ADDOPTS": (
            f"--benchmark-storage=file://{workspace / 'pytest-benchmark'}"
        ),
        "PATH": RELEASE_FIXED_EXECUTABLE_PATH,
        "TERM": "dumb",
        "TMPDIR": str(workspace / "tmp"),
        "TRANSFORMERS_OFFLINE": "1",
        "TZ": "UTC",
    }
    for child in (
        "home",
        "hypothesis",
        "ipfs-repo",
        "pycache",
        "pytest-benchmark",
        "tmp",
        "pytest-cache",
        "pytest-tmp",
    ):
        (workspace / child).mkdir(parents=True, exist_ok=True)
    return environment


def _validate_release_ipfs_preflight(errors: list[str]) -> None:
    """Never let the reviewed kit suites discover and launch a real IPFS binary."""

    resolved = shutil.which("ipfs", path=RELEASE_FIXED_EXECUTABLE_PATH)
    if resolved is not None:
        errors.append(f"release runner refused because fixed PATH resolves ipfs at {resolved}")


_RELEASE_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("private-key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("github-token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws-access-key", re.compile(rb"\bAKIA[A-Z0-9]{16}\b")),
    (
        "jwt",
        re.compile(
            rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
    ),
    (
        "named-secret",
        re.compile(
            rb"(?i)\b(?:api[_-]?key|authorization|password|secret|token)\b"
            rb"\s*(?:=|:)\s*['\"]?[A-Za-z0-9_./+~-]{20,}"
        ),
    ),
)


def _validate_release_public_log(raw: bytes, errors: list[str]) -> None:
    try:
        raw.decode("utf-8")
    except UnicodeError:
        errors.append("IPS-056 retained public log must be strict UTF-8 text")
        return
    if b"\x00" in raw:
        errors.append("IPS-056 retained public log may not contain NUL bytes")
        return
    for label, pattern in _RELEASE_SECRET_PATTERNS:
        if pattern.search(raw):
            errors.append(f"IPS-056 retained public log refused by secret scan: {label}")
            return


def _release_baseline_observations(errors: list[str]) -> dict[str, dict[str, Any]]:
    config = _load_json(CONFIG_PATH, errors)
    receipts = _validate_operator_baseline_bundle(
        config,
        errors,
        enforce_current_sources=False,
    )
    observations: dict[str, dict[str, Any]] = {}
    for receipt in receipts.values():
        commands = receipt.get("commands")
        if not isinstance(commands, list):
            continue
        for command in commands:
            if not isinstance(command, Mapping) or not isinstance(command.get("id"), str):
                continue
            command_id = str(command["id"])
            if command_id in observations:
                errors.append(f"release baseline repeats command id {command_id}")
                continue
            observations[command_id] = {
                "receipt_digest": receipt.get("receipt_digest"),
                "capture_status": command.get("capture_status"),
                "exit_code": command.get("exit_code"),
                "collected_count": command.get("collected_count"),
                "collection_complete": command.get("collection_complete"),
                "outcome_counts": command.get("outcome_counts"),
                "non_pass_nodes": command.get("non_pass_nodes"),
            }
    return observations


def _release_acceptance_status(
    spec: Mapping[str, Any], observation: Mapping[str, Any]
) -> str:
    """Classify a release observation without turning an inherited red into green."""

    status = observation.get("capture_status")
    exit_code = observation.get("exit_code")
    counts = observation.get("outcome_counts")
    current_nonpass = observation.get("non_pass_nodes")
    complete = observation.get("collection_complete")
    if (
        status != "completed"
        or not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not isinstance(counts, Mapping)
        or not isinstance(current_nonpass, list)
        or not isinstance(complete, bool)
    ):
        return "regressed"
    if any(
        isinstance(counts.get(field), bool)
        or not isinstance(counts.get(field), int)
        or counts.get(field, -1) < 0
        for field in BASELINE_OUTCOME_FIELDS
    ):
        return "regressed"
    if counts.get("selected", 0) <= 0:
        return "regressed"
    collected_count = observation.get("collected_count")
    if complete and (
        isinstance(collected_count, bool)
        or not isinstance(collected_count, int)
        or collected_count != counts["selected"] + counts["deselected"]
    ):
        return "regressed"

    if spec.get("suite_origin") == "incremental_proof_sealer_current_tree_suite":
        if (
            exit_code == 0
            and complete
            and counts.get("selected", 0) > 0
            and not any(
                counts.get(field, 0)
                for field in (
                    "failed",
                    "errors",
                    "xpassed",
                    "skipped",
                    "xfailed",
                    "deselected",
                )
            )
        ):
            return "green"
        return "regressed"

    baseline = spec.get("baseline_observation")
    if not isinstance(baseline, Mapping):
        return "regressed"
    baseline_counts = baseline.get("outcome_counts")
    baseline_nonpass = baseline.get("non_pass_nodes")
    baseline_complete = baseline.get("collection_complete")
    baseline_exit = baseline.get("exit_code")
    if (
        baseline.get("capture_status") != "completed"
        or not isinstance(baseline_counts, Mapping)
        or not isinstance(baseline_nonpass, list)
        or not isinstance(baseline_complete, bool)
        or not isinstance(baseline_exit, int)
        or isinstance(baseline_exit, bool)
    ):
        return "regressed"
    if any(
        isinstance(baseline_counts.get(field), bool)
        or not isinstance(baseline_counts.get(field), int)
        or baseline_counts.get(field, -1) < 0
        for field in BASELINE_OUTCOME_FIELDS
    ):
        return "regressed"

    if counts.get("passed", -1) < baseline_counts.get("passed", 0):
        return "regressed"
    for field in (
        "failed",
        "errors",
        "xpassed",
        "skipped",
        "xfailed",
        "deselected",
    ):
        if counts.get(field, -1) > baseline_counts.get(field, 0):
            return "regressed"
    baseline_nodes = {
        (item.get("status"), item.get("node_id"))
        for item in baseline_nonpass
        if isinstance(item, Mapping)
    }
    current_nodes = {
        (item.get("status"), item.get("node_id"))
        for item in current_nonpass
        if isinstance(item, Mapping)
    }
    if len(current_nodes) != len(current_nonpass) or not current_nodes <= baseline_nodes:
        return "regressed"
    if baseline_complete:
        if not complete:
            return "regressed"
    elif not complete and (
        observation.get("collected_count") != baseline.get("collected_count")
        or counts != baseline_counts
        or current_nonpass != baseline_nonpass
        or exit_code != baseline_exit
    ):
        return "regressed"
    if baseline_exit == 0 and exit_code != 0:
        return "regressed"

    non_green = (
        exit_code != 0
        or not complete
        or any(
            counts.get(field, 0)
            for field in (
                "failed",
                "errors",
                "xpassed",
                "skipped",
                "xfailed",
                "deselected",
            )
        )
    )
    return "baseline_compatible_non_green" if non_green else "green"


def _release_suite_specs(errors: list[str]) -> list[dict[str, Any]]:
    registry = _reviewed_suite_registry(errors)
    baseline_observations = _release_baseline_observations(errors)
    specs: list[dict[str, Any]] = []
    work_root = REPO_ROOT / RELEASE_WORK_ROOT
    for task_id in BASELINE_RECEIPT_SPECS:
        for command_id in BASELINE_RECEIPT_SPECS[task_id]["command_ids"]:
            suite = registry.get(str(command_id))
            if not isinstance(suite, Mapping):
                errors.append(f"release suite registry is missing {command_id}")
                continue
            workspace = work_root / str(command_id)
            substitutions = {
                "{python}": sys.executable,
                "{cache_dir}": str(workspace / "pytest-cache"),
                "{basetemp}": str(workspace / "pytest-tmp"),
            }
            argv = []
            for token in suite.get("argv_template", []):
                rendered = str(token)
                for placeholder, replacement in substitutions.items():
                    rendered = rendered.replace(placeholder, replacement)
                argv.append(rendered)
            specs.append(
                {
                    "id": command_id,
                    "suite_origin": "reviewed_existing_zk_suite",
                    "cwd": str(suite.get("cwd")),
                    "argv": argv,
                    "timeout_seconds": int(suite.get("timeout_seconds", 0)),
                    "baseline_observation": baseline_observations.get(str(command_id)),
                }
            )
    for command_id, cwd, test_path, timeout_seconds in RELEASE_NEW_SUITES:
        workspace = work_root / command_id
        specs.append(
            {
                "id": command_id,
                "suite_origin": "incremental_proof_sealer_current_tree_suite",
                "cwd": cwd,
                "argv": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-vv",
                    "-ra",
                    "--tb=line",
                    "--color=no",
                    "--trace-config",
                    "-o",
                    f"cache_dir={workspace / 'pytest-cache'}",
                    f"--basetemp={workspace / 'pytest-tmp'}",
                    test_path,
                ],
                "timeout_seconds": timeout_seconds,
            }
        )
    ids = [str(spec["id"]) for spec in specs]
    if len(ids) != 20 or len(set(ids)) != 20:
        errors.append("release runner must bind exactly 17 existing and 3 new suites")
    return specs


def _release_assurance() -> dict[str, Any]:
    return {
        "process_observed": True,
        "test_execution_cryptographically_proven": False,
        "cryptographic_proof": None,
        "signature": None,
        "public_log_witness_policy": RELEASE_PUBLIC_LOG_POLICY,
        "sensitive_witness_data_logged": False,
        "claim": "pytest_process_outputs_observed_only_not_a_proof_of_test_execution",
    }


def _validate_release_log_slice(
    command: Mapping[str, Any],
    raw_log: bytes,
    expected_offset: int,
    label: str,
    errors: list[str],
) -> tuple[bytes, int]:
    offset = command.get("log_offset")
    size = command.get("log_bytes")
    if offset != expected_offset or not isinstance(size, int) or isinstance(size, bool) or size < 0:
        errors.append(f"{label} log slice is not contiguous and bounded")
        return b"", expected_offset
    end = offset + size
    if end > len(raw_log):
        errors.append(f"{label} log slice exceeds the retained release log")
        return b"", expected_offset
    retained = raw_log[offset:end]
    expected_digest = "sha256:" + hashlib.sha256(retained).hexdigest()
    if command.get("log_sha256") != expected_digest:
        errors.append(f"{label} log slice digest does not match retained bytes")
    return retained, end


def _validate_release_validation(errors: list[str]) -> None:
    retained_receipt = _secure_read_repo_file(
        RELEASE_VALIDATION_JSON,
        required_parent=str(Path(RELEASE_VALIDATION_JSON).parent),
        label="IPS-056 release validation receipt",
        maximum_bytes=BASELINE_MAX_RECEIPT_BYTES,
        bound_label="two-MiB",
        errors=errors,
    )
    retained_log = _secure_read_repo_file(
        RELEASE_VALIDATION_LOG,
        required_parent=str(Path(RELEASE_VALIDATION_LOG).parent),
        label="IPS-056 retained release log",
        maximum_bytes=RELEASE_MAX_LOG_BYTES,
        bound_label="six-MiB",
        errors=errors,
    )
    if retained_receipt is None or retained_log is None:
        return
    raw_receipt, _ = retained_receipt
    raw_log, log_digest = retained_log
    _validate_release_public_log(raw_log, errors)
    try:
        receipt = json.loads(
            raw_receipt.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"IPS-056 release validation receipt is invalid JSON: {exc}")
        return
    if not isinstance(receipt, Mapping):
        errors.append("IPS-056 release validation receipt must be an object")
        return
    _closed_keys(
        receipt,
        (
            "schema_version",
            "runner_id",
            "validation_worktree_parent_revision",
            "source_revisions",
            "source_trees",
            "environment_policy_id",
            "terminal_gate",
            "pytest_commands",
            "retained_log",
            "assurance",
            "receipt_digest",
        ),
        "IPS-056 release validation receipt",
        errors,
    )
    if receipt.get("schema_version") != RELEASE_VALIDATION_SCHEMA:
        errors.append("IPS-056 release validation schema is not reviewed")
    if receipt.get("runner_id") != RELEASE_RUNNER_ID:
        errors.append("IPS-056 release validation runner is not protected")
    if receipt.get("environment_policy_id") != RELEASE_ENVIRONMENT_POLICY:
        errors.append("IPS-056 release environment policy is not reviewed")
    declared_digest = receipt.get("receipt_digest")
    body = dict(receipt)
    body.pop("receipt_digest", None)
    expected_digest = "sha256:" + hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    if declared_digest != expected_digest:
        errors.append("IPS-056 release receipt digest does not match canonical content")
    if raw_receipt != _canonical_json_bytes(receipt) + b"\n":
        errors.append("IPS-056 release receipt is not canonical JSON")
    _validate_parent_bound_output_lifecycle(
        label="IPS-056 release evidence",
        completion_task_id="IPS-056",
        parent_revision=receipt.get("validation_worktree_parent_revision"),
        source_revisions=receipt.get("source_revisions"),
        source_trees=receipt.get("source_trees"),
        completion_outputs={RELEASE_REPORT, RELEASE_VALIDATION_JSON, RELEASE_VALIDATION_LOG},
        errors=errors,
    )
    revisions_value = receipt.get("source_revisions")
    revisions = dict(revisions_value) if isinstance(revisions_value, Mapping) else {}
    if receipt.get("assurance") != _release_assurance():
        errors.append("IPS-056 release receipt assurance overstates process observation")
    log_ref = receipt.get("retained_log")
    expected_log_ref = {
        "path": RELEASE_VALIDATION_LOG,
        "bytes": len(raw_log),
        "sha256": f"sha256:{log_digest}",
    }
    if log_ref != expected_log_ref:
        errors.append("IPS-056 retained_log binding does not match full retained bytes")
    offset = 0
    terminal = receipt.get("terminal_gate")
    if not isinstance(terminal, Mapping):
        errors.append("IPS-056 terminal_gate must be an object")
        terminal = {}
    _closed_keys(
        terminal,
        (
            "id",
            "argv",
            "cwd",
            "timeout_seconds",
            "duration_ns",
            "capture_status",
            "exit_code",
            "log_offset",
            "log_bytes",
            "log_sha256",
        ),
        "IPS-056 terminal_gate",
        errors,
    )
    expected_terminal_argv = [
        sys.executable,
        "scripts/validate_incremental_proof_sealer_board.py",
        "--check-terminal",
    ]
    if terminal.get("id") != "terminal-board-gate" or terminal.get("argv") != expected_terminal_argv:
        errors.append("IPS-056 terminal gate argv is not exact")
    if terminal.get("cwd") != "." or terminal.get("timeout_seconds") != 120:
        errors.append("IPS-056 terminal gate cwd/timeout drifted")
    if terminal.get("capture_status") != "completed" or terminal.get("exit_code") != 0:
        errors.append("IPS-056 terminal structural gate was not observed successful")
    if not isinstance(terminal.get("duration_ns"), int) or terminal.get("duration_ns", 0) <= 0:
        errors.append("IPS-056 terminal gate duration must be measured")
    _, offset = _validate_release_log_slice(terminal, raw_log, offset, "IPS-056 terminal gate", errors)
    expected_suites = _release_suite_specs(errors)
    commands = receipt.get("pytest_commands")
    if not isinstance(commands, list) or len(commands) != len(expected_suites):
        errors.append("IPS-056 pytest_commands must exactly cover 17 existing and 3 new suites")
        commands = []
    baseline_non_green_ids: list[str] = []
    for index, expected_suite in enumerate(expected_suites):
        if index >= len(commands) or not isinstance(commands[index], Mapping):
            errors.append(f"IPS-056 release suite {expected_suite['id']} is missing")
            continue
        command = commands[index]
        label = f"IPS-056 release suite {expected_suite['id']}"
        _closed_keys(
            command,
            (
                "id",
                "suite_origin",
                "argv",
                "cwd",
                "timeout_seconds",
                "duration_ns",
                "capture_status",
                "exit_code",
                "collected_count",
                "collection_complete",
                "outcome_counts",
                "non_pass_nodes",
                "summary_line",
                "log_offset",
                "log_bytes",
                "log_sha256",
                "assurance",
                "acceptance_status",
            ),
            label,
            errors,
        )
        for field in ("id", "suite_origin", "argv", "cwd", "timeout_seconds"):
            if command.get(field) != expected_suite[field]:
                errors.append(f"{label}.{field} drifts from the fixed runner suite")
        if not isinstance(command.get("duration_ns"), int) or command.get("duration_ns", 0) <= 0:
            errors.append(f"{label}.duration_ns must be measured")
        retained, offset = _validate_release_log_slice(command, raw_log, offset, label, errors)
        text = _ANSI_ESCAPE.sub("", retained.decode("utf-8", "replace"))
        summary = command.get("summary_line")
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        if not isinstance(summary, str) or not nonempty or nonempty[-1] != summary:
            errors.append(f"{label}.summary_line is not the final retained output line")
            parsed = {field: 0 for field in BASELINE_OUTCOME_FIELDS}
        else:
            parsed = _summary_counts(summary)
        if command.get("outcome_counts") != parsed:
            errors.append(f"{label}.outcome_counts do not parse from retained output")
        collected = _collection_count(text)
        complete = _collection_complete(text)
        if command.get("collected_count") != collected or command.get("collection_complete") != complete:
            errors.append(f"{label} collection metadata does not parse from retained output")
        if command.get("non_pass_nodes") != _nonpass_nodes(text):
            errors.append(f"{label}.non_pass_nodes do not parse from retained output")
        if command.get("assurance") != "process_observed_only":
            errors.append(f"{label}.assurance must not claim cryptographic execution proof")
        acceptance_status = _release_acceptance_status(expected_suite, command)
        if command.get("acceptance_status") != acceptance_status:
            errors.append(f"{label}.acceptance_status does not derive from protected evidence")
        if acceptance_status == "regressed":
            errors.append(f"{label} regressed relative to its protected acceptance policy")
        elif acceptance_status == "baseline_compatible_non_green":
            baseline_non_green_ids.append(str(expected_suite["id"]))
    if offset != len(raw_log):
        errors.append("IPS-056 release log contains unbound trailing or missing bytes")
    report = _require_nonempty_file(
        RELEASE_REPORT,
        errors,
        maximum_bytes=RELEASE_MAX_REPORT_BYTES,
        bound_label="one-MiB",
    )
    if RELEASE_REPORT_REQUEST_MARKER in report:
        errors.append("IPS-056 completed report retains its materialization request marker")
    folded = report.casefold()
    required_report_terms = (
        RELEASE_VALIDATION_SCHEMA,
        str(declared_digest or ""),
        *revisions.values(),
        "existing zk systems",
        "real proving",
        "simulated",
        "structural validation",
        "direct execution proof",
        "proof-unit granularity",
        "complete cache key",
        "invalidation rules",
        "full-proof fallback",
        "merkle manifest aggregation",
        "40-transition benchmark",
        "average proof reuse rate",
        "average proving-compute reduction",
        "best incremental case",
        "worst incremental case",
        "proof size",
        "seal size",
        "verification latency",
        "storage overhead",
        "crash-recovery results",
        "tamper-test results",
        "trusted signed receipts",
        "integrity commitments",
        "remaining work before production use",
        RELEASE_PUBLIC_LOG_POLICY,
        "live ipfs was refused before release suite execution",
        "three new incremental-sealing suites require fully green execution",
        "pytest process outputs were observed but test execution was not cryptographically proven",
        "repository verification was decomposed into content-addressed proof units",
        "stale or simulated evidence",
    )
    for term in required_report_terms:
        if term and term.casefold() not in folded:
            errors.append(f"IPS-056 final report omits {term!r}")
    for command_id in baseline_non_green_ids:
        if command_id.casefold() not in folded:
            errors.append(
                f"IPS-056 final report omits remaining baseline issue {command_id!r}"
            )
    if baseline_non_green_ids and "baseline_compatible_non_green" not in folded:
        errors.append(
            "IPS-056 final report must label retained baseline issues "
            "baseline_compatible_non_green"
        )


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_OBSERVED_PROCESS_LOCK = threading.Lock()


def _linux_child_subreaper_state() -> bool:
    if not sys.platform.startswith("linux") or not Path("/proc/self/task").is_dir():
        raise OSError(errno.ENOTSUP, "Linux /proc child tracking is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    current = ctypes.c_int()
    if libc.prctl(
        _PR_GET_CHILD_SUBREAPER,
        ctypes.byref(current),
        0,
        0,
        0,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return bool(current.value)


def _set_linux_child_subreaper(enabled: bool) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(
        _PR_SET_CHILD_SUBREAPER,
        int(enabled),
        0,
        0,
        0,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _proc_identity(pid: int) -> tuple[int, str] | None:
    """Return one Linux PID's immutable start tick and current state."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    if len(fields) < 20:
        raise OSError(errno.EIO, f"cannot parse /proc/{pid}/stat")
    return int(fields[19]), fields[0]


def _proc_children(pid: int) -> set[int]:
    """Read children for every thread in a Linux process."""

    children: set[int] = set()
    task_root = Path(f"/proc/{pid}/task")
    try:
        task_ids = sorted(task_root.iterdir(), key=lambda item: item.name)
    except (FileNotFoundError, ProcessLookupError):
        return children
    for task in task_ids:
        try:
            raw = (task / "children").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        for value in raw.split():
            if value.isdecimal():
                children.add(int(value))
    return children


class _ObservedDescendants:
    """Track one runner lineage across sessions and double-fork adoption."""

    def __init__(self) -> None:
        self.owner_pid = os.getpid()
        self.root_pid: int | None = None
        self.known: dict[int, int] = {}
        self.error: str | None = None
        self.baseline_children = {
            (pid, identity[0])
            for pid in _proc_children(self.owner_pid)
            if (identity := _proc_identity(pid)) is not None
        }

    def bind_root(self, pid: int) -> None:
        self.root_pid = pid
        identity = _proc_identity(pid)
        if identity is not None:
            self.known[pid] = identity[0]
        self.observe()

    def _remember(self, pid: int, queue: deque[int]) -> None:
        identity = _proc_identity(pid)
        if identity is None:
            return
        start_tick, _state = identity
        if self.known.get(pid) == start_tick:
            return
        self.known[pid] = start_tick
        queue.append(pid)
        if len(self.known) > RUNNER_MAX_DESCENDANT_PROCESSES:
            self.error = (
                "observed process tree exceeded its fixed descendant bound"
            )

    def observe(self) -> None:
        queue: deque[int] = deque()
        for pid, start_tick in tuple(self.known.items()):
            identity = _proc_identity(pid)
            if identity is not None and identity[0] == start_tick:
                queue.append(pid)
        for pid in _proc_children(self.owner_pid):
            identity = _proc_identity(pid)
            if identity is None:
                continue
            if (pid, identity[0]) not in self.baseline_children:
                self._remember(pid, queue)
        visited: set[tuple[int, int]] = set()
        while queue:
            parent = queue.popleft()
            parent_identity = _proc_identity(parent)
            if parent_identity is None:
                continue
            marker = (parent, parent_identity[0])
            if marker in visited:
                continue
            visited.add(marker)
            for child in _proc_children(parent):
                self._remember(child, queue)

    def reap_adopted(self) -> None:
        for pid in tuple(self.known):
            if pid == self.root_pid:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                continue

    def live(self) -> dict[int, int]:
        self.observe()
        result: dict[int, int] = {}
        for pid, start_tick in self.known.items():
            identity = _proc_identity(pid)
            if identity is not None and identity[0] == start_tick and identity[1] != "Z":
                result[pid] = start_tick
        return result

    def signal(self, signal_number: int) -> None:
        self.observe()
        for pid, start_tick in sorted(self.live().items(), reverse=True):
            identity = _proc_identity(pid)
            if identity is None or identity[0] != start_tick:
                continue
            try:
                os.kill(pid, signal_number)
            except (ProcessLookupError, PermissionError):
                continue


def _signal_process_group(process_group: int, signal_number: int) -> None:
    try:
        os.killpg(process_group, signal_number)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate_observed_process_tree(
    process: subprocess.Popen[bytes],
    process_group: int,
    descendants: _ObservedDescendants,
) -> bool:
    """Terminate the session and every adopted descendant, then reap them."""

    for signal_number, grace_seconds in (
        (signal.SIGTERM, 1.0),
        (signal.SIGKILL, 2.0),
    ):
        _signal_process_group(process_group, signal_number)
        descendants.signal(signal_number)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()
            descendants.reap_adopted()
            descendants.observe()
            if (
                process.poll() is not None
                and not descendants.live()
                and not _process_group_exists(process_group)
            ):
                return descendants.error is None
            descendants.signal(signal_number)
            time.sleep(0.01)
    process.poll()
    descendants.reap_adopted()
    descendants.observe()
    return (
        descendants.error is None
        and process.poll() is not None
        and not descendants.live()
        and not _process_group_exists(process_group)
    )


def _run_observed_process(
    argv: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    maximum_output_bytes: int = RELEASE_PROCESS_MAX_OUTPUT_BYTES,
) -> tuple[str, int | None, int, bytes]:
    with _OBSERVED_PROCESS_LOCK:
        return _run_observed_process_serialized(
            argv,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
            maximum_output_bytes=maximum_output_bytes,
        )


def _run_observed_process_serialized(
    argv: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    maximum_output_bytes: int,
) -> tuple[str, int | None, int, bytes]:
    started = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    descendants: _ObservedDescendants | None = None
    process_group: int | None = None
    output = bytearray()
    status = "launch_failed"
    exit_code: int | None = None
    previous_subreaper = False
    subreaper_configured = False
    try:
        previous_subreaper = _linux_child_subreaper_state()
        _set_linux_child_subreaper(True)
        subreaper_configured = True
        descendants = _ObservedDescendants()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        process_group = process.pid
        descendants.bind_root(process.pid)
        try:
            observed_process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            observed_process_group = process_group
        if observed_process_group != process_group:
            raise OSError("observed process does not own its dedicated process group")
        os.set_blocking(process.stdout.fileno(), False)
        deadline = time.monotonic() + timeout_seconds
        status = "completed"
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                descendants.observe()
                if descendants.error:
                    status = "cleanup_failed"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timed_out"
                    break
                events = selector.select(min(0.25, remaining))
                if not events and process.poll() is not None:
                    events = selector.select(0)
                    if not events:
                        break
                for key, _ in events:
                    try:
                        chunk = os.read(key.fd, 64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining_capacity = maximum_output_bytes - len(output)
                    output.extend(chunk[: max(0, remaining_capacity)])
                    if len(chunk) > remaining_capacity:
                        status = "output_limit"
                        break
                if status != "completed":
                    break
        if status != "completed":
            _terminate_observed_process_tree(
                process, process_group, descendants
            )
        elif process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timed_out"
            else:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    status = "timed_out"
        parent_exit_code = process.returncode
        descendants.reap_adopted()
        residual_processes = status == "completed" and (
            bool(descendants.live()) or _process_group_exists(process_group)
        )
        if not _terminate_observed_process_tree(
            process, process_group, descendants
        ):
            status = "cleanup_failed"
        elif residual_processes:
            status = "residual_process_terminated"
        exit_code = parent_exit_code if status == "completed" else None
    except OSError as exc:
        status = "launch_failed"
        exit_code = None
        output = bytearray(f"{type(exc).__name__}: {exc}\n".encode("utf-8", "replace"))
    finally:
        if (
            process is not None
            and descendants is not None
            and process_group is not None
            and not _terminate_observed_process_tree(
                process, process_group, descendants
            )
        ):
            status = "cleanup_failed"
            exit_code = None
        if subreaper_configured:
            try:
                _set_linux_child_subreaper(previous_subreaper)
            except OSError as exc:
                status = "cleanup_failed"
                exit_code = None
                diagnostic = f"subreaper restore failed: {type(exc).__name__}\n".encode()
                capacity = max(0, maximum_output_bytes - len(output))
                output.extend(diagnostic[:capacity])
    return status, exit_code, max(1, time.monotonic_ns() - started), bytes(output)


def _atomic_write_artifact(relative: str, raw: bytes) -> None:
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or candidate.as_posix() != relative
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise OSError(f"refusing unsafe artifact path {relative!r}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    held_directories: list[int] = []
    descriptor = -1
    temporary_name = (
        f".{candidate.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        held_directories.append(os.open(REPO_ROOT, directory_flags))
        for part in candidate.parts[:-1]:
            held_directories.append(
                os.open(part, directory_flags, dir_fd=held_directories[-1])
            )
        parent_descriptor = held_directories[-1]
        try:
            existing = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise OSError(f"refusing to replace non-regular output {relative}")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary_name,
            candidate.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if held_directories:
            try:
                os.unlink(temporary_name, dir_fd=held_directories[-1])
            except FileNotFoundError:
                pass
            for directory_descriptor in reversed(held_directories):
                os.close(directory_descriptor)


def _materialization_request_state(
    expected: Mapping[str, bytes],
    *,
    label: str,
    errors: list[str],
) -> str:
    """Classify a declared bundle as absent, requested, complete-candidate, or invalid."""

    present = {relative: os.path.lexists(REPO_ROOT / relative) for relative in expected}
    if not any(present.values()):
        return "absent"
    if not all(present.values()):
        errors.append(f"{label} output bundle is partial")
        return "invalid"
    request_errors: list[str] = []
    for relative, exact in expected.items():
        retained = _secure_read_repo_file(
            relative,
            required_parent=Path(relative).parent.as_posix(),
            label=f"{label} request {relative}",
            maximum_bytes=max(4096, len(exact)),
            bound_label="4-KiB request",
            errors=request_errors,
        )
        if retained is None or retained[0] != exact:
            request_errors.append(f"{label} request bytes drifted for {relative}")
    if not request_errors:
        return "requested"
    return "complete-candidate"


def _consume_materialization_request(
    expected: Mapping[str, bytes], label: str, errors: list[str]
) -> bool:
    """Remove only an already byte-validated, exact declared request bundle."""

    for relative, exact in expected.items():
        retained = _secure_read_repo_file(
            relative,
            required_parent=Path(relative).parent.as_posix(),
            label=f"{label} request {relative}",
            maximum_bytes=max(4096, len(exact)),
            bound_label="4-KiB request",
            errors=errors,
        )
        if retained is None or retained[0] != exact:
            errors.append(f"{label} refuses a noncanonical request at {relative}")
    if errors:
        return False
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for relative, exact in expected.items():
        candidate = Path(relative)
        held_directories: list[int] = []
        descriptor = -1
        try:
            held_directories.append(os.open(REPO_ROOT, directory_flags))
            for part in candidate.parts[:-1]:
                held_directories.append(
                    os.open(part, directory_flags, dir_fd=held_directories[-1])
                )
            descriptor = os.open(
                candidate.name,
                file_flags,
                dir_fd=held_directories[-1],
            )
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != len(exact):
                raise OSError("materialization request identity or size drifted")
            raw = bytearray()
            while len(raw) <= len(exact):
                chunk = os.read(descriptor, len(exact) + 1 - len(raw))
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(descriptor)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if identity != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or bytes(raw) != exact:
                raise OSError("materialization request changed before consumption")
            path_after = os.stat(
                candidate.name,
                dir_fd=held_directories[-1],
                follow_symlinks=False,
            )
            if identity[:4] != (
                path_after.st_dev,
                path_after.st_ino,
                path_after.st_mode,
                path_after.st_size,
            ):
                raise OSError("materialization request path identity changed")
            os.unlink(candidate.name, dir_fd=held_directories[-1])
        except OSError as exc:
            errors.append(
                f"{label} cannot consume request {relative}: {type(exc).__name__}"
            )
            return False
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            for directory_descriptor in reversed(held_directories):
                os.close(directory_descriptor)
    return True


def _remove_release_work_contents(
    descriptor: int,
    *,
    relative: PurePosixPath,
    depth: int,
    visited: list[int],
    errors: list[str],
) -> None:
    """Unlink one held directory without following or chmodding any leaf."""

    if depth > RUNNER_MATERIALIZATION_MAX_DEPTH:
        errors.append("release-work cleanup exceeded its directory-depth bound")
        return
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        errors.append(
            f"cannot enumerate fixed release-work directory {relative}: "
            f"{type(exc).__name__}"
        )
        return
    for name in names:
        entry_error_count = len(errors)
        visited[0] += 1
        child_relative = relative / name
        if visited[0] > RUNNER_MATERIALIZATION_MAX_ENTRIES:
            errors.append("release-work cleanup exceeded its entry bound")
            return
        child_descriptor = -1
        try:
            child_descriptor = os.open(
                name, directory_flags, dir_fd=descriptor
            )
        except OSError as exc:
            if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
                errors.append(
                    f"cannot inspect fixed release-work entry {child_relative}: "
                    f"{type(exc).__name__}"
                )
                continue
            try:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    errors.append(
                        f"fixed release-work entry changed during cleanup: "
                        f"{child_relative}"
                    )
                    continue
                os.unlink(name, dir_fd=descriptor)
            except OSError as unlink_error:
                errors.append(
                    f"cannot unlink fixed release-work leaf {child_relative}: "
                    f"{type(unlink_error).__name__}"
                )
            continue
        try:
            os.fchmod(child_descriptor, 0o700)
            _remove_release_work_contents(
                child_descriptor,
                relative=child_relative,
                depth=depth + 1,
                visited=visited,
                errors=errors,
            )
        except OSError as exc:
            errors.append(
                f"cannot prepare fixed release-work directory {child_relative}: "
                f"{type(exc).__name__}"
            )
        finally:
            os.close(child_descriptor)
        if len(errors) != entry_error_count:
            continue
        try:
            os.rmdir(name, dir_fd=descriptor)
        except OSError as exc:
            errors.append(
                f"cannot remove fixed release-work directory {child_relative}: "
                f"{type(exc).__name__}"
            )


def _clean_release_work_root(errors: list[str]) -> None:
    """Remove only the fixed transient tree through held, no-follow dirfds."""

    relative = PurePosixPath(RELEASE_WORK_ROOT)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    held_directories: list[int] = []
    target_descriptor = -1
    initial_error_count = len(errors)
    try:
        held_directories.append(os.open(REPO_ROOT, directory_flags))
        for part in relative.parts[:-1]:
            try:
                held_directories.append(
                    os.open(part, directory_flags, dir_fd=held_directories[-1])
                )
            except FileNotFoundError:
                return
        try:
            target_descriptor = os.open(
                relative.name,
                directory_flags,
                dir_fd=held_directories[-1],
            )
        except FileNotFoundError:
            return
        os.fchmod(target_descriptor, 0o700)
        _remove_release_work_contents(
            target_descriptor,
            relative=relative,
            depth=0,
            visited=[0],
            errors=errors,
        )
        if len(errors) != initial_error_count:
            return
        os.close(target_descriptor)
        target_descriptor = -1
        os.rmdir(relative.name, dir_fd=held_directories[-1])
    except OSError as exc:
        errors.append(
            f"cannot clean fixed release-work directory: {type(exc).__name__}"
        )
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
        for directory_descriptor in reversed(held_directories):
            os.close(directory_descriptor)


def _release_report_has_exact_request(errors: list[str]) -> bool:
    retained = _secure_read_repo_file(
        RELEASE_REPORT,
        required_parent=Path(RELEASE_REPORT).parent.as_posix(),
        label="IPS-056 report request",
        maximum_bytes=RELEASE_MAX_REPORT_BYTES,
        bound_label="one-MiB",
        errors=errors,
    )
    if retained is None:
        return False
    try:
        text = retained[0].decode("utf-8")
    except UnicodeError:
        errors.append("IPS-056 report request is not UTF-8")
        return False
    if text.count(RELEASE_REPORT_REQUEST_MARKER) != 1:
        errors.append("IPS-056 report must contain exactly one release-evidence request marker")
        return False
    return True


def _materialize_release_report_binding(
    receipt: Mapping[str, Any], errors: list[str]
) -> None:
    """Replace only the closed report marker with the freshly observed binding."""

    retained = _secure_read_repo_file(
        RELEASE_REPORT,
        required_parent=Path(RELEASE_REPORT).parent.as_posix(),
        label="IPS-056 report materialization request",
        maximum_bytes=RELEASE_MAX_REPORT_BYTES,
        bound_label="one-MiB",
        errors=errors,
    )
    if retained is None:
        return
    try:
        text = retained[0].decode("utf-8")
    except UnicodeError:
        errors.append("IPS-056 report materialization request is not UTF-8")
        return
    if text.count(RELEASE_REPORT_REQUEST_MARKER) != 1:
        errors.append("IPS-056 report request marker is missing or duplicated")
        return
    revisions = receipt.get("source_revisions")
    if not isinstance(revisions, Mapping) or set(revisions) != set(REPOSITORY_PATHS):
        errors.append("IPS-056 report cannot bind an incomplete source revision map")
        return
    digest = receipt.get("receipt_digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        errors.append("IPS-056 report cannot bind an invalid receipt digest")
        return
    retained_baseline_ids = [
        str(command.get("id"))
        for command in receipt.get("pytest_commands", ())
        if isinstance(command, Mapping)
        and command.get("acceptance_status") == "baseline_compatible_non_green"
    ]
    retained_baseline_value = ",".join(retained_baseline_ids) or "none"
    replacement = "\n".join(
        (
            "<!-- IPS-056 RELEASE EVIDENCE (materialized by protected runner)",
            f"receipt_digest: {digest}",
            f"accelerate_revision: {revisions['accelerate']}",
            f"datasets_revision: {revisions['datasets']}",
            f"kit_revision: {revisions['kit']}",
            f"baseline_compatible_non_green: {retained_baseline_value}",
            "-->",
        )
    )
    _atomic_write_artifact(
        RELEASE_REPORT,
        text.replace(RELEASE_REPORT_REQUEST_MARKER, replacement).encode("utf-8"),
    )


def run_benchmark_validation() -> dict[str, Any]:
    """Convergently ensure fixed benchmark evidence, then validate read-only."""

    errors: list[str] = []
    requests = {
        BENCHMARK_JSON: BENCHMARK_REQUEST_JSON,
        BENCHMARK_CSV: BENCHMARK_REQUEST_CSV,
    }
    request_state = _materialization_request_state(
        requests, label="IPS-053 benchmark", errors=errors
    )
    if request_state == "complete-candidate":
        source_binding = _capture_runner_source_binding("benchmark", errors)
        if errors:
            return {
                "valid": False,
                "runner": "benchmark",
                "materialization": "invalid_preexisting_bundle",
                "errors": errors,
            }
        checked = validate_artifact("IPS-053")
        errors.extend(str(item) for item in checked["errors"])
        _verify_runner_source_binding("benchmark", source_binding, errors)
        if checked["valid"] and not errors:
            return {
                "valid": True,
                "runner": "benchmark",
                "materialization": "already_complete_read_only",
                "errors": [],
            }
        return {
            "valid": False,
            "runner": "benchmark",
            "materialization": "invalid_preexisting_bundle",
            "errors": errors,
        }
    if request_state == "invalid":
        return {"valid": False, "runner": "benchmark", "errors": errors}
    if request_state == "absent":
        errors.append("IPS-053 benchmark exact two-file materialization request is absent")
        return {"valid": False, "runner": "benchmark", "errors": errors}
    cli = REPO_ROOT / BENCHMARK_CLI
    if not cli.is_file() or cli.is_symlink():
        errors.append("protected benchmark runner cannot find a regular IPS-052 CLI")
        return {"valid": False, "runner": "benchmark", "errors": errors}
    source_binding = _capture_runner_source_binding("benchmark", errors)
    if errors:
        return {"valid": False, "runner": "benchmark", "errors": errors}
    _clean_release_work_root(errors)
    _verify_runner_source_binding("benchmark", source_binding, errors)
    if errors:
        return {"valid": False, "runner": "benchmark", "errors": errors}
    materialized = _materialize_runner_source("benchmark", source_binding, errors)
    _verify_runner_source_binding("benchmark", source_binding, errors)
    if materialized is None or errors:
        _clean_release_work_root(errors)
        return {"valid": False, "runner": "benchmark", "errors": errors}
    source_root, stage_root, materialized_digests = materialized
    if not _consume_materialization_request(
        requests, "IPS-053 benchmark", errors
    ):
        _clean_release_work_root(errors)
        return {"valid": False, "runner": "benchmark", "errors": errors}
    workspace = source_root.parent / "runtime" / "benchmark"
    environment = _release_environment(workspace, source_root=source_root)
    status, exit_code, duration_ns, output = _run_observed_process(
        _benchmark_workload_argv(),
        cwd=source_root,
        environment=environment,
        timeout_seconds=10_200,
    )
    if status != "completed" or exit_code != 0:
        errors.append(
            f"fixed benchmark process did not complete successfully: {status}/{exit_code}"
        )
        diagnostic = output.decode("utf-8", "replace")[-4000:]
        if diagnostic:
            errors.append(f"benchmark diagnostic tail: {diagnostic}")
    _verify_materialized_source(
        source_root,
        "benchmark",
        source_binding,
        errors,
        expected_digests=materialized_digests,
    )
    _verify_runner_source_binding("benchmark", source_binding, errors)
    if not errors:
        try:
            _publish_staged_benchmark_outputs(stage_root, errors)
        except OSError as exc:
            errors.append(
                f"cannot publish staged benchmark outputs: {type(exc).__name__}"
            )
    if not errors:
        checked = validate_artifact("IPS-053")
        errors.extend(str(item) for item in checked["errors"])
    _clean_release_work_root(errors)
    _verify_runner_source_binding("benchmark", source_binding, errors)
    return {
        "valid": not errors,
        "runner": "benchmark",
        "materialization": "materialized_once",
        "process": {
            "argv": _benchmark_workload_argv(),
            "capture_status": status,
            "exit_code": exit_code,
            "duration_ns": duration_ns,
        },
        "errors": errors,
    }


def _pytest_observation(
    spec: Mapping[str, Any],
    *,
    offset: int,
    output: bytes,
    status: str,
    exit_code: int | None,
    duration_ns: int,
) -> dict[str, Any]:
    text = _ANSI_ESCAPE.sub("", output.decode("utf-8", "replace"))
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    summary = nonempty[-1] if nonempty else ""
    observation = {
        key: spec.get(key)
        for key in ("id", "suite_origin", "argv", "cwd", "timeout_seconds")
    }
    observation.update({
        "duration_ns": duration_ns,
        "capture_status": status,
        "exit_code": exit_code,
        "collected_count": _collection_count(text),
        "collection_complete": _collection_complete(text),
        "outcome_counts": _summary_counts(summary),
        "non_pass_nodes": _nonpass_nodes(text),
        "summary_line": summary,
        "log_offset": offset,
        "log_bytes": len(output),
        "log_sha256": "sha256:" + hashlib.sha256(output).hexdigest(),
        "assurance": "process_observed_only",
    })
    observation["acceptance_status"] = _release_acceptance_status(spec, observation)
    return observation


def run_release_validation() -> dict[str, Any]:
    """Convergently ensure release observations, then validate read-only."""

    errors: list[str] = []
    requests = {
        RELEASE_VALIDATION_JSON: RELEASE_REQUEST_JSON,
        RELEASE_VALIDATION_LOG: RELEASE_REQUEST_LOG,
    }
    request_state = _materialization_request_state(
        requests, label="IPS-056 release", errors=errors
    )
    if request_state == "complete-candidate":
        source_binding = _capture_runner_source_binding("release", errors)
        if errors:
            return {
                "valid": False,
                "runner": "release",
                "materialization": "invalid_preexisting_bundle",
                "errors": errors,
            }
        checked = validate_artifact("IPS-056")
        errors.extend(str(item) for item in checked["errors"])
        _verify_runner_source_binding("release", source_binding, errors)
        if checked["valid"] and not errors:
            return {
                "valid": True,
                "runner": "release",
                "materialization": "already_complete_read_only",
                "errors": [],
            }
        return {
            "valid": False,
            "runner": "release",
            "materialization": "invalid_preexisting_bundle",
            "errors": errors,
        }
    if request_state == "invalid":
        return {"valid": False, "runner": "release", "errors": errors}
    if request_state == "absent":
        errors.append("IPS-056 release exact JSON/log materialization request is absent")
        return {"valid": False, "runner": "release", "errors": errors}
    if not _release_report_has_exact_request(errors):
        return {"valid": False, "runner": "release", "errors": errors}
    source_binding = _capture_runner_source_binding("release", errors)
    if errors:
        return {"valid": False, "runner": "release", "errors": errors}
    _clean_release_work_root(errors)
    revisions_before, trees_before = _current_repository_bindings(errors)
    specs = _release_suite_specs(errors)
    _validate_release_ipfs_preflight(errors)
    _verify_runner_source_binding("release", source_binding, errors)
    if errors:
        _clean_release_work_root(errors)
        return {"valid": False, "runner": "release", "errors": errors}
    materialized = _materialize_runner_source("release", source_binding, errors)
    _verify_runner_source_binding("release", source_binding, errors)
    if materialized is None or errors:
        _clean_release_work_root(errors)
        return {"valid": False, "runner": "release", "errors": errors}
    source_root, stage_root, materialized_digests = materialized
    if not _consume_materialization_request(
        requests, "IPS-056 release", errors
    ):
        _clean_release_work_root(errors)
        return {"valid": False, "runner": "release", "errors": errors}
    work_root = source_root.parent / "runtime"
    combined = bytearray()
    terminal_argv = [
        sys.executable,
        "scripts/validate_incremental_proof_sealer_board.py",
        "--check-terminal",
    ]
    terminal_status, terminal_exit, terminal_duration, terminal_output = _run_observed_process(
        terminal_argv,
        cwd=source_root,
        environment=_release_environment(
            work_root / "terminal-board-gate", source_root=source_root
        ),
        timeout_seconds=120,
    )
    terminal = {
        "id": "terminal-board-gate",
        "argv": terminal_argv,
        "cwd": ".",
        "timeout_seconds": 120,
        "duration_ns": terminal_duration,
        "capture_status": terminal_status,
        "exit_code": terminal_exit,
        "log_offset": 0,
        "log_bytes": len(terminal_output),
        "log_sha256": "sha256:" + hashlib.sha256(terminal_output).hexdigest(),
    }
    combined.extend(terminal_output)
    commands: list[dict[str, Any]] = []
    _verify_runner_source_binding("release", source_binding, errors)
    for spec in specs:
        if errors:
            break
        workspace = work_root / str(spec["id"])
        status, exit_code, duration_ns, output = _run_observed_process(
            list(spec["argv"]),
            cwd=source_root / str(spec["cwd"]),
            environment=_release_environment(workspace, source_root=source_root),
            timeout_seconds=int(spec["timeout_seconds"]),
        )
        commands.append(
            _pytest_observation(
                spec,
                offset=len(combined),
                output=output,
                status=status,
                exit_code=exit_code,
                duration_ns=duration_ns,
            )
        )
        combined.extend(output)
        _verify_runner_source_binding("release", source_binding, errors)
    _verify_materialized_source(
        source_root,
        "release",
        source_binding,
        errors,
        expected_digests=materialized_digests,
    )
    try:
        with os.scandir(stage_root) as stage_entries:
            if next(stage_entries, None) is not None:
                errors.append(
                    "protected release runner staged output channel is not empty"
                )
    except OSError as exc:
        errors.append(
            f"protected release runner cannot inspect staged output channel: "
            f"{type(exc).__name__}"
        )
    revisions_after, trees_after = _current_repository_bindings(errors)
    if revisions_after != revisions_before or trees_after != trees_before:
        errors.append("source revisions or trees changed during release validation")
    body: dict[str, Any] = {
        "schema_version": RELEASE_VALIDATION_SCHEMA,
        "runner_id": RELEASE_RUNNER_ID,
        "validation_worktree_parent_revision": revisions_before.get("accelerate"),
        "source_revisions": revisions_before,
        "source_trees": trees_before,
        "environment_policy_id": RELEASE_ENVIRONMENT_POLICY,
        "terminal_gate": terminal,
        "pytest_commands": commands,
        "retained_log": {
            "path": RELEASE_VALIDATION_LOG,
            "bytes": len(combined),
            "sha256": "sha256:" + hashlib.sha256(combined).hexdigest(),
        },
        "assurance": _release_assurance(),
    }
    body["receipt_digest"] = "sha256:" + hashlib.sha256(
        _canonical_json_bytes(body)
    ).hexdigest()
    canonical_receipt = _canonical_json_bytes(body) + b"\n"
    retained_report = _secure_read_repo_file(
        RELEASE_REPORT,
        required_parent=Path(RELEASE_REPORT).parent.as_posix(),
        label="IPS-056 report before evidence binding",
        maximum_bytes=RELEASE_MAX_REPORT_BYTES,
        bound_label="one-MiB",
        errors=errors,
    )
    if len(combined) > RELEASE_MAX_LOG_BYTES:
        errors.append("IPS-056 combined retained log exceeds the fixed six-MiB bound")
    if len(canonical_receipt) > BASELINE_MAX_RECEIPT_BYTES:
        errors.append("IPS-056 release receipt exceeds the fixed two-MiB bound")
    if retained_report is not None:
        projected_report_bytes = (
            len(retained_report[0])
            - len(RELEASE_REPORT_REQUEST_MARKER.encode("utf-8"))
            + 1024
        )
        if projected_report_bytes > RELEASE_MAX_REPORT_BYTES:
            errors.append("IPS-056 materialized report would exceed the one-MiB bound")
        if len(combined) + len(canonical_receipt) + projected_report_bytes > 12_000_000:
            errors.append("IPS-056 declared evidence exceeds its 12-MiB patch envelope")
    _validate_release_public_log(bytes(combined), errors)
    _verify_runner_source_binding("release", source_binding, errors)
    if not errors:
        try:
            _atomic_write_artifact(RELEASE_VALIDATION_LOG, bytes(combined))
            _atomic_write_artifact(RELEASE_VALIDATION_JSON, canonical_receipt)
            _materialize_release_report_binding(body, errors)
        except OSError as exc:
            errors.append(f"cannot publish release validation artifacts: {type(exc).__name__}")
    if not errors:
        checked = validate_artifact("IPS-056")
        errors.extend(str(item) for item in checked["errors"])
    _clean_release_work_root(errors)
    _verify_runner_source_binding("release", source_binding, errors)
    return {
        "valid": not errors,
        "runner": "release",
        "materialization": "materialized_once",
        "receipt_digest": body.get("receipt_digest"),
        "errors": errors,
    }
def _validate_published_inventory_artifacts(errors: list[str]) -> None:
    """Revalidate every completed pre-synthesis inventory transaction."""

    current = _git_stdout(
        REPO_ROOT, errors, "resolve committed inventory control HEAD", "rev-parse", "HEAD"
    )
    if not _HEX_40.fullmatch(current):
        return
    taskboard = _git_text_at_revision(
        current,
        "docs/architecture/incremental_proof_sealer.todo.md",
        errors,
        "current inventory taskboard",
    )
    records = _parse_markdown_records(
        taskboard,
        re.compile(r"^## (IPS-\d{3})\s+([^\n]+)$", re.MULTILINE),
        "current inventory task",
        errors,
    )
    for task_id, spec in BASELINE_RECEIPT_SPECS.items():
        status = (
            records.get(task_id, {})
            .get("fields", {})
            .get("status", "")
            .casefold()
        )
        if status != "completed":
            continue
        outputs = {str(spec[field]) for field in ("inventory", "report")}
        missing = sorted(
            relative
            for relative in outputs
            if not _task_output_exists_at_control_revision(current, relative)
        )
        if missing:
            errors.append(
                f"{task_id} is completed but committed outputs are missing: {missing}"
            )
            continue
        result = validate_artifact(task_id, require_published=True)
        errors.extend(str(item) for item in result.get("errors", ()))


def validate_artifact(
    task_id: str,
    *,
    require_published: bool = False,
) -> dict[str, Any]:
    """Validate the bounded data/document tasks that cannot use inline eval.

    The implementation supervisor deliberately rejects ``python -c`` and
    other dynamic-eval validation commands.  These bounded, standard-library
    checks retain the reviewed assertions as a normal executable entry point.
    """

    task_id = str(task_id or "").strip().upper()
    errors: list[str] = []
    _validate_no_capture_lock(errors)
    if task_id in BASELINE_RECEIPT_SPECS:
        spec = BASELINE_RECEIPT_SPECS[task_id]
        relative = str(spec["inventory"])
        revision = str(spec["revision"])
        receipt = _validate_baseline_receipt(task_id, spec, errors)
        config = _load_json(CONFIG_PATH, errors)
        _validate_config(config, errors)
        configured_receipts = config.get("operator_baseline_receipts")
        configured_pin = (
            configured_receipts.get(task_id)
            if isinstance(configured_receipts, Mapping)
            else None
        )
        if not isinstance(configured_pin, Mapping):
            errors.append(f"{task_id} has no protected operator receipt pin")
            configured_pin = {}
        expected_pin = _expected_receipt_pin(spec, receipt)
        if configured_pin != expected_pin:
            errors.append(f"{task_id} protected operator receipt pin is not exact")

        payload = _artifact_json(relative, errors)
        serialized = json.dumps(payload, sort_keys=True)
        forbidden_self_commit_fields = {
            "completioncommit",
            "finalrevision",
            "finaltaskcommit",
            "inventorycommit",
            "outputcommit",
            "taskcommit",
        }
        embedded_self_commits = {
            str(key)
            for key in payload
            if _normalized_field_name(key) in forbidden_self_commit_fields
        }
        if embedded_self_commits:
            errors.append(
                f"{task_id} inventory may not self-embed a future completion commit: "
                f"{sorted(embedded_self_commits)}"
            )
        if payload.get("planning_revision") != revision:
            errors.append(f"{task_id} planning_revision does not match {revision}")
        parent_revision = payload.get("inventory_worktree_parent_revision")
        if not isinstance(parent_revision, str) or not _HEX_40.fullmatch(parent_revision):
            errors.append(
                f"{task_id} inventory_worktree_parent_revision must be a concrete commit"
            )
        else:
            inventory_repository = REPO_ROOT / REPOSITORY_PATHS[str(spec["repository"])]
            source_revision = receipt.get("source_revision")
            parent_tree = _git(
                "rev-parse", f"{parent_revision}^{{tree}}", cwd=inventory_repository
            )
            if parent_tree.returncode != 0:
                errors.append(f"{task_id} inventory_worktree_parent_revision is unknown")
            if isinstance(source_revision, str):
                ancestry = _git(
                    "merge-base",
                    "--is-ancestor",
                    source_revision,
                    parent_revision,
                    cwd=inventory_repository,
                )
                if ancestry.returncode != 0:
                    errors.append(
                        f"{task_id} inventory parent does not descend from tested source"
                    )
            _validate_inventory_source_relevance(
                task_id=task_id,
                spec=spec,
                receipt=receipt,
                parent_revision=parent_revision,
                configured_receipts=(
                    configured_receipts
                    if isinstance(configured_receipts, Mapping)
                    else {}
                ),
                require_published=require_published,
                errors=errors,
            )
        if "repository_commit" in payload and payload.get("repository_commit") != revision:
            errors.append(f"{task_id} repository_commit conflicts with planning_revision")

        baseline_evidence = payload.get("baseline_evidence")
        expected_baseline_evidence = {
            "path": spec["receipt"],
            "receipt_digest": receipt.get("receipt_digest"),
            "required_command_ids": list(spec["command_ids"]),
            "evidence_origin": BASELINE_OPERATOR_ORIGIN,
            "assurance": "process_observed_only",
            "nonclaim": "pytest_execution_not_cryptographically_proven",
        }
        if baseline_evidence != expected_baseline_evidence:
            errors.append(
                f"{task_id} baseline_evidence must be the exact reference-only operator projection"
            )
        for forbidden in (
            "baseline",
            "baselines",
            "baseline_commands",
            "baseline_results",
            "commands",
            "outcome_counts",
            "non_pass_nodes",
            "pytest_results",
        ):
            if forbidden in payload:
                errors.append(f"{task_id} may not copy operator evidence into {forbidden!r}")
        for mapping in _walk_mappings(payload):
            mapping_keys = {str(key).casefold() for key in mapping}
            copied_fields = mapping_keys & {
                "argv",
                "capture_status",
                "duration_ns",
                "exit_code",
                "non_pass_nodes",
                "outcome_counts",
                "summary_line",
            }
            if copied_fields:
                errors.append(
                    f"{task_id} copies protected command evidence fields: {sorted(copied_fields)}"
                )
                break
        _validate_inventory_provenance(task_id, payload, errors)
        _validate_reference_only_inventory_namespace(task_id, payload, errors)

        classifications = payload.get("classifications")
        if not isinstance(classifications, (list, dict)) or not classifications:
            errors.append(f"{task_id} classifications must be non-empty")
        required_terms = {
            "IPS-001": (
                "proof_attestation",
                "proof_reuse_real_groth16_fixture",
                "kernel_verification.py",
                "prover_conformance.py",
                "proof_fallbacks.py",
                "proof_metrics.py",
                "prover_evidence_store.py",
                "manual_completion_seal.py",
                "release_evidence.py",
                "repository_forest",
            ),
            "IPS-002": (
                "cec_zkp_integration.py",
                "cec_proof_cache.py",
                "tdfol_zkp_integration.py",
                "tdfol_proof_cache.py",
                "flogic_zkp_integration.py",
                "flogic_proof_cache.py",
                "event_dag_zkp.py",
                "provekit_ffi.py",
                "wallet/proofs.py",
                "test_execution_certificate.py",
                "test_pass.py",
                "proof_receipt_attestation.py",
                "ensure_setup",
            ),
            "IPS-003": (
                "profile_d_policy.py",
                "mcplusplus/artifacts.py",
                "iroh/release.py",
                "test_joined_release_receipt.py",
                "install_lotus.py",
                "proof_certificate_store.py",
                "event_dag.py",
            ),
        }
        for term in required_terms[task_id]:
            if term not in serialized:
                errors.append(f"{task_id} inventory is missing required surface {term}")
        report = _require_nonempty_file(str(spec["report"]), errors)
        for term in (
            str(spec["receipt"]),
            str(receipt.get("receipt_digest", "")),
            *tuple(str(command_id) for command_id in spec["command_ids"]),
            "process_observed_only",
            "pytest_execution_not_cryptographically_proven",
        ):
            if term and term not in report:
                errors.append(f"{task_id} Markdown inventory omits baseline reference {term!r}")
        report_folded = report.casefold()
        if re.search(
            r"\b\d+\s+(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b",
            report_folded,
        ):
            errors.append(f"{task_id} Markdown inventory copies protected pytest counts")
        if re.search(
            r"\b(?:pytest|tests?|test\s+cases|execution|suite|run)\b.{0,48}?\b"
            r"(?:passed|failed|succeeded|successful|verified|green|red|ok)\b|"
            r"\b(?:passed|failed|succeeded|successful|verified|green|red|ok)\b"
            r".{0,32}?\b(?:pytest|tests?|cases?|execution|suite|run)\b",
            report_folded,
        ):
            errors.append(f"{task_id} Markdown inventory restates operator outcomes")
        if re.search(
            r"\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine)\s+"
            r"(?:passed|failed|successful|green|red|ok)\s+(?:pytest\s+)?(?:tests?|cases?)\b",
            report_folded,
        ):
            errors.append(f"{task_id} Markdown inventory copies provider-owned outcome counts")
        if re.search(
            r"\b(?:(?:passed|failed|error|skipped|selected|xfailed|xpassed)[_-]?count|"
            r"retained[_-]?transcript|command[_-]?line|pytest[_-]?result)\b",
            report_folded,
        ):
            errors.append(f"{task_id} Markdown inventory copies protected evidence aliases")
        if any(
            token in report_folded
            for token in (
                "pending-local-run",
                "plan-derived",
                "plan_derived",
                "declared-for-rerun",
                "declared_for_rerun",
            )
        ):
            errors.append(f"{task_id} Markdown inventory contains placeholder provenance")
        for candidate in re.findall(r"(?:sha256:)?([0-9a-f]{64})", report_folded):
            if _looks_patterned_digest(candidate):
                errors.append(f"{task_id} Markdown inventory contains a patterned fake hash")
    elif task_id == "IPS-004":
        config = _load_json(CONFIG_PATH, errors)
        _validate_config(config, errors)
        receipts = _validate_operator_baseline_bundle(
            config,
            errors,
            enforce_current_sources=True,
        )
        if set(receipts) == set(BASELINE_RECEIPT_SPECS):
            _validated_baseline_synthesis(
                config,
                receipts,
                errors,
                candidate=True,
            )
    elif task_id == "IPS-053":
        _validate_benchmark_artifacts(errors)
    elif task_id == "IPS-054":
        _validate_benchmark_summary(errors)
    elif task_id == "IPS-055":
        _validate_trust_and_migration_docs(errors)
    elif task_id == "IPS-056":
        _validate_release_validation(errors)
    else:
        errors.append(f"unsupported artifact check task: {task_id}")

    return {
        "valid": not errors,
        "check_artifact": task_id,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the IncrementalProofSealer fixed supervisor board; no flag "
            "runs the same phase-aware full gate as --check-all"
        )
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="verify the current preflight baseline or its committed IPS-004 milestone",
    )
    parser.add_argument(
        "--check-bootstrap",
        action="store_true",
        help="verify pristine committed pre-capture controls with exact empty pins",
    )
    parser.add_argument(
        "--check-terminal",
        action="store_true",
        help="verify terminal structure and the historical committed IPS-004 milestone",
    )
    parser.add_argument(
        "--check-artifact",
        choices=ARTIFACT_CHECK_TASKS,
        help="run a bounded non-eval validation for one data/document task",
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="run the fixed 40-transition benchmark argv and validate JSON/CSV",
    )
    parser.add_argument(
        "--run-release-validation",
        action="store_true",
        help="observe the fixed current-tree terminal/pytest suite and validate release artifacts",
    )
    args = parser.parse_args(argv)
    modes = (
        args.check_all,
        args.check_bootstrap,
        args.check_terminal,
        args.check_artifact,
        args.run_benchmark,
        args.run_release_validation,
    )
    if sum(bool(item) for item in modes) > 1:
        parser.error("validation, artifact, benchmark, and release modes are mutually exclusive")
    if args.check_artifact:
        result = validate_artifact(args.check_artifact)
    elif args.check_bootstrap:
        result = validate_bootstrap()
    elif args.run_benchmark:
        result = run_benchmark_validation()
    elif args.run_release_validation:
        result = run_release_validation()
    else:
        result = validate(
            check_all=True,
            check_terminal=args.check_terminal,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
