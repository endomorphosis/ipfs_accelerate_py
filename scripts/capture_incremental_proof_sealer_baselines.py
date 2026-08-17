#!/usr/bin/env python3
"""Capture and validate the fixed IncrementalProofSealer pytest baselines.

This module intentionally has no third-party imports and performs no work at
import time.  ``capture`` runs only the reviewed commands declared below.  It
does not accept arbitrary argv, install or download anything, generate proof
keys, or start network services.  A receipt is an integrity-protected record of
an observed pytest subprocess; it is not a signature or a cryptographic proof
that the tests executed.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import secrets
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "incremental-proof-sealer-baseline-receipt@4"
OPERATOR_ORIGIN = "operator_capture"
ENVIRONMENT_POLICY_ID = "incremental-proof-sealer-controlled-offline-pytest@3"
IGNORED_INPUT_POLICY_ID = "incremental-proof-sealer-clean-materialized-trees@1"
GIT_ENVIRONMENT_POLICY_ID = "incremental-proof-sealer-fixed-git-environment@2"
SUITE_REGISTRY_SCHEMA_VERSION = "incremental-proof-sealer-reviewed-suite-registry@1"
SUITE_REGISTRY_RELATIVE = "config/incremental_proof_sealer_baseline_suite_registry.json"
ARTIFACT_RELATIVE_ROOT = Path(
    "artifacts/agent_supervisor/incremental_proof_sealer/baseline_receipts"
)
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_SUITE_REGISTRY_BYTES = 256 * 1024
MAX_LOG_BYTES = 64 * 1024 * 1024
MAX_GIT_OUTPUT_BYTES = 16 * 1024 * 1024
GIT_READ_TIMEOUT_SECONDS = 30
GIT_MATERIALIZE_TIMEOUT_SECONDS = 120
CAPTURE_LOCK_SCHEMA_VERSION = "incremental-proof-sealer-capture-lock@1"
CAPTURE_LOCK_NAME = ".capture.lock"
MAX_CAPTURE_LOCK_BYTES = 4096
MAX_SOURCE_SCAN_ENTRIES = 250_000
MAX_SOURCE_SCAN_DEPTH = 64
MAX_EXECUTION_LEAF_BYTES = 256 * 1024 * 1024
MAX_EXECUTION_TREE_HASH_BYTES = 4 * 1024 * 1024 * 1024
FIXED_EXECUTABLE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SCHEDULER_CONFIG_RELATIVE = (
    "config/agent_supervisor_incremental_proof_sealer_scheduler.json"
)
PRE_CAPTURE_PROTECTED_PATHS = (
    ".gitignore",
    "docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md",
    "docs/architecture/incremental_proof_sealer.objectives.md",
    "docs/architecture/incremental_proof_sealer.todo.md",
    SCHEDULER_CONFIG_RELATIVE,
    SUITE_REGISTRY_RELATIVE,
    "scripts/validate_incremental_proof_sealer_board.py",
    "scripts/capture_incremental_proof_sealer_baselines.py",
)

PLANNING_REVISIONS = {
    "accelerate": "8881344bb2162f3f8d82f22d8348bc0ac7536f95",
    "datasets": "bd2ff6245ebe476fc744d45c7c66235c92b0e19c",
    "kit": "5a7a2df8181cfdc33bc19be09989df7ff83f2d4e",
}
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
UNMATERIALIZED_OUTER_GITLINKS = {
    "docs/fastmcp": "1d932cc778a24cc0bf46fc4baad8306d4fed9c4b",
    "docs/mcp-python-sdk": "0da9a074d09267a927d72faa58c26d828f0f8edb",
    "ipfs_accelerate_py/mcplusplus": "15c1816d6c63a2b11edd505704f6a04a9abc6167",
    "ipfs_model_manager_py": "f6151d2113f42e75ea7d83a1b2362fc97e55e44d",
    "ipfs_transformers_py": "b397988ed9e3e656475c1cf4417b84efdb95daf3",
    "test/doc-builder": "6108e850ae1cf2f71bb0815a600bcd50c39abfa7",
    "test/huggingface_doc_builder": "6108e850ae1cf2f71bb0815a600bcd50c39abfa7",
    "test/huggingface_transformers": "44752c8dd99f3fb0da23006dc4fde4a07d9c417f",
}
UNMATERIALIZED_GITLINKS = {
    "accelerate": UNMATERIALIZED_OUTER_GITLINKS,
    "datasets": {
        ".tools/ipfs_kit_py": "80afdad2fa6db5875f40e5f495f26b98b7f3c767",
        "ipfs_accelerate_py": "48f13ab632dec4c3575acaad6e309ef60420904b",
        "ipfs_datasets_py/logic/CEC/DCEC_Library": "a4beb5b3280595be6b9221cac3c91dd019e6d371",
        "ipfs_datasets_py/logic/CEC/Eng-DCEC": "df518c21ef81b8001e6db59f5fd70f10cc04ff6c",
        "ipfs_datasets_py/logic/CEC/ShadowProver": "3060ede1ac1ec3f8ef9f9c9e41386aed1dbbe7f9",
        "ipfs_datasets_py/logic/CEC/Talos": "e0b7650d3e3a403924773f8253e924c719748d36",
        "ipfs_datasets_py/multimedia/convert_to_txt_based_on_mime_type": "d58933631a5362b1e2fdc45254ef620fa231223a",
        "ipfs_datasets_py/multimedia/omni_converter_mk2": "c1d9b0d517cea022516aab5b5d8fa5e3bc9a65aa",
        "ipfs_datasets_py/processors/web_archiving/common_crawl_search_engine": "5c7c2ab8a509073f39359b2a35446183855f460a",
        "ipfs_kit_py": "80afdad2fa6db5875f40e5f495f26b98b7f3c767",
    },
    "kit": {
        "docs/filesystem_spec": "fec09b04ad626df44a03bc605cb2e526b752b042",
        "docs/ipfs-docs": "4cf83720b59738d93db4068976f9c2a11f023e45",
        "docs/ipfs_cluster": "c7ca8b5f87b41fcc795297ca65b0bb41c10234bf",
        "docs/ipfsspec": "03f5199b9bf5a96c7ebf5e2e6f5dce8cf58b655f",
        "docs/lassie": "c6ba777810d03fed23aea11b5969b7d8a97f1edf",
        "docs/libp2p-universal-connectivity": "e18a6de9c020c5e406d9f61b638f5d276054798d",
        "docs/libp2p_docs": "17cee4a438797313d1e878b103abc1dbefdf423e",
        "docs/lighthouse-python-sdk": "6b2c86693090c770d2c9a4d82ba315000a77068b",
        "docs/mcp-python-sdk": "d3133ae6ce7333a501e38046aff4275c44326f90",
        "docs/storacha_specs": "3b6791869635735ddb1a54aed7450ad6ef687c06",
        "ipfs_accelerate_py": "48f13ab632dec4c3575acaad6e309ef60420904b",
    },
}


@dataclasses.dataclass(frozen=True)
class Suite:
    id: str
    repository: str
    cwd: str
    test_args: tuple[str, ...]
    timeout_seconds: int
    observation_note: str

    @property
    def argv_template(self) -> tuple[str, ...]:
        return (
            "{python}",
            "-m",
            "pytest",
            "-vv",
            "-ra",
            "--tb=line",
            "--color=no",
            "--trace-config",
            "-o",
            "cache_dir={cache_dir}",
            "--basetemp={basetemp}",
            *self.test_args,
        )


ACCELERATE_FOCUSED_PATHS = (
    "test/api/test_agent_supervisor_provekit_setup.py",
    "test/api/test_agent_supervisor_ipfs_datasets_zk_attestation.py",
    "test/api/test_agent_supervisor_program_analysis_zkp.py",
    "test/api/test_agent_supervisor_program_analysis_zkp_conformance.py",
    "test/api/test_agent_supervisor_proof_scheduler.py",
    "test/api/test_agent_supervisor_proof_resource_scheduler.py",
    "test/api/test_agent_supervisor_adaptive_resources.py",
    "test/api/test_agent_supervisor_multi_prover_resources.py",
    "test/api/test_agent_supervisor_multi_prover_router.py",
    "test/api/test_agent_supervisor_formal_verification_contracts.py",
    "test/api/test_agent_supervisor_formal_verification_cache.py",
    "test/api/test_agent_supervisor_formal_verification_capabilities.py",
    "test/api/test_agent_supervisor_formal_verification_provider.py",
    "test/api/test_agent_supervisor_formal_verification_policy.py",
    "test/api/test_agent_supervisor_code_proof_attestation_policy.py",
    "test/api/test_agent_supervisor_test_execution_identity.py",
    "test/api/test_agent_supervisor_test_execution_identity_vectors.py",
    "test/api/test_agent_supervisor_test_proof_reuse_doctrine.py",
    "test/api/test_proof_reuse_activation_contracts.py",
    "test/api/test_proof_reuse_receipt.py",
    "test/api/test_proof_reuse_runtime_activation_report.py",
    "test/api/test_proof_reuse_controller_issuance.py",
    "test/api/test_proof_reuse_candidate_publication_context.py",
    "test/api/test_proof_reuse_locator_first_collection.py",
    "test/api/test_proof_reuse_two_stage_warm_lookup.py",
    "test/api/test_proof_reuse_runtime_revalidation.py",
    "test/api/test_proof_reuse_degradation_matrix.py",
    "test/api/test_proof_reuse_cold_pass_publication.py",
    "test/api/test_proof_reuse_default_identity_services.py",
    "test/api/test_proof_reuse_default_runtime_services.py",
    "test/api/test_proof_reuse_security_concurrency.py",
    "test/api/test_proof_reuse_invalidation_mutations.py",
    "test/api/test_proof_reuse_issued_material_retention.py",
    "test/api/test_proof_reuse_setup_provisioning.py",
    "test/api/test_proof_reuse_service_injection.py",
    "test/api/test_proof_reuse_lazy_provisioning.py",
)

CURRENT_OBSERVATION = (
    "Current controlled-offline pytest observation; historical counts are not "
    "reconstructed or claimed."
)
CORE_15_OBSERVATION = (
    "New current controlled-offline 15-path pytest observation; it does not "
    "reconstruct or claim the historical 257-result slice."
)

SUITES: tuple[Suite, ...] = (
    Suite(
        "accelerate-proof-focused-core-15",
        "accelerate",
        ".",
        (
            *ACCELERATE_FOCUSED_PATHS[:15],
            "-k",
            "not entry_point and not pyproject and not setup_py and not pytest_ini_registers and not root_conftest_has_optional",
        ),
        300,
        CORE_15_OBSERVATION,
    ),
    Suite(
        "accelerate-proof-focused-wide-36",
        "accelerate",
        ".",
        (
            *ACCELERATE_FOCUSED_PATHS,
            "-k",
            "not entry_point and not pyproject and not setup_py and not pytest_ini_registers and not root_conftest_has_optional",
        ),
        600,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "accelerate-proof-reuse-migration",
        "accelerate",
        ".",
        (
            "test/api/test_proof_reuse_v4_publication_integration.py",
            "test/api/test_proof_reuse_runtime_activation_e2e.py",
            "test/api/test_proof_reuse_runtime_composition.py",
            "test/api/test_pytest_proof_reuse_item_identity.py",
            "test/api/test_pytest_proof_reuse_lookup.py",
            "test/api/test_pytest_proof_reuse_plugin.py",
            "test/api/test_pytest_proof_reuse_receipt.py",
            "test/api/test_pytest_proof_reuse_xdist.py",
        ),
        600,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "accelerate-proof-reuse-cross-repo",
        "accelerate",
        ".",
        (
            "test/api/test_proof_reuse_cross_repository_e2e.py",
            "test/api/test_proof_reuse_accelerator_bootstrap.py",
        ),
        300,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "datasets-zkp-focused-current",
        "datasets",
        "ipfs_datasets_py",
        (
            "tests/unit/logic/zkp",
            "tests/integration/test_provekit_zkp.py",
            "tests/integration/test_groth16_local_evm_verification.py",
            "tests/integration/logic/test_proof_receipt_attestation.py",
            "-k",
            "not test_nargo_check_when_toolchain_available",
        ),
        600,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "datasets-zkp-unit-wide-current",
        "datasets",
        "ipfs_datasets_py",
        (
            "tests/unit_tests/logic/zkp",
            "-k",
            "not test_import_backends_quiet and not test_py_ecc_not_imported_on_backends_import",
        ),
        600,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "datasets-proof-cache-adapters",
        "datasets",
        "ipfs_datasets_py",
        (
            "tests/unit_tests/logic/CEC/native/test_cec_zkp_integration.py",
            "tests/unit_tests/logic/CEC/native/test_cec_proof_cache.py",
            "tests/unit_tests/logic/TDFOL/test_tdfol_proof_cache.py",
            "tests/unit/logic/test_flogic_cache_zkp.py",
            "tests/unit/logic/test_flogic_integration.py",
            "tests/unit/logic/test_flogic_semantic_cid.py",
        ),
        300,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "datasets-zkp-broad-safe-current",
        "datasets",
        "ipfs_datasets_py",
        (
            "tests/unit_tests/logic/zkp",
            "tests/unit/logic/zkp",
            "tests/unit_tests/logic/CEC/native/test_cec_zkp_integration.py",
            "tests/unit_tests/logic/CEC/native/test_cec_proof_cache.py",
            "tests/unit_tests/logic/TDFOL/test_tdfol_proof_cache.py",
            "tests/unit/logic/test_flogic_cache_zkp.py",
            "tests/unit/logic/test_flogic_integration.py",
            "tests/unit/logic/test_flogic_semantic_cid.py",
            "tests/integration/test_provekit_zkp.py",
            "tests/integration/test_groth16_local_evm_verification.py",
            "tests/integration/logic/test_proof_receipt_attestation.py",
            "tests/mcp/unit/test_mcplusplus_spec_session50.py",
            "tests/mcp/integration/test_profile_f_ceremony_p2p.py",
            "tests/mcp/integration/test_profile_d_policy_p2p.py",
            "tests/contract/processors/wallets/test_worldcoin_differential.py",
            "tests/integration/test_pdf_form_agent.py",
            "-k",
            "not test_profile_e_node_starts_with_the_installed_multiformats_runtime and not test_nargo_check_when_toolchain_available and not test_import_backends_quiet and not test_py_ecc_not_imported_on_backends_import",
        ),
        600,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-proof-certificate",
        "kit",
        "ipfs_kit_py",
        ("tests/test_proof_certificate_store.py",),
        120,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-reuse-capabilities",
        "kit",
        "ipfs_kit_py",
        ("tests/test_reuse_capabilities.py",),
        120,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-profile-d",
        "kit",
        "ipfs_kit_py",
        ("tests/test_profile_d_policy.py",),
        120,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-coordination",
        "kit",
        "ipfs_kit_py",
        ("tests/test_coordination_storage.py",),
        120,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-modern-wal",
        "kit",
        "ipfs_kit_py",
        (
            "tests/runtime_readiness/wal/test_wal_contracts.py",
            "tests/runtime_readiness/wal/test_wal_recovery.py",
            "tests/runtime_readiness/wal/test_wal_writer.py",
            "tests/runtime_readiness/wal/test_joined_crash_matrix.py",
        ),
        300,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-proof-reuse-bootstrap",
        "kit",
        "ipfs_kit_py",
        ("tests/test_proof_reuse_bootstrap.py",),
        300,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-agent-receipts",
        "kit",
        "ipfs_kit_py",
        ("tests/test_agent_supervisor_receipts.py",),
        120,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-iroh-release",
        "kit",
        "ipfs_kit_py",
        ("tests/test_iroh_release_readiness.py",),
        120,
        CURRENT_OBSERVATION,
    ),
    Suite(
        "kit-release-receipt",
        "kit",
        "ipfs_kit_py",
        ("tests/runtime_readiness/release/test_joined_release_receipt.py",),
        120,
        CURRENT_OBSERVATION,
    ),
)

SUITES_BY_REPOSITORY = {
    repository: tuple(suite for suite in SUITES if suite.repository == repository)
    for repository in REPOSITORY_PATHS
}
SUITES_BY_ID = {suite.id: suite for suite in SUITES}

OUTCOME_KEYS = (
    "passed",
    "failed",
    "errors",
    "skipped",
    "deselected",
    "xfailed",
    "xpassed",
)
NON_PASS_STATUSES = frozenset({"failed", "error", "skipped", "xfailed", "xpassed"})
TOP_LEVEL_KEYS = frozenset(
    {
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
    }
)
COMMAND_KEYS = frozenset(
    {
        "id",
        "evidence_type",
        "suite_definition_digest",
        "command_digest",
        "argv",
        "cwd",
        "workspace_relative_path",
        "environment",
        "python",
        "pytest",
        "started_at",
        "finished_at",
        "duration_ns",
        "timeout_seconds",
        "capture_status",
        "exit_code",
        "collected_count",
        "collection_complete",
        "outcome_counts",
        "non_pass_nodes",
        "summary_line",
        "parse_error",
        "log",
        "assurance",
    }
)
ASSURANCE_KEYS = frozenset(
    {
        "process_observed",
        "test_execution_cryptographically_proven",
        "cryptographic_proof",
        "signature",
        "network_isolation_enforced",
        "offline_controls_requested",
        "pytest_plugin_allowlist_enforced",
        "public_log_witness_policy",
        "inherited_secrets_forwarded",
        "remaining_trust",
        "claim",
    }
)
REMAINING_TRUST = (
    "Host socket access is not isolated; fixed offline controls are requested and trusted.",
    "Installed pytest11 plugins are recorded but not allowlisted.",
    "Selected tests and subprocesses are trusted not to bypass fixed controls.",
)
ANSI_RE = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY_COUNT_RE = re.compile(
    r"\b(?P<count>\d+)\s+(?P<kind>passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b",
    re.IGNORECASE,
)
SUMMARY_DURATION_RE = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")
ITEM_OUTCOME_RE = re.compile(
    r"^(?P<node>\S.*?::\S.*?)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)
SUMMARY_NODE_RE = re.compile(
    r"^(?P<status>FAILED|ERROR|XFAIL|XPASS)\s+(?P<node>\S.*?)(?:\s+-\s+.*)?$"
)
SKIPPED_SUMMARY_RE = re.compile(
    r"^SKIPPED\s+(?:\[\d+\]\s+)?(?P<node>\S+?:\d+)(?::\s+.*)?$"
)
COLLECTION_ERROR_RE = re.compile(
    r"^(?:[=_-]+\s+)?ERROR collecting (?P<node>\S.*?)(?:\s+[=_-]+)?$"
)
COLLECTED_RE = re.compile(r"\bcollected\s+(?P<count>\d+)\s+items?\b", re.IGNORECASE)
COLLECTED_ERROR_RE = re.compile(
    r"\bcollected\s+\d+\s+items?\b.*(?:/|,)\s*\d+\s+errors?\b",
    re.IGNORECASE,
)


class BaselineError(RuntimeError):
    """Raised for a fail-closed capture or receipt validation error."""


class AtomicWriteError(BaselineError):
    """An atomic write failed, with an explicit post-rename visibility state."""

    def __init__(self, message: str, *, replaced: bool) -> None:
        super().__init__(message)
        self.replaced = replaced


@dataclasses.dataclass
class CaptureLock:
    """Held one-shot admission lock; its on-disk presence is the exclusion token."""

    root_descriptor: int
    artifact_descriptor: int
    lock_descriptor: int
    artifact_binding: os.stat_result
    lock_binding: os.stat_result
    metadata: bytes
    closed: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def receipt_digest(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("receipt_digest", None)
    return _sha256(_canonical_bytes(body))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BaselineError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise BaselineError(f"non-finite JSON value {value!r}")


def _load_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, BaselineError) as exc:
        raise BaselineError(f"{label} is not duplicate-free UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise BaselineError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BaselineError(
            f"{label} has a non-canonical schema; missing={missing}, extra={extra}"
        )


def suite_definition_payload(suite: Suite) -> dict[str, Any]:
    return {
        "id": suite.id,
        "repository": suite.repository,
        "cwd": suite.cwd,
        "argv_template": list(suite.argv_template),
        "environment_policy_id": ENVIRONMENT_POLICY_ID,
        "timeout_seconds": suite.timeout_seconds,
        "observation_note": suite.observation_note,
    }


def suite_definition_digest(suite: Suite) -> str:
    return _sha256(_canonical_bytes(suite_definition_payload(suite)))


def reviewed_registry_payload() -> dict[str, Any]:
    """Return the fixed reviewed registry without importing pytest or running code."""

    return {
        "schema_version": SUITE_REGISTRY_SCHEMA_VERSION,
        "environment_policy_id": ENVIRONMENT_POLICY_ID,
        "repositories": {
            repository: [suite_definition_payload(suite) for suite in suites]
            for repository, suites in SUITES_BY_REPOSITORY.items()
        },
    }


def reviewed_suite_registry() -> dict[str, Any]:
    """Compatibility name for independent validators of the reviewed registry."""

    return reviewed_registry_payload()


def _validate_protected_suite_registry(repo_root: Path) -> dict[str, Any]:
    raw = _safe_fixed_repository_file(
        repo_root, SUITE_REGISTRY_RELATIVE, maximum=MAX_SUITE_REGISTRY_BYTES
    )
    payload = _load_json_object(raw, "protected baseline suite registry")
    _require_exact_keys(
        payload,
        frozenset({"schema_version", "environment_policy_id", "repositories"}),
        "protected baseline suite registry",
    )
    if raw != _canonical_bytes(payload) + b"\n":
        raise BaselineError("protected baseline suite registry is not canonical JSON")
    expected = reviewed_registry_payload()
    if payload != expected:
        raise BaselineError(
            "protected baseline suite registry differs from the compiled projection"
        )
    return payload


def _run_readonly(
    argv: Sequence[str], *, cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    return _run_bounded_command(
        argv,
        cwd=cwd,
        environment={
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": FIXED_EXECUTABLE_PATH,
        },
        timeout_seconds=GIT_READ_TIMEOUT_SECONDS,
        maximum_bytes=MAX_GIT_OUTPUT_BYTES,
        label="bounded read-only command",
    )


def _git_text(repo: Path, *args: str) -> str:
    result = _run_readonly(("git", *args), cwd=repo)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise BaselineError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return result.stdout.decode("utf-8", "strict").strip()


def _assert_no_git_object_replacement(repo: Path, label: str) -> None:
    """Reject local object-replacement mechanisms even though Git ignores them."""

    replacement_refs = _git_text(
        repo, "for-each-ref", "--format=%(refname)", "refs/replace"
    )
    if replacement_refs:
        raise BaselineError(
            f"{label} contains forbidden Git replacement refs: "
            f"{replacement_refs.splitlines()[:8]}"
        )
    common_directory = Path(
        _git_text(
            repo,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    )
    grafts = common_directory / "info" / "grafts"
    try:
        os.lstat(grafts)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise BaselineError(f"cannot inspect {label} legacy Git grafts") from exc
    raise BaselineError(f"{label} contains forbidden legacy Git grafts")


def _ignored_sensitive_binding(
    fingerprint: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Bind the policy-required empty set for clean materialized execution trees."""

    if fingerprint:
        raise BaselineError("materialized execution tree has non-source inputs")
    return {"count": 0, "digest": _sha256(_canonical_bytes([]))}


def _ignored_sensitive_inputs_payload(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    repositories: dict[str, dict[str, Any]] = {}
    if set(snapshots) != set(REPOSITORY_PATHS):
        raise BaselineError("ignored sensitive input snapshots are incomplete")
    for repository in REPOSITORY_PATHS:
        fingerprint = snapshots[repository].get("ignored_sensitive_fingerprint")
        if not isinstance(fingerprint, Mapping):
            raise BaselineError(
                f"{repository} ignored sensitive input snapshot is unavailable"
            )
        repositories[repository] = _ignored_sensitive_binding(fingerprint)
    return {"policy_id": IGNORED_INPUT_POLICY_ID, "repositories": repositories}


def _validate_ignored_sensitive_inputs_shape(value: Any, label: str) -> None:
    _require_exact_keys(
        value, frozenset({"policy_id", "repositories"}), f"{label} ignored inputs"
    )
    if value.get("policy_id") != IGNORED_INPUT_POLICY_ID:
        raise BaselineError(f"{label} ignored input policy is invalid")
    repositories = value.get("repositories")
    _require_exact_keys(
        repositories, frozenset(REPOSITORY_PATHS), f"{label} ignored repositories"
    )
    for repository in REPOSITORY_PATHS:
        binding = repositories.get(repository)
        _require_exact_keys(
            binding,
            frozenset({"count", "digest"}),
            f"{label} {repository} ignored input binding",
        )
        count = binding.get("count")
        digest = binding.get("digest")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            raise BaselineError(
                f"{label} {repository} ignored input binding is malformed"
            )
        if binding != _ignored_sensitive_binding({}):
            raise BaselineError(
                f"{label} {repository} has execution-relevant ignored inputs"
            )


def _assert_ignored_sensitive_input_policy(
    repository: str, fingerprint: Mapping[str, Sequence[Any]], phase: str
) -> None:
    if fingerprint:
        raise BaselineError(
            f"{repository} materialized execution tree has non-source inputs {phase}"
        )


def _repository_snapshot(repo_root: Path, repository: str) -> dict[str, Any]:
    repo = (repo_root / REPOSITORY_PATHS[repository]).resolve()
    _assert_no_git_object_replacement(
        repo, f"{repository} authoritative repository"
    )
    revision = _git_text(repo, "rev-parse", "HEAD")
    tree = _git_text(repo, "rev-parse", "HEAD^{tree}")
    planning_revision = PLANNING_REVISIONS[repository]
    planning_tree = _git_text(repo, "rev-parse", f"{planning_revision}^{{tree}}")
    if planning_tree != PLANNING_TREES[repository]:
        raise BaselineError(
            f"{repository} planning tree does not match the reviewed tree"
        )
    ancestor = _run_readonly(
        ("git", "merge-base", "--is-ancestor", planning_revision, revision), cwd=repo
    )
    if ancestor.returncode != 0:
        raise BaselineError(
            f"{repository} tested revision {revision} does not descend from {planning_revision}"
        )
    unstaged = _run_readonly(
        ("git", "diff", "--quiet", "--ignore-submodules=none"), cwd=repo
    )
    staged = _run_readonly(
        ("git", "diff", "--cached", "--quiet", "--ignore-submodules=none"), cwd=repo
    )
    if unstaged.returncode not in (0, 1) or staged.returncode not in (0, 1):
        raise BaselineError(f"cannot determine clean state for {repository}")
    untracked_raw = _run_readonly(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"), cwd=repo
    )
    if untracked_raw.returncode != 0:
        raise BaselineError(f"cannot enumerate untracked files for {repository}")
    untracked = [
        item.decode("utf-8", "surrogateescape")
        for item in untracked_raw.stdout.split(b"\0")
        if item
    ]
    if repository == "accelerate":
        artifact_prefix = ARTIFACT_RELATIVE_ROOT.as_posix().rstrip("/") + "/"
        untracked = [path for path in untracked if not path.startswith(artifact_prefix)]
    clean = unstaged.returncode == 0 and staged.returncode == 0 and not untracked
    _assert_no_git_object_replacement(
        repo, f"{repository} authoritative repository"
    )
    return {
        "repository": repository,
        "path": REPOSITORY_PATHS[repository].as_posix(),
        "planning_revision": planning_revision,
        "planning_tree": planning_tree,
        "tested_revision": revision,
        "tested_tree": tree,
        "tracked_clean": unstaged.returncode == 0 and staged.returncode == 0,
        "untracked_paths": sorted(untracked),
        # Incidental ignored files in the authority checkout are not execution
        # inputs.  Commands run from fresh exact-revision Git worktrees below.
        "ignored_sensitive_fingerprint": {},
        "clean": clean,
    }


def _all_source_snapshots(repo_root: Path) -> dict[str, dict[str, Any]]:
    return {
        repository: _repository_snapshot(repo_root, repository)
        for repository in REPOSITORY_PATHS
    }


def _python_metadata() -> dict[str, str]:
    return {
        "executable": str(Path(sys.executable).resolve()),
        "implementation": platform.python_implementation(),
        "version": sys.version,
    }


def _pytest_metadata() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BaselineError(
            "pytest is unavailable; dependencies are never installed by this tool"
        ) from exc
    spec = importlib.util.find_spec("pytest")
    if spec is None or not spec.origin:
        raise BaselineError("pytest module origin is unavailable")
    plugins: list[dict[str, str | None]] = []
    for entry_point in sorted(
        importlib.metadata.entry_points(group="pytest11"),
        key=lambda item: (item.name, item.value),
    ):
        distribution = entry_point.dist
        plugins.append(
            {
                "name": entry_point.name,
                "value": entry_point.value,
                "distribution": (
                    distribution.metadata.get("Name") if distribution else None
                ),
                "version": distribution.version if distribution else None,
            }
        )
    return {
        "version": version,
        "module_path": str(Path(spec.origin).resolve()),
        "autoload_plugins": plugins,
    }


def _environment(
    repo_root: Path, workspace_relative: str, pytest_module_path: str | None = None
) -> dict[str, str]:
    workspace = _artifact_path(
        repo_root,
        workspace_relative,
        label="environment workspace",
        allow_missing=True,
    )
    pytest_path = Path(
        pytest_module_path or _pytest_metadata()["module_path"]
    ).resolve()
    site_packages = pytest_path.parents[1]
    workspace_parts = _canonical_relative(
        workspace_relative, "environment workspace"
    ).parts
    work_prefix = (*PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix()).parts, "work")
    if (
        workspace_parts[: len(work_prefix)] != work_prefix
        or len(workspace_parts) < len(work_prefix) + 2
    ):
        raise BaselineError("environment workspace lacks a capture binding")
    capture_id = workspace_parts[len(work_prefix)]
    source_root = repo_root / PurePosixPath(*work_prefix, capture_id, "source")
    python_path = os.pathsep.join(
        str(path.resolve())
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
        "PATH": FIXED_EXECUTABLE_PATH,
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


def _resolved_argv(suite: Suite, python_executable: str, basetemp: Path) -> list[str]:
    cache_dir = basetemp.parent / "pytest-cache"
    return [
        token.replace("{python}", python_executable)
        .replace("{basetemp}", str(basetemp))
        .replace("{cache_dir}", str(cache_dir))
        for token in suite.argv_template
    ]


def _assert_no_live_ipfs(repo_root: Path) -> None:
    """Refuse kit observations when a real IPFS executable could be selected."""

    resolved = shutil.which("ipfs", path=FIXED_EXECUTABLE_PATH)
    if resolved is not None:
        raise BaselineError(
            f"kit capture refused: fixed PATH resolves ipfs at {resolved}"
        )
    kit_root = (repo_root / REPOSITORY_PATHS["kit"]).resolve()
    candidates = sorted(
        path
        for name in ("ipfs", "ipfs.exe")
        for path in kit_root.rglob(name)
        if path.parent.name.casefold() == "bin"
    )
    for candidate in candidates:
        try:
            mode = candidate.stat().st_mode
        except OSError as exc:
            raise BaselineError(
                f"cannot inspect kit IPFS candidate {candidate}: {exc}"
            ) from exc
        if stat.S_ISREG(mode) and os.access(candidate, os.X_OK):
            raise BaselineError(
                f"kit capture refused: executable IPFS candidate {candidate}"
            )


def _assert_one_shot_capture_available(repo_root: Path) -> None:
    """Require the reviewed empty-pin bootstrap phase before any suite starts.

    Capture is deliberately one-shot and all-repositories-only.  It has no
    recapture or crash-recovery protocol: any canonical receipt or retained log
    is therefore an explicit operator-repair condition, never an invitation to
    overwrite or resume evidence.
    """

    raw = _safe_fixed_repository_file(
        repo_root, SCHEDULER_CONFIG_RELATIVE, maximum=MAX_RECEIPT_BYTES
    )
    scheduler = _load_json_object(raw, "pre-capture scheduler configuration")
    if scheduler.get("operator_baseline_receipts") != {}:
        raise BaselineError(
            "one-shot capture requires an empty operator receipt-pin map"
        )
    if scheduler.get("protected_paths") != list(PRE_CAPTURE_PROTECTED_PATHS):
        raise BaselineError(
            "one-shot capture requires the exact reviewed pre-capture protected paths"
        )

    artifact_pure = PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix())
    artifact_entries = _safe_directory_entries(
        repo_root,
        artifact_pure,
        label="one-shot artifact directory",
        allow_missing=False,
    )
    if artifact_entries is None:
        raise BaselineError("one-shot artifact directory is unavailable")
    for repository in REPOSITORY_PATHS:
        if f"{repository}.json" in artifact_entries:
            raise BaselineError(
                "one-shot capture refused because canonical evidence already exists; "
                "operator quarantine or repair is required"
            )

    if "logs" not in artifact_entries:
        return
    if not stat.S_ISDIR(artifact_entries["logs"].st_mode):
        raise BaselineError("one-shot retained log path is unsafe")
    log_entries = _safe_directory_entries(
        repo_root,
        artifact_pure / "logs",
        label="one-shot retained log directory",
        allow_missing=False,
    )
    if log_entries:
        raise BaselineError(
            "one-shot capture refused because retained evidence already exists; "
            "operator quarantine or repair is required"
        )


def _strip_ansi(raw: bytes) -> str:
    try:
        text = ANSI_RE.sub(b"", raw).decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise BaselineError("retained pytest log is not canonical UTF-8") from exc
    return text.replace("\r", "\n")


def parse_pytest_log(raw: bytes) -> dict[str, Any]:
    """Independently derive outcomes and non-pass nodes from retained bytes."""

    text = _strip_ansi(raw)
    lines = [line.strip() for line in text.splitlines()]
    if not any("test session starts" in line for line in lines):
        raise BaselineError("retained log lacks the pytest session banner")

    summary_line = ""
    summary_counts: dict[str, int] | None = None
    for line in reversed(lines):
        matches = list(SUMMARY_COUNT_RE.finditer(line))
        if matches and (SUMMARY_DURATION_RE.search(line) or "no tests ran" in line):
            summary_line = line
            summary_counts = {key: 0 for key in OUTCOME_KEYS}
            for match in matches:
                kind = match.group("kind").lower()
                if kind in {"error", "errors"}:
                    kind = "errors"
                summary_counts[kind] += int(match.group("count"))
            break
    if summary_counts is None:
        raise BaselineError("retained log lacks a parseable pytest outcome summary")

    selected = sum(summary_counts[key] for key in OUTCOME_KEYS if key != "deselected")
    summary_counts["selected"] = selected
    collected_count: int | None = None
    for line in lines:
        match = COLLECTED_RE.search(line)
        if match:
            collected_count = int(match.group("count"))

    collection_error_nodes = {
        match.group("node").strip()
        for line in lines
        for match in (COLLECTION_ERROR_RE.match(line),)
        if match is not None
    }
    collection_count_failed = any(COLLECTED_ERROR_RE.search(line) for line in lines)
    collection_failed = bool(collection_error_nodes) or collection_count_failed

    non_pass_nodes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    collection_skip_detected = False

    def matches_runtime_item_skip(location: str) -> bool:
        for existing in non_pass_nodes:
            if existing["status"] != "skipped" or "::" not in existing["node_id"]:
                continue
            item_path = existing["node_id"].split("::", 1)[0]
            if location.startswith(item_path + ":"):
                return True
        return False

    def add(status: str, node_id: str, detail: str) -> None:
        normalized = {
            "errors": "error",
            "xfail": "xfailed",
            "xpass": "xpassed",
        }.get(status.lower(), status.lower())
        if normalized not in NON_PASS_STATUSES:
            return
        if (
            normalized == "skipped"
            and "::" not in node_id
            and matches_runtime_item_skip(node_id)
        ):
            return
        canonical_node = node_id.strip()
        if (
            normalized == "error"
            and not canonical_node.startswith("collecting ")
            and (
                canonical_node in collection_error_nodes
                or (collection_count_failed and "::" not in canonical_node)
            )
        ):
            canonical_node = f"collecting {canonical_node}"
        key = (normalized, canonical_node)
        if key in seen:
            return
        seen.add(key)
        non_pass_nodes.append(
            {"status": normalized, "node_id": canonical_node, "detail": detail}
        )

    for line in lines:
        match = ITEM_OUTCOME_RE.match(line)
        if match:
            add(match.group("status"), match.group("node"), line)
            continue
        match = COLLECTION_ERROR_RE.match(line)
        if match:
            add("error", f"collecting {match.group('node')}", line)
            continue
        match = SKIPPED_SUMMARY_RE.match(line)
        if match:
            location = match.group("node")
            if not matches_runtime_item_skip(location):
                collection_skip_detected = True
            add("skipped", location, line)
            continue
        match = SUMMARY_NODE_RE.match(line)
        if match:
            add(match.group("status"), match.group("node"), line)

    return {
        "outcome_counts": summary_counts,
        "collected_count": collected_count,
        "collection_complete": (
            collected_count is not None
            and not collection_failed
            and not collection_skip_detected
        ),
        "non_pass_nodes": non_pass_nodes,
        "summary_line": summary_line,
    }


def _canonical_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BaselineError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or value.startswith("./")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BaselineError(f"{label} must be a canonical relative POSIX path")
    return path


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_root_directory(repo_root: Path, label: str) -> tuple[Path, int]:
    """Open every absolute root component without following a symlink."""

    root = Path(os.path.abspath(os.fspath(repo_root)))
    if not root.is_absolute():
        raise BaselineError(f"{label} root is not absolute")
    flags = _directory_open_flags()
    descriptor = os.open(os.sep, flags)
    try:
        for part in root.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise BaselineError(f"{label} root is not a directory")
    except (OSError, BaselineError) as exc:
        os.close(descriptor)
        if isinstance(exc, BaselineError):
            raise
        raise BaselineError(f"cannot safely open {label} root") from exc
    return root, descriptor


def _open_relative_directory(
    root_descriptor: int,
    parts: Sequence[str],
    *,
    create: bool,
    label: str,
) -> int:
    """Walk relative directories from a held root fd, optionally creating them."""

    descriptor = os.dup(root_descriptor)
    flags = _directory_open_flags()
    try:
        for part in parts:
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                raise BaselineError(f"{label} component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
    except (OSError, BaselineError) as exc:
        os.close(descriptor)
        if isinstance(exc, BaselineError):
            raise
        raise BaselineError(f"cannot safely walk {label}") from exc
    return descriptor


def _try_open_relative_directory(
    root_descriptor: int, parts: Sequence[str], *, label: str
) -> int | None:
    descriptor = os.dup(root_descriptor)
    try:
        for part in parts:
            try:
                next_descriptor = os.open(
                    part, _directory_open_flags(), dir_fd=descriptor
                )
            except FileNotFoundError:
                os.close(descriptor)
                return None
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        os.close(descriptor)
        raise BaselineError(f"cannot safely walk {label}") from exc
    return descriptor


def _assert_relative_directory_binding(
    root_descriptor: int,
    parts: Sequence[str],
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    current_descriptor = _try_open_relative_directory(
        root_descriptor, parts, label=label
    )
    if current_descriptor is None:
        raise BaselineError(f"{label} disappeared during the operation")
    try:
        current = os.fstat(current_descriptor)
    finally:
        os.close(current_descriptor)
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise BaselineError(f"{label} was replaced during the operation")


def _artifact_relative(value: Any, label: str) -> PurePosixPath:
    pure = _canonical_relative(value, label)
    root_parts = PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix()).parts
    if pure.parts[: len(root_parts)] != root_parts:
        raise BaselineError(f"{label} is outside the baseline artifact root")
    return pure


def _artifact_path(
    repo_root: Path,
    relative: Any,
    *,
    label: str,
    allow_missing: bool,
) -> Path:
    pure = _artifact_relative(relative, label)
    root, root_descriptor = _open_root_directory(repo_root, label)
    descriptor = os.dup(root_descriptor)
    try:
        for index, part in enumerate(pure.parts):
            final = index == len(pure.parts) - 1
            try:
                info = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                if allow_missing:
                    break
                raise BaselineError(f"{label} does not exist") from None
            if stat.S_ISLNK(info.st_mode):
                raise BaselineError(f"{label} contains a symlink component")
            if not final:
                if not stat.S_ISDIR(info.st_mode):
                    raise BaselineError(f"{label} component is not a directory")
                next_descriptor = os.open(
                    part, _directory_open_flags(), dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = next_descriptor
    except OSError as exc:
        raise BaselineError(f"cannot safely inspect {label}") from exc
    finally:
        os.close(descriptor)
        os.close(root_descriptor)
    return root / PurePosixPath(*pure.parts)


def _safe_directory_entries(
    repo_root: Path,
    pure: PurePosixPath,
    *,
    label: str,
    allow_missing: bool,
) -> dict[str, os.stat_result] | None:
    """List and lstat a directory while its root-to-leaf dirfds remain held."""

    _, root_descriptor = _open_root_directory(repo_root, label)
    directory_descriptor: int | None = None
    try:
        directory_descriptor = _try_open_relative_directory(
            root_descriptor, pure.parts, label=label
        )
        if directory_descriptor is None:
            if allow_missing:
                return None
            raise BaselineError(f"{label} does not exist")
        binding = os.fstat(directory_descriptor)
        entries: dict[str, os.stat_result] = {}
        for name in os.listdir(directory_descriptor):
            try:
                name.encode("utf-8", "strict")
            except UnicodeEncodeError as exc:
                raise BaselineError(f"{label} contains a non-UTF-8 name") from exc
            if name in entries:
                raise BaselineError(f"{label} contains a duplicate name")
            entries[name] = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
        _assert_relative_directory_binding(
            root_descriptor, pure.parts, binding, label=label
        )
        return entries
    except BaselineError:
        raise
    except OSError as exc:
        raise BaselineError(f"cannot safely enumerate {label}") from exc
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(root_descriptor)


def _ensure_artifact_directory(repo_root: Path, relative: str) -> Path:
    pure = _artifact_relative(relative, "artifact directory")
    root, root_descriptor = _open_root_directory(repo_root, "artifact directory")
    directory_descriptor: int | None = None
    try:
        directory_descriptor = _open_relative_directory(
            root_descriptor,
            pure.parts,
            create=True,
            label="artifact directory",
        )
        expected = os.fstat(directory_descriptor)
        _assert_relative_directory_binding(
            root_descriptor,
            pure.parts,
            expected,
            label="artifact directory",
        )
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(root_descriptor)
    return root / PurePosixPath(*pure.parts)


def _capture_lock_bytes(capture_id: str) -> bytes:
    if not re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z-\d+", capture_id):
        raise BaselineError("capture lock requires a canonical capture id")
    payload = {
        "schema_version": CAPTURE_LOCK_SCHEMA_VERSION,
        "operation": "one-shot-all-repositories",
        "capture_id": capture_id,
        "owner_pid": os.getpid(),
        "created_at": _utc_now(),
    }
    raw = _canonical_bytes(payload) + b"\n"
    if len(raw) > MAX_CAPTURE_LOCK_BYTES:
        raise BaselineError("capture lock metadata exceeds its fixed size limit")
    return raw


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _acquire_capture_lock(repo_root: Path, capture_id: str) -> CaptureLock:
    """Create and hold the one-shot lock without following pathname components."""

    metadata = _capture_lock_bytes(capture_id)
    _, root_descriptor = _open_root_directory(repo_root, "capture lock")
    artifact_descriptor: int | None = None
    lock_descriptor: int | None = None
    artifact_parts = PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix()).parts
    try:
        artifact_descriptor = _open_relative_directory(
            root_descriptor,
            artifact_parts,
            create=False,
            label="capture lock artifact directory",
        )
        artifact_binding = os.fstat(artifact_descriptor)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            lock_descriptor = os.open(
                CAPTURE_LOCK_NAME, flags, 0o600, dir_fd=artifact_descriptor
            )
        except FileExistsError as exc:
            raise BaselineError(
                "one-shot capture lock already exists; operator quarantine or "
                "repair is required"
            ) from exc
        _write_all(lock_descriptor, metadata)
        os.fsync(lock_descriptor)
        lock_binding = os.fstat(lock_descriptor)
        if not stat.S_ISREG(lock_binding.st_mode) or lock_binding.st_size != len(metadata):
            raise BaselineError("capture lock is not a bounded regular file")
        visible = os.stat(
            CAPTURE_LOCK_NAME,
            dir_fd=artifact_descriptor,
            follow_symlinks=False,
        )
        if (visible.st_dev, visible.st_ino) != (
            lock_binding.st_dev,
            lock_binding.st_ino,
        ):
            raise BaselineError("capture lock changed during acquisition")
        os.fsync(artifact_descriptor)
        _assert_relative_directory_binding(
            root_descriptor,
            artifact_parts,
            artifact_binding,
            label="capture lock artifact directory",
        )
        return CaptureLock(
            root_descriptor=root_descriptor,
            artifact_descriptor=artifact_descriptor,
            lock_descriptor=lock_descriptor,
            artifact_binding=artifact_binding,
            lock_binding=lock_binding,
            metadata=metadata,
        )
    except BaselineError:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)
        os.close(root_descriptor)
        raise
    except OSError as exc:
        # If O_EXCL succeeded, leave even an incomplete lock in place.  Its
        # ambiguity is an operator-repair condition, never an auto-resume cue.
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)
        os.close(root_descriptor)
        raise BaselineError("cannot durably acquire the one-shot capture lock") from exc


def _assert_capture_lock_held(lock: CaptureLock) -> None:
    if lock.closed:
        raise BaselineError("capture lock is no longer held")
    try:
        opened = os.fstat(lock.lock_descriptor)
        visible = os.stat(
            CAPTURE_LOCK_NAME,
            dir_fd=lock.artifact_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(opened.st_mode):
            raise BaselineError("capture lock is no longer a regular file")
        expected_identity = (lock.lock_binding.st_dev, lock.lock_binding.st_ino)
        if (opened.st_dev, opened.st_ino) != expected_identity or (
            visible.st_dev,
            visible.st_ino,
        ) != expected_identity:
            raise BaselineError("capture lock was replaced while capture was active")
        if opened.st_size != len(lock.metadata) or opened.st_size > MAX_CAPTURE_LOCK_BYTES:
            raise BaselineError("capture lock metadata size changed")
        observed = os.pread(
            lock.lock_descriptor, MAX_CAPTURE_LOCK_BYTES + 1, 0
        )
        if observed != lock.metadata:
            raise BaselineError("capture lock owner metadata changed")
        _assert_relative_directory_binding(
            lock.root_descriptor,
            PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix()).parts,
            lock.artifact_binding,
            label="capture lock artifact directory",
        )
    except BaselineError:
        raise
    except OSError as exc:
        raise BaselineError("cannot verify the active capture lock") from exc


def _close_capture_lock(lock: CaptureLock) -> None:
    """Close descriptors but intentionally retain an ambiguous/stale lock."""

    if lock.closed:
        return
    os.close(lock.lock_descriptor)
    os.close(lock.artifact_descriptor)
    os.close(lock.root_descriptor)
    lock.closed = True


def _release_capture_lock(lock: CaptureLock) -> None:
    """Release only after a complete success or fully cleaned prepublish failure."""

    try:
        _assert_capture_lock_held(lock)
        os.unlink(CAPTURE_LOCK_NAME, dir_fd=lock.artifact_descriptor)
        os.fsync(lock.artifact_descriptor)
        _assert_relative_directory_binding(
            lock.root_descriptor,
            PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix()).parts,
            lock.artifact_binding,
            label="capture lock artifact directory",
        )
    except BaselineError:
        raise
    except OSError as exc:
        raise BaselineError("cannot durably release the one-shot capture lock") from exc
    finally:
        _close_capture_lock(lock)


def _atomic_write(repo_root: Path, relative: str, data: bytes) -> Path:
    pure = _artifact_relative(relative, "atomic output")
    root, root_descriptor = _open_root_directory(repo_root, "atomic output")
    destination = root / PurePosixPath(*pure.parts)
    temporary_name = f".{pure.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    file_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    temporary_created = False
    replaced = False
    directory_fd: int | None = None
    try:
        directory_fd = _open_relative_directory(
            root_descriptor,
            pure.parts[:-1],
            create=True,
            label="atomic output parent",
        )
        parent_binding = os.fstat(directory_fd)
        try:
            existing = os.stat(
                pure.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise BaselineError(
                f"atomic output must not replace a symlink: {relative}"
            )
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise BaselineError(
                f"atomic output must replace only a regular file: {relative}"
            )
        fd = os.open(temporary_name, file_flags, 0o600, dir_fd=directory_fd)
        temporary_created = True
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            written_binding = os.fstat(stream.fileno())
        os.replace(
            temporary_name,
            pure.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replaced = True
        temporary_created = False
        os.fsync(directory_fd)
        visible = os.stat(
            pure.name, dir_fd=directory_fd, follow_symlinks=False
        )
        if (visible.st_dev, visible.st_ino) != (
            written_binding.st_dev,
            written_binding.st_ino,
        ):
            raise BaselineError("atomic output was replaced after publication")
        _assert_relative_directory_binding(
            root_descriptor,
            pure.parts[:-1],
            parent_binding,
            label="atomic output parent",
        )
    except BaselineError as exc:
        if not replaced:
            raise
        raise AtomicWriteError(
            f"atomic output parent changed after replacement: {relative}",
            replaced=True,
        ) from exc
    except OSError as exc:
        raise AtomicWriteError(
            f"atomic output failed {'after' if replaced else 'before'} replacement: {relative}",
            replaced=replaced,
        ) from exc
    finally:
        if temporary_created and directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(root_descriptor)
    return destination


def _create_once(repo_root: Path, relative: str, data: bytes) -> Path:
    """Create canonical evidence exactly once; never replace an existing inode."""

    pure = _artifact_relative(relative, "create-only output")
    root, root_descriptor = _open_root_directory(repo_root, "create-only output")
    destination = root / PurePosixPath(*pure.parts)
    directory_descriptor: int | None = None
    file_descriptor: int | None = None
    created = False
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        directory_descriptor = _open_relative_directory(
            root_descriptor,
            pure.parts[:-1],
            create=False,
            label="create-only output parent",
        )
        parent_binding = os.fstat(directory_descriptor)
        file_descriptor = os.open(
            pure.name, flags, 0o600, dir_fd=directory_descriptor
        )
        created = True
        _write_all(file_descriptor, data)
        os.fsync(file_descriptor)
        written_binding = os.fstat(file_descriptor)
        if not stat.S_ISREG(written_binding.st_mode) or written_binding.st_size != len(data):
            raise BaselineError("create-only output is not the complete regular file")
        visible = os.stat(
            pure.name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (visible.st_dev, visible.st_ino) != (
            written_binding.st_dev,
            written_binding.st_ino,
        ):
            raise BaselineError("create-only output changed during publication")
        os.fsync(directory_descriptor)
        _assert_relative_directory_binding(
            root_descriptor,
            pure.parts[:-1],
            parent_binding,
            label="create-only output parent",
        )
    except BaselineError as exc:
        if not created:
            raise
        raise AtomicWriteError(
            f"create-only output failed after publication: {relative}",
            replaced=True,
        ) from exc
    except OSError as exc:
        raise AtomicWriteError(
            f"create-only output failed {'after' if created else 'before'} publication: {relative}",
            replaced=created,
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(root_descriptor)
    return destination


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes], process_group: int | None = None
) -> None:
    process_group = process.pid if process_group is None else process_group
    try:
        os.killpg(process_group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.05, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.01)
    if not _process_group_exists(process_group):
        return
    try:
        os.killpg(process_group, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        process.wait(timeout=5)


def _bounded_marker(raw: bytes, marker: bytes) -> bytes:
    marker = b"\n" + marker.strip() + b"\n"
    marker = marker[:MAX_LOG_BYTES]
    return raw[: max(0, MAX_LOG_BYTES - len(marker))] + marker


def _communicate_bounded(
    process: subprocess.Popen[bytes], timeout_seconds: int
) -> tuple[bytes, str, int]:
    """Stream merged output with a wall timeout and an absolute byte ceiling."""

    if process.stdout is None:
        raise BaselineError("captured process has no stdout pipe")
    process_group = process.pid
    try:
        observed_process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        # Popen(start_new_session=True) fixes PGID to PID.  A very fast child may
        # already be reaped before this observation without invalidating that
        # ownership invariant.
        observed_process_group = process_group
    if observed_process_group != process_group:
        raise BaselineError("captured process does not own a dedicated process group")
    descriptor = process.stdout.fileno()
    os.set_blocking(descriptor, False)
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    total = 0
    status = "completed"
    forced_exit: int | None = None
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timed_out"
                forced_exit = 124
                _terminate_process_group(process, process_group)
                break
            for key, _ in selector.select(min(0.25, remaining)):
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                if total + len(chunk) > MAX_LOG_BYTES:
                    allowed = max(0, MAX_LOG_BYTES - total)
                    if allowed:
                        chunks.append(chunk[:allowed])
                    status = "output_limit_exceeded"
                    forced_exit = 125
                    _terminate_process_group(process, process_group)
                    selector.unregister(key.fd)
                    break
                chunks.append(chunk)
                total += len(chunk)
            if status != "completed":
                break
            if process.poll() is not None and not selector.get_map():
                break
        if process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timed_out"
                forced_exit = 124
                _terminate_process_group(process, process_group)
            else:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    status = "timed_out"
                    forced_exit = 124
                    _terminate_process_group(process, process_group)
    except BaseException:
        _terminate_process_group(process, process_group)
        raise
    finally:
        selector.close()
    if status == "completed" and _process_group_exists(process_group):
        status = "residual_process_terminated"
        forced_exit = 126
        _terminate_process_group(process, process_group)
    raw = b"".join(chunks)
    if status == "timed_out":
        raw = _bounded_marker(
            raw, f"BASELINE_CAPTURE_TIMEOUT after {timeout_seconds}s".encode()
        )
    elif status == "output_limit_exceeded":
        raw = _bounded_marker(raw, b"BASELINE_CAPTURE_OUTPUT_LIMIT_EXCEEDED")
    elif status == "residual_process_terminated":
        raw = _bounded_marker(raw, b"BASELINE_CAPTURE_RESIDUAL_PROCESS_TERMINATED")
    return (
        raw,
        status,
        forced_exit if forced_exit is not None else int(process.returncode),
    )


def _communicate_bounded_streams(
    process: subprocess.Popen[bytes],
    timeout_seconds: int,
    maximum_bytes: int,
) -> tuple[bytes, bytes, str, int]:
    """Bound stdout+stderr together and terminate the dedicated process tree."""

    if process.stdout is None or process.stderr is None:
        raise BaselineError("bounded process lacks separate output pipes")
    process_group = process.pid
    try:
        observed_process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        observed_process_group = process_group
    if observed_process_group != process_group:
        raise BaselineError("bounded process does not own a dedicated process group")
    selector = selectors.DefaultSelector()
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    chunks: dict[str, list[bytes]] = {name: [] for name in streams}
    for name, stream in streams.items():
        descriptor = stream.fileno()
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ, data=name)
    deadline = time.monotonic() + timeout_seconds
    total = 0
    status = "completed"
    forced_exit: int | None = None
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timed_out"
                forced_exit = 124
                _terminate_process_group(process, process_group)
                break
            for key, _ in selector.select(min(0.25, remaining)):
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                if total + len(chunk) > maximum_bytes:
                    allowed = max(0, maximum_bytes - total)
                    if allowed:
                        chunks[key.data].append(chunk[:allowed])
                    status = "output_limit_exceeded"
                    forced_exit = 125
                    _terminate_process_group(process, process_group)
                    break
                chunks[key.data].append(chunk)
                total += len(chunk)
            if status != "completed":
                break
            if process.poll() is not None and not selector.get_map():
                break
        if process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timed_out"
                forced_exit = 124
                _terminate_process_group(process, process_group)
            else:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    status = "timed_out"
                    forced_exit = 124
                    _terminate_process_group(process, process_group)
    except BaseException:
        _terminate_process_group(process, process_group)
        raise
    finally:
        selector.close()
    if status == "completed" and _process_group_exists(process_group):
        status = "residual_process_terminated"
        forced_exit = 126
        _terminate_process_group(process, process_group)
    return (
        b"".join(chunks["stdout"]),
        b"".join(chunks["stderr"]),
        status,
        forced_exit if forced_exit is not None else int(process.returncode),
    )


def _run_bounded_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    maximum_bytes: int,
    label: str,
) -> subprocess.CompletedProcess[bytes]:
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise BaselineError(f"{label} launch failed") from exc
    try:
        stdout, stderr, status, returncode = _communicate_bounded_streams(
            process, timeout_seconds, maximum_bytes
        )
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    if status != "completed":
        raise BaselineError(f"{label} failed closed: {status}")
    return subprocess.CompletedProcess(
        args=list(argv),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
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


def _assert_public_log_safe(raw: bytes) -> None:
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(raw):
            raise BaselineError(f"retained public log refused by secret scan: {label}")


def _assurance_payload(*, process_observed: bool, aggregate: bool) -> dict[str, Any]:
    claim = (
        "Integrity-protected observations of fixed pytest subprocesses only."
        if aggregate
        else "Observed output from the fixed pytest subprocess only."
    )
    return {
        "process_observed": process_observed,
        "test_execution_cryptographically_proven": False,
        "cryptographic_proof": False,
        "signature": None,
        "network_isolation_enforced": False,
        "offline_controls_requested": True,
        "pytest_plugin_allowlist_enforced": False,
        "public_log_witness_policy": "public-full-log-secret-scan@1",
        "inherited_secrets_forwarded": False,
        "remaining_trust": list(REMAINING_TRUST),
        "claim": claim,
    }


def _run_local_git_materialization(argv: Sequence[str], cwd: Path) -> None:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(cwd),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": FIXED_EXECUTABLE_PATH,
    }
    result = _run_bounded_command(
        argv,
        cwd=cwd,
        environment=environment,
        timeout_seconds=GIT_MATERIALIZE_TIMEOUT_SECONDS,
        maximum_bytes=MAX_GIT_OUTPUT_BYTES,
        label="bounded local Git materialization",
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise BaselineError(f"local Git materialization failed: {detail}")


def _remove_unmaterialized_gitlink_placeholders(
    execution_root: Path, repository: str
) -> None:
    """Remove only Git's empty checkout directories for reviewed outer gitlinks."""

    _, root_descriptor = _open_root_directory(
        execution_root, "unmaterialized Git link cleanup"
    )
    try:
        for relative in UNMATERIALIZED_GITLINKS[repository]:
            pure = _canonical_relative(relative, "unmaterialized Git link")
            parent_descriptor = _try_open_relative_directory(
                root_descriptor,
                pure.parts[:-1],
                label="unmaterialized Git link parent",
            )
            if parent_descriptor is None:
                continue
            try:
                try:
                    info = os.stat(
                        pure.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if not stat.S_ISDIR(info.st_mode):
                    raise BaselineError(
                        f"unmaterialized Git link is not an empty directory: {relative}"
                    )
                try:
                    os.rmdir(pure.name, dir_fd=parent_descriptor)
                except OSError as exc:
                    raise BaselineError(
                        f"unmaterialized Git link directory is not empty: {relative}"
                    ) from exc
            finally:
                os.close(parent_descriptor)
            for depth in range(len(pure.parts) - 1, 0, -1):
                parent_parts = pure.parts[:depth]
                grandparent_descriptor = _try_open_relative_directory(
                    root_descriptor,
                    parent_parts[:-1],
                    label="unmaterialized Git link ancestor",
                )
                if grandparent_descriptor is None:
                    break
                try:
                    try:
                        os.rmdir(parent_parts[-1], dir_fd=grandparent_descriptor)
                    except OSError as exc:
                        if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                            break
                        raise BaselineError(
                            "cannot remove an empty Git link placeholder ancestor"
                        ) from exc
                finally:
                    os.close(grandparent_descriptor)
    finally:
        os.close(root_descriptor)


def _materialize_execution_trees(
    repo_root: Path,
    capture_id: str,
    snapshots: Mapping[str, Mapping[str, Any]],
) -> Path:
    capture_relative = (ARTIFACT_RELATIVE_ROOT / "work" / capture_id).as_posix()
    capture_root = _ensure_artifact_directory(repo_root, capture_relative)
    execution_root = capture_root / "source"
    if execution_root.exists() or execution_root.is_symlink():
        raise BaselineError("capture execution tree already exists")
    outer_source = (repo_root / REPOSITORY_PATHS["accelerate"]).resolve()
    _assert_no_git_object_replacement(
        outer_source, "accelerate authoritative repository"
    )
    _run_local_git_materialization(
        (
            "git",
            "clone",
            "--no-local",
            "--no-checkout",
            str(outer_source),
            str(execution_root),
        ),
        capture_root,
    )
    _run_local_git_materialization(
        (
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "checkout",
            "--detach",
            snapshots["accelerate"]["tested_revision"],
        ),
        execution_root,
    )
    _run_local_git_materialization(("git", "remote", "remove", "origin"), execution_root)
    _remove_unmaterialized_gitlink_placeholders(execution_root, "accelerate")
    _assert_no_git_object_replacement(
        execution_root, "accelerate materialized repository"
    )
    for repository in ("datasets", "kit"):
        source = (repo_root / REPOSITORY_PATHS[repository]).resolve()
        _assert_no_git_object_replacement(
            source, f"{repository} authoritative repository"
        )
        target = execution_root / REPOSITORY_PATHS[repository]
        _run_local_git_materialization(
            (
                "git",
                "clone",
                "--no-local",
                "--no-checkout",
                str(source),
                str(target),
            ),
            execution_root,
        )
        _run_local_git_materialization(
            (
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "checkout",
                "--detach",
                snapshots[repository]["tested_revision"],
            ),
            target,
        )
        _run_local_git_materialization(("git", "remote", "remove", "origin"), target)
        _remove_unmaterialized_gitlink_placeholders(target, repository)
        _assert_no_git_object_replacement(
            target, f"{repository} materialized repository"
        )
    return execution_root


def _remove_execution_write_permissions(
    directory_descriptor: int,
    *,
    depth: int,
    visited: list[int],
) -> None:
    """Clear write bits without following a materialized-tree symlink."""

    if depth > MAX_SOURCE_SCAN_DEPTH:
        raise BaselineError("execution-tree hardening exceeded its depth limit")
    try:
        names = sorted(os.listdir(directory_descriptor))
    except OSError as exc:
        raise BaselineError("cannot enumerate execution tree for hardening") from exc
    for name in names:
        visited[0] += 1
        if visited[0] > MAX_SOURCE_SCAN_ENTRIES:
            raise BaselineError("execution-tree hardening exceeded its entry limit")
        try:
            name.encode("utf-8", "strict")
            before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except (OSError, UnicodeEncodeError) as exc:
            raise BaselineError("cannot inspect execution tree for hardening") from exc
        if stat.S_ISLNK(before.st_mode):
            continue
        if stat.S_ISDIR(before.st_mode):
            try:
                child_descriptor = os.open(
                    name, _directory_open_flags(), dir_fd=directory_descriptor
                )
            except OSError as exc:
                raise BaselineError(
                    "cannot bind execution directory for hardening"
                ) from exc
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise BaselineError("execution directory changed before hardening")
                _remove_execution_write_permissions(
                    child_descriptor,
                    depth=depth + 1,
                    visited=visited,
                )
                current = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    raise BaselineError(
                        "execution directory was replaced during hardening"
                    )
            finally:
                os.close(child_descriptor)
            continue
        if not stat.S_ISREG(before.st_mode):
            raise BaselineError("execution tree contains a special hardening target")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        except OSError as exc:
            raise BaselineError("cannot bind execution leaf for hardening") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise BaselineError("execution leaf changed before hardening")
            if opened.st_nlink != 1:
                raise BaselineError("execution regular leaf is hardlinked")
            os.fchmod(descriptor, stat.S_IMODE(opened.st_mode) & ~0o222)
            current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
                or current.st_size != opened.st_size
                or current.st_mtime_ns != opened.st_mtime_ns
                or stat.S_IMODE(current.st_mode) & 0o222
            ):
                raise BaselineError("execution leaf changed during hardening")
        except OSError as exc:
            raise BaselineError("cannot make execution leaf read-only") from exc
        finally:
            os.close(descriptor)
    try:
        info = os.fstat(directory_descriptor)
        os.fchmod(directory_descriptor, stat.S_IMODE(info.st_mode) & ~0o222)
        hardened = os.fstat(directory_descriptor)
    except OSError as exc:
        raise BaselineError("cannot make execution directory read-only") from exc
    if stat.S_IMODE(hardened.st_mode) & 0o222:
        raise BaselineError("execution directory remains writable after hardening")


def _harden_execution_tree_read_only(execution_root: Path) -> None:
    """Contain cache/state writes while exact post-suite checks remain authoritative."""

    root, root_descriptor = _open_root_directory(
        execution_root, "execution-tree hardening"
    )
    try:
        _remove_execution_write_permissions(
            root_descriptor,
            depth=0,
            visited=[0],
        )
    finally:
        os.close(root_descriptor)
    if os.access(root, os.W_OK):
        raise BaselineError(
            "materialized execution tree remains writable for the capture identity"
        )


def _execution_tree_structure(
    root: Path, repository: str
) -> dict[str, tuple[Any, ...]]:
    object_format = _git_text(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise BaselineError(f"unsupported Git object format: {object_format}")
    nested_roots = {
        path.parts[0]
        for name, path in REPOSITORY_PATHS.items()
        if repository == "accelerate" and name != "accelerate"
    }
    structure: dict[str, tuple[Any, ...]] = {}
    stack: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    visited = 0
    hashed_bytes = 0
    while stack:
        directory, relative_parts = stack.pop()
        if len(relative_parts) > MAX_SOURCE_SCAN_DEPTH:
            raise BaselineError("execution-tree scan exceeded its depth limit")
        try:
            with os.scandir(directory) as stream:
                entries = sorted(stream, key=lambda entry: entry.name)
        except OSError as exc:
            raise BaselineError(
                "cannot inspect the materialized execution tree"
            ) from exc
        for entry in entries:
            visited += 1
            if visited > MAX_SOURCE_SCAN_ENTRIES:
                raise BaselineError("execution-tree scan exceeded its entry limit")
            try:
                entry.name.encode("utf-8", "strict")
            except UnicodeEncodeError as exc:
                raise BaselineError("execution-tree path is not valid UTF-8") from exc
            parts = (*relative_parts, entry.name)
            if entry.name == ".git" and not relative_parts:
                continue
            if len(parts) == 1 and entry.name in nested_roots:
                continue
            relative = PurePosixPath(*parts).as_posix()
            info = entry.stat(follow_symlinks=False)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                structure[relative] = ("directory", mode)
                stack.append((Path(entry.path), parts))
            elif stat.S_ISREG(info.st_mode):
                if info.st_size > MAX_EXECUTION_LEAF_BYTES:
                    raise BaselineError(
                        f"execution-tree leaf exceeds its hash limit: {relative}"
                    )
                hashed_bytes += info.st_size
                if hashed_bytes > MAX_EXECUTION_TREE_HASH_BYTES:
                    raise BaselineError("execution-tree hash budget exceeded")
                digest = hashlib.sha256()
                git_digest = hashlib.new(object_format)
                git_digest.update(f"blob {info.st_size}\0".encode("ascii"))
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(entry.path, flags)
                    with os.fdopen(descriptor, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        identity = (
                            info.st_dev,
                            info.st_ino,
                            info.st_mode,
                            info.st_size,
                            info.st_mtime_ns,
                            info.st_ctime_ns,
                        )
                        if identity != (
                            opened.st_dev,
                            opened.st_ino,
                            opened.st_mode,
                            opened.st_size,
                            opened.st_mtime_ns,
                            opened.st_ctime_ns,
                        ):
                            raise BaselineError(
                                f"execution-tree leaf changed before hashing: {relative}"
                            )
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                            git_digest.update(chunk)
                        finished = os.fstat(stream.fileno())
                        if identity != (
                            finished.st_dev,
                            finished.st_ino,
                            finished.st_mode,
                            finished.st_size,
                            finished.st_mtime_ns,
                            finished.st_ctime_ns,
                        ):
                            raise BaselineError(
                                f"execution-tree leaf changed while hashing: {relative}"
                            )
                except BaselineError:
                    raise
                except OSError as exc:
                    raise BaselineError(
                        f"cannot hash execution-tree leaf: {relative}"
                    ) from exc
                structure[relative] = (
                    "regular",
                    mode,
                    info.st_size,
                    "sha256:" + digest.hexdigest(),
                    f"{object_format}:" + git_digest.hexdigest(),
                )
            elif stat.S_ISLNK(info.st_mode):
                target = os.readlink(entry.path)
                try:
                    target.encode("utf-8", "strict")
                except UnicodeEncodeError as exc:
                    raise BaselineError(
                        f"execution-tree symlink target is not UTF-8: {relative}"
                    ) from exc
                target_bytes = target.encode("utf-8")
                git_digest = hashlib.new(object_format)
                git_digest.update(f"blob {len(target_bytes)}\0".encode("ascii"))
                git_digest.update(target_bytes)
                structure[relative] = (
                    "symlink",
                    mode,
                    target,
                    f"{object_format}:" + git_digest.hexdigest(),
                )
            else:
                structure[relative] = ("special", mode)
    return structure


def _assert_execution_trees_clean(
    execution_root: Path,
    snapshots: Mapping[str, Mapping[str, Any]],
    expected_structure: Mapping[str, Mapping[str, tuple[Any, ...]]] | None = None,
) -> dict[str, dict[str, tuple[Any, ...]]]:
    structures: dict[str, dict[str, tuple[Any, ...]]] = {}
    for repository, relative in REPOSITORY_PATHS.items():
        unresolved_target = execution_root / relative
        if unresolved_target.is_symlink():
            raise BaselineError(f"{repository} execution root is a symlink")
        target = unresolved_target.resolve()
        if not target.is_relative_to(execution_root.resolve()):
            raise BaselineError(f"{repository} execution root escapes the capture")
        _assert_no_git_object_replacement(
            target, f"{repository} materialized repository"
        )
        if (
            _git_text(target, "rev-parse", "HEAD")
            != snapshots[repository]["tested_revision"]
        ):
            raise BaselineError(f"{repository} execution revision changed")
        if (
            _git_text(target, "rev-parse", "HEAD^{tree}")
            != snapshots[repository]["tested_tree"]
        ):
            raise BaselineError(f"{repository} execution tree changed")
        if _git_text(target, "remote"):
            raise BaselineError(f"{repository} execution clone has a writable remote")
        git_directory = Path(_git_text(target, "rev-parse", "--absolute-git-dir"))
        if not git_directory.is_relative_to(target):
            raise BaselineError(f"{repository} execution Git directory is external")
        alternates = git_directory / "objects" / "info" / "alternates"
        if alternates.exists() or alternates.is_symlink():
            raise BaselineError(
                f"{repository} execution clone uses a shared object alternate"
            )
        status = _run_readonly(
            (
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=matching",
            ),
            cwd=target,
        )
        if status.returncode != 0:
            raise BaselineError(f"cannot inspect {repository} execution status")
        records = {
            item.decode("utf-8", "strict")
            for item in status.stdout.split(b"\0")
            if item
        }
        allowed = (
            {
                "!! ipfs_datasets_py/",
                "!! ipfs_kit_py/",
                " M ipfs_datasets_py",
                " M ipfs_kit_py",
                *(f" D {path}" for path in UNMATERIALIZED_GITLINKS[repository]),
            }
            if repository == "accelerate"
            else {f" D {path}" for path in UNMATERIALIZED_GITLINKS[repository]}
        )
        if records - allowed:
            raise BaselineError(
                f"{repository} materialized execution tree is not clean: {sorted(records - allowed)}"
            )
        structures[repository] = _execution_tree_structure(target, repository)
        tracked_result = _run_readonly(
            ("git", "ls-tree", "-r", "-z", "HEAD"), cwd=target
        )
        if tracked_result.returncode != 0:
            raise BaselineError(f"cannot enumerate {repository} tracked leaves")
        tracked_entries: dict[str, tuple[str, str, str]] = {}
        try:
            for raw in tracked_result.stdout.split(b"\0"):
                if not raw:
                    continue
                header, raw_path = raw.split(b"\t", 1)
                mode, object_type, object_id = header.decode("ascii").split(" ")
                path = raw_path.decode("utf-8", "strict")
                if path in tracked_entries:
                    raise BaselineError(
                        f"{repository} exact Git tree contains a duplicate path"
                    )
                tracked_entries[path] = (mode, object_type, object_id)
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError) as exc:
            raise BaselineError(
                f"{repository} exact Git tree is malformed"
            ) from exc
        index_result = _run_readonly(
            ("git", "ls-files", "--stage", "-z"), cwd=target
        )
        if index_result.returncode != 0:
            raise BaselineError(f"cannot enumerate {repository} index leaves")
        index_entries: dict[str, tuple[str, str]] = {}
        try:
            for raw in index_result.stdout.split(b"\0"):
                if not raw:
                    continue
                header, raw_path = raw.split(b"\t", 1)
                mode, object_id, stage = header.decode("ascii").split(" ")
                path = raw_path.decode("utf-8", "strict")
                if stage != "0" or path in index_entries:
                    raise BaselineError(
                        f"{repository} execution index is not a stage-zero projection"
                    )
                index_entries[path] = (mode, object_id)
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError) as exc:
            raise BaselineError(f"{repository} execution index is malformed") from exc
        head_index_entries = {
            path: (mode, object_id)
            for path, (mode, _object_type, object_id) in tracked_entries.items()
        }
        if index_entries != head_index_entries:
            raise BaselineError(
                f"{repository} execution index differs from the exact Git tree"
            )

        if repository == "accelerate":
            reviewed_gitlinks = {
                REPOSITORY_PATHS[name].as_posix(): snapshots[name]["tested_revision"]
                for name in ("datasets", "kit")
            }
            for path, revision in reviewed_gitlinks.items():
                entry = tracked_entries.pop(path, None)
                if entry != ("160000", "commit", revision):
                    raise BaselineError(
                        f"accelerate reviewed Git link is invalid: {path}"
                    )
        for path, revision in UNMATERIALIZED_GITLINKS[repository].items():
            entry = tracked_entries.pop(path, None)
            if entry != ("160000", "commit", revision):
                raise BaselineError(
                    f"{repository} unmaterialized Git link is invalid: {path}"
                )
            if path in structures[repository]:
                raise BaselineError(
                    f"{repository} unmaterialized Git link exists on disk: {path}"
                )
        if any(object_type != "blob" for _, object_type, _ in tracked_entries.values()):
            raise BaselineError(f"{repository} exact Git tree has an unreviewed Git link")

        tracked_leaves = set(tracked_entries)
        expected_directories = {
            PurePosixPath(*path.parts[:index]).as_posix()
            for relative_path in tracked_leaves
            for path in (PurePosixPath(relative_path),)
            for index in range(1, len(path.parts))
        }
        actual_leaves = {
            path
            for path, description in structures[repository].items()
            if description[0] != "directory"
        }
        actual_directories = {
            path
            for path, description in structures[repository].items()
            if description[0] == "directory"
        }
        if actual_leaves != tracked_leaves or actual_directories != expected_directories:
            raise BaselineError(
                f"{repository} materialized filesystem differs from tracked leaves"
            )
        object_format = _git_text(target, "rev-parse", "--show-object-format")
        for path, (mode, object_type, object_id) in tracked_entries.items():
            description = structures[repository][path]
            expected_kind = "symlink" if mode == "120000" else "regular"
            expected_mode = "100755" if description[1] & 0o111 else "100644"
            if (
                object_type != "blob"
                or description[0] != expected_kind
                or (expected_kind == "regular" and mode != expected_mode)
                or description[-1] != f"{object_format}:{object_id}"
            ):
                raise BaselineError(
                    f"{repository} materialized leaf differs from exact Git tree: {path}"
                )
        if (
            expected_structure is not None
            and structures[repository] != expected_structure[repository]
        ):
            raise BaselineError(
                f"{repository} execution filesystem structure changed during capture"
            )
        _assert_no_git_object_replacement(
            target, f"{repository} materialized repository"
        )
    return structures


def _capture_command(
    repo_root: Path,
    execution_root: Path,
    suite: Suite,
    capture_id: str,
    python_info: Mapping[str, str],
    pytest_info: Mapping[str, Any],
) -> dict[str, Any]:
    workspace_relative = (
        ARTIFACT_RELATIVE_ROOT / "work" / capture_id / suite.id
    ).as_posix()
    workspace = _ensure_artifact_directory(repo_root, workspace_relative)
    basetemp = workspace / "pytest"
    environment = _environment(
        repo_root, workspace_relative, pytest_info["module_path"]
    )
    for child in (
        "home",
        "tmp",
        "ipfs-repo",
        "pycache",
        "hypothesis",
        "pytest-benchmark",
    ):
        _ensure_artifact_directory(repo_root, f"{workspace_relative}/{child}")
    argv = _resolved_argv(suite, python_info["executable"], basetemp)
    cwd = (execution_root / suite.cwd).resolve()
    start_utc = _utc_now()
    start_ns = time.monotonic_ns()
    capture_status = "completed"
    exit_code: int | None = None
    raw = b""
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        raw, capture_status, exit_code = _communicate_bounded(
            process, suite.timeout_seconds
        )
    except OSError as exc:
        capture_status = "launch_failed"
        exit_code = None
        raw = f"BASELINE_CAPTURE_LAUNCH_FAILED {type(exc).__name__}: {exc}\n".encode()
    finally:
        duration_ns = time.monotonic_ns() - start_ns
        finish_utc = _utc_now()

    _assert_public_log_safe(raw)
    log_relative = (
        ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
    ).as_posix()
    _atomic_write(repo_root, log_relative, raw)
    try:
        parsed = parse_pytest_log(raw)
        parse_error: str | None = None
    except BaselineError as exc:
        parsed = {
            "outcome_counts": {**{key: 0 for key in OUTCOME_KEYS}, "selected": 0},
            "collected_count": None,
            "collection_complete": False,
            "non_pass_nodes": [],
            "summary_line": "",
        }
        parse_error = str(exc)

    command_payload = {
        "id": suite.id,
        "evidence_type": "pytest_execution_observation",
        "suite_definition_digest": suite_definition_digest(suite),
        "command_digest": "",
        "argv": argv,
        "cwd": suite.cwd,
        "workspace_relative_path": workspace_relative,
        "environment": {
            "policy_id": ENVIRONMENT_POLICY_ID,
            "variables": environment,
        },
        "python": dict(python_info),
        "pytest": dict(pytest_info),
        "started_at": start_utc,
        "finished_at": finish_utc,
        "duration_ns": duration_ns,
        "timeout_seconds": suite.timeout_seconds,
        "capture_status": capture_status,
        "exit_code": exit_code,
        "collected_count": parsed["collected_count"],
        "collection_complete": parsed["collection_complete"],
        "outcome_counts": parsed["outcome_counts"],
        "non_pass_nodes": parsed["non_pass_nodes"],
        "summary_line": parsed["summary_line"],
        "parse_error": parse_error,
        "log": {
            "relative_path": log_relative,
            "bytes": len(raw),
            "sha256": _sha256(raw),
        },
        "assurance": _assurance_payload(
            process_observed=process is not None, aggregate=False
        ),
    }
    command_payload["command_digest"] = _sha256(
        _canonical_bytes(
            {
                "id": suite.id,
                "argv": argv,
                "cwd": suite.cwd,
                "environment": command_payload["environment"],
            }
        )
    )
    return command_payload


def _assert_capture_sources(
    snapshots: Mapping[str, Mapping[str, Any]], phase: str
) -> None:
    for repository, snapshot in snapshots.items():
        if not snapshot.get("clean"):
            paths = snapshot.get("untracked_paths", [])
            raise BaselineError(
                f"{repository} source is not clean {phase}; untracked={paths!r}"
            )
        fingerprint = snapshot.get("ignored_sensitive_fingerprint")
        if not isinstance(fingerprint, Mapping):
            raise BaselineError(
                f"{repository} ignored-input state is unavailable {phase}"
            )
        _assert_ignored_sensitive_input_policy(repository, fingerprint, phase)


def _assert_sources_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> None:
    for name in REPOSITORY_PATHS:
        if before[name]["tested_revision"] != after[name]["tested_revision"]:
            raise BaselineError(f"{name} revision changed during capture")
        if before[name]["tested_tree"] != after[name]["tested_tree"]:
            raise BaselineError(f"{name} tree changed during capture")
        if (
            before[name]["ignored_sensitive_fingerprint"]
            != after[name]["ignored_sensitive_fingerprint"]
        ):
            raise BaselineError(
                f"{name} ignored key/build/cache state changed during capture"
            )


def _remove_directory_contents(directory_descriptor: int) -> None:
    """Remove a held directory tree without following any contained symlink."""

    try:
        info = os.fstat(directory_descriptor)
        os.fchmod(
            directory_descriptor,
            stat.S_IMODE(info.st_mode) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
        )
        names = sorted(os.listdir(directory_descriptor))
    except OSError as exc:
        raise BaselineError("cannot enumerate capture workspace for cleanup") from exc
    for name in names:
        try:
            name.encode("utf-8", "strict")
            before = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
        except (OSError, UnicodeEncodeError) as exc:
            raise BaselineError("cannot inspect capture workspace for cleanup") from exc
        if stat.S_ISDIR(before.st_mode):
            try:
                child_descriptor = os.open(
                    name, _directory_open_flags(), dir_fd=directory_descriptor
                )
            except OSError as exc:
                raise BaselineError("cannot bind capture cleanup directory") from exc
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise BaselineError("capture cleanup directory changed before open")
                _remove_directory_contents(child_descriptor)
                current = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    raise BaselineError("capture cleanup directory was replaced")
            finally:
                os.close(child_descriptor)
            try:
                os.rmdir(name, dir_fd=directory_descriptor)
            except OSError as exc:
                raise BaselineError("cannot remove capture cleanup directory") from exc
        else:
            try:
                os.unlink(name, dir_fd=directory_descriptor)
            except OSError as exc:
                raise BaselineError("cannot remove capture cleanup leaf") from exc


def _safe_cleanup_capture_workspace(repo_root: Path, capture_id: str) -> None:
    if not re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z-\d+", capture_id):
        raise BaselineError("refusing to clean a malformed capture workspace")
    work_parts = (*PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix()).parts, "work")
    _, root_descriptor = _open_root_directory(repo_root, "capture cleanup")
    work_descriptor: int | None = None
    capture_descriptor: int | None = None
    try:
        work_descriptor = _try_open_relative_directory(
            root_descriptor, work_parts, label="capture cleanup parent"
        )
        if work_descriptor is None:
            return
        work_binding = os.fstat(work_descriptor)
        try:
            before = os.stat(
                capture_id, dir_fd=work_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(before.st_mode):
            raise BaselineError("refusing to clean a non-directory capture workspace")
        try:
            capture_descriptor = os.open(
                capture_id, _directory_open_flags(), dir_fd=work_descriptor
            )
        except OSError as exc:
            raise BaselineError("cannot bind capture workspace for cleanup") from exc
        opened = os.fstat(capture_descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise BaselineError("capture workspace changed before cleanup")
        _remove_directory_contents(capture_descriptor)
        current = os.stat(
            capture_id, dir_fd=work_descriptor, follow_symlinks=False
        )
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise BaselineError("capture workspace was replaced during cleanup")
        os.close(capture_descriptor)
        capture_descriptor = None
        os.rmdir(capture_id, dir_fd=work_descriptor)
        os.fsync(work_descriptor)
        _assert_relative_directory_binding(
            root_descriptor,
            work_parts,
            work_binding,
            label="capture cleanup parent",
        )
    except OSError as exc:
        raise BaselineError("capture workspace cleanup failed") from exc
    finally:
        if capture_descriptor is not None:
            os.close(capture_descriptor)
        if work_descriptor is not None:
            os.close(work_descriptor)
        os.close(root_descriptor)


def _safe_unlink_regular_file(repo_root: Path, relative: str, *, label: str) -> None:
    pure = _artifact_relative(relative, label)
    _, root_descriptor = _open_root_directory(repo_root, label)
    parent_descriptor: int | None = None
    try:
        parent_descriptor = _try_open_relative_directory(
            root_descriptor, pure.parts[:-1], label=f"{label} parent"
        )
        if parent_descriptor is None:
            return
        parent_binding = os.fstat(parent_descriptor)
        try:
            info = os.stat(
                pure.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode):
            raise BaselineError(f"refusing to remove a non-regular {label}")
        os.unlink(pure.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        _assert_relative_directory_binding(
            root_descriptor,
            pure.parts[:-1],
            parent_binding,
            label=f"{label} parent",
        )
    except OSError as exc:
        raise BaselineError(f"cannot safely remove {label}") from exc
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        os.close(root_descriptor)


def _safe_remove_capture_logs(
    repo_root: Path, repository: str, capture_id: str
) -> None:
    """Remove only diagnostics belonging to one failed, unpublishable capture."""

    if not re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z-\d+", capture_id):
        raise BaselineError("refusing to remove logs for a malformed capture id")
    for suite in SUITES_BY_REPOSITORY[repository]:
        relative = (
            ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
        ).as_posix()
        _safe_unlink_regular_file(repo_root, relative, label="failed capture log")


def _receipt_log_paths(payload: Mapping[str, Any], repository: str) -> tuple[str, ...]:
    """Extract only registry-bound log paths from one canonical receipt body."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BaselineError(f"{repository} prior receipt schema is invalid")
    if payload.get("repository") != repository:
        raise BaselineError(f"{repository} prior receipt identity is invalid")
    if payload.get("receipt_digest") != receipt_digest(payload):
        raise BaselineError(f"{repository} prior receipt digest is invalid")
    capture_id = payload.get("capture_id")
    if not isinstance(capture_id, str) or not re.fullmatch(
        r"\d{8}T\d{6}\.\d{6}Z-\d+", capture_id
    ):
        raise BaselineError(f"{repository} prior receipt capture id is invalid")
    suites = SUITES_BY_REPOSITORY[repository]
    if payload.get("required_command_ids") != [suite.id for suite in suites]:
        raise BaselineError(f"{repository} prior receipt suite set is invalid")
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != len(suites):
        raise BaselineError(f"{repository} prior receipt commands are incomplete")
    paths: list[str] = []
    for suite, command in zip(suites, commands, strict=True):
        if not isinstance(command, Mapping) or command.get("id") != suite.id:
            raise BaselineError(f"{repository} prior receipt command order is invalid")
        log = command.get("log")
        _require_exact_keys(
            log,
            frozenset({"relative_path", "bytes", "sha256"}),
            f"{suite.id} prior log",
        )
        expected = (
            ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
        ).as_posix()
        if log.get("relative_path") != expected:
            raise BaselineError(f"{suite.id} prior log path is invalid")
        paths.append(expected)
    return tuple(paths)


def _self_validate_payload(
    repo_root: Path, repository: str, payload: Mapping[str, Any]
) -> None:
    """Validate a candidate completely before the authoritative path is replaced."""

    _require_exact_keys(payload, TOP_LEVEL_KEYS, f"{repository} candidate receipt")
    if payload.get("receipt_digest") != receipt_digest(payload):
        raise BaselineError(f"{repository} candidate receipt digest is invalid")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BaselineError(f"{repository} candidate schema is invalid")
    if payload.get("operator_origin") != OPERATOR_ORIGIN:
        raise BaselineError(f"{repository} candidate origin is invalid")
    if payload.get("repository") != repository:
        raise BaselineError(f"{repository} candidate repository is invalid")
    expected_task = {"accelerate": "IPS-001", "datasets": "IPS-002", "kit": "IPS-003"}[
        repository
    ]
    if payload.get("task_id") != expected_task:
        raise BaselineError(f"{repository} candidate task is invalid")
    _parse_timestamp(payload.get("captured_at"), f"{repository}.captured_at")
    expected_ids = [suite.id for suite in SUITES_BY_REPOSITORY[repository]]
    if payload.get("required_command_ids") != expected_ids:
        raise BaselineError(f"{repository} candidate command set is invalid")
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != len(expected_ids):
        raise BaselineError(f"{repository} candidate commands are incomplete")
    capture_id = payload.get("capture_id")
    if not isinstance(capture_id, str) or not re.fullmatch(
        r"\d{8}T\d{6}\.\d{6}Z-\d+", capture_id
    ):
        raise BaselineError(f"{repository} candidate capture id is invalid")
    _validate_source_bindings(repo_root, repository, payload)
    for suite, command in zip(SUITES_BY_REPOSITORY[repository], commands, strict=True):
        if not isinstance(command, Mapping):
            raise BaselineError(f"{suite.id} candidate command is not an object")
        python_info = command.get("python")
        pytest_info = command.get("pytest")
        if not isinstance(python_info, Mapping) or not isinstance(pytest_info, Mapping):
            raise BaselineError(f"{suite.id} candidate interpreter binding is missing")
        _validate_command(
            repo_root, suite, command, python_info, pytest_info, capture_id
        )
    if payload.get("assurance") != _assurance_payload(
        process_observed=True, aggregate=True
    ):
        raise BaselineError(f"{repository} candidate assurance is invalid")


def _candidate_receipt_payload(
    repository: str,
    capture_id: str,
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tested = after[repository]
    execution = after["accelerate"]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "operator_origin": OPERATOR_ORIGIN,
        "repository": repository,
        "task_id": {
            "accelerate": "IPS-001",
            "datasets": "IPS-002",
            "kit": "IPS-003",
        }[repository],
        "capture_id": capture_id,
        "captured_at": _utc_now(),
        "required_command_ids": [
            suite.id for suite in SUITES_BY_REPOSITORY[repository]
        ],
        "planning_revision": tested["planning_revision"],
        "planning_tree": tested["planning_tree"],
        "source_revision": tested["tested_revision"],
        "source_tree": tested["tested_tree"],
        "execution_head": execution["tested_revision"],
        "execution_tree": execution["tested_tree"],
        "source_revisions": {
            name: snapshot["tested_revision"] for name, snapshot in after.items()
        },
        "source_trees": {
            name: snapshot["tested_tree"] for name, snapshot in after.items()
        },
        "source_clean_before": {
            name: snapshot["clean"] for name, snapshot in before.items()
        },
        "source_clean_after": {
            name: snapshot["clean"] for name, snapshot in after.items()
        },
        "ignored_sensitive_inputs": _ignored_sensitive_inputs_payload(after),
        "git_environment_policy_id": GIT_ENVIRONMENT_POLICY_ID,
        "commands": list(commands),
        "assurance": _assurance_payload(
            process_observed=all(
                command["assurance"]["process_observed"] for command in commands
            ),
            aggregate=True,
        ),
        "receipt_digest": "",
    }
    payload["receipt_digest"] = receipt_digest(payload)
    return payload


def capture_repositories(
    repo_root: Path, repositories: Sequence[str]
) -> dict[str, Path]:
    selected = tuple(repositories)
    if selected != tuple(REPOSITORY_PATHS):
        raise BaselineError(
            "authoritative baseline capture is one-shot and all-repositories-only"
        )
    repo_root = repo_root.resolve()
    _validate_protected_suite_registry(repo_root)
    _ensure_artifact_directory(repo_root, ARTIFACT_RELATIVE_ROOT.as_posix())
    capture_id = (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-{os.getpid()}"
    )
    capture_lock = _acquire_capture_lock(repo_root, capture_id)
    any_receipt_published = False
    publication_complete = False
    cleanup_complete = False
    try:
        _assert_capture_lock_held(capture_lock)
        _assert_one_shot_capture_available(repo_root)
        before = _all_source_snapshots(repo_root)
        _assert_capture_sources(before, "before capture")
        if "kit" in selected:
            _assert_no_live_ipfs(repo_root)
        python_info = _python_metadata()
        pytest_info = _pytest_metadata()
        execution_root = _materialize_execution_trees(repo_root, capture_id, before)
        _harden_execution_tree_read_only(execution_root)
        execution_structure = _assert_execution_trees_clean(execution_root, before)
        commands_by_repository: dict[str, list[dict[str, Any]]] = {}
        for repository in selected:
            commands_by_repository[repository] = []
            for suite in SUITES_BY_REPOSITORY[repository]:
                _assert_capture_lock_held(capture_lock)
                commands_by_repository[repository].append(
                    _capture_command(
                        repo_root,
                        execution_root,
                        suite,
                        capture_id,
                        python_info,
                        pytest_info,
                    )
                )
                _assert_execution_trees_clean(
                    execution_root, before, execution_structure
                )
                _assert_capture_lock_held(capture_lock)
        _assert_capture_lock_held(capture_lock)
        after = _all_source_snapshots(repo_root)
        _assert_capture_sources(after, "after capture")
        _assert_sources_unchanged(before, after)
        payloads = {
            repository: _candidate_receipt_payload(
                repository,
                capture_id,
                before,
                after,
                commands_by_repository[repository],
            )
            for repository in selected
        }
        encoded: dict[str, bytes] = {}
        for repository, payload in payloads.items():
            encoded[repository] = _canonical_bytes(payload) + b"\n"
            if len(encoded[repository]) > MAX_RECEIPT_BYTES:
                raise BaselineError("candidate receipt exceeds its fixed size limit")
            _self_validate_payload(repo_root, repository, payload)
        destinations: dict[str, Path] = {}
        for repository in selected:
            _assert_capture_lock_held(capture_lock)
            destination_relative = (
                ARTIFACT_RELATIVE_ROOT / f"{repository}.json"
            ).as_posix()
            try:
                destinations[repository] = _create_once(
                    repo_root, destination_relative, encoded[repository]
                )
            except AtomicWriteError as exc:
                if exc.replaced:
                    any_receipt_published = True
                raise
            any_receipt_published = True
        publication_complete = True
        return destinations
    finally:
        try:
            if not any_receipt_published:
                for repository in selected:
                    _safe_remove_capture_logs(repo_root, repository, capture_id)
            _safe_cleanup_capture_workspace(repo_root, capture_id)
            cleanup_complete = True
        finally:
            if cleanup_complete and (
                not any_receipt_published or publication_complete
            ):
                _release_capture_lock(capture_lock)
            else:
                # A partial/ambiguous publication or cleanup failure remains
                # visibly locked and cannot be mistaken for resumable evidence.
                _close_capture_lock(capture_lock)


def capture_repository(repo_root: Path, repository: str) -> Path:
    del repo_root, repository
    raise BaselineError(
        "per-repository capture is disabled; use the one-shot all-repository capture"
    )


def _parse_timestamp(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise BaselineError(f"{field} must be a UTC RFC3339 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise BaselineError(f"{field} is malformed") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise BaselineError(f"{field} must use UTC")
    return parsed


def _safe_read_relative_file(
    repo_root: Path,
    pure: PurePosixPath,
    *,
    maximum: int,
    label: str,
) -> tuple[Path, bytes]:
    root, root_descriptor = _open_root_directory(repo_root, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_descriptor: int | None = None
    try:
        parent_descriptor = _open_relative_directory(
            root_descriptor,
            pure.parts[:-1],
            create=False,
            label=f"{label} parent",
        )
        parent_binding = os.fstat(parent_descriptor)
        descriptor = os.open(pure.name, flags, dir_fd=parent_descriptor)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise BaselineError(
                    f"{label} is not a regular non-symlink file: {pure.as_posix()}"
                )
            if before.st_size > maximum:
                raise BaselineError(f"{label} exceeds size limit: {pure.as_posix()}")
            raw = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        visible = os.stat(
            pure.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (visible.st_dev, visible.st_ino) != (before.st_dev, before.st_ino):
            raise BaselineError(f"{label} was replaced while it was being read")
        _assert_relative_directory_binding(
            root_descriptor,
            pure.parts[:-1],
            parent_binding,
            label=f"{label} parent",
        )
    except BaselineError:
        raise
    except OSError as exc:
        raise BaselineError(f"cannot safely open/read {label}: {pure.as_posix()}") from exc
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        os.close(root_descriptor)
    if len(raw) > maximum:
        raise BaselineError(f"{label} exceeds size limit: {pure.as_posix()}")
    binding_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    binding_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if binding_before != binding_after or len(raw) != before.st_size:
        raise BaselineError(
            f"{label} changed while it was being read: {pure.as_posix()}"
        )
    return root / PurePosixPath(*pure.parts), raw


def _safe_artifact_file(
    repo_root: Path, relative: Any, *, maximum: int
) -> tuple[Path, bytes]:
    pure = _artifact_relative(relative, "artifact file")
    return _safe_read_relative_file(
        repo_root, pure, maximum=maximum, label="artifact file"
    )


def _safe_fixed_repository_file(
    repo_root: Path, relative: str, *, maximum: int
) -> bytes:
    pure = _canonical_relative(relative, "fixed repository file")
    _, raw = _safe_read_relative_file(
        repo_root,
        pure,
        maximum=maximum,
        label="fixed repository file",
    )
    return raw


def _expected_argv(
    repo_root: Path, suite: Suite, command: Mapping[str, Any], python_executable: str
) -> list[str]:
    workspace_relative = command.get("workspace_relative_path")
    _canonical_relative(workspace_relative, f"{suite.id} workspace_relative_path")
    required_prefix = (ARTIFACT_RELATIVE_ROOT / "work").as_posix() + "/"
    if not workspace_relative.startswith(required_prefix):
        raise BaselineError(f"{suite.id} workspace is outside the fixed artifact root")
    workspace = _artifact_path(
        repo_root,
        workspace_relative,
        label=f"{suite.id} workspace",
        allow_missing=True,
    )
    return _resolved_argv(suite, python_executable, workspace / "pytest")


def _validate_command(
    repo_root: Path,
    suite: Suite,
    command: Mapping[str, Any],
    python_info: Mapping[str, str],
    pytest_info: Mapping[str, Any],
    capture_id: str = "",
) -> None:
    _require_exact_keys(command, COMMAND_KEYS, f"{suite.id} command")
    if command.get("id") != suite.id:
        raise BaselineError(f"expected command {suite.id}")
    if command.get("evidence_type") != "pytest_execution_observation":
        raise BaselineError(f"{suite.id} is not pytest execution evidence")
    if command.get("suite_definition_digest") != suite_definition_digest(suite):
        raise BaselineError(f"{suite.id} suite definition digest is invalid")
    if command.get("cwd") != suite.cwd:
        raise BaselineError(f"{suite.id} cwd is not the fixed cwd")
    if capture_id:
        expected_workspace = (
            ARTIFACT_RELATIVE_ROOT / "work" / capture_id / suite.id
        ).as_posix()
        if command.get("workspace_relative_path") != expected_workspace:
            raise BaselineError(f"{suite.id} workspace is not bound to the capture")
    _require_exact_keys(
        command.get("python"),
        frozenset({"executable", "implementation", "version"}),
        f"{suite.id} Python",
    )
    if command.get("python") != python_info:
        raise BaselineError(f"{suite.id} Python binding is internally inconsistent")
    if (
        python_info.get("implementation") != "CPython"
        or not isinstance(python_info.get("executable"), str)
        or not Path(python_info["executable"]).is_absolute()
        or not isinstance(python_info.get("version"), str)
        or not python_info["version"]
    ):
        raise BaselineError(f"{suite.id} Python binding is malformed")
    _require_exact_keys(
        command.get("pytest"),
        frozenset({"version", "module_path", "autoload_plugins"}),
        f"{suite.id} pytest",
    )
    if command.get("pytest") != pytest_info:
        raise BaselineError(f"{suite.id} pytest binding is internally inconsistent")
    if (
        not isinstance(pytest_info.get("version"), str)
        or not pytest_info["version"]
        or not isinstance(pytest_info.get("module_path"), str)
        or not Path(pytest_info["module_path"]).is_absolute()
        or not isinstance(pytest_info.get("autoload_plugins"), list)
    ):
        raise BaselineError(f"{suite.id} pytest binding is malformed")
    for index, plugin in enumerate(pytest_info["autoload_plugins"]):
        _require_exact_keys(
            plugin,
            frozenset({"name", "value", "distribution", "version"}),
            f"{suite.id} pytest plugin {index}",
        )
    expected_argv = _expected_argv(repo_root, suite, command, python_info["executable"])
    if command.get("argv") != expected_argv:
        raise BaselineError(f"{suite.id} argv is not the fixed command")
    expected_environment = {
        "policy_id": ENVIRONMENT_POLICY_ID,
        "variables": _environment(
            repo_root,
            command["workspace_relative_path"],
            pytest_info["module_path"],
        ),
    }
    _require_exact_keys(
        command.get("environment"),
        frozenset({"policy_id", "variables"}),
        f"{suite.id} environment",
    )
    if command.get("environment") != expected_environment:
        raise BaselineError(f"{suite.id} environment is not the fixed environment")
    command_preimage = {
        "id": suite.id,
        "argv": expected_argv,
        "cwd": suite.cwd,
        "environment": expected_environment,
    }
    if command.get("command_digest") != _sha256(_canonical_bytes(command_preimage)):
        raise BaselineError(f"{suite.id} command digest is invalid")
    if command.get("timeout_seconds") != suite.timeout_seconds:
        raise BaselineError(f"{suite.id} timeout differs from the bounded timeout")
    if command.get("capture_status") != "completed":
        raise BaselineError(f"{suite.id} capture did not complete")
    exit_code = command.get("exit_code")
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or exit_code not in range(5)
    ):
        raise BaselineError(f"{suite.id} exit code is not a pytest exit code")
    duration_ns = command.get("duration_ns")
    if (
        isinstance(duration_ns, bool)
        or not isinstance(duration_ns, int)
        or duration_ns <= 0
    ):
        raise BaselineError(f"{suite.id} duration must be positive")
    started = _parse_timestamp(command.get("started_at"), f"{suite.id}.started_at")
    finished = _parse_timestamp(command.get("finished_at"), f"{suite.id}.finished_at")
    if finished <= started:
        raise BaselineError(f"{suite.id} timestamps are not increasing")
    wall_ns = int((finished - started).total_seconds() * 1_000_000_000)
    if abs(wall_ns - duration_ns) > 5_000_000_000:
        raise BaselineError(f"{suite.id} duration disagrees with UTC timestamps")

    log = command.get("log")
    _require_exact_keys(
        log, frozenset({"relative_path", "bytes", "sha256"}), f"{suite.id} log"
    )
    if capture_id:
        expected_log = (
            ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
        ).as_posix()
        if log.get("relative_path") != expected_log:
            raise BaselineError(f"{suite.id} log is not bound to the capture")
    _, raw = _safe_artifact_file(
        repo_root, log.get("relative_path"), maximum=MAX_LOG_BYTES
    )
    if log.get("bytes") != len(raw) or log.get("sha256") != _sha256(raw):
        raise BaselineError(f"{suite.id} retained log hash or size is invalid")
    _assert_public_log_safe(raw)
    parsed = parse_pytest_log(raw)
    header = re.search(
        r"^platform\s+.+?\s+--\s+Python\s+(?P<python>[0-9.]+),\s+pytest-(?P<pytest>[0-9.]+)",
        _strip_ansi(raw),
        re.MULTILINE,
    )
    if header is None:
        raise BaselineError(f"{suite.id} log lacks concrete Python/pytest versions")
    if not python_info["version"].startswith(header.group("python")):
        raise BaselineError(f"{suite.id} Python version disagrees with retained log")
    if pytest_info["version"] != header.group("pytest"):
        raise BaselineError(f"{suite.id} pytest version disagrees with retained log")
    for key in (
        "outcome_counts",
        "collected_count",
        "collection_complete",
        "non_pass_nodes",
        "summary_line",
    ):
        if command.get(key) != parsed[key]:
            raise BaselineError(f"{suite.id} {key} disagrees with retained log")
    if command.get("parse_error") is not None:
        raise BaselineError(f"{suite.id} contains a parse error")
    counts = parsed["outcome_counts"]
    _require_exact_keys(
        counts, frozenset((*OUTCOME_KEYS, "selected")), f"{suite.id} counts"
    )
    if counts["selected"] <= 0:
        raise BaselineError(f"{suite.id} has zero pytest outcome evidence")
    if parsed["collection_complete"] and parsed["collected_count"] != (
        counts["selected"] + counts["deselected"]
    ):
        raise BaselineError(f"{suite.id} collection and outcome counts disagree")
    if exit_code == 0 and (counts["failed"] or counts["errors"]):
        raise BaselineError(f"{suite.id} claims success with failed/error outcomes")
    if exit_code == 1 and not (counts["failed"] or counts["errors"]):
        raise BaselineError(f"{suite.id} failure exit lacks failed/error outcomes")
    assurance = command.get("assurance")
    _require_exact_keys(assurance, ASSURANCE_KEYS, f"{suite.id} assurance")
    expected_assurance = _assurance_payload(process_observed=True, aggregate=False)
    if assurance != expected_assurance:
        raise BaselineError(f"{suite.id} assurance overstates the evidence")


def _validate_source_bindings(
    repo_root: Path, repository: str, payload: Mapping[str, Any]
) -> None:
    """Validate immutable tested commits, independent of the current branch tip."""

    source_revisions = payload.get("source_revisions")
    source_trees = payload.get("source_trees")
    if not isinstance(source_revisions, Mapping) or set(source_revisions) != set(
        REPOSITORY_PATHS
    ):
        raise BaselineError(f"{repository} source revisions are incomplete")
    if not isinstance(source_trees, Mapping) or set(source_trees) != set(
        REPOSITORY_PATHS
    ):
        raise BaselineError(f"{repository} source trees are incomplete")
    for name, relative in REPOSITORY_PATHS.items():
        revision = source_revisions[name]
        tree = source_trees[name]
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise BaselineError(f"{repository} recorded {name} revision is malformed")
        if not isinstance(tree, str) or not re.fullmatch(r"[0-9a-f]{40}", tree):
            raise BaselineError(f"{repository} recorded {name} tree is malformed")
        source_repo = (repo_root / relative).resolve()
        if _git_text(source_repo, "rev-parse", f"{revision}^{{tree}}") != tree:
            raise BaselineError(
                f"{repository} recorded {name} tree does not bind its commit"
            )
        ancestry = _run_readonly(
            ("git", "merge-base", "--is-ancestor", PLANNING_REVISIONS[name], revision),
            cwd=source_repo,
        )
        if ancestry.returncode != 0:
            raise BaselineError(
                f"{repository} recorded {name} revision has the wrong base"
            )
    expected_source_fields = {
        "planning_revision": PLANNING_REVISIONS[repository],
        "planning_tree": PLANNING_TREES[repository],
        "source_revision": source_revisions[repository],
        "source_tree": source_trees[repository],
        "execution_head": source_revisions["accelerate"],
        "execution_tree": source_trees["accelerate"],
        "source_clean_before": {name: True for name in REPOSITORY_PATHS},
        "source_clean_after": {name: True for name in REPOSITORY_PATHS},
        "git_environment_policy_id": GIT_ENVIRONMENT_POLICY_ID,
    }
    for field, expected in expected_source_fields.items():
        if payload.get(field) != expected:
            raise BaselineError(f"{repository} {field} source binding is invalid")
    _validate_ignored_sensitive_inputs_shape(
        payload.get("ignored_sensitive_inputs"), repository
    )


def _decode_git_paths(
    result: subprocess.CompletedProcess[bytes], label: str
) -> set[str]:
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise BaselineError(f"cannot determine {label}: {detail}")
    paths: set[str] = set()
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            path = raw_path.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise BaselineError(f"{label} contains a non-UTF-8 path") from exc
        _canonical_relative(path, label)
        paths.add(path)
    return paths


def _present_receipt_evidence_contract(
    repo_root: Path,
) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    """Derive exact evidence paths and scheduler pins from present receipts."""

    allowed_paths = {
        (ARTIFACT_RELATIVE_ROOT / f"{name}.json").as_posix()
        for name in REPOSITORY_PATHS
    }
    pinned_paths: set[str] = set()
    pins: dict[str, dict[str, Any]] = {}
    task_ids = {"accelerate": "IPS-001", "datasets": "IPS-002", "kit": "IPS-003"}
    artifact_pure = PurePosixPath(ARTIFACT_RELATIVE_ROOT.as_posix())
    artifact_entries = _safe_directory_entries(
        repo_root,
        artifact_pure,
        label="present receipt directory",
        allow_missing=True,
    ) or {}
    if CAPTURE_LOCK_NAME in artifact_entries:
        raise BaselineError(
            "present receipts are unadmitted while an active or stale capture lock exists"
        )
    for repository in REPOSITORY_PATHS:
        receipt_relative = (ARTIFACT_RELATIVE_ROOT / f"{repository}.json").as_posix()
        if f"{repository}.json" not in artifact_entries:
            continue
        _, raw = _safe_artifact_file(
            repo_root, receipt_relative, maximum=MAX_RECEIPT_BYTES
        )
        receipt = _load_json_object(raw, f"{repository} present receipt")
        _require_exact_keys(receipt, TOP_LEVEL_KEYS, f"{repository} present receipt")
        if raw != _canonical_bytes(receipt) + b"\n":
            raise BaselineError(f"{repository} present receipt is not canonical JSON")
        if receipt.get("receipt_digest") != receipt_digest(receipt):
            raise BaselineError(f"{repository} present receipt digest is invalid")
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("operator_origin") != OPERATOR_ORIGIN
            or receipt.get("repository") != repository
            or receipt.get("task_id") != task_ids[repository]
        ):
            raise BaselineError(f"{repository} present receipt identity is invalid")
        _validate_ignored_sensitive_inputs_shape(
            receipt.get("ignored_sensitive_inputs"), f"{repository} present receipt"
        )
        capture_id = receipt.get("capture_id")
        if not isinstance(capture_id, str) or not re.fullmatch(
            r"\d{8}T\d{6}\.\d{6}Z-\d+", capture_id
        ):
            raise BaselineError(f"{repository} present receipt capture id is invalid")
        suites = SUITES_BY_REPOSITORY[repository]
        required_ids = [suite.id for suite in suites]
        if receipt.get("required_command_ids") != required_ids:
            raise BaselineError(f"{repository} present receipt suite set is invalid")
        commands = receipt.get("commands")
        if not isinstance(commands, list) or len(commands) != len(suites):
            raise BaselineError(f"{repository} present receipt commands are incomplete")
        retained_logs: list[str] = []
        suite_digests: dict[str, str] = {}
        for suite, command in zip(suites, commands, strict=True):
            if not isinstance(command, Mapping) or command.get("id") != suite.id:
                raise BaselineError(
                    f"{repository} present receipt command order is invalid"
                )
            digest = command.get("suite_definition_digest")
            if digest != suite_definition_digest(suite):
                raise BaselineError(f"{suite.id} present suite digest is invalid")
            log = command.get("log")
            _require_exact_keys(
                log,
                frozenset({"relative_path", "bytes", "sha256"}),
                f"{suite.id} present log binding",
            )
            expected_log = (
                ARTIFACT_RELATIVE_ROOT / "logs" / f"{suite.id}-{capture_id}.log"
            ).as_posix()
            if log.get("relative_path") != expected_log:
                raise BaselineError(f"{suite.id} present log path is invalid")
            _, log_raw = _safe_artifact_file(
                repo_root, expected_log, maximum=MAX_LOG_BYTES
            )
            if log.get("bytes") != len(log_raw) or log.get("sha256") != _sha256(
                log_raw
            ):
                raise BaselineError(f"{suite.id} present log binding is invalid")
            _assert_public_log_safe(log_raw)
            retained_logs.append(expected_log)
            suite_digests[suite.id] = digest
            allowed_paths.add(expected_log)
            pinned_paths.add(expected_log)
        if receipt.get("assurance") != _assurance_payload(
            process_observed=True, aggregate=True
        ):
            raise BaselineError(f"{repository} present receipt assurance is invalid")
        for field in ("source_revision", "source_tree"):
            if not isinstance(receipt.get(field), str) or not re.fullmatch(
                r"[0-9a-f]{40}", receipt[field]
            ):
                raise BaselineError(f"{repository} present receipt {field} is invalid")
        if receipt.get("planning_revision") != PLANNING_REVISIONS[repository]:
            raise BaselineError(
                f"{repository} present receipt planning revision is invalid"
            )
        pinned_paths.add(receipt_relative)
        pins[task_ids[repository]] = {
            "path": receipt_relative,
            "receipt_digest": receipt["receipt_digest"],
            "planning_revision": receipt["planning_revision"],
            "source_revision": receipt["source_revision"],
            "source_tree": receipt["source_tree"],
            "required_command_ids": required_ids,
            "suite_definition_digests": suite_digests,
            "retained_log_paths": retained_logs,
        }
    log_root_relative = (ARTIFACT_RELATIVE_ROOT / "logs").as_posix()
    log_entries = _safe_directory_entries(
        repo_root,
        artifact_pure / "logs",
        label="present receipt log directory",
        allow_missing=True,
    )
    if log_entries is not None:
        actual_logs: set[str] = set()
        for name, info in log_entries.items():
            if not stat.S_ISREG(info.st_mode):
                raise BaselineError("present receipt log directory has an unsafe entry")
            actual_logs.add((ARTIFACT_RELATIVE_ROOT / "logs" / name).as_posix())
        expected_logs = {
            path for path in pinned_paths if path.startswith(log_root_relative + "/")
        }
        if actual_logs != expected_logs:
            raise BaselineError(
                "present receipt log directory contains missing or orphan logs"
            )
    return allowed_paths, pinned_paths, pins


def render_pin_projection(repo_root: Path) -> dict[str, Any]:
    """Validate a complete one-shot bundle and return its read-only config patch."""

    repo_root = repo_root.resolve()
    _validate_protected_suite_registry(repo_root)
    scheduler = _load_json_object(
        _safe_fixed_repository_file(
            repo_root, SCHEDULER_CONFIG_RELATIVE, maximum=MAX_RECEIPT_BYTES
        ),
        "pre-pin scheduler configuration",
    )
    if scheduler.get("operator_baseline_receipts") != {} or scheduler.get(
        "protected_paths"
    ) != list(PRE_CAPTURE_PROTECTED_PATHS):
        raise BaselineError(
            "render-pins requires the exact reviewed empty-pin scheduler phase"
        )
    _, pinned_paths, pins = _present_receipt_evidence_contract(repo_root)
    expected_tasks = {"IPS-001", "IPS-002", "IPS-003"}
    if set(pins) != expected_tasks:
        raise BaselineError("render-pins requires all three canonical receipts")

    capture_ids: set[str] = set()
    source_epochs: set[bytes] = set()
    for repository in REPOSITORY_PATHS:
        relative = (ARTIFACT_RELATIVE_ROOT / f"{repository}.json").as_posix()
        _, raw = _safe_artifact_file(
            repo_root, relative, maximum=MAX_RECEIPT_BYTES
        )
        payload = _load_json_object(raw, f"{repository} render-pins receipt")
        _self_validate_payload(repo_root, repository, payload)
        capture_ids.add(payload["capture_id"])
        source_epochs.add(
            _canonical_bytes(
                {
                    "source_revisions": payload["source_revisions"],
                    "source_trees": payload["source_trees"],
                }
            )
        )
    if len(capture_ids) != 1 or len(source_epochs) != 1:
        raise BaselineError("render-pins receipts do not share one capture source epoch")
    return {
        "operator_baseline_receipts": pins,
        "protected_paths": sorted(set(PRE_CAPTURE_PROTECTED_PATHS) | pinned_paths),
    }


def _scheduler_config_at_revision(repo_root: Path, revision: str) -> dict[str, Any]:
    result = _run_readonly(
        ("git", "show", f"{revision}:{SCHEDULER_CONFIG_RELATIVE}"), cwd=repo_root
    )
    if result.returncode != 0 or len(result.stdout) > MAX_RECEIPT_BYTES:
        raise BaselineError("cannot load the captured scheduler configuration")
    return _load_json_object(result.stdout, "captured scheduler configuration")


def _prior_receipt_logs_at_revision(
    repo_root: Path, revision: str, repository: str
) -> tuple[str, ...]:
    receipt_relative = (ARTIFACT_RELATIVE_ROOT / f"{repository}.json").as_posix()
    exists = _run_readonly(
        ("git", "cat-file", "-e", f"{revision}:{receipt_relative}"), cwd=repo_root
    )
    if exists.returncode != 0:
        return ()
    result = _run_readonly(
        ("git", "show", f"{revision}:{receipt_relative}"), cwd=repo_root
    )
    if result.returncode != 0 or len(result.stdout) > MAX_RECEIPT_BYTES:
        raise BaselineError(f"cannot load {repository} prior receipt from tested tree")
    payload = _load_json_object(result.stdout, f"{repository} prior tested receipt")
    _require_exact_keys(payload, TOP_LEVEL_KEYS, f"{repository} prior tested receipt")
    if result.stdout != _canonical_bytes(payload) + b"\n":
        raise BaselineError(f"{repository} prior tested receipt is not canonical")
    paths = _receipt_log_paths(payload, repository)
    for command, relative in zip(payload["commands"], paths, strict=True):
        log_result = _run_readonly(
            ("git", "show", f"{revision}:{relative}"), cwd=repo_root
        )
        if log_result.returncode != 0 or len(log_result.stdout) > MAX_LOG_BYTES:
            raise BaselineError(f"cannot load prior retained log {relative}")
        log = command["log"]
        if log["bytes"] != len(log_result.stdout) or log["sha256"] != _sha256(
            log_result.stdout
        ):
            raise BaselineError(f"prior retained log binding is invalid: {relative}")
        _assert_public_log_safe(log_result.stdout)
    return paths


def _pinned_retained_log_paths(value: Any) -> set[str]:
    if not isinstance(value, Mapping):
        raise BaselineError("captured scheduler receipt pins are malformed")
    paths: set[str] = set()
    log_prefix = (ARTIFACT_RELATIVE_ROOT / "logs").as_posix() + "/"
    for task_id, pin in value.items():
        if not isinstance(task_id, str) or not isinstance(pin, Mapping):
            raise BaselineError("captured scheduler receipt pins are malformed")
        retained = pin.get("retained_log_paths")
        if not isinstance(retained, list) or any(
            not isinstance(path, str) for path in retained
        ):
            raise BaselineError("captured scheduler retained log paths are malformed")
        for path in retained:
            canonical = _canonical_relative(
                path, "captured retained log path"
            ).as_posix()
            if not canonical.startswith(log_prefix) or canonical in paths:
                raise BaselineError(
                    "captured scheduler retained log paths are malformed"
                )
            paths.add(canonical)
    return paths


def _validate_scheduler_config_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    pins: Mapping[str, Any],
    pinned_paths: set[str],
) -> None:
    before_fixed = dict(before)
    after_fixed = dict(after)
    before_pins = before_fixed.pop("operator_baseline_receipts", None)
    after_pins = after_fixed.pop("operator_baseline_receipts", None)
    before_protected = before_fixed.pop("protected_paths", None)
    after_protected = after_fixed.pop("protected_paths", None)
    if before_fixed != after_fixed:
        raise BaselineError(
            "post-capture scheduler change is not limited to exact receipt pins and protected paths"
        )
    prior_log_paths = _pinned_retained_log_paths(before_pins)
    if (
        not isinstance(before_protected, list)
        or any(not isinstance(path, str) for path in before_protected)
        or len(before_protected) != len(set(before_protected))
    ):
        raise BaselineError("captured scheduler protected paths are malformed")
    if after_pins != pins:
        raise BaselineError(
            "current scheduler receipt pins are not the exact receipt pins"
        )
    expected_protected = (set(before_protected) - prior_log_paths) | pinned_paths
    if (
        not isinstance(after_protected, list)
        or any(not isinstance(path, str) for path in after_protected)
        or len(after_protected) != len(set(after_protected))
        or set(after_protected) != expected_protected
    ):
        raise BaselineError(
            "current scheduler protected paths are not the exact pinned evidence paths"
        )


def _current_accelerate_admissible_paths(
    repo_root: Path, payload: Mapping[str, Any]
) -> set[str]:
    allowed_paths, pinned_paths, pins = _present_receipt_evidence_contract(repo_root)
    captured_config = _scheduler_config_at_revision(
        repo_root, payload["source_revisions"]["accelerate"]
    )
    current_config = _load_json_object(
        _safe_fixed_repository_file(
            repo_root, SCHEDULER_CONFIG_RELATIVE, maximum=MAX_RECEIPT_BYTES
        ),
        "current scheduler configuration",
    )
    _validate_scheduler_config_transition(
        captured_config, current_config, pins, pinned_paths
    )
    for repository in REPOSITORY_PATHS:
        allowed_paths.update(
            _prior_receipt_logs_at_revision(
                repo_root, payload["source_revisions"]["accelerate"], repository
            )
        )
    allowed_paths.add(SCHEDULER_CONFIG_RELATIVE)
    return allowed_paths


def _validate_current_ignored_sensitive_inputs(
    repo_root: Path, payload: Mapping[str, Any]
) -> None:
    del repo_root
    expected = {
        "policy_id": IGNORED_INPUT_POLICY_ID,
        "repositories": {
            repository: _ignored_sensitive_binding({})
            for repository in REPOSITORY_PATHS
        },
    }
    if payload.get("ignored_sensitive_inputs") != expected:
        raise BaselineError("receipt does not bind clean materialized execution trees")


def _validate_current_relevance(repo_root: Path, payload: Mapping[str, Any]) -> None:
    """Reject receipts whose tested bytes no longer describe the current sources."""

    source_revisions = payload["source_revisions"]
    _validate_current_ignored_sensitive_inputs(repo_root, payload)
    accelerate_allowed_paths = _current_accelerate_admissible_paths(repo_root, payload)
    for name, relative in REPOSITORY_PATHS.items():
        repo = (repo_root / relative).resolve()
        tested_revision = source_revisions[name]
        ancestry = _run_readonly(
            ("git", "merge-base", "--is-ancestor", tested_revision, "HEAD"), cwd=repo
        )
        if ancestry.returncode != 0:
            raise BaselineError(
                f"{name} current HEAD does not descend from the tested revision"
            )
        changed: set[str] = set()
        changed.update(
            _decode_git_paths(
                _run_readonly(
                    (
                        "git",
                        "diff",
                        "--name-only",
                        "-z",
                        "--ignore-submodules=none",
                        f"{tested_revision}..HEAD",
                        "--",
                    ),
                    cwd=repo,
                ),
                f"{name} committed changes",
            )
        )
        for arguments, label in (
            (
                ("git", "diff", "--name-only", "-z", "--ignore-submodules=none", "--"),
                "unstaged changes",
            ),
            (
                (
                    "git",
                    "diff",
                    "--cached",
                    "--name-only",
                    "-z",
                    "--ignore-submodules=none",
                    "--",
                ),
                "staged changes",
            ),
            (
                ("git", "ls-files", "--others", "--exclude-standard", "-z"),
                "untracked changes",
            ),
        ):
            changed.update(
                _decode_git_paths(_run_readonly(arguments, cwd=repo), f"{name} {label}")
            )
        disallowed = sorted(
            path
            for path in changed
            if name != "accelerate" or path not in accelerate_allowed_paths
        )
        if disallowed:
            raise BaselineError(
                f"{name} current state contains changes outside post-capture evidence paths: "
                f"{disallowed}"
            )


def validate_repository(repo_root: Path, repository: str) -> Path:
    if repository not in SUITES_BY_REPOSITORY:
        raise BaselineError(f"unknown repository {repository!r}")
    repo_root = repo_root.resolve()
    _validate_protected_suite_registry(repo_root)
    receipt_path = repo_root / ARTIFACT_RELATIVE_ROOT / f"{repository}.json"
    _, raw = _safe_artifact_file(
        repo_root,
        (ARTIFACT_RELATIVE_ROOT / f"{repository}.json").as_posix(),
        maximum=MAX_RECEIPT_BYTES,
    )
    payload = _load_json_object(raw, f"{repository} receipt")
    _require_exact_keys(payload, TOP_LEVEL_KEYS, f"{repository} receipt")
    if raw != _canonical_bytes(payload) + b"\n":
        raise BaselineError(f"{repository} receipt is not canonical JSON")
    if payload.get("receipt_digest") != receipt_digest(payload):
        raise BaselineError(f"{repository} canonical receipt digest is invalid")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BaselineError(f"{repository} receipt schema is unknown")
    if payload.get("operator_origin") != OPERATOR_ORIGIN:
        raise BaselineError(f"{repository} receipt is not operator capture evidence")
    if payload.get("repository") != repository:
        raise BaselineError(f"{repository} receipt repository binding is invalid")
    expected_task = {"accelerate": "IPS-001", "datasets": "IPS-002", "kit": "IPS-003"}[
        repository
    ]
    if payload.get("task_id") != expected_task:
        raise BaselineError(f"{repository} task binding is invalid")
    required_ids = [suite.id for suite in SUITES_BY_REPOSITORY[repository]]
    if payload.get("required_command_ids") != required_ids:
        raise BaselineError(
            f"{repository} required command set is incomplete or reordered"
        )
    _validate_source_bindings(repo_root, repository, payload)
    _validate_current_relevance(repo_root, payload)
    _parse_timestamp(payload.get("captured_at"), f"{repository}.captured_at")
    if not isinstance(payload.get("capture_id"), str) or not re.fullmatch(
        r"\d{8}T\d{6}\.\d{6}Z-\d+", payload["capture_id"]
    ):
        raise BaselineError(f"{repository} capture_id is missing")
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != len(required_ids):
        raise BaselineError(f"{repository} command records are incomplete")
    for suite, command in zip(SUITES_BY_REPOSITORY[repository], commands, strict=True):
        if not isinstance(command, Mapping):
            raise BaselineError(f"{suite.id} command record must be an object")
        python_info = command.get("python")
        pytest_info = command.get("pytest")
        if not isinstance(python_info, Mapping) or not isinstance(pytest_info, Mapping):
            raise BaselineError(f"{suite.id} interpreter binding is missing")
        _validate_command(
            repo_root, suite, command, python_info, pytest_info, payload["capture_id"]
        )
    assurance = payload.get("assurance")
    _require_exact_keys(assurance, ASSURANCE_KEYS, f"{repository} assurance")
    if assurance != _assurance_payload(process_observed=True, aggregate=True):
        raise BaselineError(f"{repository} assurance overstates the evidence")
    return receipt_path


def _repositories(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return tuple(REPOSITORY_PATHS)
    if selection not in REPOSITORY_PATHS:
        raise BaselineError(f"unknown repository selection {selection!r}")
    return (selection,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_command = subparsers.add_parser("capture")
    capture_command.add_argument("--repository", choices=("all",), default="all")
    validate_command = subparsers.add_parser("validate-only")
    validate_command.add_argument(
        "--repository", choices=("all", *REPOSITORY_PATHS), default="all"
    )
    subparsers.add_parser("render-pins")
    subparsers.add_parser("list")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    try:
        registry = _validate_protected_suite_registry(root)
        if args.action == "list":
            print(json.dumps(registry, sort_keys=True, indent=2))
            return 0
        if args.action == "render-pins":
            sys.stdout.buffer.write(_canonical_bytes(render_pin_projection(root)) + b"\n")
            return 0
        repositories = _repositories(args.repository)
        if args.action == "capture":
            paths = capture_repositories(root, repositories)
            for repository, path in paths.items():
                print(f"captured {repository}: {path.relative_to(root)}")
        else:
            for repository in repositories:
                path = validate_repository(root, repository)
                print(f"valid {repository}: {path.relative_to(root)}")
    except BaselineError as exc:
        print(f"baseline receipt error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
