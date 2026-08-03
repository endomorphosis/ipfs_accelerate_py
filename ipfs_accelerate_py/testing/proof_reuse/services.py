"""Lazy, allowlisted service assembly for pytest proof reuse.

This module is deliberately inert when imported.  It does not inspect the
environment, create a cache, import an optional provider, install a package,
or perform a proof operation.  :mod:`.plugin` constructs a resolver only after
proof reuse has been explicitly enabled and an exact item identity needs a
lookup or publication.

The resolver has a closed dependency vocabulary.  An installer can only be
asked for one of those entries, and a failed import/install/construction
returns an unavailable bundle.  Callers must consequently execute tests.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Final

PROOF_REUSE_AUTO_INSTALL_ENV: Final = "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"
PROOF_REUSE_CACHE_DIR_ENV: Final = "IPFS_TEST_PROOF_REUSE_CACHE_DIR"
PROOF_REUSE_DATASETS_SOURCE_ENV: Final = "IPFS_TEST_PROOF_REUSE_DATASETS_SOURCE"
PROOF_REUSE_NLTK_DOWNLOAD_ENV: Final = "IPFS_TEST_PROOF_REUSE_NLTK_DOWNLOAD"
PROOF_REUSE_NLTK_DATA_DIR_ENV: Final = "IPFS_TEST_PROOF_REUSE_NLTK_DATA_DIR"
PROOF_REUSE_GROTH16_BUILD_ENV: Final = "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD"
PROOF_REUSE_GROTH16_ENDPOINT_ENV: Final = "IPFS_TEST_PROOF_REUSE_GROTH16_ENDPOINT"
PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV: Final = "IPFS_TEST_PROOF_REUSE_GROTH16_CIRCUIT_REF"
PROOF_REUSE_PROVISION_DIR_ENV: Final = "IPFS_TEST_PROOF_REUSE_PROVISION_DIR"
DATASETS_GROTH16_BINARY_ENV: Final = "IPFS_DATASETS_GROTH16_BINARY"
DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: Final = "GROTH16_BACKEND_ARTIFACTS_ROOT"

# Exact merged PTR-144 provider commit (lazy real Groth16 test-pass issuance).
DATASETS_VERIFIER_REVISION: Final = "eb5cc89717d6132d33de912fdf392a31d08ec848"
DATASETS_VERIFIER_SOURCE_SHA256: Final = (
    "da02643318acb108e45cfd918f77e0ea669a9d0480f2550228b7eb0b0653db81"
)
# Exact, closed source/resource manifest used by the private verifier-only
# distribution.  It is the complete reviewed ZKP tree plus the parent package
# initializers/router helper and the three native error/proof/witness schemas.
# Source is read only from the immutable reviewed Git commit; no worktree glob,
# VCS install, submodule initialization, or remote source is allowed.
DATASETS_VERIFIER_SNAPSHOT_FILES: Final = (
    "ipfs_datasets_py/__init__.py",
    "ipfs_datasets_py/logic/__init__.py",
    "ipfs_datasets_py/logic/zkp/ARCHIVE/README.md",
    "ipfs_datasets_py/logic/zkp/README.md",
    "ipfs_datasets_py/logic/zkp/__init__.py",
    "ipfs_datasets_py/logic/zkp/backends/__init__.py",
    "ipfs_datasets_py/logic/zkp/backends/backend_protocol.py",
    "ipfs_datasets_py/logic/zkp/backends/groth16.py",
    "ipfs_datasets_py/logic/zkp/backends/groth16_backup.py",
    "ipfs_datasets_py/logic/zkp/backends/groth16_ffi.py",
    "ipfs_datasets_py/logic/zkp/backends/provekit.py",
    "ipfs_datasets_py/logic/zkp/backends/provekit_ffi.py",
    "ipfs_datasets_py/logic/zkp/backends/simulated.py",
    "ipfs_datasets_py/logic/zkp/canonicalization.py",
    "ipfs_datasets_py/logic/zkp/ceremony.py",
    "ipfs_datasets_py/logic/zkp/circuits.py",
    "ipfs_datasets_py/logic/zkp/eth_contract_artifacts.py",
    "ipfs_datasets_py/logic/zkp/eth_integration.py",
    "ipfs_datasets_py/logic/zkp/eth_vk_registry_payloads.py",
    "ipfs_datasets_py/logic/zkp/evm_harness.py",
    "ipfs_datasets_py/logic/zkp/evm_public_inputs.py",
    "ipfs_datasets_py/logic/zkp/examples/zkp_advanced_demo.py",
    "ipfs_datasets_py/logic/zkp/examples/zkp_basic_demo.py",
    "ipfs_datasets_py/logic/zkp/examples/zkp_ipfs_integration.py",
    "ipfs_datasets_py/logic/zkp/form_circuit.py",
    "ipfs_datasets_py/logic/zkp/legal_theorem_semantics.py",
    "ipfs_datasets_py/logic/zkp/onchain_pipeline.py",
    "ipfs_datasets_py/logic/zkp/provekit/__init__.py",
    "ipfs_datasets_py/logic/zkp/provekit/artifacts.py",
    "ipfs_datasets_py/logic/zkp/provekit/cache.py",
    "ipfs_datasets_py/logic/zkp/provekit/circuits/knowledge_of_axioms/Nargo.toml",
    "ipfs_datasets_py/logic/zkp/provekit/circuits/knowledge_of_axioms/src/main.nr",
    "ipfs_datasets_py/logic/zkp/provekit/circuits/tdfol_v1_trace/Nargo.toml",
    "ipfs_datasets_py/logic/zkp/provekit/circuits/tdfol_v1_trace/src/main.nr",
    "ipfs_datasets_py/logic/zkp/provekit/cli.py",
    "ipfs_datasets_py/logic/zkp/provekit/public_inputs.py",
    "ipfs_datasets_py/logic/zkp/provekit/test_pass_circuit.py",
    "ipfs_datasets_py/logic/zkp/provekit/trace.py",
    "ipfs_datasets_py/logic/zkp/provekit/witness.py",
    "ipfs_datasets_py/logic/zkp/setup_artifacts.py",
    "ipfs_datasets_py/logic/zkp/statement.py",
    "ipfs_datasets_py/logic/zkp/statements/legal_constraint.py",
    "ipfs_datasets_py/logic/zkp/statements/test_pass.py",
    "ipfs_datasets_py/logic/zkp/test_certificate_assurance.py",
    "ipfs_datasets_py/logic/zkp/test_certificate_issuer.py",
    "ipfs_datasets_py/logic/zkp/test_execution_certificate.py",
    "ipfs_datasets_py/logic/zkp/test_pass_groth16_provider.py",
    "ipfs_datasets_py/logic/zkp/tests/test_eth_integration.py",
    "ipfs_datasets_py/logic/zkp/ucan_zkp_bridge.py",
    "ipfs_datasets_py/logic/zkp/vk_registry.py",
    "ipfs_datasets_py/logic/zkp/witness_manager.py",
    "ipfs_datasets_py/logic/zkp/zkp_prover.py",
    "ipfs_datasets_py/logic/zkp/zkp_verifier.py",
    "ipfs_datasets_py/processors/groth16_backend/schemas/error_envelope_v1.schema.json",
    "ipfs_datasets_py/processors/groth16_backend/schemas/proof_v1.schema.json",
    "ipfs_datasets_py/processors/groth16_backend/schemas/witness_v1.schema.json",
    "ipfs_datasets_py/router_deps.py",
)
DATASETS_VERIFIER_SNAPSHOT_SHA256: Final = (
    "789339696dc10fb37dc0fd4fddd21b24af50b669479c194095f37dc904eab343"
)
DATASETS_VERIFIER_SNAPSHOT_BYTES: Final = 873_708
DATASETS_VERIFIER_ZKP_TREE_OBJECT: Final = "33fca9e5756798b7b77e417a6747b996e55d38c1"
DATASETS_VERIFIER_SCHEMA_TREE_OBJECT: Final = "343f2381e601ff4a81dab95c8b32ae0aacec65ac"
DATASETS_VERIFIER_REQUIRES_PYTHON: Final = ">=3.12"
DATASETS_PYTHON_BUILD_FILES_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "pyproject.toml": (
            "75076f609944708ee4e9bdb48d2e2dc48280f50e85fa2a2edb1267cb3d77c8a9"
        ),
        "setup.py": (
            "3d83ecf794e36982e3074afff7a625c7642c7d6d174f74aa65972ba7550c58fb"
        ),
    }
)
DATASETS_VERIFIER_DISTRIBUTION: Final = (
    "ipfs-accelerate-proof-reuse-verifier==0.2.0+eb5cc897"
)
DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED: Final = False
DATASETS_VERIFIER_RELEASE_BLOCKER: Final = "datasets_verifier_revision_unpublished"

# Exact source inputs allowed to reach ``cargo build --locked``.  The checkout
# revision pin alone is insufficient because a Git worktree can contain
# modified files while HEAD still names the reviewed commit.  Native
# provisioning therefore verifies every build input as well as the revision.
DATASETS_GROTH16_REVIEWED_FILES_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Cargo.toml": (
            "f0b4b36d08496c9660945ca69e7fab2af6074ad56668830822d93573555f1231"
        ),
        "Cargo.lock": (
            "592b3736d8e2c25f54aa1c7f5ea8cd1c1649c644762d1973f2687918bf9e470f"
        ),
        "src/circuit.rs": (
            "3d0ab0afd0f09711f4834d155d37dec228ce0d4e5608eb4371e4f4d8026cba04"
        ),
        "src/domain.rs": (
            "fb39f6b0992b2e77053bb9ca64f8d8005cd43af18b07e766816ddfe27e6aeeb2"
        ),
        "src/lib.rs": (
            "72e4c45e123d9367da3e2a2ef7e51c0616ed4e3d0f2fae5ddfcf17760e3112b1"
        ),
        "src/main.rs": (
            "cb25765d0be0fc37cf8a4a5ba8881f7f1eb8ddcb97dfddd75dfbd8a676eb5e34"
        ),
        "src/prover.rs": (
            "a469844271d89b2fd61c7b5eb97f8957a444662b5822197989e71248da9bcc03"
        ),
        "src/setup.rs": (
            "6ddf5412dcafbaaba86f385ed8ceffad3bfcae3e08d3f41ba181a8c22134a31a"
        ),
        "src/verifier.rs": (
            "5c5e4783897ed1f65d4884b4db4dc9f5890f60c97a99a59524fe3691008653b4"
        ),
    }
)
DATASETS_GROTH16_BUNDLED_BINARIES_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "linux-aarch64": (
            "7cd14c97f321c0b4220cfc881c424800f2da288b3056c49c2a6bf7a030bb02dc"
        ),
    }
)
DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "v1/proving_key.bin": (
            "0efdaa9a518082122df987297fcd05b0ce411aa859d42cbe88a1b33d3be8c0a3"
        ),
        "v1/verifying_key.bin": (
            "790157b0ab34735872fc750a60eede767299c6771cd1501afe313eca6a11f67d"
        ),
        "v2/proving_key.bin": (
            "88bb2c916824dd86bc32f3419b622705eaa9724b2af6bc5231fc20ddb1642330"
        ),
        "v2/verifying_key.bin": (
            "3c5d85cf1ac5d305237704e1b26714ee89140fc46b3f87cbd0a9695a5a65d76d"
        ),
    }
)

MULTIFORMATS_MODULE: Final = "multiformats"
JSONSCHEMA_MODULE: Final = "jsonschema"
NLTK_MODULE: Final = "nltk"
DATASETS_VERIFIER_MODULE: Final = (
    "ipfs_datasets_py.logic.zkp.test_execution_certificate"
)
STORE_MODULE: Final = "ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store"
PROVIDER_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.integrations."
    "ipfs_datasets_test_certificate_provider"
)
LOOKUP_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.lookup"


@dataclass(frozen=True, slots=True)
class ProofReuseDependency:
    """One exact optional module and its controlled installation spec."""

    module_name: str
    distribution: str
    required_symbols: tuple[str, ...] = ()
    unavailable_reason: str = "plugin_unavailable"
    pip_options: tuple[str, ...] = ()
    install_environment: tuple[tuple[str, str], ...] = ()
    packaging_source: str = "lazy_only"


MULTIFORMATS_DEPENDENCY: Final = ProofReuseDependency(
    module_name=MULTIFORMATS_MODULE,
    distribution="multiformats>=0.3,<1",
    required_symbols=("CID", "multihash"),
    unavailable_reason="cid_provider_unavailable",
    packaging_source="requirements-proof-reuse.txt",
)
JSONSCHEMA_DEPENDENCY: Final = ProofReuseDependency(
    module_name=JSONSCHEMA_MODULE,
    distribution="jsonschema>=4,<5",
    required_symbols=("validators",),
    unavailable_reason="certificate_provider_unavailable",
    packaging_source="requirements-proof-reuse.txt",
)
NLTK_DEPENDENCY: Final = ProofReuseDependency(
    module_name=NLTK_MODULE,
    distribution="nltk>=3.8.1,<4",
    required_symbols=("data", "download"),
    unavailable_reason="nltk_python_unavailable",
    packaging_source="requirements-proof-reuse.txt",
)
DATASETS_VERIFIER_DEPENDENCY: Final = ProofReuseDependency(
    module_name=DATASETS_VERIFIER_MODULE,
    distribution=DATASETS_VERIFIER_DISTRIBUTION,
    required_symbols=("verify_test_execution_certificate",),
    unavailable_reason="certificate_provider_unavailable",
    # The canonical modules are copied from exact reviewed Git blobs into an
    # atomic owner-private CAS target. No package resolver, setup hook, VCS
    # install, submodule, or global site-packages mutation is involved.
    packaging_source="private_exact_git_blob_snapshot_cas",
)

_SERVICE_DEPENDENCIES: Final = (
    MULTIFORMATS_DEPENDENCY,
    JSONSCHEMA_DEPENDENCY,
    DATASETS_VERIFIER_DEPENDENCY,
)
_DEPENDENCIES: Final = (
    MULTIFORMATS_DEPENDENCY,
    JSONSCHEMA_DEPENDENCY,
    NLTK_DEPENDENCY,
    DATASETS_VERIFIER_DEPENDENCY,
)
PROOF_REUSE_DEPENDENCY_ALLOWLIST: Final[Mapping[str, ProofReuseDependency]] = (
    MappingProxyType(
        {dependency.module_name: dependency for dependency in _DEPENDENCIES}
    )
)

_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"", "0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class NltkDataResource:
    """One bounded NLTK downloader identifier and its lookup paths."""

    package_id: str
    find_paths: tuple[str, ...]


_NLTK_DATA_RESOURCES: Final = (
    NltkDataResource("punkt", ("tokenizers/punkt",)),
    # NLTK >=3.8.2 may split language tables from ``punkt``.
    NltkDataResource("punkt_tab", ("tokenizers/punkt_tab/english",)),
    NltkDataResource(
        "averaged_perceptron_tagger",
        ("taggers/averaged_perceptron_tagger",),
    ),
    NltkDataResource(
        "averaged_perceptron_tagger_eng",
        ("taggers/averaged_perceptron_tagger_eng",),
    ),
    NltkDataResource("maxent_ne_chunker", ("chunkers/maxent_ne_chunker",)),
    NltkDataResource(
        "maxent_ne_chunker_tab",
        ("chunkers/maxent_ne_chunker_tab/english_ace_multiclass",),
    ),
    NltkDataResource("words", ("corpora/words",)),
)
NLTK_DATA_RESOURCE_ALLOWLIST: Final[Mapping[str, NltkDataResource]] = MappingProxyType(
    {resource.package_id: resource for resource in _NLTK_DATA_RESOURCES}
)
DEFAULT_NLTK_DATA_RESOURCES: Final = tuple(
    resource.package_id for resource in _NLTK_DATA_RESOURCES
)


def automatic_install_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return the lazy-install policy without mutating anything.

    Installation is enabled by default, but it remains behind the active-use
    boundary: no resolver or installer is constructed until proof reuse is
    enabled and an exact lookup/publication needs its services.  Operators can
    disable every process/network attempt with
    ``IPFS_TEST_PROOF_REUSE_AUTO_INSTALL=0``.  Invalid values deny permission.
    """

    source = os.environ if environ is None else environ
    if PROOF_REUSE_AUTO_INSTALL_ENV not in source:
        return True
    value = str(source.get(PROOF_REUSE_AUTO_INSTALL_ENV, "")).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    # An invalid policy is never interpreted as permission to install.
    return False


def _explicit_capability_opt_in(
    environment_variable: str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return an explicit, fail-closed capability provisioning choice."""

    source = os.environ if environ is None else environ
    if environment_variable not in source:
        return False
    value = str(source.get(environment_variable, "")).strip().lower()
    return value in _TRUE_VALUES


def nltk_data_download_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """NLTK corpus downloads are off unless explicitly enabled."""

    return _explicit_capability_opt_in(PROOF_REUSE_NLTK_DOWNLOAD_ENV, environ)


def groth16_build_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Native Cargo builds are off unless explicitly enabled."""

    return _explicit_capability_opt_in(PROOF_REUSE_GROTH16_BUILD_ENV, environ)


def proof_reuse_dependency_plan(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe the complete bounded dependency/capability plan.

    This is pure introspection: it does not import a provider, inspect package
    metadata, inspect the filesystem, touch the cache, or start a process.
    Python distributions, NLTK downloader data, a native Cargo build, a remote
    endpoint, and cryptographic keys/circuits remain distinct layers.
    """

    source = os.environ if environ is None else environ
    configured_datasets_source = bool(
        str(source.get(PROOF_REUSE_DATASETS_SOURCE_ENV, "")).strip()
    )
    return {
        "interface": "ProofReuseDependencyPlan@1",
        "lazy": True,
        "cold_import_inert": True,
        "fail_open_to_run": True,
        "automatic_install_enabled": automatic_install_enabled(environ),
        "disable_environment_variable": PROOF_REUSE_AUTO_INSTALL_ENV,
        "datasets_source_environment_variable": (PROOF_REUSE_DATASETS_SOURCE_ENV),
        "datasets_requested_source": (
            "configured_local_path"
            if configured_datasets_source
            else "reviewed_integration_sibling"
        ),
        "datasets_reviewed_revision": DATASETS_VERIFIER_REVISION,
        "datasets_verifier_source_sha256": (DATASETS_VERIFIER_SOURCE_SHA256),
        "datasets_python_build_input_digests": len(DATASETS_PYTHON_BUILD_FILES_SHA256),
        "datasets_install_source": "private_exact_git_blob_snapshot_cas",
        "datasets_vcs_install": False,
        "datasets_submodules_initialized": False,
        "datasets_snapshot_files": len(DATASETS_VERIFIER_SNAPSHOT_FILES),
        "datasets_snapshot_bytes": DATASETS_VERIFIER_SNAPSHOT_BYTES,
        "datasets_snapshot_sha256": DATASETS_VERIFIER_SNAPSHOT_SHA256,
        "datasets_requires_python": DATASETS_VERIFIER_REQUIRES_PYTHON,
        "datasets_private_target_install": True,
        "datasets_global_site_packages_mutated": False,
        "remote_source_published": (DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED),
        "release_blocker": DATASETS_VERIFIER_RELEASE_BLOCKER,
        "python_dependencies": [
            {
                "module_name": dependency.module_name,
                "distribution": dependency.distribution,
                "required_symbols": list(dependency.required_symbols),
                "pip_options": list(dependency.pip_options),
                "packaging_source": dependency.packaging_source,
                "provisioning_kind": "python_distribution",
                "fallback_action": "RUN",
            }
            for dependency in _DEPENDENCIES
        ],
        # Backward-compatible alias retained for the @1 interface.
        "dependencies": [
            {
                "module_name": dependency.module_name,
                "distribution": dependency.distribution,
                "required_symbols": list(dependency.required_symbols),
                "pip_options": list(dependency.pip_options),
                "packaging_source": dependency.packaging_source,
            }
            for dependency in _DEPENDENCIES
        ],
        "nltk_data": {
            "provisioning_kind": "network_data_download",
            "python_module": NLTK_MODULE,
            "resource_allowlist": [
                {
                    "package_id": resource.package_id,
                    "find_paths": list(resource.find_paths),
                }
                for resource in _NLTK_DATA_RESOURCES
            ],
            "default_resources": list(DEFAULT_NLTK_DATA_RESOURCES),
            "explicit_consent_environment_variable": (PROOF_REUSE_NLTK_DOWNLOAD_ENV),
            "download_directory_environment_variable": (PROOF_REUSE_NLTK_DATA_DIR_ENV),
            "download_enabled": nltk_data_download_enabled(environ),
            "download_on_import": False,
            "download_during_package_install": False,
            "fallback_action": "RUN",
        },
        "groth16_native_backend": {
            "provisioning_kind": "cargo_native_build",
            "python_distribution": None,
            "datasets_source_environment_variable": (PROOF_REUSE_DATASETS_SOURCE_ENV),
            "reviewed_datasets_revision": DATASETS_VERIFIER_REVISION,
            "reviewed_build_input_digests": len(DATASETS_GROTH16_REVIEWED_FILES_SHA256),
            "immutable_git_blob_snapshot": True,
            "private_cargo_home": True,
            "inherited_rust_wrappers": False,
            "reviewed_bundled_binary_platforms": sorted(
                DATASETS_GROTH16_BUNDLED_BINARIES_SHA256
            ),
            "cargo_command": [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--manifest-path",
                "<validated-local-datasets-source>/ipfs_datasets_py/"
                "processors/groth16_backend/Cargo.toml",
            ],
            "explicit_consent_environment_variable": (PROOF_REUSE_GROTH16_BUILD_ENV),
            "build_receipt_directory_environment_variable": (
                PROOF_REUSE_PROVISION_DIR_ENV
            ),
            "previous_target_binary_without_receipt_trusted": False,
            "build_enabled": groth16_build_enabled(environ),
            "build_on_import": False,
            "build_during_package_install": False,
            "trusted_setup_during_build": False,
            "fallback_action": "DEFERRED",
        },
        "groth16_runtime_inputs": {
            "endpoint": {
                "provisioning_kind": "operator_configuration",
                "environment_variable": PROOF_REUSE_GROTH16_ENDPOINT_ENV,
                "installable": False,
            },
            "binary": {
                "provisioning_kind": "native_executable",
                "environment_variable": DATASETS_GROTH16_BINARY_ENV,
                "installable_by": "groth16_native_backend",
            },
            "keys": {
                "provisioning_kind": "cryptographic_artifacts",
                "environment_variable": (DATASETS_GROTH16_ARTIFACTS_ROOT_ENV),
                "reviewed_bundled_artifacts": sorted(
                    DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256
                ),
                "auto_generate": False,
            },
            "circuit": {
                "provisioning_kind": "versioned_circuit_binding",
                "environment_variable": (PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV),
                "auto_select": False,
            },
            "fallback_action": "DEFERRED",
        },
        "external_capabilities": [
            "groth16_endpoint",
            "groth16_native_binary",
            "groth16_verifying_key",
            "groth16_circuit_binding",
            "provekit_binary_and_artifacts",
            "shared_cache_or_local_cache_directory",
        ],
        "external_capability_absence_action": "RUN_OR_DEFERRED",
        "runtime_activation": {
            "automatic_plugin_discovery": True,
            "ordinary_enabled_run_effective_action": "run",
            "default_identity_services_injected": False,
            "default_identity_service_factory_configured": False,
            "production_identity_injector_configured": False,
            "required_identity_providers": [
                "repository_forest_provider",
                "analysis_index_provider",
                "component_inputs_provider",
                "policy_inputs_provider",
                "runtime_evidence_provider",
            ],
            "default_identity_compiler_available": True,
            "candidate_context_store_configured": False,
            "two_stage_candidate_revalidation_configured": False,
            "lookup_requires_exact_execution_key_before_candidate_read": True,
            "runtime_trace_attribute_producer_configured": False,
            "post_pass_runtime_trace_capture_configured": False,
            "post_pass_receipt_requires_runtime_trace": False,
            "deferred_request_builder_configured": False,
            "deferred_request_transport_compatible": False,
            "deferred_certificate_issuer_configured": False,
            "issuer_in_lazy_service_bundle": False,
            "issuer_in_lazy_service_resolution": False,
            "candidate_certificate_publication_configured": False,
            "authoritative_candidate_publication_configured": False,
            "receipt_content_identity_profiles_conformant": False,
            "receipt_content_identity_gap": (
                "accelerator_cidv1_dag_json_vs_datasets_sha256"
            ),
            "receipt_content_identity_profiles": {
                "accelerator": "cidv1-base32-dag-json-sha2-256",
                "datasets_statement": "sha256-canonical-json-v1",
                "exact_conformance": False,
            },
            "ordinary_warm_skip_path_complete": False,
            "missing_provider_action": "run",
            "completion_authority": False,
            "activation_blocker_codes": [
                "identity_services_unconfigured",
                "candidate_lookup_identity_cycle",
                "post_pass_runtime_trace_unproduced",
                "runtime_trace_not_required_for_receipt",
                "receipt_cid_profile_mismatch",
                "deferred_request_builder_unconfigured",
                "deferred_request_transport_type_mismatch",
                "issuer_unconfigured",
                "authoritative_candidate_not_published",
            ],
            "required_implementation_sequence": [
                {
                    "goals": ["PTR-G020", "PTR-G030", "PTR-G060"],
                    "work": "production_current_identity_provider_factory",
                },
                {
                    "goals": ["PTR-G030", "PTR-G060"],
                    "work": "controlled_current_runtime_preflight_provider",
                },
                {
                    "goals": ["PTR-G010", "PTR-G040", "PTR-G050"],
                    "work": "cross_package_receipt_cid_profile_conformance",
                },
                {
                    "goals": ["PTR-G040", "PTR-G050", "PTR-G060"],
                    "work": "deferred_request_issuer_and_candidate_publication",
                },
                {
                    "goals": [
                        "PTR-G060",
                        "PTR-G080",
                        "PTR-G090",
                        "PTR-G100",
                    ],
                    "work": "unwired_cross_repository_cold_warm_e2e",
                },
                {
                    "goals": ["PTR-G110"],
                    "work": "activated_warm_benchmark_and_rollout_evidence",
                },
            ],
        },
    }


@dataclass(frozen=True, slots=True)
class ProofReuseServiceResolution:
    """All-or-nothing result of one lazy service assembly attempt."""

    available: bool
    reason_code: str
    lookup: Any = None
    store: Any = None
    provider: Any = None
    installed_modules: tuple[str, ...] = ()

    @classmethod
    def unavailable(cls, reason_code: str) -> ProofReuseServiceResolution:
        return cls(available=False, reason_code=reason_code)


_DATASETS_VERIFIER_RECEIPT_NAME: Final = ".proof-reuse-verifier-snapshot.json"
_DATASETS_VERIFIER_RECEIPT_INTERFACE: Final = "ProofReuseVerifierSnapshot@1"
_DATASETS_AUTHORITY_PREFIX: Final = "ipfs_datasets_py.logic.zkp"
_DATASETS_AUTHORITY_NAMESPACE_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ipfs_datasets_py.logic.zkp.statements": (
            "ipfs_datasets_py/logic/zkp/statements"
        ),
    }
)


def _datasets_snapshot_digest(
    payloads: Mapping[str, bytes],
) -> tuple[str, int] | None:
    """Hash the closed snapshot manifest with path, size, and blob digest."""

    if tuple(sorted(payloads)) != DATASETS_VERIFIER_SNAPSHOT_FILES:
        return None
    aggregate = hashlib.sha256()
    total_bytes = 0
    for relative in DATASETS_VERIFIER_SNAPSHOT_FILES:
        payload = payloads.get(relative)
        if not isinstance(payload, bytes) or len(payload) > 4 * 1024 * 1024:
            return None
        total_bytes += len(payload)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(len(payload)).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(hashlib.sha256(payload).digest())
    return aggregate.hexdigest(), total_bytes


def _datasets_snapshot_payloads(root: Path) -> Mapping[str, bytes] | None:
    """Read only regular, non-symlink manifest files contained by *root*."""

    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    payloads: dict[str, bytes] = {}
    for relative in DATASETS_VERIFIER_SNAPSHOT_FILES:
        path = root / relative
        try:
            if path.is_symlink() or not path.is_file():
                return None
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
            if resolved != path.absolute():
                return None
            payload = path.read_bytes()
        except (OSError, RuntimeError, ValueError):
            return None
        payloads[relative] = payload
    return MappingProxyType(payloads)


def _datasets_reviewed_namespace_is_closed(root: Path) -> bool:
    """Reject unreviewed source/resources in the authority-bearing subtrees."""

    prefixes = (
        Path("ipfs_datasets_py/logic/zkp"),
        Path("ipfs_datasets_py/processors/groth16_backend/schemas"),
    )
    expected_files = {
        relative
        for relative in DATASETS_VERIFIER_SNAPSHOT_FILES
        if any(
            relative == prefix.as_posix()
            or relative.startswith(f"{prefix.as_posix()}/")
            for prefix in prefixes
        )
    }
    expected_entries = set(expected_files)
    for relative in expected_files:
        parent = Path(relative).parent
        while any(parent == prefix or prefix in parent.parents for prefix in prefixes):
            expected_entries.add(parent.as_posix())
            parent = parent.parent
    actual_entries = {prefix.as_posix() for prefix in prefixes}
    try:
        for prefix in prefixes:
            subtree = root / prefix
            for path in subtree.rglob("*"):
                relative = path.relative_to(root).as_posix()
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                if path.is_symlink():
                    return False
                actual_entries.add(relative)
    except (OSError, ValueError):
        return False
    return actual_entries == expected_entries


def _datasets_authority_module_relative_path(module_name: str) -> str | None:
    """Map one canonical ZKP module name to its exact reviewed source file."""

    if module_name == _DATASETS_AUTHORITY_PREFIX:
        candidates = ("ipfs_datasets_py/logic/zkp/__init__.py",)
    elif module_name.startswith(f"{_DATASETS_AUTHORITY_PREFIX}."):
        suffix = module_name[len(_DATASETS_AUTHORITY_PREFIX) + 1 :].replace(".", "/")
        candidates = (
            f"ipfs_datasets_py/logic/zkp/{suffix}.py",
            f"ipfs_datasets_py/logic/zkp/{suffix}/__init__.py",
        )
    else:
        return None
    matches = tuple(
        candidate
        for candidate in candidates
        if candidate in DATASETS_VERIFIER_SNAPSHOT_FILES
    )
    return matches[0] if len(matches) == 1 else None


def _datasets_authority_modules_match_root(root: Path) -> bool:
    """Attest every loaded canonical ZKP module to one closed reviewed root."""

    try:
        selected_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    for name, loaded in tuple(sys.modules.items()):
        if name != _DATASETS_AUTHORITY_PREFIX and not name.startswith(
            f"{_DATASETS_AUTHORITY_PREFIX}."
        ):
            continue
        if loaded is None:
            return False
        loaded_file = str(getattr(loaded, "__file__", "") or "")
        if loaded_file:
            relative = _datasets_authority_module_relative_path(name)
            if relative is None:
                return False
            try:
                loaded_path = Path(loaded_file).resolve(strict=True)
                expected_path = (selected_root / relative).resolve(strict=True)
            except (OSError, RuntimeError):
                return False
            if loaded_path != expected_path:
                return False
            continue

        namespace_relative = _DATASETS_AUTHORITY_NAMESPACE_PATHS.get(name)
        if namespace_relative is None:
            return False
        namespace_path = getattr(loaded, "__path__", None)
        spec = getattr(loaded, "__spec__", None)
        spec_path = getattr(spec, "submodule_search_locations", None)
        if namespace_path is None or spec_path is None:
            return False
        try:
            paths = tuple(
                Path(str(path)).resolve(strict=True) for path in namespace_path
            )
            spec_paths = tuple(
                Path(str(path)).resolve(strict=True) for path in spec_path
            )
            expected_namespace = (selected_root / namespace_relative).resolve(
                strict=True
            )
        except (OSError, RuntimeError):
            return False
        if paths != (expected_namespace,) or spec_paths != (expected_namespace,):
            return False
    return True


class AllowlistedPipInstaller:
    """Bounded pip installer for the closed proof-reuse dependency set.

    Nothing invokes this class unless proof reuse is enabled, an exact identity
    requires services, and automatic installation has not been disabled.
    Tests and embedding applications can inject a different installer instead.
    Each dependency receives at most one process attempt per installer instance.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        timeout_seconds: float = 120.0,
        environ: Mapping[str, str] | None = None,
        provision_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 1 <= float(timeout_seconds) <= 600
        ):
            raise ValueError("timeout_seconds must be between 1 and 600")
        self._runner = runner or subprocess.run
        self._timeout_seconds = float(timeout_seconds)
        self._environ = dict(os.environ if environ is None else environ)
        configured_provision_root = str(
            self._environ.get(PROOF_REUSE_PROVISION_DIR_ENV, "")
        ).strip()
        configured_cache_root = str(
            self._environ.get(PROOF_REUSE_CACHE_DIR_ENV, "")
        ).strip()
        if provision_root is not None:
            base_provision_root = Path(provision_root).expanduser()
        elif configured_provision_root:
            base_provision_root = Path(configured_provision_root).expanduser()
        elif configured_cache_root:
            base_provision_root = (
                Path(configured_cache_root).expanduser() / "provisioning"
            )
        else:
            home = str(self._environ.get("HOME", "")).strip()
            home_path = Path(home).expanduser() if home else Path.home()
            base_provision_root = (
                home_path
                / ".cache"
                / "ipfs_accelerate_py"
                / "proof_reuse"
                / "provisioning"
            )
        self._python_snapshot_root = base_provision_root / "python-snapshots"
        self._outcomes: dict[str, bool] = {}
        self._lock = threading.Lock()

    def _selected_distribution(
        self,
        dependency: ProofReuseDependency,
    ) -> str | None:
        """Return the registry spec, or the private snapshot distribution ID."""

        return dependency.distribution

    def _requested_local_datasets_source(self) -> Path | None:
        """Resolve only the configured path or the integration sibling."""

        configured = str(self._environ.get(PROOF_REUSE_DATASETS_SOURCE_ENV, "")).strip()
        if configured:
            return Path(configured)
        try:
            return Path(__file__).resolve().parents[3].parent / "ipfs_datasets"
        except (IndexError, OSError, RuntimeError):
            return None

    def validated_local_datasets_source(self) -> Path | None:
        """Return a repository containing the exact reviewed immutable object."""

        requested = self._requested_local_datasets_source()
        if requested is None:
            return None
        return self._validated_local_datasets_source(
            requested,
            require_reviewed_revision=True,
        )

    def reviewed_datasets_blobs(
        self, requested: Mapping[str, str]
    ) -> Mapping[str, bytes] | None:
        """Read an allowlisted set of exact-revision blobs, digest checked."""

        backend_prefix = "ipfs_datasets_py/processors/groth16_backend/"
        allowlist = {
            "ipfs_datasets_py/logic/zkp/test_execution_certificate.py": (
                DATASETS_VERIFIER_SOURCE_SHA256
            ),
            **DATASETS_PYTHON_BUILD_FILES_SHA256,
            **{
                f"{backend_prefix}{relative}": digest
                for relative, digest in (DATASETS_GROTH16_REVIEWED_FILES_SHA256.items())
            },
        }
        normalized = dict(requested)
        if not normalized or any(
            allowlist.get(relative) != digest for relative, digest in normalized.items()
        ):
            return None
        source = self.validated_local_datasets_source()
        if source is None:
            return None
        blobs: dict[str, bytes] = {}
        for relative, expected_digest in normalized.items():
            blob = self._git_object_output(
                source,
                "show",
                f"{DATASETS_VERIFIER_REVISION}:{relative}",
            )
            if blob is None or hashlib.sha256(blob).hexdigest() != expected_digest:
                return None
            blobs[relative] = blob
        return MappingProxyType(blobs)

    def reviewed_datasets_verifier_snapshot_blobs(
        self,
    ) -> Mapping[str, bytes] | None:
        """Read and SHA-256 seal the closed verifier snapshot from Git objects."""

        source = self.validated_local_datasets_source()
        if source is None:
            return None
        for relative, expected_tree in (
            ("ipfs_datasets_py/logic/zkp", DATASETS_VERIFIER_ZKP_TREE_OBJECT),
            (
                "ipfs_datasets_py/processors/groth16_backend/schemas",
                DATASETS_VERIFIER_SCHEMA_TREE_OBJECT,
            ),
        ):
            tree = self._git_object_output(
                source,
                "rev-parse",
                f"{DATASETS_VERIFIER_REVISION}:{relative}",
            )
            if tree is None or tree.decode("ascii", "strict").strip() != expected_tree:
                return None
        blobs: dict[str, bytes] = {}
        for relative in DATASETS_VERIFIER_SNAPSHOT_FILES:
            blob = self._git_object_output(
                source,
                "show",
                f"{DATASETS_VERIFIER_REVISION}:{relative}",
            )
            if blob is None:
                return None
            blobs[relative] = blob
        digest = _datasets_snapshot_digest(blobs)
        if digest != (
            DATASETS_VERIFIER_SNAPSHOT_SHA256,
            DATASETS_VERIFIER_SNAPSHOT_BYTES,
        ):
            return None
        return MappingProxyType(blobs)

    def _datasets_snapshot_target(self) -> Path:
        return (
            self._python_snapshot_root
            / f"datasets-verifier-{DATASETS_VERIFIER_SNAPSHOT_SHA256}"
        )

    @staticmethod
    def _private_directory(path: Path, *, create: bool) -> Path | None:
        """Resolve an owner-private directory without following path aliases."""

        try:
            absolute = path.absolute()
            existed = path.exists()
            if create:
                path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not path.is_dir() or path.is_symlink():
                return None
            resolved = path.resolve(strict=True)
            if resolved != absolute:
                return None
            metadata = os.lstat(path)
            if os.name != "nt":
                getuid = getattr(os, "getuid", None)
                if callable(getuid) and metadata.st_uid != getuid():
                    return None
                if metadata.st_mode & 0o077:
                    if existed:
                        return None
                    path.chmod(0o700)
                    metadata = os.lstat(path)
                    if metadata.st_mode & 0o077:
                        return None
            return resolved
        except OSError:
            return None

    @staticmethod
    def _snapshot_receipt() -> dict[str, Any]:
        return {
            "interface": _DATASETS_VERIFIER_RECEIPT_INTERFACE,
            "datasets_revision": DATASETS_VERIFIER_REVISION,
            "distribution": DATASETS_VERIFIER_DISTRIBUTION,
            "requires_python": DATASETS_VERIFIER_REQUIRES_PYTHON,
            "snapshot_sha256": DATASETS_VERIFIER_SNAPSHOT_SHA256,
            "snapshot_bytes": DATASETS_VERIFIER_SNAPSHOT_BYTES,
            "snapshot_files": len(DATASETS_VERIFIER_SNAPSHOT_FILES),
            "vcs_install": False,
            "pip_install": False,
            "stdlib_materialization": True,
            "submodules_initialized": False,
            "remote_source_allowed": False,
        }

    @staticmethod
    def _snapshot_tree_is_private(root: Path) -> bool:
        try:
            root_resolved = root.resolve(strict=True)
            for entry in (root, *root.rglob("*")):
                if entry.is_symlink():
                    return False
                resolved = entry.resolve(strict=True)
                resolved.relative_to(root_resolved)
                metadata = os.lstat(entry)
                if os.name != "nt":
                    getuid = getattr(os, "getuid", None)
                    if callable(getuid) and metadata.st_uid != getuid():
                        return False
                    if metadata.st_mode & 0o077:
                        return False
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def _validated_private_snapshot_target(self) -> Path | None:
        root = self._datasets_snapshot_target()
        if self._private_directory(root, create=False) is None:
            return None
        receipt_path = root / _DATASETS_VERIFIER_RECEIPT_NAME
        try:
            if receipt_path.is_symlink() or receipt_path.stat().st_size > 16_384:
                return None
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        if receipt != self._snapshot_receipt():
            return None
        payloads = _datasets_snapshot_payloads(root)
        if payloads is None or _datasets_snapshot_digest(payloads) != (
            DATASETS_VERIFIER_SNAPSHOT_SHA256,
            DATASETS_VERIFIER_SNAPSHOT_BYTES,
        ):
            return None
        package_root = root / "ipfs_datasets_py"
        try:
            actual_entries = {
                path.relative_to(root).as_posix() for path in root.rglob("*")
            }
        except OSError:
            return None
        expected_entries = {
            _DATASETS_VERIFIER_RECEIPT_NAME,
            *DATASETS_VERIFIER_SNAPSHOT_FILES,
        }
        for relative in DATASETS_VERIFIER_SNAPSHOT_FILES:
            expected_entries.update(
                parent.as_posix()
                for parent in Path(relative).parents
                if parent != Path(".")
            )
        if actual_entries != expected_entries or not package_root.is_dir():
            return None
        if not self._snapshot_tree_is_private(root):
            return None
        return root.resolve(strict=True)

    @staticmethod
    def _harden_snapshot_tree(root: Path) -> bool:
        try:
            entries = sorted(
                root.rglob("*"), key=lambda path: len(path.parts), reverse=True
            )
            if os.name != "nt":
                for entry in entries:
                    if entry.is_symlink():
                        return False
                    entry.chmod(0o500 if entry.is_dir() else 0o400)
                root.chmod(0o500)
            return True
        except OSError:
            return False

    def _activate_private_snapshot(self, root: Path) -> bool:
        """Overlay reviewed ZKP code without executing an unreviewed parent."""

        validated = self._validated_private_snapshot_target()
        if validated != root.resolve(strict=False):
            return False
        reviewed_source: Path | None = None
        reviewed_source_clean: bool | None = None

        def clean_reviewed_source() -> tuple[Path | None, bool]:
            nonlocal reviewed_source, reviewed_source_clean
            if reviewed_source_clean is not None:
                return reviewed_source, reviewed_source_clean
            reviewed_source = self.validated_local_datasets_source()
            source_payloads = (
                _datasets_snapshot_payloads(reviewed_source)
                if reviewed_source is not None
                else None
            )
            reviewed_source_clean = bool(
                source_payloads is not None
                and _datasets_snapshot_digest(source_payloads)
                == (
                    DATASETS_VERIFIER_SNAPSHOT_SHA256,
                    DATASETS_VERIFIER_SNAPSHOT_BYTES,
                )
                and reviewed_source is not None
                and _datasets_reviewed_namespace_is_closed(reviewed_source)
            )
            return reviewed_source, reviewed_source_clean

        def loaded_path_is_reviewed(loaded: Any, relative: str) -> bool:
            loaded_file = str(getattr(loaded, "__file__", "") or "")
            if not loaded_file:
                return bool(
                    getattr(loaded, "_proof_reuse_snapshot_parent", False)
                    and getattr(loaded, "_proof_reuse_snapshot_parent_root", "")
                    == str(root)
                )
            try:
                path = Path(loaded_file).resolve(strict=True)
                if path == (root / relative).resolve(strict=True):
                    return True
                source, source_is_clean = clean_reviewed_source()
                return bool(
                    source_is_clean
                    and source is not None
                    and path == (source / relative).resolve(strict=True)
                )
            except (OSError, RuntimeError):
                return False

        loaded_authority = any(
            name == _DATASETS_AUTHORITY_PREFIX
            or name.startswith(f"{_DATASETS_AUTHORITY_PREFIX}.")
            for name in sys.modules
        )
        if loaded_authority and not _datasets_authority_modules_match_root(root):
            source, source_is_clean = clean_reviewed_source()
            if (
                not source_is_clean
                or source is None
                or not _datasets_authority_modules_match_root(source)
            ):
                return False

        package_name = "ipfs_datasets_py"
        package_path = root / package_name
        package = sys.modules.get(package_name)
        discovered_package_paths: list[str] = []
        if package is None:
            try:
                discovered = importlib.machinery.PathFinder.find_spec(
                    package_name, list(sys.path)
                )
                if discovered is not None and discovered.submodule_search_locations:
                    discovered_package_paths.extend(
                        str(location)
                        for location in discovered.submodule_search_locations
                        if str(location) != str(package_path)
                    )
            except Exception:
                discovered_package_paths = []
            loaded_router = sys.modules.get(f"{package_name}.router_deps")
            if loaded_router is not None and not loaded_path_is_reviewed(
                loaded_router, "ipfs_datasets_py/router_deps.py"
            ):
                return False
            package_paths = [str(package_path), *discovered_package_paths]
            package_initializer = package_path / "__init__.py"
            package_spec = importlib.util.spec_from_file_location(
                package_name,
                package_initializer,
                submodule_search_locations=package_paths,
            )
            if package_spec is None or package_spec.loader is None:
                return False
            package = importlib.util.module_from_spec(package_spec)
            sys.modules[package_name] = package
            minimal_imports_name = "IPFS_DATASETS_PY_MINIMAL_IMPORTS"
            previous_minimal_imports = os.environ.get(minimal_imports_name)
            os.environ[minimal_imports_name] = "1"
            try:
                package_spec.loader.exec_module(package)
            except Exception:
                if sys.modules.get(package_name) is package:
                    sys.modules.pop(package_name, None)
                loaded_router = sys.modules.get(f"{package_name}.router_deps")
                if loaded_router is not None and loaded_path_is_reviewed(
                    loaded_router, "ipfs_datasets_py/router_deps.py"
                ):
                    sys.modules.pop(f"{package_name}.router_deps", None)
                return False
            finally:
                if previous_minimal_imports is None:
                    os.environ.pop(minimal_imports_name, None)
                else:
                    os.environ[minimal_imports_name] = previous_minimal_imports
        elif not loaded_path_is_reviewed(package, "ipfs_datasets_py/__init__.py"):
            return False
        else:
            search_path = getattr(package, "__path__", None)
            if search_path is None:
                return False
            if str(package_path) not in search_path:
                search_path.insert(0, str(package_path))

        logic_name = "ipfs_datasets_py.logic"
        logic_path = root / "ipfs_datasets_py/logic"
        loaded_logic = sys.modules.get(logic_name)
        if loaded_logic is None:
            root_search = tuple(getattr(package, "__path__", ()))
            extra_logic_paths = []
            for location in root_search:
                candidate = Path(location) / "logic"
                if candidate.is_dir() and str(candidate) != str(logic_path):
                    extra_logic_paths.append(str(candidate))
            loaded_logic = ModuleType(logic_name)
            loaded_logic.__package__ = logic_name
            loaded_logic.__path__ = [str(logic_path), *extra_logic_paths]
            logic_spec = importlib.machinery.ModuleSpec(
                logic_name, loader=None, is_package=True
            )
            logic_spec.submodule_search_locations = loaded_logic.__path__
            loaded_logic.__spec__ = logic_spec
            loaded_logic._proof_reuse_snapshot_parent = True
            loaded_logic._proof_reuse_snapshot_parent_root = str(root)
            sys.modules[logic_name] = loaded_logic
            package.logic = loaded_logic
        elif not loaded_path_is_reviewed(
            loaded_logic, "ipfs_datasets_py/logic/__init__.py"
        ):
            return False
        else:
            search_path = getattr(loaded_logic, "__path__", None)
            if search_path is None:
                return False
            if str(logic_path) in search_path:
                search_path.remove(str(logic_path))
            search_path.insert(0, str(logic_path))

        root_text = str(root)
        if root_text not in sys.path:
            sys.path.append(root_text)
        try:
            importlib.invalidate_caches()
        except Exception:
            return False
        return True

    def activate_cached_datasets_verifier(self) -> bool:
        if sys.version_info < (3, 12):
            return False
        target = self._validated_private_snapshot_target()
        return bool(target is not None and self._activate_private_snapshot(target))

    def _install_datasets_verifier_snapshot(
        self, dependency: ProofReuseDependency
    ) -> bool:
        del dependency
        if sys.version_info < (3, 12):
            return False
        if self.activate_cached_datasets_verifier():
            return True
        target = self._datasets_snapshot_target()
        if target.exists():
            # A corrupt content-addressed target is never replaced in place.
            return False
        parent = self._private_directory(self._python_snapshot_root, create=True)
        if parent is None:
            return False
        blobs = self.reviewed_datasets_verifier_snapshot_blobs()
        if blobs is None:
            return False
        try:
            temporary = tempfile.TemporaryDirectory(
                prefix=".verifier-install-",
                dir=parent,
                ignore_cleanup_errors=True,
            )
        except OSError:
            return False
        staging = Path(temporary.name) / "target"
        try:
            staging.mkdir(mode=0o700)
            for relative, payload in blobs.items():
                destination = staging / relative
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                destination.write_bytes(payload)
            payloads = _datasets_snapshot_payloads(staging)
            if payloads is None or _datasets_snapshot_digest(payloads) != (
                DATASETS_VERIFIER_SNAPSHOT_SHA256,
                DATASETS_VERIFIER_SNAPSHOT_BYTES,
            ):
                return False
            receipt = staging / _DATASETS_VERIFIER_RECEIPT_NAME
            receipt.write_text(
                json.dumps(
                    self._snapshot_receipt(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            installed_target = False
            try:
                os.replace(staging, target)
                installed_target = True
            except OSError:
                if not target.exists():
                    return False
            if installed_target and not self._harden_snapshot_tree(target):
                return False
            return self.activate_cached_datasets_verifier()
        except Exception:
            return False
        finally:
            temporary.cleanup()

    def validate_module_provenance(
        self, dependency: ProofReuseDependency, module: Any
    ) -> bool:
        """Attest every loaded authority module to the CAS or reviewed tree."""

        if dependency.module_name != DATASETS_VERIFIER_MODULE:
            return True
        module_file = str(getattr(module, "__file__", "") or "")
        if not module_file:
            return False
        expected_relative = Path(
            "ipfs_datasets_py/logic/zkp/test_execution_certificate.py"
        )
        try:
            loaded_module_path = Path(module_file).resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        selected_root: Path | None = None
        private = self._validated_private_snapshot_target()
        if private is not None:
            try:
                if loaded_module_path == (private / expected_relative).resolve(
                    strict=True
                ):
                    selected_root = private.resolve(strict=True)
            except (OSError, RuntimeError):
                return False
        if selected_root is None:
            reviewed_source = self.validated_local_datasets_source()
            if reviewed_source is not None:
                payloads = _datasets_snapshot_payloads(reviewed_source)
                reviewed_source_is_clean = bool(
                    payloads is not None
                    and _datasets_snapshot_digest(payloads)
                    == (
                        DATASETS_VERIFIER_SNAPSHOT_SHA256,
                        DATASETS_VERIFIER_SNAPSHOT_BYTES,
                    )
                    and _datasets_reviewed_namespace_is_closed(reviewed_source)
                )
                try:
                    source_module = (reviewed_source / expected_relative).resolve(
                        strict=True
                    )
                except (OSError, RuntimeError):
                    source_module = None
                if reviewed_source_is_clean and loaded_module_path == source_module:
                    selected_root = reviewed_source.resolve(strict=True)
        if selected_root is None:
            return False
        return _datasets_authority_modules_match_root(selected_root)

    @staticmethod
    def _git_object_output(source: Path, *arguments: str) -> bytes | None:
        """Read one immutable Git fact with configs/replacements disabled."""

        git = shutil.which("git")
        if not git:
            return None
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
        command = (
            str(Path(git).resolve()),
            "--no-replace-objects",
            "-c",
            f"core.hooksPath={os.devnull}",
            "-C",
            str(source),
            *arguments,
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=30,
                env=environment,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        output = bytes(completed.stdout)
        return output if len(output) <= 4 * 1024 * 1024 else None

    @staticmethod
    def _detached_git_head(source: Path) -> str | None:
        """Read a reviewed checkout HEAD without starting a git process.

        Accepts detached HEADs and worktree/branch refs (``ref: refs/...``)
        resolved through the git directory and any linked ``commondir``.
        Symbolic-ref cycles and missing objects return ``None`` so the
        installer fails closed to RUN.
        """

        def _is_commit_id(value: str) -> bool:
            return len(value) == 40 and all(
                character in "0123456789abcdef" for character in value
            )

        def _search_roots(git_dir: Path) -> tuple[Path, ...]:
            roots = [git_dir]
            try:
                common_pointer = git_dir / "commondir"
                if common_pointer.is_file():
                    common = Path(common_pointer.read_text(encoding="utf-8").strip())
                    if not common.is_absolute():
                        common = (git_dir / common).resolve(strict=True)
                    else:
                        common = common.resolve(strict=True)
                    if common not in roots:
                        roots.append(common)
            except (OSError, RuntimeError, UnicodeError):
                pass
            return tuple(roots)

        def _read_ref(git_dir: Path, ref_name: str, *, depth: int = 0) -> str | None:
            if depth > 8 or not ref_name.startswith("refs/"):
                return None
            for root in _search_roots(git_dir):
                ref_path = root / ref_name
                try:
                    if ref_path.is_file():
                        payload = ref_path.read_text(encoding="ascii").strip()
                        if payload.lower().startswith("ref:"):
                            return _read_ref(
                                git_dir,
                                payload.split(":", 1)[1].strip(),
                                depth=depth + 1,
                            )
                        if _is_commit_id(payload):
                            return payload
                        continue
                    packed = root / "packed-refs"
                    if not packed.is_file():
                        continue
                    for line in packed.read_text(encoding="ascii").splitlines():
                        text = line.strip()
                        if not text or text.startswith("#") or text.startswith("^"):
                            continue
                        parts = text.split()
                        if len(parts) != 2:
                            continue
                        commit_id, name = parts
                        if name == ref_name and _is_commit_id(commit_id):
                            return commit_id
                except (OSError, RuntimeError, UnicodeError):
                    continue
            return None

        dot_git = source / ".git"
        try:
            if dot_git.is_file():
                pointer = dot_git.read_text(encoding="utf-8").strip()
                prefix = "gitdir:"
                if not pointer.lower().startswith(prefix):
                    return None
                git_dir = Path(pointer[len(prefix) :].strip())
                if not git_dir.is_absolute():
                    git_dir = (source / git_dir).resolve(strict=True)
            elif dot_git.is_dir():
                git_dir = dot_git
            else:
                return None
            head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        except (OSError, RuntimeError, UnicodeError):
            return None
        if _is_commit_id(head):
            return head
        if head.lower().startswith("ref:"):
            return _read_ref(git_dir, head.split(":", 1)[1].strip())
        return None

    def _validated_local_datasets_source(
        self,
        configured: str | os.PathLike[str],
        *,
        require_reviewed_revision: bool,
    ) -> Path | None:
        try:
            source = Path(configured).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not source.is_dir() or source.is_symlink():
            return None
        if not (source / ".git").exists():
            return None
        top_level = self._git_object_output(source, "rev-parse", "--show-toplevel")
        if top_level is None:
            return None
        try:
            reported_root = Path(top_level.decode("utf-8").strip()).resolve(strict=True)
        except (OSError, RuntimeError, UnicodeError):
            return None
        if reported_root != source:
            return None
        if require_reviewed_revision:
            commit = self._git_object_output(
                source,
                "rev-parse",
                f"{DATASETS_VERIFIER_REVISION}^{{commit}}",
            )
            head = self._git_object_output(source, "rev-parse", "HEAD^{commit}")
            if commit is None or head is None:
                return None
            if commit.decode("ascii").strip() != DATASETS_VERIFIER_REVISION:
                return None
            if head.decode("ascii").strip() != DATASETS_VERIFIER_REVISION:
                return None
        reviewed_blobs = {
            "ipfs_datasets_py/logic/zkp/test_execution_certificate.py": (
                DATASETS_VERIFIER_SOURCE_SHA256
            ),
            **DATASETS_PYTHON_BUILD_FILES_SHA256,
        }
        for relative, expected_digest in reviewed_blobs.items():
            blob = self._git_object_output(
                source,
                "show",
                f"{DATASETS_VERIFIER_REVISION}:{relative}",
            )
            if blob is None or hashlib.sha256(blob).hexdigest() != expected_digest:
                return None
        return source

    def _validated_local_datasets_distribution(
        self,
        configured: str | os.PathLike[str],
        *,
        require_reviewed_revision: bool,
    ) -> str | None:
        """Return the private snapshot distribution ID after source validation."""

        source = self._validated_local_datasets_source(
            configured,
            require_reviewed_revision=require_reviewed_revision,
        )
        if source is None:
            return None
        return DATASETS_VERIFIER_DISTRIBUTION

    def prepare_dependency(
        self,
        dependency: ProofReuseDependency,
        *,
        allow_install: bool,
    ) -> bool:
        """Activate a validated CAS snapshot before the first verifier import."""

        if dependency.module_name != DATASETS_VERIFIER_MODULE:
            return True
        if self.activate_cached_datasets_verifier():
            return True
        return bool(allow_install and self.install(dependency))

    def install(self, dependency: ProofReuseDependency) -> bool:
        allowed = PROOF_REUSE_DEPENDENCY_ALLOWLIST.get(
            getattr(dependency, "module_name", "")
        )
        if allowed != dependency:
            return False
        with self._lock:
            previous = self._outcomes.get(dependency.module_name)
            if previous is not None:
                return previous
            if dependency.module_name == DATASETS_VERIFIER_MODULE:
                succeeded = self._install_datasets_verifier_snapshot(dependency)
                self._outcomes[dependency.module_name] = succeeded
                return succeeded
            distribution = self._selected_distribution(dependency)
            if distribution is None:
                self._outcomes[dependency.module_name] = False
                return False
            command = (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                *dependency.pip_options,
                distribution,
            )
            run_environment = dict(self._environ)
            run_environment.update(dict(dependency.install_environment))
            try:
                completed = self._runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    env=run_environment,
                )
                succeeded = getattr(completed, "returncode", 1) == 0
            except Exception:
                succeeded = False
            self._outcomes[dependency.module_name] = succeeded
            return succeeded


def _is_requested_module_absence(
    exc: ModuleNotFoundError,
    requested: str,
) -> bool:
    missing = str(getattr(exc, "name", "") or "")
    return bool(
        missing and (missing == requested or requested.startswith(f"{missing}."))
    )


def _installer_callable(installer: Any) -> Callable[[Any], Any] | None:
    install = getattr(installer, "install", None)
    if callable(install):
        return install
    if callable(installer):
        return installer
    return None


class LazyProofReuseServiceResolver:
    """Resolve and construct proof-reuse services exactly once.

    The importer and installer are injectable so unit tests and managed
    environments do not need network or process access.  No arbitrary module
    name or package spec can reach the installer.
    """

    def __init__(
        self,
        *,
        importer: Callable[[str], Any] | None = None,
        installer: Any = None,
    ) -> None:
        if importer is not None and not callable(importer):
            raise TypeError("importer must be callable")
        if installer is not None and _installer_callable(installer) is None:
            raise TypeError("installer must be callable or expose install()")
        self._importer = importer or importlib.import_module
        self._installer = installer
        self._resolution: ProofReuseServiceResolution | None = None
        self._lock = threading.Lock()

    def _load_dependency(
        self,
        dependency: ProofReuseDependency,
    ) -> tuple[Any | None, bool]:
        if PROOF_REUSE_DEPENDENCY_ALLOWLIST.get(dependency.module_name) != dependency:
            return None, False
        provenance_owner = self._installer
        if dependency.module_name == DATASETS_VERIFIER_MODULE:
            if provenance_owner is None:
                # Read-only validator/activator: constructor is inert and
                # allow_install=False forbids pip, writes, and source materialization.
                provenance_owner = AllowlistedPipInstaller(environ=os.environ)
            prepare = getattr(provenance_owner, "prepare_dependency", None)
            if callable(prepare):
                try:
                    if isinstance(provenance_owner, AllowlistedPipInstaller):
                        prepare(dependency, allow_install=False)
                    else:
                        prepare(dependency)
                except Exception:
                    pass
        try:
            module = self._importer(dependency.module_name)
        except ModuleNotFoundError as exc:
            if not _is_requested_module_absence(
                exc,
                dependency.module_name,
            ):
                return None, False
            install = _installer_callable(self._installer)
            if install is None:
                return None, False
            try:
                installed = install(dependency) is True
            except Exception:
                return None, False
            if not installed:
                return None, False
            importlib.invalidate_caches()
            try:
                module = self._importer(dependency.module_name)
            except Exception:
                return None, False
            was_installed = True
        except Exception:
            return None, False
        else:
            was_installed = False

        if any(
            getattr(module, symbol, None) is None
            for symbol in dependency.required_symbols
        ):
            return None, False
        if dependency.module_name == DATASETS_VERIFIER_MODULE:
            validate = getattr(provenance_owner, "validate_module_provenance", None)
            if not callable(validate):
                return None, False
            try:
                if validate(dependency, module) is not True:
                    return None, False
            except Exception:
                return None, False
        return module, was_installed

    def _resolve_once(
        self,
        cache_root: str | os.PathLike[str],
    ) -> ProofReuseServiceResolution:
        installed_modules: list[str] = []
        # NLTK is an allowlisted first-use capability, but certificate lookup
        # does not use it.  Do not install/import it while assembling the
        # narrower proof-reuse service bundle.
        for dependency in _SERVICE_DEPENDENCIES:
            _module, installed = self._load_dependency(dependency)
            if _module is None:
                return ProofReuseServiceResolution.unavailable(
                    dependency.unavailable_reason
                )
            if installed:
                installed_modules.append(dependency.module_name)

        try:
            store_module = self._importer(STORE_MODULE)
            provider_module = self._importer(PROVIDER_MODULE)
            lookup_module = self._importer(LOOKUP_MODULE)
            store_type = store_module.TestCertificateStore
            provider_type = provider_module.IpfsDatasetsTestCertificateProvider
            lookup_type = lookup_module.ProofReuseLookup
        except Exception:
            return ProofReuseServiceResolution.unavailable("plugin_unavailable")

        try:
            provider = provider_type()
            capabilities = provider.capabilities()
            if getattr(capabilities, "prove_on_lookup", None) is not False:
                return ProofReuseServiceResolution.unavailable(
                    "certificate_provider_unavailable"
                )
        except Exception:
            return ProofReuseServiceResolution.unavailable(
                "certificate_provider_unavailable"
            )

        try:
            root = Path(cache_root)
            store = store_type(root)
            if not all(
                callable(getattr(store, name, None))
                for name in ("lookup", "put_candidate", "put_receipt")
            ):
                return ProofReuseServiceResolution.unavailable("cache_unavailable")
        except Exception:
            return ProofReuseServiceResolution.unavailable("cache_unavailable")

        try:
            # Prefer production two-stage lookup when the module exports it;
            # fall back to the legacy single-stage constructor for hermetic
            # doubles that only expose ProofReuseLookup.
            two_stage_type = getattr(lookup_module, "ProofReuseTwoStageLookup", None)
            build_two_stage = getattr(
                lookup_module, "build_proof_reuse_two_stage_lookup", None
            )
            if callable(build_two_stage):
                lookup = build_two_stage(
                    proof_cache_store=store,
                    certificate_provider=provider,
                    require_runtime_frontier=True,
                )
            elif two_stage_type is not None:
                lookup = two_stage_type(
                    proof_cache_store=store,
                    certificate_provider=provider,
                    provider=provider,
                )
            else:
                lookup = lookup_type(store=store, provider=provider)
        except Exception:
            try:
                lookup = lookup_type(store=store, provider=provider)
            except Exception:
                return ProofReuseServiceResolution.unavailable("plugin_unavailable")
        return ProofReuseServiceResolution(
            available=True,
            reason_code="",
            lookup=lookup,
            store=store,
            provider=provider,
            installed_modules=tuple(installed_modules),
        )

    def resolve(
        self,
        *,
        cache_root: str | os.PathLike[str],
    ) -> ProofReuseServiceResolution:
        """Return the memoized all-or-nothing service bundle."""

        if self._resolution is not None:
            return self._resolution
        with self._lock:
            if self._resolution is None:
                self._resolution = self._resolve_once(cache_root)
        return self._resolution


DEFAULT_PROOF_REUSE_SERVICES_INTERFACE: Final = "ProofReuseServices@1"
LAZY_REAL_TEST_CERTIFICATE_ISSUER_INTERFACE: Final = (
    "LazyRealTestCertificateIssuer@1"
)
DATASETS_GROTH16_ENABLE_ENV: Final = "IPFS_DATASETS_ENABLE_GROTH16"
CANDIDATE_CONTEXT_CACHE_SUBDIR: Final = "candidate-context"
CERTIFICATE_CACHE_SUBDIR: Final = "certificates"


class LazyRealTestCertificateIssuer:
    """Non-None lazy real datasets issuer (PTR-147).

    Construction and attribute access never import optional datasets modules,
    never start a native build, and never prove.  The first controller
    :meth:`issue` call may invoke the bounded Groth16 provisioner and runtime
    readiness inspection only when the explicit native-build policy permits it.
    The generic pre-PTR-144 knowledge-of-axioms backend alone is never treated
    as certificate authority.  ``IPFS_DATASETS_ENABLE_GROTH16=1`` is published
    into the process environment only after the test-pass-specific circuit/key
    capability and exact provenance are ready.
    """

    interface: str = LAZY_REAL_TEST_CERTIFICATE_ISSUER_INTERFACE

    def __init__(
        self,
        *,
        store: Any = None,
        installer: Any = None,
        environ: Mapping[str, str] | None = None,
        artifacts_root: str | os.PathLike[str] | None = None,
        binary_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._store = store
        self._installer = installer
        self._environ = environ
        self._artifacts_root = (
            Path(artifacts_root) if artifacts_root is not None else None
        )
        self._binary_path = Path(binary_path) if binary_path is not None else None
        self._lock = threading.RLock()
        self._factory: Any = None
        self._enable_published = False
        self._last_bindings: Any = None
        self._last_reason: str = ""

    @property
    def factory(self) -> Any:
        return self._factory

    @property
    def enable_env_published(self) -> bool:
        return self._enable_published

    @property
    def last_artifact_bindings(self) -> Any:
        return self._last_bindings

    def _env_view(self) -> Mapping[str, str]:
        return self._environ if self._environ is not None else os.environ

    def _maybe_provision_native(self) -> None:
        """Call the bounded provisioner only under explicit native-build policy."""

        if not groth16_build_enabled(self._env_view()):
            return
        installer = self._installer
        if installer is None:
            return
        ensure = getattr(installer, "ensure_groth16_native_backend", None)
        if not callable(ensure):
            return
        try:
            ensure(consent=True)
        except Exception:
            return
        inspect = getattr(installer, "inspect_groth16_runtime", None)
        if callable(inspect):
            try:
                inspect()
            except Exception:
                return

    def _derive_bindings(self) -> Any:
        """Derive circuit/VK CIDs from exact activated artifact bytes."""

        try:
            from .publication import Groth16ArtifactIdentityBindings

            return Groth16ArtifactIdentityBindings.from_activated_artifacts(
                artifacts_root=self._artifacts_root,
                environ=self._env_view(),
                binary_path=self._binary_path,
            )
        except Exception:
            return None

    def _publish_enable_env_if_ready(self, bindings: Any) -> bool:
        if bindings is None or not getattr(bindings, "provenance_ready", False):
            return False
        if self._enable_published:
            return True
        # Publish only after test-pass-specific circuit/key provenance is ready.
        target = self._environ
        if target is None:
            os.environ[DATASETS_GROTH16_ENABLE_ENV] = "1"
        elif isinstance(target, dict):
            target[DATASETS_GROTH16_ENABLE_ENV] = "1"
        self._enable_published = True
        return True

    def _ensure_factory(self) -> Any | None:
        if self._factory is not None:
            return self._factory
        with self._lock:
            if self._factory is not None:
                return self._factory
            self._maybe_provision_native()
            bindings = self._derive_bindings()
            self._last_bindings = bindings
            if bindings is not None and not getattr(
                bindings, "provenance_ready", False
            ):
                # Missing/synthetic/stale/mismatched provenance: keep factory
                # unbuilt so issue() returns typed DEFERRED without proving.
                self._last_reason = str(
                    getattr(bindings, "reason_code", "") or "artifact_provenance_unready"
                )
                return None
            self._publish_enable_env_if_ready(bindings)
            try:
                from ipfs_datasets_py.logic.zkp.test_certificate_issuer import (
                    build_default_test_certificate_issuer,
                )
            except Exception:
                self._last_reason = "issuer_import_unavailable"
                return None
            try:
                kwargs: dict[str, Any] = {
                    "store": self._store,
                    "environ": self._env_view(),
                }
                # Prefer the real Groth16 provider factory path (PTR-144).
                try:
                    from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
                        LazyGroth16TestCertificateProvider,
                        build_default_test_certificate_issuer as build_groth16,
                    )

                    provider = LazyGroth16TestCertificateProvider(
                        binary_path=self._binary_path,
                        artifacts_root=self._artifacts_root,
                        environ=self._env_view(),
                        require_enable_env=True,
                    )
                    self._factory = build_groth16(
                        store=self._store,
                        provider=provider,
                        environ=self._env_view(),
                    )
                except Exception:
                    self._factory = build_default_test_certificate_issuer(**kwargs)
            except Exception:
                self._last_reason = "issuer_construction_failed"
                self._factory = None
            return self._factory

    def issue(self, request: Any) -> Any:
        """Issue or defer; never raises into the pytest outcome path."""

        class _Deferred:
            status = "certificate_deferred"
            reason = "issuer_unavailable"
            certificate = None
            certificate_cid = ""
            indexed = False

            def to_dict(self) -> dict[str, Any]:
                return {
                    "status": self.status,
                    "reason": self.reason,
                    "certificate_cid": self.certificate_cid,
                    "indexed": False,
                }

        try:
            factory = self._ensure_factory()
            if factory is None:
                deferred = _Deferred()
                deferred.reason = self._last_reason or "issuer_unavailable"
                return deferred
            issue = getattr(factory, "issue", None)
            if not callable(issue):
                deferred = _Deferred()
                deferred.reason = "issuer_method_unavailable"
                return deferred
            return issue(request)
        except Exception:
            deferred = _Deferred()
            deferred.reason = "issuer_exception"
            return deferred

    issue_deferred = issue
    __call__ = issue


@dataclass(frozen=True, slots=True)
class DefaultProofReuseServices:
    """Session-scoped default dependency injection for one pytest process.

    Explicit injected handles always override lazy defaults.  Construction and
    attribute access never open a network socket or install a package; callers
    resolve optional lookup/store/provider services through the lazy resolver.

    Defaults use separate persistent candidate-context and certificate stores,
    a current-context provider, revalidator, two-stage lookup, and a non-None
    lazy real issuer without eager optional imports.
    """

    interface: str = DEFAULT_PROOF_REUSE_SERVICES_INTERFACE
    identity_services: Any = None
    lookup: Any = None
    store: Any = None
    candidate_store: Any = None
    provider: Any = None
    issuer: Any = None
    revalidator: Any = None
    current_context_provider: Any = None
    resolver: Any = None
    resolution: Any = None
    source: str = "defaults"
    degraded: bool = False
    reason_code: str = ""

    @property
    def available(self) -> bool:
        return not self.degraded

    def with_overrides(
        self,
        *,
        identity_services: Any = None,
        lookup: Any = None,
        store: Any = None,
        candidate_store: Any = None,
        provider: Any = None,
        issuer: Any = None,
        revalidator: Any = None,
        current_context_provider: Any = None,
    ) -> DefaultProofReuseServices:
        """Return a copy with only the provided non-None fields replaced."""

        return DefaultProofReuseServices(
            interface=self.interface,
            identity_services=(
                identity_services
                if identity_services is not None
                else self.identity_services
            ),
            lookup=lookup if lookup is not None else self.lookup,
            store=store if store is not None else self.store,
            candidate_store=(
                candidate_store
                if candidate_store is not None
                else self.candidate_store
            ),
            provider=provider if provider is not None else self.provider,
            issuer=issuer if issuer is not None else self.issuer,
            revalidator=(revalidator if revalidator is not None else self.revalidator),
            current_context_provider=(
                current_context_provider
                if current_context_provider is not None
                else self.current_context_provider
            ),
            resolver=self.resolver,
            resolution=self.resolution,
            source=self.source if identity_services is None else "explicit",
            degraded=self.degraded,
            reason_code=self.reason_code,
        )


def _try_build_candidate_context_store(
    cache_root: str | os.PathLike[str] | None,
) -> Any | None:
    if cache_root is None:
        return None
    try:
        from ...agent_supervisor.proof.test_candidate_context_store import (
            TestCandidateContextStore,
        )

        root = Path(cache_root) / CANDIDATE_CONTEXT_CACHE_SUBDIR
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return TestCandidateContextStore(root)
    except Exception:
        return None


def _try_build_certificate_store(
    cache_root: str | os.PathLike[str] | None,
) -> Any | None:
    if cache_root is None:
        return None
    try:
        from ...agent_supervisor.proof.test_certificate_store import (
            TestCertificateStore,
        )

        root = Path(cache_root) / CERTIFICATE_CACHE_SUBDIR
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return TestCertificateStore(root)
    except Exception:
        return None


def compose_default_proof_reuse_services(
    *,
    mode: Any = None,
    root_path: str | os.PathLike[str] | None = None,
    config: Any = None,
    cache_root: str | os.PathLike[str] | None = None,
    resolver: Any = None,
    installer: Any = None,
    identity_services: Any = None,
    lookup: Any = None,
    store: Any = None,
    candidate_store: Any = None,
    provider: Any = None,
    issuer: Any = None,
    revalidator: Any = None,
    current_context_provider: Any = None,
    environ: Mapping[str, str] | None = None,
) -> DefaultProofReuseServices:
    """Assemble scoped defaults without item monkeypatches or path registries.

    Production defaults (PTR-147):

    * advance reviewed datasets revision/manifest pins (module constants);
    * persistent dedicated candidate-context + certificate stores;
    * current-context provider, revalidator, two-stage lookup;
    * non-None lazy real issuer without eager optional imports.

    Collection and lookup never build or prove.  Every optional-boundary
    failure returns a degraded-but-usable bundle so execution continues.
    Explicit service arguments always win.
    """

    reason_code = ""
    degraded = False
    env = environ if environ is not None else os.environ

    resolved_identity = identity_services
    if resolved_identity is None and mode is not None:
        try:
            from .default_identity_services import build_default_identity_services

            resolved_identity = build_default_identity_services(
                mode=mode,
                root_path=root_path,
                config=config,
            )
        except Exception:
            resolved_identity = None
            degraded = True
            reason_code = "identity_services_unavailable"

    resolved_candidate_store = candidate_store
    if resolved_candidate_store is None and cache_root is not None:
        resolved_candidate_store = _try_build_candidate_context_store(cache_root)
        if resolved_candidate_store is None:
            degraded = True
            reason_code = reason_code or "candidate_store_unavailable"

    resolved_store = store
    resolved_provider = provider
    resolved_lookup = lookup
    resolution: ProofReuseServiceResolution | None = None
    resolved_resolver = resolver

    # Prefer dedicated certificate-store layout under cache_root when the
    # caller did not inject an explicit store.  Optional provider resolution
    # remains fail-open.
    if resolved_store is None and cache_root is not None:
        resolved_store = _try_build_certificate_store(cache_root)

    if lookup is None or store is None or provider is None:
        if resolved_resolver is None:
            try:
                # Composition never upgrades a read-only caller into an
                # installer. Policy-aware facades/plugins must inject the
                # strict lazy installer explicitly when both consent gates
                # permit mutation.
                resolved_resolver = LazyProofReuseServiceResolver(installer=installer)
            except Exception:
                resolved_resolver = None
                degraded = True
                reason_code = reason_code or "service_resolver_unavailable"
        if resolved_resolver is not None and cache_root is not None:
            try:
                # Legacy all-or-nothing optional provider path; used only to
                # fill gaps. Never proves. May fail open.
                resolution = resolved_resolver.resolve(cache_root=cache_root)
            except Exception:
                resolution = ProofReuseServiceResolution.unavailable(
                    "plugin_unavailable"
                )
                degraded = True
                reason_code = reason_code or "plugin_unavailable"

    if isinstance(resolution, ProofReuseServiceResolution) and resolution.available:
        if resolved_store is None:
            resolved_store = resolution.store
        if resolved_provider is None:
            resolved_provider = resolution.provider
        if resolved_lookup is None:
            # Preserve the resolver's lookup object identity for hermetic
            # injection tests and production memoized resolution handles.
            resolved_lookup = resolution.lookup
    elif (
        isinstance(resolution, ProofReuseServiceResolution) and not resolution.available
    ):
        degraded = True
        reason_code = reason_code or resolution.reason_code or "plugin_unavailable"

    resolved_current_context = current_context_provider
    if resolved_current_context is None and resolved_identity is not None:
        try:
            from .current_context_provider import (
                build_default_current_context_provider,
            )

            resolved_current_context = build_default_current_context_provider(
                identity_services=resolved_identity,
                environ=env,
            )
        except Exception:
            resolved_current_context = None
            degraded = True
            reason_code = reason_code or "current_context_provider_unavailable"

    resolved_revalidator = revalidator
    if resolved_revalidator is None and (
        resolved_candidate_store is not None or resolved_store is not None
    ):
        try:
            from .runtime_revalidation import build_runtime_context_revalidator

            resolved_revalidator = build_runtime_context_revalidator(
                candidate_store=resolved_candidate_store,
                current_context_provider=resolved_current_context,
                require_runtime_frontier=True,
            )
        except Exception:
            resolved_revalidator = None
            degraded = True
            reason_code = reason_code or "revalidator_unavailable"

    if resolved_lookup is None:
        try:
            from .lookup import build_proof_reuse_two_stage_lookup

            resolved_lookup = build_proof_reuse_two_stage_lookup(
                candidate_context_store=resolved_candidate_store,
                certificate_provider=resolved_provider,
                proof_cache_store=resolved_store,
                revalidator=resolved_revalidator,
                current_context_provider=resolved_current_context,
                identity_services=resolved_identity,
                environ=env,
                require_runtime_frontier=True,
            )
        except Exception:
            resolved_lookup = None
            degraded = True
            reason_code = reason_code or "two_stage_lookup_unavailable"

    # Non-None lazy real issuer without eager optional imports.  Explicit
    # injections always win; construction never imports datasets.
    resolved_issuer = issuer
    if resolved_issuer is None:
        resolved_issuer = LazyRealTestCertificateIssuer(
            store=resolved_store,
            installer=installer,
            environ=env if isinstance(env, dict) else None,
        )

    return DefaultProofReuseServices(
        identity_services=resolved_identity,
        lookup=resolved_lookup,
        store=resolved_store,
        candidate_store=resolved_candidate_store,
        provider=resolved_provider,
        issuer=resolved_issuer,
        revalidator=resolved_revalidator,
        current_context_provider=resolved_current_context,
        resolver=resolved_resolver,
        resolution=resolution,
        source="defaults" if identity_services is None else "explicit",
        degraded=degraded,
        reason_code=reason_code,
    )


__all__ = [
    "AllowlistedPipInstaller",
    "CANDIDATE_CONTEXT_CACHE_SUBDIR",
    "CERTIFICATE_CACHE_SUBDIR",
    "DATASETS_GROTH16_ARTIFACTS_ROOT_ENV",
    "DATASETS_GROTH16_BINARY_ENV",
    "DATASETS_GROTH16_BUNDLED_BINARIES_SHA256",
    "DATASETS_GROTH16_ENABLE_ENV",
    "DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256",
    "DATASETS_GROTH16_REVIEWED_FILES_SHA256",
    "DATASETS_PYTHON_BUILD_FILES_SHA256",
    "DATASETS_VERIFIER_DEPENDENCY",
    "DATASETS_VERIFIER_DISTRIBUTION",
    "DATASETS_VERIFIER_MODULE",
    "DATASETS_VERIFIER_REVISION",
    "DATASETS_VERIFIER_RELEASE_BLOCKER",
    "DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED",
    "DATASETS_VERIFIER_SOURCE_SHA256",
    "DATASETS_VERIFIER_SNAPSHOT_BYTES",
    "DATASETS_VERIFIER_SNAPSHOT_FILES",
    "DATASETS_VERIFIER_SNAPSHOT_SHA256",
    "DATASETS_VERIFIER_ZKP_TREE_OBJECT",
    "DEFAULT_PROOF_REUSE_SERVICES_INTERFACE",
    "DefaultProofReuseServices",
    "DEFAULT_NLTK_DATA_RESOURCES",
    "LAZY_REAL_TEST_CERTIFICATE_ISSUER_INTERFACE",
    "LOOKUP_MODULE",
    "JSONSCHEMA_DEPENDENCY",
    "JSONSCHEMA_MODULE",
    "LazyProofReuseServiceResolver",
    "LazyRealTestCertificateIssuer",
    "MULTIFORMATS_DEPENDENCY",
    "MULTIFORMATS_MODULE",
    "NLTK_DATA_RESOURCE_ALLOWLIST",
    "NLTK_DEPENDENCY",
    "NLTK_MODULE",
    "NltkDataResource",
    "PROOF_REUSE_AUTO_INSTALL_ENV",
    "PROOF_REUSE_CACHE_DIR_ENV",
    "PROOF_REUSE_DATASETS_SOURCE_ENV",
    "PROOF_REUSE_DEPENDENCY_ALLOWLIST",
    "PROOF_REUSE_GROTH16_BUILD_ENV",
    "PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV",
    "PROOF_REUSE_GROTH16_ENDPOINT_ENV",
    "PROOF_REUSE_NLTK_DATA_DIR_ENV",
    "PROOF_REUSE_NLTK_DOWNLOAD_ENV",
    "PROOF_REUSE_PROVISION_DIR_ENV",
    "PROVIDER_MODULE",
    "ProofReuseDependency",
    "ProofReuseServiceResolution",
    "STORE_MODULE",
    "automatic_install_enabled",
    "compose_default_proof_reuse_services",
    "groth16_build_enabled",
    "nltk_data_download_enabled",
    "proof_reuse_dependency_plan",
]
