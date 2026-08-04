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
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, ClassVar, Final

PROOF_REUSE_AUTO_INSTALL_ENV: Final = "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"
# Package-wide installation consent is a second, independent gate.  It lives
# here (rather than only in ``lazy_dependencies``) so the pure dependency plan
# can report the *effective* policy without importing the runtime installer.
PACKAGE_AUTO_INSTALL_ENV: Final = "IPFS_ACCEL_AUTO_INSTALL"
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
PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_ENV: Final = (
    "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST"
)
PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256_ENV: Final = (
    "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256"
)
PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV: Final = (
    "IPFS_TEST_PROOF_REUSE_GROTH16_NATIVE_RECEIPT"
)

GROTH16_TEST_PASS_ARTIFACT_MANIFEST_INTERFACE: Final = (
    "Groth16TestPassArtifactManifest@1"
)
GROTH16_NATIVE_BUILD_RECEIPT_INTERFACE: Final = "Groth16NativeBuildReceipt@2"
TEST_PASS_GROTH16_CIRCUIT_VERSION: Final = 4
TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS: Final = (1, 2, 3, 4)
TEST_PASS_GROTH16_PROVIDER_RELATIVE_PATH: Final = (
    "ipfs_datasets_py/logic/zkp/test_pass_groth16_provider.py"
)
TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256: Final = (
    "4e00956c627a0e2e9a59ec241697a663f64a56a4a346ea05e701cf02c2e3254a"
)
TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256: Final = (
    "c674f630154212abd5e77ebeb4614dace5890b29ea7eddce44d92d5280ca472a"
)
TEST_PASS_GROTH16_CIRCUIT_CID: Final = (
    "baguqeerayz2pmmaviijkxvphp27liyknvtsysczj5j7n3tse3ewvfagki4va"
)
TEST_PASS_GROTH16_RULESET_ID: Final = "test_pass_v2"
TEST_PASS_GROTH16_STATEMENT_INTERFACE: Final = "TestPassStatementV2"
TEST_PASS_GROTH16_STATEMENT_VERSION: Final = 2

# Exact merged PTR-151 commit: v4-capable native release + lazy real test-pass
# provider.  Labels never substitute for the byte/object checks below.
DATASETS_VERIFIER_REVISION: Final = "1894e9dca7dced0690893d468e40751a14f0b15b"
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
            "5c70be1b69fb189d97b2f2b137b19000eaf8f13f7605bb1ec0ea8df6df6eb073"
        ),
        "setup.py": (
            "f0640649d73a23654274180e76e35703e38bee210c0780ed3f6841030a091825"
        ),
    }
)
DATASETS_VERIFIER_DISTRIBUTION: Final = (
    "ipfs-accelerate-proof-reuse-verifier==0.2.0+1894e9dc"
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
            "b82ac5c233f74a758d6d5f9d31edefa41dbee686cfb4d1a60bd2e9df53c2dac0"
        ),
        "Cargo.lock": (
            "592b3736d8e2c25f54aa1c7f5ea8cd1c1649c644762d1973f2687918bf9e470f"
        ),
        "build.rs": (
            "ead50ca34f9fa9cf3c9b31f0c33b1db08b3da5a7ed40b73dd51004166f724a3d"
        ),
        "build.sh": (
            "8f1fce11b3342303af3f3e54354c9b1d127fe9dda69e135716af2b172ff98b47"
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
            "86f15d779b37b6766101d165945895577df6c0fa71472395863ae4e7e7b8b3fa"
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
            "d883348d24a6dc6c0ab25745b3dab7a759e1566799ddaaf90429f21a0e469055"
        ),
    }
)
# Capabilities are accepted only when the binary digest, release manifest, and
# bounded ``capabilities --json`` output all match their exact reviewed pins.
DATASETS_GROTH16_BUNDLED_BINARY_CAPABILITIES: Final[
    Mapping[str, tuple[int, ...]]
] = MappingProxyType(
    {
        "linux-aarch64": (1, 2, 3, 4),
    }
)
DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256: Final[Mapping[str, str]] = (
    MappingProxyType(
        {
            "linux-aarch64": (
                "033990805b50b7229c394809b3c549eda88f705b9358826313d79da0714fea33"
            ),
        }
    )
)
DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256: Final[Mapping[str, str]] = (
    MappingProxyType(
        {
            "linux-aarch64": (
                "7625046099fc44760dd858af3f976bd37341ff1ca327fad30e0654ee8ad6109f"
            ),
        }
    )
)
DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY: Final = (
    "sha256:93dbdcb273114f6ec578f8f80bea185ac57f67f0b86daa6f0ff1d2575903691c"
)
DATASETS_GROTH16_TEST_PASS_CAPABILITY_PAYLOAD_SHA256: Final = (
    "7625046099fc44760dd858af3f976bd37341ff1ca327fad30e0654ee8ad6109f"
)
# Deliberately empty until an operator-reviewed v4 trusted-setup ceremony
# publishes exact proving/verifying key digests.  Tests may monkeypatch the
# imported value with a test-only manifest digest; production code must never
# invent or learn authority from a self-pinned environment variable.
DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256: Final[frozenset[str]] = (
    frozenset()
)


def validate_groth16_capability_payload(
    payload: bytes,
    *,
    required_circuit_version: int = TEST_PASS_GROTH16_CIRCUIT_VERSION,
) -> bool:
    """Validate the exact PTR-151 artifact-free native capability document."""

    if (
        type(payload) is not bytes
        or not payload
        or len(payload) >= 16_384
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
        or hashlib.sha256(payload).hexdigest()
        != DATASETS_GROTH16_TEST_PASS_CAPABILITY_PAYLOAD_SHA256
    ):
        return False
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict):
        return False
    if (
        document.get("locked_source_identity")
        != DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY
        or document.get("locked_source_identity_schema")
        != "ipfs-datasets-groth16-locked-source-v1"
    ):
        return False
    circuits = document.get("supported_circuits")
    if not isinstance(circuits, list):
        return False
    by_version = {
        item.get("version"): item for item in circuits if isinstance(item, dict)
    }
    required = by_version.get(required_circuit_version)
    if required_circuit_version == TEST_PASS_GROTH16_CIRCUIT_VERSION:
        if required != {
            "version": TEST_PASS_GROTH16_CIRCUIT_VERSION,
            "profile": "test-pass-v2",
            "ruleset_id": TEST_PASS_GROTH16_RULESET_ID,
            "can_setup": True,
            "can_prove": True,
            "can_verify": True,
        }:
            return False
    elif not isinstance(required, dict):
        return False
    return document.get("trusted_setup") == {
        "automatic_during_build": False,
        "explicit_command_required": True,
        "deterministic_seed_is_test_only": True,
        "capabilities_reads_or_writes_artifacts": False,
    }


def validate_groth16_release_manifest_payload(
    payload: bytes,
    *,
    platform_name: str,
    binary_sha256: str,
) -> bool:
    """Validate exact reviewed release-manifest bytes and bound identities."""

    expected_digest = DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256.get(platform_name)
    if (
        type(payload) is not bytes
        or not expected_digest
        or hashlib.sha256(payload).hexdigest() != expected_digest
    ):
        return False
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 1
        and manifest.get("platform") == platform_name
        and manifest.get("binary") == "groth16"
        and manifest.get("binary_sha256") == binary_sha256
        and manifest.get("capabilities_sha256")
        == DATASETS_GROTH16_TEST_PASS_CAPABILITY_PAYLOAD_SHA256
        and manifest.get("locked_source_identity")
        == DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY
        and manifest.get("locked_source_identity_schema")
        == "ipfs-datasets-groth16-locked-source-v1"
        and manifest.get("supported_circuit_versions")
        == list(TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS)
        and manifest.get("test_pass_circuit")
        == {
            "version": TEST_PASS_GROTH16_CIRCUIT_VERSION,
            "profile": "test-pass-v2",
            "ruleset_id": TEST_PASS_GROTH16_RULESET_ID,
        }
        and manifest.get("trusted_setup_included") is False
        and manifest.get("v4_keys_included") is False
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


def reviewed_groth16_source_fingerprint() -> str:
    """Return the exact reviewed revision/build-input identity."""

    digest = hashlib.sha256()
    digest.update(f"revision:{DATASETS_VERIFIER_REVISION}\n".encode())
    for relative, expected in sorted(
        DATASETS_GROTH16_REVIEWED_FILES_SHA256.items()
    ):
        digest.update(f"{relative}:{expected}\n".encode())
    return digest.hexdigest()


DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT: Final = (
    reviewed_groth16_source_fingerprint()
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
_PRIVATE_TARGET_PUBLICATION_LOCK = threading.Lock()


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


def package_auto_install_policy_permits(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return the independent package-wide lazy-install policy.

    An unset package policy follows the package's existing safe default:
    permission is granted only inside a virtual environment.  Invalid values
    deny permission.  This function is pure and starts no process.
    """

    source = os.environ if environ is None else environ
    if PACKAGE_AUTO_INSTALL_ENV not in source:
        try:
            return sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        except Exception:
            return False
    value = str(source.get(PACKAGE_AUTO_INSTALL_ENV, "")).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return False


def proof_reuse_install_permitted(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Require both proof-reuse and package-wide installation consent."""

    return automatic_install_enabled(environ) and package_auto_install_policy_permits(
        environ
    )


def isolated_pip_environment(
    environ: Mapping[str, str] | None = None,
    *,
    additions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a pip environment that ignores inherited Python/pip config.

    The command still receives ordinary OS variables (notably ``PATH`` and
    certificate/proxy settings), but inherited ``PIP_*`` configuration and
    Python import-path injection cannot expand the allowlisted install.
    Dependency-specific fixed variables are applied only after sanitization.
    """

    source = os.environ if environ is None else environ
    cleaned = {
        str(key): str(value)
        for key, value in source.items()
        if not str(key).upper().startswith("PIP_")
        and str(key).upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}
    }
    cleaned.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if additions:
        cleaned.update({str(key): str(value) for key, value in additions.items()})
    return cleaned


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
    proof_gate = automatic_install_enabled(environ)
    package_gate = package_auto_install_policy_permits(environ)
    effective_install = proof_gate and package_gate
    return {
        "interface": "ProofReuseDependencyPlan@1",
        "lazy": True,
        "cold_import_inert": True,
        "fail_open_to_run": True,
        # Backward-compatible key now reports the policy that is actually
        # enforced by the lazy installer: both independent gates must allow.
        "automatic_install_enabled": effective_install,
        "proof_reuse_auto_install_enabled": proof_gate,
        "package_auto_install_enabled": package_gate,
        "effective_auto_install_enabled": effective_install,
        "disable_environment_variable": PROOF_REUSE_AUTO_INSTALL_ENV,
        "package_disable_environment_variable": PACKAGE_AUTO_INSTALL_ENV,
        "installation_policy": {
            "proof_reuse_gate": proof_gate,
            "package_gate": package_gate,
            "effective": effective_install,
            "operator": "logical_and",
        },
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
            "reviewed_bundled_binary_capabilities": {
                platform_name: list(versions)
                for platform_name, versions in sorted(
                    DATASETS_GROTH16_BUNDLED_BINARY_CAPABILITIES.items()
                )
            },
            "reviewed_release_manifest_sha256": dict(
                DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256
            ),
            "reviewed_capability_payload_sha256": (
                DATASETS_GROTH16_TEST_PASS_CAPABILITY_PAYLOAD_SHA256
            ),
            "reviewed_locked_source_identity": (
                DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY
            ),
            "capability_probe_required": True,
            "installed_distribution_package_data_discovery": True,
            "installed_distribution_import_required": False,
            "test_pass_required_circuit_version": (
                TEST_PASS_GROTH16_CIRCUIT_VERSION
            ),
            "reviewed_source_supported_circuit_versions": list(
                TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS
            ),
            "reviewed_source_fingerprint": (
                DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT
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
                "authority_manifest_environment_variable": (
                    PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_ENV
                ),
                "authority_manifest_sha256_environment_variable": (
                    PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256_ENV
                ),
                "approved_v4_manifest_digests": len(
                    DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256
                ),
                "authority_ready_before_reviewed_ceremony": bool(
                    DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256
                ),
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
        # Cold static plan only.  Live readiness/composition claims must come
        # from ``live_runtime_activation_inventory`` /
        # ``ProofReuseRuntimeActivationReport@1`` (PTR-149), never from these
        # hard-coded booleans alone.
        "runtime_activation_live_report_interface": (
            "ProofReuseRuntimeActivationReport@1"
        ),
        "runtime_activation_live_probe_required": True,
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
_PYTHON_DEPENDENCY_RECEIPT_NAME: Final = ".proof-reuse-python-dependency.json"
_PYTHON_DEPENDENCY_RECEIPT_INTERFACE: Final = "ProofReusePythonDependencyTarget@1"
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


def _read_bounded_regular_file(path: Path, *, max_bytes: int) -> bytes | None:
    """Read a stable regular-file descriptor without following a leaf link."""

    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        return None
    descriptor = -1
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            return None
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_size < 0
            or opened.st_size > max_bytes
        ):
            return None
        payload = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                return None
            payload.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            return None
        after = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or getattr(opened, "st_mtime_ns", None)
            != getattr(after, "st_mtime_ns", None)
            or getattr(opened, "st_ctime_ns", None)
            != getattr(after, "st_ctime_ns", None)
        ):
            return None
        return bytes(payload)
    except (OSError, TypeError, ValueError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _datasets_snapshot_payloads(root: Path) -> Mapping[str, bytes] | None:
    """Read only regular, non-symlink manifest files contained by *root*."""

    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    payloads: dict[str, bytes] = {}
    total_bytes = 0
    for relative in DATASETS_VERIFIER_SNAPSHOT_FILES:
        path = root / relative
        try:
            if path.is_symlink() or not path.is_file():
                return None
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
            if resolved != path.absolute():
                return None
            remaining = DATASETS_VERIFIER_SNAPSHOT_BYTES - total_bytes
            payload = _read_bounded_regular_file(path, max_bytes=remaining)
            if payload is None:
                return None
        except (OSError, RuntimeError, ValueError):
            return None
        payloads[relative] = payload
        total_bytes += len(payload)
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
        self._provision_root = base_provision_root
        self._python_snapshot_root = base_provision_root / "python-snapshots"
        self._python_dependency_root = base_provision_root / "python-dependencies"
        self._outcomes: dict[str, bool] = {}
        self._install_diagnostics: dict[
            str, tuple[bool, int | None, str, BaseException | None]
        ] = {}
        self._active_dependency_targets: dict[str, Path] = {}
        self._active_dependency_import_roots: dict[str, frozenset[str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _bounded_completed_output(completed: Any) -> str:
        """Retain bounded diagnostics supplied by an injected process runner.

        The production subprocess continues to write to ``DEVNULL`` so an
        installer cannot accumulate unbounded pip output.  Test and embedding
        runners may nevertheless return ``stdout``/``stderr`` fields; keeping
        a bounded head and tail lets the lazy policy classify expected pip
        failures without trusting or retaining an arbitrary-sized payload.
        """

        per_stream_limit = 32 * 1024
        truncation_marker = "\n...[truncated]...\n"

        def fragment(name: str) -> str:
            try:
                value = getattr(completed, name, "")
            except Exception:
                return ""
            if isinstance(value, bytes):
                if len(value) > per_stream_limit:
                    half = (per_stream_limit - len(truncation_marker)) // 2
                    value = (
                        value[:half]
                        + truncation_marker.encode()
                        + value[-half:]
                    )
                return value.decode("utf-8", errors="replace")
            if not isinstance(value, str):
                return ""
            if len(value) <= per_stream_limit:
                return value
            half = (per_stream_limit - len(truncation_marker)) // 2
            return value[:half] + truncation_marker + value[-half:]

        return "\n".join(
            part for part in (fragment("stdout"), fragment("stderr")) if part
        )

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

    def _private_provision_directory(
        self, path: Path, *, create: bool
    ) -> Path | None:
        """Validate every directory from the provision root to *path*."""

        try:
            base = self._provision_root.absolute()
            requested = path.absolute()
            requested.relative_to(base)
            # Reject aliases and attacker-controlled ancestors before creating
            # through them.  Root-owned and current-user-owned ancestors are
            # accepted when they are not group/other writable.  The one normal
            # exception is the platform temporary directory itself: it must be
            # sticky, while the provision root below it remains owner-private.
            current_uid = getattr(os, "getuid", lambda: None)()
            try:
                platform_temp = Path(tempfile.gettempdir()).resolve(strict=True)
            except (OSError, RuntimeError):
                platform_temp = None
            for ancestor in (base, *base.parents):
                if ancestor.is_symlink():
                    return None
                if not ancestor.exists():
                    continue
                metadata = os.lstat(ancestor)
                if not ancestor.is_dir():
                    return None
                if os.name != "nt":
                    if current_uid is not None and metadata.st_uid not in {
                        0,
                        current_uid,
                    }:
                        return None
                    if metadata.st_mode & 0o022:
                        sticky_platform_temp = (
                            platform_temp is not None
                            and ancestor.resolve(strict=True) == platform_temp
                            and bool(metadata.st_mode & 0o1000)
                        )
                        if not sticky_platform_temp:
                            return None
            if not base.exists() and not create:
                return None
            validated_base = self._private_directory(base, create=create)
            if validated_base != base:
                return None
            current = base
            for component in requested.relative_to(base).parts:
                current = current / component
                validated = self._private_directory(current, create=create)
                if validated != current:
                    return None
            return requested
        except (OSError, RuntimeError, ValueError):
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
        if self._private_provision_directory(root, create=False) is None:
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

    @staticmethod
    def _dependency_descriptor(dependency: ProofReuseDependency) -> dict[str, Any]:
        return {
            "module_name": dependency.module_name,
            "distribution": dependency.distribution,
            "required_symbols": list(dependency.required_symbols),
            "pip_options": list(dependency.pip_options),
            "install_environment": [list(item) for item in dependency.install_environment],
            "python_cache_tag": str(sys.implementation.cache_tag or ""),
            "python_version": [sys.version_info.major, sys.version_info.minor],
            "platform_tag": sysconfig.get_platform(),
        }

    def _dependency_target_prefix(self, dependency: ProofReuseDependency) -> str:
        descriptor = json.dumps(
            self._dependency_descriptor(dependency),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        identity = hashlib.sha256(descriptor).hexdigest()
        safe_module = dependency.module_name.replace(".", "-")[:80]
        return f"{safe_module}-{identity}"

    def _dependency_target(
        self, dependency: ProofReuseDependency, *, tree_sha256: str
    ) -> Path:
        return self._python_dependency_root / (
            f"{self._dependency_target_prefix(dependency)}-{tree_sha256}"
        )

    def _dependency_target_candidates(
        self, dependency: ProofReuseDependency
    ) -> tuple[Path, ...]:
        try:
            if not self._python_dependency_root.is_dir():
                return ()
            prefix = f"{self._dependency_target_prefix(dependency)}-"
            candidates: list[Path] = []
            for child in self._python_dependency_root.iterdir():
                if child.name.startswith(prefix):
                    candidates.append(child)
                    if len(candidates) > 2:
                        return tuple(candidates)
            return tuple(sorted(candidates, key=lambda item: item.name))
        except OSError:
            return ()

    @staticmethod
    def _dependency_tree_digest(root: Path) -> tuple[str, int, int] | None:
        try:
            resolved_root = root.resolve(strict=True)
            files: list[Path] = []
            entries = 0
            for path in root.rglob("*"):
                entries += 1
                if entries > 40_000 or path.is_symlink():
                    return None
                if (
                    path.name != _PYTHON_DEPENDENCY_RECEIPT_NAME
                    and path.is_file()
                ):
                    files.append(path)
                    if len(files) > 20_000:
                        return None
            files.sort(key=lambda path: path.relative_to(root).as_posix())
        except (OSError, RuntimeError, ValueError):
            return None
        if not files:
            return None
        digest = hashlib.sha256()
        total = 0
        for path in files:
            try:
                metadata = os.lstat(path)
                if not stat.S_ISREG(metadata.st_mode):
                    return None
                resolved = path.resolve(strict=True)
                resolved.relative_to(resolved_root)
                relative_text = path.relative_to(root).as_posix()
                if len(relative_text.encode("utf-8")) > 512:
                    return None
                if metadata.st_size < 0 or metadata.st_size > 64 * 1024 * 1024:
                    return None
                if total + metadata.st_size > 512 * 1024 * 1024:
                    return None
                payload = _read_bounded_regular_file(
                    path,
                    max_bytes=min(
                        64 * 1024 * 1024,
                        512 * 1024 * 1024 - total,
                    ),
                )
            except (OSError, RuntimeError, ValueError):
                return None
            if payload is None or len(payload) != metadata.st_size:
                return None
            total += len(payload)
            relative = relative_text.encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(str(len(payload)).encode("ascii"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(payload).digest())
        return digest.hexdigest(), total, len(files)

    def _validated_dependency_target_at(
        self, dependency: ProofReuseDependency, root: Path
    ) -> Path | None:
        if self._private_provision_directory(root, create=False) is None:
            return None
        receipt_path = root / _PYTHON_DEPENDENCY_RECEIPT_NAME
        try:
            if receipt_path.is_symlink() or receipt_path.stat().st_size > 16_384:
                return None
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        expected_keys = {
            "interface",
            "descriptor",
            "descriptor_sha256",
            "tree_sha256",
            "tree_bytes",
            "tree_files",
            "private_target",
        }
        if not isinstance(receipt, dict) or set(receipt) != expected_keys:
            return None
        descriptor = self._dependency_descriptor(dependency)
        descriptor_bytes = json.dumps(
            descriptor, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if (
            receipt.get("interface") != _PYTHON_DEPENDENCY_RECEIPT_INTERFACE
            or receipt.get("descriptor") != descriptor
            or receipt.get("descriptor_sha256")
            != hashlib.sha256(descriptor_bytes).hexdigest()
            or receipt.get("private_target") is not True
        ):
            return None
        tree = self._dependency_tree_digest(root)
        if tree is None or tree != (
            receipt.get("tree_sha256"),
            receipt.get("tree_bytes"),
            receipt.get("tree_files"),
        ):
            return None
        if root != self._dependency_target(dependency, tree_sha256=tree[0]):
            return None
        top_level = dependency.module_name.split(".", 1)[0]
        if not ((root / top_level).is_dir() or (root / f"{top_level}.py").is_file()):
            return None
        if not self._snapshot_tree_is_private(root):
            return None
        return root.resolve(strict=True)

    def _validated_dependency_target(
        self, dependency: ProofReuseDependency
    ) -> Path | None:
        candidates = self._dependency_target_candidates(dependency)
        if len(candidates) != 1:
            return None
        return self._validated_dependency_target_at(dependency, candidates[0])

    @staticmethod
    def _dependency_import_roots(root: Path) -> frozenset[str] | None:
        roots: set[str] = set()
        try:
            children = tuple(root.iterdir())
            if len(children) > 2_048:
                return None
            for child in children:
                if child.is_symlink() or child.suffix == ".pth":
                    return None
                name = child.name
                if name == _PYTHON_DEPENDENCY_RECEIPT_NAME:
                    continue
                if child.is_dir():
                    if name.endswith((".dist-info", ".egg-info", ".data")) or name in {
                        "bin",
                        "__pycache__",
                    }:
                        continue
                    if name.isidentifier():
                        roots.add(name)
                elif child.is_file() and child.suffix == ".py":
                    if child.stem.isidentifier():
                        roots.add(child.stem)
            return frozenset(roots) if roots else None
        except OSError:
            return None

    @staticmethod
    def _loaded_import_roots_match_target(
        root: Path, import_roots: frozenset[str]
    ) -> bool:
        try:
            resolved_root = root.resolve(strict=True)
            for name, module in tuple(sys.modules.items()):
                if name.split(".", 1)[0] not in import_roots:
                    continue
                module_file = str(getattr(module, "__file__", "") or "")
                if not module_file:
                    return False
                Path(module_file).resolve(strict=True).relative_to(resolved_root)
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    def activate_cached_dependency(self, dependency: ProofReuseDependency) -> bool:
        """Activate an intact private target without importing its module."""

        if dependency.module_name == DATASETS_VERIFIER_MODULE:
            return self.activate_cached_datasets_verifier()
        target = self._validated_dependency_target(dependency)
        if target is None:
            return False
        import_roots = self._dependency_import_roots(target)
        if import_roots is None or not self._loaded_import_roots_match_target(
            target, import_roots
        ):
            return False
        target_text = str(target)
        while target_text in sys.path:
            sys.path.remove(target_text)
        # The private target must control every package it resolved.  Existing
        # loaded modules from those roots were rejected above.
        sys.path.insert(0, target_text)
        self._active_dependency_targets[dependency.module_name] = target
        self._active_dependency_import_roots[dependency.module_name] = import_roots
        importlib.invalidate_caches()
        return True

    def _install_dependency_private_target(
        self,
        dependency: ProofReuseDependency,
    ) -> tuple[bool, int | None, str, BaseException | None]:
        if self.activate_cached_dependency(dependency):
            return True, 0, "cached private target", None
        if self._dependency_target_candidates(dependency):
            return False, None, "corrupt private target", None
        parent = self._private_provision_directory(
            self._python_dependency_root, create=True
        )
        if parent is None:
            return False, None, "private target unavailable", None
        distribution = self._selected_distribution(dependency)
        if not distribution:
            return False, None, "no matching distribution found", None
        try:
            temporary = tempfile.TemporaryDirectory(
                prefix=".python-dependency-install-",
                dir=parent,
                ignore_cleanup_errors=True,
            )
        except OSError as exc:
            return False, None, "private target unavailable", exc
        with temporary:
            staging = Path(temporary.name) / "target"
            try:
                staging.mkdir(mode=0o700)
            except OSError as exc:
                return False, None, "private target unavailable", exc
            command = (
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--no-compile",
                "--target",
                str(staging),
                *dependency.pip_options,
                distribution,
            )
            run_environment = isolated_pip_environment(
                self._environ,
                additions=dict(dependency.install_environment),
            )
            try:
                completed = self._runner(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self._timeout_seconds,
                    env=run_environment,
                )
            except Exception as exc:
                return False, None, "", exc
            output = self._bounded_completed_output(completed)
            try:
                raw_code = getattr(completed, "returncode", 1)
                if isinstance(raw_code, bool) or not isinstance(raw_code, int):
                    raise TypeError("invalid process return code")
                code = raw_code
            except Exception as exc:
                return False, None, output, exc
            if code != 0:
                return False, code, output, None
            tree = self._dependency_tree_digest(staging)
            if tree is None:
                return False, code, "installed target is empty or invalid", None
            target = self._dependency_target(dependency, tree_sha256=tree[0])
            descriptor = self._dependency_descriptor(dependency)
            descriptor_bytes = json.dumps(
                descriptor, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            receipt = {
                "interface": _PYTHON_DEPENDENCY_RECEIPT_INTERFACE,
                "descriptor": descriptor,
                "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
                "tree_sha256": tree[0],
                "tree_bytes": tree[1],
                "tree_files": tree[2],
                "private_target": True,
            }
            try:
                (staging / _PYTHON_DEPENDENCY_RECEIPT_NAME).write_text(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                with _PRIVATE_TARGET_PUBLICATION_LOCK:
                    if self.activate_cached_dependency(dependency):
                        return True, 0, "concurrent private target reused", None
                    if self._dependency_target_candidates(dependency):
                        return False, code, "conflicting private target", None
                    os.replace(staging, target)
                    if not self._harden_snapshot_tree(target):
                        return False, code, "private target hardening failed", None
            except OSError as exc:
                if self.activate_cached_dependency(dependency):
                    return True, 0, "concurrent private target reused", None
                return False, code, output, exc
        if not self.activate_cached_dependency(dependency):
            return False, 0, "private target provenance validation failed", None
        return True, 0, output, None

    def install_with_diagnostics(
        self, dependency: ProofReuseDependency
    ) -> tuple[bool, int | None, str, BaseException | None]:
        """Install to a private target and retain typed process diagnostics."""

        succeeded = self.install(dependency)
        return self._install_diagnostics.get(
            dependency.module_name,
            (succeeded, 0 if succeeded else None, "", None),
        )

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
        parent = self._private_provision_directory(
            self._python_snapshot_root, create=True
        )
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
            active = self._active_dependency_targets.get(dependency.module_name)
            if active is None:
                # A dependency already provided by the interpreter remains an
                # operator/environment responsibility.  Once this installer
                # activates a private target, provenance is exact and local.
                return True
            module_file = str(getattr(module, "__file__", "") or "")
            if not module_file:
                return False
            try:
                loaded = Path(module_file).resolve(strict=True)
                loaded.relative_to(active.resolve(strict=True))
            except (OSError, RuntimeError, ValueError):
                return False
            import_roots = self._active_dependency_import_roots.get(
                dependency.module_name
            )
            return bool(
                import_roots
                and self._validated_dependency_target(dependency) == active
                and self._loaded_import_roots_match_target(active, import_roots)
            )
        return self.validate_datasets_authority_module(module)

    def validate_datasets_authority_module(self, module: Any) -> bool:
        """Attest one datasets ZKP module and the loaded authority closure."""

        module_name = str(getattr(module, "__name__", "") or "")
        expected_text = _datasets_authority_module_relative_path(module_name)
        if expected_text is None:
            return False
        module_file = str(getattr(module, "__file__", "") or "")
        if not module_file:
            return False
        expected_relative = Path(expected_text)
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
                self._install_diagnostics[dependency.module_name] = (
                    succeeded,
                    0 if succeeded else None,
                    "private reviewed snapshot" if succeeded else "snapshot unavailable",
                    None,
                )
                self._outcomes[dependency.module_name] = succeeded
                return succeeded
            result = self._install_dependency_private_target(dependency)
            succeeded = result[0]
            self._install_diagnostics[dependency.module_name] = result
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
        validate = getattr(provenance_owner, "validate_module_provenance", None)
        if dependency.module_name == DATASETS_VERIFIER_MODULE and not callable(
            validate
        ):
            return None, False
        if callable(validate):
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
# PTR-153: public proof-bearing material interface (never carries witness).
PROOF_BEARING_ISSUANCE_MATERIAL_INTERFACE: Final = "IssuedTestCertificateMaterial@1"
ISSUED_MATERIAL_DISPOSITION_INTERFACE: Final = "IssuedMaterialDisposition@1"
DATASETS_GROTH16_ENABLE_ENV: Final = "IPFS_DATASETS_ENABLE_GROTH16"
CANDIDATE_CONTEXT_CACHE_SUBDIR: Final = "candidate-context"
CERTIFICATE_CACHE_SUBDIR: Final = "certificates"

# Public certificate/proof material size bounds (PTR-153 fail-closed).
MAX_ISSUED_CERTIFICATE_BYTES: Final = 1_048_576
MAX_ISSUED_PROOF_BYTES: Final = 4 * 1024 * 1024
MAX_ISSUED_MATERIAL_JSON_BYTES: Final = 4 * 1024 * 1024

# Strict child-process environment for native prove/verify.  Loader and
# interpreter injection variables are never inherited.
_NATIVE_CHILD_ENV_ALLOWLIST: Final = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "USER",
        "LOGNAME",
        "UID",
        "SHELL",
        "TERM",
        "XDG_RUNTIME_DIR",
        "XDG_CACHE_HOME",
    }
)
_NATIVE_CHILD_ENV_DENY_PREFIXES: Final = (
    "LD_",
    "DYLD_",
    "PYTHON",
    "PERL",
    "RUBYOPT",
    "NODE_",
    "JAVA_",
    "JVM_",
    "OPENSSL_",
    "SSL_",
    "CURL_",
    "GIT_",
    "CARGO_",
    "RUST",
    "PIP_",
    "NPM_",
    "BUN_",
    "DENO_",
)
_PRIVATE_MATERIAL_FIELD_MARKERS: Final = frozenset(
    {
        "witness",
        "private",
        "secret",
        "opening",
        "proving_key",
        "proving-key",
        "receipt_opening",
        "retained_receipt",
        "retained_candidate",
        "private_axioms",
        "local_witness",
        "sk",
        "seed_phrase",
    }
)


def _sha256_file_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_private_material_key(name: str) -> bool:
    lowered = str(name or "").strip().lower().replace("-", "_")
    if not lowered:
        return False
    if lowered in _PRIVATE_MATERIAL_FIELD_MARKERS:
        return True
    return any(marker in lowered for marker in _PRIVATE_MATERIAL_FIELD_MARKERS)


def redact_private_material_fields(value: Any, *, depth: int = 0) -> Any:
    """Return a public-only projection; drop private/secret keys recursively."""

    if depth > 12:
        return None
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _is_private_material_key(name):
                continue
            cleaned[name] = redact_private_material_fields(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [redact_private_material_fields(item, depth=depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray)):
        # Never surface raw private-looking blobs through the public interface.
        return f"bytes:{len(value)}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            mapped = to_dict(include_proof=True, include_ids=True)
        except TypeError:
            try:
                mapped = to_dict()
            except Exception:
                mapped = None
        except Exception:
            mapped = None
        if isinstance(mapped, Mapping):
            return redact_private_material_fields(mapped, depth=depth + 1)
    return str(type(value).__name__)


def allowlisted_native_child_environment(
    source: Mapping[str, str] | None = None,
    *,
    artifacts_root: str | os.PathLike[str],
    binary_path: str | os.PathLike[str] | None = None,
    extras: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a strict child env for native prove/verify.

    * Overwrites (never inherits) the pinned artifacts root.
    * Excludes loader/interpreter injection (``LD_PRELOAD``, ``DYLD_*``, …).
    * Allowlists only a closed set of ambient OS variables.
    """

    ambient = os.environ if source is None else source
    cleaned: dict[str, str] = {}
    for key, value in ambient.items():
        name = str(key)
        upper = name.upper()
        if upper.startswith(_NATIVE_CHILD_ENV_DENY_PREFIXES):
            continue
        if upper in {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "LD_AUDIT",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "DYLD_FRAMEWORK_PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONSTARTUP",
            DATASETS_GROTH16_ARTIFACTS_ROOT_ENV.upper()
            if hasattr(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "upper")
            else "GROTH16_BACKEND_ARTIFACTS_ROOT",
            DATASETS_GROTH16_BINARY_ENV,
        }:
            continue
        if name not in _NATIVE_CHILD_ENV_ALLOWLIST and upper not in {
            n.upper() for n in _NATIVE_CHILD_ENV_ALLOWLIST
        }:
            continue
        cleaned[name] = str(value)
    # Force pinned artifacts root (overwrite any ambient value).
    cleaned[DATASETS_GROTH16_ARTIFACTS_ROOT_ENV] = str(Path(artifacts_root))
    if binary_path is not None:
        cleaned[DATASETS_GROTH16_BINARY_ENV] = str(Path(binary_path))
    cleaned[DATASETS_GROTH16_ENABLE_ENV] = "1"
    if extras:
        for key, value in extras.items():
            name = str(key)
            upper = name.upper()
            if upper.startswith(_NATIVE_CHILD_ENV_DENY_PREFIXES):
                continue
            if upper in {"LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH"}:
                continue
            cleaned[name] = str(value)
    # Final overwrite guarantees: never inherit ambient artifacts/binary pins.
    cleaned[DATASETS_GROTH16_ARTIFACTS_ROOT_ENV] = str(Path(artifacts_root))
    if binary_path is not None:
        cleaned[DATASETS_GROTH16_BINARY_ENV] = str(Path(binary_path))
    return cleaned


class ImmutableNativeArtifactSession:
    """Private immutable snapshot of a native binary and v4 key material.

    Capability probes may hash mutable paths; prove/verify must never execute
    those mutable paths.  This session copies exact reviewed bytes into a
    private directory (or binds an FD-backed executable path) and revalidates
    digests atomically before every use.
    """

    __slots__ = (
        "_root",
        "_binary_path",
        "_binary_sha256",
        "_binary_size",
        "_binary_fd",
        "_artifacts_root",
        "_proving_key_sha256",
        "_verifying_key_sha256",
        "_proving_key_size",
        "_verifying_key_size",
        "_closed",
        "_lock",
    )

    def __init__(
        self,
        *,
        binary_bytes: bytes,
        proving_key_bytes: bytes,
        verifying_key_bytes: bytes,
        expected_proving_key_sha256: str = "",
        expected_verifying_key_sha256: str = "",
        binary_mode: int = 0o500,
    ) -> None:
        if not binary_bytes:
            raise ValueError("binary_bytes required")
        if not proving_key_bytes or not verifying_key_bytes:
            raise ValueError("key bytes required")
        binary_digest = _sha256_file_bytes(binary_bytes)
        pk_digest = _sha256_file_bytes(proving_key_bytes)
        vk_digest = _sha256_file_bytes(verifying_key_bytes)
        if expected_proving_key_sha256 and pk_digest != expected_proving_key_sha256:
            raise ValueError("proving_key_digest_mismatch")
        if (
            expected_verifying_key_sha256
            and vk_digest != expected_verifying_key_sha256
        ):
            raise ValueError("verifying_key_digest_mismatch")

        root = Path(
            tempfile.mkdtemp(prefix="proof-reuse-native-snap-", dir=None)
        )
        try:
            os.chmod(root, 0o700)
            binary_path = root / "groth16.snap"
            binary_path.write_bytes(binary_bytes)
            os.chmod(binary_path, binary_mode)
            artifacts = root / "artifacts"
            version_dir = artifacts / f"v{TEST_PASS_GROTH16_CIRCUIT_VERSION}"
            version_dir.mkdir(parents=True, mode=0o700)
            pk_path = version_dir / "proving_key.bin"
            vk_path = version_dir / "verifying_key.bin"
            pk_path.write_bytes(proving_key_bytes)
            vk_path.write_bytes(verifying_key_bytes)
            os.chmod(pk_path, 0o400)
            os.chmod(vk_path, 0o400)
            # Best-effort FD bind so replacements of the snapshot path are
            # detectable via revalidation against the open descriptor.
            binary_fd = -1
            try:
                binary_fd = os.open(
                    binary_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                )
            except OSError:
                binary_fd = -1
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

        self._root = root
        self._binary_path = binary_path
        self._binary_sha256 = binary_digest
        self._binary_size = len(binary_bytes)
        self._binary_fd = binary_fd
        self._artifacts_root = artifacts
        self._proving_key_sha256 = pk_digest
        self._verifying_key_sha256 = vk_digest
        self._proving_key_size = len(proving_key_bytes)
        self._verifying_key_size = len(verifying_key_bytes)
        self._closed = False
        self._lock = threading.RLock()

    @property
    def binary_path(self) -> Path:
        return self._binary_path

    @property
    def artifacts_root(self) -> Path:
        return self._artifacts_root

    @property
    def binary_sha256(self) -> str:
        return self._binary_sha256

    @property
    def proving_key_sha256(self) -> str:
        return self._proving_key_sha256

    @property
    def verifying_key_sha256(self) -> str:
        return self._verifying_key_sha256

    @property
    def fd_bound(self) -> bool:
        return self._binary_fd >= 0

    def revalidate(self) -> bool:
        """Atomically re-hash snapshot bytes; return False on any drift."""

        with self._lock:
            if self._closed:
                return False
            try:
                binary = self._binary_path.read_bytes()
                if (
                    len(binary) != self._binary_size
                    or _sha256_file_bytes(binary) != self._binary_sha256
                ):
                    return False
                if self._binary_fd >= 0:
                    try:
                        # Ensure the FD still refers to the same inode size.
                        fd_stat = os.fstat(self._binary_fd)
                        if int(fd_stat.st_size) != self._binary_size:
                            return False
                    except OSError:
                        return False
                version = (
                    self._artifacts_root / f"v{TEST_PASS_GROTH16_CIRCUIT_VERSION}"
                )
                pk = (version / "proving_key.bin").read_bytes()
                vk = (version / "verifying_key.bin").read_bytes()
                if (
                    len(pk) != self._proving_key_size
                    or _sha256_file_bytes(pk) != self._proving_key_sha256
                ):
                    return False
                if (
                    len(vk) != self._verifying_key_size
                    or _sha256_file_bytes(vk) != self._verifying_key_sha256
                ):
                    return False
            except OSError:
                return False
            return True

    def child_environment(
        self,
        source: Mapping[str, str] | None = None,
        *,
        extras: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        if not self.revalidate():
            raise RuntimeError("native_snapshot_identity_drift")
        return allowlisted_native_child_environment(
            source,
            artifacts_root=self._artifacts_root,
            binary_path=self._binary_path,
            extras=extras,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._binary_fd >= 0:
                try:
                    os.close(self._binary_fd)
                except OSError:
                    pass
                self._binary_fd = -1
            shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> "ImmutableNativeArtifactSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _mapping_certificate_payload(certificate: Any) -> dict[str, Any] | None:
    if certificate is None:
        return None
    if isinstance(certificate, Mapping):
        return redact_private_material_fields(dict(certificate))
    to_dict = getattr(certificate, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict(include_proof=True, include_ids=True)
        except TypeError:
            try:
                payload = to_dict()
            except Exception:
                return None
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return redact_private_material_fields(dict(payload))
    return None


def _bounded_json_size(payload: Mapping[str, Any], *, limit: int) -> bool:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        return False
    return len(encoded) <= limit


@dataclass(frozen=True, slots=True)
class ProofBearingIssuanceMaterial:
    """Complete bounded public proof-bearing issuance material (PTR-153).

    Contains the certificate/proof needed for local verification plus reviewed
    artifact bindings.  Private witness openings and proving-key bytes never
    appear.  Material alone does not grant warm-skip or publication authority;
    the controller (PTR-155) must reverify under its own context.
    """

    __test__: ClassVar[bool] = False

    certificate: Mapping[str, Any]
    proof_digest: str
    proof_artifact_cid: str
    circuit_cid: str
    verifying_key_cid: str
    artifact_bindings: Mapping[str, Any] = field(default_factory=dict)
    proof_json: Mapping[str, Any] = field(default_factory=dict)
    backend_circuit_version: int = TEST_PASS_GROTH16_CIRCUIT_VERSION
    verified_locally: bool = True
    interface: str = PROOF_BEARING_ISSUANCE_MATERIAL_INTERFACE
    status: str = "issued"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "certificate",
            MappingProxyType(dict(self.certificate)),
        )
        object.__setattr__(
            self,
            "artifact_bindings",
            MappingProxyType(dict(self.artifact_bindings)),
        )
        object.__setattr__(
            self,
            "proof_json",
            MappingProxyType(dict(self.proof_json)),
        )

    @property
    def issued(self) -> bool:
        return True

    @property
    def deferred(self) -> bool:
        return False

    @property
    def can_authorize_skip(self) -> bool:
        # Public material is not skip authority until controller-side V2
        # verification and atomic publication (PTR-155).
        return False

    @property
    def authority(self) -> str:
        return "non_authoritative_until_controller_verify"

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize public material only (never witness/key secrets)."""

        return {
            "interface": self.interface,
            "status": self.status,
            "verified_locally": self.verified_locally,
            "backend_circuit_version": int(self.backend_circuit_version),
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "proof_digest": self.proof_digest,
            "proof_artifact_cid": self.proof_artifact_cid,
            "certificate": redact_private_material_fields(dict(self.certificate)),
            "proof_json": redact_private_material_fields(dict(self.proof_json)),
            "artifact_bindings": redact_private_material_fields(
                dict(self.artifact_bindings)
            ),
            "can_authorize_skip": False,
            "authority": self.authority,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_public_dict()


@dataclass(frozen=True, slots=True)
class IssuedMaterialDisposition:
    """Typed deferred/rejected/unavailable result with no authority."""

    __test__: ClassVar[bool] = False

    status: str = "certificate_deferred"
    reason: str = "issuer_unavailable"
    certificate: None = None
    certificate_cid: str = ""
    material: None = None
    indexed: bool = False
    interface: str = ISSUED_MATERIAL_DISPOSITION_INTERFACE

    @property
    def issued(self) -> bool:
        return False

    @property
    def deferred(self) -> bool:
        status = str(self.status or "").lower()
        return "defer" in status or status in {"", "run", "unavailable"}

    @property
    def can_authorize_skip(self) -> bool:
        return False

    @property
    def authority(self) -> str:
        return "non_attested"

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "status": self.status,
            "reason": self.reason,
            "certificate_cid": self.certificate_cid,
            "indexed": False,
            "material": None,
            "can_authorize_skip": False,
            "authority": self.authority,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return self.to_dict()


def _mapping_contains_private_keys(value: Any, *, depth: int = 0) -> bool:
    if depth > 12:
        return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_private_material_key(str(key)):
                return True
            if _mapping_contains_private_keys(item, depth=depth + 1):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _mapping_contains_private_keys(item, depth=depth + 1) for item in value
        )
    return False


def admit_proof_bearing_issuance_material(
    material: Any,
    *,
    expected_circuit_cid: str = "",
    expected_verifying_key_cid: str = "",
    max_certificate_bytes: int = MAX_ISSUED_CERTIFICATE_BYTES,
    max_proof_bytes: int = MAX_ISSUED_PROOF_BYTES,
    max_material_bytes: int = MAX_ISSUED_MATERIAL_JSON_BYTES,
) -> tuple[ProofBearingIssuanceMaterial | None, str]:
    """Admit provider output as public material, or return a reject reason.

    Malformed, oversized, provenance-mismatched, or structurally incomplete
    provider output is rejected.  Private fields are stripped.
    """

    if material is None:
        return None, "material_missing"
    if isinstance(material, ProofBearingIssuanceMaterial):
        candidate = material
        public = candidate.to_public_dict()
    elif isinstance(material, Mapping):
        # Refuse private-bearing input before redaction can hide it.
        if _mapping_contains_private_keys(material):
            return None, "private_material_present"
        public = redact_private_material_fields(dict(material))
        if not isinstance(public, Mapping):
            return None, "material_malformed"
        certificate = public.get("certificate")
        if not isinstance(certificate, Mapping):
            return None, "certificate_missing"
        try:
            candidate = ProofBearingIssuanceMaterial(
                certificate=dict(certificate),
                proof_digest=str(public.get("proof_digest") or ""),
                proof_artifact_cid=str(public.get("proof_artifact_cid") or ""),
                circuit_cid=str(public.get("circuit_cid") or ""),
                verifying_key_cid=str(public.get("verifying_key_cid") or ""),
                artifact_bindings=dict(public.get("artifact_bindings") or {}),
                proof_json=dict(public.get("proof_json") or {}),
                backend_circuit_version=int(
                    public.get("backend_circuit_version")
                    or TEST_PASS_GROTH16_CIRCUIT_VERSION
                ),
                verified_locally=bool(public.get("verified_locally", True)),
            )
        except Exception:
            return None, "material_structurally_incomplete"
        public = candidate.to_public_dict()
    else:
        # datasets IssuedTestCertificateMaterial (duck-typed).
        certificate = _mapping_certificate_payload(
            getattr(material, "certificate", None)
        )
        if certificate is None:
            return None, "certificate_missing"
        proof_json = getattr(material, "proof_json", None)
        if not isinstance(proof_json, Mapping):
            proof_json = {}
        try:
            candidate = ProofBearingIssuanceMaterial(
                certificate=certificate,
                proof_digest=str(getattr(material, "proof_digest", "") or ""),
                proof_artifact_cid=str(
                    getattr(material, "proof_artifact_cid", "") or ""
                ),
                circuit_cid=str(getattr(material, "circuit_cid", "") or ""),
                verifying_key_cid=str(
                    getattr(material, "verifying_key_cid", "") or ""
                ),
                artifact_bindings={},
                proof_json=redact_private_material_fields(dict(proof_json)),
                backend_circuit_version=int(
                    getattr(
                        material,
                        "backend_circuit_version",
                        TEST_PASS_GROTH16_CIRCUIT_VERSION,
                    )
                    or TEST_PASS_GROTH16_CIRCUIT_VERSION
                ),
                verified_locally=bool(
                    getattr(material, "verified_locally", True)
                ),
            )
        except Exception:
            return None, "material_structurally_incomplete"
        public = candidate.to_public_dict()

    if not candidate.proof_digest or not candidate.proof_artifact_cid:
        return None, "proof_identity_missing"
    if not candidate.circuit_cid or not candidate.verifying_key_cid:
        return None, "provenance_pins_missing"
    if not isinstance(candidate.certificate, Mapping) or not candidate.certificate:
        return None, "certificate_empty"
    cert_map = dict(candidate.certificate)
    for required in (
        "receipt_cid",
        "circuit_cid",
        "verifying_key_cid",
        "proof_digest",
    ):
        if not str(cert_map.get(required) or "").strip():
            # Some schemas nest ids; accept proof_digest at material top-level.
            if required == "proof_digest" and candidate.proof_digest:
                continue
            if required in {"circuit_cid", "verifying_key_cid"} and getattr(
                candidate, required, ""
            ):
                continue
            if required == "receipt_cid" and str(
                cert_map.get("receipt_id") or ""
            ).strip():
                continue
            return None, f"certificate_missing_{required}"

    if expected_circuit_cid and candidate.circuit_cid != expected_circuit_cid:
        return None, "circuit_cid_provenance_mismatch"
    if (
        expected_verifying_key_cid
        and candidate.verifying_key_cid != expected_verifying_key_cid
    ):
        return None, "verifying_key_cid_provenance_mismatch"

    if not _bounded_json_size(cert_map, limit=max_certificate_bytes):
        return None, "certificate_oversized"
    if candidate.proof_json and not _bounded_json_size(
        dict(candidate.proof_json), limit=max_proof_bytes
    ):
        return None, "proof_oversized"
    if not _bounded_json_size(public, limit=max_material_bytes):
        return None, "material_oversized"

    # Reject private leakage that survived redaction.
    encoded = json.dumps(public, sort_keys=True, default=str)
    lowered = encoded.lower()
    for marker in (
        "receipt_opening_hex",
        "private_axioms",
        "proving_key",
        "local_witness",
        "retained_receipt_bytes",
    ):
        if marker in lowered:
            return None, "private_material_present"
    return candidate, ""


class LazyRealTestCertificateIssuer:
    """Non-None lazy real datasets issuer (PTR-147 / PTR-153).

    Construction and attribute access never import optional datasets modules,
    never start a native build, and never prove.  :meth:`issue` remains the
    lightweight publication-path disposition (PTR-155 joins controller
    authority).  :meth:`issue_material` performs hardened prove/verify under an
    immutable private snapshot and strict child environment, returning public
    :class:`ProofBearingIssuanceMaterial` without private witness bytes.
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
        self._last_material: ProofBearingIssuanceMaterial | None = None
        self._last_session: ImmutableNativeArtifactSession | None = None

    @property
    def factory(self) -> Any:
        return self._factory

    @property
    def enable_env_published(self) -> bool:
        return self._enable_published

    @property
    def last_artifact_bindings(self) -> Any:
        return self._last_bindings

    @property
    def last_issued_material(self) -> ProofBearingIssuanceMaterial | None:
        return self._last_material

    def validate_authority_module_provenance(self, module: Any) -> bool:
        """Require the resolver's exact datasets authority snapshot."""

        validator = getattr(
            self._installer, "validate_authority_module_provenance", None
        )
        return bool(callable(validator) and validator(module))

    def verify_certificate_locally(
        self,
        certificate: Any,
        bindings: Any,
        context: Mapping[str, Any],
    ) -> Any:
        """Reconstruct controller pins and run the exact local v4 verifier.

        This is deliberately unavailable for generic/injected factories.  The
        The binding comes only from a controller-reconstructed typed request,
        never certificate metadata.  Its receipt identity is cross-checked
        before the reviewed datasets verifier invokes the same pinned native
        provider used for issuance.
        """

        if (
            bindings is None
            or bindings is not self._last_bindings
            or not getattr(bindings, "provenance_ready", False)
            or self._factory is None
            or not isinstance(certificate, Mapping)
            or not isinstance(context, Mapping)
        ):
            return False
        request = context.get("deferred_request")
        receipt = context.get("receipt")
        if not isinstance(receipt, Mapping):
            return False

        provider = getattr(self._factory, "provider", None)
        if provider is None or not callable(getattr(provider, "verify_proof_json", None)):
            return False
        try:
            provider_module = importlib.import_module(type(provider).__module__)
            verifier_module = importlib.import_module(
                "ipfs_datasets_py.logic.zkp.test_execution_certificate"
            )
            binding_module = importlib.import_module(
                "ipfs_datasets_py.logic.zkp.provekit.test_pass_circuit"
            )
            issuer_module = importlib.import_module(
                "ipfs_datasets_py.logic.zkp.test_certificate_issuer"
            )
        except Exception:
            return False
        if not all(
            self.validate_authority_module_provenance(module)
            for module in (
                provider_module,
                verifier_module,
                binding_module,
                issuer_module,
            )
        ):
            return False
        request_type = getattr(issuer_module, "DeferredTestCertificateRequest", None)
        if request_type is None or not isinstance(request, request_type):
            # A flattened xdist mapping cannot reconstruct the admitted-pass
            # statement.  It remains non-authoritative until the controller
            # retrieves the retained public CAS bytes and rebuilds this type.
            return False
        public_inputs = request.statement.to_public_inputs()
        receipt_cid = str(
            receipt.get("receipt_id") or receipt.get("receipt_cid") or ""
        )
        if not receipt_cid:
            try:
                from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
                    TestPassReceipt,
                )

                receipt_cid = TestPassReceipt.from_dict(receipt).receipt_id
            except Exception:
                return False
        if not receipt_cid or receipt_cid != request.receipt_cid:
            return False
        if (
            request.circuit_cid != bindings.circuit_cid
            or request.verifying_key_cid != bindings.verifying_key_cid
            or request.backend_id != "groth16"
            or request.proof_system_id != "groth16"
        ):
            return False
        try:
            provider_root = Path(provider.artifacts_root()).resolve(strict=True)
            expected_root = Path(bindings.artifacts_root).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        if provider_root != expected_root:
            return False

        class PinnedGroth16NativeVerifier:
            backend_id = "groth16"

            def verify_proof(self, proof: Any) -> bool:
                try:
                    proof_json = json.loads(bytes(proof.proof_data).decode("utf-8"))
                except Exception:
                    return False
                return bool(
                    isinstance(proof_json, Mapping)
                    and provider.verify_proof_json(proof_json) is True
                )

        try:
            binding = binding_module.TestPassCircuitBinding(
                request.statement,
                expected_public_inputs=public_inputs,
                backend_id="groth16",
                proof_system_id="groth16",
                circuit_cid=bindings.circuit_cid,
                verifying_key_cid=bindings.verifying_key_cid,
                statement_cid=request.statement_cid,
                issuer_id=request.issuer_id,
                policy_cid=request.policy_cid,
                epoch=request.epoch,
                candidate_context_cid=request.candidate_context_cid,
                verifier_artifacts={
                    "verifying_key_path": str(
                        expected_root / "v4" / "verifying_key.bin"
                    )
                },
            )
            return verifier_module.verify_test_execution_certificate_v2(
                certificate,
                binding,
                PinnedGroth16NativeVerifier(),
                expected_candidate_context_cid=request.candidate_context_cid,
            )
        except Exception:
            return False

    def _env_view(self) -> Mapping[str, str]:
        return self._environ if self._environ is not None else os.environ

    def _deferred_material(
        self,
        reason: str,
        *,
        status: str = "certificate_deferred",
    ) -> IssuedMaterialDisposition:
        self._last_reason = str(reason or "issuer_unavailable")[:96]
        self._last_material = None
        return IssuedMaterialDisposition(
            status=status,
            reason=self._last_reason,
        )

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
            ensure(
                consent=True,
                required_circuit_version=TEST_PASS_GROTH16_CIRCUIT_VERSION,
            )
        except TypeError:
            # Narrow compatibility for injected test/application installers.
            try:
                ensure(consent=True)
            except Exception:
                return
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

    def _resolve_binary_path(self, bindings: Any) -> Path | None:
        if self._binary_path is not None and self._binary_path.is_file():
            return self._binary_path
        env_bin = str(self._env_view().get(DATASETS_GROTH16_BINARY_ENV, "") or "").strip()
        if env_bin:
            path = Path(env_bin)
            if path.is_file():
                return path
        return None

    def _resolve_key_paths(self, bindings: Any) -> tuple[Path, Path] | None:
        root: Path | None = None
        if bindings is not None and getattr(bindings, "artifacts_root", ""):
            try:
                root = Path(str(bindings.artifacts_root))
            except (TypeError, ValueError):
                root = None
        if root is None and self._artifacts_root is not None:
            root = self._artifacts_root
        if root is None:
            override = str(
                self._env_view().get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "") or ""
            ).strip()
            if override:
                root = Path(override)
        if root is None:
            return None
        version = root / f"v{TEST_PASS_GROTH16_CIRCUIT_VERSION}"
        pk = version / "proving_key.bin"
        vk = version / "verifying_key.bin"
        if pk.is_file() and vk.is_file():
            return pk, vk
        return None

    def _open_immutable_session(
        self, bindings: Any
    ) -> ImmutableNativeArtifactSession | IssuedMaterialDisposition:
        """Bind exact reviewed bytes into a private immutable snapshot."""

        binary_path = self._resolve_binary_path(bindings)
        key_paths = self._resolve_key_paths(bindings)
        if binary_path is None:
            return self._deferred_material("binary_unavailable")
        if key_paths is None:
            return self._deferred_material("key_unavailable")
        pk_path, vk_path = key_paths
        try:
            binary_bytes = binary_path.read_bytes()
            pk_bytes = pk_path.read_bytes()
            vk_bytes = vk_path.read_bytes()
        except OSError:
            return self._deferred_material("artifact_read_failed")
        if not binary_bytes or not pk_bytes or not vk_bytes:
            return self._deferred_material("artifact_empty")
        expected_pk = str(getattr(bindings, "proving_key_sha256", "") or "")
        expected_vk = str(getattr(bindings, "verifying_key_sha256", "") or "")
        try:
            session = ImmutableNativeArtifactSession(
                binary_bytes=binary_bytes,
                proving_key_bytes=pk_bytes,
                verifying_key_bytes=vk_bytes,
                expected_proving_key_sha256=expected_pk,
                expected_verifying_key_sha256=expected_vk,
            )
        except ValueError as exc:
            reason = str(exc) or "artifact_digest_mismatch"
            return self._deferred_material(reason[:96])
        except Exception:
            return self._deferred_material("immutable_snapshot_failed")
        if not session.revalidate():
            session.close()
            return self._deferred_material("immutable_snapshot_unready")
        return session

    def _harden_provider_execution(
        self,
        provider: Any,
        session: ImmutableNativeArtifactSession,
    ) -> None:
        """Patch provider native execution onto the immutable snapshot.

        Replaces mutable-path / ambient-env subprocess launches with:
        * revalidation of snapshot digests at each use
        * strict allowlisted child environment
        * overwrite (not inherit) of the pinned artifacts root
        * execution of the private snapshot binary only
        """

        def _run_cli(
            args: list[str],
            *,
            stdin_bytes: bytes,
            timeout: float,
            env: dict[str, str] | None = None,
        ) -> subprocess.CompletedProcess[bytes]:
            if not session.revalidate():
                raise RuntimeError("native_snapshot_identity_drift")
            # Ignore caller-supplied env for injection vectors; rebuild strict.
            child_env = session.child_environment(self._env_view(), extras=None)
            # Explicitly drop any residual injection keys.
            for banned in list(child_env):
                upper = banned.upper()
                if upper.startswith(("LD_", "DYLD_")) or upper in {
                    "LD_PRELOAD",
                    "PYTHONPATH",
                    "PYTHONHOME",
                }:
                    child_env.pop(banned, None)
            # Final overwrite of pinned artifacts/binary.
            child_env[DATASETS_GROTH16_ARTIFACTS_ROOT_ENV] = str(
                session.artifacts_root
            )
            child_env[DATASETS_GROTH16_BINARY_ENV] = str(session.binary_path)
            if env:
                # Only permit non-injection extras that do not override pins.
                for key, value in env.items():
                    upper = str(key).upper()
                    if upper.startswith(("LD_", "DYLD_", "PYTHON")):
                        continue
                    if upper in {
                        DATASETS_GROTH16_ARTIFACTS_ROOT_ENV.upper()
                        if hasattr(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "upper")
                        else "GROTH16_BACKEND_ARTIFACTS_ROOT",
                        "GROTH16_BACKEND_ARTIFACTS_ROOT",
                        DATASETS_GROTH16_BINARY_ENV,
                    }:
                        continue
                    child_env[str(key)] = str(value)
            # Prefer /proc/self/fd execution when FD-bound (Linux).
            executable = str(session.binary_path)
            if session.fd_bound and Path("/proc/self/fd").is_dir():
                try:
                    fd_path = f"/proc/self/fd/{session._binary_fd}"
                    if Path(fd_path).exists():
                        executable = fd_path
                except Exception:
                    executable = str(session.binary_path)
            return subprocess.run(
                [executable, *list(args)],
                input=stdin_bytes,
                capture_output=True,
                timeout=timeout,
                env=child_env,
                check=False,
            )

        provider._run_cli = _run_cli  # type: ignore[attr-defined]
        # Point path resolvers at the immutable snapshot.
        provider._binary_path = session.binary_path
        provider._artifacts_root = session.artifacts_root
        provider._resolved_binary = session.binary_path

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

    def _bindings_public_payload(self, bindings: Any) -> dict[str, Any]:
        if bindings is None:
            return {"provenance_ready": False}
        to_dict = getattr(bindings, "to_dict", None)
        if callable(to_dict):
            try:
                payload = to_dict()
                if isinstance(payload, Mapping):
                    return redact_private_material_fields(dict(payload))
            except Exception:
                pass
        return {
            "provenance_ready": bool(getattr(bindings, "provenance_ready", False)),
            "circuit_cid": str(getattr(bindings, "circuit_cid", "") or "")[:128],
            "verifying_key_cid": str(
                getattr(bindings, "verifying_key_cid", "") or ""
            )[:128],
            "reason_code": str(getattr(bindings, "reason_code", "") or "")[:96],
            "backend_circuit_version": int(
                getattr(bindings, "backend_circuit_version", 0) or 0
            ),
            # Never include raw key bytes; digests only.
            "proving_key_sha256": str(
                getattr(bindings, "proving_key_sha256", "") or ""
            )[:64],
            "verifying_key_sha256": str(
                getattr(bindings, "verifying_key_sha256", "") or ""
            )[:64],
        }

    def issue_material(
        self,
        request: Any,
        *,
        local_witness: Any = None,
        local_receipt: Any = None,
        timeout_seconds: float | None = None,
        idempotency_key: str = "",
        **_ignored: Any,
    ) -> ProofBearingIssuanceMaterial | IssuedMaterialDisposition:
        """Prove under immutable inputs and return public material only.

        Private witness bytes stay process-local and never appear on the
        returned object, in logs, or in disposition payloads.  Post-binding
        binary/key replacement, ambient artifacts-root overrides, and loader
        injection all defer without executing substituted inputs.
        """

        _ = idempotency_key  # routing only; never logged
        # Close any prior session so mutated ambient paths cannot be reused.
        if self._last_session is not None:
            try:
                self._last_session.close()
            except Exception:
                pass
            self._last_session = None

        try:
            self._maybe_provision_native()
            bindings = self._derive_bindings()
            self._last_bindings = bindings
            if bindings is None or not getattr(bindings, "provenance_ready", False):
                reason = "artifact_provenance_unready"
                if bindings is not None:
                    reason = str(
                        getattr(bindings, "reason_code", "") or reason
                    )
                return self._deferred_material(reason)

            session_or_defer = self._open_immutable_session(bindings)
            if isinstance(session_or_defer, IssuedMaterialDisposition):
                return session_or_defer
            session = session_or_defer
            self._last_session = session

            # Revalidate immediately before constructing the provider so a
            # TOCTOU replacement of the source paths cannot influence the
            # snapshot (we already copied bytes) and snapshot drift defers.
            if not session.revalidate():
                session.close()
                self._last_session = None
                return self._deferred_material("post_binding_identity_drift")

            self._publish_enable_env_if_ready(bindings)

            try:
                from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
                    LazyGroth16TestCertificateProvider,
                )
            except Exception:
                session.close()
                self._last_session = None
                return self._deferred_material("provider_import_unavailable")

            # Child env is strict; do not pass ambient attacker variables.
            strict_env = session.child_environment(self._env_view())
            provider = LazyGroth16TestCertificateProvider(
                binary_path=session.binary_path,
                artifacts_root=session.artifacts_root,
                environ=strict_env,
                require_enable_env=False,
            )
            if timeout_seconds is not None:
                try:
                    provider.prove_timeout_seconds = float(timeout_seconds)
                except Exception:
                    pass
            self._harden_provider_execution(provider, session)

            # Remember a hardened factory handle for optional local verify.
            try:
                from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
                    build_default_test_certificate_issuer as build_groth16,
                )

                self._factory = build_groth16(
                    store=self._store,
                    provider=provider,
                    environ=strict_env,
                )
            except Exception:
                self._factory = None

            issue = getattr(provider, "issue", None)
            if not callable(issue):
                session.close()
                self._last_session = None
                return self._deferred_material("issuer_method_unavailable")

            # Detect ambient path substitution after binding: if the original
            # mutable source paths changed, refuse rather than trust them
            # (we already execute only the snapshot, but acceptance requires
            # explicit deferral when substituted inputs appear).
            source_binary = self._resolve_binary_path(bindings)
            source_keys = self._resolve_key_paths(bindings)
            if source_binary is not None and source_binary.is_file():
                try:
                    live_digest = _sha256_file_bytes(source_binary.read_bytes())
                    if live_digest != session.binary_sha256:
                        session.close()
                        self._last_session = None
                        return self._deferred_material(
                            "post_binding_binary_replacement"
                        )
                except OSError:
                    session.close()
                    self._last_session = None
                    return self._deferred_material("post_binding_binary_unreadable")
            if source_keys is not None:
                try:
                    live_pk = _sha256_file_bytes(source_keys[0].read_bytes())
                    live_vk = _sha256_file_bytes(source_keys[1].read_bytes())
                    if (
                        live_pk != session.proving_key_sha256
                        or live_vk != session.verifying_key_sha256
                    ):
                        session.close()
                        self._last_session = None
                        return self._deferred_material(
                            "post_binding_key_replacement"
                        )
                except OSError:
                    session.close()
                    self._last_session = None
                    return self._deferred_material("post_binding_key_unreadable")

            # Ambient artifacts root injection must not redirect execution.
            ambient_root = str(
                self._env_view().get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "") or ""
            ).strip()
            if ambient_root:
                try:
                    ambient_resolved = str(Path(ambient_root).resolve())
                    pinned_resolved = str(session.artifacts_root.resolve())
                    if ambient_resolved != pinned_resolved:
                        # Allowed only when ambient equals the original
                        # reviewed root; the session still overwrites child env.
                        reviewed = str(
                            getattr(bindings, "artifacts_root", "") or ""
                        )
                        if reviewed and ambient_resolved != str(
                            Path(reviewed).resolve()
                        ):
                            # Still proceed with overwrite — do not execute
                            # ambient.  Record diagnostic via reason only if
                            # we later fail; execution uses session root.
                            pass
                except (OSError, RuntimeError, ValueError):
                    pass

            try:
                raw = issue(
                    request,
                    local_witness=local_witness,
                    local_receipt=local_receipt,
                )
            except Exception:
                session.close()
                self._last_session = None
                # Never surface exception text (may contain paths/secrets).
                return self._deferred_material("issuer_exception")

            # Dispose of process-local witness reference as soon as possible.
            local_witness = None
            local_receipt = None

            status_text = str(getattr(raw, "status", "") or "").lower()
            if raw is None:
                session.close()
                self._last_session = None
                return self._deferred_material("issuer_unavailable")

            # Typed deferred/rejected dispositions from the datasets provider.
            if status_text in {
                "deferred",
                "certificate_deferred",
                "rejected",
                "certificate_rejected",
                "disabled",
            } or getattr(raw, "deferred", None) is True:
                reason = str(
                    getattr(getattr(raw, "reason", None), "value", None)
                    or getattr(raw, "reason", None)
                    or status_text
                    or "certificate_deferred"
                )
                session.close()
                self._last_session = None
                status = (
                    "certificate_rejected"
                    if "reject" in status_text
                    else "certificate_deferred"
                )
                return self._deferred_material(reason[:96], status=status)

            bindings_payload = self._bindings_public_payload(bindings)
            admitted, reject_reason = admit_proof_bearing_issuance_material(
                raw,
                expected_circuit_cid=str(
                    getattr(bindings, "circuit_cid", "") or ""
                ),
                expected_verifying_key_cid=str(
                    getattr(bindings, "verifying_key_cid", "") or ""
                ),
            )
            if admitted is None:
                session.close()
                self._last_session = None
                return self._deferred_material(
                    reject_reason or "provider_output_rejected",
                    status="certificate_rejected",
                )

            # Attach reviewed bindings (public digests/CIDs only).
            material = ProofBearingIssuanceMaterial(
                certificate=dict(admitted.certificate),
                proof_digest=admitted.proof_digest,
                proof_artifact_cid=admitted.proof_artifact_cid,
                circuit_cid=admitted.circuit_cid,
                verifying_key_cid=admitted.verifying_key_cid,
                artifact_bindings=bindings_payload,
                proof_json=dict(admitted.proof_json),
                backend_circuit_version=admitted.backend_circuit_version,
                verified_locally=bool(admitted.verified_locally),
            )
            self._last_material = material
            self._last_reason = "issued"
            # Keep session open only for the duration of this call; close after
            # successful material extraction so secrets/key snapshots do not
            # linger beyond the issuance boundary.
            session.close()
            self._last_session = None
            return material
        except Exception:
            if self._last_session is not None:
                try:
                    self._last_session.close()
                except Exception:
                    pass
                self._last_session = None
            return self._deferred_material("issue_material_exception")

    def issue(self, request: Any) -> Any:
        """Lightweight publication-path disposition; never raises into pytest.

        PTR-153 preserves public material via :meth:`issue_material`.  Controller
        publication authority remains deferred until PTR-155 joins material with
        controller-owned context under exact local V2 verification.  This method
        therefore returns a typed deferral without constructing an unsafe
        mutable-path provider for the publication path.
        """

        # Do not provision or construct the datasets provider on the
        # publication-path entry.  issue_material owns hardened prove/verify.
        self._last_reason = "positive_v4_issuance_pending_ptr155"
        return IssuedMaterialDisposition(
            status="certificate_deferred",
            reason=self._last_reason,
        )

    issue_deferred = issue
    __call__ = issue


# Live composition / capability probe interfaces (PTR-149).  These never claim
# readiness from source-symbol inventory or cold dependency-plan constants.
LIVE_TYPED_SERVICE_PROBE_INTERFACE: Final = "ProofReuseLiveTypedServiceProbe@1"
NATIVE_GROTH16_READINESS_PROBE_INTERFACE: Final = (
    "ProofReuseNativeGroth16ReadinessProbe@1"
)
TEST_CERTIFICATE_AUTHORITY_PROBE_INTERFACE: Final = (
    "ProofReuseTestCertificateAuthorityProbe@1"
)
# Pre-PTR-144 generic circuit family — never certificate-authority alone.
_GENERIC_KNOWLEDGE_OF_AXIOMS_CIRCUIT_PREFIX: Final = "knowledge_of_axioms@"
_TEST_PASS_ARTIFACT_VERSION: Final = 4


def _service_handle_probe(value: Any) -> dict[str, Any]:
    """Describe one already-composed service handle without importing packages."""

    present = value is not None
    interface = ""
    type_name = ""
    if present:
        interface = str(getattr(value, "interface", "") or "")[:96]
        type_name = type(value).__name__[:96]
    return {
        "present": present,
        "interface": interface,
        "type_name": type_name,
    }


def probe_live_typed_services(
    services: Any,
    *,
    source: str = "defaults",
) -> dict[str, Any]:
    """Report which typed default services are actually composed in-process.

    Pure attribute inspection: no import, install, network, or process start.
    Source-symbol presence is never treated as availability.
    """

    if services is None:
        handles = {
            name: _service_handle_probe(None)
            for name in (
                "identity_services",
                "lookup",
                "store",
                "candidate_store",
                "provider",
                "issuer",
                "revalidator",
                "current_context_provider",
                "resolver",
            )
        }
        return {
            "interface": LIVE_TYPED_SERVICE_PROBE_INTERFACE,
            "source": str(source or "missing")[:32],
            "degraded": True,
            "reason_code": "services_missing",
            "handles": handles,
            "required_handles_present": False,
            "ordinary_default_composition_usable": False,
            "network_attempted": False,
            "install_attempted": False,
            "import_for_readiness": False,
        }

    handles = {
        "identity_services": _service_handle_probe(
            getattr(services, "identity_services", None)
        ),
        "lookup": _service_handle_probe(getattr(services, "lookup", None)),
        "store": _service_handle_probe(getattr(services, "store", None)),
        "candidate_store": _service_handle_probe(
            getattr(services, "candidate_store", None)
        ),
        "provider": _service_handle_probe(getattr(services, "provider", None)),
        "issuer": _service_handle_probe(getattr(services, "issuer", None)),
        "revalidator": _service_handle_probe(
            getattr(services, "revalidator", None)
        ),
        "current_context_provider": _service_handle_probe(
            getattr(services, "current_context_provider", None)
        ),
        "resolver": _service_handle_probe(getattr(services, "resolver", None)),
    }
    # Production default path needs these for ordinary cold/warm composition.
    required = (
        "identity_services",
        "lookup",
        "store",
        "candidate_store",
        "issuer",
        "revalidator",
        "current_context_provider",
    )
    required_present = all(handles[name]["present"] for name in required)
    degraded = bool(getattr(services, "degraded", False))
    reason = str(getattr(services, "reason_code", "") or "")
    return {
        "interface": LIVE_TYPED_SERVICE_PROBE_INTERFACE,
        "source": str(
            getattr(services, "source", None) or source or "defaults"
        )[:32],
        "degraded": degraded,
        "reason_code": reason[:96],
        "handles": handles,
        "required_handles_present": required_present,
        "ordinary_default_composition_usable": required_present and not degraded,
        "network_attempted": False,
        "install_attempted": False,
        "import_for_readiness": False,
    }


def probe_native_groth16_readiness(
    *,
    installer: Any = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bounded non-mutating native Groth16 installation/readiness probe.

    Never starts a cargo build, trusted setup, network call, or package install.
    Uses the lazy installer's consent=False path when an installer is supplied.
    """

    env = environ if environ is not None else os.environ
    diagnostics: dict[str, Any] = {
        "build_policy_enabled": groth16_build_enabled(env),
        "binary_env": str(env.get(DATASETS_GROTH16_BINARY_ENV, "") or "")[:256],
        "artifacts_env": str(
            env.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "") or ""
        )[:256],
        "circuit_ref": str(
            env.get(PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV, "") or ""
        )[:96],
        "endpoint_configured": bool(
            str(env.get(PROOF_REUSE_GROTH16_ENDPOINT_ENV, "") or "").strip()
        ),
    }
    circuit_ref = diagnostics["circuit_ref"]
    diagnostics["knowledge_of_axioms_circuit"] = circuit_ref.startswith(
        _GENERIC_KNOWLEDGE_OF_AXIOMS_CIRCUIT_PREFIX
    )

    installed = False
    ready = False
    reason = "installer_unavailable"
    runtime_status: dict[str, Any] = {}
    process_started = False
    network_attempted = False

    if installer is not None:
        # Prefer inspect_groth16_runtime (non-mutating aggregate).
        inspect_runtime = getattr(installer, "inspect_groth16_runtime", None)
        if callable(inspect_runtime):
            try:
                raw = inspect_runtime()
                if isinstance(raw, Mapping):
                    runtime_status = {
                        key: raw[key]
                        for key in (
                            "ready",
                            "readiness_scope",
                            "action",
                            "test_certificate_authority_ready",
                            "test_certificate_authority_reason",
                            "skip_authority",
                            "network_attempted",
                            "process_started",
                            "trusted_setup_attempted",
                        )
                        if key in raw
                    }
                    ready = bool(raw.get("ready"))
                    network_attempted = bool(raw.get("network_attempted"))
                    process_started = bool(raw.get("process_started"))
                    reason = "ready" if ready else "native_or_endpoint_unready"
            except Exception:
                reason = "runtime_inspect_failed"
        # Non-mutating native install probe (consent=False never builds).
        ensure = getattr(installer, "ensure_groth16_native_backend", None)
        if callable(ensure):
            try:
                try:
                    resolution = ensure(
                        consent=False,
                        required_circuit_version=TEST_PASS_GROTH16_CIRCUIT_VERSION,
                    )
                except TypeError:
                    resolution = ensure(consent=False)
                available = bool(getattr(resolution, "available", False))
                resolution_diagnostics = getattr(resolution, "diagnostics", {})
                if isinstance(resolution_diagnostics, Mapping):
                    process_started = process_started or bool(
                        resolution_diagnostics.get("process_started")
                    )
                installed = available
                if ready and not available:
                    runtime_status["generic_runtime_ready"] = True
                    ready = False
                    reason = "native_v4_capability_unready"
                if available and not ready:
                    # Binary present but runtime aggregate may still be unready.
                    reason = str(
                        getattr(resolution, "reason_code", "") or "native_present"
                    )[:96]
                elif not available:
                    reason = str(
                        getattr(resolution, "reason_code", "") or reason
                    )[:96]
            except Exception:
                reason = "native_probe_failed"
    else:
        # Filesystem-only binary existence when no installer is injected.
        binary = str(env.get(DATASETS_GROTH16_BINARY_ENV, "") or "").strip()
        if binary and Path(binary).is_file():
            installed = True
            ready = False  # binary alone never implies full runtime readiness
            reason = "binary_present_runtime_unconfirmed"
        else:
            reason = "installer_unavailable"

    return {
        "interface": NATIVE_GROTH16_READINESS_PROBE_INTERFACE,
        "installed": installed,
        "ready": ready,
        "reason_code": reason[:96],
        "readiness_scope": str(
            runtime_status.get("readiness_scope")
            or "generic_native_or_endpoint_capability"
        )[:96],
        "knowledge_of_axioms_circuit": bool(
            diagnostics["knowledge_of_axioms_circuit"]
        ),
        "skip_authority": False,  # native readiness never grants skip alone
        "required_circuit_version": TEST_PASS_GROTH16_CIRCUIT_VERSION,
        "network_attempted": network_attempted,
        "process_started": process_started,
        "install_attempted": False,
        "trusted_setup_attempted": bool(
            runtime_status.get("trusted_setup_attempted")
        ),
        "diagnostics": diagnostics,
        "runtime_status": runtime_status,
    }


def probe_test_certificate_authority(
    *,
    environ: Mapping[str, str] | None = None,
    artifacts_root: str | os.PathLike[str] | None = None,
    binary_path: str | os.PathLike[str] | None = None,
    installer: Any = None,
) -> dict[str, Any]:
    """Probe real test-pass certificate authority without proving or installing.

    Separates native Groth16 installation/readiness from certificate authority.
    The generic pre-PTR-144 ``knowledge_of_axioms`` backend alone can never
    satisfy this probe, even when native readiness is true.
    """

    env = environ if environ is not None else os.environ
    circuit_ref = str(env.get(PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV, "") or "").strip()
    knowledge_of_axioms = circuit_ref.startswith(
        _GENERIC_KNOWLEDGE_OF_AXIOMS_CIRCUIT_PREFIX
    )

    # Native aggregate may already refuse certificate authority (always for
    # knowledge_of_axioms-style generic readiness).
    native = probe_native_groth16_readiness(installer=installer, environ=env)

    bindings_payload: dict[str, Any] = {
        "provenance_ready": False,
        "reason_code": "artifact_probe_not_run",
        "circuit_cid": "",
        "verifying_key_cid": "",
        "backend_circuit_version": _TEST_PASS_ARTIFACT_VERSION,
    }
    try:
        from .publication import Groth16ArtifactIdentityBindings

        bindings = Groth16ArtifactIdentityBindings.from_activated_artifacts(
            artifacts_root=artifacts_root,
            environ=env,
            binary_path=binary_path,
            circuit_version=_TEST_PASS_ARTIFACT_VERSION,
        )
        bindings_payload = {
            "provenance_ready": bool(bindings.provenance_ready),
            "reason_code": str(bindings.reason_code or "")[:96],
            "circuit_cid": str(bindings.circuit_cid or "")[:128],
            "verifying_key_cid": str(bindings.verifying_key_cid or "")[:128],
            "backend_circuit_version": int(bindings.backend_circuit_version),
            "artifacts_root": str(bindings.artifacts_root or "")[:256],
            "process_started": bool(
                getattr(bindings, "diagnostics", {}).get("process_started", False)
            ),
        }
    except Exception as exc:
        bindings_payload["reason_code"] = f"binding_probe_failed:{type(exc).__name__}"[
            :96
        ]

    # Authority requires exact test-pass artifact provenance.  Never promote
    # knowledge_of_axioms, generic native readiness, or an unmanifested binary
    # to certificate authority.
    ready = bool(bindings_payload.get("provenance_ready"))
    reason = str(bindings_payload.get("reason_code") or "unready")
    unmanifested_binary = bool(
        native.get("installed") and not bindings_payload.get("provenance_ready")
    )
    if knowledge_of_axioms and not ready:
        reason = "knowledge_of_axioms_cannot_satisfy_test_certificate_authority"
    elif knowledge_of_axioms and ready:
        # Even with co-located keys, a knowledge_of_axioms circuit binding is
        # not the test-pass ruleset; refuse certificate authority.
        ready = False
        reason = "knowledge_of_axioms_cannot_satisfy_test_certificate_authority"
    elif not ready and native.get("ready") and not bindings_payload.get(
        "provenance_ready"
    ):
        reason = "native_ready_without_test_pass_provenance"
    elif not ready and unmanifested_binary:
        # Binary presence without an approved reviewed key/manifest is never
        # certificate authority.
        if reason in {"unready", "artifact_probe_not_run", "installer_unavailable"}:
            reason = "unmanifested_native_binary"

    return {
        "interface": TEST_CERTIFICATE_AUTHORITY_PROBE_INTERFACE,
        "ready": ready,
        "reason_code": reason[:96],
        "skip_authority": ready,  # only exact test-pass provenance may skip
        "knowledge_of_axioms_rejected": knowledge_of_axioms,
        "knowledge_of_axioms_circuit": knowledge_of_axioms,
        "unmanifested_native_binary_rejected": unmanifested_binary and not ready,
        "native_groth16_ready": bool(native.get("ready")),
        "native_groth16_installed": bool(native.get("installed")),
        "artifact_bindings": bindings_payload,
        "network_attempted": False,
        "process_started": bool(bindings_payload.get("process_started", False)),
        "install_attempted": False,
        "import_for_readiness": False,
        "prove_attempted": False,
    }


def _activation_gap_from_certificate(
    certificate: Mapping[str, Any],
    *,
    native: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit activation-gap packet when reviewed authority artifacts are absent.

    Missing operator-provided reviewed v4 keys or a trusted-setup/key manifest
    is never warm-skip or closeout authority.  Tests and the supervisor continue
    under typed RUN/DEFERRED; this packet only surfaces the gap truthfully.
    """

    ready = bool(certificate.get("ready"))
    reason = str(certificate.get("reason_code") or "")[:96]
    bindings = certificate.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        bindings = {}
    native_installed = bool(
        (native or {}).get("installed")
        if native is not None
        else certificate.get("native_groth16_installed")
    )
    gap_reasons = {
        "artifact_manifest_unapproved",
        "artifact_manifest_missing",
        "artifact_manifest_digest_mismatch",
        "artifacts_root_missing",
        "test_pass_keys_missing",
        "artifact_read_failed",
        "artifact_probe_not_run",
        "native_ready_without_test_pass_provenance",
        "knowledge_of_axioms_cannot_satisfy_test_certificate_authority",
        "binary_alone_non_authoritative",
        "unmanifested_native_binary",
    }
    # Unmanifested native binary alone can never close the gap.
    unmanifested_binary = bool(
        native_installed
        and not ready
        and not bindings.get("provenance_ready")
    )
    present = (not ready) and (
        reason in gap_reasons
        or unmanifested_binary
        or not bindings.get("provenance_ready")
    )
    gap_reason = reason
    if present and unmanifested_binary and reason not in gap_reasons:
        gap_reason = "unmanifested_native_binary_cannot_satisfy_test_certificate_authority"
    elif present and not gap_reason:
        gap_reason = "reviewed_v4_keys_or_manifest_absent"
    return {
        "present": present,
        "reason_code": gap_reason[:96] if present else "",
        "warm_skip_authorized": False,
        "closeout_authorized": False,
        "tests_continue": True,
        "reviewed_v4_keys_or_manifest_required": True,
        "native_binary_alone_non_authoritative": True,
        "knowledge_of_axioms_cannot_satisfy": True,
    }


def live_runtime_activation_inventory(
    services: Any,
    *,
    installer: Any = None,
    environ: Mapping[str, str] | None = None,
    artifacts_root: str | os.PathLike[str] | None = None,
    binary_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Derive runtime-activation inventory from live handles and probes.

    Replaces hard-coded cold dependency-plan booleans for readiness claims.
    """

    composition = probe_live_typed_services(services)
    handles = composition["handles"]
    native = probe_native_groth16_readiness(installer=installer, environ=environ)
    certificate = probe_test_certificate_authority(
        environ=environ,
        artifacts_root=artifacts_root,
        binary_path=binary_path,
        installer=installer,
    )
    activation_gap = _activation_gap_from_certificate(
        certificate, native=native
    )

    identity_configured = handles["identity_services"]["present"]
    candidate_store_configured = handles["candidate_store"]["present"]
    revalidator_configured = handles["revalidator"]["present"]
    two_stage = handles["lookup"]["present"] and revalidator_configured
    issuer_configured = handles["issuer"]["present"]
    current_context = handles["current_context_provider"]["present"]
    store_configured = handles["store"]["present"]

    blockers: list[str] = []
    if not identity_configured:
        blockers.append("identity_services_unconfigured")
    if not candidate_store_configured:
        blockers.append("candidate_store_unconfigured")
    if not revalidator_configured:
        blockers.append("revalidator_unconfigured")
    if not two_stage:
        blockers.append("two_stage_lookup_unconfigured")
    if not current_context:
        blockers.append("current_context_provider_unconfigured")
    if not issuer_configured:
        blockers.append("issuer_unconfigured")
    if not store_configured:
        blockers.append("certificate_store_unconfigured")
    if not certificate["ready"]:
        blockers.append("test_certificate_authority_unready")
    if activation_gap["present"]:
        blockers.append("activation_gap_reviewed_authority_absent")

    ordinary_warm = (
        identity_configured
        and candidate_store_configured
        and revalidator_configured
        and two_stage
        and current_context
        and store_configured
        and issuer_configured
    )
    # Warm skip and completion authority require exact certificate authority;
    # an activation gap can never invent either.
    warm_complete = (
        ordinary_warm
        and bool(certificate["ready"])
        and not activation_gap["present"]
    )

    return {
        "automatic_plugin_discovery": True,
        "ordinary_enabled_run_effective_action": "run",
        "default_identity_services_injected": False,
        "default_identity_service_factory_configured": identity_configured,
        "production_identity_injector_configured": identity_configured,
        "required_identity_providers": [
            "repository_forest_provider",
            "analysis_index_provider",
            "component_inputs_provider",
            "policy_inputs_provider",
            "runtime_evidence_provider",
        ],
        "default_identity_compiler_available": identity_configured,
        "candidate_context_store_configured": candidate_store_configured,
        "two_stage_candidate_revalidation_configured": two_stage,
        "lookup_requires_exact_execution_key_before_candidate_read": True,
        "runtime_trace_attribute_producer_configured": ordinary_warm,
        "post_pass_runtime_trace_capture_configured": ordinary_warm,
        "post_pass_receipt_requires_runtime_trace": ordinary_warm,
        "deferred_request_builder_configured": issuer_configured,
        "deferred_request_transport_compatible": issuer_configured,
        "deferred_certificate_issuer_configured": issuer_configured,
        "issuer_in_lazy_service_bundle": issuer_configured,
        "issuer_in_lazy_service_resolution": issuer_configured,
        "candidate_certificate_publication_configured": (
            candidate_store_configured and store_configured
        ),
        "authoritative_candidate_publication_configured": (
            candidate_store_configured and store_configured and issuer_configured
        ),
        "ordinary_warm_skip_path_complete": warm_complete,
        "missing_provider_action": "run",
        "completion_authority": False,
        "native_groth16_installed": native["installed"],
        "native_groth16_ready": native["ready"],
        "test_certificate_authority_ready": certificate["ready"],
        "test_certificate_authority_reason": certificate["reason_code"],
        "knowledge_of_axioms_cannot_satisfy_test_certificate_authority": True,
        "unmanifested_native_binary_cannot_satisfy_test_certificate_authority": True,
        "activation_gap": activation_gap,
        "activation_gap_present": bool(activation_gap["present"]),
        "activation_blocker_codes": blockers,
        "live_probe": True,
        "network_attempted": False,
        "install_attempted": False,
        "import_for_readiness": False,
        "composition": composition,
        "native_groth16": native,
        "test_certificate_authority": certificate,
    }


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

    def probe_live_composition(self) -> dict[str, Any]:
        """Non-mutating probe of which typed handles this bundle holds."""

        return probe_live_typed_services(self, source=self.source)

    def live_runtime_activation_inventory(
        self,
        *,
        installer: Any = None,
        environ: Mapping[str, str] | None = None,
        artifacts_root: str | os.PathLike[str] | None = None,
        binary_path: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """Live activation inventory derived from this bundle and local probes."""

        return live_runtime_activation_inventory(
            self,
            installer=installer,
            environ=environ,
            artifacts_root=artifacts_root,
            binary_path=binary_path,
        )

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
    effective_installer = (
        installer if proof_reuse_install_permitted(env) else None
    )

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
                resolved_resolver = LazyProofReuseServiceResolver(
                    installer=effective_installer
                )
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
            # injection tests only when it already carries the dedicated
            # candidate-context store, or when no dedicated store exists.
            # Production composition must wire the candidate-context store
            # into two-stage lookup (never leave stage-1 store detached).
            resolver_lookup = resolution.lookup
            resolver_has_context = (
                resolver_lookup is not None
                and getattr(resolver_lookup, "candidate_context_store", None)
                is not None
            )
            if resolver_has_context or resolved_candidate_store is None:
                resolved_lookup = resolver_lookup
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
            installer=effective_installer,
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
    "ISSUED_MATERIAL_DISPOSITION_INTERFACE",
    "ImmutableNativeArtifactSession",
    "IssuedMaterialDisposition",
    "LAZY_REAL_TEST_CERTIFICATE_ISSUER_INTERFACE",
    "LIVE_TYPED_SERVICE_PROBE_INTERFACE",
    "LOOKUP_MODULE",
    "JSONSCHEMA_DEPENDENCY",
    "JSONSCHEMA_MODULE",
    "LazyProofReuseServiceResolver",
    "LazyRealTestCertificateIssuer",
    "MAX_ISSUED_CERTIFICATE_BYTES",
    "MAX_ISSUED_MATERIAL_JSON_BYTES",
    "MAX_ISSUED_PROOF_BYTES",
    "MULTIFORMATS_DEPENDENCY",
    "MULTIFORMATS_MODULE",
    "NATIVE_GROTH16_READINESS_PROBE_INTERFACE",
    "NLTK_DATA_RESOURCE_ALLOWLIST",
    "NLTK_DEPENDENCY",
    "NLTK_MODULE",
    "NltkDataResource",
    "PROOF_BEARING_ISSUANCE_MATERIAL_INTERFACE",
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
    "ProofBearingIssuanceMaterial",
    "ProofReuseDependency",
    "ProofReuseServiceResolution",
    "STORE_MODULE",
    "TEST_CERTIFICATE_AUTHORITY_PROBE_INTERFACE",
    "admit_proof_bearing_issuance_material",
    "allowlisted_native_child_environment",
    "automatic_install_enabled",
    "compose_default_proof_reuse_services",
    "groth16_build_enabled",
    "live_runtime_activation_inventory",
    "nltk_data_download_enabled",
    "probe_live_typed_services",
    "probe_native_groth16_readiness",
    "probe_test_certificate_authority",
    "proof_reuse_dependency_plan",
    "redact_private_material_fields",
]
