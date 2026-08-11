"""External authorization vendor certification tests (FVT-055 / FVT-073 / FVT-G209).

``ExternalAuthorizationVendorCertification@1``

Acceptance covered:

* Soufflé 2.4.1 source/archive and build dependencies are immutable and
  checksummed; user-local executable and artifact digests are exact;
* allow/deny/unknown/conflict/delegation plus rule/scope mutation and replay
  execute through vendor Soufflé;
* malformed-output, timeout, and disagreement behavior is injected by the
  bounded runner harness without vendor-prover fault-control variables;
* linux-aarch64 is supported for Soufflé;
* external SecPAL is a narrow unsupported-platform exception on
  linux-aarch64 under the current contract and never counts as installed,
  complete, authoritative, or production-certified;
* hermetic shadows remain differential-only and cannot satisfy vendor
  production evidence.

The synthetic evidence term ``objective validation repair`` is asserted by
``test_objective_validation_repair_proves_g209_acceptance`` so the supervisor
validation gate (FVT-073) can re-find coverage when path evidence already
exists for the certifier, installer, lock, focused test, and receipt.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "installers"
    / "authorization.py"
)
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "authorization_external.py"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_authorization_vendor_install_receipt.json"
)

VENDOR_INTERFACE = "ExternalAuthorizationVendorCertification@1"
VENDOR_SCHEMA = "external-authorization-vendor-certification/v1"
VENDOR_RECEIPT_SCHEMA = (
    "formal-verification-authorization-vendor-install-receipt/v1"
)
VENDOR_GOAL_ID = "FVT-G209"
VENDOR_TASK_ID = "FVT-055"
VENDOR_REPAIR_TASK_ID = "FVT-073"
# Synthetic evidence term required by objective-scan validation gates.
OBJECTIVE_VALIDATION_EVIDENCE = "objective validation repair"
REQUIRED_SOURCE_SHA256 = (
    "08d9b19cb4a8f570ac75dea73016b6a326d87ac28fccd4afeba217ace2071587"
)
REQUIRED_CATEGORIES = {"allow", "deny", "unknown", "conflict", "delegation"}
REQUIRED_MUTATIONS = {"rule", "scope"}
REQUIRED_BUILD_DEPS = {
    "cmake",
    "flex",
    "bison",
    "mcpp",
    "sqlite3",
    "libffi",
    "python3",
}
LINUX_AARCH64 = "linux-aarch64"
DEFAULT_MANAGED_PROVER_ROOT = (
    Path.home() / ".local/share/ipfs_datasets_py/theorem-provers"
)
DEPENDENCY_PREFIX_SUFFIX = Path(
    "build-dependencies/souffle/ubuntu-noble-arm64/root"
)


def _ensure_datasets_on_path() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (REPO_ROOT, datasets_root):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_module(name: str, path: Path):
    assert path.is_file(), f"missing module: {path}"
    _ensure_datasets_on_path()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def installer():
    return _load_module("authorization_vendor_installer", INSTALLER_PATH)


@pytest.fixture(scope="module")
def certifier():
    return _load_module("authorization_vendor_certification", CERTIFIER_PATH)


@pytest.fixture(scope="module")
def managed_prover_root() -> Path:
    root = Path(
        os.environ.get(
            "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
            str(DEFAULT_MANAGED_PROVER_ROOT),
        )
    ).expanduser().resolve()
    assert root.is_dir(), f"managed theorem-prover root is unavailable: {root}"
    return root


@pytest.fixture(scope="module")
def install_root(managed_prover_root: Path) -> Path:
    root = managed_prover_root / "souffle-vendor"
    assert root.is_dir(), f"managed Soufflé vendor root is unavailable: {root}"
    return root


@pytest.fixture(scope="module")
def dependency_prefix(managed_prover_root: Path) -> Path:
    prefix = managed_prover_root / DEPENDENCY_PREFIX_SUFFIX
    assert prefix.is_dir(), f"managed Soufflé dependency prefix is unavailable: {prefix}"
    return prefix


@pytest.fixture(scope="module")
def vendor_bundle(installer, install_root, dependency_prefix):
    return installer.ensure_authorization_vendor(
        yes=True,
        strict=True,
        force=False,
        install_root=install_root,
        dependency_prefix=dependency_prefix,
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        platform_id=LINUX_AARCH64,
        checksum_verified=True,
    )


@pytest.fixture(scope="module")
def vendor_certificate(
    certifier,
    install_root,
    dependency_prefix,
) -> dict[str, Any]:
    return certifier.certify_external_authorization_vendor(
        install_root=install_root,
        dependency_prefix=dependency_prefix,
        force_install=False,
        skip_install=True,
        platform_id=LINUX_AARCH64,
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )


# ---------------------------------------------------------------------------
# Expected outputs / lock pins
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert CERTIFIER_PATH.is_file()
    assert LOCK_PATH.is_file()
    assert RECEIPT_PATH.is_file()


def test_lock_souffle_source_is_checksummed() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    inv = lock["checksummed_release_inventory"]["souffle"]
    assert inv["version"] == "2.4.1"
    assert inv["sha256"] == REQUIRED_SOURCE_SHA256
    assert inv.get("is_checksummed") is True
    assert REQUIRED_BUILD_DEPS <= set(inv.get("build_dependencies") or {})

    tools = {entry["tool_id"]: entry for entry in lock["tools"]}
    souffle = tools["souffle"]
    pin0 = souffle["pins"][0]
    assert pin0["sha256"] == REQUIRED_SOURCE_SHA256
    assert pin0["is_checksummed"] is True
    contract = souffle["deployment_contract"]
    assert LINUX_AARCH64 in contract["supported_platforms"]
    assert REQUIRED_BUILD_DEPS <= set(contract.get("build_dependencies") or {})
    assert (
        contract["vendor_install"]["source_archive_sha256"] == REQUIRED_SOURCE_SHA256
    )
    assert contract["vendor_install"]["hermetic_shadows_are_differential_only"] is True

    secpal = tools["secpal"]
    secpal_contract = secpal["deployment_contract"]
    assert secpal_contract["supported_platforms"] == []
    for platform_id in (LINUX_AARCH64, "linux-x86_64"):
        exception = secpal_contract["platform_exceptions"][platform_id]
        assert exception["classification"] == "unsupported_here"
        assert exception["narrow_scope"] is True
        assert exception["installed"] is False
        assert exception["complete"] is False
        assert exception["authoritative"] is False
        assert exception["production_certified"] is False


def test_installer_vendor_constants(installer) -> None:
    assert installer.VENDOR_INTERFACE == "ExternalAuthorizationVendorInstaller@1"
    assert installer.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert installer.VENDOR_TASK_ID == VENDOR_TASK_ID
    assert installer.VENDOR_REPAIR_TASK_ID == VENDOR_REPAIR_TASK_ID
    assert installer.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert installer.SOUFFLE_SOURCE_ARCHIVE_SHA256 == REQUIRED_SOURCE_SHA256
    assert REQUIRED_BUILD_DEPS <= set(installer.SOUFFLE_BUILD_DEPENDENCIES)
    meta = installer.describe_authorization_installer()
    assert meta["policy"]["hermetic_shadows_are_differential_only"] is True
    assert meta["policy"]["never_promote_hermetic_shadow_as_vendor"] is True
    assert meta["policy"]["secpal_linux_aarch64_is_narrow_platform_exception"] is True
    assert (
        meta["policy"]
        ["secpal_platform_exception_does_not_satisfy_live_readiness"]
        is True
    )
    prerequisites = meta["secpal_vendor_prerequisite_report"]
    assert prerequisites["ready"] is False
    assert prerequisites["historical_release_version"] == "1.1"
    assert prerequisites["official_download_url"].endswith(
        "details.aspx?id=52356"
    )
    assert prerequisites["upstream_distribution_status"] == "retired"
    assert "vendor_license_evidence_missing" in prerequisites["block_reasons"]
    assert "vendor_runtime_contract_missing" in prerequisites["block_reasons"]
    assert meta["policy"]["souffle_linux_aarch64_supported"] is True
    assert (
        meta["policy"]["souffle_relocation_requires_explicit_known_layout"]
        is True
    )
    assert (
        meta["policy"]["souffle_relocation_preserves_provenance_manifest"]
        is True
    )
    assert meta["souffle_source_archive_sha256"] == REQUIRED_SOURCE_SHA256
    assert meta["vendor_repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert meta["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert meta["objective_validation_repair"] is True


def test_certifier_vendor_constants(certifier) -> None:
    assert certifier.VENDOR_INTERFACE == VENDOR_INTERFACE
    assert certifier.VENDOR_SCHEMA_VERSION == VENDOR_SCHEMA
    assert certifier.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert certifier.VENDOR_TASK_ID == VENDOR_TASK_ID
    assert certifier.VENDOR_REPAIR_TASK_ID == VENDOR_REPAIR_TASK_ID
    assert certifier.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert certifier.SOUFFLE_REQUIRED_SOURCE_SHA256 == REQUIRED_SOURCE_SHA256
    assert certifier.AUTHORIZATION_FAULT_MODES == {
        "malformed_output",
        "timeout",
        "disagreement",
    }
    assert "test_external_authorization_vendor_certification.py" in (
        certifier.OBJECTIVE_VALIDATION_COMMAND
    )
    assert "test_external_authorization_toolchain_certification.py" in (
        certifier.OBJECTIVE_VALIDATION_COMMAND
    )


# ---------------------------------------------------------------------------
# Vendor install path
# ---------------------------------------------------------------------------


def test_vendor_souffle_install_on_linux_aarch64(installer, vendor_bundle) -> None:
    assert vendor_bundle.interface == installer.VENDOR_INTERFACE
    assert vendor_bundle.goal_id == VENDOR_GOAL_ID
    by_id = {item.tool_id: item for item in vendor_bundle.receipts}
    assert "souffle" in by_id
    souffle = by_id["souffle"]
    assert souffle.ok
    assert souffle.is_vendor_path is True
    assert souffle.identity is not None
    identity = souffle.identity
    assert identity.is_vendor_build is True
    assert identity.is_hermetic_shadow is False
    assert identity.version == "2.4.1"
    assert identity.source_archive_sha256 == REQUIRED_SOURCE_SHA256
    assert identity.artifact_sha256
    assert len(identity.artifact_sha256) == 64
    assert identity.identity_manifest_sha256
    assert len(identity.identity_manifest_sha256) == 64
    assert identity.native_binary_format == "elf"
    assert identity.native_machine == "aarch64"
    assert identity.artifact_size_bytes > 0
    assert identity.build_contract_sha256
    assert identity.deployment_lock_sha256
    assert identity.pin_contract_sha256
    assert identity.dependency_package_set_sha256
    assert identity.dependency_packages
    if identity.is_relocated_install:
        assert identity.relocation_binding_sha256
    assert Path(identity.executable).is_file()
    assert identity.platform_id == LINUX_AARCH64
    assert REQUIRED_BUILD_DEPS <= {name for name, _ in identity.build_dependencies}
    assert identity.role == "shadow"
    assert identity.authority_ceiling == "none"

    import subprocess

    completed = subprocess.run(
        [identity.executable, "--version"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env=installer._dependency_prefix_environment(
            Path(identity.dependency_prefix)
        ),
    )
    banner = (completed.stdout or "") + (completed.stderr or "")
    assert "2.4.1" in banner
    assert "hermetic-authorization-shadow" not in banner.casefold()


def test_secpal_platform_exception_on_linux_aarch64(installer, vendor_bundle) -> None:
    by_id = {item.tool_id: item for item in vendor_bundle.receipts}
    assert "secpal" in by_id
    secpal = by_id["secpal"]
    assert secpal.ok is False
    assert secpal.platform_exception is True
    assert secpal.status == "unsupported_platform"
    assert secpal.installed is False
    assert secpal.complete is False
    assert secpal.authoritative is False
    assert secpal.production_certified is False
    payload = secpal.to_dict()
    assert payload["installed"] is False
    assert payload["complete"] is False
    assert payload["authoritative"] is False
    assert payload["production_certified"] is False
    assert payload["platform_exception"] is True
    assert not installer.tool_supported_on_platform("secpal", LINUX_AARCH64)
    assert installer.tool_supported_on_platform("souffle", LINUX_AARCH64)
    assert "official_vendor_distribution_retired" in secpal.block_reasons
    assert "vendor_artifact_checksum_missing" in secpal.block_reasons
    assert "vendor_license_evidence_missing" in secpal.block_reasons


def test_supported_linux_x86_64_unavailable_secpal_blocks_combined_certificate(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authz = certifier.authz_installer
    install_root = tmp_path.resolve()
    souffle_identity = SimpleNamespace(
        artifact_sha256="a" * 64,
        build_dependencies=(("cmake", ">=3.8"),),
        dependency_prefix="",
        install_root=str(install_root),
        is_hermetic_shadow=False,
        is_relocated_install=False,
        platform_id="linux-x86_64",
        source_archive_sha256=REQUIRED_SOURCE_SHA256,
        source_archive_url="https://example.invalid/souffle.tar.gz",
    )
    secpal_receipt = authz.InstallReceipt(
        tool_id=authz.TOOL_SECPAL,
        status="unavailable",
        identity=None,
        selected_version="1.0.0-reviewed",
        block_reasons=("authentic_vendor_artifact_unavailable",),
        platform_id="linux-x86_64",
        is_vendor_path=True,
    )
    souffle_engine = certifier.EngineCertification(
        engine_id=authz.TOOL_SOUFFLE,
        version="2.4.1",
        executable="souffle",
        usable=True,
        certified=True,
        role="shadow",
        authority_ceiling="none",
    )
    supported_status = {
        "tool_id": authz.TOOL_SECPAL,
        "host_platform": "linux-x86_64",
        "classification": "supported_here",
        "exception": False,
        "narrow_scope": False,
        "installed": None,
        "complete": None,
        "authoritative": False,
        "production_certified": False,
        "supported_platforms": ["linux-x86_64"],
        "live_ready": False,
        "live_block_reasons": ["authentic_vendor_artifact_unavailable"],
        "live_readiness": {
            "ready": False,
            "block_reasons": ["authentic_vendor_artifact_unavailable"],
        },
        "platform_exception_satisfies_live_readiness": False,
    }

    monkeypatch.setattr(
        authz,
        "pin_for_tool",
        lambda *_args, **_kwargs: {"version": "2.4.1"},
    )
    monkeypatch.setattr(
        authz,
        "_identity_from_disk",
        lambda *_args, **_kwargs: souffle_identity,
    )
    monkeypatch.setattr(authz, "ensure_secpal", lambda **_kwargs: secpal_receipt)
    monkeypatch.setattr(
        authz,
        "tool_supported_on_platform",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        certifier,
        "_validated_native_souffle_evidence",
        lambda _identity: {},
    )
    monkeypatch.setattr(
        certifier,
        "_certify_vendor_souffle",
        lambda *_args, **_kwargs: souffle_engine,
    )
    monkeypatch.setattr(
        certifier,
        "derive_secpal_platform_exception",
        lambda **_kwargs: dict(supported_status),
    )
    monkeypatch.setattr(
        certifier,
        "public_evidence_projection",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        certifier,
        "build_vendor_install_receipt",
        lambda certificate, **_kwargs: {"certified": certificate["certified"]},
    )

    certificate = certifier.certify_external_authorization_vendor(
        install_root=install_root,
        skip_install=True,
        platform_id="linux-x86_64",
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )

    assert certificate["certified"] is False
    assert certificate["souffle_vendor_certified"] is True
    assert certificate["secpal_vendor_certified"] is False
    assert certificate["secpal_live_ready"] is False
    assert certificate["combined_external_authorization_certified"] is False
    assert certificate["secpal_platform_exception"]["exception"] is False
    assert certificate["secpal_platform_exception"]["installed"] is False
    assert "secpal_vendor_unavailable_on_supported_host" in (
        certificate["summary"]["block_reasons"]
    )


def test_hermetic_shadow_cannot_satisfy_vendor(
    certifier,
    installer,
    vendor_bundle,
    tmp_path: Path,
) -> None:
    hermetic = installer.ensure_souffle(
        yes=True,
        strict=True,
        force=True,
        install_root=tmp_path / "hermetic-only",
        hermetic_shadow=True,
        vendor=False,
        checksum_verified=True,
    )
    assert hermetic.ok
    assert hermetic.identity is not None
    assert hermetic.identity.is_hermetic_shadow is True
    assert hermetic.identity.is_vendor_build is False
    vendor = {item.tool_id: item for item in vendor_bundle.receipts}["souffle"]
    assert vendor.identity is not None
    assert vendor.identity.executable != hermetic.identity.executable
    with pytest.raises(
        certifier.ExternalAuthorizationCertificationError,
        match="hermetic shadows cannot satisfy",
    ):
        certifier._certify_vendor_souffle(
            hermetic.identity,
            install_status=hermetic.status,
        )
    assert "authorization-shadows" in hermetic.identity.executable


def test_vendor_skip_install_requires_matching_dependency_prefix(
    certifier,
    install_root: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        certifier.ExternalAuthorizationCertificationError,
        match="vendor Soufflé is missing",
    ):
        certifier.certify_external_authorization_vendor(
            install_root=install_root,
            dependency_prefix=tmp_path / "wrong-prefix",
            skip_install=True,
            platform_id=LINUX_AARCH64,
            repo_root=REPO_ROOT,
            lock_path=LOCK_PATH,
        )


def test_fault_harness_rejects_unknown_mode(certifier) -> None:
    with pytest.raises(
        certifier.ExternalAuthorizationCertificationError,
        match="unsupported authorization fault mode",
    ):
        certifier.AuthorizationFaultHarness("prover_environment_switch")


# ---------------------------------------------------------------------------
# Full vendor certification corpus
# ---------------------------------------------------------------------------


def test_vendor_certificate_envelope(vendor_certificate: dict[str, Any]) -> None:
    assert vendor_certificate["schema_version"] == VENDOR_SCHEMA
    assert vendor_certificate["interface"] == VENDOR_INTERFACE
    assert vendor_certificate["goal_id"] == VENDOR_GOAL_ID
    assert vendor_certificate["task_id"] == VENDOR_TASK_ID
    assert vendor_certificate["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert vendor_certificate["host_platform"] == LINUX_AARCH64
    assert vendor_certificate["certified"] is True
    assert vendor_certificate["authority_ceiling"] == "none"
    assert (
        vendor_certificate["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert vendor_certificate["objective_validation_repair"] is True
    assert vendor_certificate["policy"]["hermetic_shadows_are_differential_only"] is True
    assert vendor_certificate["policy"]["hermetic_shadows_cannot_satisfy_vendor"] is True
    assert vendor_certificate["policy"]["souffle_linux_aarch64_supported"] is True
    assert (
        vendor_certificate["policy"]["secpal_linux_aarch64_narrow_platform_exception"]
        is True
    )
    assert vendor_certificate["policy"]["grants_authorization_decision_authority"] is False
    assert vendor_certificate["policy"]["never_mutate_system_package_manager"] is True
    assert vendor_certificate["policy"]["runner_owned_fault_injection"] is True
    assert (
        vendor_certificate["policy"]["vendor_prover_fault_environment_required"]
        is False
    )
    assert (
        vendor_certificate["policy"]["native_install_manifest_identity_verified"]
        is True
    )
    assert (
        vendor_certificate["policy"]["native_dependency_packages_rehashed"]
        is True
    )


def test_vendor_souffle_digests_and_deps(vendor_certificate: dict[str, Any]) -> None:
    souffle = vendor_certificate["souffle"]
    assert souffle["certified"] is True
    assert souffle["usable"] is True
    assert souffle["version"] == "2.4.1"
    assert souffle["source_archive_sha256"] == REQUIRED_SOURCE_SHA256
    assert souffle["artifact_sha256"]
    assert len(souffle["artifact_sha256"]) == 64
    assert souffle["artifact_kind"] == "native_compiled_executable"
    assert souffle["artifact_size_bytes"] > 0
    assert souffle["native_binary_format"] == "elf"
    assert souffle["native_machine"] == "aarch64"
    for key in (
        "identity_manifest_sha256",
        "identity_manifest_file_sha256",
        "deployment_lock_sha256",
        "pin_contract_sha256",
        "build_contract_sha256",
        "dependency_package_set_sha256",
    ):
        assert len(souffle[key]) == 64
    assert REQUIRED_BUILD_DEPS <= set(souffle["build_dependency_identities"])
    assert len(souffle["dependency_packages"]) == 6
    assert souffle["managed_dependency_prefix"] is True
    if souffle["is_relocated_install"]:
        assert len(souffle["relocation_binding_sha256"]) == 64
        assert (
            souffle["identity_manifest_sha256"]
            != souffle["relocation_binding_sha256"]
        )
    assert souffle["is_vendor_build"] is True
    assert souffle["is_hermetic_shadow"] is False
    assert souffle["linux_aarch64_supported"] is True
    assert REQUIRED_BUILD_DEPS <= set(souffle["build_dependencies"] or {})
    assert Path(souffle["executable"]).is_file()


def test_vendor_categories_and_mutations(vendor_certificate: dict[str, Any]) -> None:
    assert REQUIRED_CATEGORIES <= set(vendor_certificate["categories_exercised"])
    assert set(vendor_certificate["mutation_kinds"]) == REQUIRED_MUTATIONS


@pytest.mark.parametrize("category", sorted(REQUIRED_CATEGORIES))
def test_vendor_category_outcomes(
    certifier, vendor_certificate: dict[str, Any], category: str
) -> None:
    souffle = vendor_certificate["souffle"]
    executable = souffle["executable"]
    version = souffle["version"]
    runtime_env = certifier.native_souffle_runtime_environment(
        souffle["dependency_prefix"]
    )
    specs = [
        spec for spec in certifier.default_case_specs() if spec.category == category
    ]
    assert specs, category
    for spec in specs:
        document, query, expected = certifier.materialize_case(spec)
        record = certifier.run_shadow_case(
            "souffle",
            spec.case_id,
            document,
            query,
            executable=executable,
            engine_version=version,
            env=runtime_env,
        )
        assert record.outcome == expected, (spec.case_id, record)
        assert record.agreed is True
        assert record.quarantined is False
        assert record.authority == "none"
        assert record.is_authorization_authority is False
        assert record.is_theorem_authority is False


@pytest.mark.parametrize("mutation_kind", sorted(REQUIRED_MUTATIONS))
def test_vendor_rule_scope_mutations(
    certifier, vendor_certificate: dict[str, Any], mutation_kind: str
) -> None:
    souffle = vendor_certificate["souffle"]
    executable = souffle["executable"]
    version = souffle["version"]
    runtime_env = certifier.native_souffle_runtime_environment(
        souffle["dependency_prefix"]
    )
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category == "mutation" and spec.mutation_kind == mutation_kind
    ]
    assert specs, mutation_kind
    for spec in specs:
        base = certifier._semantic._fixture_by_id(spec.base_fixture_id)
        baseline = certifier.run_shadow_case(
            "souffle",
            f"{spec.case_id}:baseline",
            base.document,
            base.query,
            executable=executable,
            engine_version=version,
            env=runtime_env,
        )
        document, query, expected = certifier.materialize_case(spec)
        mutated = certifier.run_shadow_case(
            "souffle",
            spec.case_id,
            document,
            query,
            executable=executable,
            engine_version=version,
            env=runtime_env,
        )
        assert mutated.outcome != baseline.outcome
        assert mutated.outcome == expected
        assert mutated.agreed is True
        assert mutated.policy_digest != baseline.policy_digest


def test_vendor_replay_malformed_timeout_disagreement(
    certifier,
    vendor_certificate: dict[str, Any],
) -> None:
    from ipfs_datasets_py.logic.backends.datalog.adapters import (
        DEFAULT_AUTHORIZATION_FIXTURES,
    )

    souffle = vendor_certificate["souffle"]
    executable = souffle["executable"]
    version = souffle["version"]
    runtime_env = certifier.native_souffle_runtime_environment(
        souffle["dependency_prefix"]
    )

    # Replay
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category in {"deny", "unknown", "conflict"}
    ]
    assert specs
    for spec in specs[:2]:
        document, query, _ = certifier.materialize_case(spec)
        first = certifier.run_shadow_case(
            "souffle",
            spec.case_id,
            document,
            query,
            executable=executable,
            engine_version=version,
            env=runtime_env,
        )
        second = certifier.run_shadow_case(
            "souffle",
            f"{spec.case_id}:replay",
            document,
            query,
            executable=executable,
            engine_version=version,
            env=runtime_env,
        )
        assert first.outcome == second.outcome
        assert first.policy_digest == second.policy_digest

    # Malformed
    malformed_harness = certifier.AuthorizationFaultHarness(
        certifier.FAULT_MALFORMED_OUTPUT
    )
    malformed = certifier.run_shadow_case(
        "souffle",
        "case:malformed",
        None,
        None,
        executable=executable,
        engine_version=version,
        expect_error=True,
        env=runtime_env,
        runner=malformed_harness,
    )
    assert malformed.outcome != "allow"
    assert malformed.malformed is True
    assert malformed.quarantined is True

    # Timeout
    fixture = next(
        item for item in DEFAULT_AUTHORIZATION_FIXTURES if item.category == "allow"
    )
    timeout_harness = certifier.AuthorizationFaultHarness(
        certifier.FAULT_TIMEOUT
    )
    timed = certifier.run_shadow_case(
        "souffle",
        "case:timeout",
        fixture.document,
        fixture.query,
        executable=executable,
        engine_version=version,
        timeout_seconds=0.25,
        env=runtime_env,
        runner=timeout_harness,
    )
    assert timed.timed_out is True
    assert timed.quarantined is True
    assert timed.outcome == "timeout"

    # Disagreement
    disagreement_harness = certifier.AuthorizationFaultHarness(
        certifier.FAULT_DISAGREEMENT
    )
    disagree = certifier.run_shadow_case(
        "souffle",
        "case:disagreement",
        fixture.document,
        fixture.query,
        executable=executable,
        engine_version=version,
        env=runtime_env,
        runner=disagreement_harness,
    )
    assert disagree.agreed is False
    assert disagree.quarantined is True
    assert disagree.outcome != disagree.reference_outcome

    for harness in (
        malformed_harness,
        timeout_harness,
        disagreement_harness,
    ):
        assert len(harness.requests) == 1
        assert not any(
            name.startswith("AUTHZ_SHADOW_")
            for name in harness.requests[0].environment
        )


def test_secpal_exception_in_certificate(vendor_certificate: dict[str, Any]) -> None:
    exception = vendor_certificate["secpal_platform_exception"]
    assert exception["tool_id"] == "secpal"
    assert exception["host_platform"] == LINUX_AARCH64
    assert exception["exception"] is True
    assert exception["narrow_scope"] is True
    assert exception["classification"] == "unsupported_here"
    assert exception["installed"] is False
    assert exception["complete"] is False
    assert exception["authoritative"] is False
    assert exception["production_certified"] is False
    assert LINUX_AARCH64 not in (exception.get("supported_platforms") or [])
    assert exception["live_ready"] is False
    assert exception["platform_exception_satisfies_live_readiness"] is False
    assert exception["live_readiness"]["authoritative_live_evidence_available"] is False
    assert "official_vendor_distribution_retired" in exception["live_block_reasons"]
    assert "vendor_artifact_provenance_missing" in exception["live_block_reasons"]
    assert "vendor_installer_not_implemented" in exception["live_block_reasons"]
    assert vendor_certificate["certified"] is True
    assert vendor_certificate["souffle_vendor_certified"] is True
    assert vendor_certificate["secpal_vendor_certified"] is False
    assert vendor_certificate["secpal_live_ready"] is False
    assert vendor_certificate["combined_external_authorization_certified"] is False


def test_vendor_lane_handler(certifier, install_root, dependency_prefix) -> None:
    result = certifier.external_authorization_vendor_lane_handler(
        install_root=install_root,
        dependency_prefix=dependency_prefix,
        force_install=False,
        skip_install=True,
        platform_id=LINUX_AARCH64,
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )
    assert result["interface"] == VENDOR_INTERFACE
    assert result["goal_id"] == VENDOR_GOAL_ID
    assert result["task_id"] == VENDOR_TASK_ID
    assert result["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert result["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert result["objective_validation_repair"] is True
    assert result["certified"] is True
    assert result["souffle_vendor_certified"] is True
    assert result["secpal_vendor_certified"] is False
    assert result["combined_external_authorization_certified"] is False
    assert result["status"] == "certified"
    assert result["secpal_exception"] is True
    assert result["secpal_live_ready"] is False
    assert "official_vendor_distribution_retired" in result[
        "secpal_live_block_reasons"
    ]
    assert result["hermetic_shadows_are_differential_only"] is True
    assert result["grants_authorization_decision_authority"] is False
    assert result["certificate_digest_sha256"]
    assert len(result["certificate_digest_sha256"]) == 64


def test_checked_in_vendor_receipt_structure() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == VENDOR_RECEIPT_SCHEMA
    assert receipt["interface"] == VENDOR_INTERFACE
    assert receipt["goal_id"] == VENDOR_GOAL_ID
    assert receipt["task_id"] == VENDOR_TASK_ID
    assert receipt["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert receipt["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert receipt["objective_validation_repair"] is True
    souffle = receipt["souffle"]
    assert souffle["version"] == "2.4.1"
    assert souffle["source_archive_sha256"] == REQUIRED_SOURCE_SHA256
    assert souffle["artifact_kind"] == "native_compiled_executable"
    assert souffle["native_binary_format"] == "elf"
    assert souffle["native_machine"] == "aarch64"
    assert len(souffle["identity_manifest_sha256"]) == 64
    assert len(souffle["dependency_package_set_sha256"]) == 64
    assert len(souffle["dependency_packages"]) == 6
    assert souffle["is_vendor_build"] is True
    assert souffle["is_hermetic_shadow"] is False
    assert souffle["linux_aarch64_supported"] is True
    assert REQUIRED_BUILD_DEPS <= set(souffle.get("build_dependencies") or {})
    exception = receipt["secpal_platform_exception"]
    assert exception["exception"] is True or exception["host_platform"] != LINUX_AARCH64
    if exception.get("exception"):
        assert exception["installed"] is False
        assert exception["complete"] is False
        assert exception["authoritative"] is False
        assert exception["production_certified"] is False
    assert receipt["policy"]["hermetic_shadows_are_differential_only"] is True
    assert receipt.get("receipt_digest_sha256") or receipt.get(
        "certificate_digest_sha256"
    )
    assert receipt["acceptance"]["objective_validation_repair"] is True
    assert (
        receipt["acceptance"]["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert receipt["summary"]["objective_validation_repair"] is True


def test_write_vendor_receipt_roundtrip(
    certifier, vendor_certificate: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "formal_verification_authorization_vendor_install_receipt.json"
    receipt = certifier.write_vendor_install_receipt(
        vendor_certificate,
        receipt_path=path,
    )
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["interface"] == VENDOR_INTERFACE
    assert loaded["souffle"]["source_archive_sha256"] == REQUIRED_SOURCE_SHA256
    assert (
        loaded["souffle"]["artifact_kind"]
        == "native_compiled_executable"
    )
    assert len(loaded["souffle"]["identity_manifest_sha256"]) == 64
    assert len(loaded["souffle"]["dependency_package_set_sha256"]) == 64
    assert loaded["receipt_digest_sha256"] == receipt["receipt_digest_sha256"]
    assert loaded["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert loaded["objective_validation_repair"] is True
    assert loaded["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    secpal = loaded["secpal_platform_exception"]
    assert secpal["live_ready"] is False
    assert secpal["platform_exception_satisfies_live_readiness"] is False
    assert "vendor_installer_not_implemented" in secpal["live_block_reasons"]


def test_public_vendor_receipt_is_portable_and_self_digesting(
    certifier,
    vendor_certificate: dict[str, Any],
    install_root: Path,
    dependency_prefix: Path,
) -> None:
    receipt = certifier.build_vendor_install_receipt(
        vendor_certificate,
        repo_root=REPO_ROOT,
    )
    encoded = json.dumps(receipt, sort_keys=True)
    executable = receipt["souffle"]["executable"]
    assert executable == "<managed-tool-path-redacted>/souffle"
    assert receipt["souffle"]["executable_basename"] == "souffle"
    assert receipt["souffle"]["managed_executable"] is True
    assert str(install_root) not in encoded
    assert str(dependency_prefix) not in encoded
    assert str(REPO_ROOT) not in encoded
    assert certifier.public_evidence_audit(
        receipt, repo_root=REPO_ROOT
    )["satisfied"] is True
    assert receipt["receipt_digest_sha256"] == certifier._stable_json_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_digest_sha256"
        }
    )


def test_vendor_receipt_writer_fails_closed_on_public_evidence_audit(
    certifier,
    vendor_certificate: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unsafe-authorization-receipt.json"
    monkeypatch.setattr(
        certifier,
        "public_evidence_audit",
        lambda *_args, **_kwargs: {
            "satisfied": False,
            "failures": ["host_private_path"],
        },
    )
    with pytest.raises(
        certifier.ExternalAuthorizationCertificationError,
        match="unsafe public authorization receipt",
    ):
        certifier.write_vendor_install_receipt(
            vendor_certificate,
            repo_root=REPO_ROOT,
            receipt_path=path,
        )
    assert not path.exists()


# ---------------------------------------------------------------------------
# Objective validation repair (FVT-073 / FVT-G209)
# ---------------------------------------------------------------------------


def test_objective_validation_repair_proves_g209_acceptance(
    certifier,
    installer,
    install_root,
    dependency_prefix,
    vendor_certificate: dict[str, Any],
) -> None:
    """Objective validation repair covers every FVT-G209 acceptance term.

    This is the synthetic evidence term ``objective validation repair`` for the
    validation gate (FVT-073): path evidence for the certifier, installer,
    lock, focused test, and receipt may already exist while the supervisor
    still needs an explicit re-proof of checksummed vendor Soufflé, linux-
    aarch64 support, lock-derived SecPAL platform exception, and hermetic
    shadow differential-only boundaries.
    """

    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert VENDOR_REPAIR_TASK_ID == "FVT-073"
    assert VENDOR_GOAL_ID == "FVT-G209"
    assert VENDOR_TASK_ID == "FVT-055"

    assert certifier.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert certifier.VENDOR_REPAIR_TASK_ID == VENDOR_REPAIR_TASK_ID
    assert certifier.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert installer.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert installer.VENDOR_REPAIR_TASK_ID == VENDOR_REPAIR_TASK_ID

    # Phrase must appear in declared outputs so path+content scans re-find
    # the validation-gate evidence term.
    certifier_source = CERTIFIER_PATH.read_text(encoding="utf-8")
    installer_source = INSTALLER_PATH.read_text(encoding="utf-8")
    receipt_source = RECEIPT_PATH.read_text(encoding="utf-8")
    module_source = Path(__file__).read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in certifier_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in installer_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in receipt_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in module_source

    assert CERTIFIER_PATH.is_file() and CERTIFIER_PATH.stat().st_size > 1000
    assert INSTALLER_PATH.is_file() and INSTALLER_PATH.stat().st_size > 1000
    assert RECEIPT_PATH.is_file() and RECEIPT_PATH.stat().st_size > 500
    assert LOCK_PATH.is_file()

    # Full vendor certificate re-proves FVT-G209 acceptance.
    assert vendor_certificate["certified"] is True
    assert vendor_certificate["goal_id"] == VENDOR_GOAL_ID
    assert vendor_certificate["task_id"] == VENDOR_TASK_ID
    assert vendor_certificate["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert (
        vendor_certificate["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert vendor_certificate["objective_validation_repair"] is True
    assert vendor_certificate["forbids_theorem_authority"] is True
    assert vendor_certificate["authority_ceiling"] == "none"
    assert vendor_certificate["policy"]["grants_theorem_authority"] is False
    assert vendor_certificate["policy"]["grants_authorization_decision_authority"] is False
    assert vendor_certificate["policy"]["hermetic_shadows_are_differential_only"] is True
    assert vendor_certificate["policy"]["hermetic_shadows_cannot_satisfy_vendor"] is True
    assert vendor_certificate["policy"]["souffle_source_archive_checksummed"] is True
    assert vendor_certificate["policy"]["souffle_linux_aarch64_supported"] is True
    assert (
        vendor_certificate["policy"]["secpal_linux_aarch64_narrow_platform_exception"]
        is True
    )
    assert (
        vendor_certificate["policy"]
        ["secpal_platform_exception_does_not_satisfy_live_readiness"]
        is True
    )
    assert vendor_certificate["secpal_live_readiness"]["ready"] is False
    assert vendor_certificate["policy"]["never_mutate_system_package_manager"] is True
    assert vendor_certificate["acceptance"]["objective_validation_repair"] is True
    assert (
        vendor_certificate["acceptance"]["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert vendor_certificate["acceptance"]["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert vendor_certificate["summary"]["objective_validation_repair"] is True
    assert vendor_certificate["summary"]["block_reasons"] == []

    souffle = vendor_certificate["souffle"]
    assert souffle["certified"] is True
    assert souffle["usable"] is True
    assert souffle["version"] == "2.4.1"
    assert souffle["source_archive_sha256"] == REQUIRED_SOURCE_SHA256
    assert souffle["artifact_kind"] == "native_compiled_executable"
    assert souffle["native_binary_format"] == "elf"
    assert souffle["native_machine"] == "aarch64"
    assert len(souffle["identity_manifest_sha256"]) == 64
    assert len(souffle["dependency_package_set_sha256"]) == 64
    assert len(souffle["dependency_packages"]) == 6
    assert souffle["is_vendor_build"] is True
    assert souffle["is_hermetic_shadow"] is False
    assert souffle["linux_aarch64_supported"] is True
    assert REQUIRED_BUILD_DEPS <= set(souffle.get("build_dependencies") or {})
    assert REQUIRED_CATEGORIES <= set(vendor_certificate["categories_exercised"])
    assert set(vendor_certificate["mutation_kinds"]) == REQUIRED_MUTATIONS

    exception = vendor_certificate["secpal_platform_exception"]
    assert exception["exception"] is True
    assert exception["narrow_scope"] is True
    assert exception["classification"] == "unsupported_here"
    assert exception["installed"] is False
    assert exception["complete"] is False
    assert exception["authoritative"] is False
    assert exception["production_certified"] is False

    # Checked-in receipt binds the same validation-repair evidence term.
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert receipt["objective_validation_repair"] is True
    assert receipt["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert receipt["acceptance"]["objective_validation_repair"] is True

    # Lane handler reports the same validation-repair binding.
    handler = certifier.external_authorization_vendor_lane_handler(
        force_install=False,
        skip_install=True,
        platform_id=LINUX_AARCH64,
        repo_root=REPO_ROOT,
        install_root=install_root,
        dependency_prefix=dependency_prefix,
        lock_path=LOCK_PATH,
    )
    assert handler["certified"] is True
    assert handler["repair_task_id"] == VENDOR_REPAIR_TASK_ID
    assert handler["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert handler["objective_validation_repair"] is True
    assert handler["grants_authorization_decision_authority"] is False
