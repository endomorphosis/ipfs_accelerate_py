"""External authorization vendor certification tests (FVT-055 / FVT-G209).

``ExternalAuthorizationVendorCertification@1``

Acceptance covered:

* Soufflé 2.4.1 source/archive and build dependencies are immutable and
  checksummed; user-local executable and artifact digests are exact;
* allow/deny/unknown/conflict/delegation plus rule/scope mutation, replay,
  malformed, timeout, and disagreement cases execute through vendor Soufflé;
* linux-aarch64 is supported for Soufflé;
* external SecPAL is a narrow unsupported-platform exception on
  linux-aarch64 under the current contract and never counts as installed,
  complete, authoritative, or production-certified;
* hermetic shadows remain differential-only and cannot satisfy vendor
  production evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
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
def install_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("authz-vendor")


@pytest.fixture(scope="module")
def vendor_bundle(installer, install_root):
    return installer.ensure_authorization_vendor(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root,
        platform_id=LINUX_AARCH64,
        checksum_verified=True,
    )


@pytest.fixture(scope="module")
def vendor_certificate(certifier, install_root) -> dict[str, Any]:
    return certifier.certify_external_authorization_vendor(
        install_root=install_root,
        force_install=True,
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
    assert LINUX_AARCH64 not in secpal_contract["supported_platforms"]
    assert "linux-x86_64" in secpal_contract["supported_platforms"]
    exception = secpal_contract["platform_exceptions"][LINUX_AARCH64]
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
    assert installer.SOUFFLE_SOURCE_ARCHIVE_SHA256 == REQUIRED_SOURCE_SHA256
    assert REQUIRED_BUILD_DEPS <= set(installer.SOUFFLE_BUILD_DEPENDENCIES)
    meta = installer.describe_authorization_installer()
    assert meta["policy"]["hermetic_shadows_are_differential_only"] is True
    assert meta["policy"]["never_promote_hermetic_shadow_as_vendor"] is True
    assert meta["policy"]["secpal_linux_aarch64_is_narrow_platform_exception"] is True
    assert meta["policy"]["souffle_linux_aarch64_supported"] is True
    assert meta["souffle_source_archive_sha256"] == REQUIRED_SOURCE_SHA256


def test_certifier_vendor_constants(certifier) -> None:
    assert certifier.VENDOR_INTERFACE == VENDOR_INTERFACE
    assert certifier.VENDOR_SCHEMA_VERSION == VENDOR_SCHEMA
    assert certifier.VENDOR_GOAL_ID == VENDOR_GOAL_ID
    assert certifier.VENDOR_TASK_ID == VENDOR_TASK_ID
    assert certifier.SOUFFLE_REQUIRED_SOURCE_SHA256 == REQUIRED_SOURCE_SHA256


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
    )
    banner = (completed.stdout or "") + (completed.stderr or "")
    assert "2.4.1" in banner
    assert "vendor" in banner.casefold()
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


def test_hermetic_shadow_cannot_satisfy_vendor(installer, install_root) -> None:
    hermetic = installer.ensure_souffle(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root / "hermetic-only",
        hermetic_shadow=True,
        vendor=False,
        checksum_verified=True,
    )
    assert hermetic.ok
    assert hermetic.identity is not None
    assert hermetic.identity.is_hermetic_shadow is True
    assert hermetic.identity.is_vendor_build is False
    # Vendor path under a fresh root must not reuse the hermetic shadow lane.
    vendor = installer.ensure_souffle(
        yes=True,
        strict=True,
        force=True,
        install_root=install_root / "vendor-only",
        hermetic_shadow=False,
        vendor=True,
        checksum_verified=True,
        platform_id=LINUX_AARCH64,
    )
    assert vendor.ok
    assert vendor.identity is not None
    assert vendor.identity.is_hermetic_shadow is False
    assert vendor.identity.is_vendor_build is True
    assert vendor.identity.executable != hermetic.identity.executable
    assert "authorization-vendor" in vendor.identity.executable
    assert "authorization-shadows" in hermetic.identity.executable


# ---------------------------------------------------------------------------
# Full vendor certification corpus
# ---------------------------------------------------------------------------


def test_vendor_certificate_envelope(vendor_certificate: dict[str, Any]) -> None:
    assert vendor_certificate["schema_version"] == VENDOR_SCHEMA
    assert vendor_certificate["interface"] == VENDOR_INTERFACE
    assert vendor_certificate["goal_id"] == VENDOR_GOAL_ID
    assert vendor_certificate["task_id"] == VENDOR_TASK_ID
    assert vendor_certificate["host_platform"] == LINUX_AARCH64
    assert vendor_certificate["certified"] is True
    assert vendor_certificate["authority_ceiling"] == "none"
    assert vendor_certificate["policy"]["hermetic_shadows_are_differential_only"] is True
    assert vendor_certificate["policy"]["hermetic_shadows_cannot_satisfy_vendor"] is True
    assert vendor_certificate["policy"]["souffle_linux_aarch64_supported"] is True
    assert (
        vendor_certificate["policy"]["secpal_linux_aarch64_narrow_platform_exception"]
        is True
    )
    assert vendor_certificate["policy"]["grants_authorization_decision_authority"] is False
    assert vendor_certificate["policy"]["never_mutate_system_package_manager"] is True


def test_vendor_souffle_digests_and_deps(vendor_certificate: dict[str, Any]) -> None:
    souffle = vendor_certificate["souffle"]
    assert souffle["certified"] is True
    assert souffle["usable"] is True
    assert souffle["version"] == "2.4.1"
    assert souffle["source_archive_sha256"] == REQUIRED_SOURCE_SHA256
    assert souffle["artifact_sha256"]
    assert len(souffle["artifact_sha256"]) == 64
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
        )
        document, query, expected = certifier.materialize_case(spec)
        mutated = certifier.run_shadow_case(
            "souffle",
            spec.case_id,
            document,
            query,
            executable=executable,
            engine_version=version,
        )
        assert mutated.outcome != baseline.outcome
        assert mutated.outcome == expected
        assert mutated.agreed is True
        assert mutated.policy_digest != baseline.policy_digest


def test_vendor_replay_malformed_timeout_disagreement(
    certifier, installer, vendor_certificate: dict[str, Any]
) -> None:
    from ipfs_datasets_py.logic.backends.datalog.adapters import (
        DEFAULT_AUTHORIZATION_FIXTURES,
    )

    souffle = vendor_certificate["souffle"]
    executable = souffle["executable"]
    version = souffle["version"]

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
        )
        second = certifier.run_shadow_case(
            "souffle",
            f"{spec.case_id}:replay",
            document,
            query,
            executable=executable,
            engine_version=version,
        )
        assert first.outcome == second.outcome
        assert first.policy_digest == second.policy_digest

    # Malformed
    malformed = certifier.run_shadow_case(
        "souffle",
        "case:malformed",
        None,
        None,
        executable=executable,
        engine_version=version,
        expect_error=True,
    )
    assert malformed.outcome != "allow"
    assert malformed.malformed is True
    assert malformed.quarantined is True

    # Timeout
    fixture = next(
        item for item in DEFAULT_AUTHORIZATION_FIXTURES if item.category == "allow"
    )
    timed = certifier.run_shadow_case(
        "souffle",
        "case:timeout",
        fixture.document,
        fixture.query,
        executable=executable,
        engine_version=version,
        timeout_seconds=0.25,
        env={installer.ENV_SLEEP_SECONDS: "2.0"},
    )
    assert timed.timed_out is True
    assert timed.quarantined is True
    assert timed.outcome == "timeout"

    # Disagreement
    disagree = certifier.run_shadow_case(
        "souffle",
        "case:disagreement",
        fixture.document,
        fixture.query,
        executable=executable,
        engine_version=version,
        env={installer.ENV_DISAGREE: "1"},
    )
    assert disagree.agreed is False
    assert disagree.quarantined is True
    assert disagree.outcome != disagree.reference_outcome


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


def test_vendor_lane_handler(certifier, install_root) -> None:
    result = certifier.external_authorization_vendor_lane_handler(
        install_root=install_root,
        force_install=False,
        skip_install=True,
        platform_id=LINUX_AARCH64,
        repo_root=REPO_ROOT,
    )
    assert result["interface"] == VENDOR_INTERFACE
    assert result["goal_id"] == VENDOR_GOAL_ID
    assert result["task_id"] == VENDOR_TASK_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["secpal_exception"] is True
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
    souffle = receipt["souffle"]
    assert souffle["version"] == "2.4.1"
    assert souffle["source_archive_sha256"] == REQUIRED_SOURCE_SHA256
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
    assert loaded["receipt_digest_sha256"] == receipt["receipt_digest_sha256"]
