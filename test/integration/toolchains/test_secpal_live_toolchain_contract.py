"""Fail-closed SecPAL artifact intake and live-toolchain contract (FVT-086).

``SecPALLiveToolchainContract@1`` / FVT-G217

The recovered Microsoft MSI has strong immutable provenance, but its EULA
forbids redistribution/live use and its payload has no reviewed general policy
CLI.  These tests prove that:

* the lock binds official publisher/archived URLs, ProductVersion/ProductCode,
  exact size and SHA-256, Authenticode identity, EULA digest, and reviewed
  redistribution/execution limits;
* intake requires an explicit local path and explicit operator license
  acceptance, performs no download/extraction/execution/redistribution, stages
  non-executable bytes atomically, and restores the previous exact tree on
  failure;
* the live execution platform matrix is empty so every host fails closed for
  vendor installation and semantic certification;
* authentic sample execution remains non-promotable operator compatibility and
  cannot create an engine identity, arbitrary-policy interface, platform
  authority, production permission, or FVT-G219 completion;
* fixtures, in-process ``secpal-authorization``, and hermetic adapters remain
  separately named and cannot impersonate the external vendor tool.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from ipfs_datasets_py.logic.backends.installers import authorization as installer
from tools.logic.certification import authorization_external as certifier


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
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

LIVE_PLATFORMS = ("linux-aarch64", "linux-x86_64")


def _secpal_operator(lock: dict[str, object]) -> dict[str, object]:
    tools = {item["tool_id"]: item for item in lock["tools"]}  # type: ignore[index]
    return tools["secpal"]["deployment_contract"]["vendor_install"][  # type: ignore[index]
        "operator_artifact"
    ]


def _tools(lock: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        item["tool_id"]: item  # type: ignore[index]
        for item in lock["tools"]  # type: ignore[index]
        if isinstance(item, dict) and isinstance(item.get("tool_id"), str)
    }


def _fixture_lock(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    operator = _secpal_operator(payload)
    operator.update(
        {
            "status": "test_fixture",
            "evidence_class": installer.SECPAL_TEST_FIXTURE_EVIDENCE_CLASS,
            "artifact_filename": "SecPalFixture.msi",
            "artifact_sha256": hashlib.sha256(content).hexdigest(),
            "artifact_size_bytes": len(content),
            "release_version": "fixture-1.0",
        }
    )
    lock_path = tmp_path / "fixture.lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = tmp_path / "SecPalFixture.msi"
    artifact.write_bytes(content)
    return lock_path, artifact


def test_expected_surfaces_and_contract_identity() -> None:
    assert LOCK_PATH.is_file()
    assert INSTALLER_PATH.is_file()
    assert CERTIFIER_PATH.is_file()
    assert certifier.SECPAL_LIVE_INTERFACE == "SecPALLiveToolchainContract@1"
    assert certifier.SECPAL_LIVE_SCHEMA_VERSION == (
        "secpal-live-toolchain-contract/v1"
    )
    assert certifier.SECPAL_LIVE_GOAL_ID == "FVT-G217"
    assert certifier.SECPAL_LIVE_TASK_ID == "FVT-086"
    assert "test_secpal_live_toolchain_contract.py" in (
        certifier.OBJECTIVE_VALIDATION_COMMAND
    )
    contract = certifier.build_secpal_live_toolchain_contract(
        platform_id="linux-aarch64",
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )
    assert contract["interface"] == "SecPALLiveToolchainContract@1"
    assert contract["goal_id"] == "FVT-G217"
    assert contract["task_id"] == "FVT-086"
    assert contract["contract_complete"] is False
    assert contract["can_promote"] is False
    assert contract["can_complete_fvt_g219"] is False


def test_lock_binds_recovered_official_msi_without_claiming_a_live_engine() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    secpal = _tools(lock)["secpal"]
    deployment = secpal["deployment_contract"]
    vendor = deployment["vendor_install"]
    operator = vendor["operator_artifact"]
    authenticode = operator["authenticode_evidence"]
    license_evidence = operator["license_evidence"]

    assert deployment["supported_platforms"] == []
    assert vendor["installer_implemented"] is False
    assert vendor["artifact_intake_implemented"] is True
    assert operator["status"] == "reviewed_restricted_artifact"
    assert operator["evidence_class"] == "reviewed_official_artifact"
    assert operator["artifact_provisioning"] == "explicit_local_path_only"
    assert operator["artifact_filename"] == "SecPal_Research_Release.msi"
    assert operator["artifact_sha256"] == installer.SECPAL_OFFICIAL_ARTIFACT_SHA256
    assert operator["artifact_size_bytes"] == 2_458_624
    assert operator["release_version"] == "1.0.0"
    assert operator["publisher"] == "Microsoft Research"
    assert operator["product_name"] == "Microsoft SecPAL Research Release"
    assert operator["product_code"] == "{957BD905-629C-45B0-AA93-EC1AAD218115}"
    assert operator["upgrade_code"] == installer.SECPAL_OFFICIAL_MSI_UPGRADE_CODE
    assert operator["package_revision"] == installer.SECPAL_OFFICIAL_MSI_REVISION
    assert (
        operator["original_download_url"]
        == installer.SECPAL_OFFICIAL_ORIGINAL_ARTIFACT_URL
    )
    assert (
        operator["archived_project_metadata_url"]
        == installer.SECPAL_ARCHIVED_PROJECT_METADATA_URL
    )
    assert (
        operator["archived_binary_evidence_url"]
        == installer.SECPAL_ARCHIVED_BINARY_EVIDENCE_URL
    )
    assert operator["archive_is_acquisition_authority"] is False
    assert authenticode["verified"] is True
    assert authenticode["signer"] == "Microsoft Corporation"
    assert authenticode["sha1"] == installer.SECPAL_AUTHENTICODE_SHA1
    assert license_evidence["sha256"] == installer.SECPAL_EULA_SHA256
    assert license_evidence["acceptance_required"] is True
    assert license_evidence["redistribution_terms"] == (
        "prohibited_third_parties_must_obtain_from_microsoft"
    )
    assert license_evidence["execution_terms"] == (
        "design_develop_test_or_research_only_not_intended_for_live_environment"
    )
    assert operator["redistribution_permitted"] is False
    assert operator["production_use_permitted"] is False
    assert operator["downloads_permitted"] is False
    assert operator["transactional_install"] is True
    assert operator["rollback_required"] is True
    assert operator["fixtures_can_promote"] is False
    assert operator["artifact_intake_only"] is True
    assert operator["live_certification_eligible"] is False
    assert operator["executable_contract"]["cli_available"] is False
    assert operator["runtime_contract"]["runtime_validated"] is False
    assert operator["platform_matrix_evidence"][
        "live_execution_supported_platforms"
    ] == []


def test_offline_report_separates_intake_from_live_readiness() -> None:
    report = installer.secpal_vendor_prerequisite_report(lock_path=LOCK_PATH)
    contract = certifier.build_secpal_live_toolchain_contract(
        platform_id="linux-aarch64",
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )

    assert report["reviewed_official_artifact"] is True
    assert report["artifact_intake_implemented"] is True
    assert report["artifact_intake_ready"] is True
    assert report["ready"] is False
    assert report["installable"] is False
    assert report["authoritative_live_evidence_available"] is False
    assert report["redistribution_permitted"] is False
    assert report["production_use_permitted"] is False
    assert report["arbitrary_policy_cli_available"] is False
    assert report["downloads_permitted"] is False
    assert report["supported_platforms"] == []
    assert report["archive_is_acquisition_authority"] is False
    assert report["python_compatible_engine_is_shadow_only"] is True
    assert {
        "vendor_license_evidence_missing",
        "vendor_runtime_contract_missing",
        "vendor_executable_contract_missing",
        "vendor_platform_matrix_unverified",
        "vendor_installer_not_implemented",
    } <= set(report["block_reasons"])

    assert contract["artifact_intake_ready"] is True
    assert contract["live_ready"] is False
    assert contract["installable"] is False
    assert contract["live_semantic_runner_available"] is False
    assert contract["arbitrary_policy_interface_verified"] is False
    assert contract["supported_platforms"] == []
    assert contract["live_execution_supported_platforms"] == []
    assert contract["policy"]["no_download"] is True
    assert contract["policy"]["no_extraction"] is True
    assert contract["policy"]["no_execution_during_intake"] is True
    assert contract["policy"]["cannot_complete_fvt_g219"] is True


def test_operator_compatibility_is_real_but_unbound_and_nonpromotable() -> None:
    report = installer.secpal_vendor_prerequisite_report(lock_path=LOCK_PATH)
    probe = report["operator_compatibility_probe"]
    contract = certifier.build_secpal_live_toolchain_contract(
        platform_id="linux-aarch64",
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )

    assert probe["evidence_class"] == "operator_compatibility_probe"
    assert probe["authentic_core_loaded"] is True
    assert probe["core_dll_sha256"] == (
        "2afacfc4121332c7fec5df32911e9a8b8d9926807af15d8c71b41a2928ee8b0a"
    )
    assert probe["samples_executable_sha256"] == (
        "9a46e3bbf7bc58f0b9964814def3dd1224cd3893bfc34c72c13f3b8190599a07"
    )
    assert probe["sample_scenarios_executed"] == 18
    assert probe["contract_complete"] is False
    assert probe["arbitrary_policy_interface_verified"] is False
    assert probe["vendor_supported_platform"] is False
    assert probe["production_use_permitted"] is False
    assert probe["live_certification_eligible"] is False
    assert report["operator_compatibility_classified"] is True
    assert report["operator_compatibility_bound"] is False
    assert report["operator_compatibility_can_promote"] is False
    assert contract["operator_compatibility_only"] is True
    assert contract["operator_compatibility_can_promote"] is False
    assert contract["can_complete_fvt_g219"] is False


def test_in_process_and_hermetic_remain_separately_named() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    tools = _tools(lock)
    assert "secpal" in tools
    assert "secpal-authorization" in tools
    assert tools["secpal"]["display_name"] == "External SecPAL engine"
    assert tools["secpal"]["identity_kind"] == "operator_bound_artifact"
    assert tools["secpal-authorization"]["display_name"] == (
        "In-process SecPAL-style authorization engine"
    )
    assert tools["secpal-authorization"]["identity_kind"] == "in_process"
    assert tools["secpal"]["tool_id"] != tools["secpal-authorization"]["tool_id"]

    banner = installer.build_shadow_shim_source(
        installer.TOOL_SECPAL,
        "1.0.0-reviewed",
        identity_file="/tmp/secpal-hermetic-identity.json",
    )
    assert "hermetic-authorization-shadow" in banner
    assert "secpal" in banner.lower()

    contract = certifier.build_secpal_live_toolchain_contract(
        platform_id="linux-aarch64",
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )
    separation = contract["identity_separation"]
    assert separation["external_vendor_tool_id"] == "secpal"
    assert separation["in_process_tool_id"] == "secpal-authorization"
    assert separation["external_cannot_be_impersonated_by_in_process"] is True
    assert separation["hermetic_shadow_is_differential_only"] is True
    assert separation["never_promote_hermetic_shadow_as_vendor"] is True
    assert contract["policy"][
        "in_process_and_hermetic_remain_separately_named"
    ] is True


@pytest.mark.parametrize(
    ("platform_id", "expected_status", "platform_exception"),
    [
        ("linux-aarch64", "unsupported_platform", True),
        ("linux-x86_64", "unsupported_platform", True),
    ],
)
def test_no_live_platform_can_install_or_write(
    tmp_path: Path,
    platform_id: str,
    expected_status: str,
    platform_exception: bool,
) -> None:
    install_root = tmp_path / platform_id
    receipt = installer.ensure_secpal(
        yes=True,
        strict=True,
        install_root=install_root,
        lock_path=LOCK_PATH,
        platform_id=platform_id,
        hermetic_shadow=False,
        vendor=True,
        test_mode=True,
    )

    assert receipt.status == expected_status
    assert receipt.platform_exception is platform_exception
    assert receipt.identity is None
    assert receipt.operator_artifact is None
    assert receipt.installed is False
    assert receipt.production_certified is False
    assert receipt.complete is False
    assert not install_root.exists()

    exception = certifier.derive_secpal_platform_exception(
        platform_id=platform_id,
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )
    assert exception["exception"] is True
    assert exception["installed"] is False
    assert exception["complete"] is False
    assert exception["authoritative"] is False
    assert exception["production_certified"] is False
    assert exception["live_ready"] is False
    assert exception["platform_exception_satisfies_live_readiness"] is False


def test_semantic_certification_fails_closed_on_every_host() -> None:
    for platform_id in LIVE_PLATFORMS:
        contract = certifier.build_secpal_live_toolchain_contract(
            platform_id=platform_id,
            repo_root=REPO_ROOT,
            lock_path=LOCK_PATH,
        )
        assert contract["live_ready"] is False
        assert contract["installable"] is False
        assert contract["authoritative_live_evidence_available"] is False
        assert contract["live_semantic_runner_available"] is False
        assert contract["can_complete_fvt_g219"] is False
        assert installer.tool_supported_on_platform(
            installer.TOOL_SECPAL,
            platform_id,
            lock_path=LOCK_PATH,
        ) is False


def test_license_acceptance_required_and_public_boundary_is_strict(
    tmp_path: Path,
) -> None:
    content = b"bounded non-vendor SecPAL MSI fixture\n"
    lock_path, artifact = _fixture_lock(tmp_path, content)
    install_root = tmp_path / "managed"

    with pytest.raises(
        installer.AuthorizationInstallerError,
        match="license_accepted",
    ):
        installer._stage_secpal_operator_artifact(
            artifact,
            license_accepted=False,
            install_root=install_root,
            lock_path=lock_path,
            platform_id="linux-aarch64",
            allow_test_fixture=True,
        )
    assert not install_root.exists()

    with pytest.raises(
        installer.AuthorizationInstallerError,
        match="not ready|test_fixture",
    ):
        installer.stage_secpal_operator_artifact(
            artifact,
            license_accepted=True,
            install_root=install_root,
            lock_path=lock_path,
        )
    assert not install_root.exists()


def test_private_fixture_exercises_atomic_intake_but_cannot_promote(
    tmp_path: Path,
) -> None:
    content = b"bounded non-vendor SecPAL MSI fixture\n"
    lock_path, artifact = _fixture_lock(tmp_path, content)
    install_root = tmp_path / "managed"

    receipt = installer._stage_secpal_operator_artifact(
        artifact,
        license_accepted=True,
        install_root=install_root,
        lock_path=lock_path,
        platform_id="linux-aarch64",
        allow_test_fixture=True,
    )

    assert receipt.status == "staged_only"
    assert receipt.fixture is True
    assert receipt.evidence_class == "test_fixture"
    assert receipt.execution_eligible is False
    assert receipt.live_certification_eligible is False
    assert receipt.production_certified is False
    assert receipt.redistribution_permitted is False
    assert receipt.license_accepted is True
    assert receipt.transactional is True
    assert receipt.rollback_preserved is True
    staged = Path(receipt.artifact_path)
    assert staged.read_bytes() == content
    # Staged bytes are non-executable; intake must never produce an engine.
    assert staged.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) == 0
    manifest = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    assert manifest["license_evidence_sha256"] == installer.SECPAL_EULA_SHA256
    assert manifest["artifact_intake_only"] is True
    assert manifest["execution_eligible"] is False
    assert "extracted" not in manifest
    assert "download" not in json.dumps(manifest).lower()

    vendor_receipt = installer.InstallReceipt(
        tool_id=installer.TOOL_SECPAL,
        status="unsupported_platform",
        identity=None,
        selected_version="1.0.0-reviewed",
        platform_id="linux-aarch64",
        platform_exception=True,
        is_vendor_path=True,
        operator_artifact=receipt,
    )
    readiness = installer.secpal_vendor_prerequisite_report(lock_path=lock_path)
    with pytest.raises(
        certifier.ExternalAuthorizationCertificationError,
        match="test fixture",
    ):
        certifier._validated_nonpromotable_secpal_evidence(
            vendor_receipt,
            readiness,
        )


def test_public_boundary_rejects_fixture_even_in_test_mode(tmp_path: Path) -> None:
    lock_path, artifact = _fixture_lock(tmp_path, b"fixture-public-rejection\n")
    install_root = tmp_path / "managed"

    with pytest.raises(
        installer.AuthorizationInstallerError,
        match="not ready|test_fixture",
    ):
        installer.stage_secpal_operator_artifact(
            artifact,
            license_accepted=True,
            install_root=install_root,
            lock_path=lock_path,
        )
    assert not install_root.exists()

    receipt = installer.ensure_secpal(
        yes=True,
        strict=False,
        install_root=install_root,
        lock_path=lock_path,
        platform_id="linux-aarch64",
        hermetic_shadow=False,
        vendor=True,
        test_mode=True,
        operator_artifact_path=artifact,
        license_accepted=True,
    )
    assert receipt.status == "failed"
    assert receipt.operator_artifact is None
    assert receipt.production_certified is False
    assert not install_root.exists()


def test_failed_force_intake_preserves_previous_exact_tree(tmp_path: Path) -> None:
    content = b"first bounded SecPAL fixture\n"
    lock_path, artifact = _fixture_lock(tmp_path, content)
    install_root = tmp_path / "managed"
    first = installer._stage_secpal_operator_artifact(
        artifact,
        license_accepted=True,
        install_root=install_root,
        lock_path=lock_path,
        platform_id="linux-aarch64",
        allow_test_fixture=True,
    )
    old_artifact = Path(first.artifact_path).read_bytes()
    old_manifest = Path(first.manifest_path).read_bytes()
    artifact.write_bytes(b"corrupted replacement")

    with pytest.raises(installer.AuthorizationInstallerError, match="size or sha256"):
        installer._stage_secpal_operator_artifact(
            artifact,
            license_accepted=True,
            install_root=install_root,
            lock_path=lock_path,
            platform_id="linux-aarch64",
            allow_test_fixture=True,
            force=True,
        )

    assert Path(first.artifact_path).read_bytes() == old_artifact
    assert Path(first.manifest_path).read_bytes() == old_manifest


def test_post_publish_revalidation_failure_restores_previous_exact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"post-publication rollback SecPAL fixture\n"
    lock_path, artifact = _fixture_lock(tmp_path, content)
    install_root = tmp_path / "managed"
    first = installer._stage_secpal_operator_artifact(
        artifact,
        license_accepted=True,
        install_root=install_root,
        lock_path=lock_path,
        platform_id="linux-aarch64",
        allow_test_fixture=True,
    )
    destination = Path(first.manifest_path).parent
    sentinel = destination / "previous-tree-sentinel"
    sentinel.write_bytes(b"previous exact tree\n")
    old_artifact = Path(first.artifact_path).read_bytes()
    old_manifest = Path(first.manifest_path).read_bytes()

    monkeypatch.setattr(
        installer,
        "_secpal_operator_artifact_receipt_from_disk",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(
        installer.AuthorizationInstallerError,
        match="failed exact identity revalidation",
    ):
        installer._stage_secpal_operator_artifact(
            artifact,
            license_accepted=True,
            install_root=install_root,
            lock_path=lock_path,
            platform_id="linux-aarch64",
            allow_test_fixture=True,
            force=True,
        )

    assert Path(first.artifact_path).read_bytes() == old_artifact
    assert Path(first.manifest_path).read_bytes() == old_manifest
    assert sentinel.read_bytes() == b"previous exact tree\n"
    assert not list(destination.parent.glob(f".{destination.name}.backup-*"))
    assert not list(destination.parent.glob(f".{destination.name}.failed-*"))


def test_materialize_vendor_secpal_never_yields_engine_identity(
    tmp_path: Path,
) -> None:
    """Artifact intake cannot promote into a vendor engine identity."""
    with pytest.raises(
        installer.AuthorizationInstallerError,
        match="unsupported|unavailable|prerequisite",
    ):
        installer.materialize_vendor_secpal(
            install_root=tmp_path / "managed",
            lock_path=LOCK_PATH,
            platform_id="linux-aarch64",
        )
    with pytest.raises(
        installer.AuthorizationInstallerError,
        match="unsupported|unavailable|prerequisite",
    ):
        installer.materialize_vendor_secpal(
            install_root=tmp_path / "managed-x86",
            lock_path=LOCK_PATH,
            platform_id="linux-x86_64",
        )
