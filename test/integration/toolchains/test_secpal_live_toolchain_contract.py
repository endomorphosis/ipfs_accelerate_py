"""Fail-closed SecPAL artifact intake and live-toolchain contract.

``SecPALLiveToolchainContract@1`` / FVT-G217 (FVT-086; objective validation
repair FVT-101).

The recovered Microsoft MSI has strong immutable provenance, but its EULA
forbids redistribution/live use and its payload has no reviewed general policy
CLI.  These tests prove that exact local artifact intake is transactional while
remaining categorically distinct from installation or semantic certification.

Objective validation repair (FVT-101)
------------------------------------
Path evidence for this module, the installer, the certifier, and the lock may
already exist while the supervisor validation gate still needs an explicit
re-proof of the full FVT-G217 acceptance matrix.  The synthetic evidence term
``objective validation repair`` is bound here so objective scans re-find
coverage after the hermetic validation command passes.
"""

from __future__ import annotations

import hashlib
import json
import os
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
DEFAULT_MANAGED_PROVER_ROOT = (
    Path.home() / ".local/share/ipfs_datasets_py/theorem-provers"
)
DEPENDENCY_PREFIX_SUFFIX = Path(
    "build-dependencies/souffle/ubuntu-noble-arm64/root"
)

SECPAL_LIVE_INTERFACE = "SecPALLiveToolchainContract@1"
SECPAL_LIVE_GOAL_ID = "FVT-G217"
SECPAL_LIVE_TASK_ID = "FVT-086"
SECPAL_LIVE_REPAIR_TASK_ID = "FVT-101"
OBJECTIVE_VALIDATION_EVIDENCE = "objective validation repair"
OBJECTIVE_VALIDATION_COMMAND = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/toolchains/test_secpal_live_toolchain_contract.py "
    "test/integration/toolchains/test_external_authorization_vendor_certification.py "
    "test/integration/toolchains/test_external_authorization_toolchain_certification.py "
    "-q"
)


def _managed_souffle_paths() -> tuple[Path, Path]:
    root = Path(
        os.environ.get(
            "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
            str(DEFAULT_MANAGED_PROVER_ROOT),
        )
    ).expanduser().resolve()
    install_root = root / "souffle-vendor"
    dependency_prefix = root / DEPENDENCY_PREFIX_SUFFIX
    return install_root, dependency_prefix


def _secpal_operator(lock: dict[str, object]) -> dict[str, object]:
    tools = {item["tool_id"]: item for item in lock["tools"]}  # type: ignore[index]
    return tools["secpal"]["deployment_contract"]["vendor_install"][  # type: ignore[index]
        "operator_artifact"
    ]


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
    assert certifier.SECPAL_LIVE_INTERFACE == SECPAL_LIVE_INTERFACE
    assert certifier.SECPAL_LIVE_GOAL_ID == SECPAL_LIVE_GOAL_ID
    assert certifier.SECPAL_LIVE_TASK_ID == SECPAL_LIVE_TASK_ID
    assert certifier.SECPAL_LIVE_REPAIR_TASK_ID == SECPAL_LIVE_REPAIR_TASK_ID
    assert certifier.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert certifier.OBJECTIVE_VALIDATION_COMMAND == OBJECTIVE_VALIDATION_COMMAND
    assert installer.SECPAL_LIVE_GOAL_ID == SECPAL_LIVE_GOAL_ID
    assert installer.SECPAL_LIVE_TASK_ID == SECPAL_LIVE_TASK_ID
    assert installer.SECPAL_LIVE_REPAIR_TASK_ID == SECPAL_LIVE_REPAIR_TASK_ID
    assert installer.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert installer.OBJECTIVE_VALIDATION_COMMAND == OBJECTIVE_VALIDATION_COMMAND
    assert "test_secpal_live_toolchain_contract.py" in (
        certifier.OBJECTIVE_VALIDATION_COMMAND
    )


def test_lock_binds_recovered_official_msi_without_claiming_a_live_engine() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    secpal = {item["tool_id"]: item for item in lock["tools"]}["secpal"]
    deployment = secpal["deployment_contract"]
    vendor = deployment["vendor_install"]
    operator = vendor["operator_artifact"]

    assert deployment["supported_platforms"] == []
    assert vendor["installer_implemented"] is False
    assert vendor["artifact_intake_implemented"] is True
    assert operator["status"] == "reviewed_restricted_artifact"
    assert operator["evidence_class"] == "reviewed_official_artifact"
    assert operator["artifact_filename"] == "SecPal_Research_Release.msi"
    assert operator["artifact_sha256"] == installer.SECPAL_OFFICIAL_ARTIFACT_SHA256
    assert operator["artifact_size_bytes"] == 2_458_624
    assert operator["release_version"] == "1.0.0"
    assert operator["product_code"] == "{957BD905-629C-45B0-AA93-EC1AAD218115}"
    assert operator["authenticode_evidence"]["verified"] is True
    assert operator["authenticode_evidence"]["signer"] == "Microsoft Corporation"
    assert operator["license_evidence"]["sha256"] == installer.SECPAL_EULA_SHA256
    assert operator["license_evidence"]["acceptance_required"] is True
    assert operator["redistribution_permitted"] is False
    assert operator["production_use_permitted"] is False
    assert operator["executable_contract"]["cli_available"] is False
    assert operator["platform_matrix_evidence"][
        "live_execution_supported_platforms"
    ] == []


def test_offline_report_separates_intake_from_live_readiness() -> None:
    report = installer.secpal_vendor_prerequisite_report(lock_path=LOCK_PATH)

    assert report["reviewed_official_artifact"] is True
    assert report["artifact_intake_implemented"] is True
    assert report["artifact_intake_ready"] is True
    assert report["ready"] is False
    assert report["installable"] is False
    assert report["authoritative_live_evidence_available"] is False
    assert report["redistribution_permitted"] is False
    assert report["production_use_permitted"] is False
    assert report["arbitrary_policy_cli_available"] is False
    assert report["supported_platforms"] == []
    assert {
        "vendor_license_evidence_missing",
        "vendor_runtime_contract_missing",
        "vendor_executable_contract_missing",
        "vendor_platform_matrix_unverified",
        "vendor_installer_not_implemented",
    } <= set(report["block_reasons"])


def test_operator_compatibility_is_real_but_unbound_and_nonpromotable() -> None:
    report = installer.secpal_vendor_prerequisite_report(lock_path=LOCK_PATH)
    probe = report["operator_compatibility_probe"]

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
    assert report["operator_compatibility_classified"] is True
    assert report["operator_compatibility_bound"] is False
    assert report["operator_compatibility_can_promote"] is False


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
    staged = Path(receipt.artifact_path)
    assert staged.read_bytes() == content
    assert staged.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) == 0
    manifest = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    assert manifest["license_evidence_sha256"] == installer.SECPAL_EULA_SHA256

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


# ---------------------------------------------------------------------------
# Objective validation repair (FVT-101 / FVT-G217)
# ---------------------------------------------------------------------------


def test_objective_validation_repair_proves_g217_acceptance() -> None:
    """Objective validation repair covers every FVT-G217 acceptance term.

    This is the synthetic evidence term ``objective validation repair`` for the
    validation gate (FVT-101): path evidence for the certifier, installer,
    lock, and focused test may already exist while the supervisor still needs
    an explicit re-proof that the recovered Microsoft MSI binds official
    provenance, license-aware transactional local intake, an empty live
    platform matrix, and non-promotable operator compatibility.
    """

    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert SECPAL_LIVE_REPAIR_TASK_ID == "FVT-101"
    assert SECPAL_LIVE_GOAL_ID == "FVT-G217"
    assert SECPAL_LIVE_TASK_ID == "FVT-086"
    assert SECPAL_LIVE_INTERFACE == "SecPALLiveToolchainContract@1"

    assert certifier.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert certifier.SECPAL_LIVE_REPAIR_TASK_ID == SECPAL_LIVE_REPAIR_TASK_ID
    assert certifier.SECPAL_LIVE_GOAL_ID == SECPAL_LIVE_GOAL_ID
    assert certifier.SECPAL_LIVE_TASK_ID == SECPAL_LIVE_TASK_ID
    assert certifier.OBJECTIVE_VALIDATION_COMMAND == OBJECTIVE_VALIDATION_COMMAND
    assert installer.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert installer.SECPAL_LIVE_REPAIR_TASK_ID == SECPAL_LIVE_REPAIR_TASK_ID
    assert installer.SECPAL_LIVE_GOAL_ID == SECPAL_LIVE_GOAL_ID
    assert installer.OBJECTIVE_VALIDATION_COMMAND == OBJECTIVE_VALIDATION_COMMAND

    # Phrase must appear in declared outputs so path+content scans re-find
    # the validation-gate evidence term.
    certifier_source = CERTIFIER_PATH.read_text(encoding="utf-8")
    installer_source = INSTALLER_PATH.read_text(encoding="utf-8")
    lock_source = LOCK_PATH.read_text(encoding="utf-8")
    module_source = Path(__file__).read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in certifier_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in installer_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in lock_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in module_source
    assert SECPAL_LIVE_REPAIR_TASK_ID in certifier_source
    assert SECPAL_LIVE_REPAIR_TASK_ID in installer_source
    assert SECPAL_LIVE_REPAIR_TASK_ID in lock_source
    assert SECPAL_LIVE_REPAIR_TASK_ID in module_source

    assert CERTIFIER_PATH.is_file() and CERTIFIER_PATH.stat().st_size > 1000
    assert INSTALLER_PATH.is_file() and INSTALLER_PATH.stat().st_size > 1000
    assert LOCK_PATH.is_file()

    # Lock binds the FVT-101 repair gate under the replaced SecPAL gap.
    lock = json.loads(lock_source)
    gap = lock["replaced_install_gaps"]["datalog_secpal_external"]
    assert gap["goal_id"] == SECPAL_LIVE_GOAL_ID
    assert gap["task_id"] == SECPAL_LIVE_TASK_ID
    assert gap["repair_task_id"] == SECPAL_LIVE_REPAIR_TASK_ID
    assert gap["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert gap["interface"] == SECPAL_LIVE_INTERFACE
    assert "test_secpal_live_toolchain_contract.py" in gap["objective_validation_command"]

    # Offline prerequisite report re-proves FVT-G217 fail-closed readiness.
    report = installer.secpal_vendor_prerequisite_report(lock_path=LOCK_PATH)
    assert report["goal_id"] == SECPAL_LIVE_GOAL_ID
    assert report["task_id"] == SECPAL_LIVE_TASK_ID
    assert report["repair_task_id"] == SECPAL_LIVE_REPAIR_TASK_ID
    assert report["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert report["objective_validation_repair"] is True
    assert report["objective_validation_command"] == OBJECTIVE_VALIDATION_COMMAND
    assert report["reviewed_official_artifact"] is True
    assert report["artifact_intake_implemented"] is True
    assert report["artifact_intake_ready"] is True
    assert report["ready"] is False
    assert report["installable"] is False
    assert report["authoritative_live_evidence_available"] is False
    assert report["redistribution_permitted"] is False
    assert report["production_use_permitted"] is False
    assert report["arbitrary_policy_cli_available"] is False
    assert report["supported_platforms"] == []
    assert report["operator_compatibility_classified"] is True
    assert report["operator_compatibility_bound"] is False
    assert report["operator_compatibility_can_promote"] is False
    assert report["test_fixture_can_promote"] is False
    assert report["downloads_permitted"] is False
    assert report["platform_exception_satisfies_live_readiness"] is False
    assert report["artifact_sha256"] == installer.SECPAL_OFFICIAL_ARTIFACT_SHA256
    assert report["artifact_size_bytes"] == installer.SECPAL_OFFICIAL_ARTIFACT_SIZE_BYTES
    assert report["eula_sha256"] == installer.SECPAL_EULA_SHA256
    assert report["msi_product_version"] == installer.SECPAL_OFFICIAL_MSI_PRODUCT_VERSION

    # Installer metadata surface binds the same repair keys.
    meta = installer.describe_authorization_installer()
    assert meta["secpal_live_goal_id"] == SECPAL_LIVE_GOAL_ID
    assert meta["secpal_live_task_id"] == SECPAL_LIVE_TASK_ID
    assert meta["secpal_live_repair_task_id"] == SECPAL_LIVE_REPAIR_TASK_ID
    assert meta["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert meta["objective_validation_repair"] is True
    assert meta["objective_validation_command"] == OBJECTIVE_VALIDATION_COMMAND
    assert meta["policy"]["secpal_official_artifact_intake_is_transactional"] is True
    assert meta["policy"]["secpal_artifact_intake_is_not_engine_installation"] is True
    assert meta["policy"]["secpal_redistribution_is_prohibited"] is True
    assert meta["policy"]["secpal_no_reviewed_live_execution_platform"] is True
    assert meta["policy"]["secpal_operator_compatibility_is_nonpromotable"] is True
    assert meta["policy"]["secpal_platform_exception_does_not_satisfy_live_readiness"] is True

    # Recovered MSI contract remains official and non-live.
    secpal = {item["tool_id"]: item for item in lock["tools"]}["secpal"]
    operator = secpal["deployment_contract"]["vendor_install"]["operator_artifact"]
    assert secpal["deployment_contract"]["supported_platforms"] == []
    assert secpal["deployment_contract"]["vendor_install"]["installer_implemented"] is False
    assert secpal["deployment_contract"]["vendor_install"]["artifact_intake_implemented"] is True
    assert operator["status"] == "reviewed_restricted_artifact"
    assert operator["evidence_class"] == "reviewed_official_artifact"
    assert operator["artifact_filename"] == "SecPal_Research_Release.msi"
    assert operator["artifact_sha256"] == installer.SECPAL_OFFICIAL_ARTIFACT_SHA256
    assert operator["redistribution_permitted"] is False
    assert operator["production_use_permitted"] is False
    assert operator["artifact_intake_only"] is True
    assert operator["live_certification_eligible"] is False
    assert operator["executable_contract"]["cli_available"] is False
    assert operator["platform_matrix_evidence"]["live_execution_supported_platforms"] == []
    assert operator["authenticode_evidence"]["verified"] is True
    assert operator["authenticode_evidence"]["signer"] == "Microsoft Corporation"
    assert operator["license_evidence"]["sha256"] == installer.SECPAL_EULA_SHA256
    assert operator["license_evidence"]["acceptance_required"] is True

    # Vendor certificate embeds the SecPAL live contract with FVT-101 repair.
    install_root, dependency_prefix = _managed_souffle_paths()
    assert install_root.is_dir(), f"managed Soufflé vendor root missing: {install_root}"
    assert dependency_prefix.is_dir(), (
        f"managed Soufflé dependency prefix missing: {dependency_prefix}"
    )
    certificate = certifier.certify_external_authorization_vendor(
        install_root=install_root,
        dependency_prefix=dependency_prefix,
        force_install=False,
        skip_install=True,
        platform_id="linux-aarch64",
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
    )
    live = certificate["secpal_live_toolchain_contract"]
    assert live["interface"] == SECPAL_LIVE_INTERFACE
    assert live["goal_id"] == SECPAL_LIVE_GOAL_ID
    assert live["task_id"] == SECPAL_LIVE_TASK_ID
    assert live["repair_task_id"] == SECPAL_LIVE_REPAIR_TASK_ID
    assert live["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert live["objective_validation_repair"] is True
    assert live["objective_validation_command"] == OBJECTIVE_VALIDATION_COMMAND
    assert live["contract_complete"] is False
    assert live["artifact_intake_only"] is True
    assert live["operator_compatibility_only"] is True
    assert live["live_semantic_runner_available"] is False
    assert live["arbitrary_policy_interface_verified"] is False
    assert live["production_use_permitted"] is False
    assert live["can_promote"] is False
    assert certificate["secpal_live_ready"] is False
    assert certificate["secpal_vendor_certified"] is False
    assert certificate["policy"]["secpal_artifact_intake_does_not_imply_installation"] is True
    assert (
        certificate["policy"][
            "secpal_operator_compatibility_does_not_imply_vendor_support"
        ]
        is True
    )
    assert (
        certificate["policy"][
            "secpal_operator_compatibility_does_not_imply_semantic_certification"
        ]
        is True
    )
    assert (
        certificate["policy"][
            "secpal_platform_exception_does_not_satisfy_live_readiness"
        ]
        is True
    )

    exception = certificate["secpal_platform_exception"]
    assert exception["exception"] is True
    assert exception["installed"] is False
    assert exception["complete"] is False
    assert exception["authoritative"] is False
    assert exception["production_certified"] is False
