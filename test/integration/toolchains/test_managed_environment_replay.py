"""Managed-environment dependency/capability/platform/freshness replay.

FVT-094 / FVT-G226 — ``FormalVerificationManagedEnvironmentReplay@1``.

Acceptance covered:

* expected outputs exist (certifier, receipt, this test);
* acquisition is separately invoked and requires explicit authorization + yes;
* certification runs offline with network/download/install/ambient PATH/
  user-site/source-tree/system-package mutation disabled;
* every required external tool binds dependency, capability, platform, and
  freshness identities independently;
* Maude, OPAM, Stack, and Temurin remain non-semantic and non-authoritative;
* missing, partial, stale, relocated-without-rebinding, wrong-architecture,
  byte-mutated, and dependency-mutated trees fail only their owned axes;
* stale receipts cannot repair failures;
* installation is never semantic certification;
* sealed managed-root replay succeeds under the approved immutable deployment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_managed_environment.py"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_managed_environment_replay_receipt.json"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "FormalVerificationManagedEnvironmentReplay@1"
SCHEMA_VERSION = "formal-verification-managed-environment-replay-receipt/v1"
GOAL_ID = "FVT-G226"
TASK_ID = "FVT-094"
PROGRAM = "formal-verification-tactician/managed-environment-replay"

PRIMARY_TOOL_IDS = (
    "apalache",
    "autohyper",
    "coq",
    "eprover",
    "hyperltl",
    "isabelle",
    "mchyper",
    "proverif",
    "souffle",
    "tamarin",
    "tlc",
    "vampire",
    "ergoai",
    "runtime-mtl-external",
)
SUPPORT_TOOL_IDS = (
    "maude",
    "opam",
    "stack",
    "temurin-jdk",
)
REQUIRED_TOOL_IDS = PRIMARY_TOOL_IDS + SUPPORT_TOOL_IDS
REPLAY_AXES = ("dependency", "capability", "platform", "freshness")
FAILURE_CLASSES = (
    "missing",
    "partial",
    "stale",
    "relocated_without_rebinding",
    "wrong_architecture",
    "byte_mutated",
    "dependency_mutated",
)

DEFAULT_SEALED_ROOT = Path(
    "/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers"
)


def _load_module():
    assert CERTIFIER_PATH.is_file(), f"missing expected output: {CERTIFIER_PATH}"
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    name = "tools_logic_certify_formal_verification_managed_environment"
    spec = importlib.util.spec_from_file_location(name, CERTIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load_module()


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing lock: {LOCK_PATH}"
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def receipt_document() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"missing expected output: {RECEIPT_PATH}"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def sealed_root() -> Path | None:
    configured = os.environ.get("IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT", "").strip()
    candidate = Path(configured) if configured else DEFAULT_SEALED_ROOT
    if candidate.is_dir() and (candidate / "bin").is_dir():
        return candidate
    return None


@pytest.fixture(scope="module")
def live_receipt(certifier, sealed_root) -> dict[str, Any]:
    return certifier.certify_managed_environment_replay(
        repo_root=REPO_ROOT,
        managed_root=sealed_root,
        require_approved_immutable=sealed_root is not None
        and str(sealed_root).startswith("/opt/"),
    )


# ---------------------------------------------------------------------------
# Expected outputs / surface constants
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert CERTIFIER_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    assert Path(__file__).is_file()
    assert LOCK_PATH.is_file()


def test_module_surface_constants(certifier) -> None:
    assert certifier.INTERFACE == INTERFACE
    assert certifier.SCHEMA_VERSION == SCHEMA_VERSION
    assert certifier.GOAL_ID == GOAL_ID
    assert certifier.TASK_ID == TASK_ID
    assert certifier.PROGRAM == PROGRAM
    assert tuple(certifier.PRIMARY_TOOL_IDS) == PRIMARY_TOOL_IDS
    assert tuple(certifier.SUPPORT_TOOL_IDS) == SUPPORT_TOOL_IDS
    assert tuple(certifier.REQUIRED_TOOL_IDS) == REQUIRED_TOOL_IDS
    assert tuple(certifier.REPLAY_AXES) == REPLAY_AXES
    assert set(certifier.FAILURE_CLASS_AXES) == set(FAILURE_CLASSES)
    assert (
        certifier.DEFAULT_RECEIPT_RELATIVE.as_posix()
        == "docs/architecture/formal_verification_managed_environment_replay_receipt.json"
    )


# ---------------------------------------------------------------------------
# Acquisition phase boundary
# ---------------------------------------------------------------------------


def test_acquisition_not_run_by_default(certifier, lock) -> None:
    phase = certifier.run_acquisition_phase(lock=lock)
    assert phase["status"] == "not_run"
    assert phase["authorized"] is False
    assert phase["installed"] is False
    assert "acquisition_not_authorized" in phase["reason_codes"]
    policy = phase["policy"]
    assert policy["requires_explicit_authorization"] is True
    assert policy["requires_explicit_yes"] is True
    assert policy["never_during_offline_certification"] is True
    assert policy["installation_is_not_semantic_certification"] is True
    assert policy["user_local_only"] is True
    assert policy["single_flight"] is True
    assert policy["symlink_safe"] is True
    assert policy["atomic_publication"] is True
    assert policy["rollback_preserving"] is True


def test_acquisition_requires_explicit_yes(certifier, lock) -> None:
    phase = certifier.run_acquisition_phase(
        lock=lock,
        authorize_acquisition=True,
        yes=False,
    )
    assert phase["status"] == "blocked"
    assert phase["authorized"] is False
    assert phase["installed"] is False
    assert "explicit_yes_required" in phase["reason_codes"]


def test_acquisition_authorization_validates_policy_without_install(
    certifier, lock
) -> None:
    phase = certifier.run_acquisition_phase(
        lock=lock,
        authorize_acquisition=True,
        yes=True,
    )
    assert phase["status"] == "authorized_policy_validated"
    assert phase["authorized"] is True
    assert phase["installed"] is False
    assert phase["publication_properties"]["user_local"] is True
    assert phase["publication_properties"]["single_flight"] is True
    assert phase["publication_properties"]["symlink_safe"] is True
    assert phase["publication_properties"]["atomic"] is True
    assert phase["publication_properties"]["rollback_preserving"] is True
    reviewed = phase["reviewed_inputs"]
    assert reviewed["immutable_urls"] is True
    assert reviewed["versions"] is True
    assert reviewed["sizes"] is True
    assert reviewed["checksums"] is True
    assert reviewed["signatures_or_publisher_evidence"] is True
    assert reviewed["licenses"] is True
    assert reviewed["os_architecture_pins"] is True


# ---------------------------------------------------------------------------
# Offline certification policy
# ---------------------------------------------------------------------------


def test_offline_env_blocks_install_and_network(certifier) -> None:
    env = certifier.offline_env({"PATH": "/usr/bin", "HOME": "/tmp/x"})
    assert env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_INSTALL"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_NETWORK"] == "1"
    assert env["FORMAL_VERIFICATION_MANAGED_ENVIRONMENT_REPLAY_OFFLINE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PIP_NO_INDEX"] == "1"
    assert env["NPM_CONFIG_OFFLINE"] == "true"


def test_certify_never_installs_or_mutates_source_tree(
    certifier, tmp_path: Path
) -> None:
    # Point managed root at an empty fixture so certification remains offline.
    fixture_root = tmp_path / "managed"
    (fixture_root / "bin").mkdir(parents=True)
    before = {
        path.relative_to(REPO_ROOT): path.stat().st_mtime_ns
        for path in (
            LOCK_PATH,
            CERTIFIER_PATH,
        )
        if path.is_file()
    }

    receipt = certifier.certify_managed_environment_replay(
        repo_root=REPO_ROOT,
        managed_root=fixture_root,
        require_approved_immutable=False,
    )
    phase = receipt["certification_phase"]
    assert phase["offline"] is True
    assert phase["network"] is False
    assert phase["download"] is False
    assert phase["install"] is False
    assert phase["ambient_path_mutated"] is False
    assert phase["user_site_mutated"] is False
    assert phase["source_tree_mutated"] is False
    assert phase["system_package_mutated"] is False
    assert receipt["semantic_certification"] is False
    assert receipt["installation_is_semantic_certification"] is False
    assert receipt["acquisition_phase"]["installed"] is False
    for relative, mtime in before.items():
        assert (REPO_ROOT / relative).stat().st_mtime_ns == mtime


def test_receipt_policy_flags(certifier) -> None:
    receipt = certifier.certify_managed_environment_replay(
        repo_root=REPO_ROOT,
        managed_root=None,
        require_approved_immutable=True,
        env={},
    )
    policy = receipt["policy"]
    for key in (
        "offline_certification_forbids_network",
        "offline_certification_forbids_download",
        "offline_certification_forbids_install",
        "offline_certification_forbids_ambient_path_mutation",
        "offline_certification_forbids_user_site_mutation",
        "offline_certification_forbids_source_tree_mutation",
        "offline_certification_forbids_system_package_mutation",
        "acquisition_is_separately_invoked",
        "acquisition_requires_explicit_authorization",
        "installation_is_not_semantic_certification",
        "support_dependencies_non_semantic",
        "support_dependencies_non_authoritative",
        "stale_receipts_cannot_repair_failures",
        "axes_do_not_inherit_success",
    ):
        assert policy[key] is True, key


# ---------------------------------------------------------------------------
# Inventory / axis structure
# ---------------------------------------------------------------------------


def test_every_required_tool_has_independent_axes(live_receipt) -> None:
    tools = live_receipt["tools"]
    assert set(tools) >= set(REQUIRED_TOOL_IDS)
    for tool_id in REQUIRED_TOOL_IDS:
        row = tools[tool_id]
        assert set(row["axes"]) == set(REPLAY_AXES)
        for axis_name in REPLAY_AXES:
            axis = row["axes"][axis_name]
            assert axis["status"] in {"ready", "blocked", "not_applicable"}
            assert axis["required"] is True
            assert isinstance(axis["reason_codes"], list)
        assert row["grants_semantic_certification"] is False
        identities = row["identities"]
        assert "executable_digest_sha256" in identities
        assert "artifact_digest_sha256" in identities
        assert "runtime_digest_sha256" in identities
        assert "lock_version" in identities
        assert "host_platform" in identities


def test_support_dependencies_are_non_semantic_and_non_authoritative(
    live_receipt,
) -> None:
    for tool_id in SUPPORT_TOOL_IDS:
        row = live_receipt["tools"][tool_id]
        assert row["support_only"] is True
        assert row["non_semantic"] is True
        assert row["non_authoritative"] is True
        assert row["authority_tool"] is False
        assert row["grants_semantic_certification"] is False


def test_primary_tools_are_authority_tools_not_support(live_receipt) -> None:
    for tool_id in PRIMARY_TOOL_IDS:
        row = live_receipt["tools"][tool_id]
        assert row["support_only"] is False
        assert row["authority_tool"] is True
        # Still never grants semantic certification from this surface.
        assert row["grants_semantic_certification"] is False


# ---------------------------------------------------------------------------
# Failure-class isolation
# ---------------------------------------------------------------------------


def test_failure_classes_block_only_owned_axes(certifier) -> None:
    host = certifier.observed_host_platform()
    baseline = certifier.ToolBindingObservation(
        tool_id="vampire",
        display_name="Vampire",
        support_only=False,
        authority_tool=True,
        lock_present=True,
        pin={
            "version": "5.0.1",
            "platform": host,
            "sha256": "a" * 64,
            "artifact_url": "https://example.invalid/vampire",
            "license": "BSD-3-Clause",
        },
        host_platform=host,
        platform_supported=True,
        executable_path=f"/opt/managed/bin/vampire",
        executable_basename="vampire",
        executable_digest_sha256="b" * 64,
        artifact_path="/opt/managed/vampire-bin",
        artifact_digest_sha256="c" * 64,
        runtime_path="/opt/managed/vampire-bin",
        runtime_digest_sha256="c" * 64,
        artifact_kind="native_binary",
        under_managed_root=True,
        under_approved_immutable_root=True,
        lock_version="5.0.1",
        lock_artifact_sha256="a" * 64,
        lock_platform=host,
        lock_license="BSD-3-Clause",
        lock_source="https://example.invalid/vampire",
        lock_artifact_url="https://example.invalid/vampire",
    )
    baseline_eval = certifier.evaluate_tool_axes(baseline)
    assert baseline_eval["ready"] is True

    for failure_class in FAILURE_CLASSES:
        isolation = certifier.evaluate_failure_class_isolation(baseline, failure_class)
        assert isolation["owned_axis"] == certifier.FAILURE_CLASS_AXES[failure_class]
        assert isolation["owned_axis_blocked"] is True
        assert isolation["readiness_failed"] is True
        assert isolation["mutated_ready"] is False
        assert isolation["stale_receipt_cannot_repair"] is True
        assert isolation["isolated"] is True
        owned = isolation["owned_axis"]
        assert isolation["mutated_axes"][owned]["status"] == "blocked"


def test_stale_receipt_cannot_repair_mutated_binding(certifier) -> None:
    host = certifier.observed_host_platform()
    observation = certifier.ToolBindingObservation(
        tool_id="tlc",
        display_name="TLC",
        support_only=False,
        authority_tool=True,
        lock_present=True,
        pin={"version": "1.8.0", "platform": "any", "sha256": "d" * 64},
        host_platform=host,
        platform_supported=True,
        executable_path="/opt/managed/bin/tlc",
        executable_basename="tlc",
        executable_digest_sha256="e" * 64,
        artifact_path="/opt/managed/tlc.jar",
        artifact_digest_sha256="f" * 64,
        runtime_path="/opt/managed/tlc.jar",
        runtime_digest_sha256="f" * 64,
        artifact_kind="regular_file",
        under_managed_root=True,
        under_approved_immutable_root=True,
        lock_version="1.8.0",
        lock_artifact_sha256="d" * 64,
        lock_platform="any",
        lock_license="MIT",
        lock_source="https://example.invalid/tlc",
        lock_artifact_url="https://example.invalid/tlc",
    )
    good = certifier.evaluate_tool_axes(observation)
    stale = certifier.evaluate_tool_axes(
        certifier.apply_failure_class_to_observation(observation, "stale")
    )
    byte_mutated = certifier.evaluate_tool_axes(
        certifier.apply_failure_class_to_observation(observation, "byte_mutated")
    )
    assert good["ready"] is True
    assert stale["ready"] is False
    assert stale["axes"]["freshness"]["status"] == "blocked"
    assert byte_mutated["ready"] is False
    assert byte_mutated["axes"]["freshness"]["status"] == "blocked"
    # A previously-good receipt payload cannot flip a failed binding to ready.
    repaired = dict(good)
    repaired["ready"] = True
    repaired["axes"] = byte_mutated["axes"]
    assert repaired["axes"]["freshness"]["status"] == "blocked"
    assert not (
        repaired["ready"] is True
        and all(axis["status"] == "ready" for axis in repaired["axes"].values())
    )


def test_missing_tool_blocks_capability_only_when_dependency_platform_present(
    certifier, tmp_path: Path
) -> None:
    fixture_root = tmp_path / "managed"
    (fixture_root / "bin").mkdir(parents=True)
    receipt = certifier.certify_managed_environment_replay(
        repo_root=REPO_ROOT,
        managed_root=fixture_root,
        require_approved_immutable=False,
    )
    # Vampire is required and lock-present; empty bin means capability blocked.
    vampire = receipt["tools"]["vampire"]
    assert vampire["ready"] is False
    assert vampire["axes"]["capability"]["status"] == "blocked"
    assert vampire["axes"]["dependency"]["status"] == "ready"
    assert vampire["axes"]["platform"]["status"] == "ready"


# ---------------------------------------------------------------------------
# Checked-in receipt contract
# ---------------------------------------------------------------------------


def test_checked_in_receipt_matches_contract(receipt_document, certifier) -> None:
    failures = certifier.validate_receipt(receipt_document)
    assert failures == []
    assert receipt_document["interface"] == INTERFACE
    assert receipt_document["schema_version"] == SCHEMA_VERSION
    assert receipt_document["goal_id"] == GOAL_ID
    assert receipt_document["task_id"] == TASK_ID
    assert receipt_document["semantic_certification"] is False
    assert receipt_document["installation_is_semantic_certification"] is False
    assert set(receipt_document["tools"]) >= set(REQUIRED_TOOL_IDS)
    assert set(receipt_document["failure_class_isolation"]) == set(FAILURE_CLASSES)
    for failure_class, row in receipt_document["failure_class_isolation"].items():
        assert row["owned_axis"] == certifier.FAILURE_CLASS_AXES[failure_class]
        assert row["stale_receipt_cannot_repair"] is True
    for tool_id in SUPPORT_TOOL_IDS:
        row = receipt_document["tools"][tool_id]
        assert row["support_only"] is True
        assert row["non_semantic"] is True
        assert row["non_authoritative"] is True


def test_checked_in_receipt_digest_is_bound(receipt_document) -> None:
    digest = receipt_document.get("receipt_digest_sha256") or ""
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# Live sealed managed-root replay
# ---------------------------------------------------------------------------


def test_live_sealed_root_binds_required_tools(certifier, sealed_root, live_receipt) -> None:
    if sealed_root is None:
        pytest.skip("sealed managed prover root unavailable")
    assert live_receipt["managed_root_present"] is True
    assert live_receipt["certification_phase"]["offline"] is True
    assert live_receipt["semantic_certification"] is False

    # Every required tool must have a lock pin and an executable under the root.
    missing = []
    for tool_id in REQUIRED_TOOL_IDS:
        row = live_receipt["tools"][tool_id]
        identities = row["identities"]
        if not identities.get("executable_basename"):
            missing.append(tool_id)
            continue
        assert identities.get("under_managed_root") is True
        assert identities.get("executable_digest_sha256")
        assert identities.get("lock_version")
    assert missing == [], f"tools missing executables under sealed root: {missing}"

    # Support tools stay non-authoritative even when present.
    for tool_id in SUPPORT_TOOL_IDS:
        assert live_receipt["tools"][tool_id]["non_authoritative"] is True

    # When the sealed root is the approved immutable deployment, production
    # binding readiness should succeed.
    if str(sealed_root).startswith("/opt/"):
        assert live_receipt["managed_root_approved_immutable"] is True
        assert live_receipt["certified"] is True
        assert live_receipt["production_bindings_ready"] is True
        assert live_receipt["summary"]["blocked_tools"] == []


def test_relocated_tree_without_rebinding_fails_capability(
    certifier, sealed_root, tmp_path: Path
) -> None:
    if sealed_root is None:
        pytest.skip("sealed managed prover root unavailable")
    # A path outside the managed root cannot satisfy capability by ambient PATH.
    foreign = tmp_path / "foreign-bin"
    foreign.mkdir()
    vampire = foreign / "vampire"
    vampire.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    vampire.chmod(0o755)

    # Managed root is empty; ambient foreign binary must not be used.
    empty_root = tmp_path / "empty-managed"
    (empty_root / "bin").mkdir(parents=True)
    receipt = certifier.certify_managed_environment_replay(
        repo_root=REPO_ROOT,
        managed_root=empty_root,
        require_approved_immutable=False,
        env={"PATH": str(foreign)},
    )
    row = receipt["tools"]["vampire"]
    assert row["ready"] is False
    assert row["axes"]["capability"]["status"] == "blocked"
    assert row["identities"]["executable"] is None or row["identities"][
        "under_managed_root"
    ] is not True


def test_cli_help_and_offline_default(
    certifier,
) -> None:
    result = subprocess.run(
        [sys.executable, str(CERTIFIER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=certifier.offline_env(os.environ),
    )
    assert result.returncode == 0
    assert "FormalVerificationManagedEnvironmentReplay" in result.stdout or "managed" in result.stdout.lower()
