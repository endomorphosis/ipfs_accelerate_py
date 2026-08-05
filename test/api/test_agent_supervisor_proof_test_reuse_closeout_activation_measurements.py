"""Tests for closeout activation measurement runners."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_activation_measurements import (
    discover_real_groth16_fixture,
    materialize_local_nonproduction_e2e_manifest,
    run_closeout_activation_measurements,
)


@pytest.fixture
def clear_closeout_e2e_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PTR_CLOSEOUT_LOCAL_SETUP",
        "PTR_CLOSEOUT_DEV_E2E",
        "PTR_CLOSEOUT_HEAVY_MEASUREMENTS",
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST",
        "IPFS_TEST_PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)


def test_discover_fixture_reports_binary_or_key_gap() -> None:
    result = discover_real_groth16_fixture()
    # Discovery itself must not raise; ambient trees may lack keys.
    assert result.reason
    assert isinstance(result.available, bool)
    # When binary path is set, file existence is reported honestly.
    if result.binary_path:
        from pathlib import Path

        assert Path(result.binary_path).is_file() or result.reason != "ready"


def test_measurements_skip_heavy_by_default(clear_closeout_e2e_env: None) -> None:
    report = run_closeout_activation_measurements(
        attempt_heavy_measurements=False,
        attempt_local_setup=False,
    )
    assert report.authority is False
    by_name = {item.name: item for item in report.attempts}
    assert "fixture_discover" in by_name
    assert by_name["subprocess_proof_reuse_benchmark"].skipped is True
    assert by_name["single_repo_cold_warm"].skipped is True
    assert by_name["controller_owned_context_api"].attempted is True
    assert by_name["ordinary_default_composition"].attempted is True
    assert by_name["candidate_store_path"].attempted is True
    assert by_name["reviewed_manifest_pin_status"].attempted is True
    # Heavy measured claims stay false without heavy runners.
    assert report.claims_supported.get("measured_subprocess_benchmark") is False
    assert report.claims_supported.get("three_repository_cold_warm") is False
    if by_name["ordinary_default_composition"].succeeded:
        assert report.claims_supported.get("ordinary_default_composition_usable") is True
    # Without local e2e env pins, reviewed pin is not ready even if allowlist
    # carries a development digest.
    assert by_name["reviewed_manifest_pin_status"].succeeded is False
    assert report.authority is False


def test_measurements_heavy_skip_when_keys_unavailable(
    clear_closeout_e2e_env: None,
) -> None:
    report = run_closeout_activation_measurements(
        attempt_heavy_measurements=True,
        require_available_fixture=True,
        attempt_local_setup=False,
    )
    by_name = {item.name: item for item in report.attempts}
    sub = by_name["subprocess_proof_reuse_benchmark"]
    cold = by_name["single_repo_cold_warm"]
    if report.fixture and not report.fixture.available:
        assert sub.skipped is True
        assert cold.skipped is True
        assert "fixture_" in sub.detail or "skipped" in sub.detail
    # Never grants production authority from ambient discovery alone.
    assert report.authority is False


def test_artifact_inventory_reports_existing_versions() -> None:
    from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_activation_measurements import (
        inventory_artifact_versions,
        discover_real_groth16_fixture,
    )

    fixture = discover_real_groth16_fixture()
    inventory = inventory_artifact_versions(fixture.artifacts_root)
    assert "versions" in inventory
    versions = inventory["versions"]
    assert isinstance(versions, dict)
    if "v1" in versions:
        assert versions["v1"]["complete"] is True
    if "v2" in versions:
        assert versions["v2"]["complete"] is True


def test_local_nonproduction_manifest_materialize_when_keys_present() -> None:
    fixture = discover_real_groth16_fixture()
    if not fixture.available:
        pytest.skip("local v4 keys not present")
    result = materialize_local_nonproduction_e2e_manifest(
        artifacts_root=fixture.artifacts_root or None,
        binary_path=fixture.binary_path or None,
    )
    assert result.get("ok") is True
    assert result.get("manifest_sha256")
    assert result.get("production_authority") is False
    assert result.get("local_operational_only") is True
