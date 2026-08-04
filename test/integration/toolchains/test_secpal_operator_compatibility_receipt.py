from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path

import pytest
from tools.logic import certify_secpal_operator_compatibility as cert

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = REPO_ROOT / (
    "docs/architecture/"
    "formal_verification_secpal_operator_compatibility_receipt.json"
)


def _checked_receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def _resign(receipt: dict) -> None:
    receipt["receipt_sha256"] = cert._canonical_sha256(
        cert._receipt_digest_basis(receipt)
    )


def test_checked_receipt_is_public_safe_replayable_and_strictly_nonpromotable() -> None:
    receipt = _checked_receipt()

    assert cert.validate_receipt(receipt) == []
    assert receipt["status"] == "operator_compatibility_observed_non_authoritative"
    assert receipt["scope"] == {
        "operator_compatibility_only": True,
        "completes_fvt_086": False,
        "completes_fvt_g219": False,
        "vendor_sample_suite_only": True,
    }
    assert receipt["vendor_supported_platform"] is False
    assert receipt["arbitrary_policy_interface_verified"] is False
    assert receipt["production_use_permitted"] is False
    assert receipt["live_authority"] is False
    assert receipt["deployment_ready"] is False
    assert receipt["license_evidence"][
        "research_code_not_intended_for_live_environment"
    ] is True
    assert receipt["execution_contract"]["network_or_download_requested"] is False
    assert receipt["execution_contract"]["installer_invoked"] is False
    assert receipt["execution_contract"]["raw_output_retained"] is False

    scenarios = receipt["scenarios"]
    assert [item["name"] for item in scenarios] == list(cert.SCENARIOS)
    assert len(scenarios) == 18
    assert all(item["attempt_count"] == 2 for item in scenarios)
    assert all(item["return_codes"] == [0, 0] for item in scenarios)
    assert all(item["required_markers_present"] is True for item in scenarios)
    assert all(
        item["replay_equal_after_normalization"] is True for item in scenarios
    )
    assert receipt["missing_comprehensive_cases"] == list(
        cert.MISSING_COMPREHENSIVE_CASES
    )

    rendered = json.dumps(receipt, sort_keys=True)
    assert "/tmp/" not in rendered
    assert "/home/" not in rendered
    assert "raw_stdout" not in rendered
    assert "raw_stderr" not in rendered


def test_validator_fails_closed_for_promotions_coverage_and_replay_mutations() -> None:
    baseline = _checked_receipt()
    mutations = (
        lambda value: value.__setitem__("vendor_supported_platform", True),
        lambda value: value.__setitem__(
            "arbitrary_policy_interface_verified", True
        ),
        lambda value: value.__setitem__("production_use_permitted", True),
        lambda value: value.__setitem__("live_authority", True),
        lambda value: value.__setitem__("deployment_ready", True),
        lambda value: value["scope"].__setitem__("completes_fvt_086", True),
        lambda value: value["scenarios"].pop(),
        lambda value: value["scenarios"][0].__setitem__(
            "replay_equal_after_normalization", False
        ),
        lambda value: value["scenarios"][0].__setitem__(
            "return_codes", [0, 1]
        ),
        lambda value: value.__setitem__("missing_comprehensive_cases", []),
        lambda value: value["verified_inputs"]["msi"].__setitem__(
            "sha256", "0" * 64
        ),
    )

    for mutate in mutations:
        candidate = copy.deepcopy(baseline)
        mutate(candidate)
        _resign(candidate)
        assert cert.validate_receipt(candidate), mutate

    bad_self_digest = copy.deepcopy(baseline)
    bad_self_digest["receipt_sha256"] = "0" * 64
    assert "receipt_sha256:mismatch" in cert.validate_receipt(bad_self_digest)


def test_observation_normalization_removes_only_declared_volatile_classes() -> None:
    first = b"""eventSource="/tmp/first/SecPalSamples.exe" machineId="host-a"
2026-08-03T22:31:44.319501Z
recordId="5a291a58-0530-4292-9983-c7e980ae6fdf"
<dsig:Modulus>AAAA</dsig:Modulus>
stable-policy-result
"""
    second = b"""eventSource="/tmp/other/SecPalSamples.exe" machineId="host-b"
2027-01-02T03:04:05.999Z
recordId="6b391b69-1641-43a3-a094-d8f091bf70e0"
<dsig:Modulus>BBBB</dsig:Modulus>
stable-policy-result
"""

    normalized_first = cert.normalize_observation(
        first, temporary_cwd=Path("/tmp/first")
    )
    normalized_second = cert.normalize_observation(
        second, temporary_cwd=Path("/tmp/other")
    )

    assert normalized_first == normalized_second
    assert b"stable-policy-result" in normalized_first
    assert b"AAAA" not in normalized_first
    assert b"host-a" not in normalized_first


def test_known_identity_verification_rejects_size_and_digest_mismatch(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"reviewed")
    expected = {
        "name": "candidate.bin",
        "bytes": len(b"reviewed"),
        "sha256": hashlib.sha256(b"reviewed").hexdigest(),
    }

    assert cert.verify_known_file(candidate, expected, label="candidate") == {
        **expected,
        "identity_verified": True,
    }
    candidate.write_bytes(b"mutation")
    with pytest.raises(cert.SecPALOperatorCompatibilityError, match="SHA-256"):
        cert.verify_known_file(candidate, expected, label="candidate")
    candidate.write_bytes(b"longer mutation")
    with pytest.raises(cert.SecPALOperatorCompatibilityError, match="byte-size"):
        cert.verify_known_file(candidate, expected, label="candidate")


def test_license_acceptance_is_checked_before_any_local_input() -> None:
    absent = Path("/definitely/absent/operator-input")
    with pytest.raises(
        cert.SecPALOperatorCompatibilityError,
        match="license acceptance",
    ):
        cert.certify_secpal_operator_compatibility(
            msi=absent,
            bin_dir=absent,
            eula=absent,
            mono=absent,
            mono_framework_dir=absent,
            mono_config_dir=absent,
            mono_native_lib_dir=absent,
            license_acceptance="",
        )


def test_certifier_runs_every_named_sample_twice_with_explicit_local_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "Bin"
    framework_dir = tmp_path / "mono" / "4.5"
    config_dir = tmp_path / "mono-config"
    native_dir = tmp_path / "mono-native"
    for directory in (bin_dir, framework_dir, config_dir, native_dir):
        directory.mkdir(parents=True)

    payloads = {
        "msi": (tmp_path / "SecPal_Research_Release.msi", b"msi"),
        "sample_runner": (bin_dir / "SecPalSamples.exe", b"runner"),
        "authorization_library": (
            bin_dir / "Microsoft.Research.SecPal.Dll",
            b"library",
        ),
        "audit_viewer": (bin_dir / "AuditLogViewer.exe", b"viewer"),
        "eula": (tmp_path / "EULA.rtf", b"license"),
    }
    reviewed: dict[str, dict[str, object]] = {}
    for key, (path, content) in payloads.items():
        path.write_bytes(content)
        reviewed[key] = {
            "name": path.name,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    monkeypatch.setattr(cert, "KNOWN_ARTIFACTS", reviewed)

    (framework_dir / "mscorlib.dll").write_bytes(b"framework")
    (config_dir / "config").write_text(
        '<configuration><dllmap target="$mono_libdir/libmono-native.so" />'
        "</configuration>",
        encoding="utf-8",
    )
    (native_dir / "libmono-native.so").write_bytes(b"native")
    fake_mono = tmp_path / "mono-sgen"
    fake_mono.write_text(
        """#!/usr/bin/python3
import datetime
import os
import pathlib
import sys
import uuid

if "--version" in sys.argv:
    print("Mono JIT compiler version 6.8.0.105")
    print("Architecture: arm64")
    raise SystemExit(0)

scenario = sys.argv[-1]
print("*** Policies ***")
print("*** Tokens ***")
print("*** Query ***")
print("*** Query Result ***")
print(scenario)
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
print(uuid.uuid4())
if scenario == "AuditLogScenario":
    store = pathlib.Path("%USERPROFILE%\\Application Data\\Microsoft\\SecurityPolicyStore")
    store.mkdir(parents=True)
    (store / "auditlog.xml").write_text(
        '<audit machineId="fake-host" eventSource="'
        + os.getcwd()
        + '"><when>'
        + datetime.datetime.now(datetime.timezone.utc).isoformat()
        + '</when><id>'
        + str(uuid.uuid4())
        + '</id><dsig:Modulus>RANDOM</dsig:Modulus></audit>',
        encoding="utf-8",
    )
""",
        encoding="utf-8",
    )
    fake_mono.chmod(fake_mono.stat().st_mode | stat.S_IXUSR)

    receipt = cert.certify_secpal_operator_compatibility(
        msi=payloads["msi"][0],
        bin_dir=bin_dir,
        eula=payloads["eula"][0],
        mono=fake_mono,
        mono_framework_dir=framework_dir,
        mono_config_dir=config_dir,
        mono_native_lib_dir=native_dir,
        license_acceptance=cert.LICENSE_ACCEPTANCE_PHRASE,
        timeout_seconds=2.0,
        observed_at="2026-08-03T00:00:00Z",
    )

    assert cert.validate_receipt(receipt) == []
    assert [item["name"] for item in receipt["scenarios"]] == list(
        cert.SCENARIOS
    )
    assert all(item["return_codes"] == [0, 0] for item in receipt["scenarios"])
    assert all(
        item["replay_equal_after_normalization"] is True
        for item in receipt["scenarios"]
    )
    assert receipt["scenarios"][1]["side_effect_file_count"] == 1
    assert receipt["deployment_ready"] is False
