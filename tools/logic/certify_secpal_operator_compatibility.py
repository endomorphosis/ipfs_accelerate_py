#!/usr/bin/env python3
"""Offline, fail-closed SecPAL operator-compatibility certification.

This certifier is intentionally narrower than a SecPAL toolchain or production
certification.  It proves that the 18 samples in the pinned Microsoft research
release can be executed twice by an explicitly supplied local Mono runtime and
that the normalized observations replay.  It does not download or install
anything, redistribute the Microsoft payload, certify an arbitrary-policy
adapter, make Mono a Microsoft-supported platform, or grant live authority.

The caller must supply the already-acquired MSI, its extracted ``Bin``
directory, the extracted EULA, and every Mono runtime location.  The exact
license-acceptance phrase is required before any artifact is inspected or
executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import escape as xml_escape

SCHEMA_VERSION: Final = "secpal-operator-compatibility-receipt/v1"
INTERFACE: Final = "SecPALOperatorCompatibilityReceipt@1"
CERTIFICATION_CLASS: Final = "operator_compatibility_only"
LICENSE_ACCEPTANCE_PHRASE: Final = (
    "ACCEPT-MICROSOFT-SECPAL-RESEARCH-TERMS"
)
DEFAULT_TIMEOUT_SECONDS: Final = 15.0
MAX_TIMEOUT_SECONDS: Final = 60.0
MAX_CAPTURE_BYTES: Final = 2 * 1024 * 1024

# These identities bind the exact archived Microsoft release reviewed by an
# operator.  The MSI's Microsoft Authenticode signature was verified during
# provenance review; this script binds that reviewed identity by digest and
# deliberately does not pretend to re-run platform trust validation.
KNOWN_ARTIFACTS: Final[Mapping[str, Mapping[str, Any]]] = {
    "msi": {
        "name": "SecPal_Research_Release.msi",
        "bytes": 2_458_624,
        "sha256": (
            "c1988b9f1f6a2fb602bac4fc777a1765e59e74126285a095684a4743ea683159"
        ),
    },
    "sample_runner": {
        "name": "SecPalSamples.exe",
        "bytes": 55_368,
        "sha256": (
            "9a46e3bbf7bc58f0b9964814def3dd1224cd3893bfc34c72c13f3b8190599a07"
        ),
    },
    "authorization_library": {
        "name": "Microsoft.Research.SecPal.Dll",
        "bytes": 403_528,
        "sha256": (
            "2afacfc4121332c7fec5df32911e9a8b8d9926807af15d8c71b41a2928ee8b0a"
        ),
    },
    "audit_viewer": {
        "name": "AuditLogViewer.exe",
        "bytes": 321_608,
        "sha256": (
            "46a85221d2ab794b2b36e2f491c918edc35778a8bdafcd3dcbddb4aa18bdab92"
        ),
    },
    "eula": {
        "name": "EULA.rtf",
        "bytes": 92_171,
        "sha256": (
            "de075e7848fb737b9da3cfec5ce7c906742f4767fa04ed2bc38e69e2dd5e4fad"
        ),
    },
}

SCENARIOS: Final[tuple[str, ...]] = (
    "AttributeScenario",
    "AuditLogScenario",
    "AuthorizationQueryTemplateScenario",
    "CanActAsScenario",
    "CapabilityScenario",
    "CodeAuthorizationScenario",
    "DelegationScenario",
    "ExclusionScenario",
    "ExistsQuantifierScenario",
    "PrincipalIdentityScenario",
    "RevocationScenario",
    "RoleBasedScenario",
    "RoleExclusionScenario",
    "SerializedScenario",
    "TwoLevelDelegationScenario",
    "TwoManScenario",
    "UserAndApplicationScenario",
    "UserOrApplicationScenario",
)

REQUIRED_OUTPUT_MARKERS: Final[tuple[str, ...]] = (
    "*** Policies ***",
    "*** Tokens ***",
    "*** Query ***",
    "*** Query Result ***",
)

MISSING_COMPREHENSIVE_CASES: Final[tuple[str, ...]] = (
    "arbitrary policy and query submission through the public Logic API",
    "expected-denial and no-proof outcomes independent of vendor samples",
    "malformed policy, malformed query, and parser diagnostic behavior",
    "adversarial delegation cycles, depth limits, and resource exhaustion",
    "deterministic temporal-boundary behavior for DateTime.UtcNow sample facts",
    "cross-runtime conformance on a Microsoft-supported Windows and .NET Framework host",
    "concurrent and persistent audit-store isolation and recovery",
    "lazy-installer lifecycle, rollback, and clean-install public-package behavior",
    "license and patent review authorizing any production deployment",
    "vendor-supported Linux and arm64 execution",
)

_HEX_64_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE: Final = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_ISO_UTC_RE: Final = re.compile(
    r"(?<![0-9])\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})"
)
_MACHINE_ID_RE: Final = re.compile(r'(?<=machineId=")[^"]*(?=")')
_XML_SECRET_VALUE_RE: Final = re.compile(
    r"(<(?:[A-Za-z_][\w.-]*:)?(?:Modulus|SignatureValue)>).*?"
    r"(</(?:[A-Za-z_][\w.-]*:)?(?:Modulus|SignatureValue)>)",
    re.DOTALL,
)
_MONO_VERSION_RE: Final = re.compile(
    r"Mono JIT compiler version\s+([^\s]+)", re.IGNORECASE
)
_MONO_ARCH_RE: Final = re.compile(r"^\s*Architecture:\s*(\S+)\s*$", re.MULTILINE)


class SecPALOperatorCompatibilityError(ValueError):
    """Raised when certification cannot establish its narrow contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


def _resolved_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SecPALOperatorCompatibilityError(
            f"{label} does not resolve to a local file: {path}"
        ) from exc
    if not resolved.is_file():
        raise SecPALOperatorCompatibilityError(
            f"{label} is not a local regular file: {path}"
        )
    return resolved


def _resolved_dir(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SecPALOperatorCompatibilityError(
            f"{label} does not resolve to a local directory: {path}"
        ) from exc
    if not resolved.is_dir():
        raise SecPALOperatorCompatibilityError(
            f"{label} is not a local directory: {path}"
        )
    return resolved


def verify_known_file(
    path: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Verify a pinned file and return only public identity metadata."""

    resolved = _resolved_file(path, label)
    observed_bytes = resolved.stat().st_size
    expected_bytes = int(expected["bytes"])
    if observed_bytes != expected_bytes:
        raise SecPALOperatorCompatibilityError(
            f"{label} byte-size mismatch: expected {expected_bytes}, "
            f"observed {observed_bytes}"
        )
    observed_sha256 = _sha256_file(resolved)
    expected_sha256 = str(expected["sha256"])
    if observed_sha256 != expected_sha256:
        raise SecPALOperatorCompatibilityError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return {
        "name": str(expected["name"]),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "identity_verified": True,
    }


def normalize_observation(
    payload: bytes,
    *,
    temporary_cwd: Path | None = None,
) -> bytes:
    """Normalize volatile sample output without preserving raw observations."""

    text = payload.decode("utf-8-sig", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if temporary_cwd is not None:
        cwd = str(temporary_cwd.resolve())
        text = text.replace(cwd, "<TEMP_CWD>")
        text = text.replace(cwd.replace("/", "\\"), "<TEMP_CWD>")
    text = _MACHINE_ID_RE.sub("<MACHINE>", text)
    text = _UUID_RE.sub("<UUID>", text)
    text = _ISO_UTC_RE.sub("<UTC_TIMESTAMP>", text)
    text = _XML_SECRET_VALUE_RE.sub(r"\1<NORMALIZED_CRYPTO_VALUE>\2", text)
    # A final newline avoids platform-dependent trailing-newline drift.
    return (text.rstrip("\n") + "\n").encode("utf-8")


def _bounded_run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise SecPALOperatorCompatibilityError(
            f"bounded execution exceeded {timeout_seconds:.3f} seconds"
        ) from exc
    if len(result.stdout) > MAX_CAPTURE_BYTES or len(result.stderr) > MAX_CAPTURE_BYTES:
        raise SecPALOperatorCompatibilityError(
            f"bounded execution exceeded {MAX_CAPTURE_BYTES} captured bytes"
        )
    return result


def _runtime_environment(
    *,
    framework_dir: Path,
    native_lib_dir: Path,
    attempt_dir: Path,
) -> dict[str, str]:
    # Deliberately do not inherit credentials, proxies, user-specific paths, or
    # a caller PATH.  Every executed program is addressed by a resolved path.
    tmp_dir = attempt_dir / "_runtime_tmp"
    registry_dir = attempt_dir / "_mono_registry"
    tmp_dir.mkdir()
    registry_dir.mkdir()
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "TMPDIR": str(tmp_dir),
        "MONO_PATH": str(framework_dir),
        "MONO_REGISTRY_PATH": str(registry_dir),
        "LD_LIBRARY_PATH": str(native_lib_dir),
    }


def _prepare_runtime_config(
    *,
    config_dir: Path,
    native_lib_dir: Path,
    destination: Path,
) -> Path:
    shutil.copytree(config_dir, destination)
    config_file = destination / "config"
    if not config_file.is_file():
        raise SecPALOperatorCompatibilityError(
            "Mono config directory does not contain a config file"
        )
    text = config_file.read_text(encoding="utf-8")
    # Debian's relocatable config expands $mono_libdir to the compiled prefix.
    # Replace it only in the ephemeral copy so an explicitly supplied extracted
    # runtime remains hermetic and the operator's input is never modified.
    text = text.replace("$mono_libdir", xml_escape(str(native_lib_dir)))
    config_file.write_text(text, encoding="utf-8")
    return config_file


def _side_effect_manifest(
    attempt_dir: Path,
    *,
    excluded_names: frozenset[str],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(attempt_dir.rglob("*")):
        if not path.is_file() or path.name in excluded_names:
            continue
        payload = path.read_bytes()
        if len(payload) > MAX_CAPTURE_BYTES:
            raise SecPALOperatorCompatibilityError(
                "sample side effect exceeded the per-file capture limit"
            )
        normalized = normalize_observation(payload, temporary_cwd=attempt_dir)
        entries.append(
            {
                "relative_name": path.relative_to(attempt_dir).as_posix(),
                "bytes": len(payload),
                "normalized_sha256": _sha256_bytes(normalized),
            }
        )
    return {
        "file_count": len(entries),
        "manifest_sha256": _canonical_sha256(entries),
    }


def _run_scenario_once(
    *,
    scenario: str,
    mono: Path,
    runtime_config: Path,
    framework_dir: Path,
    native_lib_dir: Path,
    sample_runner: Path,
    authorization_library: Path,
    timeout_seconds: float,
    suite_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    attempt_dir = suite_dir / f"{scenario}-attempt-{attempt}"
    attempt_dir.mkdir()
    runner_copy = attempt_dir / "SecPalSamples.exe"
    # Mono's Linux assembly resolver is case-sensitive, while the reviewed MSI
    # stores the library with a capitalized .Dll suffix.
    library_copy = attempt_dir / "Microsoft.Research.SecPal.dll"
    shutil.copy2(sample_runner, runner_copy)
    shutil.copy2(authorization_library, library_copy)
    env = _runtime_environment(
        framework_dir=framework_dir,
        native_lib_dir=native_lib_dir,
        attempt_dir=attempt_dir,
    )
    result = _bounded_run(
        [
            str(mono),
            "--config",
            str(runtime_config),
            "./SecPalSamples.exe",
            scenario,
        ],
        cwd=attempt_dir,
        env=env,
        timeout_seconds=timeout_seconds,
    )
    normalized_stdout = normalize_observation(
        result.stdout, temporary_cwd=attempt_dir
    )
    normalized_stderr = normalize_observation(
        result.stderr, temporary_cwd=attempt_dir
    )
    decoded_stdout = normalized_stdout.decode("utf-8", errors="replace")
    markers_present = all(
        marker in decoded_stdout for marker in REQUIRED_OUTPUT_MARKERS
    )
    side_effects = _side_effect_manifest(
        attempt_dir,
        excluded_names=frozenset(
            {runner_copy.name, library_copy.name}
        ),
    )
    observation = {
        "return_code": int(result.returncode),
        "stdout_bytes": len(result.stdout),
        "stderr_bytes": len(result.stderr),
        "normalized_stdout_sha256": _sha256_bytes(normalized_stdout),
        "normalized_stderr_sha256": _sha256_bytes(normalized_stderr),
        "required_markers_present": markers_present,
        "side_effect_file_count": side_effects["file_count"],
        "side_effect_manifest_sha256": side_effects["manifest_sha256"],
    }
    observation["normalized_observation_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in observation.items()
            if key not in {"stdout_bytes", "stderr_bytes"}
        }
    )
    return observation


def _scenario_receipt(
    *,
    scenario: str,
    mono: Path,
    runtime_config: Path,
    framework_dir: Path,
    native_lib_dir: Path,
    sample_runner: Path,
    authorization_library: Path,
    timeout_seconds: float,
    suite_dir: Path,
) -> dict[str, Any]:
    attempts = [
        _run_scenario_once(
            scenario=scenario,
            mono=mono,
            runtime_config=runtime_config,
            framework_dir=framework_dir,
            native_lib_dir=native_lib_dir,
            sample_runner=sample_runner,
            authorization_library=authorization_library,
            timeout_seconds=timeout_seconds,
            suite_dir=suite_dir,
            attempt=attempt,
        )
        for attempt in (1, 2)
    ]
    for index, observation in enumerate(attempts, start=1):
        if observation["return_code"] != 0:
            raise SecPALOperatorCompatibilityError(
                f"{scenario} attempt {index} returned "
                f"{observation['return_code']}"
            )
        if not observation["required_markers_present"]:
            raise SecPALOperatorCompatibilityError(
                f"{scenario} attempt {index} omitted required sample markers"
            )
    replay_fields = (
        "return_code",
        "normalized_stdout_sha256",
        "normalized_stderr_sha256",
        "required_markers_present",
        "side_effect_file_count",
        "side_effect_manifest_sha256",
        "normalized_observation_sha256",
    )
    replay_equal = all(
        attempts[0][field] == attempts[1][field] for field in replay_fields
    )
    if not replay_equal:
        raise SecPALOperatorCompatibilityError(
            f"{scenario} did not replay after approved normalization"
        )
    return {
        "name": scenario,
        "attempt_count": 2,
        "return_codes": [attempt["return_code"] for attempt in attempts],
        "required_markers_present": True,
        "replay_equal_after_normalization": True,
        "normalized_stdout_sha256": attempts[0]["normalized_stdout_sha256"],
        "normalized_stderr_sha256": attempts[0]["normalized_stderr_sha256"],
        "normalized_observation_sha256": attempts[0][
            "normalized_observation_sha256"
        ],
        "side_effect_file_count": attempts[0]["side_effect_file_count"],
        "side_effect_manifest_sha256": attempts[0][
            "side_effect_manifest_sha256"
        ],
    }


def _runtime_identity(
    *,
    mono: Path,
    framework_dir: Path,
    config_dir: Path,
    native_lib_dir: Path,
    runtime_config: Path,
    timeout_seconds: float,
    suite_dir: Path,
) -> dict[str, Any]:
    mscorlib = _resolved_file(framework_dir / "mscorlib.dll", "Mono mscorlib")
    original_config = _resolved_file(config_dir / "config", "Mono config")
    native_library = _resolved_file(
        native_lib_dir / "libmono-native.so", "Mono native library"
    )
    probe_dir = suite_dir / "runtime-probe"
    probe_dir.mkdir()
    env = _runtime_environment(
        framework_dir=framework_dir,
        native_lib_dir=native_lib_dir,
        attempt_dir=probe_dir,
    )
    result = _bounded_run(
        [str(mono), "--config", str(runtime_config), "--version"],
        cwd=probe_dir,
        env=env,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise SecPALOperatorCompatibilityError(
            f"Mono version probe returned {result.returncode}"
        )
    output = (result.stdout + b"\n" + result.stderr).decode(
        "utf-8", errors="replace"
    )
    version_match = _MONO_VERSION_RE.search(output)
    if not version_match:
        raise SecPALOperatorCompatibilityError(
            "runtime version probe did not identify Mono"
        )
    architecture_match = _MONO_ARCH_RE.search(output)
    return {
        "implementation": "Mono",
        "version": version_match.group(1),
        "reported_architecture": (
            architecture_match.group(1) if architecture_match else "unreported"
        ),
        "host_system": platform.system().casefold(),
        "host_machine": platform.machine().casefold(),
        "mono_executable_sha256": _sha256_file(mono),
        "mscorlib_sha256": _sha256_file(mscorlib),
        "mono_config_sha256": _sha256_file(original_config),
        "mono_native_library_sha256": _sha256_file(native_library),
        "runtime_identity_recorded": True,
        "vendor_supported_runtime": False,
    }


def _public_safety_failures(value: Any, *, key_path: str = "$") -> list[str]:
    failures: list[str] = []
    forbidden_keys = {
        "raw_stdout",
        "raw_stderr",
        "command",
        "environment",
        "cwd",
        "local_path",
        "username",
        "hostname",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{key_path}.{key}"
            if str(key).casefold() in forbidden_keys:
                failures.append(f"{child_path}:forbidden_public_field")
            failures.extend(_public_safety_failures(child, key_path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(
                _public_safety_failures(child, key_path=f"{key_path}[{index}]")
            )
    elif isinstance(value, str):
        if re.search(r"(?:/home/|/tmp/|[A-Za-z]:\\Users\\)", value):
            failures.append(f"{key_path}:local_path_leak")
    return failures


def _receipt_digest_basis(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Return every structural failure; success is an empty list."""

    failures: list[str] = []
    if receipt.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if receipt.get("interface") != INTERFACE:
        failures.append("interface")
    if receipt.get("certification_class") != CERTIFICATION_CLASS:
        failures.append("certification_class")
    for field in (
        "vendor_supported_platform",
        "arbitrary_policy_interface_verified",
        "production_use_permitted",
        "live_authority",
        "deployment_ready",
    ):
        if receipt.get(field) is not False:
            failures.append(f"{field}:must_be_false")

    scope = receipt.get("scope")
    if not isinstance(scope, Mapping):
        failures.append("scope")
    else:
        if scope.get("operator_compatibility_only") is not True:
            failures.append("scope.operator_compatibility_only")
        if scope.get("completes_fvt_086") is not False:
            failures.append("scope.completes_fvt_086")
        if scope.get("completes_fvt_g219") is not False:
            failures.append("scope.completes_fvt_g219")

    inputs = receipt.get("verified_inputs")
    if not isinstance(inputs, Mapping):
        failures.append("verified_inputs")
    else:
        for key, expected in KNOWN_ARTIFACTS.items():
            item = inputs.get(key)
            if not isinstance(item, Mapping):
                failures.append(f"verified_inputs.{key}")
                continue
            for identity_field in ("name", "bytes", "sha256"):
                if item.get(identity_field) != expected[identity_field]:
                    failures.append(
                        f"verified_inputs.{key}.{identity_field}"
                    )
            if item.get("identity_verified") is not True:
                failures.append(f"verified_inputs.{key}.identity_verified")

    license_evidence = receipt.get("license_evidence")
    if not isinstance(license_evidence, Mapping):
        failures.append("license_evidence")
    else:
        for field in (
            "explicit_acceptance_supplied",
            "research_code_not_intended_for_live_environment",
            "software_redistribution_permitted_by_receipt",
            "production_use_authorized_by_receipt",
        ):
            expected = field in {
                "explicit_acceptance_supplied",
                "research_code_not_intended_for_live_environment",
            }
            if license_evidence.get(field) is not expected:
                failures.append(f"license_evidence.{field}")

    scenarios = receipt.get("scenarios")
    if not isinstance(scenarios, list):
        failures.append("scenarios")
        scenarios = []
    if [item.get("name") for item in scenarios if isinstance(item, Mapping)] != list(
        SCENARIOS
    ):
        failures.append("scenarios:exact_named_coverage")
    for index, item in enumerate(scenarios):
        if not isinstance(item, Mapping):
            failures.append(f"scenarios[{index}]")
            continue
        if item.get("attempt_count") != 2:
            failures.append(f"scenarios[{index}].attempt_count")
        if item.get("return_codes") != [0, 0]:
            failures.append(f"scenarios[{index}].return_codes")
        if item.get("required_markers_present") is not True:
            failures.append(f"scenarios[{index}].required_markers_present")
        if item.get("replay_equal_after_normalization") is not True:
            failures.append(
                f"scenarios[{index}].replay_equal_after_normalization"
            )
        for digest_field in (
            "normalized_stdout_sha256",
            "normalized_stderr_sha256",
            "normalized_observation_sha256",
            "side_effect_manifest_sha256",
        ):
            if not _HEX_64_RE.fullmatch(str(item.get(digest_field, ""))):
                failures.append(f"scenarios[{index}].{digest_field}")
        if not isinstance(item.get("side_effect_file_count"), int) or int(
            item.get("side_effect_file_count", -1)
        ) < 0:
            failures.append(f"scenarios[{index}].side_effect_file_count")

    expected_suite_digest = _canonical_sha256(
        [
            {
                "name": item.get("name"),
                "normalized_observation_sha256": item.get(
                    "normalized_observation_sha256"
                ),
            }
            for item in scenarios
            if isinstance(item, Mapping)
        ]
    )
    if receipt.get("normalized_suite_sha256") != expected_suite_digest:
        failures.append("normalized_suite_sha256")

    missing_cases = receipt.get("missing_comprehensive_cases")
    if missing_cases != list(MISSING_COMPREHENSIVE_CASES):
        failures.append("missing_comprehensive_cases")

    receipt_sha256 = str(receipt.get("receipt_sha256", ""))
    if not _HEX_64_RE.fullmatch(receipt_sha256):
        failures.append("receipt_sha256:format")
    elif receipt_sha256 != _canonical_sha256(_receipt_digest_basis(receipt)):
        failures.append("receipt_sha256:mismatch")
    failures.extend(_public_safety_failures(receipt))
    return sorted(set(failures))


def certify_secpal_operator_compatibility(
    *,
    msi: Path,
    bin_dir: Path,
    eula: Path,
    mono: Path,
    mono_framework_dir: Path,
    mono_config_dir: Path,
    mono_native_lib_dir: Path,
    license_acceptance: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Run the pinned sample suite and return a public-safe narrow receipt."""

    if license_acceptance != LICENSE_ACCEPTANCE_PHRASE:
        raise SecPALOperatorCompatibilityError(
            "explicit Microsoft SecPAL research-license acceptance is required"
        )
    if not (0 < timeout_seconds <= MAX_TIMEOUT_SECONDS):
        raise SecPALOperatorCompatibilityError(
            f"timeout must be greater than zero and at most {MAX_TIMEOUT_SECONDS}"
        )

    bin_resolved = _resolved_dir(bin_dir, "extracted SecPAL Bin directory")
    mono_resolved = _resolved_file(mono, "Mono executable")
    framework_resolved = _resolved_dir(
        mono_framework_dir, "Mono framework directory"
    )
    config_resolved = _resolved_dir(mono_config_dir, "Mono config directory")
    native_resolved = _resolved_dir(
        mono_native_lib_dir, "Mono native library directory"
    )
    input_paths = {
        "msi": msi,
        "sample_runner": bin_resolved / str(KNOWN_ARTIFACTS["sample_runner"]["name"]),
        "authorization_library": bin_resolved
        / str(KNOWN_ARTIFACTS["authorization_library"]["name"]),
        "audit_viewer": bin_resolved / str(KNOWN_ARTIFACTS["audit_viewer"]["name"]),
        "eula": eula,
    }
    verified_inputs = {
        key: verify_known_file(path, KNOWN_ARTIFACTS[key], label=key)
        for key, path in input_paths.items()
    }
    sample_runner = _resolved_file(input_paths["sample_runner"], "sample_runner")
    authorization_library = _resolved_file(
        input_paths["authorization_library"], "authorization_library"
    )

    with tempfile.TemporaryDirectory(prefix="secpal-operator-cert-") as temporary:
        suite_dir = Path(temporary)
        runtime_config = _prepare_runtime_config(
            config_dir=config_resolved,
            native_lib_dir=native_resolved,
            destination=suite_dir / "mono-config",
        )
        runtime_identity = _runtime_identity(
            mono=mono_resolved,
            framework_dir=framework_resolved,
            config_dir=config_resolved,
            native_lib_dir=native_resolved,
            runtime_config=runtime_config,
            timeout_seconds=timeout_seconds,
            suite_dir=suite_dir,
        )
        scenario_receipts = [
            _scenario_receipt(
                scenario=scenario,
                mono=mono_resolved,
                runtime_config=runtime_config,
                framework_dir=framework_resolved,
                native_lib_dir=native_resolved,
                sample_runner=sample_runner,
                authorization_library=authorization_library,
                timeout_seconds=timeout_seconds,
                suite_dir=suite_dir,
            )
            for scenario in SCENARIOS
        ]

    normalized_suite_sha256 = _canonical_sha256(
        [
            {
                "name": item["name"],
                "normalized_observation_sha256": item[
                    "normalized_observation_sha256"
                ],
            }
            for item in scenario_receipts
        ]
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "certification_class": CERTIFICATION_CLASS,
        "status": "operator_compatibility_observed_non_authoritative",
        "observed_at": observed_at
        or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "description": (
            "Offline replay of all 18 Microsoft SecPAL research-release samples "
            "on an operator-supplied Mono runtime; compatibility evidence only."
        ),
        "scope": {
            "operator_compatibility_only": True,
            "completes_fvt_086": False,
            "completes_fvt_g219": False,
            "vendor_sample_suite_only": True,
        },
        "vendor_supported_platform": False,
        "arbitrary_policy_interface_verified": False,
        "production_use_permitted": False,
        "live_authority": False,
        "deployment_ready": False,
        "verified_inputs": verified_inputs,
        "provenance_binding": {
            "publisher": "Microsoft Corporation",
            "product_name": "Microsoft SecPAL Research Release",
            "msi_product_version": "1.0.0",
            "msi_product_code": "{957BD905-629C-45B0-AA93-EC1AAD218115}",
            "authenticode_subject": "Microsoft Corporation",
            "authenticode_description": "Microsoft.Research.Secpal",
            "authenticode_timestamp_utc": "2007-06-09T03:54:32Z",
            "signature_verification_basis": "operator_reviewed_out_of_band",
            "signature_reverified_by_this_offline_script": False,
        },
        "license_evidence": {
            "eula_identity_bound": True,
            "explicit_acceptance_supplied": True,
            "acceptance_scope": "local_research_compatibility_run_only",
            "research_code_not_intended_for_live_environment": True,
            "software_redistribution_permitted_by_receipt": False,
            "production_use_authorized_by_receipt": False,
        },
        "execution_contract": {
            "network_or_download_requested": False,
            "installer_invoked": False,
            "temporary_working_directories": True,
            "attempts_per_scenario": 2,
            "per_process_timeout_seconds": timeout_seconds,
            "normalizations": [
                "line_endings_and_trailing_newline",
                "temporary_working_directory",
                "machine_identifier",
                "uuid",
                "utc_timestamp",
                "generated_xml_crypto_value",
            ],
            "raw_output_retained": False,
        },
        "operator_runtime": runtime_identity,
        "scenarios": scenario_receipts,
        "normalized_suite_sha256": normalized_suite_sha256,
        "missing_comprehensive_cases": list(MISSING_COMPREHENSIVE_CASES),
        "authority_ceiling": (
            "operator_compatibility_only_no_live_or_production_elevation"
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha256(_receipt_digest_basis(receipt))
    failures = validate_receipt(receipt)
    if failures:
        raise SecPALOperatorCompatibilityError(
            "generated receipt failed closed: " + ", ".join(failures)
        )
    return receipt


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved_parent = path.expanduser().resolve().parent
    if not resolved_parent.is_dir():
        raise SecPALOperatorCompatibilityError(
            f"output parent directory does not exist: {resolved_parent}"
        )
    rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=resolved_parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary_name = stream.name
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, path.expanduser().resolve())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msi", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--eula", type=Path, required=True)
    parser.add_argument("--mono", type=Path, required=True)
    parser.add_argument("--mono-framework-dir", type=Path, required=True)
    parser.add_argument("--mono-config-dir", type=Path, required=True)
    parser.add_argument("--mono-native-lib-dir", type=Path, required=True)
    parser.add_argument("--license-acceptance", required=True)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--observed-at",
        help="optional fixed timestamp for deterministic receipt regeneration",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = certify_secpal_operator_compatibility(
            msi=args.msi,
            bin_dir=args.bin_dir,
            eula=args.eula,
            mono=args.mono,
            mono_framework_dir=args.mono_framework_dir,
            mono_config_dir=args.mono_config_dir,
            mono_native_lib_dir=args.mono_native_lib_dir,
            license_acceptance=args.license_acceptance,
            timeout_seconds=args.timeout_seconds,
            observed_at=args.observed_at,
        )
        if args.output is not None:
            _write_json(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=False))
    except SecPALOperatorCompatibilityError as exc:
        print(
            f"SecPAL operator compatibility certification refused: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
