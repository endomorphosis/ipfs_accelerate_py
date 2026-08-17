"""Controller-owned atomic issuance and certificate publication (PTR-147/155).

Cold publication writes and rehashes candidate components and the pass receipt.
Positive v4 publication (PTR-155) reconstructs controller-owned V2 context and
issued public material, invokes the exact datasets
``verify_test_execution_certificate_v2`` path, and performs the sole atomic
``put_candidate`` only after ``CertificateVerificationStatus.VERIFIED``.

No structural boolean, injected verifier, certificate self-claim, alternate
module/provider, stale or swapped binary/key artifact, changed context, or
missing proof can reach ``put_candidate``.  Workers serialize no witness or
private material.  A crash or failure may leave an immutable non-authoritative
candidate/receipt for retry but never a partial skip candidate.  Import,
collection, ordinary setup, and verification never perform trusted setup, key
generation, build, download, or network calls.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import json
import os
import re
import stat
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Optional

from .services import (
    DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT,
    DATASETS_GROTH16_ARTIFACTS_ROOT_ENV,
    DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256,
    DATASETS_GROTH16_BINARY_ENV,
    DATASETS_GROTH16_BUNDLED_BINARIES_SHA256,
    DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256,
    DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY,
    DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256,
    DATASETS_VERIFIER_REVISION,
    GROTH16_NATIVE_BUILD_RECEIPT_INTERFACE,
    GROTH16_TEST_PASS_ARTIFACT_MANIFEST_INTERFACE,
    PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_ENV,
    PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256_ENV,
    PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV,
    TEST_PASS_GROTH16_CIRCUIT_CID,
    TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256,
    TEST_PASS_GROTH16_CIRCUIT_INTERFACE,
    TEST_PASS_GROTH16_CIRCUIT_VERSION,
    TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256,
    TEST_PASS_GROTH16_RULESET_ID,
    TEST_PASS_GROTH16_STATEMENT_INTERFACE,
    TEST_PASS_GROTH16_STATEMENT_VERSION,
    TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS,
    validate_groth16_capability_payload,
    validate_groth16_release_manifest_payload,
)

PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE: Final = (
    "ProofReuseControllerPublicationTransaction@1"
)
CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE: Final = "ControllerCandidatePublisher@2"
ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE: Final = (
    "IssuedCertificatePublicationResult@1"
)
GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE: Final = (
    "Groth16ArtifactIdentityBindings@1"
)
CONTROLLER_V2_VERIFICATION_CONTEXT_INTERFACE: Final = (
    "ControllerV2VerificationContext@1"
)
# Disposable test-only fixture reason prefix — never reviewed production authority.
_TEST_ONLY_DISPOSABLE_REASON_PREFIX: Final = "test_only"
_PRODUCTION_READY_REASON: Final = "ready"

# Test-pass circuit version introduced by PTR-144.
_TEST_PASS_CIRCUIT_VERSION: Final = TEST_PASS_GROTH16_CIRCUIT_VERSION
_GROTH16_MANIFEST_MAX_BYTES: Final = 64 * 1024
_GROTH16_RECEIPT_MAX_BYTES: Final = 64 * 1024
# V5 exact-byte proving keys are ~169 MiB.
_GROTH16_KEY_MAX_BYTES: Final = 256 * 1024 * 1024
_GROTH16_BINARY_MAX_BYTES: Final = 128 * 1024 * 1024
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_TEST_PASS_CID_PROFILE: Final = "cidv1-base32-dag-json-sha2-256"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dag_json_cid(value: Mapping[str, Any]) -> str:
    """Return the frozen CIDv1/base32/dag-json/sha2-256 identity."""

    canonical = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    raw = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def _read_regular_file(
    path: Path,
    *,
    max_bytes: int,
    owner_private: bool = False,
) -> tuple[bytes, os.stat_result] | None:
    """Read a bounded regular file without following a leaf symlink."""

    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size <= 0
            or file_stat.st_size > max_bytes
        ):
            return None
        if owner_private and os.name != "nt":
            getuid = getattr(os, "getuid", None)
            if callable(getuid) and file_stat.st_uid != getuid():
                return None
            if file_stat.st_mode & 0o077:
                return None
        chunks: list[bytes] = []
        remaining = file_stat.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != file_stat.st_size:
            return None
        return payload, file_stat
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _safe_relative_file(root: Path, relative: str) -> Path | None:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return None
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        return None
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (resolved_root / candidate_relative).resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _json_object(
    data: bytes,
) -> dict[str, Any] | None:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_pinned_groth16_manifest(
    environ: Mapping[str, str],
) -> tuple[dict[str, Any] | None, Path | None, str, str]:
    """Load and structurally validate the separately SHA-pinned manifest."""

    manifest_text = str(
        environ.get(PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_ENV, "") or ""
    ).strip()
    expected_digest = str(
        environ.get(PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256_ENV, "") or ""
    ).strip()
    if not manifest_text or _SHA256_RE.fullmatch(expected_digest) is None:
        return None, None, "", "artifact_manifest_pin_missing"
    manifest_path = Path(manifest_text).expanduser()
    loaded = _read_regular_file(
        manifest_path,
        max_bytes=_GROTH16_MANIFEST_MAX_BYTES,
    )
    if loaded is None:
        return None, manifest_path, "", "artifact_manifest_unreadable"
    data, _manifest_stat = loaded
    actual_digest = _sha256_hex(data)
    if actual_digest != expected_digest:
        return None, manifest_path, actual_digest, "artifact_manifest_digest_mismatch"
    if actual_digest not in DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256:
        return None, manifest_path, actual_digest, "artifact_manifest_unapproved"
    payload = _json_object(data)
    if payload is None:
        return None, manifest_path, actual_digest, "artifact_manifest_invalid"

    expected_top = {
        "interface",
        "reviewed_datasets_revision",
        "reviewed_source_fingerprint",
        "provider_source_sha256",
        "circuit",
        "artifacts",
        "native",
    }
    if set(payload) != expected_top:
        return None, manifest_path, actual_digest, "artifact_manifest_schema_mismatch"
    if payload.get("interface") != GROTH16_TEST_PASS_ARTIFACT_MANIFEST_INTERFACE:
        return None, manifest_path, actual_digest, "artifact_manifest_interface_mismatch"
    if payload.get("reviewed_datasets_revision") != DATASETS_VERIFIER_REVISION:
        return None, manifest_path, actual_digest, "artifact_manifest_revision_mismatch"
    if (
        payload.get("reviewed_source_fingerprint")
        != DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT
    ):
        return None, manifest_path, actual_digest, "artifact_manifest_source_mismatch"
    if (
        payload.get("provider_source_sha256")
        != TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256
    ):
        return None, manifest_path, actual_digest, "artifact_manifest_source_mismatch"

    circuit = payload.get("circuit")
    expected_circuit_keys = {
        "version",
        "identity_sha256",
        "circuit_cid",
        "proof_system",
        "ruleset_id",
        "statement_interface",
        "statement_version",
    }
    if not isinstance(circuit, dict) or set(circuit) != expected_circuit_keys:
        return None, manifest_path, actual_digest, "artifact_manifest_circuit_mismatch"
    identity = {
        "backend_circuit_version": TEST_PASS_GROTH16_CIRCUIT_VERSION,
        "interface": TEST_PASS_GROTH16_CIRCUIT_INTERFACE,
        "proof_system": "groth16",
        "ruleset_id": TEST_PASS_GROTH16_RULESET_ID,
        "statement_interface": TEST_PASS_GROTH16_STATEMENT_INTERFACE,
        "statement_version": TEST_PASS_GROTH16_STATEMENT_VERSION,
    }
    identity_digest = _sha256_hex(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if identity_digest != TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256:
        return None, manifest_path, actual_digest, "reviewed_circuit_constant_mismatch"
    expected_circuit = {
        "version": TEST_PASS_GROTH16_CIRCUIT_VERSION,
        "identity_sha256": TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256,
        "circuit_cid": TEST_PASS_GROTH16_CIRCUIT_CID,
        "proof_system": "groth16",
        "ruleset_id": TEST_PASS_GROTH16_RULESET_ID,
        "statement_interface": TEST_PASS_GROTH16_STATEMENT_INTERFACE,
        "statement_version": TEST_PASS_GROTH16_STATEMENT_VERSION,
    }
    if circuit != expected_circuit or _dag_json_cid(identity) != TEST_PASS_GROTH16_CIRCUIT_CID:
        return None, manifest_path, actual_digest, "artifact_manifest_circuit_mismatch"

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "proving_key",
        "verifying_key",
    }:
        return None, manifest_path, actual_digest, "artifact_manifest_keys_mismatch"
    expected_relatives = {
        "proving_key": f"v{TEST_PASS_GROTH16_CIRCUIT_VERSION}/proving_key.bin",
        "verifying_key": f"v{TEST_PASS_GROTH16_CIRCUIT_VERSION}/verifying_key.bin",
    }
    for name, expected_relative in expected_relatives.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict) or set(artifact) != {
            "relative_path",
            "sha256",
            "size",
        }:
            return None, manifest_path, actual_digest, "artifact_manifest_keys_mismatch"
        digest = artifact.get("sha256")
        size = artifact.get("size")
        if (
            artifact.get("relative_path") != expected_relative
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > _GROTH16_KEY_MAX_BYTES
        ):
            return None, manifest_path, actual_digest, "artifact_manifest_keys_mismatch"

    native = payload.get("native")
    if not isinstance(native, dict) or set(native) != {
        "provenance",
        "binary_sha256",
        "binary_size",
        "supported_circuit_versions",
        "release_manifest_sha256",
        "capability_payload_sha256",
        "locked_source_identity",
    }:
        return None, manifest_path, actual_digest, "artifact_manifest_native_mismatch"
    native_digest = native.get("binary_sha256")
    native_size = native.get("binary_size")
    native_versions = native.get("supported_circuit_versions")
    if (
        native.get("provenance")
        not in {"reviewed_bundled_release", "validated_build_receipt"}
        or not isinstance(native_digest, str)
        or _SHA256_RE.fullmatch(native_digest) is None
        or isinstance(native_size, bool)
        or not isinstance(native_size, int)
        or native_size <= 0
        or native_size > _GROTH16_BINARY_MAX_BYTES
        or not isinstance(native_versions, list)
        or not native_versions
        or any(
            isinstance(version, bool) or not isinstance(version, int) or version <= 0
            for version in native_versions
        )
        or native_versions != sorted(set(native_versions))
        or native.get("release_manifest_sha256")
        not in set(DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256.values())
        or native.get("capability_payload_sha256")
        not in set(DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256.values())
        or native.get("locked_source_identity")
        != DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY
    ):
        return None, manifest_path, actual_digest, "artifact_manifest_native_mismatch"
    return payload, manifest_path, actual_digest, "ready"


def _validate_native_build_receipt(
    *,
    receipt_path: Path,
    binary_path: Path,
    required_circuit_version: int,
) -> tuple[bool, str, dict[str, Any]]:
    loaded = _read_regular_file(
        receipt_path,
        max_bytes=_GROTH16_RECEIPT_MAX_BYTES,
        owner_private=True,
    )
    if loaded is None:
        return False, "native_build_receipt_unreadable", {}
    receipt_data, _receipt_stat = loaded
    receipt = _json_object(receipt_data)
    expected_keys = {
        "interface",
        "reviewed_datasets_revision",
        "reviewed_source_fingerprint",
        "native_platform",
        "binary_relative_path",
        "binary_sha256",
        "binary_size",
        "cargo_locked",
        "trusted_setup",
        "supported_circuit_versions",
        "test_pass_circuit_version",
        "test_pass_circuit_identity_sha256",
        "test_pass_circuit_cid",
        "test_pass_provider_source_sha256",
        "locked_source_identity",
        "capability_payload_sha256",
    }
    if receipt is None or set(receipt) != expected_keys:
        return False, "native_build_receipt_invalid", {}
    if (
        receipt.get("interface") != GROTH16_NATIVE_BUILD_RECEIPT_INTERFACE
        or receipt.get("reviewed_datasets_revision") != DATASETS_VERIFIER_REVISION
        or receipt.get("reviewed_source_fingerprint")
        != DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT
        or receipt.get("cargo_locked") is not True
        or receipt.get("trusted_setup") is not False
        or receipt.get("supported_circuit_versions")
        != list(TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS)
        or required_circuit_version
        not in receipt.get("supported_circuit_versions", ())
        or receipt.get("test_pass_circuit_version")
        != TEST_PASS_GROTH16_CIRCUIT_VERSION
        or receipt.get("test_pass_circuit_identity_sha256")
        != TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256
        or receipt.get("test_pass_circuit_cid") != TEST_PASS_GROTH16_CIRCUIT_CID
        or receipt.get("test_pass_provider_source_sha256")
        != TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256
        or receipt.get("locked_source_identity")
        != DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY
        or receipt.get("capability_payload_sha256")
        not in set(DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256.values())
    ):
        return False, "native_build_receipt_identity_mismatch", {}
    relative = receipt.get("binary_relative_path")
    receipt_binary = (
        _safe_relative_file(receipt_path.parent, relative)
        if isinstance(relative, str)
        else None
    )
    try:
        selected_binary = binary_path.resolve(strict=True)
    except (OSError, RuntimeError):
        selected_binary = None
    if receipt_binary is None or selected_binary != receipt_binary:
        return False, "native_build_receipt_binary_mismatch", {}
    return True, "ready", {"native_build_receipt_sha256": _sha256_hex(receipt_data)}


def _probe_native_capabilities(
    binary_path: Path,
    *,
    required_circuit_version: int,
    expected_sha256: str,
    expected_size: int,
) -> tuple[bool, str]:
    """Execute reviewed bytes, not a mutable path, for the capability probe."""

    environment = {
        "PATH": os.defpath,
        DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(
            binary_path.parent / ".proof-reuse-capability-probe-no-artifacts"
        ),
    }
    for name in ("SYSTEMROOT", "WINDIR"):
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            environment[name] = value
    loaded = _read_regular_file(binary_path, max_bytes=_GROTH16_BINARY_MAX_BYTES)
    if loaded is None:
        return False, "native_binary_unreadable"
    binary_bytes, binary_stat = loaded
    if (
        binary_stat.st_size != expected_size
        or _sha256_hex(binary_bytes) != expected_sha256
    ):
        return False, "native_binary_manifest_mismatch"
    executable_fd = -1
    try:
        command_path = str(binary_path.resolve(strict=True))
        run_kwargs: dict[str, Any] = {}
        if os.name != "nt":
            memfd_create = getattr(os, "memfd_create", None)
            if not callable(memfd_create) or not Path("/proc/self/fd").is_dir():
                return False, "native_fd_execution_unavailable"
            executable_fd = memfd_create(
                "proof-reuse-reviewed-groth16",
                getattr(os, "MFD_CLOEXEC", 0),
            )
            view = memoryview(binary_bytes)
            while view:
                written = os.write(executable_fd, view)
                if written <= 0:
                    return False, "native_fd_copy_failed"
                view = view[written:]
            os.fchmod(executable_fd, 0o500)
            command_path = f"/proc/self/fd/{executable_fd}"
            run_kwargs["pass_fds"] = (executable_fd,)
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            completed = subprocess.run(
                (command_path, "capabilities", "--json"),
                cwd=str(Path(binary_path.anchor)),
                check=False,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=5,
                env=environment,
                **run_kwargs,
            )
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(65_537)
            stderr = stderr_file.read(65_537)
    except Exception:
        return False, "native_capability_probe_failed"
    finally:
        if executable_fd >= 0:
            os.close(executable_fd)
    after = _read_regular_file(binary_path, max_bytes=_GROTH16_BINARY_MAX_BYTES)
    if (
        after is None
        or after[1].st_size != expected_size
        or _sha256_hex(after[0]) != expected_sha256
    ):
        return False, "native_binary_changed_during_probe"
    if (
        getattr(completed, "returncode", 1) != 0
        or len(stdout) > 65_536
        or len(stderr) > 65_536
        or stderr
        or not validate_groth16_capability_payload(
            stdout,
            required_circuit_version=required_circuit_version,
        )
    ):
        return False, "native_capability_payload_mismatch"
    return True, "ready"


def validate_groth16_native_manifest_identity(
    *,
    binary_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    required_circuit_version: int = _TEST_PASS_CIRCUIT_VERSION,
) -> tuple[bool, str, Mapping[str, Any]]:
    """Validate the v4 capability claim for one exact native executable.

    This helper performs only bounded file reads.  It never executes the
    binary, imports datasets, builds Cargo source, performs setup, or contacts
    an endpoint.
    """

    env = os.environ if environ is None else environ
    manifest, manifest_path, manifest_digest, reason = _load_pinned_groth16_manifest(
        env
    )
    if manifest is None:
        return False, reason, {"manifest_sha256": manifest_digest}
    native = manifest["native"]
    versions = native["supported_circuit_versions"]
    if (
        versions != list(TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS)
        or required_circuit_version not in versions
    ):
        return False, "native_v4_capability_missing", {
            "manifest_sha256": manifest_digest,
            "supported_circuit_versions": list(versions),
        }
    selected_text = (
        str(binary_path)
        if binary_path is not None
        else str(env.get(DATASETS_GROTH16_BINARY_ENV, "") or "").strip()
    )
    if not selected_text:
        return False, "native_binary_missing", {"manifest_sha256": manifest_digest}
    selected = Path(selected_text).expanduser()
    loaded = _read_regular_file(selected, max_bytes=_GROTH16_BINARY_MAX_BYTES)
    if loaded is None:
        return False, "native_binary_unreadable", {"manifest_sha256": manifest_digest}
    binary_data, binary_stat = loaded
    binary_digest = _sha256_hex(binary_data)
    if (
        binary_digest != native["binary_sha256"]
        or binary_stat.st_size != native["binary_size"]
    ):
        return False, "native_binary_manifest_mismatch", {
            "manifest_sha256": manifest_digest
        }
    if os.name != "nt" and not os.access(selected, os.X_OK):
        return False, "native_binary_not_executable", {
            "manifest_sha256": manifest_digest
        }
    provenance = native["provenance"]
    receipt_diagnostics: dict[str, Any] = {}
    if provenance == "reviewed_bundled_release":
        platform_name = next(
            (
                name
                for name, digest in DATASETS_GROTH16_BUNDLED_BINARIES_SHA256.items()
                if digest == binary_digest
                and DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256.get(name)
                == native["release_manifest_sha256"]
                and DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256.get(name)
                == native["capability_payload_sha256"]
            ),
            "",
        )
        release_loaded = _read_regular_file(
            selected.parent / "release-manifest.json",
            max_bytes=_GROTH16_MANIFEST_MAX_BYTES,
        )
        if (
            not platform_name
            or release_loaded is None
            or not validate_groth16_release_manifest_payload(
                release_loaded[0],
                platform_name=platform_name,
                binary_sha256=binary_digest,
            )
        ):
            return False, "native_release_manifest_mismatch", {
                "manifest_sha256": manifest_digest
            }
        receipt_diagnostics["native_release_platform"] = platform_name
    elif provenance == "validated_build_receipt":
        receipt_text = str(
            env.get(PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV, "") or ""
        ).strip()
        if not receipt_text:
            return False, "native_build_receipt_missing", {
                "manifest_sha256": manifest_digest
            }
        valid, receipt_reason, receipt_diagnostics = _validate_native_build_receipt(
            receipt_path=Path(receipt_text).expanduser(),
            binary_path=selected,
            required_circuit_version=required_circuit_version,
        )
        if not valid:
            return False, receipt_reason, {"manifest_sha256": manifest_digest}
    capability_ready, capability_reason = _probe_native_capabilities(
        selected,
        required_circuit_version=required_circuit_version,
        expected_sha256=native["binary_sha256"],
        expected_size=native["binary_size"],
    )
    if not capability_ready:
        return False, capability_reason, {
            "manifest_sha256": manifest_digest,
            "process_started": True,
        }
    return True, "ready", MappingProxyType(
        {
            "manifest_path": str(manifest_path or "")[:256],
            "manifest_sha256": manifest_digest,
            "native_provenance": provenance,
            "native_binary_sha256": native["binary_sha256"],
            "supported_circuit_versions": list(versions),
            "locked_source_identity": DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY,
            "capability_payload_sha256": native["capability_payload_sha256"],
            "process_started": True,
            **receipt_diagnostics,
        }
    )


def inspect_pinned_groth16_artifact_versions(
    *,
    artifacts_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], str, str]:
    """Inspect only SHA-pinned key pairs, including test-pass v4.

    This deliberately does not promote key presence to certificate authority;
    native and provider identity are checked separately by
    :meth:`Groth16ArtifactIdentityBindings.from_activated_artifacts`.
    """

    env = os.environ if environ is None else environ
    manifest, _manifest_path, digest, reason = _load_pinned_groth16_manifest(env)
    if manifest is None:
        return (), (), reason, digest
    root_text = (
        str(artifacts_root)
        if artifacts_root is not None
        else str(env.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "") or "").strip()
    )
    if not root_text:
        return (), (), "artifacts_root_missing", digest
    root = Path(root_text).expanduser()
    if root.is_symlink() or not root.is_dir():
        return (), (), "artifacts_root_missing", digest
    results: dict[str, bool] = {}
    for name in ("proving_key", "verifying_key"):
        entry = manifest["artifacts"][name]
        path = _safe_relative_file(root, entry["relative_path"])
        loaded = (
            _read_regular_file(path, max_bytes=_GROTH16_KEY_MAX_BYTES)
            if path is not None
            else None
        )
        results[name] = bool(
            loaded is not None
            and _sha256_hex(loaded[0]) == entry["sha256"]
            and loaded[1].st_size == entry["size"]
        )
    return (
        (_TEST_PASS_CIRCUIT_VERSION,) if results.get("verifying_key") else (),
        (_TEST_PASS_CIRCUIT_VERSION,) if results.get("proving_key") else (),
        "ready" if all(results.values()) else "artifact_manifest_digest_mismatch",
        digest,
    )


def _bounded_text(value: Any, *, max_chars: int = 256) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ""
    text = value.strip()
    return text[:max_chars] if len(text) > max_chars else text


def _mapping_of(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return payload
    return None


def _status_text(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    try:
        return str(raw).strip().lower()
    except Exception:
        return ""


def _extract_certificate_payload(issue_result: Any) -> Mapping[str, Any] | None:
    """Return public certificate mapping from an immediate or deferred result."""

    if issue_result is None:
        return None
    # Direct certificate attribute (material / disposition / deferred result).
    for attr in ("certificate", "issued_certificate", "certificate_payload"):
        cert = getattr(issue_result, attr, None)
        if cert is None:
            continue
        mapped = _mapping_of(cert)
        if mapped is not None:
            # Prefer include_proof=True when available.
            to_dict = getattr(cert, "to_dict", None)
            if callable(to_dict):
                try:
                    with_proof = to_dict(include_proof=True, include_ids=True)
                    if isinstance(with_proof, Mapping):
                        return with_proof
                except TypeError:
                    pass
                except Exception:
                    pass
            return mapped
    # Nested material (IssuedTestCertificateMaterial).
    material = getattr(issue_result, "material", None)
    if material is not None:
        mapped = _extract_certificate_payload(material)
        if mapped is not None:
            return mapped
        public = getattr(material, "to_public_dict", None)
        if callable(public):
            try:
                payload = public()
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                nested = payload.get("certificate")
                if isinstance(nested, Mapping):
                    return nested
    mapped = _mapping_of(issue_result)
    if mapped is not None:
        nested = mapped.get("certificate")
        if isinstance(nested, Mapping):
            return nested
        # Some dispositions expose only certificate_cid without body — not enough.
    return None


def _issue_succeeded(issue_result: Any) -> bool:
    if issue_result is None:
        return False
    if getattr(issue_result, "issued", None) is True:
        return True
    status = _status_text(getattr(issue_result, "status", None))
    if status in {
        "issued",
        "certificate_issued",
        "success",
        "ok",
    }:
        return True
    # Material present implies success even if status attribute differs.
    if _extract_certificate_payload(issue_result) is not None:
        if status in {"", "issued", "certificate_issued", "success", "ok"}:
            return True
        if status not in {
            "deferred",
            "certificate_deferred",
            "queued",
            "rejected",
            "certificate_rejected",
            "failed",
        }:
            # Unknown status but certificate body present — treat as success so
            # flush_publications never discards a returned certificate.
            return True
    return False


def _issue_deferred(issue_result: Any) -> bool:
    if issue_result is None:
        return True
    if getattr(issue_result, "deferred", None) is True:
        return True
    status = _status_text(getattr(issue_result, "status", None))
    return status in {
        "deferred",
        "certificate_deferred",
        "queued",
        "run",
    }


def _is_test_only_disposable_bindings(
    bindings: "Groth16ArtifactIdentityBindings | None",
) -> bool:
    if bindings is None:
        return False
    reason = _bounded_text(getattr(bindings, "reason_code", ""), max_chars=96).lower()
    return reason.startswith(_TEST_ONLY_DISPOSABLE_REASON_PREFIX)


def _bindings_production_ready(
    bindings: "Groth16ArtifactIdentityBindings | None",
) -> bool:
    """True only for reviewed production-ready artifact pins (not test fixtures)."""

    if not isinstance(bindings, Groth16ArtifactIdentityBindings):
        return False
    if not (
        bindings.provenance_ready
        and bindings.circuit_cid
        and bindings.verifying_key_cid
        and bindings.artifacts_root
        and bindings.verifying_key_sha256
        and bindings.proving_key_sha256
        and bindings.backend_circuit_version == _TEST_PASS_CIRCUIT_VERSION
        and bindings.reviewed_revision == DATASETS_VERIFIER_REVISION
        and bindings.reason_code == _PRODUCTION_READY_REASON
    ):
        return False
    diagnostics = dict(bindings.diagnostics or {})
    approved = DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256
    if not approved:
        # No hardcoded-reviewed key-manifest allowlist is configured.
        return False
    manifest_sha = _bounded_text(diagnostics.get("manifest_sha256"), max_chars=64)
    if not manifest_sha or manifest_sha not in approved:
        return False
    if not diagnostics.get("native_v4_capability_validated"):
        return False
    return True


def _bindings_usable_for_publication(
    bindings: "Groth16ArtifactIdentityBindings | None",
) -> bool:
    """Accept production-ready pins or an explicitly disposable test-only fixture."""

    if _bindings_production_ready(bindings):
        return True
    if not isinstance(bindings, Groth16ArtifactIdentityBindings):
        return False
    if not (
        bindings.provenance_ready
        and bindings.circuit_cid
        and bindings.verifying_key_cid
        and bindings.backend_circuit_version == _TEST_PASS_CIRCUIT_VERSION
        and bindings.reviewed_revision == DATASETS_VERIFIER_REVISION
        and _is_test_only_disposable_bindings(bindings)
    ):
        return False
    return True


@dataclass(frozen=True, slots=True)
class ControllerV2VerificationContext:
    """Publication-side reconstruction of exact V2 expected inputs (PTR-155).

    Built exclusively from controller-owned PTR-154 pins and public statement
    inputs.  Certificate fields never fill missing pins.  Context alone never
    grants publication or skip authority.
    """

    interface: str = CONTROLLER_V2_VERIFICATION_CONTEXT_INTERFACE
    receipt_cid: str = ""
    execution_key_cid: str = ""
    candidate_context_cid: str = ""
    expected_candidate_context_cid: str = ""
    policy_cid: str = ""
    statement_cid: str = ""
    circuit_cid: str = ""
    verifying_key_cid: str = ""
    issuer_id: str = ""
    epoch: str = ""
    backend_id: str = "groth16"
    proof_system_id: str = "groth16"
    locator_cid: str = ""
    content_profile: str = _TEST_PASS_CID_PROFILE
    public_inputs: Mapping[str, Any] = field(default_factory=dict)
    statement: Any = None
    verifying_key_path: str = ""
    is_test_only_disposable: bool = False
    reason_code: str = "ok"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def may_publish_candidate(self) -> bool:
        return False

    @property
    def is_complete(self) -> bool:
        required = (
            self.receipt_cid,
            self.execution_key_cid,
            self.expected_candidate_context_cid or self.candidate_context_cid,
            self.policy_cid,
            self.statement_cid,
            self.circuit_cid,
            self.verifying_key_cid,
            self.issuer_id,
            self.epoch,
            self.backend_id,
        )
        return all(bool(value) for value in required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "expected_candidate_context_cid": (
                self.expected_candidate_context_cid or self.candidate_context_cid
            ),
            "policy_cid": self.policy_cid,
            "statement_cid": self.statement_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "issuer_id": self.issuer_id,
            "epoch": self.epoch,
            "backend_id": self.backend_id,
            "proof_system_id": self.proof_system_id,
            "locator_cid": self.locator_cid,
            "content_profile": self.content_profile,
            "verifying_key_path": self.verifying_key_path,
            "is_test_only_disposable": self.is_test_only_disposable,
            "is_complete": self.is_complete,
            "may_authorize_skip": False,
            "may_publish_candidate": False,
            "reason_code": self.reason_code,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_sources(
        cls,
        *,
        deferred_request: Any = None,
        receipt: Any = None,
        bindings: "Groth16ArtifactIdentityBindings | None" = None,
        verifying_key_path: str = "",
        is_test_only_disposable: bool = False,
    ) -> "ControllerV2VerificationContext | None":
        """Build from controller-owned request/receipt pins; never certificate fill-in."""

        try:
            request_map: dict[str, Any] = {}
            public_inputs: dict[str, Any] = {}
            statement_obj: Any = None
            if deferred_request is not None:
                mapped = _mapping_of(deferred_request)
                if mapped is not None:
                    request_map = dict(mapped)
                # Typed DeferredTestCertificateRequest / ControllerOwned context.
                for attr in (
                    "receipt_cid",
                    "execution_key_cid",
                    "candidate_context_cid",
                    "policy_cid",
                    "statement_cid",
                    "circuit_cid",
                    "verifying_key_cid",
                    "issuer_id",
                    "epoch",
                    "backend_id",
                    "proof_system_id",
                    "locator_cid",
                ):
                    if not request_map.get(attr):
                        value = getattr(deferred_request, attr, None)
                        if value not in (None, ""):
                            request_map[attr] = value
                statement_obj = getattr(deferred_request, "statement", None)
                if statement_obj is not None:
                    to_public = getattr(statement_obj, "to_public_inputs", None)
                    if callable(to_public):
                        try:
                            pi = to_public()
                            if isinstance(pi, Mapping):
                                public_inputs = dict(pi)
                        except Exception:
                            public_inputs = {}
                nested_pi = request_map.get("public_inputs")
                if not public_inputs and isinstance(nested_pi, Mapping):
                    public_inputs = dict(nested_pi)
                nested_statement = request_map.get("statement")
                if statement_obj is None and nested_statement is not None:
                    statement_obj = nested_statement
                # Flat deferred envelopes may already be public-input shaped.
                if not public_inputs and request_map.get("candidate_context_cid"):
                    public_inputs = {
                        key: request_map[key]
                        for key in (
                            "receipt_cid",
                            "execution_key_cid",
                            "candidate_context_cid",
                            "policy_cid",
                            "statement_cid",
                            "circuit_cid",
                            "verifying_key_cid",
                            "issuer_id",
                            "epoch",
                            "locator_cid",
                            "content_profile",
                            "completeness_policy_cid",
                            "statement_version",
                        )
                        if key in request_map
                    }

            receipt_map = _mapping_of(receipt) or {}
            if not isinstance(receipt_map, Mapping):
                receipt_map = {}

            def _pin(*names: str) -> str:
                for name in names:
                    for source in (public_inputs, request_map, receipt_map):
                        if isinstance(source, Mapping):
                            text = _bounded_text(source.get(name), max_chars=256)
                            if text:
                                return text
                    if name == "receipt_cid":
                        text = _bounded_text(
                            getattr(receipt, "receipt_id", None)
                            or getattr(receipt, "receipt_cid", None),
                            max_chars=256,
                        )
                        if text:
                            return text
                    if name == "execution_key_cid":
                        text = _bounded_text(
                            getattr(receipt, "execution_key_cid", None),
                            max_chars=256,
                        )
                        if text:
                            return text
                    if name == "locator_cid":
                        text = _bounded_text(
                            getattr(receipt, "locator_cid", None),
                            max_chars=256,
                        )
                        if text:
                            return text
                return ""

            circuit_cid = _pin("circuit_cid")
            verifying_key_cid = _pin("verifying_key_cid")
            if bindings is not None and getattr(bindings, "provenance_ready", False):
                # Controller pins must match activated artifact pins when both set.
                if circuit_cid and bindings.circuit_cid and circuit_cid != bindings.circuit_cid:
                    return None
                if (
                    verifying_key_cid
                    and bindings.verifying_key_cid
                    and verifying_key_cid != bindings.verifying_key_cid
                ):
                    return None
                if not circuit_cid:
                    circuit_cid = bindings.circuit_cid
                if not verifying_key_cid:
                    verifying_key_cid = bindings.verifying_key_cid

            candidate_cid = _pin("candidate_context_cid")
            vk_path = _bounded_text(verifying_key_path, max_chars=4096)
            if not vk_path and bindings is not None and bindings.artifacts_root:
                vk_path = str(Path(bindings.artifacts_root) / "v4" / "verifying_key.bin")

            if not public_inputs and candidate_cid:
                public_inputs = {
                    "receipt_cid": _pin("receipt_cid"),
                    "execution_key_cid": _pin("execution_key_cid"),
                    "candidate_context_cid": candidate_cid,
                    "policy_cid": _pin("policy_cid"),
                    "statement_cid": _pin("statement_cid"),
                    "circuit_cid": circuit_cid,
                    "verifying_key_cid": verifying_key_cid,
                    "issuer_id": _pin("issuer_id"),
                    "epoch": _pin("epoch"),
                    "locator_cid": _pin("locator_cid"),
                    "content_profile": _pin("content_profile") or _TEST_PASS_CID_PROFILE,
                }

            return cls(
                receipt_cid=_pin("receipt_cid"),
                execution_key_cid=_pin("execution_key_cid"),
                candidate_context_cid=candidate_cid,
                expected_candidate_context_cid=candidate_cid,
                policy_cid=_pin("policy_cid"),
                statement_cid=_pin("statement_cid"),
                circuit_cid=circuit_cid,
                verifying_key_cid=verifying_key_cid,
                issuer_id=_pin("issuer_id"),
                epoch=_pin("epoch"),
                backend_id=_pin("backend_id") or "groth16",
                proof_system_id=_pin("proof_system_id") or "groth16",
                locator_cid=_pin("locator_cid"),
                content_profile=_pin("content_profile") or _TEST_PASS_CID_PROFILE,
                public_inputs=MappingProxyType(dict(public_inputs)),
                statement=statement_obj,
                verifying_key_path=vk_path,
                is_test_only_disposable=is_test_only_disposable
                or _is_test_only_disposable_bindings(bindings),
                reason_code="ok" if candidate_cid else "incomplete",
                diagnostics=MappingProxyType(
                    {
                        "source": "controller_reconstruction",
                        "has_public_inputs": bool(public_inputs),
                        "has_statement": statement_obj is not None,
                    }
                ),
            )
        except Exception:
            return None


def _import_datasets_v2_surface(
    *,
    module_provenance_validator: Any = None,
) -> tuple[Any, Any, Any, str] | None:
    """Import exact datasets V2 verifier/binding modules without network or build.

    Returns ``(verifier_module, binding_module, zkp_module, reason)`` or None.
    """

    try:
        verifier_module = importlib.import_module(
            "ipfs_datasets_py.logic.zkp.test_execution_certificate"
        )
        binding_module = importlib.import_module(
            "ipfs_datasets_py.logic.zkp.provekit.test_pass_circuit"
        )
        zkp_module = importlib.import_module("ipfs_datasets_py.logic.zkp")
    except Exception:
        return None
    modules = (verifier_module, binding_module, zkp_module)
    if callable(module_provenance_validator):
        try:
            if not all(module_provenance_validator(module) for module in modules):
                return None
        except Exception:
            return None
    return verifier_module, binding_module, zkp_module, "ready"


def _resolve_publication_backend(
    *,
    bindings: "Groth16ArtifactIdentityBindings",
    test_only_backend: Any = None,
    is_test_only: bool = False,
) -> tuple[Any | None, str]:
    """Resolve the sole verifier backend for publication.

    Disposable test-only backends are accepted only for explicitly labeled
    test fixtures.  Production uses the FD-bound verifying key under the
    provenance-ready artifacts root via the reviewed provider when available.
    """

    if is_test_only and test_only_backend is not None:
        backend_id = _status_text(getattr(test_only_backend, "backend_id", "groth16"))
        if backend_id and backend_id not in {"", "groth16"}:
            return None, "test_only_backend_id_rejected"
        if not callable(getattr(test_only_backend, "verify_proof", None)):
            return None, "test_only_backend_missing_verify_proof"
        return test_only_backend, "test_only_disposable_backend"

    if not _bindings_production_ready(bindings):
        return None, "production_backend_unavailable"

    try:
        provider_module = importlib.import_module(
            "ipfs_datasets_py.logic.zkp.test_pass_groth16_provider"
        )
        provider_cls = getattr(provider_module, "LazyGroth16TestCertificateProvider", None)
        if provider_cls is None:
            return None, "provider_unavailable"
        provider = provider_cls(
            artifacts_root=bindings.artifacts_root,
            require_enable_env=False,
        )
        provider_root = Path(provider.artifacts_root()).resolve(strict=True)
        expected_root = Path(bindings.artifacts_root).resolve(strict=True)
        if provider_root != expected_root:
            return None, "provider_artifact_root_mismatch"
        if not callable(getattr(provider, "verify_proof_json", None)):
            return None, "provider_verify_unavailable"

        class _PinnedGroth16NativeVerifier:
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

        return _PinnedGroth16NativeVerifier(), "pinned_native_backend"
    except Exception:
        return None, "production_backend_resolution_failed"


def verify_test_execution_certificate_v2_for_publication(
    certificate: Mapping[str, Any] | Any,
    *,
    bindings: "Groth16ArtifactIdentityBindings",
    controller_context: ControllerV2VerificationContext,
    test_only_backend: Any = None,
    module_provenance_validator: Any = None,
) -> tuple[bool, str, Any]:
    """Sole publication authority adapter around exact local V2 verification.

    Requires ``CertificateVerificationStatus.VERIFIED`` from
    ``verify_test_execution_certificate_v2`` with the controller's expected
    candidate-context CID.  Structural completeness, certificate self-claims,
    and arbitrary injected verifier callbacks are never authority.
    """

    try:
        if not _bindings_usable_for_publication(bindings):
            return False, "artifact_provenance_unready", None
        if controller_context is None or not controller_context.is_complete:
            return False, "controller_v2_context_incomplete", None
        expected_cid = (
            controller_context.expected_candidate_context_cid
            or controller_context.candidate_context_cid
        )
        if not expected_cid:
            return False, "expected_candidate_context_missing", None

        # Pin agreement: certificate identity must match controller + bindings.
        cert_map = _mapping_of(certificate) or {}
        cert_circuit = _bounded_text(
            cert_map.get("circuit_cid") or getattr(certificate, "circuit_cid", "")
        )
        cert_vk = _bounded_text(
            cert_map.get("verifying_key_cid")
            or getattr(certificate, "verifying_key_cid", "")
        )
        if cert_circuit and cert_circuit != bindings.circuit_cid:
            return False, "circuit_cid_mismatch", None
        if cert_vk and cert_vk != bindings.verifying_key_cid:
            return False, "verifying_key_cid_mismatch", None
        if controller_context.circuit_cid != bindings.circuit_cid:
            return False, "controller_circuit_pin_mismatch", None
        if controller_context.verifying_key_cid != bindings.verifying_key_cid:
            return False, "controller_verifying_key_pin_mismatch", None

        surface = _import_datasets_v2_surface(
            module_provenance_validator=module_provenance_validator,
        )
        if surface is None:
            return False, "datasets_v2_surface_unavailable", None
        verifier_module, binding_module, zkp_module, _ready = surface

        status_enum = getattr(verifier_module, "CertificateVerificationStatus", None)
        verify_v2 = getattr(verifier_module, "verify_test_execution_certificate_v2", None)
        binding_cls = getattr(binding_module, "TestPassCircuitBinding", None)
        if status_enum is None or not callable(verify_v2) or binding_cls is None:
            return False, "datasets_v2_symbols_unavailable", None

        public_inputs = dict(controller_context.public_inputs or {})
        statement_obj = getattr(controller_context, "statement", None)
        if not public_inputs and statement_obj is not None:
            to_public = getattr(statement_obj, "to_public_inputs", None)
            if callable(to_public):
                try:
                    pi = to_public()
                    if isinstance(pi, Mapping):
                        public_inputs = dict(pi)
                except Exception:
                    public_inputs = {}
        if not public_inputs and statement_obj is None:
            return False, "controller_public_inputs_missing", None
        # Force controller-owned pins over any certificate-supplied inputs.
        if public_inputs:
            public_inputs.update(
                {
                    "receipt_cid": controller_context.receipt_cid,
                    "execution_key_cid": controller_context.execution_key_cid,
                    "candidate_context_cid": expected_cid,
                    "policy_cid": controller_context.policy_cid,
                    "statement_cid": controller_context.statement_cid,
                    "circuit_cid": controller_context.circuit_cid,
                    "verifying_key_cid": controller_context.verifying_key_cid,
                    "issuer_id": controller_context.issuer_id,
                    "epoch": controller_context.epoch,
                }
            )
            if controller_context.locator_cid:
                public_inputs.setdefault(
                    "locator_cid", controller_context.locator_cid
                )
            public_inputs.setdefault("content_profile", _TEST_PASS_CID_PROFILE)

        verifier_artifacts: dict[str, str] = {}
        # Only pin a verifying-key path when the FD-bound file actually exists.
        # Missing paths make the datasets verifier return backend_unavailable
        # even for disposable test-only backends that do not need disk keys.
        candidate_vk_paths: list[str] = []
        if controller_context.verifying_key_path:
            candidate_vk_paths.append(controller_context.verifying_key_path)
        if bindings.artifacts_root:
            candidate_vk_paths.append(
                str(Path(bindings.artifacts_root) / "v4" / "verifying_key.bin")
            )
        for vk_path_text in candidate_vk_paths:
            try:
                vk_path = Path(vk_path_text)
                if vk_path.is_file() and not vk_path.is_symlink():
                    verifier_artifacts["verifying_key_path"] = str(vk_path)
                    break
            except OSError:
                continue

        binding_kwargs: dict[str, Any] = {
            "backend_id": controller_context.backend_id or "groth16",
            "proof_system_id": controller_context.proof_system_id or "groth16",
            "circuit_cid": bindings.circuit_cid,
            "verifying_key_cid": bindings.verifying_key_cid,
            "statement_cid": controller_context.statement_cid,
            "issuer_id": controller_context.issuer_id,
            "policy_cid": controller_context.policy_cid,
            "epoch": controller_context.epoch,
            "candidate_context_cid": expected_cid,
            "verifier_artifacts": verifier_artifacts or None,
        }
        try:
            if statement_obj is not None and not isinstance(statement_obj, Mapping):
                # Prefer the live statement object (to_public_inputs may include
                # statement-level fields that cannot round-trip through from_dict).
                binding = binding_cls(statement_obj, **binding_kwargs)
            else:
                # Rebuild a V2 statement envelope so circuit_version/ruleset_id
                # are not treated as unknown public-input fields.
                statement_level = {
                    key: public_inputs.pop(key)
                    for key in ("circuit_version", "ruleset_id", "interface")
                    if key in public_inputs
                }
                envelope: dict[str, Any] = {
                    "interface": statement_level.get(
                        "interface", "TestPassStatementV2"
                    ),
                    "public_inputs": public_inputs,
                }
                if "circuit_version" in statement_level:
                    envelope["circuit_version"] = statement_level["circuit_version"]
                if "ruleset_id" in statement_level:
                    envelope["ruleset_id"] = statement_level["ruleset_id"]
                binding = binding_cls(envelope, **binding_kwargs)
        except Exception:
            return False, "test_pass_circuit_binding_failed", None

        is_test_only = (
            controller_context.is_test_only_disposable
            or _is_test_only_disposable_bindings(bindings)
        )
        backend, backend_reason = _resolve_publication_backend(
            bindings=bindings,
            test_only_backend=test_only_backend,
            is_test_only=is_test_only,
        )
        if backend is None:
            return False, backend_reason or "verification_backend_unavailable", None

        # Prefer include_proof certificate mapping when available.
        cert_payload: Any = certificate
        if isinstance(certificate, Mapping):
            cert_payload = dict(certificate)
        else:
            to_dict = getattr(certificate, "to_dict", None)
            if callable(to_dict):
                try:
                    with_proof = to_dict(include_proof=True, include_ids=True)
                    if isinstance(with_proof, Mapping):
                        cert_payload = with_proof
                except TypeError:
                    mapped = _mapping_of(certificate)
                    if mapped is not None:
                        cert_payload = dict(mapped)
                except Exception:
                    mapped = _mapping_of(certificate)
                    if mapped is not None:
                        cert_payload = dict(mapped)

        try:
            result = verify_v2(
                cert_payload,
                binding,
                backend,
                expected_candidate_context_cid=expected_cid,
            )
        except Exception:
            return False, "exact_v2_verify_exception", None

        # Only the exhaustive VERIFIED status is authority.  Booleans, self
        # claims, and alternate result shapes are rejected.
        verified_status = getattr(status_enum, "VERIFIED", None)
        status = getattr(result, "status", None)
        if verified_status is not None and status is verified_status:
            if getattr(result, "verified", None) is True:
                reason = "verified"
                if is_test_only or backend_reason == "test_only_disposable_backend":
                    reason = "verified_test_only_disposable"
                return True, reason, result
        status_text = _status_text(getattr(status, "value", status))
        detail = _bounded_text(
            getattr(result, "reason", None) or getattr(result, "detail", None) or "",
            max_chars=96,
        )
        return False, detail or status_text or "exact_v2_not_verified", result
    except Exception:
        return False, "exact_v2_verification_exception", None


def _local_verify_certificate(
    certificate: Mapping[str, Any] | Any,
    *,
    bindings: "Groth16ArtifactIdentityBindings | None" = None,
    require_cryptographic_verify: bool = False,
    cryptographic_verifier: Any = None,
    module_provenance_validator: Any = None,
    verification_context: Mapping[str, Any] | None = None,
    controller_context: ControllerV2VerificationContext | None = None,
    test_only_backend: Any = None,
) -> tuple[bool, str]:
    """Locally verify one returned certificate via exact V2; never raise.

    Injected ``cryptographic_verifier`` callbacks and certificate self-claims
    are intentionally ignored for publication authority.  Only
    :func:`verify_test_execution_certificate_v2_for_publication` can authorize.
    """

    del cryptographic_verifier  # never authority
    try:
        if not isinstance(bindings, Groth16ArtifactIdentityBindings):
            return False, "artifact_provenance_unready"
        if not _bindings_usable_for_publication(bindings):
            return False, "artifact_provenance_unready"
        context = controller_context
        if context is None and isinstance(verification_context, Mapping):
            context = ControllerV2VerificationContext.from_sources(
                deferred_request=verification_context.get("deferred_request"),
                receipt=verification_context.get("receipt"),
                bindings=bindings,
                is_test_only_disposable=_is_test_only_disposable_bindings(bindings),
            )
        if context is None:
            return False, "controller_v2_context_unavailable"
        if require_cryptographic_verify is False:
            # Structural-only path is never publication authority.
            return False, "structural_verification_rejected"
        ok, reason, _result = verify_test_execution_certificate_v2_for_publication(
            certificate,
            bindings=bindings,
            controller_context=context,
            test_only_backend=test_only_backend,
            module_provenance_validator=module_provenance_validator,
        )
        return ok, reason
    except Exception:
        return False, "local_verification_exception"


@dataclass(frozen=True, slots=True)
class Groth16ArtifactIdentityBindings:
    """Circuit and verifying-key CIDs derived from exact activated bytes.

    Labels and certificate metadata are never authoritative for these pins.
    Missing, synthetic, stale, substituted, or mismatched provenance yields
    ``provenance_ready=False`` so callers return RUN/DEFERRED.
    """

    interface: str = GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE
    circuit_cid: str = ""
    verifying_key_cid: str = ""
    artifacts_root: str = ""
    verifying_key_sha256: str = ""
    proving_key_sha256: str = ""
    backend_circuit_version: int = _TEST_PASS_CIRCUIT_VERSION
    reviewed_revision: str = DATASETS_VERIFIER_REVISION
    provenance_ready: bool = False
    reason_code: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "artifacts_root": self.artifacts_root,
            "verifying_key_sha256": self.verifying_key_sha256,
            "proving_key_sha256": self.proving_key_sha256,
            "backend_circuit_version": self.backend_circuit_version,
            "reviewed_revision": self.reviewed_revision,
            "provenance_ready": self.provenance_ready,
            "reason_code": self.reason_code,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def unready(
        cls,
        reason_code: str,
        **diagnostics: Any,
    ) -> "Groth16ArtifactIdentityBindings":
        bounded: dict[str, Any] = {}
        for key, value in list(diagnostics.items())[:16]:
            name = str(key)[:64]
            if value is None or isinstance(value, (bool, int)):
                bounded[name] = value
            elif isinstance(value, str):
                bounded[name] = value[:128]
            else:
                bounded[name] = type(value).__name__[:64]
        return cls(
            provenance_ready=False,
            reason_code=str(reason_code)[:64],
            diagnostics=MappingProxyType(bounded),
        )

    @classmethod
    def from_activated_artifacts(
        cls,
        *,
        artifacts_root: str | os.PathLike[str] | None = None,
        environ: Mapping[str, str] | None = None,
        binary_path: str | os.PathLike[str] | None = None,
        circuit_version: int = _TEST_PASS_CIRCUIT_VERSION,
    ) -> "Groth16ArtifactIdentityBindings":
        """Derive pins from exact reviewed circuit + activated key bytes.

        A native binary or nonempty keys are non-authoritative.  Authority
        requires a separately SHA-pinned operator/review manifest, exact key
        bytes, reviewed provider/circuit identity, and a manifest-bound
        executable whose v4 capability is either operator-reviewed or tied to
        a validated current-source Cargo build receipt.
        """

        env = environ if environ is not None else os.environ
        try:
            if circuit_version != _TEST_PASS_CIRCUIT_VERSION:
                return cls.unready(
                    "unsupported_test_pass_circuit_version",
                    requested_version=int(circuit_version),
                    required_version=_TEST_PASS_CIRCUIT_VERSION,
                )

            manifest, manifest_path, manifest_digest, manifest_reason = (
                _load_pinned_groth16_manifest(env)
            )
            if manifest is None:
                return cls.unready(
                    manifest_reason,
                    manifest_path=str(manifest_path or ""),
                    manifest_sha256=manifest_digest,
                    arbitrary_keys_non_authoritative=True,
                )

            root: Path | None
            if artifacts_root is not None:
                root = Path(artifacts_root)
            else:
                override = str(env.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "") or "").strip()
                if override:
                    root = Path(override)
                else:
                    root = None
                    # Prefer datasets default when importable (lazy).
                    try:
                        from ipfs_datasets_py.logic.zkp.test_pass_groth16_provider import (
                            default_artifacts_root,
                        )

                        root = default_artifacts_root()
                    except Exception:
                        root = None
            if root is None or not root.is_dir():
                return cls.unready(
                    "artifacts_root_missing",
                    binary_alone_non_authoritative=True,
                )

            if root.is_symlink():
                return cls.unready("artifacts_root_symlink", artifacts_root=str(root))
            artifacts_manifest = manifest["artifacts"]
            pk_path = _safe_relative_file(
                root, artifacts_manifest["proving_key"]["relative_path"]
            )
            vk_path = _safe_relative_file(
                root, artifacts_manifest["verifying_key"]["relative_path"]
            )
            if pk_path is None or vk_path is None:
                return cls.unready(
                    "test_pass_keys_missing",
                    artifacts_root=str(root),
                    version=int(circuit_version),
                    binary_alone_non_authoritative=True,
                )
            pk_loaded = _read_regular_file(pk_path, max_bytes=_GROTH16_KEY_MAX_BYTES)
            vk_loaded = _read_regular_file(vk_path, max_bytes=_GROTH16_KEY_MAX_BYTES)
            if pk_loaded is None or vk_loaded is None:
                return cls.unready("artifact_read_failed", artifacts_root=str(root))
            pk_bytes, pk_stat = pk_loaded
            vk_bytes, vk_stat = vk_loaded

            pk_digest = _sha256_hex(pk_bytes)
            vk_digest = _sha256_hex(vk_bytes)
            if (
                pk_digest != artifacts_manifest["proving_key"]["sha256"]
                or pk_stat.st_size != artifacts_manifest["proving_key"]["size"]
                or vk_digest != artifacts_manifest["verifying_key"]["sha256"]
                or vk_stat.st_size != artifacts_manifest["verifying_key"]["size"]
            ):
                return cls.unready(
                    "artifact_manifest_digest_mismatch",
                    artifacts_root=str(root),
                    manifest_sha256=manifest_digest,
                )

            native_ready, native_reason, native_diagnostics = (
                validate_groth16_native_manifest_identity(
                    binary_path=binary_path,
                    environ=env,
                    required_circuit_version=_TEST_PASS_CIRCUIT_VERSION,
                )
            )
            if not native_ready:
                return cls.unready(
                    native_reason,
                    artifacts_root=str(root),
                    manifest_sha256=manifest_digest,
                    process_started=bool(
                        native_diagnostics.get("process_started", False)
                    ),
                )

            # Import only the narrow provider after all file-system trust
            # anchors have validated.  Missing optional dependencies remain a
            # typed unready result; no sha256-envelope fallback can authorize.
            try:
                provider = importlib.import_module(
                    "ipfs_datasets_py.logic.zkp.test_pass_groth16_provider"
                )
            except Exception as exc:
                return cls.unready(
                    "reviewed_provider_unavailable",
                    error_type=type(exc).__name__,
                    manifest_sha256=manifest_digest,
                )
            provider_source_text = inspect.getsourcefile(provider) or str(
                getattr(provider, "__file__", "") or ""
            )
            provider_loaded = (
                _read_regular_file(
                    Path(provider_source_text), max_bytes=4 * 1024 * 1024
                )
                if provider_source_text
                else None
            )
            if (
                provider_loaded is None
                or _sha256_hex(provider_loaded[0])
                != TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256
            ):
                return cls.unready(
                    "reviewed_provider_source_mismatch",
                    manifest_sha256=manifest_digest,
                )
            # Legacy provider re-exported circuit version/ruleset constants.
            # V5 native provider exports TestPassStatementV5 version instead.
            legacy_version = getattr(provider, "TEST_PASS_GROTH16_CIRCUIT_VERSION", None)
            legacy_ruleset = getattr(provider, "TEST_PASS_GROTH16_RULESET_ID", None)
            v5_version = getattr(provider, "TEST_PASS_STATEMENT_V5_VERSION", None)
            v5_ruleset = getattr(provider, "TEST_PASS_V5_RULESET_ID", None)
            if legacy_version is not None or legacy_ruleset is not None:
                if (
                    legacy_version != _TEST_PASS_CIRCUIT_VERSION
                    or legacy_ruleset != TEST_PASS_GROTH16_RULESET_ID
                ):
                    return cls.unready("reviewed_provider_identity_mismatch")
            elif v5_version is not None or v5_ruleset is not None:
                if (
                    v5_version != _TEST_PASS_CIRCUIT_VERSION
                    or v5_ruleset != TEST_PASS_GROTH16_RULESET_ID
                ):
                    return cls.unready("reviewed_provider_identity_mismatch")
            else:
                return cls.unready("reviewed_provider_identity_mismatch")
            try:
                # Prefer provider helpers when present; otherwise derive from
                # accelerate-reviewed circuit constants + verifying-key bytes.
                if hasattr(provider, "reviewed_circuit_cid") and hasattr(
                    provider, "verifying_key_cid_for_bytes"
                ):
                    circuit_cid = provider.reviewed_circuit_cid()
                    verifying_key_cid = provider.verifying_key_cid_for_bytes(vk_bytes)
                else:
                    circuit_cid = TEST_PASS_GROTH16_CIRCUIT_CID
                    verifying_key_cid = _dag_json_cid(
                        {
                            "artifact": "groth16_verifying_key",
                            "backend_circuit_version": _TEST_PASS_CIRCUIT_VERSION,
                            "sha256": vk_digest,
                            "size": len(vk_bytes),
                        }
                    )
            except Exception as exc:
                return cls.unready(
                    "cid_derivation_failed",
                    error_type=type(exc).__name__,
                    artifacts_root=str(root),
                )
            independent_vk_cid = _dag_json_cid(
                {
                    "artifact": "groth16_verifying_key",
                    "backend_circuit_version": _TEST_PASS_CIRCUIT_VERSION,
                    "sha256": vk_digest,
                    "size": len(vk_bytes),
                }
            )
            if (
                circuit_cid != TEST_PASS_GROTH16_CIRCUIT_CID
                or verifying_key_cid != independent_vk_cid
            ):
                return cls.unready(
                    "cid_derivation_mismatch",
                    artifacts_root=str(root),
                )

            return cls(
                circuit_cid=circuit_cid,
                verifying_key_cid=verifying_key_cid,
                artifacts_root=str(root.resolve()),
                verifying_key_sha256=vk_digest,
                proving_key_sha256=pk_digest,
                backend_circuit_version=int(circuit_version),
                reviewed_revision=DATASETS_VERIFIER_REVISION,
                provenance_ready=True,
                reason_code="ready",
                diagnostics=MappingProxyType(
                    {
                        "binary_present": True,
                        "binary_alone_non_authoritative": True,
                        "key_version": int(circuit_version),
                        "manifest_sha256": manifest_digest,
                        "manifest_path": str(manifest_path or "")[:256],
                        "native_provenance": str(
                            native_diagnostics.get("native_provenance", "")
                        )[:64],
                        "native_v4_capability_validated": True,
                        "provider_source_sha256": (
                            TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256
                        ),
                    }
                ),
            )
        except Exception as exc:
            return cls.unready(
                "artifact_binding_exception",
                error_type=type(exc).__name__,
            )


@dataclass(frozen=True, slots=True)
class IssuedCertificatePublicationResult:
    """Outcome of one controller publication transaction."""

    interface: str = ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE
    published: bool = False
    status: str = "deferred"
    reason_code: str = ""
    receipt_cid: str = ""
    certificate_cid: str = ""
    candidate_context_cid: str = ""
    indexed: bool = False
    put_candidate_called: bool = False
    non_authoritative_retained: bool = False
    action: str = "DEFERRED"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def authorizes_skip(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "published": self.published,
            "status": self.status,
            "reason_code": self.reason_code,
            "receipt_cid": self.receipt_cid,
            "certificate_cid": self.certificate_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "indexed": self.indexed,
            "put_candidate_called": self.put_candidate_called,
            "non_authoritative_retained": self.non_authoritative_retained,
            "action": self.action,
            "authorizes_skip": False,
            "diagnostics": dict(self.diagnostics),
        }


class ProofReuseControllerPublicationTransaction:
    """Atomic controller-owned issue → exact-V2-verify → put_candidate sequence.

    Implements ``ProofReuseControllerPublicationTransaction@1``.

    Authority order (PTR-155):

    1. Cold-write non-authoritative candidate components + receipt (rehash).
    2. Reconstruct public deferred request / controller V2 context (no private
       worker material; certificate fields never fill missing pins).
    3. Call issuer when no certificate is attached (immediate or deferred).
    4. Require provenance-ready PTR-151 bindings (or an explicit disposable
       test-only fixture that is never production authority).
    5. Invoke ``verify_test_execution_certificate_v2`` with the pinned backend
       and expected candidate-context CID; require
       ``CertificateVerificationStatus.VERIFIED``.
    6. Atomically ``put_candidate`` exactly once after VERIFIED only.
    7. On failure/deferral, retain immutable non-authoritative candidate/receipt
       for retry; never publish a partial skip candidate.
    """

    interface: str = PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE

    def __init__(
        self,
        *,
        store: Any = None,
        candidate_store: Any = None,
        issuer: Any = None,
        owner_id: str = "",
        artifact_bindings: Groth16ArtifactIdentityBindings | None = None,
        metrics: Any = None,
        test_only_verification_backend: Any = None,
    ) -> None:
        self.store = store
        self.candidate_store = candidate_store
        self.issuer = issuer
        self.owner_id = _bounded_text(owner_id, max_chars=128)
        self.artifact_bindings = artifact_bindings
        self.metrics = metrics
        # Disposable test-only backend; only consulted for test_only bindings.
        self.test_only_verification_backend = test_only_verification_backend
        self._lock = threading.RLock()
        self._completed_intents: set[str] = set()

    def _retain_non_authoritative(
        self,
        *,
        receipt: Any,
        locator_cid: str,
        candidate_components: Mapping[str, bytes] | None = None,
        publication_envelope: Any = None,
    ) -> tuple[bool, str]:
        """Write immutable candidate/receipt without index skip authority."""

        retained = False
        candidate_cid = ""
        store = self.candidate_store
        if store is not None:
            try:
                # Prefer full envelope publish when available.
                if publication_envelope is not None:
                    publish = getattr(store, "publish", None)
                    if callable(publish):
                        try:
                            result = publish(publication_envelope)
                            retained = True
                            candidate_cid = _bounded_text(
                                getattr(result, "candidate_context_cid", "")
                                or getattr(result, "cid", "")
                                or getattr(publication_envelope, "candidate_context_cid", "")
                            )
                        except TypeError:
                            # Signature variants — fall through to put_canonical.
                            pass
                        except Exception:
                            pass
                if candidate_components:
                    put_bytes = getattr(store, "put_canonical_bytes", None)
                    if callable(put_bytes):
                        for _name, payload in candidate_components.items():
                            if isinstance(payload, (bytes, bytearray)) and payload:
                                try:
                                    put_bytes(bytes(payload))
                                    retained = True
                                except Exception:
                                    continue
                # Always attempt receipt retention on the candidate store CAS.
                if receipt is not None:
                    put_bytes = getattr(store, "put_canonical_bytes", None)
                    receipt_bytes = None
                    if isinstance(receipt, (bytes, bytearray)):
                        receipt_bytes = bytes(receipt)
                    else:
                        canonical = getattr(receipt, "canonical_bytes", None)
                        if callable(canonical):
                            try:
                                receipt_bytes = bytes(canonical())
                            except Exception:
                                receipt_bytes = None
                        elif isinstance(receipt, Mapping):
                            try:
                                receipt_bytes = json.dumps(
                                    dict(receipt),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            except Exception:
                                receipt_bytes = None
                    if receipt_bytes and callable(put_bytes):
                        try:
                            put_bytes(receipt_bytes)
                            retained = True
                        except Exception:
                            pass
            except Exception:
                retained = retained or False

        # Certificate-store put_receipt is intentionally deferred until the
        # no-certificate path: a later successful put_candidate must remain the
        # sole authority write and must not be preceded by a partial indexable
        # receipt publication on the certificate store.
        return retained, candidate_cid

    def _put_candidate_once(
        self,
        *,
        receipt: Any,
        certificate: Any,
        locator_cid: str,
    ) -> tuple[bool, bool, str]:
        """Atomically publish complete candidate; never partial index."""

        store = self.store
        if store is None:
            return False, False, "store_unavailable"
        method = getattr(store, "put_candidate", None)
        if not callable(method):
            return False, False, "put_candidate_unavailable"
        kwargs: dict[str, Any] = {}
        if locator_cid:
            kwargs["locator_cid"] = locator_cid
        if self.owner_id:
            kwargs["owner_id"] = self.owner_id
        try:
            signature = inspect.signature(method)
            try:
                signature.bind(receipt, certificate, **kwargs)
                selected_kwargs = kwargs
            except TypeError:
                signature.bind(receipt, certificate)
                selected_kwargs = {}
        except (TypeError, ValueError):
            return False, False, "put_candidate_signature_incompatible"
        try:
            # Exactly one invocation.  A TypeError raised inside the store is
            # a failed write, not permission to retry and possibly double-index.
            result = method(receipt, certificate, **selected_kwargs)
        except Exception:
            return False, False, "put_candidate_failed"

        stored = getattr(result, "stored", None)
        indexed = getattr(result, "indexed", None)
        stored_ok = stored is True
        indexed_ok = indexed is True
        if not (stored_ok and indexed_ok):
            return False, False, "put_candidate_rejected"
        return True, True, "published"

    def publish_intent(
        self,
        intent: Any,
        *,
        store: Any = None,
        candidate_store: Any = None,
        issuer: Any = None,
        deferred_request: Mapping[str, Any] | None = None,
        candidate_components: Mapping[str, bytes] | None = None,
        publication_envelope: Any = None,
    ) -> IssuedCertificatePublicationResult:
        """Run one complete controller publication transaction.

        Never raises; failures return DEFERRED/RUN with optional non-authoritative
        retention for retry.
        """

        if store is not None:
            self.store = store
        if candidate_store is not None:
            self.candidate_store = candidate_store
        if issuer is not None:
            self.issuer = issuer

        receipt = getattr(intent, "receipt", intent)
        receipt_cid = _bounded_text(
            getattr(intent, "receipt_cid", None)
            or (receipt.get("receipt_id") if isinstance(receipt, Mapping) else "")
            or getattr(receipt, "receipt_id", "")
        )
        locator_cid = _bounded_text(
            getattr(intent, "locator_cid", None)
            or (receipt.get("locator_cid") if isinstance(receipt, Mapping) else "")
            or getattr(receipt, "locator_cid", "")
        )
        intent_id = _bounded_text(
            getattr(intent, "intent_id", None) or receipt_cid, max_chars=128
        )
        existing_certificate = getattr(intent, "certificate", None)

        with self._lock:
            if intent_id and intent_id in self._completed_intents:
                return IssuedCertificatePublicationResult(
                    published=True,
                    status="already_published",
                    reason_code="idempotent_skip",
                    receipt_cid=receipt_cid,
                    action="DEFERRED",
                )

        # 1. Cold retain candidate + receipt before issuance (non-authoritative).
        retained, candidate_cid = self._retain_non_authoritative(
            receipt=receipt,
            locator_cid=locator_cid,
            candidate_components=candidate_components,
            publication_envelope=publication_envelope,
        )

        certificate_payload = (
            dict(existing_certificate)
            if isinstance(existing_certificate, Mapping)
            else _mapping_of(existing_certificate)
        )

        # Retain any attached public certificate bytes without index authority.
        if certificate_payload is not None:
            try:
                certificate_bytes = json.dumps(
                    dict(certificate_payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                certificate_bytes = b""
            if certificate_bytes and len(certificate_bytes) <= 4 * 1024 * 1024:
                cert_retained, cert_candidate_cid = self._retain_non_authoritative(
                    receipt=None,
                    locator_cid=locator_cid,
                    candidate_components={"certificate": certificate_bytes},
                )
                retained = retained or cert_retained
                candidate_cid = candidate_cid or cert_candidate_cid

        # 2–3. Issue when no certificate is already attached.
        issue_result = None
        request = deferred_request
        if request is None:
            request = getattr(intent, "deferred_request", None)
        if request is None:
            request = {
                "receipt_cid": receipt_cid,
                "locator_cid": locator_cid,
            }
        if certificate_payload is None and self.issuer is not None:
            try:
                # Prefer issue_material when available (PTR-153 public material).
                issue_material = getattr(self.issuer, "issue_material", None)
                issue = getattr(self.issuer, "issue", None)
                if callable(issue_material):
                    issue_result = issue_material(request)
                elif callable(issue):
                    issue_result = issue(request)
                elif callable(self.issuer):
                    issue_result = self.issuer(request)
            except Exception:
                issue_result = None
            if _issue_succeeded(issue_result):
                certificate_payload = _extract_certificate_payload(issue_result)
            elif _issue_deferred(issue_result):
                reason = _bounded_text(
                    getattr(issue_result, "reason", None)
                    or getattr(issue_result, "reason_code", None)
                    or "certificate_deferred"
                )
                if self.metrics is not None:
                    try:
                        self.metrics.deferred(
                            reason_code=reason or "certificate_deferred"
                        )
                    except Exception:
                        pass
                return IssuedCertificatePublicationResult(
                    published=False,
                    status="certificate_deferred",
                    reason_code=reason or "certificate_deferred",
                    receipt_cid=receipt_cid,
                    candidate_context_cid=candidate_cid,
                    non_authoritative_retained=retained,
                    action="DEFERRED",
                    diagnostics=MappingProxyType({"stage": "issuance"}),
                )

        if certificate_payload is None:
            # No certificate available: retain non-authoritative receipt only.
            if self.store is not None:
                put_receipt = getattr(self.store, "put_receipt", None)
                if callable(put_receipt):
                    try:
                        put_receipt(receipt)
                        retained = True
                    except Exception:
                        pass
            if self.metrics is not None:
                try:
                    self.metrics.deferred(reason_code="certificate_deferred")
                except Exception:
                    pass
            return IssuedCertificatePublicationResult(
                published=False,
                status="certificate_deferred",
                reason_code="certificate_unavailable",
                receipt_cid=receipt_cid,
                candidate_context_cid=candidate_cid,
                non_authoritative_retained=retained,
                action="DEFERRED",
            )

        # 4. Artifact provenance pins (production ready or explicit test-only).
        bindings = self.artifact_bindings
        if bindings is None:
            bindings = getattr(self.issuer, "last_artifact_bindings", None)
        if not _bindings_usable_for_publication(bindings):
            reason = "artifact_provenance_unready"
            # Self-asserted provenance_ready without production/test-only path.
            if (
                isinstance(bindings, Groth16ArtifactIdentityBindings)
                and bindings.provenance_ready
                and not _is_test_only_disposable_bindings(bindings)
                and bindings.reason_code != _PRODUCTION_READY_REASON
            ):
                reason = "artifact_provenance_unready"
            elif (
                isinstance(bindings, Groth16ArtifactIdentityBindings)
                and bindings.provenance_ready
                and bindings.reason_code == _PRODUCTION_READY_REASON
                and not _bindings_production_ready(bindings)
            ):
                reason = "production_key_manifest_unapproved"
            if self.metrics is not None:
                try:
                    self.metrics.deferred(reason_code=reason)
                except Exception:
                    pass
            certificate_cid = ""
            if certificate_payload is not None:
                certificate_cid = _bounded_text(
                    certificate_payload.get("certificate_id")
                    or certificate_payload.get("certificate_cid")
                    or getattr(intent, "certificate_cid", "")
                    or ""
                )
            else:
                certificate_cid = _bounded_text(
                    getattr(intent, "certificate_cid", "") or ""
                )
            return IssuedCertificatePublicationResult(
                published=False,
                status="certificate_deferred",
                reason_code=reason,
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
                candidate_context_cid=candidate_cid,
                non_authoritative_retained=retained,
                action="DEFERRED",
                diagnostics=MappingProxyType({"stage": "artifact_provenance"}),
            )
        assert isinstance(bindings, Groth16ArtifactIdentityBindings)

        # 5. Reconstruct controller-owned V2 expected context (PTR-154 join).
        is_test_only = _is_test_only_disposable_bindings(bindings)
        controller_context = ControllerV2VerificationContext.from_sources(
            deferred_request=request,
            receipt=receipt,
            bindings=bindings,
            is_test_only_disposable=is_test_only,
        )
        if controller_context is None or not controller_context.is_complete:
            reason = "controller_v2_context_incomplete"
            if self.metrics is not None:
                try:
                    self.metrics.deferred(reason_code=reason)
                except Exception:
                    pass
            return IssuedCertificatePublicationResult(
                published=False,
                status="certificate_deferred",
                reason_code=reason,
                receipt_cid=receipt_cid,
                candidate_context_cid=candidate_cid,
                non_authoritative_retained=retained,
                action="DEFERRED",
                diagnostics=MappingProxyType({"stage": "controller_v2_context"}),
            )

        # Prefer controller candidate CID for the publication result envelope.
        expected_candidate_cid = (
            controller_context.expected_candidate_context_cid
            or controller_context.candidate_context_cid
        )
        if expected_candidate_cid:
            candidate_cid = candidate_cid or expected_candidate_cid

        module_provenance_validator = getattr(
            self.issuer, "validate_authority_module_provenance", None
        )
        # Injected issuer.verify_certificate_locally is intentionally ignored.
        verified, verify_reason, verify_result = (
            verify_test_execution_certificate_v2_for_publication(
                certificate_payload,
                bindings=bindings,
                controller_context=controller_context,
                test_only_backend=self.test_only_verification_backend,
                module_provenance_validator=module_provenance_validator,
            )
        )
        if not verified:
            if self.metrics is not None:
                try:
                    self.metrics.deferred(
                        reason_code=verify_reason or "local_verification_failed"
                    )
                except Exception:
                    pass
            return IssuedCertificatePublicationResult(
                published=False,
                status="certificate_deferred",
                reason_code=verify_reason or "local_verification_failed",
                receipt_cid=receipt_cid,
                candidate_context_cid=candidate_cid,
                non_authoritative_retained=retained,
                action="DEFERRED",
                diagnostics=MappingProxyType(
                    {
                        "stage": "exact_v2_verification",
                        "verify_status": _status_text(
                            getattr(
                                getattr(verify_result, "status", None),
                                "value",
                                getattr(verify_result, "status", None),
                            )
                        ),
                        "test_only_disposable": is_test_only,
                    }
                ),
            )

        # 6. Atomic put_candidate exactly once — never discard a returned cert.
        certificate_cid = _bounded_text(
            certificate_payload.get("certificate_id")
            or certificate_payload.get("certificate_cid")
            or getattr(issue_result, "certificate_cid", "")
        )
        ok, indexed, pub_reason = self._put_candidate_once(
            receipt=receipt,
            certificate=certificate_payload,
            locator_cid=locator_cid,
        )
        if not ok:
            if self.metrics is not None:
                try:
                    self.metrics.degraded(
                        reason_code=pub_reason or "publication_failed"
                    )
                except Exception:
                    pass
            return IssuedCertificatePublicationResult(
                published=False,
                status="publication_failed",
                reason_code=pub_reason or "publication_failed",
                receipt_cid=receipt_cid,
                certificate_cid=certificate_cid,
                candidate_context_cid=candidate_cid,
                put_candidate_called=True,
                non_authoritative_retained=retained,
                action="DEFERRED",
                diagnostics=MappingProxyType({"stage": "put_candidate"}),
            )

        with self._lock:
            if intent_id:
                self._completed_intents.add(intent_id)

        published_reason = "published"
        if is_test_only or str(verify_reason).startswith("verified_test_only"):
            published_reason = "published_test_only_disposable"
        return IssuedCertificatePublicationResult(
            published=True,
            status="certificate_issued",
            reason_code=published_reason,
            receipt_cid=receipt_cid,
            certificate_cid=certificate_cid,
            candidate_context_cid=candidate_cid,
            indexed=indexed,
            put_candidate_called=True,
            non_authoritative_retained=retained,
            action="RUN",
            diagnostics=MappingProxyType(
                {
                    "stage": "complete",
                    "verify_reason": verify_reason,
                    "expected_candidate_context_cid": expected_candidate_cid,
                    "test_only_disposable": is_test_only
                    or str(verify_reason).startswith("verified_test_only"),
                    "production_authority": (
                        not is_test_only
                        and _bindings_production_ready(bindings)
                        and not str(verify_reason).startswith("verified_test_only")
                    ),
                }
            ),
        )


class ControllerCandidatePublisher:
    """Controller-only signed-receipt and candidate publication authority (PTR-164).

    Implements ``ControllerCandidatePublisher@2``.

    * Only the controller role may sign terminal setup/call/teardown passes.
    * Workers supply bounded public envelopes and never private keys or witnesses.
    * Signing produces public runner-attestation bytes retained with the candidate.
    * Publication delegates to
      :class:`ProofReuseControllerPublicationTransaction` so partial or racing
      writes never become skip-authorizing index entries.
    * Private key material is never serialized into intents, packets, or logs.
    """

    interface: str = CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE

    def __init__(
        self,
        *,
        role: str = "controller",
        private_key: Any = None,
        trust_policy: Any = None,
        nonce_registry: Any = None,
        transaction: ProofReuseControllerPublicationTransaction | None = None,
        store: Any = None,
        candidate_store: Any = None,
        issuer: Any = None,
        owner_id: str = "",
        artifact_bindings: Groth16ArtifactIdentityBindings | None = None,
        metrics: Any = None,
        clock: Callable[[], int] | None = None,
        test_only_verification_backend: Any = None,
    ) -> None:
        self.role = _bounded_text(role, max_chars=32).lower() or "controller"
        self._private_key = private_key
        self._trust_policy = trust_policy
        self._nonce_registry = nonce_registry
        self._transaction = transaction
        self._store = store
        self._candidate_store = candidate_store
        self._issuer = issuer
        self._owner_id = _bounded_text(owner_id, max_chars=128)
        self._artifact_bindings = artifact_bindings
        self._metrics = metrics
        self._clock = clock
        self._test_only_verification_backend = test_only_verification_backend
        self._lock = threading.RLock()
        self._signed_intents: set[str] = set()

    @property
    def is_controller(self) -> bool:
        return self.role in {"controller", "master", "gwmaster", ""}

    @property
    def can_sign(self) -> bool:
        return (
            self.is_controller
            and self._private_key is not None
            and self._trust_policy is not None
        )

    @property
    def can_publish(self) -> bool:
        return self.is_controller

    def _ensure_transaction(self) -> ProofReuseControllerPublicationTransaction:
        if self._transaction is not None:
            return self._transaction
        self._transaction = ProofReuseControllerPublicationTransaction(
            store=self._store,
            candidate_store=self._candidate_store,
            issuer=self._issuer,
            owner_id=self._owner_id,
            artifact_bindings=self._artifact_bindings,
            metrics=self._metrics,
            test_only_verification_backend=self._test_only_verification_backend,
        )
        return self._transaction

    def sign_complete_pass(
        self,
        receipt: Any,
        *,
        candidate_context_cid: str,
        issuance_nonce: str | None = None,
        issued_at: int | None = None,
    ) -> tuple[Any | None, str]:
        """Controller-sign one admitted complete pass; workers always fail.

        Returns ``(attestation_or_none, reason_code)``.  Never raises.  Never
        returns private key material.
        """

        if not self.is_controller:
            return None, "worker_cannot_sign"
        if self._private_key is None or self._trust_policy is None:
            return None, "controller_signing_material_unavailable"
        try:
            from ...agent_supervisor.proof.test_execution_contracts import (
                TestPassReceipt,
            )
            from .runner_pass_attestation import attest_test_pass_receipt

            if not isinstance(receipt, TestPassReceipt):
                if isinstance(receipt, Mapping):
                    receipt = TestPassReceipt.from_dict(receipt)
                else:
                    return None, "receipt_invalid"
            if not receipt.admitted or not receipt.all_phases_pass:
                return None, "receipt_not_complete_pass"
            now = (
                int(issued_at)
                if issued_at is not None
                else (int(self._clock()) if self._clock is not None else None)
            )
            attestation = attest_test_pass_receipt(
                receipt,
                private_key=self._private_key,
                policy=self._trust_policy,
                candidate_context_cid=str(candidate_context_cid or ""),
                issuance_nonce=issuance_nonce,
                issued_at=now,
                nonce_registry=self._nonce_registry,
            )
            return attestation, "signed"
        except Exception as exc:
            return None, f"controller_sign_failed:{type(exc).__name__}"[:128]

    def public_attestation_envelope(
        self,
        attestation: Any,
        *,
        signed_receipt: Any = None,
    ) -> dict[str, Any] | None:
        """Project attestation to a public-only mapping (no private fields)."""

        try:
            payload: dict[str, Any] = {}
            if attestation is None:
                return None
            if hasattr(attestation, "to_dict"):
                raw = attestation.to_dict()
                if isinstance(raw, Mapping):
                    payload = {
                        key: value
                        for key, value in raw.items()
                        if "private" not in str(key).lower()
                        and "secret" not in str(key).lower()
                        and "witness" not in str(key).lower()
                        and "seed" not in str(key).lower()
                    }
            elif isinstance(attestation, Mapping):
                payload = {
                    key: value
                    for key, value in attestation.items()
                    if "private" not in str(key).lower()
                    and "secret" not in str(key).lower()
                    and "witness" not in str(key).lower()
                }
            else:
                return None
            if signed_receipt is not None and hasattr(signed_receipt, "to_dict"):
                try:
                    payload["signed_receipt"] = signed_receipt.to_dict()
                except Exception:
                    pass
            # Never include private key handles.
            payload.pop("private_key", None)
            payload.pop("signing_key", None)
            return payload
        except Exception:
            return None

    def retain_signed_attestation(
        self,
        *,
        attestation: Any,
        receipt: Any = None,
    ) -> tuple[bool, str]:
        """Retain immutable public attestation/receipt bytes without indexing."""

        if not self.is_controller:
            return False, "worker_cannot_publish"
        store = self._candidate_store
        if store is None:
            return False, "candidate_store_unavailable"
        put_bytes = getattr(store, "put_canonical_bytes", None)
        if not callable(put_bytes):
            return False, "put_canonical_bytes_unavailable"
        retained = False
        try:
            for material in (attestation, receipt):
                if material is None:
                    continue
                raw = None
                if isinstance(material, (bytes, bytearray)):
                    raw = bytes(material)
                else:
                    canonical = getattr(material, "canonical_bytes", None)
                    if callable(canonical):
                        raw = bytes(canonical())
                if raw:
                    put_bytes(raw)
                    retained = True
            return retained, "retained" if retained else "nothing_to_retain"
        except Exception:
            return False, "retain_failed"

    def publish(
        self,
        intent: Any,
        *,
        store: Any = None,
        candidate_store: Any = None,
        issuer: Any = None,
        deferred_request: Mapping[str, Any] | None = None,
        candidate_components: Mapping[str, bytes] | None = None,
        publication_envelope: Any = None,
        candidate_context_cid: str = "",
        sign: bool = True,
    ) -> IssuedCertificatePublicationResult:
        """Controller-only sign (optional) then atomic publish.

        Workers always receive a non-published RUN result.  Partial or racing
        writes never produce an indexed skip-authorizing candidate.
        """

        if not self.is_controller:
            return IssuedCertificatePublicationResult(
                published=False,
                indexed=False,
                put_candidate_called=False,
                status="run",
                reason_code="worker_cannot_publish",
                action="RUN",
                diagnostics=MappingProxyType(
                    {
                        "stage": "controller_only",
                        "role": self.role,
                    }
                ),
            )

        with self._lock:
            # Optional controller signature of the terminal pass receipt.
            attestation = None
            sign_reason = "not_requested"
            if sign and self.can_sign:
                receipt = getattr(intent, "receipt", intent)
                context_cid = (
                    candidate_context_cid
                    or str(
                        getattr(publication_envelope, "candidate_context_cid", "")
                        or getattr(intent, "locator_cid", "")
                        or ""
                    )
                )
                attestation, sign_reason = self.sign_complete_pass(
                    receipt,
                    candidate_context_cid=context_cid,
                )
                if attestation is not None:
                    self.retain_signed_attestation(
                        attestation=attestation,
                        receipt=receipt,
                    )
                    # Attach public attestation bytes into candidate components
                    # for later warm-path trust verification.
                    try:
                        att_bytes = bytes(attestation.canonical_bytes())
                        components = dict(candidate_components or {})
                        components["runner_attestation"] = att_bytes
                        candidate_components = components
                    except Exception:
                        pass

            transaction = self._ensure_transaction()
            try:
                outcome = transaction.publish_intent(
                    intent,
                    store=store if store is not None else self._store,
                    candidate_store=(
                        candidate_store
                        if candidate_store is not None
                        else self._candidate_store
                    ),
                    issuer=issuer if issuer is not None else self._issuer,
                    deferred_request=deferred_request,
                    candidate_components=candidate_components,
                    publication_envelope=publication_envelope,
                )
            except Exception as exc:
                return IssuedCertificatePublicationResult(
                    published=False,
                    indexed=False,
                    put_candidate_called=False,
                    status="run",
                    reason_code="controller_publish_exception",
                    action="RUN",
                    diagnostics=MappingProxyType(
                        {
                            "stage": "controller_publish",
                            "exception_type": type(exc).__name__[:64],
                            "controller_sign_reason": sign_reason,
                            "controller_signed": attestation is not None,
                        }
                    ),
                )
            # Annotate diagnostics with signing disposition (public only).
            diagnostics = dict(getattr(outcome, "diagnostics", {}) or {})
            diagnostics["controller_sign_reason"] = sign_reason
            diagnostics["controller_signed"] = attestation is not None
            if attestation is not None:
                diagnostics["runner_attestation_cid"] = _bounded_text(
                    getattr(attestation, "cid", "")
                )
            return IssuedCertificatePublicationResult(
                interface=getattr(
                    outcome,
                    "interface",
                    ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE,
                ),
                published=bool(getattr(outcome, "published", False)),
                status=str(getattr(outcome, "status", "deferred") or "deferred"),
                reason_code=str(getattr(outcome, "reason_code", "") or ""),
                receipt_cid=str(getattr(outcome, "receipt_cid", "") or ""),
                certificate_cid=str(getattr(outcome, "certificate_cid", "") or ""),
                candidate_context_cid=str(
                    getattr(outcome, "candidate_context_cid", "") or ""
                ),
                indexed=bool(getattr(outcome, "indexed", False)),
                put_candidate_called=bool(
                    getattr(outcome, "put_candidate_called", False)
                ),
                non_authoritative_retained=bool(
                    getattr(outcome, "non_authoritative_retained", False)
                ),
                action=str(getattr(outcome, "action", "DEFERRED") or "DEFERRED"),
                diagnostics=MappingProxyType(diagnostics),
            )

    def reject_worker_private_material(
        self, payload: Mapping[str, Any] | None
    ) -> tuple[bool, str]:
        """Return ``(accepted, reason)`` after scanning for private markers."""

        if payload is None:
            return True, "empty"
        if not isinstance(payload, Mapping):
            return False, "payload_not_mapping"
        private_markers = (
            "private",
            "secret",
            "witness",
            "seed",
            "proving_key",
            "signing_key",
            "private_key",
        )
        stack: list[Any] = [payload]
        seen = 0
        while stack and seen < 256:
            seen += 1
            current = stack.pop()
            if isinstance(current, Mapping):
                for key, value in current.items():
                    lowered = str(key).lower().replace("-", "_")
                    if any(marker in lowered for marker in private_markers):
                        return False, f"private_field:{lowered[:64]}"
                    if isinstance(value, (Mapping, list, tuple)):
                        stack.append(value)
            elif isinstance(current, (list, tuple)):
                stack.extend(list(current)[:64])
        return True, "public_only"


def build_controller_candidate_publisher(
    *,
    role: str = "controller",
    private_key: Any = None,
    trust_policy: Any = None,
    nonce_registry: Any = None,
    transaction: ProofReuseControllerPublicationTransaction | None = None,
    store: Any = None,
    candidate_store: Any = None,
    issuer: Any = None,
    owner_id: str = "",
    artifact_bindings: Groth16ArtifactIdentityBindings | None = None,
    metrics: Any = None,
    clock: Callable[[], int] | None = None,
    test_only_verification_backend: Any = None,
) -> ControllerCandidatePublisher:
    """Factory for the controller-only candidate publisher."""

    return ControllerCandidatePublisher(
        role=role,
        private_key=private_key,
        trust_policy=trust_policy,
        nonce_registry=nonce_registry,
        transaction=transaction,
        store=store,
        candidate_store=candidate_store,
        issuer=issuer,
        owner_id=owner_id,
        artifact_bindings=artifact_bindings,
        metrics=metrics,
        clock=clock,
        test_only_verification_backend=test_only_verification_backend,
    )


__all__ = [
    "CONTROLLER_CANDIDATE_PUBLISHER_INTERFACE",
    "CONTROLLER_V2_VERIFICATION_CONTEXT_INTERFACE",
    "ControllerCandidatePublisher",
    "ControllerV2VerificationContext",
    "GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE",
    "Groth16ArtifactIdentityBindings",
    "ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE",
    "IssuedCertificatePublicationResult",
    "PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE",
    "ProofReuseControllerPublicationTransaction",
    "build_controller_candidate_publisher",
    "verify_test_execution_certificate_v2_for_publication",
]
