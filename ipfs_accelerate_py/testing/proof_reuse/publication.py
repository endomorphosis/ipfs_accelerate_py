"""Controller-owned atomic issuance and certificate publication (PTR-147).

Cold publication writes and rehashes candidate components and the pass receipt.
Positive v4 issuance and verification remain closed until PTR-155 supplies an
unforgeable exact authority path; no asserted binding or injected verifier can
reach a native provider or ``put_candidate`` in the interim.

Workers serialize no witness/private material.  A crash or failure may leave an
immutable non-authoritative candidate/receipt for retry but never a partial
skip candidate.  Cache, issuer, Groth16, transport, lock, permission, or
controller absence preserves the pass and returns RUN/DEFERRED.
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
from collections.abc import Mapping
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
ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE: Final = (
    "IssuedCertificatePublicationResult@1"
)
GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE: Final = (
    "Groth16ArtifactIdentityBindings@1"
)

# Test-pass circuit version introduced by PTR-144.
_TEST_PASS_CIRCUIT_VERSION: Final = TEST_PASS_GROTH16_CIRCUIT_VERSION
_GROTH16_MANIFEST_MAX_BYTES: Final = 64 * 1024
_GROTH16_RECEIPT_MAX_BYTES: Final = 64 * 1024
_GROTH16_KEY_MAX_BYTES: Final = 64 * 1024 * 1024
_GROTH16_BINARY_MAX_BYTES: Final = 128 * 1024 * 1024
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_POSITIVE_V4_AUTHORITY_ENABLED: Final = False


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
        "interface": "TestPassGroth16CircuitV4",
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
        "proving_key": "v4/proving_key.bin",
        "verifying_key": "v4/verifying_key.bin",
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


def _local_verify_certificate(
    certificate: Mapping[str, Any] | Any,
    *,
    bindings: "Groth16ArtifactIdentityBindings | None" = None,
    require_cryptographic_verify: bool = False,
    cryptographic_verifier: Any = None,
    module_provenance_validator: Any = None,
    verification_context: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Locally verify one returned certificate; never raise.

    When *bindings* are provenance-ready, the certificate circuit/VK CIDs must
    match pins derived from exact activated bytes, and the backend artifact
    root (when present) must match those pins.  Missing/mismatched provenance
    returns failure so callers surface RUN/DEFERRED.

    Controller publication passes ``require_cryptographic_verify=True``.  In
    that mode missing artifact provenance, verifier-module provenance, or a
    real verification result is a hard DEFERRED outcome; structural shape is
    never certificate authority.
    """

    try:
        cert_map = _mapping_of(certificate) or {}
        enforce_pins = (
            bindings is not None and getattr(bindings, "provenance_ready", False)
        )
        if not enforce_pins:
            return False, "artifact_provenance_unready"
        if enforce_pins:
            assert bindings is not None
            cert_circuit = _bounded_text(
                cert_map.get("circuit_cid")
                or getattr(certificate, "circuit_cid", "")
            )
            cert_vk = _bounded_text(
                cert_map.get("verifying_key_cid")
                or getattr(certificate, "verifying_key_cid", "")
            )
            if cert_circuit and bindings.circuit_cid and cert_circuit != bindings.circuit_cid:
                return False, "circuit_cid_mismatch"
            if cert_vk and bindings.verifying_key_cid and cert_vk != bindings.verifying_key_cid:
                return False, "verifying_key_cid_mismatch"
            # Prove the backend used the artifact root matching those pins.
            claimed_root = _bounded_text(
                cert_map.get("artifacts_root")
                or cert_map.get("artifact_root")
                or (
                    (cert_map.get("extra") or {}).get("artifacts_root")
                    if isinstance(cert_map.get("extra"), Mapping)
                    else ""
                )
            )
            if (
                claimed_root
                and bindings.artifacts_root
                and Path(claimed_root).resolve()
                != Path(bindings.artifacts_root).resolve()
            ):
                return False, "artifact_root_mismatch"

        # Structural completeness: required public identity fields.
        required = (
            "receipt_cid",
            "execution_key_cid",
            "circuit_cid",
            "verifying_key_cid",
        )
        structural_ok = all(
            _bounded_text(cert_map.get(name) or getattr(certificate, name, ""))
            for name in required
        )
        if not structural_ok:
            return False, "certificate_structurally_incomplete"

        if callable(cryptographic_verifier):
            try:
                result = cryptographic_verifier(
                    certificate,
                    bindings,
                    verification_context or MappingProxyType({}),
                )
            except Exception:
                result = None
            if result is True and not require_cryptographic_verify:
                return True, "verified"
            if result is not None:
                status = _status_text(getattr(result, "status", result))
                authority = _status_text(getattr(result, "authority", ""))
                if (
                    getattr(result, "verified", None) is True
                    and getattr(result, "authoritative", None) is True
                    and getattr(result, "can_authorize_skip", None) is True
                    and status == "verified"
                    and authority == "authoritative"
                ):
                    return True, "verified"
            return False, "local_verification_failed"

        # There is intentionally no ambient import or structural fallback.
        # PTR-153/154/155 will preserve proof-bearing material and construct the
        # controller-owned V2 context needed by the callback above.
        return False, "local_verification_unavailable"
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
            expected_provider_constants = {
                "TEST_PASS_GROTH16_CIRCUIT_VERSION": _TEST_PASS_CIRCUIT_VERSION,
                "TEST_PASS_GROTH16_RULESET_ID": TEST_PASS_GROTH16_RULESET_ID,
            }
            if any(
                getattr(provider, name, None) != value
                for name, value in expected_provider_constants.items()
            ):
                return cls.unready("reviewed_provider_identity_mismatch")
            try:
                circuit_cid = provider.reviewed_circuit_cid()
                verifying_key_cid = provider.verifying_key_cid_for_bytes(vk_bytes)
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
    """Atomic controller-owned issue → verify → put_candidate sequence.

    Implements ``ProofReuseControllerPublicationTransaction@1``.

    Authority-enabled order (currently fenced after step 1):

    1. Cold-write non-authoritative candidate components + receipt (rehash).
    2. Reconstruct public deferred request (no worker private material).
    3. Call issuer (immediate or deferred result).
    4. Locally verify every success against artifact pins.
    5. Atomically ``put_candidate`` exactly once for complete authority.
    6. On failure/deferral, retain immutable non-authoritative candidate/receipt
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
    ) -> None:
        self.store = store
        self.candidate_store = candidate_store
        self.issuer = issuer
        self.owner_id = _bounded_text(owner_id, max_chars=128)
        self.artifact_bindings = artifact_bindings
        self.metrics = metrics
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

        # PTR-152 is a denial boundary.  The current datasets provider proves
        # and verifies through mutable executable/key paths and inherits an
        # ambient child environment.  Do not invoke the issuer or local
        # verifier until PTR-155 supplies FD-bound inputs, a strict child
        # environment, and exact authority-module provenance.  An attached
        # public certificate may be retained in the candidate CAS, but it is
        # never indexed or treated as authority.
        if not _POSITIVE_V4_AUTHORITY_ENABLED:
            certificate_cid = ""
            if certificate_payload is not None:
                certificate_cid = _bounded_text(
                    certificate_payload.get("certificate_id")
                    or certificate_payload.get("certificate_cid")
                    or getattr(intent, "certificate_cid", "")
                )
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
                if (
                    certificate_bytes
                    and len(certificate_bytes) <= 4 * 1024 * 1024
                ):
                    cert_retained, cert_candidate_cid = (
                        self._retain_non_authoritative(
                            receipt=None,
                            locator_cid=locator_cid,
                            candidate_components={
                                "certificate": certificate_bytes,
                            },
                        )
                    )
                    retained = retained or cert_retained
                    candidate_cid = candidate_cid or cert_candidate_cid
            reason = "positive_v4_publication_pending_ptr155"
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
                certificate_cid=certificate_cid,
                candidate_context_cid=candidate_cid,
                indexed=False,
                put_candidate_called=False,
                non_authoritative_retained=retained,
                action="DEFERRED",
                diagnostics=MappingProxyType(
                    {"stage": "positive_publication_gate"}
                ),
            )

        # 2–3. Issue when no certificate is already attached.
        issue_result = None
        if certificate_payload is None and self.issuer is not None:
            request = deferred_request
            if request is None:
                request = getattr(intent, "deferred_request", None)
            if request is None:
                request = {
                    "receipt_cid": receipt_cid,
                    "locator_cid": locator_cid,
                }
            try:
                issue = getattr(self.issuer, "issue", None)
                if callable(issue):
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
                        self.metrics.deferred(reason_code=reason or "certificate_deferred")
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

        # 4. Local verification of every success.
        bindings = self.artifact_bindings
        if bindings is None:
            bindings = getattr(self.issuer, "last_artifact_bindings", None)
        if not isinstance(bindings, Groth16ArtifactIdentityBindings) or not (
            bindings.provenance_ready
            and bindings.circuit_cid
            and bindings.verifying_key_cid
            and bindings.artifacts_root
            and bindings.verifying_key_sha256
            and bindings.proving_key_sha256
            and bindings.backend_circuit_version == _TEST_PASS_CIRCUIT_VERSION
            and bindings.reviewed_revision == DATASETS_VERIFIER_REVISION
        ):
            reason = "artifact_provenance_unready"
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
                diagnostics=MappingProxyType({"stage": "artifact_provenance"}),
            )
        cryptographic_verifier = getattr(
            self.issuer, "verify_certificate_locally", None
        )
        module_provenance_validator = getattr(
            self.issuer, "validate_authority_module_provenance", None
        )
        verified, verify_reason = _local_verify_certificate(
            certificate_payload,
            bindings=bindings,
            require_cryptographic_verify=True,
            cryptographic_verifier=cryptographic_verifier,
            module_provenance_validator=module_provenance_validator,
            verification_context=MappingProxyType(
                {
                    "receipt": receipt,
                    "deferred_request": (
                        deferred_request
                        or getattr(intent, "deferred_request", None)
                        or MappingProxyType({})
                    ),
                }
            ),
        )
        if not verified:
            if self.metrics is not None:
                try:
                    self.metrics.deferred(reason_code=verify_reason or "local_verification_failed")
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
                diagnostics=MappingProxyType({"stage": "local_verification"}),
            )

        # 5. Atomic put_candidate exactly once — never discard a returned cert.
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
                    self.metrics.degraded(reason_code=pub_reason or "publication_failed")
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

        return IssuedCertificatePublicationResult(
            published=True,
            status="certificate_issued",
            reason_code="published",
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
                }
            ),
        )


__all__ = [
    "GROTH16_ARTIFACT_IDENTITY_BINDINGS_INTERFACE",
    "Groth16ArtifactIdentityBindings",
    "ISSUED_CERTIFICATE_PUBLICATION_RESULT_INTERFACE",
    "IssuedCertificatePublicationResult",
    "PROOF_REUSE_CONTROLLER_PUBLICATION_INTERFACE",
    "ProofReuseControllerPublicationTransaction",
]
