#!/usr/bin/env python3
"""Evaluate the frozen proof-context clean-install matrix without weakening it.

PCCE-053 qualifies dependency *resolution* for one environment.  It does not
qualify builds of the source distributions selected by those resolutions.
This harness therefore performs the immutable-artifact, lock, CID, and source
isolation preflight, then records an explicit clean-install no-go before pip,
imports, CLI smoke tests, or an image build can run.

An artifact directory is deliberately an explicit input.  The repository does
not know, discover, or fall back to an operator's private frozen-artifact path.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCRIPT_PATH = Path(__file__).resolve()
ACCELERATOR_ROOT = SCRIPT_PATH.parents[2]
LOCK_ROOT = ACCELERATOR_ROOT / "packaging" / "proof_context" / "locks"
ENVIRONMENT_SLUG = "cpython312-linux-aarch64"
PROFILE_ORDER = ("core", "verification", "codex", "local-model", "evaluation")
SCHEMA_PREFIX = "lift_coding.proof-carrying-context-engine"

ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}.artifact-hashes@1"
RESOLVER_SCHEMA = f"{SCHEMA_PREFIX}.resolver-receipt@1"
REPORT_SCHEMA = f"{SCHEMA_PREFIX}.clean-install-matrix@1"
RESOLUTION_STATUS = "supported-hash-bound"
CLEAN_INSTALL_NO_GO = "no-go-sdist-builds-not-qualified"
BYTE_AVAILABILITY_STATUS = "partial-one-admitted-sdist-unavailable"
PYPI_ORIGIN = "https://pypi.org/simple"
LOCAL_ORIGIN = "local-admitted-artifact"
EXIT_EVIDENCE_ERROR = 2
EXIT_REQUIRE_QUALIFIED_NO_GO = 5

EXPECTED_ENVIRONMENT = {
    "implementation": "CPython",
    "python": "3.12",
    "operating_system": "Linux",
    "architecture": "aarch64",
}
DIRECT_DISTRIBUTIONS = {
    "ipfs-accelerate-py",
    "ipfs-datasets-py",
    "ipfs-kit-py",
    "mcp-plus-plus-contracts",
}
SOURCE_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
EMPTY_UNSAFE_REQUIREMENTS = {
    "editable": [],
    "local-path": [],
    "mutable-vcs": [],
    "unadmitted-direct-url": [],
}
FROZEN_ARTIFACT_MANIFEST_IDENTITY = {
    "sha256": "b5b38995520aedd3392a205173182dcb07bc43361a5825b53639b985cb460ade",
    "cid_v1_raw": "bafkreifvwoezkuqk5xjtskrakfzrqlola66egnq2las3knrzxgc4wrqk3y",
}
FROZEN_PROFILE_INPUT_IDENTITIES = {
    "core": {
        "lock": {
            "sha256": "daf48e32318af4b07cc732b42ab8fe81e52862da3d251146acac45efd514f119",
            "cid_v1_raw": "bafkreig26shdemmk6syhzrzswqvlr7ub4uugfwr5euiunlfmixx5kfhrde",
        },
        "resolver_receipt": {
            "sha256": "e9e471da36b4ddc29d874a0a5561cfeb651c4e7e501f927aeb6aec79adc8fa43",
            "cid_v1_raw": "bafkreihj4ry5unvu3xbj3b2kbjkwdt7lmuoe47sqd6jhv23k5r423sh2im",
        },
    },
    "verification": {
        "lock": {
            "sha256": "3e85dbe837b4c6535b32fc69afeca7143936d63a9fed0b47bc101a4aa8ff36c3",
            "cid_v1_raw": "bafkreib6qxn6qn5uyzjvwmx4ngx6zjyuhe3nmou75ufuppaqdjfkr7zwym",
        },
        "resolver_receipt": {
            "sha256": "0ddc6427a613ddd198ff2994f16d4f43349cd9dbb77082eebbf6dcd008be3595",
            "cid_v1_raw": "bafkreian3rscpjqt3xizr7zjstyw2t2dgsontw5xocbo5o7w3tiarprvsu",
        },
    },
    "codex": {
        "lock": {
            "sha256": "5f95721c8b5f57d874803811ff9dad8f0b47a406c684ead25b00786ebf5862ec",
            "cid_v1_raw": "bafkreic7svzbzc27k7mhjabych7z3lmpbnd2ibwgqtvnewyapbxl6wdc5q",
        },
        "resolver_receipt": {
            "sha256": "e4e5ba494b4170924812684c3f1c1e720711ae8d694bd96447ff8675190e27ac",
            "cid_v1_raw": "bafkreihe4w5ess2bocjeqetijq7ryhtsa4i25dljjpmwir77qz2rsdrhvq",
        },
    },
    "local-model": {
        "lock": {
            "sha256": "332079812d0347d7aade896e59011913cbb754681e3f35c2ce7cf320cb41a904",
            "cid_v1_raw": "bafkreibteb4yclidi7l2vxujnzmqcgitzo3vi2a6h424ftt46mqmwqnjaq",
        },
        "resolver_receipt": {
            "sha256": "4845cd20ffc09ad4afaa4928e04a33cec379c0bb69e87daeefbf8741be74987d",
            "cid_v1_raw": "bafkreiciixgsb76atlkk7ksjfdqeum6oyn44bo3j5b625357q5a345eypu",
        },
    },
    "evaluation": {
        "lock": {
            "sha256": "87209db6618f8f9317352e35aa799eee4263bbc0e16ebb1004e7bfd238228618",
            "cid_v1_raw": "bafkreieheco3mympr6jronjogwvhthxoijr3xqhbn25rabhhx7jdqiugda",
        },
        "resolver_receipt": {
            "sha256": "79675f8c9b51568a023480e3cfa08c05772273b8eedbcb6b61033704242facaf",
            "cid_v1_raw": "bafkreidzm5pyzg2rk2faenea4ph2bdafo4rhhoho3pfwwyidg4ccil5mv4",
        },
    },
}


class EvidenceError(RuntimeError):
    """Raised when a claim would exceed the frozen evidence."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_cid_v1_from_sha256(sha256: str) -> str:
    """Return CIDv1(raw, sha2-256) for an exact SHA-256 digest."""
    if HEX_SHA256.fullmatch(sha256) is None:
        raise EvidenceError(f"invalid SHA-256 for raw CID derivation: {sha256!r}")
    payload = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    cid = "b" + base64.b32encode(payload).decode("ascii").lower().rstrip("=")
    _verify_raw_cid_v1(cid, sha256)
    return cid


def _verify_raw_cid_v1(cid: str, sha256: str) -> None:
    """Require canonical base32 CIDv1 raw bytes bound to ``sha256``."""
    if HEX_SHA256.fullmatch(sha256) is None:
        raise EvidenceError(f"invalid expected SHA-256: {sha256!r}")
    if not isinstance(cid, str) or not cid.startswith("b") or cid != cid.lower():
        raise EvidenceError(f"raw CIDv1 is not canonical base32-lower: {cid!r}")
    encoded = cid[1:].upper()
    encoded += "=" * ((8 - len(encoded) % 8) % 8)
    try:
        payload = base64.b32decode(encoded, casefold=False)
    except (ValueError, binascii.Error) as exc:
        raise EvidenceError(f"raw CIDv1 cannot be decoded: {cid!r}") from exc
    expected = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    if payload != expected:
        raise EvidenceError(f"raw CIDv1 does not bind expected SHA-256: {cid!r}")


def _identity_for_bytes(value: bytes) -> dict[str, str]:
    sha256 = _sha256_bytes(value)
    return {
        "sha256": sha256,
        "cid_v1_raw": _raw_cid_v1_from_sha256(sha256),
        "cid_binding_status": "bytes-verified",
    }


def _identity_for_path(path: Path) -> dict[str, str]:
    return _identity_for_bytes(path.read_bytes())


def _require_frozen_identity(
    label: str, actual: Mapping[str, str], expected: Mapping[str, str]
) -> None:
    expected_sha256 = expected.get("sha256", "")
    expected_cid = expected.get("cid_v1_raw", "")
    _verify_raw_cid_v1(expected_cid, expected_sha256)
    if actual.get("sha256") != expected_sha256 or actual.get("cid_v1_raw") != expected_cid:
        raise EvidenceError(
            f"{label} is not the exact frozen PCCE-053 identity: "
            f"expected {expected_sha256}/{expected_cid}, "
            f"observed {actual.get('sha256')}/{actual.get('cid_v1_raw')}"
        )


def _load_canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected a JSON object: {path}")
    if raw != _canonical_json_bytes(value):
        raise EvidenceError(f"JSON evidence is not in canonical PCCE encoding: {path}")
    return value, raw


def _normalized_name(value: Any) -> str:
    if not isinstance(value, str):
        raise EvidenceError(f"distribution name is not a string: {value!r}")
    name = re.sub(r"[-_.]+", "-", value).lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise EvidenceError(f"invalid distribution name: {value!r}")
    return name


def _safe_filename(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"artifact filename is invalid: {value!r}")
    if Path(value).name != value or value in {".", ".."} or "\\" in value:
        raise EvidenceError(f"artifact filename leaves its explicit root: {value!r}")
    return value


def _safe_artifact_path(root: Path, filename: str) -> Path:
    path = root / _safe_filename(filename)
    if path.is_symlink():
        raise EvidenceError(f"artifact must be a regular in-root file, not a symlink: {path}")
    return path


def _artifact_key(descriptor: Mapping[str, Any]) -> tuple[str, str]:
    version = descriptor.get("version")
    if not isinstance(version, str) or not version:
        raise EvidenceError(f"artifact version is invalid: {version!r}")
    return _normalized_name(descriptor.get("distribution")), version


def _validate_artifact_manifest(
    manifest_path: Path, artifact_root: Path | None
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, dict[str, Any]]], dict[str, Any]]:
    """Verify admitted identities and, when explicitly supplied, archive bytes."""
    manifest, raw = _load_canonical_object(manifest_path)
    identity = _identity_for_bytes(raw)
    _require_frozen_identity("artifact manifest", identity, FROZEN_ARTIFACT_MANIFEST_IDENTITY)
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise EvidenceError("artifact manifest schema is not supported")
    if manifest.get("resolution_status") != RESOLUTION_STATUS:
        raise EvidenceError("artifact manifest lost its hash-bound resolution status")
    if manifest.get("artifact_clean_install_status") != CLEAN_INSTALL_NO_GO:
        raise EvidenceError("artifact manifest clean-install disposition drifted")
    if manifest.get("artifact_byte_availability_status") != BYTE_AVAILABILITY_STATUS:
        raise EvidenceError("artifact manifest byte-availability disposition drifted")
    if manifest.get("semantic_surrogates") != []:
        raise EvidenceError("semantic surrogate artifacts are forbidden")
    if manifest.get("identity_policy") != "SHA-256 over the exact admitted archive bytes":
        raise EvidenceError("artifact identity policy is not the frozen exact-byte policy")
    if manifest.get("cid_policy") != "CIDv1 with raw multicodec and sha2-256 multihash":
        raise EvidenceError("artifact CID policy is not CIDv1(raw,sha2-256)")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 8:
        raise EvidenceError("artifact manifest must contain four wheel/sdist pairs")
    root = artifact_root.resolve() if artifact_root is not None else None
    if root is not None and not root.is_dir():
        raise EvidenceError(f"explicit artifact root is not a directory: {root}")

    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    filenames: set[str] = set()
    verified_files: list[dict[str, Any]] = []
    unavailable_files: list[dict[str, Any]] = []
    for raw_descriptor in artifacts:
        if not isinstance(raw_descriptor, dict):
            raise EvidenceError("artifact descriptors must be objects")
        descriptor = dict(raw_descriptor)
        key = _artifact_key(descriptor)
        if key[0] not in DIRECT_DISTRIBUTIONS:
            raise EvidenceError(f"unadmitted direct distribution: {key[0]}")
        kind = descriptor.get("kind")
        if kind not in {"wheel", "sdist"}:
            raise EvidenceError(f"invalid artifact kind for {key}: {kind!r}")
        filename = _safe_filename(descriptor.get("filename"))
        if filename in filenames:
            raise EvidenceError(f"duplicate artifact filename: {filename}")
        filenames.add(filename)
        if kind == "wheel" and not filename.endswith(".whl"):
            raise EvidenceError(f"wheel descriptor does not name a wheel: {filename}")
        if kind == "sdist" and not filename.endswith(SOURCE_ARCHIVE_SUFFIXES):
            raise EvidenceError(f"sdist descriptor has an unsupported suffix: {filename}")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
            raise EvidenceError(f"artifact SHA-256 is invalid: {filename}")
        expected_cid = _raw_cid_v1_from_sha256(digest)
        if descriptor.get("cid_v1_raw") != expected_cid:
            raise EvidenceError(f"artifact raw CID does not bind its SHA-256: {filename}")
        _verify_raw_cid_v1(expected_cid, digest)
        available = descriptor.get("bytes_available")
        verified = descriptor.get("bytes_verified")
        binding = descriptor.get("cid_binding_status")
        path = _safe_artifact_path(root, filename) if root is not None else None
        if available is True:
            size = descriptor.get("size")
            if verified is not True or binding != "bytes-verified":
                raise EvidenceError(
                    f"available artifact lacks byte-verified CID status: {filename}"
                )
            if not isinstance(size, int) or size <= 0:
                raise EvidenceError(f"available artifact size is invalid: {filename}")
            if path is not None:
                if not path.is_file():
                    raise EvidenceError(f"available admitted artifact is absent: {path}")
                actual_size = path.stat().st_size
                actual_sha256 = _sha256_path(path)
                if actual_size != size or actual_sha256 != digest:
                    raise EvidenceError(f"admitted artifact bytes do not match: {path}")
                actual_cid = _raw_cid_v1_from_sha256(actual_sha256)
                if actual_cid != expected_cid:
                    raise EvidenceError(f"admitted artifact raw CID does not match: {path}")
                verified_files.append(
                    {
                        "distribution": key[0],
                        "version": key[1],
                        "kind": kind,
                        "filename": filename,
                        "size": actual_size,
                        "sha256": actual_sha256,
                        "cid_v1_raw": actual_cid,
                    }
                )
        elif available is False:
            if verified is not False or binding != "identity-derived-bytes-unavailable":
                raise EvidenceError(
                    f"unavailable artifact has an overstated CID binding: {filename}"
                )
            if descriptor.get("size") is not None:
                raise EvidenceError(f"unavailable artifact size must be null: {filename}")
            if path is not None and path.exists():
                raise EvidenceError(
                    f"manifest marks artifact bytes unavailable but the named path exists: {path}"
                )
            unavailable_files.append(
                {
                    "distribution": key[0],
                    "version": key[1],
                    "kind": kind,
                    "filename": filename,
                    "sha256": digest,
                    "cid_v1_raw": expected_cid,
                    "cid_binding_status": binding,
                }
            )
        else:
            raise EvidenceError(f"artifact byte availability is not boolean: {filename}")
        by_kind = pairs.setdefault(key, {})
        if kind in by_kind:
            raise EvidenceError(f"duplicate {kind} descriptor for {key}")
        by_kind[kind] = descriptor

    if len(pairs) != 4 or {key[0] for key in pairs} != DIRECT_DISTRIBUTIONS:
        raise EvidenceError("artifact manifest does not cover the four direct distributions")
    if any(set(pair) != {"wheel", "sdist"} for pair in pairs.values()):
        raise EvidenceError("each direct distribution must have one wheel and one sdist")
    expected_unavailable = {
        "distribution": "ipfs-kit-py",
        "version": "0.3.0",
        "kind": "sdist",
        "filename": "ipfs_kit_py-0.3.0.tar.gz",
        "sha256": "8db7299f2cc144814d6b1b01a8476ba2daa67830856513f2863c6fac4af3ed15",
        "cid_v1_raw": "bafkreienw4uz6lgbisau22y3agueo25c3kthqmefmuj7fbr4n6wev47ncu",
        "cid_binding_status": "identity-derived-bytes-unavailable",
    }
    if unavailable_files != [expected_unavailable]:
        raise EvidenceError("the sole unavailable identity is not the admitted kit sdist")
    if sum(descriptor.get("bytes_verified") is True for descriptor in artifacts) != 7:
        raise EvidenceError("artifact manifest must declare exactly seven byte-verified archives")

    evidence = {
        "path": str(manifest_path.resolve()),
        **identity,
        "schema": ARTIFACT_SCHEMA,
        "artifact_root": str(root) if root is not None else None,
        "artifact_root_input": "explicit" if root is not None else "not-provided",
        "artifact_bytes_verification_status": (
            "bytes-verified"
            if root is not None
            else "artifact-bytes-not-verified-artifact-root-not-provided"
        ),
        "descriptor_count": len(artifacts),
        "byte_verified_count": len(verified_files),
        "manifest_declared_byte_verified_count": sum(
            descriptor.get("bytes_verified") is True for descriptor in artifacts
        ),
        "identity_derived_unavailable_count": len(unavailable_files),
        "verified_artifacts": verified_files,
        "unavailable_artifacts": unavailable_files,
        "semantic_surrogates_rejected": True,
        "frozen_pcce053_identity_binding": "passed",
    }
    return manifest, pairs, evidence


def _lock_bytes(receipt: Mapping[str, Any]) -> bytes:
    profile = receipt["profile"]
    lines = [
        "# Generated by scripts/proof_context/build_environment_manifest.py.",
        f"# Profile: {profile}; environment: {ENVIRONMENT_SLUG}.",
        f"# Resolution: {receipt['resolution_status']} using exact admitted direct-package wheels.",
        (
            "# Artifact clean-install: "
            f"{receipt['artifact_clean_install_status']}; selected sdist builds are not qualified."
        ),
        f"# Native build: {receipt['native_build_status']}.",
        "# Index: https://pypi.org/simple; no extra indexes.",
        "",
    ]
    packages = receipt.get("packages")
    if not isinstance(packages, list):
        raise EvidenceError("resolver receipt packages are not a list")
    for package in packages:
        hashes = package.get("lock_hashes")
        if not isinstance(hashes, list) or not hashes:
            raise EvidenceError(f"lock entry has no hashes: {package.get('name')!r}")
        extras = package.get("requested_extras")
        if not isinstance(extras, list) or not all(isinstance(item, str) for item in extras):
            raise EvidenceError(f"lock entry extras are invalid: {package.get('name')!r}")
        requirement_name = str(package.get("name", ""))
        if extras:
            requirement_name += f"[{','.join(extras)}]"
        lines.append(f"{requirement_name}=={package.get('version')} \\")
        for index, digest in enumerate(hashes):
            suffix = " \\" if index + 1 < len(hashes) else ""
            lines.append(f"    --hash=sha256:{digest}{suffix}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _validate_remote_archive_url(url: Any) -> None:
    if not isinstance(url, str):
        raise EvidenceError("resolver archive URL is not a string")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
        raise EvidenceError(f"resolver archive origin is not admitted: {url!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise EvidenceError(f"resolver archive URL has mutable or credential data: {url!r}")


def _validate_profile_inputs(
    profile: str,
    artifact_pairs: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if profile not in FROZEN_PROFILE_INPUT_IDENTITIES:
        raise EvidenceError(f"profile is not part of the frozen PCCE-053 matrix: {profile}")
    receipt_path = LOCK_ROOT / ENVIRONMENT_SLUG / f"{profile}.resolver.json"
    lock_path = LOCK_ROOT / ENVIRONMENT_SLUG / f"{profile}.txt"
    receipt, receipt_raw = _load_canonical_object(receipt_path)
    try:
        lock_raw = lock_path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read profile lock {lock_path}: {exc}") from exc
    receipt_identity = _identity_for_bytes(receipt_raw)
    lock_identity = _identity_for_bytes(lock_raw)
    frozen_inputs = FROZEN_PROFILE_INPUT_IDENTITIES[profile]
    _require_frozen_identity(f"{profile} lock", lock_identity, frozen_inputs["lock"])
    _require_frozen_identity(
        f"{profile} resolver receipt",
        receipt_identity,
        frozen_inputs["resolver_receipt"],
    )
    if receipt.get("schema") != RESOLVER_SCHEMA or receipt.get("profile") != profile:
        raise EvidenceError(f"invalid resolver receipt for profile {profile}")
    if receipt.get("environment") != EXPECTED_ENVIRONMENT:
        raise EvidenceError(f"resolver environment drifted for profile {profile}")
    if receipt.get("resolution_status") != RESOLUTION_STATUS:
        raise EvidenceError(f"resolution status drifted for profile {profile}")
    if receipt.get("artifact_clean_install_status") != CLEAN_INSTALL_NO_GO:
        raise EvidenceError(f"clean-install disposition drifted for profile {profile}")
    if receipt.get("native_build_status") not in {
        "not-required-by-profile",
        "no-go-native-sdist-only",
    }:
        raise EvidenceError(f"native-build disposition is invalid for profile {profile}")
    resolver = receipt.get("resolver")
    if not isinstance(resolver, dict):
        raise EvidenceError(f"resolver metadata is absent for profile {profile}")
    if (
        resolver.get("name") != "pip"
        or resolver.get("index") != PYPI_ORIGIN
        or resolver.get("extra_indexes") != []
        or resolver.get("resolution_only") is not True
    ):
        raise EvidenceError(f"resolver policy drifted for profile {profile}")
    report_sha256 = resolver.get("raw_report_sha256")
    if not isinstance(report_sha256, str) or HEX_SHA256.fullmatch(report_sha256) is None:
        raise EvidenceError(f"raw resolver report identity is invalid for profile {profile}")

    packages = receipt.get("packages")
    if not isinstance(packages, list) or not packages:
        raise EvidenceError(f"resolver package set is empty for profile {profile}")
    package_names: list[str] = []
    filenames_by_origin = {LOCAL_ORIGIN: [], PYPI_ORIGIN: []}
    direct_seen: set[tuple[str, str]] = set()
    selected_sources: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise EvidenceError(f"resolver package entry is invalid for profile {profile}")
        name = _normalized_name(package.get("name"))
        version = package.get("version")
        if not isinstance(version, str) or not version:
            raise EvidenceError(f"resolver package version is invalid: {name}")
        package_names.append(name)
        filename = _safe_filename(package.get("filename"))
        digest = package.get("sha256")
        resolution_digest = package.get("resolution_sha256")
        lock_hashes = package.get("lock_hashes")
        if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
            raise EvidenceError(f"resolver package SHA-256 is invalid: {name}")
        if resolution_digest != digest:
            raise EvidenceError(f"resolver and admitted identities differ for {name}")
        if (
            not isinstance(lock_hashes, list)
            or not lock_hashes
            or any(
                not isinstance(item, str) or HEX_SHA256.fullmatch(item) is None
                for item in lock_hashes
            )
            or lock_hashes != sorted(set(lock_hashes))
            or digest not in lock_hashes
        ):
            raise EvidenceError(f"resolver lock hashes are invalid for {name}")
        if package.get("artifact_availability") != "available":
            raise EvidenceError(f"resolver selected an unavailable artifact: {name}")
        if package.get("resolution_surrogate") is not None:
            raise EvidenceError(f"resolver selected a semantic surrogate: {name}")
        origin = package.get("origin")
        if origin not in {LOCAL_ORIGIN, PYPI_ORIGIN}:
            raise EvidenceError(f"resolver selected an undeclared origin for {name}")
        if origin == PYPI_ORIGIN:
            _validate_remote_archive_url(package.get("download_url"))
        filenames_by_origin[origin].append(filename)

        build_system = package.get("build_system")
        if not isinstance(build_system, dict):
            raise EvidenceError(f"resolver omitted build-system evidence for {name}")
        source_build_required = build_system.get("source_build_required")
        artifact_kind = build_system.get("artifact_kind")
        if source_build_required is True:
            if artifact_kind != "sdist" or not filename.endswith(SOURCE_ARCHIVE_SUFFIXES):
                raise EvidenceError(f"source-build evidence is inconsistent for {name}")
            if build_system.get("source_archive_sha256") != digest:
                raise EvidenceError(f"source-build archive identity drifted for {name}")
            backend = build_system.get("build_backend")
            requires = build_system.get("build_requires")
            if not isinstance(backend, str) or not backend:
                raise EvidenceError(f"source-build backend evidence is absent for {name}")
            if not isinstance(requires, list) or not all(
                isinstance(item, str) for item in requires
            ):
                raise EvidenceError(f"source-build requirements are invalid for {name}")
            selected_sources.append(
                {
                    "distribution": name,
                    "version": version,
                    "filename": filename,
                    "sha256": digest,
                    "cid_v1_raw": _raw_cid_v1_from_sha256(digest),
                    "build_backend": backend,
                    "build_requires": requires,
                    "native": name == "llama-cpp-python",
                }
            )
        elif source_build_required is False:
            if artifact_kind != "wheel" or not filename.endswith(".whl"):
                raise EvidenceError(f"wheel evidence is inconsistent for {name}")
        else:
            raise EvidenceError(f"source-build requirement is not boolean for {name}")

        key = (name, version)
        if key in artifact_pairs:
            pair = artifact_pairs[key]
            wheel = pair["wheel"]
            expected_hashes = sorted(item["sha256"] for item in pair.values())
            if (
                origin != LOCAL_ORIGIN
                or package.get("download_url") != f"artifact://{wheel['filename']}"
                or filename != wheel["filename"]
                or digest != wheel["sha256"]
                or lock_hashes != expected_hashes
                or source_build_required is not False
            ):
                raise EvidenceError(
                    f"direct package is not bound to its admitted wheel pair: {name}"
                )
            direct_seen.add(key)
        elif name in DIRECT_DISTRIBUTIONS:
            raise EvidenceError(f"direct package version is not admitted: {name}=={version}")
        elif origin != PYPI_ORIGIN:
            raise EvidenceError(
                f"transitive package did not come from the sole resolver index: {name}"
            )

    if package_names != sorted(package_names) or len(set(package_names)) != len(package_names):
        raise EvidenceError(f"resolver package order/set is not canonical for profile {profile}")
    if direct_seen != set(artifact_pairs):
        raise EvidenceError(f"profile {profile} does not contain all admitted direct packages")
    if not selected_sources:
        raise EvidenceError(
            f"profile {profile} no longer has the PCCE-053 source-build no-go; "
            "this harness revision cannot infer clean-install qualification"
        )

    risk_ledger = receipt.get("requirement_risk_ledger")
    if not isinstance(risk_ledger, dict):
        raise EvidenceError(f"requirement risk ledger is absent for profile {profile}")
    if risk_ledger.get("policy_status") != "passed":
        raise EvidenceError(f"requirement risk policy did not pass for profile {profile}")
    if (
        risk_ledger.get("selected_archive_origin_policy")
        != "exact local admitted archive or credential-free files.pythonhosted.org URL with SHA-256"
    ):
        raise EvidenceError(f"selected archive origin policy drifted for profile {profile}")
    if risk_ledger.get("selected_unsafe_requirements_by_class") != EMPTY_UNSAFE_REQUIREMENTS:
        raise EvidenceError(f"selected unsafe requirements are nonempty for profile {profile}")
    if risk_ledger.get("selected_unsafe_vcs_direct_editable_path_requirements") != []:
        raise EvidenceError(f"selected unsafe requirement aggregate is nonempty for {profile}")
    inactive_vcs = risk_ledger.get("inactive_mutable_vcs_core_metadata")
    if (
        not isinstance(inactive_vcs, list)
        or len(inactive_vcs) != 3
        or any(
            not isinstance(item, dict)
            or item.get("selection_status") != "inactive-unrequested-extra-target-absent"
            or item.get("required_extra_requested") is not False
            or item.get("target_distribution_selected") is not False
            for item in inactive_vcs
        )
    ):
        raise EvidenceError(f"inactive mutable-VCS metadata ledger drifted for {profile}")
    expected_origins = {
        origin: sorted(filenames) for origin, filenames in filenames_by_origin.items()
    }
    if risk_ledger.get("selected_archive_origins") != expected_origins:
        raise EvidenceError(f"selected archive origin ledger drifted for profile {profile}")

    if lock_raw != _lock_bytes(receipt):
        raise EvidenceError(f"profile lock differs from its resolver receipt: {lock_path}")
    return {
        "profile": profile,
        "receipt": receipt,
        "selected_source_archives": selected_sources,
        "lock": {
            "path": str(lock_path.relative_to(ACCELERATOR_ROOT)),
            **lock_identity,
            "frozen_pcce053_identity_binding": "passed",
        },
        "resolver_receipt": {
            "path": str(receipt_path.relative_to(ACCELERATOR_ROOT)),
            **receipt_identity,
            "frozen_pcce053_identity_binding": "passed",
        },
        "raw_pip_report": {
            "sha256": report_sha256,
            "retained": resolver.get("raw_report_retained"),
        },
        "distribution_count": len(packages),
        "requirement_risk_ledger": {
            "policy_status": "passed",
            "selected_unsafe_requirements_by_class": dict(EMPTY_UNSAFE_REQUIREMENTS),
            "selected_unsafe_vcs_direct_editable_path_requirements": [],
            "inactive_mutable_vcs_core_metadata_count": len(inactive_vcs),
            "selected_archive_origins": expected_origins,
        },
    }


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_source_path_isolation(
    *, cwd: Path, sys_path: Sequence[str], source_roots: Sequence[Path]
) -> None:
    """Reject a child-process trace that can import from a source checkout."""
    resolved_cwd = cwd.resolve()
    roots = tuple(root.resolve() for root in source_roots)
    candidates = [resolved_cwd if entry == "" else Path(entry).resolve() for entry in sys_path]
    for root in roots:
        if _path_is_within(resolved_cwd, root):
            raise EvidenceError(f"qualified subprocess cwd is inside source tree: {resolved_cwd}")
        for candidate in candidates:
            if _path_is_within(candidate, root):
                raise EvidenceError(
                    f"qualified subprocess sys.path reaches source tree: {candidate}"
                )


def _offline_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    for name in ("PYTHONHOME", "PYTHONPATH", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
        result.pop(name, None)
    result.update(
        {
            "LC_ALL": "C.UTF-8",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TZ": "UTC",
        }
    )
    return result


def _capture_source_path_trace() -> dict[str, Any]:
    probe = (
        "import json, os, sys; "
        "print(json.dumps({'cwd': os.getcwd(), 'isolated': sys.flags.isolated, "
        "'sys_path': sys.path, 'pythonhome_present': 'PYTHONHOME' in os.environ, "
        "'pythonpath_present': 'PYTHONPATH' in os.environ}, sort_keys=True))"
    )
    environment = _offline_environment()
    with tempfile.TemporaryDirectory(prefix="pcce054-isolation-") as temporary:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            cwd=temporary,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        raise EvidenceError("isolated source-path probe failed: " + completed.stderr.strip())
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EvidenceError("isolated source-path probe returned malformed JSON") from exc
    if (
        not isinstance(observed, dict)
        or observed.get("isolated") != 1
        or observed.get("pythonhome_present") is not False
        or observed.get("pythonpath_present") is not False
        or not isinstance(observed.get("sys_path"), list)
    ):
        raise EvidenceError("isolated source-path probe did not enforce the subprocess contract")
    _assert_source_path_isolation(
        cwd=Path(str(observed["cwd"])),
        sys_path=[str(item) for item in observed["sys_path"]],
        source_roots=[ACCELERATOR_ROOT],
    )
    transcript = {
        "status": "passed",
        "command": [sys.executable, "-I", "-c", "<source-path-probe>"],
        "cwd_policy": "disposable-directory-outside-source-tree",
        "environment_removed": [
            "PIP_EXTRA_INDEX_URL",
            "PIP_INDEX_URL",
            "PYTHONHOME",
            "PYTHONPATH",
        ],
        "environment_forced": {
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "isolated_flag": observed["isolated"],
        "observed_sys_path": observed["sys_path"],
        "source_roots_excluded": [str(ACCELERATOR_ROOT)],
    }
    return {
        **transcript,
        "transcript_identity": _identity_for_bytes(_canonical_json_bytes(transcript)),
    }


def _observed_environment() -> dict[str, str]:
    implementation = platform.python_implementation()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    machine = platform.machine().lower()
    aliases = {"arm64": "aarch64", "amd64": "x86_64"}
    return {
        "implementation": implementation,
        "python": version,
        "operating_system": platform.system(),
        "architecture": aliases.get(machine, machine),
    }


def _profile_no_go_record(
    inputs: Mapping[str, Any],
    *,
    artifact_manifest_identity: Mapping[str, Any],
    observed_environment: Mapping[str, str],
    isolation_identity: Mapping[str, str],
) -> dict[str, Any]:
    selected = inputs["selected_source_archives"]
    reasons = []
    artifact_bytes_verified = (
        artifact_manifest_identity["artifact_bytes_verification_status"] == "bytes-verified"
    )
    if not artifact_bytes_verified:
        reasons.append(
            {
                "code": "artifact-bytes-not-verified-artifact-root-not-provided",
                "detail": (
                    "No artifact root was supplied. Manifest identities were checked, but the "
                    "admitted archive bytes were not opened or described as verified by this run."
                ),
            }
        )
    reasons.append(
        {
            "code": "selected-source-archives-require-unqualified-builds",
            "detail": (
                "The frozen resolver selected source distributions. PCCE-054 may consume "
                "frozen artifacts but may not build replacement wheels or repair PCCE-053 locks."
            ),
            "archives": [item["filename"] for item in selected],
        }
    )
    if observed_environment != EXPECTED_ENVIRONMENT:
        reasons.append(
            {
                "code": "unsupported-runner-environment",
                "detail": "The runner does not match CPython 3.12 on Linux aarch64.",
            }
        )
    transcript = {
        "profile": inputs["profile"],
        "qualification_status": "no-go",
        "artifact_manifest": {
            "sha256": artifact_manifest_identity["sha256"],
            "cid_v1_raw": artifact_manifest_identity["cid_v1_raw"],
            "artifact_bytes_verification_status": artifact_manifest_identity[
                "artifact_bytes_verification_status"
            ],
            "byte_verified_count": artifact_manifest_identity["byte_verified_count"],
        },
        "lock": inputs["lock"],
        "resolver_receipt": inputs["resolver_receipt"],
        "raw_pip_report": inputs["raw_pip_report"],
        "requirement_risk_ledger": inputs["requirement_risk_ledger"],
        "selected_source_archives": selected,
        "no_go_reason_codes": [reason["code"] for reason in reasons],
        "observed_environment": dict(observed_environment),
        "source_path_trace_identity": dict(isolation_identity),
        "pip_policy": [
            "--no-index",
            "--only-binary=:all:",
            "--require-hashes",
        ],
    }
    return {
        "profile": inputs["profile"],
        "qualification_status": "no-go",
        "resolution_status": inputs["receipt"]["resolution_status"],
        "artifact_clean_install_status": inputs["receipt"]["artifact_clean_install_status"],
        "native_build_status": inputs["receipt"]["native_build_status"],
        "distribution_count": inputs["distribution_count"],
        "frozen_pcce053_inputs": {
            "artifact_manifest": transcript["artifact_manifest"],
            "lock": inputs["lock"],
            "resolver_receipt": inputs["resolver_receipt"],
        },
        "requirement_risk_ledger": inputs["requirement_risk_ledger"],
        "selected_source_archives": selected,
        "no_go_reasons": reasons,
        "stages": {
            "frozen_artifact_sha256_and_raw_cid": (
                "passed"
                if artifact_bytes_verified
                else "manifest-identities-only-bytes-not-verified"
            ),
            "semantic_surrogate_rejection": "passed",
            "frozen_pcce053_identity_binding": "passed",
            "hash_bound_lock_validation": "passed",
            "source_path_isolation_probe": "passed",
            "wheel_only_offline_install": "not-run-preflight-no-go",
            "installed_distribution_hashes": "not-produced",
            "installed_resource_byte_validation": "not-run-preflight-no-go",
            "installed_imports": "not-run-preflight-no-go",
            "cli_smoke": "not-run-preflight-no-go",
        },
        "transcript": transcript,
        "transcript_identity": _identity_for_bytes(_canonical_json_bytes(transcript)),
    }


def build_report(
    *, artifacts_path: Path, artifact_root: Path | None, profiles: Sequence[str]
) -> dict[str, Any]:
    _, artifact_pairs, artifact_evidence = _validate_artifact_manifest(
        artifacts_path.resolve(), artifact_root.resolve() if artifact_root is not None else None
    )
    isolation = _capture_source_path_trace()
    observed = _observed_environment()
    profile_records = [
        _profile_no_go_record(
            _validate_profile_inputs(profile, artifact_pairs),
            artifact_manifest_identity=artifact_evidence,
            observed_environment=observed,
            isolation_identity=isolation["transcript_identity"],
        )
        for profile in profiles
    ]
    return {
        "schema": REPORT_SCHEMA,
        "task_id": "PCCE-054",
        "environment_id": ENVIRONMENT_SLUG,
        "qualification_status": "no-go",
        "artifact_clean_install_status": CLEAN_INSTALL_NO_GO,
        "resolution_status": RESOLUTION_STATUS,
        "requested_profiles": list(profiles),
        "qualified_install_profile_count": 0,
        "no_go_profile_count": len(profile_records),
        "expected_environment": dict(EXPECTED_ENVIRONMENT),
        "observed_environment": observed,
        "artifact_manifest": artifact_evidence,
        "source_path_isolation": isolation,
        "install_policy": {
            "artifact_root_input": artifact_evidence["artifact_root_input"],
            "network": "disabled",
            "index": "none",
            "require_hashes": True,
            "wheel_only": True,
            "source_builds": "forbidden",
            "semantic_surrogates": "forbidden",
            "editable_or_source_siblings": "forbidden",
        },
        "profiles": profile_records,
        "execution_disposition": {
            "artifact_and_lock_preflight": (
                "completed"
                if artifact_evidence["artifact_bytes_verification_status"] == "bytes-verified"
                else "manifest-and-locks-only-artifact-bytes-not-verified"
            ),
            "fresh_environment_created": False,
            "pip_invoked": False,
            "imports_invoked": False,
            "cli_invoked": False,
            "container_build_invoked": False,
            "workflow_result": "not-observed-by-local-harness",
            "container_result": "explicit-no-go-not-built",
            "reason": (
                "Every frozen profile selects at least one source archive; the conflict "
                "policy forbids building or substituting wheels during validation."
            ),
        },
        "exit_policy": {
            "evidence_only": 0,
            "require_qualified_no_go": EXIT_REQUIRE_QUALIFIED_NO_GO,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="canonical PCCE-053 artifact_hashes.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "explicit directory containing the exact admitted artifact bytes; omission is "
            "recorded as an evidence-only no-go and never triggers path discovery"
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILE_ORDER,
        dest="profiles",
        help="evaluate one profile; repeat as needed (default: all five)",
    )
    parser.add_argument("--output", type=Path, help="also write the canonical JSON report")
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help=f"exit {EXIT_REQUIRE_QUALIFIED_NO_GO} when any requested profile is no-go",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profiles = tuple(args.profiles or PROFILE_ORDER)
    if len(set(profiles)) != len(profiles):
        print("clean-install evidence error: duplicate --profile", file=sys.stderr)
        return EXIT_EVIDENCE_ERROR
    profiles = tuple(profile for profile in PROFILE_ORDER if profile in profiles)
    try:
        report = build_report(
            artifacts_path=args.artifacts,
            artifact_root=args.artifact_root,
            profiles=profiles,
        )
        encoded = _canonical_json_bytes(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(encoded)
        sys.stdout.buffer.write(encoded)
    except (EvidenceError, OSError) as exc:
        print(f"clean-install evidence error: {exc}", file=sys.stderr)
        return EXIT_EVIDENCE_ERROR
    if args.require_qualified and report["qualification_status"] != "qualified":
        return EXIT_REQUIRE_QUALIFIED_NO_GO
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
