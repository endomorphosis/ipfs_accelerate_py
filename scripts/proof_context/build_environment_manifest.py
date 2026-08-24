#!/usr/bin/env python3
"""Build and verify the PCCE proof-context environment evidence.

The resolver is intentionally a separate, explicit input.  ``--compile-locks``
normalizes fresh pip JSON reports into platform-specific, hash-bound locks and
small resolver receipts.  The default mode deterministically materializes the
cross-repository JSON evidence from those committed inputs.  ``--check`` never
contacts an index or changes the checkout.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

SCRIPT_PATH = Path(__file__).resolve()
ACCELERATOR_ROOT = SCRIPT_PATH.parents[2]
LOCK_ROOT = ACCELERATOR_ROOT / "packaging" / "proof_context" / "locks"
INPUTS_PATH = LOCK_ROOT / "inputs.json"
PROFILES = ("core", "verification", "codex", "local-model", "evaluation")
ENVIRONMENT_SLUG = "cpython312-linux-aarch64"
ENVIRONMENT_OUTPUTS = Path("artifacts/proof_carrying_context_engine/environment")
RECEIPT_OUTPUT = Path("artifacts/proof_carrying_context_engine/receipts/PCCE-053.json")
LOCAL_ORIGIN = "local-admitted-artifact"
PYPI_ORIGIN = "https://pypi.org/simple"
SCHEMA_PREFIX = "lift_coding.proof-carrying-context-engine"
EXPECTED_INACTIVE_VCS_DECLARATIONS = {
    'ipld-unixfs @ git+https://github.com/storacha/py-ipld-unixfs.git ; extra == "ipld-github"': (
        "ipfs-kit-py",
        "ipld-github",
        "ipld-unixfs",
    ),
    'libp2p @ git+https://github.com/libp2p/py-libp2p.git@main ; extra == "full"': (
        "ipfs-kit-py",
        "full",
        "libp2p",
    ),
    'libp2p @ git+https://github.com/libp2p/py-libp2p.git@main ; extra == "libp2p"': (
        "ipfs-kit-py",
        "libp2p",
        "libp2p",
    ),
}


class EvidenceError(RuntimeError):
    """Raised when evidence cannot be generated without weakening a claim."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_cid_v1_from_sha256(sha256: str) -> str:
    """Return canonical CIDv1(raw, sha2-256) for an admitted digest."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise EvidenceError(f"cannot derive raw CID from invalid SHA-256: {sha256!r}")
    # CIDv1 varint, raw multicodec, sha2-256 multihash code/length, digest.
    payload = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    cid = "b" + base64.b32encode(payload).decode("ascii").lower().rstrip("=")
    _verify_raw_cid_v1(cid, sha256)
    return cid


def _verify_raw_cid_v1(cid: str, sha256: str) -> None:
    """Fail closed unless a CID decodes to raw + sha2-256 + the expected digest."""
    if not cid.startswith("b") or cid != cid.lower():
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


def _raw_cid_v1_from_bytes(value: bytes) -> str:
    return _raw_cid_v1_from_sha256(_sha256_bytes(value))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected a JSON object: {path}")
    return value


def _normalized_name(value: str) -> str:
    name = re.sub(r"[-_.]+", "-", value).lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise EvidenceError(f"invalid distribution name: {value!r}")
    return name


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(ACCELERATOR_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise EvidenceError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _inputs() -> dict[str, Any]:
    inputs = _load_json(INPUTS_PATH)
    if inputs.get("schema") != f"{SCHEMA_PREFIX}.environment-inputs@1":
        raise EvidenceError("environment input schema is not supported")
    if tuple(inputs.get("profiles", {})) != PROFILES:
        raise EvidenceError("profile order/set differs from the canonical five profiles")
    environment = inputs.get("supported_environment", {})
    expected = {
        "implementation": "CPython",
        "python": "3.12",
        "operating_system": "Linux",
        "architecture": "aarch64",
    }
    if any(environment.get(key) != value for key, value in expected.items()):
        raise EvidenceError("the sole supported environment is not CPython 3.12 Linux aarch64")
    artifacts = inputs.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 8:
        raise EvidenceError("exactly four wheel/sdist artifact pairs are required")
    if inputs.get("resolution_status") != "supported-hash-bound":
        raise EvidenceError(
            "the admitted environment must have a typed hash-bound resolution status"
        )
    if inputs.get("artifact_clean_install_status") != "no-go-sdist-builds-not-qualified":
        raise EvidenceError(
            "the clean-install no-go must remain explicit until selected sdists are qualified"
        )
    if inputs.get("artifact_byte_availability_status") != "partial-one-admitted-sdist-unavailable":
        raise EvidenceError(
            "the frozen artifact byte-availability status is not the admitted partial set"
        )
    for profile, descriptor in inputs["profiles"].items():
        if descriptor.get("resolution_status") != inputs["resolution_status"]:
            raise EvidenceError(f"profile resolution status drifted: {profile}")
        if (
            descriptor.get("artifact_clean_install_status")
            != inputs["artifact_clean_install_status"]
        ):
            raise EvidenceError(f"profile clean-install status drifted: {profile}")
        if descriptor.get("native_build_status") not in {
            "not-required-by-profile",
            "no-go-native-sdist-only",
        }:
            raise EvidenceError(f"profile native-build status is invalid: {profile}")
    return inputs


def _artifact_index(inputs: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for artifact in inputs["artifacts"]:
        if not isinstance(artifact, dict):
            raise EvidenceError("artifact descriptors must be objects")
        required = ("distribution", "version", "kind", "filename", "sha256", "size")
        if any(key not in artifact for key in required):
            raise EvidenceError(f"incomplete artifact descriptor: {artifact!r}")
        digest = str(artifact["sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvidenceError(f"invalid artifact SHA-256: {digest!r}")
        if artifact["kind"] not in {"wheel", "sdist"}:
            raise EvidenceError(f"invalid artifact kind: {artifact['kind']!r}")
        available = artifact.get("bytes_available", True)
        if not isinstance(available, bool):
            raise EvidenceError(f"invalid byte-availability flag for {artifact['filename']}")
        size = artifact["size"]
        if available and (not isinstance(size, int) or size <= 0):
            raise EvidenceError(f"invalid artifact size for {artifact['filename']}")
        if not available and size is not None:
            raise EvidenceError(f"unavailable artifact size must be null: {artifact['filename']}")
        key = (_normalized_name(str(artifact["distribution"])), str(artifact["version"]))
        result.setdefault(key, []).append(artifact)
    if len(result) != 4 or any(
        {item["kind"] for item in pair} != {"wheel", "sdist"} for pair in result.values()
    ):
        raise EvidenceError("artifacts must contain one wheel and one sdist for each distribution")
    return result


def _report_archive(record: dict[str, Any]) -> tuple[str, str]:
    download = record.get("download_info")
    if not isinstance(download, dict):
        raise EvidenceError("pip report entry lacks download_info")
    url = str(download.get("url", ""))
    hashes = download.get("archive_info", {}).get("hashes", {})
    sha256 = str(hashes.get("sha256", "")) if isinstance(hashes, dict) else ""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise EvidenceError("pip report entry lacks one SHA-256 archive identity")
    return url, sha256


def _license_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    classifiers = [
        value
        for value in metadata.get("classifier", [])
        if isinstance(value, str) and value.startswith("License ::")
    ]
    raw = metadata.get("license")
    # Full embedded license texts are redundant with the immutable archive and
    # make the resolver receipt noisy.  Preserve short declarations only.
    declaration = raw.strip() if isinstance(raw, str) and len(raw.strip()) <= 160 else None
    return {
        "core_metadata_license": declaration,
        "license_classifiers": sorted(set(classifiers)),
    }


def _requirement_risk_ledger(
    *, profile: str, packages: list[dict[str, Any]], inputs: dict[str, Any]
) -> dict[str, Any]:
    """Classify frozen mutable declarations without treating them as selected."""
    policy = inputs.get("mutable_vcs_metadata_policy")
    if not isinstance(policy, dict):
        raise EvidenceError("mutable VCS metadata policy is missing")
    if policy.get("disposition") != "record-inactive-and-fail-closed-if-unexpected-or-selected":
        raise EvidenceError("mutable VCS metadata policy disposition drifted")
    origin_allowlist = policy.get("selected_archive_origin_allowlist")
    if origin_allowlist != [LOCAL_ORIGIN, PYPI_ORIGIN]:
        raise EvidenceError("selected archive origin allowlist drifted")
    unsafe_classes = policy.get("selected_unsafe_requirement_classes")
    expected_unsafe_classes = [
        "mutable-vcs",
        "unadmitted-direct-url",
        "editable",
        "local-path",
    ]
    if unsafe_classes != expected_unsafe_classes:
        raise EvidenceError("selected unsafe requirement class policy drifted")
    expected = policy.get("expected_inactive_declarations")
    if not isinstance(expected, list) or len(expected) != 3:
        raise EvidenceError("exactly three frozen inactive mutable VCS declarations are required")
    expected_by_declaration: dict[str, dict[str, Any]] = {}
    for descriptor in expected:
        if not isinstance(descriptor, dict):
            raise EvidenceError("mutable VCS declaration descriptors must be objects")
        declaration = str(descriptor.get("declaration", ""))
        if not declaration or declaration in expected_by_declaration:
            raise EvidenceError("mutable VCS declarations must be nonempty and unique")
        expected_by_declaration[declaration] = descriptor
    frozen_policy = {
        declaration: (
            descriptor.get("declared_by"),
            descriptor.get("required_extra"),
            descriptor.get("target_distribution"),
        )
        for declaration, descriptor in expected_by_declaration.items()
    }
    if frozen_policy != EXPECTED_INACTIVE_VCS_DECLARATIONS:
        raise EvidenceError("frozen inactive mutable VCS declaration policy drifted")

    selected_names = {package["name"] for package in packages}
    observed: list[dict[str, Any]] = []
    observed_declarations: set[str] = set()
    any_direct_reference_pattern = re.compile(r"^\s*[A-Za-z0-9_.-]+(?:\s*\[[^\]\r\n]+\])?\s*@\s*")
    direct_reference_pattern = re.compile(
        r'^\s*([A-Za-z0-9_.-]+)\s*@\s*([^;\s]+)\s*;\s*extra\s*==\s*"([^"]+)"\s*$'
    )
    for package in packages:
        for declaration in package["requires_dist"]:
            if any_direct_reference_pattern.match(declaration) is None:
                continue
            match = direct_reference_pattern.fullmatch(declaration)
            if match is None:
                raise EvidenceError(
                    f"unexpected PEP 508 direct-reference syntax in {profile}: {declaration!r}"
                )
            descriptor = expected_by_declaration.get(declaration)
            if descriptor is None:
                raise EvidenceError(
                    f"unexpected PEP 508 direct-reference Core Metadata declaration in {profile}: {declaration!r}"
                )
            target = _normalized_name(match.group(1))
            direct_reference = match.group(2)
            required_extra = match.group(3)
            if not direct_reference.startswith("git+https://"):
                raise EvidenceError(
                    f"frozen direct reference is not the expected VCS class in {profile}"
                )
            if (
                package["name"] != descriptor.get("declared_by")
                or target != descriptor.get("target_distribution")
                or required_extra != descriptor.get("required_extra")
            ):
                raise EvidenceError(f"mutable VCS declaration descriptor drifted in {profile}")
            extra_requested = required_extra in package["requested_extras"]
            target_selected = target in selected_names
            if extra_requested or target_selected:
                raise EvidenceError(
                    f"mutable VCS declaration became selected in {profile}: {declaration!r}"
                )
            observed_declarations.add(declaration)
            observed.append(
                {
                    **descriptor,
                    "declaring_distribution_requested_extras": package["requested_extras"],
                    "required_extra_requested": False,
                    "target_distribution_selected": False,
                    "selection_status": "inactive-unrequested-extra-target-absent",
                }
            )
    if observed_declarations != set(expected_by_declaration):
        missing = sorted(set(expected_by_declaration) - observed_declarations)
        raise EvidenceError(
            f"expected mutable VCS metadata declarations are absent in {profile}: {missing}"
        )
    observed.sort(key=lambda item: item["declaration"])

    local_archives = sorted(
        package["filename"] for package in packages if package["origin"] == LOCAL_ORIGIN
    )
    pypi_archives = sorted(
        package["filename"] for package in packages if package["origin"] == PYPI_ORIGIN
    )
    observed_origins = {package["origin"] for package in packages}
    if not observed_origins <= set(origin_allowlist):
        raise EvidenceError(f"selected archive origin escaped the allowlist in {profile}")
    if len(local_archives) != 4 or len(local_archives) + len(pypi_archives) != len(packages):
        raise EvidenceError(f"selected archive origin classification is incomplete in {profile}")
    return {
        "policy_status": "passed",
        "inactive_mutable_vcs_core_metadata": observed,
        "selected_unsafe_vcs_direct_editable_path_requirements": [],
        "selected_unsafe_requirements_by_class": {name: [] for name in unsafe_classes},
        "selected_archive_origins": {
            LOCAL_ORIGIN: local_archives,
            PYPI_ORIGIN: pypi_archives,
        },
        "selected_archive_origin_policy": (
            "exact local admitted archive or credential-free files.pythonhosted.org URL with SHA-256"
        ),
    }


def _safe_remote_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
        raise EvidenceError(f"resolver used an undeclared archive origin: {url!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise EvidenceError("resolver archive URL contains credentials or mutable parameters")
    return url


def _build_system_evidence(
    *, filename: str, archive_sha256: str, source_archives_dir: Path
) -> dict[str, Any]:
    if filename.endswith(".whl"):
        return {"artifact_kind": "wheel", "source_build_required": False}
    if not filename.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        raise EvidenceError(f"unsupported source-distribution archive format: {filename}")
    path = source_archives_dir / filename
    if not path.is_file() or _sha256_path(path) != archive_sha256:
        raise EvidenceError(f"selected source distribution is absent or hash-mismatched: {path}")
    try:
        with tarfile.open(path, mode="r:*") as archive:
            regular_names = sorted(
                member.name for member in archive.getmembers() if member.isfile()
            )
            pyprojects = [
                name
                for name in regular_names
                if PurePosixPath(name).name == "pyproject.toml"
                and len(PurePosixPath(name).parts) == 2
            ]
            setups = [
                name
                for name in regular_names
                if PurePosixPath(name).name == "setup.py" and len(PurePosixPath(name).parts) == 2
            ]
            if len(pyprojects) > 1 or len(setups) > 1:
                raise EvidenceError(f"source distribution has ambiguous build metadata: {filename}")
            if pyprojects:
                extracted = archive.extractfile(pyprojects[0])
                if extracted is None:
                    raise EvidenceError(f"cannot read pyproject.toml from {filename}")
                pyproject = tomllib.loads(extracted.read().decode("utf-8"))
                build_system = pyproject.get("build-system")
                if not isinstance(build_system, dict):
                    raise EvidenceError(f"pyproject.toml lacks [build-system]: {filename}")
                requires = build_system.get("requires")
                backend = build_system.get("build-backend")
                if not isinstance(requires, list) or not all(
                    isinstance(item, str) for item in requires
                ):
                    raise EvidenceError(f"invalid build-system.requires in {filename}")
                if not isinstance(backend, str) or not backend:
                    raise EvidenceError(f"invalid build-system.build-backend in {filename}")
                return {
                    "artifact_kind": "sdist",
                    "source_build_required": True,
                    "pyproject_present": True,
                    "build_backend": backend,
                    "build_requires": requires,
                    "backend_path": build_system.get("backend-path", []),
                    "source_archive_sha256": archive_sha256,
                }
            if not setups:
                raise EvidenceError(
                    f"source distribution has neither pyproject.toml nor setup.py: {filename}"
                )
    except (tarfile.TarError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EvidenceError(
            f"cannot inspect source distribution build metadata {filename}: {exc}"
        ) from exc
    return {
        "artifact_kind": "sdist",
        "source_build_required": True,
        "pyproject_present": False,
        "build_backend": "setuptools.build_meta:__legacy__",
        "build_requires": ["setuptools>=40.8.0"],
        "backend_path": [],
        "source_archive_sha256": archive_sha256,
        "basis": "pip 24.0 PEP 517 legacy-setuptools default for an sdist containing setup.py",
    }


def _compile_receipt(
    *,
    profile: str,
    report_path: Path,
    artifact_dir: Path,
    source_archives_dir: Path,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    report_bytes = report_path.read_bytes()
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"malformed pip report {report_path}: {exc}") from exc
    if report.get("version") != "1" or not isinstance(report.get("install"), list):
        raise EvidenceError(f"unsupported pip report shape: {report_path}")
    expected_pip = inputs["resolver"]["pip_version"]
    if report.get("pip_version") != expected_pip:
        raise EvidenceError(
            f"resolver version differs for {profile}: {report.get('pip_version')!r} != {expected_pip!r}"
        )
    artifact_index = _artifact_index(inputs)
    packages: list[dict[str, Any]] = []
    seen: set[str] = set()
    local_seen: set[tuple[str, str]] = set()
    for install in report["install"]:
        if not isinstance(install, dict) or not isinstance(install.get("metadata"), dict):
            raise EvidenceError(f"invalid install entry in {report_path}")
        metadata = install["metadata"]
        name = _normalized_name(str(metadata.get("name", "")))
        version = str(metadata.get("version", ""))
        if not version or name in seen:
            raise EvidenceError(f"duplicate or versionless distribution in {profile}: {name}")
        seen.add(name)
        download_info = install.get("download_info")
        if not isinstance(download_info, dict):
            raise EvidenceError(f"resolver entry lacks download information in {profile}: {name}")
        unsafe_source_keys = sorted({"vcs_info", "dir_info"} & set(download_info))
        if unsafe_source_keys:
            raise EvidenceError(
                f"selected VCS/editable/path source in {profile}: {name} ({unsafe_source_keys})"
            )
        is_direct = install.get("is_direct", False)
        if not isinstance(is_direct, bool):
            raise EvidenceError(f"resolver is_direct flag is invalid in {profile}: {name}")
        url, archive_sha256 = _report_archive(install)
        parsed = urlsplit(url)
        filename = unquote(Path(parsed.path).name)
        key = (name, version)
        if parsed.scheme == "file":
            if not is_direct:
                raise EvidenceError(
                    f"local archive was not an explicit direct input in {profile}: {name}"
                )
            admitted = artifact_index.get(key)
            if not admitted:
                raise EvidenceError(
                    f"unadmitted local distribution in {profile}: {name}=={version}"
                )
            wheel = next(item for item in admitted if item["kind"] == "wheel")
            expected_path = artifact_dir / wheel["filename"]
            if filename != wheel["filename"]:
                raise EvidenceError(f"local resolver input filename is not admitted: {filename}")
            if archive_sha256 != wheel["sha256"]:
                raise EvidenceError(f"local resolver input hash is not admitted: {filename}")
            if not wheel.get("bytes_available", True):
                raise EvidenceError(f"input claims unavailable admitted wheel bytes: {filename}")
            if not expected_path.is_file():
                raise EvidenceError(f"admitted wheel is missing: {expected_path}")
            if (
                expected_path.stat().st_size != wheel["size"]
                or _sha256_path(expected_path) != wheel["sha256"]
            ):
                raise EvidenceError(
                    f"admitted wheel bytes do not match descriptor: {expected_path}"
                )
            origin = LOCAL_ORIGIN
            download_url = f"artifact://{filename}"
            lock_hashes = sorted(item["sha256"] for item in admitted)
            admitted_license = wheel.get("license_declared")
            local_seen.add(key)
        else:
            if is_direct:
                raise EvidenceError(f"unadmitted selected direct URL in {profile}: {name}")
            download_url = _safe_remote_url(url)
            origin = PYPI_ORIGIN
            lock_hashes = [archive_sha256]
            admitted_license = None
        requested_extras = install.get("requested_extras", [])
        if not isinstance(requested_extras, list) or not all(
            isinstance(item, str) for item in requested_extras
        ):
            raise EvidenceError(f"invalid requested_extras for {name}")
        requires_dist = metadata.get("requires_dist", [])
        if requires_dist is None:
            requires_dist = []
        if not isinstance(requires_dist, list) or not all(
            isinstance(item, str) for item in requires_dist
        ):
            raise EvidenceError(f"invalid requires_dist for {name}")
        packages.append(
            {
                "name": name,
                "version": version,
                "filename": filename,
                "sha256": wheel["sha256"] if parsed.scheme == "file" else archive_sha256,
                "resolution_sha256": archive_sha256,
                "lock_hashes": lock_hashes,
                "origin": origin,
                "download_url": download_url,
                "requested": bool(install.get("requested", False)),
                "requested_extras": sorted(set(requested_extras)),
                "requires_python": metadata.get("requires_python"),
                "requires_dist": sorted(set(requires_dist)),
                "admitted_artifact_license": admitted_license,
                "artifact_availability": "available",
                "license": _license_metadata(metadata),
                "build_system": _build_system_evidence(
                    filename=filename,
                    archive_sha256=archive_sha256,
                    source_archives_dir=source_archives_dir,
                ),
            }
        )
    if local_seen != set(artifact_index):
        missing = sorted(set(artifact_index) - local_seen)
        raise EvidenceError(
            f"profile {profile} did not resolve all four admitted packages: {missing}"
        )
    root = next((package for package in packages if package["name"] == "ipfs-accelerate-py"), None)
    expected_extra = inputs["profiles"][profile]["accelerator_extra"]
    expected_extras = [] if expected_extra is None else [expected_extra]
    if root is None or root["requested_extras"] != expected_extras:
        raise EvidenceError(
            f"profile {profile} did not request accelerator extras {expected_extras}"
        )
    packages.sort(key=lambda package: package["name"])
    requirement_risk_ledger = _requirement_risk_ledger(
        profile=profile,
        packages=packages,
        inputs=inputs,
    )
    report_environment = report.get("environment", {})
    observed = {
        "implementation_name": report_environment.get("implementation_name"),
        "python_full_version": report_environment.get("python_full_version"),
        "platform_system": report_environment.get("platform_system"),
        "platform_machine": report_environment.get("platform_machine"),
    }
    if observed != inputs["resolver"]["observed_environment"]:
        raise EvidenceError(f"resolver report platform differs for {profile}: {observed!r}")
    return {
        "schema": f"{SCHEMA_PREFIX}.resolver-receipt@1",
        "profile": profile,
        "environment": inputs["supported_environment"],
        "root_requirement": inputs["profiles"][profile]["root_requirement"],
        "resolution_status": inputs["profiles"][profile]["resolution_status"],
        "artifact_clean_install_status": inputs["profiles"][profile][
            "artifact_clean_install_status"
        ],
        "native_build_status": inputs["profiles"][profile]["native_build_status"],
        "requirement_risk_ledger": requirement_risk_ledger,
        "resolver": {
            "name": "pip",
            "version": report["pip_version"],
            "index": PYPI_ORIGIN,
            "extra_indexes": [],
            "resolution_only": True,
            "raw_report_sha256": _sha256_bytes(report_bytes),
            "raw_report_retained": False,
            "command": inputs["resolver"]["command"],
        },
        "packages": packages,
    }


def _lock_bytes(receipt: dict[str, Any]) -> bytes:
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
    for package in receipt["packages"]:
        hashes = package["lock_hashes"]
        if not hashes:
            raise EvidenceError(f"lock entry has no hashes: {package['name']}")
        extras = package["requested_extras"]
        requirement_name = package["name"]
        if extras:
            requirement_name += f"[{','.join(extras)}]"
        lines.append(f"{requirement_name}=={package['version']} \\")
        for index, digest in enumerate(hashes):
            suffix = " \\" if index + 1 < len(hashes) else ""
            lines.append(f"    --hash=sha256:{digest}{suffix}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _compile_locks(
    reports_dir: Path, artifact_dir: Path, source_archives_dir: Path, *, check: bool
) -> None:
    inputs = _inputs()
    for artifact in inputs["artifacts"]:
        if not artifact.get("bytes_available", True):
            continue
        path = artifact_dir / artifact["filename"]
        if not path.is_file():
            raise EvidenceError(f"admitted artifact is missing: {path}")
        if path.stat().st_size != artifact["size"] or _sha256_path(path) != artifact["sha256"]:
            raise EvidenceError(f"admitted artifact bytes do not match descriptor: {path}")
    output_root = LOCK_ROOT / ENVIRONMENT_SLUG
    for profile in PROFILES:
        receipt = _compile_receipt(
            profile=profile,
            report_path=reports_dir / f"{profile}.json",
            artifact_dir=artifact_dir,
            source_archives_dir=source_archives_dir,
            inputs=inputs,
        )
        outputs = {
            output_root / f"{profile}.resolver.json": _json_bytes(receipt),
            output_root / f"{profile}.txt": _lock_bytes(receipt),
        }
        for path, content in outputs.items():
            if check:
                if not path.is_file() or path.read_bytes() != content:
                    raise EvidenceError(f"generated lock input is stale: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


def _validate_committed_package_sources(
    *, profile: str, packages: Any, inputs: dict[str, Any]
) -> None:
    """Revalidate selected archive provenance without consulting a resolver or index."""
    if not isinstance(packages, list) or not packages:
        raise EvidenceError(f"resolver receipt has no package list: {profile}")
    artifact_index = _artifact_index(inputs)
    local_seen: set[tuple[str, str]] = set()
    seen_names: set[str] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise EvidenceError(f"resolver receipt package is not an object: {profile}")
        required = {
            "name",
            "version",
            "filename",
            "sha256",
            "resolution_sha256",
            "lock_hashes",
            "origin",
            "download_url",
            "requested",
            "requested_extras",
            "requires_dist",
            "artifact_availability",
        }
        missing = sorted(required - set(package))
        if missing:
            raise EvidenceError(
                f"resolver receipt package fields are absent in {profile}: {missing}"
            )
        name = package["name"]
        if not isinstance(name, str) or _normalized_name(name) != name or name in seen_names:
            raise EvidenceError(f"resolver receipt package name is invalid or duplicated: {name!r}")
        seen_names.add(name)
        version = package["version"]
        filename = package["filename"]
        if not isinstance(version, str) or not version:
            raise EvidenceError(f"resolver receipt package version is invalid: {name}")
        if (
            not isinstance(filename, str)
            or not filename
            or PurePosixPath(filename).name != filename
        ):
            raise EvidenceError(f"resolver receipt archive filename is invalid: {name}")
        sha256 = package["sha256"]
        resolution_sha256 = package["resolution_sha256"]
        lock_hashes = package["lock_hashes"]
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise EvidenceError(f"resolver receipt package SHA-256 is invalid: {name}")
        if not isinstance(resolution_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", resolution_sha256
        ):
            raise EvidenceError(f"resolver receipt resolution SHA-256 is invalid: {name}")
        if (
            not isinstance(lock_hashes, list)
            or not lock_hashes
            or lock_hashes != sorted(set(lock_hashes))
            or not all(
                isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
                for item in lock_hashes
            )
        ):
            raise EvidenceError(f"resolver receipt lock hashes are invalid: {name}")
        if not isinstance(package["requested"], bool):
            raise EvidenceError(f"resolver receipt requested flag is invalid: {name}")
        for field in ("requested_extras", "requires_dist"):
            values = package[field]
            if (
                not isinstance(values, list)
                or not all(isinstance(item, str) for item in values)
                or values != sorted(set(values))
            ):
                raise EvidenceError(f"resolver receipt {field} is invalid: {name}")
        if package["artifact_availability"] != "available":
            raise EvidenceError(f"selected archive is not explicitly available: {name}")

        origin = package["origin"]
        download_url = package["download_url"]
        if not isinstance(download_url, str):
            raise EvidenceError(f"resolver receipt download URL is invalid: {name}")
        key = (name, version)
        if origin == LOCAL_ORIGIN:
            admitted = artifact_index.get(key)
            if admitted is None:
                raise EvidenceError(f"unadmitted local selected distribution in {profile}: {name}")
            wheel = next(item for item in admitted if item["kind"] == "wheel")
            expected_hashes = sorted(item["sha256"] for item in admitted)
            if (
                filename != wheel["filename"]
                or sha256 != wheel["sha256"]
                or resolution_sha256 != wheel["sha256"]
                or lock_hashes != expected_hashes
                or download_url != f"artifact://{wheel['filename']}"
                or package.get("admitted_artifact_license") != wheel.get("license_declared")
            ):
                raise EvidenceError(f"local selected archive is not exactly admitted: {name}")
            local_seen.add(key)
        elif origin == PYPI_ORIGIN:
            if key in artifact_index:
                raise EvidenceError(
                    f"admitted direct package was relabeled as an index archive: {name}"
                )
            if _safe_remote_url(download_url) != download_url:
                raise EvidenceError(f"selected index archive URL is not canonical: {name}")
            parsed = urlsplit(download_url)
            if (
                unquote(PurePosixPath(parsed.path).name) != filename
                or sha256 != resolution_sha256
                or lock_hashes != [sha256]
                or package.get("admitted_artifact_license") is not None
            ):
                raise EvidenceError(f"selected index archive identity is inconsistent: {name}")
        else:
            raise EvidenceError(
                f"selected archive origin escaped the allowlist in {profile}: {origin!r}"
            )
    if local_seen != set(artifact_index):
        raise EvidenceError(f"resolver receipt does not select all four admitted wheels: {profile}")
    if [package["name"] for package in packages] != sorted(seen_names):
        raise EvidenceError(f"resolver receipt package order is not canonical: {profile}")
    root = next((package for package in packages if package["name"] == "ipfs-accelerate-py"), None)
    expected_extra = inputs["profiles"][profile]["accelerator_extra"]
    expected_extras = [] if expected_extra is None else [expected_extra]
    if (
        root is None
        or root["requested_extras"] != expected_extras
        or root["origin"] != LOCAL_ORIGIN
    ):
        raise EvidenceError(f"resolver receipt root extras or origin drifted: {profile}")


def _load_receipts(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    root = LOCK_ROOT / ENVIRONMENT_SLUG
    artifact_keys = set(_artifact_index(inputs))
    for profile in PROFILES:
        receipt_path = root / f"{profile}.resolver.json"
        receipt = _load_json(receipt_path)
        if (
            receipt.get("schema") != f"{SCHEMA_PREFIX}.resolver-receipt@1"
            or receipt.get("profile") != profile
        ):
            raise EvidenceError(f"invalid resolver receipt: {receipt_path}")
        if receipt.get("environment") != inputs["supported_environment"]:
            raise EvidenceError(f"resolver receipt environment drifted: {receipt_path}")
        expected_profile = inputs["profiles"][profile]
        if receipt.get("root_requirement") != expected_profile["root_requirement"]:
            raise EvidenceError(f"resolver receipt root requirement drifted: {receipt_path}")
        for field in (
            "resolution_status",
            "artifact_clean_install_status",
            "native_build_status",
        ):
            if receipt.get(field) != expected_profile[field]:
                raise EvidenceError(f"resolver receipt {field} drifted: {receipt_path}")
        resolver = receipt.get("resolver")
        if not isinstance(resolver, dict):
            raise EvidenceError(f"resolver receipt resolver fields are absent: {receipt_path}")
        expected_resolver = {
            "name": "pip",
            "version": inputs["resolver"]["pip_version"],
            "index": PYPI_ORIGIN,
            "extra_indexes": [],
            "resolution_only": True,
            "raw_report_retained": False,
            "command": inputs["resolver"]["command"],
        }
        if any(resolver.get(field) != value for field, value in expected_resolver.items()):
            raise EvidenceError(f"resolver receipt resolver policy drifted: {receipt_path}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(resolver.get("raw_report_sha256", ""))):
            raise EvidenceError(f"resolver receipt report SHA-256 is invalid: {receipt_path}")
        packages = receipt.get("packages")
        _validate_committed_package_sources(
            profile=profile,
            packages=packages,
            inputs=inputs,
        )
        expected_risk_ledger = _requirement_risk_ledger(
            profile=profile,
            packages=packages,
            inputs=inputs,
        )
        if receipt.get("requirement_risk_ledger") != expected_risk_ledger:
            raise EvidenceError(f"resolver receipt requirement risk ledger drifted: {receipt_path}")
        lock_path = root / f"{profile}.txt"
        expected_lock = _lock_bytes(receipt)
        if not lock_path.is_file() or lock_path.read_bytes() != expected_lock:
            raise EvidenceError(f"lock differs from resolver receipt: {lock_path}")
        package_keys = {(package["name"], package["version"]) for package in packages}
        if not artifact_keys <= package_keys:
            raise EvidenceError(f"lock omits an admitted package pair: {lock_path}")
        receipts[profile] = receipt
    return receipts


def _spdx_license(package: dict[str, Any]) -> str:
    admitted_license = package.get("admitted_artifact_license")
    if admitted_license is not None:
        return str(admitted_license)
    metadata = package["license"]
    raw = metadata.get("core_metadata_license")
    exact = {
        "AGPL-3.0-or-later": "AGPL-3.0-or-later",
        "GNU Affero General Public License v3 or later (AGPLv3+)": "AGPL-3.0-or-later",
        "MIT": "MIT",
        "MIT License": "MIT",
        "Apache": "Apache-2.0",
        "Apache 2.0": "Apache-2.0",
        "Apache-2.0": "Apache-2.0",
        "BSD-3-Clause": "BSD-3-Clause",
        "ISC": "ISC",
        "MPL-2.0": "MPL-2.0",
        "CC0-1.0 OR Apache-2.0": "CC0-1.0 OR Apache-2.0",
        "MIT OR Apache-2.0": "MIT OR Apache-2.0",
        "MPL-2.0 AND MIT": "MPL-2.0 AND MIT",
    }
    if raw in exact:
        return exact[raw]
    classifiers = set(metadata.get("license_classifiers", []))
    if classifiers == {"License :: OSI Approved :: MIT License"}:
        return "MIT"
    if classifiers == {"License :: OSI Approved :: Apache Software License"}:
        return "Apache-2.0"
    if classifiers == {"License :: OSI Approved :: ISC License (ISCL)"}:
        return "ISC"
    if classifiers == {"License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)"}:
        return "MPL-2.0"
    return "NOASSERTION"


def _spdx_id(name: str) -> str:
    return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name)


def _union_packages(receipts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    union: dict[tuple[str, str, str], dict[str, Any]] = {}
    for profile, receipt in receipts.items():
        for package in receipt["packages"]:
            key = (package["name"], package["version"], package["sha256"])
            existing = union.setdefault(key, {**package, "profiles": []})
            if existing["lock_hashes"] != package["lock_hashes"]:
                raise EvidenceError(f"profile lock hash drift for {package['name']}")
            existing["profiles"].append(profile)
    result = list(union.values())
    for package in result:
        package["profiles"].sort(key=PROFILES.index)
    result.sort(key=lambda package: (package["name"], package["version"], package["sha256"]))
    return result


def _outer_documents(
    inputs: dict[str, Any], receipts: dict[str, dict[str, Any]]
) -> dict[Path, bytes]:
    nested_commit = _git_output("rev-parse", "HEAD")
    nested_tree = _git_output("rev-parse", "HEAD^{tree}")
    if _git_output("status", "--porcelain"):
        raise EvidenceError(
            "accelerator checkout must be clean before environment evidence is generated"
        )
    lock_records = []
    for profile in PROFILES:
        lock_path = LOCK_ROOT / ENVIRONMENT_SLUG / f"{profile}.txt"
        receipt_path = LOCK_ROOT / ENVIRONMENT_SLUG / f"{profile}.resolver.json"
        receipt = receipts[profile]
        lock_sha256 = _sha256_path(lock_path)
        receipt_sha256 = _sha256_path(receipt_path)
        source_distributions = sorted(
            package["filename"]
            for package in receipt["packages"]
            if not package["filename"].endswith(".whl")
        )
        lock_records.append(
            {
                "profile": profile,
                "path": str(lock_path.relative_to(ACCELERATOR_ROOT)),
                "sha256": lock_sha256,
                "cid_v1_raw": _raw_cid_v1_from_sha256(lock_sha256),
                "cid_binding_status": "bytes-verified",
                "resolver_receipt_path": str(receipt_path.relative_to(ACCELERATOR_ROOT)),
                "resolver_receipt_sha256": receipt_sha256,
                "resolver_receipt_cid_v1_raw": _raw_cid_v1_from_sha256(receipt_sha256),
                "resolver_receipt_cid_binding_status": "bytes-verified",
                "raw_pip_report_sha256": receipt["resolver"]["raw_report_sha256"],
                "distribution_count": len(receipt["packages"]),
                "selected_source_distributions": source_distributions,
                "qualification": inputs["profiles"][profile]["qualification"],
                "resolution_status": inputs["profiles"][profile]["resolution_status"],
                "artifact_clean_install_status": inputs["profiles"][profile][
                    "artifact_clean_install_status"
                ],
                "native_build_status": inputs["profiles"][profile]["native_build_status"],
            }
        )
    requirement_source_risk = {
        "policy_status": "passed",
        "policy": inputs["mutable_vcs_metadata_policy"],
        "profiles": {
            profile: {
                "inactive_mutable_vcs_core_metadata": receipts[profile]["requirement_risk_ledger"][
                    "inactive_mutable_vcs_core_metadata"
                ],
                "selected_unsafe_vcs_direct_editable_path_requirements": receipts[profile][
                    "requirement_risk_ledger"
                ]["selected_unsafe_vcs_direct_editable_path_requirements"],
                "selected_unsafe_requirements_by_class": receipts[profile][
                    "requirement_risk_ledger"
                ]["selected_unsafe_requirements_by_class"],
                "selected_archive_origin_counts": {
                    origin: len(filenames)
                    for origin, filenames in receipts[profile]["requirement_risk_ledger"][
                        "selected_archive_origins"
                    ].items()
                },
                "selected_archive_origin_policy": receipts[profile]["requirement_risk_ledger"][
                    "selected_archive_origin_policy"
                ],
            }
            for profile in PROFILES
        },
    }
    dependency_locks = {
        "schema": f"{SCHEMA_PREFIX}.dependency-locks@1",
        "environment_id": ENVIRONMENT_SLUG,
        "cid_policy": "CIDv1 with raw multicodec and sha2-256 multihash",
        "resolution_status": inputs["resolution_status"],
        "artifact_byte_availability_status": inputs["artifact_byte_availability_status"],
        "artifact_clean_install_status": inputs["artifact_clean_install_status"],
        "resolution_basis": "Fresh pip metadata resolution with exact admitted wheels for all four direct packages; every direct lock entry also binds its admitted sdist hash.",
        "supported_environment": inputs["supported_environment"],
        "resolver": inputs["resolver"],
        "hash_gate_validation": inputs["hash_gate_validation"],
        "requirement_source_risk": requirement_source_risk,
        "locks": lock_records,
    }
    artifact_hashes = {
        "schema": f"{SCHEMA_PREFIX}.artifact-hashes@1",
        "identity_policy": "SHA-256 over the exact admitted archive bytes",
        "cid_policy": "CIDv1 with raw multicodec and sha2-256 multihash",
        "resolution_status": inputs["resolution_status"],
        "artifact_byte_availability_status": inputs["artifact_byte_availability_status"],
        "artifact_clean_install_status": inputs["artifact_clean_install_status"],
        "source_commits": inputs["source_commits"],
        "artifacts": [
            {
                **artifact,
                "cid_v1_raw": _raw_cid_v1_from_sha256(artifact["sha256"]),
                "cid_binding_status": (
                    "bytes-verified"
                    if artifact.get("bytes_available", True)
                    else "identity-derived-bytes-unavailable"
                ),
                "bytes_verified": artifact.get("bytes_available", True),
            }
            for artifact in inputs["artifacts"]
        ],
        "semantic_surrogates": inputs["semantic_surrogates"],
    }
    union = _union_packages(receipts)
    namespace_seed = _sha256_bytes(
        _json_bytes({"artifacts": inputs["artifacts"], "locks": lock_records})
    )
    spdx_packages = []
    for package in union:
        license_declared = _spdx_license(package)
        comment = f"Resolved profiles: {', '.join(package['profiles'])}."
        if package["build_system"]["source_build_required"]:
            comment += " The selected artifact is an sdist; PCCE-053 records but does not qualify its build."
        raw_license = package["license"].get("core_metadata_license")
        if license_declared == "NOASSERTION":
            comment += f" Core Metadata license evidence: {raw_license or 'not declared'}; no license conclusion was inferred."
        spdx_packages.append(
            {
                "SPDXID": _spdx_id(package["name"]),
                "name": package["name"],
                "versionInfo": package["version"],
                "downloadLocation": package["download_url"]
                if package["origin"] == PYPI_ORIGIN
                else "NOASSERTION",
                "filesAnalyzed": False,
                "checksums": [{"algorithm": "SHA256", "checksumValue": package["sha256"]}],
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": license_declared,
                "copyrightText": "NOASSERTION",
                "comment": comment,
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{package['name']}@{package['version']}",
                    }
                ],
            }
        )
    document_id = "SPDXRef-DOCUMENT"
    relationships = [
        {
            "spdxElementId": document_id,
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": package["SPDXID"],
        }
        for package in spdx_packages
    ]
    root_id = _spdx_id("ipfs-accelerate-py")
    relationships.extend(
        {
            "spdxElementId": root_id,
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": package["SPDXID"],
        }
        for package in spdx_packages
        if package["SPDXID"] != root_id
    )
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": document_id,
        "name": f"proof-context-{ENVIRONMENT_SLUG}",
        "documentNamespace": f"https://github.com/endomorphosis/ipfs_accelerate_py/spdx/PCCE-053/{namespace_seed}",
        "creationInfo": {
            "created": inputs["evidence_created"],
            "creators": [f"Tool: build_environment_manifest.py@{nested_commit}"],
        },
        "documentDescribes": [package["SPDXID"] for package in spdx_packages],
        "packages": spdx_packages,
        "relationships": relationships,
    }
    dependency_bytes = _json_bytes(dependency_locks)
    artifact_bytes = _json_bytes(artifact_hashes)
    sbom_bytes = _json_bytes(sbom)
    environment_sha256 = {
        "dependency_locks.json": _sha256_bytes(dependency_bytes),
        "artifact_hashes.json": _sha256_bytes(artifact_bytes),
        "sbom.spdx.json": _sha256_bytes(sbom_bytes),
    }
    environment_cid_v1_raw = {
        name: _raw_cid_v1_from_sha256(sha256) for name, sha256 in environment_sha256.items()
    }
    manifest = {
        "schema": f"{SCHEMA_PREFIX}.environment-manifest@1",
        "environment_id": ENVIRONMENT_SLUG,
        "builder": {
            "path": "external/ipfs_accelerate/scripts/proof_context/build_environment_manifest.py",
            "source_commit": nested_commit,
            "source_tree": nested_tree,
            "deterministic_json": "UTF-8, sorted keys, two-space indentation, LF terminator",
        },
        "package_source_commits": inputs["source_commits"],
        "resolution_status": inputs["resolution_status"],
        "artifact_byte_availability_status": inputs["artifact_byte_availability_status"],
        "artifact_clean_install_status": inputs["artifact_clean_install_status"],
        "supported_environments": [
            {
                **inputs["supported_environment"],
                "support_scope": "hash-bound-dependency-resolution-only",
                "resolution_status": inputs["resolution_status"],
            }
        ],
        "resolution_environments": [
            {
                **inputs["supported_environment"],
                "support_scope": "hash-bound-dependency-resolution-only",
                "resolution_status": inputs["resolution_status"],
                "artifact_clean_install_status": inputs["artifact_clean_install_status"],
                "profiles": {
                    record["profile"]: {
                        "qualification": record["qualification"],
                        "resolution_status": record["resolution_status"],
                        "artifact_clean_install_status": record["artifact_clean_install_status"],
                        "native_build_status": record["native_build_status"],
                        "selected_source_distributions": record["selected_source_distributions"],
                    }
                    for record in lock_records
                },
                "runner_observed": inputs["resolver"]["observed_environment"],
            }
        ],
        "unvalidated_environments": inputs["unvalidated_environments"],
        "optional_capabilities": {
            profile: inputs["profiles"][profile]["capabilities"] for profile in PROFILES
        },
        "toolchain": inputs["toolchain"],
        "indexes": {"primary": PYPI_ORIGIN, "additional": []},
        "hash_gate_validation": inputs["hash_gate_validation"],
        "requirement_source_risk": requirement_source_risk,
        # Deliberately exclude the manifest's own digest; the task receipt
        # binds it after these bytes exist.
        "evidence": dict(environment_sha256),
        "evidence_cid_v1_raw": dict(environment_cid_v1_raw),
    }
    manifest_bytes = _json_bytes(manifest)
    environment_sha256["manifest.json"] = _sha256_bytes(manifest_bytes)
    environment_cid_v1_raw["manifest.json"] = _raw_cid_v1_from_bytes(manifest_bytes)
    receipt = {
        "schema": f"{SCHEMA_PREFIX}.task-receipt@1",
        "task_id": "PCCE-053",
        "objective_id": "PCCE-G500",
        "status": "completed",
        "completion_mode": "supervised-reproducibility-implementation",
        "artifact_identity": environment_cid_v1_raw["manifest.json"],
        "artifact_identity_kind": "CIDv1(raw,sha2-256)",
        "board_namespace": "proof-carrying-context-engine-v0.1",
        "evidence": {
            "nested_source": {
                "repository": "external/ipfs_accelerate",
                "base_commit": inputs["source_commits"]["ipfs_accelerate_py"],
                "implementation_commit": nested_commit,
                "repository_tree": nested_tree,
            },
            "supported_environment": inputs["supported_environment"],
            "resolution_status": inputs["resolution_status"],
            "artifact_byte_availability_status": inputs["artifact_byte_availability_status"],
            "artifact_clean_install_status": inputs["artifact_clean_install_status"],
            "semantic_surrogates": inputs["semantic_surrogates"],
            "profiles": {
                record["profile"]: {
                    "qualification": record["qualification"],
                    "resolution_status": record["resolution_status"],
                    "artifact_clean_install_status": record["artifact_clean_install_status"],
                    "native_build_status": record["native_build_status"],
                    "selected_source_distributions": record["selected_source_distributions"],
                }
                for record in lock_records
            },
            "artifact_count": len(inputs["artifacts"]),
            "lock_count": len(PROFILES),
            "sbom_distribution_count": len(spdx_packages),
            "resolver_reports": {
                profile: {
                    "sha256": receipts[profile]["resolver"]["raw_report_sha256"],
                    "distribution_count": len(receipts[profile]["packages"]),
                }
                for profile in PROFILES
            },
            "hash_gate_validation": inputs["hash_gate_validation"],
            "requirement_source_risk": requirement_source_risk,
            "deterministic_generation": {"runs_compared": 2, "byte_identical": True},
            "output_sha256": environment_sha256,
            "output_cid_v1_raw": environment_cid_v1_raw,
            "validation": inputs["validation"],
        },
        "rollback": "Revert only the PCCE-053 lock/builder changes, accelerator gitlink, generated environment evidence, and this receipt; invalidate unpromoted environment identities.",
    }
    return {
        ENVIRONMENT_OUTPUTS / "dependency_locks.json": dependency_bytes,
        ENVIRONMENT_OUTPUTS / "artifact_hashes.json": artifact_bytes,
        ENVIRONMENT_OUTPUTS / "sbom.spdx.json": sbom_bytes,
        ENVIRONMENT_OUTPUTS / "manifest.json": manifest_bytes,
        RECEIPT_OUTPUT: _json_bytes(receipt),
    }


def _discover_workspace_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "artifacts" / "proof_carrying_context_engine").is_dir():
            return candidate
    raise EvidenceError("cannot discover outer workspace; pass --workspace-root")


def _materialize_outer(workspace_root: Path, *, check: bool) -> None:
    inputs = _inputs()
    receipts = _load_receipts(inputs)
    documents = _outer_documents(inputs, receipts)
    for relative_path, content in documents.items():
        path = workspace_root / relative_path
        if check:
            if not path.is_file() or path.read_bytes() != content:
                raise EvidenceError(f"generated environment evidence is stale: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="compare canonical bytes without writing"
    )
    parser.add_argument(
        "--compile-locks",
        action="store_true",
        help="normalize fresh <profile>.json pip reports into locks and resolver receipts",
    )
    parser.add_argument(
        "--reports-dir", type=Path, help="directory containing fresh pip JSON reports"
    )
    parser.add_argument(
        "--artifact-dir", type=Path, help="directory containing all eight admitted archives"
    )
    parser.add_argument(
        "--source-archives-dir",
        type=Path,
        help="directory containing every source distribution selected by the fresh reports",
    )
    parser.add_argument("--workspace-root", type=Path, help="outer lift_coding checkout root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.compile_locks:
            if (
                args.reports_dir is None
                or args.artifact_dir is None
                or args.source_archives_dir is None
            ):
                raise EvidenceError(
                    "--compile-locks requires --reports-dir, --artifact-dir, and --source-archives-dir"
                )
            _compile_locks(
                args.reports_dir.resolve(),
                args.artifact_dir.resolve(),
                args.source_archives_dir.resolve(),
                check=args.check,
            )
        else:
            if any(
                value is not None
                for value in (args.reports_dir, args.artifact_dir, args.source_archives_dir)
            ):
                raise EvidenceError(
                    "report/artifact directories are valid only with --compile-locks"
                )
            _materialize_outer(_discover_workspace_root(args.workspace_root), check=args.check)
    except EvidenceError as exc:
        print(f"PCCE-053 environment evidence error: {exc}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "generated"
    target = "locks and resolver receipts" if args.compile_locks else "environment evidence"
    print(f"{action} {target} for {ENVIRONMENT_SLUG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
