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
        ["git", "-C", str(ACCELERATOR_ROOT), *args],
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
        url, archive_sha256 = _report_archive(install)
        parsed = urlsplit(url)
        filename = unquote(Path(parsed.path).name)
        key = (name, version)
        if parsed.scheme == "file":
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
        lock_path = root / f"{profile}.txt"
        expected_lock = _lock_bytes(receipt)
        if not lock_path.is_file() or lock_path.read_bytes() != expected_lock:
            raise EvidenceError(f"lock differs from resolver receipt: {lock_path}")
        package_keys = {(package["name"], package["version"]) for package in receipt["packages"]}
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
        source_distributions = sorted(
            package["filename"]
            for package in receipt["packages"]
            if not package["filename"].endswith(".whl")
        )
        lock_records.append(
            {
                "profile": profile,
                "path": str(lock_path.relative_to(ACCELERATOR_ROOT)),
                "sha256": _sha256_path(lock_path),
                "resolver_receipt_path": str(receipt_path.relative_to(ACCELERATOR_ROOT)),
                "resolver_receipt_sha256": _sha256_path(receipt_path),
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
    dependency_locks = {
        "schema": f"{SCHEMA_PREFIX}.dependency-locks@1",
        "environment_id": ENVIRONMENT_SLUG,
        "resolution_status": inputs["resolution_status"],
        "artifact_byte_availability_status": inputs["artifact_byte_availability_status"],
        "artifact_clean_install_status": inputs["artifact_clean_install_status"],
        "resolution_basis": "Fresh pip metadata resolution with exact admitted wheels for all four direct packages; every direct lock entry also binds its admitted sdist hash.",
        "supported_environment": inputs["supported_environment"],
        "resolver": inputs["resolver"],
        "hash_gate_validation": inputs["hash_gate_validation"],
        "locks": lock_records,
    }
    artifact_hashes = {
        "schema": f"{SCHEMA_PREFIX}.artifact-hashes@1",
        "identity_policy": "SHA-256 over the exact admitted archive bytes",
        "resolution_status": inputs["resolution_status"],
        "artifact_byte_availability_status": inputs["artifact_byte_availability_status"],
        "artifact_clean_install_status": inputs["artifact_clean_install_status"],
        "source_commits": inputs["source_commits"],
        "artifacts": inputs["artifacts"],
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
    environment_refs = {
        "dependency_locks.json": _sha256_bytes(dependency_bytes),
        "artifact_hashes.json": _sha256_bytes(artifact_bytes),
        "sbom.spdx.json": _sha256_bytes(sbom_bytes),
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
        # Deliberately exclude the manifest's own digest; the task receipt
        # binds it after these bytes exist.
        "evidence": dict(environment_refs),
    }
    manifest_bytes = _json_bytes(manifest)
    environment_refs["manifest.json"] = _sha256_bytes(manifest_bytes)
    receipt = {
        "schema": f"{SCHEMA_PREFIX}.task-receipt@1",
        "task_id": "PCCE-053",
        "objective_id": "PCCE-G500",
        "status": "completed",
        "completion_mode": "supervised-reproducibility-implementation",
        "artifact_identity": f"urn:pcce:environment:{ENVIRONMENT_SLUG}:{environment_refs['manifest.json']}",
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
            "deterministic_generation": {"runs_compared": 2, "byte_identical": True},
            "output_sha256": environment_refs,
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
