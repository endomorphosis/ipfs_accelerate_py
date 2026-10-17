#!/usr/bin/env python3
"""Stage the frozen ordinary-worker wheel closure without executing wheel code.

Invoke only inside the caller's genuine, storage-admitted native build owner.
This stdlib source does not create a lease, invoke Docker, run an installer,
import installed packages, inspect credentials, or grant dispatch authority.
The caller owns image construction, exact base binding, runtime qualification,
and the independently reviewed ordinary-route integration.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import email.parser
import hashlib
import io
import json
import os
import re
import ssl
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

LOCK_SHA256 = "eed4f3a61cdba4930473acd3c4f28debe24966218cb74506fcd38fd6ec5f82d8"
CONTAINERFILE_SHA256 = "b256241b381c9611c1a2fd42677641bed45d18135f693a9d82effb167f5c5794"
BASE_IMAGE_ID = "sha256:95d16dfcd4ab8b61154e2280ab9fe4afa2ef03e5eefa8a689e86571e8a1b32cf"
PROFILE = "ordinary-task-toolchain-v1"
LABEL = "2026-10-16-ordinary-v1"
SOURCE_DATE_EPOCH = 1792108800  # 2026-10-16 00:00:00 UTC.
SITE = PurePosixPath("opt/ipfs-validation-site-packages")
MAX_LOCK_BYTES = 64 * 1024
MAX_DOWNLOAD_BYTES = 32 * 1024**2
MAX_WHEEL_ENTRIES = 10_000
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_CLOSURE_BYTES = 512 * 1024**2
MAX_METADATA_BYTES = 4 * 1024**2
MAX_DOWNLOAD_SECONDS = 180
EXPECTED_PACKAGES = {
    "pytest": "9.0.3", "pluggy": "1.6.0", "packaging": "26.2",
    "iniconfig": "2.3.0", "pygments": "2.19.2", "cryptography": "49.0.0",
    "cffi": "2.1.0", "pycparser": "3.0", "duckdb": "1.5.2",
}
PLACEHOLDER = b"#!/bin/sh\nprintf '%s\\n' 'Trusted host Python bind mount required' >&2\nexit 125\n"


class BuildInputError(ValueError):
    """A bounded frozen-input or archive check failed."""


@dataclass(frozen=True)
class WheelInput:
    name: str
    version: str
    filename: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BuildLock:
    wheels: tuple[WheelInput, ...]
    sha256: str
    base_image_id: str


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _safe_path(value: str, *, directory: bool = False) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BuildInputError("unsafe_archive_path")
    stripped = value[:-1] if directory and value.endswith("/") else value
    components = stripped.split("/")
    if (not stripped or len(stripped.encode("utf-8")) > 240
            or any(c in {"", ".", ".."} for c in components)
            or any(ord(c) < 32 or ord(c) == 127 for c in stripped)
            or any(":" in c for c in components)):
        raise BuildInputError("unsafe_archive_path")
    path = PurePosixPath(stripped)
    if path.is_absolute() or str(path) != stripped:
        raise BuildInputError("noncanonical_archive_path")
    if any(part.endswith(".data") for part in path.parts):
        raise BuildInputError("wheel_data_layout_not_admitted")
    if (path.name.endswith(".pth")
            or path.name.startswith(("sitecustomize.", "usercustomize."))):
        raise BuildInputError("python_startup_hook_not_admitted")
    if any(part in {"sitecustomize", "usercustomize"} for part in path.parts):
        raise BuildInputError("python_startup_hook_not_admitted")
    return path


def load_lock(path: Path) -> BuildLock:
    """Accept only the exact reviewed nine-wheel manifest bytes."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOCK_BYTES:
        raise BuildInputError("invalid_lock_file")
    raw = path.read_bytes()
    if _digest(raw) != LOCK_SHA256:
        raise BuildInputError("frozen_lock_sha256_mismatch")
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get("schema") != "legal-ir-ordinary-worker-wheels@1"
            or value.get("python") != "cp312" or value.get("os") != "linux"
            or value.get("architecture") != "arm64"
            or value.get("base_image_id") != BASE_IMAGE_ID
            or value.get("old_authority_image_required") is not False):
        raise BuildInputError("invalid_lock_contract")
    records = value.get("packages")
    if not isinstance(records, list) or len(records) != len(EXPECTED_PACKAGES):
        raise BuildInputError("invalid_wheel_population")
    wheels = []
    seen = set()
    for item in records:
        if not isinstance(item, dict):
            raise BuildInputError("invalid_wheel_record")
        fields = ("name", "version", "filename", "url", "sha256")
        if any(not isinstance(item.get(key), str) for key in fields):
            raise BuildInputError("invalid_wheel_record")
        name = _name(item["name"])
        if name in seen or EXPECTED_PACKAGES.get(name) != item["version"]:
            raise BuildInputError("unexpected_or_duplicate_wheel")
        seen.add(name)
        filename = item["filename"]
        if ("/" in filename or "\\" in filename or not filename.endswith(".whl")
                or re.fullmatch(r"[A-Za-z0-9_.-]+", filename) is None):
            raise BuildInputError("invalid_wheel_filename")
        url = urllib.parse.urlsplit(item["url"])
        if (url.scheme != "https" or url.hostname != "files.pythonhosted.org"
                or url.username is not None or url.password is not None
                or url.port not in (None, 443) or url.query or url.fragment
                or not url.path.startswith("/packages/") or not url.path.endswith("/" + filename)):
            raise BuildInputError("wheel_url_outside_frozen_public_origin")
        size = item.get("size")
        if type(size) is not int or not 1 <= size <= MAX_DOWNLOAD_BYTES:
            raise BuildInputError("invalid_wheel_size")
        if re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None:
            raise BuildInputError("invalid_wheel_sha256")
        wheels.append(WheelInput(name, item["version"], filename, item["url"], size, item["sha256"]))
    if seen != set(EXPECTED_PACKAGES) or sum(w.size for w in wheels) > MAX_DOWNLOAD_BYTES:
        raise BuildInputError("wheel_population_or_byte_bound_failed")
    return BuildLock(tuple(sorted(wheels, key=lambda w: w.name)), LOCK_SHA256, BASE_IMAGE_ID)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise BuildInputError("wheel_download_redirect_not_admitted")


def _download(wheel: WheelInput, destination: Path) -> None:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context(
            cafile="/etc/ssl/certs/ca-certificates.crt")),
    )
    request = urllib.request.Request(wheel.url, headers={
        "User-Agent": "legal-ir-frozen-wheel-stager/1", "Accept-Encoding": "identity",
    })
    size = 0
    digest = hashlib.sha256()
    deadline = time.monotonic() + MAX_DOWNLOAD_SECONDS
    with opener.open(request, timeout=20) as response, destination.open("xb") as target:
        if response.status != 200 or response.geturl() != wheel.url:
            raise BuildInputError("wheel_download_response_not_admitted")
        header_size = response.headers.get("Content-Length")
        if header_size is not None and header_size != str(wheel.size):
            raise BuildInputError("wheel_download_declared_size_mismatch")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise BuildInputError("wheel_download_encoding_not_admitted")
        while True:
            if time.monotonic() > deadline:
                raise BuildInputError("wheel_download_deadline_exceeded")
            block = response.read(min(1024 * 1024, wheel.size - size + 1))
            if not block:
                break
            size += len(block)
            if size > wheel.size:
                raise BuildInputError("wheel_download_size_exceeded")
            digest.update(block)
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
    if size != wheel.size or digest.hexdigest() != wheel.sha256:
        raise BuildInputError("wheel_download_size_or_sha256_mismatch")
    destination.chmod(0o444)


def _record_rows(raw: bytes, expected_record: str) -> dict[str, tuple[str, int | None]]:
    if len(raw) > MAX_METADATA_BYTES:
        raise BuildInputError("wheel_record_byte_bound_exceeded")
    rows: dict[str, tuple[str, int | None]] = {}
    for row in csv.reader(io.StringIO(raw.decode("utf-8"), newline="")):
        if len(row) != 3:
            raise BuildInputError("malformed_wheel_record")
        path = str(_safe_path(row[0]))
        if path in rows or len(rows) >= MAX_WHEEL_ENTRIES:
            raise BuildInputError("duplicate_or_excessive_record_member")
        if path == expected_record:
            if row[1:] != ["", ""]:
                raise BuildInputError("wheel_record_self_hash_not_empty")
            rows[path] = ("", None)
        else:
            if (not row[1].startswith("sha256=")
                    or re.fullmatch(r"[A-Za-z0-9_-]{43}", row[1][7:]) is None
                    or re.fullmatch(r"0|[1-9][0-9]*", row[2]) is None):
                raise BuildInputError("wheel_record_hash_or_size_invalid")
            expected = base64.urlsafe_b64decode(row[1][7:] + "=").hex()
            size = int(row[2])
            if size > MAX_MEMBER_BYTES:
                raise BuildInputError("wheel_record_member_byte_bound_exceeded")
            rows[path] = (expected, size)
    return rows


def _extract(wheel: WheelInput, archive: Path, rootfs: Path,
             inventory: dict[str, dict[str, object]], budget: list[int]) -> None:
    """Verify a complete wheel's RECORD, then write only verified regular files."""
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if not members or len(members) > MAX_WHEEL_ENTRIES:
            raise BuildInputError("wheel_entry_population_invalid")
        files: dict[str, zipfile.ZipInfo] = {}
        seen = set()
        for member in members:
            directory = member.is_dir()
            name = str(_safe_path(member.filename, directory=directory))
            if name in seen:
                raise BuildInputError("duplicate_zip_member")
            seen.add(name)
            mode = member.external_attr >> 16
            if (member.flag_bits & 1 or stat.S_ISLNK(mode)
                    or stat.S_IFMT(mode) not in (0, stat.S_IFDIR if directory else stat.S_IFREG)):
                raise BuildInputError("nonregular_or_encrypted_wheel_member")
            if directory:
                if member.file_size != 0:
                    raise BuildInputError("nonempty_wheel_directory")
                continue
            if (member.file_size > MAX_MEMBER_BYTES or member.compress_size < 0
                    or (member.file_size > 1024**2 and member.file_size > 500 * max(1, member.compress_size))):
                raise BuildInputError("wheel_member_expansion_bound_exceeded")
            files[name] = member
        dist_infos = {PurePosixPath(name).parts[0] for name in files
                      if PurePosixPath(name).parts[0].endswith(".dist-info")}
        if len(dist_infos) != 1:
            raise BuildInputError("wheel_dist_info_population_invalid")
        dist_info = next(iter(dist_infos))
        expected_info = wheel.name.replace("-", "_") + "-" + wheel.version + ".dist-info"
        if dist_info.lower() != expected_info.lower():
            raise BuildInputError("wheel_dist_info_identity_mismatch")
        record_name = dist_info + "/RECORD"
        if record_name not in files or files[record_name].file_size > MAX_METADATA_BYTES:
            raise BuildInputError("wheel_record_missing_or_excessive")
        records = _record_rows(source.read(files[record_name]), record_name)
        if set(records) != set(files):
            raise BuildInputError("wheel_record_does_not_cover_exact_member_set")
        metadata_name, wheel_name = dist_info + "/METADATA", dist_info + "/WHEEL"
        if any(name not in files or files[name].file_size > MAX_METADATA_BYTES
               for name in (metadata_name, wheel_name)):
            raise BuildInputError("wheel_metadata_missing_or_excessive")
        metadata = email.parser.BytesParser().parsebytes(source.read(files[metadata_name]))
        wheel_metadata = email.parser.BytesParser().parsebytes(source.read(files[wheel_name]))
        if _name(str(metadata["Name"])) != wheel.name or metadata["Version"] != wheel.version:
            raise BuildInputError("wheel_metadata_identity_mismatch")
        if wheel_metadata["Wheel-Version"] != "1.0":
            raise BuildInputError("unsupported_wheel_version")
        for name, member in sorted(files.items()):
            relative = str(SITE / name)
            if relative in inventory:
                raise BuildInputError("cross_wheel_file_collision")
            budget[0] += member.file_size
            budget[1] += 1
            if budget[0] > MAX_CLOSURE_BYTES or budget[1] > MAX_WHEEL_ENTRIES:
                raise BuildInputError("installed_closure_bound_exceeded")
            data = source.read(member)
            if len(data) != member.file_size:
                raise BuildInputError("wheel_member_actual_size_mismatch")
            expected_hash, expected_size = records[name]
            digest = _digest(data)
            if name != record_name and (expected_size != len(data) or expected_hash != digest):
                raise BuildInputError("wheel_record_hash_or_size_mismatch")
            target = rootfs / relative
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
            target.chmod(0o444)
            inventory[relative] = {"kind": "file", "bytes": len(data), "sha256": digest,
                                   "sha1": hashlib.sha1(data).hexdigest(), "mode": "0444",
                                   "package": wheel.name}


def _write_generated(rootfs: Path, relative: str, data: bytes, mode: int,
                     inventory: dict[str, dict[str, object]]) -> None:
    target = rootfs / relative
    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(data)
    target.chmod(mode)
    inventory[relative] = {"kind": "file", "bytes": len(data), "sha256": _digest(data),
                           "sha1": hashlib.sha1(data).hexdigest(), "mode": f"{mode:04o}",
                           "package": "generated-profile"}


def _spdx(lock: BuildLock, inventory: dict[str, dict[str, object]]) -> dict[str, object]:
    inventory_digest = _digest(_canonical(inventory))
    created = dt.datetime.fromtimestamp(SOURCE_DATE_EPOCH, dt.timezone.utc).isoformat().replace("+00:00", "Z")
    packages = []
    files = []
    relationships = []
    for wheel in lock.wheels:
        package_id = "SPDXRef-Package-" + wheel.name
        packages.append({"SPDXID": package_id, "name": wheel.name, "versionInfo": wheel.version,
                         "downloadLocation": wheel.url, "filesAnalyzed": False,
                         "checksums": [{"algorithm": "SHA256", "checksumValue": wheel.sha256}],
                         "licenseConcluded": "NOASSERTION", "licenseDeclared": "NOASSERTION",
                         "copyrightText": "NOASSERTION",
                         "comment": "Frozen wheel input. Installed files are independently inventoried; base and host tools are outside this package inventory."})
        relationships.append({"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": package_id})
    for index, (path, item) in enumerate(sorted(inventory.items())):
        file_id = "SPDXRef-File-" + str(index)
        if item["kind"] != "file":
            continue
        files.append({"SPDXID": file_id, "fileName": "./" + path,
                      "checksums": [{"algorithm": "SHA256", "checksumValue": item["sha256"]},
                                    {"algorithm": "SHA1", "checksumValue": item["sha1"]}],
                      "licenseConcluded": "NOASSERTION", "copyrightText": "NOASSERTION"})
        if item["package"] != "generated-profile":
            relationships.append({"spdxElementId": "SPDXRef-Package-" + str(item["package"]), "relationshipType": "CONTAINS", "relatedSpdxElement": file_id})
    return {"spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
            "name": PROFILE + "-installed-overlay", "documentNamespace": "https://justicedao.org/spdx/ordinary-worker/" + inventory_digest,
            "creationInfo": {"created": created, "creators": ["Tool: stage_ordinary_worker_context.py@1"]},
            "documentComment": "Overlay inventory only. Ubuntu base ID: " + lock.base_image_id + ". Trusted host /usr and CA mount closures require separate runtime qualification. No protected-route authority or provider execution is certified.",
            "packages": packages, "files": files, "relationships": relationships}


def _context_tar(context: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="x", format=tarfile.USTAR_FORMAT) as archive:
        for source in sorted(context.rglob("*"), key=lambda p: p.relative_to(context).as_posix()):
            relative = source.relative_to(context).as_posix()
            info = tarfile.TarInfo(relative + ("/" if source.is_dir() else ""))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = SOURCE_DATE_EPOCH
            if source.is_symlink():
                if relative != "rootfs/opt/ipfs-task-tools/bin/python3" or os.readlink(source) != "python":
                    raise BuildInputError("generated_context_symlink_not_admitted")
                info.type = tarfile.SYMTYPE
                info.mode = 0o777
                info.linkname = "python"
                archive.addfile(info)
            elif source.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
            elif source.is_file():
                info.mode = stat.S_IMODE(source.stat().st_mode)
                info.size = source.stat().st_size
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                raise BuildInputError("nonregular_context_member")
    destination.chmod(0o444)


def stage_worker_context(*, output: Path, lock_path: Path, containerfile: Path) -> dict[str, object]:
    """Download/extract frozen inputs in one new caller-owned TMPDIR subtree."""
    lock = load_lock(lock_path)
    temporary_raw = os.environ.get("TMPDIR", "")
    temporary = Path(temporary_raw)
    if (not temporary_raw or not temporary.is_absolute() or temporary.is_symlink()
            or not temporary.is_dir() or temporary.resolve(strict=True) != temporary
            or temporary.stat().st_uid != os.getuid()):
        raise BuildInputError("canonical_owned_native_tmpdir_required")
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.parent.is_dir() or output.parent.resolve(strict=True) != output.parent
            or not output.is_relative_to(temporary) or output == temporary
            or any(part in {".", ".."} for part in output.parts)):
        raise BuildInputError("new_output_below_native_tmpdir_required")
    if containerfile.is_symlink() or not containerfile.is_file() or containerfile.stat().st_size > MAX_LOCK_BYTES:
        raise BuildInputError("invalid_containerfile")
    recipe = containerfile.read_bytes()
    if (_digest(recipe) != CONTAINERFILE_SHA256
            or b"org.ipfs-accelerate.ordinary-worker=\"" + LABEL.encode() + b"\"" not in recipe
            or b"74c4a6ff" in recipe or b"authority-validation=" in recipe
            or any(line.lstrip().startswith((b"RUN ", b"VOLUME ")) for line in recipe.splitlines())):
        raise BuildInputError("containerfile_contract_mismatch")
    output.mkdir(mode=0o700)
    downloads, context = output / "wheels", output / "context"
    downloads.mkdir(mode=0o700)
    context.mkdir(mode=0o700)
    rootfs = context / "rootfs"
    rootfs.mkdir(mode=0o755)
    (context / "Containerfile").write_bytes(recipe)
    (context / "Containerfile").chmod(0o444)
    inventory: dict[str, dict[str, object]] = {}
    budget = [0, 0]
    for wheel in lock.wheels:
        archive = downloads / wheel.filename
        _download(wheel, archive)
        _extract(wheel, archive, rootfs, inventory, budget)
    _write_generated(rootfs, "opt/ipfs-task-tools/bin/python", PLACEHOLDER, 0o555, inventory)
    # The genuine staging owner forbids physical symlinks. A regular sealed
    # launcher forwards python3 to the explicitly mounted trusted interpreter.
    _write_generated(rootfs, "opt/ipfs-task-tools/bin/python3",
                     b'#!/bin/sh\nexec /opt/ipfs-task-tools/bin/python "$@"\n',
                     0o555, inventory)
    for directory in ("workspace", "opt/codex-home"):
        (rootfs / directory).mkdir(mode=0o755, parents=True, exist_ok=True)
    profile = {"schema": "legal-ir-ordinary-worker-profile@1", "profile": PROFILE,
               "label": LABEL, "base_image_id": lock.base_image_id, "wheel_lock_sha256": lock.sha256,
               "architecture": "arm64", "python": "cp312", "host_mounts_required": [
                   "/usr:read-only", "/etc/ssl/certs:read-only",
                   "/usr/bin/python3.12:/opt/ipfs-task-tools/bin/python:read-only"],
               "provider_binaries": "explicit trusted host mounts; not embedded",
               "protected_route_authority": False, "runtime_qualified": False}
    _write_generated(rootfs, "opt/ipfs-ordinary-worker/profile.json", _canonical(profile), 0o444, inventory)
    installed = {"schema": "legal-ir-ordinary-worker-installed-files@1", "profile": PROFILE,
                 "base_image_id": lock.base_image_id, "wheel_lock_sha256": lock.sha256,
                 "scope": "complete installed overlay file/symlink inventory; base and host mounts excluded",
                 "files": dict(sorted(inventory.items()))}
    (output / "installed-files.json").write_bytes(_canonical(installed))
    (output / "sbom.spdx.json").write_bytes(_canonical(_spdx(lock, inventory)))
    _context_tar(context, output / "build-context.tar")
    context_sha = hashlib.sha256()
    with (output / "build-context.tar").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            context_sha.update(block)
    report = {"schema": "legal-ir-ordinary-worker-staging-receipt@1", "status": "staged_unqualified",
              "profile": PROFILE, "base_image_id": lock.base_image_id, "wheel_lock_sha256": lock.sha256,
              "containerfile_sha256": _digest(recipe), "build_context_sha256": context_sha.hexdigest(),
              "build_context_bytes": (output / "build-context.tar").stat().st_size,
              "installed_manifest_sha256": _digest((output / "installed-files.json").read_bytes()),
              "sbom_sha256": _digest((output / "sbom.spdx.json").read_bytes()),
              "wheel_count": len(lock.wheels), "download_bytes": sum(w.size for w in lock.wheels),
              "wheel_installed_bytes": budget[0], "installed_file_and_symlink_count": len(inventory),
              "docker_invoked": False, "wheel_code_executed": False,
              "provider_invoked": False, "runtime_qualified": False, "protected_route_authority": False}
    (output / "staging-receipt.json").write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("wheels.lock.json"))
    parser.add_argument("--containerfile", type=Path,
                        default=Path(__file__).with_name("ordinary-task-toolchain.Containerfile"))
    args = parser.parse_args()
    try:
        report = stage_worker_context(output=args.output, lock_path=args.lock, containerfile=args.containerfile)
    except BuildInputError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, UnicodeError,
            json.JSONDecodeError, csv.Error, ValueError):
        # Network diagnostics and arbitrary archive strings never enter logs.
        print('{"reason":"bounded_io_or_parse_failure","status":"failed"}', file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
