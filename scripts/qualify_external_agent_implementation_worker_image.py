#!/usr/bin/env python3
"""Offline qualification for the closed EAAEF implementation-worker image.

This command copies two explicitly named, local, static native executables into
a script-owned temporary Docker build context.  It neither reads provider
authentication nor invokes a provider.  Its output is an unsigned candidate
report with zero worker capacity; independent network/provider authorization
and the existing admission path remain mandatory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

REPORT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/eaaef-implementation-worker-image-candidate@1"
INPUT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/eaaef-implementation-worker-image-inputs@1"
EXPECTED_BASE_REFERENCE = "ipfs-accelerate-authority-validation:20260803-v2"
EXPECTED_BASE_IMAGE_ID = "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
EXPECTED_ARCHITECTURE = "arm64"
NONROOT_USER = "65532:65532"
MAXIMUM_COMMAND_OUTPUT_BYTES = 128 * 1024
MAXIMUM_REPORT_BYTES = 1024 * 1024
MAXIMUM_SBOM_BYTES = 1024 * 1024
SBOM_CANONICALIZATION = "sorted-key-compact-utf8-json-with-one-lf"
MAXIMUM_EXPORT_ENTRIES = 2_000_000
MAXIMUM_EXPORT_LOGICAL_BYTES = 32 * 1024**3
MAXIMUM_CREDENTIAL_FINDINGS = 128
MAXIMUM_CREDENTIAL_SCAN_SECONDS = 600
CONTAINERFILE = Path("containers/external-agent/implementation-worker.Containerfile")
RUNTIME_ENVIRONMENT = {
    "BASH_ENV": "",
    "CODEX_HOME": "/opt/codex-home",
    "ENV": "",
    "HOME": "/opt/codex-home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/opt/eaaef/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TERM": "dumb",
}
REQUIRED_BASE_TOOLS = {
    "git": ("/usr/bin/git", "--version"),
    "python": ("/usr/bin/python3", "--version"),
    "rg": ("/usr/local/bin/rg", "--version"),
}
SENSITIVE_ENV_FRAGMENTS = (
    "API_KEY",
    "AUTH_TOKEN",
    "BEARER",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
)
CREDENTIAL_PATH_BASENAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "application_default_credentials.json",
        "auth.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
)
CREDENTIAL_CONTENT_PATTERNS = (
    (
        "private_key_pem",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("aws_access_key_id", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("provider_secret_token", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,255}\b")),
    (
        "assigned_secret_value",
        re.compile(
            rb"(?i)(?:api[_-]?key|auth[_-]?token|client[_-]?secret|private[_-]?key)"
            rb"[ \t]{0,8}[:=][ \t]{0,8}['\"]?[A-Za-z0-9_./+-]{20,255}"
        ),
    ),
)
KNOWN_CREDENTIAL_PATHS = (
    "/opt/codex-home/auth.json",
    "/root/.codex/auth.json",
    "/root/.config/grok",
    "/root/.aws/credentials",
    "/run/secrets",
)


class QualificationError(ValueError):
    """A deterministic offline qualification failure."""


@dataclass(frozen=True)
class SourceBinding:
    name: str
    absolute_path: str
    uid: int
    gid: int
    mode: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    version: str
    sha256: str
    elf_machine: str
    static: bool


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _cid(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _bounded_text(result: subprocess.CompletedProcess[str]) -> None:
    for value in (result.stdout, result.stderr):
        if len(value.encode("utf-8", errors="replace")) > (MAXIMUM_COMMAND_OUTPUT_BYTES):
            raise QualificationError("command output exceeded its bound")


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    # Never let the container client consult the operator's credential-bearing
    # Docker config.  This empty directory is owned and removed by this call.
    with tempfile.TemporaryDirectory(prefix="eaaef-empty-docker-config-") as config:
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env={
                "DOCKER_BUILDKIT": "1",
                "DOCKER_CONFIG": config,
                "HOME": "/nonexistent-eaaef-home",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
        )
    _bounded_text(result)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise QualificationError(f"command failed ({result.returncode}): {detail}")
    return result


def _docker_json(
    docker: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
) -> Any:
    result = _run([docker, *arguments], cwd=cwd)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError("container runtime returned invalid JSON") from exc


def _image_id(docker: str, reference: str, *, cwd: Path) -> str:
    return _run(
        [docker, "image", "inspect", reference, "--format", "{{.Id}}"],
        cwd=cwd,
    ).stdout.strip()


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _source_stat(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise QualificationError("native binary path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"native binary is unavailable: {path}") from exc
    if resolved != path or path.is_symlink():
        raise QualificationError("native binary path must be canonical, not a symlink")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise QualificationError("native binary must be a singly linked regular file")
    if info.st_mode & 0o022:
        raise QualificationError("native binary must not be group/world writable")
    if not info.st_mode & stat.S_IXUSR:
        raise QualificationError("native binary must be owner-executable")
    return info


def _elf_static_arm64(fd: int) -> None:
    header = os.pread(fd, 64, 0)
    if len(header) != 64 or header[:4] != b"\x7fELF":
        raise QualificationError("native binary is not ELF")
    if header[4:6] != b"\x02\x01":
        raise QualificationError("native binary is not 64-bit little-endian ELF")
    machine = struct.unpack_from("<H", header, 18)[0]
    if machine != 183:
        raise QualificationError("native binary is not Linux arm64")
    program_offset = struct.unpack_from("<Q", header, 32)[0]
    entry_size = struct.unpack_from("<H", header, 54)[0]
    entry_count = struct.unpack_from("<H", header, 56)[0]
    if entry_size < 56 or not 1 <= entry_count <= 512:
        raise QualificationError("native binary has invalid ELF program headers")
    for index in range(entry_count):
        entry = os.pread(fd, entry_size, program_offset + index * entry_size)
        if len(entry) != entry_size:
            raise QualificationError("native binary ELF program headers are truncated")
        program_type = struct.unpack_from("<I", entry, 0)[0]
        if program_type in {2, 3}:  # PT_DYNAMIC or PT_INTERP
            raise QualificationError("native binary is dynamically linked")


def _snapshot(path: Path) -> tuple[os.stat_result, str]:
    expected = _source_stat(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
        ) != (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
            expected.st_mtime_ns,
            expected.st_mode,
            expected.st_uid,
            expected.st_gid,
        ):
            raise QualificationError("native binary identity drifted while opening")
        _elf_static_arm64(fd)
        digest = _hash_fd(fd)
        if os.fstat(fd) != opened:
            raise QualificationError("native binary drifted while hashing")
    finally:
        os.close(fd)
    return expected, digest


def _version(path: Path, *, repo_root: Path) -> str:
    result = _run([str(path), "--version"], cwd=repo_root, timeout=30)
    version = result.stdout.strip() or result.stderr.strip()
    if not version or "\n" in version or len(version.encode()) > 4096:
        raise QualificationError("native binary returned an invalid version")
    return version


def _bind_source(name: str, path: Path, *, repo_root: Path) -> SourceBinding:
    before, before_hash = _snapshot(path)
    version = _version(path, repo_root=repo_root)
    after, after_hash = _snapshot(path)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before_hash,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after_hash,
    )
    if before_identity != after_identity:
        raise QualificationError("native binary drifted during version binding")
    return SourceBinding(
        name=name,
        absolute_path=str(path),
        uid=before.st_uid,
        gid=before.st_gid,
        mode=f"{stat.S_IMODE(before.st_mode):04o}",
        size=before.st_size,
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
        version=version,
        sha256=before_hash,
        elf_machine="aarch64",
        static=True,
    )


def _binding_identity(binding: SourceBinding) -> tuple[object, ...]:
    return (
        binding.absolute_path,
        binding.uid,
        binding.gid,
        binding.mode,
        binding.size,
        binding.device,
        binding.inode,
        binding.mtime_ns,
        binding.sha256,
    )


def _opened_identity(path: Path, info: os.stat_result, digest: str) -> tuple[object, ...]:
    return (
        str(path),
        info.st_uid,
        info.st_gid,
        f"{stat.S_IMODE(info.st_mode):04o}",
        info.st_size,
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        digest,
    )


def _stage_binary(
    binding: SourceBinding,
    destination: Path,
    *,
    normalized_mtime_ns: int | None = None,
) -> None:
    source = Path(binding.absolute_path)
    current, digest = _snapshot(source)
    observed = _opened_identity(source, current, digest)
    if observed != _binding_identity(binding):
        raise QualificationError(f"{binding.name} source/hash drift before staging")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    try:
        opened = os.fstat(source_fd)
        if _opened_identity(source, opened, binding.sha256) != _binding_identity(binding):
            raise QualificationError(f"{binding.name} source identity drift during staging")
        with destination.open("xb") as output:
            copied = hashlib.sha256()
            while True:
                block = os.read(source_fd, 1024 * 1024)
                if not block:
                    break
                output.write(block)
                copied.update(block)
            output.flush()
            os.fsync(output.fileno())
        if copied.hexdigest() != binding.sha256:
            raise QualificationError(f"{binding.name} hash drift during staging")
        if _opened_identity(source, os.fstat(source_fd), copied.hexdigest()) != (
            _binding_identity(binding)
        ):
            raise QualificationError(f"{binding.name} source identity drift during staging")
        staged = destination.stat(follow_symlinks=False)
        if staged.st_size != binding.size:
            raise QualificationError(f"{binding.name} size drift during staging")
        os.chmod(destination, 0o555)
        if normalized_mtime_ns is not None:
            os.utime(
                destination,
                ns=(normalized_mtime_ns, normalized_mtime_ns),
                follow_symlinks=False,
            )
    finally:
        os.close(source_fd)
    final, final_hash = _snapshot(source)
    final_observed = _opened_identity(source, final, final_hash)
    if final_observed != _binding_identity(binding):
        raise QualificationError(f"{binding.name} source/hash drift after staging")


def _canonical_rootfs_tar(
    destination: Path,
    *,
    staged_bindings: Sequence[tuple[SourceBinding, Path]],
    manifest: bytes,
    source_date_epoch: int,
) -> tuple[str, int]:
    expected_files = {
        "opt/eaaef/bin/codex": (staged_bindings[0][1], staged_bindings[0][0], 0o555),
        "opt/eaaef/bin/grok": (staged_bindings[1][1], staged_bindings[1][0], 0o555),
    }

    def member(
        name: str,
        *,
        mode: int,
        size: int = 0,
        uid: int = 65532,
        gid: int = 65532,
    ) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.uid = uid
        info.gid = gid
        info.uname = ""
        info.gname = ""
        info.mode = mode
        info.mtime = source_date_epoch
        info.size = size
        return info

    with destination.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for directory, uid, gid, mode in (
                (".", 0, 0, 0o755),
                ("opt", 0, 0, 0o755),
                ("opt/eaaef", 65532, 65532, 0o555),
                ("opt/eaaef/bin", 65532, 65532, 0o555),
            ):
                info = member(directory, mode=mode, uid=uid, gid=gid)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            for name, (path, binding, mode) in expected_files.items():
                info = member(name, mode=mode, size=binding.size)
                with path.open("rb") as payload:
                    archive.addfile(info, payload)
            manifest_info = member(
                "opt/eaaef/candidate-inputs.json",
                mode=0o444,
                size=len(manifest),
            )
            archive.addfile(manifest_info, io.BytesIO(manifest))
        output.flush()
        os.fsync(output.fileno())
    os.chmod(destination, 0o444)
    normalized_mtime_ns = source_date_epoch * 1_000_000_000
    os.utime(
        destination,
        ns=(normalized_mtime_ns, normalized_mtime_ns),
        follow_symlinks=False,
    )

    expected = {
        ".": (tarfile.DIRTYPE, 0o755, 0, "", 0, 0),
        "opt": (tarfile.DIRTYPE, 0o755, 0, "", 0, 0),
        "opt/eaaef": (tarfile.DIRTYPE, 0o555, 0, "", 65532, 65532),
        "opt/eaaef/bin": (tarfile.DIRTYPE, 0o555, 0, "", 65532, 65532),
        "opt/eaaef/bin/codex": (
            tarfile.REGTYPE,
            0o555,
            staged_bindings[0][0].size,
            staged_bindings[0][0].sha256,
            65532,
            65532,
        ),
        "opt/eaaef/bin/grok": (
            tarfile.REGTYPE,
            0o555,
            staged_bindings[1][0].size,
            staged_bindings[1][0].sha256,
            65532,
            65532,
        ),
        "opt/eaaef/candidate-inputs.json": (
            tarfile.REGTYPE,
            0o444,
            len(manifest),
            hashlib.sha256(manifest).hexdigest(),
            65532,
            65532,
        ),
    }
    with tarfile.open(destination, mode="r:") as archive:
        members = archive.getmembers()
        if [item.name for item in members] != list(expected):
            raise QualificationError("canonical rootfs tar member set/order drifted")
        for info in members:
            (
                expected_type,
                expected_mode,
                expected_size,
                expected_hash,
                expected_uid,
                expected_gid,
            ) = expected[info.name]
            if (
                info.type != expected_type
                or info.uid != expected_uid
                or info.gid != expected_gid
                or info.uname != ""
                or info.gname != ""
                or info.mode != expected_mode
                or info.mtime != source_date_epoch
                or info.size != expected_size
                or info.issym()
                or info.islnk()
            ):
                raise QualificationError("canonical rootfs tar metadata drifted")
            if expected_hash:
                payload = archive.extractfile(info)
                if payload is None:
                    raise QualificationError("canonical rootfs tar member is unreadable")
                digest = hashlib.sha256()
                while block := payload.read(1024 * 1024):
                    digest.update(block)
                if digest.hexdigest() != expected_hash:
                    raise QualificationError("canonical rootfs tar content drifted")
    digest = hashlib.sha256()
    with destination.open("rb") as payload:
        for block in iter(lambda: payload.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), destination.stat().st_size


def _input_manifest(bindings: Sequence[SourceBinding], *, source_date_epoch: int) -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "source_date_epoch": source_date_epoch,
        "base_image": {
            "reference": EXPECTED_BASE_REFERENCE,
            "image_id": EXPECTED_BASE_IMAGE_ID,
        },
        "native_binaries": [asdict(binding) for binding in bindings],
        "base_tools_policy": {
            "install_or_copy_from_host": False,
            "required_exact_paths": {
                name: values[0] for name, values in REQUIRED_BASE_TOOLS.items()
            },
        },
        "authority": {
            "signed": False,
            "network_authorized": False,
            "provider_authorized": False,
            "worker_capacity": 0,
        },
    }


def _build(
    *,
    docker: str,
    repo_root: Path,
    image_tag: str,
    bindings: Sequence[SourceBinding],
    source_date_epoch: int,
) -> dict[str, Any]:
    containerfile = (repo_root / CONTAINERFILE).resolve(strict=True)
    if containerfile.is_symlink() or containerfile.parent.parent.parent != repo_root:
        raise QualificationError("implementation-worker Containerfile path drifted")
    before = _image_id(docker, EXPECTED_BASE_REFERENCE, cwd=repo_root)
    if before != EXPECTED_BASE_IMAGE_ID:
        raise QualificationError("local base image identity drifted before build")
    manifest = _canonical(_input_manifest(bindings, source_date_epoch=source_date_epoch)) + b"\n"
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    build_result: subprocess.CompletedProcess[str] | None = None
    after = ""
    with (
        tempfile.TemporaryDirectory(prefix="eaaef-worker-build-") as raw_context,
        tempfile.TemporaryDirectory(prefix="eaaef-worker-stage-") as raw_staging,
    ):
        context = Path(raw_context)
        staging = Path(raw_staging)
        normalized_mtime_ns = source_date_epoch * 1_000_000_000
        _stage_binary(
            bindings[0],
            staging / "codex",
            normalized_mtime_ns=normalized_mtime_ns,
        )
        _stage_binary(
            bindings[1],
            staging / "grok",
            normalized_mtime_ns=normalized_mtime_ns,
        )
        rootfs_tar = context / "worker-rootfs.tar"
        rootfs_tar_hash, rootfs_tar_bytes = _canonical_rootfs_tar(
            rootfs_tar,
            staged_bindings=(
                (bindings[0], staging / "codex"),
                (bindings[1], staging / "grok"),
            ),
            manifest=manifest,
            source_date_epoch=source_date_epoch,
        )
        os.utime(
            context,
            ns=(normalized_mtime_ns, normalized_mtime_ns),
            follow_symlinks=False,
        )
        if {item.name for item in context.iterdir()} != {"worker-rootfs.tar"}:
            raise QualificationError("temporary build context has unexpected inputs")
        command = [
            docker,
            "build",
            "--no-cache",
            "--pull=false",
            "--network=none",
            "--provenance=false",
            "--sbom=false",
            "--output",
            "type=docker,rewrite-timestamp=true",
            "--build-arg",
            f"BASE_IMAGE={EXPECTED_BASE_REFERENCE}",
            "--build-arg",
            f"CODEX_SHA256={bindings[0].sha256}",
            "--build-arg",
            f"GROK_SHA256={bindings[1].sha256}",
            "--build-arg",
            f"INPUT_MANIFEST_SHA256={manifest_hash}",
            "--build-arg",
            f"ROOTFS_TAR_SHA256={rootfs_tar_hash}",
            "--build-arg",
            f"SOURCE_DATE_EPOCH={source_date_epoch}",
            "-f",
            str(containerfile),
            "-t",
            image_tag,
            str(context),
        ]
        image_ids: list[str] = []
        output_hashes: list[str] = []
        try:
            for _attempt in range(2):
                # Sending the first context may advance access times under a
                # relatime filesystem.  Re-normalize every context entry so
                # the second clean build receives byte-for-byte equivalent
                # metadata as well as equivalent file content.
                for staged_path in (
                    rootfs_tar,
                    context,
                ):
                    os.utime(
                        staged_path,
                        ns=(normalized_mtime_ns, normalized_mtime_ns),
                        follow_symlinks=False,
                    )
                build_result = _run(command, cwd=repo_root, timeout=900)
                image_ids.append(_image_id(docker, image_tag, cwd=repo_root))
                output_hashes.append(
                    "sha256:" + hashlib.sha256(build_result.stdout.encode()).hexdigest()
                )
        finally:
            after = _image_id(docker, EXPECTED_BASE_REFERENCE, cwd=repo_root)
            if after != before:
                raise QualificationError("local base image identity changed during build")
    assert build_result is not None
    image_id = _image_id(docker, image_tag, cwd=repo_root)
    reproducible = len(image_ids) == 2 and image_ids[0] == image_ids[1] == image_id
    return {
        "attempted": True,
        "succeeded": True,
        "base_image_reference": EXPECTED_BASE_REFERENCE,
        "base_image_id_before": before,
        "base_image_id_after": after,
        "image_id": image_id,
        "clean_build_image_ids": image_ids,
        "clean_build_attempts": len(image_ids),
        "clean_build_reproducible": reproducible,
        "input_manifest_sha256": "sha256:" + manifest_hash,
        "rootfs_tar_sha256": "sha256:" + rootfs_tar_hash,
        "rootfs_tar_bytes": rootfs_tar_bytes,
        "context_inputs": ["worker-rootfs.tar"],
        "context_retained": False,
        "network": "none",
        "pull": False,
        "cache": False,
        "provenance": False,
        "embedded_buildkit_sbom": False,
        "stdout_sha256": output_hashes,
    }


def _probe_program() -> str:
    tools = {
        **{name: [path, option] for name, (path, option) in REQUIRED_BASE_TOOLS.items()},
        "codex": ["/opt/eaaef/bin/codex", "--version"],
        "grok": ["/opt/eaaef/bin/grok", "--version"],
    }
    credential_paths = list(KNOWN_CREDENTIAL_PATHS)
    return (
        "import hashlib,json,os,pathlib,stat,subprocess;"
        f"tools={tools!r};credential_paths={credential_paths!r};"
        "versions={};"
        "\nfor name,argv in tools.items():"
        "\n try:"
        "\n  p=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=20);versions[name]={'path':argv[0],'returncode':p.returncode,'version':p.stdout.strip()}"
        "\n except (FileNotFoundError,subprocess.TimeoutExpired) as exc: versions[name]={'path':argv[0],'returncode':127,'version':'','error':type(exc).__name__}"
        "\ndef sha(path):"
        "\n h=hashlib.sha256()"
        "\n with open(path,'rb') as stream:"
        "\n  for block in iter(lambda:stream.read(1048576),b''): h.update(block)"
        "\n return h.hexdigest()"
        "\ndef metadata(path):"
        "\n s=os.stat(path,follow_symlinks=False)"
        "\n return {'gid':s.st_gid,'links':s.st_nlink,'mode':format(stat.S_IMODE(s.st_mode),'04o'),'regular':stat.S_ISREG(s.st_mode),'size':s.st_size,'uid':s.st_uid}"
        "\ndef readable(path):"
        "\n try:"
        "\n  with open(path,'rb'): return True"
        "\n except (IsADirectoryError,FileNotFoundError,PermissionError,OSError): return False"
        "\ndenied=False"
        "\ntry: pathlib.Path('/eaaef-root-write-probe').write_text('x')"
        "\nexcept OSError: denied=True"
        "\nprint(json.dumps({'credential_paths_readable':[p for p in credential_paths if readable(p)],'docker_socket_present':pathlib.Path('/var/run/docker.sock').exists(),'environment':dict(os.environ),'file_metadata':{'candidate-inputs':metadata('/opt/eaaef/candidate-inputs.json'),'codex':metadata('/opt/eaaef/bin/codex'),'grok':metadata('/opt/eaaef/bin/grok')},'gid':os.getgid(),'hashes':{'candidate-inputs':sha('/opt/eaaef/candidate-inputs.json'),'codex':sha('/opt/eaaef/bin/codex'),'grok':sha('/opt/eaaef/bin/grok')},'root_write_denied':denied,'tools':versions,'uid':os.getuid()},sort_keys=True,separators=(',',':')))"
    )


def _probe(
    *, docker: str, repo_root: Path, image_tag: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    create = [
        docker,
        "create",
        "--read-only",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--cpus",
        "2",
        "--memory",
        str(4 * 1024**3),
        "--memory-swap",
        str(4 * 1024**3),
        "--user",
        NONROOT_USER,
        image_tag,
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        "-c",
        _probe_program(),
    ]
    container_id = _run(create, cwd=repo_root).stdout.strip()
    if not container_id:
        raise QualificationError("container runtime returned no container ID")
    try:
        inspected = _docker_json(docker, ["container", "inspect", container_id], cwd=repo_root)
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise QualificationError("container inspection is invalid")
        started = _run([docker, "start", "-a", container_id], cwd=repo_root)
        try:
            observed = json.loads(started.stdout)
        except json.JSONDecodeError as exc:
            raise QualificationError("diagnostic probe output is invalid") from exc
        if not isinstance(observed, dict):
            raise QualificationError("diagnostic probe output is not an object")
        return inspected[0], observed
    finally:
        _run([docker, "rm", "-f", container_id], cwd=repo_root, check=False)


def _credential_path_detectors(path: str) -> set[str]:
    normalized = path.lstrip("./").lower()
    basename = normalized.rsplit("/", 1)[-1]
    detectors: set[str] = set()
    if basename in CREDENTIAL_PATH_BASENAMES:
        detectors.add("credential_filename")
    if basename == ".env" or (
        basename.startswith(".env.") and not basename.endswith((".example", ".sample", ".template"))
    ):
        detectors.add("environment_secret_filename")
    if normalized.startswith("etc/ssl/private/") and not normalized.endswith("/.gitkeep"):
        detectors.add("private_tls_material_path")
    if "/.ssh/" in "/" + normalized and basename not in {
        "authorized_keys",
        "config",
        "known_hosts",
    }:
        detectors.add("ssh_secret_material_path")
    return detectors


def _scan_export_tar(
    stream: BinaryIO,
    *,
    deadline: float,
) -> dict[str, Any]:
    """Scan a merged Docker export without extracting or retaining its bytes."""

    entries = 0
    regular_files = 0
    logical_bytes = 0
    scanned_content_bytes = 0
    finding_signals = 0
    finding_files = 0
    findings: list[dict[str, Any]] = []
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if time.monotonic() > deadline:
                raise QualificationError("credential scan exceeded its time bound")
            entries += 1
            if entries > MAXIMUM_EXPORT_ENTRIES:
                raise QualificationError("credential scan exceeded its entry bound")
            logical_bytes += max(0, member.size)
            if logical_bytes > MAXIMUM_EXPORT_LOGICAL_BYTES:
                raise QualificationError("credential scan exceeded its logical-byte bound")
            safe_path = member.name.encode("utf-8", errors="replace").decode(
                "utf-8", errors="strict"
            )[:4096]
            detectors = _credential_path_detectors(safe_path)
            digest = ""
            if member.isfile():
                regular_files += 1
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise QualificationError("credential scan could not read a regular file")
                file_digest = hashlib.sha256()
                overlap = b""
                while True:
                    if time.monotonic() > deadline:
                        raise QualificationError("credential scan exceeded its time bound")
                    chunk = extracted.read(1024 * 1024)
                    if not chunk:
                        break
                    scanned_content_bytes += len(chunk)
                    file_digest.update(chunk)
                    window = overlap + chunk
                    lowered = window.lower()
                    prefilters = (
                        b"-----begin " in lowered,
                        b"akia" in lowered,
                        any(
                            prefix in window
                            for prefix in (b"ghp_", b"gho_", b"ghu_", b"ghs_", b"ghr_")
                        ),
                        b"sk-" in lowered,
                        any(
                            marker in lowered
                            for marker in (
                                b"api_key",
                                b"api-key",
                                b"auth_token",
                                b"auth-token",
                                b"client_secret",
                                b"client-secret",
                                b"private_key",
                                b"private-key",
                            )
                        ),
                    )
                    for enabled, (detector, pattern) in zip(
                        prefilters,
                        CREDENTIAL_CONTENT_PATTERNS,
                        strict=True,
                    ):
                        if enabled and pattern.search(window):
                            detectors.add(detector)
                    overlap = window[-512:]
                digest = "sha256:" + file_digest.hexdigest()
            if detectors:
                finding_files += 1
                finding_signals += len(detectors)
                if len(findings) < MAXIMUM_CREDENTIAL_FINDINGS:
                    findings.append(
                        {
                            "path": safe_path,
                            "detectors": sorted(detectors),
                            "sha256": digest,
                            "size": member.size,
                        }
                    )
            # TarFile retains every member even in stream mode; the scanner
            # does not extract links or need prior entries, so drop metadata
            # already accounted for to keep memory bounded on large images.
            archive.members.clear()
    findings.sort(key=lambda item: (str(item["path"]), list(item["detectors"])))
    return {
        "schema": ("ipfs_accelerate_py/agent-supervisor/eaaef-bounded-image-credential-scan@1"),
        "mode": "merged-docker-export-tar-stream",
        "complete": True,
        "stop_reason": "",
        "entries": entries,
        "regular_files": regular_files,
        "logical_bytes": logical_bytes,
        "scanned_content_bytes": scanned_content_bytes,
        "path_scope": "every exported filesystem entry",
        "content_scope": "every byte of every exported regular file",
        "findings": findings,
        "finding_signals": finding_signals,
        "finding_files": finding_files,
        "findings_truncated": finding_files > len(findings),
        "raw_secret_values_recorded": False,
        "limits": {
            "maximum_entries": MAXIMUM_EXPORT_ENTRIES,
            "maximum_logical_bytes": MAXIMUM_EXPORT_LOGICAL_BYTES,
            "maximum_seconds": MAXIMUM_CREDENTIAL_SCAN_SECONDS,
            "maximum_recorded_findings": MAXIMUM_CREDENTIAL_FINDINGS,
        },
    }


def _scan_exported_filesystem(
    *,
    docker: str,
    repo_root: Path,
    image_tag: str,
) -> dict[str, Any]:
    """Create a stopped, unmounted container and stream its merged filesystem."""

    container_id = _run(
        [
            docker,
            "create",
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "32",
            "--memory",
            str(256 * 1024**2),
            "--memory-swap",
            str(256 * 1024**2),
            "--user",
            NONROOT_USER,
            image_tag,
        ],
        cwd=repo_root,
    ).stdout.strip()
    if not container_id:
        raise QualificationError("credential-scan container identity is empty")
    process: subprocess.Popen[bytes] | None = None
    try:
        inspected = _docker_json(
            docker,
            ["container", "inspect", container_id],
            cwd=repo_root,
        )
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise QualificationError("credential-scan container inspection is invalid")
        container = inspected[0]
        host = container.get("HostConfig") or {}
        if (
            container.get("State", {}).get("Running") is not False
            or container.get("Mounts")
            or host.get("Binds")
            or host.get("Devices")
            or host.get("DeviceRequests")
            or host.get("NetworkMode") != "none"
        ):
            raise QualificationError("credential-scan container is not inert and unmounted")
        with (
            tempfile.TemporaryDirectory(prefix="eaaef-empty-docker-config-") as config,
            tempfile.TemporaryFile() as error_stream,
        ):
            process = subprocess.Popen(
                [docker, "export", container_id],
                cwd=repo_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=error_stream,
                env={
                    "DOCKER_CONFIG": config,
                    "HOME": "/nonexistent-eaaef-home",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                },
            )
            if process.stdout is None:
                raise QualificationError("container export returned no byte stream")
            try:
                scan = _scan_export_tar(
                    process.stdout,
                    deadline=time.monotonic() + MAXIMUM_CREDENTIAL_SCAN_SECONDS,
                )
                process.stdout.close()
                return_code = process.wait(timeout=30)
            except Exception:
                process.kill()
                process.wait(timeout=30)
                raise
            if return_code != 0:
                error_stream.seek(0)
                detail = error_stream.read(MAXIMUM_COMMAND_OUTPUT_BYTES).decode(
                    "utf-8", errors="replace"
                )
                raise QualificationError("container export failed: " + detail.strip()[-4000:])
            scan["container_started"] = False
            scan["mounts_present"] = False
            scan["device_requests_present"] = False
            return scan
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        _run([docker, "rm", "-f", container_id], cwd=repo_root, check=False)


def _sensitive_environment(values: object) -> list[str]:
    if not isinstance(values, list):
        return ["invalid_image_environment"]
    findings: list[str] = []
    for item in values:
        key = str(item).partition("=")[0].upper()
        if any(fragment in key for fragment in SENSITIVE_ENV_FRAGMENTS):
            findings.append(key)
    return sorted(set(findings))


def _environment_values(values: object, key: str) -> list[str]:
    if not isinstance(values, list):
        return []
    prefix = key + "="
    return [str(item)[len(prefix) :] for item in values if str(item).startswith(prefix)]


def _evaluate_probe(
    container: Mapping[str, Any],
    observed: Mapping[str, Any],
    bindings: Sequence[SourceBinding],
    *,
    input_manifest_sha256: str = "",
) -> tuple[list[str], dict[str, str]]:
    blockers: list[str] = []
    host = container.get("HostConfig") or {}
    config = container.get("Config") or {}
    if not (
        host.get("ReadonlyRootfs") is True
        and host.get("NetworkMode") == "none"
        and host.get("CapDrop") == ["ALL"]
        and "no-new-privileges" in host.get("SecurityOpt", [])
        and host.get("PidsLimit") == 256
        and host.get("NanoCpus") == 2_000_000_000
        and host.get("Memory") == 4 * 1024**3
        and host.get("MemorySwap") == 4 * 1024**3
        and host.get("Privileged") is False
        and host.get("PidMode") == ""
        and host.get("IpcMode") == "private"
        and not host.get("Binds")
        and not host.get("PortBindings")
        and not host.get("Devices")
        and not host.get("DeviceRequests")
        and not container.get("Mounts")
        and config.get("User") == NONROOT_USER
        and observed.get("uid") == 65532
        and observed.get("gid") == 65532
        and observed.get("environment") == RUNTIME_ENVIRONMENT
        and observed.get("root_write_denied") is True
        and observed.get("docker_socket_present") is False
    ):
        blockers.append("nonroot_readonly_network_none_probe_failed")
    if observed.get("credential_paths_readable"):
        blockers.append("embedded_credential_path_detected")
    if _sensitive_environment(config.get("Env") or []):
        blockers.append("embedded_sensitive_environment_detected")
    tools = observed.get("tools")
    versions: dict[str, str] = {}
    if not isinstance(tools, Mapping):
        blockers.append("tool_version_probe_invalid")
        tools = {}
    for name in ("git", "python", "rg"):
        record = tools.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != REQUIRED_BASE_TOOLS[name][0]
            or record.get("returncode") != 0
            or not record.get("version")
        ):
            blockers.append(f"base_tool_{name}_unavailable")
        else:
            versions[name] = str(record["version"]).splitlines()[0]
    hashes = observed.get("hashes") or {}
    metadata = observed.get("file_metadata") or {}
    for binding in bindings:
        record = tools.get(binding.name)
        file_record = metadata.get(binding.name)
        if (
            not isinstance(record, Mapping)
            or not isinstance(file_record, Mapping)
            or record.get("path") != f"/opt/eaaef/bin/{binding.name}"
            or record.get("returncode") != 0
            or record.get("version") != binding.version
            or hashes.get(binding.name) != binding.sha256
            or file_record
            != {
                "gid": 65532,
                "links": 1,
                "mode": "0555",
                "regular": True,
                "size": binding.size,
                "uid": 65532,
            }
        ):
            blockers.append(f"embedded_{binding.name}_identity_drift")
        else:
            versions[binding.name] = binding.version
    if input_manifest_sha256:
        manifest_hash = input_manifest_sha256.removeprefix("sha256:")
        manifest_record = metadata.get("candidate-inputs")
        if (
            len(manifest_hash) != 64
            or hashes.get("candidate-inputs") != manifest_hash
            or not isinstance(manifest_record, Mapping)
            or manifest_record.get("gid") != 65532
            or manifest_record.get("links") != 1
            or manifest_record.get("mode") != "0444"
            or manifest_record.get("regular") is not True
            or not isinstance(manifest_record.get("size"), int)
            or manifest_record.get("size") <= 0
            or manifest_record.get("uid") != 65532
        ):
            blockers.append("embedded_input_manifest_identity_drift")
    return blockers, versions


def _spdx_document(
    *,
    image_id: str,
    image_tag: str,
    bindings: Sequence[SourceBinding],
    base_tool_versions: Mapping[str, str],
    source_date_epoch: int,
) -> dict[str, Any]:
    image_hash = image_id.removeprefix("sha256:")
    created = dt.datetime.fromtimestamp(source_date_epoch, tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": "SPDXRef-ImplementationWorkerImage",
            "name": image_tag,
            "versionInfo": image_id,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": image_hash}],
            "supplier": "NOASSERTION",
            "copyrightText": "NOASSERTION",
        },
        {
            "SPDXID": "SPDXRef-BaseImage",
            "name": EXPECTED_BASE_REFERENCE,
            "versionInfo": EXPECTED_BASE_IMAGE_ID,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": EXPECTED_BASE_IMAGE_ID.removeprefix("sha256:"),
                }
            ],
            "supplier": "NOASSERTION",
            "copyrightText": "NOASSERTION",
        },
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-ImplementationWorkerImage",
        },
        {
            "spdxElementId": "SPDXRef-ImplementationWorkerImage",
            "relationshipType": "DESCENDANT_OF",
            "relatedSpdxElement": "SPDXRef-BaseImage",
        },
    ]
    for binding in bindings:
        spdx_id = f"SPDXRef-Native-{binding.name.title()}"
        packages.append(
            {
                "SPDXID": spdx_id,
                "name": binding.name,
                "versionInfo": binding.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "checksums": [{"algorithm": "SHA256", "checksumValue": binding.sha256}],
                "supplier": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "comment": (
                    "Exact static native input; source owner "
                    f"{binding.uid}:{binding.gid}, mode {binding.mode}."
                ),
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-ImplementationWorkerImage",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": spdx_id,
            }
        )
    admitted_base_tools = {
        name: base_tool_versions[name] for name in REQUIRED_BASE_TOOLS if name in base_tool_versions
    }
    for index, (name, version) in enumerate(sorted(admitted_base_tools.items()), start=1):
        spdx_id = f"SPDXRef-BaseTool-{index}"
        packages.append(
            {
                "SPDXID": spdx_id,
                "name": name,
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "supplier": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "comment": "Observed in the exact pinned base image; not copied.",
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-ImplementationWorkerImage",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": spdx_id,
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "EAAEF implementation-worker unsigned candidate SBOM",
        "documentNamespace": ("urn:ipfs-accelerate:eaaef:implementation-worker-sbom:" + image_hash),
        "creationInfo": {
            "created": created,
            "creators": ["Tool: qualify_external_agent_implementation_worker_image.py"],
        },
        "documentComment": (
            "Closed unsigned candidate. Package-level exact image/base/native "
            "input and observed base-tool statement; filesystem files and "
            "transitive base packages were not analyzed. Worker capacity is 0."
        ),
        "packages": packages,
        "relationships": relationships,
    }


def _sbom_identity(sbom_bytes: bytes, *, image_id: str) -> dict[str, object]:
    """Return the detached identity that the candidate report binds.

    The identity is intentionally outside the SPDX bytes: embedding a digest
    of those same bytes in the document would make the content identity
    cyclic and unverifiable.
    """

    return {
        "content_cid": "sha256:" + hashlib.sha256(sbom_bytes).hexdigest(),
        "canonicalization": SBOM_CANONICALIZATION,
        "bytes": len(sbom_bytes),
        "subject_image_id": image_id,
    }


def _base_report(*, source_date_epoch: int) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "source_date_epoch": source_date_epoch,
        "decision": "no_go",
        "status": "host_capability_no_go",
        "workload_class": "offline_implementation_worker_image_diagnostic",
        "task_dispatch_admitted": False,
        "worker_capacity": 0,
        "maximum_parallel_workers": 0,
        "network_authorized": False,
        "provider_authorized": False,
        "independent_authorization_verified": False,
        "candidate_signed": False,
        "image_signature_minted": False,
        "sbom_signature_minted": False,
        "production_receipt_minted": False,
        "authority_mutated": False,
        "provider_auth_accessed": False,
        "provider_invoked": False,
        "supervisor_process_started": False,
        "diagnostic_container_process_started": False,
        "blockers": [
            "network_authorization_not_independently_signed",
            "provider_authorization_not_independently_signed",
        ],
        "sources": [],
        "build": {"attempted": False, "succeeded": False, "image_id": ""},
        "probe": {},
        "credential_scan": {
            "schema": ("ipfs_accelerate_py/agent-supervisor/eaaef-bounded-image-credential-scan@1"),
            "mode": "not-run",
            "complete": False,
            "stop_reason": "container_runtime_unavailable",
            "findings": [],
            "raw_secret_values_recorded": False,
        },
        "sbom": {
            "format": "",
            "content_cid": "",
            "canonicalization": "",
            "bytes": 0,
            "files_analyzed": False,
            "subject_image_id": "",
        },
    }


def qualify(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    repo_root = Path(args.repo_root).resolve(strict=True)
    report = _base_report(source_date_epoch=args.source_date_epoch)
    bindings = [
        _bind_source("codex", Path(args.codex_binary), repo_root=repo_root),
        _bind_source("grok", Path(args.grok_binary), repo_root=repo_root),
    ]
    report["sources"] = [asdict(binding) for binding in bindings]
    docker = shutil.which(args.runtime)
    if docker is None:
        report["blockers"].append("container_runtime_unavailable")
        report["report_cid"] = _cid(report)
        return report, b""
    build = _build(
        docker=docker,
        repo_root=repo_root,
        image_tag=args.image_tag,
        bindings=bindings,
        source_date_epoch=args.source_date_epoch,
    )
    report["build"] = build
    inspected = _docker_json(docker, ["image", "inspect", args.image_tag], cwd=repo_root)
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise QualificationError("built image inspection is invalid")
    image = inspected[0]
    config = image.get("Config") or {}
    labels = config.get("Labels") or {}
    if (
        image.get("Id") != build["image_id"]
        or image.get("Os") != "linux"
        or image.get("Architecture") != EXPECTED_ARCHITECTURE
        or config.get("User") != NONROOT_USER
        or labels.get("org.ipfs-accelerate.eaaef.worker-capacity") != "0"
        or labels.get("org.ipfs-accelerate.eaaef.unsigned") != "true"
        or labels.get("org.ipfs-accelerate.eaaef.codex.sha256") != bindings[0].sha256
        or labels.get("org.ipfs-accelerate.eaaef.grok.sha256") != bindings[1].sha256
        or labels.get("org.ipfs-accelerate.eaaef.input-manifest.sha256")
        != str(build["input_manifest_sha256"]).removeprefix("sha256:")
        or labels.get("org.ipfs-accelerate.eaaef.rootfs-tar.sha256")
        != str(build["rootfs_tar_sha256"]).removeprefix("sha256:")
        or labels.get("org.ipfs-accelerate.eaaef.source-date-epoch") != str(args.source_date_epoch)
        or labels.get("org.opencontainers.image.base.digest") != EXPECTED_BASE_IMAGE_ID
        or _environment_values(config.get("Env"), "NVIDIA_VISIBLE_DEVICES") != ["void"]
        or _environment_values(config.get("Env"), "NVIDIA_DRIVER_CAPABILITIES") != [""]
    ):
        raise QualificationError("built image identity or labels drifted")
    try:
        created = dt.datetime.fromisoformat(str(image.get("Created") or "").replace("Z", "+00:00"))
        created_epoch = int(created.timestamp())
    except (ValueError, OverflowError) as exc:
        raise QualificationError("built image creation timestamp is invalid") from exc
    build["image_created_at"] = str(image.get("Created"))
    build["source_date_epoch_applied"] = created_epoch == args.source_date_epoch
    if not build.get("clean_build_reproducible"):
        report["blockers"].append("clean_offline_build_not_reproducible")
    if not build["source_date_epoch_applied"]:
        report["blockers"].append("source_date_epoch_not_applied")
    container, observed = _probe(docker=docker, repo_root=repo_root, image_tag=args.image_tag)
    report["diagnostic_container_process_started"] = True
    blockers, tool_versions = _evaluate_probe(
        container,
        observed,
        bindings,
        input_manifest_sha256=str(build["input_manifest_sha256"]),
    )
    report["blockers"].extend(blockers)
    try:
        credential_scan = _scan_exported_filesystem(
            docker=docker,
            repo_root=repo_root,
            image_tag=args.image_tag,
        )
    except (OSError, QualificationError, subprocess.TimeoutExpired, tarfile.TarError) as exc:
        credential_scan = {
            "schema": ("ipfs_accelerate_py/agent-supervisor/eaaef-bounded-image-credential-scan@1"),
            "mode": "merged-docker-export-tar-stream",
            "complete": False,
            "stop_reason": str(exc)[:4096],
            "findings": [],
            "raw_secret_values_recorded": False,
        }
        report["blockers"].append("bounded_credential_scan_incomplete")
    else:
        if credential_scan.get("finding_files"):
            report["blockers"].append("credential_like_material_detected")
    report["credential_scan"] = credential_scan
    report["probe"] = {
        "hardening_valid": "nonroot_readonly_network_none_probe_failed" not in blockers,
        "known_credential_path_probe": {
            "paths": list(KNOWN_CREDENTIAL_PATHS),
            "clear": "embedded_credential_path_detected" not in blockers,
            "claim_scope": "only_the_listed_paths",
        },
        "sensitive_image_environment_probe_clear": (
            "embedded_sensitive_environment_detected" not in blockers
        ),
        "base_tool_versions": tool_versions,
        "host_usr_bind_present": bool((container.get("HostConfig") or {}).get("Binds")),
        "observed": observed,
    }
    sbom = _spdx_document(
        image_id=str(build["image_id"]),
        image_tag=args.image_tag,
        bindings=bindings,
        base_tool_versions=tool_versions,
        source_date_epoch=args.source_date_epoch,
    )
    sbom_bytes = _canonical(sbom) + b"\n"
    if len(sbom_bytes) > MAXIMUM_SBOM_BYTES:
        raise QualificationError("SPDX document exceeded its bound")
    report["sbom"] = {
        "format": "spdx-json",
        "spdx_version": "SPDX-2.3",
        "files_analyzed": False,
        **_sbom_identity(sbom_bytes, image_id=str(build["image_id"])),
    }
    report["blockers"] = list(dict.fromkeys(report["blockers"]))
    capability_blockers = [
        item
        for item in report["blockers"]
        if item
        not in {
            "network_authorization_not_independently_signed",
            "provider_authorization_not_independently_signed",
        }
    ]
    report["status"] = (
        "host_capability_no_go"
        if capability_blockers
        else "closed_unsigned_candidate_for_independent_review"
    )
    report["report_cid"] = _cid(report)
    if len(_canonical(report)) > MAXIMUM_REPORT_BYTES:
        raise QualificationError("candidate report exceeded its bound")
    return report, sbom_bytes


def _atomic_write(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime", choices=("docker",), default="docker")
    parser.add_argument("--codex-binary", type=Path, required=True)
    parser.add_argument("--grok-binary", type=Path, required=True)
    parser.add_argument(
        "--image-tag",
        default="eaaef-implementation-worker:local-unsigned-candidate",
    )
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source_date_epoch <= 0:
        raise SystemExit("--source-date-epoch must be positive")
    try:
        report, sbom = qualify(args)
        if sbom:
            _atomic_write(args.sbom, sbom)
        _atomic_write(args.report, _canonical(report) + b"\n")
    except (OSError, QualificationError, subprocess.TimeoutExpired) as exc:
        print(f"qualification_error: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
