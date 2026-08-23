#!/usr/bin/env python3
"""Build and qualify one closed minimal EAAEF worker-image candidate.

The command never pulls an image, installs a package, reads provider
authentication, or invokes a provider.  It copies a deterministic allowlist of
Python/Git/ripgrep runtime files out of one exact *local* source image and adds
two separately bound native clients.  The resulting report is deliberately an
unsigned no-go record with zero dispatch capacity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.parser
import email.policy
import hashlib
import importlib.util
import io
import json
import os
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

_SHARED_PATH = Path(__file__).with_name(
    "qualify_external_agent_implementation_worker_image.py"
)
_SPEC = importlib.util.spec_from_file_location("eaaef_worker_image_shared", _SHARED_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure is fatal
    raise RuntimeError("shared worker-image qualification module is unavailable")
shared = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = shared
_SPEC.loader.exec_module(shared)

REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-implementation-worker-minimal-image-candidate@1"
)
INPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-implementation-worker-minimal-image-inputs@1"
)
BASE_REFERENCE = "ubuntu:24.04"
BASE_IMAGE_ID = "sha256:ea17ec341c4211d1dd7f184a0dedf7dcb7945e92db20a5dde20544262214b84f"
TOOL_SOURCE_REFERENCE = "ipfs-accelerate-authority-validation:20260803-v2"
TOOL_SOURCE_IMAGE_ID = (
    "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
)
CONTAINERFILE = Path(
    "containers/external-agent/implementation-worker-minimal.Containerfile"
)
NONROOT_USER = "65532:65532"
MAXIMUM_MANIFEST_BYTES = 4 * 1024 * 1024
MAXIMUM_SBOM_BYTES = 1024 * 1024
MAXIMUM_CLOSURE_FILES = 10_000
MAXIMUM_CLOSURE_BYTES = 1024 * 1024**2

RUNTIME_ENVIRONMENT = {
    "BASH_ENV": "",
    "CODEX_HOME": "/opt/codex-home",
    "ENV": "",
    "GIT_CONFIG_NOSYSTEM": "1",
    "HOME": "/opt/codex-home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": "/opt/eaaef/lib",
    "PATH": "/opt/eaaef/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "TERM": "dumb",
}

COPY_PATHS = (
    ("/usr/bin/python", "usr/bin/python"),
    ("/usr/bin/python3", "usr/bin/python3"),
    ("/usr/bin/python3.12", "usr/bin/python3.12"),
    ("/usr/bin/git", "usr/bin/git"),
    ("/usr/local/bin/rg", "usr/local/bin/rg"),
    ("/usr/lib/python3.12", "usr/lib/python3.12"),
    ("/usr/lib/git-core", "usr/lib/git-core"),
    ("/usr/share/git-core/templates", "usr/share/git-core/templates"),
    ("/etc/ssl/certs/ca-certificates.crt", "etc/ssl/certs/ca-certificates.crt"),
)
PYTHON_VALIDATION_IMPORT_ROOT = "/opt/eaaef/python-validation"
PYTHON_VALIDATION_DRIVER = "/opt/eaaef/libexec/python-validation-driver.py"
PYTHON_VALIDATION_WRAPPER = "/opt/eaaef/bin/python"
PYTHON_VALIDATION_COMPONENTS = (
    {
        "name": "pytest",
        "version": "9.0.3",
        "runtime_dependencies": (
            "iniconfig==2.3.0",
            "packaging==26.2",
            "pluggy==1.6.0",
            "Pygments==2.19.2",
        ),
        "paths": (
            ("/opt/ipfs-validation-site-packages/_pytest", "_pytest"),
            ("/opt/ipfs-validation-site-packages/pytest", "pytest"),
            ("/opt/ipfs-validation-site-packages/py.py", "py.py"),
            (
                "/opt/ipfs-validation-site-packages/pytest-9.0.3.dist-info",
                "pytest-9.0.3.dist-info",
            ),
        ),
        "metadata": "pytest-9.0.3.dist-info/METADATA",
    },
    {
        "name": "pluggy",
        "version": "1.6.0",
        "runtime_dependencies": (),
        "paths": (
            ("/opt/ipfs-validation-site-packages/pluggy", "pluggy"),
            (
                "/opt/ipfs-validation-site-packages/pluggy-1.6.0.dist-info",
                "pluggy-1.6.0.dist-info",
            ),
        ),
        "metadata": "pluggy-1.6.0.dist-info/METADATA",
    },
    {
        "name": "packaging",
        "version": "26.2",
        "runtime_dependencies": (),
        "paths": (
            ("/opt/ipfs-validation-site-packages/packaging", "packaging"),
            (
                "/opt/ipfs-validation-site-packages/packaging-26.2.dist-info",
                "packaging-26.2.dist-info",
            ),
        ),
        "metadata": "packaging-26.2.dist-info/METADATA",
    },
    {
        "name": "iniconfig",
        "version": "2.3.0",
        "runtime_dependencies": (),
        "paths": (
            ("/opt/ipfs-validation-site-packages/iniconfig", "iniconfig"),
            (
                "/opt/ipfs-validation-site-packages/iniconfig-2.3.0.dist-info",
                "iniconfig-2.3.0.dist-info",
            ),
        ),
        "metadata": "iniconfig-2.3.0.dist-info/METADATA",
    },
    {
        "name": "Pygments",
        "version": "2.19.2",
        "runtime_dependencies": (),
        "paths": (
            ("/usr/local/lib/python3.12/dist-packages/pygments", "pygments"),
            (
                "/usr/local/lib/python3.12/dist-packages/pygments-2.19.2.dist-info",
                "pygments-2.19.2.dist-info",
            ),
        ),
        "metadata": "pygments-2.19.2.dist-info/METADATA",
    },
    {
        "name": "cryptography",
        "version": "49.0.0",
        "runtime_dependencies": ("cffi==2.1.0",),
        "paths": (
            ("/opt/ipfs-validation-site-packages/cryptography", "cryptography"),
            (
                "/opt/ipfs-validation-site-packages/cryptography-49.0.0.dist-info",
                "cryptography-49.0.0.dist-info",
            ),
        ),
        "metadata": "cryptography-49.0.0.dist-info/METADATA",
    },
    {
        "name": "cffi",
        "version": "2.1.0",
        "runtime_dependencies": ("pycparser==3.0",),
        "paths": (
            ("/opt/ipfs-validation-site-packages/cffi", "cffi"),
            (
                "/opt/ipfs-validation-site-packages/cffi-2.1.0.dist-info",
                "cffi-2.1.0.dist-info",
            ),
            (
                "/opt/ipfs-validation-site-packages/_cffi_backend.cpython-312-aarch64-linux-gnu.so",
                "_cffi_backend.cpython-312-aarch64-linux-gnu.so",
            ),
        ),
        "metadata": "cffi-2.1.0.dist-info/METADATA",
    },
    {
        "name": "pycparser",
        "version": "3.0",
        "runtime_dependencies": (),
        "paths": (
            ("/opt/ipfs-validation-site-packages/pycparser", "pycparser"),
            (
                "/opt/ipfs-validation-site-packages/pycparser-3.0.dist-info",
                "pycparser-3.0.dist-info",
            ),
        ),
        "metadata": "pycparser-3.0.dist-info/METADATA",
    },
    {
        "name": "duckdb",
        "version": "1.5.2",
        "runtime_dependencies": (),
        "paths": (
            ("/opt/ipfs-validation-site-packages/duckdb", "duckdb"),
            (
                "/opt/ipfs-validation-site-packages/duckdb-1.5.2.dist-info",
                "duckdb-1.5.2.dist-info",
            ),
            (
                "/opt/ipfs-validation-site-packages/_duckdb-stubs",
                "_duckdb-stubs",
            ),
            (
                "/opt/ipfs-validation-site-packages/_duckdb.cpython-312-aarch64-linux-gnu.so",
                "_duckdb.cpython-312-aarch64-linux-gnu.so",
            ),
        ),
        "metadata": "duckdb-1.5.2.dist-info/METADATA",
    },
)
PYTHON_VALIDATION_EXPECTED_VERSIONS = {
    str(item["name"]): str(item["version"])
    for item in PYTHON_VALIDATION_COMPONENTS
}
PYTHON_VALIDATION_NATIVE_SOURCES = (
    "/opt/ipfs-validation-site-packages/_duckdb.cpython-312-aarch64-linux-gnu.so",
    "/opt/ipfs-validation-site-packages/_cffi_backend.cpython-312-aarch64-linux-gnu.so",
    "/opt/ipfs-validation-site-packages/cryptography/hazmat/bindings/_rust.abi3.so",
)

# The wrapper deliberately does not honor PYTHONPATH, user/site directories or
# ``.pth`` files.  The worker runtime resolves the task worktree separately;
# this driver admits only its real current working directory plus the exact
# manifest-bound validation root.
PYTHON_VALIDATION_DRIVER_BYTES = b"""from __future__ import annotations
import os
import runpy
import sys

VALIDATION_ROOT = \"/opt/eaaef/python-validation\"

if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
    raise SystemExit(125)
if \"site\" in sys.modules or any(name.endswith(\".pth\") for name in os.listdir(VALIDATION_ROOT)):
    raise SystemExit(126)

project_root = os.path.realpath(os.getcwd())
sys.path[:] = [VALIDATION_ROOT, project_root, *[
    path for path in sys.path
    if path and path != project_root and \"site-packages\" not in path and \"dist-packages\" not in path
]]
arguments = sys.argv[1:]
if arguments[:2] == [\"-m\", \"pytest\"]:
    import pytest
    sys.argv = [\"pytest\", *arguments[2:]]
    raise SystemExit(pytest.console_main())
if arguments and not arguments[0].startswith(\"-\"):
    script = os.path.realpath(arguments[0])
    if os.path.commonpath((project_root, script)) != project_root:
        raise SystemExit(126)
    sys.argv = arguments
    runpy.run_path(script, run_name=\"__main__\")
    raise SystemExit(0)
raise SystemExit(126)
"""
PYTHON_VALIDATION_WRAPPER_BYTES = (
    b"#!/bin/sh\nexec /usr/bin/python3 -I -S -B "
    b"/opt/eaaef/libexec/python-validation-driver.py \"$@\"\n"
)
SKIPPED_RELATIVE_PATHS = frozenset(
    {
        "usr/lib/python3.12/sitecustomize.py",
    }
)
SKIPPED_PREFIXES = (
    "usr/lib/python3.12/__pycache__/",
    "usr/lib/python3.12/config-3.12-aarch64-linux-gnu/",
)
EXPECTED_TOOL_PATHS = {
    "python": "/usr/bin/python3",
    "git": "/usr/bin/git",
    "rg": "/usr/local/bin/rg",
    "codex": "/opt/eaaef/bin/codex",
    "grok": "/opt/eaaef/bin/grok",
}

# These are exact artifact-scoped adjudications, not path-only exclusions.
# The full byte scan still reports each heuristic signal.  The listed hashes
# were separately inspected: the Grok match is its documented
# ``api_key = \"your ... token\"`` placeholder, libssh contains OpenSSH key
# algorithm/PEM parser constants, and libgnutls contains PEM parser constants.
# Any path, digest, or detector drift becomes an unresolved material finding.
KNOWN_STATIC_MARKER_FINDINGS = {
    (
        "opt/eaaef/python-validation/_duckdb.cpython-312-aarch64-linux-gnu.so",
        "sha256:c378b8f61040764fdc904cf7c0643a005d547f491ab9303e6bd13c33aa353f2a",
    ): {
        "detectors": ["private_key_pem"],
        "classification": "duckdb_extension_crypto_parser_marker_not_credential",
    },
    (
        "opt/eaaef/python-validation/cryptography/hazmat/bindings/_rust/openssl/hpke.pyi",
        "sha256:a7f8462e7e981fe11aac91755796d4b14b638a9be2100a5c4793b4b141c92ed7",
    ): {
        "detectors": ["assigned_secret_value"],
        "classification": "cryptography_typed_api_parameter_not_credential",
    },
    (
        "opt/eaaef/python-validation/cryptography/hazmat/primitives/asymmetric/ec.py",
        "sha256:2e495508b2f16db433aa213a1d16984d16ae2cca61614c37b44e9cf28048c584",
    ): {
        "detectors": ["assigned_secret_value"],
        "classification": "cryptography_private_key_api_variable_not_credential",
    },
    (
        "opt/eaaef/python-validation/cryptography/hazmat/primitives/serialization/base.py",
        "sha256:8a4ab9309230a7fa149e389a05ca3f3e643039362e1a2f979185181cacbc568d",
    ): {
        "detectors": ["assigned_secret_value"],
        "classification": "cryptography_abstract_api_parameter_not_credential",
    },
    (
        "opt/eaaef/python-validation/cryptography/hazmat/primitives/serialization/pkcs7.py",
        "sha256:98533b385b99f1c0b1506528a380a87fa06708773f21bc750108e37f19cf971a",
    ): {
        "detectors": ["assigned_secret_value"],
        "classification": "cryptography_pkcs7_api_variable_not_credential",
    },
    (
        "opt/eaaef/python-validation/cryptography/hazmat/primitives/serialization/ssh.py",
        "sha256:1d5e99a888ea34d68d50be8ce2043234926d67be44b22499b3acb175da81ad62",
    ): {
        "detectors": ["assigned_secret_value", "private_key_pem"],
        "classification": "cryptography_ssh_key_parser_source_not_credential",
    },
    (
        "opt/eaaef/python-validation/cryptography/x509/base.py",
        "sha256:93dcc4142ec7ec6cace7c02624d7847e94de242be883ca7ca0311af0383408ed",
    ): {
        "detectors": ["assigned_secret_value"],
        "classification": "cryptography_x509_api_parameter_not_credential",
    },
    (
        "opt/eaaef/python-validation/cryptography/x509/ocsp.py",
        "sha256:61ecba35d155d4c3e3a29db8323fd57a78c4a60de451a328a5258e2b401b781b",
    ): {
        "detectors": ["assigned_secret_value"],
        "classification": "cryptography_ocsp_api_parameter_not_credential",
    },
    (
        "opt/eaaef/bin/grok",
        "sha256:1c1fe67d7c35497fb09f44a451f57acc3787add4c9aea2c56f5c7c75dc5ffcf1",
    ): {
        "detectors": ["assigned_secret_value"],
        "classification": "documented_placeholder_literal_not_credential",
        "matched_literal_sha256": (
            "sha256:9bfee49ff9af15d50f704da5d60561c1e0e5c4233e9acc9f0d7c5c362aa49e2a"
        ),
    },
    (
        "opt/eaaef/lib/libgnutls.so.30",
        "sha256:6b12c4675dbf7fca76bd47228f22b19f4218bc8c72e2f8b5cb6de172dee14961",
    ): {
        "detectors": ["private_key_pem"],
        "classification": "pem_parser_marker_not_credential",
    },
    (
        "opt/eaaef/lib/libssh.so.4",
        "sha256:420097fc2d7ea28265e1ffe6c8f7eaeb7c6eeadfea82a0bc4a995f7e8b9abf02",
    ): {
        "detectors": ["private_key_pem", "provider_secret_token"],
        "classification": "ssh_algorithm_and_pem_parser_markers_not_credentials",
    },
    (
        "usr/lib/aarch64-linux-gnu/libgnutls.so.30.37.1",
        "sha256:17ffd34fd0bc239903ca37e95bb14eab6ee986c8d1260a5a08f16e0bfc669739",
    ): {
        "detectors": ["private_key_pem"],
        "classification": "pem_parser_marker_not_credential",
    },
}


class MinimalQualificationError(ValueError):
    """A deterministic minimal-image qualification failure."""


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


def _image_id(docker: str, reference: str, *, repo_root: Path) -> str:
    return shared._image_id(docker, reference, cwd=repo_root)


def _dependency_discovery_program() -> str:
    program = r'''import json,os,re,subprocess
candidates=__EAAEF_NATIVE_CANDIDATES__
for root in ("/usr/lib/python3.12/lib-dynload","/usr/lib/git-core"):
    for parent,dirs,files in os.walk(root):
        dirs.sort();files.sort()
        for name in files:
            path=os.path.join(parent,name)
            if name.endswith(".so") or os.access(path,os.X_OK): candidates.append(path)
deps=set();checked=[]
for path in sorted(set(candidates)):
    result=subprocess.run(["/usr/bin/ldd",path],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=20,env={"LANG":"C","LC_ALL":"C","PATH":"/usr/bin:/bin"})
    if result.returncode not in (0,1): raise SystemExit("ldd_failed:"+path)
    checked.append(path)
    for line in result.stdout.splitlines():
        match=re.search(r"=> (/[^ ]+) \(",line) or re.match(r"\s*(/[^ ]+) \(",line)
        if match: deps.add(match.group(1))
records=[]
for path in sorted(deps):
    resolved=os.path.realpath(path)
    if not resolved.startswith(("/lib/aarch64-linux-gnu/","/usr/lib/aarch64-linux-gnu/")): raise SystemExit("dependency_outside_allowlist:"+path)
    if not os.path.isfile(resolved): raise SystemExit("dependency_not_regular:"+path)
    records.append({"path":path,"resolved":resolved})
print(json.dumps({"candidates":checked,"dependencies":records},sort_keys=True,separators=(",",":")))'''
    return program.replace(
        "__EAAEF_NATIVE_CANDIDATES__",
        repr(
            [
                "/usr/bin/python3.12",
                "/usr/bin/git",
                "/usr/local/bin/rg",
                *PYTHON_VALIDATION_NATIVE_SOURCES,
            ]
        ),
    )


def _discover_dependencies(
    docker: str, *, repo_root: Path
) -> dict[str, Any]:
    result = shared._run(
        [
            docker,
            "run",
            "--rm",
            "--pull=never",
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--user",
            "0:0",
            "--entrypoint",
            "/usr/bin/env",
            TOOL_SOURCE_REFERENCE,
            "-i",
            "PATH=/usr/bin:/bin",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            _dependency_discovery_program(),
        ],
        cwd=repo_root,
        timeout=180,
    )
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MinimalQualificationError("dependency discovery returned invalid JSON") from exc
    dependencies = record.get("dependencies") if isinstance(record, Mapping) else None
    candidates = record.get("candidates") if isinstance(record, Mapping) else None
    if not isinstance(dependencies, list) or not isinstance(candidates, list):
        raise MinimalQualificationError("dependency discovery record is invalid")
    if not dependencies or len(dependencies) > 256:
        raise MinimalQualificationError("dependency closure is empty or unbounded")
    return {"candidates": candidates, "dependencies": dependencies}


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise MinimalQualificationError("closure destination escaped staging root")
    return Path(*pure.parts)


def _docker_cp(
    docker: str,
    container_id: str,
    source: str,
    destination: Path,
    *,
    repo_root: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shared._run(
        [docker, "cp", f"{container_id}:{source}", str(destination)],
        cwd=repo_root,
        timeout=180,
    )


def _stage_closure(
    docker: str,
    *,
    repo_root: Path,
    overlay: Path,
    dependencies: Sequence[Mapping[str, Any]],
) -> None:
    container_id = shared._run(
        [
            docker,
            "create",
            "--pull=never",
            "--network",
            "none",
            "--entrypoint",
            "/bin/true",
            TOOL_SOURCE_REFERENCE,
        ],
        cwd=repo_root,
    ).stdout.strip()
    if not container_id:
        raise MinimalQualificationError("source container identity is empty")
    try:
        inspected = shared._docker_json(
            docker, ["container", "inspect", container_id], cwd=repo_root
        )
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise MinimalQualificationError("source container inspection is invalid")
        source_container = inspected[0]
        host = source_container.get("HostConfig") or {}
        if (
            source_container.get("State", {}).get("Running") is not False
            or source_container.get("Mounts")
            or host.get("NetworkMode") != "none"
            or host.get("Binds")
            or host.get("Devices")
            or host.get("DeviceRequests")
        ):
            raise MinimalQualificationError("source copy container is not inert")
        for source, relative in COPY_PATHS:
            _docker_cp(
                docker,
                container_id,
                source,
                overlay / _safe_relative(relative),
                repo_root=repo_root,
            )
        validation_root = overlay / _safe_relative(
            PYTHON_VALIDATION_IMPORT_ROOT.removeprefix("/")
        )
        validation_root.mkdir(parents=True, exist_ok=True)
        for component in PYTHON_VALIDATION_COMPONENTS:
            for source, relative in component["paths"]:
                _docker_cp(
                    docker,
                    container_id,
                    str(source),
                    validation_root / _safe_relative(str(relative)),
                    repo_root=repo_root,
                )
        driver = overlay / _safe_relative(PYTHON_VALIDATION_DRIVER.removeprefix("/"))
        driver.parent.mkdir(parents=True, exist_ok=True)
        driver.write_bytes(PYTHON_VALIDATION_DRIVER_BYTES)
        driver.chmod(0o444)
        wrapper = overlay / _safe_relative(PYTHON_VALIDATION_WRAPPER.removeprefix("/"))
        wrapper.write_bytes(PYTHON_VALIDATION_WRAPPER_BYTES)
        wrapper.chmod(0o555)
        wrapper_python3 = wrapper.with_name("python3")
        wrapper_python3.symlink_to("python")
        library_dir = overlay / "opt/eaaef/lib"
        library_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, str] = {}
        for dependency in dependencies:
            source_path = str(dependency.get("path") or "")
            resolved = str(dependency.get("resolved") or "")
            # The ELF interpreter is supplied, and content-addressed, by the
            # exact Ubuntu base.  LD_LIBRARY_PATH cannot replace it and the
            # closure must not overlay this base-critical path.
            if source_path == "/lib/ld-linux-aarch64.so.1":
                if resolved != "/usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1":
                    raise MinimalQualificationError("ELF interpreter identity drifted")
                continue
            if not source_path.startswith(("/lib/aarch64-linux-gnu/", "/usr/lib/aarch64-linux-gnu/")):
                raise MinimalQualificationError("dependency path escaped allowlist")
            if not resolved.startswith(("/lib/aarch64-linux-gnu/", "/usr/lib/aarch64-linux-gnu/")):
                raise MinimalQualificationError("resolved dependency escaped allowlist")
            name = posixpath.basename(source_path)
            if not name or "/" in name:
                raise MinimalQualificationError("dependency soname is invalid")
            destination = library_dir / name
            if destination.exists():
                raise MinimalQualificationError("dependency soname collision")
            _docker_cp(
                docker,
                container_id,
                resolved,
                destination,
                repo_root=repo_root,
            )
            if not destination.is_file() or destination.is_symlink():
                raise MinimalQualificationError("copied dependency is not regular")
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if name in seen and seen[name] != digest:
                raise MinimalQualificationError("dependency soname content collision")
            seen[name] = digest
    finally:
        shared._run(
            [docker, "rm", "-f", container_id],
            cwd=repo_root,
            check=False,
        )


def _is_skipped(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return (
        relative in SKIPPED_RELATIVE_PATHS
        or relative.startswith(SKIPPED_PREFIXES)
        or "__pycache__" in parts
        or relative.endswith((".pyc", ".pyo"))
    )


def _normalized_archive_metadata(
    relative: str, kind: str, *, executable: bool = False
) -> tuple[int, int, str]:
    owner = (
        65532
        if relative.startswith(("opt/eaaef/bin/", "opt/codex-home"))
        else 0
    )
    if kind == "directory":
        mode = 0o700 if relative == "opt/codex-home" else 0o755
    elif kind == "symlink":
        mode = 0o777
    else:
        mode = 0o555 if executable else 0o444
    return owner, owner, f"{mode:04o}"


def _closure_records(overlay: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    logical_bytes = 0
    for path in sorted(overlay.rglob("*"), key=lambda item: item.relative_to(overlay).as_posix()):
        relative = path.relative_to(overlay).as_posix()
        if _is_skipped(relative):
            continue
        if relative.startswith("opt/eaaef/python-validation/") and (
            relative.endswith(".pth")
            or PurePosixPath(relative).name
            in {"sitecustomize.py", "usercustomize.py"}
        ):
            raise MinimalQualificationError(
                "validation closure contains site-startup executable content"
            )
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            uid, gid, mode = _normalized_archive_metadata(relative, "directory")
            records.append(
                {
                    "gid": gid,
                    "mode": mode,
                    "path": relative,
                    "type": "directory",
                    "uid": uid,
                }
            )
        elif stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            if target.startswith("/") or ".." in PurePosixPath(target).parts:
                raise MinimalQualificationError("closure contains an unsafe symlink")
            uid, gid, mode = _normalized_archive_metadata(relative, "symlink")
            records.append(
                {
                    "gid": gid,
                    "mode": mode,
                    "path": relative,
                    "target": target,
                    "type": "symlink",
                    "uid": uid,
                }
            )
        elif stat.S_ISREG(info.st_mode):
            logical_bytes += info.st_size
            executable = bool(info.st_mode & 0o111)
            uid, gid, mode = _normalized_archive_metadata(
                relative, "file", executable=executable
            )
            records.append(
                {
                    "executable": executable,
                    "gid": gid,
                    "mode": mode,
                    "path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": info.st_size,
                    "type": "file",
                    "uid": uid,
                }
            )
        else:
            raise MinimalQualificationError("closure contains a special file")
        if len(records) > MAXIMUM_CLOSURE_FILES or logical_bytes > MAXIMUM_CLOSURE_BYTES:
            raise MinimalQualificationError("closure exceeded file or byte bound")
    return records


def _validation_toolchain_manifest(
    overlay: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validation_relative = PYTHON_VALIDATION_IMPORT_ROOT.removeprefix("/")
    by_path = {str(item["path"]): dict(item) for item in records}
    components: list[dict[str, Any]] = []
    claimed_paths: set[str] = set()
    for specification in PYTHON_VALIDATION_COMPONENTS:
        metadata_relative = (
            f"{validation_relative}/{specification['metadata']}"
        )
        metadata_path = overlay / _safe_relative(metadata_relative)
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise MinimalQualificationError(
                f"validation metadata missing for {specification['name']}"
            )
        message = email.parser.BytesParser(policy=email.policy.default).parsebytes(
            metadata_path.read_bytes()
        )
        observed_name = str(message.get("Name") or "")
        observed_version = str(message.get("Version") or "")
        if (
            observed_name.lower().replace("-", "_")
            != str(specification["name"]).lower().replace("-", "_")
            or observed_version != specification["version"]
        ):
            raise MinimalQualificationError(
                f"validation distribution identity drifted for {specification['name']}"
            )
        component_paths: list[str] = []
        for _source, relative in specification["paths"]:
            target = f"{validation_relative}/{relative}"
            matches = sorted(
                path
                for path in by_path
                if path == target or path.startswith(target + "/")
            )
            if not matches:
                raise MinimalQualificationError(
                    f"validation component path missing for {specification['name']}"
                )
            component_paths.extend(matches)
        if len(component_paths) != len(set(component_paths)):
            raise MinimalQualificationError("validation component paths overlap")
        claimed_paths.update(component_paths)
        entries = [by_path[path] for path in sorted(component_paths)]
        component = {
            "name": specification["name"],
            "version": specification["version"],
            "metadata_path": "/" + metadata_relative,
            "runtime_dependencies": list(specification["runtime_dependencies"]),
            "source_image_paths": [
                str(source) for source, _relative in specification["paths"]
            ],
            "closure_entries": entries,
        }
        component["content_cid"] = _cid(component)
        components.append(component)
    all_validation_paths = {
        path
        for path in by_path
        if path == validation_relative or path.startswith(validation_relative + "/")
    }
    unclaimed = sorted(
        path
        for path in all_validation_paths - claimed_paths
        if path != validation_relative
    )
    if unclaimed:
        raise MinimalQualificationError(
            "validation import root contains unassigned component paths"
        )
    entrypoint_paths = (
        "opt/eaaef/bin/python",
        "opt/eaaef/bin/python3",
        "opt/eaaef/libexec/python-validation-driver.py",
        "usr/bin/python",
        "usr/bin/python3",
        "usr/bin/python3.12",
    )
    try:
        entrypoints = [by_path[path] for path in entrypoint_paths]
    except KeyError as exc:
        raise MinimalQualificationError(
            "validation entrypoint closure is incomplete"
        ) from exc
    manifest = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-minimal-python-validation-toolchain@1"
        ),
        "approved_import_root": PYTHON_VALIDATION_IMPORT_ROOT,
        "components": components,
        "direct_board_executable_aliases": {
            "python": PYTHON_VALIDATION_WRAPPER,
            "python3": PYTHON_VALIDATION_WRAPPER + "3",
        },
        "entrypoints": entrypoints,
        "isolation_argv_prefix": [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            PYTHON_VALIDATION_DRIVER,
        ],
        "supported_board_forms": [
            ["python", "-m", "pytest"],
            ["python3", "-m", "pytest"],
            ["python", "<worktree-relative-script.py>"],
            ["python3", "<worktree-relative-script.py>"],
        ],
        "startup_code_policy": {
            "environment_pythonpath_accepted": False,
            "site_module_loaded": False,
            "user_site_accepted": False,
            "pth_files_accepted": False,
            "plugin_autoload": False,
        },
        "scope": (
            "bootstrap reconciliation Python validation; DuckDB 1.5.2 is a "
            "test dependency and is not the separately qualified DuckDB/Quack "
            "1.5.5 mutable control-plane authority"
        ),
    }
    manifest["content_cid"] = _cid(manifest)
    return manifest


def _member(name: str, *, mode: int, source_date_epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = mode
    info.mtime = source_date_epoch
    return info


def _canonical_rootfs_tar(
    destination: Path,
    *,
    overlay: Path,
    records: Sequence[Mapping[str, Any]],
    manifest_bytes: bytes,
    source_date_epoch: int,
) -> tuple[str, int]:
    all_records = [dict(item) for item in records]
    manifest_path = "opt/eaaef/minimal-candidate-inputs.json"
    if any(item.get("path") == manifest_path for item in all_records):
        raise MinimalQualificationError("manifest path collides with closure")
    with destination.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for record in all_records:
                relative = str(record["path"])
                source = overlay / _safe_relative(relative)
                kind = record["type"]
                owner = int(record["uid"])
                group = int(record["gid"])
                mode = int(str(record["mode"]), 8)
                if kind == "directory":
                    info = _member(relative, mode=mode, source_date_epoch=source_date_epoch)
                    info.uid = owner
                    info.gid = group
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                elif kind == "symlink":
                    info = _member(relative, mode=mode, source_date_epoch=source_date_epoch)
                    info.uid = owner
                    info.gid = group
                    info.type = tarfile.SYMTYPE
                    info.linkname = str(record["target"])
                    archive.addfile(info)
                else:
                    info = _member(relative, mode=mode, source_date_epoch=source_date_epoch)
                    info.uid = owner
                    info.gid = group
                    info.size = int(record["size"])
                    with source.open("rb") as payload:
                        archive.addfile(info, payload)
            manifest_info = _member(
                manifest_path, mode=0o444, source_date_epoch=source_date_epoch
            )
            manifest_info.uid = 65532
            manifest_info.gid = 65532
            manifest_info.size = len(manifest_bytes)
            archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        raw.flush()
        os.fsync(raw.fileno())
    os.chmod(destination, 0o444)
    normalized = source_date_epoch * 1_000_000_000
    os.utime(destination, ns=(normalized, normalized), follow_symlinks=False)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return digest, destination.stat().st_size


def _build(
    docker: str,
    *,
    repo_root: Path,
    image_tag: str,
    bindings: Sequence[Any],
    source_date_epoch: int,
) -> dict[str, Any]:
    containerfile = (repo_root / CONTAINERFILE).resolve(strict=True)
    if containerfile.is_symlink() or containerfile.parent.parent.parent != repo_root:
        raise MinimalQualificationError("minimal Containerfile path drifted")
    before_base = _image_id(docker, BASE_REFERENCE, repo_root=repo_root)
    before_source = _image_id(docker, TOOL_SOURCE_REFERENCE, repo_root=repo_root)
    if before_base != BASE_IMAGE_ID or before_source != TOOL_SOURCE_IMAGE_ID:
        raise MinimalQualificationError("local base or tool-source image identity drifted")
    discovery = _discover_dependencies(docker, repo_root=repo_root)
    with (
        tempfile.TemporaryDirectory(prefix="eaaef-minimal-overlay-") as raw_overlay,
        tempfile.TemporaryDirectory(prefix="eaaef-minimal-context-") as raw_context,
    ):
        overlay = Path(raw_overlay)
        context = Path(raw_context)
        (overlay / "opt/eaaef/bin").mkdir(parents=True)
        (overlay / "opt/codex-home").mkdir(parents=True)
        normalized = source_date_epoch * 1_000_000_000
        shared._stage_binary(
            bindings[0], overlay / "opt/eaaef/bin/codex", normalized_mtime_ns=normalized
        )
        shared._stage_binary(
            bindings[1], overlay / "opt/eaaef/bin/grok", normalized_mtime_ns=normalized
        )
        _stage_closure(
            docker,
            repo_root=repo_root,
            overlay=overlay,
            dependencies=discovery["dependencies"],
        )
        records = _closure_records(overlay)
        validation_toolchain = _validation_toolchain_manifest(overlay, records)
        manifest = {
            "schema": INPUT_SCHEMA,
            "source_date_epoch": source_date_epoch,
            "base_image": {"reference": BASE_REFERENCE, "image_id": BASE_IMAGE_ID},
            "tool_source_image": {
                "reference": TOOL_SOURCE_REFERENCE,
                "image_id": TOOL_SOURCE_IMAGE_ID,
                "copy_container_started": False,
                "dependency_discovery_container_started": True,
            },
            "native_binaries": [asdict(binding) for binding in bindings],
            "copied_paths": [list(item) for item in COPY_PATHS],
            "skipped_paths": sorted(SKIPPED_RELATIVE_PATHS),
            "skipped_prefixes": list(SKIPPED_PREFIXES),
            "dependency_discovery": discovery,
            "closure": records,
            "python_validation_toolchain": validation_toolchain,
            "authority": {
                "signed": False,
                "network_authorized": False,
                "provider_authorized": False,
                "validation_dependencies_admitted": (
                    "pending_exact_in_container_runtime_smoke"
                ),
                "worker_capacity": 0,
            },
        }
        manifest_bytes = _canonical(manifest) + b"\n"
        if len(manifest_bytes) > MAXIMUM_MANIFEST_BYTES:
            raise MinimalQualificationError("closure manifest exceeded its byte bound")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        rootfs = context / "worker-rootfs-minimal.tar"
        rootfs_hash, rootfs_bytes = _canonical_rootfs_tar(
            rootfs,
            overlay=overlay,
            records=records,
            manifest_bytes=manifest_bytes,
            source_date_epoch=source_date_epoch,
        )
        os.utime(context, ns=(normalized, normalized), follow_symlinks=False)
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
            f"BASE_IMAGE={BASE_REFERENCE}",
            "--build-arg",
            f"CODEX_SHA256={bindings[0].sha256}",
            "--build-arg",
            f"GROK_SHA256={bindings[1].sha256}",
            "--build-arg",
            f"INPUT_MANIFEST_SHA256={manifest_hash}",
            "--build-arg",
            f"ROOTFS_TAR_SHA256={rootfs_hash}",
            "--build-arg",
            f"SOURCE_DATE_EPOCH={source_date_epoch}",
            "--build-arg",
            f"TOOL_SOURCE_IMAGE_ID={TOOL_SOURCE_IMAGE_ID}",
            "--build-arg",
            (
                "VALIDATION_TOOLCHAIN_SHA256="
                + validation_toolchain["content_cid"].removeprefix("sha256:")
            ),
            "-f",
            str(containerfile),
            "-t",
            image_tag,
            str(context),
        ]
        image_ids: list[str] = []
        output_hashes: list[str] = []
        for _attempt in range(2):
            os.utime(rootfs, ns=(normalized, normalized), follow_symlinks=False)
            os.utime(context, ns=(normalized, normalized), follow_symlinks=False)
            result = shared._run(command, cwd=repo_root, timeout=900)
            image_ids.append(_image_id(docker, image_tag, repo_root=repo_root))
            output_hashes.append(
                "sha256:" + hashlib.sha256(result.stdout.encode()).hexdigest()
            )
    after_base = _image_id(docker, BASE_REFERENCE, repo_root=repo_root)
    after_source = _image_id(docker, TOOL_SOURCE_REFERENCE, repo_root=repo_root)
    if after_base != before_base or after_source != before_source:
        raise MinimalQualificationError("local source image identity changed during build")
    image_id = _image_id(docker, image_tag, repo_root=repo_root)
    return {
        "attempted": True,
        "succeeded": True,
        "base_image_reference": BASE_REFERENCE,
        "base_image_id_before": before_base,
        "base_image_id_after": after_base,
        "tool_source_reference": TOOL_SOURCE_REFERENCE,
        "tool_source_image_id_before": before_source,
        "tool_source_image_id_after": after_source,
        "dependency_discovery_container_started": True,
        "source_copy_container_started": False,
        "image_id": image_id,
        "clean_build_image_ids": image_ids,
        "clean_build_attempts": len(image_ids),
        "clean_build_reproducible": image_ids == [image_id, image_id],
        "input_manifest_sha256": "sha256:" + manifest_hash,
        "input_manifest_bytes": len(manifest_bytes),
        "rootfs_tar_sha256": "sha256:" + rootfs_hash,
        "rootfs_tar_bytes": rootfs_bytes,
        "closure_entries": len(records),
        "python_validation_toolchain": validation_toolchain,
        "context_inputs": ["worker-rootfs-minimal.tar"],
        "context_retained": False,
        "network": "none",
        "pull": False,
        "cache": False,
        "package_install": False,
        "provider_auth_accessed": False,
        "stdout_sha256": output_hashes,
    }


def _probe_program() -> str:
    tools = {
        "python": ["/usr/bin/python3", "--version"],
        "git": ["/usr/bin/git", "--version"],
        "rg": ["/usr/local/bin/rg", "--version"],
        "codex": ["/opt/eaaef/bin/codex", "--version"],
        "grok": ["/opt/eaaef/bin/grok", "--version"],
    }
    program = r'''import contextlib,hashlib,importlib.metadata,io,json,os,pathlib,ssl,sqlite3,stat,subprocess,sys
tools=__EAAEF_TOOLS__;expected_versions=__EAAEF_VALIDATION_VERSIONS__;credential_paths=__EAAEF_CREDENTIAL_PATHS__;versions={}
for name,argv in tools.items():
 p=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=20);versions[name]={'path':argv[0],'returncode':p.returncode,'version':(p.stdout.strip() or p.stderr.strip()).splitlines()[0] if (p.stdout.strip() or p.stderr.strip()) else ''}
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def cid(value):return 'sha256:'+hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def meta(path):
 s=os.stat(path,follow_symlinks=False);return {'gid':s.st_gid,'mode':format(stat.S_IMODE(s.st_mode),'04o'),'regular':stat.S_ISREG(s.st_mode),'size':s.st_size,'uid':s.st_uid}
def readable(path):
 try:
  with open(path,'rb'):return True
 except (IsADirectoryError,FileNotFoundError,PermissionError,OSError):return False
pathlib.Path('/tmp/repo').mkdir();subprocess.run(['/usr/bin/git','init','-q','/tmp/repo'],check=True,timeout=20);pathlib.Path('/tmp/repo/a.py').write_text('x=1\n');subprocess.run(['/usr/bin/git','-C','/tmp/repo','add','a.py'],check=True,timeout=20);git_ok=subprocess.run(['/usr/bin/git','-C','/tmp/repo','diff','--cached','--quiet'],check=False,timeout=20).returncode==1
python_ok=hashlib.sha256(b'x').hexdigest().startswith('2d71') and sqlite3.connect(':memory:').execute('select 1').fetchone()==(1,) and bool(ssl.OPENSSL_VERSION)
denied=False
try:pathlib.Path('/eaaef-root-write-probe').write_text('x')
except OSError:denied=True
manifest=json.loads(pathlib.Path('/opt/eaaef/minimal-candidate-inputs.json').read_bytes());toolchain=manifest['python_validation_toolchain'];toolchain_without_cid=dict(toolchain);toolchain_cid=toolchain_without_cid.pop('content_cid');manifest_ok=toolchain_cid==cid(toolchain_without_cid);checked_entries=0
for component in toolchain['components']:
 component_without_cid=dict(component);component_cid=component_without_cid.pop('content_cid');manifest_ok=manifest_ok and component_cid==cid(component_without_cid)
 for record in component['closure_entries']:
  path='/'+record['path'];checked_entries+=1
  try:s=os.lstat(path)
  except OSError:manifest_ok=False;continue
  manifest_ok=manifest_ok and s.st_uid==record['uid'] and s.st_gid==record['gid'] and format(stat.S_IMODE(s.st_mode),'04o')==record['mode']
  if record['type']=='directory':manifest_ok=manifest_ok and stat.S_ISDIR(s.st_mode)
  elif record['type']=='symlink':manifest_ok=manifest_ok and stat.S_ISLNK(s.st_mode) and os.readlink(path)==record['target']
  else:manifest_ok=manifest_ok and stat.S_ISREG(s.st_mode) and s.st_size==record['size'] and sha(path)==record['sha256']
for record in toolchain['entrypoints']:
 path='/'+record['path'];checked_entries+=1
 try:s=os.lstat(path)
 except OSError:manifest_ok=False;continue
 manifest_ok=manifest_ok and s.st_uid==record['uid'] and s.st_gid==record['gid'] and format(stat.S_IMODE(s.st_mode),'04o')==record['mode']
 if record['type']=='symlink':manifest_ok=manifest_ok and stat.S_ISLNK(s.st_mode) and os.readlink(path)==record['target']
 else:manifest_ok=manifest_ok and stat.S_ISREG(s.st_mode) and s.st_size==record['size'] and sha(path)==record['sha256']
validation_root=pathlib.Path('/opt/eaaef/python-validation');startup_files=sorted(str(p.relative_to(validation_root)) for p in validation_root.rglob('*') if p.suffix=='.pth' or p.name in ('sitecustomize.py','usercustomize.py'));site_loaded_before='site' in sys.modules;path_before=list(sys.path);sys.path.insert(0,str(validation_root));validation_versions={name:importlib.metadata.version(name) for name in expected_versions};import pytest,cryptography,duckdb
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
key=Ed25519PrivateKey.generate();message=b'eaaef-validation';key.public_key().verify(key.sign(message),message);connection=duckdb.connect(':memory:');duckdb_ok=connection.execute('select 1').fetchone()==(1,);connection.close();imported_paths={'pytest':pytest.__file__,'cryptography':cryptography.__file__,'duckdb':duckdb.__file__};imports_bounded=all(os.path.commonpath((str(validation_root),os.path.realpath(path)))==str(validation_root) for path in imported_paths.values())
smoke=pathlib.Path('/tmp/eaaef_validation_smoke.py');smoke.write_text("import importlib.metadata\nimport duckdb\nfrom cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey\ndef test_exact_validation_closure():\n assert importlib.metadata.version('pytest') == '9.0.3'\n assert importlib.metadata.version('duckdb') == '1.5.2'\n key=Ed25519PrivateKey.generate(); value=b'eaaef'; key.public_key().verify(key.sign(value),value)\n connection=duckdb.connect(':memory:'); assert connection.execute('select 7').fetchone() == (7,); connection.close()\n")
smoke_result=subprocess.run(['/opt/eaaef/bin/python3','-m','pytest','-q','-p','no:cacheprovider','--confcutdir=/tmp',str(smoke)],cwd='/tmp',stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=60)
script=pathlib.Path('/tmp/eaaef_validation_script.py');script.write_text("import importlib.metadata,json\nprint(json.dumps({'pytest':importlib.metadata.version('pytest'),'duckdb':importlib.metadata.version('duckdb'),'cryptography':importlib.metadata.version('cryptography')},sort_keys=True,separators=(',',':')))\n");script_result=subprocess.run(['/opt/eaaef/bin/python',str(script)],cwd='/tmp',stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=20)
validation={'approved_import_root':str(validation_root),'closure_entries_checked':checked_entries,'direct_script_returncode':script_result.returncode,'direct_script_stdout_sha256':'sha256:'+hashlib.sha256(script_result.stdout.encode()).hexdigest(),'driver_isolation_argv':toolchain['isolation_argv_prefix'],'duckdb_query':duckdb_ok,'expected_versions':expected_versions,'imported_module_paths':imported_paths,'imports_bounded_to_approved_root':imports_bounded,'manifest_verified':manifest_ok,'observed_versions':validation_versions,'path_before':path_before,'plugin_autoload_disabled':os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD')=='1','pytest_smoke_returncode':smoke_result.returncode,'pytest_stdout_sha256':'sha256:'+hashlib.sha256(smoke_result.stdout.encode()).hexdigest(),'pytest_stderr_sha256':'sha256:'+hashlib.sha256(smoke_result.stderr.encode()).hexdigest(),'site_loaded_before_validation':site_loaded_before,'startup_files':startup_files,'sys_flags':{'dont_write_bytecode':sys.flags.dont_write_bytecode,'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'no_user_site':sys.flags.no_user_site}}
print(json.dumps({'ca_sha256':sha('/etc/ssl/certs/ca-certificates.crt'),'credential_paths_readable':[p for p in credential_paths if readable(p)],'docker_socket_present':pathlib.Path('/var/run/docker.sock').exists(),'environment':dict(os.environ),'file_metadata':{'codex':meta('/opt/eaaef/bin/codex'),'grok':meta('/opt/eaaef/bin/grok'),'manifest':meta('/opt/eaaef/minimal-candidate-inputs.json')},'gid':os.getgid(),'git_worktree_probe':git_ok,'hashes':{'codex':sha('/opt/eaaef/bin/codex'),'grok':sha('/opt/eaaef/bin/grok'),'manifest':sha('/opt/eaaef/minimal-candidate-inputs.json')},'python_stdlib_probe':python_ok,'python_validation':validation,'root_write_denied':denied,'tools':versions,'uid':os.getuid()},sort_keys=True,separators=(',',':')))
'''
    return (
        program.replace("__EAAEF_TOOLS__", repr(tools))
        .replace(
            "__EAAEF_VALIDATION_VERSIONS__",
            repr(PYTHON_VALIDATION_EXPECTED_VERSIONS),
        )
        .replace(
            "__EAAEF_CREDENTIAL_PATHS__",
            repr(list(shared.KNOWN_CREDENTIAL_PATHS)),
        )
    )


def _probe(
    docker: str, *, repo_root: Path, image_tag: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    container_id = shared._run(
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
            "256",
            "--cpus",
            "2",
            "--memory",
            str(4 * 1024**3),
            "--memory-swap",
            str(4 * 1024**3),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=0700,uid=65532,gid=65532",
            "--user",
            NONROOT_USER,
            image_tag,
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            _probe_program(),
        ],
        cwd=repo_root,
    ).stdout.strip()
    if not container_id:
        raise MinimalQualificationError("probe container identity is empty")
    try:
        inspected = shared._docker_json(
            docker, ["container", "inspect", container_id], cwd=repo_root
        )
        started = shared._run(
            [docker, "start", "-a", container_id], cwd=repo_root, timeout=120
        )
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise MinimalQualificationError("probe inspection is invalid")
        try:
            observed = json.loads(started.stdout)
        except json.JSONDecodeError as exc:
            raise MinimalQualificationError("probe output is invalid") from exc
        if not isinstance(observed, Mapping):
            raise MinimalQualificationError("probe output is not an object")
        return inspected[0], observed
    finally:
        shared._run([docker, "rm", "-f", container_id], cwd=repo_root, check=False)


def _evaluate_probe(
    container: Mapping[str, Any],
    observed: Mapping[str, Any],
    bindings: Sequence[Any],
    *,
    manifest_sha256: str,
    validation_toolchain: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    blockers: list[str] = []
    host = container.get("HostConfig") or {}
    config = container.get("Config") or {}
    tmpfs = host.get("Tmpfs") or {}
    if not (
        host.get("ReadonlyRootfs") is True
        and host.get("NetworkMode") == "none"
        and host.get("CapDrop") == ["ALL"]
        and "no-new-privileges" in (host.get("SecurityOpt") or [])
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
        and set(tmpfs) == {"/tmp"}
        and all(token in str(tmpfs["/tmp"]) for token in ("noexec", "nosuid", "nodev", "size=16m", "uid=65532", "gid=65532"))
        and config.get("User") == NONROOT_USER
        and observed.get("uid") == 65532
        and observed.get("gid") == 65532
        and observed.get("environment") == RUNTIME_ENVIRONMENT
        and observed.get("root_write_denied") is True
        and observed.get("docker_socket_present") is False
    ):
        blockers.append("minimal_hardening_probe_failed")
    if observed.get("credential_paths_readable"):
        blockers.append("embedded_credential_path_detected")
    if shared._sensitive_environment(config.get("Env") or []):
        blockers.append("embedded_sensitive_environment_detected")
    if observed.get("python_stdlib_probe") is not True:
        blockers.append("python_runtime_closure_incomplete")
    if observed.get("git_worktree_probe") is not True:
        blockers.append("git_runtime_closure_incomplete")
    validation = observed.get("python_validation")
    expected_validation_entries = sum(
        len(component.get("closure_entries") or [])
        for component in validation_toolchain.get("components") or []
        if isinstance(component, Mapping)
    ) + len(validation_toolchain.get("entrypoints") or [])
    if not (
        isinstance(validation, Mapping)
        and validation.get("approved_import_root")
        == PYTHON_VALIDATION_IMPORT_ROOT
        and validation.get("closure_entries_checked")
        == expected_validation_entries
        and validation.get("direct_script_returncode") == 0
        and validation.get("driver_isolation_argv")
        == validation_toolchain.get("isolation_argv_prefix")
        and validation.get("duckdb_query") is True
        and validation.get("expected_versions")
        == PYTHON_VALIDATION_EXPECTED_VERSIONS
        and validation.get("observed_versions")
        == PYTHON_VALIDATION_EXPECTED_VERSIONS
        and validation.get("imports_bounded_to_approved_root") is True
        and validation.get("manifest_verified") is True
        and validation.get("plugin_autoload_disabled") is True
        and validation.get("pytest_smoke_returncode") == 0
        and validation.get("site_loaded_before_validation") is False
        and validation.get("startup_files") == []
        and validation.get("sys_flags")
        == {
            "dont_write_bytecode": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
        }
        and not any(
            "site-packages" in str(path) or "dist-packages" in str(path)
            for path in validation.get("path_before") or []
        )
    ):
        blockers.append("project_validation_dependencies_not_admitted")
    tools = observed.get("tools") or {}
    versions: dict[str, str] = {}
    for name, expected_path in EXPECTED_TOOL_PATHS.items():
        record = tools.get(name) if isinstance(tools, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or record.get("path") != expected_path
            or record.get("returncode") != 0
            or not record.get("version")
        ):
            blockers.append(f"minimal_tool_{name}_unavailable")
        else:
            versions[name] = str(record["version"])
    hashes = observed.get("hashes") or {}
    metadata = observed.get("file_metadata") or {}
    for binding in bindings:
        expected = {
            "gid": 65532,
            "mode": "0555",
            "regular": True,
            "size": binding.size,
            "uid": 65532,
        }
        if hashes.get(binding.name) != binding.sha256 or metadata.get(binding.name) != expected:
            blockers.append(f"embedded_{binding.name}_identity_drift")
        if versions.get(binding.name) != binding.version:
            blockers.append(f"embedded_{binding.name}_version_drift")
    manifest_hash = manifest_sha256.removeprefix("sha256:")
    manifest_record = metadata.get("manifest") or {}
    if (
        hashes.get("manifest") != manifest_hash
        or manifest_record.get("uid") != 65532
        or manifest_record.get("gid") != 65532
        or manifest_record.get("mode") != "0444"
        or manifest_record.get("regular") is not True
    ):
        blockers.append("embedded_minimal_manifest_identity_drift")
    ca_hash = str(observed.get("ca_sha256") or "")
    if len(ca_hash) != 64:
        blockers.append("ca_bundle_unavailable")
    return list(dict.fromkeys(blockers)), versions


def _adjudicate_credential_scan(scan: Mapping[str, Any]) -> dict[str, Any]:
    """Separate exact static-code markers from unresolved material signals.

    This does not remove or rewrite the underlying whole-filesystem scan.  It
    adds an exact-digest review layer whose closed allowlist fails on all
    drift.  No matched bytes or potential secret values enter the report.
    """

    result = dict(scan)
    findings = scan.get("findings")
    if not isinstance(findings, list):
        raise MinimalQualificationError("credential scan findings are invalid")
    static_markers: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for item in findings:
        if not isinstance(item, Mapping):
            raise MinimalQualificationError("credential scan finding is invalid")
        path = str(item.get("path") or "")
        digest = str(item.get("sha256") or "")
        detectors = sorted(str(value) for value in (item.get("detectors") or []))
        admitted = KNOWN_STATIC_MARKER_FINDINGS.get((path, digest))
        if admitted is not None and detectors == sorted(admitted["detectors"]):
            static_markers.append(
                {
                    "path": path,
                    "sha256": digest,
                    "detectors": detectors,
                    "classification": admitted["classification"],
                    **(
                        {"matched_literal_sha256": admitted["matched_literal_sha256"]}
                        if "matched_literal_sha256" in admitted
                        else {}
                    ),
                }
            )
        else:
            unresolved.append(dict(item))
    result["pattern_finding_files"] = int(scan.get("finding_files") or len(findings))
    result["static_marker_files"] = len(static_markers)
    result["static_marker_adjudications"] = static_markers
    result["credential_material_finding_files"] = len(unresolved)
    result["credential_material_findings"] = unresolved
    result["adjudication_policy"] = (
        "exact-path-digest-detector-allowlist; any drift remains unresolved"
    )
    result["adjudication_independently_signed"] = False
    return result


def _spdx(
    *,
    image_id: str,
    image_tag: str,
    bindings: Sequence[Any],
    versions: Mapping[str, str],
    rootfs_sha256: str,
    source_date_epoch: int,
    validation_toolchain: Mapping[str, Any],
) -> bytes:
    image_hash = image_id.removeprefix("sha256:")
    packages = [
        {
            "SPDXID": "SPDXRef-MinimalWorkerImage",
            "name": image_tag,
            "versionInfo": image_id,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": image_hash}],
            "copyrightText": "NOASSERTION",
        },
        {
            "SPDXID": "SPDXRef-UbuntuBase",
            "name": BASE_REFERENCE,
            "versionInfo": BASE_IMAGE_ID,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": BASE_IMAGE_ID.removeprefix('sha256:')}],
            "copyrightText": "NOASSERTION",
        },
        {
            "SPDXID": "SPDXRef-ToolSource",
            "name": TOOL_SOURCE_REFERENCE,
            "versionInfo": TOOL_SOURCE_IMAGE_ID,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": TOOL_SOURCE_IMAGE_ID.removeprefix('sha256:')}],
            "copyrightText": "NOASSERTION",
        },
        {
            "SPDXID": "SPDXRef-ClosureArchive",
            "name": "worker-rootfs-minimal.tar",
            "versionInfo": rootfs_sha256,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [{"algorithm": "SHA256", "checksumValue": rootfs_sha256.removeprefix('sha256:')}],
            "copyrightText": "NOASSERTION",
        },
    ]
    for index, binding in enumerate(bindings, start=1):
        packages.append(
            {
                "SPDXID": f"SPDXRef-Native-{index}",
                "name": binding.name,
                "versionInfo": binding.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "checksums": [{"algorithm": "SHA256", "checksumValue": binding.sha256}],
                "copyrightText": "NOASSERTION",
            }
        )
    validation_spdx_ids: list[str] = []
    for index, component in enumerate(
        validation_toolchain.get("components") or [], start=1
    ):
        if not isinstance(component, Mapping):
            raise MinimalQualificationError("SPDX validation component is invalid")
        content_cid = str(component.get("content_cid") or "")
        if not content_cid.startswith("sha256:"):
            raise MinimalQualificationError(
                "SPDX validation component identity is invalid"
            )
        spdx_id = f"SPDXRef-PythonValidation-{index}"
        validation_spdx_ids.append(spdx_id)
        packages.append(
            {
                "SPDXID": spdx_id,
                "name": str(component.get("name") or ""),
                "versionInfo": str(component.get("version") or ""),
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": content_cid.removeprefix("sha256:"),
                    }
                ],
                "copyrightText": "NOASSERTION",
                "comment": (
                    "Exact file identities and normalized uid/gid/mode are "
                    "bound by the embedded Python validation-toolchain manifest."
                ),
            }
        )
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-MinimalWorkerImage",
        },
        *[
            {
                "spdxElementId": "SPDXRef-MinimalWorkerImage",
                "relationshipType": "CONTAINS" if item.startswith("SPDXRef-Native") or item == "SPDXRef-ClosureArchive" else "DESCENDANT_OF",
                "relatedSpdxElement": item,
            }
            for item in (
                "SPDXRef-UbuntuBase",
                "SPDXRef-ToolSource",
                "SPDXRef-ClosureArchive",
                *[f"SPDXRef-Native-{i}" for i in range(1, len(bindings) + 1)],
                *validation_spdx_ids,
            )
        ],
    ]
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "EAAEF minimal implementation-worker unsigned candidate SBOM",
        "documentNamespace": "urn:ipfs-accelerate:eaaef:minimal-worker-sbom:" + image_hash,
        "creationInfo": {
            "created": dt.datetime.fromtimestamp(source_date_epoch, tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": ["Tool: qualify_external_agent_implementation_worker_minimal_image.py"],
        },
        "documentComment": (
            "Exact image/base/tool-source/native/closure identities. The closure's "
            "file-level identities are in its embedded content-addressed manifest; "
            "this detached package SBOM does not claim transitive license analysis. "
            f"Observed tools: {json.dumps(dict(sorted(versions.items())), sort_keys=True)}. "
            "The exact isolated Python validation closure is identified by "
            f"{validation_toolchain.get('content_cid')}. "
            "Unsigned candidate; worker capacity is zero."
        ),
        "packages": packages,
        "relationships": relationships,
    }
    payload = _canonical(document) + b"\n"
    if len(payload) > MAXIMUM_SBOM_BYTES:
        raise MinimalQualificationError("SPDX document exceeded its bound")
    return payload


def qualify(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    repo_root = Path(args.repo_root).resolve(strict=True)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "source_date_epoch": args.source_date_epoch,
        "decision": "no_go",
        "status": "host_capability_no_go",
        "workload_class": "offline_minimal_implementation_worker_image_diagnostic",
        "task_dispatch_admitted": False,
        "worker_capacity": 0,
        "maximum_parallel_workers": 0,
        "network_authorized": False,
        "provider_authorized": False,
        "validation_dependencies_admitted": False,
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
            "project_validation_dependencies_not_admitted",
            "network_authorization_not_independently_signed",
            "provider_authorization_not_independently_signed",
            "image_and_sbom_not_independently_signed",
            (
                "rootless_runtime_or_independently_approved_"
                "rootful_control_not_admitted"
            ),
        ],
    }
    bindings = [
        shared._bind_source("codex", Path(args.codex_binary), repo_root=repo_root),
        shared._bind_source("grok", Path(args.grok_binary), repo_root=repo_root),
    ]
    report["sources"] = [asdict(binding) for binding in bindings]
    docker = shutil.which(args.runtime)
    if docker is None:
        report["blockers"].append("container_runtime_unavailable")
        report["report_cid"] = _cid(report)
        return report, b""
    build = _build(
        docker,
        repo_root=repo_root,
        image_tag=args.image_tag,
        bindings=bindings,
        source_date_epoch=args.source_date_epoch,
    )
    report["build"] = build
    inspected = shared._docker_json(
        docker, ["image", "inspect", args.image_tag], cwd=repo_root
    )
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise MinimalQualificationError("built image inspection is invalid")
    image = inspected[0]
    config = image.get("Config") or {}
    labels = config.get("Labels") or {}
    if (
        image.get("Id") != build["image_id"]
        or image.get("Os") != "linux"
        or image.get("Architecture") != "arm64"
        or config.get("User") != NONROOT_USER
        or labels.get("org.ipfs-accelerate.eaaef.worker-capacity") != "0"
        or labels.get("org.ipfs-accelerate.eaaef.unsigned") != "true"
        or labels.get("org.ipfs-accelerate.eaaef.codex.sha256") != bindings[0].sha256
        or labels.get("org.ipfs-accelerate.eaaef.grok.sha256") != bindings[1].sha256
        or labels.get("org.ipfs-accelerate.eaaef.input-manifest.sha256") != str(build["input_manifest_sha256"]).removeprefix("sha256:")
        or labels.get("org.ipfs-accelerate.eaaef.rootfs-tar.sha256") != str(build["rootfs_tar_sha256"]).removeprefix("sha256:")
        or labels.get("org.ipfs-accelerate.eaaef.tool-source.digest") != TOOL_SOURCE_IMAGE_ID
        or labels.get(
            "org.ipfs-accelerate.eaaef.python-validation-toolchain.sha256"
        )
        != str(build["python_validation_toolchain"]["content_cid"]).removeprefix(
            "sha256:"
        )
        or labels.get("org.opencontainers.image.base.digest") != BASE_IMAGE_ID
    ):
        raise MinimalQualificationError("built image identity or labels drifted")
    try:
        created = dt.datetime.fromisoformat(str(image.get("Created") or "").replace("Z", "+00:00"))
        created_epoch = int(created.timestamp())
    except (ValueError, OverflowError) as exc:
        raise MinimalQualificationError("image creation timestamp is invalid") from exc
    build["image_created_at"] = str(image.get("Created"))
    build["source_date_epoch_applied"] = created_epoch == args.source_date_epoch
    if not build["clean_build_reproducible"]:
        report["blockers"].append("clean_offline_build_not_reproducible")
    if not build["source_date_epoch_applied"]:
        report["blockers"].append("source_date_epoch_not_applied")
    container, observed = _probe(docker, repo_root=repo_root, image_tag=args.image_tag)
    report["diagnostic_container_process_started"] = True
    blockers, versions = _evaluate_probe(
        container,
        observed,
        bindings,
        manifest_sha256=str(build["input_manifest_sha256"]),
        validation_toolchain=build["python_validation_toolchain"],
    )
    if "project_validation_dependencies_not_admitted" not in blockers:
        report["blockers"] = [
            blocker
            for blocker in report["blockers"]
            if blocker != "project_validation_dependencies_not_admitted"
        ]
        report["validation_dependencies_admitted"] = True
    report["blockers"].extend(blockers)
    try:
        scan = shared._scan_exported_filesystem(
            docker=docker, repo_root=repo_root, image_tag=args.image_tag
        )
    except (OSError, ValueError, subprocess.TimeoutExpired, tarfile.TarError) as exc:
        scan = {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-bounded-image-credential-scan@1",
            "mode": "merged-docker-export-tar-stream",
            "complete": False,
            "stop_reason": str(exc)[:4096],
            "findings": [],
            "raw_secret_values_recorded": False,
        }
        report["blockers"].append("bounded_credential_scan_incomplete")
    else:
        scan = _adjudicate_credential_scan(scan)
        if scan.get("credential_material_finding_files"):
            report["blockers"].append("credential_like_material_detected")
    report["credential_scan"] = scan
    report["probe"] = {
        "hardening_valid": "minimal_hardening_probe_failed" not in blockers,
        "tool_versions": versions,
        "python_validation": observed.get("python_validation"),
        "observed": observed,
    }
    sbom = _spdx(
        image_id=str(build["image_id"]),
        image_tag=args.image_tag,
        bindings=bindings,
        versions=versions,
        rootfs_sha256=str(build["rootfs_tar_sha256"]),
        source_date_epoch=args.source_date_epoch,
        validation_toolchain=build["python_validation_toolchain"],
    )
    report["sbom"] = {
        "format": "spdx-json",
        "spdx_version": "SPDX-2.3",
        "files_analyzed": False,
        "content_cid": "sha256:" + hashlib.sha256(sbom).hexdigest(),
        "bytes": len(sbom),
        "subject_image_id": build["image_id"],
        "python_validation_toolchain_cid": build["python_validation_toolchain"][
            "content_cid"
        ],
    }
    report["blockers"] = list(dict.fromkeys(report["blockers"]))
    technical = {
        "clean_offline_build_not_reproducible",
        "source_date_epoch_not_applied",
        "minimal_hardening_probe_failed",
        "embedded_credential_path_detected",
        "embedded_sensitive_environment_detected",
        "python_runtime_closure_incomplete",
        "git_runtime_closure_incomplete",
        "embedded_codex_identity_drift",
        "embedded_grok_identity_drift",
        "embedded_codex_version_drift",
        "embedded_grok_version_drift",
        "embedded_minimal_manifest_identity_drift",
        "ca_bundle_unavailable",
        "bounded_credential_scan_incomplete",
        "credential_like_material_detected",
        "project_validation_dependencies_not_admitted",
    }
    report["status"] = (
        "host_capability_no_go"
        if technical.intersection(report["blockers"])
        else "closed_unsigned_minimal_candidate_for_independent_review"
    )
    report["report_cid"] = _cid(report)
    return report, sbom


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime", choices=("docker",), default="docker")
    parser.add_argument("--codex-binary", type=Path, required=True)
    parser.add_argument("--grok-binary", type=Path, required=True)
    parser.add_argument(
        "--image-tag",
        default="eaaef-implementation-worker:minimal-unsigned-20260818",
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
            shared._atomic_write(args.sbom, sbom)
        shared._atomic_write(args.report, _canonical(report) + b"\n")
    except (
        OSError,
        MinimalQualificationError,
        shared.QualificationError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"qualification_error: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
