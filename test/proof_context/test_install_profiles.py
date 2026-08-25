"""PCCE-052 immutable accelerator install-profile qualification."""

from __future__ import annotations

import csv
import email
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATE_EPOCH = "0"
CORE_REQUIREMENTS = (
    "ipfs_datasets_py==0.2.0",
    "ipfs_kit_py==0.3.0",
    "mcp-plus-plus-contracts==0.1.0",
)
CORE_EXTRA = "proof-context"
PROFILE_REQUIREMENTS = {
    "verification": ("jsonschema>=4.22,<5",),
    "codex": (),
    "local-model": (
        "llama-cpp-python[server]>=0.3.10,<0.4",
        "huggingface-hub>=0.24,<1",
    ),
    "evaluation": (
        "jsonschema>=4.22,<5",
        "llama-cpp-python[server]>=0.3.10,<0.4",
        "huggingface-hub>=0.24,<1",
        "pytest>=8,<9",
    ),
}
CONSOLE_TARGET = "ipfs_accelerate_py.proof_context.cli.__main__:main"


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    contents_manifest_sha256: str
    member_count: int


@dataclass(frozen=True)
class ArtifactPair:
    wheel: Artifact
    sdist: Artifact


def _run(
    args: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {args!r}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def _build_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "HOME": os.devnull,
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "TZ": "UTC",
        }
    )
    return environment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_manifest(path: Path) -> tuple[str, int]:
    records: list[dict[str, object]] = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                archive_path = PurePosixPath(member.filename)
                assert not archive_path.is_absolute()
                assert ".." not in archive_path.parts
                if member.is_dir():
                    continue
                payload = archive.read(member)
                records.append(
                    {
                        "path": member.filename,
                        "mode": (member.external_attr >> 16) & 0o777,
                        "size": member.file_size,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    else:
        with tarfile.open(path) as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                archive_path = PurePosixPath(member.name)
                assert not archive_path.is_absolute()
                assert ".." not in archive_path.parts
                record: dict[str, object] = {
                    "path": member.name,
                    "type": member.type.decode("ascii"),
                    "mode": member.mode,
                    "size": member.size,
                }
                if member.isfile():
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    record["sha256"] = hashlib.sha256(extracted.read()).hexdigest()
                elif member.issym() or member.islnk():
                    record["linkname"] = member.linkname
                records.append(record)
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), len(records)


def _artifact(path: Path) -> Artifact:
    contents_manifest_sha256, member_count = _archive_manifest(path)
    return Artifact(path, _sha256_file(path), contents_manifest_sha256, member_count)


def _linked_source_copy(destination: Path) -> Path:
    def link_or_copy(source: str, target: str) -> str:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return target

    shutil.copytree(
        PROJECT_ROOT,
        destination,
        copy_function=link_or_copy,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".ruff_cache",
            "*.egg-info",
            "__pycache__",
            "build",
            "dist",
        ),
    )
    return destination


def _build_artifacts(source: Path, output: Path) -> ArtifactPair:
    output.mkdir()
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--sdist",
            "--outdir",
            str(output),
            str(source),
        ],
        cwd=output.parent,
        env=_build_environment(),
    )
    (wheel,) = output.glob("*.whl")
    (sdist,) = output.glob("*.tar.gz")
    return ArtifactPair(_artifact(wheel), _artifact(sdist))


def _restrict_build_tree_modes(root: Path) -> None:
    """Model a warm supervisor build tree created under umask 077."""

    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(0o700 if mode & 0o111 else 0o600)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> ArtifactPair:
    root = tmp_path_factory.mktemp("pcce-052-artifacts")
    source = _linked_source_copy(root / "source")
    first = _build_artifacts(source, root / "first")
    _restrict_build_tree_modes(source / "build")
    second = _build_artifacts(source, root / "second")

    assert first.wheel.path.name == second.wheel.path.name
    assert first.sdist.path.name == second.sdist.path.name
    assert first.wheel.sha256 == second.wheel.sha256
    assert first.sdist.sha256 == second.sdist.sha256
    assert first.wheel.contents_manifest_sha256 == second.wheel.contents_manifest_sha256
    assert first.sdist.contents_manifest_sha256 == second.sdist.contents_manifest_sha256
    return first


def _canonical_requirement(value: str) -> tuple[str, tuple[str, ...], str]:
    requirement = Requirement(value)
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
    )


def _is_bounded(requirement: Requirement) -> bool:
    specifiers = tuple(requirement.specifier)
    lower = any(item.operator in {">", ">=", "~=", "==", "==="} for item in specifiers)
    upper = any(item.operator in {"<", "<=", "==", "==="} for item in specifiers)
    return lower and upper


def _wheel_metadata(wheel: Path) -> tuple[email.message.Message, str, set[str]]:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        (metadata_name,) = (
            name for name in members if name.endswith(".dist-info/METADATA")
        )
        (entry_points_name,) = (
            name for name in members if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
        entry_points = archive.read(entry_points_name).decode("utf-8")
    return metadata, entry_points, members


def test_metadata_declares_closed_bounded_profile_graph() -> None:
    import tomllib

    document = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project = document["project"]
    extras = project["optional-dependencies"]
    inventory = document["tool"]["ipfs-accelerate"]["proof-context-profiles"]

    # Composition keeps the accelerator's established interpreter and dynamic
    # core dependency contract. PCCE's cross-repository authorities are an
    # explicit capability extra instead of replacing the whole project graph.
    assert project["requires-python"] == ">=3.8"
    assert project["dynamic"] == ["dependencies"]
    assert tuple(extras[CORE_EXTRA]) == CORE_REQUIREMENTS
    assert tuple(inventory["core-distributions"]) == CORE_REQUIREMENTS
    for profile, expected in PROFILE_REQUIREMENTS.items():
        assert tuple(extras[profile]) == expected

    assert inventory == {
        "schema": "ipfs-accelerate.proof-context.v0.1/install-profiles@1",
        "core-distributions": list(CORE_REQUIREMENTS),
        "core-extra": CORE_EXTRA,
        "verification-extra": "verification",
        "codex-extra": "codex",
        "codex-command-contract": "codex exec",
        "local-model-extra": "local-model",
        "local-model-inventory": "ipfs_accelerate_py.utils.llama_cpp",
        "local-model-adapter": (
            "ipfs_accelerate_py.proof_context.adapters.command:CommandAdapter"
        ),
        "local-model-command-contract": "local-agent-json-argv@1",
        "evaluation-extra": "evaluation",
        "evaluation-surface": (
            "ipfs_accelerate_py.agent_supervisor.self_hosting:"
            "SelfHostingQualificationHarness"
        ),
        "console-script": "proof-context",
        "console-target": CONSOLE_TARGET,
        "runtime-authority-widened": False,
    }

    pcce_requirements = [
        *extras[CORE_EXTRA],
        *(
            value
            for profile in PROFILE_REQUIREMENTS
            for value in extras[profile]
        ),
    ]
    parsed = [Requirement(value) for value in pcce_requirements]
    assert all(requirement.url is None for requirement in parsed)
    assert all(_is_bounded(requirement) for requirement in parsed)
    assert not any(
        marker in value.lower()
        for value in pcce_requirements
        for marker in ("git+", "@main", "@master", "file:", "../")
    )


def test_existing_local_inventory_is_packaged_without_a_new_provider() -> None:
    from ipfs_accelerate_py.proof_context.adapters.command import (
        COMMAND_CONTRACT,
        CommandAdapter,
    )
    from ipfs_accelerate_py.proof_context.adapters.registry import ADAPTER_NAMES
    from ipfs_accelerate_py.utils import llama_cpp

    assert COMMAND_CONTRACT == "local-agent-json-argv@1"
    assert CommandAdapter.__module__.endswith(".adapters.command")
    assert llama_cpp.DEFAULT_LLAMA_CPP_PORT == 8080
    assert ADAPTER_NAMES == ("codex", "command", "replay", "external-patch")


def test_artifacts_bind_metadata_entrypoint_and_evaluation_surface(
    artifacts: ArtifactPair,
) -> None:
    metadata, entry_points, members = _wheel_metadata(artifacts.wheel.path)
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    proof_context_core = [
        requirement
        for requirement in requirements
        if requirement.marker is not None
        and requirement.marker.evaluate({"extra": CORE_EXTRA})
    ]

    assert metadata["Name"] == "ipfs_accelerate_py"
    assert metadata["Version"] == "0.0.45"
    assert metadata["Requires-Python"] == ">=3.8"
    assert sorted(
        _canonical_requirement(str(item)) for item in proof_context_core
    ) == sorted(
        _canonical_requirement(item) for item in CORE_REQUIREMENTS
    )
    assert all(item.url is None and _is_bounded(item) for item in proof_context_core)
    assert {CORE_EXTRA, *PROFILE_REQUIREMENTS} <= set(
        metadata.get_all("Provides-Extra", [])
    )
    assert f"proof-context = {CONSOLE_TARGET}" in entry_points
    assert "ipfs_accelerate_py/proof_context/cli/__main__.py" in members
    assert "ipfs_accelerate_py/agent_supervisor/self_hosting/harness.py" in members

    for profile, expected in PROFILE_REQUIREMENTS.items():
        selected = [
            requirement
            for requirement in requirements
            if requirement.marker is not None
            and requirement.marker.evaluate({"extra": profile})
        ]
        assert sorted(_canonical_requirement(str(item)) for item in selected) == sorted(
            _canonical_requirement(item) for item in expected
        )

    with zipfile.ZipFile(artifacts.wheel.path) as archive:
        (record_name,) = (
            name for name in members if name.endswith(".dist-info/RECORD")
        )
        rows = {
            name: (digest, size)
            for name, digest, size in csv.reader(
                io.StringIO(archive.read(record_name).decode("utf-8"))
            )
        }
        assert set(rows) == {name for name in members if not name.endswith("/")}

    with tarfile.open(artifacts.sdist.path) as archive:
        names = set(archive.getnames())
    assert any(name.endswith("/pyproject.toml") for name in names)
    assert any(name.endswith("/setup.py") for name in names)
    assert any(name.endswith("/self_hosting/harness.py") for name in names)


def test_clean_wheel_import_entrypoint_and_absent_optionals(
    artifacts: ArtifactPair, tmp_path: Path
) -> None:
    environment_root = tmp_path / "clean-environment"
    venv.EnvBuilder(with_pip=True, clear=True).create(environment_root)
    python = environment_root / "bin" / "python"
    console = environment_root / "bin" / "proof-context"
    environment = _build_environment()
    environment["HOME"] = str(tmp_path / "home")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    environment["XDG_DATA_HOME"] = str(tmp_path / "data")

    _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(artifacts.wheel.path)],
        cwd=tmp_path,
        env=environment,
    )
    smoke = _run([str(console), "--help"], cwd=tmp_path, env=environment)
    assert smoke.stdout.startswith("usage: proof-context")

    code = """
import importlib.util
import json
import sys

for name in ('jsonschema', 'llama_cpp', 'huggingface_hub', 'pytest'):
    assert importlib.util.find_spec(name) is None, name
import ipfs_accelerate_py.proof_context as proof_context
from ipfs_accelerate_py.agent_supervisor.self_hosting import SelfHostingQualificationHarness
from ipfs_accelerate_py.proof_context.cli.__main__ import main
assert proof_context.SCHEMA == 'ipfs-accelerate.proof-context.v0.1'
assert SelfHostingQualificationHarness.__name__ == 'SelfHostingQualificationHarness'
assert callable(main)
assert not {'jsonschema', 'llama_cpp', 'huggingface_hub', 'pytest'} & set(sys.modules)
print(json.dumps({'entrypoint': 'proof-context', 'harness': True}, sort_keys=True))
"""
    imported = _run([str(python), "-I", "-c", code], cwd=tmp_path, env=environment)
    assert json.loads(imported.stdout) == {
        "entrypoint": "proof-context",
        "harness": True,
    }
