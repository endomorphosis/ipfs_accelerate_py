from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import struct
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/qualify_external_agent_implementation_worker_minimal_image.py"
)
CONTAINERFILE_PATH = (
    REPO_ROOT
    / "containers/external-agent/implementation-worker-minimal.Containerfile"
)
SPEC = importlib.util.spec_from_file_location("eaaef_minimal_worker_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


def _static_arm64_elf(path: Path) -> None:
    header = bytearray(64)
    header[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<HHIQQQIHHHHHH", header, 16, 2, 183, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0
    )
    program = bytearray(56)
    struct.pack_into("<IIQQQQQQ", program, 0, 1, 5, 0, 0, 0, 120, 120, 4096)
    path.write_bytes(bytes(header + program))
    path.chmod(0o755)


def _binding(name: str, path: Path, version: str):
    info = path.stat()
    return qualification.shared.SourceBinding(
        name=name,
        absolute_path=str(path),
        uid=info.st_uid,
        gid=info.st_gid,
        mode=f"{stat.S_IMODE(info.st_mode):04o}",
        size=info.st_size,
        device=info.st_dev,
        inode=info.st_ino,
        mtime_ns=info.st_mtime_ns,
        version=version,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        elf_machine="aarch64",
        static=True,
    )


def test_containerfile_is_distinct_minimal_offline_copy_only_candidate():
    text = CONTAINERFILE_PATH.read_text(encoding="utf-8")

    assert not text.startswith("# syntax=")
    assert "ARG BASE_IMAGE=ubuntu:24.04" in text
    assert qualification.BASE_IMAGE_ID in text
    assert 'profile="implementation-worker-minimal-candidate-v1"' in text
    assert 'worker-capacity="0"' in text
    assert "USER 65532:65532" in text
    assert "LD_LIBRARY_PATH=/opt/eaaef/lib" in text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in text
    assert "VALIDATION_TOOLCHAIN_SHA256" in text
    assert text.count("ADD ") == 1
    assert "COPY " not in text
    assert "RUN " not in text
    assert not any(value in text for value in ("apt-get", " apk ", "curl ", "wget "))


def test_safe_relative_and_skip_policy_fail_closed():
    assert qualification._safe_relative("usr/bin/git") == Path("usr/bin/git")
    for value in ("/usr/bin/git", "../git", "usr/../../git", ""):
        with pytest.raises(qualification.MinimalQualificationError):
            qualification._safe_relative(value)
    assert qualification._is_skipped("usr/lib/python3.12/sitecustomize.py")
    assert qualification._is_skipped("usr/lib/python3.12/__pycache__/x.pyc")
    assert not qualification._is_skipped("usr/lib/python3.12/json/__init__.py")


def test_canonical_minimal_tar_is_reproducible_and_normalized(tmp_path: Path):
    overlay = tmp_path / "overlay"
    (overlay / "opt/eaaef/bin").mkdir(parents=True)
    (overlay / "opt/codex-home").mkdir(parents=True)
    executable = overlay / "opt/eaaef/bin/codex"
    executable.write_bytes(b"native")
    executable.chmod(0o755)
    symlink = overlay / "opt/eaaef/bin/current"
    symlink.symlink_to("codex")
    records = qualification._closure_records(overlay)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    arguments = {
        "overlay": overlay,
        "records": records,
        "manifest_bytes": b'{"schema":"test"}\n',
        "source_date_epoch": 1_800_000_000,
    }

    first_identity = qualification._canonical_rootfs_tar(first, **arguments)
    second_identity = qualification._canonical_rootfs_tar(second, **arguments)

    assert first_identity == second_identity
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert members["opt/eaaef/bin/codex"].mode == 0o555
    assert members["opt/eaaef/bin/codex"].uid == 65532
    assert members["opt/codex-home"].mode == 0o700
    assert members["opt/codex-home"].uid == 65532
    assert members["opt/eaaef/bin/current"].issym()
    assert members["opt/eaaef/bin/current"].linkname == "codex"
    assert members["opt/eaaef/minimal-candidate-inputs.json"].uid == 65532
    assert all(member.mtime == 1_800_000_000 for member in members.values())


def test_closure_rejects_unsafe_symlink_and_special_file(tmp_path: Path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "bad").symlink_to("../../outside")

    with pytest.raises(qualification.MinimalQualificationError, match="unsafe symlink"):
        qualification._closure_records(overlay)


def test_exact_static_marker_adjudication_preserves_raw_findings():
    findings = [
        {
            "path": path,
            "sha256": digest,
            "detectors": list(record["detectors"]),
            "size": 1,
        }
        for (path, digest), record in qualification.KNOWN_STATIC_MARKER_FINDINGS.items()
    ]

    result = qualification._adjudicate_credential_scan(
        {"complete": True, "finding_files": len(findings), "findings": findings}
    )

    assert result["findings"] == findings
    assert result["pattern_finding_files"] == len(
        qualification.KNOWN_STATIC_MARKER_FINDINGS
    )
    assert result["static_marker_files"] == len(
        qualification.KNOWN_STATIC_MARKER_FINDINGS
    )
    assert result["credential_material_finding_files"] == 0
    assert result["adjudication_independently_signed"] is False
    assert not any(
        "raw" in item or "value" in item
        for item in result["static_marker_adjudications"]
    )


def test_static_marker_adjudication_fails_on_any_digest_or_detector_drift():
    (path, digest), record = next(iter(qualification.KNOWN_STATIC_MARKER_FINDINGS.items()))
    finding = {
        "path": path,
        "sha256": digest[:-1] + ("0" if digest[-1] != "0" else "1"),
        "detectors": list(record["detectors"]),
        "size": 1,
    }

    result = qualification._adjudicate_credential_scan(
        {"complete": True, "finding_files": 1, "findings": [finding]}
    )

    assert result["static_marker_files"] == 0
    assert result["credential_material_finding_files"] == 1
    assert result["credential_material_findings"] == [finding]


def test_probe_evaluation_requires_exact_hardening_and_runtime_identity(tmp_path: Path):
    codex = tmp_path / "codex"
    grok = tmp_path / "grok"
    _static_arm64_elf(codex)
    _static_arm64_elf(grok)
    bindings = [
        _binding("codex", codex, "codex-cli test"),
        _binding("grok", grok, "grok test"),
    ]
    manifest_hash = "a" * 64
    container = {
        "HostConfig": {
            "ReadonlyRootfs": True,
            "NetworkMode": "none",
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "PidsLimit": 256,
            "NanoCpus": 2_000_000_000,
            "Memory": 4 * 1024**3,
            "MemorySwap": 4 * 1024**3,
            "Privileged": False,
            "PidMode": "",
            "IpcMode": "private",
            "Binds": None,
            "PortBindings": {},
            "Devices": [],
            "DeviceRequests": [],
            "Tmpfs": {
                "/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=0700,uid=65532,gid=65532"
            },
        },
        "Config": {"User": "65532:65532", "Env": ["PATH=/usr/bin:/bin"]},
    }
    versions = {
        "python": "Python 3.12.3",
        "git": "git version 2.43.0",
        "rg": "ripgrep 14.1.0",
        "codex": bindings[0].version,
        "grok": bindings[1].version,
    }
    observed = {
        "uid": 65532,
        "gid": 65532,
        "environment": qualification.RUNTIME_ENVIRONMENT,
        "root_write_denied": True,
        "docker_socket_present": False,
        "credential_paths_readable": [],
        "python_stdlib_probe": True,
        "git_worktree_probe": True,
        "python_validation": {
            "approved_import_root": qualification.PYTHON_VALIDATION_IMPORT_ROOT,
            "closure_entries_checked": 0,
            "direct_script_returncode": 0,
            "driver_isolation_argv": [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                qualification.PYTHON_VALIDATION_DRIVER,
            ],
            "duckdb_query": True,
            "expected_versions": qualification.PYTHON_VALIDATION_EXPECTED_VERSIONS,
            "observed_versions": qualification.PYTHON_VALIDATION_EXPECTED_VERSIONS,
            "imports_bounded_to_approved_root": True,
            "manifest_verified": True,
            "path_before": ["/usr/lib/python312.zip", "/usr/lib/python3.12"],
            "plugin_autoload_disabled": True,
            "pytest_smoke_returncode": 0,
            "site_loaded_before_validation": False,
            "startup_files": [],
            "sys_flags": {
                "dont_write_bytecode": 1,
                "isolated": 1,
                "no_site": 1,
                "no_user_site": 1,
            },
        },
        "ca_sha256": "b" * 64,
        "tools": {
            name: {
                "path": qualification.EXPECTED_TOOL_PATHS[name],
                "returncode": 0,
                "version": value,
            }
            for name, value in versions.items()
        },
        "hashes": {
            "codex": bindings[0].sha256,
            "grok": bindings[1].sha256,
            "manifest": manifest_hash,
        },
        "file_metadata": {
            **{
                binding.name: {
                    "gid": 65532,
                    "mode": "0555",
                    "regular": True,
                    "size": binding.size,
                    "uid": 65532,
                }
                for binding in bindings
            },
            "manifest": {
                "gid": 65532,
                "mode": "0444",
                "regular": True,
                "size": 100,
                "uid": 65532,
            },
        },
    }

    blockers, observed_versions = qualification._evaluate_probe(
        container,
        observed,
        bindings,
        manifest_sha256="sha256:" + manifest_hash,
        validation_toolchain={
            "components": [],
            "entrypoints": [],
            "isolation_argv_prefix": [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                qualification.PYTHON_VALIDATION_DRIVER,
            ],
        },
    )

    assert blockers == []
    assert observed_versions == versions
    container["HostConfig"]["NetworkMode"] = "bridge"
    blockers, _ = qualification._evaluate_probe(
        container,
        observed,
        bindings,
        manifest_sha256="sha256:" + manifest_hash,
        validation_toolchain={
            "components": [],
            "entrypoints": [],
            "isolation_argv_prefix": [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                qualification.PYTHON_VALIDATION_DRIVER,
            ],
        },
    )
    assert blockers == ["minimal_hardening_probe_failed"]
    container["HostConfig"]["NetworkMode"] = "none"
    observed["python_validation"]["pytest_smoke_returncode"] = 1
    blockers, _ = qualification._evaluate_probe(
        container,
        observed,
        bindings,
        manifest_sha256="sha256:" + manifest_hash,
        validation_toolchain={
            "components": [],
            "entrypoints": [],
            "isolation_argv_prefix": [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                qualification.PYTHON_VALIDATION_DRIVER,
            ],
        },
    )
    assert blockers == ["project_validation_dependencies_not_admitted"]


def test_validation_toolchain_manifest_binds_exact_files_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    overlay = tmp_path / "overlay"
    package = overlay / "opt/eaaef/python-validation/example"
    distribution = overlay / "opt/eaaef/python-validation/example-1.0.dist-info"
    package.mkdir(parents=True)
    distribution.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: example\nVersion: 1.0\n",
        encoding="utf-8",
    )
    driver = overlay / qualification.PYTHON_VALIDATION_DRIVER.removeprefix("/")
    wrapper = overlay / qualification.PYTHON_VALIDATION_WRAPPER.removeprefix("/")
    driver.parent.mkdir(parents=True)
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    driver.write_bytes(qualification.PYTHON_VALIDATION_DRIVER_BYTES)
    wrapper.write_bytes(qualification.PYTHON_VALIDATION_WRAPPER_BYTES)
    wrapper.chmod(0o755)
    wrapper.with_name("python3").symlink_to("python")
    usr_bin = overlay / "usr/bin"
    usr_bin.mkdir(parents=True)
    (usr_bin / "python3.12").write_bytes(b"python")
    (usr_bin / "python3.12").chmod(0o755)
    (usr_bin / "python3").symlink_to("python3.12")
    (usr_bin / "python").symlink_to("python3")
    monkeypatch.setattr(
        qualification,
        "PYTHON_VALIDATION_COMPONENTS",
        (
            {
                "name": "example",
                "version": "1.0",
                "runtime_dependencies": (),
                "paths": (
                    ("/source/example", "example"),
                    (
                        "/source/example-1.0.dist-info",
                        "example-1.0.dist-info",
                    ),
                ),
                "metadata": "example-1.0.dist-info/METADATA",
            },
        ),
    )

    manifest = qualification._validation_toolchain_manifest(
        overlay, qualification._closure_records(overlay)
    )

    assert manifest["components"][0]["name"] == "example"
    assert manifest["components"][0]["version"] == "1.0"
    assert manifest["components"][0]["content_cid"].startswith("sha256:")
    file_record = next(
        item
        for item in manifest["components"][0]["closure_entries"]
        if item["path"].endswith("/__init__.py")
    )
    assert file_record["uid"] == 0
    assert file_record["gid"] == 0
    assert file_record["mode"] == "0444"
    assert len(file_record["sha256"]) == 64
    assert manifest["startup_code_policy"]["pth_files_accepted"] is False


def test_validation_driver_is_fixed_isolated_and_denies_site_startup():
    driver = qualification.PYTHON_VALIDATION_DRIVER_BYTES.decode()
    wrapper = qualification.PYTHON_VALIDATION_WRAPPER_BYTES.decode()

    assert 'VALIDATION_ROOT = "/opt/eaaef/python-validation"' in driver
    assert "sys.flags.isolated" in driver
    assert "sys.flags.no_site" in driver
    assert '"site" in sys.modules' in driver
    assert 'arguments[:2] == ["-m", "pytest"]' in driver
    assert "/usr/bin/python3 -I -S -B" in wrapper


def test_spdx_is_deterministic_and_truthful_about_analysis(tmp_path: Path):
    codex = tmp_path / "codex"
    grok = tmp_path / "grok"
    _static_arm64_elf(codex)
    _static_arm64_elf(grok)
    bindings = [
        _binding("codex", codex, "codex-cli test"),
        _binding("grok", grok, "grok test"),
    ]
    values = {
        "image_id": "sha256:" + "a" * 64,
        "image_tag": "minimal:test",
        "bindings": bindings,
        "versions": {"python": "Python 3.12.3"},
        "rootfs_sha256": "sha256:" + "b" * 64,
        "source_date_epoch": 1_800_000_000,
        "validation_toolchain": {
            "content_cid": "sha256:" + "c" * 64,
            "components": [
                {
                    "name": "pytest",
                    "version": "9.0.3",
                    "content_cid": "sha256:" + "d" * 64,
                }
            ],
        },
    }

    first = qualification._spdx(**values)
    second = qualification._spdx(**values)
    document = json.loads(first)

    assert first == second
    assert document["spdxVersion"] == "SPDX-2.3"
    assert all(package["filesAnalyzed"] is False for package in document["packages"])
    assert any(package["name"] == "pytest" for package in document["packages"])
    assert "worker capacity is zero" in document["documentComment"]
