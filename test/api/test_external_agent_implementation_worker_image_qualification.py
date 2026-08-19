from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import stat
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/qualify_external_agent_implementation_worker_image.py"
CONTAINERFILE_PATH = REPO_ROOT / "containers/external-agent/implementation-worker.Containerfile"
SPEC = importlib.util.spec_from_file_location(
    "eaaef_implementation_worker_image_qualification", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


def _static_arm64_elf(path: Path) -> None:
    header = bytearray(64)
    header[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into("<HHIQQQIHHHHHH", header, 16, 2, 183, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    program = bytearray(56)
    struct.pack_into("<IIQQQQQQ", program, 0, 1, 5, 0, 0, 0, 120, 120, 4096)
    path.write_bytes(bytes(header + program))
    path.chmod(0o755)


def _binding(name: str, path: Path, version: str):
    info = path.stat()
    return qualification.SourceBinding(
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


def _bindings(tmp_path: Path):
    codex = tmp_path / "codex-source"
    grok = tmp_path / "grok-source"
    _static_arm64_elf(codex)
    _static_arm64_elf(grok)
    return [
        _binding("codex", codex, "codex-cli test"),
        _binding("grok", grok, "grok test"),
    ]


def _valid_probe(bindings):
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
        },
        "Config": {
            "User": "65532:65532",
            "Env": [
                "PATH=/usr/bin:/bin",
                "NVIDIA_VISIBLE_DEVICES=void",
                "NVIDIA_DRIVER_CAPABILITIES=",
            ],
        },
        "Mounts": [],
    }
    observed = {
        "uid": 65532,
        "gid": 65532,
        "environment": qualification.RUNTIME_ENVIRONMENT,
        "file_metadata": {
            **{
                binding.name: {
                    "gid": 65532,
                    "links": 1,
                    "mode": "0555",
                    "regular": True,
                    "size": binding.size,
                    "uid": 65532,
                }
                for binding in bindings
            },
            "candidate-inputs": {
                "gid": 65532,
                "links": 1,
                "mode": "0444",
                "regular": True,
                "size": 100,
                "uid": 65532,
            },
        },
        "root_write_denied": True,
        "docker_socket_present": False,
        "credential_paths_readable": [],
        "hashes": {
            **{binding.name: binding.sha256 for binding in bindings},
            "candidate-inputs": manifest_hash,
        },
        "tools": {
            "git": {
                "path": "/usr/bin/git",
                "returncode": 0,
                "version": "git version 2.43.0",
            },
            "python": {
                "path": "/usr/bin/python3",
                "returncode": 0,
                "version": "Python 3.12.3",
            },
            "rg": {
                "path": "/usr/local/bin/rg",
                "returncode": 0,
                "version": "ripgrep 14.1.0",
            },
            **{
                binding.name: {
                    "path": f"/opt/eaaef/bin/{binding.name}",
                    "returncode": 0,
                    "version": binding.version,
                }
                for binding in bindings
            },
        },
    }
    return container, observed


def test_containerfile_is_distinct_offline_copy_only_candidate():
    text = CONTAINERFILE_PATH.read_text(encoding="utf-8")

    assert not text.startswith("# syntax=")
    assert "ARG BASE_IMAGE=ipfs-accelerate-authority-validation:20260803-v2" in text
    assert qualification.EXPECTED_BASE_IMAGE_ID in text
    assert "USER 65532:65532" in text
    assert "NVIDIA_VISIBLE_DEVICES=void" in text
    assert 'NVIDIA_DRIVER_CAPABILITIES=""' in text
    assert 'ENTRYPOINT ["/usr/bin/env", "-i"' in text
    assert "COPY " not in text
    assert text.count("ADD ") == 1
    assert "ADD worker-rootfs.tar /" in text
    assert "RUN " not in text
    assert 'worker-capacity="0"' in text
    assert not any(command in text for command in (" apt ", "apt-get", " apk ", "curl ", "wget "))
    assert "COPY /usr" not in text


def test_canonical_rootfs_tar_is_closed_and_byte_reproducible(tmp_path: Path):
    bindings = _bindings(tmp_path)
    manifest = b'{"schema":"candidate-inputs@test"}\n'
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    values = {
        "staged_bindings": (
            (bindings[0], Path(bindings[0].absolute_path)),
            (bindings[1], Path(bindings[1].absolute_path)),
        ),
        "manifest": manifest,
        "source_date_epoch": 1_800_000_000,
    }

    first_hash, first_size = qualification._canonical_rootfs_tar(first, **values)
    second_hash, second_size = qualification._canonical_rootfs_tar(second, **values)

    assert first.read_bytes() == second.read_bytes()
    assert first_hash == second_hash == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_size == second_size == len(first.read_bytes())
    with tarfile.open(first, mode="r:") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == [
        ".",
        "opt",
        "opt/eaaef",
        "opt/eaaef/bin",
        "opt/eaaef/bin/codex",
        "opt/eaaef/bin/grok",
        "opt/eaaef/candidate-inputs.json",
    ]
    assert [(member.uid, member.gid) for member in members] == [
        (0, 0),
        (0, 0),
        (65532, 65532),
        (65532, 65532),
        (65532, 65532),
        (65532, 65532),
        (65532, 65532),
    ]
    assert all(member.mtime == 1_800_000_000 for member in members)


def test_native_source_requires_absolute_canonical_static_arm64(tmp_path: Path):
    binary = tmp_path / "codex"
    _static_arm64_elf(binary)

    info, digest = qualification._snapshot(binary)

    assert info.st_size == 120
    assert digest == hashlib.sha256(binary.read_bytes()).hexdigest()
    with pytest.raises(qualification.QualificationError, match="absolute"):
        qualification._snapshot(Path("codex"))
    symlink = tmp_path / "codex-link"
    symlink.symlink_to(binary)
    with pytest.raises(qualification.QualificationError, match="canonical"):
        qualification._snapshot(symlink)


def test_staging_fails_on_bound_source_or_hash_drift(tmp_path: Path):
    source = tmp_path / "codex"
    _static_arm64_elf(source)
    binding = _binding("codex", source, "codex-cli test")
    source.write_bytes(source.read_bytes() + b"drift")
    source.chmod(0o755)

    with pytest.raises(qualification.QualificationError, match="source/hash drift"):
        qualification._stage_binary(binding, tmp_path / "staged-codex")


def test_staging_normalizes_context_mtime_for_clean_build_reproducibility(tmp_path: Path):
    source = tmp_path / "codex"
    destination = tmp_path / "staged-codex"
    _static_arm64_elf(source)
    binding = _binding("codex", source, "codex-cli test")
    normalized_mtime_ns = 1_800_000_000 * 1_000_000_000

    qualification._stage_binary(
        binding,
        destination,
        normalized_mtime_ns=normalized_mtime_ns,
    )

    assert destination.stat().st_mtime_ns == normalized_mtime_ns
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555


def test_exact_tool_versions_and_missing_base_rg_are_fail_closed(tmp_path: Path):
    bindings = _bindings(tmp_path)
    container, observed = _valid_probe(bindings)

    blockers, versions = qualification._evaluate_probe(container, observed, bindings)
    assert blockers == []
    assert versions == {
        "git": "git version 2.43.0",
        "python": "Python 3.12.3",
        "rg": "ripgrep 14.1.0",
        "codex": "codex-cli test",
        "grok": "grok test",
    }

    observed["tools"]["rg"] = {
        "path": "/usr/local/bin/rg",
        "returncode": 127,
        "version": "",
    }
    blockers, versions = qualification._evaluate_probe(container, observed, bindings)
    assert blockers == ["base_tool_rg_unavailable"]
    assert "rg" not in versions


def test_probe_rejects_native_metadata_and_input_manifest_drift(tmp_path: Path):
    bindings = _bindings(tmp_path)
    container, observed = _valid_probe(bindings)
    manifest_hash = "a" * 64

    blockers, _versions = qualification._evaluate_probe(
        container,
        observed,
        bindings,
        input_manifest_sha256="sha256:" + manifest_hash,
    )
    assert blockers == []

    observed["file_metadata"]["codex"]["mode"] = "0777"
    observed["hashes"]["candidate-inputs"] = "b" * 64
    blockers, _versions = qualification._evaluate_probe(
        container,
        observed,
        bindings,
        input_manifest_sha256="sha256:" + manifest_hash,
    )
    assert blockers == [
        "embedded_codex_identity_drift",
        "embedded_input_manifest_identity_drift",
    ]


def test_probe_rejects_embedded_credentials_and_host_usr_bind(tmp_path: Path):
    bindings = _bindings(tmp_path)
    container, observed = _valid_probe(bindings)
    container["Config"]["Env"].append("PROVIDER_API_KEY=embedded")
    container["HostConfig"]["Binds"] = ["/usr:/usr:ro"]
    observed["credential_paths_readable"] = ["/root/.codex/auth.json"]

    blockers, _versions = qualification._evaluate_probe(container, observed, bindings)

    assert "nonroot_readonly_network_none_probe_failed" in blockers
    assert "embedded_credential_path_detected" in blockers
    assert "embedded_sensitive_environment_detected" in blockers


def test_real_probe_orchestration_targets_candidate_after_all_hardening_options(
    monkeypatch,
    tmp_path: Path,
):
    image_tag = "eaaef-implementation-worker:probe-regression"
    bindings = _bindings(tmp_path)
    container, observed = _valid_probe(bindings)
    run_calls: list[list[str]] = []
    inspect_calls: list[list[str]] = []

    def fake_run(argv, *, cwd, timeout=120, check=True):
        del cwd, timeout, check
        command = list(argv)
        run_calls.append(command)
        if command[1:2] == ["create"]:
            return subprocess.CompletedProcess(command, 0, "probe-container-id\n", "")
        if command[1:3] == ["start", "-a"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(observed), "")
        if command[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, "probe-container-id\n", "")
        raise AssertionError(f"unexpected runtime invocation: {command!r}")

    def fake_docker_json(docker, arguments, *, cwd):
        del cwd
        inspect_calls.append([docker, *arguments])
        return [container]

    monkeypatch.setattr(qualification, "_run", fake_run)
    monkeypatch.setattr(qualification, "_docker_json", fake_docker_json)

    returned_container, returned_observed = qualification._probe(
        docker="/usr/bin/docker",
        repo_root=REPO_ROOT,
        image_tag=image_tag,
    )

    assert returned_container == container
    assert returned_observed == observed
    assert run_calls[0] == [
        "/usr/bin/docker",
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
        qualification.NONROOT_USER,
        image_tag,
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        "-c",
        qualification._probe_program(),
    ]
    assert inspect_calls == [["/usr/bin/docker", "container", "inspect", "probe-container-id"]]
    assert run_calls[-1] == [
        "/usr/bin/docker",
        "rm",
        "-f",
        "probe-container-id",
    ]


def test_spdx_binds_exact_image_base_and_native_digests(tmp_path: Path):
    bindings = _bindings(tmp_path)
    image_id = "sha256:" + "1" * 64
    values = {
        "image_id": image_id,
        "image_tag": "eaaef-implementation-worker:test",
        "bindings": bindings,
        "base_tool_versions": {
            "git": "git version 2.43.0",
            "python": "Python 3.12.3",
            "rg": "ripgrep 14.1.0",
            # Native tools may be present in the probe's combined version map,
            # but must not be duplicated as base-image packages in the SBOM.
            "codex": "codex-cli test",
            "grok": "grok test",
        },
        "source_date_epoch": 1_800_000_000,
    }

    first = qualification._canonical(qualification._spdx_document(**values))
    second = qualification._canonical(qualification._spdx_document(**values))

    assert first == second
    assert len(first) < qualification.MAXIMUM_SBOM_BYTES
    sbom = json.loads(first)
    packages = {package["name"]: package for package in sbom["packages"]}
    image = packages["eaaef-implementation-worker:test"]
    base = packages[qualification.EXPECTED_BASE_REFERENCE]
    assert image["versionInfo"] == image_id
    assert image["checksums"][0]["checksumValue"] == "1" * 64
    assert base["versionInfo"] == qualification.EXPECTED_BASE_IMAGE_ID
    for binding in bindings:
        assert packages[binding.name]["checksums"] == [
            {"algorithm": "SHA256", "checksumValue": binding.sha256}
        ]
    assert len(sbom["packages"]) == 2 + len(bindings) + len(qualification.REQUIRED_BASE_TOOLS)
    assert len(packages) == len(sbom["packages"])
    assert all(package["filesAnalyzed"] is False for package in sbom["packages"])
    assert "Worker capacity is 0" in sbom["documentComment"]


def test_detached_sbom_identity_binds_exact_canonical_bytes_and_subject():
    sbom_bytes = b'{"spdxVersion":"SPDX-2.3"}\n'
    image_id = "sha256:" + "1" * 64

    identity = qualification._sbom_identity(sbom_bytes, image_id=image_id)

    assert identity == {
        "content_cid": "sha256:" + hashlib.sha256(sbom_bytes).hexdigest(),
        "canonicalization": qualification.SBOM_CANONICALIZATION,
        "bytes": len(sbom_bytes),
        "subject_image_id": image_id,
    }


def test_merged_export_scan_records_only_redacted_credential_evidence():
    archive_bytes = io.BytesIO()
    secret = b"api_key=" + b"A" * 32
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for path, payload in (
            ("etc/ordinary.conf", b"safe=true\n"),
            ("root/.aws/credentials", secret + b"\n"),
        ):
            member = tarfile.TarInfo(path)
            member.size = len(payload)
            member.mode = 0o600
            archive.addfile(member, io.BytesIO(payload))
    archive_bytes.seek(0)

    scan = qualification._scan_export_tar(
        archive_bytes,
        deadline=time.monotonic() + 10,
    )

    assert scan["complete"] is True
    assert scan["entries"] == 2
    assert scan["regular_files"] == 2
    assert scan["finding_files"] == 1
    assert scan["findings"] == [
        {
            "path": "root/.aws/credentials",
            "detectors": ["assigned_secret_value", "credential_filename"],
            "sha256": "sha256:" + hashlib.sha256(secret + b"\n").hexdigest(),
            "size": len(secret) + 1,
        }
    ]
    assert scan["raw_secret_values_recorded"] is False
    assert secret.decode() not in json.dumps(scan, sort_keys=True)


def test_missing_runtime_emits_typed_unsigned_zero_capacity_no_go(monkeypatch, tmp_path: Path):
    bindings = _bindings(tmp_path)
    monkeypatch.setattr(
        qualification,
        "_bind_source",
        lambda name, _path, repo_root: next(
            binding for binding in bindings if binding.name == name
        ),
    )
    monkeypatch.setattr(qualification.shutil, "which", lambda _name: None)
    args = argparse.Namespace(
        repo_root=REPO_ROOT,
        runtime="docker",
        codex_binary=Path("/absolute/codex"),
        grok_binary=Path("/absolute/grok"),
        image_tag="eaaef-implementation-worker:test",
        source_date_epoch=1_800_000_000,
    )

    report, sbom = qualification.qualify(args)

    assert report["decision"] == "no_go"
    assert report["status"] == "host_capability_no_go"
    assert report["worker_capacity"] == 0
    assert report["maximum_parallel_workers"] == 0
    assert report["task_dispatch_admitted"] is False
    assert report["candidate_signed"] is False
    assert report["image_signature_minted"] is False
    assert report["sbom_signature_minted"] is False
    assert report["production_receipt_minted"] is False
    assert report["network_authorized"] is False
    assert report["provider_authorized"] is False
    assert report["provider_auth_accessed"] is False
    assert report["provider_invoked"] is False
    assert report["supervisor_process_started"] is False
    assert report["credential_scan"]["mode"] == "not-run"
    assert report["credential_scan"]["complete"] is False
    assert "no_embedded_credentials" not in json.dumps(report, sort_keys=True)
    assert report["blockers"] == [
        "network_authorization_not_independently_signed",
        "provider_authorization_not_independently_signed",
        "container_runtime_unavailable",
    ]
    assert report["report_cid"] == qualification._cid(
        {key: value for key, value in report.items() if key != "report_cid"}
    )
    assert sbom == b""


def test_build_contract_names_only_disposable_context_and_offline_flags():
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'TemporaryDirectory(prefix="eaaef-worker-build-")' in text
    assert 'TemporaryDirectory(prefix="eaaef-worker-stage-")' in text
    assert '"--pull=false"' in text
    assert '"--no-cache"' in text
    assert '"--network=none"' in text
    assert '"--provenance=false"' in text
    assert '"--sbom=false"' in text
    assert '"type=docker,rewrite-timestamp=true"' in text
    assert '"context_retained": False' in text
    assert '"base_image_id_before": before' in text
    assert '"base_image_id_after": after' in text
    assert 'f"SOURCE_DATE_EPOCH={source_date_epoch}"' in text
    assert 'f"ROOTFS_TAR_SHA256={rootfs_tar_hash}"' in text
    assert "provider_process" not in text
    assert "agent_supervisor.runtime" not in text
