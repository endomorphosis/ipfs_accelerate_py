#!/usr/bin/env python3
"""Build and probe the EAAEF bootstrap image without granting admission.

The command emits a canonical SPDX image/toolchain statement and a typed host
capability report.  The committed image is a diagnostic verifier, not an agent
worker, so its report always carries zero task-dispatch capacity.  A rootful
daemon may produce useful nonroot hardening evidence but is not admitted by the
current policy.  This command has no signing key and cannot create the
independently reviewed provider/container qualification or launch a supervisor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-provider-container-policy@1"
)
REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-container-host-capability-report@1"
)
DAEMON_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-container-daemon-identity@1"
)
DAEMON_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-container-daemon-policy@1"
)
EXPECTED_BASE_IMAGE_ID = (
    "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
)
EXPECTED_BASE_REFERENCE = (
    "ipfs-accelerate-authority-validation:20260803-v2"
)
MAXIMUM_COMMAND_OUTPUT_BYTES = 128 * 1024
MAXIMUM_SBOM_BYTES = 1024 * 1024
AUTHORITY_REGISTRY_PREFIX = (
    "data/agent_supervisor/external_agent_autonomous_execution_fabric/"
    "authority/"
)


class QualificationError(ValueError):
    """A deterministic configuration or local-runtime qualification error."""


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


def _read_json(path: Path, *, maximum_bytes: int) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > maximum_bytes:
        raise QualificationError(f"JSON input is empty or oversized: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"JSON input is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"JSON input must be an object: {path}")
    return value


def _validate_policy(policy: Mapping[str, Any], *, repo_root: Path) -> None:
    identity = policy.get("policy_cid")
    body = dict(policy)
    body.pop("policy_cid", None)
    provider_route = policy.get("provider_route")
    image = policy.get("image_build")
    runtime = policy.get("runtime_policy")
    modes = runtime.get("execution_modes") if isinstance(runtime, Mapping) else None
    rootless_mode = modes.get("rootless_engine") if isinstance(modes, Mapping) else None
    rootful_mode = (
        modes.get("rootful_daemon_nonroot_worker")
        if isinstance(modes, Mapping)
        else None
    )
    sbom = policy.get("sbom_policy")
    ceremony = policy.get("qualification_ceremony")
    launch = policy.get("launch_policy")
    if (
        set(policy)
        != {
            "schema",
            "board_namespace",
            "task_id",
            "policy_cid",
            "provider_route",
            "image_build",
            "runtime_policy",
            "sbom_policy",
            "qualification_ceremony",
            "launch_policy",
        }
        or policy.get("schema") != POLICY_SCHEMA
        or identity != _cid(body)
        or policy.get("board_namespace")
        != "external-agent-autonomous-execution-fabric-v1"
        or policy.get("task_id") != "EAAEF-000"
        or not isinstance(provider_route, Mapping)
        or set(provider_route)
        != {
            "route_id",
            "primary_provider_id",
            "primary_model_id",
            "fallback_provider_id",
            "fallback_model_id",
            "fallback_reasoning_effort",
            "required_effects",
            "authorization_path_prefix",
            "source_addressed",
        }
        or provider_route.get("route_id")
        != "agent-supervisor-eaaef-v1-grok46-terra56-high-auth-or-hard-quota-v1"
        or provider_route.get("primary_provider_id") != "grok_cli"
        or provider_route.get("primary_model_id") != "grok-4.6"
        or provider_route.get("fallback_provider_id") != "codex"
        or provider_route.get("fallback_model_id") != "gpt-5.6-terra"
        or provider_route.get("fallback_reasoning_effort") != "high"
        or provider_route.get("required_effects")
        != ["edit", "isolated_worktree", "test"]
        or provider_route.get("authorization_path_prefix")
        != AUTHORITY_REGISTRY_PREFIX + "provider-route-authorization-"
        or provider_route.get("source_addressed") is not True
        or not isinstance(image, Mapping)
        or set(image)
        != {
            "containerfile",
            "base_image_local_reference",
            "base_image_id",
            "pull",
            "build_network",
            "provenance_attestation",
            "embedded_sbom_attestation",
            "expected_os",
            "expected_architecture",
        }
        or image.get("base_image_local_reference")
        != EXPECTED_BASE_REFERENCE
        or image.get("base_image_id") != EXPECTED_BASE_IMAGE_ID
        or image.get("pull") is not False
        or image.get("build_network") != "none"
        or image.get("provenance_attestation") is not False
        or image.get("embedded_sbom_attestation") is not False
        or image.get("expected_os") != "linux"
        or image.get("expected_architecture") != "arm64"
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "workload_class",
            "task_dispatch_admitted",
            "maximum_parallel_workers",
            "maximum_parallel_containers",
            "execution_modes",
            "nonroot_user",
            "read_only_base",
            "network_mode",
            "cap_drop",
            "no_new_privileges",
            "pids_limit",
            "cpu_limit",
            "memory_limit_bytes",
            "disk_limit_bytes",
            "gpu",
            "privileged",
            "host_pid",
            "host_ipc",
            "devices",
            "docker_socket_mounted",
            "inherit_host_environment",
            "environment",
            "mount_targets",
        }
        or runtime.get("workload_class") != "bootstrap_diagnostic_only"
        or runtime.get("task_dispatch_admitted") is not False
        or runtime.get("maximum_parallel_workers") != 0
        or runtime.get("maximum_parallel_containers") != 1
        or not isinstance(modes, Mapping)
        or set(modes) != {"rootless_engine", "rootful_daemon_nonroot_worker"}
        or not isinstance(rootless_mode, Mapping)
        or rootless_mode
        != {
            "admitted_when_supported": True,
            "requires_rootless_supported": True,
            "requires_rootless_verified": True,
        }
        or not isinstance(rootful_mode, Mapping)
        or rootful_mode
        != {
            "admitted": False,
            "requires_rootless_supported_false": True,
            "requires_exact_daemon_identity": True,
            "requires_exact_daemon_policy": True,
            "requires_independent_security_approval": True,
            "docker_socket_mounted": False,
        }
        or runtime.get("nonroot_user") != "65532:65532"
        or runtime.get("read_only_base") is not True
        or runtime.get("network_mode") != "none"
        or runtime.get("cap_drop") != ["ALL"]
        or runtime.get("no_new_privileges") is not True
        or runtime.get("pids_limit") != 256
        or runtime.get("cpu_limit") != 2
        or runtime.get("memory_limit_bytes") != 4 * 1024**3
        or runtime.get("disk_limit_bytes") != 16 * 1024**3
        or runtime.get("gpu")
        != {"mode": "none", "device_ids": [], "memory_limit_bytes": 0}
        or runtime.get("privileged") is not False
        or runtime.get("host_pid") is not False
        or runtime.get("host_ipc") is not False
        or runtime.get("devices") != []
        or runtime.get("docker_socket_mounted") is not False
        or runtime.get("inherit_host_environment") is not False
        or runtime.get("environment")
        != {
            "BASH_ENV": "",
            "CODEX_HOME": "/opt/codex-home",
            "ENV": "",
            "HOME": "/opt/codex-home",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/opt/ipfs-task-tools/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/opt/ipfs-validation-site-packages",
            "TERM": "dumb",
        }
        or runtime.get("mount_targets")
        != {
            "worktree": "/workspace",
            "provider_auth": "/opt/codex-home/auth.json",
            "secret_prefix": "/run/secrets/",
        }
        or not isinstance(sbom, Mapping)
        or dict(sbom)
        != {
            "format": "spdx-json",
            "spdx_version": "SPDX-2.3",
            "maximum_bytes": MAXIMUM_SBOM_BYTES,
            "canonical_json": True,
            "files_analyzed_may_be_false": True,
            "package_inventory_must_disclose_scope": True,
        }
        or not isinstance(ceremony, Mapping)
        or set(ceremony)
        != {
            "schema",
            "artifact_path_prefix",
            "source_addressed",
            "prepare_function",
            "signing_bytes_function",
            "seal_function",
            "verify_function",
            "signer_role",
            "independent_security_reviewer_required",
            "independent_qualification_signer_required",
            "maximum_lifetime_ms",
        }
        or ceremony.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-provider-container-qualification@1"
        )
        or ceremony.get("artifact_path_prefix")
        != AUTHORITY_REGISTRY_PREFIX + "provider-container-qualification--"
        or ceremony.get("source_addressed") is not True
        or ceremony.get("prepare_function")
        != "prepare_eaaef_provider_container_qualification"
        or ceremony.get("signing_bytes_function")
        != "eaaef_provider_container_qualification_signing_bytes"
        or ceremony.get("seal_function")
        != "seal_eaaef_provider_container_qualification"
        or ceremony.get("verify_function")
        != "verify_eaaef_provider_container_qualification"
        or ceremony.get("signer_role")
        != "independent_bootstrap_admission_reviewer"
        or ceremony.get("independent_security_reviewer_required") is not True
        or ceremony.get("independent_qualification_signer_required") is not True
        or ceremony.get("maximum_lifetime_ms") != 86_400_000
        or not isinstance(launch, Mapping)
        or set(launch)
        != {
            "diagnostic_probe_may_start_non_authoritative_container",
            "this_policy_grants_launch",
            "this_qualification_grants_launch",
            "final_materializer_admission_required",
            "authority_mutated",
            "process_started",
        }
        or launch.get("diagnostic_probe_may_start_non_authoritative_container")
        is not True
        or any(
            launch.get(field) is not False
            for field in (
                "this_policy_grants_launch",
                "this_qualification_grants_launch",
                "authority_mutated",
                "process_started",
            )
        )
        or launch.get("final_materializer_admission_required") is not True
    ):
        raise QualificationError("bootstrap provider/container policy is invalid")
    containerfile = repo_root / str(image.get("containerfile") or "")
    if (
        containerfile.resolve(strict=True) != containerfile
        or not containerfile.is_file()
        or containerfile.is_symlink()
    ):
        raise QualificationError("bootstrap Containerfile is unavailable")


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env={
            "HOME": os.environ.get("HOME", "/nonexistent"),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "DOCKER_BUILDKIT": "1",
        },
    )
    if (
        len(result.stdout.encode("utf-8", errors="replace"))
        > MAXIMUM_COMMAND_OUTPUT_BYTES
        or len(result.stderr.encode("utf-8", errors="replace"))
        > MAXIMUM_COMMAND_OUTPUT_BYTES
    ):
        raise QualificationError("runtime command output exceeded its bound")
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise QualificationError(
            f"runtime command failed ({result.returncode}): {detail}"
        )
    return result


def _docker_json(
    docker: str,
    args: Sequence[str],
    *,
    cwd: Path,
) -> Any:
    result = _run([docker, *args], cwd=cwd)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError("Docker returned invalid JSON") from exc


def _image_id(docker: str, reference: str, *, cwd: Path) -> str:
    result = _run(
        [docker, "image", "inspect", reference, "--format", "{{.Id}}"],
        cwd=cwd,
    )
    return result.stdout.strip()


def _probe_user_namespace(*, cwd: Path) -> tuple[bool, str]:
    executable = shutil.which("unshare")
    if executable is None:
        return False, "unshare_unavailable"
    result = _run([executable, "-Ur", "true"], cwd=cwd, check=False)
    return (
        result.returncode == 0,
        "" if result.returncode == 0 else "user_namespace_unavailable",
    )


def _spdx_document(
    *,
    image_id: str,
    image_tag: str,
    base_image_id: str,
    toolchains: Mapping[str, str],
    source_date_epoch: int,
) -> dict[str, Any]:
    created = dt.datetime.fromtimestamp(
        source_date_epoch,
        tz=dt.UTC,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    image_checksum = image_id.removeprefix("sha256:")
    packages: list[dict[str, Any]] = [
        {
            "SPDXID": "SPDXRef-Package-EAAEFBootstrapImage",
            "name": image_tag,
            "versionInfo": image_id,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [
                {"algorithm": "SHA256", "checksumValue": image_checksum}
            ],
            "supplier": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "comment": (
                "Package-level OCI image statement; filesystem files and "
                "transitive packages were not analyzed."
            ),
        }
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-Package-EAAEFBootstrapImage",
        }
    ]
    for index, (name, version) in enumerate(sorted(toolchains.items()), 1):
        spdx_id = f"SPDXRef-Package-Toolchain-{index}"
        packages.append(
            {
                "SPDXID": spdx_id,
                "name": name,
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "supplier": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Package-EAAEFBootstrapImage",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": spdx_id,
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "EAAEF bootstrap reconciliation OCI image statement",
        "documentNamespace": (
            "urn:ipfs-accelerate:eaaef:bootstrap-sbom:"
            + image_checksum
        ),
        "creationInfo": {
            "created": created,
            "creators": [
                "Tool: qualify_external_agent_bootstrap_container.py"
            ],
        },
        "documentComment": (
            "Bounded package-level statement generated without a package "
            "scanner; filesystem files and transitive packages were not "
            "analyzed, and filesAnalyzed=false is intentional. Base "
            f"image identity: {base_image_id}."
        ),
        "packages": packages,
        "relationships": relationships,
    }


def _probe_container(
    *,
    docker: str,
    image_tag: str,
    policy: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    runtime = policy["runtime_policy"]
    assert isinstance(runtime, Mapping)
    probe_program = (
        "import json,os,pathlib,subprocess;"
        "denied=False;"
        "\ntry:\n pathlib.Path('/eaaef-root-write-probe').write_text('x')"
        "\nexcept OSError:\n denied=True"
        "\nversions={};"
        "\nfor name,argv in "
        "{'python':['/usr/bin/python3','--version'],"
        "'git':['/usr/bin/git','--version']}.items():"
        "\n p=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False); versions[name]=p.stdout.strip()"
        "\nprint(json.dumps({'uid':os.getuid(),'gid':os.getgid(),"
        "'environment':dict(os.environ),'root_write_denied':denied,"
        "'docker_socket_present':pathlib.Path('/var/run/docker.sock').exists(),"
        "'toolchains':versions},sort_keys=True,separators=(',',':')))"
    )
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
        str(runtime["pids_limit"]),
        "--cpus",
        str(runtime["cpu_limit"]),
        "--memory",
        str(runtime["memory_limit_bytes"]),
        "--memory-swap",
        str(runtime["memory_limit_bytes"]),
        "--user",
        str(runtime["nonroot_user"]),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=67108864",
        image_tag,
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        "-c",
        probe_program,
    ]
    container_id = _run(create, cwd=repo_root).stdout.strip()
    if not container_id:
        raise QualificationError("Docker did not return a container identity")
    try:
        inspect = _docker_json(
            docker,
            ["container", "inspect", container_id],
            cwd=repo_root,
        )
        if not isinstance(inspect, list) or len(inspect) != 1:
            raise QualificationError("Docker container inspection is invalid")
        started = _run(
            [docker, "start", "-a", container_id],
            cwd=repo_root,
        )
        try:
            observed = json.loads(started.stdout)
        except json.JSONDecodeError as exc:
            raise QualificationError("container probe output is invalid") from exc
        if not isinstance(observed, dict):
            raise QualificationError("container probe result must be an object")
        return inspect[0], observed
    finally:
        _run(
            [docker, "rm", "-f", container_id],
            cwd=repo_root,
            check=False,
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def qualify(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    config_path = args.config.resolve(strict=True)
    repo_root = config_path.parent.parent.resolve(strict=True)
    policy = _read_json(config_path, maximum_bytes=256 * 1024)
    _validate_policy(policy, repo_root=repo_root)
    docker = shutil.which(args.runtime)
    blockers: list[str] = [
        "agent_worker_image_unavailable",
        "provider_task_dispatch_not_admitted",
    ]
    build_result: dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "image_id": "",
    }
    runtime_result: dict[str, Any] = {
        "runtime": args.runtime,
        "available": docker is not None,
        "daemon_rootless": False,
        "user_namespace_available": False,
        "security_options": [],
        "execution_mode": "unavailable",
        "daemon_identity_cid": "",
        "daemon_policy_cid": "",
    }
    sbom_bytes = b""
    probe_result: dict[str, Any] = {}
    if docker is None:
        blockers.append("container_runtime_unavailable")
    else:
        version = _docker_json(
            docker,
            ["version", "--format", "{{json .}}"],
            cwd=repo_root,
        )
        options = _docker_json(
            docker,
            ["info", "--format", "{{json .SecurityOptions}}"],
            cwd=repo_root,
        )
        security_options = options if isinstance(options, list) else []
        daemon_rootless = any(
            "rootless" in str(option).casefold()
            for option in security_options
        )
        userns_ok, userns_reason = _probe_user_namespace(cwd=repo_root)
        execution_mode = (
            "rootless_engine"
            if daemon_rootless
            else "rootful_daemon_nonroot_worker"
        )
        daemon_identity_cid = _cid(
            {
                "schema": DAEMON_IDENTITY_SCHEMA,
                "runtime": args.runtime,
                "server": version.get("Server")
                if isinstance(version, Mapping)
                else None,
            }
        )
        daemon_policy_cid = _cid(
            {
                "schema": DAEMON_POLICY_SCHEMA,
                "execution_mode": execution_mode,
                "security_options": sorted(str(item) for item in security_options),
                "rootless_supported": daemon_rootless,
                "user_namespace_available": userns_ok,
                "runtime_policy": dict(policy["runtime_policy"]),
            }
        )
        runtime_result.update(
            {
                "version": version,
                "daemon_rootless": daemon_rootless,
                "user_namespace_available": userns_ok,
                "security_options": security_options,
                "execution_mode": execution_mode,
                "daemon_identity_cid": daemon_identity_cid,
                "daemon_policy_cid": daemon_policy_cid,
            }
        )
        if not daemon_rootless:
            blockers.append("rootless_container_runtime_unavailable")
            blockers.append("rootful_daemon_fallback_not_admitted")
            if userns_reason:
                blockers.append(userns_reason)
        before = _image_id(
            docker,
            EXPECTED_BASE_REFERENCE,
            cwd=repo_root,
        )
        if before != EXPECTED_BASE_IMAGE_ID:
            raise QualificationError("local base image identity drifted")
        if args.diagnostic_build:
            build_result["attempted"] = True
            image_build = policy["image_build"]
            assert isinstance(image_build, Mapping)
            command = [
                docker,
                "build",
                "--pull=false",
                "--network=none",
                "--provenance=false",
                "--sbom=false",
                "-f",
                str(repo_root / str(image_build["containerfile"])),
                "-t",
                args.image_tag,
                str(repo_root),
            ]
            result = _run(command, cwd=repo_root, timeout=600)
            after = _image_id(
                docker,
                EXPECTED_BASE_REFERENCE,
                cwd=repo_root,
            )
            if after != before:
                raise QualificationError("local base image changed during build")
            image_id = _image_id(docker, args.image_tag, cwd=repo_root)
            image_inspect = _docker_json(
                docker,
                ["image", "inspect", args.image_tag],
                cwd=repo_root,
            )
            if not isinstance(image_inspect, list) or len(image_inspect) != 1:
                raise QualificationError("built image inspection is invalid")
            image_record = image_inspect[0]
            image_config = image_record.get("Config") or {}
            if (
                image_record.get("Id") != image_id
                or image_record.get("Os") != image_build.get("expected_os")
                or image_record.get("Architecture")
                != image_build.get("expected_architecture")
                or image_config.get("User")
                != policy["runtime_policy"].get("nonroot_user")
            ):
                raise QualificationError("built image identity or metadata drifted")
            container_inspect, observed = _probe_container(
                docker=docker,
                image_tag=args.image_tag,
                policy=policy,
                repo_root=repo_root,
            )
            host_config = container_inspect.get("HostConfig") or {}
            config = container_inspect.get("Config") or {}
            expected_environment = policy["runtime_policy"].get("environment")
            hardening_valid = bool(
                host_config.get("ReadonlyRootfs") is True
                and host_config.get("NetworkMode") == "none"
                and host_config.get("CapDrop") == ["ALL"]
                and "no-new-privileges" in host_config.get("SecurityOpt", [])
                and host_config.get("PidsLimit")
                == policy["runtime_policy"].get("pids_limit")
                and host_config.get("NanoCpus")
                == int(float(policy["runtime_policy"].get("cpu_limit")) * 1e9)
                and host_config.get("Memory")
                == policy["runtime_policy"].get("memory_limit_bytes")
                and host_config.get("Privileged") is False
                and host_config.get("PidMode") == ""
                and host_config.get("IpcMode") == "private"
                and host_config.get("Devices") == []
                and not host_config.get("DeviceRequests")
                and not host_config.get("Binds")
                and not host_config.get("PortBindings")
                and host_config.get("PublishAllPorts") is False
                and config.get("User")
                == policy["runtime_policy"].get("nonroot_user")
                and observed.get("uid") == 65532
                and observed.get("gid") == 65532
                and observed.get("environment") == expected_environment
                and observed.get("root_write_denied") is True
                and observed.get("docker_socket_present") is False
            )
            if not hardening_valid:
                blockers.append("nonroot_container_hardening_probe_failed")
            toolchains = observed.get("toolchains")
            if (
                not isinstance(toolchains, Mapping)
                or not toolchains
                or any(not str(value) for value in toolchains.values())
            ):
                raise QualificationError("toolchain probe is incomplete")
            sbom = _spdx_document(
                image_id=image_id,
                image_tag=args.image_tag,
                base_image_id=before,
                toolchains={
                    str(key): str(value) for key, value in toolchains.items()
                },
                source_date_epoch=args.source_date_epoch,
            )
            sbom_bytes = _canonical(sbom) + b"\n"
            if len(sbom_bytes) > MAXIMUM_SBOM_BYTES:
                raise QualificationError("SPDX statement exceeded its bound")
            build_result.update(
                {
                    "succeeded": True,
                    "image_id": image_id,
                    "base_image_id_before": before,
                    "base_image_id_after": after,
                    "image_labels_cid": _cid(image_config.get("Labels") or {}),
                    "inherited_exposed_ports": (
                        image_config.get("ExposedPorts") or {}
                    ),
                    "command": command,
                    "stdout_sha256": (
                        "sha256:"
                        + hashlib.sha256(result.stdout.encode()).hexdigest()
                    ),
                }
            )
            probe_result = {
                "hardening_valid": hardening_valid,
                "host_config": {
                    "network_mode": host_config.get("NetworkMode"),
                    "read_only_rootfs": host_config.get("ReadonlyRootfs"),
                    "cap_drop": host_config.get("CapDrop"),
                    "security_opt": host_config.get("SecurityOpt"),
                    "pids_limit": host_config.get("PidsLimit"),
                    "nano_cpus": host_config.get("NanoCpus"),
                    "memory": host_config.get("Memory"),
                    "privileged": host_config.get("Privileged"),
                    "pid_mode": host_config.get("PidMode"),
                    "ipc_mode": host_config.get("IpcMode"),
                    "devices": host_config.get("Devices"),
                    "device_requests": host_config.get("DeviceRequests"),
                    "binds": host_config.get("Binds"),
                    "port_bindings": host_config.get("PortBindings"),
                    "publish_all_ports": host_config.get("PublishAllPorts"),
                    "inherited_exposed_ports": config.get("ExposedPorts") or {},
                },
                "observed": observed,
            }
        else:
            blockers.append("diagnostic_image_build_not_requested")

    blockers = list(dict.fromkeys(blockers))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "policy_cid": str(policy.get("policy_cid") or ""),
        "source_date_epoch": args.source_date_epoch,
        "runtime": runtime_result,
        "build": build_result,
        "probe": probe_result,
        "sbom": {
            "format": "spdx-json" if sbom_bytes else "",
            "digest": (
                "sha256:" + hashlib.sha256(sbom_bytes).hexdigest()
                if sbom_bytes
                else ""
            ),
            "bytes": len(sbom_bytes),
            "files_analyzed": False,
            "scope": "package-level image and observed toolchains",
        },
        "prior_failed_attempts": sorted(set(args.prior_failed_attempt)),
        "workload_class": "bootstrap_diagnostic_only",
        "task_dispatch_admitted": False,
        "maximum_parallel_workers": 0,
        "maximum_parallel_containers": 1,
        "status": (
            "qualified_for_independent_security_review"
            if not blockers
            else "host_capability_no_go"
        ),
        "blockers": blockers,
        "image_qualification_minted": False,
        "provider_container_qualification_minted": False,
        "authority_mutated": False,
        "diagnostic_container_process_started": bool(probe_result),
        "supervisor_process_started": False,
        "provider_process_started": False,
        "provider_invoked": False,
    }
    report["report_cid"] = _cid(report)
    return report, sbom_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/external_agent_autonomous_execution_fabric_bootstrap.json"
        ),
    )
    parser.add_argument("--runtime", choices=("docker",), default="docker")
    parser.add_argument(
        "--image-tag",
        default="eaaef-bootstrap-reconciliation:local-qualification",
    )
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--diagnostic-build", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument(
        "--prior-failed-attempt",
        action="append",
        default=[],
        choices=(
            "dockerfile_frontend_network_fetch_and_bare_digest_resolution_failed",
        ),
    )
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
    return 0 if report["status"] == "qualified_for_independent_security_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
