"""Live seal for the offline TypeScript 5.9.3 authority image.

These tests intentionally require the real local image.  A missing image is a
failed predecessor, not an optional integration dependency: later DCR tasks
must never fall back to a host compiler or to an image pull.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.validation import (
    typescript_validation_image,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "docs/architecture/agent_supervisor_typescript_validation_image_receipt.json"
)


def _docker_environment(docker_config: Path) -> dict[str, str]:
    return {
        "DOCKER_CONFIG": str(docker_config),
        "DOCKER_HOST": grok_cli_runner._CODEX_FALLBACK_DOCKER_HOST,
        "HOME": "/nonexistent/ipfs-typescript-validation-test",
        "PATH": os.defpath,
    }


def _canary_payload(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if not line.startswith("{"):
            continue
        candidate = json.loads(line)
        if candidate.get("schema") == (
            "ipfs_accelerate_py.agent_supervisor." "typescript-validation-canary@1"
        ):
            return candidate
    raise AssertionError("sealed TypeScript canary output is absent")


def _generic_authority_environment(
    workspace: Path,
    command: str,
) -> dict[str, str]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    target_commit = git("rev-parse", "HEAD")
    target_tree = git("rev-parse", "HEAD^{tree}")
    task_id = "TEST-TYPESCRIPT-CANARY"
    task_cid = content_identity({"test_authority_task": task_id})
    commands = [command]
    scope = "pre_merge"
    plan_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "authority-validation-command-plan@2"
        ),
        "authority_profile": "generic_workspace_only@1",
        "task_id": task_id,
        "canonical_task_cid": task_cid,
        "scope": scope,
        "commands": commands,
        "target_commit": target_commit,
        "target_tree": target_tree,
        "git_common_anchor": "",
        "git_dir": "",
    }
    return {
        implementation_daemon._AUTHORITY_VALIDATION_SCOPE_ENV: scope,
        implementation_daemon._AUTHORITY_VALIDATION_PROFILE_ENV: (
            "generic_workspace_only@1"
        ),
        implementation_daemon._AUTHORITY_VALIDATION_COMMANDS_ENV: json.dumps(
            commands,
            separators=(",", ":"),
        ),
        implementation_daemon._AUTHORITY_VALIDATION_TASK_ENV: task_id,
        implementation_daemon._AUTHORITY_VALIDATION_TASK_CID_ENV: task_cid,
        implementation_daemon._AUTHORITY_VALIDATION_PLAN_ENV: content_identity(
            plan_body
        ),
        implementation_daemon._AUTHORITY_VALIDATION_TARGET_COMMIT_ENV: (
            target_commit
        ),
        implementation_daemon._AUTHORITY_VALIDATION_TARGET_TREE_ENV: target_tree,
        implementation_daemon._AUTHORITY_VALIDATION_GIT_COMMON_ANCHOR_ENV: "",
        implementation_daemon._AUTHORITY_VALIDATION_GIT_DIR_ENV: "",
        implementation_daemon._AUTHORITY_VALIDATION_PRODUCER_TRUST_ENV: "1",
    }


def test_receipt_pins_one_offline_layer_over_the_sealed_base(
    tmp_path: Path,
) -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    assert receipt["receipt_id"] == content_identity(receipt_body)
    image_id = typescript_validation_image.TYPESCRIPT_VALIDATION_IMAGE
    base_id = typescript_validation_image.TYPESCRIPT_VALIDATION_BASE_IMAGE
    assert receipt["image"]["image_id"] == image_id
    assert receipt["image"]["base_image_id"] == base_id
    assert receipt["build"]["network_mode"] == "none"
    assert receipt["build"]["pull_allowed"] is False
    assert grok_cli_runner._CODEX_FALLBACK_IMAGE == image_id
    assert (
        implementation_daemon.DEFAULT_AUTHORITY_VALIDATION_CONTAINER_IMAGE == image_id
    )

    docker_config = tmp_path / "docker-config"
    docker_config.mkdir(mode=0o700)
    completed = subprocess.run(
        [
            str(grok_cli_runner._CODEX_FALLBACK_DOCKER_BIN),
            f"--host={grok_cli_runner._CODEX_FALLBACK_DOCKER_HOST}",
            "--config",
            str(docker_config),
            "image",
            "inspect",
            base_id,
            image_id,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
        env=_docker_environment(docker_config),
    )
    assert completed.returncode == 0, completed.stdout
    base, image = json.loads(completed.stdout)
    assert base["Id"] == base_id
    assert image["Id"] == image_id
    assert image["Architecture"] == "arm64"
    assert image["Os"] == "linux"
    base_layers = base["RootFS"]["Layers"]
    image_layers = image["RootFS"]["Layers"]
    assert image_layers[:-1] == base_layers
    assert len(image_layers) == len(base_layers) + 1
    assert image_layers[-1] == receipt["image"]["toolchain_layer_diff_id"]
    labels = image["Config"]["Labels"]
    expected = typescript_validation_image.typescript_validation_toolchain_contract()
    assert labels["org.ipfs-accelerate.authority-validation.base"] == base_id
    assert labels["org.ipfs-accelerate.validation-build-network"] == "none"
    assert labels["org.ipfs-accelerate.node.version"] == (expected["node"]["version"])
    assert labels["org.ipfs-accelerate.node.sha256"] == (expected["node"]["sha256"])
    assert labels["org.ipfs-accelerate.typescript.version"] == (
        expected["typescript"]["version"]
    )
    assert labels["org.ipfs-accelerate.typescript.sha256"] == (
        expected["typescript"]["compiler_sha256"]
    )
    assert labels["org.ipfs-accelerate.typescript.package-sha256"] == (
        expected["typescript"]["package_sha256"]
    )
    assert labels["org.ipfs-accelerate.toolchain-manifest.sha256"] == (
        expected["manifest_sha256"]
    )


def test_host_authority_runner_parses_the_bounded_typescript_canary() -> None:
    toolchain = typescript_validation_image.typescript_validation_toolchain_contract()
    command = typescript_validation_image.TYPESCRIPT_CANARY_COMMAND
    result = implementation_daemon.PortalImplementationDaemon._authority_validation_command_runner(
        spec=SimpleNamespace(command=command, raw_command=command),
        workspace_path=REPOSITORY_ROOT,
        timeout_seconds=30,
        environment=_generic_authority_environment(REPOSITORY_ROOT, command),
    )
    assert result["returncode"] == 0, result.get("output")
    assert result["timed_out"] is False
    assert result["infrastructure_failure"] is False
    contract = result["authority_validation_isolation"]
    isolation = result["authority_validation_isolation_receipt"]
    assert contract["available"] is True
    assert contract["image_id"] == toolchain["image_id"]
    assert contract["typescript_validation_toolchain"] == toolchain
    assert isolation["image_id"] == toolchain["image_id"]
    assert isolation["typescript_validation_toolchain"] == toolchain
    assert isolation["network_mode"] == "none"
    assert isolation["workspace_read_only"] is True
    assert isolation["host_filesystem"] == "workspace_only_read_only"
    assert isolation["git_metadata_read_only"] is False
    assert isolation["git_metadata_drift_detected"] is False
    assert isolation["git_metadata_replay"]["mode"] == "not_requested"
    assert isolation["container_root_read_only"] is True
    assert isolation["capabilities_dropped"] == "all"
    assert isolation["no_new_privileges"] is True
    assert isolation["container_removed"] is True
    assert isolation["process_tree_quiesced"] is True
    canary = _canary_payload(str(result["output"]))
    assert canary == {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "typescript-validation-canary@1"
        ),
        "node_version": toolchain["node"]["version"],
        "node_sha256": toolchain["node"]["sha256"],
        "typescript_version": toolchain["typescript"]["version"],
        "compiler_sha256": toolchain["typescript"]["compiler_sha256"],
        "package_sha256": toolchain["typescript"]["package_sha256"],
        "asset_count": toolchain["asset_count"],
        "parse_diagnostic_count": 0,
        "statement_count": 2,
        "source_sha256": toolchain["canary"]["source_sha256"],
    }
    sealed = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))[
        "authority_typescript_canary"
    ]
    assert sealed["outer_policy"] == {
        "network_mode": "none",
        "workspace_read_only": True,
        "container_root_read_only": True,
        "capabilities_dropped": "all",
        "no_new_privileges": True,
        "provider_auth_mounted": False,
    }
    assert sealed["result"] == canary
    assert sealed["returncode"] == 0


def test_generic_authority_runner_denies_linked_git_metadata_projection(
    tmp_path: Path,
) -> None:
    def git(root: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-c", "protocol.file.allow=always", *arguments),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env={
                **os.environ,
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            },
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    def initialize(root: Path, label: str) -> None:
        root.mkdir()
        git(root, "init", "-q", "-b", "main")
        git(root, "config", "user.name", "Sealed linked Git test")
        git(root, "config", "user.email", "sealed-git@example.invalid")
        (root / "tracked.txt").write_text(label + "\n", encoding="utf-8")
        git(root, "add", "tracked.txt")
        git(root, "commit", "-qm", f"seed {label}")

    nested_source = tmp_path / "nested-source"
    initialize(nested_source, "nested")
    source = tmp_path / "source"
    initialize(source, "root")
    git(source, "submodule", "add", "-q", str(nested_source), "nested")
    git(source, "commit", "-qam", "add nested root")
    linked = tmp_path / "linked"
    git(source, "worktree", "add", "-q", "-b", "sealed", str(linked))
    git(linked, "submodule", "update", "--init", "--recursive")
    command = "git rev-parse --verify HEAD"

    result = implementation_daemon.PortalImplementationDaemon._authority_validation_command_runner(
        spec=SimpleNamespace(command=command, raw_command=command),
        workspace_path=linked,
        timeout_seconds=30,
        environment=_generic_authority_environment(linked, command),
    )

    assert result["returncode"] not in {0, 75, 78}, result
    isolation = result["authority_validation_isolation_receipt"]
    assert isolation["host_filesystem"] == "workspace_only_read_only"
    assert isolation["git_metadata_read_only"] is False
    assert isolation["git_metadata_replay"]["mode"] == "not_requested"
    assert isolation["git_metadata_replay"]["external_mount_count"] == 0
    assert isolation["git_metadata_drift_detected"] is False


def _codex_sandbox_boundary(
    tmp_path: Path,
) -> tuple[list[str], str, Path]:
    workspace = REPOSITORY_ROOT.resolve()
    checkpoint = (tmp_path / "checkpoint").resolve()
    checkpoint.mkdir(mode=0o700)
    docker_config = (tmp_path / "docker-config").resolve()
    docker_config.mkdir(mode=0o700)
    cidfile = (tmp_path / "container.cid").resolve()
    environment = dict(os.environ)
    environment[grok_cli_runner._CODEX_FALLBACK_CHECKPOINT_ENV] = str(checkpoint)
    (
        _docker,
        auth_path,
        package_root,
        bwrap_path,
        checkpoint_path,
    ) = grok_cli_runner._resolve_containerized_codex_fallback_assets(
        workspace=workspace,
        base_env=environment,
        require_checkpoint=True,
    )
    host_route = grok_cli_runner._codex_quota_fallback_host_command(
        codex="/usr/local/bin/codex",
        workspace_text=str(workspace),
        reasoning_effort="high",
    )
    provider_boundary = (
        grok_cli_runner._build_containerized_codex_quota_fallback_command(
            host_fallback_command=host_route,
            workspace=workspace,
            auth_path=auth_path,
            package_root=package_root,
            bwrap_path=bwrap_path,
            checkpoint_path=checkpoint_path,
            docker_config=docker_config,
            container_name=("ipfs-accelerate-codex-1-" + uuid.uuid4().hex),
            cidfile=cidfile,
            git_controls=(),
        )
    )
    # The smoke invokes no provider, so it closes the provider boundary's only
    # network grant.  Every other outer control and the root wrapper are the
    # exact production values validated immediately above.
    network_index = provider_boundary.index("--network=bridge")
    provider_boundary[network_index] = "--network=none"
    image_index = provider_boundary.index(grok_cli_runner._CODEX_FALLBACK_IMAGE)
    marker_index = image_index + 3
    assert provider_boundary[marker_index] == "codex-fallback-root-wrapper"
    prefix = provider_boundary[: marker_index + 1]
    profile = next(
        item
        for item in provider_boundary
        if item.startswith("permissions.dcr_fallback=")
    )
    assert "--pull=never" in prefix
    assert "--read-only" in prefix
    assert "--cap-drop=ALL" in prefix
    assert "--network=none" in prefix
    assert "--network=bridge" not in prefix
    # no-new-privileges is deliberately absent here: after the root wrapper's
    # zero-capability UID/GID transition, nested bwrap must regain only its
    # file-authorized setuid privilege.  The independent authority canary
    # above is the NNP=true validation boundary.
    assert "--security-opt=no-new-privileges:true" not in prefix
    wrapper = provider_boundary[image_index + 2]
    assert "/bin/chown 0:0 /usr/local/bin/bwrap" in wrapper
    assert "/bin/chmod 4755 /usr/local/bin/bwrap" in wrapper
    assert "exec /usr/bin/setpriv" in wrapper
    assert f"--reuid={os.getuid()} --regid={os.getgid()}" in wrapper
    assert "--clear-groups" in wrapper
    assert str(auth_path) not in profile
    assert f'"{grok_cli_runner._CODEX_FALLBACK_AUTH_DESTINATION}"="deny"' in profile
    assert (
        f'"{grok_cli_runner._CODEX_FALLBACK_CONTAINER_BWRAP_SOURCE}"="deny"' in profile
    )
    assert (
        f'"{grok_cli_runner._CODEX_FALLBACK_CONTAINER_INSTALLED_BWRAP}"="deny"'
        in profile
    )
    return prefix, profile, cidfile


def test_no_provider_codex_sandbox_smoke_and_profile_denials(
    tmp_path: Path,
) -> None:
    prefix, profile, cidfile = _codex_sandbox_boundary(tmp_path)
    fixed = [
        "sandbox",
        "-c",
        profile,
        "-c",
        'default_permissions="dcr_fallback"',
        "-P",
        "dcr_fallback",
    ]
    cases = (
        ("true", ["/bin/true"], 0),
        (
            "identity",
            [
                "/usr/bin/python3",
                "-c",
                (
                    "import os;"
                    "s=dict(x.split(':',1) for x in "
                    "open('/proc/self/status') if ':' in x);"
                    "print(os.getuid(),os.getgid(),"
                    "s['CapEff'].strip(),s['CapPrm'].strip(),"
                    "s['CapAmb'].strip(),s['CapBnd'].strip(),"
                    "s['NoNewPrivs'].strip())"
                ),
            ],
            0,
        ),
        (
            "auth_denied",
            [
                "/bin/cat",
                str(grok_cli_runner._CODEX_FALLBACK_AUTH_DESTINATION),
            ],
            1,
        ),
        (
            "copied_bwrap_denied",
            [
                str(grok_cli_runner._CODEX_FALLBACK_CONTAINER_INSTALLED_BWRAP),
                "--version",
            ],
            101,
        ),
        (
            "host_bwrap_denied",
            [
                str(grok_cli_runner._CODEX_FALLBACK_CONTAINER_BWRAP_SOURCE),
                "--version",
            ],
            101,
        ),
    )
    observed: dict[str, subprocess.CompletedProcess[str]] = {}
    for name, arguments, expected_returncode in cases:
        if cidfile.exists():
            cidfile.unlink()
        command = [*prefix, *fixed, *arguments]
        image_index = command.index(grok_cli_runner._CODEX_FALLBACK_IMAGE)
        assert command[image_index + 4] == "sandbox"
        assert "exec" not in command[image_index + 4 :]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
            env=grok_cli_runner._docker_control_env(),
        )
        observed[name] = completed
        assert completed.returncode == expected_returncode, completed.stdout

    assert observed["true"].stdout == ""
    assert observed["identity"].stdout.strip() == (
        f"{os.getuid()} {os.getgid()} "
        "0000000000000000 0000000000000000 0000000000000000 "
        "0000000000000000 1"
    )
    assert "Permission denied" in observed["auth_denied"].stdout
    assert "Permission denied" in observed["copied_bwrap_denied"].stdout
    assert "Permission denied" in observed["host_bwrap_denied"].stdout

    sealed = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))[
        "codex_no_provider_sandbox"
    ]
    recorded_argv = sealed["full_smoke_argv"]
    encoded_argv = json.dumps(
        recorded_argv,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sealed["full_smoke_argv_sha256"] == hashlib.sha256(encoded_argv).hexdigest()
    assert (
        sealed["root_wrapper_sha256"]
        == hashlib.sha256(
            grok_cli_runner._codex_fallback_root_wrapper().encode("utf-8")
        ).hexdigest()
    )
    recorded_image_index = recorded_argv.index(grok_cli_runner._CODEX_FALLBACK_IMAGE)
    assert "--network=none" in recorded_argv[:recorded_image_index]
    assert "--network=bridge" not in recorded_argv[:recorded_image_index]
    assert "--read-only" in recorded_argv[:recorded_image_index]
    assert "--cap-drop=ALL" in recorded_argv[:recorded_image_index]
    assert (
        "--security-opt=no-new-privileges:true"
        not in recorded_argv[:recorded_image_index]
    )
    assert recorded_argv[recorded_image_index + 4] == "sandbox"
    permission_index = recorded_argv.index("-P", recorded_image_index + 4)
    assert recorded_argv[permission_index + 1 : permission_index + 3] == [
        "dcr_fallback",
        "/bin/true",
    ]
    assert sealed["outer_policy"]["outer_no_new_privileges"] is False
    assert sealed["inner_policy"]["NoNewPrivs"] == 1
    for capability_field in ("CapEff", "CapPrm", "CapAmb", "CapBnd"):
        assert sealed["inner_policy"][capability_field] == ("0000000000000000")
    assert sealed["provider_invoked"] is False
    assert sealed["results"]["true"]["returncode"] == 0
    assert sealed["results"]["identity"]["returncode"] == 0
    assert sealed["results"]["auth_denied"]["returncode"] != 0
    assert sealed["results"]["copied_bwrap_denied"]["returncode"] != 0
    assert sealed["results"]["host_bwrap_denied"]["returncode"] != 0
