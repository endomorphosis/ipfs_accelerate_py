"""Security regressions for signed local profiles and once-only attempts."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    local_profile as local_profile_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    provider_attempt_store as provider_attempt_store_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileRevoked,
    LocalProfileTampered,
    ed25519_public_key_from_did,
    initialize_local_profile,
    load_local_profile,
    revoke_local_profile,
    rotate_local_profile,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.provider_attempt_store import (
    DurableProviderAttemptCAS,
    ProviderAttemptStoreError,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    grok_cli_runner as grok_cli_runner_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)


@pytest.fixture(autouse=True)
def _isolated_lifecycle_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_profile_module,
        "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE",
        tmp_path / "root-registry",
    )


def _locations(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "profile", tmp_path / "lifecycle"


def _launch_context(tag: str = "one") -> dict[str, str]:
    def identity(value: str) -> str:
        return "sha256:" + hashlib.sha256(value.encode()).hexdigest()

    return {
        "provider_id": "codex",
        "command_id": identity(f"command:{tag}"),
        "runtime_id": identity(f"runtime:{tag}"),
        "image_id": identity(f"image:{tag}"),
        "mount_id": identity(f"mount:{tag}"),
        "environment_id": identity(f"environment:{tag}"),
        "container_name": f"test-container-{tag}",
        "container_id": identity(f"container:{tag}"),
    }


def _initialize(tmp_path: Path):
    profile_dir, lifecycle_dir = _locations(tmp_path)
    return initialize_local_profile(
        repository_cid="repository:one",
        baseline_commit="a" * 40,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )


@pytest.mark.parametrize("boundary", ["profile", "attempt_store", "capsule"])
def test_authority_parent_creation_rejects_inserted_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    """A parent inserted after the lexical walk can never redirect creation."""

    attacker = tmp_path / f"attacker-{boundary}"
    attacker.mkdir(mode=0o700)
    marker = f"missing-{boundary}-parent"
    original_mkdir = os.mkdir
    inserted = False

    def insert_then_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal inserted
        if str(path) == marker and not inserted:
            inserted = True
            os.symlink(attacker, path, dir_fd=dir_fd)
        original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", insert_then_mkdir)
    target = tmp_path / marker / "authority"
    if boundary == "profile":
        lifecycle = tmp_path / "existing-lifecycle"
        lifecycle.mkdir(mode=0o700)
        with pytest.raises(LocalProfileTampered):
            initialize_local_profile(
                repository_cid="repository:parent-race",
                baseline_commit="a" * 40,
                profile_dir=target,
                lifecycle_dir=lifecycle,
            )
    elif boundary == "attempt_store":
        with pytest.raises(ProviderAttemptStoreError):
            DurableProviderAttemptCAS(target)
    else:
        with pytest.raises(ValueError):
            llm_router._agent_create_private_directory_chain(
                target,
                final_mode=0o700,
            )
    assert inserted
    assert not (attacker / "authority").exists()


def _base58btc_decode(value: str) -> bytes:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    integer = 0
    for character in value:
        integer = integer * 58 + alphabet.index(character)
    body = (
        integer.to_bytes((integer.bit_length() + 7) // 8, "big")
        if integer
        else b""
    )
    return b"\0" * (len(value) - len(value.lstrip("1"))) + body


def test_provider_output_cannot_inject_protected_lf_records() -> None:
    prefix = grok_cli_runner_module.AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX

    class Chunks:
        def __init__(self, values: list[str]) -> None:
            self.values = iter(values)

        def read(self, _size: int) -> str:
            return next(self.values, "")

    destination = io.StringIO()
    grok_cli_runner_module._stream_provider_pipe_without_reserved_records(
        Chunks([prefix[:7], prefix[7:] + "{}\n"]),  # type: ignore[arg-type]
        destination,
    )
    rendered = destination.getvalue()
    assert rendered.startswith("[provider-child-output-escaped] ")
    assert not rendered.startswith(prefix)

    destination = io.StringIO()
    grok_cli_runner_module._stream_provider_pipe_without_reserved_records(
        Chunks(
            [
                "harmless\r" + prefix + "{}\n",
                prefix + ("x" * (20 * 1024)) + "\n",
                "harmless\u2028" + prefix + "{}\n",
            ]
        ),  # type: ignore[arg-type]
        destination,
    )
    lines = destination.getvalue().split("\n")
    assert all(not line.startswith(prefix) for line in lines)
    assert "\r" not in destination.getvalue()
    assert "\u2028" not in destination.getvalue()


def test_owned_log_tail_rejects_link_mode_and_stability_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "runner.log"
    log.write_text("discarded-partial\ntrusted-record\n", encoding="utf-8")
    log.chmod(0o600)
    text, receipt_text = implementation_daemon_module._stable_owned_log_tail(
        log,
        15,
    )
    assert text.endswith("trusted-record\n")
    assert receipt_text == "trusted-record\n"

    symlink = tmp_path / "runner-symlink.log"
    symlink.symlink_to(log)
    with pytest.raises(OSError):
        implementation_daemon_module._stable_owned_log_tail(symlink, 1024)
    hardlink = tmp_path / "runner-hardlink.log"
    os.link(log, hardlink)
    with pytest.raises(OSError):
        implementation_daemon_module._stable_owned_log_tail(log, 1024)
    hardlink.unlink()
    log.chmod(0o620)
    with pytest.raises(OSError):
        implementation_daemon_module._stable_owned_log_tail(log, 1024)
    log.chmod(0o600)

    original_read = implementation_daemon_module.os.read
    mutated = False

    def growing_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        value = original_read(descriptor, size)
        if not mutated:
            mutated = True
            with log.open("ab") as stream:
                stream.write(b"attacker-growth\n")
        return value

    monkeypatch.setattr(
        implementation_daemon_module.os,
        "read",
        growing_read,
    )
    with pytest.raises(OSError, match="changed"):
        implementation_daemon_module._stable_owned_log_tail(log, 1024)


def test_private_attempt_log_repairs_umask_0002_without_truncating_existing_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "attempt.log"
    created = subprocess.run(
        [
            "bash",
            "-c",
            'umask 0002; printf stale-authority > "$1"',
            "attempt-log",
            str(log),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    assert log.stat().st_mode & 0o777 == 0o664
    original_inode = log.stat().st_ino

    def forbidden_truncate(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("private log writes must never truncate an inode")

    monkeypatch.setattr(
        implementation_daemon_module.os,
        "ftruncate",
        forbidden_truncate,
    )

    with implementation_daemon_module._open_private_implementation_log(
        log,
        "w",
    ) as stream:
        stream.write("trusted-record\n")
    assert log.stat().st_mode & 0o777 == 0o600
    assert log.stat().st_ino != original_inode
    with implementation_daemon_module._open_private_implementation_log(
        log,
        "a",
    ) as stream:
        stream.write("trusted-append\n")
    text, receipt_text = implementation_daemon_module._stable_owned_log_tail(
        log,
        1024,
    )
    assert text == "trusted-record\ntrusted-append\n"
    assert receipt_text == text

    victim = tmp_path / "victim.log"
    victim.write_text("must-survive\n", encoding="utf-8")
    hardlink = tmp_path / "hardlink.log"
    os.link(victim, hardlink)
    with implementation_daemon_module._open_private_implementation_log(
        hardlink,
        "w",
    ) as stream:
        stream.write("replacement\n")
    assert victim.read_text(encoding="utf-8") == "must-survive\n"
    assert hardlink.read_text(encoding="utf-8") == "replacement\n"
    assert hardlink.stat().st_ino != victim.stat().st_ino

    symlink = tmp_path / "symlink.log"
    symlink.symlink_to(victim)
    with implementation_daemon_module._open_private_implementation_log(
        symlink,
        "w",
    ) as stream:
        stream.write("symlink-replacement\n")
    assert victim.read_text(encoding="utf-8") == "must-survive\n"
    assert not symlink.is_symlink()
    assert symlink.read_text(encoding="utf-8") == "symlink-replacement\n"


def test_private_attempt_log_fails_closed_on_hardlink_at_replace_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "attempt.log"
    log.write_text("old-record\n", encoding="utf-8")
    protected_old_inode = tmp_path / "old-inode.log"
    os.link(log, protected_old_inode)
    real_replace = implementation_daemon_module.os.replace

    def same_inode_rename_boundary(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        os.unlink(destination, dir_fd=dst_dir_fd)
        os.link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=False,
        )
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(
        implementation_daemon_module.os,
        "replace",
        same_inode_rename_boundary,
    )
    with pytest.raises(OSError, match="changed while opening"):
        implementation_daemon_module._open_private_implementation_log(
            log,
            "w",
        )

    assert protected_old_inode.read_text(encoding="utf-8") == "old-record\n"
    assert log.stat().st_nlink == 1
    assert not list(tmp_path.glob(".implementation-log-*.tmp"))


def test_private_attempt_log_append_rechecks_hardlink_after_mode_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "attempt.log"
    log.write_text("trusted-record\n", encoding="utf-8")
    log.chmod(0o664)
    injected_alias = tmp_path / "injected-alias.log"
    real_fchmod = implementation_daemon_module.os.fchmod

    def hardlink_then_fchmod(descriptor: int, mode: int) -> None:
        os.link(log, injected_alias)
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(
        implementation_daemon_module.os,
        "fchmod",
        hardlink_then_fchmod,
    )
    with pytest.raises(OSError, match="identity changed before use"):
        implementation_daemon_module._open_private_implementation_log(
            log,
            "a",
        )

    assert log.read_text(encoding="utf-8") == "trusted-record\n"
    assert injected_alias.read_text(encoding="utf-8") == "trusted-record\n"


def test_codex_effect_is_created_then_cas_claimed_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}", encoding="utf-8")
    source_auth.chmod(0o600)
    fake_home_path = tmp_path / "codex-home"
    fake_home_path.mkdir()
    events: list[str] = []

    class FakeHome:
        name = str(fake_home_path)

        def cleanup(self) -> None:
            events.append("home_cleanup")

    class FakeLease:
        container_name = "ipfs-accelerate-codex-123-" + "a" * 32
        lease_root = tmp_path / "lease-root"
        docker_config = lease_root / "docker-config"
        cidfile = lease_root / "container.cid"
        provider_home = fake_home_path
        prompt_path = tmp_path / "prompt.txt"
        _watchdog = SimpleNamespace(pid=os.getpid())

        def mark_cas_owned(self) -> None:
            events.append("mark_cas_owned")

        def mark_cas_terminal(self) -> None:
            events.append("mark_cas_terminal")

        def close(self, *, docker_run_finished: bool) -> None:
            assert docker_run_finished
            events.append("lease_close")

    monkeypatch.setattr(
        grok_cli_runner_module,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_isolated_codex_quota_fallback_home",
        lambda **_kwargs: (
            FakeHome(),
            grok_cli_runner_module._codex_task_container_environment(),
            source_auth,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner_module._DockerContainerLease,
        "create",
        lambda *_args, **_kwargs: FakeLease(),
    )
    image_id = grok_cli_runner_module._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_docker_codex_task_toolchain_image_id",
        lambda *_args, **_kwargs: image_id,
    )
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_validated_codex_auth_path",
        lambda **_kwargs: source_auth,
    )
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_git_metadata_roots",
        lambda _workspace: (),
    )
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_robust_remove_runner_temp_tree",
        lambda _path: None,
    )

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        assert "create" in command
        assert "run" not in command
        assert "--rm" not in command
        events.append("create")
        return SimpleNamespace(
            returncode=0,
            stdout=("8" * 64 + "\n").encode("ascii"),
            stderr=b"",
        )

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: object) -> FakeProcess:
        assert command[4] == "start"
        assert events[-1] == "mark_cas_owned"
        events.append("start")
        return FakeProcess()

    monkeypatch.setattr(grok_cli_runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(grok_cli_runner_module.subprocess, "Popen", fake_popen)

    command = [
        str(codex),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="high"',
        "-",
    ]

    def claim(context: dict[str, str]) -> None:
        assert events[-1] == "post_create_validate"
        assert context["container_id"] == "sha256:" + "8" * 64
        events.append("claim")

    validation_count = 0

    def validate() -> None:
        nonlocal validation_count
        validation_count += 1
        events.append(
            "post_create_validate" if "create" in events else "pre_create_validate"
        )

    def terminal(returncode: int) -> None:
        assert returncode == 0
        assert events[-1] == "start"
        events.append("terminal")

    returncode = grok_cli_runner_module._run_codex_quota_fallback_in_docker(
        command,
        workspace=workspace,
        prompt="prompt",
        prompt_path=prompt_path,
        base_env={},
        pre_effect_validator=validate,
        effect_claim=claim,
        effect_terminal=terminal,
    )
    assert returncode == 0
    assert validation_count == 3
    assert events[:6] == [
        "pre_create_validate",
        "pre_create_validate",
        "create",
        "post_create_validate",
        "claim",
        "mark_cas_owned",
    ]
    assert events[6:9] == ["start", "terminal", "mark_cas_terminal"]


def test_local_profile_key_must_remain_private_owned_regular_file(tmp_path: Path) -> None:
    _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)
    key = profile_dir / "local_dev_profile.key"
    key.chmod(0o640)
    with pytest.raises(LocalProfileTampered):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )


@pytest.mark.parametrize("linked_directory", ["profile", "lifecycle"])
def test_local_authority_rejects_symlinked_directory_components(
    tmp_path: Path,
    linked_directory: str,
) -> None:
    _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)
    target = profile_dir if linked_directory == "profile" else lifecycle_dir
    alias = tmp_path / f"{linked_directory}-alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(LocalProfileTampered, match="symlink"):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=alias if linked_directory == "profile" else profile_dir,
            lifecycle_dir=(
                alias if linked_directory == "lifecycle" else lifecycle_dir
            ),
        )


def test_local_authority_rejects_dangling_directory_symlink(tmp_path: Path) -> None:
    dangling = tmp_path / "dangling-profile"
    dangling.symlink_to(tmp_path / "missing-profile", target_is_directory=True)
    with pytest.raises(LocalProfileTampered, match="symlink"):
        initialize_local_profile(
            repository_cid="repository:one",
            baseline_commit="a" * 40,
            profile_dir=dangling,
            lifecycle_dir=tmp_path / "lifecycle",
        )


@pytest.mark.parametrize("artifact", ["profile", "key", "signature", "anchor"])
def test_local_authority_rejects_multiply_linked_read_artifacts(
    tmp_path: Path,
    artifact: str,
) -> None:
    _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)
    paths = {
        "profile": profile_dir / "local_dev_profile.json",
        "key": profile_dir / "local_dev_profile.key",
        "signature": profile_dir / "local_dev_profile.sig",
        "anchor": next(lifecycle_dir.glob("*.json")),
    }
    os.link(paths[artifact], tmp_path / f"{artifact}-hardlink")
    with pytest.raises(LocalProfileTampered):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )


def test_local_authority_rejects_oversized_and_duplicate_json(tmp_path: Path) -> None:
    _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)
    profile_path = profile_dir / "local_dev_profile.json"
    original = profile_path.read_bytes()
    profile_path.write_bytes(b'{"schema":"duplicate",' + original[1:])
    with pytest.raises(LocalProfileTampered):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )

    profile_path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(LocalProfileTampered):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )


def test_local_authority_rejects_path_swap_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)
    profile_path = profile_dir / "local_dev_profile.json"
    replacement = tmp_path / "replacement-profile.json"
    replacement.write_bytes(profile_path.read_bytes())
    replacement.chmod(0o600)
    original_read = local_profile_module.os.read
    swapped = False

    def swap_after_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if not swapped:
            swapped = True
            os.replace(replacement, profile_path)
        return chunk

    monkeypatch.setattr(local_profile_module.os, "read", swap_after_read)
    with pytest.raises(LocalProfileTampered):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )


def test_profile_identity_is_a_standard_ed25519_did_key(tmp_path: Path) -> None:
    profile = _initialize(tmp_path)
    assert profile.identity_did.startswith("did:key:z")
    decoded = _base58btc_decode(profile.identity_did.removeprefix("did:key:z"))
    assert decoded[:2] == b"\xed\x01"
    assert len(decoded) == 34
    public = ed25519_public_key_from_did(profile.identity_did)
    assert public.public_bytes(Encoding.Raw, PublicFormat.Raw) == decoded[2:]
    assert profile.reviewer_identity == profile.identity_did
    assert profile.reviewer_provider == "local_operator"
    assert profile.route_id.endswith("auth-or-hard-quota-v1")
    assert (
        profile.fallback_provider_id,
        profile.fallback_model_id,
        profile.fallback_reasoning_effort,
    ) == ("codex", "gpt-5.6-terra", "high")


def test_rotation_uses_a_fresh_key_and_copied_old_authority_cannot_revive(
    tmp_path: Path,
) -> None:
    original = _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)
    backup = tmp_path / "old-profile-copy"
    shutil.copytree(profile_dir, backup)
    old_key = (profile_dir / "local_dev_profile.key").read_bytes()

    receipt = rotate_local_profile(
        repository_cid="repository:one",
        baseline_commit="b" * 40,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )
    rotated = load_local_profile(
        repository_cid="repository:one",
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )
    assert receipt.old_identity_did == original.identity_did
    assert receipt.new_identity_did == rotated.identity_did
    assert rotated.identity_did != original.identity_did
    assert receipt.lifecycle_generation == 2

    with pytest.raises(LocalProfileTampered, match="fresh"):
        rotate_local_profile(
            repository_cid="repository:one",
            baseline_commit="c" * 40,
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
            signing_key=(profile_dir / "local_dev_profile.key").read_bytes(),
        )
    assert old_key != (profile_dir / "local_dev_profile.key").read_bytes()

    shutil.rmtree(profile_dir)
    shutil.copytree(backup, profile_dir)
    with pytest.raises(LocalProfileTampered, match="monotonic lifecycle"):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )


def test_serialized_concurrent_rotations_advance_monotonically(tmp_path: Path) -> None:
    _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)

    def rotate(baseline: str):
        return rotate_local_profile(
            repository_cid="repository:one",
            baseline_commit=baseline,
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = tuple(
            executor.map(rotate, ("b" * 40, "c" * 40))
        )
    assert sorted(receipt.lifecycle_generation for receipt in receipts) == [2, 3]
    current = load_local_profile(
        repository_cid="repository:one",
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )
    assert current.lifecycle_generation == 3


def test_attempt_cas_adopts_only_the_identical_logical_attempt(tmp_path: Path) -> None:
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    first = store.reserve_or_adopt(
        logical_attempt_id="attempt:1", route_id="route:1", decision_id="decision:1",
        task_id="task:1", worktree_id="worktree:1", authorized=True,
    )
    adopted = store.reserve_or_adopt(
        logical_attempt_id="attempt:1", route_id="route:1", decision_id="decision:1",
        task_id="task:1", worktree_id="worktree:1", authorized=True,
    )
    assert first.created and adopted.adopted
    with pytest.raises(ProviderAttemptStoreError):
        store.reserve_or_adopt(
            logical_attempt_id="attempt:1", route_id="route:other", decision_id="decision:1",
            task_id="task:1", worktree_id="worktree:1", authorized=True,
        )


def test_attempt_cas_rejects_symlinked_and_dangling_store_directories(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual-attempts"
    DurableProviderAttemptCAS(actual)
    alias = tmp_path / "attempts-alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ProviderAttemptStoreError, match="symlink"):
        DurableProviderAttemptCAS(alias)

    dangling = tmp_path / "dangling-attempts"
    dangling.symlink_to(tmp_path / "missing-attempts", target_is_directory=True)
    with pytest.raises(ProviderAttemptStoreError, match="symlink"):
        DurableProviderAttemptCAS(dangling)


@pytest.mark.parametrize(
    "corruption",
    ["symlink", "dangling_symlink", "hardlink", "permissions", "oversized", "duplicate"],
)
def test_attempt_cas_rejects_unsafe_reservation_storage(
    tmp_path: Path,
    corruption: str,
) -> None:
    directory = tmp_path / "attempts"
    store = DurableProviderAttemptCAS(directory)
    values = {
        "logical_attempt_id": "attempt:unsafe",
        "route_id": "route:1",
        "decision_id": "decision:1",
        "task_id": "task:1",
        "worktree_id": "worktree:1",
        "authorized": True,
    }
    store.reserve_or_adopt(**values)
    reservation_path = next(directory.glob("*.json"))

    if corruption in {"symlink", "dangling_symlink"}:
        saved = tmp_path / "saved-reservation"
        reservation_path.replace(saved)
        reservation_path.symlink_to(
            saved if corruption == "symlink" else tmp_path / "missing-reservation"
        )
    elif corruption == "hardlink":
        os.link(reservation_path, tmp_path / "reservation-hardlink")
    elif corruption == "permissions":
        reservation_path.chmod(0o640)
    elif corruption == "oversized":
        reservation_path.write_bytes(b" " * (256 * 1024 + 1))
    else:
        original = json.loads(reservation_path.read_text(encoding="utf-8"))
        encoded = json.dumps(original, sort_keys=True, separators=(",", ":"))
        reservation_path.write_text(
            '{"schema":"duplicate",' + encoded[1:],
            encoding="utf-8",
        )

    with pytest.raises(ProviderAttemptStoreError):
        store.reserve_or_adopt(**values)


def test_attempt_cas_rejects_file_growth_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "attempts"
    store = DurableProviderAttemptCAS(directory)
    store.reserve_or_adopt(
        logical_attempt_id="attempt:growing",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
    )
    reservation_path = next(directory.glob("*.json"))
    original_read = provider_attempt_store_module.os.read
    grew = False

    def grow_after_read(descriptor: int, size: int) -> bytes:
        nonlocal grew
        chunk = original_read(descriptor, size)
        if not grew:
            grew = True
            with reservation_path.open("ab") as stream:
                stream.write(b" ")
        return chunk

    monkeypatch.setattr(provider_attempt_store_module.os, "read", grow_after_read)
    with pytest.raises(ProviderAttemptStoreError):
        store.read("attempt:growing")


def test_attempt_cas_authorizes_exactly_one_concurrent_effect(tmp_path: Path) -> None:
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    reserved = store.reserve_or_adopt(
        logical_attempt_id="attempt:concurrent",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
    ).reservation

    with ThreadPoolExecutor(max_workers=12) as executor:
        claims = tuple(
            executor.map(
                lambda _: store.claim_effect(
                    reserved,
                    launch_context=_launch_context("concurrent"),
                ),
                range(24),
            )
        )
    winners = tuple(claim for claim in claims if claim.launch_authorized)
    assert len(winners) == 1
    assert all(
        claim.reservation.reservation_id == reserved.reservation_id
        for claim in claims
    )
    assert sum(claim.adopted for claim in claims) == 23
    with pytest.raises(ProviderAttemptStoreError, match="effect winner"):
        store.complete(
            winners[0].reservation,
            completion_capability="0" * 64,
        )
    store.complete(
        winners[0].reservation,
        completion_capability=winners[0].completion_capability,
    )


def test_restart_adopts_effect_started_and_terminal_without_replay(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "attempts"
    first_store = DurableProviderAttemptCAS(directory)
    initial = first_store.reserve_or_adopt(
        logical_attempt_id="attempt:crash",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
    )
    claimed = first_store.claim_effect(
        initial.reservation,
        launch_context=_launch_context("crash"),
    )
    assert claimed.launch_authorized

    restarted = DurableProviderAttemptCAS(directory)
    adopted = restarted.reserve_or_adopt(
        logical_attempt_id="attempt:crash",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
    )
    assert adopted.adopted and adopted.reservation.state == "effect_started"
    restarted_claim = restarted.claim_effect(
        adopted.reservation,
        launch_context=_launch_context("crash"),
    )
    assert not restarted_claim.launch_authorized
    with pytest.raises(ProviderAttemptStoreError, match="effect winner"):
        restarted.complete(
            adopted.reservation,
            returncode=17,
            outcome={"result": "failed", "returncode": 17},
            completion_capability=restarted_claim.completion_capability,
        )

    terminal = first_store.complete(
        adopted.reservation,
        returncode=17,
        outcome={"result": "failed", "returncode": 17},
        completion_capability=claimed.completion_capability,
    )
    second_restart = DurableProviderAttemptCAS(directory)
    terminal_adoption = second_restart.reserve_or_adopt(
        logical_attempt_id="attempt:crash",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
    )
    assert terminal_adoption.reservation == terminal
    assert terminal_adoption.reservation.terminal_returncode == 17
    assert not second_restart.claim_effect(
        terminal,
        launch_context=_launch_context("crash"),
    ).launch_authorized


def test_effect_adoption_uses_only_the_store_owned_native_inspector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    reserved = store.reserve_or_adopt(
        logical_attempt_id="attempt:adoption",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
    )
    claimed = store.claim_effect(
        reserved.reservation,
        launch_context=_launch_context("adoption"),
    )
    with pytest.raises(TypeError):
        store.adopt_effect(
            claimed.reservation,
            inspect_effect=lambda _receipt, _timestamp: {  # type: ignore[call-arg]
                "status": "absent"
            },
        )

    monkeypatch.setattr(
        provider_attempt_store_module,
        "_process_identity_alive",
        lambda _pid, _start: False,
    )

    def native_inspection(
        launch: dict[str, object],
        timestamp: int,
    ) -> dict[str, object]:
        return {
            "status": "exited",
            "inspection_runtime_id": launch["runtime_id"],
            "inspection_command_id": "sha256:" + "a" * 64,
            "observed_at_ms": timestamp,
            "provider_id": launch["provider_id"],
            "command_id": launch["command_id"],
            "runtime_id": launch["runtime_id"],
            "image_id": launch["image_id"],
            "mount_id": launch["mount_id"],
            "environment_id": launch["environment_id"],
            "container_name": launch["container_name"],
            "container_id": launch["container_id"],
            "returncode": 17,
        }

    monkeypatch.setattr(
        provider_attempt_store_module,
        "_inspect_recorded_docker_effect",
        native_inspection,
    )
    adopter_id = "sha256:" + "b" * 64
    adopted = store.adopt_effect(
        claimed.reservation,
        effect_owner_id=adopter_id,
        now_ms=int(claimed.reservation.effect_started_at_ms or 0) + 1,
    )
    assert adopted.adoption_authorized
    assert adopted.reservation.effect_adoption_generation == 1
    assert (
        adopted.reservation.effect_adoption_receipt["inspection_status"]
        == "exited"
    )
    terminal = store.complete(
        adopted.reservation,
        returncode=17,
        outcome={"result": "failed", "returncode": 17},
        completion_capability=adopted.completion_capability,
        effect_owner_id=adopter_id,
        now_ms=int(claimed.reservation.effect_started_at_ms or 0) + 2,
    )
    assert terminal.terminal


def test_protected_attempt_latch_categorically_fences_counter_restore(
    tmp_path: Path,
) -> None:
    del tmp_path
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    revision = "task-revision:one"
    task = PortalTask(
        task_id="task:one",
        title="one",
        status="ready",
        completion="merged",
        priority="high",
        track="implementation",
        canonical_task_key="task-key:one",
        canonical_task_cid=revision,
        board_namespace="board:one",
    )
    daemon._identity_for_task = lambda _task: SimpleNamespace(
        canonical_task_cid=revision,
        canonical_task_key="task-key:one",
    )
    daemon.todo_path = Path("board:one")
    daemon.max_task_attempts = 3
    daemon._scoped_recovery_attempts = {}
    body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "protected-implementation-attempt-latch@1"
        ),
        "task_id": task.task_id,
        "attempt": 3,
        "task_revision_cid": revision,
        "board_namespace": task.board_namespace,
        "route_id": "route:protected",
        "invocation_id": "invocation:one",
        "logical_attempt_id": "logical:one",
        "worktree_id": "worktree:one",
        "provider_attempt_store": "/private/attempts",
        "provider_attempt_store_identity": "sha256:" + "c" * 64,
    }
    record = {**body, "latch_id": content_identity(body)}
    key = daemon._protected_attempt_latch_key(task.task_id, 3, revision)
    event = {"type": "protected_implementation_attempt_latched", **record}
    daemon._iter_events = lambda: iter((event,))
    state = PortalTaskState(
        implementation_attempts={task.task_id: 3},
        implementation_attempts_by_cid={revision: 3},
        protected_implementation_attempts={key: record},
        active_task_cid=revision,
        last_implementation_task_cid=revision,
    )
    daemon._restore_task_attempt(state, task, 0)
    assert state.implementation_attempts == {task.task_id: 3}
    assert state.implementation_attempts_by_cid == {revision: 3}
    released = daemon._release_unfinished_active_attempt(
        state,
        task_id=task.task_id,
        attempt=3,
    )
    assert released["protected_scoped_attempt"] is True
    assert released["released"] is False
    assert daemon._task_attempt(state, task) == 3
    selectable, limited = daemon._partition_tasks_at_attempt_limit(
        [task],
        {task.task_id: "ready"},
        state,
    )
    assert selectable == [task]
    assert limited == []

    # A state-file truncation cannot erase the append-only protected latch.
    event_only = PortalTaskState(
        implementation_attempts={task.task_id: 3},
        implementation_attempts_by_cid={revision: 3},
        active_task_cid=revision,
        last_implementation_task_cid=revision,
    )
    daemon._restore_task_attempt(event_only, task, 0)
    assert event_only.implementation_attempts[task.task_id] == 3
    assert not daemon._release_unfinished_active_attempt(
        event_only,
        task_id=task.task_id,
        attempt=3,
    )["released"]

    # A new revision with the same display ID does not inherit the old latch.
    new_revision = "task-revision:two"
    daemon._identity_for_task = lambda _task: SimpleNamespace(
        canonical_task_cid=new_revision,
        canonical_task_key="task-key:two",
    )
    replacement = PortalTask(
        **{
            **task.__dict__,
            "canonical_task_key": "task-key:two",
            "canonical_task_cid": new_revision,
        }
    )
    later = PortalTaskState(
        implementation_attempts={task.task_id: 1},
        implementation_attempts_by_cid={new_revision: 1},
        protected_implementation_attempts={key: record},
    )
    daemon._restore_task_attempt(later, replacement, 0)
    assert task.task_id not in later.implementation_attempts
    assert new_revision not in later.implementation_attempts_by_cid


def test_attempt_cas_rejects_backdated_transitions_without_mutating_state(
    tmp_path: Path,
) -> None:
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    reserved = store.reserve_or_adopt(
        logical_attempt_id="attempt:time",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
        now_ms=100,
    ).reservation
    with pytest.raises(ProviderAttemptStoreError, match="predates"):
        store.claim_effect(
            reserved,
            launch_context=_launch_context("time"),
            now_ms=99,
        )
    assert store.read("attempt:time").state == "reserved"

    claim = store.claim_effect(
        reserved,
        launch_context=_launch_context("time"),
        now_ms=110,
    )
    claimed = claim.reservation
    with pytest.raises(ProviderAttemptStoreError, match="predates"):
        store.complete(
            claimed,
            completion_capability=claim.completion_capability,
            now_ms=109,
        )
    assert store.read("attempt:time").state == "effect_started"


@pytest.mark.parametrize("invalid", [True, -1, 1.5, "100"])
def test_attempt_cas_rejects_noncanonical_transition_timestamp_types(
    tmp_path: Path,
    invalid: object,
) -> None:
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    with pytest.raises(ProviderAttemptStoreError, match="timestamp"):
        store.reserve_or_adopt(
            logical_attempt_id="attempt:invalid-time",
            route_id="route:1",
            decision_id="decision:1",
            task_id="task:1",
            worktree_id="worktree:1",
            authorized=True,
            now_ms=invalid,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field,value,state",
    [
        ("created_at_ms", True, "reserved"),
        ("created_at_ms", "100", "reserved"),
        ("created_at_ms", -1, "reserved"),
        ("effect_started_at_ms", 99, "effect_started"),
        ("terminal_at_ms", 109, "terminal"),
    ],
)
def test_attempt_cas_rejects_malformed_or_nonmonotonic_durable_timestamps(
    tmp_path: Path,
    field: str,
    value: object,
    state: str,
) -> None:
    directory = tmp_path / "attempts"
    store = DurableProviderAttemptCAS(directory)
    reservation = store.reserve_or_adopt(
        logical_attempt_id="attempt:malformed-time",
        route_id="route:1",
        decision_id="decision:1",
        task_id="task:1",
        worktree_id="worktree:1",
        authorized=True,
        now_ms=100,
    ).reservation
    completion_capability = ""
    if state in {"effect_started", "terminal"}:
        claim = store.claim_effect(
            reservation,
            launch_context=_launch_context("malformed"),
            now_ms=110,
        )
        reservation = claim.reservation
        completion_capability = claim.completion_capability
    if state == "terminal":
        reservation = store.complete(
            reservation,
            completion_capability=completion_capability,
            now_ms=120,
        )

    path = next(directory.glob("*.json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[field] = value
    path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProviderAttemptStoreError):
        store.read("attempt:malformed-time")


def test_signed_revocation_history_survives_marker_removal(tmp_path: Path) -> None:
    _initialize(tmp_path)
    profile_dir, lifecycle_dir = _locations(tmp_path)
    backup = tmp_path / "pre-revocation-copy"
    shutil.copytree(profile_dir, backup)
    revoke_local_profile(profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    (profile_dir / "local_dev_profile.revoked").unlink()
    with pytest.raises(LocalProfileRevoked):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )

    shutil.rmtree(profile_dir)
    shutil.copytree(backup, profile_dir)
    with pytest.raises(LocalProfileRevoked):
        load_local_profile(
            repository_cid="repository:one",
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
        )


def test_implementation_finish_requires_exact_revision_and_board_namespace() -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    task_id = "reused-display-id"
    attempt = 2
    revision = "task-revision:current"
    namespace = "board:current"
    exact_event = {
        "type": "implementation_finished",
        "task_id": task_id,
        "attempt": attempt,
        "canonical_task_cid": revision,
        "board_namespace": namespace,
    }

    def has_finish(
        event: dict[str, object],
        *,
        requested_revision: str = revision,
        requested_namespace: str = namespace,
    ) -> bool:
        daemon._iter_events = lambda: iter((event,))
        return daemon._task_attempt_has_implementation_finish(
            task_id,
            attempt,
            task_revision_cid=requested_revision,
            board_namespace=requested_namespace,
        )

    assert has_finish(exact_event)
    assert not has_finish(
        {**exact_event, "canonical_task_cid": "task-revision:foreign"}
    )
    assert not has_finish(
        {key: value for key, value in exact_event.items() if key != "canonical_task_cid"}
    )
    assert not has_finish({**exact_event, "board_namespace": "board:foreign"})
    assert not has_finish(
        {key: value for key, value in exact_event.items() if key != "board_namespace"}
    )
    assert not has_finish(exact_event, requested_revision="")
    assert not has_finish(exact_event, requested_namespace="")


def test_docker_cleanup_watchdog_launch_is_python_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = tmp_path / "hostile-python"
    hostile.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(hostile))
    monkeypatch.setenv("PYTHONHOME", str(hostile / "fake-home"))
    captured: dict[str, object] = {}
    real_popen = grok_cli_runner_module.subprocess.Popen

    class FakeWatchdog:
        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FakeWatchdog:
        captured["command"] = command
        captured.update(kwargs)
        return FakeWatchdog()

    monkeypatch.setattr(grok_cli_runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_remove_exact_docker_container",
        lambda **_kwargs: None,
    )
    lease = grok_cli_runner_module._DockerContainerLease.create(
        "/usr/bin/docker",
        provider="codex",
        provider_home=tmp_path / "asref-codex-home-test",
        prompt_path=tmp_path / "asref-grok-prompt-test",
    )
    try:
        command = captured["command"]
        assert isinstance(command, list)
        assert command[:2] == [grok_cli_runner_module.sys.executable, "-I"]
        assert command[2] == str(Path(grok_cli_runner_module.__file__).resolve())
        assert command[3] == grok_cli_runner_module._DOCKER_CLEANUP_WATCHDOG_ARG
        assert captured["cwd"] == "/"
        assert captured["env"] == {
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
        }
        assert captured["start_new_session"] is True
        assert captured["close_fds"] is True
        assert captured["pass_fds"] == ()
        isolation_probe = real_popen(
            [*command[:2], "-c", "pass"],
            stdin=grok_cli_runner_module.subprocess.DEVNULL,
            stdout=grok_cli_runner_module.subprocess.DEVNULL,
            stderr=grok_cli_runner_module.subprocess.DEVNULL,
            cwd=captured["cwd"],
            env=captured["env"],
        )
        assert isolation_probe.wait(timeout=5) == 0
        assert not marker.exists()
    finally:
        lease.close(docker_run_finished=False)
