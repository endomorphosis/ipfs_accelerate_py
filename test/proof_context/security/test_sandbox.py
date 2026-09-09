from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import ipfs_accelerate_py.proof_context.sandbox as sandbox
import pytest
from ipfs_accelerate_py.proof_context.adapters.base import CancellationToken
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    MalformedError,
    PartialEffectError,
    PseudoCidError,
    RepairRequiredError,
    SchemaMismatchError,
    UnknownFieldError,
)
from ipfs_accelerate_py.proof_context.sandbox import (
    CANONICAL_BRANCH_AUTHORITY,
    ENFORCEMENT_DISPOSITION,
    PRODUCTION_ELIGIBLE,
    PUBLICATION_AUTHORITY,
    RUNTIME_INTEGRATION_STATUS,
    DescriptorRoot,
    DisposableWorktreeGuard,
    SandboxCapabilityReport,
    SandboxCommandAdapter,
    SandboxDenialTrace,
    SandboxExecutionPermit,
    SandboxExecutionReceipt,
    SandboxExecutionResult,
    SandboxExecutor,
    SandboxPolicy,
    sandbox_descriptor,
    sandbox_descriptor_cid,
)

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "PCCE sandbox test",
    "GIT_AUTHOR_EMAIL": "sandbox@example.invalid",
    "GIT_COMMITTER_NAME": "PCCE sandbox test",
    "GIT_COMMITTER_EMAIL": "sandbox@example.invalid",
}

_STATIC_HELPER_SOURCE = r"""
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

static int denied(void) {
    if (errno == EPERM || errno == EACCES) {
        puts("denied");
        return 0;
    }
    fprintf(stderr, "unexpected errno=%d\n", errno);
    return 42;
}

static int copy_stdin(void) {
    char buffer[4096];
    ssize_t count;
    while ((count = read(STDIN_FILENO, buffer, sizeof(buffer))) > 0) {
        ssize_t offset = 0;
        while (offset < count) {
            ssize_t written = write(STDOUT_FILENO, buffer + offset, (size_t)(count - offset));
            if (written < 0) return 43;
            offset += written;
        }
    }
    return count < 0 ? 44 : 0;
}

static int path_read(const char *path) {
    char buffer[64];
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return denied();
    ssize_t count = read(fd, buffer, sizeof(buffer));
    close(fd);
    if (count < 0) return 45;
    return write(STDOUT_FILENO, buffer, (size_t)count) == count ? 0 : 46;
}

static int path_write(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return denied();
    close(fd);
    puts("write-breached");
    return 47;
}

static int socket_probe(int domain, int type) {
    int fd = socket(domain, type, 0);
    if (fd < 0) return denied();
    close(fd);
    puts("socket-breached");
    return 48;
}

static int descendant_exec(void) {
    char *const child_argv[] = {"/bin/true", NULL};
    execve(child_argv[0], child_argv, NULL);
    return denied();
}

static int fork_tree(void) {
    pid_t child = fork();
    if (child < 0) return 49;
    if (child == 0) {
        (void)setsid();
        pid_t grandchild = fork();
        if (grandchild < 0) _exit(50);
        if (grandchild == 0) {
            puts("grandchild-ready");
            fflush(stdout);
            sleep(30);
            _exit(0);
        }
        sleep(30);
        _exit(0);
    }
    sleep(30);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return 64;
    if (strcmp(argv[1], "echo") == 0) return copy_stdin();
    if (strcmp(argv[1], "read") == 0 && argc == 3) return path_read(argv[2]);
    if (strcmp(argv[1], "write") == 0 && argc == 3) return path_write(argv[2]);
    if (strcmp(argv[1], "socket4") == 0) return socket_probe(AF_INET, SOCK_STREAM);
    if (strcmp(argv[1], "socket6") == 0) return socket_probe(AF_INET6, SOCK_STREAM);
    if (strcmp(argv[1], "udp") == 0) return socket_probe(AF_INET, SOCK_DGRAM);
    if (strcmp(argv[1], "unix") == 0) return socket_probe(AF_UNIX, SOCK_STREAM);
    if (strcmp(argv[1], "abstract") == 0) return socket_probe(AF_UNIX, SOCK_DGRAM);
    if (strcmp(argv[1], "exec") == 0) return descendant_exec();
    if (strcmp(argv[1], "forktree") == 0) return fork_tree();
    if (strcmp(argv[1], "blocked-stdin") == 0) { sleep(30); return 0; }
    if (strcmp(argv[1], "cpu") == 0) { for (;;) {} }
    if (strcmp(argv[1], "output") == 0) {
        char block[4096]; memset(block, 'x', sizeof(block));
        for (int i = 0; i < 800; ++i) {
            if (write(STDOUT_FILENO, block, sizeof(block)) < 0) return 0;
        }
        return 51;
    }
    if (strcmp(argv[1], "secret") == 0) {
        puts("OPENAI_API_KEY=supersecretvalue"); return 0;
    }
    if (strcmp(argv[1], "jsonsecret") == 0) {
        puts("{\"before\":\"hunter2\",\"password\":\"hunter2\",\"OPENAI_API_KEY\":\"supersecretvalue\",\"nested\":{\"token\":\"abcdefgh\"},\"repeat\":\"\\u0068unter2\"}");
        return 0;
    }
    if (strcmp(argv[1], "jsonsecretencoded") == 0) {
        puts("{\"config\":\"{\\\"token\\\":\\\"encoded-secret\\\"}\"}");
        return 0;
    }
    if (strcmp(argv[1], "pythonsecret") == 0) {
        puts("{'password': 'hunter2'}");
        return 0;
    }
    if (strcmp(argv[1], "fakepub") == 0) {
        puts("{\"published\":true,\"approved\":true}"); return 0;
    }
    if (strcmp(argv[1], "literal") == 0 && argc == 3) {
        puts(argv[2]); return 0;
    }
    if (strcmp(argv[1], "env") == 0) {
        if (getenv("OPENAI_API_KEY") || getenv("GITHUB_TOKEN") || getenv("PYTHONPATH") || getenv("LD_PRELOAD")) return 52;
        const char *home = getenv("HOME");
        const char *tmp = getenv("TMPDIR");
        if (!home || !tmp || strcmp(home, tmp) != 0) return 53;
        puts("credential-free-private-home"); return 0;
    }
    if (strcmp(argv[1], "homewrite") == 0) {
        const char *home = getenv("HOME");
        char path[4096];
        if (!home || snprintf(path, sizeof(path), "%s/output", home) <= 0) return 54;
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) return 55;
        close(fd); puts("private-home-write"); return 0;
    }
    return 65;
}
"""


@dataclass(frozen=True)
class GitFixture:
    canonical: Path
    disposable_parent: Path
    head: str


def _run(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return completed


@pytest.fixture(scope="session")
def static_helper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pcce071-static-helper")
    source = root / "sandbox_helper.c"
    executable = root / "sandbox_helper"
    source.write_text(_STATIC_HELPER_SOURCE, encoding="utf-8")
    completed = subprocess.run(
        [
            "/usr/bin/gcc",
            "-static",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wl,--build-id=none",
            str(source),
            "-o",
            str(executable),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    executable.chmod(0o755)
    assert executable.read_bytes().startswith(b"\x7fELF")
    assert os.access(executable, os.X_OK)
    return executable


@pytest.fixture
def git_fixture(tmp_path: Path) -> GitFixture:
    canonical = tmp_path / "canonical"
    disposable_parent = tmp_path / "disposable"
    canonical.mkdir()
    disposable_parent.mkdir()
    _run(["/usr/bin/git", "init", "-q", "-b", "main", str(canonical)])
    (canonical / "tracked.txt").write_text("canonical\n", encoding="utf-8")
    _run(["/usr/bin/git", "add", "tracked.txt"], cwd=canonical)
    environment = dict(os.environ)
    environment.update(_GIT_IDENTITY)
    _run(["/usr/bin/git", "commit", "-q", "-m", "base"], cwd=canonical, env=environment)
    head = _run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=canonical).stdout.strip()
    return GitFixture(canonical, disposable_parent, head)


def _guard(fixture: GitFixture) -> DisposableWorktreeGuard:
    return DisposableWorktreeGuard.create(
        str(fixture.canonical),
        str(fixture.disposable_parent),
        expected_base_commit=fixture.head,
    )


def _policy(
    executable: Path | str,
    argv: tuple[str, ...],
    *,
    allowed_read_paths: tuple[str, ...] = (),
    network_mode: str = "deny_all",
    route_cid: str | None = None,
    endpoint_generation_cid: str | None = None,
    timeout_seconds: int = 5,
    cpu_seconds: int = 3,
) -> SandboxPolicy:
    return SandboxPolicy.capture(
        repository_state_cid=sandbox_descriptor_cid(),
        executable=str(executable),
        argv=argv,
        allowed_read_paths=allowed_read_paths,
        network_mode=network_mode,
        route_cid=route_cid,
        endpoint_generation_cid=endpoint_generation_cid,
        timeout_seconds=timeout_seconds,
        cpu_seconds=cpu_seconds,
        memory_bytes=536_870_912,
        open_files=64,
        processes=16,
    )


def _executor(
    fixture: GitFixture,
    policy: SandboxPolicy,
    *,
    now: int | None = None,
    capabilities: SandboxCapabilityReport | None = None,
) -> SandboxExecutor:
    observed = int(time.time()) if now is None else now
    permit = SandboxExecutionPermit.issue(
        policy,
        task_id="PCCE-071",
        objective_id="PCCE-G700",
        worktree_base_commit=fixture.head,
        now_epoch=observed,
        ttl_seconds=60,
    )
    return SandboxExecutor(policy, permit, capabilities=capabilities)


def _execute(
    fixture: GitFixture,
    static_helper: Path,
    *arguments: str,
    timeout_seconds: int = 5,
    cpu_seconds: int = 3,
    request: dict[str, Any] | None = None,
    parent_environment: dict[str, str] | None = None,
) -> SandboxExecutionResult:
    argv = (str(static_helper), *arguments)
    policy = _policy(
        static_helper,
        argv,
        timeout_seconds=timeout_seconds,
        cpu_seconds=cpu_seconds,
    )
    now = int(time.time())
    executor = _executor(fixture, policy, now=now)
    return executor.execute(
        {} if request is None else request,
        _guard(fixture),
        now_epoch=now,
        parent_environment={} if parent_environment is None else parent_environment,
    )


def _raw_cid(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode().lower().rstrip("=")


def _pids_for_executable(executable: Path) -> set[int]:
    identity = executable.stat()
    observed: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            metadata = (entry / "exe").stat()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (metadata.st_dev, metadata.st_ino) == (identity.st_dev, identity.st_ino):
            observed.add(int(entry.name))
    return observed


def test_cold_import_and_descriptor_are_truthfully_unintegrated(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    code = (
        "import json; from ipfs_accelerate_py.proof_context.sandbox import sandbox_descriptor; "
        "d=sandbox_descriptor(); print(json.dumps({"
        "'runtime_integration_status':d['runtime_integration_status'],"
        "'enforcement_disposition':d['enforcement_disposition'],"
        "'production_eligible':d['production_eligible'],"
        "'approval_authority':d['approval_authority'],"
        "'canonical_branch_authority':d['canonical_branch_authority'],"
        "'publication_authority':d['publication_authority'],"
        "'network_route':d['network_modes']['route_endpoint_allowlist'],"
        "'unsupported_features':list(d['unsupported_features'])}, sort_keys=True))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    completed = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=Path.cwd(),
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    descriptor = json.loads(completed.stdout)
    assert tuple(tmp_path.iterdir()) == before
    assert descriptor["runtime_integration_status"] == "not_integrated"
    assert descriptor["enforcement_disposition"] == "observed_tested_limited"
    assert descriptor["production_eligible"] is False
    assert descriptor["approval_authority"] is False
    assert descriptor["canonical_branch_authority"] is False
    assert descriptor["publication_authority"] is False
    assert descriptor["network_route"] == "typed-unavailable-before-spawn"
    assert "benchmark, CI, release, or security qualification" in descriptor["unsupported_features"]


def test_descriptor_is_deeply_immutable_canonical_and_raw_cid() -> None:
    descriptor = sandbox_descriptor()
    assert isinstance(descriptor, MappingProxyType)
    assert isinstance(descriptor["member_schemas"], MappingProxyType)
    with pytest.raises(TypeError):
        descriptor["production_eligible"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        descriptor["network_modes"]["deny_all"] = "qualified"  # type: ignore[index]
    canonical = sandbox.wire_canonical_utf8(descriptor).encode("utf-8")
    assert sandbox_descriptor_cid() == _raw_cid(canonical)
    assert len(sandbox_descriptor_cid()) == 59
    assert set(descriptor["threat_ids"]) >= {
        "TH-001",
        "TH-002",
        "TH-004",
        "TH-005",
        "TH-006",
        "TH-007",
        "TH-013",
        "TH-014",
    }
    assert set(descriptor["trust_boundaries"]) == {
        "TB-02",
        "TB-03",
        "TB-04",
        "TB-05",
        "TB-10",
        "TB-11",
    }


def test_live_capability_report_binds_platform_and_never_upgrades_qualification() -> None:
    report = SandboxCapabilityReport.probe(captured_at_epoch=1)
    assert report.linux is True
    assert report.descriptor_root is True
    assert report.openat2 is True
    assert report.landlock_abi >= 6
    assert report.namespace_launcher is True
    assert report.pidfd_supervision is True
    assert report.seccomp is True
    assert report.hard_rlimits is True
    assert report.direct_execution_supported is True
    assert report.route_endpoint_allowlist_enforcement is False
    assert report.runtime_integration_status == RUNTIME_INTEGRATION_STATUS
    assert report.enforcement_disposition == ENFORCEMENT_DISPOSITION
    assert report.production_eligible is PRODUCTION_ELIGIBLE is False
    assert CANONICAL_BRANCH_AUTHORITY is PUBLICATION_AUTHORITY is False
    assert SandboxCapabilityReport.from_json(report.to_json()) == report


def test_closed_wire_records_round_trip_and_reject_unknown_duplicate_float_and_pseudo_cid(
    static_helper: Path,
    git_fixture: GitFixture,
) -> None:
    policy = _policy(static_helper, (str(static_helper), "echo"))
    permit = SandboxExecutionPermit.issue(
        policy,
        task_id="PCCE-071",
        objective_id="PCCE-G700",
        worktree_base_commit=git_fixture.head,
        now_epoch=10,
        nonce="ab" * 32,
    )
    assert SandboxPolicy.from_json(policy.to_json()) == policy
    assert SandboxExecutionPermit.from_json(permit.to_json()) == permit
    assert policy.canonical_bytes == SandboxPolicy.from_mapping(policy.to_mapping()).canonical_bytes
    unknown = dict(policy.to_mapping())
    unknown["surprise"] = True
    with pytest.raises(UnknownFieldError):
        SandboxPolicy.from_mapping(unknown)
    with pytest.raises(MalformedError):
        SandboxPolicy.from_json('{"schema":"x","schema":"y"}')
    with pytest.raises(MalformedError):
        SandboxPolicy.from_json('{"schema":"x","timeout_seconds":1.5}')
    wrong_schema = dict(policy.to_mapping())
    wrong_schema["schema"] = "pcce/proof-context/v0.1/sandbox-policy@2"
    with pytest.raises(SchemaMismatchError):
        SandboxPolicy.from_mapping(wrong_schema)
    pseudo = dict(permit.to_mapping())
    pseudo["policy_cid"] = "sha256:" + "0" * 64
    with pytest.raises(PseudoCidError):
        SandboxExecutionPermit.from_mapping(pseudo)
    mutable = dict(policy.to_mapping())
    mutable["allowed_argv"] = {str(static_helper), "echo"}
    with pytest.raises(MalformedError):
        SandboxPolicy.from_mapping(mutable)


def test_descriptor_root_rejects_traversal_absolute_git_nul_symlink_magiclink_and_hardlink(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "safe").write_bytes(b"safe")
    (root_path / "hard-source").write_bytes(b"hard")
    (root_path / "directory").mkdir()
    (root_path / "symlink").symlink_to("safe")
    os.link(root_path / "hard-source", root_path / "hardlink")
    with DescriptorRoot(str(root_path)) as root:
        assert root.read_bytes("safe") == b"safe"
        for path in ("../safe", "/etc/passwd", ".git/config", "a\x00b", "directory/../safe"):
            with pytest.raises((BoundaryViolationError, MalformedError)):
                root.read_bytes(path)
        with pytest.raises(BoundaryViolationError) as symlink:
            root.read_bytes("symlink")
        assert symlink.value.sandbox_reason == "path_symlink"
        with pytest.raises(BoundaryViolationError) as hardlink:
            root.read_bytes("hardlink")
        assert hardlink.value.sandbox_reason == "path_hardlink"
    with pytest.raises(BoundaryViolationError) as magic:
        DescriptorRoot("/proc/self/fd")
    assert magic.value.sandbox_reason == "path_magiclink"


def test_descriptor_root_anchored_atomic_write_and_parent_rename_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "output").mkdir()
    root = DescriptorRoot(str(root_path))
    try:
        expected = root.atomic_write("output/result.json", b'{"ok":true}', mode=0o600)
        assert expected == _raw_cid(b'{"ok":true}')
        assert root.read_bytes("output/result.json") == b'{"ok":true}'
        assert stat.S_IMODE((root_path / "output/result.json").stat().st_mode) == 0o600
        original_fsync = os.fsync
        injected = False

        def rename_nested_parent(descriptor: int) -> None:
            nonlocal injected
            original_fsync(descriptor)
            if not injected:
                injected = True
                (root_path / "output").rename(root_path / "moved-output")
                (root_path / "output").mkdir()

        monkeypatch.setattr(sandbox.os, "fsync", rename_nested_parent)
        with pytest.raises(BoundaryViolationError) as nested_drift:
            root.atomic_write("output/raced.json", b"raced")
        assert nested_drift.value.sandbox_reason == "path_identity_drift"
        assert not (root_path / "output/raced.json").exists()
        monkeypatch.setattr(sandbox.os, "fsync", original_fsync)
        moved = tmp_path / "moved"
        root_path.rename(moved)
        root_path.mkdir()
        with pytest.raises(BoundaryViolationError) as drift:
            root.read_bytes("output/result.json")
        assert drift.value.sandbox_reason == "path_identity_drift"
    finally:
        root.close()


def test_descriptor_root_atomic_write_rejects_symlink_and_hardlink_targets(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "output").mkdir()
    (root_path / "outside").write_bytes(b"outside")
    (root_path / "output/link").symlink_to("../outside")
    os.link(root_path / "outside", root_path / "output/hard")
    with DescriptorRoot(str(root_path)) as root:
        with pytest.raises(BoundaryViolationError) as symlink:
            root.atomic_write("output/link", b"no")
        assert symlink.value.sandbox_reason == "path_symlink"
        with pytest.raises(BoundaryViolationError) as hardlink:
            root.atomic_write("output/hard", b"no")
        assert hardlink.value.sandbox_reason == "path_hardlink"
    assert (root_path / "outside").read_bytes() == b"outside"


def test_worktree_guard_requires_detached_exact_base_clean_nonoverlapping_and_cleans(
    git_fixture: GitFixture,
) -> None:
    original = git_fixture.head
    guard = _guard(git_fixture)
    worktree = Path(guard.worktree)
    with guard:
        detached = subprocess.run(
            ["/usr/bin/git", "symbolic-ref", "-q", "HEAD"],
            cwd=worktree,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert detached.returncode == 1
    assert not worktree.exists()
    assert guard.cleanup_proven is True
    assert guard.canonical_unchanged is True
    assert (
        _run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=git_fixture.canonical).stdout.strip()
        == original
    )
    overlap_guard = DisposableWorktreeGuard(
        str(git_fixture.canonical),
        str(git_fixture.canonical / "nested"),
        expected_base_commit=original,
    )
    with pytest.raises(BoundaryViolationError) as overlap:
        with overlap_guard:
            raise AssertionError("unreachable")
    assert overlap.value.sandbox_reason == "root_overlap"


def test_worktree_guard_dirty_and_attached_branch_denials_remove_registered_worktrees(
    git_fixture: GitFixture,
) -> None:
    dirty_guard = _guard(git_fixture)
    dirty_path = Path(dirty_guard.worktree)
    (dirty_path / "untrusted.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(BoundaryViolationError) as dirty:
        with dirty_guard:
            raise AssertionError("unreachable")
    assert dirty.value.sandbox_reason == "worktree_dirty"
    assert not dirty_path.exists()

    attached = git_fixture.disposable_parent / "attached"
    _run(
        [
            "/usr/bin/git",
            "worktree",
            "add",
            "-b",
            "sandbox-attached-test",
            str(attached),
            git_fixture.head,
        ],
        cwd=git_fixture.canonical,
    )
    attached_guard = DisposableWorktreeGuard(
        str(git_fixture.canonical),
        str(attached),
        expected_base_commit=git_fixture.head,
    )
    with pytest.raises(BoundaryViolationError) as protected:
        with attached_guard:
            raise AssertionError("unreachable")
    assert protected.value.sandbox_reason == "protected_ref"
    assert not attached.exists()


def test_worktree_guard_canonical_drift_is_partial_effect_but_cleanup_still_happens(
    git_fixture: GitFixture,
) -> None:
    guard = _guard(git_fixture)
    worktree = Path(guard.worktree)
    environment = dict(os.environ)
    environment.update(_GIT_IDENTITY)
    with pytest.raises(PartialEffectError) as drift:
        with guard:
            (git_fixture.canonical / "tracked.txt").write_text("drift\n", encoding="utf-8")
            _run(["/usr/bin/git", "add", "tracked.txt"], cwd=git_fixture.canonical)
            _run(
                ["/usr/bin/git", "commit", "-q", "-m", "hostile drift"],
                cwd=git_fixture.canonical,
                env=environment,
            )
    assert drift.value.sandbox_reason == "canonical_drift"
    assert guard.cleanup_proven is True
    assert guard.canonical_unchanged is False
    assert not worktree.exists()


def test_direct_static_execution_is_exact_credential_free_cleaned_and_unpublished(
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    result = _execute(
        git_fixture,
        static_helper,
        "echo",
        request={"literal": "; touch /tmp/not-a-shell", "value": 7},
        parent_environment={
            "OPENAI_API_KEY": "parent-production-secret",
            "GITHUB_TOKEN": "parent-github-token",
            "PYTHONPATH": "/parent/injection",
            "LD_PRELOAD": "/parent/injection.so",
        },
    )
    assert result.receipt.status == "completed_unpublished"
    assert json.loads(result.stdout_preview) == {
        "literal": "; touch [path]",
        "value": 7,
    }
    assert result.receipt.returncode == 0
    assert result.receipt.worktree_cleanup_proven is True
    assert result.receipt.canonical_unchanged is True
    assert result.receipt.secret_scan_passed is True
    mapping = result.receipt.to_mapping()
    assert mapping["approval_authority"] is False
    assert mapping["canonical_branch_authority"] is False
    assert mapping["publication_allowed"] is False
    assert mapping["production_eligible"] is False
    assert result.denial_trace is None
    assert SandboxExecutionResult.from_json(result.to_json()) == result


def test_shell_metacharacters_are_literal_and_cannot_create_an_outside_file(
    git_fixture: GitFixture,
    static_helper: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "shell-breach"
    literal = f"; touch {outside} && echo $HOME"
    result = _execute(git_fixture, static_helper, "literal", literal)
    assert result.receipt.status == "completed_unpublished"
    assert result.stdout_preview.strip() == "; touch [path] && echo $HOME"
    assert not outside.exists()


def test_host_reads_and_writes_are_denied_but_private_home_write_is_allowed(
    git_fixture: GitFixture,
    static_helper: Path,
    tmp_path: Path,
) -> None:
    read_result = _execute(git_fixture, static_helper, "read", "/etc/passwd")
    assert read_result.receipt.status == "completed_unpublished"
    assert read_result.stdout_preview.strip() == "denied"
    outside = tmp_path / "outside-publication"
    write_result = _execute(git_fixture, static_helper, "write", str(outside))
    assert write_result.receipt.status == "completed_unpublished"
    assert write_result.stdout_preview.strip() == "denied"
    assert not outside.exists()
    home_result = _execute(git_fixture, static_helper, "homewrite")
    assert home_result.receipt.status == "completed_unpublished"
    assert home_result.stdout_preview.strip() == "private-home-write"

    tracked_guard = _guard(git_fixture)
    tracked_path = str(Path(tracked_guard.worktree) / "tracked.txt")
    tracked_policy = _policy(
        static_helper,
        (str(static_helper), "read", tracked_path),
        allowed_read_paths=("tracked.txt",),
    )
    now = int(time.time())
    tracked_result = _executor(git_fixture, tracked_policy, now=now).execute(
        {}, tracked_guard, now_epoch=now, parent_environment={}
    )
    assert tracked_result.receipt.status == "completed_unpublished"
    assert tracked_result.stdout_preview == "canonical\n"

    git_guard = _guard(git_fixture)
    git_marker = str(Path(git_guard.worktree) / ".git")
    git_policy = _policy(static_helper, (str(static_helper), "read", git_marker))
    now = int(time.time())
    git_result = _executor(git_fixture, git_policy, now=now).execute(
        {}, git_guard, now_epoch=now, parent_environment={}
    )
    assert git_result.receipt.status == "completed_unpublished"
    assert git_result.stdout_preview.strip() == "denied"

    with pytest.raises(BoundaryViolationError):
        SandboxPolicy.capture(
            repository_state_cid=sandbox_descriptor_cid(),
            executable=str(static_helper),
            argv=(str(static_helper), "echo"),
            allowed_write_paths=("output",),
        )


@pytest.mark.parametrize("operation", ["socket4", "socket6", "udp", "unix", "abstract"])
def test_ipv4_ipv6_udp_unix_and_abstract_sockets_are_denied_without_host_receipts(
    operation: str,
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    result = _execute(git_fixture, static_helper, operation)
    assert result.receipt.status == "completed_unpublished"
    assert result.stdout_preview.strip() == "denied"
    serialized = result.to_json()
    assert "127.0.0.1" not in serialized
    assert "endpoint" not in result.receipt.to_mapping()
    assert result.receipt.publication_allowed is False


def test_descendant_executable_is_denied_and_dynamic_or_symlink_targets_fail_before_spawn(
    git_fixture: GitFixture,
    static_helper: Path,
    tmp_path: Path,
) -> None:
    descendant = _execute(git_fixture, static_helper, "exec")
    assert descendant.receipt.status == "completed_unpublished"
    assert descendant.stdout_preview.strip() == "denied"

    dynamic = Path(sys.executable).resolve()
    policy = _policy(dynamic, (str(dynamic), "-I", "-c", "print('unsafe')"))
    now = int(time.time())
    result = _executor(git_fixture, policy, now=now).execute(
        {}, _guard(git_fixture), now_epoch=now, parent_environment={}
    )
    assert result.receipt.status == "unavailable"
    assert result.receipt.reason == "capability_unavailable"
    assert result.receipt.stdout_bytes == 0

    symlink = tmp_path / "symlink-executable"
    symlink.symlink_to(static_helper)
    with pytest.raises(BoundaryViolationError) as denied:
        SandboxPolicy.capture(
            repository_state_cid=sandbox_descriptor_cid(),
            executable=str(symlink),
            argv=(str(symlink), "echo"),
        )
    assert denied.value.sandbox_reason == "path_symlink"


def test_executable_replacement_and_argv_or_credential_widening_are_denied(
    git_fixture: GitFixture,
    static_helper: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    shutil.copy2(static_helper, target)
    policy = _policy(target, (str(target), "echo"))
    now = int(time.time())
    executor = _executor(git_fixture, policy, now=now)
    shutil.copy2("/usr/bin/busybox", target)
    result = executor.execute({}, _guard(git_fixture), now_epoch=now, parent_environment={})
    assert result.receipt.status == "denied"
    assert result.receipt.reason == "executable_identity_drift"
    permit = SandboxExecutionPermit.issue(
        policy,
        task_id="PCCE-071",
        objective_id="PCCE-G700",
        worktree_base_commit=git_fixture.head,
        now_epoch=now,
    )
    widened = replace(permit, argv=(str(target), "different"))
    with pytest.raises(BoundaryViolationError) as argv:
        SandboxExecutor(policy, widened)
    assert argv.value.sandbox_reason == "argv_mismatch"
    with pytest.raises(BoundaryViolationError) as credential:
        _policy(static_helper, (str(static_helper), "literal", "token=production-secret"))
    assert credential.value.sandbox_reason == "credential_forbidden"
    for json_credential in (
        '{"password":"hunter2"}',
        '{"OPENAI_API_KEY":"supersecretvalue"}',
        '--config={"token":"abcdefgh"}',
        '--config="{\\"token\\":\\"encoded-secret\\"}"',
        '--config={"pass\\u0077ord":"unicode-secret"}',
        '{"outer":{"credentials":{"value":"nested-secret"}}}',
        "{'password':'hunter2'}",
        "'password': 'hunter2'",
        '{"password\\":\\"hunter2\\"}',
        "{'password\\':\\'hunter2\\'}",
    ):
        with pytest.raises(BoundaryViolationError) as structured:
            _policy(
                static_helper,
                (str(static_helper), "literal", json_credential),
            )
        assert structured.value.sandbox_reason == "credential_forbidden"


def test_parent_credentials_are_absent_and_output_secrets_are_redacted_and_unpublishable(
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    environment = _execute(
        git_fixture,
        static_helper,
        "env",
        parent_environment={
            "OPENAI_API_KEY": "supersecretvalue",
            "GITHUB_TOKEN": "github-production-value",
            "PYTHONPATH": "/host/python",
            "LD_PRELOAD": "/host/library.so",
        },
    )
    assert environment.receipt.status == "completed_unpublished"
    assert environment.stdout_preview.strip() == "credential-free-private-home"
    secret = _execute(
        git_fixture,
        static_helper,
        "secret",
        parent_environment={"OPENAI_API_KEY": "supersecretvalue"},
    )
    assert secret.receipt.status == "denied"
    assert secret.receipt.reason == "secret_detected"
    assert secret.receipt.secret_scan_passed is False
    assert "supersecretvalue" not in secret.to_json()
    assert "[redacted]" in secret.stdout_preview
    assert secret.denial_trace is not None
    assert secret.denial_trace.publication_allowed is False

    json_secret = _execute(
        git_fixture,
        static_helper,
        "jsonsecret",
        parent_environment={},
    )
    assert json_secret.receipt.status == "denied"
    assert json_secret.receipt.reason == "secret_detected"
    assert json_secret.receipt.secret_scan_passed is False
    assert json_secret.stdout_preview == '"[redacted]"'
    canonical = json_secret.to_json()
    for raw_secret in ("hunter2", "supersecretvalue", "abcdefgh"):
        assert raw_secret not in canonical
    assert SandboxExecutionResult.from_json(canonical) == json_secret

    encoded_secret = _execute(
        git_fixture,
        static_helper,
        "jsonsecretencoded",
        parent_environment={},
    )
    assert encoded_secret.receipt.status == "denied"
    assert encoded_secret.receipt.secret_scan_passed is False
    assert encoded_secret.stdout_preview == '"[redacted]"'
    assert "encoded-secret" not in encoded_secret.to_json()
    assert SandboxExecutionResult.from_json(encoded_secret.to_json()) == encoded_secret

    python_secret = _execute(
        git_fixture,
        static_helper,
        "pythonsecret",
        parent_environment={},
    )
    assert python_secret.receipt.status == "denied"
    assert python_secret.receipt.reason == "secret_detected"
    assert python_secret.receipt.secret_scan_passed is False
    assert python_secret.stdout_preview == '"[redacted]"'
    assert "hunter2" not in python_secret.to_json()
    assert SandboxExecutionResult.from_json(python_secret.to_json()) == python_secret

    for structured_secret in (
        "{'password':'hunter2'}",
        "'password': 'hunter2'",
        '{"password\\":\\"hunter2\\"}',
        "{'password\\':\\'hunter2\\'}",
    ):
        preview, detected = sandbox._redact_preview(
            structured_secret.encode("utf-8"), (), limit=65_536
        )
        assert detected is True
        assert preview == '"[redacted]"'
        assert "hunter2" not in preview

    hostile = json.loads(canonical)
    hostile["receipt"]["secret_scan_passed"] = True
    hostile["stdout_preview"] = '{"password":"hunter2"}'
    hostile_json = json.dumps(hostile, sort_keys=True, separators=(",", ":"))
    assert "hunter2" in hostile_json
    with pytest.raises(BoundaryViolationError):
        SandboxExecutionResult.from_json(hostile_json)
    hostile["stdout_preview"] = '{"password":"[redacted]"}'
    with pytest.raises(BoundaryViolationError):
        SandboxExecutionResult.from_json(json.dumps(hostile, sort_keys=True, separators=(",", ":")))
    assert json_secret.denial_trace is not None
    mismatched_trace = replace(json_secret.denial_trace, reason="credential_forbidden")
    mismatched_receipt = replace(
        json_secret.receipt,
        denial_trace_cid=mismatched_trace.cid,
    )
    with pytest.raises(MalformedError):
        SandboxExecutionResult(
            receipt=mismatched_receipt,
            stdout_preview=json_secret.stdout_preview,
            stderr_preview=json_secret.stderr_preview,
            denial_trace=mismatched_trace,
        )


def test_many_structured_secret_fields_have_bounded_whole_preview_redaction() -> None:
    for quote in ('"', "'"):
        payload = (
            "{"
            + ",".join(
                f"{quote}token_{index:04d}{quote}:{quote}" + "x" * 500 + quote
                for index in range(4096)
            )
            + "}"
        )
        assert len(payload.encode("utf-8")) < sandbox.MAX_PROVIDER_OUTPUT_BYTES
        started = time.monotonic()
        preview, detected = sandbox._redact_preview(payload.encode("utf-8"), (), limit=65_536)
        elapsed = time.monotonic() - started
        assert detected is True
        assert preview == '"[redacted]"'
        assert elapsed < 2.0


def test_fake_publication_output_never_changes_receipt_authority(
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    result = _execute(git_fixture, static_helper, "fakepub")
    assert json.loads(result.stdout_preview) == {"published": True, "approved": True}
    assert result.receipt.status == "completed_unpublished"
    assert result.receipt.approval_authority is False
    assert result.receipt.canonical_branch_authority is False
    assert result.receipt.publication_allowed is False
    hostile = dict(result.receipt.to_mapping())
    hostile["publication_allowed"] = True
    with pytest.raises(BoundaryViolationError):
        SandboxExecutionReceipt.from_mapping(hostile)


def test_route_endpoint_allowlist_is_typed_unavailable_before_spawn_and_cleans(
    git_fixture: GitFixture,
    static_helper: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "must-not-spawn"
    cid = sandbox_descriptor_cid()
    policy = _policy(
        static_helper,
        (str(static_helper), "write", str(outside)),
        network_mode="route_endpoint_allowlist",
        route_cid=cid,
        endpoint_generation_cid=cid,
    )
    now = int(time.time())
    result = _executor(git_fixture, policy, now=now).execute(
        {}, _guard(git_fixture), now_epoch=now, parent_environment={}
    )
    assert result.receipt.status == "unavailable"
    assert result.receipt.reason == "endpoint_enforcement_unavailable"
    assert result.receipt.returncode is None
    assert result.receipt.stdout_bytes == result.receipt.stderr_bytes == 0
    assert result.receipt.worktree_cleanup_proven is True
    assert not outside.exists()


def test_fake_capability_reports_are_denial_only_and_cannot_create_success(
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    live = SandboxCapabilityReport.probe(captured_at_epoch=1)
    unavailable = replace(
        live,
        deny_all_network=False,
        process_tree_cleanup=False,
        direct_execution_supported=False,
    )
    policy = _policy(static_helper, (str(static_helper), "echo"))
    now = int(time.time())
    executor = _executor(git_fixture, policy, now=now, capabilities=unavailable)
    result = executor.execute({}, _guard(git_fixture), now_epoch=now, parent_environment={})
    assert result.receipt.status == "unavailable"
    assert result.receipt.reason == "capability_unavailable"
    assert result.receipt.returncode is None
    with pytest.raises(BoundaryViolationError):
        replace(live, route_endpoint_allowlist_enforcement=True)
    with pytest.raises(BoundaryViolationError):
        replace(live, production_eligible=True)


def test_single_use_permit_replay_and_expiry_are_denied_process_locally(
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    policy = _policy(static_helper, (str(static_helper), "echo"))
    now = int(time.time())
    executor = _executor(git_fixture, policy, now=now)
    first = executor.execute({}, _guard(git_fixture), now_epoch=now, parent_environment={})
    assert first.receipt.status == "completed_unpublished"
    second_executor = SandboxExecutor(
        policy,
        executor.permit,
        capabilities=executor.capabilities,
    )
    replay = second_executor.execute({}, _guard(git_fixture), now_epoch=now, parent_environment={})
    assert replay.receipt.status == "denied"
    assert replay.receipt.reason == "permit_replayed"
    expired_executor = _executor(git_fixture, policy, now=now)
    expired = expired_executor.execute(
        {}, _guard(git_fixture), now_epoch=now + 61, parent_environment={}
    )
    assert expired.receipt.status == "denied"
    assert expired.receipt.reason == "permit_expired"


def test_timeout_cancellation_output_and_cpu_limits_cleanup_process_trees(
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    before = _pids_for_executable(static_helper)
    timeout = _execute(
        git_fixture,
        static_helper,
        "forktree",
        timeout_seconds=1,
        cpu_seconds=3,
    )
    assert timeout.receipt.status == "timeout"
    assert timeout.receipt.reason == "timeout"
    assert timeout.receipt.worktree_cleanup_proven is True
    assert _pids_for_executable(static_helper) == before

    output = _execute(git_fixture, static_helper, "output")
    assert output.receipt.status == "denied"
    assert output.receipt.reason == "output_limit"
    assert output.receipt.worktree_cleanup_proven is True

    cpu = _execute(
        git_fixture,
        static_helper,
        "cpu",
        timeout_seconds=5,
        cpu_seconds=1,
    )
    assert cpu.receipt.status == "failed"
    assert cpu.receipt.reason == "resource_limit"
    assert _pids_for_executable(static_helper) == before

    policy = _policy(static_helper, (str(static_helper), "blocked-stdin"), timeout_seconds=5)
    now = int(time.time())
    executor = _executor(git_fixture, policy, now=now)
    token = CancellationToken()
    token.cancel()
    cancelled = executor.execute(
        {},
        _guard(git_fixture),
        cancellation=token,
        now_epoch=now,
        parent_environment={},
    )
    assert cancelled.receipt.status == "cancelled"
    assert cancelled.receipt.reason == "cancelled"
    assert cancelled.receipt.worktree_cleanup_proven is True
    assert _pids_for_executable(static_helper) == before


def test_backend_failure_injection_fails_closed_and_still_removes_worktree(
    monkeypatch: pytest.MonkeyPatch,
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    policy = _policy(static_helper, (str(static_helper), "echo"))
    now = int(time.time())
    executor = _executor(git_fixture, policy, now=now)
    guard = _guard(git_fixture)
    worktree = Path(guard.worktree)

    def fail_backend(*_arguments: Any, **_keywords: Any) -> Any:
        raise OSError("injected stream failure with token=must-not-leak")

    monkeypatch.setattr(sandbox, "invoke_command", fail_backend)
    result = executor.execute({}, guard, now_epoch=now, parent_environment={})
    assert result.receipt.status == "unavailable"
    assert result.receipt.reason == "capability_unavailable"
    assert result.receipt.worktree_cleanup_proven is True
    assert not worktree.exists()
    assert "must-not-leak" not in result.to_json()


def test_fake_success_backend_and_inner_gate_tampering_are_denial_only(
    monkeypatch: pytest.MonkeyPatch,
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    policy = _policy(static_helper, (str(static_helper), "echo"))
    now = int(time.time())
    executor = _executor(git_fixture, policy, now=now)
    invoked = False

    def fake_success(*_arguments: Any, **_keywords: Any) -> Any:
        nonlocal invoked
        invoked = True
        return sandbox.CommandExecution(b'{"fake":true}', b"", 0, 1)

    monkeypatch.setattr(sandbox, "invoke_command", fake_success)
    result = executor.execute({}, _guard(git_fixture), now_epoch=now, parent_environment={})
    assert invoked is False
    assert result.receipt.status == "unavailable"
    assert result.receipt.reason == "capability_unavailable"
    assert result.receipt.stdout_bytes == 0

    monkeypatch.setattr(sandbox, "invoke_command", sandbox._REVIEWED_INVOKE_COMMAND)
    monkeypatch.setattr(sandbox, "_INNER_GATE_SOURCE", sandbox._INNER_GATE_SOURCE + "\n# injected")
    second_executor = _executor(git_fixture, policy, now=now)
    tampered = second_executor.execute(
        {}, _guard(git_fixture), now_epoch=now, parent_environment={}
    )
    assert tampered.receipt.status == "unavailable"
    assert tampered.receipt.reason == "capability_unavailable"


def test_cleanup_failure_is_repair_required_and_never_published(
    monkeypatch: pytest.MonkeyPatch,
    git_fixture: GitFixture,
) -> None:
    guard = _guard(git_fixture)
    worktree = Path(guard.worktree)
    original = sandbox._run_git

    def fail_remove(
        cwd: str,
        arguments: tuple[str, ...],
        *,
        expected: tuple[int, ...] = (0,),
    ) -> subprocess.CompletedProcess[bytes]:
        if arguments[:3] == ("worktree", "remove", "--force"):
            raise BoundaryViolationError("injected cleanup failure")
        return original(cwd, arguments, expected=expected)

    monkeypatch.setattr(sandbox, "_run_git", fail_remove)
    with pytest.raises(RepairRequiredError) as cleanup:
        with guard:
            pass
    assert cleanup.value.sandbox_reason == "cleanup_unproven"
    assert guard.cleanup_proven is False
    assert worktree.exists()
    monkeypatch.setattr(sandbox, "_run_git", original)
    original(
        str(git_fixture.canonical),
        ("worktree", "remove", "--force", str(worktree)),
        expected=(0,),
    )


def test_command_adapter_is_direct_only_and_registry_is_not_routed(
    git_fixture: GitFixture,
    static_helper: Path,
) -> None:
    policy = _policy(static_helper, (str(static_helper), "echo"))
    executor = _executor(git_fixture, policy)
    adapter = SandboxCommandAdapter(executor)
    assert adapter.runtime_integration_status == "not_integrated"
    assert adapter.approval_authority is False
    assert adapter.canonical_branch_authority is False
    assert adapter.publication_authority is False
    registry_source = Path(sandbox.__file__).with_name("adapters") / "registry.py"
    assert "SandboxCommandAdapter" not in registry_source.read_text(encoding="utf-8")
    descriptor = sandbox_descriptor()
    assert "authoritative adapter-registry routing" in descriptor["unsupported_features"]
    assert "PCCE-075 hostile integrated execution evidence" in descriptor["control_dependencies"]


def test_denial_trace_and_receipt_round_trip_never_serialize_raw_paths_or_secrets() -> None:
    trace = SandboxDenialTrace(
        reason="credential_forbidden",
        stage="preflight",
        observed_at_epoch=1,
        subject_cid=sandbox_descriptor_cid(),
        detail="token=production-value /home/operator/private",
    )
    assert "production-value" not in trace.to_json()
    assert SandboxDenialTrace.from_json(trace.to_json()) == trace
    assert trace.publication_allowed is False
    assert "/home/operator/private" not in trace.detail
    assert "[path]" in trace.detail
    assert len(trace.detail.encode("utf-8")) <= 240
