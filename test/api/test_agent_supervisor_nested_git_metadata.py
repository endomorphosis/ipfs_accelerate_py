"""Real linked-worktree metadata discovery stays bounded to declared submodules."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner


def git(repo, *args):
    return subprocess.check_output([
        "git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
        "-c", "user.name=Metadata Test", "-c", "user.email=metadata@example.invalid",
        "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


def initialize(path):
    path.mkdir()
    git(path, "init", "--quiet")
    (path / "source.txt").write_text("source\n")
    git(path, "add", "source.txt")
    git(path, "commit", "--quiet", "-m", "fixture")


@pytest.fixture
def linked_submodules(tmp_path):
    inner, library, parent = [tmp_path / name for name in ("inner", "library", "parent")]
    for repo in (inner, library, parent):
        initialize(repo)
    git(library, "-c", "protocol.file.allow=always", "submodule", "add", str(inner), "deps/inner")
    git(library, "commit", "--quiet", "-am", "declare inner")
    git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(library), "external/library")
    git(parent, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive")
    git(parent, "commit", "--quiet", "-am", "declare library")
    workspace = tmp_path / "worker"
    git(parent, "worktree", "add", "--quiet", "--detach", str(workspace), "HEAD")
    # Match the native runtime's local-source worktree reuse: initialized
    # submodules point to independent canonical repositories, outside the
    # outer parent's .git/modules tree.
    canonical_library = library
    worker_library = workspace / "external/library"
    git(canonical_library, "worktree", "add", "--quiet", "--detach", str(worker_library), "HEAD")
    worker_inner = worker_library / "deps/inner"
    git(inner, "worktree", "add", "--quiet", "--detach", str(worker_inner), "HEAD")
    return parent, workspace, worker_library, worker_inner


def metadata(repo):
    return {Path(git(repo, "rev-parse", "--path-format=absolute", flag))
            for flag in ("--git-common-dir", "--absolute-git-dir")}


def test_real_nested_linked_submodule_metadata_is_discovered(linked_submodules):
    _parent, workspace, library, inner = linked_submodules
    expected = metadata(workspace) | metadata(library) | metadata(inner)
    assert set(runner._git_metadata_roots(workspace)) == expected
    # These are precisely the external metadata roots whose absence caused
    # git status to fail inside the real fallback container.
    assert metadata(library) - metadata(workspace)
    assert metadata(inner) - metadata(workspace)


def test_uncommitted_declarations_and_unrelated_repos_are_not_mounts(linked_submodules, tmp_path):
    _parent, workspace, _library, _inner = linked_submodules
    before = runner._git_metadata_roots(workspace)
    unrelated = workspace / "unrelated"
    initialize(unrelated)
    with (workspace / ".gitmodules").open("a") as handle:
        handle.write('[submodule "unrelated"]\n\tpath = unrelated\n\turl = file:///unrelated\n')
    assert runner._git_metadata_roots(workspace) == before


def test_declared_repository_without_committed_gitlink_is_not_a_mount(linked_submodules):
    _parent, workspace, _library, _inner = linked_submodules
    before = runner._git_metadata_roots(workspace)
    unrelated = workspace / "unrelated"
    initialize(unrelated)
    with (workspace / ".gitmodules").open("a") as handle:
        handle.write('[submodule "unrelated"]\n\tpath = unrelated\n\turl = file:///unrelated\n')
    git(workspace, "add", ".gitmodules")
    git(workspace, "commit", "--quiet", "-m", "declaration without gitlink")
    assert runner._git_metadata_roots(workspace) == before


def test_standard_absorbed_submodules_are_supported(linked_submodules):
    parent, _workspace, _library, _inner = linked_submodules
    library = parent / "external/library"
    inner = library / "deps/inner"
    assert set(runner._git_metadata_roots(parent)) == (
        metadata(parent) | metadata(library) | metadata(inner))


@pytest.mark.parametrize("mutation", ["submodule_symlink", "ancestor_symlink", "marker_symlink",
    "foreign_worktree", "home_directory", "fifo_commondir", "fifo_gitfile"])
def test_unsafe_initialized_metadata_is_rejected(linked_submodules, tmp_path, mutation):
    _parent, workspace, library, _inner = linked_submodules
    marker = library / ".git"
    if mutation == "submodule_symlink":
        moved = tmp_path / "moved-library"
        library.rename(moved)
        library.symlink_to(moved, target_is_directory=True)
    elif mutation == "ancestor_symlink":
        external = library.parent
        moved = tmp_path / "moved-external"
        external.rename(moved)
        external.symlink_to(moved, target_is_directory=True)
    elif mutation == "marker_symlink":
        saved = library / "saved-marker"
        marker.rename(saved)
        marker.symlink_to(saved)
    elif mutation == "foreign_worktree":
        marker.write_text((workspace / ".git").read_text())
    elif mutation == "home_directory":
        marker.write_text(f"gitdir: {tmp_path}\n")
    elif mutation == "fifo_gitfile":
        marker.unlink()
        os.mkfifo(marker, mode=0o600)
    else:
        gitdir = Path(git(library, "rev-parse", "--absolute-git-dir"))
        common = gitdir / "commondir"
        common.unlink()
        os.mkfifo(common, mode=0o600)
    with pytest.raises(ValueError):
        runner._git_metadata_roots(workspace)


def test_uninitialized_declared_submodule_does_not_require_metadata(linked_submodules):
    _parent, workspace, library, _inner = linked_submodules
    (library / ".git").unlink()
    assert set(runner._git_metadata_roots(workspace)) == metadata(workspace)


@pytest.mark.parametrize("unsafe", ["../outside", "/tmp/foreign-repository", "."])
def test_committed_unsafe_submodule_paths_are_rejected(linked_submodules, unsafe):
    _parent, workspace, _library, _inner = linked_submodules
    with (workspace / ".gitmodules").open("a") as handle:
        handle.write(f'[submodule "unsafe"]\n\tpath = {unsafe}\n\turl = file:///unrelated\n')
    git(workspace, "add", ".gitmodules")
    git(workspace, "commit", "--quiet", "-m", "unsafe fixture declaration")
    with pytest.raises(ValueError, match="path is unsafe"):
        runner._git_metadata_roots(workspace)


@pytest.mark.skipif(os.environ.get("IPFS_ACCELERATE_TEST_DOCKER_GIT_METADATA") != "1",
                    reason="explicit disposable local Docker qualification")
def test_real_docker_git_status_requires_declared_linked_metadata(linked_submodules, tmp_path):
    _parent, workspace, library, inner = linked_submodules
    docker = runner._docker_isolation_binary()
    assert docker, "Local pinned Docker runtime must be available for qualification"
    config = tmp_path / "docker-config"
    config.mkdir(mode=0o700)
    image = runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    def run_with_roots(roots, repository, *git_args):
        command = [docker, "--host=unix:///var/run/docker.sock", "--config", str(config),
                   "run", "--rm", "--pull=never", "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128",
                   "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", str(repository),
                   "--entrypoint", "/usr/bin/env"]
        for root in roots:
            command.extend(runner._docker_mount(root, read_only=True))
        for root in (workspace, Path("/usr")):
            command.extend(runner._docker_mount(root, read_only=True))
        command.extend([image, "-i", "HOME=/nonexistent", "PATH=/usr/bin:/bin", "git",
                        "--no-optional-locks", "-c", "core.fsmonitor=false", *git_args])
        return subprocess.run(command, env=runner._docker_control_env(), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=30, check=False)
    old_roots = tuple(metadata(workspace))
    denied = run_with_roots(old_roots, workspace, "status", "--porcelain")
    assert denied.returncode != 0
    assert "not a git repository" in denied.stderr
    roots = runner._git_metadata_roots(workspace)
    for repository in (workspace, library, inner):
        status = run_with_roots(roots, repository, "status", "--porcelain")
        assert status.returncode == 0, status.stderr
        assert status.stdout == ""
        revision = run_with_roots(roots, repository, "rev-parse", "HEAD")
        assert revision.returncode == 0, revision.stderr
        assert revision.stdout.strip() == git(repository, "rev-parse", "HEAD")
