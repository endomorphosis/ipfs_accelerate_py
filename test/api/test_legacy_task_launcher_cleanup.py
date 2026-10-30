"""Controlled launcher-cleanup negatives; no Docker or provider execution.

Run only within the actual admitted validation owner against the patched runner.
"""
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as G


def fixture_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(G.tempfile, "gettempdir", lambda: str(tmp_path))
    root = tmp_path / "asref-grok-container-owned"
    root.mkdir(mode=0o700)
    directory = root / "task-launchers"
    directory.mkdir(mode=0o700)
    files = [directory / "provider-command-env", directory / "task-bootstrap.py"]
    for path in files:
        path.write_text("generated inert fixture\n")
        path.chmod(0o500)
    return root, directory, files


def test_only_owned_generated_names_are_removed_without_read_permission(tmp_path, monkeypatch):
    root, directory, files = fixture_tree(tmp_path, monkeypatch)
    files[0].chmod(0)
    assert G._remove_legacy_task_container_launchers(root)
    assert not directory.exists()
    assert root.exists()
    assert G._remove_legacy_task_container_launchers(root)


@pytest.mark.parametrize("mutation", ["extra", "symlink", "hardlink", "writable", "directory_mode"])
def test_unsafe_or_unexpected_members_preserve_the_entire_directory(tmp_path, monkeypatch, mutation):
    root, directory, files = fixture_tree(tmp_path, monkeypatch)
    external = tmp_path / "external-inert-fixture"
    external.write_text("must remain\n")
    external.chmod(0o500)
    if mutation == "extra":
        (directory / "unexpected-result").write_text("preserve unknown content\n")
    elif mutation in {"symlink", "hardlink"}:
        files[0].unlink()
        if mutation == "symlink":
            files[0].symlink_to(external)
        else:
            os.link(external, files[0])
    elif mutation == "writable":
        files[0].chmod(0o502)
    else:
        directory.chmod(0o750)
    assert not G._remove_legacy_task_container_launchers(root)
    assert os.path.lexists(files[0]) and files[1].exists()
    assert external.read_text() == "must remain\n"


@pytest.mark.parametrize("where", ["lease_root", "launcher_directory"])
def test_symlinked_directory_never_traverses_into_another_tree(tmp_path, monkeypatch, where):
    root, directory, files = fixture_tree(tmp_path, monkeypatch)
    if where == "lease_root":
        link = tmp_path / "asref-grok-container-link"
        link.symlink_to(root, target_is_directory=True)
        selected = link
    else:
        external = tmp_path / "external-launchers"
        directory.rename(external)
        directory.symlink_to(external, target_is_directory=True)
        selected = root
    assert not G._remove_legacy_task_container_launchers(selected)
    assert all(path.exists() for path in files)


@pytest.mark.parametrize("container_absence_proven", [False, True])
def test_eof_watchdog_cleans_launchers_only_after_container_absence(
    tmp_path, monkeypatch, container_absence_proven,
):
    root, directory, files = fixture_tree(tmp_path, monkeypatch)
    config = root / "docker-config"
    config.mkdir(mode=0o700)
    home = tmp_path / "asref-grok-home-owned"
    home.mkdir(mode=0o700)
    prompt = tmp_path / "asref-grok-prompt-owned"
    prompt.write_text("inert fixture\n")
    cidfile = root / "container.cid"
    cidfile.write_text("a" * 64)
    removals = []

    def controlled_removal(**kwargs):
        removals.append(kwargs)
        if not container_absence_proven:
            raise ValueError("controlled container absence failure")

    monkeypatch.setattr(G, "_legacy_remove_exact_docker_container", controlled_removal)
    monkeypatch.setattr(G.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"")))
    monkeypatch.setattr(G.signal, "signal", lambda *args: None)
    result = G._legacy_docker_cleanup_watchdog_main([
        "--provider", "grok", "--docker-bin", "/usr/bin/docker",
        "--container-name", "ipfs-accelerate-grok-123-" + "a" * 32,
        "--cidfile", str(cidfile), "--lease-root", str(root),
        "--provider-home", str(home), "--prompt-path", str(prompt),
    ])
    assert len(removals) == 1 and removals[0]["settle_for_creation"] is True
    if container_absence_proven:
        assert result == 0
        assert not root.exists() and not prompt.exists() and not home.exists()
    else:
        assert result == 125
        assert all(path.exists() for path in files)
        assert prompt.exists() and home.exists() and config.exists()
