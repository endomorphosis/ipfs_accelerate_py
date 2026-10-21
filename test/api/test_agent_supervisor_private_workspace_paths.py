"""Private-path admission and real mount-command construction, without providers.

These are hermetic construction tests. They do not claim a live Docker/Grok
qualification or that a prompt instruction is a confidentiality boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tomllib

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as G


def _environment(tmp_path: Path, patterns: list[str] | None = None) -> dict[str, str]:
    home = tmp_path / "operator"
    home.mkdir(exist_ok=True)
    env = {"PATH": "", "HOME": str(home), "GROK_HOME": str(home / ".grok")}
    if patterns is not None:
        env[G.GROK_PRIVATE_PATHS_ENV] = json.dumps(patterns)
    return env


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "applications").mkdir()
    (root / "applications" / "service.py").write_text("# allowed synthetic source\n")
    (root / ".complaint_workspace" / "sessions").mkdir(parents=True)
    (root / ".complaint_workspace" / "sessions" / "synthetic.json").write_text('{"synthetic":true}')
    (root / "data").mkdir()
    (root / "data" / "synthetic.txt").write_text("synthetic private fixture")
    (root / ".env.test").write_text("SYNTHETIC_ONLY=1")
    return root


@pytest.mark.parametrize("raw", [
    "", "{", "{}", "[]", '["/etc"]', '["../data"]', '["data/../source"]',
    '["."]', '["data//secret"]', '["data/"]', '["~/data"]',
    '["C:\\\\data"]', '["**/*.key"]', '["data/*/secret"]',
    '[".env[abc]"]', '[".git"]', '[".git/config"]', '[1]', '[null]',
    '["data\\nsecret"]', json.dumps(["data"] * 65),
])
def test_malformed_or_escaping_policy_fails_closed(raw):
    with pytest.raises(ValueError):
        G.validate_grok_private_path_patterns(raw)


def test_absent_policy_keeps_legacy_resolution_and_rules(tmp_path):
    env = _environment(tmp_path)
    assert G._grok_private_workspace_paths(base_env=env, workspace=None) == ()
    path = tmp_path / "legacy"
    assert len(G._grok_filesystem_deny_rules((path,))) == 8
    assert all(not rule.startswith("List(") for rule in G._grok_filesystem_deny_rules((path,)))


def test_patterns_are_bounded_and_directory_recursion_is_not_scanned(tmp_path):
    root = _workspace(tmp_path)
    env = _environment(tmp_path, [".complaint_workspace/sessions/**", "data/**", ".env*", "missing"])
    actual = set(G._grok_private_workspace_paths(base_env=env, workspace=root))
    assert actual == {root / ".complaint_workspace/sessions", root / "data", root / ".env.test"}
    assert root / "applications/service.py" not in actual
    assert G.validate_grok_private_path_patterns(env[G.GROK_PRIVATE_PATHS_ENV]) == (
        ".complaint_workspace/sessions", ".env*", "data", "missing",
    )


def test_expansion_limit_prevents_unbounded_policy_enumeration(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    for index in range(4):
        (root / f"secret-{index}.key").write_text("synthetic key fixture")
    monkeypatch.setattr(G, "_GROK_PRIVATE_PATH_MAX_MATCHES", 3)
    with pytest.raises(ValueError, match="bound"):
        G._grok_private_workspace_paths(base_env=_environment(tmp_path, ["*.key"]), workspace=root)


@pytest.mark.parametrize("pattern", ["data", "data/*.key"])
def test_private_path_symlink_escape_is_rejected_before_glob(tmp_path, pattern):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "synthetic.key").write_text("synthetic-only")
    (root / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escape"):
        G._grok_private_workspace_paths(base_env=_environment(tmp_path, [pattern]), workspace=root)


def test_dangling_private_symlink_fails_closed(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "data").symlink_to(root / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="resolved safely"):
        G._grok_private_workspace_paths(base_env=_environment(tmp_path, ["data"]), workspace=root)


def test_direct_tools_and_kernel_policy_receive_the_same_private_paths(tmp_path):
    root = _workspace(tmp_path)
    env = _environment(tmp_path, [".complaint_workspace", "data", ".env*"])
    temporary, _child, policy_path, denied = G._isolated_grok_home(
        base_env=env, child_env={}, codex_fallback_command=(), workspace=root,
    )
    try:
        expected = {root / ".complaint_workspace", root / "data", root / ".env.test"}
        assert expected <= set(denied)
        policy = tomllib.loads(policy_path.read_text())
        kernel_denied = set(policy["profiles"][G.GROK_PRIMARY_SANDBOX_PROFILE]["deny"])
        assert {str(path) for path in expected} <= kernel_denied
        rules = G._grok_filesystem_deny_rules(denied, include_list=True)
        for path in expected:
            for operation in ("Read", "Grep", "Edit", "Write", "List"):
                assert f"{operation}({path})" in rules
                assert f"{operation}({path}/**)" in rules
        assert not any(str(root / "applications") in rule for rule in rules)
        (root / "public-alias").symlink_to(root / "data", target_is_directory=True)
        assert root / "public-alias" in G._workspace_symlinks_reach_denied_paths(
            workspace=root, denied_paths=denied,
        )
    finally:
        temporary.cleanup()


def test_docker_mask_mounts_hide_private_bytes_and_leave_source_mount(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    env = _environment(tmp_path, [".complaint_workspace", "data", ".env*"])
    git_store = tmp_path / "private-git-store"
    git_store.mkdir()
    (root / ".git").write_text(f"gitdir: {git_store}\n")
    monkeypatch.setattr(G, "_git_metadata_roots", lambda _workspace: (git_store,))
    temporary, child, _policy, denied = G._isolated_grok_home(
        base_env=env, child_env={}, codex_fallback_command=(), workspace=root,
    )
    try:
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("synthetic instructions")
        binary = tmp_path / "fixture-grok"
        binary.write_text("inert fixture; never executed")
        command = G._docker_grok_command(
            grok_command=[str(binary)], grok_bin=binary, workspace=root,
            prompt_path=prompt, grok_home=Path(temporary.name), base_env=env, child_env=child,
            denied_paths=denied, mask_root=tmp_path / "masks", docker_config=tmp_path / "docker-config",
            container_name="ipfs-accelerate-grok-123-" + "a" * 32,
            cidfile=tmp_path / "cid", docker_bin="/fixture/docker", isolation_image="sha256:" + "a" * 64,
        )
        mounts = [command[index + 1] for index, part in enumerate(command) if part == "--mount"]
        for private in (root / ".complaint_workspace", root / "data", root / ".env.test", root / ".git"):
            matching = [value for value in mounts if f"dst={private}," in value]
            assert len(matching) == 1
            fields = dict(part.split("=", 1) for part in matching[0].split(",") if "=" in part)
            masked = Path(fields["src"])
            assert masked != private and masked.is_relative_to(tmp_path / "masks")
            assert stat.S_IMODE(masked.stat().st_mode) == 0
            assert "readonly" in matching[0]
        assert any(f"dst={root}" in value.split(",") and "readonly" not in value for value in mounts)
        assert not any(f"dst={root / 'applications'}," in value for value in mounts)
        assert not any(f"src={git_store}" in value.split(",") for value in mounts)
        assert "synthetic private fixture" not in " ".join(command)
    finally:
        temporary.cleanup()


def test_hardlink_alias_of_private_file_is_rejected_even_inside_workspace(tmp_path):
    root = _workspace(tmp_path)
    private = root / "data" / "synthetic.txt"
    alias = root / "applications" / "synthetic-alias.txt"
    os.link(private, alias)
    # Existing legacy audit permits internally complete link groups. Privacy
    # additionally refuses an alias outside the masked subtree.
    assert G._workspace_regular_file_hardlinks(root) == ()
    assert set(G._workspace_regular_file_hardlinks(root, private_paths=(root / "data",))) == {private, alias}


def test_invalid_policy_is_rejected_before_provider_discovery(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    monkeypatch.setenv(G.GROK_PRIVATE_PATHS_ENV, '["../outside"]')
    import ipfs_accelerate_py.llm_router as router

    def forbidden(*_args, **_kwargs):
        pytest.fail("provider discovery or subprocess preceded private-path validation")

    monkeypatch.setattr(router, "find_grok_cli", forbidden)
    monkeypatch.setattr(G.subprocess, "Popen", forbidden)
    assert G.main(["--workspace", str(root)]) == 2


def test_private_launch_requires_real_container_admission_even_with_receipt_fd(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    monkeypatch.setenv(G.GROK_PRIVATE_PATHS_ENV, '["data"]')
    monkeypatch.setenv(G.GROK_TERMINAL_RECEIPT_FD_ENV, "invalid")
    import ipfs_accelerate_py.llm_router as router

    monkeypatch.setattr(router, "find_grok_cli", lambda **_kwargs: "/fixture/grok")
    requested = []

    def no_container(*, require_container_boundary=False):
        requested.append(require_container_boundary)
        raise ValueError("fixture container unavailable")

    monkeypatch.setattr(G, "_select_grok_isolation_backend", no_container)

    def forbidden(*_args, **_kwargs):
        pytest.fail("provider subprocess bypassed the private container boundary")

    monkeypatch.setattr(G.subprocess, "Popen", forbidden)
    monkeypatch.setattr(G.sys, "stdin", __import__("io").StringIO("synthetic instructions"))
    assert G.main(["--workspace", str(root), "--grok-bin", "/fixture/grok"]) != 0
    assert requested == [True]
