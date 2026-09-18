"""Real Git integration tests for publication holds, ordering, and retry safety."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue/fleet_completion.py"
SPEC = importlib.util.spec_from_file_location("fleet_completion_under_test", SOURCE)
fleet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fleet)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


def commit(root: Path, content: str, name: str = "feature.txt") -> str:
    (root / name).write_text(content)
    git(root, "add", name)
    git(root, "commit", "-m", content)
    return git(root, "rev-parse", "HEAD")


def repository(tmp_path: Path, name: str = "repo") -> tuple[Path, Path]:
    remote = tmp_path / f"{name}.git"
    root = tmp_path / name
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(tmp_path, "clone", str(remote), str(root))
    git(root, "config", "user.name", "Fleet tests")
    git(root, "config", "user.email", "fleet@example.invalid")
    commit(root, "baseline", "baseline.txt")
    git(root, "push", "origin", "HEAD:main")
    git(root, "switch", "-c", "accepted")
    return root, remote


def manifest(tmp_path: Path, roots: dict[str, Path]) -> tuple[dict, Path]:
    evidence_path = tmp_path / "completion-evidence.json"
    evidence_path.write_text(json.dumps({
        "authoritative": True, "complete": True, "board_id": "BOARD",
        "active_claims": 0, "pending_merges": 0, "blocking_obligations": 0,
        "source_heads": {name: git(root, "rev-parse", "HEAD") for name, root in roots.items()},
    }))
    return {
        "schema": fleet.SCHEMA, "board_id": "BOARD",
        "completion_gate": {
            "cwd": str(tmp_path),
            "argv": [sys.executable, "-c", "import pathlib; print(pathlib.Path(__import__('sys').argv[1]).read_text())", str(evidence_path)],
        },
        "repositories": [{
            "id": name, "root": str(root), "source_ref": "refs/heads/accepted",
            "validation": [{"argv": [sys.executable, "-c", "from pathlib import Path; assert Path('baseline.txt').read_text() == 'baseline'"]}],
            "dependencies": [],
        } for name, root in roots.items()],
    }, evidence_path


@pytest.fixture(autouse=True)
def isolated_git_environment(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")


class FakeGitHub:
    """Local GitHub stand-in: required checks/reviews, never a production main push."""

    def __init__(self):
        self.commands: list[tuple[str, ...]] = []
        self.slugs: dict[str, Path] = {}
        self.prs: dict[tuple[str, str], dict] = {}
        self.next_number = 1
        self.required_checks = [{"name": "ci", "state": "SUCCESS", "bucket": "pass"}]
        self.review_decision = ""
        self.mergeable = "MERGEABLE"
        self.draft = False
        self.billing = False
        self.head_override = None
        self.state = "OPEN"

    def bind(self, slug: str, origin: str | Path) -> None:
        self.slugs[slug] = Path(origin)

    def _flag(self, args: tuple[str, ...], name: str) -> str:
        return args[args.index(name) + 1] if name in args else ""

    def _pr(self, repo: str, head: str) -> dict:
        pr = self.prs.get((repo, head))
        if pr is None:
            raise fleet.PublicationHold("pull request is absent")
        head_oid = self.head_override or pr["headRefOid"]
        return {
            **pr,
            "isDraft": self.draft,
            "mergeable": self.mergeable,
            "reviewDecision": self.review_decision,
            "headRefOid": head_oid,
            "state": self.state,
        }

    def _git_dir(self, origin: Path) -> Path:
        if (origin / "HEAD").exists() and (origin / "refs").exists():
            return origin
        if (origin / ".git").exists():
            return origin / ".git"
        return origin

    def cli(self, *args: str, timeout: float = 120) -> str:
        self.commands.append(args)
        if fleet._FORBIDDEN_GITHUB_FLAGS & set(args):
            raise fleet.PublicationHold("publication must not use a GitHub admin or ruleset bypass")
        if self.billing:
            raise fleet.PublicationHold(
                "GitHub required checks cannot start: account billing, quota, or Actions outage"
            )
        repo = self._flag(args, "--repo")
        if args[:2] == ("pr", "list"):
            head = self._flag(args, "--head")
            pr = self.prs.get((repo, head))
            return json.dumps([] if pr is None else [self._pr(repo, head)])
        if args[:2] == ("pr", "create"):
            head = self._flag(args, "--head")
            number = self.next_number
            self.next_number += 1
            url = f"https://github.com/{repo}/pull/{number}"
            # The isolated branch was already pushed; read its tip from the bound origin.
            origin = self.slugs[repo]
            head_oid = subprocess.check_output(
                ["git", "--git-dir", str(self._git_dir(origin)), "rev-parse", f"refs/heads/{head}"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            self.prs[(repo, head)] = {
                "number": number, "url": url, "headRefOid": head_oid, "head": head,
            }
            return url
        if args[:2] == ("pr", "view"):
            return json.dumps(self._pr(repo, args[2]))
        if args[:2] == ("pr", "checks"):
            return json.dumps(self.required_checks)
        if args[:2] == ("pr", "merge"):
            if "--merge" not in args or "--match-head-commit" not in args:
                raise fleet.PublicationHold("merge requires --merge --match-head-commit")
            sha = self._flag(args, "--match-head-commit")
            number = int(args[2])
            match = next(pr for pr in self.prs.values() if pr["number"] == number)
            observed = self._pr(repo, match["head"])
            if observed["headRefOid"] != sha:
                raise fleet.PublicationHold("pull request head does not match the exact candidate commit")
            subprocess.check_call(
                ["git", "--git-dir", str(self._git_dir(self.slugs[repo])),
                 "update-ref", "refs/heads/main", sha],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return ""
        raise fleet.PublicationHold(f"unsupported GitHub CLI argv {args!r}")


def local_publication(monkeypatch, github: FakeGitHub | None = None) -> FakeGitHub:
    github = github or FakeGitHub()
    monkeypatch.setattr(fleet, "_github_origin", lambda root: git(root, "remote", "get-url", "origin"))

    def slug(origin: str) -> str:
        ident = f"endomorphosis/{Path(origin).name.replace('.git', '') or 'repo'}"
        github.bind(ident, origin)
        return ident

    monkeypatch.setattr(fleet, "_github_repository_slug", slug)
    monkeypatch.setattr(fleet, "_github_cli", github.cli)
    return github


def test_publishes_accepted_source_and_retries_without_duplicate_commit(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    source = commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    first = fleet.publish_completed_board(config, tmp_path / "state")
    assert first["status"] == "published", first
    main = git(remote, "rev-parse", "main")
    assert git(remote, "show", "main:feature.txt") == "accepted feature"
    assert git(root, "rev-parse", "HEAD") == source
    assert git(root, "branch", "--show-current") == "accepted"
    second = fleet.publish_completed_board(config, tmp_path / "state")
    assert second["status"] == "published", second
    assert second["repositories"][0]["status"] == "already_published"
    assert git(remote, "rev-parse", "main") == main


@pytest.mark.parametrize("patch,expected", [
    ({"complete": False}, "authoritative successful completion"),
    ({"authoritative": False}, "authoritative successful completion"),
    ({"active_claims": 1}, "active_claims=0"),
    ({"pending_merges": False}, "pending_merges=0"),
    ({"blocking_obligations": 2}, "blocking_obligations=0"),
    ({"source_heads": {"repo": "0" * 40}}, "exact accepted source heads"),
    ({"board_id": "WRONG"}, "identity mismatch"),
])
def test_incomplete_or_stale_gate_never_publishes(tmp_path, monkeypatch, patch, expected):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, evidence = manifest(tmp_path, {"repo": root})
    evidence.write_text(json.dumps({**json.loads(evidence.read_text()), **patch}))
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert expected in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline
    assert not (tmp_path / "state/integrations").exists()


def test_dirty_source_is_preserved_and_not_published(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    config, _ = manifest(tmp_path, {"repo": root})
    (root / "baseline.txt").write_text("user work")
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "dirty" in result["reason"]
    assert (root / "baseline.txt").read_text() == "user work"
    assert git(remote, "show", "main:baseline.txt") == "baseline"


def test_conflict_holds_and_keeps_isolated_checkout(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    git(root, "switch", "main")
    commit(root, "main change", "baseline.txt")
    git(root, "push", "origin", "main")
    baseline = git(remote, "rev-parse", "main")
    git(root, "switch", "accepted")
    source = commit(root, "accepted conflicting change", "baseline.txt")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held", result
    assert "merge conflict" in result["reason"]
    integration = Path(result["repositories"][0]["integration_worktree"])
    assert git(integration, "diff", "--name-only", "--diff-filter=U") == "baseline.txt"
    assert git(root, "rev-parse", "HEAD") == source
    assert git(remote, "rev-parse", "main") == baseline


def test_failed_validation_never_pushes(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = [{"argv": [sys.executable, "-c", "raise SystemExit(7)"]}]
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "exit code 7" in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline


def test_nested_repository_published_before_parent_gitlink(tmp_path, monkeypatch):
    child, child_remote = repository(tmp_path, "child")
    child_source = commit(child, "child feature")
    parent, parent_remote = repository(tmp_path, "parent")
    git(parent, "submodule", "add", str(child_remote), "nested")
    git(parent / "nested", "fetch", str(child), "accepted")
    git(parent / "nested", "checkout", child_source)
    git(parent, "add", ".gitmodules", "nested")
    git(parent, "commit", "-m", "accepted nested feature")
    config, _ = manifest(tmp_path, {"parent": parent, "child": child})
    config["repositories"][0]["dependencies"] = [{"repository": "child", "path": "nested"}]
    config["repositories"][0]["validation"].append({"argv": [sys.executable, "-c", "from pathlib import Path; assert Path('nested/feature.txt').read_text() == 'child feature'"]})
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "published", result
    assert [r["id"] for r in result["repositories"]] == ["child", "parent"]
    published_child = git(child_remote, "rev-parse", "main")
    assert published_child != child_source
    assert fleet._gitlinks(parent_remote, "main") == {"nested": published_child}
    assert git(parent / "nested", "rev-parse", "HEAD") == child_source


def test_undeclared_changed_gitlink_holds(tmp_path, monkeypatch):
    child, child_remote = repository(tmp_path, "child")
    parent, parent_remote = repository(tmp_path, "parent")
    baseline = git(parent_remote, "rev-parse", "main")
    git(parent, "submodule", "add", str(child_remote), "nested")
    git(parent, "commit", "-am", "new undeclared dependency")
    config, _ = manifest(tmp_path, {"parent": parent})
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "changed gitlinks lack" in result["reason"]
    assert git(parent_remote, "rev-parse", "main") == baseline


def test_origin_resolution_follows_local_clone_without_mutation(tmp_path):
    root, remote = repository(tmp_path)
    git(remote, "remote", "add", "origin", "https://github.com/endomorphosis/ipfs_accelerate_py.git")
    assert fleet._github_origin(root) == "https://github.com/endomorphosis/ipfs_accelerate_py.git"
    assert git(root, "remote", "get-url", "origin") == str(remote)
    git(remote, "remote", "set-url", "origin", str(root))
    with pytest.raises(fleet.PublicationHold, match="cycle"):
        fleet._github_origin(root)


def test_manifest_requires_validation_and_rejects_dependency_cycles(tmp_path):
    root, _ = repository(tmp_path)
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = []
    with pytest.raises(fleet.PublicationHold, match="validation is required"):
        fleet._ordered_repositories(config)
    config["repositories"][0]["validation"] = [{"argv": ["true"]}]
    config["repositories"][0]["dependencies"] = [{"repository": "repo", "path": "self"}]
    with pytest.raises(fleet.PublicationHold, match="cycle"):
        fleet._ordered_repositories(config)


def test_remote_advancement_during_validation_is_preserved(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    source = commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    gate = fleet._completion_gate
    calls = 0
    concurrent = None

    def gate_with_concurrent_push(manifest, heads):
        nonlocal calls, concurrent
        calls += 1
        if calls == 2:
            tree = git(root, "rev-parse", baseline + "^{tree}")
            concurrent = git(root, "commit-tree", tree, "-p", baseline, "-m", "concurrent origin main update")
            git(root, "push", "origin", f"{concurrent}:refs/heads/main")
        return gate(manifest, heads)

    monkeypatch.setattr(fleet, "_completion_gate", gate_with_concurrent_push)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held", result
    assert "advanced during validation" in result["reason"]
    assert git(remote, "rev-parse", "main") == concurrent
    assert git(root, "rev-parse", "HEAD") == source


def test_validation_cannot_replace_commit_then_publish(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = [
        {"argv": ["git", "commit", "--allow-empty", "-m", "unexpected validation commit"]},
    ]
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held", result
    assert "validation changed the integration commit" in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline


def test_timeout_kills_descendants_even_when_they_redirect_output(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    child_script = (
        "import os,signal,time,pathlib; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(__import__('sys').argv[1]).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent_script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(60)"
    )
    with pytest.raises(fleet.PublicationHold, match="timed out"):
        fleet._run([sys.executable, "-c", parent_script, child_script, str(child_pid_path)], tmp_path, timeout=.5)
    child_pid = child_pid_path.read_text()
    stat = Path("/proc") / child_pid / "stat"
    for _ in range(50):
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(.02)
    else:
        pytest.fail("timed-out command left a live descendant")


def test_successful_publish_pushes_isolated_branch_not_main(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    github = local_publication(monkeypatch)
    git_calls = []
    real_git = fleet._git

    def spy(path, *args, timeout=300):
        git_calls.append(args)
        return real_git(path, *args, timeout=timeout)

    monkeypatch.setattr(fleet, "_git", spy)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "published", result
    pushes = [args for args in git_calls if args and args[0] == "push"]
    assert pushes
    assert any("refs/heads/fleet-publication/" in arg for args in pushes for arg in args)
    assert all(
        "refs/heads/main" not in arg and arg not in {"main", "HEAD:main"}
        for args in pushes for arg in args
    )
    assert any("--match-head-commit" in command for command in github.commands)
    assert any(command[:2] == ("pr", "merge") for command in github.commands)
    assert all("--admin" not in command for command in github.commands)
    row = result["repositories"][0]
    assert row["pull_request"].startswith("https://github.com/")
    assert row["publication_branch"].startswith("fleet-publication/BOARD/repo/")
    assert git(remote, "show", "main:feature.txt") == "accepted feature"


def test_github_cli_rejects_admin_bypass_before_starting_gh(monkeypatch):
    monkeypatch.setattr(fleet, "_communicate", lambda *args, **kwargs: pytest.fail("gh must not start"))
    with pytest.raises(fleet.PublicationHold, match="admin or ruleset bypass"):
        fleet._github_cli("pr", "merge", "1", "--admin", "--merge")


def test_billing_lock_holds_without_merging(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    github = local_publication(monkeypatch)
    github.billing = True
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "billing" in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline
    assert not any(command[:2] == ("pr", "merge") for command in github.commands)


@pytest.mark.parametrize("field,value,expected", [
    ("draft", True, "draft"),
    ("mergeable", "CONFLICTING", "MERGEABLE"),
    ("review_decision", "REVIEW_REQUIRED", "required review"),
    ("review_decision", "CHANGES_REQUESTED", "required review"),
    ("required_checks", [], "have not started"),
    ("required_checks", [{"name": "ci", "state": "PENDING", "bucket": "pending"}], "has not succeeded"),
    ("required_checks", [{"name": "ci", "state": "FAILURE", "bucket": "fail"}], "has not succeeded"),
])
def test_unqualified_pull_request_never_merges(tmp_path, monkeypatch, field, value, expected):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    github = local_publication(monkeypatch)
    setattr(github, field if field != "draft" else "draft", value)
    if field == "draft":
        github.draft = True
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held", result
    assert expected in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline
    assert not any(command[:2] == ("pr", "merge") for command in github.commands)


def test_head_changed_before_merge_is_held(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    github = local_publication(monkeypatch)
    github.head_override = "0" * 40
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "head does not match" in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline
    assert not any(command[:2] == ("pr", "merge") for command in github.commands)


def test_direct_main_push_guard_rejects_refspecs():
    with pytest.raises(fleet.PublicationHold, match="must not push directly"):
        fleet._reject_direct_main_push(("push", "origin", "abc:refs/heads/main"))
    with pytest.raises(fleet.PublicationHold, match="must not push directly"):
        fleet._reject_direct_main_push(("push", "origin", "main"))
    fleet._reject_direct_main_push(("push", "origin", "abc:refs/heads/fleet-publication/x"))
    fleet._reject_direct_main_push(("fetch", "origin", "refs/heads/main:refs/remotes/x/main"))


@pytest.mark.parametrize("origin,slug", [
    ("git@github.com:endomorphosis/ipfs_accelerate_py", "endomorphosis/ipfs_accelerate_py"),
    ("https://github.com/endomorphosis/lift_coding", "endomorphosis/lift_coding"),
    ("https://github.com/endomorphosis/ipfs_accelerate_py.git", "endomorphosis/ipfs_accelerate_py"),
])
def test_github_repository_slug_parses_supported_origins(origin, slug):
    assert fleet._github_repository_slug(origin) == slug


def test_github_cli_classifies_billing_stderr(monkeypatch):
    monkeypatch.setattr(
        fleet, "_communicate",
        lambda argv, cwd, timeout: (1, "", "The job was not started because your account is locked due to a billing issue."),
    )
    with pytest.raises(fleet.PublicationHold, match="billing, quota, or Actions outage"):
        fleet._github_cli("pr", "checks", "1", "--repo", "owner/repo", "--required")
