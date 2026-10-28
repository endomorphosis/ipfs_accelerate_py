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


def local_publication(monkeypatch):
    monkeypatch.setattr(fleet, "_github_origin", lambda root: git(root, "remote", "get-url", "origin"))
    # Local bare repositories stand in for GitHub's normal PR merge service.
    monkeypatch.setattr(fleet, "_merge_reviewed_pull_request",
                        lambda root, remote, candidate: git(root, "push", remote, f"{candidate}:refs/heads/main"))


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


def test_validation_failure_includes_safe_stderr(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    baseline = git(remote, "rev-parse", "main")
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = [{"argv": [
        sys.executable, "-c",
        "import sys; sys.stderr.write('ImportError: cannot import name X\\n'); raise SystemExit(1)",
    ]}]
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "held"
    assert "publication validation failed" in result["reason"]
    assert "ImportError: cannot import name X" in result["reason"]
    assert git(remote, "rev-parse", "main") == baseline


def test_source_identity_validation_publishes_diverged_integration(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    git(root, "switch", "main")
    commit(root, "main only", "main_only.txt")
    git(root, "push", "origin", "HEAD:main")
    git(root, "switch", "accepted")
    commit(root, "source only", "source_only.txt")
    config, _ = manifest(tmp_path, {"repo": root})
    config["repositories"][0]["validation"] = [{"argv": [
        sys.executable, "-c",
        "from pathlib import Path; assert Path('source_only.txt').read_text() == 'source only'; "
        "assert not Path('main_only.txt').exists()",
    ]}]
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "published", result
    assert result["repositories"][0].get("validation_authority") == "accepted_source"
    assert git(remote, "show", "main:source_only.txt") == "source only"


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


def test_nested_stale_source_ref_uses_parent_gitlink(tmp_path, monkeypatch):
    child, child_remote = repository(tmp_path, "child")
    stale = git(child, "rev-parse", "HEAD")
    child_source = commit(child, "child feature")
    parent, parent_remote = repository(tmp_path, "parent")
    git(parent, "submodule", "add", str(child_remote), "nested")
    git(parent / "nested", "fetch", str(child), "accepted")
    git(parent / "nested", "checkout", child_source)
    git(parent, "add", ".gitmodules", "nested")
    git(parent, "commit", "-m", "accepted nested feature")
    config, _ = manifest(tmp_path, {"parent": parent, "child": child})
    config["repositories"][0]["dependencies"] = [{"repository": "child", "path": "nested"}]
    child_repo = next(repo for repo in config["repositories"] if repo["id"] == "child")
    child_repo["source_ref"] = stale
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "published", result
    assert git(child, "rev-parse", "HEAD") == child_source
    assert stale != child_source


def test_leaf_repository_gitlinks_travel_with_source_head(tmp_path, monkeypatch):
    grandchild, grandchild_remote = repository(tmp_path, "grandchild")
    child, child_remote = repository(tmp_path, "child")
    git(child, "submodule", "add", str(grandchild_remote), "vendor")
    git(child, "commit", "-am", "leaf gitlink")
    child_source = git(child, "rev-parse", "HEAD")
    git(child, "push", "origin", "HEAD:accepted")
    parent, parent_remote = repository(tmp_path, "parent")
    git(parent, "submodule", "add", str(child_remote), "nested")
    git(parent / "nested", "fetch", str(child), "accepted")
    git(parent / "nested", "checkout", child_source)
    git(parent, "add", ".gitmodules", "nested")
    git(parent, "commit", "-m", "accepted nested leaf")
    config, _ = manifest(tmp_path, {"parent": parent, "child": child})
    config["repositories"][0]["dependencies"] = [{"repository": "child", "path": "nested"}]
    local_publication(monkeypatch)
    result = fleet.publish_completed_board(config, tmp_path / "state")
    assert result["status"] == "published", result


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


BILLING_MESSAGE = "The job was not started because your account is locked due to a billing issue."
CHECK_RUN_ENDPOINT = "repos/owner/repo/commits/" + "a" * 40 + "/check-runs?per_page=8&filter=latest"
ANNOTATION_ENDPOINT = "repos/owner/repo/check-runs/9/annotations?per_page=8&page=1"


def billing_check_run(candidate="a" * 40, **patch):
    return {
        "id": 9, "head_sha": candidate, "app": {"id": 15368, "slug": "github-actions"},
        "status": "completed", "conclusion": "failure", "output": {"annotations_count": 1},
        **patch,
    }


def billing_annotation(**patch):
    return {"annotation_level": "failure", "message": BILLING_MESSAGE, **patch}


def assert_no_billing_bypass(calls):
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)
    assert not any("--admin" in argv or "--auto" in argv for argv in calls)
    assert not any(str(arg).endswith("run_documentation_gates.py") for argv in calls for arg in argv)
    assert not any(argv[:2] == ["gh", "run"] for argv in calls)


def billing_api(monkeypatch, *, runs=None, annotations=None, raw_runs=None,
                raw_annotations=None, unavailable=None):
    """Only the two bounded candidate-specific REST reads are allowed."""
    calls = []
    rows = [billing_check_run()] if runs is None else runs
    payload = {"total_count": len(rows), "check_runs": rows}

    def run(argv, root, timeout=300, **kwargs):
        calls.append(list(argv))
        assert argv[:7] == ["gh", "api", "--hostname", "github.com", "--method", "GET", argv[-1]]
        assert len(argv) == 7
        assert timeout == 15
        assert not kwargs
        endpoint = argv[-1]
        if endpoint == CHECK_RUN_ENDPOINT:
            if unavailable == "runs":
                raise fleet.PublicationHold("check-runs unavailable")
            return raw_runs if raw_runs is not None else json.dumps(payload)
        if endpoint.startswith("repos/owner/repo/check-runs/") and endpoint.endswith("/annotations?per_page=8&page=1"):
            ident = endpoint.split("/")[4]
            assert ident in {str(row.get("id")) for row in rows if isinstance(row, dict)}
            if unavailable == "annotations":
                raise fleet.PublicationHold("annotations unavailable")
            if raw_annotations is not None:
                return raw_annotations
            if isinstance(annotations, dict):
                return json.dumps(annotations[int(ident)])
            return json.dumps([billing_annotation()] if annotations is None else annotations)
        raise AssertionError(f"unexpected or unbounded billing evidence read: {argv!r}")

    monkeypatch.setattr(fleet, "_run", run)
    return calls


def github_pr(monkeypatch, *, patch=None, checks_fail=False, second_patch=None, listed=True,
              billing_locked=False, merged=False, merged_head=None,
              listed_after_create=None, commit_pulls=None):
    candidate = "a" * 40
    calls = []
    views = 0
    creates = 0
    ready = {"number": 7, "state": "OPEN", "isDraft": False, "baseRefName": "main",
             "headRefOid": candidate, "mergeable": "MERGEABLE", "mergeStateStatus": "CLEAN",
             "reviewDecision": "APPROVED"}

    def run(argv, root, *args, **kwargs):
        nonlocal views, creates
        calls.append(argv)
        if argv[:3] == ["gh", "pr", "list"]:
            state = argv[argv.index("--state") + 1] if "--state" in argv else "open"
            if state == "merged":
                if merged:
                    return json.dumps([{
                        "number": 7, "headRefOid": merged_head or candidate, "state": "MERGED",
                    }])
                return "[]"
            show = listed_after_create if creates and listed_after_create is not None else listed
            return json.dumps([{"number": 7, "headRefOid": candidate}] if show else [])
        if argv[:3] == ["gh", "pr", "create"]:
            creates += 1
            return "https://github.com/owner/repo/pull/7"
        if argv[:3] == ["gh", "pr", "view"]:
            views += 1
            return json.dumps({**ready, **(patch or {}), **(second_patch or {} if views == 2 else {})})
        if argv[:3] == ["gh", "pr", "checks"] and checks_fail:
            raise fleet.PublicationHold("hosted check unavailable due to billing")
        if argv[:2] == ["gh", "api"]:
            endpoint = argv[-1]
            if "/pulls" in endpoint:
                return json.dumps(commit_pulls if commit_pulls is not None else [])
            if endpoint == f"repos/owner/repo/commits/{candidate}/check-runs?per_page=8&filter=latest":
                return json.dumps({"total_count": int(billing_locked), "check_runs": [
                    billing_check_run(candidate),
                ] if billing_locked else []})
            if endpoint == "repos/owner/repo/check-runs/9/annotations?per_page=8&page=1":
                return json.dumps([billing_annotation()])
        if argv[:3] in (["gh", "pr", "checks"], ["gh", "pr", "merge"]):
            return ""
        if argv[:2] == ["git", "push"]:
            return ""
        raise AssertionError(f"unexpected publication command: {argv!r}")

    monkeypatch.setattr(fleet, "_run", run)
    return candidate, calls


def test_github_publication_merges_exact_reviewed_head_without_direct_main_push(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch)
    fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    pushes = [argv for argv in calls if argv[:2] == ["git", "push"]]
    assert pushes == [["git", "push", "https://github.com/owner/repo.git",
                       f"{candidate}:refs/heads/fleet-publication/{candidate}"]]
    assert ["gh", "pr", "checks", "7", "--repo", "owner/repo", "--required"] in calls
    assert calls[-1] == ["gh", "pr", "merge", "7", "--repo", "owner/repo", "--merge",
                         "--match-head-commit", candidate]
    assert not any("--admin" in argv or "--auto" in argv for argv in calls)


@pytest.mark.parametrize("patch", [
    {"mergeStateStatus": "BLOCKED"}, {"mergeStateStatus": "UNKNOWN"},
    {"mergeStateStatus": "BEHIND"}, {"mergeable": "CONFLICTING"},
    {"reviewDecision": "REVIEW_REQUIRED"}, {"reviewDecision": "CHANGES_REQUESTED"},
    {"reviewDecision": None}, {"headRefOid": "b" * 40}, {"isDraft": True},
    {"baseRefName": "other"}, {"state": "CLOSED"},
])
def test_unready_github_pr_cannot_use_account_bypass_rights(tmp_path, monkeypatch, patch):
    candidate, calls = github_pr(monkeypatch, patch=patch)
    with pytest.raises(fleet.PublicationHold, match="not ready"):
        fleet._merge_reviewed_pull_request(tmp_path, "git@github.com:owner/repo.git", candidate)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_failed_or_unavailable_hosted_checks_hold_despite_local_validation(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, checks_fail=True)
    with pytest.raises(fleet.PublicationHold, match="checks are unsuccessful or unavailable"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo", candidate)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_github_actions_billing_lock_holds_without_local_required_check_substitute(tmp_path, monkeypatch):
    candidate, calls = github_pr(
        monkeypatch, patch={"mergeStateStatus": "BLOCKED"}, billing_locked=True,
    )
    with pytest.raises(fleet.PublicationHold, match="^GitHub Actions account is locked due to a billing issue; retain the PR for retry$"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert_no_billing_bypass(calls)
    assert any("/commits/" + candidate + "/check-runs?" in argv[-1] for argv in calls)


def test_billing_lock_after_failed_hosted_checks_never_runs_local_substitute(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, checks_fail=True, billing_locked=True)
    with pytest.raises(fleet.PublicationHold, match="^GitHub Actions account is locked due to a billing issue; retain the PR for retry$"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert ["gh", "pr", "checks", "7", "--repo", "owner/repo", "--required"] in calls
    assert_no_billing_bypass(calls)


def test_pr_head_movement_during_checks_refuses_merge(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, second_patch={"headRefOid": "b" * 40})
    with pytest.raises(fleet.PublicationHold, match="not ready"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo", candidate)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_new_pr_waits_for_hosted_checks(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, listed=False)
    with pytest.raises(fleet.PublicationHold, match="pull request created"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo", candidate)
    assert any(argv[:3] == ["gh", "pr", "create"] for argv in calls)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_merged_publication_pr_at_exact_candidate_is_already_published(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, listed=False, merged=True)
    fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert not any(argv[:3] == ["gh", "pr", "create"] for argv in calls)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_merged_publication_pr_associated_commit_is_already_published(tmp_path, monkeypatch):
    candidate = "a" * 40
    candidate, calls = github_pr(
        monkeypatch, listed=False,
        commit_pulls=[{
            "number": 266, "state": "closed", "merged_at": "2026-09-15T23:52:12Z",
            "merged": True, "head": {"sha": candidate},
        }],
    )
    fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert not any(argv[:3] == ["gh", "pr", "create"] for argv in calls)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)
    assert any(argv[:2] == ["gh", "api"] for argv in calls)


def test_merged_publication_pr_with_moved_head_holds(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, listed=False, merged=True, merged_head="b" * 40)
    with pytest.raises(fleet.PublicationHold, match="head differs"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert not any(argv[:3] == ["gh", "pr", "create"] for argv in calls)
    assert not any(argv[:3] == ["gh", "pr", "merge"] for argv in calls)


def test_new_pr_billing_lock_holds_without_local_required_check_substitute(tmp_path, monkeypatch):
    candidate, calls = github_pr(
        monkeypatch, listed=False, listed_after_create=True, billing_locked=True,
    )
    with pytest.raises(fleet.PublicationHold, match="^GitHub Actions account is locked due to a billing issue; retain the PR for retry$"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert any(argv[:3] == ["gh", "pr", "create"] for argv in calls)
    assert_no_billing_bypass(calls)


def test_pending_pr_retry_reuses_validated_commit_and_checkout(tmp_path, monkeypatch):
    root, remote = repository(tmp_path)
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    merge = fleet._merge_reviewed_pull_request
    attempted = []

    def pending(integration, remote, candidate):
        attempted.append((integration, candidate))
        if len(attempted) == 1:
            raise fleet.PublicationHold("required checks pending")
        merge(integration, remote, candidate)

    monkeypatch.setattr(fleet, "_merge_reviewed_pull_request", pending)
    first = fleet.publish_completed_board(config, tmp_path / "state")
    assert first["status"] == "held"
    second = fleet.publish_completed_board(config, tmp_path / "state")
    assert second["status"] == "published", second
    assert attempted[0] == attempted[1]
    assert len(list((tmp_path / "state/integrations").iterdir())) == 1


@pytest.mark.parametrize("mutation", ["dirty", "head", "record"])
def test_pending_pr_retry_refuses_changed_retained_candidate(tmp_path, monkeypatch, mutation):
    root, _ = repository(tmp_path)
    commit(root, "accepted feature")
    config, _ = manifest(tmp_path, {"repo": root})
    local_publication(monkeypatch)
    calls = []

    def pending(*args):
        calls.append(args)
        raise fleet.PublicationHold("required checks pending")

    monkeypatch.setattr(fleet, "_merge_reviewed_pull_request", pending)
    first = fleet.publish_completed_board(config, tmp_path / "state")
    integration = Path(first["repositories"][0]["integration_worktree"])
    if mutation == "dirty":
        (integration / "baseline.txt").write_text("unexpected changes")
    elif mutation == "head":
        commit(integration, "unexpected commit")
    else:
        record = next((tmp_path / "state").glob("candidate-*.json"))
        value = json.loads(record.read_text())
        value["integration"] = str(root)
        record.write_text(json.dumps(value))
    second = fleet.publish_completed_board(config, tmp_path / "state")
    assert second["status"] == "held"
    assert len(calls) == 1


def test_exact_candidate_actions_failure_annotation_is_only_billing_evidence(tmp_path, monkeypatch):
    calls = billing_api(monkeypatch)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is True
    assert [argv[-1] for argv in calls] == [CHECK_RUN_ENDPOINT, ANNOTATION_ENDPOINT]


@pytest.mark.parametrize("patch", [
    {"head_sha": "b" * 40, "pull_requests": [{"number": 99}]},
    {"head_sha": None}, {"id": True}, {"id": 0}, {"id": "9"},
    {"app": {"id": 15368, "slug": "other-app"}},
    {"app": {"id": 7, "slug": "github-actions"}},
    {"app": {"id": True, "slug": "github-actions"}},
    {"app": {"name": "GitHub Actions"}}, {"app": None},
    {"status": "queued"}, {"conclusion": "success"},
    {"output": {"annotations_count": 0, "summary": BILLING_MESSAGE}},
    {"output": {"annotations_count": True}}, {"output": {"annotations_count": -1}},
    {"output": {"annotations_count": 9}}, {"output": None},
])
def test_unrelated_or_ineligible_run_cannot_establish_billing_lock(tmp_path, monkeypatch, patch):
    calls = billing_api(monkeypatch, runs=[billing_check_run(**patch)])
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert [argv[-1] for argv in calls] == [CHECK_RUN_ENDPOINT]


@pytest.mark.parametrize("pull_requests", [[], [{"number": 99}]])
def test_same_commit_push_run_does_not_need_pr_association(tmp_path, monkeypatch, pull_requests):
    calls = billing_api(monkeypatch, runs=[billing_check_run(pull_requests=pull_requests)])
    # This is commit-scoped outage diagnosis, never a grant to merge either PR.
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is True
    assert len(calls) == 2


@pytest.mark.parametrize("raw_runs", [
    "null", "[]", "{", "{}", '{"total_count":true,"check_runs":[]}',
    '{"total_count":0,"check_runs":[{}]}',
    '{"total_count":1,"check_runs":{}}',
    json.dumps({"total_count": 1, "check_runs": [billing_check_run()], "padding": "é" * (128 * 1024)}, ensure_ascii=False),
])
def test_malformed_or_oversized_check_run_payload_cannot_diagnose_billing(tmp_path, monkeypatch, raw_runs):
    calls = billing_api(monkeypatch, raw_runs=raw_runs)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert [argv[-1] for argv in calls] == [CHECK_RUN_ENDPOINT]


@pytest.mark.parametrize("annotations", [
    [], [billing_annotation(), billing_annotation()], [None],
    [billing_annotation(annotation_level="warning")],
    [billing_annotation(annotation_level="notice")],
    [billing_annotation(annotation_level=True)], [billing_annotation(annotation_level="fatal")],
    [billing_annotation(message="billing issue")],
    [billing_annotation(message="Earlier run: " + BILLING_MESSAGE)],
    [billing_annotation(message=BILLING_MESSAGE + " This is a test fixture.")],
    [billing_annotation(message=True)], [billing_annotation(message="")],
    [billing_annotation(message="x" * 4097)],
])
def test_unverified_or_nonliteral_annotations_do_not_establish_billing_lock(tmp_path, monkeypatch, annotations):
    calls = billing_api(monkeypatch, annotations=annotations)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert len(calls) == 2


@pytest.mark.parametrize("raw_annotations", [
    "null", "{}", "{",
    json.dumps([billing_annotation(padding="é" * (32 * 1024))], ensure_ascii=False),
])
def test_malformed_or_oversized_annotation_body_does_not_establish_billing_lock(tmp_path, monkeypatch, raw_annotations):
    calls = billing_api(monkeypatch, raw_annotations=raw_annotations)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert len(calls) == 2


@pytest.mark.parametrize("unavailable,expected_calls", [("runs", 1), ("annotations", 2)])
def test_unavailable_candidate_billing_evidence_is_not_inferred(tmp_path, monkeypatch, unavailable, expected_calls):
    calls = billing_api(monkeypatch, unavailable=unavailable)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert len(calls) == expected_calls


def test_oversized_or_duplicate_run_listing_stops_before_annotations(tmp_path, monkeypatch):
    runs = [billing_check_run(id=ident) for ident in range(1, 10)]
    calls = billing_api(monkeypatch, runs=runs)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert len(calls) == 1
    calls = billing_api(monkeypatch, runs=[billing_check_run(), billing_check_run()])
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert len(calls) == 1


def test_billing_annotation_reads_are_capped_without_pagination_or_broad_fallback(tmp_path, monkeypatch):
    runs = [billing_check_run(id=ident) for ident in range(1, 9)]
    annotations = {ident: [billing_annotation(message="ordinary check failure")] for ident in range(1, 9)}
    annotations[4] = [billing_annotation()]
    calls = billing_api(monkeypatch, runs=runs, annotations=annotations)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert [argv[-1] for argv in calls] == [CHECK_RUN_ENDPOINT] + [
        f"repos/owner/repo/check-runs/{ident}/annotations?per_page=8&page=1" for ident in range(1, 4)
    ]


def test_eight_declared_annotations_remain_bounded_and_can_diagnose(tmp_path, monkeypatch):
    annotations = [billing_annotation(message="ordinary check failure") for _ in range(7)] + [billing_annotation()]
    calls = billing_api(monkeypatch, runs=[billing_check_run(output={"annotations_count": 8})], annotations=annotations)
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is True
    assert len(calls) == 2


def test_successful_hosted_checks_do_not_consult_billing_or_local_substitutes(tmp_path, monkeypatch):
    candidate, calls = github_pr(monkeypatch, billing_locked=True)
    fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert not any("check-runs" in argv[-1] for argv in calls)
    assert not any("--admin" in argv for argv in calls)
    assert calls[-1] == ["gh", "pr", "merge", "7", "--repo", "owner/repo", "--merge", "--match-head-commit", candidate]


def test_restored_hosted_checks_retry_the_same_candidate_with_normal_merge(tmp_path, monkeypatch):
    candidate, first = github_pr(monkeypatch, checks_fail=True, billing_locked=True)
    with pytest.raises(fleet.PublicationHold, match="^GitHub Actions account is locked due to a billing issue; retain the PR for retry$"):
        fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", candidate)
    assert_no_billing_bypass(first)
    same_candidate, retry = github_pr(monkeypatch, billing_locked=True)
    assert same_candidate == candidate
    fleet._merge_reviewed_pull_request(tmp_path, "https://github.com/owner/repo.git", same_candidate)
    assert ["gh", "pr", "checks", "7", "--repo", "owner/repo", "--required"] in retry
    assert retry[-1] == ["gh", "pr", "merge", "7", "--repo", "owner/repo", "--merge", "--match-head-commit", candidate]
    assert not any("check-runs" in argv[-1] or "--admin" in argv for argv in retry)


@pytest.mark.parametrize("repo,candidate", [
    ("other/owner/repo", "a" * 40), ("owner/repo?head=old", "a" * 40),
    ("https://github.com/owner/repo", "a" * 40), (None, "a" * 40),
    ("owner/repo", "a" * 39), ("owner/repo", "b" * 41),
    ("owner/repo", None), ("owner/repo", True),
])
def test_invalid_billing_scope_is_rejected_before_any_api_read(tmp_path, monkeypatch, repo, candidate):
    calls = billing_api(monkeypatch)
    assert fleet._github_actions_billing_locked(repo, candidate, tmp_path) is False
    assert calls == []


def test_empty_candidate_run_listing_never_reads_repo_wide_stale_billing_evidence(tmp_path, monkeypatch):
    calls = billing_api(monkeypatch, runs=[])
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    # The closed fake forbids run-list, logs, other commits, repos or pagination.
    assert [argv[-1] for argv in calls] == [CHECK_RUN_ENDPOINT]


def test_untrusted_run_urls_cannot_redirect_annotation_read(tmp_path, monkeypatch):
    run = billing_check_run(
        details_url="https://attacker.invalid/unrelated-billing",
        output={"annotations_count": 1, "annotations_url": "https://api.github.com/repos/other/repo/check-runs/4/annotations"},
    )
    calls = billing_api(monkeypatch, runs=[run])
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is True
    assert [argv[-1] for argv in calls] == [CHECK_RUN_ENDPOINT, ANNOTATION_ENDPOINT]


def test_malformed_later_annotation_does_not_leave_early_billing_diagnosis(tmp_path, monkeypatch):
    calls = billing_api(
        monkeypatch, runs=[billing_check_run(output={"annotations_count": 2})],
        annotations=[billing_annotation(), billing_annotation(message=True)],
    )
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert len(calls) == 2


def test_malformed_later_queried_run_does_not_leave_early_billing_diagnosis(tmp_path, monkeypatch):
    calls = billing_api(
        monkeypatch, runs=[billing_check_run(id=9), billing_check_run(id=10)],
        annotations={9: [billing_annotation()], 10: [billing_annotation(annotation_level=True)]},
    )
    assert fleet._github_actions_billing_locked("owner/repo", "a" * 40, tmp_path) is False
    assert len(calls) == 3
