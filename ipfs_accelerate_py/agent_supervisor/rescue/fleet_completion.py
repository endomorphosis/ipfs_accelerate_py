"""Publish explicitly accepted board integration refs without disturbing workers.

The operator-owned manifest supplies a live completion gate, accepted integration
refs, a dependency graph of gitlinks, and validation commands. The gate must
return JSON with ``authoritative`` and ``complete`` true, the exact ``board_id``,
zero ``active_claims``, ``pending_merges``, and ``blocking_obligations``, and
``source_heads`` mapping every repository id to its accepted commit. It must
verify the board's receipts/seals and current-tree completion contracts; a
Markdown projection or a task count is not a completion gate.

Repositories contain ``id``, ``root``, ``source_ref``, nonempty ``validation``
(objects with argv and optional repository-relative cwd), and ``dependencies``
(objects with repository id and gitlink path). Only those refs are considered.
The publisher holds on conflicts or unverifiable completion and never force
pushes, resets a live checkout, or marks a task complete. GitHub publication
pushes an isolated branch, opens or updates a pull request, and merges only
after GitHub reports that exact head mergeable, required checks successful,
and required reviews satisfied. A successful branch push is not publication.
Admin, ruleset, and branch-protection bypasses are forbidden.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Mapping
from urllib.parse import urlparse

SCHEMA = "agent-supervisor/fleet-publication@1"
GATE_SCHEMA = "agent-supervisor/fleet-completion-gate@1"
_OID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_GITHUB_SLUG = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_BILLING_MARKERS = ("billing issue", "account is locked", "quota", "actions is disabled")
_FORBIDDEN_GITHUB_FLAGS = frozenset({
    "--admin", "--bypass-rules", "--bypass-policy", "--bypass-ruleset",
    "--disable-rules",
})


class PublicationHold(RuntimeError):
    """Publication requires additional accepted evidence or a conflict repair."""


def _communicate(argv: list[str], cwd: Path, timeout: float = 300) -> tuple[int, str, str]:
    if not argv or not all(isinstance(arg, str) for arg in argv):
        raise PublicationHold("command argv must be a nonempty string list")
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GH_PROMPT_DISABLED": "1"},
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Keep the leader unreaped until the whole group has been killed:
            # a child can ignore TERM and redirect both output streams, letting
            # communicate() return while that child continues mutating Git.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(1)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise PublicationHold(f"command timed out after {timeout:g}s") from None
    except OSError as exc:
        raise PublicationHold(f"command could not start: {type(exc).__name__}") from None
    return process.returncode, (stdout or "").strip(), (stderr or "").strip()


def _run(argv: list[str], cwd: Path, timeout: float = 300) -> str:
    code, stdout, _stderr = _communicate(argv, cwd, timeout)
    if code:
        # Do not copy output or argv containing credentials into fleet receipts.
        raise PublicationHold(f"command failed with exit code {code}")
    return stdout


def _reject_direct_main_push(args: tuple[str, ...]) -> None:
    """A normal push to main can bypass required checks when the account may."""
    if not args or args[0] != "push":
        return
    for arg in args[1:]:
        if arg.startswith("-"):
            continue
        dest = arg.split(":", 1)[1] if ":" in arg else arg
        if dest in {"main", "master", "refs/heads/main", "refs/heads/master"}:
            raise PublicationHold("publication must not push directly to GitHub main")


def _git(root: Path, *args: str, timeout: float = 300) -> str:
    _reject_direct_main_push(args)
    return _run(["git", *args], root, timeout)


def _github_repository_slug(origin: str) -> str:
    """Map a GitHub origin URL to owner/name without mutating remotes."""
    ssh = re.fullmatch(
        r"git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?", origin,
    )
    if ssh:
        return f"{ssh.group(1)}/{ssh.group(2)}"
    parsed = urlparse(origin)
    if parsed.scheme in {"https", "ssh"} and parsed.hostname == "github.com":
        parts = parsed.path.strip("/").removesuffix(".git").split("/")
        if len(parts) == 2 and all(parts):
            return f"{parts[0]}/{parts[1]}"
    raise PublicationHold("GitHub origin is not a parseable owner/repository")


def _publication_branch(board_id: str, ident: str, candidate: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", board_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
        raise PublicationHold("publication branch requires safe board and repository ids")
    if not _OID.fullmatch(candidate):
        raise PublicationHold("publication branch requires an exact candidate commit")
    return f"fleet-publication/{board_id}/{ident}/{candidate[:12]}"


def _push_isolated_branch(root: Path, remote: str, candidate: str, branch: str) -> None:
    if branch in {"main", "master"} or branch.startswith("refs/heads/main"):
        raise PublicationHold("publication must not push to GitHub main")
    dest = f"refs/heads/{branch}"
    if dest in {"refs/heads/main", "refs/heads/master"}:
        raise PublicationHold("publication must not push to GitHub main")
    _git(root, "push", remote, f"{candidate}:{dest}")


def _github_cli(*args: str, timeout: float = 120) -> str:
    """Run gh. Bypass flags are rejected before the process starts."""
    if not args or not all(isinstance(arg, str) and arg for arg in args):
        raise PublicationHold("GitHub CLI argv must be a nonempty string list")
    if _FORBIDDEN_GITHUB_FLAGS & set(args):
        raise PublicationHold("publication must not use a GitHub admin or ruleset bypass")
    code, stdout, stderr = _communicate(["gh", *args], Path.cwd(), timeout)
    combined = f"{stdout}\n{stderr}".lower()
    if any(marker in combined for marker in _BILLING_MARKERS):
        raise PublicationHold(
            "GitHub required checks cannot start: account billing, quota, or Actions outage"
        )
    if code:
        raise PublicationHold(f"GitHub CLI failed with exit code {code}")
    return stdout


def _load_json_payload(raw: str, *, noun: str) -> Any:
    try:
        return json.loads(raw or "null")
    except ValueError as exc:
        raise PublicationHold(f"{noun} is not JSON") from exc


def _observe_pull_request(repo: str, head: str, candidate: str) -> dict[str, Any]:
    if not _GITHUB_SLUG.fullmatch(repo):
        raise PublicationHold("GitHub repository slug is malformed")
    payload = _load_json_payload(
        _github_cli(
            "pr", "view", head, "--repo", repo,
            "--json", "number,url,isDraft,mergeable,reviewDecision,headRefOid,state",
        ),
        noun="pull request observation",
    )
    if not isinstance(payload, dict):
        raise PublicationHold("pull request observation is not an object")
    number, url, head_oid = payload.get("number"), payload.get("url"), payload.get("headRefOid")
    if type(number) is not int or number < 1 or not isinstance(url, str) or not url:
        raise PublicationHold("pull request observation is missing identity")
    if payload.get("state") != "OPEN":
        raise PublicationHold("pull request is not open")
    if payload.get("isDraft") is True:
        raise PublicationHold("pull request is draft; required review/check success is not established")
    if payload.get("mergeable") != "MERGEABLE":
        raise PublicationHold("pull request is not MERGEABLE at the exact publication head")
    review = payload.get("reviewDecision") or ""
    if review in {"REVIEW_REQUIRED", "CHANGES_REQUESTED"}:
        raise PublicationHold("required review is not satisfied")
    if review not in {"", "APPROVED"}:
        raise PublicationHold("pull request review decision is unknown")
    if head_oid != candidate:
        raise PublicationHold("pull request head does not match the exact candidate commit")
    return {"number": number, "url": url, "head": head_oid}


def _required_checks_passed(repo: str, pr_number: int) -> None:
    raw = _github_cli(
        "pr", "checks", str(pr_number), "--repo", repo, "--required",
        "--json", "name,state,bucket",
    )
    checks = _load_json_payload(raw, noun="required check observation")
    if not isinstance(checks, list) or not checks:
        raise PublicationHold("required hosted checks have not started or are not reported")
    for check in checks:
        if not isinstance(check, dict):
            raise PublicationHold("required check observation is malformed")
        state = str(check.get("state") or "")
        bucket = str(check.get("bucket") or "")
        if state != "SUCCESS" or bucket != "pass":
            raise PublicationHold(
                f"required hosted check {check.get('name')!r} has not succeeded"
            )


def _ensure_pull_request(repo: str, head: str, *, title: str, body: str) -> None:
    listed = _load_json_payload(
        _github_cli(
            "pr", "list", "--repo", repo, "--head", head, "--base", "main",
            "--state", "open", "--json", "number,url,headRefOid,isDraft",
        ),
        noun="pull request list",
    )
    if isinstance(listed, list) and listed:
        return
    _github_cli(
        "pr", "create", "--repo", repo, "--base", "main", "--head", head,
        "--title", title, "--body", body,
    )


def _publish_through_required_pull_request(
    integration: Path, *, remote: str, ident: str, candidate: str, main: str,
    board_id: str,
) -> dict[str, str]:
    """Push an isolated branch and merge only through a qualified GitHub PR."""
    if candidate == main:
        raise PublicationHold(f"{ident}: candidate commit is already GitHub main")
    branch = _publication_branch(board_id, ident, candidate)
    _push_isolated_branch(integration, remote, candidate, branch)
    repo = _github_repository_slug(remote)
    title = f"Complete {board_id}: integrate accepted {ident} work"
    body = (
        f"Fleet publication of accepted `{ident}` commit `{candidate}` onto GitHub main.\n"
        "Merge only after required checks and reviews succeed for this exact head."
    )
    _ensure_pull_request(repo, branch, title=title, body=body)
    observation = _observe_pull_request(repo, branch, candidate)
    _required_checks_passed(repo, observation["number"])
    # Recheck the exact head immediately before merge; stale observations cannot merge.
    observation = _observe_pull_request(repo, branch, candidate)
    _github_cli(
        "pr", "merge", str(observation["number"]), "--repo", repo, "--merge",
        "--match-head-commit", candidate,
    )
    observed = _fetch_main(integration, remote, ident)
    if not _ancestor(integration, candidate, observed):
        raise PublicationHold(f"{ident}: merged pull request is not reachable from GitHub main")
    return {
        "published_head": observed,
        "publication_branch": branch,
        "pull_request": observation["url"],
    }


def _github_origin(root: Path) -> str:
    """Resolve local clone origins without changing anyone's remote config."""
    visited: set[Path] = set()
    for _ in range(12):
        root = root.resolve()
        if root in visited:
            raise PublicationHold("local origin chain contains a cycle")
        visited.add(root)
        origin = _git(root, "remote", "get-url", "origin")
        if re.fullmatch(r"git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", origin):
            return origin
        parsed = urlparse(origin)
        if parsed.scheme in {"https", "ssh"} and parsed.hostname == "github.com":
            if parsed.password or parsed.query or parsed.fragment or parsed.port:
                raise PublicationHold("GitHub origin contains unsupported credentials or options")
            if parsed.username and not (parsed.scheme == "ssh" and parsed.username == "git"):
                raise PublicationHold("GitHub origin must not contain credentials")
            if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", parsed.path):
                raise PublicationHold("GitHub origin has an invalid repository path")
            return origin
        if parsed.scheme == "file" and not parsed.netloc:
            root = Path(parsed.path)
        elif not parsed.scheme and ":" not in origin:
            root = (root / origin).resolve()
        else:
            raise PublicationHold("origin chain does not resolve to a GitHub repository")
        if not root.is_dir():
            raise PublicationHold("local origin repository does not exist")
    raise PublicationHold("local origin chain exceeds twelve repositories")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _command(spec: Mapping[str, Any], root: Path) -> str:
    cwd = (root / spec.get("cwd", ".")).resolve()
    if not cwd.is_relative_to(root.resolve()):
        raise PublicationHold("validation cwd must remain inside the isolated repository")
    return _run(spec.get("argv", []), cwd, float(spec.get("timeout_seconds", 600)))


def _ordered_repositories(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise PublicationHold("publication requires an explicit nonempty repository list")
    by_id: dict[str, dict[str, Any]] = {}
    for repo in repositories:
        if not isinstance(repo, dict):
            raise PublicationHold("every repository entry must be an object")
        ident = repo.get("id", "")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in by_id:
            raise PublicationHold("repository ids must be unique safe names")
        if not repo.get("root") or not repo.get("source_ref"):
            raise PublicationHold(f"{ident}: root and accepted source_ref are required")
        if not str(repo["source_ref"]).startswith("refs/heads/") and not _OID.fullmatch(str(repo["source_ref"])):
            raise PublicationHold(f"{ident}: source_ref must be a full branch ref or commit id")
        validations = repo.get("validation")
        if not isinstance(validations, list) or not validations:
            raise PublicationHold(f"{ident}: publication validation is required")
        if any(not isinstance(v, dict) or not v.get("argv") for v in validations):
            raise PublicationHold(f"{ident}: validation commands require argv")
        for field in ("dependencies", "initialize_submodules"):
            if not isinstance(repo.get(field, []), list):
                raise PublicationHold(f"{ident}: {field} must be a list")
        for submodule_path in repo.get("initialize_submodules", []):
            path = Path(submodule_path)
            if path.is_absolute() or ".." in path.parts or str(path) == ".":
                raise PublicationHold("initialized submodule paths must be repository relative")
        by_id[ident] = dict(repo)
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(ident: str) -> None:
        if ident in visited:
            return
        if ident in visiting or ident not in by_id:
            raise PublicationHold("repository dependencies contain a cycle or unknown id")
        visiting.add(ident)
        paths: set[str] = set()
        for dep in by_id[ident].get("dependencies", []):
            if not isinstance(dep, dict):
                raise PublicationHold("every repository dependency must be an object")
            path = Path(dep.get("path", ""))
            if not dep.get("path") or path.is_absolute() or ".." in path.parts or str(path) == ".":
                raise PublicationHold("dependency gitlink paths must be repository relative")
            if str(path) in paths:
                raise PublicationHold("dependency gitlink paths must be unique")
            paths.add(str(path))
            visit(dep.get("repository", ""))
        visiting.remove(ident)
        visited.add(ident)
        ordered.append(by_id[ident])

    for ident in by_id:
        visit(ident)
    return ordered


def _source_heads(repositories: list[dict[str, Any]]) -> dict[str, str]:
    heads = {}
    for repo in repositories:
        root = Path(repo["root"]).resolve()
        if _git(root, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none"):
            raise PublicationHold(f"{repo['id']}: accepted source checkout is dirty")
        head = _git(root, "rev-parse", "--verify", repo["source_ref"] + "^{commit}")
        if not _OID.fullmatch(head) or _git(root, "rev-parse", "HEAD") != head:
            raise PublicationHold(f"{repo['id']}: source_ref must match its clean integration checkout HEAD")
        heads[repo["id"]] = head
    return heads


def _completion_gate(manifest: Mapping[str, Any], heads: Mapping[str, str]) -> dict[str, Any]:
    gate = manifest.get("completion_gate") or {}
    if not gate.get("cwd") or not gate.get("argv"):
        raise PublicationHold("live authoritative completion gate command is required")
    try:
        evidence = json.loads(_run(gate["argv"], Path(gate["cwd"]), float(gate.get("timeout_seconds", 300))))
    except (ValueError, TypeError):
        raise PublicationHold("completion gate did not return a JSON object") from None
    if not isinstance(evidence, dict):
        raise PublicationHold("completion gate did not return a JSON object")
    if evidence.get("authoritative") is not True or evidence.get("complete") is not True:
        raise PublicationHold("board lacks authoritative successful completion")
    if evidence.get("board_id") != manifest.get("board_id"):
        raise PublicationHold("completion gate board identity mismatch")
    for key in ("active_claims", "pending_merges", "blocking_obligations"):
        if type(evidence.get(key)) is not int or evidence[key] != 0:
            raise PublicationHold(f"completion gate requires {key}=0")
    if evidence.get("source_heads") != dict(heads):
        raise PublicationHold("completion evidence does not bind the exact accepted source heads")
    return {key: evidence[key] for key in (
        "board_id", "authoritative", "complete", "active_claims", "pending_merges",
        "blocking_obligations", "source_heads",
    )}


def _gitlinks(root: Path, ref: str) -> dict[str, str]:
    links = {}
    for entry in _git(root, "ls-tree", "-r", "-z", ref).split("\0"):
        if entry.startswith("160000 "):
            descriptor, path = entry.split("\t", 1)
            links[path] = descriptor.split()[2]
    return links


def _fetch_main(root: Path, remote: str, ident: str) -> str:
    ref = f"refs/remotes/fleet-publication/{ident}/main"
    _git(root, "fetch", "--no-tags", remote, f"refs/heads/main:{ref}")
    return _git(root, "rev-parse", "--verify", ref + "^{commit}")


def _ancestor(root: Path, older: str, newer: str) -> bool:
    # Exit code 1 is a normal non-ancestor result; other failures are holds.
    result = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=root,
                            capture_output=True, timeout=30)
    if result.returncode not in (0, 1):
        raise PublicationHold("could not establish repository ancestry")
    return result.returncode == 0


def publish_completed_board(manifest: Mapping[str, Any], state_dir: str | Path) -> dict[str, Any]:
    """Try publication once, returning and atomically recording a typed receipt.

    The state directory is private to this board. ``held`` is retryable after
    repairing the recorded condition. Successful dependencies remain published
    when a parent holds, and later attempts safely reuse their merged commits.
    """
    state = Path(state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "board_id": manifest.get("board_id"), "status": "held",
        "started_at": time.time(), "repositories": [],
    }
    with (state / "publication.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**receipt, "reason": "another publisher holds this board's lock"}
        try:
            if manifest.get("schema") != SCHEMA or not manifest.get("board_id"):
                raise PublicationHold("unsupported publication manifest or missing board id")
            repositories = _ordered_repositories(manifest)
            heads = _source_heads(repositories)
            receipt["completion_evidence"] = _completion_gate(manifest, heads)
            remotes = {repo["id"]: _github_origin(Path(repo["root"])) for repo in repositories}
            published: dict[str, str] = {}
            # Validate the whole source graph before publishing even a leaf.
            for repo in repositories:
                links = _gitlinks(Path(repo["root"]), heads[repo["id"]])
                for dep in repo.get("dependencies", []):
                    if links.get(dep["path"]) != heads[dep["repository"]]:
                        raise PublicationHold(f"{repo['id']}: dependency gitlink differs from accepted source")
            for repo in repositories:
                ident, root = repo["id"], Path(repo["root"]).resolve()
                remote, source = remotes[ident], heads[ident]
                row: dict[str, Any] = {"id": ident, "source_head": source, "status": "preparing"}
                receipt["repositories"].append(row)
                main = _fetch_main(root, remote, ident)
                row["main_before"] = main
                source_links, main_links = _gitlinks(root, source), _gitlinks(root, main)
                dependencies = {d["path"]: d["repository"] for d in repo.get("dependencies", [])}
                unknown = [p for p, oid in source_links.items()
                           if main_links.get(p) != oid and p not in dependencies]
                if unknown:
                    raise PublicationHold(f"{ident}: changed gitlinks lack declared repository dependencies")
                # Already merged and all required children are present: idempotent retry.
                if _ancestor(root, source, main) and all(
                    main_links.get(path) == published[dep] for path, dep in dependencies.items()
                ):
                    row.update(status="already_published", published_head=main)
                    published[ident] = main
                    _atomic_json(state / "publication.json", receipt)
                    continue
                key = hashlib.sha256(f"{ident}:{source}:{main}:{time.time_ns()}".encode()).hexdigest()[:16]
                integration = state / "integrations" / f"{ident}-{key}"
                integration.parent.mkdir(exist_ok=True)
                _git(root, "worktree", "add", "--detach", str(integration), main)
                row["integration_worktree"] = str(integration)
                try:
                    _git(integration, "merge", "--no-ff", "--no-commit", source)
                except PublicationHold:
                    unresolved = _git(integration, "diff", "--name-only", "--diff-filter=U").splitlines()
                    if not unresolved or any(path not in dependencies for path in unresolved):
                        raise PublicationHold(f"{ident}: merge conflict; preserved isolated integration checkout") from None
                for path, dep in dependencies.items():
                    # The dependency's already-published commit is authoritative
                    # for resolving a declared submodule conflict, never a worker ref.
                    _git(integration, "update-index", "--add", "--cacheinfo", f"160000,{published[dep]},{path}")
                if _git(integration, "diff", "--name-only", "--diff-filter=U"):
                    raise PublicationHold(f"{ident}: unresolved integration conflict")
                staged = _git(integration, "diff", "--cached", "--name-only")
                merge_head = Path(_git(integration, "rev-parse", "--git-path", "MERGE_HEAD"))
                if staged or merge_head.exists():
                    _git(integration, "commit", "-m", f"Complete {manifest['board_id']}: integrate accepted {ident} work")
                paths = sorted(set(dependencies) | set(repo.get("initialize_submodules", [])))
                if paths:
                    _git(integration, "submodule", "update", "--init", "--recursive", "--", *paths)
                candidate = _git(integration, "rev-parse", "HEAD")
                for command in repo["validation"]:
                    _command(command, integration)
                if _git(integration, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none"):
                    raise PublicationHold(f"{ident}: publication validation changed the integration checkout")
                if _git(integration, "rev-parse", "HEAD") != candidate:
                    raise PublicationHold(f"{ident}: publication validation changed the integration commit")
                if _source_heads(repositories) != heads:
                    raise PublicationHold("accepted source changed during publication")
                _completion_gate(manifest, heads)
                # Fetch again to detect movement during validation. Publication
                # then goes through an isolated branch and a qualified PR merge.
                if _fetch_main(root, remote, ident) != main:
                    raise PublicationHold(f"{ident}: GitHub main advanced during validation; retry required")
                publication = _publish_through_required_pull_request(
                    integration, remote=remote, ident=ident, candidate=candidate,
                    main=main, board_id=str(manifest["board_id"]),
                )
                observed = _fetch_main(root, remote, ident)
                if observed != publication["published_head"]:
                    raise PublicationHold(f"{ident}: GitHub main moved after the qualified pull request merge")
                if not _ancestor(root, candidate, observed):
                    raise PublicationHold(f"{ident}: published commit is not reachable from GitHub main")
                row.update(
                    status="published", published_head=observed,
                    publication_branch=publication["publication_branch"],
                    pull_request=publication["pull_request"],
                )
                published[ident] = observed
                _atomic_json(state / "publication.json", receipt)
            receipt["status"] = "published"
        except (PublicationHold, OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
            receipt["reason"] = str(exc) if isinstance(exc, PublicationHold) else type(exc).__name__
            if receipt["repositories"] and receipt["repositories"][-1]["status"] == "preparing":
                receipt["repositories"][-1]["status"] = "held"
        receipt["finished_at"] = time.time()
        _atomic_json(state / "publication.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = publish_completed_board(json.loads(args.manifest.read_text()), args.state_dir)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "published" else 2


if __name__ == "__main__":
    raise SystemExit(main())
