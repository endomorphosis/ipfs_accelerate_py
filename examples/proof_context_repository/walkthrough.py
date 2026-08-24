#!/usr/bin/env python3
"""Run the credential-free proof-context v0.1 example.

The script is deliberately a consumer of the installed runtime. It does not
implement lifecycle authority, contact a provider, or infer a repository from
the current directory. A successful transcript is emitted only after a real
governed rejection, exact-tree selected-test evidence, live runtime acceptance,
verification-receipt reuse, assurance, and sealing all succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.proof_context.bootstrap import RuntimeOptions, open_runtime
from ipfs_accelerate_py.proof_context.facade import EngineIdentities, EngineRecord
from ipfs_accelerate_py.proof_context.lifecycle import STAGES, mint_lifecycle_cid

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/example-walkthrough@1"
TASK_ID: Final[str] = "PCCE-055"
MODE: Final[str] = "production"
TOKEN_ESTIMATOR: Final[str] = "whitespace-v1"
TARGET: Final[str] = "src/demo/__init__.py"
GOOD_TEST: Final[str] = "tests/test_double.py"
SELECTED_TESTS: Final[tuple[str, ...]] = (
    "tests/test_demo.py",
    "tests/test_labels.py",
    GOOD_TEST,
)
COMPRESSED_CONTEXT: Final[tuple[str, ...]] = (TARGET,)
EXPANDED_CONTEXT: Final[tuple[str, ...]] = (
    TARGET,
    "src/demo/labels.py",
    "tests/test_demo.py",
    "tests/test_labels.py",
)
GOOD_SOURCE: Final[str] = '''"""Tiny ordinary-Python package used by the PCCE walkthrough."""


def increment(value: int) -> int:
    """Return the next integer."""

    return value + 1


def double(value: int) -> int:
    """Return twice an integer."""

    return value * 2


__all__ = ["double", "increment"]
'''
GOOD_TEST_SOURCE: Final[str] = """from unittest import TestCase

from demo import double


class TestDouble(TestCase):
    def test_double(self) -> None:
        self.assertEqual(double(6), 12)
"""
GOOD_FILES: Final[Mapping[str, str]] = {
    TARGET: GOOD_SOURCE,
    GOOD_TEST: GOOD_TEST_SOURCE,
}


class WalkthroughError(RuntimeError):
    """A fail-closed example qualification failure."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _command_environment(home: Path, *, source_root: Path | None = None) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "pcce-example",
        "GIT_AUTHOR_EMAIL": "pcce-example@invalid.example",
        "GIT_COMMITTER_NAME": "pcce-example",
        "GIT_COMMITTER_EMAIL": "pcce-example@invalid.example",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    python_path = os.environ.get("PYTHONPATH", "")
    prefixes = [str(source_root)] if source_root is not None else []
    environment["PYTHONPATH"] = os.pathsep.join(item for item in (*prefixes, python_path) if item)
    return environment


def _git(repository: Path, *args: str, home: Path) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
        env=_command_environment(home),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git failed").strip()
        raise WalkthroughError(f"git {' '.join(args)} failed: {detail[:240]}")
    return completed.stdout.strip()


def _record(record: EngineRecord) -> dict[str, Any]:
    if record.provenance != "live":
        raise WalkthroughError(f"{record.operation} returned non-live provenance")
    return {
        "operation": record.operation,
        "status": record.status,
        "artifact_cid": record.artifact_cid,
        "provenance": record.provenance,
        "patch_id": record.identities.patch_id,
        "artifact_id": record.identities.artifact_id,
    }


def _failure_evidence(record: EngineRecord) -> tuple[str | None, str | None]:
    trace = record.payload.get("trace")
    if not isinstance(trace, (list, tuple)):
        return None, None
    for item in reversed(trace):
        if isinstance(item, Mapping) and item.get("error"):
            artifact_cid = item.get("artifact_cid")
            return str(item["error"]), str(artifact_cid) if artifact_cid else None
    return None, None


def _identity(repository: Path, run_name: str, *, home: Path) -> EngineIdentities:
    head = _git(repository, "rev-parse", "HEAD", home=home)
    ref = _git(repository, "rev-parse", "--abbrev-ref", "HEAD", home=home)
    state_cid = mint_lifecycle_cid({"kind": "example-repository-state", "head": head, "ref": ref})
    return EngineIdentities(
        repository_id="pcce/example-repository",
        repository_state_cid=state_cid,
        task_id=TASK_ID,
        run_id=mint_lifecycle_cid({"kind": "example-run", "name": run_name, "head": head}),
        trace_id=mint_lifecycle_cid({"kind": "example-trace", "name": run_name, "head": head}),
    )


def _tracked_files(repository: Path, *, home: Path) -> tuple[str, ...]:
    raw = _git(repository, "ls-files", "-z", home=home)
    return tuple(item for item in raw.split("\0") if item)


def _context_identity(repository: Path, paths: Sequence[str]) -> tuple[str, int]:
    payload = bytearray()
    tokens = 0
    for relative in sorted(dict.fromkeys(paths)):
        path = repository / relative
        if not path.is_file() or not path.resolve().is_relative_to(repository.resolve()):
            raise WalkthroughError(f"context path is unavailable or unsafe: {relative}")
        content = path.read_bytes()
        encoded_path = relative.encode("utf-8")
        payload.extend(len(encoded_path).to_bytes(4, "big"))
        payload.extend(encoded_path)
        payload.extend(len(content).to_bytes(8, "big"))
        payload.extend(content)
        tokens += len(re.findall(rb"\S+", content))
    return _sha256(bytes(payload)), tokens


def _apply_files(root: Path, files: Mapping[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        if not target.resolve().is_relative_to(root.resolve()):
            raise WalkthroughError(f"candidate path escaped its worktree: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _selected_test_receipt(
    repository: Path,
    candidate: Path,
    state_dir: Path,
    *,
    home: Path,
) -> dict[str, Any]:
    _git(repository, "worktree", "add", "--detach", str(candidate), "HEAD", home=home)
    _apply_files(candidate, GOOD_FILES)
    _git(candidate, "add", "--", *GOOD_FILES, home=home)
    _git(candidate, "commit", "-m", "pcce disposable external patch", home=home)
    candidate_tree = _git(candidate, "rev-parse", "HEAD^{tree}", home=home)
    selected_modules = [path.removesuffix(".py").replace("/", ".") for path in SELECTED_TESTS]
    argv = [sys.executable, "-m", "unittest", "-q", *selected_modules]
    completed = subprocess.run(
        argv,
        cwd=candidate,
        check=False,
        capture_output=True,
        text=True,
        env=_command_environment(state_dir / "home", source_root=candidate / "src"),
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()
        raise WalkthroughError(f"selected tests failed: {detail[-1200:]}")
    body = {
        "schema": "ipfs-accelerate.proof-context.v0.1/example-selected-test-receipt@1",
        "candidate_tree": candidate_tree,
        "command": ["python", "-m", "unittest", "-q", *selected_modules],
        "selected_tests": list(SELECTED_TESTS),
        "status": "passed",
        "provenance": "live",
        "execution_count": 1,
    }
    return {**body, "receipt_cid": mint_lifecycle_cid(body)}


def _remove_worktree(repository: Path, candidate: Path, *, home: Path) -> None:
    if candidate.exists():
        _git(repository, "worktree", "remove", "--force", str(candidate), home=home)
    _git(repository, "worktree", "prune", home=home)


def run_walkthrough(repository: Path, state_dir: Path) -> dict[str, Any]:
    repository = repository.resolve()
    state_dir = state_dir.resolve()
    home = state_dir / "home"
    if not (repository / ".git").exists():
        raise WalkthroughError("repository must be a fresh ordinary Git clone")
    if state_dir == repository or state_dir.is_relative_to(repository):
        raise WalkthroughError("state directory must be outside the governed repository")
    state_dir.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    initial_status = _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all", home=home
    )
    if initial_status:
        raise WalkthroughError("fresh clone must be clean")
    initial_head = _git(repository, "rev-parse", "HEAD", home=home)
    initial_tree = _git(repository, "rev-parse", "HEAD^{tree}", home=home)
    tracked = _tracked_files(repository, home=home)
    full_cid, full_tokens = _context_identity(repository, tracked)
    compressed_cid, compressed_tokens = _context_identity(repository, COMPRESSED_CONTEXT)
    expanded_cid, expanded_tokens = _context_identity(repository, EXPANDED_CONTEXT)
    if not 0 < compressed_tokens < expanded_tokens < full_tokens:
        raise WalkthroughError("fixture does not demonstrate bounded context expansion")

    context_bundle = open_runtime(
        repository,
        identities=_identity(repository, "context", home=home),
        mode=MODE,
        options=RuntimeOptions(
            kit_root=state_dir / "context" / "store",
            worktree_parent=state_dir / "context" / "worktrees",
            operator_id="pcce-example-context",
        ),
    )
    scan = context_bundle.engine.scan()
    plan = context_bundle.engine.plan()
    pack = context_bundle.engine.context_pack()
    expansion = context_bundle.engine.expand_context()
    for record in (scan, plan, pack, expansion):
        if record.status != "succeeded":
            raise WalkthroughError(f"{record.operation} did not succeed")
    if list(pack.payload.get("declared_files") or ()) != [TARGET]:
        raise WalkthroughError("runtime context pack did not bind the fixture target")

    escape_path = repository.parent / "pcce-example-escape.txt"
    if escape_path.exists():
        raise WalkthroughError("bad-patch escape sentinel already exists")
    bad_bundle = open_runtime(
        repository,
        identities=_identity(repository, "bad-patch", home=home),
        mode=MODE,
        options=RuntimeOptions(
            kit_root=state_dir / "bad" / "store",
            worktree_parent=state_dir / "bad" / "worktrees",
            operator_id="pcce-example-bad-patch",
        ),
    )
    bad = bad_bundle.engine.run(
        {
            "declared_files": ["../pcce-example-escape.txt"],
            "files": {"../pcce-example-escape.txt": "escape rejected\n"},
            "adapter_id": "external-patch",
            "approver_id": "coordinator",
        }
    )
    bad_reason, bad_artifact_cid = _failure_evidence(bad)
    if (
        bad.status != "rejected"
        or bad.provenance != "live"
        or bad_reason != "boundary_violation"
        or not str(bad_artifact_cid or "").startswith("b")
        or escape_path.exists()
    ):
        raise WalkthroughError("unsafe patch was not rejected for a governed boundary")
    if _git(repository, "rev-parse", "HEAD", home=home) != initial_head:
        raise WalkthroughError("bad patch changed the canonical head")

    candidate = state_dir / "candidate"
    if candidate.exists():
        raise WalkthroughError("candidate worktree already exists")
    selected_receipt: dict[str, Any] | None = None
    good_bundle = None
    discard: Mapping[str, Any] = {"discarded": False}
    try:
        selected_receipt = _selected_test_receipt(repository, candidate, state_dir, home=home)
        good_bundle = open_runtime(
            repository,
            identities=_identity(repository, "good-patch", home=home),
            mode=MODE,
            options=RuntimeOptions(
                kit_root=state_dir / "good" / "store",
                worktree_parent=state_dir / "good" / "worktrees",
                operator_id="pcce-example-good-patch",
            ),
        )
        good = good_bundle.engine.run(
            {
                "declared_files": list(GOOD_FILES),
                "files": dict(GOOD_FILES),
                "adapter_id": "external-patch",
                "approver_id": "coordinator",
            }
        )
        if (
            good.status != "succeeded"
            or good.provenance != "live"
            or good.payload.get("published") is not True
            or good.payload.get("sealed") is not True
            or tuple(good.payload.get("stages") or ()) != STAGES
        ):
            raise WalkthroughError("good patch was not accepted through every governed stage")
        worktree_payload = good.payload.get("worktree")
        if not isinstance(worktree_payload, Mapping):
            raise WalkthroughError("accepted patch omitted its disposable worktree evidence")
        worktree_path = Path(str(worktree_payload.get("worktree_path") or "")).resolve()
        expected_parent = (state_dir / "good" / "worktrees").resolve()
        if not worktree_path.is_relative_to(expected_parent) or not worktree_path.is_dir():
            raise WalkthroughError("accepted patch worktree escaped the configured state root")
        admitted_tree = _git(worktree_path, "rev-parse", "HEAD^{tree}", home=home)
        if admitted_tree != selected_receipt["candidate_tree"]:
            raise WalkthroughError("selected-test receipt key does not match accepted tree")

        verify_first = good_bundle.engine.verify()
        verify_second = good_bundle.engine.verify()
        if (
            verify_first.status != "succeeded"
            or verify_second.status != "succeeded"
            or verify_first.artifact_cid != verify_second.artifact_cid
        ):
            raise WalkthroughError("runtime verification receipt was not reused exactly")
        verify_trace = [
            item
            for item in good.payload.get("trace") or ()
            if isinstance(item, Mapping) and item.get("stage") == "incremental-verify"
        ]
        if len(verify_trace) != 1:
            raise WalkthroughError("accepted lifecycle must contain one verification stage")

        assurance = good_bundle.engine.assurance()
        seal = good_bundle.engine.seal()
        report = good_bundle.engine.report()
        if any(item.status != "succeeded" for item in (assurance, seal, report)):
            raise WalkthroughError("assurance, seal, and report must all succeed")
        seal_cid = str(good.payload.get("seal_cid") or seal.payload.get("seal_cid") or "")
        if not seal_cid.startswith("b") or seal.artifact_cid == verify_first.artifact_cid:
            raise WalkthroughError("final seal identity is missing or aliases verification")

        bad_command = _record(bad)
        bad_command.update(
            {
                "failure_artifact_cid": bad_artifact_cid,
                "failure_reason": bad_reason,
            }
        )
        commands = [
            {
                "operation": "initialize",
                "status": "succeeded",
                "artifact_cid": _sha256(initial_tree.encode("ascii")),
                "provenance": "live-local",
            },
            _record(scan),
            _record(plan),
            _record(pack),
            _record(expansion),
            bad_command,
            {
                "operation": "incremental-tests",
                "status": "succeeded",
                "artifact_cid": selected_receipt["receipt_cid"],
                "provenance": "live",
            },
            _record(good),
            _record(verify_first),
            _record(verify_second),
            _record(assurance),
            _record(seal),
            _record(report),
        ]
        transcript = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "status": "succeeded",
            "mode": MODE,
            "provider_bound": False,
            "network_required": False,
            "credentials_required": False,
            "fixture": {
                "commit": initial_head,
                "tree": initial_tree,
                "clean_at_start": True,
                "tracked_file_count": len(tracked),
            },
            "context": {
                "estimator": TOKEN_ESTIMATOR,
                "before": {"cid": full_cid, "tokens": full_tokens},
                "compressed": {
                    "cid": compressed_cid,
                    "tokens": compressed_tokens,
                    "files": list(COMPRESSED_CONTEXT),
                    "record_cid": pack.artifact_cid,
                },
                "expanded": {
                    "cid": expanded_cid,
                    "tokens": expanded_tokens,
                    "files": list(EXPANDED_CONTEXT),
                    "record_cid": expansion.artifact_cid,
                },
            },
            "rejection": {
                "status": bad.status,
                "reason": bad_reason,
                "artifact_cid": bad_artifact_cid,
                "execution_receipt_cid": bad.artifact_cid,
                "canonical_head_unchanged": True,
                "escape_created": False,
            },
            "incremental_tests": selected_receipt,
            "proof_reuse": {
                "reused": True,
                "reason": "exact_candidate_tree_and_runtime_receipt_identity",
                "test_receipt_cid": selected_receipt["receipt_cid"],
                "source_candidate_tree": selected_receipt["candidate_tree"],
                "admitted_candidate_tree": admitted_tree,
                "verification_receipt_cid": verify_first.artifact_cid,
                "verification_calls": 2,
                "verification_stage_count": len(verify_trace),
                "reuse_count": 1,
            },
            "acceptance": {
                "status": good.status,
                "patch_id": good.identities.patch_id,
                "execution_receipt_cid": good.artifact_cid,
                "all_stages": list(good.payload.get("stages") or ()),
                "published": good.payload.get("published"),
                "canonical_mutated": good.payload.get("canonical_mutated"),
            },
            "seal": {
                "status": seal.status,
                "seal_cid": seal_cid,
                "artifact_cid": seal.artifact_cid,
                "provenance": seal.provenance,
            },
            "commands": commands,
        }
    finally:
        try:
            if good_bundle is not None:
                discard = good_bundle.session.worktree.discard(
                    good_bundle.session.lifecycle_identities, repository
                )
        finally:
            _remove_worktree(repository, candidate, home=home)

    if discard.get("discarded") is not True:
        raise WalkthroughError("accepted disposable worktree was not removed")
    if _git(repository, "rev-parse", "HEAD", home=home) != initial_head:
        raise WalkthroughError("walkthrough changed the canonical head")
    final_status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all", home=home)
    if final_status:
        raise WalkthroughError("walkthrough left the canonical clone dirty")
    transcript["fixture"]["clean_at_finish"] = True
    transcript["acceptance"]["disposable_worktree_removed"] = True
    transcript["transcript_cid"] = mint_lifecycle_cid(transcript)
    return transcript


def _outside(repository: Path, candidate: Path, *, label: str) -> Path:
    resolved_repository = repository.resolve()
    resolved = candidate.resolve()
    if resolved == resolved_repository or resolved.is_relative_to(resolved_repository):
        raise WalkthroughError(f"{label} must be outside the governed repository")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    repository = arguments.repository.resolve()
    state_dir = _outside(repository, arguments.state_dir, label="state directory")
    output = _outside(repository, arguments.output, label="output")
    transcript = run_walkthrough(repository, state_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json(transcript) + b"\n")
    print(json.dumps(transcript, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
