"""PCCE-055: fresh-clone qualification for the synthetic walkthrough."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ACCELERATOR_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ACCELERATOR_ROOT / "examples" / "proof_context_repository"
SCHEMA = "ipfs-accelerate.proof-context.v0.1/example-walkthrough@1"
EXPECTED_STAGES = [
    "identify-operator",
    "resolve-repository",
    "scan-semantic",
    "invalidate",
    "context-pack",
    "sufficiency",
    "route",
    "proposal",
    "scope-check",
    "isolated-apply",
    "impact",
    "incremental-verify",
    "escalate",
    "assurance",
    "seal",
    "disposition",
]


def _environment(home: Path) -> dict[str, str]:
    return {
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
        "PYTHONPATH": str(ACCELERATOR_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }


def _run(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    expected: int = 0,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    return completed


def _git(repository: Path, environment: dict[str, str], *args: str) -> str:
    return _run(
        ["/usr/bin/git", "-C", str(repository), *args],
        cwd=repository.parent,
        environment=environment,
    ).stdout.strip()


def _seed(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    seed = tmp_path / "seed"
    home = tmp_path / "home"
    home.mkdir()
    environment = _environment(home)
    shutil.copytree(EXAMPLE_ROOT, seed)
    _run(
        ["/usr/bin/git", "init", "--initial-branch=main", str(seed)],
        cwd=tmp_path,
        environment=environment,
    )
    _git(seed, environment, "add", "-A")
    _git(seed, environment, "commit", "-m", "initial proof-context example")
    assert _git(seed, environment, "status", "--porcelain=v1") == ""
    return seed, environment


def _fresh_run(
    tmp_path: Path,
    seed: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    clone = tmp_path / "clone"
    state = tmp_path / "state"
    output = state / "transcript.json"
    _run(
        ["/usr/bin/git", "clone", "--local", "--no-hardlinks", str(seed), str(clone)],
        cwd=tmp_path,
        environment=environment,
    )
    completed = _run(
        [
            sys.executable,
            str(clone / "walkthrough.py"),
            "--repository",
            str(clone),
            "--state-dir",
            str(state),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        environment=environment,
    )
    from_file = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == from_file
    assert _git(clone, environment, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert (clone / "src" / "demo" / "__init__.py").read_text(encoding="utf-8").count(
        "def double"
    ) == 0
    assert not (tmp_path / "pcce-example-escape.txt").exists()
    return from_file


def _assert_transcript(transcript: dict[str, Any]) -> None:
    assert transcript["schema"] == SCHEMA
    assert transcript["task_id"] == "PCCE-055"
    assert transcript["status"] == "succeeded"
    assert transcript["mode"] == "production"
    assert transcript["provider_bound"] is False
    assert transcript["network_required"] is False
    assert transcript["credentials_required"] is False
    assert transcript["fixture"]["clean_at_start"] is True
    assert transcript["fixture"]["clean_at_finish"] is True
    assert len(transcript["fixture"]["commit"]) == 40
    assert len(transcript["fixture"]["tree"]) == 40

    context = transcript["context"]
    assert context["estimator"] == "whitespace-v1"
    assert 0 < context["compressed"]["tokens"] < context["expanded"]["tokens"]
    assert context["expanded"]["tokens"] < context["before"]["tokens"]
    assert context["compressed"]["record_cid"].startswith("b")
    assert context["expanded"]["record_cid"].startswith("b")

    rejection = transcript["rejection"]
    assert rejection == {
        "status": "rejected",
        "reason": "boundary_violation",
        "artifact_cid": rejection["artifact_cid"],
        "execution_receipt_cid": rejection["execution_receipt_cid"],
        "canonical_head_unchanged": True,
        "escape_created": False,
    }
    assert rejection["artifact_cid"].startswith("b")
    assert rejection["execution_receipt_cid"].startswith("b")

    tests = transcript["incremental_tests"]
    assert tests["status"] == "passed"
    assert tests["provenance"] == "live"
    assert tests["execution_count"] == 1
    assert tests["selected_tests"] == [
        "tests/test_demo.py",
        "tests/test_labels.py",
        "tests/test_double.py",
    ]
    assert tests["receipt_cid"].startswith("b")

    reuse = transcript["proof_reuse"]
    assert reuse["reused"] is True
    assert reuse["reason"] == "runtime_verification_receipt_identity"
    assert reuse["exact_candidate_tree_match"] is True
    assert reuse["source_candidate_tree"] == reuse["admitted_candidate_tree"]
    assert reuse["selected_test_receipt_cid"] == tests["receipt_cid"]
    assert reuse["verification_calls"] == 2
    assert reuse["verification_stage_count"] == 1
    assert reuse["reuse_count"] == 1
    assert reuse["verification_receipt_cid"].startswith("b")

    acceptance = transcript["acceptance"]
    assert acceptance["status"] == "succeeded"
    assert acceptance["all_stages"] == EXPECTED_STAGES
    assert acceptance["published"] is True
    assert acceptance["canonical_mutated"] is False
    assert acceptance["disposable_worktree_removed"] is True
    assert acceptance["patch_id"].startswith("b")
    assert acceptance["execution_receipt_cid"].startswith("b")
    assert rejection["execution_receipt_cid"] != acceptance["execution_receipt_cid"]

    seal = transcript["seal"]
    assert seal["status"] == "succeeded"
    assert seal["provenance"] == "live"
    assert seal["seal_cid"].startswith("b")
    assert seal["artifact_cid"].startswith("b")
    assert transcript["transcript_cid"].startswith("b")
    assert all(item["artifact_cid"] for item in transcript["commands"])
    assert all(item["status"] in {"succeeded", "rejected"} for item in transcript["commands"])


def test_fresh_clone_walkthrough_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    seed, environment = _seed(tmp_path)
    first = _fresh_run(tmp_path, seed, environment)
    _assert_transcript(first)

    shutil.rmtree(tmp_path / "clone")
    shutil.rmtree(tmp_path / "state")
    second = _fresh_run(tmp_path, seed, environment)
    _assert_transcript(second)
    assert second == first


def test_fixture_is_credential_free_and_self_contained() -> None:
    files = sorted(
        path
        for path in EXAMPLE_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".toml"}
    )
    assert files
    assert all(path.is_relative_to(EXAMPLE_ROOT) for path in files)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    forbidden = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROK_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "BEGIN PRIVATE KEY",
        "http://",
        "https://",
    )
    assert not any(marker in combined for marker in forbidden)


def test_walkthrough_rejects_state_inside_governed_clone(tmp_path: Path) -> None:
    seed, environment = _seed(tmp_path)
    clone = tmp_path / "clone"
    _run(
        ["/usr/bin/git", "clone", "--local", "--no-hardlinks", str(seed), str(clone)],
        cwd=tmp_path,
        environment=environment,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(clone / "walkthrough.py"),
            "--repository",
            str(clone),
            "--state-dir",
            str(clone / ".state"),
            "--output",
            str(tmp_path / "transcript.json"),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "state directory must be outside" in result.stderr
    assert _git(clone, environment, "status", "--porcelain=v1", "--untracked-files=all") == ""
