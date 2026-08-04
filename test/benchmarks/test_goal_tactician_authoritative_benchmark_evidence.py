"""Adversarial tests for live GoalTactician benchmark authority.

These tests create an isolated Git repository and never promote or rewrite the
checked-in FVT-G063 fixture benchmark.  The positive case establishes only
that the repository-bound verification contract is replayable; production
authority still requires a genuine live cohort published on trusted main.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_metrics import (
    GOAL_TACTICIAN_AUTHORITATIVE_COHORT_INTERFACE,
    GOAL_TACTICIAN_AUTHORITATIVE_COHORT_SCHEMA,
    GOAL_TACTICIAN_BENCHMARK_AUTHORITY_GOAL_ID,
    GOAL_TACTICIAN_BENCHMARK_AUTHORITY_INTERFACE,
    GOAL_TACTICIAN_BENCHMARK_AUTHORITY_SCHEMA,
    GOAL_TACTICIAN_BENCHMARK_VERIFIER_FUNCTION,
    GOAL_TACTICIAN_BENCHMARK_VERIFIER_PATH,
    CacheOutcome,
    EvidenceClass,
    GoalTacticianRunReceipt,
    architecture_benchmark_document,
    build_goal_tactician_benchmark_report,
    fixture_cohort_receipts,
    verify_authoritative_benchmark_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_MODULE = REPO_ROOT / GOAL_TACTICIAN_BENCHMARK_VERIFIER_PATH
BUILDER_PATH = (
    REPO_ROOT / "tools" / "logic" / "build_formal_verification_tactician_receipt.py"
)
COHORT_PATH = "evidence/goal-tactician-live-cohort.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _content_id(value: Any, *, prefix: str = "sha256:") -> str:
    return prefix + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_id(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _live_receipt() -> GoalTacticianRunReceipt:
    """Return a strict live-contract sample used only in an isolated test repo."""

    return GoalTacticianRunReceipt(
        receipt_id="receipt:fvt063:ci-z3:0001",
        run_id="run:fvt063:ci-z3:0001",
        goal_id=GOAL_TACTICIAN_BENCHMARK_AUTHORITY_GOAL_ID,
        repository_tree_id="sha256:" + ("a" * 64),
        policy_id="policy:formal-verification-tactician@1",
        provider_id="provider:z3@1",
        evidence_class=EvidenceClass.LIVE,
        formalization_attempted=True,
        formalization_succeeded=True,
        formalization_required=True,
        proof_gap_true_positive=3,
        proof_gap_false_positive=0,
        proof_gap_false_negative=0,
        proof_gap_true_negative=2,
        plan_steps_total=4,
        plan_steps_solvable=4,
        plan_admitted=True,
        claimed_assurance="kernel_verified",
        authoritative_assurance="kernel_verified",
        authority_boundary_violation=False,
        false_completion=False,
        privacy_violation=False,
        counterexample_count=1,
        counterexample_replayable_count=1,
        counterexample_reduced_count=1,
        counterexample_explained_count=1,
        providers_queried=("provider:z3@1", "provider:cvc5@1"),
        providers_agreeing=("provider:z3@1", "provider:cvc5@1"),
        wall_time_ms=25,
        cpu_time_ms=20,
        memory_peak_bytes=8 * 1024 * 1024,
        cancelled=False,
        cancellation_honored=True,
        calibration_receipt_id="",
        cache_outcome=CacheOutcome.MISS,
        cache_key="",
        cache_authority_preserved=True,
        cache_identity_preserved=True,
        unresolved_hole_ids=(),
        witness_ids=("witness:counterexample:0001",),
        critical_path_step_ids=("step:formalize", "step:check"),
        budget_cpu_ms_remaining=1_000,
        budget_memory_bytes_remaining=64 * 1024 * 1024,
        budget_token_remaining=2_000,
        next_actions=("publish-live-cohort-receipt",),
    )


def _commit_identity(root: Path, relative_path: str) -> str:
    return _git(root, "rev-parse", f"HEAD:{relative_path}")


def _refresh_authority_identity(document: dict[str, Any]) -> None:
    authority = document["authoritative_measurement"]
    authority.pop("content_id", None)
    authority["content_id"] = _content_id(authority)


def _refresh_report_and_authority_identities(document: dict[str, Any]) -> None:
    report = document["report"]
    report.pop("report_id", None)
    report["report_id"] = _content_id(
        report,
        prefix="goal-tactician-bench-",
    )
    document["authoritative_measurement"]["report_id"] = report["report_id"]
    _refresh_authority_identity(document)


def _bound_document(
    tmp_path: Path,
    *,
    receipts: tuple[GoalTacticianRunReceipt, ...] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    repository = tmp_path / "authority-repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Benchmark Authority Test")
    _git(repository, "config", "user.email", "benchmark-authority@example.invalid")

    verifier_path = repository / GOAL_TACTICIAN_BENCHMARK_VERIFIER_PATH
    verifier_path.parent.mkdir(parents=True)
    shutil.copyfile(METRICS_MODULE, verifier_path)

    typed_receipts = receipts or (_live_receipt(),)
    strict_receipts = [receipt.to_dict() for receipt in typed_receipts]
    receipt_ids = [receipt.receipt_id for receipt in typed_receipts]
    notes = "Repository-bound cohort contract test; not production evidence."
    generated_at = "2026-08-03T03:30:00Z"
    cohort = {
        "schema": GOAL_TACTICIAN_AUTHORITATIVE_COHORT_SCHEMA,
        "interface": GOAL_TACTICIAN_AUTHORITATIVE_COHORT_INTERFACE,
        "goal_id": GOAL_TACTICIAN_BENCHMARK_AUTHORITY_GOAL_ID,
        "task_id": "FVT-033",
        "cohort_id": "formal-verification-tactician/ci-live-contract",
        "generated_at": generated_at,
        "source": "live_cohort_receipts",
        "synthetic_distributions": False,
        "notes": notes,
        "receipt_count": len(strict_receipts),
        "receipt_ids": receipt_ids,
        "receipt_content_ids": [
            {
                "receipt_id": receipt["receipt_id"],
                "content_id": _content_id(receipt),
            }
            for receipt in strict_receipts
        ],
        "receipt_set_id": _content_id(
            strict_receipts,
            prefix="goal-tactician-receipts-",
        ),
        "receipts": strict_receipts,
    }
    cohort["content_id"] = _content_id(cohort)
    cohort_path = repository / COHORT_PATH
    cohort_path.parent.mkdir(parents=True)
    cohort_path.write_text(_canonical_json(cohort) + "\n", encoding="utf-8")

    _git(repository, "add", GOAL_TACTICIAN_BENCHMARK_VERIFIER_PATH, COHORT_PATH)
    _git(repository, "commit", "-m", "Bind benchmark verifier and cohort evidence")
    commit_sha = _git(repository, "rev-parse", "HEAD^{commit}")

    report = build_goal_tactician_benchmark_report(
        typed_receipts,
        goal_id=cohort["goal_id"],
        task_id=cohort["task_id"],
        cohort_id=cohort["cohort_id"],
        generated_at=cohort["generated_at"],
        notes=cohort["notes"],
    )
    document = architecture_benchmark_document(report)
    authority = {
        "schema": GOAL_TACTICIAN_BENCHMARK_AUTHORITY_SCHEMA,
        "interface": GOAL_TACTICIAN_BENCHMARK_AUTHORITY_INTERFACE,
        "goal_id": GOAL_TACTICIAN_BENCHMARK_AUTHORITY_GOAL_ID,
        "report_id": report["report_id"],
        "receipt_artifact": {
            "path": COHORT_PATH,
            "sha256": _file_id(cohort_path),
            "content_id": cohort["content_id"],
        },
        "repository_binding": {
            "trusted_ref": "refs/heads/main",
            "commit_sha": commit_sha,
            "tree_sha": _git(repository, "rev-parse", f"{commit_sha}^{{tree}}"),
            "receipt_blob_sha": _commit_identity(repository, COHORT_PATH),
            "verifier_blob_sha": _commit_identity(
                repository,
                GOAL_TACTICIAN_BENCHMARK_VERIFIER_PATH,
            ),
        },
        "verifier": {
            "path": GOAL_TACTICIAN_BENCHMARK_VERIFIER_PATH,
            "function": GOAL_TACTICIAN_BENCHMARK_VERIFIER_FUNCTION,
            "sha256": _file_id(verifier_path),
        },
    }
    authority["content_id"] = _content_id(authority)
    document["authoritative_measurement"] = authority
    return document, repository, cohort_path


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "goal_tactician_benchmark_authority_builder_test",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_bound_live_cohort_recomputes_and_binds_report(
    tmp_path: Path,
) -> None:
    document, repository, _ = _bound_document(tmp_path)

    result = verify_authoritative_benchmark_evidence(
        document,
        repo_root=repository,
    )

    assert result["valid"] is True
    assert result["failures"] == []
    assert result["report_id"] == document["report"]["report_id"]
    assert (
        result["authority_content_id"]
        == document["authoritative_measurement"]["content_id"]
    )
    assert result["receipt_count"] == 1
    assert result["evidence_classes"] == ["live"]
    assert result["trusted_commit"] == _git(repository, "rev-parse", "HEAD")


def test_release_builder_can_load_and_call_repository_verifier(
    tmp_path: Path,
) -> None:
    document, repository, _ = _bound_document(tmp_path)
    builder = _load_builder()

    evidence = builder._benchmark_hard_gate_evidence(
        document,
        repo_root=repository,
    )

    assert evidence["authoritative"] is True
    assert evidence["failures"] == []
    assert evidence["authority_anchor"]["bound"] is True


def test_checked_in_fixture_is_not_authoritative() -> None:
    fixture_document = json.loads(
        (
            REPO_ROOT
            / "docs"
            / "architecture"
            / "formal_verification_tactician_benchmark.json"
        ).read_text(encoding="utf-8")
    )

    result = verify_authoritative_benchmark_evidence(
        fixture_document,
        repo_root=REPO_ROOT,
    )

    assert result["valid"] is False
    assert "authoritative_measurement_missing_or_invalid" in result["failures"]
    assert fixture_document["report"]["metrics"]["evidence_classes"] == ["fixture"]


def test_self_asserted_anchor_without_replayable_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    document, repository, _ = _bound_document(tmp_path)
    document["authoritative_measurement"] = {
        "schema": GOAL_TACTICIAN_BENCHMARK_AUTHORITY_SCHEMA,
        "interface": GOAL_TACTICIAN_BENCHMARK_AUTHORITY_INTERFACE,
        "goal_id": GOAL_TACTICIAN_BENCHMARK_AUTHORITY_GOAL_ID,
        "report_id": document["report"]["report_id"],
        "authoritative": True,
        "verified": True,
        "content_id": "sha256:" + ("0" * 64),
    }

    result = verify_authoritative_benchmark_evidence(
        document,
        repo_root=repository,
    )

    assert result["valid"] is False
    assert "authority_fields_malformed_or_self_asserted" in result["failures"]
    assert "receipt_artifact_claim_missing_or_invalid" in result["failures"]


def test_fixture_receipts_cannot_be_promoted_by_a_valid_git_envelope(
    tmp_path: Path,
) -> None:
    document, repository, _ = _bound_document(
        tmp_path,
        receipts=fixture_cohort_receipts(),
    )

    result = verify_authoritative_benchmark_evidence(
        document,
        repo_root=repository,
    )

    assert result["valid"] is False
    assert "receipt_evidence_class_not_authoritative" in result["failures"]
    assert "receipt_identity_fixture_or_synthetic" in result["failures"]


def test_mutated_artifact_bytes_and_declared_digest_are_rejected(
    tmp_path: Path,
) -> None:
    document, repository, cohort_path = _bound_document(tmp_path)
    cohort_path.write_bytes(cohort_path.read_bytes() + b" ")

    result = verify_authoritative_benchmark_evidence(
        document,
        repo_root=repository,
    )

    assert result["valid"] is False
    assert "receipt_artifact_sha256_mismatch" in result["failures"]
    assert "receipt_artifact_not_committed_exactly" in result["failures"]

    document, repository, _ = _bound_document(tmp_path / "declared-digest")
    authority = document["authoritative_measurement"]
    authority["receipt_artifact"]["sha256"] = "sha256:" + ("0" * 64)
    _refresh_authority_identity(document)
    result = verify_authoritative_benchmark_evidence(
        document,
        repo_root=repository,
    )
    assert result["valid"] is False
    assert "receipt_artifact_sha256_mismatch" in result["failures"]


def test_report_mutation_is_rejected_even_when_all_outer_ids_are_recomputed(
    tmp_path: Path,
) -> None:
    document, repository, _ = _bound_document(tmp_path)
    document["report"]["metrics"]["proof_gap"]["true_positive"] += 100
    _refresh_report_and_authority_identities(document)

    result = verify_authoritative_benchmark_evidence(
        document,
        repo_root=repository,
    )

    assert result["valid"] is False
    assert "report_not_exact_recomputation" in result["failures"]


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        (
            lambda authority: authority["receipt_artifact"].__setitem__(
                "path",
                "../outside.json",
            ),
            "receipt_artifact_path_not_safe_relative",
        ),
        (
            lambda authority: authority["repository_binding"].__setitem__(
                "trusted_ref",
                "refs/heads/unreviewed",
            ),
            "trusted_ref_mismatch",
        ),
        (
            lambda authority: authority["verifier"].__setitem__(
                "function",
                "claimed_valid_without_replay",
            ),
            "verifier_function_mismatch",
        ),
    ],
)
def test_malformed_or_unverifiable_anchor_fails_closed(
    tmp_path: Path,
    mutation,
    expected_failure: str,
) -> None:
    document, repository, _ = _bound_document(tmp_path)
    mutation(document["authoritative_measurement"])
    _refresh_authority_identity(document)

    result = verify_authoritative_benchmark_evidence(
        document,
        repo_root=repository,
    )

    assert result["valid"] is False
    assert expected_failure in result["failures"]
