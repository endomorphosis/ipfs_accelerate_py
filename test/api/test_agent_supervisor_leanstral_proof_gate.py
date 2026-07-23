"""Security and provenance tests for the Leanstral proof/patch gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.kernel_verification import (
    KernelFailureCode,
    KernelVerificationBindings,
)
from ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider import (
    LEANSTRAL_DRAFT_SCHEMA_VERSION,
    LeanstralGateStatus,
    LeanstralPatchGatePolicy,
    LeanstralProofDraft,
    check_leanstral_patch_proposal,
    verify_leanstral_draft,
)
from ipfs_accelerate_py.agent_supervisor.proof_context import (
    FixedTheoremIdentity,
)


def _theorem() -> FixedTheoremIdentity:
    return FixedTheoremIdentity(
        theorem_id="Fixed.identity",
        obligation_id="obligation-1",
        declaration_name="Fixed.identity",
        assumptions=("P",),
        conclusion="P",
        template_id="identity",
        template_version="1",
        source_scope=("Fixed.identity",),
        canonical_source_digest="sha256:canonical",
    )


def _draft(proof_text: str) -> LeanstralProofDraft:
    theorem = _theorem()
    output_sha256 = hashlib.sha256(proof_text.encode("utf-8")).hexdigest()
    identity = {
        "schema_version": LEANSTRAL_DRAFT_SCHEMA_VERSION,
        "llm_provider": "leanstral_local",
        "model": "Leanstral",
        "obligation_ids": [theorem.obligation_id],
        "canonical_source_digest": theorem.canonical_source_digest,
        "theorem_id": theorem.theorem_id,
        "theorem_equivalence_key": theorem.equivalence_key,
        "context_capsule_id": "capsule-1",
        "proposal_kind": "proof",
        "prompt_sha256": "a" * 64,
        "output_sha256": output_sha256,
    }
    artifact_id = "leanstral-draft-" + hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return LeanstralProofDraft(
        artifact_id=artifact_id,
        draft_text=proof_text,
        request_id="request-1",
        llm_provider="leanstral_local",
        model="Leanstral",
        obligation_ids=(theorem.obligation_id,),
        canonical_source_digest=theorem.canonical_source_digest,
        prompt_sha256="a" * 64,
        output_sha256=output_sha256,
        timeout_ms=1_000,
        token_budget=128,
        theorem_id=theorem.theorem_id,
        theorem_equivalence_key=theorem.equivalence_key,
        context_capsule_id="capsule-1",
        proposal_kind="proof",
    )


def _bindings() -> KernelVerificationBindings:
    return KernelVerificationBindings(
        obligation_id="obligation-1",
        request_id="request-1",
        candidate_id="candidate-1",
        kernel_id="kernel:lean@test",
        toolchain_id="toolchain:lean@test",
    )


NATIVE_SOURCE = (
    "theorem Fixed.identity (h : P) : P := sorry\n"
    "#print axioms Fixed.identity\n"
)


def _accepting_kernel_runner(**kwargs):
    assert kwargs["source"] == (
        "theorem Fixed.identity (h : P) : P := by exact h\n"
        "#print axioms Fixed.identity\n"
    )
    return {
        "command": ["/tools/lean", "--json", "Reconstruction.lean"],
        "stdout": json.dumps(
            {
                "severity": "information",
                "data": "'Fixed.identity' does not depend on any axioms",
            }
        ),
        "stderr": "",
        "returncode": 0,
        "version": "Lean 4.test",
        "executable": "/tools/lean",
    }


@pytest.mark.parametrize(
    ("proof_text", "failure_code"),
    (
        ("import Mathlib\nby exact h", KernelFailureCode.FORBIDDEN_IMPORT),
        ("axiom forged : P\nby exact h", KernelFailureCode.FORBIDDEN_DECLARATION),
        ("unsafe def forged : P := h\nby exact h", KernelFailureCode.FORBIDDEN_DECLARATION),
        ("by sorry", KernelFailureCode.INCOMPLETE_PROOF),
        ("by admit", KernelFailureCode.INCOMPLETE_PROOF),
        (
            "theorem substitute : P := by exact h",
            KernelFailureCode.FORBIDDEN_DECLARATION,
        ),
        ("by exact Fixed.identity h", KernelFailureCode.THEOREM_SUBSTITUTION),
        (
            "diff --git a/Fixed.lean b/Fixed.lean\nby exact h",
            KernelFailureCode.SOURCE_COPY,
        ),
    ),
)
def test_forbidden_model_proof_text_fails_before_kernel(
    proof_text: str, failure_code: KernelFailureCode
) -> None:
    called = False

    def kernel_runner(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("rejected proof must not reach the kernel")

    result = verify_leanstral_draft(
        _draft(proof_text),
        _theorem(),
        native_source=NATIVE_SOURCE,
        bindings=_bindings(),
        kernel_runner=kernel_runner,
    )

    assert result.status is LeanstralGateStatus.REJECTED
    assert result.admission.failure_code is failure_code
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert not result.authoritative
    assert called is False


def test_accepted_draft_uses_independent_reconstruction_and_separate_provenance() -> None:
    draft = _draft("by exact h")

    result = verify_leanstral_draft(
        draft,
        _theorem(),
        native_source=NATIVE_SOURCE,
        bindings=_bindings(),
        kernel_runner=_accepting_kernel_runner,
    )
    payload = result.to_dict()

    assert result.accepted
    assert result.kernel_verification.accepted
    assert result.assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.kernel_verification.evidence.independent is True
    assert payload["model_artifact"]["assurance"] == "unverified"
    assert payload["model_artifact"]["kernel_checked"] is False
    assert payload["kernel_artifact"]["authoritative_assurance"] == "kernel_verified"
    assert payload["provenance"]["model"]["artifact_id"] == draft.artifact_id
    assert (
        payload["provenance"]["kernel"]["artifact_id"]
        == result.kernel_verification.verification_id
    )
    assert payload["provenance"]["model"]["resource_class"] == "model"
    assert payload["provenance"]["kernel"]["resource_class"] == "kernel"


def test_corrupt_persisted_model_artifact_cannot_enter_kernel() -> None:
    payload = _draft("by exact h").to_dict()
    payload["proof_text"] = "by exact forged"
    payload["draft_text"] = "by exact forged"

    with pytest.raises(ValueError, match="output digest"):
        verify_leanstral_draft(
            payload,
            _theorem(),
            native_source=NATIVE_SOURCE,
            bindings=_bindings(),
            kernel_runner=_accepting_kernel_runner,
        )


def test_canonical_source_copy_is_rejected() -> None:
    copied_line = "-- canonical implementation detail that must never be model output"
    source = copied_line + "\n" + NATIVE_SOURCE
    result = verify_leanstral_draft(
        _draft(f"by\n  {copied_line}\n  exact h"),
        _theorem(),
        native_source=NATIVE_SOURCE,
        canonical_source=source,
        bindings=_bindings(),
        kernel_runner=_accepting_kernel_runner,
    )

    assert not result.accepted
    assert result.admission.failure_code is KernelFailureCode.SOURCE_COPY


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def _patch_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "proof-gate@example.invalid")
    _git(repo, "config", "user.name", "Proof Gate")
    (repo / "allowed.txt").write_text("before\n", encoding="utf-8")
    (repo / "outside.txt").write_text("outside\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    (repo / "allowed.txt").write_text("after\n", encoding="utf-8")
    patch = _git(repo, "diff", "--", "allowed.txt")
    _git(repo, "checkout", "--", "allowed.txt")
    return repo, patch


def test_patch_is_scope_checked_apply_checked_and_validated_in_isolation(
    tmp_path: Path,
) -> None:
    repo, patch = _patch_repo(tmp_path)

    result = check_leanstral_patch_proposal(
        patch,
        model_artifact_id="leanstral-patch-1",
        repo_root=repo,
        task_declared_paths=("allowed.txt",),
        validation_commands=(
            (
                "python",
                "-c",
                "from pathlib import Path; assert Path('allowed.txt').read_text() == 'after\\n'",
            ),
        ),
    )

    assert result.accepted
    assert result.touched_paths == ("allowed.txt",)
    assert result.apply_check["passed"] is True
    assert result.validation_results[0]["passed"] is True
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert result.authoritative is False
    assert (repo / "allowed.txt").read_text(encoding="utf-8") == "before\n"
    assert result.to_dict()["can_apply"] is False


def test_patch_outside_task_scope_is_rejected_before_apply(tmp_path: Path) -> None:
    repo, patch = _patch_repo(tmp_path)

    result = check_leanstral_patch_proposal(
        patch,
        model_artifact_id="leanstral-patch-2",
        repo_root=repo,
        task_declared_paths=("outside.txt",),
    )

    assert not result.accepted
    assert result.reason_codes == ("path_outside_task_scope",)


def test_patch_policy_cannot_expand_task_declared_scope(tmp_path: Path) -> None:
    repo, patch = _patch_repo(tmp_path)

    result = check_leanstral_patch_proposal(
        patch,
        model_artifact_id="leanstral-patch-policy",
        repo_root=repo,
        task_declared_paths=("outside.txt",),
        policy=LeanstralPatchGatePolicy(allowed_paths=("allowed.txt",)),
    )

    assert result.reason_codes == ("path_outside_task_scope",)


def test_patch_must_pass_git_apply_check_and_configured_validation(
    tmp_path: Path,
) -> None:
    repo, patch = _patch_repo(tmp_path)
    malformed = patch.replace("-before", "-not-the-current-content")
    apply_result = check_leanstral_patch_proposal(
        malformed,
        model_artifact_id="leanstral-patch-3",
        repo_root=repo,
        task_declared_paths=("allowed.txt",),
    )
    validation_result = check_leanstral_patch_proposal(
        patch,
        model_artifact_id="leanstral-patch-4",
        repo_root=repo,
        task_declared_paths=("allowed.txt",),
        validation_commands=(("python", "-c", "raise SystemExit(7)"),),
    )

    assert apply_result.reason_codes == ("git_apply_check_failed",)
    assert validation_result.reason_codes == ("configured_validation_failed",)
    assert validation_result.validation_results[0]["returncode"] == 7
