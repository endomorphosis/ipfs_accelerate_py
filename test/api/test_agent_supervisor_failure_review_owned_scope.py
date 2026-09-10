"""Failure feedback must use the proposal's task-owned paths, not just outputs."""

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as runtime,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_failure_review import (
    ImplementationFailureReviewReceipt,
    review_implementation_failure,
)

PRODUCTION = "src/pctdd"
OUTPUTS = ("tests/test_contract.py",)
CHANNEL_DENIAL = "validation_channel_tampering_forbidden"


def _review(tmp_path, monkeypatch, *, metadata, extra_paths=(), missing_output=False):
    task = runtime.PortalTask(
        task_id="PCTDD-005",
        title="Define prepared canonical block contracts",
        status="ready",
        completion="auto",
        priority="P0",
        track="PCTDD-G021",
        outputs=list(OUTPUTS),
        metadata=metadata,
    )
    daemon = object.__new__(runtime.PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon._implementation_scope_adjudications = {}
    monkeypatch.setattr(runtime, "completion_gap_edit_scope", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "validation_ast_companion_paths", lambda *a, **k: ())
    monkeypatch.setattr(runtime, "validation_ast_relocation_hints", lambda *a, **k: ())
    monkeypatch.setattr(
        daemon, "_authoritative_validation_environment_guidance", lambda: ""
    )
    events = []
    monkeypatch.setattr(
        daemon, "_record_event", lambda kind, body: events.append((kind, body))
    )
    monkeypatch.setattr(
        daemon,
        "_validate_implementation_patch",
        lambda *a, **k: pytest.fail("content rejection cannot authorize revalidation"),
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        lambda *a, **k: pytest.fail("content rejection cannot dispatch validation"),
    )
    # Generic normalization currently retains two path items. Exercise the
    # real sanitized caller within that budget; native tests cover the full
    # four-path PCTDD incident without this unrelated existing truncation.
    changed = [*OUTPUTS, *(extra_paths or (PRODUCTION + "/contracts.py",))]
    for name in OUTPUTS[:-1] if missing_output else OUTPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n")
    if missing_output:
        changed.remove(OUTPUTS[-1])
    rejected = {
        "passed": False,
        "attempted": False,
        "returncode": 78,
        "reason": "proposal_gate_failed",
        "error": "proposal_validation_failed",
        "proposal_gate": {
            "accepted": False,
            "reason_codes": [CHANNEL_DENIAL],
            "changed_paths": changed,
        },
    }
    result = daemon._apply_implementation_failure_review(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
        validation_result=rejected,
        proposal_validation=SimpleNamespace(accepted=False),
    )
    review = ImplementationFailureReviewReceipt.from_dict(result["failure_review"])
    assert result["passed"] is False and result["returncode"] == 78
    assert result["proposal_gate"] == rejected["proposal_gate"]
    assert not review.accepted
    assert CHANNEL_DENIAL in review.finding_codes
    assert len(events) == 1 and events[0][0] == "implementation_failure_reviewed"
    return review


@pytest.mark.parametrize("field", ["predicted files", "allowed paths"])
def test_non_scope_rejection_keeps_declared_production_paths(
    tmp_path, monkeypatch, field
):
    review = _review(tmp_path, monkeypatch, metadata={field: PRODUCTION})
    assert review.out_of_scope_paths == ()
    assert "scope_expansion_denied" not in review.reason_codes
    assert "large_or_undeclared_refactor" not in review.reason_codes
    assert (
        "Do not modify these out-of-scope paths"
        not in review.next_attempt_prompt_addendum
    )
    assert review.expected_outputs == tuple(sorted(OUTPUTS))


def test_declared_directory_does_not_cover_prefix_sibling(tmp_path, monkeypatch):
    foreign = PRODUCTION + "_foreign/other.py"
    review = _review(
        tmp_path,
        monkeypatch,
        metadata={"predicted files": PRODUCTION},
        extra_paths=(foreign,),
    )
    assert review.out_of_scope_paths == (foreign,)
    assert "scope_expansion_denied" in review.reason_codes


def test_missing_scope_does_not_infer_production_ownership(tmp_path, monkeypatch):
    review = _review(tmp_path, monkeypatch, metadata={})
    assert review.out_of_scope_paths == (PRODUCTION + "/contracts.py",)


def test_allowed_directory_does_not_replace_required_outputs(tmp_path, monkeypatch):
    review = _review(
        tmp_path,
        monkeypatch,
        metadata={"predicted files": PRODUCTION},
        missing_output=True,
    )
    assert review.out_of_scope_paths == ()
    assert review.missing_expected_outputs == (OUTPUTS[-1],)
    assert "incomplete_expected_outputs" in review.reason_codes


def test_candidate_validation_payload_cannot_nominate_owned_paths():
    foreign = "foreign/production.py"
    review = review_implementation_failure(
        task_id="PCTDD-005",
        attempt=1,
        expected_outputs=OUTPUTS,
        changed_paths=(*OUTPUTS, foreign),
        proposal_accepted=False,
        validation_result={
            "passed": False,
            "reason": "proposal_gate_failed",
            "task_owned_paths": [foreign],
            "allowed_edit_paths": [foreign],
            "proposal_gate": {"reason_codes": [CHANNEL_DENIAL]},
        },
    )
    assert review.out_of_scope_paths == (foreign,)
    assert not review.accepted and CHANNEL_DENIAL in review.finding_codes
