"""Omitted diagnostic paths cannot become scope violations or admission."""

import copy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as runtime,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_failure_review import (
    review_implementation_failure,
)
from test.api.test_agent_supervisor_failure_review_owned_scope import (
    PRODUCTION,
    _review,
)

INCOMPLETE = "diagnostic_path_evidence_incomplete"
MARKER = "[truncated original_bytes=41 sha256=" + "a" * 64 + " omitted_items=2]"


def test_four_path_sanitized_review_retains_incomplete_evidence(tmp_path, monkeypatch):
    review = _review(
        tmp_path,
        monkeypatch,
        metadata={"allowed paths": PRODUCTION},
        extra_paths=tuple(PRODUCTION + f"/{name}.py" for name in ("a", "b", "c")),
    )
    assert review.out_of_scope_paths == ()
    assert len(review.changed_paths) == 2
    assert INCOMPLETE in review.reason_codes
    assert "scope_expansion_denied" not in review.reason_codes
    assert "large_or_undeclared_refactor" not in review.reason_codes
    assert "incomplete" in review.next_attempt_prompt_addendum
    assert "truncated original_bytes=" not in review.next_attempt_prompt_addendum
    assert not review.accepted


def test_retained_real_violation_survives_four_path_truncation(tmp_path, monkeypatch):
    review = _review(
        tmp_path,
        monkeypatch,
        metadata={"allowed paths": PRODUCTION},
        extra_paths=("foreign/real.py", PRODUCTION + "/a.py", PRODUCTION + "/b.py"),
    )
    assert review.out_of_scope_paths == ("foreign/real.py",)
    assert "scope_expansion_denied" in review.reason_codes
    assert INCOMPLETE in review.reason_codes
    assert not review.accepted


def _scope_review(changed):
    validation = {
        "passed": False,
        "reason": "proposal_gate_failed",
        "proposal_gate": {"reason_codes": ["path_outside_scope"], "changed_paths": changed},
    }
    return review_implementation_failure(
        task_id="T-1",
        attempt=1,
        validation_result=validation,
        scope_adjudication={"accepted": True, "justified_paths": ["src"]},
        proposal_accepted=False,
    )


@pytest.mark.parametrize("changed", [[MARKER], ["src/a.py " + MARKER], None, {"src/a.py": True}, "src/a.py", [None], ["../foreign.py"]])
def test_spoofed_or_unsupported_paths_only_reduce_diagnostic_completeness(changed):
    review = _scope_review(changed)
    assert INCOMPLETE in review.reason_codes
    assert review.changed_paths == ()
    assert review.out_of_scope_paths == ()
    assert not review.accepted
    assert not review.proof_authoritative and not review.completion_authoritative


def test_missing_path_projection_cannot_authorize_scope_override():
    review = review_implementation_failure(
        task_id="T-1", attempt=1,
        validation_result={"passed": False, "reason": "proposal_gate_failed", "reason_codes": ["path_outside_scope"]},
        scope_adjudication={"accepted": True, "justified_paths": ["src"]},
        proposal_accepted=False,
    )
    assert INCOMPLETE in review.reason_codes
    assert not review.accepted


@pytest.mark.parametrize(
    ("proposal", "incomplete"),
    [({"changed_paths": [MARKER]}, True), ({"changed_paths": []}, False),
     ({"changed_paths": None}, True), ({}, True), ([], True), (None, True)],
)
def test_opaque_proposal_projection_does_not_fall_back_to_selection_paths(proposal, incomplete):
    review = review_implementation_failure(
        task_id="T-1", attempt=1, expected_outputs=("src",),
        proposal_accepted=False,
        validation_result={
            "passed": False,
            "proposal_gate": proposal,
            "selection": {"changed_files": ["foreign/read_only_validation.py"]},
        },
    )
    assert review.changed_paths == ()
    assert review.out_of_scope_paths == ()
    assert (INCOMPLETE in review.reason_codes) is incomplete
    assert not review.accepted


def test_path_scan_limit_retains_observed_prefix_and_denies_acceptance():
    review = _scope_review(["src/a.py"] * 128 + ["foreign/omitted.py"])
    assert review.changed_paths == ("src/a.py",)
    assert INCOMPLETE in review.reason_codes
    assert not review.accepted
    assert len(json.dumps(review.to_dict())) < 8192


def test_complete_justified_scope_review_keeps_existing_admission_policy():
    review = _scope_review(["src/a.py"])
    assert INCOMPLETE not in review.reason_codes
    assert review.accepted
    assert not review.proof_authoritative and not review.completion_authoritative


@pytest.mark.parametrize("name", ["foreign/truncated.py", "foreign/[truncated].py", "foreign/[truncated original_bytes=41 sha256=not-a-digest].py"])
def test_ordinary_pathnames_are_still_real_scope_evidence(name):
    review = _scope_review([name])
    assert review.changed_paths == (name,)
    assert review.out_of_scope_paths == (name,)
    assert INCOMPLETE not in review.reason_codes


def test_bounded_path_inspection_does_not_invoke_candidate_hooks():
    class Hostile:
        def __str__(self):
            raise AssertionError("candidate __str__ called")

        def __iter__(self):
            pytest.fail("candidate __iter__ called")

    for changed in ([Hostile()], Hostile(), ["x" * 100_000]):
        review = _scope_review(changed)
        assert INCOMPLETE in review.reason_codes
        assert not review.accepted
        assert len(json.dumps(review.to_dict())) < 8192


def test_sanitizer_budget_privacy_and_original_projection_unchanged(tmp_path):
    daemon = object.__new__(runtime.PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    original = {
        "passed": False,
        "reason": "proposal_gate_failed",
        "proposal_gate": {"changed_paths": ["src/a.py", "src/b.py", "private/c.py", "private/d.py"]},
    }
    before = copy.deepcopy(original)
    safe = daemon._sanitize_failed_validation_result(original)
    encoded = json.dumps(safe)
    assert original == before
    assert len(safe["proposal_gate"]["changed_paths"]) == 3
    assert "omitted_items=2]" in encoded
    assert "private/c.py" not in encoded and "private/d.py" not in encoded
    assert len(encoded) < 32768
    review = _scope_review(safe["proposal_gate"]["changed_paths"])
    assert review.changed_paths == ("src/a.py", "src/b.py")
    assert INCOMPLETE in review.reason_codes
    assert not review.accepted
