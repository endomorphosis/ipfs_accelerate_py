from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.qualification import (
    PAPER_CLAIM_INCOMPLETE,
    QUALIFICATION_INCOMPLETE,
    QUALIFICATION_SUFFICIENT,
    normalize_qualification,
    paper_claim_complete,
    pending_dimensions,
    qualification_blocks_completion,
    qualification_view,
    update_qualification,
)


def _sufficient() -> dict[str, object]:
    return {
        "location": "main",
        "implementation": "present",
        "integration": "called",
        "validation": "reproduced",
        "rollout": "guarded",
        "paper_claim": {
            "mechanism": "closed recovery PlanDelta",
            "population": "autonomy wakes",
            "scope": "existing board owner",
            "limitations": "TypeSafe is not authority",
        },
    }


def test_missing_qualification_payload_does_not_block() -> None:
    view = qualification_view({})
    assert view["claimed"] is False
    assert view["blocks_completion"] is False
    assert view["completes_task"] is False
    assert qualification_blocks_completion(None) is False


def test_dimensions_update_independently() -> None:
    first = update_qualification({}, {"implementation": "present"})
    assert first["implementation"] == "present"
    assert first["integration"] == "unknown"
    assert first["validation"] == "not_run"
    second = update_qualification(first, {"integration": "adapter_only"})
    assert second["implementation"] == "present"
    assert second["integration"] == "adapter_only"
    assert second["validation"] == "not_run"
    assert "integration" in pending_dimensions(second)


def test_landing_date_cannot_replace_a_pending_dimension() -> None:
    prior = update_qualification({}, {"implementation": "present"})
    with pytest.raises(ValueError, match="landing dates"):
        update_qualification(prior, {"landing_date": "2026-10-01"})


def test_importable_is_not_called_and_reported_is_not_reproduced() -> None:
    record = normalize_qualification(
        {
            "location": "branch",
            "implementation": "present",
            "integration": "unknown",
            "validation": "reported",
            "rollout": "shadow",
        }
    )
    pending = pending_dimensions(record)
    assert "integration" in pending
    assert "validation" in pending
    assert "rollout" in pending
    assert "paper_claim" in pending
    assert qualification_blocks_completion({"qualification": record}) is True


def test_paper_claim_needs_mechanism_population_scope_and_limitations() -> None:
    claim = {
        "mechanism": "exact reuse",
        "population": "tasks",
        "scope": "runtime",
        "limitations": "",
    }
    assert paper_claim_complete(claim) is False
    view = qualification_view(
        {
            **_sufficient(),
            "paper_claim": claim,
        }
    )
    assert view["reason_code"] == PAPER_CLAIM_INCOMPLETE
    assert view["blocks_completion"] is True


def test_sufficient_qualification_is_not_task_completion() -> None:
    view = qualification_view({"qualification": _sufficient()})
    assert view["sufficient"] is True
    assert view["blocks_completion"] is False
    assert view["reason_code"] == QUALIFICATION_SUFFICIENT
    assert view["completes_task"] is False
    assert view["accepted_as_authority"] is False


def test_empty_qualification_claim_is_incomplete() -> None:
    view = qualification_view({"qualification": {}})
    assert view["claimed"] is True
    assert view["reason_code"] == QUALIFICATION_INCOMPLETE
    assert "location" in view["pending"]
    assert "implementation" in view["pending"]


def test_sealed_program_catalog_is_honest_and_not_sufficient() -> None:
    from ipfs_accelerate_py.agent_supervisor.autonomy.qualification import (
        PROGRAM_IDS,
        apply_program_catalog,
        pending_dimensions,
        program_catalog_view,
        sealed_program_catalog,
    )

    catalog = sealed_program_catalog()
    assert tuple(catalog) == PROGRAM_IDS
    for program_id, record in catalog.items():
        assert record["location"] == "branch"
        assert record["rollout"] == "off"
        assert record["completes_task"] is False
        assert pending_dimensions(record)
    view = program_catalog_view({"program_catalog": True})
    assert view["claimed"] is True
    assert view["sufficient"] is False
    assert view["blocks_completion"] is True
    assert "SAWM" in view["pending"]
    assert "PCTDD" in view["pending"]
    assert "SPAR" in view["pending"]
    assert "DOEP" in view["pending"]
    assert "ASEH" in view["pending"]

    overlay = apply_program_catalog(
        {
            "program_catalog": True,
            "programs": {
                "DOEP": {"rollout": "required", "ancestor_of_main": True},
                "SPAR-2": {"implementation": "present", "rollout": "required"},
            },
        }
    )
    assert overlay["DOEP"]["rollout"] == "required"
    assert overlay["DOEP"]["location"] == "branch"
    assert "SPAR-2" not in overlay
    assert pending_dimensions(overlay["SAWM"])
    assert "rollout" not in pending_dimensions(overlay["DOEP"])


def test_program_catalog_claim_blocks_without_collapsing_status() -> None:
    view = qualification_view({"program_catalog": True})
    assert view["claimed"] is True
    assert view["blocks_completion"] is True
    assert view["reason_code"] == QUALIFICATION_INCOMPLETE
    assert view["completes_task"] is False


def test_ancestor_of_main_is_still_branch_and_bootstrap_is_off() -> None:
    from ipfs_accelerate_py.agent_supervisor.autonomy.qualification import (
        apply_program_catalog,
        observe_location,
        observe_rollout,
        pending_dimensions,
    )

    assert (
        observe_location(
            {
                "current_branch": "chore/fmt-check-main",
                "ancestor_of_main": True,
            }
        )
        == "branch"
    )
    assert observe_location({"current_branch": "main"}) == "main"
    assert observe_location({"dirty_worktree": True, "current_branch": "main"}) == (
        "local_overlay"
    )
    assert observe_rollout({"current_rollout_mode": "bootstrap"}) == "off"
    assert observe_rollout({"current_rollout_mode": "shadow_plan"}) == "shadow"
    assert observe_rollout({"current_rollout_mode": "required"}) == "required"
    assert observe_rollout({"current_rollout_mode": "canary"}) == "off"

    catalog = apply_program_catalog(
        {
            "program_catalog": True,
            "current_branch": "main",
            "current_rollout_mode": "required",
        }
    )
    assert catalog["DOEP"]["location"] == "main"
    assert catalog["SPAR"]["rollout"] == "required"
    assert catalog["SAWM"]["rollout"] == "off"
    assert catalog["SAWM"]["implementation"] == "partial"
    assert pending_dimensions(catalog["SAWM"])
    assert pending_dimensions(catalog["DOEP"]) == ("rollout",)
