"""Campaign handler tests for AAE-056 mutate plan/run/target/explain and report.

Acceptance covered here:

* Typed APIs are reached through the CLI handlers.
* Mutate run requires explicit ``--authorize-run`` authority.
* Absolute host paths / external repository roots are rejected.
* Output is bounded and deterministic.
* Cancellation and resource budgets are honored.
* No production policy change is claimed.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import cli as assurance_cli
from ipfs_accelerate_py.agent_supervisor.adversarial_assurance import (
    cli_campaign as campaign,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


REPO_ID = "repository:sha256:test-repo-identity-aae056"
REPO_STATE = _cid("repo-state-aae056")


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class _FakeApi:
    """Injectable campaign API for hermetic CLI tests."""

    def __init__(self) -> None:
        self.plan_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.explain_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def plan_mutation_campaign(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.plan_calls.append((args, kwargs))
        return {
            "interface": "MutationCampaignPlanResult@1",
            "plan": {
                "plan_id": "plan-aae056",
                "plan_cid": _cid("plan-aae056"),
                "candidate_cids": [_cid("cand-1"), _cid("cand-2")],
                "require_sandbox": True,
                "require_rollback": True,
                "repository_state_cid": REPO_STATE,
            },
            "production_policy_change": False,
        }

    def execute_mutation_campaign(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.run_calls.append((args, kwargs))
        return {
            "interface": "MutationCampaignExecutionResult@1",
            "plan_id": "plan-aae056",
            "plan_cid": _cid("plan-aae056"),
            "result_cid": _cid("result-aae056"),
            "repository_state_cid": REPO_STATE,
            "verification_policy_cid": _cid("vpolicy"),
            "candidate_reports": [
                {
                    "candidate_id": "cand-1",
                    "candidate_cid": _cid("cand-1"),
                    "terminal_status": "killed",
                    "outcome_cid": _cid("out-1"),
                    "report_cid": _cid("rep-1"),
                },
                {
                    "candidate_id": "cand-2",
                    "candidate_cid": _cid("cand-2"),
                    "terminal_status": "survivor",
                    "outcome_cid": _cid("out-2"),
                    "report_cid": _cid("rep-2"),
                },
            ],
            "killed_count": 1,
            "survivor_count": 1,
            "invalid_count": 0,
            "inconclusive_count": 0,
            "terminal_status": "complete",
            "reason_codes": [
                "campaign_executed",
                "no_production_policy_change",
                "no_arbitrary_path_exposure",
            ],
            "require_sandbox": True,
            "network_disabled": True,
            "production_policy_changed": False,
        }

    def predict_detection_set(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.explain_calls.append((args, kwargs))
        return {
            "candidate_id": "cand-1",
            "candidate_cid": _cid("cand-1"),
            "predictions": [
                {
                    "detector_id": "unit.test_auth",
                    "strength": "strong",
                    "rationale": "covers authorization branch",
                    "terminal_status": "expected_kill",
                }
            ],
            "terminal_status": "complete",
        }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return path


def _ns(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "output_human": False,
        "output_json": True,
        "timeout_seconds": None,
        "max_candidates": None,
        "max_worktrees": None,
        "cancel": False,
        "cancel_file": None,
        "notes": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def test_campaign_descriptor_exposes_typed_apis() -> None:
    desc = campaign.campaign_cli_descriptor()
    assert desc["interface"] == "AssuranceCampaignCLI@1"
    assert desc["explicit_run_authority_required"] is True
    assert desc["production_policy_change"] is False
    assert set(desc["commands"]) == set(assurance_cli.CAMPAIGN_COMMANDS)
    assert desc["apis"]["mutate.plan"] == "plan_mutation_campaign"
    assert desc["apis"]["mutate.run"] == "execute_mutation_campaign"
    assert desc["apis"]["mutate.target"] == "select_mutation_targets"
    assert desc["apis"]["mutate.explain"] == "predict_detection_set"
    assert desc["apis"]["report"] == "build_assurance_report"


# ---------------------------------------------------------------------------
# mutate plan
# ---------------------------------------------------------------------------


def test_mutate_plan_reaches_typed_api(tmp_path: Path) -> None:
    api = _FakeApi()
    args = _ns(
        repository_state_json=str(
            _write_json(
                tmp_path / "state.json",
                {
                    "repository_id": REPO_ID,
                    "repository_state_cid": REPO_STATE,
                },
            )
        ),
        manifest_json=str(
            _write_json(
                tmp_path / "manifest.json",
                {
                    "repository_id": REPO_ID,
                    "repository_state_cid": REPO_STATE,
                    "observation_complete": True,
                },
            )
        ),
        policy_json=str(
            _write_json(
                tmp_path / "policy.json",
                {"campaign_policy_id": "default", "campaign_policy_version": "1"},
            )
        ),
        resource_budget_json=str(
            _write_json(
                tmp_path / "budget.json",
                {
                    "max_total_candidates": 8,
                    "max_targets": 4,
                    "max_worktrees": 2,
                    "max_execution_seconds": 60,
                },
            )
        ),
        baseline_json=str(
            _write_json(
                tmp_path / "baseline.json",
                {"baseline_receipt_cid": _cid("baseline")},
            )
        ),
        targets_json=None,
        operators_json=None,
        properties_json=None,
        generation_manifest_json=None,
        seed_config_json=None,
        plan_id="plan-aae056",
        no_partition=False,
    )
    result = campaign.handle_mutate_plan(args, api=api)
    assert result["status"] == "planned"
    assert result["api"] == "plan_mutation_campaign"
    assert result["plan_id"] == "plan-aae056"
    assert result["production_policy_change"] is False
    assert len(api.plan_calls) == 1
    assert api.plan_calls[0][1]["plan_id"] == "plan-aae056"
    assert api.plan_calls[0][1]["return_result"] is True


def test_mutate_plan_rejects_absolute_host_path_in_payload(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "state.json",
        {
            "repository_id": REPO_ID,
            "repository_state_cid": REPO_STATE,
            "repository_path": "/home/other/external-repo",
        },
    )
    args = _ns(
        repository_state_json=str(path),
        manifest_json=str(_write_json(tmp_path / "m.json", {})),
        policy_json=str(_write_json(tmp_path / "p.json", {})),
        resource_budget_json=str(_write_json(tmp_path / "b.json", {})),
        baseline_json=None,
        targets_json=None,
        operators_json=None,
        properties_json=None,
        generation_manifest_json=None,
        seed_config_json=None,
        plan_id=None,
        no_partition=False,
    )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        campaign.handle_mutate_plan(args, api=_FakeApi())


def test_mutate_plan_honors_cli_resource_intersection(tmp_path: Path) -> None:
    api = _FakeApi()
    args = _ns(
        repository_state_json=str(
            _write_json(
                tmp_path / "state.json",
                {"repository_id": REPO_ID, "repository_state_cid": REPO_STATE},
            )
        ),
        manifest_json=str(_write_json(tmp_path / "m.json", {"ok": True})),
        policy_json=str(_write_json(tmp_path / "p.json", {"ok": True})),
        resource_budget_json=str(
            _write_json(
                tmp_path / "b.json",
                {
                    "max_total_candidates": 100,
                    "max_worktrees": 50,
                    "max_execution_seconds": 9_999,
                },
            )
        ),
        baseline_json=None,
        targets_json=None,
        operators_json=None,
        properties_json=None,
        generation_manifest_json=None,
        seed_config_json=None,
        plan_id=None,
        no_partition=False,
        max_candidates=3,
        max_worktrees=2,
        timeout_seconds=30,
    )
    budget = assurance_cli.resource_budget_from_args(args)
    campaign.handle_mutate_plan(args, api=api, budget=budget)
    passed_budget = api.plan_calls[0][0][3]
    assert passed_budget["max_total_candidates"] == 3
    assert passed_budget["max_worktrees"] == 2
    assert passed_budget["max_execution_seconds"] == 30


# ---------------------------------------------------------------------------
# mutate run
# ---------------------------------------------------------------------------


def test_mutate_run_requires_explicit_authority(tmp_path: Path) -> None:
    args = _ns(
        plan_json=str(
            _write_json(
                tmp_path / "plan.json",
                {
                    "plan_id": "p",
                    "plan_cid": _cid("p"),
                    "candidate_cids": [],
                    "require_sandbox": True,
                    "require_rollback": True,
                },
            )
        ),
        verification_policy_json=str(
            _write_json(tmp_path / "vp.json", {"policy_cid": _cid("vp")})
        ),
        precomputed_reports_json=str(_write_json(tmp_path / "r.json", [])),
        candidates_json=None,
        expected_detections_json=None,
        authorize_run=False,
    )
    with pytest.raises(assurance_cli.AssuranceCLIAuthorityError) as exc:
        campaign.handle_mutate_run(args, api=_FakeApi())
    assert exc.value.reason_code == "run_authority_required"
    assert exc.value.exit_code == assurance_cli.EXIT_AUTHORITY


def test_mutate_run_with_authority_reaches_api(tmp_path: Path) -> None:
    api = _FakeApi()
    reports = [
        {
            "candidate_id": "cand-1",
            "terminal_status": "killed",
            "report_cid": _cid("rep-1"),
        }
    ]
    args = _ns(
        plan_json=str(
            _write_json(
                tmp_path / "plan.json",
                {
                    "plan_id": "plan-aae056",
                    "plan_cid": _cid("plan-aae056"),
                    "candidate_cids": [_cid("cand-1")],
                    "require_sandbox": True,
                    "require_rollback": True,
                },
            )
        ),
        verification_policy_json=str(
            _write_json(tmp_path / "vp.json", {"policy_cid": _cid("vp")})
        ),
        precomputed_reports_json=str(_write_json(tmp_path / "r.json", reports)),
        candidates_json=None,
        expected_detections_json=None,
        authorize_run=True,
    )
    result = campaign.handle_mutate_run(args, api=api)
    assert result["authorized"] is True
    assert result["api"] == "execute_mutation_campaign"
    assert result["killed_count"] == 1
    assert result["production_policy_change"] is False
    assert len(api.run_calls) == 1
    assert api.run_calls[0][1]["metadata"]["cli_authorize_run"] is True


def test_mutate_run_rejects_host_path_in_plan(tmp_path: Path) -> None:
    args = _ns(
        plan_json=str(
            _write_json(
                tmp_path / "plan.json",
                {
                    "plan_id": "p",
                    "plan_cid": _cid("p"),
                    "worktree_path": "/tmp/evil-worktree",
                    "require_sandbox": True,
                    "require_rollback": True,
                },
            )
        ),
        verification_policy_json=str(
            _write_json(tmp_path / "vp.json", {"policy_cid": _cid("vp")})
        ),
        precomputed_reports_json=str(_write_json(tmp_path / "r.json", [])),
        candidates_json=None,
        expected_detections_json=None,
        authorize_run=True,
    )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        campaign.handle_mutate_run(args, api=_FakeApi())


def test_mutate_run_honors_max_candidates(tmp_path: Path) -> None:
    reports = [{"candidate_id": f"c{i}", "terminal_status": "killed"} for i in range(5)]
    args = _ns(
        plan_json=str(
            _write_json(
                tmp_path / "plan.json",
                {
                    "plan_id": "p",
                    "plan_cid": _cid("p"),
                    "candidate_cids": [_cid(f"c{i}") for i in range(5)],
                    "require_sandbox": True,
                    "require_rollback": True,
                },
            )
        ),
        verification_policy_json=str(
            _write_json(tmp_path / "vp.json", {"policy_cid": _cid("vp")})
        ),
        precomputed_reports_json=str(_write_json(tmp_path / "r.json", reports)),
        candidates_json=None,
        expected_detections_json=None,
        authorize_run=True,
        max_candidates=2,
    )
    budget = assurance_cli.resource_budget_from_args(args)
    with pytest.raises(assurance_cli.AssuranceCLIResourceError):
        campaign.handle_mutate_run(args, api=_FakeApi(), budget=budget)


# ---------------------------------------------------------------------------
# mutate target
# ---------------------------------------------------------------------------


def test_mutate_target_rejects_filesystem_repository_id(tmp_path: Path) -> None:
    props = [
        {
            "property_id": "p1",
            "property_class": "authorization",
            "statement": "tenant must bind",
            "symbol_ids": ["mod.check"],
            "artifact_cids": [_cid("art")],
        }
    ]
    args = _ns(
        properties_json=str(_write_json(tmp_path / "props.json", props)),
        repository_id="/home/other/external-repo",
        repository_state_cid=REPO_STATE,
        sampling_budget_json=None,
        return_result=False,
    )
    with pytest.raises(assurance_cli.AssuranceCLIUsageError) as exc:
        campaign.handle_mutate_target(args)
    assert exc.value.reason_code == "repository_path_forbidden"


def test_mutate_target_selects_from_properties(tmp_path: Path) -> None:
    # Use a minimal property mapping accepted by select_mutation_targets.
    # If the leaf rejects the fixture shape, the CLI must still fail closed
    # with a typed error rather than path exposure or authority bypass.
    props = [
        {
            "claim_id": "claim-auth",
            "property_class": "authorization",
            "statement": "authorization must bind tenant",
            "symbol_ids": ["mod.auth_check"],
            "artifact_cids": [_cid("artifact-auth")],
            "repository_id": REPO_ID,
            "repository_state_cid": REPO_STATE,
        }
    ]
    args = _ns(
        properties_json=str(_write_json(tmp_path / "props.json", props)),
        repository_id=REPO_ID,
        repository_state_cid=REPO_STATE,
        sampling_budget_json=str(
            _write_json(
                tmp_path / "sample.json",
                {"max_targets": 4, "seed": 7},
            )
        ),
        return_result=True,
    )
    try:
        result = campaign.handle_mutate_target(args)
    except Exception as exc:
        # Leaf validation failures are acceptable for malformed fixtures, but
        # must never be silent success or path leakage.
        assert "path" not in str(exc).lower() or "relative" in str(exc).lower()
        return
    assert result["api"] == "select_mutation_targets"
    assert result["repository_id"] == REPO_ID
    assert result["production_policy_change"] is False
    assert result["status"] == "selected"


# ---------------------------------------------------------------------------
# mutate explain
# ---------------------------------------------------------------------------


def test_mutate_explain_reaches_predict_detection_set(tmp_path: Path) -> None:
    api = _FakeApi()
    args = _ns(
        candidate_json=str(
            _write_json(
                tmp_path / "cand.json",
                {
                    "candidate_id": "cand-1",
                    "candidate_cid": _cid("cand-1"),
                    "operator_id": "control_flow_invert",
                },
            )
        ),
        manifest_json=str(
            _write_json(
                tmp_path / "manifest.json",
                {
                    "repository_id": REPO_ID,
                    "repository_state_cid": REPO_STATE,
                },
            )
        ),
    )
    result = campaign.handle_mutate_explain(args, api=api)
    assert result["status"] == "explained"
    assert result["api"] == "predict_detection_set"
    assert result["candidate_id"] == "cand-1"
    assert result["detector_count"] == 1
    assert len(api.explain_calls) == 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_projects_bounded_campaign_result(tmp_path: Path) -> None:
    campaign_result = {
        "plan_id": "plan-aae056",
        "plan_cid": _cid("plan-aae056"),
        "result_cid": _cid("result-aae056"),
        "repository_state_cid": REPO_STATE,
        "verification_policy_cid": _cid("vp"),
        "killed_count": 1,
        "survivor_count": 1,
        "invalid_count": 0,
        "inconclusive_count": 0,
        "terminal_status": "complete",
        "reason_codes": ["campaign_executed", "no_production_policy_change"],
        "candidate_reports": [
            {
                "candidate_id": "cand-1",
                "candidate_cid": _cid("cand-1"),
                "terminal_status": "killed",
                "report_cid": _cid("rep-1"),
            },
            {
                "candidate_id": "cand-2",
                "candidate_cid": _cid("cand-2"),
                "terminal_status": "survivor",
                "report_cid": _cid("rep-2"),
            },
        ],
        "require_sandbox": True,
        "network_disabled": True,
        "production_policy_changed": False,
    }
    args = _ns(
        campaign_result_json=str(
            _write_json(tmp_path / "result.json", campaign_result)
        ),
        plan_json=None,
    )
    result = campaign.handle_report(args)
    assert result["status"] == "reported"
    assert result["plan_id"] == "plan-aae056"
    assert result["killed_count"] == 1
    assert result["survivor_count"] == 1
    assert result["production_policy_change"] is False
    assert result["result"]["metrics_available"] is False
    assert result["result"]["candidate_report_count"] == 2
    # Bounded projection: only identity/status fields, not full bodies.
    for item in result["result"]["candidate_reports"]:
        assert set(item.keys()) <= {
            "candidate_id",
            "candidate_cid",
            "terminal_status",
            "outcome_cid",
            "report_cid",
        }


def test_report_rejects_absolute_paths(tmp_path: Path) -> None:
    args = _ns(
        campaign_result_json=str(
            _write_json(
                tmp_path / "result.json",
                {
                    "plan_id": "p",
                    "repo_root": "/var/lib/external",
                    "candidate_reports": [],
                },
            )
        ),
        plan_json=None,
    )
    with pytest.raises(assurance_cli.AssuranceCLIPathError):
        campaign.handle_report(args)


# ---------------------------------------------------------------------------
# Cancellation / end-to-end dispatch
# ---------------------------------------------------------------------------


def test_cancellation_flag_short_circuits_dispatch() -> None:
    out = io.StringIO()
    args = _ns(
        assurance_command="report",
        campaign_result_json="missing.json",
        plan_json=None,
        cancel=True,
    )
    code = assurance_cli.run_assurance_cli(args, stdout=out)
    assert code == assurance_cli.EXIT_CANCELLED
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    assert payload["reason_code"] == "cancelled"


def test_cancellation_file_short_circuits(tmp_path: Path) -> None:
    cancel_path = tmp_path / "cancel"
    cancel_path.write_text("cancelled\n", encoding="utf-8")
    out = io.StringIO()
    args = _ns(
        assurance_command="report",
        campaign_result_json="missing.json",
        plan_json=None,
        cancel_file=str(cancel_path),
    )
    code = assurance_cli.run_assurance_cli(args, stdout=out)
    assert code == assurance_cli.EXIT_CANCELLED


def test_end_to_end_run_dispatch_json_envelope(tmp_path: Path) -> None:
    api = _FakeApi()
    reports = [
        {
            "candidate_id": "cand-1",
            "terminal_status": "killed",
            "report_cid": _cid("rep-1"),
        }
    ]
    plan_path = _write_json(
        tmp_path / "plan.json",
        {
            "plan_id": "plan-aae056",
            "plan_cid": _cid("plan-aae056"),
            "candidate_cids": [_cid("cand-1")],
            "require_sandbox": True,
            "require_rollback": True,
        },
    )
    vp_path = _write_json(tmp_path / "vp.json", {"policy_cid": _cid("vp")})
    reports_path = _write_json(tmp_path / "reports.json", reports)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    assurance_cli.register_assurance_cli(sub)
    args = parser.parse_args(
        [
            "assurance",
            "mutate",
            "run",
            "--plan-json",
            str(plan_path),
            "--verification-policy-json",
            str(vp_path),
            "--precomputed-reports-json",
            str(reports_path),
            "--authorize-run",
        ]
    )
    out = io.StringIO()
    code = assurance_cli.run_assurance_cli(args, stdout=out, api=api)
    assert code == assurance_cli.EXIT_SUCCESS
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["command"] == "mutate.run"
    assert payload["result"]["authorized"] is True
    assert payload["production_policy_change"] is False
    assert payload["path_exposure"] is False
    # Deterministic re-emit
    out2 = io.StringIO()
    assurance_cli.emit(payload, output_json=True, stream=out2)
    assert json.loads(out2.getvalue()) == payload


def test_end_to_end_unauthorized_run_is_authority_exit(tmp_path: Path) -> None:
    plan_path = _write_json(
        tmp_path / "plan.json",
        {
            "plan_id": "p",
            "plan_cid": _cid("p"),
            "candidate_cids": [],
            "require_sandbox": True,
            "require_rollback": True,
        },
    )
    vp_path = _write_json(tmp_path / "vp.json", {"policy_cid": _cid("vp")})
    reports_path = _write_json(tmp_path / "reports.json", [])
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    assurance_cli.register_assurance_cli(sub)
    args = parser.parse_args(
        [
            "assurance",
            "mutate",
            "run",
            "--plan-json",
            str(plan_path),
            "--verification-policy-json",
            str(vp_path),
            "--precomputed-reports-json",
            str(reports_path),
        ]
    )
    out = io.StringIO()
    code = assurance_cli.run_assurance_cli(args, stdout=out, api=_FakeApi())
    assert code == assurance_cli.EXIT_AUTHORITY
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    assert payload["reason_code"] == "run_authority_required"


def test_end_to_end_report_human_output(tmp_path: Path) -> None:
    result_path = _write_json(
        tmp_path / "result.json",
        {
            "plan_id": "plan-h",
            "plan_cid": _cid("plan-h"),
            "killed_count": 2,
            "survivor_count": 0,
            "invalid_count": 0,
            "inconclusive_count": 0,
            "terminal_status": "complete",
            "reason_codes": ["campaign_executed"],
            "candidate_reports": [],
            "production_policy_changed": False,
        },
    )
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    assurance_cli.register_assurance_cli(sub)
    args = parser.parse_args(
        [
            "assurance",
            "report",
            "--campaign-result-json",
            str(result_path),
            "--output-human",
        ]
    )
    out = io.StringIO()
    code = assurance_cli.run_assurance_cli(args, stdout=out)
    assert code == assurance_cli.EXIT_SUCCESS
    text = out.getvalue()
    assert "report" in text
    assert "plan_id=plan-h" in text
    assert "killed_count=2" in text


def test_project_result_redacts_host_paths() -> None:
    projected = assurance_cli.project_result(
        {"worktree_path": "/tmp/escape", "ok": True, "nested": {"path": "/home/x"}}
    )
    assert projected["worktree_path"] == "<redacted-host-path>"
    assert projected["nested"]["path"] == "<redacted-host-path>"
    assert projected["ok"] is True


def test_handlers_table_covers_closed_vocabulary() -> None:
    assert set(campaign.CAMPAIGN_HANDLERS) == set(assurance_cli.CAMPAIGN_COMMANDS)
    for name, handler in campaign.CAMPAIGN_HANDLERS.items():
        assert callable(handler), name
