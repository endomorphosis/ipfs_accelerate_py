"""Provider-free planning is observed without pretending numerical model use."""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding import terminal_numerical_observation as observer
from benchmarks.agent_supervisor.container_coding.terminal_numerical_policy import PROFILE
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original  # noqa: F401


def build(original, *, full=False):
    root, instruction, state = original
    preparation.prepare(repository=root, instruction=instruction, state=state,
                        planning_mode="declared-contract@1")
    initial = preparation.initial_context(state=state, numerical_formalization_profile=PROFILE) if full else None
    plan = preparation.plan(state=state, provider_callable=lambda *a, **kw: pytest.fail("no model planning"))
    assert plan["qualified"], plan
    report = dict(arm="full" if full else "no-index", planning_mode="declared-contract@1",
                  planning=plan, provider_invocations=[], initial_context=initial)
    return root, state, report


def test_legacy_model_census_cannot_relabel_provider_free_planning(tmp_path):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    with pytest.raises(ValueError, match="model planning provenance"):
        driver.run(instruction=tmp_path / "absent-instruction", state=tmp_path / "state",
                   arm="full", planning_mode="declared-contract@1", metadata_census_only=True)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_driver_forwards_explicit_planning_mode_to_preparation(tmp_path, monkeypatch, arm):
    from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
    from benchmarks.agent_supervisor.container_coding.test_formalization_resource_profile import _simulate
    run, observed = driver.run, []

    def selected(**kwargs):
        prepare = driver.preparation.prepare

        def capture(**options):
            observed.append(options["planning_mode"])
            return prepare(**options)

        monkeypatch.setattr(driver.preparation, "prepare", capture)
        return run(**kwargs, planning_mode="declared-contract@1")

    monkeypatch.setattr(driver, "run", selected)
    report, creates, _, _, _, _ = _simulate(tmp_path, monkeypatch, arm=arm)
    assert observed == ["declared-contract@1"]
    assert report["planning_mode"] == "declared-contract@1"
    assert creates  # The optional mode still reaches the normal runtime owner.


@pytest.mark.parametrize("full", [False, True])
def test_declared_plan_observation_keeps_numerical_delivery_honest(original, full):
    _, state, report = build(original, full=full)
    result = observer.observe_numerical_planning(state=state, report=report, profile=PROFILE)
    assert result["status"] == "verified", result
    assert result["administrative_plan_verified"] is result["provider_free"] is True
    assert result["numerical_context_used_for_plan_selection"] is False
    assert result["live_planner_delivery_verified"] is result["worker_delivery_verified"] is False
    assert result["request_checked"] is result["request_hash_join_verified"] is False
    assert result["numerical_summary_count"] == int(full)
    assert result["source_obligations_discharged"] == 0
    assert result["source_semantics_verified"] is result["proof_authority"] is False
    assert not (state / "planner-provider-request.json").exists()
    assert all(value is None or type(value) in (str, bool, int) for value in result.values())


@pytest.mark.parametrize("mutation", ["mode", "provider", "calls_bool", "graph", "artifact",
                                      "receipt", "source", "signed_policy"])
def test_missing_or_conflicting_declared_route_cannot_claim_verified(original, mutation):
    root, state, report = build(original)
    report = deepcopy(report)
    if mutation == "mode":
        report["planning_mode"] = "model"
    elif mutation == "provider":
        report["provider_invocations"] = [{"phase": "planning"}]
    elif mutation == "calls_bool":
        report["planning"]["provider_calls"] = False
    elif mutation == "graph":
        report["planning"]["task_cids"] = ["foreign"]
    elif mutation == "artifact":
        (state / "planner-provider-request.json").symlink_to(state / "missing")
    elif mutation == "receipt":
        (state / "declared-contract-planning-receipt.json").write_text("{}")
    elif mutation == "source":
        (root / "bottle.py").write_text("# changed after planning\n")
    elif mutation == "signed_policy":
        p = state / "prepared.json"
        value = json.loads(p.read_text())
        value["request"]["planning_policy"]["allow_model"] = True
        p.write_text(json.dumps(value))
    # Agreement with an unsigned report alone cannot replace native replay.
    (state / "planning-result.json").write_text(json.dumps(report["planning"]))
    result = observer.observe_numerical_planning(state=state, report=report, profile=PROFILE)
    assert result["status"] != "verified", (mutation, result)
    assert result["live_planner_delivery_verified"] is result["provider_free"] is False
