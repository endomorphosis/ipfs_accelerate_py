"""Real structural projection and lake replay in provider-free preflight.

The START tests double only the lifecycle boundary; they make no claim of a
live worker, model delivery, benchmark completion, or numerical proof.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_native_preflight as prepflight
from benchmarks.agent_supervisor.container_coding import terminal_native_start_preflight as native
from benchmarks.agent_supervisor.container_coding import terminal_context_rebind as rebind
from benchmarks.agent_supervisor.container_coding.test_terminal_native_preflight import (
    _case, _qualify, no_provider_calls,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_context import SOURCE
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical


@pytest.fixture
def numerical_case(tmp_path, monkeypatch):
    case = _case(tmp_path, "generic")
    (case["repository"] / "module.py").write_text(SOURCE)
    graph = prepflight._authored_graph
    observed = []

    def before_graph(prepared):
        assert not (case["state"] / "admission.json").exists()
        initial = json.loads((case["state"] / "initial-context-result.json").read_bytes())
        formal = initial["numerical_formalization"]
        assert numerical.validate_numerical_context(repository=case["repository"], expected_receipt=formal) == formal
        assert initial["canonical_tasks_created"] is False
        observed.append(formal["receipt_sha256"])
        return graph(prepared)

    monkeypatch.setattr(prepflight, "_authored_graph", before_graph)
    result = _qualify(case, numerical_formalization_profile=numerical.PROFILE)
    assert observed == [result["initial_context"]["numerical_formalization"]["receipt_sha256"]]
    return case, result


def test_selected_numerical_preflight_replays_real_source_and_both_lakes_before_goals(numerical_case):
    case, result = numerical_case
    observation = result["numerical_formalization_observation"]
    assert observation == prepflight.numerical_preflight_observation(state=case["state"], profile=numerical.PROFILE)
    assert result["source384_enabled"] is False
    assert "source384_timeout_seconds" not in result
    assert result["numerical_formalization_profile"] == numerical.PROFILE
    assert observation["profile"] == numerical.PROFILE
    assert observation["provider_free"] is observation["admitted_context_matches_initial"] is True
    assert observation["summary"]["families_with_artifacts"] == ["first_order", "higher_order", "program"]
    assert observation["summary"]["pending_obligation_count"] > 0
    assert observation["hydration"]["ducklake_verified"] is True
    assert observation["hydration"]["ducklake_catalogs"] == observation["hydration"]["ducklake_links"] == 3
    assert observation["live_planner_delivery_verified"] is observation["worker_delivery_verified"] is False
    assert all(observation[key] is value for key, value in numerical.AUTHORITY.items())
    assert result["qualified"] is True and result["supervisor_execution_qualified"] is False
    assert not (case["state"] / "planner-provider-request.json").exists()
    assert not (case["state"] / "planning-result.json").exists()


@pytest.mark.parametrize("mutation", ["source", "selection", "admitted", "receipt", "lake"])
def test_stale_numerical_evidence_cannot_qualify_start_observation(numerical_case, mutation):
    case, result = numerical_case
    state = case["state"]
    if mutation == "source":
        (case["repository"] / "module.py").write_text(SOURCE + "\n# edited\n")
    elif mutation == "selection":
        (state / "numerical-formalization-selection.json").write_text('{"profile":"different"}')
    elif mutation == "admitted":
        path = state / "context-result.json"
        value = json.loads(path.read_bytes())
        del value["numerical_formalization"]
        path.write_text(json.dumps(value))
    else:
        path = state / "numerical-formalization" / (
            "receipt.json" if mutation == "receipt" else "metadata-lake/metadata.ducklake")
        with path.open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError):
        prepflight.numerical_preflight_observation(state=state, profile=numerical.PROFILE)


@pytest.mark.parametrize("profile", ["unknown", "", True, {"profile": numerical.PROFILE}])
def test_unsupported_profile_is_refused_before_preparation(tmp_path, monkeypatch, profile):
    case = _case(tmp_path, "data")
    monkeypatch.setattr(prepflight.prep, "prepare", lambda **kwargs: pytest.fail("unexpected preparation"))
    with pytest.raises(ValueError, match="numerical"):
        _qualify(case, numerical_formalization_profile=profile)
    assert not case["state"].exists()


def test_legacy_default_does_not_create_or_claim_numerical_evidence(tmp_path):
    case = _case(tmp_path, "data")
    result = _qualify(case)
    assert "numerical_formalization_observation" not in result
    assert "numerical_formalization_profile" not in result
    assert "initial_context" not in result
    assert not (case["state"] / "numerical-formalization-selection.json").exists()


@pytest.mark.parametrize("changed", [False, True])
def test_native_start_replays_selected_numerical_context_and_preserves_cleanup(
        tmp_path, monkeypatch, numerical_case, changed):
    case, preparation = numerical_case
    root = tmp_path / "container"
    output = root / "state/native"
    output.parent.mkdir(parents=True)
    events = []
    monkeypatch.setattr(native, "ROOT", root)
    monkeypatch.setattr(native.os, "getuid", lambda: 1000)
    monkeypatch.setattr(native.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(native, "_workers", lambda: [])
    monkeypatch.setattr(native, "_native_diagnostics", lambda _: {"fixture_boundary": True})

    @contextmanager
    def owner(**kwargs):
        yield SimpleNamespace(server=object(), source=object())

    class Runtime:
        state = output / "launch-state"
        manifest = {"environment": {}}
        bootstrap_receipts = []
        bootstrap_errors = []
        profile = object()
        process = SimpleNamespace(snapshot=lambda _: SimpleNamespace(members=()))

        def start(self):
            events.append("start")
            # A deliberately unsuccessful doubled START ends this control-flow
            # test before any native process or actual worker could be claimed.
            return SimpleNamespace(to_dict=lambda: {"status": "failed", "fixture_boundary": True})

        def stop(self):
            events.append("stop")
            return SimpleNamespace(to_dict=lambda: {"status": "succeeded", "fixture_boundary": True})

        def close(self):
            events.append("close")

    def git_boundary(argv, **kwargs):
        assert argv == ["git", "-C", str(root / "source"), "rev-parse", "--verify", "HEAD"]
        if changed:
            path = case["state"] / "context-result.json"
            value = json.loads(path.read_bytes())
            value["numerical_formalization"] = deepcopy(value["numerical_formalization"])
            value["numerical_formalization"]["receipt_sha256"] = "0" * 64
            path.write_text(json.dumps(value))
        return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")

    monkeypatch.setattr(native, "open_existing_native_owner", owner)
    monkeypatch.setattr(rebind, "rebind_full_context", lambda **kwargs: {"context_bundle": {}})
    monkeypatch.setattr(native.AdmittedBenchmarkRuntime, "create", lambda *a, **k: Runtime())
    # Replace this module's boundary, not the shared subprocess module used by
    # real source replay (including its isolated semantic snapshots).
    monkeypatch.setattr(native, "subprocess", SimpleNamespace(run=git_boundary))
    result = native.qualify(prepared_state=case["state"], output=output, full_context=True,
        numerical_formalization_profile=numerical.PROFILE)
    assert result["qualified"] is False
    assert result["numerical_formalization_observation"] == preparation["numerical_formalization_observation"]
    assert result["full_context"]["numerical_formalization_observation"] == result["numerical_formalization_observation"]
    assert result["error_phase"] == ("numerical_context_replay" if changed else "start")
    assert events == (["stop", "close"] if changed else ["start", "stop", "close"])
    assert result["remaining_native_processes"] == 0
    assert result["workers_after_finally"] == []
    assert json.loads((output / "result.json").read_bytes()) == result


@pytest.mark.parametrize("module", [prepflight, native])
@pytest.mark.parametrize("selected", [False, True])
def test_cli_forwards_only_explicit_numerical_selection(tmp_path, monkeypatch, module, selected):
    seen = []
    monkeypatch.setattr(module, "qualify", lambda **kwargs: seen.append(kwargs) or {"qualified": True})
    arguments = (["--repository", str(tmp_path), "--instruction", str(tmp_path / "instruction"),
        "--state", str(tmp_path / "state")] if module is prepflight else
        ["--prepared-state", str(tmp_path), "--output", str(tmp_path / "output"), "--full-context"])
    if selected:
        arguments += ["--numerical-formalization-profile", numerical.PROFILE]
    monkeypatch.setattr("sys.argv", ["preflight", *arguments])
    module.main()
    assert seen[0]["numerical_formalization_profile"] == (numerical.PROFILE if selected else None)


def test_native_numerical_selection_requires_full_context_before_side_effects(tmp_path):
    with pytest.raises(ValueError, match="requires full context"):
        native.qualify(prepared_state=tmp_path / "absent", output=tmp_path / "output",
            numerical_formalization_profile=numerical.PROFILE)
    assert not (tmp_path / "output").exists()
