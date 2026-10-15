"""Durable attempt lifecycle around real native checks and authored providers.

Provider calls and usage in this suite are authored; none establishes model
quality, provider ingestion, total workflow usage, billing or completion.
"""
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_attempt_runner as attempt_runner
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_attempt_journal import (
    CONTROL_FIELDS, START_SCHEMA, RepositoryWorldAttemptJournal as Journal,
)
from test.api.test_repository_world_dispatch import (
    TASK, dispatch_case, world_session, required, scheduler,  # noqa: F401
)


def make_journal(tmp_path, *, workspace=None):
    workspace = workspace or tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    root = tmp_path / "journal"
    root.mkdir(mode=0o700)
    start = dict(schema=START_SCHEMA, attempt_id="case:baseline:1", case_id="arithmetic",
        arm="baseline", parent_attempt_id=None,
        observed_controls=dict.fromkeys(CONTROL_FIELDS, None))
    return Journal(root, workspace=workspace, start=start), root


def options(case, **changes):
    return dict(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
        max_output_tokens=128, world_task_packet=case[2]) | changes


def minimal_receipt():
    return dict(schema="router-implementation-invocation@1", invocation_id="a" * 32,
        provider="codex_cli", model="pinned", purpose="coding", status="provider_returned",
        world_task_packet=dict(validation_count=1, dispatch_rechecked=True, provider_attempted=True),
        usage={"prompt_tokens": 37}, native_rollout_usage=None, completion_authority=False)


def read(root, name):
    return json.loads((root / name).read_bytes())


def test_start_exists_before_runner_and_terminal_follows_exact_receipt(tmp_path, monkeypatch):
    journal, root = make_journal(tmp_path)
    actual = minimal_receipt()
    def authored(**kwargs):
        assert kwargs == {"literal": "unchanged", "world_task_attempt_journal": journal}
        assert sorted(path.name for path in root.iterdir()) == ["start.json"]
        Journal.retain_receipt(journal, actual)
        assert not (root / "terminal.json").exists()
        return "unchanged output", actual
    monkeypatch.setattr(runner, "run", authored)
    output, receipt, diagnostic = attempt_runner.run_repository_world_packet_attempt(
        journal=journal, runner_options={"literal": "unchanged"})
    assert output == "unchanged output" and receipt is actual
    assert read(root, "receipt.json") == actual
    final = read(root, "terminal.json")
    assert final["outcome"] == "returned" and final["error_type"] is None
    assert final["receipt_sha256"] == diagnostic["journal"]["receipt"]["sha256"]
    assert diagnostic["journal"]["closed"] is True
    assert diagnostic["terminal_finish_failed"] is diagnostic["close_failed"] is False
    assert diagnostic["completion_authority"] is diagnostic["billing_verified"] is False
    actual["usage"]["prompt_tokens"] = 999
    assert read(root, "receipt.json")["usage"]["prompt_tokens"] == 37


@pytest.mark.parametrize("bad", [None, [], {"world_task_attempt_journal": None}])
def test_invalid_wrapper_options_keep_begin_and_terminal_without_inventing_receipt(tmp_path, monkeypatch, bad):
    journal, root = make_journal(tmp_path)
    monkeypatch.setattr(runner, "run", lambda **_kw: pytest.fail("invalid wrapper options dispatched"))
    with pytest.raises(ValueError):
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=bad)
    assert read(root, "terminal.json")["outcome"] == "raised"
    assert read(root, "terminal.json")["receipt_sha256"] is None
    assert not (root / "receipt.json").exists()
    assert Journal.receipt(journal)["closed"] is True


@pytest.mark.parametrize("failure", [RuntimeError("authored provider failure"), KeyboardInterrupt()])
def test_same_primary_error_survives_terminal_and_close_failures(tmp_path, monkeypatch, failure):
    journal, root = make_journal(tmp_path)
    real_close = Journal.close
    def authored(**_kwargs): raise failure
    def fail_finish(*_args, **_kwargs): raise OSError("authored terminal failure")
    def fail_close(*_args, **_kwargs): raise OSError("authored close failure")
    monkeypatch.setattr(runner, "run", authored)
    monkeypatch.setattr(Journal, "finish", fail_finish)
    monkeypatch.setattr(Journal, "close", fail_close)
    try:
        with pytest.raises(type(failure)) as caught:
            attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options={})
        assert caught.value is failure
        assert sorted(path.name for path in root.iterdir()) == ["start.json"]
    finally:
        real_close(journal)


def test_success_survives_terminal_close_and_diagnostic_failures(tmp_path, monkeypatch):
    journal, _root = make_journal(tmp_path)
    actual = minimal_receipt()
    real_close = Journal.close
    def authored(**_kwargs): return "provider success", actual
    def fail(*_args, **_kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(runner, "run", authored)
    monkeypatch.setattr(Journal, "finish", fail)
    monkeypatch.setattr(Journal, "close", fail)
    monkeypatch.setattr(Journal, "receipt", fail)
    try:
        output, receipt, diagnostic = attempt_runner.run_repository_world_packet_attempt(
            journal=journal, runner_options={})
        assert output == "provider success" and receipt is actual
        assert diagnostic["terminal_finish_failed"] is diagnostic["close_failed"] is True
        assert diagnostic["journal_diagnostic_unavailable"] is True and diagnostic["journal"] is None
    finally:
        real_close(journal)


def test_initial_root_drift_closes_valid_owner_and_preserves_refusal(tmp_path):
    journal, root = make_journal(tmp_path)
    old = tmp_path / "old-journal"
    root.rename(old)
    root.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="root_changed"):
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options={})
    assert Journal.receipt(journal)["closed"] is True
    assert sorted(path.name for path in old.iterdir()) == ["start.json"]
    assert list(root.iterdir()) == []


def test_real_router_captures_complete_accounting_before_wrapper_terminal(dispatch_case, tmp_path):
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    output, receipt, diagnostic = attempt_runner.run_repository_world_packet_attempt(
        journal=journal, runner_options=options(dispatch_case))
    retained = read(root, "receipt.json")
    assert retained == receipt
    assert retained["usage"]["prompt_tokens"] == 15
    assert retained["world_task_packet"]["provider_attempted"] is True
    assert "native_rollout_usage" in retained and "context_delivery_observation" in retained
    assert output.startswith("Authored") and len(dispatch_case[3]["seen"]) == 1
    assert read(root, "terminal.json")["outcome"] == "returned"
    assert diagnostic["journal"]["capture_error"] is False
    assert "world_task_attempt_journal" not in receipt


def test_failed_provider_receipt_is_durable_without_stdout_extraction(dispatch_case, tmp_path):
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    failure = TimeoutError("authored provider timeout")
    def fail(_prompt, _options): raise failure
    dispatch_case[3]["callback"] = fail
    with pytest.raises(TimeoutError) as caught:
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=options(dispatch_case))
    assert caught.value is failure
    receipt = read(root, "receipt.json")
    assert receipt["status"] == "failed" and receipt["failure_phase"] == "provider_invocation"
    assert receipt["world_task_packet"]["provider_attempted"] is True
    assert receipt["usage"]["prompt_tokens"] == 15
    assert read(root, "terminal.json")["outcome"] == "raised"


@pytest.mark.parametrize("provider_failure", [False, True])
def test_capture_failure_does_not_replace_success_or_original_error(dispatch_case, tmp_path, monkeypatch, provider_failure):
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    def failed_capture(*_args, **_kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(Journal, "retain_receipt", failed_capture)
    if provider_failure:
        failure = RuntimeError("authored provider failure")
        def fail(_prompt, _options): raise failure
        dispatch_case[3]["callback"] = fail
        with pytest.raises(RuntimeError) as caught:
            attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=options(dispatch_case))
        assert caught.value is failure
    else:
        output, receipt, diagnostic = attempt_runner.run_repository_world_packet_attempt(
            journal=journal, runner_options=options(dispatch_case))
        assert output.startswith("Authored") and receipt["status"] == "provider_returned"
        assert diagnostic["journal"]["capture_error"] is True
    assert Journal.receipt(journal)["capture_error"] is True
    assert read(root, "terminal.json")["receipt_sha256"] is None
    assert not (root / "receipt.json").exists()


def test_accounting_receipt_precedes_a_stdout_failure(dispatch_case, tmp_path, monkeypatch):
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    failure = OSError("authored broken output stream")
    def failed_print(*_args, **_kwargs): raise failure
    # The unchanged runner restores its environment after printing. Track both
    # values so this deliberately broken stream cannot leak fixture state.
    monkeypatch.setenv("ipfs_accelerate_py_CODEX_SANDBOX", "workspace-write")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(runner, "print", failed_print, raising=False)
    with pytest.raises(OSError) as caught:
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=options(dispatch_case))
    assert caught.value is failure
    retained = read(root, "receipt.json")
    assert retained["status"] == "provider_returned" and retained["usage"]["prompt_tokens"] == 15
    assert read(root, "terminal.json")["outcome"] == "raised"


def test_workspace_refusal_keeps_unknown_receipt_and_declared_slot(dispatch_case, tmp_path, monkeypatch):
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    other = tmp_path / "wrong-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(subprocess.CalledProcessError):
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=options(dispatch_case))
    assert not (root / "receipt.json").exists()
    assert read(root, "terminal.json")["receipt_sha256"] is None
    assert read(root, "start.json")["attempt_id"] == "case:baseline:1"
    assert dispatch_case[3]["seen"] == []


@pytest.mark.parametrize("changes", [{"world_task_packet": None}, {"provider": "grok_cli"},
    {"purpose": "planning"}, {"world_packet_artifact_dir": Path("/unused")}])
def test_journal_never_grants_another_route_or_original_only_retention(dispatch_case, tmp_path, monkeypatch, changes):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    def forbidden(*_args, **_kwargs): pytest.fail("unsupported journal route reached discovery")
    monkeypatch.setattr(intelligence_index, "discover_available_providers", forbidden)
    with pytest.raises(ValueError):
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=options(dispatch_case, **changes))
    assert not (root / "receipt.json").exists() and dispatch_case[3]["seen"] == []


def test_provider_initialization_journal_drift_refuses_before_dispatch(dispatch_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    def init(*_args, **_kwargs):
        (root / "start.json").unlink()
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", init)
    with pytest.raises((ValueError, OSError)):
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=options(dispatch_case))
    assert dispatch_case[3]["seen"] == []
    assert Journal.receipt(journal)["closed"] is True
    assert not (root / "receipt.json").exists()


def test_journal_recheck_is_before_final_native_source_fences(dispatch_case, tmp_path, monkeypatch):
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    real_require = Journal.require_open
    calls = []
    def recheck(owner, **kwargs):
        real_require(owner, **kwargs)
        calls.append(owner)
        if len(calls) == 3:
            # Wrapper entry, router entry, then post-initialization journal
            # fence: source changes here must still hit native closing checks.
            (dispatch_case[0] / "calc.py").write_text("def add(value):\n    return value + 99\n")
    monkeypatch.setattr(Journal, "require_open", recheck)
    with pytest.raises(ValueError):
        attempt_runner.run_repository_world_packet_attempt(journal=journal, runner_options=options(dispatch_case))
    assert len(calls) >= 3 and dispatch_case[3]["seen"] == []
    retained = read(root, "receipt.json")
    assert retained["world_task_packet"]["provider_attempted"] is False
    assert retained["failure_phase"] == "world_task_packet_currentness"


def test_journal_for_a_foreign_workspace_refuses_before_provider(dispatch_case, tmp_path):
    foreign = tmp_path / "other-workspace"
    journal, root = make_journal(tmp_path, workspace=foreign)
    with pytest.raises(ValueError, match="workspace"):
        attempt_runner.run_repository_world_packet_attempt(journal=journal,
            runner_options=options(dispatch_case))
    assert dispatch_case[3]["seen"] == []
    retained = read(root, "receipt.json")
    assert retained["world_task_packet"]["provider_attempted"] is False
    assert retained["failure_phase"] == "world_task_attempt_journal"


def test_presented_receipt_retains_both_original_and_actual_input_bindings(dispatch_case, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_presentation import (
        prepare_repository_world_packet_presentation,
    )
    journal, root = make_journal(tmp_path, workspace=dispatch_case[0])
    presentation = prepare_repository_world_packet_presentation(dispatch_case[2].prompt)
    _, actual, _ = attempt_runner.run_repository_world_packet_attempt(journal=journal,
        runner_options=options(dispatch_case, world_task_presentation=presentation))
    retained = read(root, "receipt.json")
    assert retained == actual
    assert retained["world_task_presentation"]["dispatch_rechecked"] is True
    assert retained["world_task_presentation"]["sidecar_sha256"] == presentation.receipt()["sidecar_sha256"]
    assert retained["world_task_packet"]["provider_attempted"] is True


def test_default_none_does_not_touch_journal_or_add_receipt_fields(dispatch_case, monkeypatch):
    def forbidden(*_args, **_kwargs): pytest.fail("default route touched attempt journal")
    monkeypatch.setattr(Journal, "require_open", forbidden)
    monkeypatch.setattr(Journal, "retain_receipt", forbidden)
    _, omitted = runner.run(**options(dispatch_case))
    _, explicit = runner.run(**options(dispatch_case), world_task_attempt_journal=None)
    assert set(omitted) == set(explicit)
    assert "world_task_attempt_journal" not in explicit
    assert omitted["model_prompt_sha256"] == explicit["model_prompt_sha256"]


def test_unissued_journal_is_not_a_capture_or_provider_capability(dispatch_case, monkeypatch):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    clone = object.__new__(Journal)
    def forbidden(*_args, **_kwargs): pytest.fail("unissued journal reached provider discovery")
    monkeypatch.setattr(intelligence_index, "discover_available_providers", forbidden)
    with pytest.raises(ValueError, match="exact_issued_required"):
        runner.run(**options(dispatch_case), world_task_attempt_journal=clone)
    assert dispatch_case[3]["seen"] == []


def test_readonly_guard_precedes_any_journal_handling(dispatch_case, required, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import (
        prepare_repository_world_packet, repository_world_task_binding,
    )
    from test.api.test_repository_context_compiler import CORE, compile_context
    adapter = dispatch_case[2]._adapter
    effects, paths = ("repository_read",), ("calc.py",)
    compilation = compile_context(adapter, required, authority={**CORE["authority"],
        "world_task_packet": repository_world_task_binding("task:arithmetic", TASK, effects, paths)})
    readonly = prepare_repository_world_packet(adapter=adapter, compilation=compilation, generation=1,
        task_id="task:arithmetic", task_text=TASK, unresolved_obligations=("Observe only.",),
        validation_commands=(), validation_profile="read_only_observation@1", allowed_effects=effects, write_paths=paths)
    def forbidden(*_args, **_kwargs): pytest.fail("read-only packet reached journal handling")
    monkeypatch.setattr(Journal, "require_open", forbidden)
    with pytest.raises(ValueError, match="readonly_provider_dispatch_refused"):
        runner.run(**options(dispatch_case, world_task_packet=readonly), world_task_attempt_journal=object())
    assert dispatch_case[3]["seen"] == []
