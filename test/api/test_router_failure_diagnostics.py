"""Wrapper failures remain visible without exporting input or granting authority."""
import io
import json
import linecache
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner


def _run_main(monkeypatch, capsys, *, prompt=b"private-input-canary", argv=None):
    monkeypatch.setattr(sys, "argv", ["router", *(argv or ["--provider", "grok_cli", "--model", "wrong-model"])])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(prompt)))
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # No invented invocation/usage receipt.
    assert "private-input-canary" not in captured.err
    return runner.validate_runner_error_envelope(json.loads(captured.err))


def test_actual_argument_failure_retains_source_phase_without_reading_source(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics tried to read source or dispatch")
    monkeypatch.setattr(linecache, "getline", forbidden)
    monkeypatch.setattr(linecache, "getlines", forbidden)
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "generate_text", forbidden)
    failure = _run_main(monkeypatch, capsys)
    diagnostic = failure["diagnostic"]
    assert diagnostic["phase"] == "argument_validation"
    assert failure["error_type"] == "ValueError"
    assert diagnostic["exceptions"][0]["frames"]
    assert all(frame["file"] == "router_implementation_runner.py"
               for frame in diagnostic["exceptions"][0]["frames"])
    assert diagnostic["provider_dispatch_observed"] is None
    assert diagnostic["completion_authority"] is False
    assert diagnostic["settlement_authority"] is False
    assert diagnostic["automatic_retry_admitted"] is False


def test_actual_grok_missing_boundary_has_separate_phase(monkeypatch, capsys):
    failure = _run_main(monkeypatch, capsys, argv=["--provider", "grok_cli", "--model", "grok-4.7"])
    assert failure["diagnostic"]["phase"] == "container_boundary"


def test_invalid_utf8_still_returns_bounded_error(monkeypatch, capsys):
    failure = _run_main(monkeypatch, capsys, prompt=b"\xffprivate-input-canary")
    assert failure["error_type"] == "UnicodeDecodeError"
    assert failure["diagnostic"]["phase"] == "unclassified"


def test_provider_failure_keeps_invocation_receipt_and_phase(tmp_path, monkeypatch, capsys):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())

    def timeout(*args, **kwargs):
        raise TimeoutError("private-provider-canary")

    monkeypatch.setattr(llm_router, "generate_text", timeout)
    monkeypatch.setattr(sys, "argv", ["router", "--model", "pinned"])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"private-input-canary")))
    assert runner.main() == 1
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert receipt["schema"] == "router-implementation-invocation@1"
    assert receipt["status"] == "failed" and receipt["usage"] == {}
    failure = runner.validate_runner_error_envelope(json.loads(captured.err))
    assert failure["diagnostic"]["phase"] == "provider_invocation"
    assert "private" not in captured.err
    assert failure["diagnostic"]["provider_dispatch_observed"] is None


def _exception():
    try:
        raise ValueError("private-error-canary")
    except ValueError as error:
        return error


def test_foreign_frames_dynamic_types_and_cycles_are_bounded():
    secret_type = type("private_type_canary", (ValueError,), {})
    error = secret_type("private-error-canary")
    inner = _exception()
    error.__cause__ = inner
    inner.__cause__ = error
    diagnostic = runner._runner_failure_diagnostic(error)
    encoded = json.dumps(diagnostic)
    assert "private" not in encoded and "test_router" not in encoded
    assert diagnostic["exceptions"][0]["exception_type"] == "other"
    assert len(diagnostic["exceptions"]) == 2 and diagnostic["chain_truncated"] is True
    assert all(item["frames"] == [] for item in diagnostic["exceptions"])


def test_diagnostic_walk_limits_deep_chains():
    error = _exception()
    for _ in range(12):
        outer = _exception()
        outer.__cause__ = error
        error = outer
    result = runner._runner_failure_diagnostic(error)
    assert len(result["exceptions"]) == 4 and result["chain_truncated"] is True


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra="untrusted"),
    lambda value: value.update(phase="arbitrary-phase"),
    lambda value: value.update(provider_dispatch_observed=False),
    lambda value: value.update(completion_authority=True),
    lambda value: value.update(settlement_authority=True),
    lambda value: value.update(automatic_retry_admitted=True),
    lambda value: value.update(chain_truncated=0),
    lambda value: value.update(exceptions=[]),
    lambda value: value["exceptions"][0].update(exception_type="arbitrary-name"),
    lambda value: value["exceptions"][0].update(message="private"),
    lambda value: value["exceptions"][0].update(frames=[{"file": "foreign.py", "line": 1}]),
    lambda value: value["exceptions"][0].update(frames=[{"file": "router_implementation_runner.py", "line": True}]),
    lambda value: value["exceptions"][0].update(frames=[{"file": "router_implementation_runner.py", "line": 0}]),
])
def test_closed_validator_rejects_expansion_and_authority(mutation):
    value = runner._runner_failure_diagnostic(_exception())
    mutation(value)
    with pytest.raises(ValueError):
        runner.validate_runner_failure_diagnostic(value)


def test_envelope_is_exact_and_returns_copy():
    diagnostic = runner._runner_failure_diagnostic(_exception())
    value = {"schema": "router-implementation-error@2", "error_type": "ValueError", "diagnostic": diagnostic}
    observed = runner.validate_runner_error_envelope(value)
    observed["diagnostic"]["exceptions"].clear()
    assert value["diagnostic"]["exceptions"]
    for replacement in ({"error_type": "RuntimeError"}, {"schema": "router-implementation-error@1"}, {"raw_error": "private"}):
        with pytest.raises(ValueError):
            runner.validate_runner_error_envelope({**value, **replacement})


def test_failure_diagnostic_failure_does_not_change_exit(monkeypatch, capsys):
    monkeypatch.setattr(runner, "_runner_failure_diagnostic", lambda error: (_ for _ in ()).throw(RuntimeError("private")))
    monkeypatch.setattr(sys, "argv", ["router", "--provider", "grok_cli", "--model", "wrong-model"])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"private-input-canary")))
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"schema": "router-implementation-error@1", "error_type": "other"}
