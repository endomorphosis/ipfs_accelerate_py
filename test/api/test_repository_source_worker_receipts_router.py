"""Worker report custody with an authored provider; no provider calls or owner work."""
import builtins
import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as binding_api
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_worker_receipts as receipts
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from test.api.test_repository_source_owner_router import owner_route
from test.api.test_repository_source_router import route, _run


def _empty_receipt(target):
    return dict(schema="repository-source-worker-forwarding@1", bridge_id="f" * 32,
        status="observed", complete=True, reason=None, session_id=target["session_id"],
        descriptor_sha256=target["descriptor_sha256"], returncode=0,
        owner_exchange_attempts=0, owner_exchange_unconfirmed=0, observed_responses=0, events=[],
        worker_reported=True, independently_verified=False, provider_ingestion_verified=False,
        token_savings_measured=False, proof_authority=False, execution_authority=False,
        completion_authority=False, raw_content_exported=False, observer_provider_calls=0)


def _row(capsys):
    return next(json.loads(line) for line in capsys.readouterr().out.splitlines()
                if line.startswith('{"') and json.loads(line).get("schema") == "router-implementation-invocation@1")


def _publish(binding):
    target = binding["worker_forwarding"]
    receipts.publish_receipt(target, _empty_receipt(target))


def test_owner_router_binds_captures_and_removes_worker_outbox(owner_route, route):
    options, state = owner_route
    state["callback"] = _publish
    output, row = _run(**options)
    binding, capture = row["repository_source_policy"], row["repository_source_worker_forwarding"]
    assert output == "completed authored provider result"
    assert capture["status"] == "captured" and capture["receipt_count"] == 1
    assert capture["cleanup_complete"] is True and capture["slot_scan_complete"] is True
    assert capture["invocation_id"] == row["invocation_id"] == binding["worker_forwarding"]["invocation_id"]
    assert capture["worker_reported"] is True and capture["independently_verified"] is False
    assert not Path(binding["worker_forwarding"]["directory"]).exists()
    assert not Path(binding["worker_forwarding"]["directory"]).is_relative_to(route[0])
    assert binding_api.validate_binding(binding)  # Historical replay survives cleanup.
    prompt = state["seen"][0][0]
    replay, _ = runner.render_model_prompt(prompt="Repair the declared file.", purpose="coding",
        workspace=route[0], repository_source_binding=binding)
    assert replay == prompt and "worker-outbox" not in prompt
    assert row["usage"]["prompt_tokens"] == 15 and row["usage"]["completion_tokens"] == 4
    assert len(state["seen"]) == 1


def test_missing_helper_receipt_stays_unknown_and_provider_result_survives(owner_route):
    options, state = owner_route
    output, row = _run(**options)
    capture = row["repository_source_worker_forwarding"]
    assert output and capture["status"] == "unavailable" and capture["receipt_count"] is None
    assert capture["receipts"] == [] and capture["cleanup_complete"] is True
    assert not Path(row["repository_source_policy"]["worker_forwarding"]["directory"]).exists()
    assert len(state["seen"]) == 1


@pytest.mark.parametrize("failure", ["timeout", "rate_limit", "interruption"])
def test_provider_failures_keep_original_outcome_and_capture(owner_route, failure, capsys):
    options, state = owner_route
    if failure == "rate_limit":
        from ipfs_accelerate_py.llm_allocation.observations import CallErrorKind
        error = RuntimeError("private-provider-message")
        error.codex_error_kind = CallErrorKind.RATE_LIMIT
        error.codex_error_status = 429
        error.codex_retry_after_seconds = 12
    else:
        error = TimeoutError("private-provider-message") if failure == "timeout" else KeyboardInterrupt()
    def fail(binding):
        _publish(binding)
        raise error
    state["callback"] = fail
    with pytest.raises(type(error)) as caught:
        _run(**options)
    assert caught.value is error
    row = _row(capsys)
    assert row["status"] == "failed" and row["router_calls"] == 1
    assert row["repository_source_worker_forwarding"]["status"] == "captured"
    assert row["repository_source_worker_forwarding"]["cleanup_complete"] is True
    assert "private-provider-message" not in json.dumps(row)
    if failure == "rate_limit":
        assert row["provider_failure"]["reason_code"] == "rate_limit"
        assert row["provider_failure"]["http_status"] == 429
        assert row["provider_failure"]["retry_after_seconds"] == 12
    assert len(state["seen"]) == 1


@pytest.mark.parametrize("during", ["creation", "binding"])
def test_optional_setup_failure_keeps_original_binding_and_provider(owner_route, monkeypatch, during):
    options, state = owner_route
    created = []
    original = receipts.create_outbox
    def create(**kwargs):
        if during == "creation":
            raise OSError("private-outbox-location")
        handle = original(**kwargs)
        created.append(handle.target["directory"])
        return handle
    monkeypatch.setattr(receipts, "create_outbox", create)
    if during == "binding":
        monkeypatch.setattr(binding_api, "with_worker_forwarding",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("private-target")))
    output, row = _run(**options)
    assert output and "worker_forwarding" not in row["repository_source_policy"]
    assert row["repository_source_worker_forwarding"]["status"] == "unavailable"
    assert "private-outbox-location" not in json.dumps(row) and "private-target" not in json.dumps(row)
    assert not any(Path(path).exists() for path in created)
    assert len(state["seen"]) == 1


@pytest.mark.parametrize("provider_failed", [False, True])
@pytest.mark.parametrize("interrupt_capture", [False, True])
def test_capture_errors_do_not_replace_provider_error(owner_route, monkeypatch, capsys,
                                                     provider_failed, interrupt_capture):
    options, state = owner_route
    original = receipts.create_outbox
    capture_error = KeyboardInterrupt() if interrupt_capture else OSError("private-capture-error")
    provider_error = TimeoutError("original-provider-error")
    class Handle:
        def __init__(self, wrapped):
            self.wrapped, self.target = wrapped, wrapped.target
        def collect_and_close(self):
            self.wrapped.collect_and_close()
            raise capture_error
    monkeypatch.setattr(receipts, "create_outbox", lambda **kw: Handle(original(**kw)))
    def callback(binding):
        _publish(binding)
        if provider_failed:
            raise provider_error
    state["callback"] = callback
    expected = provider_error if provider_failed else capture_error if interrupt_capture else None
    if expected is None:
        output, row = _run(**options)
        assert output
    else:
        with pytest.raises(type(expected)) as caught:
            _run(**options)
        assert caught.value is expected
        row = _row(capsys)
    assert row["repository_source_worker_forwarding"]["status"] == "unavailable"
    assert row["repository_source_worker_forwarding"]["receipt_count"] is None
    assert not Path(row["repository_source_policy"]["worker_forwarding"]["directory"]).exists()
    assert len(state["seen"]) == 1


def test_legacy_route_does_not_provision_receipt_files(route, monkeypatch):
    monkeypatch.setattr(receipts, "create_outbox", lambda **kw: pytest.fail("legacy route provisioned outbox"))
    _, row = _run()
    assert "repository_source_worker_forwarding" not in row


@pytest.mark.parametrize("failure", ["capture_import", "binding_import", "initial_metadata"])
def test_optional_accounting_import_and_metadata_failure_keeps_provider_route(
        owner_route, monkeypatch, failure):
    options, state = owner_route
    original_import = builtins.__import__
    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if ((failure == "capture_import" and "repository_source_worker_receipts" in (fromlist or ()))
                or (failure == "binding_import" and "with_worker_forwarding" in (fromlist or ()))):
            raise ImportError("private-optional-import")
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", imports)
    if failure == "initial_metadata":
        monkeypatch.setattr(receipts, "unavailable_capture",
            lambda **kw: (_ for _ in ()).throw(ValueError("private-initial-capture")))
    output, row = _run(**options)
    assert output and row["status"] == "provider_returned"
    assert "worker_forwarding" not in row["repository_source_policy"]
    assert "repository_source_worker_forwarding" not in row
    assert len(state["seen"]) == 1


@pytest.mark.parametrize("provider_outcome", ["success", "timeout", "rate_limit", "interruption"])
@pytest.mark.parametrize("capture_interrupted", [False, True])
@pytest.mark.parametrize("metadata_interrupted", [False, True])
def test_capture_failure_does_not_reenter_failed_metadata_or_skip_final_accounting(
        owner_route, monkeypatch, capsys, provider_outcome, capture_interrupted, metadata_interrupted):
    options, state = owner_route
    original_create = receipts.create_outbox
    capture_error = KeyboardInterrupt("PRIVATE_CAPTURE") if capture_interrupted else OSError("PRIVATE_CAPTURE")
    metadata_error = SystemExit("PRIVATE_METADATA") if metadata_interrupted else ValueError("PRIVATE_METADATA")
    provider_error = None
    if provider_outcome == "timeout":
        provider_error = TimeoutError("PRIVATE_PROVIDER")
    elif provider_outcome == "interruption":
        provider_error = KeyboardInterrupt("PRIVATE_PROVIDER")
    elif provider_outcome == "rate_limit":
        from ipfs_accelerate_py.llm_allocation.observations import CallErrorKind
        provider_error = RuntimeError("PRIVATE_PROVIDER")
        provider_error.codex_error_kind = CallErrorKind.RATE_LIMIT
        provider_error.codex_error_status = 429
        provider_error.codex_retry_after_seconds = 12
    secondary_calls, handles, observed = [], [], []

    def broken_metadata(**kwargs):
        secondary_calls.append(kwargs)
        raise metadata_error

    class Handle:
        def __init__(self, wrapped):
            self.wrapped, self.target = wrapped, wrapped.target
            handles.append(self)

        def collect_and_close(self):
            # Complete real bounded file cleanup before injecting the capture
            # failure. The newly unavailable metadata callback represents the
            # secondary fault that used to replace the provider outcome.
            self.wrapped.collect_and_close()
            monkeypatch.setattr(receipts, "unavailable_capture", broken_metadata)
            raise capture_error

    monkeypatch.setattr(receipts, "create_outbox", lambda **kwargs: Handle(original_create(**kwargs)))

    def callback(binding):
        _publish(binding)
        if provider_error is not None:
            raise provider_error

    state["callback"] = callback
    environment = {key: os.environ.get(key) for key in
                   ("ipfs_accelerate_py_CODEX_SANDBOX", "PYTHONDONTWRITEBYTECODE")}
    expected = provider_error if provider_error is not None else capture_error if capture_interrupted else None
    returned = None
    if expected is None:
        output, returned = _run(**options, receipt_observer=observed.append)
        assert output == "completed authored provider result"
    else:
        with pytest.raises(type(expected)) as caught:
            _run(**options, receipt_observer=observed.append)
        assert caught.value is expected
    row = _row(capsys)
    if returned is not None:
        assert returned == row
    assert secondary_calls == []  # Finalization uses the already established snapshot.
    assert observed == [row]
    assert row["status"] == ("provider_returned" if provider_error is None else "failed")
    assert row["router_calls"] == 1 and len(state["seen"]) == 1
    assert len(handles) == 1 and not Path(handles[0].target["directory"]).exists()
    assert handles[0].wrapped._fd is None
    capture = row["repository_source_worker_forwarding"]
    assert capture["status"] == "unavailable" and capture["reason"] == "capture_failed"
    assert capture["receipt_count"] is capture["receipt_bytes"] is None
    assert capture["receipts"] == [] and capture["cleanup_complete"] is False
    assert capture["invocation_id"] == row["invocation_id"]
    assert row["repository_source_policy"]["usage"] == binding_api.unavailable_usage()
    assert row["usage"]["prompt_tokens"] == 15 and row["usage"]["completion_tokens"] == 4
    assert "native_rollout_usage" in row and "context_delivery_observation" in row
    assert {key: os.environ.get(key) for key in environment} == environment
    assert "PRIVATE_" not in json.dumps(row)
    if provider_outcome == "rate_limit":
        assert row["provider_failure"]["reason_code"] == "rate_limit"
        assert row["provider_failure"]["http_status"] == 429
        assert row["provider_failure"]["retry_after_seconds"] == 12


@pytest.mark.parametrize("provider_failed", [False, True])
def test_first_capture_interruption_survives_later_receipt_observer_interruption(
        owner_route, monkeypatch, capsys, provider_failed):
    options, state = owner_route
    original_create = receipts.create_outbox
    capture_error = KeyboardInterrupt("PRIVATE_FIRST_CAPTURE")
    observer_error = SystemExit("PRIVATE_LATER_OBSERVER")
    provider_error = TimeoutError("PRIVATE_PROVIDER") if provider_failed else None
    observed, handles = [], []

    class Handle:
        def __init__(self, wrapped):
            self.wrapped, self.target = wrapped, wrapped.target
            handles.append(self)

        def collect_and_close(self):
            self.wrapped.collect_and_close()
            raise capture_error

    monkeypatch.setattr(receipts, "create_outbox", lambda **kwargs: Handle(original_create(**kwargs)))

    def callback(binding):
        _publish(binding)
        if provider_error is not None:
            raise provider_error

    def observer(row):
        observed.append(row)
        raise observer_error

    state["callback"] = callback
    expected = provider_error if provider_error is not None else capture_error
    with pytest.raises(type(expected)) as caught:
        _run(**options, receipt_observer=observer)
    assert caught.value is expected
    row = _row(capsys)
    assert len(observed) == 1 and len(state["seen"]) == row["router_calls"] == 1
    assert row["receipt_observer_error_type"] == "SystemExit"
    assert observed[0]["usage"] == row["usage"]
    assert observed[0]["native_rollout_usage"] == row["native_rollout_usage"]
    assert row["repository_source_policy"]["usage"] == binding_api.unavailable_usage()
    assert row["repository_source_worker_forwarding"]["reason"] == "capture_failed"
    assert row["repository_source_worker_forwarding"]["receipt_count"] is None
    assert not Path(handles[0].target["directory"]).exists() and handles[0].wrapped._fd is None
    assert "PRIVATE_" not in json.dumps(row)
