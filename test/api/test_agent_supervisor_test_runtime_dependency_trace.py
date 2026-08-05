"""PTR-021 contract tests for bounded runtime dependency tracing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    ContentIdentity,
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_runtime_dependency_trace import (
    RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE,
    RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA,
    RUNTIME_TRACE_INSTRUMENTATION_INTERFACE,
    RUNTIME_TRACE_LIMITS_INTERFACE,
    RuntimeTestDependencyTracer,
    RuntimeTraceCompleteness,
    RuntimeTraceError,
    RuntimeTraceLimits,
    trace_runtime_dependencies,
)
from multiformats import CID


def _cid(label: str) -> str:
    return mint_content_identity({"label": label}).cid


def _tracer(tmp_path: Path, **kwargs: object) -> RuntimeTestDependencyTracer:
    values: dict[str, object] = {
        "allowed_roots": {"repo": tmp_path},
        "capture_code_objects": False,
    }
    values.update(kwargs)
    return RuntimeTestDependencyTracer(**values)  # type: ignore[arg-type]


def _assert_profile_cid(trace_cid: str, canonical_bytes: bytes) -> None:
    parsed = CID.decode(trace_cid)
    assert parsed.version == 1
    assert parsed.base.name == "base32"
    assert parsed.codec.name == "dag-json"
    assert parsed.hashfun.name == "sha2-256"
    assert bytes(parsed.raw_digest) == hashlib.sha256(canonical_bytes).digest()


def test_empty_trace_is_complete_canonical_and_content_addressed(tmp_path: Path) -> None:
    tracer = _tracer(tmp_path)
    with tracer:
        pass
    trace = tracer.result

    assert trace is not None
    assert trace.interface == RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE
    assert trace.schema == RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA
    assert trace.completeness is RuntimeTraceCompleteness.COMPLETE
    assert trace.complete is True
    assert trace.completeness_reasons == ()
    assert trace.content_identity is not None
    assert trace.verify() is trace
    assert (
        json.dumps(
            trace.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        == trace.canonical_bytes
    )
    _assert_profile_cid(trace.cid, trace.canonical_bytes)

    payload = trace.to_dict()
    assert payload["limits"]["interface"] == RUNTIME_TRACE_LIMITS_INTERFACE
    assert payload["instrumentation"]["interface"] == RUNTIME_TRACE_INSTRUMENTATION_INTERFACE
    assert payload["instrumentation_cid"].startswith("b")
    assert payload["health"]["audit_hook_healthy"] is True


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"max_events": 0}, "max_events"),
        ({"max_events": True}, "max_events"),
        ({"max_file_bytes": 100_000_000}, "max_file_bytes"),
        ({"max_trace_seconds": 0}, "max_trace_seconds"),
        ({"max_text_chars": 8}, "max_text_chars"),
    ],
)
def test_trace_limits_are_hard_validated(changes: dict[str, object], match: str) -> None:
    with pytest.raises(RuntimeTraceError, match=match):
        RuntimeTraceLimits(**changes)  # type: ignore[arg-type]


def test_configuration_rejects_unreviewed_environment_and_fake_tool_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeTraceError, match="non-reviewed"):
        _tracer(tmp_path, environment_allowlist=("AWS_SECRET_ACCESS_KEY",))
    with pytest.raises(RuntimeTraceError, match="canonical CID"):
        _tracer(tmp_path, subprocess_allowlist={"python": "sha256:not-a-cid"})
    with pytest.raises(RuntimeTraceError, match="canonical CID"):
        _tracer(tmp_path, subprocess_allowlist={"python": "b" + "a" * 60})


def test_dependency_order_does_not_change_canonical_trace(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text('{"ok":true}', encoding="utf-8")

    first = _tracer(tmp_path, environment_allowlist=("TZ",))
    first.start()
    assert first.record_environment_read("TZ", "UTC")
    assert first.record_module("demo.module", source_path=data)
    assert first.record_file_read(data)
    trace_a = first.stop()

    second = _tracer(tmp_path, environment_allowlist=("TZ",))
    second.start()
    assert second.record_file_read(data)
    assert second.record_module("demo.module", source_path=data)
    assert second.record_environment_read("TZ", "UTC")
    trace_b = second.stop()

    assert trace_a.complete and trace_b.complete
    assert trace_a.cid == trace_b.cid
    assert trace_a.canonical_bytes == trace_b.canonical_bytes
    assert trace_a.recorded_fact_count == 3


def test_file_fact_is_relative_and_binds_content(tmp_path: Path) -> None:
    nested = tmp_path / "fixtures"
    nested.mkdir()
    data = nested / "payload.bin"
    data.write_bytes(b"dependency bytes")

    tracer = _tracer(tmp_path)
    tracer.start()
    assert tracer.record_file_read(data)
    trace = tracer.stop()

    fact = trace.to_dict()["dependencies"]["files"][0]
    assert fact == {
        "root_id": "repo",
        "path": "fixtures/payload.bin",
        "size_bytes": len(b"dependency bytes"),
        "content_sha256": hashlib.sha256(b"dependency bytes").hexdigest(),
    }
    assert str(tmp_path) not in trace.canonical_bytes.decode()
    assert trace.complete


def test_private_path_marks_incomplete_without_leaking_path_or_body(
    tmp_path: Path,
) -> None:
    private_dir = tmp_path.parent / "private-runtime-trace"
    private_dir.mkdir(exist_ok=True)
    private_file = private_dir / "credential-token.txt"
    private_file.write_text("super-secret-body", encoding="utf-8")

    tracer = _tracer(tmp_path)
    tracer.start()
    assert tracer.record_file_read(private_file) is False
    trace = tracer.stop()

    encoded = trace.canonical_bytes.decode()
    assert not trace.complete
    assert "private_event" in trace.completeness_reasons
    assert "credential-token" not in encoded
    assert "super-secret-body" not in encoded
    assert str(private_file) not in encoded
    assert trace.to_dict()["health"]["private_event_kinds"] == ["file_path"]


def test_symlink_is_private_and_never_retained(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("public content", encoding="utf-8")
    link = tmp_path / "linked-secret-name.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    tracer = _tracer(tmp_path)
    tracer.start()
    assert tracer.record_file_read(link) is False
    trace = tracer.stop()
    assert not trace.complete
    assert "linked-secret-name" not in trace.canonical_bytes.decode()
    assert "symlink" in trace.to_dict()["health"]["private_event_kinds"]


def test_file_size_and_event_overflow_are_explicit(tmp_path: Path) -> None:
    data = tmp_path / "large.bin"
    data.write_bytes(b"0123456789")
    limits = RuntimeTraceLimits(max_file_bytes=4, max_events=2)
    tracer = _tracer(tmp_path, limits=limits)
    tracer.start()
    assert tracer.record_file_read(data) is False
    assert tracer.record_module("one")
    assert tracer.record_module("two")
    assert tracer.record_module("three") is False
    trace = tracer.stop()

    assert trace.completeness is RuntimeTraceCompleteness.INCOMPLETE
    assert "overflow" in trace.completeness_reasons
    assert trace.dropped_event_count == 1
    assert len(trace.to_dict()["dependencies"]["modules"]) == 2
    assert trace.to_dict()["dependencies"]["files"] == []


def test_allowlisted_environment_is_value_cid_only(tmp_path: Path) -> None:
    tracer = _tracer(tmp_path, environment_allowlist=("TZ",))
    tracer.start()
    assert tracer.record_environment_read("TZ", "Never/Retain/This/Raw")
    trace = tracer.stop()

    encoded = trace.canonical_bytes.decode()
    fact = trace.to_dict()["dependencies"]["environment"][0]
    assert fact["name"] == "TZ"
    assert fact["value_cid"].startswith("b")
    assert "Never/Retain/This/Raw" not in encoded
    assert trace.complete


def test_internal_identity_minting_is_not_observed_as_test_activity(
    tmp_path: Path,
) -> None:
    private_probe = str(tmp_path.parent / "private-cid-provider-input")

    def auditing_identity_minter(value: object) -> ContentIdentity:
        sys.audit("open", private_probe, "r", 0)
        sys.audit("provider.private-operation")
        return mint_content_identity(value)

    tracer = _tracer(
        tmp_path,
        environment_allowlist=("TZ",),
        identity_minter=auditing_identity_minter,
    )
    tracer.start()
    assert tracer.record_environment_read("TZ", "UTC")
    trace = tracer.stop()

    assert trace.complete
    assert trace.to_dict()["health"]["private_event_kinds"] == []
    assert trace.to_dict()["health"]["unsupported_event_kinds"] == []
    assert private_probe not in trace.canonical_bytes.decode()


def test_private_environment_event_is_incomplete_and_secret_free(tmp_path: Path) -> None:
    tracer = _tracer(tmp_path, environment_allowlist=("TZ",))
    tracer.start()
    assert tracer.record_environment_read("AWS_SECRET_ACCESS_KEY", "do-not-retain") is False
    trace = tracer.stop()

    encoded = trace.canonical_bytes.decode()
    assert not trace.complete
    assert "private_event" in trace.completeness_reasons
    assert "AWS_SECRET_ACCESS_KEY" not in encoded
    assert "do-not-retain" not in encoded


def test_getenv_adapter_preserves_value_and_records_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "UTC")
    tracer = _tracer(tmp_path, environment_allowlist=("TZ",))
    tracer.start()
    assert tracer.getenv("TZ") == "UTC"
    trace = tracer.stop()
    assert trace.complete
    assert trace.to_dict()["dependencies"]["environment"][0]["name"] == "TZ"
    assert '"value":"UTC"' not in trace.canonical_bytes.decode()


def test_subprocess_arguments_are_digest_only_and_tool_identity_is_bound(
    tmp_path: Path,
) -> None:
    tool_cid = _cid("python-tool")
    tracer = _tracer(tmp_path, subprocess_allowlist={"python3": tool_cid})
    tracer.start()
    assert tracer.record_subprocess("/usr/bin/python3", ["python3", "--token", "do-not-retain"])
    trace = tracer.stop()

    fact = trace.to_dict()["dependencies"]["subprocesses"][0]
    assert fact["executable"] == "python3"
    assert fact["argument_count"] == 3
    assert fact["tool_identity"] == tool_cid
    assert len(fact["arguments_sha256"]) == 64
    encoded = trace.canonical_bytes.decode()
    assert "do-not-retain" not in encoded
    assert "--token" not in encoded
    assert "/usr/bin" not in encoded
    assert trace.complete


def test_unadmitted_subprocess_and_unsupported_audit_event_are_incomplete(
    tmp_path: Path,
) -> None:
    tracer = _tracer(tmp_path)
    tracer.start()
    assert tracer.record_subprocess("curl", ["curl", "https://private.invalid"]) is True
    tracer.observe_audit_event("vendor.private", ("secret-audit-body",))
    trace = tracer.stop()

    encoded = trace.canonical_bytes.decode()
    assert not trace.complete
    assert "unsupported_event" in trace.completeness_reasons
    assert "https://private.invalid" not in encoded
    assert "secret-audit-body" not in encoded
    assert trace.to_dict()["health"]["unsupported_event_kinds"] == [
        "subprocess_tool",
        "unknown_audit_event",
    ]


def test_process_audit_hook_marks_custom_events_without_raising(tmp_path: Path) -> None:
    tracer = _tracer(tmp_path)
    tracer.start()
    sys.audit("vendor.unsupported-dependency", "private audit detail")
    trace = tracer.stop()
    assert not trace.complete
    assert "private audit detail" not in trace.canonical_bytes.decode()
    assert "unknown_audit_event" in trace.to_dict()["health"]["unsupported_event_kinds"]


def test_silently_suppressed_audit_hook_is_never_reported_healthy() -> None:
    script = textwrap.dedent(
        """
        import sys

        def suppress_new_hook(event, _arguments):
            if event == "sys.addaudithook":
                raise RuntimeError("silently suppress hook registration")

        sys.addaudithook(suppress_new_hook)
        sys.path.insert(0, sys.argv[1])

        from ipfs_accelerate_py.agent_supervisor.analysis.test_runtime_dependency_trace import (
            RuntimeTestDependencyTracer,
        )

        tracer = RuntimeTestDependencyTracer(capture_code_objects=False)
        with tracer:
            pass
        trace = tracer.result
        assert trace is not None
        payload = trace.to_dict()
        assert payload["health"]["audit_hook_healthy"] is False
        assert "instrumentation_failure" in trace.completeness_reasons
        assert not trace.complete
        """
    )

    result = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            script,
            str(Path(__file__).parents[2]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_service_policy_and_capability_adapters_bind_cids(tmp_path: Path) -> None:
    adapter = _cid("adapter")
    snapshot = _cid("snapshot")
    clock = _cid("clock-policy")
    random = _cid("random-policy")
    state = _cid("cuda-state")
    tracer = _tracer(tmp_path, eligibility_profile="snapshot_bound")
    tracer.start()
    assert tracer.record_service("postgres", adapter_identity=adapter, snapshot_identity=snapshot)
    assert tracer.record_clock_policy(clock)
    assert tracer.record_randomness_policy(random)
    assert tracer.record_capability("cuda", adapter_identity=adapter, state_identity=state)
    trace = tracer.stop()

    assert trace.complete
    payload = trace.to_dict()
    assert payload["eligibility_profile"] == "snapshot_bound"
    assert payload["dependencies"]["services"][0]["snapshot_identity"] == snapshot
    assert {item["kind"] for item in payload["dependencies"]["policies"]} == {
        "clock",
        "randomness",
    }
    assert payload["dependencies"]["capabilities"][0]["state_identity"] == state


def test_invalid_adapter_identity_is_unsupported_not_an_exception(tmp_path: Path) -> None:
    tracer = _tracer(tmp_path)
    tracer.start()
    assert (
        tracer.record_service("database", adapter_identity="cid:fake", snapshot_identity="secret")
        is False
    )
    assert tracer.record_clock_policy("sha256:fake") is False
    assert tracer.record_capability("gpu", adapter_identity="bad", state_identity="worse") is False
    trace = tracer.stop()
    assert not trace.complete
    assert "unsupported_event" in trace.completeness_reasons
    assert "cid:fake" not in trace.canonical_bytes.decode()
    assert "sha256:fake" not in trace.canonical_bytes.decode()


def test_explicit_code_object_uses_relative_path_and_normalized_digest() -> None:
    test_file = Path(__file__).resolve()
    root = test_file.parents[2]
    tracer = RuntimeTestDependencyTracer(
        allowed_roots={"accelerate": root}, capture_code_objects=False
    )
    tracer.start()
    assert tracer.record_code_object(
        test_explicit_code_object_uses_relative_path_and_normalized_digest.__code__,
        __name__,
    )
    trace = tracer.stop()

    fact = trace.to_dict()["dependencies"]["code_objects"][0]
    assert not fact["path"].startswith("/")
    assert fact["path"].endswith("test/api/test_agent_supervisor_test_runtime_dependency_trace.py")
    assert len(fact["code_sha256"]) == 64
    assert str(root) not in trace.canonical_bytes.decode()
    assert trace.complete


def test_native_module_is_retained_as_unknown_frontier(tmp_path: Path) -> None:
    module = tmp_path / "extension.so"
    module.write_bytes(b"native-placeholder")
    tracer = _tracer(tmp_path)
    tracer.start()
    assert tracer.record_module("demo.native", source_path=module, native=True)
    trace = tracer.stop()
    assert not trace.complete
    assert trace.to_dict()["dependencies"]["modules"][0]["kind"] == "native"
    assert "native_module" in trace.to_dict()["health"]["unsupported_event_kinds"]


def test_audited_open_records_read_and_write_is_unsupported(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("observed", encoding="utf-8")
    output = tmp_path / "output.txt"
    tracer = _tracer(tmp_path)
    tracer.start()
    with source.open("rb") as stream:
        assert stream.read() == b"observed"
    with output.open("w", encoding="utf-8") as stream:
        stream.write("effect")
    trace = tracer.stop()

    paths = {fact["path"] for fact in trace.to_dict()["dependencies"]["files"]}
    assert "source.txt" in paths
    assert "output.txt" not in paths
    assert not trace.complete
    assert "file_write" in trace.to_dict()["health"]["unsupported_event_kinds"]


def test_recording_failure_never_changes_test_exception(tmp_path: Path) -> None:
    class TestSentinel(RuntimeError):
        __test__ = False

    tracer = _tracer(tmp_path)
    sentinel = TestSentinel("test outcome")
    with pytest.raises(TestSentinel) as caught:
        with tracer:
            tracer.observe_audit_event("unsupported", (object(),))
            raise sentinel
    assert caught.value is sentinel
    assert tracer.result is not None
    assert not tracer.result.complete


def test_identity_provider_failure_is_incomplete_and_does_not_escape(
    tmp_path: Path,
) -> None:
    def fail_identity(_value: object) -> ContentIdentity:
        raise RuntimeError("provider failure with private detail")

    tracer = _tracer(tmp_path, identity_minter=fail_identity)
    with tracer:
        observed_test_value = 42
    trace = tracer.result
    assert observed_test_value == 42
    assert trace is not None
    assert not trace.complete
    assert trace.cid == ""
    assert "instrumentation_failure" in trace.completeness_reasons
    assert "private detail" not in trace.canonical_bytes.decode()


def test_identity_provider_cannot_substitute_different_canonical_bytes(
    tmp_path: Path,
) -> None:
    def wrong_value_identity(_value: object) -> ContentIdentity:
        return mint_content_identity(
            {"schema": "unrelated/provider-output@1", "secret": "do-not-retain"}
        )

    tracer = _tracer(tmp_path, identity_minter=wrong_value_identity)
    with tracer:
        observed_test_value = 42
    trace = tracer.result

    assert observed_test_value == 42
    assert trace is not None
    assert not trace.complete
    assert trace.cid == ""
    assert "instrumentation_failure" in trace.completeness_reasons
    assert "do-not-retain" not in trace.canonical_bytes.decode()


def test_stop_is_idempotent(tmp_path: Path) -> None:
    tracer = _tracer(tmp_path)
    tracer.start()
    first = tracer.stop()
    second = tracer.stop()
    assert second is first


def test_trace_runtime_dependencies_returns_value_and_trace(tmp_path: Path) -> None:
    tracer = _tracer(tmp_path)
    value, trace = trace_runtime_dependencies(lambda left, right: left + right, 2, 3, tracer=tracer)
    assert value == 5
    assert trace is tracer.result
    assert trace.complete


def test_trace_runtime_dependencies_preserves_operation_exception(tmp_path: Path) -> None:
    class OperationError(Exception):
        pass

    error = OperationError("original")
    tracer = _tracer(tmp_path)

    def fail() -> None:
        raise error

    with pytest.raises(OperationError) as caught:
        trace_runtime_dependencies(fail, tracer=tracer)
    assert caught.value is error
    assert tracer.result is not None
