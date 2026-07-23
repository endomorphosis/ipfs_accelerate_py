from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_capabilities import (
    PROOF_PROVIDER_CAPABILITY_SCHEMA_VERSION,
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_provider import (
    PROOF_PROVIDER_PROTOCOL_VERSION,
    PROOF_PROVIDER_REQUEST_SCHEMA,
    PROOF_PROVIDER_RESPONSE_SCHEMA,
    CancellationToken,
    InProcessProofProvider,
    NetworkAccessDenied,
    ProofProviderRegistry,
    ProviderFailureCode,
    ProviderInvocationConfig,
    ProviderInvocationError,
    ProviderRequest,
    ProviderResponse,
    SubprocessProofProvider,
    serve_provider_json,
)


ALL_OPERATIONS = tuple(ProofProviderOperation)


class FixtureProvider:
    provider_id = "fixture"
    provider_version = "1.2.3"
    protocol_version = PROOF_PROVIDER_PROTOCOL_VERSION

    def __init__(self) -> None:
        self.calls: list[ProviderRequest] = []

    def _result(self, request: ProviderRequest) -> dict[str, object]:
        self.calls.append(request)
        return {
            "operation": request.operation.value,
            "payload": dict(request.payload),
            # Status text remains opaque provider output, not supervisor
            # assurance or a ProofReceipt.
            "provider_claimed_status": "kernel_verified",
        }

    capability = _result
    translate = _result
    prove = _result
    reconstruct = _result
    verify = _result
    attest = _result


def _capability(*, network_required: bool = False) -> ProofProviderCapability:
    return ProofProviderCapability(
        provider_id="fixture",
        provider_version="1.2.3",
        protocol_versions=(1,),
        operations=ALL_OPERATIONS,
        isolation=(
            ProofProviderIsolation.IN_PROCESS,
            ProofProviderIsolation.SUBPROCESS,
        ),
        network_access_required=network_required,
        resource_limits_supported=True,
        metadata={"toolchain": "fixture"},
    )


def test_operation_capability_is_versioned_deterministic_and_not_proof_evidence() -> None:
    capability = _capability()
    payload = capability.to_dict()

    assert payload["schema_version"] == PROOF_PROVIDER_CAPABILITY_SCHEMA_VERSION
    assert payload["operations"] == [operation.value for operation in ALL_OPERATIONS]
    assert payload["proof_attempted"] is False
    assert payload["proof_success"] is False
    assert ProofProviderCapability.from_dict(payload) == capability
    assert all(capability.supports(operation) for operation in ALL_OPERATIONS)
    assert not capability.supports("not-an-operation")
    assert not capability.supports(ProofProviderOperation.PROVE, protocol_version=99)

    with pytest.raises(ValueError, match="capability"):
        ProofProviderCapability(
            provider_id="broken",
            provider_version="1",
            operations=(ProofProviderOperation.PROVE,),
        )


def test_request_response_envelopes_are_correlated_and_fail_closed() -> None:
    request = ProviderRequest(
        operation=ProofProviderOperation.PROVE,
        payload={"obligation_id": "bafy-obligation"},
        request_id="request-1",
        resource_budget=ResourceBudget(
            wall_time_ms=1000,
            memory_bytes=64 * 1024 * 1024,
            max_output_bytes=4096,
            network_allowed=False,
        ),
    )
    assert request.to_dict()["schema_version"] == PROOF_PROVIDER_REQUEST_SCHEMA
    assert ProviderRequest.from_dict(request.to_dict()) == request

    response = ProviderResponse.success(
        request,
        {"candidate_id": "candidate-1"},
        provider_id="fixture",
        provider_version="1",
    )
    assert response.to_dict()["schema_version"] == PROOF_PROVIDER_RESPONSE_SCHEMA
    assert ProviderResponse.from_dict(response.to_dict()) == response
    assert response.require_result()["candidate_id"] == "candidate-1"

    failure = ProviderResponse.failure(
        request,
        ProviderFailureCode.RESOURCE_EXHAUSTED,
        "memory budget exceeded",
    )
    with pytest.raises(ProviderInvocationError) as raised:
        failure.require_result()
    assert raised.value.code is ProviderFailureCode.RESOURCE_EXHAUSTED

    with pytest.raises(ValueError, match="successful"):
        ProviderResponse(
            request_id=request.request_id,
            operation=request.operation,
            ok=True,
            result={"candidate": True},
            error={"code": "provider_error", "message": "contradiction"},
        )
    with pytest.raises(ValueError, match="strict JSON"):
        ProviderRequest(operation="prove", payload={"unsafe": float("nan")})


def test_all_protocol_operations_execute_in_process_without_promoting_status() -> None:
    provider = FixtureProvider()
    client = InProcessProofProvider(
        provider,
        config=ProviderInvocationConfig(timeout_seconds=1),
        expected_capability=_capability(),
    )

    for operation in ALL_OPERATIONS:
        response = client.call(operation, {"value": operation.value})
        assert response.ok
        assert response.operation is operation
        assert response.provider_id == provider.provider_id
        assert response.result is not None
        assert response.result["provider_claimed_status"] == "kernel_verified"
        assert "authoritative_assurance" not in response.to_dict()
    assert [request.operation for request in provider.calls] == list(ALL_OPERATIONS)


def test_in_process_timeout_cancellation_network_and_malformed_fail_explicitly() -> None:
    class SlowProvider(FixtureProvider):
        def prove(self, request: ProviderRequest) -> dict[str, object]:
            time.sleep(0.2)
            return super().prove(request)

    timeout_response = InProcessProofProvider(
        SlowProvider(),
        config=ProviderInvocationConfig(timeout_seconds=0.02),
    ).prove({})
    assert not timeout_response.ok
    assert timeout_response.error is not None
    assert timeout_response.error.code is ProviderFailureCode.TIMED_OUT

    token = CancellationToken()
    threading.Timer(0.02, token.cancel).start()
    cancelled = InProcessProofProvider(
        SlowProvider(),
        config=ProviderInvocationConfig(timeout_seconds=1),
    ).prove({}, cancellation=token)
    assert not cancelled.ok
    assert cancelled.error is not None
    assert cancelled.error.code is ProviderFailureCode.CANCELLED

    denied = InProcessProofProvider(
        FixtureProvider(),
        config=ProviderInvocationConfig(allow_network=False),
        expected_capability=_capability(network_required=True),
    ).prove({}, network_allowed=True)
    assert not denied.ok
    assert denied.error is not None
    assert denied.error.code is ProviderFailureCode.NETWORK_DENIED

    class MalformedProvider(FixtureProvider):
        def prove(self, request: ProviderRequest) -> object:
            return object()

    malformed = InProcessProofProvider(MalformedProvider()).prove({})
    assert not malformed.ok
    assert malformed.error is not None
    assert malformed.error.code is ProviderFailureCode.MALFORMED_RESPONSE

    class NetworkProvider(FixtureProvider):
        def prove(self, request: ProviderRequest) -> dict[str, object]:
            raise NetworkAccessDenied()

    network_failure = InProcessProofProvider(NetworkProvider()).prove({})
    assert network_failure.error is not None
    assert network_failure.error.code is ProviderFailureCode.NETWORK_DENIED


def _write_subprocess_fixture(tmp_path: Path) -> Path:
    script = tmp_path / "proof_provider_fixture.py"
    script.write_text(
        """
import json
import sys
import time

request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
mode = request["payload"].get("mode", "ok")
if mode == "sleep":
    time.sleep(5)
if mode == "malformed":
    sys.stdout.write("{ definitely-not-json")
    raise SystemExit(0)
if mode == "multiple":
    sys.stdout.write("{}\\n{}\\n")
    raise SystemExit(0)
if mode == "oversized":
    sys.stdout.write("x" * 100000)
    raise SystemExit(0)
if mode == "exit":
    sys.stderr.write("provider failed")
    raise SystemExit(7)

response = {
    "schema_version": "ipfs_accelerate_py/agent-supervisor/proof-provider-response@1",
    "protocol_version": 1,
    "request_id": "wrong-id" if mode == "mismatch" else request["request_id"],
    "operation": request["operation"],
    "ok": True,
    "result": {
        "echo": request["payload"],
        "network_allowed": request["network_allowed"],
    },
    "error": None,
    "provider_id": "fixture-subprocess",
    "provider_version": "1",
    "duration_ms": 1,
}
sys.stdout.write(json.dumps(response, sort_keys=True))
""".lstrip(),
        encoding="utf-8",
    )
    return script


def test_subprocess_protocol_round_trip_and_all_failure_classes(tmp_path: Path) -> None:
    script = _write_subprocess_fixture(tmp_path)
    client = SubprocessProofProvider(
        (sys.executable, str(script)),
        config=ProviderInvocationConfig(
            timeout_seconds=0.2,
            max_request_bytes=4096,
            max_response_bytes=4096,
            memory_bytes=256 * 1024 * 1024,
            cpu_time_seconds=2,
            max_processes=4,
        ),
        expected_capability=_capability(),
    )

    success = client.translate({"source": "x"})
    assert success.ok
    assert success.provider_id == "fixture-subprocess"
    assert success.result == {
        "echo": {"source": "x"},
        "network_allowed": False,
    }

    expectations = {
        "sleep": ProviderFailureCode.TIMED_OUT,
        "malformed": ProviderFailureCode.MALFORMED_RESPONSE,
        "multiple": ProviderFailureCode.MALFORMED_RESPONSE,
        "oversized": ProviderFailureCode.RESOURCE_EXHAUSTED,
        "exit": ProviderFailureCode.PROVIDER_ERROR,
        "mismatch": ProviderFailureCode.MALFORMED_RESPONSE,
    }
    for mode, expected_code in expectations.items():
        response = client.prove({"mode": mode})
        assert not response.ok, mode
        assert response.error is not None
        assert response.error.code is expected_code

    too_large = client.prove({"source": "x" * 5000})
    assert too_large.error is not None
    assert too_large.error.code is ProviderFailureCode.RESOURCE_EXHAUSTED

    missing = SubprocessProofProvider(("/path/that/does/not/exist",)).prove({})
    assert missing.error is not None
    assert missing.error.code is ProviderFailureCode.UNAVAILABLE


def test_subprocess_cancellation_and_declared_network_policy(tmp_path: Path) -> None:
    script = _write_subprocess_fixture(tmp_path)
    client = SubprocessProofProvider(
        (sys.executable, str(script)),
        config=ProviderInvocationConfig(timeout_seconds=2, allow_network=False),
        expected_capability=_capability(),
    )
    token = CancellationToken()
    threading.Timer(0.03, token.cancel).start()
    response = client.prove({"mode": "sleep"}, cancellation=token)
    assert response.error is not None
    assert response.error.code is ProviderFailureCode.CANCELLED

    network_client = SubprocessProofProvider(
        (sys.executable, str(script)),
        config=ProviderInvocationConfig(allow_network=False),
        expected_capability=_capability(network_required=True),
    )
    denied = network_client.prove({}, network_allowed=True)
    assert denied.error is not None
    assert denied.error.code is ProviderFailureCode.NETWORK_DENIED


def test_stdio_server_emits_protocol_response_and_rejects_bad_requests() -> None:
    request = ProviderRequest(
        operation=ProofProviderOperation.VERIFY,
        payload={"candidate": "proof"},
        request_id="stdio-request",
    )
    output = io.BytesIO()
    assert (
        serve_provider_json(
            FixtureProvider(),
            input_stream=io.BytesIO(json.dumps(request.to_dict()).encode()),
            output_stream=output,
        )
        == 0
    )
    response = ProviderResponse.from_dict(json.loads(output.getvalue()))
    assert response.ok
    assert response.request_id == request.request_id
    assert response.operation is ProofProviderOperation.VERIFY

    malformed_output = io.BytesIO()
    malformed_envelope = request.to_dict()
    malformed_envelope["protocol_version"] = True
    assert (
        serve_provider_json(
            FixtureProvider(),
            input_stream=io.BytesIO(json.dumps(malformed_envelope).encode()),
            output_stream=malformed_output,
        )
        == 0
    )
    malformed_response = ProviderResponse.from_dict(
        json.loads(malformed_output.getvalue())
    )
    assert malformed_response.error is not None
    assert malformed_response.error.code is ProviderFailureCode.MALFORMED_REQUEST

    assert (
        serve_provider_json(
            FixtureProvider(),
            input_stream=io.BytesIO(b"not json"),
            output_stream=io.BytesIO(),
        )
        == 2
    )


def test_registry_discovers_entry_points_without_loading_optional_package() -> None:
    loaded: list[str] = []

    class FakeEntryPoint:
        name = "optional"
        group = "test.proof.providers"

        def load(self) -> FixtureProvider:
            loaded.append("loaded")
            return FixtureProvider()

    class FakeEntryPoints(tuple):
        def select(self, *, group: str) -> tuple[FakeEntryPoint, ...]:
            assert group == "test.proof.providers"
            return (FakeEntryPoint(),)

    registry = ProofProviderRegistry(
        entry_point_group="test.proof.providers",
        environ={},
        entry_points=lambda: FakeEntryPoints(),
    )
    registrations = registry.discover()
    assert [registration.provider_id for registration in registrations] == ["optional"]
    assert loaded == []

    client = registry.client("optional")
    assert client is not None
    assert loaded == []
    assert client.capability({}).ok
    assert loaded == ["loaded"]


def test_missing_provider_is_normal_and_does_not_import_ipfs_datasets_py() -> None:
    def no_entry_points() -> SimpleNamespace:
        return SimpleNamespace(select=lambda **_: ())

    registry = ProofProviderRegistry(environ={}, entry_points=no_entry_points)
    assert registry.discover() == ()
    assert registry.client("ipfs-datasets") is None
    assert "ipfs_datasets_py" not in sys.modules
