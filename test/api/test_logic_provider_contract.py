"""Cross-package contract tests for the canonical logic-provider boundary."""

from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.logic_provider_contract import (
    SupervisorLogicProviderFacade,
    to_logic_provider_request,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderOperation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    PROOF_PROVIDER_REQUEST_SCHEMA,
    CancellationToken,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
    dispatch_provider_request,
)
from ipfs_datasets_py.logic.backends.provider import (
    LOGIC_PROVIDER_REQUEST_SCHEMA,
    LOGIC_PROVIDER_RESPONSE_SCHEMA,
    LogicProvider,
    LogicProviderContractError,
    LogicProviderFailureCode,
    LogicProviderRequest,
    LogicProviderResponse,
    ProviderCancellation,
    ProviderResourceBudget,
    dispatch_logic_provider_request,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_PROVIDER_SOURCE = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "provider.py"
)


class FixtureLogicProvider:
    provider_id = "fixture.logic"
    provider_version = "1.2.3"
    protocol_version = 1

    def __init__(self) -> None:
        self.requests: list[LogicProviderRequest] = []

    def _invoke(self, request: LogicProviderRequest) -> dict[str, object]:
        self.requests.append(request)
        return {
            "echo": dict(request.payload),
            "operation": request.operation.value,
            "provider_claimed_authority": "authoritative",
        }

    capability = _invoke
    translate = _invoke
    prove = _invoke
    reconstruct = _invoke
    verify = _invoke
    attest = _invoke


def _canonical_request(**overrides: object) -> LogicProviderRequest:
    values: dict[str, object] = {
        "operation": "prove",
        "request_id": "request-canonical-1",
        "payload": {"obligation_id": "obligation:1", "premises": ["p", "p=>q"]},
        "resource_budget": ProviderResourceBudget(
            wall_time_ms=2_000,
            cpu_time_ms=1_000,
            memory_bytes=64 * 1024 * 1024,
            disk_bytes=1_024,
            max_processes=2,
            max_premises=8,
            max_output_bytes=4_096,
            model_token_limit=128,
            provider_quota=1,
            network_allowed=True,
        ),
        "cancellation": ProviderCancellation(
            cancellation_id="cancel:request-canonical-1"
        ),
        "network_allowed": True,
        "deadline_unix_ms": 4_102_444_800_000,
    }
    values.update(overrides)
    return LogicProviderRequest(**values)


def _supervisor_request(**overrides: object) -> ProviderRequest:
    values: dict[str, object] = {
        "operation": ProofProviderOperation.PROVE,
        "request_id": "request-supervisor-1",
        "payload": {"obligation_id": "obligation:supervisor"},
        "resource_budget": ResourceBudget(
            wall_time_ms=2_000,
            cpu_time_ms=1_000,
            memory_bytes=32 * 1024 * 1024,
            disk_bytes=2_048,
            max_processes=2,
            max_premises=16,
            max_output_bytes=8_192,
            model_token_limit=256,
            provider_quota=1,
            network_allowed=True,
        ),
        "network_allowed": True,
        "deadline_unix_ms": 4_102_444_800_000,
    }
    values.update(overrides)
    return ProviderRequest(**values)


def test_dataset_request_response_round_trip_is_canonical_and_lossless() -> None:
    request = _canonical_request()

    assert request.to_dict()["schema_version"] == LOGIC_PROVIDER_REQUEST_SCHEMA
    assert LogicProviderRequest.from_json(request.to_json()) == request
    assert request.to_json() == json.dumps(
        request.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert request.resource_budget.to_dict() == {
        "schema_version": "ipfs_datasets_py/logic-provider-resource-budget@1",
        "wall_time_ms": 2_000,
        "cpu_time_ms": 1_000,
        "memory_bytes": 64 * 1024 * 1024,
        "disk_bytes": 1_024,
        "max_processes": 2,
        "max_premises": 8,
        "max_output_bytes": 4_096,
        "model_token_limit": 128,
        "provider_quota": 1,
        "network_allowed": True,
    }
    assert request.cancellation is not None
    assert request.cancellation.cancelled is False

    response = LogicProviderResponse.success(
        request,
        {"candidate_id": "candidate:1", "proof_attempted": True},
        provider_id="fixture.logic",
        provider_version="1.2.3",
        duration_ms=4,
    )
    assert response.to_dict()["schema_version"] == LOGIC_PROVIDER_RESPONSE_SCHEMA
    assert LogicProviderResponse.from_json(response.to_json()) == response

    failure = LogicProviderResponse.failure(
        request,
        LogicProviderFailureCode.RESOURCE_EXHAUSTED,
        "memory budget exceeded",
        details={"limit": "memory_bytes"},
    )
    assert LogicProviderResponse.from_dict(failure.to_dict()) == failure


def test_wire_contract_fails_closed_on_ambiguous_or_unsafe_values() -> None:
    with pytest.raises(LogicProviderContractError, match="floating-point"):
        _canonical_request(payload={"score": 0.5})
    with pytest.raises(LogicProviderContractError, match="duplicate object key"):
        LogicProviderRequest.from_json(
            '{"schema_version":"ipfs_datasets_py/logic-provider-request@1",'
            '"schema_version":"ipfs_datasets_py/logic-provider-request@1"}'
        )
    with pytest.raises(LogicProviderContractError, match="unknown provider request"):
        LogicProviderRequest.from_dict(
            {**_canonical_request().to_dict(), "future_unreviewed_field": True}
        )
    with pytest.raises(LogicProviderContractError, match="exceeds"):
        _canonical_request(
            resource_budget=ProviderResourceBudget(network_allowed=False),
            network_allowed=True,
        )
    with pytest.raises(LogicProviderContractError, match="requires cancelled"):
        ProviderCancellation(
            cancellation_id="cancel:bad",
            cancelled=False,
            reason="not actually cancelled",
        )
    with pytest.raises(LogicProviderContractError, match="successful"):
        LogicProviderResponse(
            request_id="request:bad",
            operation="prove",
            ok=True,
            result={"candidate": True},
            error={
                "code": "provider_error",
                "message": "contradictory failure",
            },
        )


def test_dataset_dispatch_represents_cancellation_deadlines_and_correlation() -> None:
    provider = FixtureLogicProvider()
    assert isinstance(provider, LogicProvider)

    cancelled = dispatch_logic_provider_request(
        provider,
        _canonical_request(
            request_id="request-cancelled",
            cancellation=ProviderCancellation(
                cancellation_id="cancel:request-cancelled",
                cancelled=True,
                reason="caller stopped",
            ),
        ),
    )
    assert cancelled.error is not None
    assert cancelled.error.code is LogicProviderFailureCode.CANCELLED
    assert provider.requests == []

    expired = dispatch_logic_provider_request(
        provider,
        _canonical_request(
            request_id="request-expired",
            deadline_unix_ms=0,
        ),
    )
    assert expired.error is not None
    assert expired.error.code is LogicProviderFailureCode.TIMED_OUT
    assert provider.requests == []

    class MismatchedProvider(FixtureLogicProvider):
        def prove(self, request: LogicProviderRequest) -> LogicProviderResponse:
            return LogicProviderResponse.success(
                _canonical_request(request_id="a-different-request"),
                {"candidate": True},
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )

    mismatch = dispatch_logic_provider_request(
        MismatchedProvider(), _canonical_request()
    )
    assert mismatch.error is not None
    assert mismatch.error.code is LogicProviderFailureCode.MALFORMED_RESPONSE

    class WrongProtocolProvider(FixtureLogicProvider):
        protocol_version = 2

    protocol_error = dispatch_logic_provider_request(
        WrongProtocolProvider(), _canonical_request()
    )
    assert protocol_error.error is not None
    assert protocol_error.error.code is LogicProviderFailureCode.PROTOCOL_ERROR


def test_supervisor_conversion_preserves_every_wire_resource_and_policy_field() -> None:
    supervisor_request = _supervisor_request()

    canonical = to_logic_provider_request(supervisor_request)

    assert canonical.request_id == supervisor_request.request_id
    assert canonical.operation.value == supervisor_request.operation.value
    assert canonical.payload == supervisor_request.payload
    assert canonical.deadline_unix_ms == supervisor_request.deadline_unix_ms
    assert canonical.network_allowed is True
    expected_budget = supervisor_request.resource_budget.to_dict()
    expected_budget.pop("schema")
    actual_budget = canonical.resource_budget.to_dict()
    actual_budget.pop("schema_version")
    assert actual_budget == expected_budget
    # The existing supervisor contract remains unchanged and is adapted rather
    # than replaced by the new datasets schema.
    assert supervisor_request.to_dict()["schema_version"] == PROOF_PROVIDER_REQUEST_SCHEMA


def test_supervisor_facade_is_additive_and_does_not_promote_provider_claims() -> None:
    provider = FixtureLogicProvider()
    facade = SupervisorLogicProviderFacade(
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        provider=provider,
    )
    request = _supervisor_request()

    response = dispatch_provider_request(facade, request)

    assert isinstance(response, ProviderResponse)
    assert response.ok
    assert response.request_id == request.request_id
    assert response.provider_id == provider.provider_id
    assert response.result is not None
    assert response.result["echo"] == request.payload
    assert response.result["provider_claimed_authority"] == "authoritative"
    assert "authoritative_assurance" not in response.to_dict()
    assert provider.requests[0].resource_budget.memory_bytes == 32 * 1024 * 1024

    token = CancellationToken()
    token.cancel()
    cancelled = facade.invoke(_supervisor_request(request_id="cancel-me"), cancellation=token)
    assert cancelled.error is not None
    assert cancelled.error.code is ProviderFailureCode.CANCELLED
    assert len(provider.requests) == 1


def test_provider_reference_and_contract_discovery_stay_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "lfv_lazy_logic_provider_fixture"
    marker = tmp_path / "provider-imported"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        f"""
from pathlib import Path
Path({str(marker)!r}).write_text("imported", encoding="utf-8")
from ipfs_datasets_py.logic.backends.provider import LogicProviderResponse

class Provider:
    provider_id = "lazy.fixture"
    provider_version = "1"
    protocol_version = 1

    def prove(self, request):
        return LogicProviderResponse.success(
            request,
            {{"loaded_only_for": request.operation.value}},
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )

provider = Provider()
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    facade = SupervisorLogicProviderFacade.from_reference(
        f"{module_name}:provider",
        provider_id="lazy.fixture",
        provider_version="1",
    )
    assert facade.loaded is False
    assert module_name not in sys.modules
    assert not marker.exists()

    response = facade.invoke(_supervisor_request(request_id="lazy-request"))

    assert response.ok
    assert response.result == {"loaded_only_for": "prove"}
    assert facade.loaded is True
    assert marker.read_text(encoding="utf-8") == "imported"
    importlib.invalidate_caches()


def test_datasets_contract_has_no_parent_package_dependency() -> None:
    tree = ast.parse(DATASETS_PROVIDER_SOURCE.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert "ipfs_accelerate_py" not in imported_roots
