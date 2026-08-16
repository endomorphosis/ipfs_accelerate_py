"""DCR-023 hermetic MCP observer tests: no network transport is constructed."""

from __future__ import annotations

from dataclasses import dataclass, field

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    build_mcp_contract_graph,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    McpObservationTemplate,
    ObservationFailureCode,
    ObservationStatus,
    TemplateStatus,
    observe_mcp_template,
)
from ipfs_accelerate_py.agent_supervisor.analysis.runtime_service_identity import (
    RuntimeServiceIdentity,
    RuntimeServiceObservation,
    ServiceIdentityStatus,
)


@dataclass
class FakeLocalTransport:
    response: bytes = b'{"jsonrpc":"2.0","result":{}}'
    calls: list[tuple[str, bytes]] = field(default_factory=list)

    def exchange(self, *, endpoint: str, request: bytes) -> bytes:
        self.calls.append((endpoint, request))
        return self.response


def _graph() -> dict[str, object]:
    span = {"root": "desktop", "path": "app.ts", "sha256": "sha256:x"}
    desktop = {
        "consumers": [{"root": "desktop", "path": "app.ts", "sha256": "sha256:x"}],
        "evidence": [
            {"operation": "catalog.read", "declaration_kind": "orb_idl", "source_span": span}
        ],
        "effective_expectations": [
            {"operation": "catalog.read", "request": "CatalogRead", "source_span": span}
        ],
        "blockers": [],
    }
    provider = {
        "rows": [
            {
                "operation": "catalog.read",
                "status": "resolved",
                "dispatcher": "Dispatcher",
                "handler": "read_catalog",
                "effect": "catalog.read",
                "source_digest": "sha256:provider",
            }
        ]
    }
    identities = [
        {
            "semantic_cid": "bafy-semantic",
            "declaration_cid": "bafy-declaration",
            "semantic_key": {"operation": "catalog.read"},
        }
    ]
    return build_mcp_contract_graph(
        provider_surfaces=provider, desktop_expectations=desktop, identities=identities
    )


def _identity(
    status: ServiceIdentityStatus = ServiceIdentityStatus.VALID,
) -> RuntimeServiceIdentity:
    observation = RuntimeServiceObservation(
        role="accelerate",
        interpreter="/py",
        module_origin="/checkout/module.py",
        module_digest="sha256:module",
        checkout_commit="commit",
        checkout_tree="tree",
        overlay_id="overlay",
        argv=("/py", "serve"),
        environment={"MODE": "safe"},
        config_cid="bafy-config",
        state_cid="bafy-state",
        transport="mcp",
        endpoint="http://127.0.0.1:9010",
        pid=12,
        started_at="start",
        process_identity="diagnostic-only",
        observed_port=9010,
    )
    return RuntimeServiceIdentity(observation, status, ())


def _template(status: TemplateStatus = TemplateStatus.VALID) -> McpObservationTemplate:
    return McpObservationTemplate(
        operation="catalog.read",
        request_bytes=b'{"jsonrpc":"2.0","method":"catalog.read","params":{}}',
        status=status,
        read_only=True,
    )


def test_pending_identity_or_template_defers_before_transport() -> None:
    transport = FakeLocalTransport()
    pending_template = observe_mcp_template(
        graph=_graph(),
        runtime_identity=_identity(),
        template=_template(TemplateStatus.INTEGRATION_PENDING),
        transport=transport,
    )
    assert pending_template.status is ObservationStatus.DEFERRED
    assert pending_template.failure is ObservationFailureCode.INTEGRATION_PENDING
    pending_identity = observe_mcp_template(
        graph=_graph(),
        runtime_identity=_identity(ServiceIdentityStatus.INTEGRATION_PENDING),
        template=_template(),
        transport=transport,
    )
    assert pending_identity.status is ObservationStatus.DEFERRED
    assert transport.calls == []


def test_valid_graph_witness_and_template_capture_raw_bytes_without_empty_success() -> None:
    transport = FakeLocalTransport()
    result = observe_mcp_template(
        graph=_graph(), runtime_identity=_identity(), template=_template(), transport=transport
    )
    assert result.status is ObservationStatus.OBSERVED
    assert result.failure is None
    assert result.request_bytes == _template().request_bytes
    assert result.response_bytes == transport.response
    assert result.to_dict()["completion_authoritative"] is False
    assert len(transport.calls) == 1

    empty = observe_mcp_template(
        graph=_graph(),
        runtime_identity=_identity(),
        template=_template(),
        transport=FakeLocalTransport(b""),
    )
    assert empty.status is ObservationStatus.UNKNOWN_RESPONSE
    assert empty.failure is ObservationFailureCode.EMPTY_RESPONSE


def test_invalid_graph_remote_endpoint_and_dynamic_template_never_invoke_transport() -> None:
    transport = FakeLocalTransport()
    graph = _graph()
    graph["graph_cid"] = "forged"
    invalid_graph = observe_mcp_template(
        graph=graph, runtime_identity=_identity(), template=_template(), transport=transport
    )
    assert invalid_graph.failure is ObservationFailureCode.GRAPH_INVALID

    remote_observation = _identity().observation
    remote_identity = RuntimeServiceIdentity(
        RuntimeServiceObservation(
            **{**remote_observation.__dict__, "endpoint": "http://remote.invalid:9010"}
        ),
        ServiceIdentityStatus.VALID,
        (),
    )
    remote = observe_mcp_template(
        graph=_graph(), runtime_identity=remote_identity, template=_template(), transport=transport
    )
    assert remote.failure is ObservationFailureCode.ENDPOINT_REJECTED

    dynamic = McpObservationTemplate(
        operation="catalog.read",
        request_bytes=b'{"jsonrpc":"2.0","method":"catalog.read","params":{"user":"data"}}',
        status=TemplateStatus.VALID,
        read_only=True,
    )
    rejected = observe_mcp_template(
        graph=_graph(), runtime_identity=_identity(), template=dynamic, transport=transport
    )
    assert rejected.failure is ObservationFailureCode.TEMPLATE_INVALID
    assert transport.calls == []


def test_untyped_template_status_and_non_mcp_identity_are_rejected_without_transport() -> None:
    transport = FakeLocalTransport()
    untyped_status = McpObservationTemplate(
        operation="catalog.read",
        request_bytes=b'{"jsonrpc":"2.0","method":"catalog.read","params":{}}',
        status="valid",  # type: ignore[arg-type]  # Hostile untyped integration input.
        read_only=True,
    )
    rejected_template = observe_mcp_template(
        graph=_graph(),
        runtime_identity=_identity(),
        template=untyped_status,
        transport=transport,
    )
    assert rejected_template.failure is ObservationFailureCode.TEMPLATE_INVALID

    non_mcp = RuntimeServiceIdentity(
        RuntimeServiceObservation(**{**_identity().observation.__dict__, "transport": "http"}),
        ServiceIdentityStatus.VALID,
        (),
    )
    rejected_identity = observe_mcp_template(
        graph=_graph(), runtime_identity=non_mcp, template=_template(), transport=transport
    )
    assert rejected_identity.failure is ObservationFailureCode.IDENTITY_INVALID
    assert transport.calls == []
