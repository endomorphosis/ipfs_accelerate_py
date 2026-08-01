"""SCA-171: model-server route and inference contract extraction tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.model_server_contract_extractor import (
    CANONICAL_JSON_RPC_SELECTORS,
    MODEL_SERVER_CONTRACT_CATALOG_INTERFACE,
    MODEL_SERVER_CONTRACT_EXTRACTOR_INTERFACE,
    REQUIRED_INFERENCE_FIELDS,
    AgreementState,
    InvocationMode,
    ModelServerContractExtractor,
    ModelServerContractExtractorError,
    ModelServerRouteKind,
    PreservationState,
    ProofEligibility,
    ReviewedAdapter,
    RouteSurface,
    compare_route_tables,
    extract_fastapi_routes_from_source,
    extract_mcp_tools_from_source,
    extract_model_server_contracts,
    extract_typescript_jsonrpc_routes_from_source,
    preserve_inference_fields,
)
from ipfs_accelerate_py.agent_supervisor.analysis.runtime_component_catalog import (
    ImplementationAuthorityKind,
    RuntimeComponentKind,
    build_runtime_component_catalog,
)


def _route(
    *,
    surface: str,
    kind: str,
    transport: str,
    selector: str,
    source_path: str = "src/fixture.ts",
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "surface": surface,
        "kind": kind,
        "transport": transport,
        "selector": selector,
        "source_path": source_path,
        "source_ids": [f"{source_path}:{selector}"],
    }
    payload.update(extra)
    return payload


def _matching_launcher_connector() -> dict[str, object]:
    operational = [
        _route(
            surface="launcher",
            kind="health",
            transport="http",
            selector="/health/ready",
            source_path="src/entrypoints/mcp.ts",
        ),
        _route(
            surface="launcher",
            kind="list",
            transport="json-rpc",
            selector="tools/list",
            source_path="src/entrypoints/mcp.ts",
        ),
        _route(
            surface="launcher",
            kind="call",
            transport="json-rpc",
            selector="tools/call",
            source_path="src/entrypoints/mcp.ts",
        ),
    ]
    connector = [
        {
            **item,
            "surface": "connector",
            "source_path": "src/services/mcp/mcp-plus-plus-connector.ts",
            "source_ids": [
                f"src/services/mcp/mcp-plus-plus-connector.ts:{item['selector']}"
            ],
        }
        for item in operational
    ]
    return {
        "component_id": "model-server",
        "launcher_routes": operational,
        "connector_routes": connector,
    }


def _runtime_catalog_payload() -> dict[str, object]:
    return {
        "components": [
            {
                "componentId": "model-server",
                "displayName": "SwissKnife MCP model server",
                "kind": "model_server",
                "implementationSymbol": "startMCPServer",
                "sourcePath": "src/entrypoints/mcp.ts",
                "routeProfileId": "mcp-plus-plus-v1",
                "authority": {
                    "kind": "primary",
                    "canonicalComponentId": "model-server",
                    "decision": "canonical_runtime_root",
                    "sourcePath": "src/entrypoints/mcp.ts",
                },
            },
            {
                "componentId": "orchestrator",
                "displayName": "orchestrator",
                "kind": "orchestrator",
                "implementationSymbol": "MCPCapabilityRouter",
                "sourcePath": "src/services/mcp/mcp-orb-capability-router.ts",
                "routeProfileId": "mcp-plus-plus-v1",
                "authority": {
                    "kind": "primary",
                    "canonicalComponentId": "orchestrator",
                    "decision": "canonical_runtime_root",
                    "sourcePath": "src/services/mcp/mcp-orb-capability-router.ts",
                },
            },
            {
                "componentId": "scheduler",
                "displayName": "scheduler",
                "kind": "scheduler",
                "implementationSymbol": "MCPScheduler",
                "sourcePath": "src/services/mcp/mcp-scheduler.ts",
                "routeProfileId": "mcp-plus-plus-v1",
                "authority": {
                    "kind": "primary",
                    "canonicalComponentId": "scheduler",
                    "decision": "canonical_runtime_root",
                    "sourcePath": "src/services/mcp/mcp-scheduler.ts",
                },
            },
            {
                "componentId": "supervisor",
                "displayName": "supervisor",
                "kind": "supervisor",
                "implementationSymbol": "MCPServerController",
                "sourcePath": "src/patches/mcp/mcp-server-controller.ts",
                "routeProfileId": "mcp-plus-plus-v1",
                "authority": {
                    "kind": "primary",
                    "canonicalComponentId": "supervisor",
                    "decision": "canonical_runtime_root",
                    "sourcePath": "src/patches/mcp/mcp-server-controller.ts",
                },
            },
        ],
        "routeProfiles": [
            {
                "profileId": "mcp-plus-plus-v1",
                "routes": [
                    {
                        "kind": "connector",
                        "transport": "typescript",
                        "selector": "MCPPPServerConnector",
                        "sourcePath": "src/services/mcp/mcp-plus-plus-connector.ts",
                    },
                    {
                        "kind": "launcher",
                        "transport": "stdio",
                        "selector": "startMCPServer",
                        "sourcePath": "src/entrypoints/mcp.ts",
                    },
                    {
                        "kind": "health",
                        "transport": "http",
                        "selector": "/health/ready",
                        "sourcePath": "src/services/mcp/mcp-plus-plus-connector.ts",
                    },
                    {
                        "kind": "list",
                        "transport": "json-rpc",
                        "selector": "tools/list",
                        "sourcePath": "src/services/mcp/mcp-plus-plus-connector.ts",
                    },
                    {
                        "kind": "call",
                        "transport": "json-rpc",
                        "selector": "tools/call",
                        "sourcePath": "src/services/mcp/mcp-plus-plus-connector.ts",
                    },
                ],
            }
        ],
    }


def test_interfaces_and_required_fields_are_stable() -> None:
    assert MODEL_SERVER_CONTRACT_EXTRACTOR_INTERFACE == "ModelServerContractExtractor@1"
    assert MODEL_SERVER_CONTRACT_CATALOG_INTERFACE == "ModelServerContractCatalog@1"
    assert "tools/list" in CANONICAL_JSON_RPC_SELECTORS
    assert "tools/call" in CANONICAL_JSON_RPC_SELECTORS
    assert set(REQUIRED_INFERENCE_FIELDS) == {
        "model_id",
        "model_revision",
        "parameters",
        "result",
        "error",
        "provenance",
    }


def test_launcher_and_connector_route_tables_agree() -> None:
    catalog = extract_model_server_contracts(_matching_launcher_connector())

    agreement = catalog.launcher_connector_agreement()
    assert agreement is not None
    assert agreement.state is AgreementState.AGREED
    assert agreement.counterexamples == ()
    assert agreement.matched_route_ids
    kinds = {
        route.kind
        for route in catalog.routes
        if route.surface in {RouteSurface.LAUNCHER, RouteSurface.CONNECTOR}
    }
    assert ModelServerRouteKind.LIST in kinds
    assert ModelServerRouteKind.CALL in kinds
    assert ModelServerRouteKind.HEALTH in kinds


def test_launcher_connector_mismatch_emits_exact_counterexamples() -> None:
    payload = _matching_launcher_connector()
    # Connector drifts to a legacy REST call path.
    payload["connector_routes"] = [
        _route(
            surface="connector",
            kind="health",
            transport="http",
            selector="/health/ready",
            source_path="src/services/mcp/mcp-plus-plus-connector.ts",
        ),
        _route(
            surface="connector",
            kind="list",
            transport="json-rpc",
            selector="tools/list",
            source_path="src/services/mcp/mcp-plus-plus-connector.ts",
        ),
        _route(
            surface="connector",
            kind="call",
            transport="http",
            selector="/api/v0/inference",
            source_path="src/services/mcp/mcp-plus-plus-connector.ts",
        ),
    ]
    catalog = extract_model_server_contracts(payload)
    agreement = catalog.launcher_connector_agreement()

    assert agreement is not None
    assert agreement.state is AgreementState.REFUTED
    assert agreement.counterexamples
    call_cx = next(
        item
        for item in agreement.counterexamples
        if item.path == "routes.call"
    )
    assert call_cx.reason_code == "route_selector_mismatch"
    assert call_cx.expected["selector"] == "tools/call"
    assert call_cx.actual["selector"] == "/api/v0/inference"
    assert call_cx.expected["transport"] == "json-rpc"
    assert call_cx.actual["transport"] == "http"


def test_canonical_json_rpc_invocation_is_proof_eligible() -> None:
    catalog = extract_model_server_contracts(
        {
            "component_id": "model-server",
            "invocations": [
                {
                    "operation_id": "accelerate.inference",
                    "mode": "canonical_json_rpc",
                    "transport": "json-rpc",
                    "selector": "tools/call",
                    "surface": "connector",
                    "source_ids": ["connector:tools/call"],
                }
            ],
        }
    )

    assert len(catalog.invocations) == 1
    invocation = catalog.invocations[0]
    assert invocation.mode is InvocationMode.CANONICAL_JSON_RPC
    assert invocation.proof_eligibility is ProofEligibility.PROOF_ELIGIBLE
    assert invocation.can_prove_success is True
    assert catalog.proof_eligible_invocations() == (invocation,)


def test_reviewed_adapter_can_prove_when_bound() -> None:
    adapter = {
        "adapter_id": "compat-v1",
        "from_surface": "compatibility_adapter",
        "to_surface": "connector",
        "version": "1.0.0",
        "review_id": "review:model-server-compat-1",
        "source_ids": ["docs/adapters/compat-v1.md"],
        "maps": {"tools/call": "tools/call"},
    }
    catalog = extract_model_server_contracts(
        {
            "component_id": "model-server",
            "reviewed_adapters": [adapter],
            "invocations": [
                {
                    "operation_id": "accelerate.inference",
                    "mode": "reviewed_adapter",
                    "transport": "adapter",
                    "selector": "tools/call",
                    "surface": "compatibility_adapter",
                    "adapter_id": "compat-v1",
                    "source_ids": ["compat:tools/call"],
                }
            ],
        }
    )
    invocation = catalog.invocations[0]
    assert invocation.mode is InvocationMode.REVIEWED_ADAPTER
    assert invocation.adapter_identity
    assert invocation.can_prove_success is True
    assert catalog.reviewed_adapters[0].version == "1.0.0"


def test_reviewed_adapter_without_binding_cannot_prove() -> None:
    catalog = extract_model_server_contracts(
        {
            "invocations": [
                {
                    "operation_id": "accelerate.inference",
                    "mode": "reviewed_adapter",
                    "transport": "adapter",
                    "selector": "tools/call",
                    "surface": "compatibility_adapter",
                    "source_ids": ["compat:unbound"],
                }
            ]
        }
    )
    invocation = catalog.invocations[0]
    assert invocation.can_prove_success is False
    assert invocation.proof_eligibility is ProofEligibility.NON_PROVING


def test_mock_degraded_and_synthesized_aliases_cannot_prove_success() -> None:
    catalog = extract_model_server_contracts(
        {
            "component_id": "model-server",
            "connector_routes": [
                _route(
                    surface="connector",
                    kind="call",
                    transport="json-rpc",
                    selector="tools/call",
                    mock=True,
                ),
                _route(
                    surface="connector",
                    kind="list",
                    transport="json-rpc",
                    selector="tools/list",
                    degraded=True,
                ),
            ],
            "invocations": [
                {
                    "operation_id": "alias.inference",
                    "mode": "synthesized_alias",
                    "transport": "alias",
                    "selector": "fake.tools/call",
                    "surface": "connector",
                    "source_ids": ["synth:1"],
                    # Explicit claim is ignored for non-proving modes.
                    "can_prove_success": True,
                }
            ],
        }
    )

    modes = {item.mode for item in catalog.invocations}
    assert InvocationMode.MOCK_TRANSPORT in modes
    assert InvocationMode.DEGRADED_TRANSPORT in modes
    assert InvocationMode.SYNTHESIZED_ALIAS in modes
    assert catalog.proof_eligible_invocations() == ()
    assert all(not item.can_prove_success for item in catalog.invocations)
    assert all(
        item.proof_eligibility is ProofEligibility.NON_PROVING
        for item in catalog.invocations
    )


def test_model_id_revision_parameters_result_error_provenance_preserved() -> None:
    catalog = extract_model_server_contracts(
        {
            "component_id": "model-server",
            "inference_contracts": [
                {
                    "operation_id": "accelerate.inference",
                    "model_id": "meta-llama/Llama-3-8B",
                    "model_revision": "rev-abc123",
                    "parameters": {
                        "temperature": 0,
                        "max_tokens": 128,
                        "top_p": 1,
                    },
                    "consumer_fields": {
                        "model_id": "meta-llama/Llama-3-8B",
                        "model_revision": "rev-abc123",
                        "parameters": {
                            "temperature": 0,
                            "max_tokens": 128,
                            "top_p": 1,
                        },
                        "result": {
                            "outputs": "string",
                            "output_cid": "cid",
                        },
                        "error": {"code": "string", "message": "string"},
                        "provenance": {
                            "provenance_cid": "cid",
                            "backend": "string",
                        },
                    },
                    "handler_fields": {
                        "model_id": "meta-llama/Llama-3-8B",
                        "model_revision": "rev-abc123",
                        "parameters": {
                            "temperature": 0,
                            "max_tokens": 128,
                            "top_p": 1,
                        },
                        "result": {
                            "outputs": "string",
                            "output_cid": "cid",
                        },
                        "error": {"code": "string", "message": "string"},
                        "provenance": {
                            "provenance_cid": "cid",
                            "backend": "string",
                        },
                    },
                    "source_ids": ["fixture:inference"],
                }
            ],
        }
    )

    contract = catalog.inference_contracts[0]
    assert contract.model_id == "meta-llama/Llama-3-8B"
    assert contract.model_revision == "rev-abc123"
    assert contract.parameters["max_tokens"] == 128
    assert contract.all_fields_preserved is True
    paths = {item.field_path: item.state for item in contract.preservations}
    for required in REQUIRED_INFERENCE_FIELDS:
        assert paths[required] is PreservationState.PRESERVED


def test_dropped_model_revision_is_refuted_with_counterexample() -> None:
    catalog = extract_model_server_contracts(
        {
            "inference_contracts": [
                {
                    "operation_id": "accelerate.inference",
                    "model_id": "gpt2",
                    "model_revision": "main",
                    "parameters": {"temperature": 0},
                    "consumer_fields": {
                        "model_id": "gpt2",
                        "model_revision": "main",
                        "parameters": {"temperature": 0},
                        "result": {"text": "ok"},
                        "error": {"code": "E"},
                        "provenance": {"cid": "p"},
                    },
                    "handler_fields": {
                        "model_id": "gpt2",
                        # model_revision dropped
                        "parameters": {"temperature": 0},
                        "result": {"text": "ok"},
                        "error": {"code": "E"},
                        "provenance": {"cid": "p"},
                    },
                }
            ]
        }
    )
    contract = catalog.inference_contracts[0]
    revision = next(
        item
        for item in contract.preservations
        if item.field_path == "model_revision"
    )
    assert revision.state is PreservationState.REFUTED
    assert revision.counterexamples
    assert revision.counterexamples[0].expected == "main"
    assert revision.counterexamples[0].actual is None
    assert contract.all_fields_preserved is False


def test_parameter_mutation_is_refuted() -> None:
    preservations = preserve_inference_fields(
        operation_id="op",
        consumer_fields={
            "model_id": "m",
            "model_revision": "r",
            "parameters": {"temperature": 0, "max_tokens": 32},
            "result": {"ok": True},
            "error": {},
            "provenance": {"cid": "x"},
        },
        handler_fields={
            "model_id": "m",
            "model_revision": "r",
            "parameters": {"temperature": 1, "max_tokens": 32},
            "result": {"ok": True},
            "error": {},
            "provenance": {"cid": "x"},
        },
    )
    params = next(item for item in preservations if item.field_path == "parameters")
    assert params.state is PreservationState.REFUTED
    assert params.counterexamples[0].reason_code == "field_value_mismatch"
    assert params.counterexamples[0].expected["temperature"] == 0
    assert params.counterexamples[0].actual["temperature"] == 1


def test_surfaces_cover_connector_registry_servers_and_native_tools() -> None:
    catalog = extract_model_server_contracts(
        {
            "component_id": "model-server",
            "capability_registry_routes": [
                _route(
                    surface="capability_registry",
                    kind="call",
                    transport="json-rpc",
                    selector="tools/call",
                    source_path="src/services/apps/registry.ts",
                )
            ],
            "cli_routes": [
                _route(
                    surface="cli_launcher",
                    kind="launcher",
                    transport="stdio",
                    selector="startMCPServer",
                    source_path="src/entrypoints/cli.tsx",
                )
            ],
            "flask_routes": [
                _route(
                    surface="flask_server",
                    kind="completions",
                    transport="http",
                    selector="/v1/completions",
                    source_path="external/ipfs_accelerate/.../legacy_flask.py",
                )
            ],
            "integrated_routes": [
                _route(
                    surface="integrated_server",
                    kind="chat",
                    transport="http",
                    selector="/v1/chat/completions",
                    source_path="external/ipfs_accelerate/.../unified.py",
                )
            ],
            "mcp_plus_plus_routes": [
                _route(
                    surface="mcp_plus_plus_server",
                    kind="list",
                    transport="json-rpc",
                    selector="tools/list",
                    source_path="src/services/mcp/mcp-plus-plus.ts",
                )
            ],
            "compatibility_adapter_routes": [
                _route(
                    surface="compatibility_adapter",
                    kind="call",
                    transport="http",
                    selector="/api/v0/inference",
                    source_path="src/patches/mcp/fix-mcp-entrypoint.ts",
                    mode="compatibility",
                )
            ],
            "hf_routes": [
                _route(
                    surface="hf_model_server",
                    kind="embeddings",
                    transport="http",
                    selector="/v1/embeddings",
                    source_path="ipfs_accelerate_py/hf_model_server/server.py",
                    function_symbol="create_embedding",
                )
            ],
            "mcp_ai_routes": [
                _route(
                    surface="mcp_ai_model_server",
                    kind="call",
                    transport="mcp",
                    selector="list_models",
                    source_path="ipfs_accelerate_py/mcp/ai_model_server.py",
                    function_symbol="list_models",
                    mode="canonical_json_rpc",
                )
            ],
            "native_tool_routes": [
                _route(
                    surface="native_model_tool",
                    kind="call",
                    transport="mcp",
                    selector="causal_lm",
                    source_path="ipfs_accelerate_py/mcp/inference_tools.py",
                    function_symbol="causal_lm",
                    mode="canonical_json_rpc",
                )
            ],
            "connector_routes": [
                _route(
                    surface="connector",
                    kind="call",
                    transport="json-rpc",
                    selector="tools/call",
                    source_path="src/services/mcp/mcp-plus-plus-connector.ts",
                )
            ],
        }
    )

    surfaces = {route.surface for route in catalog.routes}
    assert RouteSurface.CAPABILITY_REGISTRY in surfaces
    assert RouteSurface.CLI_LAUNCHER in surfaces
    assert RouteSurface.FLASK_SERVER in surfaces
    assert RouteSurface.INTEGRATED_SERVER in surfaces
    assert RouteSurface.MCP_PLUS_PLUS_SERVER in surfaces
    assert RouteSurface.COMPATIBILITY_ADAPTER in surfaces
    assert RouteSurface.HF_MODEL_SERVER in surfaces
    assert RouteSurface.MCP_AI_MODEL_SERVER in surfaces
    assert RouteSurface.NATIVE_MODEL_TOOL in surfaces
    # Exact function identities retained.
    by_selector = {route.selector: route for route in catalog.routes}
    assert by_selector["list_models"].function_symbol == "list_models"
    assert by_selector["/v1/embeddings"].function_symbol == "create_embedding"


def test_static_fastapi_and_mcp_source_extraction() -> None:
    fastapi_src = '''
class HFModelServer:
    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {"ok": True}

        @self.app.post("/v1/completions")
        async def completions():
            return {}

        @self.app.post("/v1/chat/completions")
        async def chat():
            return {}

        @self.app.get("/v1/models")
        async def models():
            return {}
'''
    mcp_src = '''
class AIModelServer:
    def _register_tools(self):
        @self.mcp.tool()
        def list_models(limit: int = 10):
            return []

        @self.mcp.tool(name="recommend_model")
        def recommend(task: str):
            return {"model_id": "x"}
'''
    fastapi_routes = extract_fastapi_routes_from_source(
        fastapi_src, source_path="hf_model_server/server.py"
    )
    mcp_routes = extract_mcp_tools_from_source(
        mcp_src, source_path="mcp/ai_model_server.py"
    )

    selectors = {route.selector for route in fastapi_routes}
    assert "/health" in selectors
    assert "/v1/completions" in selectors
    assert "/v1/chat/completions" in selectors
    assert "/v1/models" in selectors
    assert any(route.kind is ModelServerRouteKind.HEALTH for route in fastapi_routes)
    assert any(
        route.kind is ModelServerRouteKind.COMPLETIONS for route in fastapi_routes
    )

    tool_names = {route.selector for route in mcp_routes}
    assert "list_models" in tool_names
    assert "recommend_model" in tool_names
    assert all(
        route.invocation_mode is InvocationMode.CANONICAL_JSON_RPC
        for route in mcp_routes
    )


def test_static_typescript_jsonrpc_extraction() -> None:
    source = """
export class Connector {
  async list() { return this.jsonRpc('tools/list', {}); }
  async call(name: string, args: object) {
    return this.jsonRpc('tools/call', { name, arguments: args });
  }
  health() { return fetch('/health/ready'); }
}
"""
    routes = extract_typescript_jsonrpc_routes_from_source(
        source,
        source_path="src/services/mcp/mcp-plus-plus-connector.ts",
        surface=RouteSurface.CONNECTOR,
    )
    by_selector = {route.selector: route for route in routes}
    assert by_selector["tools/list"].invocation_mode is InvocationMode.CANONICAL_JSON_RPC
    assert by_selector["tools/call"].proof_eligibility is ProofEligibility.PROOF_ELIGIBLE
    assert by_selector["/health/ready"].kind is ModelServerRouteKind.HEALTH


def test_source_payload_extraction_end_to_end() -> None:
    catalog = extract_model_server_contracts(
        {
            "component_id": "model-server",
            "sources": {
                "src/services/mcp/mcp-plus-plus-connector.ts": {
                    "language": "typescript",
                    "surface": "connector",
                    "text": (
                        "this.jsonRpc('tools/list', {});"
                        "this.jsonRpc('tools/call', {});"
                        "fetch('/health/ready');"
                    ),
                },
                "ipfs_accelerate_py/hf_model_server/server.py": {
                    "language": "python",
                    "surface": "hf_model_server",
                    "text": (
                        '@self.app.get("/health")\n'
                        "async def health(): pass\n"
                        '@self.app.post("/v1/completions")\n'
                        "async def completions(): pass\n"
                    ),
                },
                "ipfs_accelerate_py/mcp/ai_model_server.py": {
                    "language": "python",
                    "surface": "mcp_ai_model_server",
                    "text": (
                        "class S:\n"
                        "    def r(self):\n"
                        "        @self.mcp.tool()\n"
                        "        def list_models():\n"
                        "            return []\n"
                    ),
                },
            },
        }
    )
    surfaces = {route.surface for route in catalog.routes}
    assert RouteSurface.CONNECTOR in surfaces
    assert RouteSurface.HF_MODEL_SERVER in surfaces
    assert RouteSurface.MCP_AI_MODEL_SERVER in surfaces
    assert any(route.selector == "tools/call" for route in catalog.routes)
    assert any(route.selector == "list_models" for route in catalog.routes)


def test_runtime_catalog_binding_pins_component_root_cid() -> None:
    runtime = build_runtime_component_catalog(_runtime_catalog_payload())
    catalog = extract_model_server_contracts(
        {
            "component_id": "model-server",
            "connector_routes": [
                _route(
                    surface="connector",
                    kind="call",
                    transport="json-rpc",
                    selector="tools/call",
                )
            ],
        },
        runtime_catalog=runtime,
    )

    assert catalog.component_id == "model-server"
    assert catalog.component_root_cid == runtime.component("model-server").root_cid
    assert catalog.component_root_cid.startswith("b")
    # Runtime profile routes are merged in.
    assert any(route.selector == "tools/list" for route in catalog.routes)
    assert any(route.selector == "startMCPServer" for route in catalog.routes)
    primary = runtime.component("model-server")
    assert primary.kind is RuntimeComponentKind.MODEL_SERVER
    assert primary.authority.kind is ImplementationAuthorityKind.PRIMARY


def test_catalog_is_content_addressed_and_order_independent() -> None:
    payload = _matching_launcher_connector()
    payload["inference_contracts"] = [
        {
            "operation_id": "accelerate.inference",
            "model_id": "m",
            "model_revision": "r",
            "parameters": {"temperature": 0},
            "consumer_fields": {
                "model_id": "m",
                "model_revision": "r",
                "parameters": {"temperature": 0},
                "result": {"ok": True},
                "error": {},
                "provenance": {"cid": "p"},
            },
            "handler_fields": {
                "model_id": "m",
                "model_revision": "r",
                "parameters": {"temperature": 0},
                "result": {"ok": True},
                "error": {},
                "provenance": {"cid": "p"},
            },
        }
    ]
    first = extract_model_server_contracts(payload)
    reversed_payload = {
        "component_id": payload["component_id"],
        "connector_routes": list(reversed(payload["connector_routes"])),
        "launcher_routes": list(reversed(payload["launcher_routes"])),
        "inference_contracts": payload["inference_contracts"],
    }
    second = extract_model_server_contracts(reversed_payload)

    assert first.catalog_id.startswith("b")
    assert first.catalog_id == second.catalog_id
    assert first.to_dict() == second.to_dict()


def test_compare_route_tables_missing_kind_counterexample() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis.model_server_contract_extractor import (
        ModelServerRoute,
    )

    left = (
        ModelServerRoute(
            surface=RouteSurface.LAUNCHER,
            kind=ModelServerRouteKind.CALL,
            transport="json-rpc",
            selector="tools/call",
            source_path="launcher.ts",
            invocation_mode=InvocationMode.CANONICAL_JSON_RPC,
            proof_eligibility=ProofEligibility.PROOF_ELIGIBLE,
            source_ids=("l:call",),
        ),
    )
    right = (
        ModelServerRoute(
            surface=RouteSurface.CONNECTOR,
            kind=ModelServerRouteKind.LIST,
            transport="json-rpc",
            selector="tools/list",
            source_path="connector.ts",
            invocation_mode=InvocationMode.CANONICAL_JSON_RPC,
            proof_eligibility=ProofEligibility.PROOF_ELIGIBLE,
            source_ids=("c:list",),
        ),
    )
    agreement = compare_route_tables(
        left,
        right,
        left_surface=RouteSurface.LAUNCHER,
        right_surface=RouteSurface.CONNECTOR,
    )
    assert agreement.state is AgreementState.REFUTED
    reasons = {item.reason_code for item in agreement.counterexamples}
    assert "route_missing_on_right" in reasons
    assert "route_missing_on_left" in reasons


def test_malformed_payload_fails_closed() -> None:
    with pytest.raises(ModelServerContractExtractorError):
        extract_model_server_contracts([])  # type: ignore[arg-type]

    with pytest.raises(ModelServerContractExtractorError):
        ReviewedAdapter(
            adapter_id="a",
            from_surface="x",
            to_surface="y",
            version="1",
            review_id="r",
            source_ids=(),
        )


def test_extractor_class_matches_module_helper() -> None:
    payload = _matching_launcher_connector()
    via_class = ModelServerContractExtractor().extract(payload)
    via_helper = extract_model_server_contracts(payload)
    assert via_class.catalog_id == via_helper.catalog_id


def test_direct_rest_is_visible_but_non_proving() -> None:
    catalog = extract_model_server_contracts(
        {
            "hf_routes": [
                _route(
                    surface="hf_model_server",
                    kind="completions",
                    transport="http",
                    selector="/v1/completions",
                    source_path="hf_model_server/server.py",
                )
            ]
        }
    )
    invocation = next(
        item for item in catalog.invocations if item.selector == "/v1/completions"
    )
    assert invocation.mode is InvocationMode.DIRECT_REST
    assert invocation.can_prove_success is False
    assert "direct_rest_non_mcp_proof" in invocation.reason_codes
