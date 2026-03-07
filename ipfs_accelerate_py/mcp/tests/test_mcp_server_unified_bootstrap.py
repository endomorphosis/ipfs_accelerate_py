#!/usr/bin/env python3
"""Bootstrap tests for the new unified `ipfs_accelerate_py.mcp_server` package."""

import unittest
import os
import tempfile
import base64
import json
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server import (
    configure_wave_a_loaders,
    get_unified_meta_tool_names,
    get_unified_supported_profiles,
    get_unified_wave_a_categories,
    HierarchicalToolManager,
    RuntimeRouter,
    ToolMetadata,
    ToolRegistry,
    UnifiedMCPServerConfig,
    get_tool_metadata,
    register_tool_metadata,
    register_legacy_tools_into_manager,
)
from ipfs_accelerate_py.mcp_server.server import create_server, _parse_preload_categories
from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.exceptions import RuntimeExecutionError
from ipfs_accelerate_py.mcp_server.mcplusplus.delegation import (
    HAVE_CRYPTO_ED25519,
    compute_delegation_proof_cid,
    compute_delegation_signature,
    compute_delegation_signature_ed25519,
    parse_delegation_chain,
)


class TestUnifiedMCPServerBootstrap(unittest.TestCase):
    """Validate core integration points for incremental mcp_server migration."""

    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        """Assert canonical success envelope and return the inner result payload."""
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    def test_registry_registers_function_tools(self):
        """ToolRegistry should wrap and index plain functions."""
        registry = ToolRegistry()

        def hello(name: str):
            return {"hello": name}

        registry.register_function(hello, category="demo")
        tools = registry.list_tools()

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "hello")
        self.assertEqual(registry.get_categories(), ["demo"])

    def test_unified_meta_tool_names_contract(self):
        """Canonical unified meta-tool list should stay stable and complete."""
        names = get_unified_meta_tool_names()
        self.assertEqual(
            names,
            [
                "tools_list_categories",
                "tools_list_tools",
                "tools_get_schema",
                "tools_dispatch",
                "tools_runtime_metrics",
            ],
        )

    def test_unified_wave_a_categories_contract(self):
        """Canonical Wave A categories should remain stable."""
        self.assertEqual(get_unified_wave_a_categories(), ["ipfs", "workflow", "p2p"])

    def test_parse_preload_categories(self):
        """Preload category parser should support subsets and `all`."""
        self.assertEqual(_parse_preload_categories(None), [])
        self.assertEqual(_parse_preload_categories(""), [])
        self.assertEqual(_parse_preload_categories("ipfs"), ["ipfs"])
        self.assertEqual(
            _parse_preload_categories("ipfs,workflow,invalid,p2p"),
            ["ipfs", "workflow", "p2p"],
        )
        self.assertEqual(_parse_preload_categories("all"), ["ipfs", "workflow", "p2p"])

    def test_unified_config_from_env(self):
        """UnifiedMCPServerConfig should parse migration flags from environment."""
        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "true",
                "IPFS_MCP_SERVER_ENABLE_RISK_FRONTIER_EXECUTION": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow,p2p",
            },
            clear=False,
        ):
            config = UnifiedMCPServerConfig.from_env(
                allowed_preload_categories=["ipfs", "workflow", "p2p"]
            )

        self.assertTrue(config.enable_unified_bridge)
        self.assertTrue(config.enable_unified_bootstrap)
        self.assertTrue(config.enable_risk_frontier_execution)
        self.assertEqual(config.preload_categories, ["workflow", "p2p"])

    def test_unified_config_normalizes_artifact_store_backend(self):
        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_SERVER_ARTIFACT_STORE_BACKEND": " JSON ",
                "IPFS_MCP_SERVER_ARTIFACT_STORE_PATH": " /tmp/cid-artifacts.json ",
            },
            clear=False,
        ):
            config = UnifiedMCPServerConfig.from_env(
                allowed_preload_categories=["ipfs", "workflow", "p2p"]
            )

        self.assertEqual(config.artifact_store_backend, "json")
        self.assertEqual(config.artifact_store_path, "/tmp/cid-artifacts.json")

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_SERVER_ARTIFACT_STORE_BACKEND": "sqlite",
            },
            clear=False,
        ):
            invalid = UnifiedMCPServerConfig.from_env(
                allowed_preload_categories=["ipfs", "workflow", "p2p"]
            )

        self.assertEqual(invalid.artifact_store_backend, "memory")

    def test_runtime_router_resolves_metadata_runtime(self):
        """RuntimeRouter should use registered metadata when available."""
        router = RuntimeRouter(default_runtime="fastapi")

        async def echo(value: int):
            return value

        # Register runtime preference via manager metadata integration.
        manager = HierarchicalToolManager(runtime_router=router)
        manager.register_tool("demo", "echo", echo, runtime="trio")

        resolved = router.resolve_runtime("demo.echo", echo)
        self.assertEqual(resolved, "trio")

    def test_runtime_router_resolution_precedence(self):
        """Runtime resolution order should be: explicit map > metadata > function attr > default."""
        router = RuntimeRouter(default_runtime="fastapi")

        async def f1():
            return "ok"

        setattr(f1, "__mcp_runtime__", "fastapi")
        register_tool_metadata(ToolMetadata(name="demo.precedence.1", runtime="trio"))
        self.assertEqual(router.resolve_runtime("demo.precedence.1", f1), "trio")

        # Explicit map should override metadata
        router.register_tool_runtime("demo.precedence.1", "fastapi")
        self.assertEqual(router.resolve_runtime("demo.precedence.1", f1), "fastapi")

        # Function attribute should apply when no map/metadata exists
        async def f2():
            return "ok"

        setattr(f2, "__mcp_runtime__", "trio")
        self.assertEqual(router.resolve_runtime("demo.precedence.2", f2), "trio")

        # Default runtime should be used as final fallback
        async def f3():
            return "ok"

        self.assertEqual(router.resolve_runtime("demo.precedence.3", f3), "fastapi")

    def test_runtime_router_timeout_from_metadata(self):
        """Metadata timeout should terminate long-running tool calls."""

        async def _run() -> None:
            router = RuntimeRouter(default_runtime="fastapi")

            async def slow_tool() -> dict:
                await anyio.sleep(0.05)
                return {"ok": True}

            register_tool_metadata(
                ToolMetadata(
                    name="demo.timeout.slow",
                    runtime="fastapi",
                    timeout_seconds=0.001,
                )
            )

            with self.assertRaises(RuntimeExecutionError):
                await router.route_tool_call("demo.timeout.slow", slow_tool)

            metrics = router.get_metrics()
            self.assertGreaterEqual(metrics["fastapi"]["timeout_count"], 1)
            self.assertGreaterEqual(metrics["fastapi"]["error_count"], 1)

        anyio.run(_run)

    def test_hierarchical_manager_dispatches(self):
        """Hierarchical manager should lazy-load and dispatch category tools."""

        async def _run():
            manager = HierarchicalToolManager(runtime_router=RuntimeRouter())

            def loader(mgr):
                async def greet(name: str):
                    return {"greet": name}

                mgr.register_tool("demo", "greet", greet, description="greet tool")

            manager.register_category_loader("demo", loader)

            self.assertEqual(manager.list_categories(), ["demo"])
            tools = manager.list_tools("demo")
            self.assertEqual(tools[0]["name"], "greet")

            result = await manager.dispatch("demo", "greet", {"name": "world"})
            self.assertEqual(result["greet"], "world")

            metadata = get_tool_metadata("demo.greet")
            self.assertIsNotNone(metadata)

        anyio.run(_run)

    def test_legacy_adapter_registers_tools(self):
        """Legacy adapter should map existing mcp.tools into new manager."""
        manager = HierarchicalToolManager(runtime_router=RuntimeRouter())
        count = register_legacy_tools_into_manager(manager)

        self.assertGreater(count, 0)
        self.assertGreater(len(manager.list_categories()), 0)

    def test_wave_a_loader_registers_ipfs_tools(self):
        """Wave A loader bootstrap should provide IPFS category tools."""
        manager = HierarchicalToolManager(runtime_router=RuntimeRouter())
        configure_wave_a_loaders(manager)

        ipfs_tools = manager.list_tools("ipfs")
        self.assertGreater(len(ipfs_tools), 0)

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_create_server_bootstrap_flag_enabled(self, mock_create):
        """create_server should attach unified components when feature flag is enabled."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        with patch.dict(os.environ, {"IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1"}, clear=False):
            server = create_server(name="dummy")

        self.assertIs(server, dummy)
        self.assertTrue(getattr(server, "_unified_bootstrap_enabled", False))
        self.assertIsNotNone(getattr(server, "_unified_tool_manager", None))
        self.assertIsNotNone(getattr(server, "_unified_runtime_router", None))
        self.assertEqual(getattr(server, "_unified_meta_tools", []), get_unified_meta_tool_names())
        self.assertEqual(getattr(server, "_unified_supported_profiles", []), get_unified_supported_profiles())
        self.assertTrue(getattr(server, "_unified_profile_negotiation", {}).get("supports_profile_negotiation"))
        self.assertEqual(getattr(server, "_unified_preloaded_categories", []), [])
        self.assertIsInstance(getattr(server, "_unified_services", None), dict)
        self.assertIsNotNone(getattr(server, "_unified_server_context", None))
        self.assertIn("task_queue_factory", server._unified_services)
        self.assertIn("workflow_scheduler_factory", server._unified_services)
        self.assertIn("workflow_engine_factory", server._unified_services)
        self.assertIn("workflow_dag_executor_factory", server._unified_services)
        self.assertIn("peer_registry_factory", server._unified_services)
        self.assertIn("peer_discovery_factory", server._unified_services)
        self.assertIn("result_cache_factory", server._unified_services)
        self.assertIs(server._unified_server_context.services, server._unified_services)
        self.assertEqual(server._unified_server_context.supported_profiles, get_unified_supported_profiles())
        self.assertIn("tools_list_categories", server.tools)
        self.assertIn("tools_list_tools", server.tools)
        self.assertIn("tools_get_schema", server.tools)
        self.assertIn("tools_dispatch", server.tools)
        self.assertIn("tools_runtime_metrics", server.tools)
        self.assertEqual(sorted(get_unified_meta_tool_names()), sorted(server.tools.keys()))

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_supported_profiles_are_attached_as_snapshots(self, mock_create):
        """Attached profile metadata should not mutate canonical defaults/context snapshots."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        expected_profiles = get_unified_supported_profiles()

        with patch.dict(os.environ, {"IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1"}, clear=False):
            server = create_server(name="dummy")

        server._unified_supported_profiles.append("mcp++/profile-z-drift")
        server._unified_profile_negotiation["profiles"].append("mcp++/profile-z-drift")

        self.assertEqual(get_unified_supported_profiles(), expected_profiles)
        self.assertEqual(server._unified_server_context.supported_profiles, expected_profiles)
        self.assertEqual(
            server._unified_server_context_snapshot.get("supported_profiles"),
            expected_profiles,
        )
        self.assertEqual(
            server._unified_server_context.profile_negotiation().get("profiles"),
            expected_profiles,
        )
        self.assertEqual(
            (server._unified_server_context_snapshot.get("profile_negotiation") or {}).get("profiles"),
            expected_profiles,
        )
        self.assertEqual(
            (server._unified_server_context_snapshot.get("profile_negotiation") or {}).get("mode"),
            "optional_additive",
        )

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_profile_capability_snapshot_matches_expected_contract(self, mock_create):
        """Canonical profile capability metadata should remain stable across bootstrap snapshots."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        expected_profiles = [
            "mcp++/profile-a-idl",
            "mcp++/profile-b-cid-artifacts",
            "mcp++/profile-c-ucan",
            "mcp++/profile-d-temporal-policy",
            "mcp++/profile-e-mcp-p2p",
        ]
        expected_negotiation = {
            "supports_profile_negotiation": True,
            "mode": "optional_additive",
            "profiles": expected_profiles,
        }

        with patch.dict(os.environ, {"IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1"}, clear=False):
            server = create_server(name="dummy")

        self.assertEqual(get_unified_supported_profiles(), expected_profiles)
        self.assertEqual(server._unified_supported_profiles, expected_profiles)
        self.assertEqual(server._unified_profile_negotiation, expected_negotiation)
        self.assertEqual(server._unified_server_context.supported_profiles, expected_profiles)
        self.assertEqual(server._unified_server_context.profile_negotiation(), expected_negotiation)
        self.assertEqual(server._unified_server_context_snapshot.get("supported_profiles"), expected_profiles)
        self.assertEqual(server._unified_server_context_snapshot.get("profile_negotiation"), expected_negotiation)

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_bootstrap_service_factories_smoke(self, mock_create):
        """Unified bootstrap should provide callable MCP++ service factories."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        with patch.dict(os.environ, {"IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1"}, clear=False):
            server = create_server(name="dummy")

        services = server._unified_services
        self.assertTrue(callable(services["task_queue_factory"]))
        self.assertTrue(callable(services["workflow_scheduler_factory"]))
        self.assertTrue(callable(services["workflow_engine_factory"]))
        self.assertTrue(callable(services["workflow_dag_executor_factory"]))
        self.assertTrue(callable(services["peer_registry_factory"]))
        self.assertTrue(callable(services["peer_discovery_factory"]))
        self.assertTrue(callable(services["result_cache_factory"]))

        # Instantiate low-risk local services to ensure factories are wired.
        task_queue = services["task_queue_factory"]()
        workflow_engine = services["workflow_engine_factory"](max_concurrent_tasks=2)
        dag_executor = services["workflow_dag_executor_factory"](max_concurrent=2)
        result_cache = services["result_cache_factory"](default_ttl=5.0)

        self.assertTrue(hasattr(task_queue, "submit"))
        self.assertTrue(hasattr(workflow_engine, "execute_workflow"))
        self.assertTrue(hasattr(dag_executor, "execute_workflow"))
        self.assertTrue(hasattr(result_cache, "get"))

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_bootstrap_attaches_secrets_vault_when_enabled(self, mock_create):
        """Unified bootstrap should attach a secrets vault when enabled."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_SECRETS_VAULT": "1",
                "IPFS_MCP_SERVER_SECRETS_MASTER_KEY": "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
            },
            clear=False,
        ):
            server = create_server(name="dummy")

        self.assertIs(server, dummy)
        vault = getattr(server, "_unified_secrets_vault", None)
        self.assertIsNotNone(vault)
        self.assertTrue(hasattr(vault, "set"))
        self.assertTrue(hasattr(vault, "get"))

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_bootstrap_attaches_risk_scorer(self, mock_create):
        """Unified bootstrap should attach canonical risk scorer component."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        with patch.dict(os.environ, {"IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1"}, clear=False):
            server = create_server(name="dummy")

        scorer = getattr(server, "_unified_risk_scorer", None)
        self.assertIsNotNone(scorer)
        assessment = scorer.score_intent(tool="smoke.echo", actor="did:model:worker", params={"k": 1})
        self.assertEqual(assessment.tool, "smoke.echo")

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_bootstrap_attaches_observability_components(self, mock_create):
        """Unified bootstrap should attach monitoring/tracing/prometheus components."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_MONITORING": "1",
                "IPFS_MCP_SERVER_ENABLE_OTEL_TRACING": "1",
                "IPFS_MCP_SERVER_ENABLE_PROMETHEUS_EXPORTER": "1",
                "IPFS_MCP_SERVER_ENABLE_POLICY_AUDIT": "1",
                "IPFS_MCP_SERVER_PROMETHEUS_NAMESPACE": "mcp_test",
            },
            clear=False,
        ):
            server = create_server(name="dummy")

        self.assertIsNotNone(getattr(server, "_unified_metrics_collector", None))
        self.assertIsNotNone(getattr(server, "_unified_p2p_metrics_collector", None))
        self.assertIsNotNone(getattr(server, "_unified_tracer", None))
        self.assertIsNotNone(getattr(server, "_unified_tracing_status", None))
        self.assertIsNotNone(getattr(server, "_unified_prometheus_status", None))
        self.assertIsNotNone(getattr(server, "_unified_audit_metrics_status", None))

        tracing = getattr(server, "_unified_tracing_status", {})
        self.assertIn("enabled", tracing)
        self.assertIn("info", tracing)

        prom = getattr(server, "_unified_prometheus_status", {})
        self.assertTrue(prom.get("enabled"))
        self.assertEqual((prom.get("info") or {}).get("namespace"), "mcp_test")

        audit_metrics = getattr(server, "_unified_audit_metrics_status", {})
        self.assertTrue(audit_metrics.get("enabled"))
        self.assertTrue(audit_metrics.get("attached"))

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_bootstrap_autoloads_secrets_into_env(self, mock_create):
        """Unified bootstrap should optionally autoload vault secrets into environment."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        with tempfile.TemporaryDirectory() as tmp:
            vault_file = os.path.join(tmp, "vault.json")
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                    "IPFS_MCP_SERVER_ENABLE_SECRETS_VAULT": "1",
                    "IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_AUTOLOAD": "1",
                    "IPFS_MCP_SERVER_SECRETS_MASTER_KEY": "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
                    "IPFS_MCP_SERVER_SECRETS_VAULT_FILE": vault_file,
                },
                clear=False,
            ):
                seed_server = create_server(name="seed")
                seed_vault = getattr(seed_server, "_unified_secrets_vault")
                seed_vault.set("MCP_TEST_SECRET", "vault-loaded")
                os.environ.pop("MCP_TEST_SECRET", None)

                server = create_server(name="autoload")

                self.assertEqual(os.environ.get("MCP_TEST_SECRET"), "vault-loaded")
                status = getattr(server, "_unified_secrets_status", {})
                self.assertTrue(status.get("attached"))
                self.assertIn("MCP_TEST_SECRET", status.get("env_loaded", []))
                self.assertEqual(status.get("error"), "")

            os.environ.pop("MCP_TEST_SECRET", None)

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_unified_bootstrap_secrets_autoload_failure_is_nonfatal(self, mock_create):
        """Secrets autoload failure should not break bootstrap and should be surfaced in status."""

        class DummyServer:
            def __init__(self):
                self.tools = {}

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        dummy = DummyServer()
        mock_create.return_value = dummy

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_SECRETS_VAULT": "1",
                "IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_AUTOLOAD": "1",
                "IPFS_MCP_SERVER_SECRETS_MASTER_KEY": "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
            },
            clear=False,
        ):
            with patch("ipfs_accelerate_py.mcp_server.server.SecretsVault.load_into_env", side_effect=RuntimeError("autoload_failed")):
                server = create_server(name="autoload-failure")

        self.assertIs(server, dummy)
        self.assertIn("tools_dispatch", server.tools)
        status = getattr(server, "_unified_secrets_status", {})
        self.assertTrue(status.get("attached"))
        self.assertEqual(status.get("error"), "autoload_failed")

    @patch("ipfs_accelerate_py.mcp.server.create_mcp_server")
    def test_create_server_bootstrap_flag_disabled(self, mock_create):
        """create_server should keep pure delegation behavior when feature flag is disabled."""

        class DummyServer:
            pass

        dummy = DummyServer()
        mock_create.return_value = dummy

        with patch.dict(os.environ, {"IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "0"}, clear=False):
            server = create_server(name="dummy")

        self.assertIs(server, dummy)
        self.assertFalse(hasattr(server, "_unified_tool_manager"))

    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_legacy_create_mcp_server_bridge_enabled(self, mock_unified_create):
        """Legacy create_mcp_server should delegate to unified server when bridge flag is enabled."""

        class DummyServer:
            pass

        dummy = DummyServer()
        mock_unified_create.return_value = dummy

        with patch.dict(os.environ, {"IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1"}, clear=False):
            server = create_mcp_server(name="bridge-enabled")

        self.assertIs(server, dummy)
        mock_unified_create.assert_called_once()

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @patch("ipfs_accelerate_py.mcp_server.server.create_server")
    def test_legacy_create_mcp_server_bridge_disabled(self, mock_unified_create, mock_wrapper):
        """Legacy create_mcp_server should stay on legacy path when bridge flag is disabled."""

        class DummyServer:
            mcp = None

        mock_wrapper.return_value = DummyServer()

        with patch.dict(os.environ, {"IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "0"}, clear=False):
            server = create_mcp_server(name="bridge-disabled")

        self.assertIsNotNone(server)
        mock_unified_create.assert_not_called()
        mock_wrapper.assert_called_once()

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_legacy_bridge_plus_bootstrap_meta_tool_flow(self, mock_wrapper):
        """Legacy entrypoint should expose working hierarchical meta-tools when both flags are enabled."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_MONITORING": "1",
                "IPFS_MCP_SERVER_ENABLE_PROMETHEUS_EXPORTER": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="bridge-bootstrap-e2e")

        self.assertTrue(getattr(server, "_unified_bootstrap_enabled", False))
        self.assertIn("tools_list_categories", server.tools)
        self.assertIn("tools_list_tools", server.tools)
        self.assertIn("tools_get_schema", server.tools)
        self.assertIn("tools_dispatch", server.tools)
        self.assertIn("tools_runtime_metrics", server.tools)
        self.assertEqual(getattr(server, "_unified_preloaded_categories", []), ["ipfs"])
        self.assertEqual(sorted(get_unified_meta_tool_names()), sorted(server.tools.keys()))

        async def _run_flow() -> None:
            # Register a lightweight smoke tool directly into unified manager.
            async def echo(value: str):
                return {"echo": value}

            manager = server._unified_tool_manager
            manager.register_tool("smoke", "echo", echo, description="echo smoke")

            list_categories = server.tools["tools_list_categories"]["function"]
            list_tools = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]
            runtime_metrics = server.tools["tools_runtime_metrics"]["function"]

            async def _dispatch_result(category: str, tool: str, params: dict) -> dict:
                return self._assert_dispatch_success_envelope(
                    await dispatch(category, tool, params)
                )

            categories = await list_categories()
            self.assertIn("admin_tools", categories["categories"])
            self.assertIn("alert_tools", categories["categories"])
            self.assertIn("audit_tools", categories["categories"])
            self.assertIn("smoke", categories["categories"])
            self.assertIn("analysis_tools", categories["categories"])
            self.assertIn("auth_tools", categories["categories"])
            self.assertIn("bespoke_tools", categories["categories"])
            self.assertIn("rate_limiting", categories["categories"])
            self.assertIn("cache_tools", categories["categories"])
            self.assertIn("cli", categories["categories"])
            self.assertIn("background_task_tools", categories["categories"])
            self.assertIn("dashboard_tools", categories["categories"])
            self.assertIn("dataset_tools", categories["categories"])
            self.assertIn("embedding_tools", categories["categories"])
            self.assertIn("monitoring_tools", categories["categories"])
            self.assertIn("ipfs_cluster_tools", categories["categories"])
            self.assertIn("p2p_workflow_tools", categories["categories"])
            self.assertIn("p2p_tools", categories["categories"])
            self.assertIn("functions", categories["categories"])
            self.assertIn("investigation_tools", categories["categories"])
            self.assertIn("logic_tools", categories["categories"])
            self.assertIn("workflow_tools", categories["categories"])
            self.assertIn("sparse_embedding_tools", categories["categories"])
            self.assertIn("web_scraping_tools", categories["categories"])
            self.assertIn("data_processing_tools", categories["categories"])
            self.assertIn("email_tools", categories["categories"])
            self.assertIn("file_detection_tools", categories["categories"])
            self.assertIn("geospatial_tools", categories["categories"])
            self.assertIn("index_management_tools", categories["categories"])
            self.assertIn("provenance_tools", categories["categories"])
            self.assertIn("security_tools", categories["categories"])
            self.assertIn("search_tools", categories["categories"])
            self.assertIn("software_engineering_tools", categories["categories"])
            self.assertIn("session_tools", categories["categories"])
            self.assertIn("storage_tools", categories["categories"])
            self.assertIn("vector_store_tools", categories["categories"])
            self.assertIn("vector_tools", categories["categories"])
            self.assertIn("web_archive_tools", categories["categories"])
            self.assertIn("rate_limiting_tools", categories["categories"])
            self.assertIn("discord_tools", categories["categories"])
            self.assertIn("file_converter_tools", categories["categories"])
            self.assertIn("development_tools", categories["categories"])
            self.assertIn("ipfs_tools", categories["categories"])
            self.assertIn("graph_tools", categories["categories"])
            self.assertIn("pdf_tools", categories["categories"])
            self.assertIn("finance_data_tools", categories["categories"])
            self.assertIn("legal_dataset_tools", categories["categories"])
            self.assertIn("legacy_mcp_tools", categories["categories"])
            self.assertIn("lizardperson_argparse_programs", categories["categories"])
            self.assertIn("lizardpersons_function_tools", categories["categories"])
            self.assertIn("media_tools", categories["categories"])
            self.assertIn("mcplusplus", categories["categories"])
            self.assertIn("medical_research_scrapers", categories["categories"])

            tools = await list_tools("smoke")
            self.assertEqual(tools["tools"][0]["name"], "echo")

            schema = await get_schema("smoke", "echo")
            self.assertEqual(schema["name"], "echo")
            self.assertEqual(schema["category"], "smoke")

            result = self._assert_dispatch_success_envelope(
                await dispatch("smoke", "echo", {"value": "ok"})
            )
            self.assertEqual(result["echo"], "ok")

            configured = self._assert_dispatch_success_envelope(
                await dispatch(
                    "rate_limiting",
                    "configure_rate_limits",
                    {
                        "limits": [
                            {
                                "name": "smoke_limit",
                                "strategy": "token_bucket",
                                "requests_per_second": 1000,
                                "burst_capacity": 1000,
                                "enabled": True,
                            }
                        ]
                    },
                )
            )
            self.assertEqual(configured.get("configured_count"), 1)

            check = await _dispatch_result(
                "rate_limiting",
                "check_rate_limit",
                {
                    "limit_name": "smoke_limit",
                    "identifier": "worker-1",
                },
            )
            self.assertTrue(check.get("allowed", False))

            listing = await _dispatch_result(
                "rate_limiting",
                "manage_rate_limits",
                {"action": "list"},
            )
            self.assertGreaterEqual(int(listing.get("total_count", 0)), 1)

            listing_tools = await _dispatch_result(
                "rate_limiting_tools",
                "manage_rate_limits",
                {"action": "list"},
            )
            self.assertIn("total_count", listing_tools)

            set_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cache_tools",
                    "cache_set",
                    {"key": "smoke-key", "value": "smoke-value", "namespace": "smoke"},
                )
            )
            self.assertTrue(set_result.get("success", False))

            get_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cache_tools",
                    "cache_get",
                    {"key": "smoke-key", "namespace": "smoke"},
                )
            )
            self.assertEqual(get_result.get("value"), "smoke-value")
            self.assertTrue(get_result.get("hit", False))

            cli_result = await _dispatch_result(
                "cli",
                "execute_command",
                {"command": "echo", "args": ["smoke"]},
            )
            self.assertIn(cli_result.get("status"), ["success", "error"])

            stats_result = await _dispatch_result(
                "cache_tools",
                "manage_cache",
                {"operation": "stats", "namespace": "smoke"},
            )
            self.assertTrue(stats_result.get("success", False))
            self.assertIn("cache_stats", stats_result)

            created_session = await _dispatch_result(
                "session_tools",
                "create_session",
                {
                    "session_name": "smoke-session",
                    "user_id": "smoke-user",
                },
            )
            self.assertEqual(created_session.get("status"), "success")
            session_id = created_session.get("session_id")
            self.assertIsInstance(session_id, str)

            get_session = await _dispatch_result(
                "session_tools",
                "manage_session_state",
                {
                    "session_id": session_id,
                    "action": "get",
                },
            )
            self.assertEqual(get_session.get("status"), "success")
            self.assertEqual(get_session.get("session", {}).get("session_id"), session_id)

            pause_session = await _dispatch_result(
                "session_tools",
                "manage_session_state",
                {
                    "session_id": session_id,
                    "action": "pause",
                },
            )
            self.assertEqual(pause_session.get("status"), "success")
            self.assertEqual(pause_session.get("session_status"), "paused")

            search_result = await _dispatch_result(
                "search_tools",
                "semantic_search",
                {
                    "query": "smoke query",
                    "top_k": 2,
                    "collection": "smoke",
                },
            )
            self.assertEqual(search_result.get("query"), "smoke query")
            self.assertGreaterEqual(int(search_result.get("total_found", 0)), 1)

            conversion_result = await _dispatch_result(
                "data_processing_tools",
                "convert_format",
                {
                    "data": {"k": "v"},
                    "source_format": "json",
                    "target_format": "json",
                },
            )
            self.assertEqual(conversion_result.get("status"), "success")
            self.assertEqual(conversion_result.get("source_format"), "json")
            self.assertEqual(conversion_result.get("target_format"), "json")

            permission_result = await _dispatch_result(
                "security_tools",
                "check_access_permission",
                {
                    "resource_id": "smoke-resource",
                    "user_id": "smoke-user",
                    "permission_type": "read",
                },
            )
            self.assertIn(permission_result.get("status"), ["success", "error"])
            self.assertEqual(permission_result.get("resource_id"), "smoke-resource")

            analysis_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "analysis_tools",
                    "analyze_data_distribution",
                    {
                        "data_source": "mock",
                        "analysis_type": "comprehensive",
                        "data_params": {"n_samples": 20, "n_features": 8},
                    },
                )
            )
            self.assertEqual(analysis_result.get("status"), "success")
            self.assertEqual(analysis_result.get("data_source"), "mock")

            geospatial_result = await _dispatch_result(
                "geospatial_tools",
                "query_geographic_context",
                {
                    "query": "find events near London",
                    "corpus_data": "Sample event data",
                    "radius_km": 25.0,
                },
            )
            self.assertEqual(geospatial_result.get("status"), "success")
            self.assertEqual(geospatial_result.get("query"), "find events near London")

            graph_result = await _dispatch_result(
                "graph_tools",
                "graph_create",
                {},
            )
            self.assertIn(graph_result.get("status"), ["success", "error"])

            index_status_result = await _dispatch_result(
                "index_management_tools",
                "load_index",
                {
                    "action": "status",
                },
            )
            self.assertIn(index_status_result.get("status"), ["success", "error"])
            self.assertEqual(index_status_result.get("action"), "status")

            auth_decode_result = await _dispatch_result(
                "auth_tools",
                "validate_token",
                {
                    "token": "smoke-token",
                    "action": "decode",
                },
            )
            self.assertEqual(auth_decode_result.get("status"), "success")
            self.assertIn("message", auth_decode_result)

            bespoke_result = await _dispatch_result(
                "bespoke_tools",
                "system_health",
                {},
            )
            self.assertTrue("success" in bespoke_result or "status" in bespoke_result)

            provenance_result = await _dispatch_result(
                "provenance_tools",
                "record_provenance",
                {
                    "dataset_id": "smoke-dataset",
                    "operation": "transform",
                    "parameters": {"step": "normalize"},
                },
            )
            self.assertIn(provenance_result.get("status"), ["success", "error"])
            self.assertEqual(provenance_result.get("dataset_id"), "smoke-dataset")

            admin_result = await _dispatch_result(
                "admin_tools",
                "manage_endpoints",
                {
                    "action": "list",
                },
            )
            self.assertEqual(admin_result.get("status"), "success")
            self.assertEqual(admin_result.get("action"), "list")

            detection_result = await _dispatch_result(
                "file_detection_tools",
                "detect_file_type",
                {
                    "file_path": "README.md",
                },
            )
            self.assertTrue("mime_type" in detection_result or "error" in detection_result)

            storage_result = await _dispatch_result(
                "storage_tools",
                "manage_collections",
                {
                    "action": "list",
                },
            )
            self.assertIn(storage_result.get("success"), [True, False])
            self.assertEqual(storage_result.get("action"), "list")

            vector_result = await _dispatch_result(
                "vector_store_tools",
                "vector_index",
                {
                    "action": "info",
                    "index_name": "smoke-index",
                },
            )
            self.assertEqual(vector_result.get("action"), "info")
            self.assertEqual(vector_result.get("index_name"), "smoke-index")

            vector_tools_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "create_vector_index",
                    {
                        "vectors": [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
                        "index_id": "smoke-vector-tools",
                        "metric": "cosine",
                    },
                )
            )
            self.assertEqual(vector_tools_result.get("status"), "success")

            audit_result = await _dispatch_result(
                "audit_tools",
                "record_audit_event",
                {
                    "action": "smoke.test",
                    "resource_id": "smoke-resource",
                    "severity": "info",
                },
            )
            self.assertIn(audit_result.get("status"), ["success", "error"])
            self.assertEqual(audit_result.get("action"), "smoke.test")

            alert_result = await _dispatch_result(
                "alert_tools",
                "list_alert_rules",
                {
                    "enabled_only": False,
                },
            )
            self.assertIn(alert_result.get("status"), ["success", "error"])

            email_result = await _dispatch_result(
                "email_tools",
                "email_test_connection",
                {
                    "protocol": "imap",
                    "server": "mail.example.com",
                    "timeout": 1,
                },
            )
            self.assertIn(email_result.get("status"), ["success", "error"])

            finance_result = await _dispatch_result(
                "finance_data_tools",
                "scrape_stock_data",
                {"symbols": ["AAPL"], "days": 1},
            )
            self.assertIn(finance_result.get("status"), ["success", "error"])

            file_converter_result = await _dispatch_result(
                "file_converter_tools",
                "file_info_tool",
                {"input_path": "README.md"},
            )
            self.assertIn(file_converter_result.get("status"), ["success", "error"])

            dashboard_result = await _dispatch_result(
                "dashboard_tools",
                "get_tdfol_metrics",
                {},
            )
            self.assertIn(dashboard_result.get("status"), ["success", "error"])

            software_engineering_result = await _dispatch_result(
                "software_engineering_tools",
                "search_repositories",
                {"query": "smoke", "max_results": 1},
            )
            self.assertIn(software_engineering_result.get("status"), ["success", "error"])

            background_task_result = await _dispatch_result(
                "background_task_tools",
                "manage_task_queue",
                {"action": "get_stats"},
            )
            self.assertIn(background_task_result.get("status"), ["success", "error"])

            dataset_result = await _dispatch_result(
                "dataset_tools",
                "load_dataset",
                {"source": "smoke_source"},
            )
            self.assertIn(dataset_result.get("status"), ["success", "error"])

            development_result = await _dispatch_result(
                "development_tools",
                "vscode_cli_status",
                {},
            )
            self.assertTrue("success" in development_result or "status" in development_result)

            discord_result = await _dispatch_result(
                "discord_tools",
                "discord_list_guilds",
                {},
            )
            self.assertIn(discord_result.get("status"), ["success", "error"])

            embedding_result = await _dispatch_result(
                "embedding_tools",
                "generate_embeddings",
                {"texts": ["smoke embedding input"]},
            )
            self.assertIn(embedding_result.get("status"), ["success", "error"])

            monitoring_result = await _dispatch_result(
                "monitoring_tools",
                "health_check",
                {},
            )
            self.assertTrue("status" in monitoring_result)

            sparse_embedding_result = await _dispatch_result(
                "sparse_embedding_tools",
                "generate_sparse_embedding",
                {"text": "smoke sparse embedding input"},
            )
            self.assertTrue("model" in sparse_embedding_result or "status" in sparse_embedding_result)

            ipfs_cluster_result = await _dispatch_result(
                "ipfs_cluster_tools",
                "manage_ipfs_cluster",
                {"action": "status"},
            )
            self.assertTrue("status" in ipfs_cluster_result)

            ipfs_tools_result = await _dispatch_result(
                "ipfs_tools",
                "get_from_ipfs",
                {"cid": "bafybeigdyrzt5examplecid"},
            )
            self.assertIn(ipfs_tools_result.get("status"), ["success", "error"])

            legal_result = await _dispatch_result(
                "legal_dataset_tools",
                "list_state_jurisdictions",
                {},
            )
            self.assertIn(legal_result.get("status"), ["success", "error"])

            media_result = await _dispatch_result(
                "media_tools",
                "ytdlp_extract_info",
                {"url": "https://example.com/media"},
            )
            self.assertIn(media_result.get("status"), ["success", "error"])

            mcplusplus_result = await _dispatch_result(
                "mcplusplus",
                "mcplusplus_engine_status",
                {},
            )
            self.assertIn(mcplusplus_result.get("status"), ["success", "error"])

            medical_research_result = await _dispatch_result(
                "medical_research_scrapers",
                "scrape_pubmed_medical_research",
                {"query": "smoke"},
            )
            self.assertIn(medical_research_result.get("status"), ["success", "error"])

            legacy_result = await _dispatch_result(
                "legacy_mcp_tools",
                "legacy_tools_inventory",
                {},
            )
            self.assertIn(legacy_result.get("status"), ["success", "error"])

            lizardperson_argparse_result = await _dispatch_result(
                "lizardperson_argparse_programs",
                "municipal_bluebook_validator_info",
                {},
            )
            self.assertIn(lizardperson_argparse_result.get("status"), ["success", "error"])

            lizardpersons_function_result = await _dispatch_result(
                "lizardpersons_function_tools",
                "get_current_time",
                {"format_type": "iso"},
            )
            self.assertIn(lizardpersons_function_result.get("status"), ["success", "error"])

            investigation_result = await _dispatch_result(
                "investigation_tools",
                "analyze_entities",
                {"corpus_data": '{"documents": []}'},
            )
            self.assertIn(investigation_result.get("status"), ["success", "error"])

            logic_result = await _dispatch_result(
                "logic_tools",
                "logic_health",
                {},
            )
            self.assertTrue("status" in logic_result or "success" in logic_result)

            p2p_workflow_result = await _dispatch_result(
                "p2p_workflow_tools",
                "get_p2p_scheduler_status",
                {},
            )
            self.assertTrue("status" in p2p_workflow_result or "success" in p2p_workflow_result)

            p2p_tools_result = await _dispatch_result(
                "p2p_tools",
                "p2p_service_status",
                {},
            )
            self.assertTrue("ok" in p2p_tools_result or "status" in p2p_tools_result)

            pdf_tools_result = await _dispatch_result(
                "pdf_tools",
                "pdf_query_corpus",
                {"query": "smoke pdf query"},
            )
            self.assertIn(pdf_tools_result.get("status"), ["success", "error"])

            function_result = await _dispatch_result(
                "functions",
                "execute_python_snippet",
                {"code": "print('smoke')"},
            )
            self.assertTrue("status" in function_result)

            workflow_tools_result = await _dispatch_result(
                "workflow_tools",
                "schedule_workflow",
                {
                    "workflow_definition": {"name": "smoke-workflow", "steps": []},
                    "schedule_config": {"cron": "0 * * * *"},
                },
            )
            self.assertTrue("status" in workflow_tools_result or "success" in workflow_tools_result)

            workflow_templates_result = await _dispatch_result(
                "workflow_tools",
                "list_templates",
                {},
            )
            self.assertTrue(
                "templates" in workflow_templates_result
                or "status" in workflow_templates_result
                or "success" in workflow_templates_result
            )

            workflow_resume_result = await _dispatch_result(
                "workflow_tools",
                "resume_workflow",
                {"workflow_id": "smoke-workflow"},
            )
            self.assertTrue(
                "workflow_id" in workflow_resume_result
                or "status" in workflow_resume_result
                or "success" in workflow_resume_result
            )

            workflow_metrics_result = await _dispatch_result(
                "workflow_tools",
                "get_workflow_metrics",
                {
                    "workflow_id": "smoke-workflow",
                    "include_performance": True,
                },
            )
            self.assertTrue(
                "metrics" in workflow_metrics_result
                or "status" in workflow_metrics_result
                or "success" in workflow_metrics_result
            )

            workflow_pause_result = await _dispatch_result(
                "workflow_tools",
                "pause_workflow",
                {"workflow_id": "smoke-workflow"},
            )
            self.assertTrue(
                "workflow_id" in workflow_pause_result
                or "status" in workflow_pause_result
                or "success" in workflow_pause_result
            )

            workflow_list_result = await _dispatch_result(
                "workflow_tools",
                "list_workflows",
                {"include_logs": False},
            )
            self.assertTrue(
                "workflows" in workflow_list_result
                or "status" in workflow_list_result
                or "success" in workflow_list_result
            )

            workflow_create_result = await _dispatch_result(
                "workflow_tools",
                "create_workflow",
                {
                    "workflow_id": "smoke-created-workflow",
                    "workflow_definition": {
                        "name": "smoke-created-workflow",
                        "steps": [],
                    },
                },
            )
            self.assertTrue(
                "workflow_id" in workflow_create_result
                or "status" in workflow_create_result
                or "success" in workflow_create_result
            )

            workflow_run_result = await _dispatch_result(
                "workflow_tools",
                "run_workflow",
                {"workflow_id": "smoke-created-workflow"},
            )
            self.assertTrue(
                "workflow_id" in workflow_run_result
                or "status" in workflow_run_result
                or "success" in workflow_run_result
            )

            workflow_assigned_result = await _dispatch_result(
                "workflow_tools",
                "get_assigned_workflows",
                {},
            )
            self.assertTrue(
                "assigned_workflows" in workflow_assigned_result
                or "status" in workflow_assigned_result
                or "success" in workflow_assigned_result
            )

            workflow_tags_result = await _dispatch_result(
                "workflow_tools",
                "get_workflow_tags",
                {},
            )
            self.assertTrue(
                "tags" in workflow_tags_result
                or "status" in workflow_tags_result
                or "success" in workflow_tags_result
            )

            workflow_init_p2p_result = await _dispatch_result(
                "workflow_tools",
                "initialize_p2p_scheduler",
                {"peer_id": "smoke-peer", "peers": ["peer-a", "peer-b"]},
            )
            self.assertTrue(
                "status" in workflow_init_p2p_result
                or "success" in workflow_init_p2p_result
                or "error" in workflow_init_p2p_result
            )

            workflow_schedule_p2p_result = await _dispatch_result(
                "workflow_tools",
                "schedule_p2p_workflow",
                {
                    "workflow_id": "smoke-p2p-workflow",
                    "name": "Smoke P2P Workflow",
                    "tags": ["p2p_eligible"],
                    "priority": 1.0,
                },
            )
            self.assertTrue(
                "workflow_id" in workflow_schedule_p2p_result
                or "status" in workflow_schedule_p2p_result
                or "success" in workflow_schedule_p2p_result
                or "error" in workflow_schedule_p2p_result
            )

            web_scraping_result = await _dispatch_result(
                "web_scraping_tools",
                "check_scraper_methods_tool",
                {},
            )
            self.assertIn(web_scraping_result.get("status"), ["success", "error"])

            web_archive_result = await _dispatch_result(
                "web_archive_tools",
                "search_common_crawl",
                {
                    "domain": "example.com",
                    "limit": 1,
                },
            )
            self.assertIn(web_archive_result.get("status"), ["success", "error"])

            metrics_payload = await runtime_metrics()
            self.assertIn("runtimes", metrics_payload)
            self.assertIn("fastapi", metrics_payload["runtimes"])
            self.assertIn("timeout_count", metrics_payload["runtimes"]["fastapi"])
            self.assertIn("observability", metrics_payload)
            self.assertIn("monitoring", metrics_payload["observability"])
            self.assertIn("tracing", metrics_payload["observability"])
            self.assertIn("prometheus", metrics_payload["observability"])

            monitoring_snapshot = metrics_payload["observability"]["monitoring"]["snapshot"]
            tool_metrics = monitoring_snapshot.get("tool_metrics", {})
            self.assertIn("smoke.echo", tool_metrics)
            self.assertGreaterEqual(tool_metrics["smoke.echo"].get("total_calls", 0), 1)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_idl_tools_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified meta-tools should dispatch native MCP-IDL `interfaces/*` tools."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="idl-dispatch-e2e")

        async def _run_flow() -> None:
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = self._assert_dispatch_success_envelope(
                await dispatch("idl", "interfaces_list", {})
            )
            self.assertIsInstance(listed, dict)
            self.assertGreaterEqual(listed.get("count", 0), 1)
            self.assertIn("interface_cids", listed)

            first_cid = listed["interface_cids"][0]
            payload = self._assert_dispatch_success_envelope(
                await dispatch("idl", "interfaces_get", {"interface_cid": first_cid})
            )
            self.assertTrue(payload.get("found"))
            self.assertEqual(payload.get("interface_cid"), first_cid)

            compat = self._assert_dispatch_success_envelope(
                await dispatch("idl", "interfaces_compat", {"interface_cid": first_cid})
            )
            self.assertIn("compatible", compat)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_idl_tools_include_preloaded_ipfs_category_descriptors(self, mock_wrapper):
        """IDL tools should include descriptors derived from preloaded Wave A categories."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="idl-preloaded-descriptors")

        async def _run_flow() -> None:
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = self._assert_dispatch_success_envelope(
                await dispatch("idl", "interfaces_list", {})
            )
            self.assertIsInstance(listed, dict)
            self.assertGreaterEqual(listed.get("count", 0), 2)

            found_ipfs_descriptor = False
            for interface_cid in listed.get("interface_cids", []):
                payload = self._assert_dispatch_success_envelope(
                    await dispatch("idl", "interfaces_get", {"interface_cid": interface_cid})
                )
                descriptor = (payload or {}).get("descriptor") or {}
                if descriptor.get("name") != "ipfs_tools":
                    continue

                methods = descriptor.get("methods", [])
                self.assertTrue(any(m.get("name") == "ipfs/ipfs_files_validate_cid" for m in methods if isinstance(m, dict)))

                compat = self._assert_dispatch_success_envelope(
                    await dispatch("idl", "interfaces_compat", {"interface_cid": interface_cid})
                )
                self.assertTrue(compat.get("compatible"))
                found_ipfs_descriptor = True
                break

            self.assertTrue(found_ipfs_descriptor)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_idl_tools_include_preloaded_workflow_category_descriptors(self, mock_wrapper):
        """IDL tools should include descriptors for preloaded workflow category tools."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="idl-preloaded-workflow-descriptors")

        async def _run_flow() -> None:
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = self._assert_dispatch_success_envelope(
                await dispatch("idl", "interfaces_list", {})
            )
            self.assertIsInstance(listed, dict)
            self.assertGreaterEqual(listed.get("count", 0), 2)

            found_workflow_descriptor = False
            for interface_cid in listed.get("interface_cids", []):
                payload = self._assert_dispatch_success_envelope(
                    await dispatch("idl", "interfaces_get", {"interface_cid": interface_cid})
                )
                descriptor = (payload or {}).get("descriptor") or {}
                if descriptor.get("name") != "workflow_tools":
                    continue

                methods = descriptor.get("methods", [])
                self.assertTrue(any(m.get("name") == "workflow/get_workflow_templates" for m in methods if isinstance(m, dict)))

                compat = self._assert_dispatch_success_envelope(
                    await dispatch("idl", "interfaces_compat", {"interface_cid": interface_cid})
                )
                self.assertTrue(compat.get("compatible"))
                found_workflow_descriptor = True
                break

            self.assertTrue(found_workflow_descriptor)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_idl_tools_include_preloaded_p2p_category_descriptors_with_loader_patch(self, mock_wrapper):
        """IDL tools should include p2p descriptor when p2p preload loader registers tools."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        def _patched_load_p2p_tools(manager):
            async def _p2p_ping(peer_id: str = ""):
                return {"ok": True, "peer_id": str(peer_id or "")}

            manager.register_tool(
                category="p2p",
                name="p2p_ping",
                func=_p2p_ping,
                description="Patched p2p ping for deterministic preload coverage.",
                input_schema={
                    "type": "object",
                    "properties": {"peer_id": {"type": "string", "default": ""}},
                    "required": [],
                },
                runtime="trio",
                tags=["native", "wave-a", "p2p"],
            )

        with patch("ipfs_accelerate_py.mcp_server.wave_a_loaders.load_p2p_tools", new=_patched_load_p2p_tools):
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                    "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
                },
                clear=False,
            ):
                server = create_mcp_server(name="idl-preloaded-p2p-descriptors")

        async def _run_flow() -> None:
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = self._assert_dispatch_success_envelope(
                await dispatch("idl", "interfaces_list", {})
            )
            self.assertIsInstance(listed, dict)
            self.assertGreaterEqual(listed.get("count", 0), 2)

            found_p2p_descriptor = False
            for interface_cid in listed.get("interface_cids", []):
                payload = self._assert_dispatch_success_envelope(
                    await dispatch("idl", "interfaces_get", {"interface_cid": interface_cid})
                )
                descriptor = (payload or {}).get("descriptor") or {}
                if descriptor.get("name") != "p2p_tools":
                    continue

                methods = descriptor.get("methods", [])
                self.assertTrue(any(m.get("name") == "p2p/p2p_ping" for m in methods if isinstance(m, dict)))

                compat = self._assert_dispatch_success_envelope(
                    await dispatch("idl", "interfaces_compat", {"interface_cid": interface_cid})
                )
                self.assertTrue(compat.get("compatible"))
                found_p2p_descriptor = True
                break

            self.assertTrue(found_p2p_descriptor)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_tool_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A IPFS tool implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-ipfs-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            ipfs_tools = await tools_list("ipfs")
            ipfs_names = [tool["name"] for tool in ipfs_tools["tools"]]
            self.assertIn("ipfs_files_validate_cid", ipfs_names)

            # Use an obviously invalid CID to keep test deterministic and offline.
            result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs",
                    "ipfs_files_validate_cid",
                    {"cid": "not_a_valid_cid"},
                )
            )

            self.assertIn("success", result)
            self.assertIn("data", result)
            self.assertIsInstance(result["data"], dict)
            self.assertIn("valid", result["data"])
            self.assertFalse(result["data"]["valid"])

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_list_files_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A IPFS list-files implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-ipfs-list-dispatch")

        class _FakeResult:
            def __init__(self):
                self.success = True
                self.data = {"path": "/", "files": [{"name": "demo.txt", "size": 4}], "count": 1}
                self.error = None

        class _FakeKit:
            def list_files(self, path: str = "/"):
                self.last_path = path
                return _FakeResult()

        fake_kit = _FakeKit()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            ipfs_tools = await tools_list("ipfs")
            ipfs_names = [tool["name"] for tool in ipfs_tools["tools"]]
            self.assertIn("ipfs_files_list_files", ipfs_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=fake_kit,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch("ipfs", "ipfs_files_list_files", {"path": "/"})
                )

            self.assertTrue(result["success"])
            self.assertIn("data", result)
            self.assertEqual(result["data"]["count"], 1)
            self.assertEqual(result["data"]["files"][0]["name"], "demo.txt")
            self.assertEqual(fake_kit.last_path, "/")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_add_file_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A IPFS add-file implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-ipfs-add-dispatch")

        class _FakeResult:
            def __init__(self):
                self.success = True
                self.data = {
                    "cid": "QmFakeCid123456789012345678901234567890123456",
                    "name": "sample.txt",
                    "size": 12,
                }
                self.error = None

        class _FakeKit:
            def add_file(self, path: str, pin: bool = True):
                self.last_path = path
                self.last_pin = pin
                return _FakeResult()

        fake_kit = _FakeKit()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            ipfs_tools = await tools_list("ipfs")
            ipfs_names = [tool["name"] for tool in ipfs_tools["tools"]]
            self.assertIn("ipfs_files_add_file", ipfs_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=fake_kit,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_add_file",
                        {"path": "/tmp/sample.txt", "pin": False},
                    )
                )

            self.assertTrue(result["success"])
            self.assertIn("data", result)
            self.assertEqual(result["data"]["name"], "sample.txt")
            self.assertEqual(fake_kit.last_path, "/tmp/sample.txt")
            self.assertFalse(fake_kit.last_pin)

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_pin_file_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A IPFS pin-file implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-ipfs-pin-dispatch")

        class _FakeResult:
            def __init__(self):
                self.success = True
                self.data = {
                    "cid": "QmPinnedCid123456789012345678901234567890123456",
                    "pinned": True,
                }
                self.error = None

        class _FakeKit:
            def pin_file(self, cid: str):
                self.last_cid = cid
                return _FakeResult()

        fake_kit = _FakeKit()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            ipfs_tools = await tools_list("ipfs")
            ipfs_names = [tool["name"] for tool in ipfs_tools["tools"]]
            self.assertIn("ipfs_files_pin_file", ipfs_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=fake_kit,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_pin_file",
                        {"cid": "QmPinnedCid123456789012345678901234567890123456"},
                    )
                )

            self.assertTrue(result["success"])
            self.assertIn("data", result)
            self.assertTrue(result["data"]["pinned"])
            self.assertEqual(
                fake_kit.last_cid,
                "QmPinnedCid123456789012345678901234567890123456",
            )

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_unpin_file_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A IPFS unpin-file implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-ipfs-unpin-dispatch")

        class _FakeResult:
            def __init__(self):
                self.success = True
                self.data = {
                    "cid": "QmPinnedCid123456789012345678901234567890123456",
                    "pinned": False,
                }
                self.error = None

        class _FakeKit:
            def unpin_file(self, cid: str):
                self.last_cid = cid
                return _FakeResult()

        fake_kit = _FakeKit()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            ipfs_tools = await tools_list("ipfs")
            ipfs_names = [tool["name"] for tool in ipfs_tools["tools"]]
            self.assertIn("ipfs_files_unpin_file", ipfs_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=fake_kit,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_unpin_file",
                        {"cid": "QmPinnedCid123456789012345678901234567890123456"},
                    )
                )

            self.assertTrue(result["success"])
            self.assertIn("data", result)
            self.assertFalse(result["data"]["pinned"])
            self.assertEqual(
                fake_kit.last_cid,
                "QmPinnedCid123456789012345678901234567890123456",
            )

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_get_file_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A IPFS get-file implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-ipfs-get-dispatch")

        class _FakeResult:
            def __init__(self):
                self.success = True
                self.data = {
                    "cid": "QmGetCid123456789012345678901234567890123456789",
                    "path": "/tmp/retrieved.txt",
                    "size": 22,
                }
                self.error = None

        class _FakeKit:
            def get_file(self, cid: str, output_path: str):
                self.last_cid = cid
                self.last_output_path = output_path
                return _FakeResult()

        fake_kit = _FakeKit()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            ipfs_tools = await tools_list("ipfs")
            ipfs_names = [tool["name"] for tool in ipfs_tools["tools"]]
            self.assertIn("ipfs_files_get_file", ipfs_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=fake_kit,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_get_file",
                        {
                            "cid": "QmGetCid123456789012345678901234567890123456789",
                            "output_path": "/tmp/retrieved.txt",
                        },
                    )
                )

            self.assertTrue(result["success"])
            self.assertIn("data", result)
            self.assertEqual(result["data"]["path"], "/tmp/retrieved.txt")
            self.assertEqual(
                fake_kit.last_cid,
                "QmGetCid123456789012345678901234567890123456789",
            )
            self.assertEqual(fake_kit.last_output_path, "/tmp/retrieved.txt")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_cat_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A IPFS cat implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "ipfs",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-ipfs-cat-dispatch")

        class _FakeResult:
            def __init__(self):
                self.success = True
                self.data = {
                    "cid": "QmCatCid123456789012345678901234567890123456789",
                    "content": "hello from ipfs",
                    "size": 15,
                }
                self.error = None

        class _FakeKit:
            def cat_file(self, cid: str):
                self.last_cid = cid
                return _FakeResult()

        fake_kit = _FakeKit()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            ipfs_tools = await tools_list("ipfs")
            ipfs_names = [tool["name"] for tool in ipfs_tools["tools"]]
            self.assertIn("ipfs_files_cat", ipfs_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=fake_kit,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_cat",
                        {"cid": "QmCatCid123456789012345678901234567890123456789"},
                    )
                )

            self.assertTrue(result["success"])
            self.assertEqual(result["data"]["content"], "hello from ipfs")
            self.assertEqual(result["data"]["size"], 15)
            self.assertEqual(
                fake_kit.last_cid,
                "QmCatCid123456789012345678901234567890123456789",
            )

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_templates_dispatch_via_meta_tools(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native Wave A workflow templates implementation."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("get_workflow_templates", workflow_names)

            result = self._assert_dispatch_success_envelope(
                await dispatch("workflow", "get_workflow_templates", {})
            )
            self.assertEqual(result["status"], "success")
            self.assertIn("templates", result)
            self.assertEqual(result["total"], 4)
            self.assertIn("image_generation", result["templates"])

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_list_dispatch_via_meta_tools_manager_unavailable(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow list tool and surface manager-unavailable state."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-list-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("list_workflows", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=None,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "list_workflows", {})
                )

            self.assertEqual(result["status"], "error")
            self.assertIn("Workflow manager not available", result["error"])

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_get_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow get tool and map workflow/task fields."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-get-dispatch")

        class _FakeTask:
            def __init__(self):
                self.task_id = "task-1"
                self.name = "demo task"
                self.type = "text-generation"
                self.status = "completed"
                self.config = {"model": "gpt-4"}
                self.result = {"output": "ok"}
                self.error = None
                self.started_at = "2026-01-01T00:00:00Z"
                self.completed_at = "2026-01-01T00:00:01Z"
                self.dependencies = []

        class _FakeWorkflow:
            def __init__(self):
                self.workflow_id = "wf-1"
                self.name = "Demo Workflow"
                self.description = "workflow for testing"
                self.status = "completed"
                self.created_at = "2026-01-01T00:00:00Z"
                self.started_at = "2026-01-01T00:00:00Z"
                self.completed_at = "2026-01-01T00:00:01Z"
                self.error = None
                self.tasks = [_FakeTask()]
                self.metadata = {"source": "test"}

            def get_progress(self):
                return {"completed": 1, "total": 1, "percent": 100.0}

        class _FakeManager:
            def get_workflow(self, workflow_id: str):
                if workflow_id == "wf-1":
                    return _FakeWorkflow()
                return None

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("get_workflow", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=_FakeManager(),
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "get_workflow", {"workflow_id": "wf-1"})
                )

            self.assertEqual(result["status"], "success")
            self.assertIn("workflow", result)
            self.assertEqual(result["workflow"]["workflow_id"], "wf-1")
            self.assertEqual(result["workflow"]["tasks"][0]["task_id"], "task-1")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_create_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow create tool and map creation response."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-create-dispatch")

        class _FakeWorkflow:
            def __init__(self):
                self.workflow_id = "wf-created"
                self.name = "Created Workflow"
                self.description = "created via test"
                self.created_at = "2026-01-02T00:00:00Z"
                self.tasks = [{"name": "task-a"}]

        class _FakeManager:
            def create_workflow(self, name: str, description: str, tasks: list[dict]):
                self.last_name = name
                self.last_description = description
                self.last_tasks = tasks
                return _FakeWorkflow()

        fake_manager = _FakeManager()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("create_workflow", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=fake_manager,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "create_workflow",
                        {
                            "name": "Created Workflow",
                            "description": "created via test",
                            "tasks": [{"name": "task-a"}],
                        },
                    )
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["workflow_id"], "wf-created")
            self.assertEqual(result["task_count"], 1)
            self.assertEqual(fake_manager.last_name, "Created Workflow")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_update_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow update tool and map update response."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-update-dispatch")

        class _FakeWorkflow:
            def __init__(self):
                self.workflow_id = "wf-updated"
                self.name = "Updated Workflow"
                self.description = "updated via test"
                self.tasks = [{"name": "task-b"}]

        class _FakeManager:
            def update_workflow(self, workflow_id: str, name=None, description=None, tasks=None):
                self.last_workflow_id = workflow_id
                self.last_name = name
                self.last_description = description
                self.last_tasks = tasks
                return _FakeWorkflow()

        fake_manager = _FakeManager()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("update_workflow", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=fake_manager,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "update_workflow",
                        {
                            "workflow_id": "wf-updated",
                            "name": "Updated Workflow",
                            "description": "updated via test",
                            "tasks": [{"name": "task-b"}],
                        },
                    )
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["workflow_id"], "wf-updated")
            self.assertEqual(result["task_count"], 1)
            self.assertEqual(fake_manager.last_workflow_id, "wf-updated")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_delete_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow delete tool and map delete response."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-delete-dispatch")

        class _FakeManager:
            def delete_workflow(self, workflow_id: str):
                self.last_workflow_id = workflow_id

        fake_manager = _FakeManager()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("delete_workflow", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=fake_manager,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "delete_workflow",
                        {"workflow_id": "wf-deleted"},
                    )
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["workflow_id"], "wf-deleted")
            self.assertIn("deleted", result["message"].lower())
            self.assertEqual(fake_manager.last_workflow_id, "wf-deleted")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_start_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow start tool and map start response."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-start-dispatch")

        class _FakeManager:
            def start_workflow(self, workflow_id: str):
                self.last_workflow_id = workflow_id

        fake_manager = _FakeManager()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("start_workflow", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=fake_manager,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "start_workflow",
                        {"workflow_id": "wf-started"},
                    )
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["workflow_id"], "wf-started")
            self.assertIn("started", result["message"].lower())
            self.assertEqual(fake_manager.last_workflow_id, "wf-started")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_pause_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow pause tool and map pause response."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-pause-dispatch")

        class _FakeManager:
            def pause_workflow(self, workflow_id: str):
                self.last_workflow_id = workflow_id

        fake_manager = _FakeManager()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("pause_workflow", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=fake_manager,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "pause_workflow",
                        {"workflow_id": "wf-paused"},
                    )
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["workflow_id"], "wf-paused")
            self.assertIn("paused", result["message"].lower())
            self.assertEqual(fake_manager.last_workflow_id, "wf-paused")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_workflow_stop_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native workflow stop tool and map stop response."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-workflow-stop-dispatch")

        class _FakeManager:
            def stop_workflow(self, workflow_id: str):
                self.last_workflow_id = workflow_id

        fake_manager = _FakeManager()

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            workflow_tools = await tools_list("workflow")
            workflow_names = [tool["name"] for tool in workflow_tools["tools"]]
            self.assertIn("stop_workflow", workflow_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=fake_manager,
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "stop_workflow",
                        {"workflow_id": "wf-stopped"},
                    )
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["workflow_id"], "wf-stopped")
            self.assertIn("stopped", result["message"].lower())
            self.assertEqual(fake_manager.last_workflow_id, "wf-stopped")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_status_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p status tool and pass through status payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-status-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_status", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._request_status",
                return_value={"ok": True, "status": "healthy"},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch("p2p", "p2p_taskqueue_status", {})
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "healthy")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_list_tasks_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p list-tasks tool and pass through task payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-list-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_list_tasks", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._list_tasks",
                return_value=[{"task_id": "t1", "status": "queued"}],
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch("p2p", "p2p_taskqueue_list_tasks", {"limit": 5})
                )

            self.assertTrue(result["ok"])
            self.assertEqual(len(result["tasks"]), 1)
            self.assertEqual(result["tasks"][0]["task_id"], "t1")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_get_task_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p get-task tool and pass through task payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-get-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_get_task", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._get_task",
                return_value={"task_id": "t1", "status": "running"},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch("p2p", "p2p_taskqueue_get_task", {"task_id": "t1"})
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["task"]["task_id"], "t1")
            self.assertEqual(result["task"]["status"], "running")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_wait_task_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p wait-task tool and pass through task payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-wait-task-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_wait_task", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._wait_task",
                return_value={"task_id": "t1", "status": "completed"},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_wait_task",
                        {"task_id": "t1", "timeout_s": 5.0},
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["task"]["task_id"], "t1")
            self.assertEqual(result["task"]["status"], "completed")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_complete_task_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p complete-task and pass through response payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-complete-task-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_complete_task", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._complete_task",
                return_value={"ok": True, "task_id": "t1", "status": "completed"},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_complete_task",
                        {"task_id": "t1", "status": "completed", "result": {"value": 7}},
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["task_id"], "t1")
            self.assertEqual(result["status"], "completed")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_heartbeat_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p heartbeat and pass through response payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-heartbeat-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_heartbeat", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._heartbeat",
                return_value={"ok": True, "heartbeat": "accepted"},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_heartbeat",
                        {"peer_id": "peer-1", "clock": {"epoch": 1}},
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["heartbeat"], "accepted")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_cache_get_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p cache-get and pass through response payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-cache-get-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_cache_get", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._cache_get",
                return_value={"ok": True, "key": "k1", "value": {"n": 1}},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_cache_get",
                        {"key": "k1", "timeout_s": 2.0},
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["key"], "k1")
            self.assertEqual(result["value"], {"n": 1})

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_cache_set_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p cache-set and pass through response payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-cache-set-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_cache_set", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._cache_set",
                return_value={"ok": True, "key": "k1", "stored": True},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_cache_set",
                        {"key": "k1", "value": {"n": 2}, "ttl_s": 30.0, "timeout_s": 2.0},
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["key"], "k1")
            self.assertTrue(result["stored"])

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_submit_docker_hub_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p submit-docker-hub and pass through task id."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-submit-docker-hub-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_submit_docker_hub", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._submit_docker_hub",
                return_value="task-123",
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_submit_docker_hub",
                        {
                            "image": "alpine:latest",
                            "command": ["echo", "hi"],
                            "environment": {"A": "1"},
                        },
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["task_id"], "task-123")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_submit_docker_github_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p submit-docker-github and pass through task id."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-submit-docker-github-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_submit_docker_github", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._submit_docker_github",
                return_value="task-456",
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_submit_docker_github",
                        {
                            "repo_url": "https://github.com/example/repo",
                            "branch": "main",
                            "dockerfile_path": "Dockerfile",
                            "context_path": ".",
                            "build_args": {"A": "1"},
                        },
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["task_id"], "task-456")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_submit_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p submit and pass through result payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-submit-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_submit", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._submit_task_with_info",
                return_value={"task_id": "task-999", "accepted": True},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_submit",
                        {
                            "task_type": "inference",
                            "model_name": "model-a",
                            "payload": {"prompt": "hi"},
                        },
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["task_id"], "task-999")
            self.assertTrue(result["accepted"])

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_claim_next_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p claim-next and pass through task payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-claim-next-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_claim_next", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._claim_next",
                return_value={"task_id": "task-claim-1", "status": "claimed"},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_claim_next",
                        {
                            "worker_id": "worker-a",
                            "supported_task_types": ["inference"],
                            "peer_id": "peer-a",
                        },
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["task"]["task_id"], "task-claim-1")
            self.assertEqual(result["task"]["status"], "claimed")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_list_peers_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native list_peers and pass through peer payload."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-list-peers-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("list_peers", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._list_peers",
                return_value={
                    "ok": True,
                    "count": 1,
                    "peers": [{"peer_id": "peer-1", "multiaddr": "/ip4/127.0.0.1/tcp/4001"}],
                },
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "list_peers",
                        {"discover": True, "limit": 10},
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["peers"][0]["peer_id"], "peer-1")

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_p2p_call_tool_dispatch_via_meta_tools_success(self, mock_wrapper):
        """Unified `tools_dispatch` should execute native p2p call_tool with payload `tool_name` key."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "p2p",
            },
            clear=False,
        ):
            server = create_mcp_server(name="native-p2p-call-tool-dispatch")

        async def _run_dispatch() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            p2p_tools = await tools_list("p2p")
            p2p_names = [tool["name"] for tool in p2p_tools["tools"]]
            self.assertIn("p2p_taskqueue_call_tool", p2p_names)

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools._call_tool",
                return_value={"ok": True, "result": {"pong": True}},
            ):
                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "p2p",
                        "p2p_taskqueue_call_tool",
                        {"tool_name": "health_ping", "args": {"x": 1}},
                    )
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"], {"pong": True})

        anyio.run(_run_dispatch)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_artifact_envelope_opt_in(self, mock_wrapper):
        """`tools_dispatch` should emit CID artifact chain when opt-in control flag is present."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-artifacts-opt-in")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]
            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__emit_artifacts": True,
                    "__correlation_id": "corr-123",
                    "__proof_cid": "cid-proof",
                    "__policy_cid": "cid-policy",
                    "__parent_event_cids": ["cid-parent-1"],
                },
            )

            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["echo"], "ok")

            artifacts = response["artifacts"]
            for key in ["input_cid", "intent_cid", "decision_cid", "output_cid", "receipt_cid", "event_cid"]:
                self.assertTrue(artifacts[key].startswith("cidv1-sha256-"))

            payloads = response["artifact_payloads"]
            self.assertEqual(payloads["intent"]["correlation_id"], "corr-123")
            self.assertEqual(payloads["decision"]["policy_cid"], "cid-policy")
            self.assertEqual(payloads["event"]["parents"], ["cid-parent-1"])

            artifact_store = response.get("artifact_store") or {}
            self.assertTrue(artifact_store.get("persisted"))
            self.assertEqual(int(artifact_store.get("written") or 0), 6)

            event_payload = server._unified_artifact_store.get(artifacts["event_cid"])
            self.assertIsNotNone(event_payload)
            self.assertEqual((event_payload or {}).get("intent_cid"), artifacts["intent_cid"])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_artifact_envelope_default_policy_from_config(self, mock_wrapper):
        """`tools_dispatch` should emit artifacts by default when config flag enables CID artifacts."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_CID_ARTIFACTS": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-artifacts-default-policy")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]
            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__correlation_id": "corr-default",
                },
            )

            self.assertTrue(response["ok"])
            self.assertIn("artifacts", response)
            self.assertIn("artifact_payloads", response)
            self.assertTrue((response.get("artifact_store") or {}).get("persisted"))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_ucan_denies_without_chain(self, mock_wrapper):
        """`tools_dispatch` should deny execution when UCAN enforcement is enabled and no chain is provided."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-deny")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [],
                },
            )

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"], "authorization_denied")
            self.assertEqual(response["authorization"]["scheme"], "ucan")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_ucan_allows_valid_chain(self, mock_wrapper):
        """`tools_dispatch` should allow execution when UCAN chain is valid for actor/capability."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-allow")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [
                        {
                            "issuer": "did:user:alice",
                            "audience": "did:model:planner",
                            "capabilities": [{"resource": "*", "ability": "invoke"}],
                        },
                        {
                            "issuer": "did:model:planner",
                            "audience": "did:model:worker",
                            "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                        },
                    ],
                },
            )

            self.assertEqual(response["echo"], "ok")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_ucan_allows_valid_signed_chain(self, mock_wrapper):
        """`tools_dispatch` should allow execution when signature-required UCAN chain validates."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-signed-allow")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            unsigned_chain = parse_delegation_chain(
                [
                    {
                        "issuer": "did:user:alice",
                        "audience": "did:model:planner",
                        "capabilities": [{"resource": "*", "ability": "invoke"}],
                    },
                    {
                        "issuer": "did:model:planner",
                        "audience": "did:model:worker",
                        "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    },
                ]
            )
            issuer_keys = {
                "did:user:alice": "pk-alice",
                "did:model:planner": "pk-planner",
            }
            signed_chain = []
            for d in unsigned_chain:
                signed_chain.append(
                    {
                        "issuer": d.issuer,
                        "audience": d.audience,
                        "capabilities": [{"resource": c.resource, "ability": c.ability} for c in d.capabilities],
                        "proof_cid": compute_delegation_proof_cid(d),
                        "signature": compute_delegation_signature(delegation=d, issuer_key_hint=issuer_keys.get(d.issuer, "")),
                    }
                )

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": signed_chain,
                    "__ucan_require_signatures": True,
                    "__ucan_issuer_public_keys": issuer_keys,
                },
            )

            self.assertEqual(response["echo"], "ok")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_ucan_denies_revoked_signed_proof(self, mock_wrapper):
        """`tools_dispatch` should deny execution when signed UCAN proof is revoked."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-signed-revoked")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            delegation = parse_delegation_chain(
                [
                    {
                        "issuer": "did:user:alice",
                        "audience": "did:model:worker",
                        "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    }
                ]
            )[0]
            issuer_key = "pk-alice"
            proof_cid = compute_delegation_proof_cid(delegation)
            signature = compute_delegation_signature(delegation=delegation, issuer_key_hint=issuer_key)

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [
                        {
                            "issuer": delegation.issuer,
                            "audience": delegation.audience,
                            "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                            "proof_cid": proof_cid,
                            "signature": signature,
                        }
                    ],
                    "__ucan_require_signatures": True,
                    "__ucan_issuer_public_keys": {"did:user:alice": issuer_key},
                    "__ucan_revoked_proof_cids": [proof_cid],
                },
            )

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"], "authorization_denied")
            self.assertEqual(response["authorization"]["reason"], "revoked_proof_at_hop_0")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    @unittest.skipUnless(HAVE_CRYPTO_ED25519, "cryptography ed25519 unavailable")
    def test_tools_dispatch_ucan_allows_valid_did_key_ed25519_chain(self, mock_wrapper):
        """`tools_dispatch` should allow Ed25519 signed UCAN chain using did:key public key material."""

        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        import base64

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        def _base58btc_encode(raw: bytes) -> str:
            alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            data = bytes(raw)
            zeros = 0
            for b in data:
                if b != 0:
                    break
                zeros += 1
            value = int.from_bytes(data, "big")
            out = ""
            while value > 0:
                value, rem = divmod(value, 58)
                out = alphabet[rem] + out
            return ("1" * zeros) + (out or "")

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-did-key-ed25519")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            delegation = parse_delegation_chain(
                [
                    {
                        "issuer": "did:user:alice",
                        "audience": "did:model:worker",
                        "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    }
                ]
            )[0]

            private = Ed25519PrivateKey.generate()
            public = private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            private_bytes = private.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )

            private_b64 = base64.urlsafe_b64encode(private_bytes).decode("ascii").rstrip("=")
            did_key = "did:key:z" + _base58btc_encode(bytes([0xED, 0x01]) + public)
            proof_cid = compute_delegation_proof_cid(delegation)
            signature = compute_delegation_signature_ed25519(
                delegation=delegation,
                private_key_b64=private_b64,
            )

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_require_signatures": True,
                    "__ucan_issuer_public_keys": {"did:user:alice": did_key},
                    "__ucan_proof_chain": [
                        {
                            "issuer": delegation.issuer,
                            "audience": delegation.audience,
                            "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                            "proof_cid": proof_cid,
                            "signature": signature,
                        }
                    ],
                },
            )

            self.assertEqual(response["echo"], "ok")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_ucan_caveat_requires_context_cids(self, mock_wrapper):
        """`tools_dispatch` should enforce UCAN caveat context CID requirements."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-caveat-context")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            chain = [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [
                        {
                            "actor_equals": "did:model:worker",
                            "context_cids_all": ["cid-required"],
                        }
                    ],
                }
            ]

            denied = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "x",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": chain,
                    "__ucan_context_cids": ["cid-other"],
                },
            )
            self.assertFalse(denied["ok"])
            self.assertEqual(denied["error"], "authorization_denied")
            self.assertEqual((denied.get("authorization") or {}).get("reason"), "caveat_denied_at_hop_0")

            allowed = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": chain,
                    "__ucan_context_cids": ["cid-required", "cid-extra"],
                },
            )
            self.assertEqual(allowed["echo"], "ok")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_ucan_allows_compact_token_envelope(self, mock_wrapper):
        """`tools_dispatch` should accept compact UCAN token envelopes in proof chain entries."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-token-envelope")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            payload = {
                "iss": "did:user:alice",
                "aud": "did:model:worker",
                "exp": 4102444800,
                "att": {
                    "smoke.echo": {
                        "invoke": [{}],
                    }
                },
            }
            payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
            token = "e30." + payload_b64 + ".sig"

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [{"token": token}],
                },
            )

            self.assertEqual(response["echo"], "ok")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_ucan_caveat_actor_and_context_set_constraints(self, mock_wrapper):
        """`tools_dispatch` should enforce actor/context set caveat constraints."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-caveat-actor-context-set")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            chain = [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [
                        {
                            "actor_in": ["did:model:worker", "did:model:fallback"],
                            "actor_regex": r"did:model:[a-z]+",
                            "context_cids_any": ["cid-allow-a", "cid-allow-b"],
                            "context_cids_none": ["cid-blocked"],
                        }
                    ],
                }
            ]

            denied = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "x",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": chain,
                    "__ucan_context_cids": ["cid-blocked"],
                },
            )
            self.assertFalse(denied["ok"])
            self.assertEqual(denied["error"], "authorization_denied")
            self.assertEqual((denied.get("authorization") or {}).get("reason"), "caveat_denied_at_hop_0")

            allowed = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": chain,
                    "__ucan_context_cids": ["cid-allow-b", "cid-extra"],
                },
            )
            self.assertEqual(allowed["echo"], "ok")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_policy_denies(self, mock_wrapper):
        """`tools_dispatch` should deny execution when temporal policy evaluator returns deny."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-policy-deny")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_clauses": [
                        {
                            "clause_type": "prohibition",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        }
                    ],
                },
            )

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"], "policy_denied")
            self.assertEqual(response["policy"]["decision"], "deny")
            policy_decision = response.get("policy_decision") or {}
            self.assertTrue(str(policy_decision.get("decision_cid") or "").startswith("cidv1-sha256-"))
            self.assertTrue(policy_decision.get("persisted"))

            stored = server._unified_artifact_store.get(policy_decision.get("decision_cid"))
            self.assertIsNotNone(stored)
            self.assertEqual((stored or {}).get("decision"), "deny")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_combined_ucan_and_policy_denial_prefers_ucan(self, mock_wrapper):
        """When both controls are enforced and both deny, response remains explicit/auditable via UCAN denial path."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-ucan-policy-combined-deny")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [],
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_clauses": [
                        {
                            "clause_type": "prohibition",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        }
                    ],
                },
            )

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"], "authorization_denied")
            self.assertEqual((response.get("authorization") or {}).get("scheme"), "ucan")
            self.assertEqual((response.get("authorization") or {}).get("reason"), "missing_delegation_chain")
            self.assertNotIn("policy", response)
            self.assertNotIn("policy_decision", response)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_policy_allows_with_obligations(self, mock_wrapper):
        """`tools_dispatch` should return policy details when allowed with obligations."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-policy-obligations")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_clauses": [
                        {
                            "clause_type": "permission",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        },
                        {
                            "clause_type": "obligation",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                            "obligation_deadline": "2030-01-01T00:00:00Z",
                        },
                    ],
                },
            )

            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["echo"], "ok")
            self.assertEqual(response["policy"]["decision"], "allow_with_obligations")
            self.assertEqual(len(response["policy"]["obligations"]), 1)
            policy_decision = response.get("policy_decision") or {}
            self.assertTrue(str(policy_decision.get("decision_cid") or "").startswith("cidv1-sha256-"))
            self.assertTrue(policy_decision.get("persisted"))

            stored = server._unified_artifact_store.get(policy_decision.get("decision_cid"))
            self.assertIsNotNone(stored)
            self.assertEqual((stored or {}).get("decision"), "allow_with_obligations")
            self.assertEqual(len((stored or {}).get("obligations") or []), 1)
            self.assertEqual(((stored or {}).get("obligations") or [])[0].get("status"), "pending")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_policy_fulfilled_obligation_returns_allow(self, mock_wrapper):
        """Fulfilled obligations should not remain outstanding in dispatch responses or persisted decisions."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-policy-fulfilled-obligations")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            outstanding = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_cid": "cid-policy-v1",
                    "__policy_version": "v1",
                    "__policy_clauses": [
                        {
                            "clause_type": "permission",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        },
                        {
                            "clause_type": "obligation",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                            "obligation_deadline": "2030-01-01T00:00:00Z",
                            "metadata": {"ticket": "obl-1"},
                        },
                    ],
                },
            )

            fulfilled = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_cid": "cid-policy-v1",
                    "__policy_version": "v1",
                    "__policy_clauses": [
                        {
                            "clause_type": "permission",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        },
                        {
                            "clause_type": "obligation",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                            "obligation_deadline": "2030-01-01T00:00:00Z",
                            "metadata": {
                                "ticket": "obl-1",
                                "fulfilled": True,
                                "fulfilled_at": "2026-02-01T00:00:00Z",
                            },
                        },
                    ],
                },
            )

            self.assertTrue(outstanding["ok"])
            self.assertEqual(outstanding["policy"]["decision"], "allow_with_obligations")
            self.assertEqual(len(outstanding["policy"]["obligations"]), 1)

            self.assertTrue(fulfilled["ok"])
            self.assertEqual(fulfilled["policy"]["decision"], "allow")
            self.assertEqual(fulfilled["policy"]["obligations"], [])
            self.assertIn("already fulfilled", fulfilled["policy"]["justification"])

            outstanding_cid = str((outstanding.get("policy_decision") or {}).get("decision_cid") or "")
            fulfilled_cid = str((fulfilled.get("policy_decision") or {}).get("decision_cid") or "")
            self.assertTrue(outstanding_cid.startswith("cidv1-sha256-"))
            self.assertTrue(fulfilled_cid.startswith("cidv1-sha256-"))
            self.assertNotEqual(outstanding_cid, fulfilled_cid)

            stored_fulfilled = server._unified_artifact_store.get(fulfilled_cid) or {}
            self.assertEqual(stored_fulfilled.get("decision"), "allow")
            self.assertEqual(stored_fulfilled.get("obligations"), [])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_policy_decision_cid_stable_across_emit_modes_and_policy_evolution(self, mock_wrapper):
        """Policy decision CID should remain stable across emit modes and change deterministically when policy identity evolves."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-policy-version-cids")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            base_payload = {
                "value": "ok",
                "__enforce_policy": True,
                "__policy_actor": "did:model:worker",
                "__policy_version": "v1",
                "__policy_clauses": [
                    {
                        "clause_type": "permission",
                        "actor": "did:model:worker",
                        "action": "smoke.echo",
                    },
                    {
                        "clause_type": "obligation",
                        "actor": "did:model:worker",
                        "action": "smoke.echo",
                        "obligation_deadline": "2030-01-01T00:00:00Z",
                        "metadata": {"migration": "phase-a"},
                    },
                ],
            }

            response_no_emit = await dispatch("smoke", "echo", dict(base_payload, **{"__emit_artifacts": False}))
            response_emit = await dispatch("smoke", "echo", dict(base_payload, **{"__emit_artifacts": True}))

            cid_no_emit = str((response_no_emit.get("policy_decision") or {}).get("decision_cid") or "")
            cid_emit = str((response_emit.get("policy_decision") or {}).get("decision_cid") or "")
            self.assertTrue(cid_no_emit.startswith("cidv1-sha256-"))
            self.assertEqual(cid_no_emit, cid_emit)

            stored_v1 = server._unified_artifact_store.get(cid_no_emit) or {}
            self.assertEqual(stored_v1.get("policy_cid"), "")
            self.assertEqual(stored_v1.get("policy_version"), "v1")
            self.assertEqual(stored_v1.get("decision"), "allow_with_obligations")

            response_v2 = await dispatch(
                "smoke",
                "echo",
                dict(base_payload, **{"__policy_version": "v2", "__emit_artifacts": False}),
            )
            cid_v2 = str((response_v2.get("policy_decision") or {}).get("decision_cid") or "")
            self.assertTrue(cid_v2.startswith("cidv1-sha256-"))
            self.assertNotEqual(cid_no_emit, cid_v2)

            stored_v2 = server._unified_artifact_store.get(cid_v2) or {}
            self.assertEqual(stored_v2.get("policy_version"), "v2")

            response_policy_cid = await dispatch(
                "smoke",
                "echo",
                dict(base_payload, **{"__policy_cid": "cid-policy-v3", "__emit_artifacts": False}),
            )
            cid_policy_cid = str((response_policy_cid.get("policy_decision") or {}).get("decision_cid") or "")
            self.assertTrue(cid_policy_cid.startswith("cidv1-sha256-"))
            self.assertNotEqual(cid_no_emit, cid_policy_cid)

            stored_policy_cid = server._unified_artifact_store.get(cid_policy_cid) or {}
            self.assertEqual(stored_policy_cid.get("policy_cid"), "cid-policy-v3")
            self.assertEqual(stored_policy_cid.get("policy_version"), "v1")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_policy_decision_uses_envelope_cid_when_prepersist_fails(self, mock_wrapper):
        """Artifact emission should return envelope decision CID when policy pre-persist binding fails."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-policy-prepersist-fallback")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")

            original_put = server._unified_artifact_store.put
            put_calls = {"count": 0}

            def flaky_put(cid, payload):
                put_calls["count"] += 1
                if put_calls["count"] == 1:
                    raise RuntimeError("simulated pre-persist failure")
                return original_put(cid, payload)

            with patch.object(server._unified_artifact_store, "put", side_effect=flaky_put):
                response = await server.tools["tools_dispatch"]["function"](
                    "smoke",
                    "echo",
                    {
                        "value": "ok",
                        "__emit_artifacts": True,
                        "__enforce_policy": True,
                        "__policy_actor": "did:model:worker",
                        "__policy_clauses": [
                            {
                                "clause_type": "permission",
                                "actor": "did:model:worker",
                                "action": "smoke.echo",
                            }
                        ],
                    },
                )

            self.assertTrue(response["ok"])
            self.assertIn("artifacts", response)
            self.assertIn("policy_decision", response)

            decision_cid = str((response.get("policy_decision") or {}).get("decision_cid") or "")
            self.assertTrue(decision_cid.startswith("cidv1-sha256-"))
            self.assertEqual(decision_cid, (response.get("artifacts") or {}).get("decision_cid"))
            self.assertTrue((response.get("policy_decision") or {}).get("persisted"))

            stored = server._unified_artifact_store.get(decision_cid) or {}
            self.assertEqual(stored.get("decision"), "allow")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_risk_gating_denies_high_risk(self, mock_wrapper):
        """`tools_dispatch` should deny execution when risk scoring exceeds threshold."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_RISK_SCORING": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-risk-gate-deny")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "x",
                    "__risk_actor": "did:model:risky",
                    "__risk_policy": {
                        "tool_risk_overrides": {"smoke.echo": 1.0},
                        "max_acceptable_risk": 0.05,
                    },
                },
            )

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"], "risk_denied")
            assessment = response.get("risk_assessment") or {}
            self.assertFalse(assessment.get("is_acceptable", True))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_risk_gating_allows_with_assessment(self, mock_wrapper):
        """`tools_dispatch` should include risk assessment metadata when risk scoring is enabled."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_RISK_SCORING": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-risk-gate-allow")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__risk_actor": "did:model:worker",
                    "__risk_policy": {
                        "tool_risk_overrides": {"smoke.echo": 0.1},
                        "max_acceptable_risk": 0.95,
                    },
                },
            )

            self.assertEqual(response["echo"], "ok")
            assessment = response.get("risk_assessment") or {}
            self.assertTrue(assessment.get("is_acceptable", False))
            self.assertEqual(assessment.get("tool"), "smoke.echo")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_policy_audit_records_allow_and_deny(self, mock_wrapper):
        """`tools_dispatch` should record policy audit entries when policy audit is enabled."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                "IPFS_MCP_SERVER_ENABLE_POLICY_AUDIT": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-policy-audit")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            denied = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "x",
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_clauses": [
                        {
                            "clause_type": "prohibition",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        }
                    ],
                },
            )

            self.assertFalse(denied["ok"])
            self.assertEqual(denied["error"], "policy_denied")
            self.assertTrue((denied.get("audit") or {}).get("enabled"))

            allowed = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_clauses": [
                        {
                            "clause_type": "permission",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        }
                    ],
                },
            )

            self.assertTrue(allowed["ok"])
            audit = allowed.get("audit") or {}
            self.assertTrue(audit.get("enabled"))
            self.assertGreaterEqual(int(audit.get("total_recorded", 0)), 2)

            entries = server._unified_policy_audit.recent(10)
            self.assertGreaterEqual(len(entries), 2)
            self.assertEqual(entries[-1].decision, "allow")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_event_dag_persistence_with_parent_lineage(self, mock_wrapper):
        """Artifact emission should persist event nodes and expose deterministic lineage metadata."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-event-dag-lineage")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            first = await dispatch("smoke", "echo", {"value": "one", "__emit_artifacts": True})
            self.assertTrue(first["event_dag"]["persisted"])
            first_event = first["artifacts"]["event_cid"]

            second = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "two",
                    "__emit_artifacts": True,
                    "__parent_event_cids": [first_event],
                },
            )
            self.assertTrue(second["event_dag"]["persisted"])
            second_event = second["artifacts"]["event_cid"]

            self.assertEqual(second["event_dag"]["lineage"], [first_event, second_event])

            stored = server._unified_event_dag.get_event(second_event)
            self.assertIsNotNone(stored)
            self.assertEqual(stored["parents"], [first_event])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_event_dag_snapshot_replay_and_rollback(self, mock_wrapper):
        """Event DAG snapshots should support deterministic replay/rollback traversal."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-event-dag-snapshot")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            root = await dispatch("smoke", "echo", {"value": "root", "__emit_artifacts": True})
            root_event = root["artifacts"]["event_cid"]

            leaf = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "leaf",
                    "__emit_artifacts": True,
                    "__parent_event_cids": [root_event],
                },
            )
            leaf_event = leaf["artifacts"]["event_cid"]

            snapshot = server._unified_event_dag.export_snapshot()
            rebuilt = type(server._unified_event_dag).from_snapshot(snapshot)

            replay = rebuilt.replay_from_root(root_event)
            self.assertEqual(replay, [root_event, leaf_event])

            rollback = rebuilt.rollback_path(leaf_event)
            self.assertEqual(rollback, [leaf_event, root_event])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_event_dag_merge_fork_snapshot_is_deterministic(self, mock_wrapper):
        """Fork/merge event DAG paths should replay and snapshot deterministically."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-event-dag-merge-fork")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            root = await dispatch("smoke", "echo", {"value": "root", "__emit_artifacts": True})
            root_event = root["artifacts"]["event_cid"]

            branch_z = await dispatch(
                "smoke",
                "echo",
                {"value": "z", "__emit_artifacts": True, "__parent_event_cids": [root_event]},
            )
            branch_a = await dispatch(
                "smoke",
                "echo",
                {"value": "a", "__emit_artifacts": True, "__parent_event_cids": [root_event]},
            )
            branch_z_event = branch_z["artifacts"]["event_cid"]
            branch_a_event = branch_a["artifacts"]["event_cid"]

            merge = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "merge",
                    "__emit_artifacts": True,
                    "__parent_event_cids": [branch_z_event, branch_a_event],
                },
            )
            merge_event = merge["artifacts"]["event_cid"]

            expected_branch = min(branch_a_event, branch_z_event)
            self.assertEqual(merge["event_dag"]["lineage"], [root_event, expected_branch, merge_event])

            replay = server._unified_event_dag.replay_from_root(root_event)
            self.assertEqual(replay, [root_event, expected_branch, max(branch_a_event, branch_z_event), merge_event])
            self.assertEqual(replay.count(merge_event), 1)

            snapshot = server._unified_event_dag.export_snapshot()
            rebuilt = type(server._unified_event_dag).from_snapshot(snapshot)
            self.assertEqual(rebuilt.get_lineage(merge_event), [root_event, expected_branch, merge_event])
            self.assertEqual(rebuilt.replay_from_root(root_event), replay)
            self.assertEqual(rebuilt.rollback_path(merge_event), [merge_event, expected_branch, root_event])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_risk_tracking_and_frontier(self, mock_wrapper):
        """Dispatch should expose risk metadata and frontier stats for deny/success flows."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-risk-frontier")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            denied = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "x",
                    "__risk_actor": "did:model:risky",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:risky",
                    "__ucan_proof_chain": [],
                },
            )
            self.assertFalse(denied["ok"])
            self.assertIn("risk", denied)
            self.assertEqual(denied["risk"]["denied_count"], 1)

            allowed = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__risk_actor": "did:model:risky",
                    "__emit_artifacts": True,
                },
            )
            self.assertTrue(allowed["ok"])
            self.assertIn("risk", allowed)
            self.assertIn("frontier", allowed)
            self.assertTrue(allowed["frontier"]["enqueued"])
            self.assertGreaterEqual(allowed["frontier"]["stats"]["frontier_size"], 1)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_frontier_execution_binds_to_workflow_scheduler(self, mock_wrapper):
        """Frontier execution should bind popped items to workflow scheduler when available."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class FakeWorkflowScheduler:
            def __init__(self):
                self.calls = []

            async def submit_workflow(self, workflow_name, tasks, metadata=None):
                self.calls.append(
                    {
                        "workflow_name": workflow_name,
                        "tasks": tasks,
                        "metadata": metadata,
                    }
                )
                return {"workflow_id": "wf-risk-1"}

        fake_scheduler = FakeWorkflowScheduler()
        factory_calls = {"workflow_scheduler_factory": 0, "task_queue_factory": 0}
        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-frontier-workflow-bind")

        # Force deterministic local scheduler binding to avoid optional dependency behavior.
        def _workflow_scheduler_factory(**kwargs):
            _ = kwargs
            factory_calls["workflow_scheduler_factory"] += 1
            return fake_scheduler

        def _task_queue_factory(**kwargs):
            _ = kwargs
            factory_calls["task_queue_factory"] += 1
            return None

        server._unified_services["workflow_scheduler_factory"] = _workflow_scheduler_factory
        server._unified_services["task_queue_factory"] = _task_queue_factory

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__emit_artifacts": True,
                    "__risk_actor": "did:model:risk-exec",
                    "__execute_frontier": True,
                },
            )

            self.assertTrue(response["ok"])
            frontier = response.get("frontier") or {}
            execution = frontier.get("execution") or {}
            self.assertTrue(execution.get("attempted"))
            self.assertTrue(execution.get("scheduled"))
            self.assertEqual(execution.get("route"), "workflow_scheduler")
            self.assertEqual(execution.get("workflow_id"), "wf-risk-1")
            self.assertEqual(frontier.get("event_cid"), execution.get("event_cid"))

            self.assertEqual(len(fake_scheduler.calls), 1)
            call = fake_scheduler.calls[0]
            self.assertEqual(call["workflow_name"], "risk_frontier_dispatch")
            self.assertEqual(call["tasks"][0]["task_type"], "mcp.frontier.execute")
            self.assertEqual(call["tasks"][0]["payload"]["event_cid"], response["artifacts"]["event_cid"])
            self.assertEqual(factory_calls["workflow_scheduler_factory"], 1)
            self.assertEqual(factory_calls["task_queue_factory"], 0)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_frontier_execution_binds_to_task_queue_fallback(self, mock_wrapper):
        """Frontier execution should bind popped items to task queue when scheduler is unavailable."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class FakeTaskQueue:
            def __init__(self):
                self.calls = []

            async def submit(self, task_type, payload, priority=0):
                self.calls.append(
                    {
                        "task_type": task_type,
                        "payload": payload,
                        "priority": priority,
                    }
                )
                return "task-risk-1"

        fake_task_queue = FakeTaskQueue()
        factory_calls = {"workflow_scheduler_factory": 0, "task_queue_factory": 0}
        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-frontier-task-queue-bind")

        # Force deterministic fallback path to task queue adapter.
        def _workflow_scheduler_factory(**kwargs):
            _ = kwargs
            factory_calls["workflow_scheduler_factory"] += 1
            return None

        def _task_queue_factory(**kwargs):
            _ = kwargs
            factory_calls["task_queue_factory"] += 1
            return fake_task_queue

        server._unified_services["workflow_scheduler_factory"] = _workflow_scheduler_factory
        server._unified_services["task_queue_factory"] = _task_queue_factory

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__emit_artifacts": True,
                    "__risk_actor": "did:model:risk-exec-fallback",
                    "__execute_frontier": True,
                },
            )

            self.assertTrue(response["ok"])
            frontier = response.get("frontier") or {}
            execution = frontier.get("execution") or {}
            self.assertTrue(execution.get("attempted"))
            self.assertTrue(execution.get("scheduled"))
            self.assertEqual(execution.get("route"), "task_queue")
            self.assertEqual(execution.get("task_id"), "task-risk-1")
            self.assertEqual(frontier.get("event_cid"), execution.get("event_cid"))

            self.assertEqual(len(fake_task_queue.calls), 1)
            call = fake_task_queue.calls[0]
            self.assertEqual(call["task_type"], "mcp_frontier_event")
            self.assertEqual(call["payload"]["event_cid"], response["artifacts"]["event_cid"])
            self.assertIsInstance(call["priority"], int)
            self.assertEqual(factory_calls["workflow_scheduler_factory"], 1)
            self.assertEqual(factory_calls["task_queue_factory"], 1)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_result_cache_factory_consumed_on_cache_hit(self, mock_wrapper):
        """tools_dispatch should consume result_cache_factory and short-circuit on cache hit."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class FakeResultCache:
            def __init__(self):
                self.get_calls = []
                self.put_calls = []

            async def get(self, task_id, inputs=None):
                self.get_calls.append({"task_id": task_id, "inputs": dict(inputs or {})})
                return {"echo": "from-cache"}

            async def put(self, task_id, value, ttl=None, inputs=None):
                self.put_calls.append(
                    {
                        "task_id": task_id,
                        "value": value,
                        "ttl": ttl,
                        "inputs": dict(inputs or {}),
                    }
                )

        fake_cache = FakeResultCache()
        factory_calls = {"result_cache_factory": 0}
        dispatch_invocations = {"count": 0}
        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-result-cache-hit")

        def _result_cache_factory(**kwargs):
            _ = kwargs
            factory_calls["result_cache_factory"] += 1
            return fake_cache

        server._unified_services["result_cache_factory"] = _result_cache_factory

        async def _run_flow() -> None:
            async def echo(value: str):
                dispatch_invocations["count"] += 1
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ignored-by-cache",
                    "__use_result_cache": True,
                },
            )

            self.assertTrue(response["ok"])
            self.assertEqual(response["result"], {"echo": "from-cache"})
            self.assertEqual(dispatch_invocations["count"], 0)
            self.assertEqual(factory_calls["result_cache_factory"], 1)
            self.assertEqual(len(fake_cache.get_calls), 1)
            self.assertEqual(len(fake_cache.put_calls), 0)
            self.assertIn("cache", response)
            self.assertTrue(response["cache"]["enabled"])
            self.assertTrue(response["cache"]["hit"])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_peer_registry_factory_consumed_for_probe(self, mock_wrapper):
        """tools_dispatch should consume peer_registry_factory when peer probing is requested."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class FakePeerRegistry:
            def __init__(self):
                self.calls = []

            async def discover_peers(self, max_peers=50):
                self.calls.append({"max_peers": max_peers})
                return [
                    {"peer_id": "peer-a", "multiaddr": "/ip4/127.0.0.1/tcp/4001"},
                    {"peer_id": "peer-b", "multiaddr": "/ip4/127.0.0.1/tcp/4002"},
                ]

        fake_registry = FakePeerRegistry()
        factory_calls = {"peer_registry_factory": 0}
        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-peer-registry-probe")

        def _peer_registry_factory(**kwargs):
            _ = kwargs
            factory_calls["peer_registry_factory"] += 1
            return fake_registry

        server._unified_services["peer_registry_factory"] = _peer_registry_factory

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__discover_peers": True,
                    "__peer_probe_limit": 1,
                },
            )

            self.assertTrue(response["ok"])
            self.assertEqual(factory_calls["peer_registry_factory"], 1)
            self.assertEqual(len(fake_registry.calls), 1)
            self.assertEqual(fake_registry.calls[0]["max_peers"], 1)
            self.assertIn("peer_registry", response)
            self.assertTrue(response["peer_registry"]["enabled"])
            self.assertTrue(response["peer_registry"]["factory_used"])
            self.assertEqual(response["peer_registry"]["peer_count"], 1)
            self.assertEqual(len(response["peer_registry"]["peers"]), 1)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_ipfs_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """ipfs_tools should expose source-compatible operations and dispatch envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="ipfs-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("ipfs_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("pin_to_ipfs", names)
            self.assertIn("get_from_ipfs", names)

            pin_schema = await get_schema("ipfs_tools", "pin_to_ipfs")
            self.assertEqual(pin_schema.get("name"), "pin_to_ipfs")
            self.assertEqual(pin_schema.get("category"), "ipfs_tools")
            self.assertIn("content_source", (pin_schema.get("input_schema") or {}).get("properties", {}))

            get_schema_payload = await get_schema("ipfs_tools", "get_from_ipfs")
            self.assertEqual(get_schema_payload.get("name"), "get_from_ipfs")
            self.assertEqual(get_schema_payload.get("category"), "ipfs_tools")
            get_schema_input = get_schema_payload.get("input_schema") or {}
            get_schema_props = get_schema_input.get("properties", {})
            self.assertIn("cid", get_schema_props)
            self.assertEqual((get_schema_props.get("timeout_seconds") or {}).get("minimum"), 1)

            pin_result = self._assert_dispatch_success_envelope(
                await dispatch(

                    "ipfs_tools",
                    "pin_to_ipfs",
                    {
                        "content_source": "ipfs-tools-smoke",
                        "recursive": True,
                    },
                )
            )
            self.assertTrue("status" in pin_result or "success" in pin_result or "message" in pin_result)

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs_tools",
                    "get_from_ipfs",
                    {
                        "cid": "QmDemoHash",
                        "timeout_seconds": "bad",
                    },
                )
            )
            self.assertEqual(invalid_timeout.get("status"), "error")
            self.assertIn("must be an integer", str(invalid_timeout.get("message", "")))

            invalid_gateway = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs_tools",
                    "get_from_ipfs",
                    {
                        "cid": "QmDemoHash",
                        "gateway": "ipfs://localhost:8080",
                    },
                )
            )
            self.assertEqual(invalid_gateway.get("status"), "error")
            self.assertIn("must start with", str(invalid_gateway.get("message", "")))

            json_entrypoint_missing_cid = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs_tools",
                    "get_from_ipfs",
                    {
                        "cid": json.dumps({"output_path": "/tmp/no-cid"}),
                    },
                )
            )
            self.assertIn("content", json_entrypoint_missing_cid)
            parsed_missing_cid = json.loads(
                ((json_entrypoint_missing_cid.get("content") or [{}])[0]).get("text", "{}")
            )
            self.assertEqual(parsed_missing_cid.get("status"), "error")
            self.assertIn("Missing required field: cid", str(parsed_missing_cid.get("error", "")))

            json_entrypoint_invalid = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs_tools",
                    "pin_to_ipfs",
                    {
                        "content_source": "{not-json",
                    },
                )
            )
            self.assertIn("content", json_entrypoint_invalid)
            parsed_invalid = json.loads(((json_entrypoint_invalid.get("content") or [{}])[0]).get("text", "{}"))
            self.assertEqual(parsed_invalid.get("status"), "error")
            self.assertEqual(parsed_invalid.get("error_type"), "validation")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_ipfs_native_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """ipfs native category should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="ipfs-native-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("ipfs")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("ipfs_files_validate_cid", names)
            self.assertIn("ipfs_files_get_file", names)

            validate_schema = await get_schema("ipfs", "ipfs_files_validate_cid")
            validate_props = (validate_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((validate_props.get("cid") or {}).get("minLength"), 1)

            get_schema_payload = await get_schema("ipfs", "ipfs_files_get_file")
            get_props = (get_schema_payload.get("input_schema") or {}).get("properties", {})
            self.assertEqual((get_props.get("output_path") or {}).get("minLength"), 1)

            invalid_cid = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs",
                    "ipfs_files_validate_cid",
                    {"cid": "   "},
                )
            )
            self.assertFalse(invalid_cid.get("success"))
            self.assertIn("cid must be a non-empty string", str(invalid_cid.get("error", "")))

            invalid_output = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs",
                    "ipfs_files_get_file",
                    {"cid": "bafy-demo", "output_path": "   "},
                )
            )
            self.assertFalse(invalid_output.get("success"))
            self.assertIn("output_path must be a non-empty string", str(invalid_output.get("error", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_workflow_native_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """workflow native category should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="workflow-native-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("workflow")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("get_workflow_templates", names)
            self.assertIn("create_workflow", names)

            get_schema_payload = await get_schema("workflow", "get_workflow")
            get_props = (get_schema_payload.get("input_schema") or {}).get("properties", {})
            self.assertEqual((get_props.get("workflow_id") or {}).get("minLength"), 1)

            create_schema = await get_schema("workflow", "create_workflow")
            create_props = (create_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((create_props.get("name") or {}).get("minLength"), 1)

            invalid_get = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow",
                    "get_workflow",
                    {"workflow_id": "   "},
                )
            )
            self.assertEqual(invalid_get.get("status"), "error")
            self.assertIn("workflow_id must be a non-empty string", str(invalid_get.get("error", "")))

            invalid_create = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow",
                    "create_workflow",
                    {
                        "name": "",
                        "description": "demo",
                        "tasks": [],
                    },
                )
            )
            self.assertEqual(invalid_create.get("status"), "error")
            self.assertIn("name must be a non-empty string", str(invalid_create.get("error", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_p2p_native_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """p2p native category should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="p2p-native-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("p2p")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("p2p_taskqueue_status", names)
            self.assertIn("p2p_taskqueue_submit", names)

            status_schema = await get_schema("p2p", "p2p_taskqueue_status")
            status_props = (status_schema.get("input_schema") or {}).get("properties", {})
            self.assertGreater((status_props.get("timeout_s") or {}).get("minimum", 0), 0)

            submit_schema = await get_schema("p2p", "p2p_taskqueue_submit")
            submit_props = (submit_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((submit_props.get("task_type") or {}).get("minLength"), 1)

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p",
                    "p2p_taskqueue_status",
                    {"timeout_s": 0},
                )
            )
            self.assertFalse(invalid_timeout.get("ok"))
            self.assertIn("timeout_s must be a number > 0", str(invalid_timeout.get("error", "")))

            invalid_submit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p",
                    "p2p_taskqueue_submit",
                    {
                        "task_type": "",
                        "model_name": "demo-model",
                        "payload": {},
                    },
                )
            )
            self.assertFalse(invalid_submit.get("ok"))
            self.assertIn("task_type must be a non-empty string", str(invalid_submit.get("error", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_rate_limiting_tools_category_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """rate_limiting_tools alias category should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="rate-limiting-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("rate_limiting_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("check_rate_limit", names)
            self.assertIn("manage_rate_limits", names)

            check_schema = await get_schema("rate_limiting_tools", "check_rate_limit")
            check_props = (check_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((check_props.get("limit_name") or {}).get("minLength"), 1)

            manage_schema = await get_schema("rate_limiting_tools", "manage_rate_limits")
            manage_props = (manage_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("update", (manage_props.get("action") or {}).get("enum", []))

            invalid_identifier = self._assert_dispatch_success_envelope(
                await dispatch(
                    "rate_limiting_tools",
                    "check_rate_limit",
                    {
                        "limit_name": "api",
                        "identifier": "   ",
                    },
                )
            )
            self.assertEqual(invalid_identifier.get("status"), "error")
            self.assertIn(
                "identifier must be a non-empty string",
                str(invalid_identifier.get("error", "")),
            )

            invalid_stats_limit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "rate_limiting_tools",
                    "manage_rate_limits",
                    {
                        "action": "stats",
                        "limit_name": "   ",
                    },
                )
            )
            self.assertEqual(invalid_stats_limit.get("status"), "error")
            self.assertIn(
                "limit_name must be a non-empty string when provided for stats",
                str(invalid_stats_limit.get("error", "")),
            )

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_rate_limiting_native_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """rate_limiting native category should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="rate-limiting-native-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("rate_limiting")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("check_rate_limit", names)
            self.assertIn("manage_rate_limits", names)

            check_schema = await get_schema("rate_limiting", "check_rate_limit")
            check_props = (check_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((check_props.get("limit_name") or {}).get("minLength"), 1)

            manage_schema = await get_schema("rate_limiting", "manage_rate_limits")
            manage_props = (manage_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("update", (manage_props.get("action") or {}).get("enum", []))

            invalid_identifier = self._assert_dispatch_success_envelope(
                await dispatch(
                    "rate_limiting",
                    "check_rate_limit",
                    {
                        "limit_name": "api",
                        "identifier": "   ",
                    },
                )
            )
            self.assertEqual(invalid_identifier.get("status"), "error")
            self.assertIn(
                "identifier must be a non-empty string",
                str(invalid_identifier.get("error", "")),
            )

            invalid_stats_limit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "rate_limiting",
                    "manage_rate_limits",
                    {
                        "action": "stats",
                        "limit_name": "   ",
                    },
                )
            )
            self.assertEqual(invalid_stats_limit.get("status"), "error")
            self.assertIn(
                "limit_name must be a non-empty string when provided for stats",
                str(invalid_stats_limit.get("error", "")),
            )

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_p2p_workflow_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """p2p_workflow_tools category should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="p2p-workflow-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("p2p_workflow_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("initialize_p2p_scheduler", names)
            self.assertIn("schedule_p2p_workflow", names)
            self.assertIn("get_workflow_tags", names)
            self.assertIn("add_p2p_peer", names)
            self.assertIn("remove_p2p_peer", names)
            self.assertIn("calculate_peer_distance", names)
            self.assertIn("merge_merkle_clock", names)

            schedule_schema = await get_schema("p2p_workflow_tools", "schedule_p2p_workflow")
            schedule_props = (schedule_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((schedule_props.get("workflow_id") or {}).get("minLength"), 1)
            self.assertGreater((schedule_props.get("priority") or {}).get("minimum", 0), 0)

            add_peer_schema = await get_schema("p2p_workflow_tools", "add_p2p_peer")
            add_peer_props = (add_peer_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((add_peer_props.get("peer_id") or {}).get("minLength"), 1)

            invalid_tags = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p_workflow_tools",
                    "schedule_p2p_workflow",
                    {
                        "workflow_id": "wf-1",
                        "name": "demo",
                        "tags": [],
                    },
                )
            )
            self.assertEqual(invalid_tags.get("status"), "error")
            self.assertIn("non-empty array of strings", str(invalid_tags.get("message", "")))

            invalid_peers = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p_workflow_tools",
                    "initialize_p2p_scheduler",
                    {
                        "peers": ["peer-1", ""],
                    },
                )
            )
            self.assertEqual(invalid_peers.get("status"), "error")
            self.assertIn("array of non-empty strings", str(invalid_peers.get("message", "")))

            invalid_peer_id = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p_workflow_tools",
                    "add_p2p_peer",
                    {
                        "peer_id": "   ",
                    },
                )
            )
            self.assertEqual(invalid_peer_id.get("status"), "error")
            self.assertIn("peer_id must be a non-empty string", str(invalid_peer_id.get("error", "")))

            workflow_tags = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p_workflow_tools",
                    "get_workflow_tags",
                    {},
                )
            )
            self.assertEqual(workflow_tags.get("status"), "success")
            self.assertIn("tags", workflow_tags)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_search_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """search_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="search-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("search_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("semantic_search", names)
            self.assertIn("similarity_search", names)

            semantic_schema = await get_schema("search_tools", "semantic_search")
            semantic_props = (semantic_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((semantic_props.get("query") or {}).get("minLength"), 1)
            self.assertEqual((semantic_props.get("top_k") or {}).get("minimum"), 1)

            similarity_schema = await get_schema("search_tools", "similarity_search")
            similarity_props = (similarity_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((similarity_props.get("embedding") or {}).get("minItems"), 1)
            self.assertEqual((similarity_props.get("threshold") or {}).get("maximum"), 1.0)

            faceted_schema = await get_schema("search_tools", "faceted_search")
            faceted_props = (faceted_schema.get("input_schema") or {}).get("properties", {})
            facets_schema = faceted_props.get("facets") or {}
            facet_items = (facets_schema.get("additionalProperties") or {}).get("items") or {}
            self.assertEqual(facet_items.get("minLength"), 1)
            self.assertEqual(
                ((faceted_props.get("aggregations") or {}).get("items") or {}).get("minLength"),
                1,
            )

            invalid_query = self._assert_dispatch_success_envelope(
                await dispatch(
                    "search_tools",
                    "semantic_search",
                    {
                        "query": "   ",
                    },
                )
            )
            self.assertEqual(invalid_query.get("status"), "error")
            self.assertIn("query is required", str(invalid_query.get("message", "")))

            invalid_filters = self._assert_dispatch_success_envelope(
                await dispatch(
                    "search_tools",
                    "semantic_search",
                    {
                        "query": "hello",
                        "filters": ["not-an-object"],
                    },
                )
            )
            self.assertEqual(invalid_filters.get("status"), "error")
            self.assertIn("filters must be an object", str(invalid_filters.get("message", "")))

            invalid_top_k = self._assert_dispatch_success_envelope(
                await dispatch(
                    "search_tools",
                    "faceted_search",
                    {
                        "query": "hello",
                        "top_k": "oops",
                    },
                )
            )
            self.assertEqual(invalid_top_k.get("status"), "error")
            self.assertIn("top_k must be a positive integer", str(invalid_top_k.get("message", "")))

            invalid_facets = self._assert_dispatch_success_envelope(
                await dispatch(
                    "search_tools",
                    "faceted_search",
                    {
                        "query": "hello",
                        "facets": {"category": "news"},
                    },
                )
            )
            self.assertEqual(invalid_facets.get("status"), "error")
            self.assertIn(
                "each facets value must be an array of non-empty strings",
                str(invalid_facets.get("message", "")),
            )

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_vector_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """vector_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="vector-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("vector_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("create_vector_index", names)
            self.assertIn("search_vector_index", names)
            self.assertIn("list_vector_indexes", names)
            self.assertIn("manage_vector_store", names)
            self.assertIn("create_store", names)
            self.assertIn("list_stores", names)
            self.assertIn("get_vector_store_info", names)
            self.assertIn("save_store", names)
            self.assertIn("delete_vector_index", names)

            create_schema = await get_schema("vector_tools", "create_vector_index")
            create_props = (create_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((create_props.get("vectors") or {}).get("minItems"), 1)
            self.assertEqual((create_props.get("dimension") or {}).get("minimum"), 1)

            search_schema = await get_schema("vector_tools", "search_vector_index")
            search_props = (search_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((search_props.get("index_id") or {}).get("minLength"), 1)
            self.assertEqual((search_props.get("query_vector") or {}).get("minItems"), 1)

            list_indexes_schema = await get_schema("vector_tools", "list_vector_indexes")
            list_indexes_props = (list_indexes_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((list_indexes_props.get("backend") or {}).get("default"), "all")

            manage_schema = await get_schema("vector_tools", "manage_vector_store")
            manage_props = (manage_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("create", (manage_props.get("operation") or {}).get("enum", []))
            self.assertEqual((manage_props.get("top_k") or {}).get("minimum"), 1)

            create_store_schema = await get_schema("vector_tools", "create_store")
            create_store_props = (create_store_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((create_store_props.get("name") or {}).get("minLength"), 1)

            list_stores_schema = await get_schema("vector_tools", "list_stores")
            list_stores_props = (list_stores_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((list_stores_props.get("include_details") or {}).get("default"), False)

            invalid_vectors = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "create_vector_index",
                    {
                        "vectors": [[1.0], []],
                    },
                )
            )
            self.assertEqual(invalid_vectors.get("status"), "error")
            self.assertIn("non-empty numeric vectors", str(invalid_vectors.get("error", "")))

            invalid_query_vector = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "search_vector_index",
                    {
                        "index_id": "idx",
                        "query_vector": [],
                    },
                )
            )
            self.assertEqual(invalid_query_vector.get("status"), "error")
            self.assertIn("query_vector must be a non-empty list of numbers", str(invalid_query_vector.get("error", "")))

            invalid_manage = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "manage_vector_store",
                    {
                        "operation": "rename",
                    },
                )
            )
            self.assertEqual(invalid_manage.get("status"), "error")
            self.assertIn("operation must be one of", str(invalid_manage.get("error", "")))

            invalid_load = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "load_store",
                    {
                        "name": "legal",
                        "create_if_missing": "yes",
                    },
                )
            )
            self.assertEqual(invalid_load.get("status"), "error")
            self.assertIn("create_if_missing must be a boolean", str(invalid_load.get("error", "")))

            created_store = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "create_store",
                    {
                        "name": "legal",
                        "backend": "faiss",
                    },
                )
            )
            self.assertEqual(created_store.get("status"), "success")
            self.assertEqual(created_store.get("store_name"), "legal")

            indexed_store = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "manage_vector_store",
                    {
                        "operation": "index",
                        "store_type": "faiss",
                        "collection_name": "legal",
                        "vectors": [[1.0, 2.0], [3.0, 4.0]],
                        "ids": ["doc-1", "doc-2"],
                        "metadata": [{"topic": "alpha"}, {"topic": "beta"}],
                    },
                )
            )
            self.assertEqual(indexed_store.get("status"), "success")
            self.assertEqual(indexed_store.get("indexed_count"), 2)

            queried_store = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "manage_vector_store",
                    {
                        "operation": "query",
                        "store_type": "faiss",
                        "collection_name": "legal",
                        "query_vector": [1.0, 2.0],
                        "top_k": 1,
                    },
                )
            )
            self.assertEqual(queried_store.get("status"), "success")
            self.assertEqual(queried_store.get("results_count"), 1)

            listed_stores = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "list_stores",
                    {
                        "backend": "all",
                        "include_details": True,
                    },
                )
            )
            self.assertEqual(listed_stores.get("status"), "success")
            self.assertTrue(any(store.get("store_name") == "legal" for store in listed_stores.get("stores", [])))

            store_info = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "get_vector_store_info",
                    {
                        "store_name": "legal",
                        "backend": "faiss",
                    },
                )
            )
            self.assertEqual(store_info.get("status"), "success")
            self.assertEqual(store_info.get("vector_count"), 2)

            saved_store = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "save_store",
                    {
                        "store_name": "legal",
                        "backend": "faiss",
                    },
                )
            )
            self.assertEqual(saved_store.get("status"), "success")
            self.assertEqual(saved_store.get("saved"), True)

            deleted_store = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_tools",
                    "delete_vector_index",
                    {
                        "index_name": "legal",
                        "backend": "faiss",
                    },
                )
            )
            self.assertIn(deleted_store.get("status"), ["success", "error"])
            self.assertEqual(deleted_store.get("backend"), "faiss")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_storage_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """storage_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="storage-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("storage_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("store_data", names)
            self.assertIn("query_storage", names)
            self.assertIn("list_storage", names)
            self.assertIn("get_storage_stats", names)
            self.assertIn("get_storage_collection_stats", names)
            self.assertIn("get_storage_lifecycle_report", names)
            self.assertIn("get_storage_backend_status", names)
            self.assertIn("list_storage_collections", names)
            self.assertIn("create_storage_collection", names)
            self.assertIn("get_storage_collection", names)
            self.assertIn("delete_storage_collection", names)
            self.assertIn("delete_data", names)

            store_schema = await get_schema("storage_tools", "store_data")
            store_props = (store_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((store_props.get("collection") or {}).get("minLength"), 1)
            self.assertEqual(
                ((store_props.get("tags") or {}).get("items") or {}).get("minLength"),
                1,
            )

            retrieve_schema = await get_schema("storage_tools", "retrieve_data")
            retrieve_props = (retrieve_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                ((retrieve_props.get("item_ids") or {}).get("items") or {}).get("minLength"),
                1,
            )
            self.assertEqual((retrieve_props.get("item_ids") or {}).get("minItems"), 1)

            query_schema = await get_schema("storage_tools", "query_storage")
            query_props = (query_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((query_props.get("limit") or {}).get("minimum"), 1)
            self.assertEqual((query_props.get("offset") or {}).get("minimum"), 0)
            self.assertEqual(
                ((query_props.get("size_range") or {}).get("items") or {}).get("minimum"),
                0,
            )
            self.assertEqual(
                ((query_props.get("date_range") or {}).get("items") or {}).get("format"),
                "date-time",
            )
            self.assertEqual(
                ((query_props.get("date_range") or {}).get("items") or {}).get("minLength"),
                1,
            )

            list_schema = await get_schema("storage_tools", "list_storage")
            list_props = (list_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((list_props.get("limit") or {}).get("minimum"), 1)
            self.assertEqual((list_props.get("offset") or {}).get("minimum"), 0)

            stats_schema = await get_schema("storage_tools", "get_storage_stats")
            stats_props = (stats_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((stats_props.get("report_format") or {}).get("default"), "summary")

            collection_stats_schema = await get_schema("storage_tools", "get_storage_collection_stats")
            collection_stats_input = collection_stats_schema.get("input_schema") or {}
            collection_stats_props = collection_stats_input.get("properties", {})
            self.assertEqual(collection_stats_input.get("required"), ["collection_name"])
            self.assertEqual((collection_stats_props.get("report_format") or {}).get("default"), "summary")

            lifecycle_alias_schema = await get_schema("storage_tools", "get_storage_lifecycle_report")
            lifecycle_alias_props = (lifecycle_alias_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((lifecycle_alias_props.get("report_format") or {}).get("default"), "detailed")
            self.assertEqual((lifecycle_alias_props.get("include_breakdown") or {}).get("default"), False)

            backend_alias_schema = await get_schema("storage_tools", "get_storage_backend_status")
            backend_alias_props = (backend_alias_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((backend_alias_props.get("availability_filter") or {}).get("default"), "all")
            self.assertEqual((backend_alias_props.get("include_capabilities") or {}).get("default"), False)
            self.assertEqual(
                ((backend_alias_props.get("unavailable_reasons") or {}).get("propertyNames") or {}).get("minLength"),
                1,
            )

            collections_alias_schema = await get_schema("storage_tools", "list_storage_collections")
            collections_alias_props = (collections_alias_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((collections_alias_props.get("include_metadata") or {}).get("default"), True)
            self.assertEqual((collections_alias_props.get("include_timestamps") or {}).get("default"), True)

            create_collection_schema = await get_schema("storage_tools", "create_storage_collection")
            create_collection_input = create_collection_schema.get("input_schema") or {}
            create_collection_props = create_collection_input.get("properties", {})
            self.assertEqual(create_collection_input.get("required"), ["collection_name"])
            self.assertEqual(
                ((create_collection_props.get("metadata") or {}).get("propertyNames") or {}).get("minLength"),
                1,
            )

            get_collection_schema = await get_schema("storage_tools", "get_storage_collection")
            get_collection_input = get_collection_schema.get("input_schema") or {}
            get_collection_props = get_collection_input.get("properties", {})
            self.assertEqual(get_collection_input.get("required"), ["collection_name"])
            self.assertEqual((get_collection_props.get("include_metadata") or {}).get("default"), True)
            self.assertEqual((get_collection_props.get("include_timestamps") or {}).get("default"), True)
            self.assertEqual((create_collection_input.get("properties", {}).get("description") or {}).get("minLength"), 1)

            delete_collection_schema = await get_schema("storage_tools", "delete_storage_collection")
            delete_collection_input = delete_collection_schema.get("input_schema") or {}
            delete_collection_props = delete_collection_input.get("properties", {})
            self.assertEqual(delete_collection_input.get("required"), ["collection_name"])
            self.assertEqual((delete_collection_props.get("delete_items") or {}).get("default"), False)

            delete_schema = await get_schema("storage_tools", "delete_data")
            delete_props = (delete_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((delete_props.get("missing_ok") or {}).get("default"), False)
            self.assertEqual((delete_props.get("item_ids") or {}).get("minItems"), 1)

            manage_schema = await get_schema("storage_tools", "manage_collections")
            manage_input_schema = manage_schema.get("input_schema") or {}
            manage_props = manage_input_schema.get("properties", {})
            self.assertEqual((manage_props.get("description") or {}).get("minLength"), 1)
            self.assertEqual(
                ((manage_props.get("metadata") or {}).get("propertyNames") or {}).get("minLength"),
                1,
            )
            report_enum = (manage_props.get("report_format") or {}).get("enum") or []
            self.assertIn("detailed", report_enum)
            self.assertIn("summary", report_enum)
            self.assertIn("analytics", report_enum)
            self.assertEqual((manage_props.get("include_capabilities") or {}).get("type"), "boolean")
            self.assertIn("backend_types", manage_props)
            self.assertIn("unavailable_backends", manage_props)
            self.assertIn("unavailable_reasons", manage_props)
            self.assertEqual(
                ((manage_props.get("unavailable_reasons") or {}).get("propertyNames") or {}).get("minLength"),
                1,
            )
            availability_filter_schema = manage_props.get("availability_filter") or {}
            self.assertIn("available", availability_filter_schema.get("enum") or [])
            self.assertIn("unavailable", availability_filter_schema.get("enum") or [])

            action_enum = (manage_props.get("action") or {}).get("enum") or []
            self.assertIn("backend_status", action_enum)
            self.assertIn("lifecycle_report", action_enum)
            all_of = manage_input_schema.get("allOf") or []
            self.assertGreaterEqual(len(all_of), 1)
            first_rule = all_of[0]
            self.assertIn("collection_name", ((first_rule.get("then") or {}).get("required") or []))
            conditional_collection_schema = (
                ((first_rule.get("then") or {}).get("properties") or {}).get("collection_name")
                or {}
            )
            self.assertEqual(conditional_collection_schema.get("type"), "string")
            self.assertEqual(conditional_collection_schema.get("minLength"), 1)

            invalid_tags = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "store_data",
                    {
                        "data": {"x": 1},
                        "tags": ["ok", ""],
                    },
                )
            )
            self.assertEqual(invalid_tags.get("status"), "error")
            self.assertIn("tags must be an array of non-empty strings", str(invalid_tags.get("error", "")))

            invalid_collection = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "store_data",
                    {
                        "data": {"x": 1},
                        "collection": "",
                    },
                )
            )
            self.assertEqual(invalid_collection.get("status"), "error")
            self.assertIn(
                "collection must be a non-empty string when provided",
                str(invalid_collection.get("error", "")),
            )

            invalid_item_ids = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "retrieve_data",
                    {
                        "item_ids": ["ok", ""],
                    },
                )
            )
            self.assertEqual(invalid_item_ids.get("status"), "error")
            self.assertIn(
                "item_ids must be an array of non-empty strings",
                str(invalid_item_ids.get("error", "")),
            )

            missing_item_ids = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "retrieve_data",
                    {
                        "item_ids": [],
                    },
                )
            )
            self.assertEqual(missing_item_ids.get("status"), "error")
            self.assertIn(
                "At least one item ID must be provided",
                str(missing_item_ids.get("error", "")),
            )

            invalid_format_type = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "retrieve_data",
                    {
                        "item_ids": ["item-1"],
                        "format_type": "",
                    },
                )
            )
            self.assertEqual(invalid_format_type.get("status"), "error")
            self.assertIn(
                "format_type must be a non-empty string",
                str(invalid_format_type.get("error", "")),
            )

            invalid_range = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "query_storage",
                    {
                        "size_range": [10, 1],
                    },
                )
            )
            self.assertEqual(invalid_range.get("status"), "error")
            self.assertIn("size_range must be non-negative and ordered", str(invalid_range.get("error", "")))

            invalid_list_limit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "list_storage",
                    {
                        "limit": 0,
                    },
                )
            )
            self.assertEqual(invalid_list_limit.get("status"), "error")
            self.assertIn("limit must be a positive integer", str(invalid_list_limit.get("error", "")))

            invalid_delete = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "delete_data",
                    {
                        "item_ids": ["ok", ""],
                    },
                )
            )
            self.assertEqual(invalid_delete.get("status"), "error")
            self.assertIn("item_ids must be an array of non-empty strings", str(invalid_delete.get("error", "")))

            invalid_date_range = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "query_storage",
                    {
                        "date_range": ["not-a-date", "2026-01-01T00:00:00Z"],
                    },
                )
            )
            self.assertEqual(invalid_date_range.get("status"), "error")
            self.assertIn(
                "date_range values must be valid ISO-8601 datetime strings",
                str(invalid_date_range.get("error", "")),
            )

            empty_date_range = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "query_storage",
                    {
                        "date_range": ["", "2026-01-01T00:00:00Z"],
                    },
                )
            )
            self.assertEqual(empty_date_range.get("status"), "error")
            self.assertIn(
                "date_range values must be non-empty strings",
                str(empty_date_range.get("error", "")),
            )

            invalid_report_format = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "stats",
                        "report_format": "xml",
                    },
                )
            )
            self.assertEqual(invalid_report_format.get("status"), "error")
            self.assertIn("report_format must be one of", str(invalid_report_format.get("error", "")))

            invalid_manage_description = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "create",
                        "collection_name": "bootstrap-manage-invalid-desc",
                        "description": "",
                    },
                )
            )
            self.assertEqual(invalid_manage_description.get("status"), "error")
            self.assertIn(
                "description must be a non-empty string when provided",
                str(invalid_manage_description.get("error", "")),
            )

            invalid_manage_metadata_key = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "create",
                        "collection_name": "bootstrap-manage-invalid-meta",
                        "metadata": {"": "invalid"},
                    },
                )
            )
            self.assertEqual(invalid_manage_metadata_key.get("status"), "error")
            self.assertIn(
                "metadata keys must be non-empty strings when provided",
                str(invalid_manage_metadata_key.get("error", "")),
            )

            invalid_stats_collection = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "stats",
                        "collection_name": "",
                    },
                )
            )
            self.assertEqual(invalid_stats_collection.get("status"), "error")
            self.assertIn(
                "collection_name must be a non-empty string when provided",
                str(invalid_stats_collection.get("error", "")),
            )

            invalid_availability_filter = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "backend_status",
                        "availability_filter": "partial",
                    },
                )
            )
            self.assertEqual(invalid_availability_filter.get("status"), "error")
            self.assertIn(
                "availability_filter must be one of",
                str(invalid_availability_filter.get("error", "")),
            )

            invalid_backend_alias_unavailable_reasons = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_backend_status",
                    {
                        "backend_types": ["memory"],
                        "unavailable_reasons": {"": "invalid"},
                    },
                )
            )
            self.assertEqual(invalid_backend_alias_unavailable_reasons.get("status"), "error")
            self.assertIn(
                "unavailable_reasons must be an object with non-empty string keys/values",
                str(invalid_backend_alias_unavailable_reasons.get("error", "")),
            )

            invalid_backend_alias_unknown_reason_backend = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_backend_status",
                    {
                        "unavailable_reasons": {"tape": "unsupported"},
                    },
                )
            )
            self.assertEqual(invalid_backend_alias_unknown_reason_backend.get("status"), "error")
            self.assertIn(
                "unavailable_reasons contains unknown storage backends",
                str(invalid_backend_alias_unknown_reason_backend.get("error", "")),
            )

            invalid_backend_alias_unknown_unavailable_backends = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_backend_status",
                    {
                        "unavailable_backends": ["tape"],
                    },
                )
            )
            self.assertEqual(invalid_backend_alias_unknown_unavailable_backends.get("status"), "error")
            self.assertIn(
                "unavailable_backends contains unknown storage backends",
                str(invalid_backend_alias_unknown_unavailable_backends.get("error", "")),
            )

            backend_status = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "backend_status",
                        "backend_types": ["memory", "ipfs"],
                        "unavailable_backends": ["ipfs"],
                        "unavailable_reasons": {"ipfs": "dial timeout"},
                        "availability_filter": "unavailable",
                        "include_breakdown": True,
                    },
                )
            )
            self.assertEqual(backend_status.get("status"), "success")
            backend_report = backend_status.get("backend_report") or {}
            self.assertEqual(backend_report.get("availability_filter"), "unavailable")
            backend_entries = backend_report.get("backends") or []
            self.assertEqual(len(backend_entries), 1)
            self.assertEqual((backend_entries[0] or {}).get("storage_type"), "ipfs")
            self.assertEqual((backend_entries[0] or {}).get("available"), False)
            self.assertEqual((backend_entries[0] or {}).get("unavailable_reason"), "dial timeout")
            self.assertEqual((backend_report.get("breakdown") or {}).get("unavailable_count"), 1)

            lifecycle_report = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "lifecycle_report",
                        "collection_name": "default",
                        "report_format": "analytics",
                        "include_breakdown": True,
                    },
                )
            )
            self.assertEqual(lifecycle_report.get("status"), "success")
            lifecycle_payload = lifecycle_report.get("lifecycle_report") or {}
            self.assertEqual(lifecycle_payload.get("scope"), "collection")
            self.assertEqual(lifecycle_payload.get("collection_name"), "default")
            self.assertIn("collections_total", lifecycle_payload)
            self.assertIn("totals", lifecycle_payload)

            lifecycle_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_lifecycle_report",
                    {
                        "collection_name": "default",
                        "report_format": "analytics",
                        "include_breakdown": True,
                    },
                )
            )
            self.assertEqual(lifecycle_alias.get("status"), "success")
            self.assertEqual(lifecycle_alias.get("scope"), "collection")
            self.assertEqual(lifecycle_alias.get("collection_name"), "default")
            self.assertIn("lifecycle_report", lifecycle_alias)

            backend_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_backend_status",
                    {
                        "backend_types": ["memory", "ipfs"],
                        "unavailable_backends": ["ipfs"],
                        "unavailable_reasons": {"ipfs": "dial timeout"},
                        "availability_filter": "unavailable",
                        "include_breakdown": True,
                    },
                )
            )
            self.assertEqual(backend_alias.get("status"), "success")
            self.assertEqual(backend_alias.get("availability_filter"), "unavailable")
            self.assertEqual(backend_alias.get("backend_count"), 1)

            collection_stats_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_collection_stats",
                    {
                        "collection_name": "default",
                        "report_format": "summary",
                        "include_breakdown": True,
                    },
                )
            )
            self.assertEqual(collection_stats_alias.get("status"), "success")
            self.assertEqual(collection_stats_alias.get("scope"), "collection")
            self.assertEqual(collection_stats_alias.get("collection_name"), "default")

            collections_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "list_storage_collections",
                    {
                        "include_metadata": False,
                        "include_timestamps": False,
                    },
                )
            )
            self.assertEqual(collections_alias.get("status"), "success")
            self.assertIn("collections", collections_alias)
            self.assertIn("total_count", collections_alias)

            create_collection_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "create_storage_collection",
                    {
                        "collection_name": "bootstrap-alias-temp",
                        "description": "bootstrap alias temp collection",
                    },
                )
            )
            self.assertEqual(create_collection_alias.get("status"), "success")
            self.assertEqual(create_collection_alias.get("collection_name"), "bootstrap-alias-temp")
            self.assertTrue(create_collection_alias.get("created"))

            get_collection_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_collection",
                    {
                        "collection_name": "bootstrap-alias-temp",
                        "include_metadata": True,
                        "include_timestamps": False,
                    },
                )
            )
            self.assertEqual(get_collection_alias.get("status"), "success")
            self.assertEqual(get_collection_alias.get("collection_name"), "bootstrap-alias-temp")
            self.assertIn("collection", get_collection_alias)
            self.assertNotIn("created_at", (get_collection_alias.get("collection") or {}))

            invalid_create_description = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "create_storage_collection",
                    {
                        "collection_name": "bootstrap-invalid-desc",
                        "description": "",
                    },
                )
            )
            self.assertEqual(invalid_create_description.get("status"), "error")
            self.assertIn(
                "description must be a non-empty string when provided",
                str(invalid_create_description.get("error", "")),
            )

            invalid_create_metadata_key = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "create_storage_collection",
                    {
                        "collection_name": "bootstrap-invalid-meta",
                        "metadata": {"": "invalid"},
                    },
                )
            )
            self.assertEqual(invalid_create_metadata_key.get("status"), "error")
            self.assertIn(
                "metadata keys must be non-empty strings when provided",
                str(invalid_create_metadata_key.get("error", "")),
            )

            invalid_get_include = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_collection",
                    {
                        "collection_name": "bootstrap-alias-temp",
                        "include_metadata": "yes",
                    },
                )
            )
            self.assertEqual(invalid_get_include.get("status"), "error")
            self.assertIn(
                "include_metadata must be a boolean",
                str(invalid_get_include.get("error", "")),
            )

            missing_collection_get = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_collection",
                    {
                        "collection_name": "bootstrap-missing-collection",
                    },
                )
            )
            self.assertEqual(missing_collection_get.get("status"), "error")
            self.assertEqual(missing_collection_get.get("collection_name"), "bootstrap-missing-collection")
            self.assertEqual(missing_collection_get.get("found"), False)
            self.assertIn("not found", str(missing_collection_get.get("error", "")).lower())

            delete_collection_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "delete_storage_collection",
                    {
                        "collection_name": "bootstrap-alias-temp",
                        "delete_items": True,
                    },
                )
            )
            self.assertEqual(delete_collection_alias.get("status"), "success")
            self.assertEqual(delete_collection_alias.get("collection_name"), "bootstrap-alias-temp")
            self.assertTrue(delete_collection_alias.get("deleted"))

            create_temp_collection = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "create",
                        "collection_name": "bootstrap-delete-temp",
                    },
                )
            )
            self.assertEqual(create_temp_collection.get("status"), "success")

            delete_collection_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "delete_storage_collection",
                    {
                        "collection_name": "bootstrap-delete-temp",
                        "delete_items": True,
                    },
                )
            )
            self.assertEqual(delete_collection_alias.get("status"), "success")
            self.assertEqual(delete_collection_alias.get("collection_name"), "bootstrap-delete-temp")
            self.assertTrue(delete_collection_alias.get("deleted"))

            stored = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "store_data",
                    {
                        "data": {"x": 1},
                        "collection": "default",
                    },
                )
            )
            self.assertEqual(stored.get("status"), "success")
            stored_item_id = stored.get("item_id")

            listed_storage = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "list_storage",
                    {
                        "limit": 5,
                    },
                )
            )
            self.assertEqual(listed_storage.get("status"), "success")
            self.assertIn("objects", listed_storage)

            storage_stats = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_stats",
                    {
                        "report_format": "summary",
                    },
                )
            )
            self.assertEqual(storage_stats.get("status"), "success")
            self.assertIn("total_objects", storage_stats)
            self.assertIn("total_bytes", storage_stats)

            invalid_stats_alias_collection = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "get_storage_stats",
                    {
                        "collection_name": "",
                    },
                )
            )
            self.assertEqual(invalid_stats_alias_collection.get("status"), "error")
            self.assertIn(
                "collection_name must be a non-empty string when provided",
                str(invalid_stats_alias_collection.get("error", "")),
            )

            deleted = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "delete_data",
                    {
                        "item_ids": [stored_item_id],
                    },
                )
            )
            self.assertEqual(deleted.get("status"), "success")
            self.assertEqual(deleted.get("deleted_count"), 1)

            missing_collection = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "create",
                    },
                )
            )
            self.assertEqual(missing_collection.get("status"), "error")
            self.assertIn(
                "collection_name required for create action",
                str(missing_collection.get("error", "")),
            )

            null_collection = self._assert_dispatch_success_envelope(
                await dispatch(
                    "storage_tools",
                    "manage_collections",
                    {
                        "action": "create",
                        "collection_name": None,
                    },
                )
            )
            self.assertEqual(null_collection.get("status"), "error")
            self.assertIn(
                "collection_name required for create action",
                str(null_collection.get("error", "")),
            )

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_dataset_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """dataset_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dataset-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("dataset_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("load_dataset", names)
            self.assertIn("text_to_fol", names)
            self.assertIn("dataset_tools_claudes", names)

            load_schema = await get_schema("dataset_tools", "load_dataset")
            load_props = (load_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((load_props.get("source") or {}).get("minLength"), 1)

            fol_schema = await get_schema("dataset_tools", "text_to_fol")
            fol_props = (fol_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((fol_props.get("confidence_threshold") or {}).get("minimum"), 0)
            self.assertEqual((fol_props.get("confidence_threshold") or {}).get("maximum"), 1)

            invalid_source = self._assert_dispatch_success_envelope(
                await dispatch(
                    "dataset_tools",
                    "load_dataset",
                    {
                        "source": "   ",
                    },
                )
            )
            self.assertEqual(invalid_source.get("status"), "error")
            self.assertIn("source must be a non-empty string", str(invalid_source.get("error", "")))

            invalid_threshold = self._assert_dispatch_success_envelope(
                await dispatch(
                    "dataset_tools",
                    "text_to_fol",
                    {
                        "text_input": "All humans are mortal",
                        "confidence_threshold": 1.5,
                    },
                )
            )
            self.assertEqual(invalid_threshold.get("status"), "error")
            self.assertIn("confidence_threshold must be between 0 and 1", str(invalid_threshold.get("error", "")))

            claudes_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "dataset_tools",
                    "dataset_tools_claudes",
                    {},
                )
            )
            self.assertIn(claudes_result.get("status"), ["success", "error"])
            self.assertIn("available_methods", claudes_result)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_embedding_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """embedding_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="embedding-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("embedding_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("generate_embedding", names)
            self.assertIn("generate_embeddings_from_file", names)
            self.assertIn("semantic_search", names)
            self.assertIn("hybrid_search", names)
            self.assertIn("search_with_filters", names)
            self.assertIn("multi_modal_search", names)
            self.assertIn("generate_embeddings", names)
            self.assertIn("chunk_text_for_embeddings", names)

            single_schema = await get_schema("embedding_tools", "generate_embedding")
            single_props = (single_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((single_props.get("batch_size") or {}).get("minimum"), 1)

            file_schema = await get_schema("embedding_tools", "generate_embeddings_from_file")
            file_props = (file_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("json", (file_props.get("output_format") or {}).get("enum", []))

            semantic_schema = await get_schema("embedding_tools", "semantic_search")
            semantic_props = (semantic_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((semantic_props.get("top_k") or {}).get("maximum"), 1000)

            hybrid_schema = await get_schema("embedding_tools", "hybrid_search")
            hybrid_props = (hybrid_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((hybrid_props.get("top_k") or {}).get("minimum"), 1)

            filter_schema = await get_schema("embedding_tools", "search_with_filters")
            filter_props = (filter_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("semantic", (filter_props.get("search_method") or {}).get("enum", []))

            multimodal_schema = await get_schema("embedding_tools", "multi_modal_search")
            multimodal_props = (multimodal_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((multimodal_props.get("top_k") or {}).get("maximum"), 1000)

            generate_schema = await get_schema("embedding_tools", "generate_embeddings")
            generate_props = (generate_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((generate_props.get("texts") or {}).get("minItems"), 1)

            shard_schema = await get_schema("embedding_tools", "shard_embeddings")
            shard_props = (shard_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((shard_props.get("shard_count") or {}).get("minimum"), 1)

            invalid_texts = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "generate_embeddings",
                    {
                        "texts": ["hello", ""],
                    },
                )
            )
            self.assertEqual(invalid_texts.get("status"), "error")
            self.assertIn("non-empty strings", str(invalid_texts.get("error", "")))

            invalid_single_batch = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "generate_embedding",
                    {
                        "text": "hello",
                        "batch_size": 0,
                    },
                )
            )
            self.assertEqual(invalid_single_batch.get("status"), "error")
            self.assertIn("batch_size must be a positive integer", str(invalid_single_batch.get("error", "")))

            invalid_file_format = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "generate_embeddings_from_file",
                    {
                        "file_path": "input.txt",
                        "output_format": "csv",
                    },
                )
            )
            self.assertEqual(invalid_file_format.get("status"), "error")
            self.assertIn("output_format must be one of", str(invalid_file_format.get("error", "")))

            invalid_semantic_top_k = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "semantic_search",
                    {
                        "query": "hello",
                        "vector_store_id": "vs-1",
                        "top_k": 0,
                    },
                )
            )
            self.assertEqual(invalid_semantic_top_k.get("status"), "error")
            self.assertIn("top_k must be between 1 and 1000", str(invalid_semantic_top_k.get("error", "")))

            invalid_hybrid_top_k = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "hybrid_search",
                    {
                        "query": "hello",
                        "vector_store_id": "vs-1",
                        "top_k": 0,
                    },
                )
            )
            self.assertEqual(invalid_hybrid_top_k.get("status"), "error")
            self.assertIn("top_k must be >= 1", str(invalid_hybrid_top_k.get("error", "")))

            invalid_filter_method = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "search_with_filters",
                    {
                        "query": "hello",
                        "vector_store_id": "vs-1",
                        "filters": {"category": "tech"},
                        "search_method": "vector",
                    },
                )
            )
            self.assertEqual(invalid_filter_method.get("status"), "error")
            self.assertIn("search_method must be one of", str(invalid_filter_method.get("error", "")))

            invalid_multimodal_inputs = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "multi_modal_search",
                    {
                        "vector_store_id": "vs-1",
                    },
                )
            )
            self.assertEqual(invalid_multimodal_inputs.get("status"), "error")
            self.assertIn(
                "either query or image_query must be provided",
                str(invalid_multimodal_inputs.get("error", "")),
            )

            invalid_overlap = self._assert_dispatch_success_envelope(
                await dispatch(
                    "embedding_tools",
                    "chunk_text_for_embeddings",
                    {
                        "text": "abc def ghi",
                        "chunk_size": 5,
                        "chunk_overlap": 5,
                    },
                )
            )
            self.assertEqual(invalid_overlap.get("status"), "error")
            self.assertIn("smaller than chunk_size", str(invalid_overlap.get("error", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_graph_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """graph_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="graph-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("graph_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("graph_add_entity", names)
            self.assertIn("graph_query_cypher", names)
            self.assertIn("query_knowledge_graph", names)
            self.assertIn("graph_search_hybrid", names)
            self.assertIn("graph_transaction_begin", names)
            self.assertIn("graph_transaction_commit", names)
            self.assertIn("graph_transaction_rollback", names)
            self.assertIn("graph_index_create", names)
            self.assertIn("graph_constraint_add", names)
            self.assertIn("graph_visualize", names)
            self.assertIn("graph_explain", names)

            add_entity_schema = await get_schema("graph_tools", "graph_add_entity")
            add_entity_props = (add_entity_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((add_entity_props.get("entity_id") or {}).get("minLength"), 1)

            query_schema = await get_schema("graph_tools", "graph_query_cypher")
            query_props = (query_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((query_props.get("query") or {}).get("minLength"), 1)

            search_schema = await get_schema("graph_tools", "graph_search_hybrid")
            search_props = (search_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                (search_props.get("search_type") or {}).get("enum"),
                ["hybrid", "keyword", "semantic"],
            )

            tx_commit_schema = await get_schema("graph_tools", "graph_transaction_commit")
            tx_commit_props = (tx_commit_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((tx_commit_props.get("transaction_id") or {}).get("minLength"), 1)

            index_schema = await get_schema("graph_tools", "graph_index_create")
            index_props = (index_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((index_props.get("properties") or {}).get("minItems"), 1)

            constraint_schema = await get_schema("graph_tools", "graph_constraint_add")
            constraint_props = (constraint_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("unique", (constraint_props.get("constraint_type") or {}).get("enum", []))

            visualize_schema = await get_schema("graph_tools", "graph_visualize")
            visualize_props = (visualize_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                (visualize_props.get("format") or {}).get("enum"),
                ["ascii", "d3_json", "dot", "mermaid"],
            )

            invalid_properties = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_add_entity",
                    {
                        "entity_id": "alice",
                        "entity_type": "Person",
                        "properties": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_properties.get("status"), "error")
            self.assertIn("properties must be an object", str(invalid_properties.get("error", "")))

            invalid_parameters = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_query_cypher",
                    {
                        "query": "MATCH (n) RETURN n",
                        "parameters": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_parameters.get("status"), "error")
            self.assertIn("parameters must be an object", str(invalid_parameters.get("error", "")))

            invalid_search_type = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_search_hybrid",
                    {
                        "query": "find people",
                        "search_type": "vector",
                    },
                )
            )
            self.assertEqual(invalid_search_type.get("status"), "error")
            self.assertIn("search_type must be one of", str(invalid_search_type.get("error", "")))

            invalid_tx = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_transaction_commit",
                    {"transaction_id": "   "},
                )
            )
            self.assertEqual(invalid_tx.get("status"), "error")
            self.assertIn("transaction_id must be a non-empty string", str(invalid_tx.get("error", "")))

            invalid_constraint = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_constraint_add",
                    {
                        "constraint_name": "person-email",
                        "constraint_type": "primary",
                        "entity_type": "Person",
                        "properties": ["email"],
                    },
                )
            )
            self.assertEqual(invalid_constraint.get("status"), "error")
            self.assertIn("constraint_type must be one of", str(invalid_constraint.get("error", "")))

            invalid_explain = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_explain",
                    {
                        "explain_type": "relationship",
                    },
                )
            )
            self.assertEqual(invalid_explain.get("status"), "error")
            self.assertIn("relationship_id is required", str(invalid_explain.get("error", "")))

            begun = self._assert_dispatch_success_envelope(
                await dispatch("graph_tools", "graph_transaction_begin", {})
            )
            self.assertEqual(begun.get("status"), "success")
            transaction_id = begun.get("transaction_id")
            self.assertIsInstance(transaction_id, str)

            committed = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_transaction_commit",
                    {"transaction_id": transaction_id},
                )
            )
            self.assertEqual(committed.get("status"), "success")

            created_index = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_index_create",
                    {
                        "index_name": "people-name",
                        "entity_type": "Person",
                        "properties": ["name"],
                    },
                )
            )
            self.assertEqual(created_index.get("status"), "success")
            self.assertEqual(created_index.get("index_name"), "people-name")

            created_constraint = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "graph_constraint_add",
                    {
                        "constraint_name": "person-email",
                        "constraint_type": "unique",
                        "entity_type": "Person",
                        "properties": ["email"],
                    },
                )
            )
            self.assertEqual(created_constraint.get("status"), "success")
            self.assertEqual(created_constraint.get("constraint_name"), "person-email")

            valid_query_kg = self._assert_dispatch_success_envelope(
                await dispatch(
                    "graph_tools",
                    "query_knowledge_graph",
                    {
                        "query": "find regulations",
                    },
                )
            )
            self.assertIn(valid_query_kg.get("status"), ["success", "error"])
            self.assertEqual(valid_query_kg.get("query"), "find regulations")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_data_processing_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """data_processing_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="data-processing-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("data_processing_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("chunk_text", names)
            self.assertIn("transform_data", names)
            self.assertIn("convert_format", names)
            self.assertIn("validate_data", names)

            chunk_schema = await get_schema("data_processing_tools", "chunk_text")
            chunk_props = (chunk_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                (chunk_props.get("strategy") or {}).get("enum"),
                ["fixed_size", "sentence", "paragraph", "semantic"],
            )

            transform_schema = await get_schema("data_processing_tools", "transform_data")
            transform_props = (transform_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("parameters", transform_props)

            invalid_strategy = self._assert_dispatch_success_envelope(
                await dispatch(
                    "data_processing_tools",
                    "chunk_text",
                    {
                        "text": "alpha beta gamma",
                        "strategy": "windowed",
                    },
                )
            )
            self.assertEqual(invalid_strategy.get("status"), "error")
            self.assertIn("strategy must be one of", str(invalid_strategy.get("message", "")))

            invalid_parameters = self._assert_dispatch_success_envelope(
                await dispatch(
                    "data_processing_tools",
                    "transform_data",
                    {
                        "data": {"x": 1},
                        "transformation": "normalize",
                        "parameters": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_parameters.get("status"), "error")
            self.assertIn("parameters must be an object", str(invalid_parameters.get("message", "")))

            invalid_rules = self._assert_dispatch_success_envelope(
                await dispatch(
                    "data_processing_tools",
                    "validate_data",
                    {
                        "data": {"x": 1},
                        "validation_type": "schema",
                        "rules": [{"rule": "required"}, "bad"],
                    },
                )
            )
            self.assertEqual(invalid_rules.get("status"), "error")
            self.assertIn("rules entries must be objects", str(invalid_rules.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_pdf_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """pdf_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="pdf-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("pdf_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("pdf_analyze_relationships", names)
            self.assertIn("pdf_cross_document_analysis", names)
            self.assertIn("pdf_optimize_for_llm", names)
            self.assertIn("pdf_query_corpus", names)
            self.assertIn("pdf_batch_process", names)
            self.assertIn("pdf_query_knowledge_graph", names)

            relationship_schema = await get_schema("pdf_tools", "pdf_analyze_relationships")
            relationship_props = (relationship_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((relationship_props.get("document_id") or {}).get("minLength"), 1)
            self.assertEqual((relationship_props.get("min_confidence") or {}).get("maximum"), 1.0)

            cross_schema = await get_schema("pdf_tools", "pdf_cross_document_analysis")
            cross_props = (cross_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((cross_props.get("document_ids") or {}).get("minItems"), 1)
            self.assertEqual((cross_props.get("similarity_threshold") or {}).get("maximum"), 1.0)

            optimize_schema = await get_schema("pdf_tools", "pdf_optimize_for_llm")
            optimize_props = (optimize_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((optimize_props.get("max_chunk_size") or {}).get("minimum"), 1)
            self.assertEqual((optimize_props.get("overlap_size") or {}).get("minimum"), 0)

            query_schema = await get_schema("pdf_tools", "pdf_query_corpus")
            query_props = (query_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((query_props.get("max_documents") or {}).get("minimum"), 1)

            batch_schema = await get_schema("pdf_tools", "pdf_batch_process")
            batch_props = (batch_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((batch_props.get("pdf_sources") or {}).get("minItems"), 1)

            graph_schema = await get_schema("pdf_tools", "pdf_query_knowledge_graph")
            graph_props = (graph_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("sparql", (graph_props.get("query_type") or {}).get("enum", []))
            self.assertEqual((graph_props.get("max_results") or {}).get("minimum"), 1)

            invalid_relationships = self._assert_dispatch_success_envelope(
                await dispatch(
                    "pdf_tools",
                    "pdf_analyze_relationships",
                    {
                        "document_id": "doc-1",
                        "relationship_types": ["SIGNED_BY", ""],
                    },
                )
            )
            self.assertEqual(invalid_relationships.get("status"), "error")
            self.assertIn("relationship_types must be a list of non-empty strings", str(invalid_relationships.get("error", "")))

            invalid_cross = self._assert_dispatch_success_envelope(
                await dispatch(
                    "pdf_tools",
                    "pdf_cross_document_analysis",
                    {
                        "document_ids": ["doc-1", "doc-2"],
                        "analysis_types": ["entities", ""],
                    },
                )
            )
            self.assertEqual(invalid_cross.get("status"), "error")
            self.assertIn("analysis_types must be a list of non-empty strings", str(invalid_cross.get("error", "")))

            invalid_query = self._assert_dispatch_success_envelope(
                await dispatch(
                    "pdf_tools",
                    "pdf_query_corpus",
                    {
                        "query": "",
                    },
                )
            )
            self.assertEqual(invalid_query.get("status"), "error")
            self.assertIn("query must be a non-empty string", str(invalid_query.get("error", "")))

            invalid_batch = self._assert_dispatch_success_envelope(
                await dispatch(
                    "pdf_tools",
                    "pdf_batch_process",
                    {
                        "pdf_sources": [""],
                    },
                )
            )
            self.assertEqual(invalid_batch.get("status"), "error")
            self.assertIn("pdf_sources entries must be non-empty strings or objects", str(invalid_batch.get("error", "")))

            invalid_optimize = self._assert_dispatch_success_envelope(
                await dispatch(
                    "pdf_tools",
                    "pdf_optimize_for_llm",
                    {
                        "pdf_source": "file.pdf",
                        "max_chunk_size": 100,
                        "overlap_size": 200,
                    },
                )
            )
            self.assertEqual(invalid_optimize.get("status"), "error")
            self.assertIn("overlap_size must be less than or equal to max_chunk_size", str(invalid_optimize.get("error", "")))

            invalid_graph_query = self._assert_dispatch_success_envelope(
                await dispatch(
                    "pdf_tools",
                    "pdf_query_knowledge_graph",
                    {
                        "graph_id": "graph-1",
                        "query": "MATCH (n) RETURN n",
                        "query_type": "sql",
                    },
                )
            )
            self.assertEqual(invalid_graph_query.get("status"), "error")
            self.assertIn("query_type must be one of", str(invalid_graph_query.get("error", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_logic_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """logic_tools should expose deterministic schema and validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="logic-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("logic_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("tdfol_parse", names)
            self.assertIn("tdfol_prove", names)
            self.assertIn("tdfol_kb_add_axiom", names)
            self.assertIn("tdfol_kb_query", names)
            self.assertIn("tdfol_kb_export", names)
            self.assertIn("cec_prove", names)
            self.assertIn("cec_parse", names)
            self.assertIn("cec_validate_formula", names)

            parse_schema = await get_schema("logic_tools", "tdfol_parse")
            parse_props = (parse_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((parse_props.get("format") or {}).get("minLength"), 1)

            prove_schema = await get_schema("logic_tools", "tdfol_prove")
            prove_props = (prove_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((prove_props.get("timeout_ms") or {}).get("minimum"), 1)

            kb_export_schema = await get_schema("logic_tools", "tdfol_kb_export")
            kb_export_props = (kb_export_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("json", (kb_export_props.get("export_format") or {}).get("enum", []))

            cec_prove_schema = await get_schema("logic_tools", "cec_prove")
            cec_prove_props = (cec_prove_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((cec_prove_props.get("timeout") or {}).get("minimum"), 1)

            cec_parse_schema = await get_schema("logic_tools", "cec_parse")
            cec_parse_props = (cec_parse_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((cec_parse_props.get("language") or {}).get("minLength"), 1)

            invalid_parse = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "tdfol_parse",
                    {
                        "text": "forall x P(x)",
                        "format": "",
                    },
                )
            )
            self.assertEqual(invalid_parse.get("success"), False)
            self.assertIn("'format' must be a non-empty string", str(invalid_parse.get("error", "")))

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "tdfol_prove",
                    {
                        "formula": "forall x P(x)",
                        "timeout_ms": 0,
                    },
                )
            )
            self.assertEqual(invalid_timeout.get("success"), False)
            self.assertIn("'timeout_ms' must be an integer greater than or equal to 1", str(invalid_timeout.get("error", "")))

            invalid_kb_export = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "tdfol_kb_export",
                    {
                        "export_format": "xml",
                    },
                )
            )
            self.assertEqual(invalid_kb_export.get("success"), False)
            self.assertIn("'export_format' must be one of", str(invalid_kb_export.get("error", "")))

            invalid_cec_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "cec_prove",
                    {
                        "goal": "P(a)",
                        "timeout": 0,
                    },
                )
            )
            self.assertEqual(invalid_cec_timeout.get("success"), False)
            self.assertIn("'timeout' must be an integer greater than or equal to 1", str(invalid_cec_timeout.get("error", "")))

            invalid_cec_parse = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "cec_parse",
                    {
                        "text": "agent knows p",
                        "language": "",
                    },
                )
            )
            self.assertEqual(invalid_cec_parse.get("success"), False)
            self.assertIn("'language' must be a non-empty string", str(invalid_cec_parse.get("error", "")))

            added_axiom = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "tdfol_kb_add_axiom",
                    {
                        "formula": "forall x P(x)",
                    },
                )
            )
            self.assertEqual(added_axiom.get("success"), True)

            queried_kb = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "tdfol_kb_query",
                    {},
                )
            )
            self.assertEqual(queried_kb.get("success"), True)
            self.assertIn("stats", queried_kb)

            exported_kb = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "tdfol_kb_export",
                    {
                        "export_format": "json",
                    },
                )
            )
            self.assertEqual(exported_kb.get("success"), True)
            self.assertEqual(exported_kb.get("format"), "json")

            cec_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "cec_prove",
                    {
                        "goal": "K(agent,P)",
                        "timeout": 30,
                    },
                )
            )
            self.assertIn(cec_result.get("success"), [True, False])

            cec_parse_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "logic_tools",
                    "cec_parse",
                    {
                        "text": "agent knows p",
                        "language": "en",
                    },
                )
            )
            self.assertIn(cec_parse_result.get("success"), [True, False])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_session_tools_enhanced_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """session_tools should expose enhanced wrappers with deterministic dispatch contracts."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="session-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("session_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("manage_session", names)
            self.assertIn("get_session_state", names)

            create_schema = await get_schema("session_tools", "create_session")
            create_props = (create_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("session_type", create_props)
            self.assertIn("metadata", create_props)
            self.assertIn("tags", create_props)

            manage_schema = await get_schema("session_tools", "manage_session")
            self.assertEqual(manage_schema.get("name"), "manage_session")
            self.assertEqual(manage_schema.get("category"), "session_tools")
            manage_props = (manage_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("action", manage_props)
            cleanup_schema = manage_props.get("cleanup_options") or {}
            cleanup_props = cleanup_schema.get("properties") or {}
            self.assertEqual((cleanup_props.get("max_age_hours") or {}).get("minimum"), 1)
            self.assertEqual((cleanup_props.get("dry_run") or {}).get("type"), "boolean")

            state_schema = await get_schema("session_tools", "get_session_state")
            self.assertEqual(state_schema.get("name"), "get_session_state")
            self.assertEqual(state_schema.get("category"), "session_tools")
            self.assertIn("session_id", (state_schema.get("input_schema") or {}).get("properties", {}))

            created = self._assert_dispatch_success_envelope(
                await dispatch(
                    "session_tools",
                    "create_session",
                    {
                        "session_name": "session-tools-bootstrap",
                        "user_id": "bootstrap-user",
                    },
                )
            )
            self.assertEqual(created.get("status"), "success")
            session_id = created.get("session_id")
            self.assertIsInstance(session_id, str)

            invalid_tags = self._assert_dispatch_success_envelope(
                await dispatch(
                    "session_tools",
                    "create_session",
                    {
                        "session_name": "invalid-session-tags",
                        "tags": ["ok", "   "],
                    },
                )
            )
            self.assertEqual(invalid_tags.get("status"), "error")
            invalid_tags_text = (
                str(invalid_tags.get("message", ""))
                + " "
                + str(invalid_tags.get("error", ""))
            )
            self.assertIn("tags must be a list of non-empty strings", invalid_tags_text)

            managed_get = self._assert_dispatch_success_envelope(
                await dispatch(
                    "session_tools",
                    "manage_session",
                    {
                        "action": "get",
                        "session_id": session_id,
                    },
                )
            )
            self.assertEqual(managed_get.get("status"), "success")
            self.assertIn("session", managed_get)

            invalid_cleanup = self._assert_dispatch_success_envelope(
                await dispatch(
                    "session_tools",
                    "manage_session",
                    {
                        "action": "cleanup",
                        "cleanup_options": {"dry_run": "yes"},
                    },
                )
            )
            self.assertEqual(invalid_cleanup.get("status"), "error")
            self.assertEqual(invalid_cleanup.get("code"), "INVALID_CLEANUP_OPTIONS")
            self.assertIn(
                "cleanup_options.dry_run must be a boolean",
                str(invalid_cleanup.get("error", "")),
            )

            invalid_state = self._assert_dispatch_success_envelope(
                await dispatch(
                    "session_tools",
                    "get_session_state",
                    {
                        "session_id": "not-a-uuid",
                    },
                )
            )
            self.assertEqual(invalid_state.get("status"), "error")
            self.assertEqual(invalid_state.get("code"), "INVALID_SESSION_ID")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_auth_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """auth_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="auth-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("auth_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("authenticate_user", names)
            self.assertIn("validate_token", names)
            self.assertIn("get_user_info", names)

            validate_schema = await get_schema("auth_tools", "validate_token")
            self.assertEqual(validate_schema.get("name"), "validate_token")
            schema_props = (validate_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((schema_props.get("action") or {}).get("default"), "validate")
            self.assertIn("decode", (schema_props.get("action") or {}).get("enum", []))
            self.assertEqual((schema_props.get("strict") or {}).get("default"), False)

            auth_schema = await get_schema("auth_tools", "authenticate_user")
            auth_props = (auth_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((auth_props.get("remember_me") or {}).get("default"), False)

            user_info_schema = await get_schema("auth_tools", "get_user_info")
            user_info_props = (user_info_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((user_info_props.get("include_permissions") or {}).get("default"), True)

            invalid_action = self._assert_dispatch_success_envelope(
                await dispatch(
                    "auth_tools",
                    "validate_token",
                    {
                        "token": "dummy-token",
                        "action": "bad",
                    },
                )
            )
            self.assertEqual(invalid_action.get("status"), "error")
            self.assertEqual(invalid_action.get("valid"), False)

            decode_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "auth_tools",
                    "validate_token",
                    {
                        "token": "dummy-token",
                        "action": "decode",
                    },
                )
            )
            self.assertIn(decode_result.get("status"), ["success", "error"])
            self.assertIn("message", decode_result)

            auth_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "auth_tools",
                    "authenticate_user",
                    {
                        "username": "demo",
                        "password": "pw",
                        "remember_me": True,
                    },
                )
            )
            self.assertIn(auth_result.get("status"), ["success", "error"])
            self.assertEqual(auth_result.get("remember_me"), True)

            user_info_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "auth_tools",
                    "get_user_info",
                    {
                        "token": "dummy-token",
                        "include_permissions": False,
                        "include_profile": False,
                    },
                )
            )
            self.assertIn(user_info_result.get("status"), ["success", "error"])
            self.assertEqual(user_info_result.get("include_permissions"), False)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_analysis_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """analysis_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="analysis-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("analysis_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("analyze_data_distribution", names)
            self.assertIn("cluster_analysis", names)
            self.assertIn("quality_assessment", names)
            self.assertIn("detect_outliers", names)
            self.assertIn("dimensionality_reduction", names)

            cluster_schema = await get_schema("analysis_tools", "cluster_analysis")
            self.assertEqual(cluster_schema.get("name"), "cluster_analysis")
            schema_props = (cluster_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((schema_props.get("algorithm") or {}).get("default"), "kmeans")
            self.assertIn("spectral", (schema_props.get("algorithm") or {}).get("enum", []))

            reduction_schema = await get_schema("analysis_tools", "dimensionality_reduction")
            method_props = ((reduction_schema.get("input_schema") or {}).get("properties", {}).get("method") or {})
            self.assertEqual(method_props.get("default"), "pca")
            self.assertIn("truncated_svd", method_props.get("enum", []))

            outlier_schema = await get_schema("analysis_tools", "detect_outliers")
            outlier_props = (outlier_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((outlier_props.get("threshold") or {}).get("exclusiveMinimum"), 0)

            invalid_clusters = self._assert_dispatch_success_envelope(
                await dispatch(
                    "analysis_tools",
                    "cluster_analysis",
                    {
                        "algorithm": "kmeans",
                        "n_clusters": "bad",
                    },
                )
            )
            self.assertEqual(invalid_clusters.get("status"), "error")
            self.assertIn("n_clusters must be a positive integer", str(invalid_clusters.get("message", "")))

            source_algorithm = self._assert_dispatch_success_envelope(
                await dispatch(
                    "analysis_tools",
                    "cluster_analysis",
                    {
                        "algorithm": "spectral",
                    },
                )
            )
            self.assertIn(source_algorithm.get("status"), ["success", "error"])

            source_method = self._assert_dispatch_success_envelope(
                await dispatch(
                    "analysis_tools",
                    "dimensionality_reduction",
                    {
                        "method": "truncated_svd",
                        "n_components": 2,
                    },
                )
            )
            self.assertIn(source_method.get("status"), ["success", "error"])

            invalid_outliers = self._assert_dispatch_success_envelope(
                await dispatch(
                    "analysis_tools",
                    "detect_outliers",
                    {
                        "data": [[0.1, 0.2], [0.2, 0.3]],
                        "threshold": 0,
                    },
                )
            )
            self.assertEqual(invalid_outliers.get("status"), "error")
            self.assertIn("threshold must be a positive number", str(invalid_outliers.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_geospatial_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """geospatial_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="geospatial-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("geospatial_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("extract_geographic_entities", names)
            self.assertIn("map_spatiotemporal_events", names)
            self.assertIn("query_geographic_context", names)
            self.assertIn("analyze_geospatial_corpus", names)

            map_schema = await get_schema("geospatial_tools", "map_spatiotemporal_events")
            self.assertEqual(map_schema.get("name"), "map_spatiotemporal_events")
            schema_props = (map_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((schema_props.get("temporal_resolution") or {}).get("default"), "day")
            self.assertIn("year", (schema_props.get("temporal_resolution") or {}).get("enum", []))

            combined_schema = await get_schema("geospatial_tools", "analyze_geospatial_corpus")
            combined_props = (combined_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((combined_props.get("confidence_threshold") or {}).get("default"), 0.7)
            self.assertIn("month", (combined_props.get("temporal_resolution") or {}).get("enum", []))

            invalid_resolution = self._assert_dispatch_success_envelope(
                await dispatch(
                    "geospatial_tools",
                    "map_spatiotemporal_events",
                    {
                        "corpus_data": "sample",
                        "temporal_resolution": "quarter",
                    },
                )
            )
            self.assertEqual(invalid_resolution.get("status"), "error")
            self.assertIn("must be one of", str(invalid_resolution.get("message", "")))

            valid_query = self._assert_dispatch_success_envelope(
                await dispatch(
                    "geospatial_tools",
                    "query_geographic_context",
                    {
                        "query": "city",
                        "corpus_data": "sample",
                        "radius_km": 25,
                    },
                )
            )
            self.assertIn(valid_query.get("status"), ["success", "error"])

            invalid_combined = self._assert_dispatch_success_envelope(
                await dispatch(
                    "geospatial_tools",
                    "analyze_geospatial_corpus",
                    {
                        "corpus_data": "sample",
                        "temporal_resolution": "quarter",
                    },
                )
            )
            self.assertEqual(invalid_combined.get("status"), "error")
            self.assertEqual(invalid_combined.get("phase"), "map_spatiotemporal_events")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_index_management_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """index_management_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="index-management-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("index_management_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("load_index", names)
            self.assertIn("manage_shards", names)
            self.assertIn("monitor_index_status", names)
            self.assertIn("manage_index_configuration", names)
            self.assertIn("orchestrate_index_lifecycle", names)

            monitor_schema = await get_schema("index_management_tools", "monitor_index_status")
            self.assertEqual(monitor_schema.get("name"), "monitor_index_status")
            schema_props = (monitor_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((schema_props.get("time_range") or {}).get("default"), "24h")
            self.assertIn("30d", (schema_props.get("time_range") or {}).get("enum", []))

            lifecycle_schema = await get_schema("index_management_tools", "orchestrate_index_lifecycle")
            lifecycle_props = (lifecycle_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((lifecycle_props.get("dataset") or {}).get("minLength"), 1)
            self.assertIn("optimize", (lifecycle_props.get("action") or {}).get("enum", []))

            invalid_shards = self._assert_dispatch_success_envelope(
                await dispatch(
                    "index_management_tools",
                    "manage_shards",
                    {
                        "action": "create_shards",
                        "num_shards": "bad",
                    },
                )
            )
            self.assertEqual(invalid_shards.get("status"), "error")
            self.assertIn("positive integer", str(invalid_shards.get("message", "")))

            invalid_time_range = self._assert_dispatch_success_envelope(
                await dispatch(
                    "index_management_tools",
                    "monitor_index_status",
                    {
                        "time_range": "2h",
                    },
                )
            )
            self.assertEqual(invalid_time_range.get("status"), "error")
            self.assertIn("must be one of", str(invalid_time_range.get("message", "")))

            valid_manage = self._assert_dispatch_success_envelope(
                await dispatch(
                    "index_management_tools",
                    "manage_index_configuration",
                    {
                        "action": "get_config",
                        "optimization_level": 1,
                    },
                )
            )
            self.assertIn(valid_manage.get("status"), ["success", "error"])

            invalid_lifecycle = self._assert_dispatch_success_envelope(
                await dispatch(
                    "index_management_tools",
                    "orchestrate_index_lifecycle",
                    {
                        "dataset": "",
                    },
                )
            )
            self.assertEqual(invalid_lifecycle.get("status"), "error")
            self.assertIn("dataset is required", str(invalid_lifecycle.get("message", "")))

            valid_lifecycle = self._assert_dispatch_success_envelope(
                await dispatch(
                    "index_management_tools",
                    "orchestrate_index_lifecycle",
                    {
                        "dataset": "bootstrap-dataset",
                        "action": "create",
                    },
                )
            )
            self.assertIn(valid_lifecycle.get("status"), ["success", "error"])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_provenance_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """provenance_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="provenance-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("provenance_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("record_provenance", names)
            self.assertIn("record_provenance_batch", names)
            self.assertIn("verify_provenance_records", names)
            self.assertIn("generate_provenance_report", names)

            schema = await get_schema("provenance_tools", "record_provenance")
            self.assertEqual(schema.get("name"), "record_provenance")
            schema_props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((schema_props.get("timestamp") or {}).get("format"), "date-time")

            batch_schema = await get_schema("provenance_tools", "record_provenance_batch")
            self.assertEqual(batch_schema.get("name"), "record_provenance_batch")
            batch_props = (batch_schema.get("input_schema") or {}).get("properties", {})
            records_schema = batch_props.get("records") or {}
            self.assertEqual(records_schema.get("type"), "array")
            item_props = (records_schema.get("items") or {}).get("properties", {})
            self.assertEqual((item_props.get("timestamp") or {}).get("format"), "date-time")

            verify_schema = await get_schema("provenance_tools", "verify_provenance_records")
            self.assertEqual(verify_schema.get("name"), "verify_provenance_records")
            verify_props = (verify_schema.get("input_schema") or {}).get("properties", {})
            verify_records_schema = verify_props.get("records") or {}
            self.assertEqual(verify_records_schema.get("type"), "array")
            self.assertEqual(verify_records_schema.get("minItems"), 1)

            report_schema = await get_schema("provenance_tools", "generate_provenance_report")
            self.assertEqual(report_schema.get("name"), "generate_provenance_report")
            report_props = (report_schema.get("input_schema") or {}).get("properties", {})
            report_records_schema = report_props.get("records") or {}
            self.assertEqual(report_records_schema.get("type"), "array")
            self.assertEqual(report_records_schema.get("minItems"), 1)

            invalid_timestamp = self._assert_dispatch_success_envelope(
                await dispatch(
                    "provenance_tools",
                    "record_provenance",
                    {
                        "dataset_id": "dataset-1",
                        "operation": "transform",
                        "timestamp": "not-a-timestamp",
                    },
                )
            )
            self.assertEqual(invalid_timestamp.get("status"), "error")
            self.assertIn("valid ISO-8601", str(invalid_timestamp.get("message", "")))

            valid_record = self._assert_dispatch_success_envelope(
                await dispatch(
                    "provenance_tools",
                    "record_provenance",
                    {
                        "dataset_id": "dataset-1",
                        "operation": "transform",
                        "timestamp": "2026-03-03T12:00:00Z",
                    },
                )
            )
            self.assertIn(valid_record.get("status"), ["success", "error"])

            invalid_batch = self._assert_dispatch_success_envelope(
                await dispatch(
                    "provenance_tools",
                    "record_provenance_batch",
                    {"records": []},
                )
            )
            self.assertEqual(invalid_batch.get("status"), "error")
            self.assertIn("non-empty array", str(invalid_batch.get("message", "")))

            valid_batch = self._assert_dispatch_success_envelope(
                await dispatch(
                    "provenance_tools",
                    "record_provenance_batch",
                    {
                        "records": [
                            {
                                "dataset_id": "dataset-1",
                                "operation": "transform",
                                "timestamp": "2026-03-03T12:00:00Z",
                            },
                            {
                                "dataset_id": "dataset-2",
                                "operation": "publish",
                            },
                        ]
                    },
                )
            )
            self.assertEqual(valid_batch.get("status"), "success")
            self.assertEqual(valid_batch.get("processed"), 2)

            invalid_verify = self._assert_dispatch_success_envelope(
                await dispatch(
                    "provenance_tools",
                    "verify_provenance_records",
                    {"records": []},
                )
            )
            self.assertEqual(invalid_verify.get("status"), "error")
            self.assertIn("non-empty array", str(invalid_verify.get("message", "")))

            valid_verify = self._assert_dispatch_success_envelope(
                await dispatch(
                    "provenance_tools",
                    "verify_provenance_records",
                    {
                        "records": [
                            {
                                "status": "success",
                                "dataset_id": "dataset-1",
                                "operation": "transform",
                            },
                            {
                                "status": "error",
                                "dataset_id": "dataset-2",
                                "operation": "publish",
                                "message": "failed",
                            },
                        ]
                    },
                )
            )
            self.assertEqual(valid_verify.get("status"), "success")
            self.assertEqual(valid_verify.get("verified_count"), 1)
            self.assertEqual(valid_verify.get("failed_count"), 1)

            valid_report = self._assert_dispatch_success_envelope(
                await dispatch(
                    "provenance_tools",
                    "generate_provenance_report",
                    {
                        "records": [
                            {
                                "status": "success",
                                "dataset_id": "dataset-1",
                                "operation": "transform",
                            },
                            {
                                "status": "error",
                                "dataset_id": "dataset-2",
                                "operation": "transform",
                                "message": "validation failed",
                            },
                        ]
                    },
                )
            )
            self.assertEqual(valid_report.get("status"), "success")
            report_payload = valid_report.get("report") or {}
            self.assertEqual(report_payload.get("success_count"), 1)
            self.assertEqual(report_payload.get("error_count"), 1)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_admin_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """admin_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="admin-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("admin_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("manage_endpoints", names)
            self.assertIn("system_maintenance", names)
            self.assertIn("configure_system", names)
            self.assertIn("system_health", names)
            self.assertIn("get_system_status", names)
            self.assertIn("manage_service", names)
            self.assertIn("update_configuration", names)
            self.assertIn("cleanup_resources", names)

            endpoint_schema = await get_schema("admin_tools", "manage_endpoints")
            props = (endpoint_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("list", (props.get("action") or {}).get("enum", []))
            self.assertEqual((props.get("ctx_length") or {}).get("minimum"), 1)

            health_schema = await get_schema("admin_tools", "system_health")
            health_props = (health_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((health_props.get("component") or {}).get("default"), "all")
            self.assertEqual((health_props.get("detailed") or {}).get("default"), False)

            status_schema = await get_schema("admin_tools", "get_system_status")
            status_props = (status_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((status_props.get("format") or {}).get("default"), "json")

            service_schema = await get_schema("admin_tools", "manage_service")
            service_props = (service_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((service_props.get("timeout_seconds") or {}).get("minimum"), 1)

            update_schema = await get_schema("admin_tools", "update_configuration")
            update_props = (update_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((update_props.get("create_backup") or {}).get("default"), True)

            cleanup_schema = await get_schema("admin_tools", "cleanup_resources")
            cleanup_props = (cleanup_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((cleanup_props.get("max_log_age_days") or {}).get("minimum"), 1)

            invalid_action = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "manage_endpoints",
                    {
                        "action": "bad",
                    },
                )
            )
            self.assertEqual(invalid_action.get("status"), "error")
            self.assertIn("must be one of", str(invalid_action.get("message", "")))

            status_alias = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "system_maintenance",
                    {
                        "operation": "status",
                    },
                )
            )
            self.assertIn(status_alias.get("status"), ["success", "error"])

            invalid_settings = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "configure_system",
                    {
                        "action": "update",
                        "settings": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_settings.get("status"), "error")
            self.assertIn("must be an object", str(invalid_settings.get("message", "")))

            invalid_health_component = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "system_health",
                    {
                        "component": "   ",
                    },
                )
            )
            self.assertEqual(invalid_health_component.get("status"), "error")
            self.assertIn("non-empty string", str(invalid_health_component.get("message", "")))

            valid_health = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "system_health",
                    {
                        "component": "all",
                        "detailed": True,
                    },
                )
            )
            self.assertIn(valid_health.get("status"), ["success", "error"])

            invalid_status_format = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "get_system_status",
                    {
                        "format": "xml",
                    },
                )
            )
            self.assertEqual(invalid_status_format.get("status"), "error")
            self.assertIn("must be one of", str(invalid_status_format.get("message", "")))

            valid_status = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "get_system_status",
                    {
                        "format": "detailed",
                        "include_services": True,
                    },
                )
            )
            self.assertIn(valid_status.get("status"), ["success", "operational"])

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "manage_service",
                    {
                        "service_name": "cache_service",
                        "action": "restart",
                        "timeout_seconds": 0,
                    },
                )
            )
            self.assertEqual(invalid_timeout.get("status"), "error")
            self.assertIn("positive integer", str(invalid_timeout.get("message", "")))

            valid_service = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "manage_service",
                    {
                        "service_name": "all",
                        "action": "status",
                    },
                )
            )
            self.assertEqual(valid_service.get("status"), "success")

            invalid_config_updates = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "update_configuration",
                    {
                        "action": "update",
                        "config_updates": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_config_updates.get("status"), "error")
            self.assertIn("must be an object", str(invalid_config_updates.get("message", "")))

            valid_update = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "update_configuration",
                    {
                        "action": "validate",
                        "config_updates": {"cache.timeout": 5},
                    },
                )
            )
            self.assertEqual(valid_update.get("status"), "success")

            invalid_cleanup = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "cleanup_resources",
                    {
                        "cleanup_type": "purge",
                    },
                )
            )
            self.assertEqual(invalid_cleanup.get("status"), "error")
            self.assertIn("must be one of", str(invalid_cleanup.get("message", "")))

            valid_cleanup = self._assert_dispatch_success_envelope(
                await dispatch(
                    "admin_tools",
                    "cleanup_resources",
                    {
                        "cleanup_type": "basic",
                    },
                )
            )
            self.assertEqual(valid_cleanup.get("status"), "success")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_file_detection_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """file_detection_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="file-detection-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("file_detection_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("detect_file_type", names)
            self.assertIn("batch_detect_file_types", names)
            self.assertIn("analyze_detection_accuracy", names)
            self.assertIn("generate_detection_report", names)

            detect_schema = await get_schema("file_detection_tools", "detect_file_type")
            schema_props = (detect_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("voting", (schema_props.get("strategy") or {}).get("enum", []))

            report_schema = await get_schema("file_detection_tools", "generate_detection_report")
            report_props = (report_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((report_props.get("top_mime_types") or {}).get("maximum"), 50)
            self.assertEqual((report_props.get("include_examples") or {}).get("default"), True)

            invalid_methods = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_detection_tools",
                    "detect_file_type",
                    {
                        "file_path": "/tmp/sample.txt",
                        "methods": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_methods.get("status"), "error")
            self.assertIn("methods entries must be one of", str(invalid_methods.get("message", "")))

            invalid_report = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_detection_tools",
                    "generate_detection_report",
                    {
                        "results": {},
                    },
                )
            )
            self.assertEqual(invalid_report.get("status"), "error")
            self.assertIn("non-empty object", str(invalid_report.get("message", "")))

            valid_report = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_detection_tools",
                    "generate_detection_report",
                    {
                        "results": {
                            "/tmp/a.txt": {"mime_type": "text/plain", "confidence": 0.9},
                            "/tmp/b.pdf": {"mime_type": "application/pdf", "confidence": 0.8},
                            "/tmp/c": {"error": "not detected"},
                        }
                    },
                )
            )
            self.assertEqual(valid_report.get("status"), "success")
            report_payload = valid_report.get("report") or {}
            self.assertEqual(report_payload.get("total_files"), 3)
            self.assertEqual(report_payload.get("successful"), 2)

            missing_batch_source = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_detection_tools",
                    "batch_detect_file_types",
                    {
                        "pattern": "*.txt",
                    },
                )
            )
            self.assertEqual(missing_batch_source.get("status"), "error")
            self.assertIn("either directory or file_paths must be provided", str(missing_batch_source.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_audit_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """audit_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="audit-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("audit_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("record_audit_event", names)
            self.assertIn("generate_audit_report", names)
            self.assertIn("audit_tools", names)

            event_schema = await get_schema("audit_tools", "record_audit_event")
            event_props = (event_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("critical", (event_props.get("severity") or {}).get("enum", []))

            tools_schema = await get_schema("audit_tools", "audit_tools")
            tools_props = (tools_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((tools_props.get("target") or {}).get("default"), ".")
            self.assertEqual((tools_props.get("action") or {}).get("default"), "audit")

            invalid_severity = self._assert_dispatch_success_envelope(
                await dispatch(
                    "audit_tools",
                    "record_audit_event",
                    {
                        "action": "dataset.read",
                        "severity": "fatal",
                    },
                )
            )
            self.assertEqual(invalid_severity.get("status"), "error")
            self.assertIn("severity must be one of", str(invalid_severity.get("message", "")))

            invalid_report_type = self._assert_dispatch_success_envelope(
                await dispatch(
                    "audit_tools",
                    "generate_audit_report",
                    {
                        "report_type": "summary",
                    },
                )
            )
            self.assertEqual(invalid_report_type.get("status"), "error")
            self.assertIn("report_type must be one of", str(invalid_report_type.get("message", "")))

            invalid_audit_tools_target = self._assert_dispatch_success_envelope(
                await dispatch(
                    "audit_tools",
                    "audit_tools",
                    {
                        "target": "   ",
                    },
                )
            )
            self.assertEqual(invalid_audit_tools_target.get("status"), "error")
            self.assertIn("target is required", str(invalid_audit_tools_target.get("message", "")))

            valid_audit_tools = self._assert_dispatch_success_envelope(
                await dispatch(
                    "audit_tools",
                    "audit_tools",
                    {
                        "target": "/tmp/audit-target",
                        "action": "scan",
                    },
                )
            )
            self.assertIn(valid_audit_tools.get("status"), ["success", "error"])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_alert_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """alert_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="alert-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("alert_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("send_discord_message", names)
            self.assertIn("evaluate_alert_rules", names)
            self.assertIn("list_alert_rules", names)
            self.assertIn("remove_alert_rule", names)

            list_schema = await get_schema("alert_tools", "list_alert_rules")
            list_props = (list_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((list_props.get("enabled_only") or {}).get("default"), False)

            remove_schema = await get_schema("alert_tools", "remove_alert_rule")
            remove_props = (remove_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((remove_props.get("rule_id") or {}).get("minLength"), 1)

            invalid_text = self._assert_dispatch_success_envelope(
                await dispatch(
                    "alert_tools",
                    "send_discord_message",
                    {
                        "text": "   ",
                    },
                )
            )
            self.assertEqual(invalid_text.get("status"), "error")
            self.assertIn("text is required", str(invalid_text.get("message", "")))

            invalid_event = self._assert_dispatch_success_envelope(
                await dispatch(
                    "alert_tools",
                    "evaluate_alert_rules",
                    {
                        "event": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_event.get("status"), "error")
            self.assertIn("event must be an object", str(invalid_event.get("message", "")))

            invalid_remove = self._assert_dispatch_success_envelope(
                await dispatch(
                    "alert_tools",
                    "remove_alert_rule",
                    {
                        "rule_id": " ",
                    },
                )
            )
            self.assertEqual(invalid_remove.get("status"), "error")
            self.assertIn("rule_id is required", str(invalid_remove.get("message", "")))

            valid_remove = self._assert_dispatch_success_envelope(
                await dispatch(
                    "alert_tools",
                    "remove_alert_rule",
                    {
                        "rule_id": "rule-1",
                    },
                )
            )
            self.assertIn(valid_remove.get("status"), ["success", "error"])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_email_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """email_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="email-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("email_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("email_test_connection", names)
            self.assertIn("email_analyze_export", names)
            self.assertIn("email_search_export", names)
            self.assertIn("email_parse_eml", names)

            search_schema = await get_schema("email_tools", "email_search_export")
            search_props = (search_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("all", (search_props.get("field") or {}).get("enum", []))

            parse_schema = await get_schema("email_tools", "email_parse_eml")
            parse_props = (parse_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((parse_props.get("include_attachments") or {}).get("default"), True)

            invalid_protocol = self._assert_dispatch_success_envelope(
                await dispatch(
                    "email_tools",
                    "email_test_connection",
                    {
                        "protocol": "smtp",
                    },
                )
            )
            self.assertEqual(invalid_protocol.get("status"), "error")
            self.assertIn("protocol must be one of", str(invalid_protocol.get("message", "")))

            missing_query = self._assert_dispatch_success_envelope(
                await dispatch(
                    "email_tools",
                    "email_search_export",
                    {
                        "file_path": "/tmp/email.json",
                        "query": "",
                    },
                )
            )
            self.assertEqual(missing_query.get("status"), "error")
            self.assertIn("query is required", str(missing_query.get("message", "")))

            missing_parse_file = self._assert_dispatch_success_envelope(
                await dispatch(
                    "email_tools",
                    "email_parse_eml",
                    {
                        "file_path": "",
                    },
                )
            )
            self.assertEqual(missing_parse_file.get("status"), "error")
            self.assertIn("file_path is required", str(missing_parse_file.get("message", "")))

            valid_parse = self._assert_dispatch_success_envelope(
                await dispatch(
                    "email_tools",
                    "email_parse_eml",
                    {
                        "file_path": "/tmp/mail.eml",
                        "include_attachments": False,
                    },
                )
            )
            self.assertIn(valid_parse.get("status"), ["success", "error"])

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_background_task_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """background_task_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="background-task-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("background_task_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("check_task_status", names)
            self.assertIn("manage_background_tasks", names)
            self.assertIn("manage_task_queue", names)
            self.assertIn("get_task_status", names)

            schema = await get_schema("background_task_tools", "check_task_status")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("limit") or {}).get("maximum"), 100)

            invalid_task_type = self._assert_dispatch_success_envelope(
                await dispatch(
                    "background_task_tools",
                    "check_task_status",
                    {
                        "task_type": "etl",
                    },
                )
            )
            self.assertEqual(invalid_task_type.get("status"), "error")
            self.assertIn("task_type must be one of", str(invalid_task_type.get("message", "")))

            missing_cancel_task_id = self._assert_dispatch_success_envelope(
                await dispatch(
                    "background_task_tools",
                    "manage_background_tasks",
                    {
                        "action": "cancel",
                    },
                )
            )
            self.assertEqual(missing_cancel_task_id.get("status"), "error")
            self.assertIn("task_id is required for cancel action", str(missing_cancel_task_id.get("message", "")))

            status_schema = await get_schema("background_task_tools", "get_task_status")
            status_props = (status_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((status_props.get("log_limit") or {}).get("maximum"), 500)

            invalid_log_limit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "background_task_tools",
                    "get_task_status",
                    {
                        "log_limit": 0,
                    },
                )
            )
            self.assertEqual(invalid_log_limit.get("status"), "error")
            self.assertIn("between 1 and 500", str(invalid_log_limit.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_dashboard_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """dashboard_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dashboard-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("dashboard_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("profile_tdfol_operation", names)
            self.assertIn("export_tdfol_statistics", names)
            self.assertIn("check_tdfol_performance_regression", names)

            export_schema = await get_schema("dashboard_tools", "export_tdfol_statistics")
            export_props = (export_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("json", (export_props.get("format") or {}).get("enum", []))

            invalid_formula = self._assert_dispatch_success_envelope(
                await dispatch(
                    "dashboard_tools",
                    "profile_tdfol_operation",
                    {
                        "formula_str": "",
                    },
                )
            )
            self.assertEqual(invalid_formula.get("status"), "error")
            self.assertIn("formula_str is required", str(invalid_formula.get("message", "")))

            invalid_top_n = self._assert_dispatch_success_envelope(
                await dispatch(
                    "dashboard_tools",
                    "get_tdfol_profiler_report",
                    {
                        "top_n": 0,
                    },
                )
            )
            self.assertEqual(invalid_top_n.get("status"), "error")
            self.assertIn("top_n must be an integer >= 1", str(invalid_top_n.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_web_scraping_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """web_scraping_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="web-scraping-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("web_scraping_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("scrape_url_tool", names)
            self.assertIn("scrape_multiple_urls_tool", names)
            self.assertIn("check_scraper_methods_tool", names)

            schema = await get_schema("web_scraping_tools", "scrape_url_tool")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("requests_only", (props.get("method") or {}).get("enum", []))

            invalid_url = self._assert_dispatch_success_envelope(
                await dispatch(
                    "web_scraping_tools",
                    "scrape_url_tool",
                    {
                        "url": "",
                    },
                )
            )
            self.assertEqual(invalid_url.get("status"), "error")
            self.assertIn("url is required", str(invalid_url.get("message", "")))

            invalid_max_concurrent = self._assert_dispatch_success_envelope(
                await dispatch(
                    "web_scraping_tools",
                    "scrape_multiple_urls_tool",
                    {
                        "urls": ["https://example.com"],
                        "max_concurrent": 0,
                    },
                )
            )
            self.assertEqual(invalid_max_concurrent.get("status"), "error")
            self.assertIn("max_concurrent must be an integer >= 1", str(invalid_max_concurrent.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_sparse_embedding_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """sparse_embedding_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="sparse-embedding-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("sparse_embedding_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("generate_sparse_embedding", names)
            self.assertIn("sparse_search", names)
            self.assertIn("manage_sparse_models", names)

            schema = await get_schema("sparse_embedding_tools", "generate_sparse_embedding")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("splade", (props.get("model") or {}).get("enum", []))

            invalid_text = self._assert_dispatch_success_envelope(
                await dispatch(
                    "sparse_embedding_tools",
                    "generate_sparse_embedding",
                    {
                        "text": "",
                    },
                )
            )
            self.assertEqual(invalid_text.get("status"), "error")
            self.assertIn("text is required", str(invalid_text.get("message", "")))

            invalid_action = self._assert_dispatch_success_envelope(
                await dispatch(
                    "sparse_embedding_tools",
                    "manage_sparse_models",
                    {
                        "action": "delete",
                    },
                )
            )
            self.assertEqual(invalid_action.get("status"), "error")
            self.assertIn("action must be one of", str(invalid_action.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_monitoring_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """monitoring_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="monitoring-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("monitoring_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("health_check", names)
            self.assertIn("get_performance_metrics", names)
            self.assertIn("generate_monitoring_report", names)
            self.assertIn("check_health", names)
            self.assertIn("collect_metrics", names)
            self.assertIn("manage_alerts", names)

            schema = await get_schema("monitoring_tools", "generate_monitoring_report")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("summary", (props.get("report_type") or {}).get("enum", []))

            enhanced_health_schema = await get_schema("monitoring_tools", "check_health")
            enhanced_health_props = (enhanced_health_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("comprehensive", (enhanced_health_props.get("check_depth") or {}).get("enum", []))

            enhanced_collect_schema = await get_schema("monitoring_tools", "collect_metrics")
            enhanced_collect_props = (enhanced_collect_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("parquet", (enhanced_collect_props.get("export_format") or {}).get("enum", []))

            enhanced_alert_schema = await get_schema("monitoring_tools", "manage_alerts")
            enhanced_alert_props = (enhanced_alert_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("configure_thresholds", (enhanced_alert_props.get("action") or {}).get("enum", []))

            invalid_check_type = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "health_check",
                    {
                        "check_type": "full",
                    },
                )
            )
            self.assertEqual(invalid_check_type.get("status"), "error")
            self.assertIn("check_type must be one of", str(invalid_check_type.get("message", "")))

            invalid_time_period = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "generate_monitoring_report",
                    {
                        "time_period": "30d",
                    },
                )
            )
            self.assertEqual(invalid_time_period.get("status"), "error")
            self.assertIn("time_period must be one of", str(invalid_time_period.get("message", "")))

            invalid_depth = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "check_health",
                    {
                        "check_depth": "deep",
                    },
                )
            )
            self.assertEqual(invalid_depth.get("status"), "error")
            self.assertIn("check_depth must be one of", str(invalid_depth.get("message", "")))

            valid_enhanced_health = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "check_health",
                    {
                        "check_depth": "comprehensive",
                    },
                )
            )
            self.assertEqual(valid_enhanced_health.get("status"), "success")
            self.assertIn("health_check", valid_enhanced_health)

            invalid_export = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "collect_metrics",
                    {
                        "export_format": "xml",
                    },
                )
            )
            self.assertEqual(invalid_export.get("status"), "error")
            self.assertIn("export_format must be one of", str(invalid_export.get("message", "")))

            valid_collect = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "collect_metrics",
                    {
                        "include_anomalies": True,
                    },
                )
            )
            self.assertEqual(valid_collect.get("status"), "success")
            self.assertIn("metrics_collection", valid_collect)

            missing_alert_id = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "manage_alerts",
                    {
                        "action": "resolve",
                    },
                )
            )
            self.assertEqual(missing_alert_id.get("status"), "error")
            self.assertIn("alert_id required", str(missing_alert_id.get("message", "")))

            valid_alerts = self._assert_dispatch_success_envelope(
                await dispatch(
                    "monitoring_tools",
                    "manage_alerts",
                    {
                        "action": "list",
                        "include_metrics": True,
                    },
                )
            )
            self.assertEqual(valid_alerts.get("status"), "success")
            self.assertIn("alerts", valid_alerts)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_ipfs_cluster_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """ipfs_cluster_tools should expose source-compatible operations with deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="ipfs-cluster-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("ipfs_cluster_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("manage_ipfs_cluster", names)
            self.assertIn("manage_ipfs_content", names)

            schema = await get_schema("ipfs_cluster_tools", "manage_ipfs_cluster")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("pin_content", (props.get("action") or {}).get("enum", []))

            invalid_action = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs_cluster_tools",
                    "manage_ipfs_cluster",
                    {
                        "action": "heal",
                    },
                )
            )
            self.assertEqual(invalid_action.get("status"), "error")
            self.assertIn("action must be one of", str(invalid_action.get("message", "")))

            missing_cid = self._assert_dispatch_success_envelope(
                await dispatch(
                    "ipfs_cluster_tools",
                    "manage_ipfs_content",
                    {
                        "action": "download",
                    },
                )
            )
            self.assertEqual(missing_cid.get("status"), "error")
            self.assertIn("cid is required for download action", str(missing_cid.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_security_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """security_tools should expose source-compatible schema and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="security-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("security_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("check_access_permission", names)
            self.assertIn("check_access_permissions_batch", names)

            schema = await get_schema("security_tools", "check_access_permission")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("resource_id") or {}).get("minLength"), 1)
            self.assertEqual((props.get("user_id") or {}).get("minLength"), 1)

            batch_schema = await get_schema("security_tools", "check_access_permissions_batch")
            batch_props = (batch_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((batch_props.get("requests") or {}).get("minItems"), 1)

            invalid_permission = self._assert_dispatch_success_envelope(
                await dispatch(
                    "security_tools",
                    "check_access_permission",
                    {
                        "resource_id": "resource-1",
                        "user_id": "user-1",
                        "permission_type": "godmode",
                    },
                )
            )
            self.assertEqual(invalid_permission.get("status"), "error")
            invalid_permission_text = (
                str(invalid_permission.get("message", ""))
                + " "
                + str(invalid_permission.get("error", ""))
            )
            self.assertIn("permission_type must be one of", invalid_permission_text)

            invalid_resource_type = self._assert_dispatch_success_envelope(
                await dispatch(
                    "security_tools",
                    "check_access_permission",
                    {
                        "resource_id": "resource-1",
                        "user_id": "user-1",
                        "resource_type": "   ",
                    },
                )
            )
            self.assertEqual(invalid_resource_type.get("status"), "error")
            invalid_resource_type_text = (
                str(invalid_resource_type.get("message", ""))
                + " "
                + str(invalid_resource_type.get("error", ""))
            )
            self.assertIn(
                "resource_type must be a non-empty string",
                invalid_resource_type_text,
            )

            invalid_batch = self._assert_dispatch_success_envelope(
                await dispatch(
                    "security_tools",
                    "check_access_permissions_batch",
                    {
                        "requests": [],
                    },
                )
            )
            self.assertEqual(invalid_batch.get("status"), "error")
            invalid_batch_text = (
                str(invalid_batch.get("message", ""))
                + " "
                + str(invalid_batch.get("error", ""))
            )
            self.assertIn("requests must be a non-empty array", invalid_batch_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_vector_store_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """vector_store_tools should expose source-compatible schema and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="vector-store-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("vector_store_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("vector_index", names)
            self.assertIn("vector_retrieval", names)
            self.assertIn("vector_metadata", names)
            self.assertIn("enhanced_vector_index", names)
            self.assertIn("enhanced_vector_search", names)
            self.assertIn("enhanced_vector_storage", names)

            schema = await get_schema("vector_store_tools", "vector_index")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("create", (props.get("action") or {}).get("enum", []))
            self.assertIn("list", (props.get("action") or {}).get("enum", []))
            self.assertEqual((props.get("index_name") or {}).get("type"), ["string", "null"])

            enhanced_search_schema = await get_schema("vector_store_tools", "enhanced_vector_search")
            enhanced_search_props = (enhanced_search_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((enhanced_search_props.get("query_vector") or {}).get("minItems"), 1)

            enhanced_storage_schema = await get_schema("vector_store_tools", "enhanced_vector_storage")
            enhanced_storage_props = (enhanced_storage_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("get_metadata", (enhanced_storage_props.get("action") or {}).get("enum", []))

            invalid_action = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "vector_index",
                    {
                        "action": "truncate",
                        "index_name": "idx",
                    },
                )
            )
            self.assertEqual(invalid_action.get("status"), "error")
            self.assertIn("action must be one of", str(invalid_action.get("message", "")))

            list_indexes = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "vector_index",
                    {
                        "action": "list",
                    },
                )
            )
            self.assertEqual(list_indexes.get("status"), "success")
            self.assertEqual(list_indexes.get("action"), "list")

            invalid_limit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "vector_retrieval",
                    {
                        "collection": "docs",
                        "limit": 0,
                    },
                )
            )
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn("limit must be an integer >= 1", str(invalid_limit.get("message", "")))

            invalid_enhanced_search = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "enhanced_vector_search",
                    {
                        "collection": "docs",
                        "query_vector": [],
                    },
                )
            )
            self.assertEqual(invalid_enhanced_search.get("status"), "error")
            self.assertIn("non-empty list of numbers", str(invalid_enhanced_search.get("message", "")))

            valid_enhanced_index = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "enhanced_vector_index",
                    {
                        "action": "list",
                    },
                )
            )
            self.assertEqual(valid_enhanced_index.get("status"), "success")

            valid_enhanced_search = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "enhanced_vector_search",
                    {
                        "collection": "docs",
                        "query_vector": [0.1, 0.2, 0.3],
                        "top_k": 3,
                    },
                )
            )
            self.assertEqual(valid_enhanced_search.get("status"), "success")
            self.assertIn("results", valid_enhanced_search)

            invalid_enhanced_storage = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "enhanced_vector_storage",
                    {
                        "action": "delete",
                        "vector_ids": [""],
                    },
                )
            )
            self.assertEqual(invalid_enhanced_storage.get("status"), "error")
            self.assertIn("list of non-empty strings", str(invalid_enhanced_storage.get("message", "")))

            valid_enhanced_storage = self._assert_dispatch_success_envelope(
                await dispatch(
                    "vector_store_tools",
                    "enhanced_vector_storage",
                    {
                        "action": "list",
                        "collection": "docs",
                    },
                )
            )
            self.assertEqual(valid_enhanced_storage.get("status"), "success")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_function_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """functions should expose source-compatible schema and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="function-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("functions")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("execute_python_snippet", names)

            schema = await get_schema("functions", "execute_python_snippet")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("timeout_seconds") or {}).get("minimum"), 1)

            invalid_code = self._assert_dispatch_success_envelope(
                await dispatch(
                    "functions",
                    "execute_python_snippet",
                    {
                        "code": " ",
                    },
                )
            )
            self.assertEqual(invalid_code.get("status"), "error")
            self.assertIn("code must be a non-empty string", str(invalid_code.get("message", "")))

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "functions",
                    "execute_python_snippet",
                    {
                        "code": "print('ok')",
                        "timeout_seconds": 0,
                    },
                )
            )
            self.assertEqual(invalid_timeout.get("status"), "error")
            self.assertIn("timeout_seconds must be an integer >= 1", str(invalid_timeout.get("message", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_discord_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """discord_tools should expose source-compatible schema and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="discord-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("discord_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("discord_list_guilds", names)
            self.assertIn("discord_list_channels", names)
            self.assertIn("discord_list_dm_channels", names)
            self.assertIn("discord_export_channel", names)
            self.assertIn("discord_analyze_export", names)
            self.assertIn("discord_convert_export", names)
            self.assertIn("discord_batch_convert_exports", names)

            schema = await get_schema("discord_tools", "discord_list_channels")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("guild_id") or {}).get("minLength"), 1)

            export_schema = await get_schema("discord_tools", "discord_export_channel")
            export_props = (export_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((export_props.get("channel_id") or {}).get("minLength"), 1)

            invalid_guild = self._assert_dispatch_success_envelope(
                await dispatch(
                    "discord_tools",
                    "discord_list_channels",
                    {
                        "guild_id": "   ",
                    },
                )
            )
            self.assertEqual(invalid_guild.get("status"), "error")
            invalid_guild_text = (
                str(invalid_guild.get("message", ""))
                + " "
                + str(invalid_guild.get("error", ""))
            )
            self.assertIn("guild_id is required", invalid_guild_text)

            invalid_token = self._assert_dispatch_success_envelope(
                await dispatch(
                    "discord_tools",
                    "discord_list_guilds",
                    {
                        "token": "   ",
                    },
                )
            )
            self.assertEqual(invalid_token.get("status"), "error")
            invalid_token_text = (
                str(invalid_token.get("message", ""))
                + " "
                + str(invalid_token.get("error", ""))
            )
            self.assertIn("token must be a non-empty string", invalid_token_text)

            invalid_export = self._assert_dispatch_success_envelope(
                await dispatch(
                    "discord_tools",
                    "discord_export_channel",
                    {
                        "channel_id": "   ",
                    },
                )
            )
            self.assertEqual(invalid_export.get("status"), "error")
            self.assertIn("channel_id is required", str(invalid_export.get("error", "")))

            invalid_convert = self._assert_dispatch_success_envelope(
                await dispatch(
                    "discord_tools",
                    "discord_convert_export",
                    {
                        "input_path": "in.json",
                        "output_path": "out.json",
                        "to_format": "xml",
                    },
                )
            )
            self.assertEqual(invalid_convert.get("status"), "error")
            self.assertIn("to_format must be one of", str(invalid_convert.get("error", "")))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_file_converter_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """file_converter_tools should expose source-compatible schema and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="file-converter-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("file_converter_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("batch_convert_tool", names)
            self.assertIn("convert_file_tool", names)
            self.assertIn("extract_knowledge_graph_tool", names)
            self.assertIn("file_info_tool", names)
            self.assertIn("generate_summary_tool", names)
            self.assertIn("generate_embeddings_tool", names)
            self.assertIn("extract_archive_tool", names)
            self.assertIn("download_url_tool", names)

            download_schema = await get_schema("file_converter_tools", "download_url_tool")
            download_props = (download_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((download_props.get("timeout") or {}).get("minimum"), 1)
            self.assertEqual((download_props.get("max_size_mb") or {}).get("minimum"), 1)

            batch_schema = await get_schema("file_converter_tools", "batch_convert_tool")
            batch_props = (batch_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((batch_props.get("input_paths") or {}).get("minItems"), 1)
            self.assertEqual((batch_props.get("max_concurrent") or {}).get("minimum"), 1)

            embedding_schema = await get_schema("file_converter_tools", "generate_embeddings_tool")
            embedding_props = (embedding_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("qdrant", (embedding_props.get("vector_store") or {}).get("enum", []))

            archive_schema = await get_schema("file_converter_tools", "extract_archive_tool")
            archive_props = (archive_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((archive_props.get("max_depth") or {}).get("minimum"), 0)

            invalid_path = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_converter_tools",
                    "convert_file_tool",
                    {
                        "input_path": "   ",
                    },
                )
            )
            self.assertEqual(invalid_path.get("status"), "error")
            invalid_path_text = (
                str(invalid_path.get("message", ""))
                + " "
                + str(invalid_path.get("error", ""))
            )
            self.assertIn("input_path is required", invalid_path_text)

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_converter_tools",
                    "download_url_tool",
                    {
                        "url": "https://example.com",
                        "timeout": 0,
                    },
                )
            )
            self.assertEqual(invalid_timeout.get("status"), "error")
            invalid_timeout_text = (
                str(invalid_timeout.get("message", ""))
                + " "
                + str(invalid_timeout.get("error", ""))
            )
            self.assertIn("timeout must be an integer >= 1", invalid_timeout_text)

            invalid_batch = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_converter_tools",
                    "batch_convert_tool",
                    {
                        "input_paths": ["ok.txt", "   "],
                    },
                )
            )
            self.assertEqual(invalid_batch.get("status"), "error")
            invalid_batch_text = (
                str(invalid_batch.get("message", ""))
                + " "
                + str(invalid_batch.get("error", ""))
            )
            self.assertIn("input_paths must be a non-empty list of strings", invalid_batch_text)

            invalid_vector_store = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_converter_tools",
                    "generate_embeddings_tool",
                    {
                        "input_path": "document.pdf",
                        "vector_store": "sqlite",
                    },
                )
            )
            self.assertEqual(invalid_vector_store.get("status"), "error")
            invalid_vector_store_text = (
                str(invalid_vector_store.get("message", ""))
                + " "
                + str(invalid_vector_store.get("error", ""))
            )
            self.assertIn("vector_store must be one of", invalid_vector_store_text)

            invalid_archive = self._assert_dispatch_success_envelope(
                await dispatch(
                    "file_converter_tools",
                    "extract_archive_tool",
                    {
                        "archive_path": "archive.zip",
                        "max_depth": -1,
                    },
                )
            )
            self.assertEqual(invalid_archive.get("status"), "error")
            invalid_archive_text = (
                str(invalid_archive.get("message", ""))
                + " "
                + str(invalid_archive.get("error", ""))
            )
            self.assertIn("max_depth must be an integer >= 0", invalid_archive_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_development_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """development_tools should expose source-compatible schema and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="development-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("development_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("codebase_search", names)
            self.assertIn("documentation_generator", names)
            self.assertIn("lint_python_codebase", names)
            self.assertIn("run_comprehensive_tests", names)
            self.assertIn("test_generator", names)
            self.assertIn("vscode_cli_execute", names)
            self.assertIn("vscode_cli_status", names)

            schema = await get_schema("development_tools", "codebase_search")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("context") or {}).get("minimum"), 0)
            self.assertIn("json", (props.get("format") or {}).get("enum", []))

            docs_schema = await get_schema("development_tools", "documentation_generator")
            docs_props = (docs_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("html", (docs_props.get("format_type") or {}).get("enum", []))

            execute_schema = await get_schema("development_tools", "vscode_cli_execute")
            execute_props = (execute_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((execute_props.get("timeout") or {}).get("maximum"), 300)

            invalid_pattern = self._assert_dispatch_success_envelope(
                await dispatch(
                    "development_tools",
                    "codebase_search",
                    {
                        "pattern": "   ",
                    },
                )
            )
            self.assertEqual(invalid_pattern.get("status"), "error")
            invalid_pattern_text = (
                str(invalid_pattern.get("message", ""))
                + " "
                + str(invalid_pattern.get("error", ""))
            )
            self.assertIn("pattern is required", invalid_pattern_text)

            invalid_context = self._assert_dispatch_success_envelope(
                await dispatch(
                    "development_tools",
                    "codebase_search",
                    {
                        "pattern": "foo",
                        "context": -1,
                    },
                )
            )
            self.assertEqual(invalid_context.get("status"), "error")
            invalid_context_text = (
                str(invalid_context.get("message", ""))
                + " "
                + str(invalid_context.get("error", ""))
            )
            self.assertIn("context must be an integer >= 0", invalid_context_text)

            invalid_doc_format = self._assert_dispatch_success_envelope(
                await dispatch(
                    "development_tools",
                    "documentation_generator",
                    {
                        "input_path": "src",
                        "format_type": "pdf",
                    },
                )
            )
            self.assertEqual(invalid_doc_format.get("status"), "error")
            invalid_doc_text = (
                str(invalid_doc_format.get("message", ""))
                + " "
                + str(invalid_doc_format.get("error", ""))
            )
            self.assertIn("format_type must be one of", invalid_doc_text)

            invalid_test_framework = self._assert_dispatch_success_envelope(
                await dispatch(
                    "development_tools",
                    "run_comprehensive_tests",
                    {
                        "path": ".",
                        "test_framework": "nose",
                    },
                )
            )
            self.assertEqual(invalid_test_framework.get("status"), "error")
            invalid_test_framework_text = (
                str(invalid_test_framework.get("message", ""))
                + " "
                + str(invalid_test_framework.get("error", ""))
            )
            self.assertIn("test_framework must be one of", invalid_test_framework_text)

            invalid_cli_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "development_tools",
                    "vscode_cli_execute",
                    {
                        "command": ["--version"],
                        "timeout": 0,
                    },
                )
            )
            self.assertEqual(invalid_cli_timeout.get("status"), "error")
            invalid_cli_timeout_text = (
                str(invalid_cli_timeout.get("message", ""))
                + " "
                + str(invalid_cli_timeout.get("error", ""))
            )
            self.assertIn("timeout must be an integer between 1 and 300", invalid_cli_timeout_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_media_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """media_tools should expose source-compatible schema and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="media-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("media_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("ffmpeg_analyze", names)
            self.assertIn("ffmpeg_mux", names)
            self.assertIn("ffmpeg_stream_output", names)
            self.assertIn("ffmpeg_batch_process", names)
            self.assertIn("ytdlp_download_playlist", names)
            self.assertIn("ytdlp_batch_download", names)
            self.assertIn("ytdlp_extract_info", names)
            self.assertIn("ytdlp_search_videos", names)

            schema = await get_schema("media_tools", "ytdlp_extract_info")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("url") or {}).get("minLength"), 1)

            mux_schema = await get_schema("media_tools", "ffmpeg_mux")
            mux_props = (mux_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((mux_props.get("output_file") or {}).get("minLength"), 1)

            stream_output_schema = await get_schema("media_tools", "ffmpeg_stream_output")
            stream_output_props = (stream_output_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((stream_output_props.get("stream_url") or {}).get("minLength"), 1)

            batch_schema = await get_schema("media_tools", "ffmpeg_batch_process")
            batch_props = (batch_schema.get("input_schema") or {}).get("properties", {})
            batch_one_of = (batch_props.get("input_files") or {}).get("oneOf", [])
            self.assertEqual((batch_one_of[0] or {}).get("minItems"), 1)

            playlist_schema = await get_schema("media_tools", "ytdlp_download_playlist")
            playlist_props = (playlist_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((playlist_props.get("playlist_url") or {}).get("minLength"), 1)

            search_schema = await get_schema("media_tools", "ytdlp_search_videos")
            search_props = (search_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((search_props.get("query") or {}).get("minLength"), 1)

            invalid_input = self._assert_dispatch_success_envelope(
                await dispatch(
                    "media_tools",
                    "ffmpeg_analyze",
                    {
                        "input_file": "   ",
                    },
                )
            )
            self.assertEqual(invalid_input.get("status"), "error")
            invalid_input_text = (
                str(invalid_input.get("message", ""))
                + " "
                + str(invalid_input.get("error", ""))
            )
            self.assertIn("input_file must be a non-empty string or object", invalid_input_text)

            invalid_mux = self._assert_dispatch_success_envelope(
                await dispatch(
                    "media_tools",
                    "ffmpeg_mux",
                    {
                        "output_file": "/tmp/output.mp4",
                    },
                )
            )
            self.assertEqual(invalid_mux.get("status"), "error")
            invalid_mux_text = str(invalid_mux.get("message", "")) + " " + str(invalid_mux.get("error", ""))
            self.assertIn("At least one input stream must be provided", invalid_mux_text)

            invalid_batch_parallelism = self._assert_dispatch_success_envelope(
                await dispatch(
                    "media_tools",
                    "ffmpeg_batch_process",
                    {
                        "input_files": ["/tmp/a.mp4"],
                        "output_directory": "/tmp/out",
                        "max_parallel": 0,
                    },
                )
            )
            self.assertEqual(invalid_batch_parallelism.get("status"), "error")
            invalid_batch_parallelism_text = (
                str(invalid_batch_parallelism.get("message", ""))
                + " "
                + str(invalid_batch_parallelism.get("error", ""))
            )
            self.assertIn(
                "max_parallel must be an integer greater than or equal to 1",
                invalid_batch_parallelism_text,
            )

            invalid_stream_url = self._assert_dispatch_success_envelope(
                await dispatch(
                    "media_tools",
                    "ffmpeg_stream_output",
                    {
                        "input_file": "/tmp/video.mp4",
                        "stream_url": "   ",
                    },
                )
            )
            self.assertEqual(invalid_stream_url.get("status"), "error")
            invalid_stream_url_text = (
                str(invalid_stream_url.get("message", ""))
                + " "
                + str(invalid_stream_url.get("error", ""))
            )
            self.assertIn("stream_url must be a non-empty string", invalid_stream_url_text)

            invalid_url = self._assert_dispatch_success_envelope(
                await dispatch(
                    "media_tools",
                    "ytdlp_extract_info",
                    {
                        "url": "example.com/video",
                    },
                )
            )
            self.assertEqual(invalid_url.get("status"), "error")
            invalid_url_text = (
                str(invalid_url.get("message", ""))
                + " "
                + str(invalid_url.get("error", ""))
            )
            self.assertIn("url must start with http:// or https://", invalid_url_text)

            invalid_playlist_window = self._assert_dispatch_success_envelope(
                await dispatch(
                    "media_tools",
                    "ytdlp_download_playlist",
                    {
                        "playlist_url": "https://example.com/playlist",
                        "start_index": 4,
                        "end_index": 2,
                    },
                )
            )
            self.assertEqual(invalid_playlist_window.get("status"), "error")
            invalid_playlist_window_text = (
                str(invalid_playlist_window.get("message", ""))
                + " "
                + str(invalid_playlist_window.get("error", ""))
            )
            self.assertIn(
                "end_index must be an integer greater than or equal to start_index when provided",
                invalid_playlist_window_text,
            )

            invalid_search_query = self._assert_dispatch_success_envelope(
                await dispatch(
                    "media_tools",
                    "ytdlp_search_videos",
                    {
                        "query": "   ",
                    },
                )
            )
            self.assertEqual(invalid_search_query.get("status"), "error")
            invalid_search_query_text = (
                str(invalid_search_query.get("message", ""))
                + " "
                + str(invalid_search_query.get("error", ""))
            )
            self.assertIn("query must be a non-empty string", invalid_search_query_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_workflow_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """workflow_tools should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="workflow-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("workflow_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("resume_workflow", names)
            self.assertIn("merge_merkle_clock", names)
            self.assertIn("enhanced_workflow_management", names)
            self.assertIn("enhanced_batch_processing", names)
            self.assertIn("enhanced_data_pipeline", names)

            schema = await get_schema("workflow_tools", "schedule_p2p_workflow")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("workflow_id") or {}).get("minLength"), 1)
            self.assertEqual((props.get("tags") or {}).get("minItems"), 1)
            self.assertEqual((props.get("priority") or {}).get("minimum"), 0)

            enhanced_management_schema = await get_schema("workflow_tools", "enhanced_workflow_management")
            management_props = (enhanced_management_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("create", (management_props.get("action") or {}).get("enum", []))

            enhanced_batch_schema = await get_schema("workflow_tools", "enhanced_batch_processing")
            batch_props = (enhanced_batch_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((batch_props.get("operation_type") or {}).get("minLength"), 1)

            enhanced_pipeline_schema = await get_schema("workflow_tools", "enhanced_data_pipeline")
            pipeline_props = (enhanced_pipeline_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((pipeline_props.get("pipeline_config") or {}).get("minProperties"), 1)

            invalid_workflow_id = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "resume_workflow",
                    {
                        "workflow_id": "   ",
                    },
                )
            )
            self.assertEqual(invalid_workflow_id.get("status"), "error")
            invalid_workflow_id_text = (
                str(invalid_workflow_id.get("message", ""))
                + " "
                + str(invalid_workflow_id.get("error", ""))
            )
            self.assertIn("workflow_id must be a non-empty string", invalid_workflow_id_text)

            invalid_counter = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "merge_merkle_clock",
                    {
                        "other_peer_id": "peer-a",
                        "other_counter": -1,
                    },
                )
            )
            self.assertEqual(invalid_counter.get("status"), "error")
            invalid_counter_text = (
                str(invalid_counter.get("message", ""))
                + " "
                + str(invalid_counter.get("error", ""))
            )
            self.assertIn("other_counter must be an integer >= 0", invalid_counter_text)

            invalid_enhanced_create = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "enhanced_workflow_management",
                    {
                        "action": "create",
                    },
                )
            )
            self.assertEqual(invalid_enhanced_create.get("status"), "error")
            invalid_enhanced_create_text = (
                str(invalid_enhanced_create.get("message", ""))
                + " "
                + str(invalid_enhanced_create.get("error", ""))
            )
            self.assertIn("workflow_definition must be a non-empty object", invalid_enhanced_create_text)

            valid_enhanced_list = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "enhanced_workflow_management",
                    {
                        "action": "list",
                        "status_filter": "active",
                    },
                )
            )
            self.assertEqual(valid_enhanced_list.get("status"), "success")
            self.assertIn("workflows", valid_enhanced_list)

            invalid_batch_source = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "enhanced_batch_processing",
                    {
                        "operation_type": "reindex",
                        "data_source": {},
                        "output_config": {"destination": "/tmp/out"},
                    },
                )
            )
            self.assertEqual(invalid_batch_source.get("status"), "error")
            invalid_batch_source_text = (
                str(invalid_batch_source.get("message", ""))
                + " "
                + str(invalid_batch_source.get("error", ""))
            )
            self.assertIn("data_source must be a non-empty object", invalid_batch_source_text)

            valid_batch = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "enhanced_batch_processing",
                    {
                        "operation_type": "reindex",
                        "data_source": {"source_type": "dataset", "name": "demo"},
                        "output_config": {"destination": "/tmp/out"},
                    },
                )
            )
            self.assertEqual(valid_batch.get("status"), "success")
            self.assertIn("processing_completed", valid_batch)

            invalid_pipeline = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "enhanced_data_pipeline",
                    {
                        "pipeline_config": {"name": "demo", "load": {"destination_type": "file"}},
                    },
                )
            )
            self.assertEqual(invalid_pipeline.get("status"), "error")
            invalid_pipeline_text = (
                str(invalid_pipeline.get("message", ""))
                + " "
                + str(invalid_pipeline.get("error", ""))
            )
            self.assertIn("must include 'extract'", invalid_pipeline_text)

            valid_pipeline = self._assert_dispatch_success_envelope(
                await dispatch(
                    "workflow_tools",
                    "enhanced_data_pipeline",
                    {
                        "pipeline_config": {
                            "name": "demo-pipeline",
                            "extract": {"source_type": "dataset"},
                            "load": {"destination_type": "file"},
                        },
                    },
                )
            )
            self.assertEqual(valid_pipeline.get("status"), "success")
            self.assertIn("pipeline_name", valid_pipeline)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_p2p_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """p2p_tools should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="p2p-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("p2p_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("p2p_cache_get", names)
            self.assertIn("p2p_remote_call_tool", names)

            schema = await get_schema("p2p_tools", "p2p_service_status")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("peers_limit") or {}).get("minimum"), 1)

            invalid_key = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p_tools",
                    "p2p_cache_get",
                    {
                        "key": "   ",
                    },
                )
            )
            self.assertEqual(invalid_key.get("status"), "error")
            invalid_key_text = (
                str(invalid_key.get("message", ""))
                + " "
                + str(invalid_key.get("error", ""))
            )
            self.assertIn("key must be a non-empty string", invalid_key_text)

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "p2p_tools",
                    "p2p_remote_status",
                    {
                        "timeout_s": 0,
                    },
                )
            )
            self.assertEqual(invalid_timeout.get("status"), "error")
            invalid_timeout_text = (
                str(invalid_timeout.get("message", ""))
                + " "
                + str(invalid_timeout.get("error", ""))
            )
            self.assertIn("timeout_s must be a number > 0", invalid_timeout_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_finance_data_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """finance_data_tools should expose expanded schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="finance-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("finance_data_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("scrape_stock_data", names)
            self.assertIn("scrape_financial_news", names)
            self.assertIn("fetch_stock_data", names)
            self.assertIn("get_stock_quote", names)
            self.assertIn("fetch_financial_news", names)
            self.assertIn("search_archive_news", names)
            self.assertIn("list_financial_theorems", names)
            self.assertIn("analyze_embedding_market_correlation", names)

            schema = await get_schema("finance_data_tools", "scrape_stock_data")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("symbols") or {}).get("minItems"), 1)
            self.assertEqual(((props.get("symbols") or {}).get("items") or {}).get("minLength"), 1)
            self.assertEqual((props.get("days") or {}).get("minimum"), 1)

            fetch_stock_schema = await get_schema("finance_data_tools", "fetch_stock_data")
            fetch_stock_props = (fetch_stock_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((fetch_stock_props.get("interval") or {}).get("enum"), ["1d", "1h", "5m"])
            self.assertEqual((fetch_stock_props.get("source") or {}).get("enum"), ["yahoo"])

            theorem_schema = await get_schema("finance_data_tools", "apply_financial_theorem")
            theorem_required = (theorem_schema.get("input_schema") or {}).get("required", [])
            self.assertEqual(
                theorem_required,
                ["theorem_id", "symbol", "event_date", "event_data"],
            )

            invalid_symbols = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "scrape_stock_data",
                    {
                        "symbols": [],
                    },
                )
            )
            self.assertEqual(invalid_symbols.get("status"), "error")
            invalid_symbols_text = (
                str(invalid_symbols.get("message", ""))
                + " "
                + str(invalid_symbols.get("error", ""))
            )
            self.assertIn("symbols must be a non-empty list", invalid_symbols_text)

            invalid_max_articles = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "scrape_financial_news",
                    {
                        "topics": ["markets"],
                        "max_articles": 0,
                    },
                )
            )
            self.assertEqual(invalid_max_articles.get("status"), "error")
            invalid_max_articles_text = (
                str(invalid_max_articles.get("message", ""))
                + " "
                + str(invalid_max_articles.get("error", ""))
            )
            self.assertIn("max_articles must be an integer >= 1", invalid_max_articles_text)

            invalid_interval = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "fetch_stock_data",
                    {
                        "symbol": "AAPL",
                        "start_date": "2026-02-01",
                        "end_date": "2026-02-28",
                        "interval": "15m",
                    },
                )
            )
            self.assertEqual(invalid_interval.get("status"), "error")
            invalid_interval_text = (
                str(invalid_interval.get("message", ""))
                + " "
                + str(invalid_interval.get("error", ""))
            )
            self.assertIn("interval must be one of", invalid_interval_text)

            invalid_sources = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "fetch_financial_news",
                    {
                        "topic": "inflation",
                        "start_date": "2026-02-01",
                        "end_date": "2026-02-28",
                        "sources": "nyt",
                    },
                )
            )
            self.assertEqual(invalid_sources.get("status"), "error")
            invalid_sources_text = (
                str(invalid_sources.get("message", ""))
                + " "
                + str(invalid_sources.get("error", ""))
            )
            self.assertIn("sources must be a comma-separated list", invalid_sources_text)

            invalid_correlation = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "find_predictive_embedding_patterns",
                    {
                        "historical_embeddings_json": "[]",
                        "min_correlation": 2,
                    },
                )
            )
            self.assertEqual(invalid_correlation.get("status"), "error")
            invalid_correlation_text = (
                str(invalid_correlation.get("message", ""))
                + " "
                + str(invalid_correlation.get("error", ""))
            )
            self.assertIn("min_correlation must be a number between 0 and 1", invalid_correlation_text)

            stock_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "fetch_corporate_actions",
                    {
                        "symbol": "AAPL",
                        "start_date": "2026-02-01",
                        "end_date": "2026-02-28",
                    },
                )
            )
            self.assertIn(stock_result.get("status"), ["success", "error"])
            self.assertEqual(stock_result.get("symbol"), "AAPL")

            theorem_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "apply_financial_theorem",
                    {
                        "theorem_id": "split-theorem",
                        "symbol": "AAPL",
                        "event_date": "2026-02-01",
                        "event_data": '{"ratio": "2:1"}',
                    },
                )
            )
            self.assertIn(theorem_result.get("status"), ["success", "error"])
            self.assertEqual((theorem_result.get("theorem") or {}).get("theorem_id"), "split-theorem")

            embedding_result = self._assert_dispatch_success_envelope(
                await dispatch(
                    "finance_data_tools",
                    "analyze_embedding_market_correlation",
                    {
                        "news_articles_json": "[]",
                        "stock_data_json": "[]",
                        "time_window": 12,
                        "n_clusters": 3,
                    },
                )
            )
            self.assertIn(embedding_result.get("status"), ["success", "error"])
            self.assertEqual((embedding_result.get("analysis") or {}).get("time_window_hours"), 12)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_legal_dataset_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """legal_dataset_tools should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="legal-dataset-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("legal_dataset_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("list_state_jurisdictions", names)
            self.assertIn("scrape_state_laws", names)
            self.assertIn("expand_legal_query", names)
            self.assertIn("get_legal_synonyms", names)
            self.assertIn("get_legal_relationships", names)

            schema = await get_schema("legal_dataset_tools", "scrape_state_laws")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("rate_limit_delay") or {}).get("minimum"), 0)
            self.assertEqual((props.get("min_full_text_chars") or {}).get("minimum"), 1)
            self.assertEqual(
                (props.get("output_format") or {}).get("enum"),
                ["json", "csv", "parquet"],
            )

            expand_schema = await get_schema("legal_dataset_tools", "expand_legal_query")
            expand_props = (expand_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                (expand_props.get("strategy") or {}).get("enum"),
                ["conservative", "balanced", "aggressive"],
            )
            self.assertEqual((expand_props.get("max_expansions") or {}).get("maximum"), 50)

            relationship_schema = await get_schema("legal_dataset_tools", "get_legal_relationships")
            relationship_props = (relationship_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn(
                "hierarchical",
                (relationship_props.get("relationship_type") or {}).get("enum", []),
            )

            invalid_output_format = self._assert_dispatch_success_envelope(
                await dispatch(
                    "legal_dataset_tools",
                    "scrape_state_laws",
                    {
                        "output_format": "xml",
                    },
                )
            )
            self.assertEqual(invalid_output_format.get("status"), "error")
            invalid_output_format_text = (
                str(invalid_output_format.get("message", ""))
                + " "
                + str(invalid_output_format.get("error", ""))
            )
            self.assertIn("output_format must be one of", invalid_output_format_text)

            invalid_chars = self._assert_dispatch_success_envelope(
                await dispatch(
                    "legal_dataset_tools",
                    "scrape_state_laws",
                    {
                        "output_format": "json",
                        "min_full_text_chars": 0,
                    },
                )
            )
            self.assertEqual(invalid_chars.get("status"), "error")
            invalid_chars_text = (
                str(invalid_chars.get("message", ""))
                + " "
                + str(invalid_chars.get("error", ""))
            )
            self.assertIn("min_full_text_chars must be an integer >= 1", invalid_chars_text)

            invalid_strategy = self._assert_dispatch_success_envelope(
                await dispatch(
                    "legal_dataset_tools",
                    "expand_legal_query",
                    {
                        "query": "epa water rules",
                        "strategy": "wide",
                    },
                )
            )
            self.assertEqual(invalid_strategy.get("status"), "error")
            invalid_strategy_text = (
                str(invalid_strategy.get("message", ""))
                + " "
                + str(invalid_strategy.get("error", ""))
            )
            self.assertIn("strategy must be one of", invalid_strategy_text)

            valid_expand = self._assert_dispatch_success_envelope(
                await dispatch(
                    "legal_dataset_tools",
                    "expand_legal_query",
                    {
                        "query": "epa water rules",
                        "strategy": "balanced",
                        "domains": ["environmental"],
                    },
                )
            )
            self.assertEqual(valid_expand.get("status"), "success")
            self.assertEqual(valid_expand.get("original_query"), "epa water rules")

            invalid_relationship = self._assert_dispatch_success_envelope(
                await dispatch(
                    "legal_dataset_tools",
                    "get_legal_relationships",
                    {
                        "relationship_type": "graph",
                    },
                )
            )
            self.assertEqual(invalid_relationship.get("status"), "error")
            invalid_relationship_text = (
                str(invalid_relationship.get("message", ""))
                + " "
                + str(invalid_relationship.get("error", ""))
            )
            self.assertIn("relationship_type must be null or one of", invalid_relationship_text)

            valid_synonyms = self._assert_dispatch_success_envelope(
                await dispatch(
                    "legal_dataset_tools",
                    "get_legal_synonyms",
                    {
                        "term": "regulation",
                    },
                )
            )
            self.assertEqual(valid_synonyms.get("status"), "success")
            self.assertEqual(valid_synonyms.get("term"), "regulation")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_investigation_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """investigation_tools should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="investigation-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("investigation_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("analyze_entities", names)
            self.assertIn("map_relationships", names)
            self.assertIn("explore_entity", names)
            self.assertIn("analyze_entity_timeline", names)
            self.assertIn("detect_patterns", names)
            self.assertIn("track_provenance", names)
            self.assertIn("ingest_news_article", names)
            self.assertIn("ingest_news_feed", names)
            self.assertIn("ingest_website", names)
            self.assertIn("ingest_document_collection", names)
            self.assertIn("analyze_deontological_conflicts", names)
            self.assertIn("query_deontic_statements", names)
            self.assertIn("query_deontic_conflicts", names)
            self.assertIn("extract_geographic_entities", names)
            self.assertIn("map_spatiotemporal_events", names)
            self.assertIn("query_geographic_context", names)

            schema = await get_schema("investigation_tools", "analyze_entities")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("analysis_type") or {}).get("minLength"), 1)
            self.assertEqual((props.get("confidence_threshold") or {}).get("minimum"), 0)
            self.assertEqual((props.get("confidence_threshold") or {}).get("maximum"), 1)

            timeline_schema = await get_schema("investigation_tools", "analyze_entity_timeline")
            timeline_props = (timeline_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                (timeline_props.get("time_granularity") or {}).get("enum"),
                ["hour", "day", "week", "month"],
            )

            geo_schema = await get_schema("investigation_tools", "query_geographic_context")
            geo_props = (geo_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((geo_props.get("radius_km") or {}).get("exclusiveMinimum"), 0)

            invalid_confidence = self._assert_dispatch_success_envelope(
                await dispatch(
                    "investigation_tools",
                    "analyze_entities",
                    {
                        "corpus_data": '{"documents": []}',
                        "confidence_threshold": 1.2,
                    },
                )
            )
            self.assertEqual(invalid_confidence.get("status"), "error")
            invalid_confidence_text = (
                str(invalid_confidence.get("message", ""))
                + " "
                + str(invalid_confidence.get("error", ""))
            )
            self.assertIn("confidence_threshold must be a number between 0 and 1", invalid_confidence_text)

            invalid_depth = self._assert_dispatch_success_envelope(
                await dispatch(
                    "investigation_tools",
                    "map_relationships",
                    {
                        "corpus_data": '{"documents": []}',
                        "max_depth": 0,
                    },
                )
            )
            self.assertEqual(invalid_depth.get("status"), "error")
            invalid_depth_text = (
                str(invalid_depth.get("message", ""))
                + " "
                + str(invalid_depth.get("error", ""))
            )
            self.assertIn("max_depth must be an integer >= 1", invalid_depth_text)

            invalid_granularity = self._assert_dispatch_success_envelope(
                await dispatch(
                    "investigation_tools",
                    "analyze_entity_timeline",
                    {
                        "corpus_data": '{"documents": []}',
                        "entity_id": "entity-1",
                        "time_granularity": "year",
                    },
                )
            )
            self.assertEqual(invalid_granularity.get("status"), "error")
            invalid_granularity_text = (
                str(invalid_granularity.get("message", ""))
                + " "
                + str(invalid_granularity.get("error", ""))
            )
            self.assertIn("time_granularity must be one of", invalid_granularity_text)

            valid_explore = self._assert_dispatch_success_envelope(
                await dispatch(
                    "investigation_tools",
                    "explore_entity",
                    {
                        "entity_id": "entity-1",
                        "corpus_data": '{"documents": []}',
                    },
                )
            )
            self.assertEqual(valid_explore.get("status"), "success")
            self.assertEqual(valid_explore.get("entity_id"), "entity-1")

            invalid_modality = self._assert_dispatch_success_envelope(
                await dispatch(
                    "investigation_tools",
                    "query_deontic_statements",
                    {
                        "corpus_data": '{"documents": []}',
                        "modality": "should",
                    },
                )
            )
            self.assertEqual(invalid_modality.get("status"), "error")
            invalid_modality_text = (
                str(invalid_modality.get("message", ""))
                + " "
                + str(invalid_modality.get("error", ""))
            )
            self.assertIn("modality must be null or one of", invalid_modality_text)

            valid_geo = self._assert_dispatch_success_envelope(
                await dispatch(
                    "investigation_tools",
                    "query_geographic_context",
                    {
                        "query": "incident near river",
                        "corpus_data": '{"documents": []}',
                        "radius_km": 25,
                    },
                )
            )
            self.assertEqual(valid_geo.get("status"), "success")
            self.assertEqual(valid_geo.get("query"), "incident near river")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_software_engineering_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """software_engineering_tools should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="software-engineering-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("software_engineering_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("analyze_github_actions", names)
            self.assertIn("parse_systemd_logs", names)
            self.assertIn("analyze_service_health", names)
            self.assertIn("parse_kubernetes_logs", names)
            self.assertIn("analyze_pod_health", names)
            self.assertIn("detect_error_patterns", names)
            self.assertIn("suggest_fixes", names)
            self.assertIn("coordinate_auto_healing", names)
            self.assertIn("monitor_healing_effectiveness", names)
            self.assertIn("scrape_repository", names)
            self.assertIn("search_repositories", names)

            schema = await get_schema("software_engineering_tools", "scrape_repository")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("max_items") or {}).get("minimum"), 1)

            actions_schema = await get_schema("software_engineering_tools", "analyze_github_actions")
            actions_props = (actions_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((actions_props.get("max_runs") or {}).get("minimum"), 1)

            systemd_schema = await get_schema("software_engineering_tools", "parse_systemd_logs")
            systemd_props = (systemd_schema.get("input_schema") or {}).get("properties", {})
            self.assertIn("warning", (systemd_props.get("priority_filter") or {}).get("enum", []))

            service_schema = await get_schema("software_engineering_tools", "analyze_service_health")
            service_required = (service_schema.get("input_schema") or {}).get("required", [])
            self.assertEqual(service_required, ["log_data", "service_name"])

            monitor_schema = await get_schema("software_engineering_tools", "monitor_healing_effectiveness")
            monitor_props = (monitor_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((monitor_props.get("healing_history") or {}).get("minItems"), 1)

            invalid_url = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "scrape_repository",
                    {
                        "repository_url": "https://example.com/repo",
                    },
                )
            )
            self.assertEqual(invalid_url.get("status"), "error")
            invalid_url_text = (
                str(invalid_url.get("message", ""))
                + " "
                + str(invalid_url.get("error", ""))
            )
            self.assertIn("repository_url must start with", invalid_url_text)

            invalid_max_results = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "search_repositories",
                    {
                        "query": "smoke",
                        "max_results": 0,
                    },
                )
            )
            self.assertEqual(invalid_max_results.get("status"), "error")
            invalid_max_results_text = (
                str(invalid_max_results.get("message", ""))
                + " "
                + str(invalid_max_results.get("error", ""))
            )
            self.assertIn("max_results must be an integer >= 1", invalid_max_results_text)

            invalid_actions_runs = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "analyze_github_actions",
                    {
                        "repository_url": "https://github.com/example/repo",
                        "max_runs": 0,
                    },
                )
            )
            self.assertEqual(invalid_actions_runs.get("status"), "error")
            invalid_actions_runs_text = (
                str(invalid_actions_runs.get("message", ""))
                + " "
                + str(invalid_actions_runs.get("error", ""))
            )
            self.assertIn("max_runs must be an integer >= 1", invalid_actions_runs_text)

            invalid_priority_filter = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "parse_systemd_logs",
                    {
                        "log_content": "svc log",
                        "priority_filter": "panic",
                    },
                )
            )
            self.assertEqual(invalid_priority_filter.get("status"), "error")
            invalid_priority_filter_text = (
                str(invalid_priority_filter.get("message", ""))
                + " "
                + str(invalid_priority_filter.get("error", ""))
            )
            self.assertIn("priority_filter must be null or one of", invalid_priority_filter_text)

            invalid_service_log_data = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "analyze_service_health",
                    {
                        "log_data": {},
                        "service_name": "api",
                    },
                )
            )
            self.assertEqual(invalid_service_log_data.get("status"), "error")
            invalid_service_log_data_text = (
                str(invalid_service_log_data.get("message", ""))
                + " "
                + str(invalid_service_log_data.get("error", ""))
            )
            self.assertIn("log_data must be a non-empty object", invalid_service_log_data_text)

            invalid_k8s_severity = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "parse_kubernetes_logs",
                    {
                        "log_content": "2026-01-01T00:00:00.000Z INFO [api] ok",
                        "severity_filter": "trace",
                    },
                )
            )
            self.assertEqual(invalid_k8s_severity.get("status"), "error")
            invalid_k8s_severity_text = (
                str(invalid_k8s_severity.get("message", ""))
                + " "
                + str(invalid_k8s_severity.get("error", ""))
            )
            self.assertIn("severity_filter must be null or one of", invalid_k8s_severity_text)

            invalid_error_logs = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "detect_error_patterns",
                    {
                        "error_logs": [],
                    },
                )
            )
            self.assertEqual(invalid_error_logs.get("status"), "error")
            invalid_error_logs_text = (
                str(invalid_error_logs.get("message", ""))
                + " "
                + str(invalid_error_logs.get("error", ""))
            )
            self.assertIn("error_logs must be a list of at least 1 non-empty strings", invalid_error_logs_text)

            invalid_fix_pattern = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "suggest_fixes",
                    {
                        "error_pattern": "   ",
                    },
                )
            )
            self.assertEqual(invalid_fix_pattern.get("status"), "error")
            invalid_fix_pattern_text = (
                str(invalid_fix_pattern.get("message", ""))
                + " "
                + str(invalid_fix_pattern.get("error", ""))
            )
            self.assertIn("error_pattern must be a non-empty string", invalid_fix_pattern_text)

            invalid_healing_history = self._assert_dispatch_success_envelope(
                await dispatch(
                    "software_engineering_tools",
                    "monitor_healing_effectiveness",
                    {
                        "healing_history": [],
                    },
                )
            )
            self.assertEqual(invalid_healing_history.get("status"), "error")
            invalid_healing_history_text = (
                str(invalid_healing_history.get("message", ""))
                + " "
                + str(invalid_healing_history.get("error", ""))
            )
            self.assertIn("healing_history must be a list of at least 1 objects", invalid_healing_history_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_cache_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """cache_tools should expose enhanced cache stats/monitor schemas and deterministic envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="cache-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("cache_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("get_cache_stats", names)
            self.assertIn("monitor_cache", names)

            schema = await get_schema("cache_tools", "get_cache_stats")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                (props.get("format") or {}).get("enum"),
                ["json", "summary"],
            )

            invalid_format = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cache_tools",
                    "get_cache_stats",
                    {
                        "format": "xml",
                    },
                )
            )
            self.assertEqual(invalid_format.get("status"), "error")
            invalid_format_text = (
                str(invalid_format.get("message", ""))
                + " "
                + str(invalid_format.get("error", ""))
            )
            self.assertIn("format must be either", invalid_format_text)

            invalid_metrics = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cache_tools",
                    "monitor_cache",
                    {
                        "metrics": ["hit_rate", "   "],
                    },
                )
            )
            self.assertEqual(invalid_metrics.get("status"), "error")
            invalid_metrics_text = (
                str(invalid_metrics.get("message", ""))
                + " "
                + str(invalid_metrics.get("error", ""))
            )
            self.assertIn("metrics must be a list of non-empty strings", invalid_metrics_text)

            monitor_ok = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cache_tools",
                    "monitor_cache",
                    {
                        "metrics": ["hit_rate", "memory_usage"],
                        "include_predictions": True,
                    },
                )
            )
            self.assertEqual(monitor_ok.get("status"), "success")
            self.assertIn("predictions", monitor_ok)

            analyze_ok = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cache_tools",
                    "manage_cache",
                    {
                        "action": "analyze",
                    },
                )
            )
            self.assertEqual(analyze_ok.get("status"), "success")
            self.assertIn("analysis", analyze_ok)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_bespoke_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """bespoke_tools should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="bespoke-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("bespoke_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("system_health", names)
            self.assertIn("cache_stats", names)
            self.assertIn("system_status", names)
            self.assertIn("execute_workflow", names)
            self.assertIn("list_indices", names)
            self.assertIn("delete_index", names)
            self.assertIn("create_vector_store", names)

            schema = await get_schema("bespoke_tools", "cache_stats")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("namespace") or {}).get("minLength"), 1)

            workflow_schema = await get_schema("bespoke_tools", "execute_workflow")
            workflow_props = (workflow_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((workflow_props.get("timeout_seconds") or {}).get("minimum"), 1)

            create_schema = await get_schema("bespoke_tools", "create_vector_store")
            create_props = (create_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((create_props.get("dimension") or {}).get("minimum"), 1)

            invalid_namespace = self._assert_dispatch_success_envelope(
                await dispatch(
                    "bespoke_tools",
                    "cache_stats",
                    {
                        "namespace": "   ",
                    },
                )
            )
            self.assertEqual(invalid_namespace.get("status"), "error")
            invalid_namespace_text = (
                str(invalid_namespace.get("message", ""))
                + " "
                + str(invalid_namespace.get("error", ""))
            )
            self.assertIn("namespace must be null or a non-empty string", invalid_namespace_text)

            invalid_workflow = self._assert_dispatch_success_envelope(
                await dispatch(
                    "bespoke_tools",
                    "execute_workflow",
                    {
                        "workflow_id": "audit_report",
                        "parameters": ["bad"],
                    },
                )
            )
            self.assertEqual(invalid_workflow.get("status"), "error")
            invalid_workflow_text = (
                str(invalid_workflow.get("message", ""))
                + " "
                + str(invalid_workflow.get("error", ""))
            )
            self.assertIn("parameters must be null or an object", invalid_workflow_text)

            invalid_store_type = self._assert_dispatch_success_envelope(
                await dispatch(
                    "bespoke_tools",
                    "list_indices",
                    {
                        "store_type": "sqlite",
                    },
                )
            )
            self.assertEqual(invalid_store_type.get("status"), "error")
            invalid_store_type_text = (
                str(invalid_store_type.get("message", ""))
                + " "
                + str(invalid_store_type.get("error", ""))
            )
            self.assertIn("store_type must be one of", invalid_store_type_text)

            create_ok = self._assert_dispatch_success_envelope(
                await dispatch(
                    "bespoke_tools",
                    "create_vector_store",
                    {
                        "store_name": "Demo Store",
                        "dimension": 384,
                    },
                )
            )
            self.assertEqual(create_ok.get("status"), "success")
            self.assertIn("store_info", create_ok)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_cli_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """cli should expose schema contracts and deterministic validation envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="cli-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("cli")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("execute_command", names)
            self.assertIn("scrape_pubmed_cli", names)
            self.assertIn("scrape_clinical_trials_cli", names)
            self.assertIn("discover_protein_binders_cli", names)
            self.assertIn("discover_enzyme_inhibitors_cli", names)
            self.assertIn("discover_biomolecules_rag_cli", names)

            schema = await get_schema("cli", "execute_command")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("command") or {}).get("minLength"), 1)
            self.assertEqual((props.get("timeout_seconds") or {}).get("minimum"), 1)

            biomolecule_schema = await get_schema("cli", "discover_biomolecules_rag_cli")
            biomolecule_props = (biomolecule_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((biomolecule_props.get("type") or {}).get("enum"), ["binders", "inhibitors", "pathway"])

            invalid_command = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cli",
                    "execute_command",
                    {
                        "command": "   ",
                    },
                )
            )
            self.assertEqual(invalid_command.get("status"), "error")
            invalid_command_text = (
                str(invalid_command.get("message", ""))
                + " "
                + str(invalid_command.get("error", ""))
            )
            self.assertIn("command must be a non-empty string", invalid_command_text)

            invalid_pubmed = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cli",
                    "scrape_pubmed_cli",
                    {
                        "query": "COVID-19",
                        "research_type": "case_report",
                    },
                )
            )
            self.assertEqual(invalid_pubmed.get("status"), "error")
            invalid_pubmed_text = (
                str(invalid_pubmed.get("message", ""))
                + " "
                + str(invalid_pubmed.get("error", ""))
            )
            self.assertIn("research_type must be one of", invalid_pubmed_text)

            invalid_trials = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cli",
                    "scrape_clinical_trials_cli",
                    {},
                )
            )
            self.assertEqual(invalid_trials.get("status"), "error")
            invalid_trials_text = (
                str(invalid_trials.get("message", ""))
                + " "
                + str(invalid_trials.get("error", ""))
            )
            self.assertIn("query or condition is required", invalid_trials_text)

            cli_ok = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cli",
                    "discover_biomolecules_rag_cli",
                    {
                        "target": "mTOR signaling",
                        "type": "pathway",
                    },
                )
            )
            self.assertEqual(cli_ok.get("status"), "success")
            self.assertEqual(cli_ok.get("type"), "pathway")

            invalid_timeout = self._assert_dispatch_success_envelope(
                await dispatch(
                    "cli",
                    "execute_command",
                    {
                        "command": "echo",
                        "timeout_seconds": 0,
                    },
                )
            )
            self.assertEqual(invalid_timeout.get("status"), "error")
            invalid_timeout_text = (
                str(invalid_timeout.get("message", ""))
                + " "
                + str(invalid_timeout.get("error", ""))
            )
            self.assertIn("timeout_seconds must be an integer >= 1", invalid_timeout_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_medical_research_scrapers_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """medical_research_scrapers should expose schema contracts and deterministic envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="medical-research-scrapers-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("medical_research_scrapers")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("scrape_pubmed_medical_research", names)
            self.assertIn("scrape_clinical_trials", names)

            schema = await get_schema("medical_research_scrapers", "scrape_pubmed_medical_research")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((props.get("query") or {}).get("minLength"), 1)
            self.assertEqual((props.get("max_results") or {}).get("minimum"), 1)

            invalid_query = self._assert_dispatch_success_envelope(
                await dispatch(
                    "medical_research_scrapers",
                    "scrape_pubmed_medical_research",
                    {
                        "query": "   ",
                    },
                )
            )
            self.assertEqual(invalid_query.get("status"), "error")
            invalid_query_text = (
                str(invalid_query.get("message", ""))
                + " "
                + str(invalid_query.get("error", ""))
            )
            self.assertIn("query must be a non-empty string", invalid_query_text)

            invalid_max = self._assert_dispatch_success_envelope(
                await dispatch(
                    "medical_research_scrapers",
                    "scrape_clinical_trials",
                    {
                        "query": "diabetes",
                        "max_results": 0,
                    },
                )
            )
            self.assertEqual(invalid_max.get("status"), "error")
            invalid_max_text = (
                str(invalid_max.get("message", ""))
                + " "
                + str(invalid_max.get("error", ""))
            )
            self.assertIn("max_results must be an integer >= 1", invalid_max_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_lizardpersons_function_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """lizardpersons_function_tools should expose schema contracts and deterministic envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="lizardpersons-function-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("lizardpersons_function_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("get_current_time", names)

            schema = await get_schema("lizardpersons_function_tools", "get_current_time")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(
                (props.get("format_type") or {}).get("enum"),
                ["iso", "human", "timestamp"],
            )

            invalid_format = self._assert_dispatch_success_envelope(
                await dispatch(
                    "lizardpersons_function_tools",
                    "get_current_time",
                    {
                        "format_type": "custom",
                    },
                )
            )
            self.assertEqual(invalid_format.get("status"), "error")
            invalid_format_text = (
                str(invalid_format.get("message", ""))
                + " "
                + str(invalid_format.get("error", ""))
            )
            self.assertIn("format_type must be one of", invalid_format_text)

            invalid_flag = self._assert_dispatch_success_envelope(
                await dispatch(
                    "lizardpersons_function_tools",
                    "get_current_time",
                    {
                        "format_type": "iso",
                        "check_if_within_working_hours": "yes",
                    },
                )
            )
            self.assertEqual(invalid_flag.get("status"), "error")
            invalid_flag_text = (
                str(invalid_flag.get("message", ""))
                + " "
                + str(invalid_flag.get("error", ""))
            )
            self.assertIn("check_if_within_working_hours must be a boolean", invalid_flag_text)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_legacy_mcp_tools_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """legacy_mcp_tools should expose schema contracts and deterministic envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="legacy-mcp-tools-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("legacy_mcp_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("legacy_tools_inventory", names)

            schema = await get_schema("legacy_mcp_tools", "legacy_tools_inventory")
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(props, {})

            dispatched = self._assert_dispatch_success_envelope(
                await dispatch(
                    "legacy_mcp_tools",
                    "legacy_tools_inventory",
                    {},
                )
            )
            self.assertIn(dispatched.get("status"), ["success", "error"])
            self.assertIs(dispatched.get("deprecated"), True)
            self.assertIn("temporal_deontic_tool_count", dispatched)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_lizardperson_argparse_programs_discovery_schema_and_dispatch_parity(self, mock_wrapper):
        """lizardperson_argparse_programs should expose schema contracts and deterministic envelopes."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="lizardperson-argparse-programs-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("lizardperson_argparse_programs")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            self.assertIn("municipal_bluebook_validator_info", names)
            self.assertIn("municipal_bluebook_validator_invoke", names)

            schema = await get_schema(
                "lizardperson_argparse_programs",
                "municipal_bluebook_validator_info",
            )
            props = (schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual(props, {})

            invoke_schema = await get_schema(
                "lizardperson_argparse_programs",
                "municipal_bluebook_validator_invoke",
            )
            invoke_props = (invoke_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((invoke_props.get("allow_execution") or {}).get("default"), False)

            dispatched = self._assert_dispatch_success_envelope(
                await dispatch(
                    "lizardperson_argparse_programs",
                    "municipal_bluebook_validator_info",
                    {},
                )
            )
            self.assertIn(dispatched.get("status"), ["success", "error"])
            self.assertEqual(
                dispatched.get("entrypoint"),
                "municipal_bluebook_citation_validator.main",
            )

            invoked = self._assert_dispatch_success_envelope(
                await dispatch(
                    "lizardperson_argparse_programs",
                    "municipal_bluebook_validator_invoke",
                    {"argv": ["--sample-size", "7"]},
                )
            )
            self.assertEqual(invoked.get("status"), "success")
            self.assertIs(invoked.get("dry_run"), True)
            self.assertIs(invoked.get("invoked"), False)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_workflow_tools_expanded_p2p_parity_operations(self, mock_wrapper):
        """workflow_tools should expose and dispatch expanded source-compatible P2P operations."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="workflow-tools-expanded-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("workflow_tools")
            names = [tool.get("name") for tool in listed.get("tools", [])]
            expected = {
                "create_template",
                "get_next_p2p_workflow",
                "add_p2p_peer",
                "remove_p2p_peer",
                "get_p2p_scheduler_status",
                "calculate_peer_distance",
                "merge_merkle_clock",
            }
            self.assertTrue(expected.issubset(set(names)))

            template_schema = await get_schema("workflow_tools", "create_template")
            self.assertIn("template", (template_schema.get("input_schema") or {}).get("properties", {}))

            add_peer_schema = await get_schema("workflow_tools", "add_p2p_peer")
            self.assertIn("peer_id", (add_peer_schema.get("input_schema") or {}).get("properties", {}))

            calls = [
                ("get_next_p2p_workflow", {}),
                ("add_p2p_peer", {"peer_id": "peer-a"}),
                ("remove_p2p_peer", {"peer_id": "peer-a"}),
                ("get_p2p_scheduler_status", {}),
                ("calculate_peer_distance", {"hash1": "0f0f", "hash2": "f0f0"}),
                (
                    "merge_merkle_clock",
                    {
                        "other_peer_id": "peer-a",
                        "other_counter": 1,
                        "other_parent_hash": None,
                    },
                ),
            ]

            for tool_name, params in calls:
                result = self._assert_dispatch_success_envelope(
                    await dispatch("workflow_tools", tool_name, params)
                )
                self.assertTrue("status" in result or "success" in result or "error" in result)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_p2p_tools_expanded_parity_operations(self, mock_wrapper):
        """p2p_tools should expose source-compatible local and remote helper operations."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="p2p-tools-expanded-parity")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("p2p_tools")
            names = {tool.get("name") for tool in listed.get("tools", [])}
            expected = {
                "p2p_cache_has",
                "p2p_cache_delete",
                "p2p_task_delete",
                "p2p_remote_status",
                "p2p_remote_call_tool",
                "p2p_remote_cache_get",
                "p2p_remote_cache_set",
                "p2p_remote_cache_has",
                "p2p_remote_cache_delete",
                "p2p_remote_submit_task",
            }
            self.assertTrue(expected.issubset(names))

            remote_status_schema = await get_schema("p2p_tools", "p2p_remote_status")
            self.assertIn("remote_multiaddr", (remote_status_schema.get("input_schema") or {}).get("properties", {}))

            remote_call_schema = await get_schema("p2p_tools", "p2p_remote_call_tool")
            self.assertIn("tool_name", (remote_call_schema.get("input_schema") or {}).get("properties", {}))

            calls = [
                ("p2p_cache_has", {"key": "smoke"}),
                ("p2p_cache_delete", {"key": "smoke"}),
                ("p2p_task_delete", {"task_id": "task-1"}),
                ("p2p_remote_status", {}),
                ("p2p_remote_call_tool", {"tool_name": "health", "args": {}}),
                ("p2p_remote_cache_get", {"key": "smoke"}),
                ("p2p_remote_cache_set", {"key": "smoke", "value": {"ok": True}}),
                ("p2p_remote_cache_has", {"key": "smoke"}),
                ("p2p_remote_cache_delete", {"key": "smoke"}),
                (
                    "p2p_remote_submit_task",
                    {"task_type": "smoke", "model_name": "demo", "payload": {}},
                ),
            ]

            for tool_name, params in calls:
                result = self._assert_dispatch_success_envelope(
                    await dispatch("p2p_tools", tool_name, params)
                )
                self.assertTrue("ok" in result or "status" in result or "success" in result or "error" in result)

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_mcplusplus_tools_engine_status_operations(self, mock_wrapper):
        """mcplusplus tools should expose engine-backed status helper operations."""

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="mcplusplus-tools-engine-status")

        async def _run_flow() -> None:
            tools_list = server.tools["tools_list_tools"]["function"]
            get_schema = server.tools["tools_get_schema"]["function"]
            dispatch = server.tools["tools_dispatch"]["function"]

            listed = await tools_list("mcplusplus")
            names = {tool.get("name") for tool in listed.get("tools", [])}
            expected = {
                "mcplusplus_engine_status",
                "mcplusplus_list_engines",
                "mcplusplus_taskqueue_get_status",
                "mcplusplus_taskqueue_submit",
                "mcplusplus_taskqueue_priority",
                "mcplusplus_taskqueue_cancel",
                "mcplusplus_taskqueue_list",
                "mcplusplus_taskqueue_set_priority",
                "mcplusplus_taskqueue_stats",
                "mcplusplus_taskqueue_retry",
                "mcplusplus_taskqueue_pause",
                "mcplusplus_taskqueue_resume",
                "mcplusplus_taskqueue_clear",
                "mcplusplus_worker_register",
                "mcplusplus_worker_unregister",
                "mcplusplus_worker_status",
                "mcplusplus_taskqueue_result",
                "mcplusplus_workflow_get_status",
                "mcplusplus_workflow_submit",
                "mcplusplus_workflow_cancel",
                "mcplusplus_workflow_list",
                "mcplusplus_workflow_dependencies",
                "mcplusplus_workflow_result",
                "mcplusplus_peer_list",
                "mcplusplus_peer_discover",
                "mcplusplus_peer_connect",
                "mcplusplus_peer_disconnect",
                "mcplusplus_peer_metrics",
                "mcplusplus_peer_bootstrap_network",
            }
            self.assertTrue(expected.issubset(names))

            task_schema = await get_schema("mcplusplus", "mcplusplus_taskqueue_get_status")
            task_props = (task_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((task_props.get("task_id") or {}).get("minLength"), 1)

            peer_schema = await get_schema("mcplusplus", "mcplusplus_peer_list")
            peer_props = (peer_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((peer_props.get("limit") or {}).get("minimum"), 1)

            task_priority_schema = await get_schema("mcplusplus", "mcplusplus_taskqueue_priority")
            task_priority_props = (task_priority_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((task_priority_props.get("new_priority") or {}).get("exclusiveMinimum"), 0)

            task_set_priority_schema = await get_schema("mcplusplus", "mcplusplus_taskqueue_set_priority")
            task_set_priority_props = (task_set_priority_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((task_set_priority_props.get("new_priority") or {}).get("exclusiveMinimum"), 0)

            peer_discover_schema = await get_schema("mcplusplus", "mcplusplus_peer_discover")
            peer_discover_props = (peer_discover_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((peer_discover_props.get("max_peers") or {}).get("minimum"), 1)

            workflow_deps_schema = await get_schema("mcplusplus", "mcplusplus_workflow_dependencies")
            workflow_deps_props = (workflow_deps_schema.get("input_schema") or {}).get("properties", {})
            self.assertEqual((workflow_deps_props.get("fmt") or {}).get("default"), "json")

            invalid_task = self._assert_dispatch_success_envelope(
                await dispatch(
                    "mcplusplus",
                    "mcplusplus_taskqueue_get_status",
                    {"task_id": "   "},
                )
            )
            self.assertEqual(invalid_task.get("status"), "error")
            self.assertIn(
                "task_id must be a non-empty string",
                str(invalid_task.get("error", "")),
            )

            invalid_limit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "mcplusplus",
                    "mcplusplus_peer_list",
                    {"limit": 0},
                )
            )
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn(
                "limit must be an integer >= 1",
                str(invalid_limit.get("error", "")),
            )

            invalid_submit = self._assert_dispatch_success_envelope(
                await dispatch(
                    "mcplusplus",
                    "mcplusplus_taskqueue_submit",
                    {
                        "task_id": "task-1",
                        "task_type": "demo",
                        "payload": {},
                        "priority": 0,
                    },
                )
            )
            self.assertEqual(invalid_submit.get("status"), "error")
            self.assertIn(
                "priority must be > 0",
                str(invalid_submit.get("error", "")),
            )

            invalid_set_priority = self._assert_dispatch_success_envelope(
                await dispatch(
                    "mcplusplus",
                    "mcplusplus_taskqueue_set_priority",
                    {
                        "task_id": "task-1",
                        "new_priority": 0,
                    },
                )
            )
            self.assertEqual(invalid_set_priority.get("status"), "error")
            self.assertIn(
                "new_priority must be > 0",
                str(invalid_set_priority.get("error", "")),
            )

            calls = [
                ("mcplusplus_engine_status", {}),
                ("mcplusplus_list_engines", {}),
                ("mcplusplus_taskqueue_get_status", {"task_id": "task-1"}),
                (
                    "mcplusplus_taskqueue_submit",
                    {
                        "task_id": "task-1",
                        "task_type": "demo",
                        "payload": {"x": 1},
                        "metadata": {"source": "bootstrap-test"},
                    },
                ),
                ("mcplusplus_taskqueue_priority", {"task_id": "task-1", "new_priority": 2}),
                ("mcplusplus_taskqueue_cancel", {"task_id": "task-1"}),
                ("mcplusplus_taskqueue_list", {"limit": 5}),
                (
                    "mcplusplus_taskqueue_set_priority",
                    {"task_id": "task-1", "new_priority": 2.0},
                ),
                ("mcplusplus_taskqueue_stats", {}),
                ("mcplusplus_taskqueue_retry", {"task_id": "task-1"}),
                ("mcplusplus_taskqueue_pause", {}),
                ("mcplusplus_taskqueue_resume", {}),
                ("mcplusplus_taskqueue_clear", {"confirm": True}),
                (
                    "mcplusplus_worker_register",
                    {"worker_id": "worker-1", "capabilities": ["inference"]},
                ),
                ("mcplusplus_worker_status", {"worker_id": "worker-1"}),
                ("mcplusplus_worker_unregister", {"worker_id": "worker-1"}),
                ("mcplusplus_taskqueue_result", {"task_id": "task-1"}),
                ("mcplusplus_workflow_get_status", {"workflow_id": "wf-1"}),
                (
                    "mcplusplus_workflow_submit",
                    {
                        "workflow_id": "wf-1",
                        "name": "demo",
                        "steps": [{"step_id": "s1", "action": "noop"}],
                        "dependencies": ["wf-0"],
                    },
                ),
                ("mcplusplus_workflow_cancel", {"workflow_id": "wf-1"}),
                ("mcplusplus_workflow_list", {"limit": 5}),
                ("mcplusplus_workflow_dependencies", {"workflow_id": "wf-1", "fmt": "json"}),
                ("mcplusplus_workflow_result", {"workflow_id": "wf-1"}),
                (
                    "mcplusplus_peer_list",
                    {"limit": 5, "capability_filter": ["inference"], "sort_by": "last_seen", "offset": 0},
                ),
                ("mcplusplus_peer_discover", {"max_peers": 2}),
                (
                    "mcplusplus_peer_connect",
                    {"peer_id": "peer-1", "multiaddr": "/ip4/127.0.0.1/tcp/4001"},
                ),
                ("mcplusplus_peer_disconnect", {"peer_id": "peer-1"}),
                ("mcplusplus_peer_metrics", {"peer_id": "peer-1"}),
                ("mcplusplus_peer_bootstrap_network", {"max_connections": 5}),
            ]

            for tool_name, params in calls:
                result = self._assert_dispatch_success_envelope(
                    await dispatch("mcplusplus", tool_name, params)
                )
                self.assertTrue("status" in result or "success" in result or "error" in result)

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
