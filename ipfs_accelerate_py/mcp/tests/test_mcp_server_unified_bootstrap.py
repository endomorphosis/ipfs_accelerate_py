#!/usr/bin/env python3
"""Bootstrap tests for the new unified `ipfs_accelerate_py.mcp_server` package."""

import unittest
import os
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server import (
    configure_wave_a_loaders,
    get_unified_meta_tool_names,
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


class TestUnifiedMCPServerBootstrap(unittest.TestCase):
    """Validate core integration points for incremental mcp_server migration."""

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
                "IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES": "workflow,p2p",
            },
            clear=False,
        ):
            config = UnifiedMCPServerConfig.from_env(
                allowed_preload_categories=["ipfs", "workflow", "p2p"]
            )

        self.assertTrue(config.enable_unified_bridge)
        self.assertTrue(config.enable_unified_bootstrap)
        self.assertEqual(config.preload_categories, ["workflow", "p2p"])

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
        self.assertEqual(getattr(server, "_unified_preloaded_categories", []), [])
        self.assertIsInstance(getattr(server, "_unified_services", None), dict)
        self.assertIn("task_queue_factory", server._unified_services)
        self.assertIn("workflow_scheduler_factory", server._unified_services)
        self.assertIn("workflow_engine_factory", server._unified_services)
        self.assertIn("workflow_dag_executor_factory", server._unified_services)
        self.assertIn("peer_registry_factory", server._unified_services)
        self.assertIn("peer_discovery_factory", server._unified_services)
        self.assertIn("result_cache_factory", server._unified_services)
        self.assertIn("tools_list_categories", server.tools)
        self.assertIn("tools_list_tools", server.tools)
        self.assertIn("tools_get_schema", server.tools)
        self.assertIn("tools_dispatch", server.tools)
        self.assertIn("tools_runtime_metrics", server.tools)
        self.assertEqual(sorted(get_unified_meta_tool_names()), sorted(server.tools.keys()))

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

            categories = await list_categories()
            self.assertIn("smoke", categories["categories"])

            tools = await list_tools("smoke")
            self.assertEqual(tools["tools"][0]["name"], "echo")

            schema = await get_schema("smoke", "echo")
            self.assertEqual(schema["name"], "echo")
            self.assertEqual(schema["category"], "smoke")

            result = await dispatch("smoke", "echo", {"value": "ok"})
            self.assertEqual(result["echo"], "ok")

            metrics_payload = await runtime_metrics()
            self.assertIn("runtimes", metrics_payload)
            self.assertIn("fastapi", metrics_payload["runtimes"])
            self.assertIn("timeout_count", metrics_payload["runtimes"]["fastapi"])

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
            result = await dispatch(
                "ipfs",
                "ipfs_files_validate_cid",
                {"cid": "not_a_valid_cid"},
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
                result = await dispatch("ipfs", "ipfs_files_list_files", {"path": "/"})

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
                result = await dispatch(
                    "ipfs",
                    "ipfs_files_add_file",
                    {"path": "/tmp/sample.txt", "pin": False},
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
                result = await dispatch(
                    "ipfs",
                    "ipfs_files_pin_file",
                    {"cid": "QmPinnedCid123456789012345678901234567890123456"},
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
                result = await dispatch(
                    "ipfs",
                    "ipfs_files_unpin_file",
                    {"cid": "QmPinnedCid123456789012345678901234567890123456"},
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
                result = await dispatch(
                    "ipfs",
                    "ipfs_files_get_file",
                    {
                        "cid": "QmGetCid123456789012345678901234567890123456789",
                        "output_path": "/tmp/retrieved.txt",
                    },
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

            result = await dispatch("workflow", "get_workflow_templates", {})
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
                result = await dispatch("workflow", "list_workflows", {})

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
                result = await dispatch("workflow", "get_workflow", {"workflow_id": "wf-1"})

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
                result = await dispatch(
                    "workflow",
                    "create_workflow",
                    {
                        "name": "Created Workflow",
                        "description": "created via test",
                        "tasks": [{"name": "task-a"}],
                    },
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
                result = await dispatch(
                    "workflow",
                    "update_workflow",
                    {
                        "workflow_id": "wf-updated",
                        "name": "Updated Workflow",
                        "description": "updated via test",
                        "tasks": [{"name": "task-b"}],
                    },
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
                result = await dispatch(
                    "workflow",
                    "delete_workflow",
                    {"workflow_id": "wf-deleted"},
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
                result = await dispatch(
                    "workflow",
                    "start_workflow",
                    {"workflow_id": "wf-started"},
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
                result = await dispatch(
                    "workflow",
                    "pause_workflow",
                    {"workflow_id": "wf-paused"},
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
                result = await dispatch(
                    "workflow",
                    "stop_workflow",
                    {"workflow_id": "wf-stopped"},
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
                result = await dispatch("p2p", "p2p_taskqueue_status", {})

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
                result = await dispatch("p2p", "p2p_taskqueue_list_tasks", {"limit": 5})

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
                result = await dispatch("p2p", "p2p_taskqueue_get_task", {"task_id": "t1"})

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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_wait_task",
                    {"task_id": "t1", "timeout_s": 5.0},
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_complete_task",
                    {"task_id": "t1", "status": "completed", "result": {"value": 7}},
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_heartbeat",
                    {"peer_id": "peer-1", "clock": {"epoch": 1}},
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_cache_get",
                    {"key": "k1", "timeout_s": 2.0},
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_cache_set",
                    {"key": "k1", "value": {"n": 2}, "ttl_s": 30.0, "timeout_s": 2.0},
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_submit_docker_hub",
                    {
                        "image": "alpine:latest",
                        "command": ["echo", "hi"],
                        "environment": {"A": "1"},
                    },
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
                result = await dispatch(
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_submit",
                    {
                        "task_type": "inference",
                        "model_name": "model-a",
                        "payload": {"prompt": "hi"},
                    },
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_claim_next",
                    {
                        "worker_id": "worker-a",
                        "supported_task_types": ["inference"],
                        "peer_id": "peer-a",
                    },
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
                result = await dispatch(
                    "p2p",
                    "list_peers",
                    {"discover": True, "limit": 10},
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
                result = await dispatch(
                    "p2p",
                    "p2p_taskqueue_call_tool",
                    {"tool_name": "health_ping", "args": {"x": 1}},
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["result"], {"pong": True})

        anyio.run(_run_dispatch)


if __name__ == "__main__":
    unittest.main()
