#!/usr/bin/env python3
"""Subprocess startup contract tests for MCP transport entrypoints."""

import os
import subprocess
import sys
import textwrap
import unittest
import json


class TestMCPTransportSubprocessContracts(unittest.TestCase):
    """Validate standalone transport entrypoints in isolated subprocesses."""

    def _run_subprocess(self, code: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_standalone_run_server_contract(self) -> None:
        """`run_server` should invoke create_mcp_server and call run()."""
        code = """
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp import standalone

        class DummyServer:
            def run(self, host=None, port=None):
                print(f'RUN_CALLED:{host}:{port}')

        with patch('ipfs_accelerate_py.mcp.server.create_mcp_server', return_value=DummyServer()):
            standalone.run_server(host='127.0.0.1', port=8991, name='demo', description='demo', verbose=False)
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("RUN_CALLED:127.0.0.1:8991", result.stdout)

    def test_standalone_run_fastapi_server_contract(self) -> None:
        """`run_fastapi_server` should delegate to integration helpers."""
        code = """
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp import standalone

        with patch('ipfs_accelerate_py.mcp.integration.create_standalone_app', return_value='APP'):
            with patch('ipfs_accelerate_py.mcp.integration.run_standalone_app', side_effect=lambda **kwargs: print('FASTAPI_RUN_CALLED', kwargs['host'], kwargs['port'])):
                standalone.run_fastapi_server(host='127.0.0.1', port=8992, mount_path='/mcp', name='demo', description='demo', verbose=True)
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("FASTAPI_RUN_CALLED 127.0.0.1 8992", result.stdout)

    def test_standalone_fastapi_app_tracks_d2_bridge_disable_override(self) -> None:
        """Standalone integration app should expose D2 telemetry when bridge-disable is ignored."""
        code = """
        import json
        import os
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp import integration
        from ipfs_accelerate_py.mcp.server import _reset_mcp_facade_telemetry, get_mcp_facade_telemetry

        class DummyUnifiedServer:
            def __init__(self):
                self.app = object()
                self.mcp = None

        _reset_mcp_facade_telemetry()

        with patch('ipfs_accelerate_py.mcp_server.server.create_server', return_value=DummyUnifiedServer()):
            with patch.dict(
                os.environ,
                {
                    'IPFS_MCP_ENABLE_UNIFIED_BRIDGE': '0',
                    'IPFS_MCP_FORCE_LEGACY_ROLLBACK': '0',
                    'IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN': '0',
                },
                clear=False,
            ):
                app = integration.create_standalone_app(mount_path='/mcp', name='demo', description='demo')

        telemetry = getattr(app, '_mcp_server')._mcp_facade_telemetry
        counts = get_mcp_facade_telemetry()
        print('D2_APP_TELEMETRY', json.dumps({'telemetry': telemetry, 'counts': counts}, sort_keys=True))
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        marker = 'D2_APP_TELEMETRY '
        payload_line = next((line for line in result.stdout.splitlines() if line.startswith(marker)), '')
        self.assertTrue(payload_line, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        payload = json.loads(payload_line[len(marker):])
        telemetry = payload['telemetry']
        counts = payload['counts']
        self.assertTrue(telemetry.get('bridge_disable_ignored'))
        self.assertTrue(telemetry.get('bridge_active'))
        self.assertEqual(telemetry.get('deprecation_phase'), 'D2_opt_in_only')
        self.assertEqual(telemetry.get('reason'), 'unified_bridge')
        self.assertEqual(counts.get('bridge_disable_ignored_calls'), 1)
        self.assertEqual(counts.get('unified_bridge_calls'), 1)

    def test_standalone_run_server_preserves_d2_rollback_telemetry(self) -> None:
        """Standalone run_server should preserve explicit rollback telemetry in subprocess mode."""
        code = """
        import json
        import os
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp import standalone
        from ipfs_accelerate_py.mcp.server import _reset_mcp_facade_telemetry, get_mcp_facade_telemetry

        class DummyServer:
            def __init__(self):
                self.app = object()
                self.mcp = None

            def run(self, host=None, port=None):
                payload = {
                    'telemetry': getattr(self, '_mcp_facade_telemetry', {}),
                    'counts': get_mcp_facade_telemetry(),
                    'host': host,
                    'port': port,
                }
                print('D2_RUN_TELEMETRY', json.dumps(payload, sort_keys=True))

        _reset_mcp_facade_telemetry()

        with patch('ipfs_accelerate_py.mcp.server.MCPServerWrapper', return_value=DummyServer()):
            with patch.dict(
                os.environ,
                {
                    'IPFS_MCP_ENABLE_UNIFIED_BRIDGE': '1',
                    'IPFS_MCP_FORCE_LEGACY_ROLLBACK': '1',
                    'IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN': '0',
                },
                clear=False,
            ):
                standalone.run_server(host='127.0.0.1', port=8995, name='demo', description='demo', verbose=False)
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        marker = 'D2_RUN_TELEMETRY '
        payload_line = next((line for line in result.stdout.splitlines() if line.startswith(marker)), '')
        self.assertTrue(payload_line, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        payload = json.loads(payload_line[len(marker):])
        telemetry = payload['telemetry']
        counts = payload['counts']
        self.assertEqual(payload['host'], '127.0.0.1')
        self.assertEqual(payload['port'], 8995)
        self.assertTrue(telemetry.get('used_legacy_wrapper'))
        self.assertTrue(telemetry.get('force_legacy_rollback'))
        self.assertFalse(telemetry.get('bridge_disable_ignored'))
        self.assertEqual(telemetry.get('deprecation_phase'), 'D2_opt_in_only')
        self.assertEqual(telemetry.get('reason'), 'force_legacy_rollback')
        self.assertEqual(counts.get('rollback_calls'), 1)
        self.assertEqual(counts.get('legacy_wrapper_calls'), 1)
        self.assertEqual(counts.get('warning_emissions'), 1)

    def test_canonical_standalone_run_server_contract(self) -> None:
        """Canonical standalone facade should invoke the canonical server builder path."""
        code = """
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp_server import standalone_server

        class _DummyServer:
            def run(self, **kwargs):
                print('CANONICAL_RUN_SERVER', kwargs['host'], kwargs['port'])

        with patch('ipfs_accelerate_py.mcp_server.standalone_server.create_server', return_value=_DummyServer()):
            standalone_server.run_server(host='127.0.0.1', port=8993, name='demo', description='demo', verbose=False)
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("CANONICAL_RUN_SERVER 127.0.0.1 8993", result.stdout)

    def test_canonical_standalone_run_fastapi_server_contract(self) -> None:
        """Canonical standalone FastAPI facade should invoke canonical FastAPI runner."""
        code = """
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp_server import standalone_server

        with patch('ipfs_accelerate_py.mcp_server.standalone_server.run_canonical_fastapi_server', side_effect=lambda cfg: print('CANONICAL_FASTAPI_RUN', cfg.host, cfg.port, cfg.mount_path)):
            standalone_server.run_fastapi_server(host='127.0.0.1', port=8994, mount_path='/mcp', name='demo', description='demo', verbose=True)
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("CANONICAL_FASTAPI_RUN 127.0.0.1 8994 /mcp", result.stdout)

    def test_canonical_module_main_contract(self) -> None:
        """Canonical mcp_server.__main__ facade should delegate to standalone_server.main."""
        code = """
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp_server import __main__ as canonical_main

        with patch('ipfs_accelerate_py.mcp_server.standalone_server.main', side_effect=lambda: print('STANDALONE_MAIN_CALLED')):
            print('MAIN_RETURN', canonical_main.main())
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("STANDALONE_MAIN_CALLED", result.stdout)
        self.assertIn("MAIN_RETURN 0", result.stdout)

    def test_canonical_server_main_contract(self) -> None:
        """Canonical mcp_server.server.main should delegate to standalone_server.main."""
        code = """
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp_server import server as canonical_server

        with patch('ipfs_accelerate_py.mcp_server.standalone_server.main', side_effect=lambda: print('SERVER_STANDALONE_MAIN_CALLED')):
            print('SERVER_MAIN_RETURN', canonical_server.main())
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("SERVER_STANDALONE_MAIN_CALLED", result.stdout)
        self.assertIn("SERVER_MAIN_RETURN None", result.stdout)

    def test_canonical_simple_server_start_contract(self) -> None:
        """Canonical simple_server facade should delegate startup to standalone run_server."""
        code = """
        from unittest.mock import patch
        from ipfs_accelerate_py.mcp_server.simple_server import start_simple_server

        with patch('ipfs_accelerate_py.mcp_server.simple_server.run_server', side_effect=lambda **kwargs: print('SIMPLE_START_CALLED', kwargs)):
            start_simple_server()
        """
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("SIMPLE_START_CALLED {}", result.stdout)

    def test_unified_bootstrap_trio_dispatch_contract(self) -> None:
        """Unified bootstrap dispatch should invoke trio runtime path in subprocess."""
        code = """
        import os
        import anyio
        from unittest.mock import AsyncMock, patch
        from ipfs_accelerate_py.mcp.server import create_mcp_server

        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    'function': function,
                    'description': description,
                    'input_schema': input_schema,
                    'execution_context': execution_context,
                    'tags': tags,
                }

        async def main() -> None:
            with patch('ipfs_accelerate_py.mcp.server.MCPServerWrapper', return_value=DummyServer()):
                with patch.dict(
                    os.environ,
                    {
                        'IPFS_MCP_ENABLE_UNIFIED_BRIDGE': '1',
                        'IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP': '1',
                    },
                    clear=False,
                ):
                    server = create_mcp_server(name='trio-contract')

            manager = server._unified_tool_manager
            router = server._unified_runtime_router

            async def trio_echo(value: str):
                return {'value': value}

            manager.register_tool('transport', 'echo_trio', trio_echo, runtime='trio', description='trio echo')

            dispatch = server.tools['tools_dispatch']['function']
            with patch.object(router, '_execute_trio', AsyncMock(return_value={'mode': 'trio', 'value': 'ok'})) as mock_trio:
                result = await dispatch('transport', 'echo_trio', {'value': 'ok'})

            assert isinstance(result, dict)
            assert result.get('mode') == 'trio'
            assert result.get('value') == 'ok'
            assert mock_trio.await_count == 1
            print('TRIO_DISPATCH_OK')

        anyio.run(main)
        """

        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("TRIO_DISPATCH_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
