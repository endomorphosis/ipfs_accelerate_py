#!/usr/bin/env python3
"""Subprocess startup contract tests for MCP transport entrypoints."""

import os
import subprocess
import sys
import textwrap
import unittest


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

            assert result == {'mode': 'trio', 'value': 'ok'}
            assert mock_trio.await_count == 1
            print('TRIO_DISPATCH_OK')

        anyio.run(main)
        """

        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=f"stderr={result.stderr}\nstdout={result.stdout}")
        self.assertIn("TRIO_DISPATCH_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
