"""Adapter boundary checks; no provider calls or Docker needed."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from harbor.models.agent.context import AgentContext
from harbor_agent import SupervisorWorkspaceAgent
from terminal_worker import raw_usage
from portable_context import localize_context
from terminal_report import audit_tools


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_stdout_and_only_declared_output_uploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SupervisorWorkspaceAgent(logs_dir=Path(directory), model_name='grok-4.6')
            environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=None)),
                                          upload_file=AsyncMock())
            async def spawn(*argv, **kwargs):
                self.assertIn('--semantic-context', argv)
                root = Path(argv[argv.index('--root') + 1])
                (root / 'repo').mkdir()
                (root / 'repo/run.py').write_text('# merged candidate')
                (root / 'repo/README.md').write_text('not an output')
                (root / 'worker-result.json').write_text(json.dumps({'merged_output_exists': True, 'provider_dispatched': True}))
                return SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0))
            with patch.dict('os.environ', {'SUPERVISOR_BENCH_PYTHON': '/usr/bin/python3', 'SUPERVISOR_BENCH_SEMANTIC_CONTEXT': '1'}), \
                 patch('harbor_agent.asyncio.create_subprocess_exec', side_effect=spawn):
                context = AgentContext()
                await agent.run('Implement run_tasks in /app/run.py', environment, context)
            environment.upload_file.assert_awaited_once_with(Path(directory) / 'supervisor/repo/run.py', '/app/run.py')
            self.assertIsNone(context.n_input_tokens)

    async def test_nonempty_workspace_is_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SupervisorWorkspaceAgent(logs_dir=Path(directory))
            environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='/app/input.txt')))
            with patch('harbor_agent.asyncio.create_subprocess_exec') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'empty /app'):
                    await agent.run('Implement run_tasks in /app/run.py', environment, AgentContext())
                spawn.assert_not_called()

    def test_tool_audit_flags_external_retrieval_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'attempt.log').write_text('\n'.join([
                json.dumps({'type': 'tool_call', 'toolName': 'read_file'}),
                json.dumps({'type': 'tool_call', 'toolName': 'run_terminal_command'}),
            ]))
            self.assertEqual(audit_tools(Path(directory))['outside_file_profile'], ['run_terminal_command'])

    def test_context_references_survive_worktree_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            board = root / 'tasks.todo.md'
            board.write_text(f'## TB-001 Task\n- Discovery evidence: {root}/.runtime/discovery/gap.md\n- External: /other/repo/evidence.md\n')
            self.assertEqual(localize_context(root), 1)
            self.assertIn('## TB-001 Task', board.read_text())
            self.assertIn('Discovery evidence: .runtime/discovery/gap.md', board.read_text())
            self.assertIn('/other/repo/evidence.md', board.read_text())
            self.assertEqual(localize_context(root), 0)

    def test_usage_counts_only_native_usage_events(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'attempt.log').write_text('\n'.join([
                json.dumps({'type':'usage','usage':{'input_tokens':10,'output_tokens':2,'cache_read_input_tokens':4}}),
                json.dumps({'type':'message','usage':{'input_tokens':999}}),
                'not json',
                json.dumps({'type':'usage','usage':{'input_tokens':7,'output_tokens':3}}),
            ]))
            usage = raw_usage(Path(directory))
            self.assertEqual(usage['events'], 2)
            self.assertEqual(usage['raw_token_counters'], {'input_tokens':17,'output_tokens':5,'cache_read_input_tokens':4})
            self.assertIsNone(usage['normalized_total_tokens'])


if __name__ == '__main__':
    unittest.main()
