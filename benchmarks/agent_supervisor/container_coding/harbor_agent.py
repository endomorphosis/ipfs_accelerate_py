"""Harbor adapter for a bounded native-supervisor workspace pilot.

Deliberately supports only two empty-/app coding tasks. The implementer sees
public instructions and smoke checks, never the Harbor tests or solution.
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import time

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class SupervisorWorkspaceAgent(BaseAgent):
    @staticmethod
    def name():
        return 'ipfs-supervisor-workspace'

    def version(self):
        return '0.1.0'

    async def setup(self, environment: BaseEnvironment):
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext):
        if '/app/run.py' in instruction and 'run_tasks' in instruction:
            task, output = 'cancel-async-tasks', 'run.py'
        elif '/app/polyglot/main.py.c' in instruction:
            task, output = 'polyglot-c-py', 'polyglot/main.py.c'
        else:
            raise ValueError('Workspace pilot supports only cancel-async-tasks and polyglot-c-py')
        # Both selected tasks have empty /app. Refuse silently dropping inputs.
        listing = await environment.exec(command='find /app -mindepth 1 -maxdepth 1 -print')
        if listing.return_code != 0 or (listing.stdout or '').strip():
            raise RuntimeError('Expected an empty /app for this workspace-only adapter')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        root = self.logs_dir.resolve() / 'supervisor'
        root.mkdir()
        instruction_path = root / 'instruction.md'
        instruction_path.write_text(instruction)
        python = os.environ['SUPERVISOR_BENCH_PYTHON']
        worker = Path(__file__).with_name('terminal_worker.py')
        source = worker.resolve().parents[3]
        started = time.monotonic()
        semantic_args = ['--semantic-context'] if os.environ.get('SUPERVISOR_BENCH_SEMANTIC_CONTEXT') == '1' else []
        with (root / 'daemon.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(python, str(worker), '--task', task,
                '--root', str(root), '--instruction', str(instruction_path), *semantic_args,
                env={**os.environ, 'PYTHONPATH': str(source)}, stdout=log, stderr=log,
                start_new_session=True)
            try:
                await asyncio.wait_for(process.wait(), timeout=480)
            finally:
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
        result_path = root / 'worker-result.json'
        if process.returncode != 0 or not result_path.exists():
            raise RuntimeError(f'Native supervisor subprocess failed ({process.returncode}); see daemon.log')
        result = json.loads(result_path.read_text())
        context.metadata = {'supervisor': result, 'adapter_seconds': time.monotonic() - started,
                            'profile': 'host native daemon + Docker worker + Harbor verifier',
                            'normalized_usage_unavailable': True}
        if not result.get('provider_dispatched'):
            raise RuntimeError('Supervisor did not dispatch a provider: ' +
                               str(result.get('selection_idle_reason')))
        delivered = root / 'repo' / output
        if result['merged_output_exists']:
            if delivered.is_symlink():
                raise RuntimeError('Refusing to export a symlink as a benchmark deliverable')
            await environment.exec(command='mkdir -p /app/polyglot' if task == 'polyglot-c-py' else 'true')
            await environment.upload_file(delivered, '/app/' + output)
        # Raw provider fields remain in metadata; cache/input semantics are not
        # assumed equivalent to Harbor's inclusive input-token accounting.
