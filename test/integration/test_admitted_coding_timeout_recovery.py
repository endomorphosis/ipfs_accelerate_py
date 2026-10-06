"""Real provider-free timeout, native daemon replacement, and custody closure."""
from __future__ import annotations

import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from test.integration.test_admitted_benchmark_runtime import _prepare_implementation_fixture
from test.integration.test_admitted_benchmark_runtime import admitted  # noqa: F401


def test_native_frontier_maintenance_keeps_sidecars_in_the_run_state(admitted, monkeypatch):
    """Exercise the actual maintenance client without waiting for its watchdog."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        DATABASE_BLOCKED_PORTAL_FRONTIER_REASON, PortalImplementationSupervisor,
        parse_args, supervisor_config_from_args,
    )
    runtime, owner, prepared = admitted
    original_task = owner.source.get_task(prepared['task_cid'])
    # Reproduce the native launch cwd; the logical store id is not a filename.
    monkeypatch.chdir(runtime.repository)
    args = parse_args(list(runtime.profile.argv[4:]))
    supervisor = PortalImplementationSupervisor(
        supervisor_config_from_args(args, repo_root=runtime.repository))
    # The native launcher installs the exact endpoint secret *handle* and
    # schema bindings before this maintenance client opens its read route.
    for key, value in supervisor.config.database_program.environment().items():
        monkeypatch.setenv(key, value)
    # This fixture owns the live server. Supply its real read credential just
    # as the admitted launcher does; never put the value in an assertion/log.
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_QUACK_TOKEN',
        owner.server._vault.resolve(owner.identity.secret_handle))
    result = supervisor._rearm_blocked_recoverable_portal_frontier()
    assert result['attempted'] is True
    assert result['reason'] == DATABASE_BLOCKED_PORTAL_FRONTIER_REASON, result
    assert result['rearm_count'] == 0
    state = runtime.state / 'run' / 'portal-frontier-rearm'
    assert (state / 'execution.duckdb').is_file()
    assert (state / 'coordination.duckdb').is_file()
    assert not list(runtime.repository.glob('control*.duckdb'))
    verify_local_benchmark_admission(prepared['admission'], initial=False)
    assert owner.source.get_task(prepared['task_cid']) == original_task


@pytest.mark.parametrize('watchdog_window_seconds', [10, 310], ids=['within-startup-grace', 'after-startup-grace'])
def test_native_coding_timeout_keeps_authenticated_tree_and_closes_custody(tmp_path, monkeypatch, watchdog_window_seconds):
    monkeypatch.setattr(profile_authority, '_LIFECYCLE_REGISTRY_ROOT_OVERRIDE', tmp_path / 'account')
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', raising=False)
    prepared = _prepare_implementation_fixture(tmp_path / 'task')
    verified = verify_local_benchmark_admission(prepared['admission'])
    with open_existing_native_owner(
        database=Path(prepared['intent_database']), checkout=Path(prepared['repository']),
        state_dir=tmp_path / 'owner', repository_id=verified['manifest']['repository_cid'],
        execution_routes={prepared['task_id']: GROK_CODEX_EXECUTION_MODE},
    ) as owner:
        bundle = write_task_context_bundle(repository=Path(prepared['repository']), prepared=[{
            'schema': 'supervisor-task-context-preparation@1', 'task_cid': prepared['task_cid'],
            'task_id': prepared['task_id'], 'metadata': {},
        }], output=Path(prepared['repository']) / '.runtime/context.json')
        worktrees = tmp_path / 'worktrees'
        worktrees.mkdir(mode=0o750)
        runtime = AdmittedBenchmarkRuntime.create(
            tmp_path / 'launch', admission=prepared['admission'], server=owner.server,
            source=owner.source, context_bundle=bundle, worker_worktree_root=worktrees,
            implement=True, implementation_timeout_seconds=2, lifetime_seconds=600,
            implementation_command=shlex.join([sys.executable, '-B', '-c', 'import time; time.sleep(60)']),
        )
        try:
            assert runtime.start().succeeded
            original = runtime.bootstrap_receipts[0]
            deadline = time.monotonic() + 65
            timeout_seen = False
            while time.monotonic() < deadline:
                # Authored fixture only; no provider or task text exported.
                events = (runtime.state / 'run' / 'admitted_database_portal_attempts').glob('*/portal-events.jsonl')
                timeout_seen = any(json.loads(line).get('type') == 'implementation_timeout'
                    for path in events for line in path.read_text().splitlines() if line)
                assert not runtime.bootstrap_errors, runtime.startup_diagnostics()
                if timeout_seen:
                    break
                time.sleep(.1)
            assert timeout_seen, runtime.startup_diagnostics()
            # Allow watchdog/recovery transitions after the real timeout.
            deadline = time.monotonic() + watchdog_window_seconds
            while time.monotonic() < deadline:
                assert not runtime.bootstrap_errors, runtime.startup_diagnostics()
                tree = runtime.process.snapshot(runtime.profile)
                assert len(tree.roots) == 1, runtime.startup_diagnostics()
                time.sleep(.1)
            # No timeout path may synthesize successful task completion.
            assert owner.source.get_task(prepared['task_cid']).status != 'completed'
            # Watchdog maintenance must not create runtime DBs in task input.
            verify_local_benchmark_admission(prepared['admission'], initial=False)
            assert not list(Path(prepared['repository']).glob('control*.duckdb'))
            tree = runtime.process.snapshot(runtime.profile)
            child = next(item for item in tree.members if item.pid == runtime.bootstrap_receipts[-1]['pid'])
            runtime.process._signal_exact(child, signal.SIGTERM)
            deadline = time.monotonic() + 35
            while len(runtime.bootstrap_receipts) < 2 and not runtime.bootstrap_errors:
                assert time.monotonic() < deadline, runtime.startup_diagnostics()
                time.sleep(.1)
            assert not runtime.bootstrap_errors, runtime.startup_diagnostics()
            assert runtime.bootstrap_receipts[-1]['process_birth_id'] != original['process_birth_id']
            assert runtime.stop().succeeded
            assert not runtime.process.snapshot(runtime.profile).members
            assert all(child.poll() is not None for child in runtime._children)
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                runtime.stop()
            # Own only these locally launched fixture roots. This cleanup must
            # also run when a regression hides the root from the native tree.
            for child in runtime._children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=10)
            runtime.close()
