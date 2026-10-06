"""Closed diagnostics from signed native START through cleanup; no model calls."""
from pathlib import Path
import json
import shlex
import sys
import time

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from benchmarks.agent_supervisor.container_coding.terminal_failure_observation import collect
from benchmarks.agent_supervisor.container_coding.terminal_native_progress import NativeProgress
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from test.integration.test_admitted_benchmark_runtime import _prepare_implementation_fixture


def test_signed_native_preflight_failure_remains_observable_after_stop_and_close(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_authority, '_LIFECYCLE_REGISTRY_ROOT_OVERRIDE', tmp_path / 'account')
    monkeypatch.setenv('IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR', str(tmp_path / 'ambient'))
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', raising=False)
    prepared = _prepare_implementation_fixture(tmp_path / 'task')
    verified = verify_local_benchmark_admission(prepared['admission'])
    repository = Path(prepared['repository'])
    script = tmp_path / 'authored_preflight_failure.py'
    source = str(Path(__file__).resolve().parents[2])
    script.write_text('import sys\nsys.path.insert(0,' + repr(source) + ')\n'
        'from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner\n'
        "sys.argv=['runner','--provider','grok_cli','--model','wrong-model']\n"
        'raise SystemExit(runner.main())\n')
    state = tmp_path / 'outer'
    state.mkdir()
    progress = NativeProgress(started=time.monotonic())
    report = {'native_progress': progress.report}
    with open_existing_native_owner(database=Path(prepared['intent_database']), checkout=repository,
            state_dir=tmp_path / 'owner', repository_id=verified['manifest']['repository_cid'],
            execution_routes={prepared['task_id']: GROK_CODEX_EXECUTION_MODE}) as owner:
        bundle = write_task_context_bundle(repository=repository, prepared=[{
            'schema': 'supervisor-task-context-preparation@1', 'task_cid': prepared['task_cid'],
            'task_id': prepared['task_id'], 'metadata': {},
        }], output=repository / '.runtime/context.json')
        runtime = AdmittedBenchmarkRuntime.create(state / 'launch', admission=prepared['admission'],
            server=owner.server, source=owner.source, context_bundle=bundle, implement=True,
            implementation_command=shlex.join([sys.executable, '-B', str(script)]),
            implementation_timeout_seconds=20, max_task_attempts=1)
        native_state = runtime.state
        assert native_state == state / 'launch/state'
        try:
            report['start'] = runtime.start().to_dict()
            assert report['start']['status'] == 'succeeded'
            progress.begin_post_start()
            deadline = time.monotonic() + 90
            while True:
                task = owner.source.get_task(prepared['task_cid'])
                report['task_state'] = {'task_cid': task.task_cid, 'status': task.status, 'revision': task.revision}
                progress.sample(task=task, task_cid=task.task_cid, state=native_state, now=time.monotonic())
                if task.status in {'failed', 'blocked', 'completed', 'cancelled'}:
                    break
                assert time.monotonic() < deadline
                time.sleep(.25)
            assert task.status == 'blocked'
        finally:
            try:
                report['stop'] = runtime.stop().to_dict()
                assert report['stop']['status'] == 'succeeded'
                assert not runtime.process.snapshot(runtime.profile).members
            finally:
                runtime.close()
        progress.finish(report)
        result = collect(state, report, native_state=native_state)
        assert result['bridge']['status'] == 'observed'
        assert result['bridge']['scope'] == 'exact_admitted_task_and_attempt'
        diagnostic = result['bridge']['diagnostic']
        assert diagnostic['phase'] == 'terminal_failure'
        assert diagnostic['reason_code'] == 'portal_provider_failed'
        assert diagnostic['callback']['state'] == 'failed_outcome_settled'
        child = diagnostic['child_reported_router_failure']
        assert child['status'] == 'observed'
        assert child['scope'] == 'bounded_native_log_tail'
        assert child['diagnostic']['diagnostic']['phase'] == 'argument_validation'
        assert child['diagnostic']['error_type'] == 'ValueError'
        assert result['planner_child']['status'] == 'missing'
        assert result['bridge']['provider_dispatch_observed'] is None
        assert result['completion_authority'] is result['retry_authority'] is result['settlement_authority'] is False
        assert task.task_cid not in json.dumps(result) and str(tmp_path) not in json.dumps(result)
        logs = list((native_state / 'run/admitted_database_portal_attempts').rglob('*.log'))
        assert not any('router-implementation-invocation@1' in path.read_text() for path in logs)
        assert collect(state, report)['bridge']['status'] == 'unavailable'
