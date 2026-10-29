"""Whole native run_once regression: a real paid-style worker stays in flight."""
from pathlib import Path
import contextlib
import json
import shlex
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_canary as canary
from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_runtime as native
from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_provider as provider
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon


class LongNativeInflight(unittest.TestCase):
    def test_public_run_once_waits_for_one_real_provider_then_native_validation_merge_and_completion(self):
        framework = Path(native.__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix='wwl-long-native-reference-') as directory:
            reference = Path(directory)
            plan = reference / 'plan.json'
            native.atomic_json(plan, {'routing': {}})
            native.atomic_json(plan.with_name('watchdog.policy.json'), {'policy_id':'fixture', 'repair_scope': {'allowed_paths':[
                'ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search*.py',
                'test/api/test_worldwide_legal_search*.py']}})
            config = reference / 'reference.json'
            native.atomic_json(config, {'site_root':str(framework),'framework_root':str(framework),
                'state_root':str(reference/'state'),'plan_path':str(plan),'run_id':'long-native-reference','repair_command':[]})
            original_open, original_pass = native.CampaignRuntime.open, PortalImplementationDaemon.run_once
            owners, threads, evidence = [], [], {}
            gates = []
            worker_reports = []
            used = [False]
            def opened(runtime):
                original_open(runtime)
                owners.append(runtime)
            @contextlib.contextmanager
            def delayed_stub(spec_path, python):
                original = provider.implementation_command
                gate = spec_path.parent / 'native-inflight-canary-release.json'
                gates.append(gate)
                code = "import time,runpy,sys,pathlib; gate=pathlib.Path(" + repr(str(gate)) + "); end=time.monotonic()+90\nwhile not gate.is_file() and time.monotonic()<end: time.sleep(.05)\nif not gate.is_file(): raise SystemExit(124)\nsys.argv=" + repr([str(Path(canary.__file__).resolve()),'worker','--spec',str(spec_path)]) + "; runpy.run_path(sys.argv[0],run_name='__main__')"
                provider.implementation_command = lambda *_args, **_kwargs: shlex.join([python,'-I','-c',code])
                try:
                    yield
                finally:
                    provider.implementation_command = original
            def observe_real_inflight(observer):
                authority = observer._database_attempt_authority or {}
                if used[0] or not str(authority.get('task_id','')).startswith('WWL-REPAIR-'):
                    return original_pass(observer)
                used[0] = True
                owner = owners[-1]
                attempt = owner._daemon.get_attempt(authority['database_attempt_id'])
                intent = owner._daemon.provider_invocation_recorded(attempt.attempt_id,idempotency_key='provider:'+attempt.attempt_id)
                self.assertEqual(intent['callback_state'],'started_outcome_unknown')
                bridge = owner._bridge._for_attempt(attempt)
                paths = bridge._paths(attempt)
                worker = bridge.portal_factory(paths,attempt.task_alias)
                self.assertIs(type(worker),PortalImplementationDaemon)
                worker.bind_database_attempt_authority(**{key:authority[key] for key in ('task_id','database_task_cid','database_attempt_id','database_claim_id','database_attempt_number','database_binding_id')})
                before = (owner.config.state_root/'provider-activity-binding.json').read_bytes()
                failures=[]
                def work():
                    try:
                        finished = original_pass(worker)
                        outcome = finished.get('implementation_result') or {}
                        worker_reports.append({key:outcome.get(key) for key in ('returncode','reason','board_completion','validation_result','commit_result','merge_result')})
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        worker.close_event_runtime()
                thread=threading.Thread(target=work,name='isolated-real-native-provider',daemon=False)
                threads.append(thread)
                thread.start()
                until=time.monotonic()+90
                while time.monotonic()<until:
                    if observer._find_live_inflight_implementation():
                        break
                    if not thread.is_alive():
                        raise AssertionError('real controlled provider stopped before live observation: '+repr(failures))
                    time.sleep(.05)
                else:
                    raise AssertionError('real controlled provider never became observable')
                pending=original_pass(observer)
                observed=pending.get('implementation_result',{})
                self.assertEqual(observed.get('reason'),'inflight_process')
                self.assertIs(observed.get('dispatch_intent_retained'),True)
                self.assertIs(observed.get('provider_dispatched'),True)
                self.assertIs(observed.get('attempt_consumed'),True)
                self.assertEqual(owner._cached['stage_deadline_at'],attempt.started_at_ms/1000+owner.config.task_timeout_seconds)
                owner._bind_provider_activity(attempt)
                self.assertEqual((owner.config.state_root/'provider-activity-binding.json').read_bytes(),before)
                evidence.update(attempt_id=attempt.attempt_id, config=owner.config, intent=intent, deadline=owner._cached['stage_deadline_at'])
                native.atomic_json(gates[-1], {'release_after_verified_native_inflight': True})
                return pending
            try:
                with patch.object(native.CampaignRuntime,'open',opened), patch.object(PortalImplementationDaemon,'run_once',observe_real_inflight), patch.object(canary,'_stub_provider_command',delayed_stub):
                    try:
                        result=canary.run_canary(reference_config=config,parent_dir=reference)
                    except BaseException as error:
                        details=[{key:owner._last_failure.get(key) for key in ('error_class','reason','phase','traceback')} for owner in owners if owner._last_failure]
                        raise AssertionError('native fixture failure: '+json.dumps(details,sort_keys=True)+' WORKER_REPORTS='+json.dumps(worker_reports,sort_keys=True)) from error
            finally:
                for gate in gates:
                    if not gate.exists(): native.atomic_json(gate, {'cleanup_release': True})
                for thread in threads:
                    thread.join(timeout=30)
                    self.assertFalse(thread.is_alive())
            self.assertTrue(used[0])
            self.assertTrue(result['validated_promotion'])
            self.assertTrue(result['native_completion_receipt'])
            child=evidence['config']
            with native.owner_lock(child):
                daemon=DatabaseImplementationDaemon(database_path=child.database,coordination_path=child.state_root/'coordination.duckdb',
                    execution_path=child.state_root/'execution.duckdb',state_dir=child.state_root/'native',state_prefix='wwl',
                    owner_session_id=child.run_id,authority_mode='embedded_exclusive',require_real_execution=True).open()
                try:
                    attempt=daemon.get_attempt(evidence['attempt_id'])
                    phases=daemon.phase_history(attempt.attempt_id)
                    self.assertEqual(attempt.status,'succeeded')
                    self.assertEqual(attempt.attempt_number,1)
                    self.assertNotIn('failed',[phase['phase'] for phase in phases])
                finally:
                    daemon.close()
            events=[json.loads(line) for path in child.state_root.glob('native/*_database_portal_attempts/*/portal-events.jsonl') for line in path.read_text().splitlines()]
            self.assertEqual(sum(event.get('type')=='implementation_started' for event in events),1)
            self.assertEqual(sum(event.get('type')=='merge_finished' and event.get('merged') is True for event in events),1)
            self.assertEqual(sum(event.get('type')=='task_completed' for event in events),1)
