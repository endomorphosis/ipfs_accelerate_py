"""SDK protocol cycles; actual Lift CLI when available, otherwise an explicit authored emitter.

The standalone SDK fixture proves only SDK protocol/budget behavior. Root
integration requires its real worker, and never substitutes this emitter.
"""
import contextlib,importlib.util,io,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
SDK_ROOT=(Path(os.environ['SVF_FOUNDATION_REPOSITORY_ROOT'])/'external/ipfs_accelerate'
          if 'SVF_FOUNDATION_REPOSITORY_ROOT'in os.environ else Path(__file__).resolve().parents[2])
LIFT_ROOT=(Path(os.environ['SVF_FOUNDATION_REPOSITORY_ROOT'])
           if 'SVF_FOUNDATION_REPOSITORY_ROOT'in os.environ else SDK_ROOT.parent.parent)
sys.path.insert(0,str(SDK_ROOT))
from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume as deployed
candidate=os.environ.get('SVF_OPERATOR_HOLD_CANDIDATE')
if candidate:
 name='ipfs_accelerate_py.agent_supervisor.rescue.operator_hold_candidate';spec=importlib.util.spec_from_file_location(name,candidate);lane=importlib.util.module_from_spec(spec);sys.modules[name]=lane;spec.loader.exec_module(lane)
else:lane=deployed
worker_path=Path(os.environ.get('SVF_HOLD_WORKER_CANDIDATE',str(LIFT_ROOT/'scripts/swissknife_visual_functional_worker.py')))
worker=None
if worker_path.is_file():
 sys.path.insert(0,str(LIFT_ROOT/'scripts'))
 name='svf_hold_cross_worker';spec=importlib.util.spec_from_file_location(name,worker_path);worker=importlib.util.module_from_spec(spec);sys.modules[name]=worker;spec.loader.exec_module(worker)
elif 'SVF_FOUNDATION_REPOSITORY_ROOT'in os.environ or 'SVF_HOLD_WORKER_CANDIDATE'in os.environ:
 raise RuntimeError('explicit Root integration requires the actual Lift worker')


def run_worker_or_sdk_fixture(root,task_id,stderr):
 if worker is not None:
  with patch.object(worker,'ROOT',root),contextlib.redirect_stderr(stderr):
   return worker.main(['--profile','profile.json','--task-id',task_id])
 # SDK-only authored protocol boundary: no Lift implementation or integration
 # claim. Tests still execute the real SDK runner/parser/budget/cooldown code.
 assert (root/'HOLD').is_file()
 print(json.dumps({'task_id':task_id,'error':{'code':'operator_hold'},'retryable':True,
                   'work_complete':False,'completion_authority':False}),file=stderr)
 return 75

class OperatorHoldTests(unittest.TestCase):
 def setUp(self):
  temporary=tempfile.TemporaryDirectory(dir='/dev/shm');self.addCleanup(temporary.cleanup);self.root=Path(temporary.name);self.task='SVF-HOLD-1';self.starts=[]
  (self.root/'worker.py').write_text('# Authored admission boundary; never a real subprocess.\n');(self.root/'app.py').write_text('VALUE = 1\n')
  (self.root/'board.md').write_text('## SVF-HOLD-1 Task\n- Status: ready\n- Priority: P1\n- Deterministic command: python3 worker.py\n- Execution inputs: worker.py, app.py\n- Outputs: result.json\n- Acceptance: actual evidence required\n\n')
  self.profile={'runtimeRoot':'data/agent_supervisor/swissknife_visual_functional','taskSpecPath':'tasks.json','appContractPath':'contracts.json'}
  (self.root/'profile.json').write_text(json.dumps(self.profile));(self.root/'tasks.json').write_text(json.dumps({'tasks':[{'task_id':self.task,'kind':'command','edit_paths':['app.py'],'checks':[['python3','app.py']]}]}))
  self.budget=self.root/self.profile['runtimeRoot']/'tasks'/self.task/'required-phase-repair-budget.json';self.budget.parent.mkdir(parents=True);self.budget.write_bytes(b'{"spent":6,"history":["existing charged work"]}\n');self.paid=self.budget.read_bytes()
 def cycle(self):return lane.run_resume_cycle(repo_root=self.root,todo_path=self.root/'board.md',task_header_prefix='## SVF-HOLD-',attempt_path=self.root/'attempts.json',log_path=self.root/'lane.log',heartbeat=lambda **kwargs:None)
 def journal(self):return json.loads((self.root/'attempts.json').read_bytes())
 def patches(self,kind='hold',keep_marker=False):
  stack=contextlib.ExitStack();self.addCleanup(stack.close)
  def spawn(argv,**kwargs):
   self.starts.append(list(argv));process=MagicMock();process.pid=999999991
   if kind=='hold':
    (self.root/'HOLD').touch();stderr=io.StringIO()
    code=run_worker_or_sdk_fixture(self.root,self.task,stderr)
    kwargs['stdout'].write(stderr.getvalue());kwargs['stdout'].flush()
    if not keep_marker:(self.root/'HOLD').unlink()
   else:
    code=1;kwargs['stdout'].write('SVF '+self.task+': RuntimeError: coordinator hold is active; no new model invocation or source apply\n');kwargs['stdout'].flush()
   process.wait.return_value=code;return process
  stack.enter_context(patch.object(lane.subprocess,'Popen',side_effect=spawn));stack.enter_context(patch.object(lane,'CommandProcessTree'));stack.enter_context(patch.object(lane,'_stop_command_group'))
  return stack
 def test_post_spawn_hold_protocol_is_retryable_resource_without_budget_or_semantic_charge(self):
  self.patches(keep_marker=True);result=self.cycle();saved=self.journal();record=saved['_validation_fixed_points'][self.task];progress=record['execution_progress']
  self.assertEqual(result['phase'],'operator_hold');self.assertEqual(saved.get(self.task,0),0);self.assertEqual(progress['semantic_failures'],0);self.assertEqual(progress['attempts_started'],1)
  self.assertEqual(progress['state'],'resource_wait');self.assertEqual(progress['resource_refusal'],'operator_hold');self.assertEqual(record['resource_wait']['code'],'operator_hold');self.assertEqual(record['fair_dispatch_sequence'],1);self.assertEqual(len(self.starts),1);self.assertEqual(self.budget.read_bytes(),self.paid)
  self.assertFalse(list(self.root.rglob('receipt.json')));self.assertIn('- Status: ready',(self.root/'board.md').read_text())
 def test_hold_cleared_before_parent_observes_exit_still_preserves_typed_refusal_and_cooldown(self):
  self.patches();result=self.cycle();saved=self.journal();self.assertEqual(result['phase'],'resource_retry_wait');self.assertEqual(saved.get(self.task,0),0);self.assertEqual(saved['_validation_fixed_points'][self.task]['resource_wait']['refusals'],1)
  self.cycle();self.assertEqual(len(self.starts),1);self.assertEqual(self.journal().get(self.task,0),0);self.assertEqual(self.budget.read_bytes(),self.paid)
 def test_real_error_with_identical_prose_still_spends_normal_attempt_and_failure(self):
  self.patches(kind='error');result=self.cycle();saved=self.journal();record=saved['_validation_fixed_points'][self.task]
  self.assertEqual(result['phase'],'deterministic_command_failed');self.assertEqual(saved[self.task],1);self.assertEqual(record['execution_progress']['semantic_failures'],1);self.assertNotIn('resource_wait',record);self.assertEqual(record['fair_dispatch_sequence'],1)
 def test_previous_real_failure_and_paid_history_are_not_refunded_by_later_hold(self):
  first=self.patches(kind='error');self.cycle();first.close();before=self.journal();prior=before['_validation_fixed_points'][self.task]['execution_progress'];self.patches();self.cycle();after=self.journal();record=after['_validation_fixed_points'][self.task]
  self.assertEqual(after[self.task],before[self.task]);self.assertEqual(record['execution_progress']['semantic_failures'],prior['semantic_failures']);self.assertEqual(record['execution_progress']['first_failure_at_ms'],prior['first_failure_at_ms']);self.assertEqual(record['execution_progress']['attempts_started'],2);self.assertEqual(record['fair_dispatch_sequence'],2);self.assertEqual(self.budget.read_bytes(),self.paid)
 def test_explicit_root_context_must_use_actual_worker_and_standalone_may_use_only_labelled_fixture(self):
  if 'SVF_FOUNDATION_REPOSITORY_ROOT'in os.environ or 'SVF_HOLD_WORKER_CANDIDATE'in os.environ:self.assertIsNotNone(worker)
  else:self.assertEqual(Path(deployed.__file__).resolve().is_relative_to(SDK_ROOT.resolve()),True)
 def test_only_exact_task_bound_closed_operator_protocol_is_recognized(self):
  correct={'task_id':self.task,'error':{'code':'operator_hold'},'retryable':True,'work_complete':False,'completion_authority':False};log=self.root/'protocol.log'
  def read(value):log.write_text(json.dumps(value)+'\n');return lane._transient_resource_refusal(log,0,self.task)
  self.assertEqual(read(correct),'operator_hold')
  for changed in [{**correct,'retryable':False},{**correct,'retryable':1},{**correct,'work_complete':True},{**correct,'completion_authority':True},{**correct,'task_id':'SVF-FOREIGN'},{**correct,'extra':'not admitted'},{**correct,'error':{'code':'operator_hold','message':'unbounded prose'}}]:
   with self.subTest(changed=changed):self.assertEqual(read(changed),'')
 def test_protocol_does_not_read_before_this_producer_log_offset(self):
  log=self.root/'protocol.log';prior={'task_id':self.task,'error':{'code':'operator_hold'},'retryable':True,'work_complete':False,'completion_authority':False};log.write_text(json.dumps(prior)+'\n');start=log.stat().st_size
  with log.open('a')as out:out.write('actual new error without typed admission refusal\n')
  self.assertEqual(lane._transient_resource_refusal(log,start,self.task),'')

if __name__=='__main__':unittest.main()
