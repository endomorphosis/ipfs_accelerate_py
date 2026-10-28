"""Persistent equal-priority dispatch fairness; subprocess effects are authored mocks."""
from pathlib import Path
import copy,importlib.util,json,os,sys,tempfile,unittest
from unittest.mock import MagicMock,patch

SDK_ROOT=(Path(os.environ['SVF_FOUNDATION_REPOSITORY_ROOT'])/'external/ipfs_accelerate'
          if 'SVF_FOUNDATION_REPOSITORY_ROOT'in os.environ else Path(__file__).resolve().parents[2])
sys.path.insert(0,str(SDK_ROOT))
from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume as deployed
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import parse_todo_task_fields
CANDIDATE=os.environ.get('SVF_FAIRNESS_CANDIDATE')
def load_lane(suffix='unit'):
 if not CANDIDATE:return deployed
 name='ipfs_accelerate_py.agent_supervisor.rescue.fairness_'+suffix
 spec=importlib.util.spec_from_file_location(name,CANDIDATE);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
lane=load_lane()

class FairDispatchTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir='/dev/shm',prefix='svf-fair-dispatch-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  (self.root/'worker.py').write_text('# Authored producer boundary; never executed.\n')
  (self.root/'shared.py').write_text('VERSION = 0\n');self.board=self.root/'board.md';self.board.write_text('');self.starts=[]
 def add(self,identity,priority='P1',depends=''):
  with self.board.open('a')as f:f.write(f'## FAIR-{identity} Work\n- Status: ready\n- Depends on: {depends}\n- Priority: {priority}\n- Deterministic command: python3 worker.py {identity}\n- Execution inputs: worker.py, shared.py\n- Outputs: output-{identity}.json\n- Acceptance: actual result required\n\n')
 def patches(self,module=lane,fail_spawn=False):
  stack=__import__('contextlib').ExitStack();self.addCleanup(stack.close)
  def spawn(argv,**kw):
   if fail_spawn:raise PermissionError('authored refused spawn')
   self.starts.append(argv[-1]);process=MagicMock();process.pid=999999991;process.wait.return_value=1;return process
  popen=stack.enter_context(patch.object(module.subprocess,'Popen',side_effect=spawn));tree=stack.enter_context(patch.object(module,'CommandProcessTree'));stop=stack.enter_context(patch.object(module,'_stop_command_group'))
  return stack,popen,tree,stop
 def cycle(self,module=lane):
  return module.run_resume_cycle(repo_root=self.root,todo_path=self.board,task_header_prefix='## FAIR-',attempt_path=self.root/'attempts.json',log_path=self.root/'lane.log',heartbeat=lambda **kw:None)
 def journal(self):return json.loads((self.root/'attempts.json').read_bytes())
 def test_rearmed_earlier_source_never_starves_equal_peers_and_cursor_survives_restart(self):
  for identity in ['453','543','553']:self.add(identity)
  self.patches();lane._ATTEMPT_LIMIT=1;self.addCleanup(setattr,lane,'_ATTEMPT_LIMIT',3)
  for n in range(6):
   (self.root/'shared.py').write_text(f'VERSION = {n}\n')
   self.assertEqual(self.cycle()['phase'],'deterministic_command_failed')
  self.assertEqual(self.starts,['453','543','553','453','543','553'])
  saved=self.journal();fixed=saved['_validation_fixed_points'];self.assertEqual([fixed['FAIR-'+tid]['fair_dispatch_sequence']for tid in ['453','543','553']],[4,5,6])
  self.assertTrue(all(fields['status']=='ready'for _,fields in parse_todo_task_fields(self.board.read_text(),task_header_prefix='## FAIR-')))
  self.assertEqual({tid:saved['FAIR-'+tid]for tid in ['453','543','553']},{'453':1,'543':1,'553':1})
  restarted=load_lane('restart');self.patches(restarted)
  (self.root/'shared.py').write_text('VERSION = 7\n');self.cycle(restarted)
  self.assertEqual(self.starts[-1],'453');self.assertEqual(self.journal()['_validation_fixed_points']['FAIR-453']['fair_dispatch_sequence'],7)
 def test_priority_dependency_and_live_cooldown_still_decide_eligibility(self):
  self.add('001','P0');self.add('002','P1');self.add('003','P0','FAIR-002');self.patches()
  with patch.object(lane,'_ATTEMPT_LIMIT',1):
   self.cycle();self.cycle()
   self.assertEqual(self.starts,['001','002']);self.assertNotIn('003',self.starts)
   before=self.journal();result=self.cycle();self.assertEqual(result['phase'],'attempt_retry_wait');self.assertEqual(self.starts,['001','002'])
  after=self.journal();self.assertEqual(before['_retry_after_ms'],after['_retry_after_ms']);self.assertEqual(before['_validation_fixed_points'],after['_validation_fixed_points'])
 def test_equal_priority_order_changes_only_by_persistent_dispatch_and_is_pure(self):
  ids=['a','unknown','b','c'];blocks={'a':{'priority':'P1'},'b':{'priority':'P0'},'c':{'priority':'p1'},'unknown':{'priority':'1'}}
  fixed={'a':{'fair_dispatch_sequence':10,'execution_progress':{'contract':'new'}},'b':{'fair_dispatch_sequence':20},'c':{'fair_dispatch_sequence':3}}
  original=copy.deepcopy((ids,blocks,fixed));self.assertEqual(lane._order_explicit_task_priorities(ids,blocks,fixed_points=fixed),['b','unknown','c','a']);self.assertEqual((ids,blocks,fixed),original)
  self.assertEqual(lane._order_explicit_task_priorities(ids,blocks),['b','unknown','a','c'])
 def test_peek_hold_and_precommand_refusals_never_record_dispatch(self):
  self.add('001');self.patches();result=lane._order_explicit_task_priorities(['FAIR-001'],{'FAIR-001':{'priority':'P1'}});self.assertEqual(result,['FAIR-001']);self.assertFalse((self.root/'attempts.json').exists())
  (self.root/'HOLD').write_text('authored hold');self.assertEqual(self.cycle()['phase'],'operator_hold');self.assertEqual(self.starts,[]);(self.root/'HOLD').unlink()
  with patch.object(lane,'recovery_commands',side_effect=lane.RecoveryPolicyError('authored pre-command refusal')):self.cycle()
  self.assertEqual(self.starts,[]);self.assertNotIn('fair_dispatch_sequence',self.journal()['_validation_fixed_points']['FAIR-001'])
 def test_foreign_authority_refusal_before_command_admission_never_rotates(self):
  self.add('001');self.patches()
  def refuse(**kw):raise RuntimeError('foreign checkout authority refused')
  with self.assertRaisesRegex(RuntimeError,'foreign checkout'):
   lane.run_resume_cycle(repo_root=self.root,todo_path=self.board,task_header_prefix='## FAIR-',attempt_path=self.root/'attempts.json',log_path=self.root/'lane.log',heartbeat=refuse)
  self.assertEqual(self.starts,[])
  if(self.root/'attempts.json').exists():self.assertFalse(any('fair_dispatch_sequence'in row for row in self.journal().get('_validation_fixed_points',{}).values()))
 def test_failed_spawn_never_claims_a_dispatch(self):
  self.add('001');self.patches(fail_spawn=True)
  with self.assertRaises(PermissionError):self.cycle()
  self.assertNotIn('fair_dispatch_sequence',self.journal()['_validation_fixed_points']['FAIR-001'])
 def test_dispatch_persistence_failure_cleans_only_the_owned_spawn(self):
  self.add('001');_,_,tree,stop=self.patches();actual=lane._store_attempt_state
  def store(path,attempts,retry,fixed=None,evidence=None):
   if any('fair_dispatch_sequence'in row for row in(fixed or{}).values()):raise OSError('authored dispatch journal fsync failed')
   return actual(path,attempts,retry,fixed,evidence)
  with patch.object(lane,'_store_attempt_state',side_effect=store):
   with self.assertRaisesRegex(OSError,'fsync failed'):self.cycle()
  self.assertEqual(self.starts,['001']);stop.assert_called_once();tree.return_value.close.assert_called_once();self.assertNotIn('fair_dispatch_sequence',self.journal()['_validation_fixed_points']['FAIR-001'])
 def test_order_state_is_bounded_and_cannot_be_bool_negative_or_unbounded(self):
  for value in [True,-1,1.0,2**63,'2']:
   with self.subTest(value=value),self.assertRaises(ValueError):lane._order_explicit_task_priorities(['a'],{'a':{'priority':'P1'}},fixed_points={'a':{'fair_dispatch_sequence':value}})
  self.assertEqual(lane._ATTEMPT_LIMIT,3);self.assertEqual(lane._BOUNDED_ATTEMPT_LIMIT,2);self.assertEqual(lane.semantic_failure_limit({}),6)

if __name__=='__main__':unittest.main()
