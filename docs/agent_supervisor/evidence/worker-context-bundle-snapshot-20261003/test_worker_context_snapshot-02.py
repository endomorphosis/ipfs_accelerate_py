"""Per-construction nomination coherence on the imported native daemon.

The bundle parser and context compiler are real. Native Source384 validation and
individual artifact readers are controlled seams; no model/provider executes.
"""
from types import SimpleNamespace
import json
import subprocess
import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask,TodoImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_context_runtime as semantic
from ipfs_accelerate_py.agent_supervisor.runtime import code_retrieval_context as retrieval
from ipfs_accelerate_py.agent_supervisor.semantic_state import intent_world_snapshot as world

@pytest.fixture
def context(tmp_path,monkeypatch):
 root=tmp_path/'repo';root.mkdir();(root/'todo.md').write_text('# Tasks\n')
 for args in [('init','-q'),('add','.'),('-c','user.name=Context','-c','user.email=local@example.invalid','commit','-qm','seed')]:
  subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True)
 daemon=TodoImplementationDaemon(todo_path=root/'todo.md',repo_root=root,
  state_path=tmp_path/'state/tasks.json',strategy_path=tmp_path/'state/strategy.json',
  events_path=tmp_path/'state/events.jsonl',task_header_prefix='## CTX-')
 task=PortalTask(task_id='CTX-001',title='Context snapshot',status='ready',completion='manual',priority='P1',
  track='context',outputs=['result.json'],validation=['true'],acceptance='Current evidence',
  canonical_task_cid='task:ctx1',metadata={'Provider role':'deterministic-only'})
 metadata={'semantic context artifact':'semantic.json','semantic context sha256':'s'*64,'semantic context refresh':'true',
  'code retrieval artifact':'retrieval.json','code retrieval sha256':'r'*64,
  'world context artifact':'world.json','world context sha256':'w'*64,'world context repository':'repo:1'}
 prepared={'schema':'supervisor-task-context-preparation@1','task_cid':task.canonical_task_cid,'task_id':task.task_id,
  'metadata':metadata,'source384_context':{'schema':'terminal-source384-repository-context@1'}}
 daemon._task_context_nomination_bundle=bundles.write_task_context_bundle(repository=root,prepared=[prepared],output=root/'bundle.json')
 validations=[];reads=[]
 monkeypatch.setattr(source384,'validate_source384_context',lambda **kw:validations.append(kw))
 monkeypatch.setattr(semantic,'load_semantic_worker_context',lambda **kw:reads.append(('semantic',kw)) or json.dumps({'semantic_root_cid':'root:new'}))
 monkeypatch.setattr(retrieval,'load_code_retrieval_context',lambda **kw:reads.append(('retrieval',kw)) or '{}')
 monkeypatch.setattr(world,'load_intent_world_context',lambda **kw:reads.append(('world',kw)) or
  {'schema':'intent-world@1','semantic_root_cid':'root:new','component_status':{'symbol_root':'current'},
   'execution_authority':False,'completion_authority':False})
 monkeypatch.setattr(world,'minify_intent_world_worker_context',lambda value:value)
 return SimpleNamespace(daemon=daemon,task=task,root=root,reads=reads,validations=validations,metadata=metadata)

def test_initial_compilation_validates_bundle_once_and_preserves_artifact_order(context):
 c=context;result=c.daemon._compile_implementation_context(c.task,attempt=1)
 assert len(c.validations)==1
 assert [name for name,_ in c.reads]==['semantic','retrieval','world']
 assert [item.kind for item in result.capsule.evidence if item.kind.endswith('context')]==[
  'semantic-context','code-retrieval-context','intent-world-context']
 calls=dict(c.reads)
 assert calls['semantic']['expected_sha256']=='s'*64
 assert calls['retrieval']['expected_sha256']=='r'*64
 assert calls['world']['expected_sha256']=='w'*64 and calls['world']['repository_id']=='repo:1'
 assert c.task.metadata=={'Provider role':'deterministic-only'}

def test_next_actual_compilation_revalidates_stale_source_before_artifact_loads(context,monkeypatch):
 c=context;c.daemon._compile_implementation_context(c.task,attempt=1)
 error=ValueError('Source384 source changed')
 def stale(**kw):raise error
 monkeypatch.setattr(source384,'validate_source384_context',stale)
 with pytest.raises(ValueError) as got:c.daemon._compile_implementation_context(c.task,attempt=2)
 assert got.value is error and len(c.reads)==3

def test_retry_compilation_validates_one_fresh_nomination_in_original_order(context,monkeypatch):
 c=context;base=c.daemon._compile_implementation_context(c.task,attempt=1)
 monkeypatch.setattr(c.daemon,'_implementation_parent',lambda task:(base.capsule,'decision:prior'))
 monkeypatch.setattr(c.daemon,'_implementation_cancel_requested',lambda:False)
 # Stop after real retry artifact construction, before unrelated diagnostic/compiler work.
 class ReachedDiagnostic(Exception):pass
 def reached():raise ReachedDiagnostic
 c.reads.clear()
 with pytest.raises(ReachedDiagnostic):c.daemon._compile_implementation_retry_context(c.task,2,SimpleNamespace(to_record=reached))
 assert len(c.validations)==2
 assert [name for name,_ in c.reads]==['semantic','world','retrieval']

def test_conflicting_task_metadata_refuses_complete_compilation(context):
 c=context;c.task.metadata['world_context_sha256']='different'
 with pytest.raises(ValueError,match='conflicts'):c.daemon._compile_implementation_context(c.task,attempt=1)
 assert len(c.validations)==1 and c.reads==[]

def test_actual_snapshot_is_detached_read_only(context):
 c=context;snapshot=c.daemon._task_metadata_snapshot(c.task,include_context=True)
 with pytest.raises(TypeError):snapshot['world context sha256']='forged'
 c.task.metadata['Provider role']='changed'
 assert snapshot['provider role']=='deterministic-only'
 assert c.daemon._task_metadata_snapshot(c.task)['provider role']=='changed'

@pytest.mark.parametrize('kind',['semantic','retrieval','world'])
def test_each_direct_consumer_still_loads_live_bundle(context,kind):
 c=context
 if kind=='semantic':c.daemon._semantic_implementation_references(c.task,1,repository_id='r',tree_id='t')
 elif kind=='retrieval':c.daemon._code_retrieval_implementation_references(c.task,repository_id='r',tree_id='t')
 else:c.daemon._world_implementation_references(c.task,repository_id='r',tree_id='t')
 assert len(c.validations)==1 and c.reads[0][0]==kind
