"""Replay retained native receipt shape after two explicit filesystem relocations.

No native inference or current-source validity is asserted: that heavy validator
is replaced by a recording sentinel. The actual bounded byte reader and bundle
writer/historical reader run unchanged, on authored local task identities.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,os,sys,tempfile,types
B=Path(__file__).resolve().parent; W=B.parents[1]
A=W/'.worktrees/ir-release-accelerate-20261002'
report=W/'artifacts/source384-initial-context-lifetime-20261003/trial-02/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__aaqEeLD/agent/supervisor-result.json'
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as owner
from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as publication
raw=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sha=lambda v:hashlib.sha256(v).hexdigest()
original=json.loads(report.read_bytes())['initial_context']['source384_context']
original_bytes=raw(original)
assert len(original_bytes)==78563
assert len(original_bytes)>32768
selected_module='ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context'
old=sys.modules.get(selected_module)
recording=types.ModuleType(selected_module);calls=[]
recording.validate_source384_context=lambda **kw:calls.append(deepcopy(kw))
try:
 sys.modules[selected_module]=recording
 with tempfile.TemporaryDirectory(prefix='retained-transport-',dir=B) as td:
  base=Path(td); root=base/'repository'; root.mkdir(); output=base/'private'; output.mkdir()
  receipt=deepcopy(original);receipt.update(repository=str(root),output=str(output))
  assert {k for k in original if original[k]!=receipt[k]}=={'repository','output'}
  (output/'receipt.json').write_bytes(raw(receipt))
  prepared=[dict(schema='supervisor-task-context-preparation@1',task_cid=f'authored:{i}',task_id=f'TEST-{i}',metadata={},source384_context=receipt) for i in range(16)]
  bundle=owner.write_task_context_bundle(repository=root,prepared=prepared,output=root/'bundle.json')
  bundle_bytes=(root/bundle['artifact']).read_bytes()
  assert len(bundle_bytes)<16384
  args=dict(repository=root,artifact=bundle['artifact'],expected_sha256=bundle['sha256'],task_cid='authored:15',task_id='TEST-15')
  historic=owner.read_task_context_historical_selection(**args)
  assert historic['source384_context']==receipt and not calls
  selected=owner.load_task_context_selection(**args)
  assert selected==historic and calls==[dict(repository=root,expected_receipt=receipt)]
  observation=publication._source384_successor_unavailable({'task_cid':'authored:15','task_id':'TEST-15'},bundle,historic['source384_context'])
  assert observation['source384_context']==receipt and observation['source384_current'] is False
  assert observation['status']=='successor_unavailable' and observation['needs_successor'] is True
  assert observation['completion_authority'] is observation['proof_authority'] is False
  (output/'receipt.json').write_bytes(raw(dict(receipt,seconds=receipt['seconds']+1)))
  try: owner.read_task_context_historical_selection(**args)
  except ValueError as exc: assert 'receipt bytes differ' in str(exc)
  else: raise AssertionError('mutated receipt accepted')
  result=dict(status='passed',scope=__doc__,source_report=str(report),source_report_sha256=sha(report.read_bytes()),
   original_receipt_bytes=len(original_bytes),original_receipt_sha256=sha(original_bytes),
   relocated_fields=['repository','output'],relocated_receipt_bytes=len(raw(receipt)),relocated_receipt_sha256=sha(raw(receipt)),
   bundle_schema=owner.SOURCE384_REFERENCE_SCHEMA,task_count=16,bundle_bytes=len(bundle_bytes),bundle_sha256=sha(bundle_bytes),
   source_inventory_rows=len(original['source_inventory']),source_hashes=len(original['source_hashes']),
   all_other_native_receipt_fields_unchanged=True,full_receipt_forwarded_exactly=True,historical_byte_mutation_refused=True,
   historical_currentness_checked=False,recording_validator_calls=len(calls),neural_inference_calls=0,provider_calls=0,
   native_publication_owner_gate_exercised=False,successor_available=False,
   source_pins={str(p.relative_to(A)):sha(p.read_bytes()) for p in [Path(owner.__file__),Path(publication.__file__)]},
   script_sha256=sha(Path(__file__).read_bytes()))
  (B/'retained-native-transport.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:result[k] for k in ('status','original_receipt_bytes','relocated_receipt_bytes','bundle_bytes','task_count')}))
finally:
 if old is None:sys.modules.pop(selected_module,None)
 else:sys.modules[selected_module]=old
