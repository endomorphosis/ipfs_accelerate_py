"""Read-only independent receipt checks; no benchmark or native model execution."""
import hashlib
import json
from pathlib import Path

B=Path(__file__).resolve().parent
W=B.parents[1]
def load(name): return json.loads((B/name).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pin(path): return {'bytes':path.stat().st_size,'sha256':sha(path)}
def save(name,value):
 p=B/name
 if p.exists(): raise ValueError('fresh audit output required')
 p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

review=load('independent-review.json')
for name,digest in review['sha256'].items(): assert sha(B/name)==digest, name
controls=load('controls-02.json')
assert controls['returncode']==0
assert 'Ran 25 tests' in (B/'controls-02.log').read_text()
for name,digest in controls['source_sha256'].items(): assert sha(B/name)==digest,name
assert load('before-pins.json')==load('after-pins.json')
for path,digest in load('before-pins.json').items(): assert sha(Path(path))==digest,path
for name,digest in load('verdict.json')['files'].items(): assert sha(B/name)==digest,name

outcome=load('source-free-outcome.json')
assert outcome['process']['returncode']==1 and outcome['raw_qualifier_passed'] is False
assert not (B/'docker-01/native-inference.json').exists()
assert outcome['inference']['verified'] is False
assert outcome['inference']['model_loads'] is None and outcome['inference']['coverage'] is None
assert outcome['provider_calls_reported']==0 and outcome['official_verifier_executed'] is False
assert outcome['actual_cgroups']['cpu_max']=='500000 100000'
assert outcome['actual_cgroups']['memory_max']=='12884901888'
assert outcome['cleanup']['no_matching_containers'] is True
assert outcome['input_preservation']['original_count']==218
assert outcome['input_preservation']['original_inputs_equal_across_deployment'] is True

archive=load('cache-advice.json');binary=load('codex-cache-advice.json');library=load('library-cache-advice.json')
for row,count,bytes_ in [(archive,26905,3914232094),(binary,4,628736528),(library,7,414442268)]:
 assert row['selected_files']==row['advised_files']==count
 assert row['selected_bytes']==row['advised_bytes']==bytes_
 assert row['errors']==[] and row['metadata_unchanged'] is True and row['body_writes']==0
 assert row['freed_bytes_claimed'] is False and row['global_drop_caches'] is False
assert binary['post_boundary_receipt_sha256']==sha(B/'docker-01/worker-boundary/installation/native-codex-binary.log')
expected_binary={'codex':('0059c73b149a1433b634e26ad8a715e02717a9920665a9cc5676941db03c45cd',247446792),
 'codex-code-mode-host':('68237b34d0bc182e99c43ca2197a25120339c154dd3fc29f908c2a1b022efa5b',66921472)}
assert len({r['path'] for r in binary['files']})==4
for r in binary['files']: assert (r['sha256'],r['bytes'])==expected_binary[r['name']]
source=load('finite-library-source-pins.json')
assert library['source_pins_sha256']==sha(B/'finite-library-source-pins.json')
expected={Path(r['path']).name:r for r in source['extensions']}
expected.update({Path(r['wheel_member']).name:r for r in source['torch']})
assert len(expected)==len(library['files'])==7
for row in library['files']:
 e=expected[row['name']]
 assert (row['sha256'],row['bytes'],row['mode'])==(e['sha256'],e['bytes'],e['mode'])
 assert row['bytes']==row['expected_bytes']
assert load('source-pin-reproduction.json')['output_sha256']==library['source_pins_sha256']

trace=(B/'docker-01/source384-probe.stdout').read_text()
for marker in ['codebase_source_units_384.py", line 248','codebase_ir.py", line 712','parent_lease.acquire_child','LeaseTimeoutError']:
 assert marker in trace,marker
D=W/'.worktrees/ir-release-datasets-20261002'
A=W/'.worktrees/ir-release-accelerate-20261002'
units=D/'ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py'
owner=D/'ipfs_datasets_py/logic/software_contracts/codebase_ir.py'
consumer=A/'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py'
assert sha(units)=='1ab971c23d2b8982d4e1c8af74b24cc10da704ec29fa260777836d7be1db68f6'
assert sha(owner)=='2cc174e71f17a23daabd1c8855d96c621811addfe5152f386810ec23918b1463'
assert sha(consumer)=='5e7da432a206b07900822044326dde963b5bf8cb9c1f39d313e1a6ceed56e66f'
lines=units.read_text().splitlines()
assert '_worker(payload' in lines[246]
assert '_validate_output(output,key,preparation);observe()' in lines[247]
assert '_stage(registry,saved)' in lines[252]
snapshot=load('post-context-scheduler.json')
assert snapshot['observation']=='after_context_unwind'
assert snapshot['snapshot']['proof_backoff']['reason']=='proof_memory_headroom'
assert all(snapshot['snapshot'][key]==0 for key in ('active_lease_count','active_child_lease_count','active_root_lease_count','waiting_request_count'))
assert outcome['failure_resources']['available_memory_mb']==8637

save('independent-outcome-audit.json',{
 'schema':'independent-seven-library-diagnostic-outcome-audit@1','reviewer':'peer_recovery_fresh',
 'decision':'receipt_and_control_flow_claims_verified_failure_retained',
 'outcome_sha256':sha(B/'source-free-outcome.json'),'verdict_sha256':sha(B/'verdict.json'),
 'helper_review_sha256':sha(B/'independent-review.json'),'focused_controls':25,
 'reviewed_source_and_controls_hashes_equal':True,'before_after_live_source_pins_equal':True,
 'actual_resource_profile':'5 CPU /12288 MiB','input_preservation':'218 originals exactly equal across deployment',
 'advice_scopes':{'archive':{'files':26905,'bytes':3914232094},'public_binary_copies':{'files':4,'bytes':628736528},
  'public_native_libraries':{'files':7,'bytes':414442268}},
 'independent_library_pin_comparison':'Seven installed row hashes/sizes/modes equal reviewed archive-extension and CPU-wheel-member metadata. No wheel payload rehash by this reviewer.',
 'root_admitted_and_initial_index_returned':'Inferred from frozen consumer control flow and exact later native traceback.',
 'worker_return_and_closed_output_validation':'Inferred from units384.py line248 observe frame after line247 _worker return and _validate_output on line248; not a persisted inference receipt.',
 'failure_boundary':{'name':'post-worker source observation child resource lease','unit_owner_line':248,'index_owner_line':712,
  'unit_owner_sha256':sha(units),'index_owner_sha256':sha(owner),'consumer_sha256':sha(consumer)},
 'durable_inference_publication_reached':False,'model_loads':None,'coverage':None,
 'post_unwind_scheduler_reason':'proof_memory_headroom','post_unwind_leases_and_waiters':0,
 'post_unwind_available_memory_mib':8637,
 'admission_evidence_limit':'The retained backoff reason and later resource sample do not capture the exact child admission decision sample or its contemporaneous scheduler state.',
 'advice_bytes_are_freed_bytes':False,'cleanup_absent':True,'provider_calls':0,'official_verifier_executed':False,
 'benchmark_result':False,'production_qualification_claimed':False,'production_cache_policy_changed':False,
 'trace_descriptor':{'path':'docker-01/source384-probe.stdout',**pin(B/'docker-01/source384-probe.stdout')},
 'payloads_copied':False,'tests_models_or_docker_run_by_reviewer':False})
print(json.dumps({'audit_sha256':sha(B/'independent-outcome-audit.json'),'outcome_sha256':sha(B/'source-free-outcome.json')}))
