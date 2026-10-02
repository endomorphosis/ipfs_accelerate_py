"""Read-only historical custody/signature audit; no checker or daemon execution."""
from pathlib import Path
import hashlib,json,subprocess,datetime,math
from ipfs_accelerate_py.agent_supervisor.runtime import repository_finite_public_context as public
from ipfs_accelerate_py.agent_supervisor.runtime import repository_behavioral_admission as behavioral
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
BASE=Path('/home/barberb/lift_coding/artifacts/repository-behavioral-supervision-20261002')
REPOS={'datasets':Path('/tmp/ir-release-datasets-20261001'),'accelerate':Path('/tmp/ir-release-accelerate-20261001')}
def sha(b):return hashlib.sha256(b).hexdigest()
r=json.loads((BASE/'managed-native-03/result.json').read_text());failed=json.loads((BASE/'managed-native-02/result.json').read_text())
assert r['qualified'] and not failed['qualified'] and failed['error']['message']=='preparation output owner is outside the exact named disk roots'
assert r['task']=={'revision':4,'status':'completed'} and r['start']['status']==r['stop']['status']=='succeeded'
assert r['remaining_processes']==0 and r['owned_remaining_leases']==[]
assert r['provider_calls']==r['provider_tokens']==r['training_steps']==0 and r['learned_interpretation'] is False
assert r['initial_public_check_exit_code']==1 and r['final_public_check_exit_code']==0
for k in ['checked_cache_replay_agrees','cold_semantic_manifest_agrees','cold_successor_agrees','published_source_matches_checked_candidate','stale_checked_cache_refused','stale_dispatch_refused','stale_index_refused','behavioral_worker_fence_replayed','delegated_worker_resources_verified']:assert r[k] is True
pin_checks=[]
for name in ['managed-native-02','managed-native-03']:
 prov=json.loads((BASE/(name+'-producers.json')).read_text())
 for row in prov['files']:
  if name.endswith('02'):
   raw=subprocess.check_output(['git','-C',str(REPOS[row['repository']]),'show',prov['heads'][row['repository']]+':'+row['path']]);current=None
  else:
   raw=(BASE/(name+'-sources')/row['repository']/(row['path']+'.txt')).read_bytes()
   current=sha((REPOS[row['repository']]/row['path']).read_bytes())
   assert current==row['sha256']
  assert sha(raw)==row['sha256']
  pin_checks.append(dict(run=name,**row,exact_snapshot_or_commit_verified=True,current_bytes_match=current==row['sha256'] if current else None))
root=BASE/'managed-native-03';signed_files=[];loaded_fence=None;pub=None
for path in sorted((root/'repository/.runtime/repository-finite-handoffs').glob('*.json')):
 data=path.read_bytes();assert sha(data)==path.stem
 envelope=json.loads(data);binding=envelope['binding'];payload=envelope['payload']
 assert binding['identity']==r['start']['caller']==r['stop']['caller']
 public.local.verify_did_key_signature(identity_did=binding['identity'],payload=payload,signature=binding['signature'])
 if payload['schema']==public.SCHEMA:
  pub=public.load_finite_public_context(artifact=path,expected_sha256=path.stem,owner_did=binding['identity'],profile_id=binding['profile_id'],task_cid=r['task_cid'],handoff_sha256=r['handoff_sha256'])
 elif payload['schema']==behavioral.SCHEMA:
  loaded_fence=behavioral._load(artifact=path,expected_sha256=path.stem,owner_did=binding['identity'],profile_id=binding['profile_id'],task_cid=r['task_cid'])
 signed_files.append(dict(path=str(path.relative_to(root)),sha256=path.stem,schema=payload['schema'],signature_verified=True,owner_equals_start_stop_caller=True))
assert len(signed_files)==3 and pub and loaded_fence
assert pub['checker_execution_performed'] is False and pub['owner_keys_used'] is False
assert loaded_fence['native_task_population']==[r['task_cid']]
assert loaded_fence['source_head']==r['initial_head']
assert r['public_context_worker_receipt']['task_cid']==r['task_cid']
assert sha((root/'repository'/r['context_bundle']['artifact']).read_bytes())==r['context_bundle']['sha256']
legacy=loaded_fence['legacy_signed_evidence']['payload'];edit=legacy['edit']
path=edit['path']
before=subprocess.check_output(['git','-C',str(root/'repository'),'show',r['baseline_commit']+':'+path])
after=subprocess.check_output(['git','-C',str(root/'repository'),'show',r['published_commit']+':'+path])
assert sha(before)==edit['before_sha256'] and sha(after)==edit['after_sha256']
assert (root/'repository'/path).read_bytes()==after
pipe=r['pipeline_resource_receipt'];host=r['full_trial_resource_receipt'];worker=r['public_context_worker_receipt']['delegated_resource_phase']
assert pipe['closed'] and host['closed'] and pipe['root']==host
assert pipe['active_phases']==pipe['queued_phases']==pipe['retained_host_phase_count']==pipe['retained_payload_bytes']==pipe['queued_payload_bytes']==0
assert not pipe['retained_disk_reservations'] and pipe['lifecycle']['reserved']==pipe['lifecycle']['pending_reap']==0
assert host['owned_active_root_count']==0 and host['native_parent']['released']
root_id=host['host_authority']['root_lease_id'];assert root_id==worker['root_lease_id']
phase=[e for e in host['events'] if e['child_lease_id']==worker['bridge_phase_lease_id']];assert len(phase)==1
assert phase[0]['parent_lease_id']==root_id and phase[0]['demand']['phase']=='validation' and phase[0]['status']=='completed'
protected=[e for e in pipe['events'] if e['disk']['reservation_id']==worker['daemon_reservation_id']];assert len(protected)==1
protected=protected[0];lease=protected['disk']['resource_lease']
assert protected['protected'] and protected['phase']=='validation' and protected['status']=='completed'
assert lease['lease_id']==worker['parent_lease_id'] and lease['parent_lease_id']==worker['bridge_phase_lease_id'] and lease['released']
assert worker['lease']['parent_lease_id']==lease['lease_id'] and worker['lease']['released'] and worker['private_token_disclosed'] is False
assert worker['task_cid']==r['task_cid']
# Private grant is read only for exact custody/ID joins. No token/secret body is exported.
grants=list((root/'private-resources').glob('*.json'));assert len(grants)==1
grant_raw=grants[0].read_bytes();assert sha(grant_raw)==grants[0].stem
grant=json.loads(grant_raw);assert grant['schema']==worker['grant_schema']
for key in ['root_lease_id','bridge_phase_lease_id','daemon_reservation_id','task_cid','repository_id']:assert grant[key]==worker[key]
assert grant['parent_token']['lease_id']==worker['parent_lease_id'] and grant['phase']==worker['demand']
assert all(e['status']=='completed' and e['disk']['status']=='released' for e in pipe['events'])
assert all(e['protected']==(e['phase'] in ['validation','cleanup']) for e in pipe['events'])
phases=r['trial_phases'];assert phases[0]['start_seconds']==0 and all(p['status']=='completed' for p in phases)
assert all(a['end_seconds']==b['start_seconds'] for a,b in zip(phases,phases[1:])) and phases[-1]['end_seconds']==r['seconds']
cas_checks=[]
for name in ['artifacts','cold-artifacts']:
 cas=ImmutableCAS(root/name);count=0
 for family in ['source','structured']:
  for p in (root/name/family).glob('*/*'):
   assert p.is_file() and not p.is_symlink()
   (cas.get_bytes if family=='source' else cas.get)(p.name);count+=1
 cas_checks.append(dict(path=name,verified_objects=count))
audit=dict(schema='repository-managed-native-independent-audit/v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scope='Historical signature/hash/identity reconstruction; no new checker, worker, daemon, native DB mutation, or live lease admission.',
 historical_failure=dict(seconds=failed['seconds'],error=failed['error'],baseline_commit='8a37a3a81b99be4cc1f8df1143f308a3b7d1875e'),
 successful_run=dict(seconds=r['seconds'],task=r['task'],provider_calls=0,provider_tokens=0,remaining_owned_processes=0,remaining_owned_leases=0),
 source_pins=pin_checks,signed_public_artifacts=signed_files,public_integrity_replay=pub,cas_checks=cas_checks,
 published_source=dict(path=path,before_sha256=sha(before),after_sha256=sha(after),baseline_commit=r['baseline_commit'],published_commit=r['published_commit']),
 resources=dict(phase_count=len(pipe['events']),protected_count=sum(e['protected'] for e in pipe['events']),all_events_completed_and_disk_reservations_released=True,
  historical_chain=[root_id,worker['bridge_phase_lease_id'],worker['parent_lease_id'],worker['lease']['lease_id']],private_grant_sha256=sha(grant_raw),private_grant_bytes_exported=False,
  phase_and_daemon_and_consumer_bindings_match=True,owned_lifecycle_inventory_zero=True,contiguous_wall_accounting=True,hard_enforcement=False),
 reviewer_limits=['The finished lease chain is checked as historical receipts and retained hash-bound grant identities, not re-admitted as a live capability.',
  'Native SQL task/lifecycle state is not reopened; completion and shutdown are cross-checked from public native receipts and source publication.',
  'Nine selected direct source pins are checked for the successful run, not a transitive attestation of every imported library.',
  'No fresh checker invocation, daemon rerun, official benchmark, learned Intent inference, or performance comparison was performed.',
  'Cooperative/sampled CPU, memory and named-root disk accounting remains bounded; hard limits, per-device authority, external worker RSS and ambient temporary storage remain outside this qualification.'])
output=BASE/'managed-native-03-independent-audit.json';output.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(output),sha256=sha(output.read_bytes()),signed_public_artifacts=len(signed_files),source_pins=len(pin_checks),cas_checks=cas_checks,protected_phases=sum(e['protected'] for e in pipe['events']))))
