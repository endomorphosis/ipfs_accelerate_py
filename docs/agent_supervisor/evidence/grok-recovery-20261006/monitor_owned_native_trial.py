"""Bounded read-only monitor for one root-authorized Docker trial container."""
import hashlib,json,pathlib,re,subprocess,sys,time
if len(sys.argv)!=3 or re.fullmatch('[0-9a-f]{64}',sys.argv[1]) is None or re.fullmatch('[0-9]{2}',sys.argv[2]) is None:
 raise SystemExit('exact authorized container ID and fresh two-digit trial required')
CID,TRIAL=sys.argv[1:];ROOT=pathlib.Path(__file__).parent
CID_SHA256=hashlib.sha256(CID.encode()).hexdigest()
path=ROOT/('native-live-monitor-'+TRIAL+'.jsonl')
stream=path.open('x');stream.write(json.dumps({'schema':'owned-native-live-monitor-binding@1','trial':TRIAL,'container_id_sha256':CID_SHA256,'monitor_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'deadline_seconds':1100,'poll_seconds':5,'authority':False},sort_keys=True)+'\n');stream.flush();started=time.monotonic();last_save=0;last_signature=None;last_oom=None;baseline_oom=None;sample_count=0;retained=0
code=r"""
import hashlib,json,os,pathlib,stat,time
R=pathlib.Path('/opt/ipfs-supervisor/state/benchmark/launch/state');RUN=R/'run'
def i(v):return v if type(v)is int and 0<=v<2**63 else None
def b(v):return v if type(v)is bool else None
def e(v,allowed):return v if type(v)is str and v in allowed else None if v is None else 'other'
def read(p,limit=1048576):
 descriptor=None
 try:
  if p.is_symlink() or not p.resolve().is_relative_to(R):return None
  descriptor=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
  before=os.fstat(descriptor)
  if not stat.S_ISREG(before.st_mode) or before.st_size>limit:return None
  raw=os.read(descriptor,limit+1);after=os.fstat(descriptor)
  if len(raw)>limit or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):return None
  v=json.loads(raw);return v if type(v)is dict else None
 except (OSError,ValueError,RecursionError):return None
 finally:
  if descriptor is not None:os.close(descriptor)
out={'schema':'owned-native-live-phase-observation@1','observed_at_ms':time.time_ns()//1000000,'authority':False}
p=RUN/'admitted_supervisor_status.json';v=read(p);s={'available':v is not None}
if v:
 s.update(status=e(v.get('status'),{'starting','running','stopping','stopped','failed','completed'}),restart_count=i(v.get('restart_count')))
 for key in ('source_update_pending','control_plane_update_pending','control_plane_reload_deferred'):s[key]=b(v.get(key))
 for key in ('supervisor_pid','daemon_pid'):
  pid=i(v.get(key));record={'pid':pid,'alive':False}
  if pid:
   try:
    raw=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    record.update(alive=raw[0]!='Z',parent_pid=int(raw[1]),birth_ticks=int(raw[19]),threads=int(raw[17]))
   except (OSError,ValueError,IndexError):pass
  s[key]=record
out['supervisor']=s
for stem in ('admitted_native_owner_heartbeat','admitted_database_daemon_pass_heartbeat'):
 p=RUN/(stem+'.json');v=read(p);item={'available':v is not None}
 if v:
  try:item['age_ms']=max(0,int((time.time()-p.stat().st_mtime)*1000))
  except OSError:pass
  item.update(sequence=i(v.get('sequence')),owner_read_succeeded=b(v.get('owner_read_succeeded')),
   active_task_present=type(v.get('active_task_id'))is str and bool(v['active_task_id']))
  item['selection_idle_reason']=e(v.get('selection_idle_reason'),{'no_ready_tasks','running','ready','provider_callback_outcome_unknown','implementation_in_progress','source384_resource_admission_deferred','native_task_blocked'})
 out[stem]=item
lifecycle={}
for name in ('start','stop','start-cleanup-process-proof','start-cleanup-repair'):
 v=read(R/(name+'-receipt.json'));item={'available':v is not None}
 if v:
  item['status']=e(v.get('status'),{'succeeded','conflict','failed','rejected','timeout','unavailable'})
  item['phase']=e(v.get('phase'),{'failed','repaired'})
  err=v.get('error');err=err if type(err)is dict else {};d=err.get('details');d=d if type(d)is dict else {}
  item['error_type']=e(d.get('exception_type'),{'ProcessIdentityMismatch','TransactionConflictError','ProcessTreeNotFenced','StaleLeaseError','RuntimeError','ValueError','TimeoutError','PartialMutationError'})
  for key in ('process_tree_absent','start_succeeded','completion_authority'):item[key]=b(v.get(key))
 lifecycle[name]=item
out['lifecycle']=lifecycle
p=RUN/'bridge-failure-observation.json';events=[]
sidecar=read(p,32768)
observation={'status':'missing' if not p.exists() else 'invalid','observation_only':True,'scope':'fixed_native_diagnostic_sidecar'}
if (type(sidecar)is dict and set(sidecar)=={'schema','task_cid_sha256','attempt_id_sha256','diagnostic','observation_only'}
 and sidecar.get('schema')=='database-bridge-failure-observation@1' and sidecar.get('observation_only')is True
 and all(type(sidecar.get(k))is str and len(sidecar[k])==64 and all(c in '0123456789abcdef' for c in sidecar[k]) for k in ('task_cid_sha256','attempt_id_sha256'))):
 v=sidecar.get('diagnostic')
 if type(v)is dict and v.get('schema')=='database-bridge-failure-diagnostic@1':
  observation.update(status='observed',task_cid_sha256=sidecar['task_cid_sha256'],attempt_id_sha256=sidecar['attempt_id_sha256'])
  c=v.get('callback');c=c if type(c)is dict else {};n=c.get('native_exit');n=n if type(n)is dict else {}
  child=v.get('child_reported_router_failure');child=child if type(child)is dict else {};wrap=child.get('diagnostic');wrap=wrap if type(wrap)is dict else {};rd=wrap.get('diagnostic');rd=rd if type(rd)is dict else {}
  item={'phase':e(v.get('phase'),{'unknown_callback','terminal_failure'}),'reason_code':e(v.get('reason_code'),{'portal_provider_failed','implementation_protected_path_mutated','portal_validation_failed','portal_result_unavailable','other'}),
   'callback_state':e(c.get('state'),{'started_outcome_unknown','failed_outcome_settled','not_dispatched','completed','other'}),
   'native_exit_present':b(n.get('present')),'returncode':n.get('returncode') if type(n.get('returncode'))is int and -255<=n['returncode']<=255 else None,
   'child_report_status':e(child.get('status'),{'observed','missing','ambiguous','invalid','unavailable'}),
   'child_phase':e(rd.get('phase'),{'runner_initialization','argument_validation','planning_contract','container_boundary','workspace_identity','semantic_context','doctor_residual','public_instruction','model_prompt','credential_isolation','provider_discovery','provider_allocation','provider_initialization','provider_invocation','provider_result_validation','semantic_response_decode','unclassified'}),
   'child_error_type':e(wrap.get('error_type'),{'ValueError','TypeError','RuntimeError','PermissionError','FileNotFoundError','CalledProcessError','TimeoutExpired','TimeoutError','JSONDecodeError','UnicodeDecodeError','ImportError','ModuleNotFoundError','OSError','SemanticTranslationError','other'}),'scope':'fixed_native_diagnostic_sidecar','authority':False}
  for key in ('reaped','process_group_absent','subreaper_children_absent','lifecycle_finalized'):item[key]=b(n.get(key))
  events.append(item)
out['bridge_observation']=observation
out['bridge_failures']=events
cg=pathlib.Path('/sys/fs/cgroup');memory={}
for name in ('memory.current','memory.max','memory.high','memory.peak','pids.current','pids.max'):
 try:
  value=(cg/name).read_text().strip();memory[name]=int(value) if value.isdigit() else 'max' if value=='max' else None
 except OSError:memory[name]=None
for name in ('memory.events','memory.events.local'):
 try:
  value=dict(line.split() for line in (cg/name).read_text().splitlines())
  memory[name]={key:int(value[key]) for key in ('low','high','max','oom','oom_kill','oom_group_kill') if key in value and value[key].isdigit()}
 except (OSError,ValueError):memory[name]=None
out['cgroup']=memory
print(json.dumps(out,sort_keys=True))
"""
try:
 while time.monotonic()-started<1100:
  try:
   result=subprocess.run(['docker','exec','-i',CID,'python3','-I','-B','-c',code],capture_output=True,text=True,timeout=12)
  except subprocess.TimeoutExpired:
   sample_count+=1
   row={'schema':'owned-native-live-monitor-unavailable@1','trial':TRIAL,'reason':'bounded_docker_exec_timeout','elapsed_seconds':round(time.monotonic()-started,3),'absence_is_not_success':True,'authority':False}
   stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();print(json.dumps(row),flush=True)
   time.sleep(min(5,max(0,1100-(time.monotonic()-started))));continue
  now=time.monotonic();sample_count+=1
  if result.returncode:
   try:
    state=subprocess.run(['docker','inspect','--format','{{.State.Running}}',CID],capture_output=True,text=True,timeout=5)
   except subprocess.TimeoutExpired:
    state=None
   closure={'schema':'owned-native-live-monitor-closure@1','trial':TRIAL,'elapsed_seconds':round(now-started,3),'samples':sample_count,'retained':retained,'docker_exec_returncode':result.returncode,'container_running':state.stdout.strip()=='true' if state is not None and state.returncode==0 else None,'container_inspect_returncode':state.returncode if state is not None else None,'absence_is_not_success':True,'authority':False}
   stream.write(json.dumps(closure,sort_keys=True)+'\n');stream.flush();print(json.dumps(closure),flush=True);break
  value=json.loads(result.stdout);value.update(trial=TRIAL,elapsed_seconds=round(now-started,3))
  c=value.get('cgroup',{});oom=(c.get('memory.events') or {}).get('oom_kill')
  if baseline_oom is None:baseline_oom=oom
  value['oom_kill_delta_since_first_observation']=oom-baseline_oom if type(oom)is int and type(baseline_oom)is int and oom>=baseline_oom else None
  signature=json.dumps({key:value[key] for key in ('supervisor','lifecycle','bridge_observation','bridge_failures')},sort_keys=True)
  if signature!=last_signature or oom!=last_oom or now-last_save>=30:
   stream.write(json.dumps(value,sort_keys=True)+'\n');stream.flush();retained+=1;last_save=now
   print(json.dumps({'trial':TRIAL,'elapsed_seconds':value['elapsed_seconds'],'native_status':value['supervisor'].get('status'),'start_status':value['lifecycle']['start'].get('status'),'bridge_failure_count':len(value['bridge_failures']),'oom_kill_delta':value['oom_kill_delta_since_first_observation'],'retained':retained}),flush=True)
  last_signature,last_oom=signature,oom
  time.sleep(min(5,max(0,1100-(time.monotonic()-started))))
 else:
  stream.write(json.dumps({'schema':'owned-native-live-monitor-closure@1','trial':TRIAL,'reason':'bounded_monitor_deadline','authority':False})+'\n');stream.flush()
finally:
 stream.close()
print(path,flush=True)
