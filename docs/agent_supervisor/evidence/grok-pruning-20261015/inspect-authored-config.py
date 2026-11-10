"""Network-isolated authored config inspection; export no native report bodies."""
from pathlib import Path
import hashlib,json,os,selectors,signal,subprocess,tempfile,time
B=Path(__file__).resolve().parent
BIN=Path('/home/barberb/lift_coding/artifacts/source-context-substitution-20261013/live-preparation/grok-1.0.46/grok')
EXPECTED='45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e'
NAMES=('enabled','keep_last_n_turns','soft_trim_threshold','soft_trim_head','soft_trim_tail','hard_clear_age_turns')
os.umask(0o077)
with BIN.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED


def bounded(command):
 p=subprocess.Popen(command,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
 buffers={'stdout':bytearray(),'stderr':bytearray()};failure=None;deadline=time.monotonic()+30
 try:
  with selectors.DefaultSelector() as s:
   for label,pipe in (('stdout',p.stdout),('stderr',p.stderr)):
    os.set_blocking(pipe.fileno(),False);s.register(pipe,selectors.EVENT_READ,label)
   while s.get_map():
    if time.monotonic()>=deadline:failure='timeout';break
    for key,_ in s.select(.1):
     block=os.read(key.fd,8192)
     if not block:s.unregister(key.fileobj);continue
     if sum(map(len,buffers.values()))+len(block)>262144:failure='output_bound';break
     buffers[key.data].extend(block)
    if failure:break
 finally:
  if failure or p.poll() is None:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  p.wait(timeout=3)
 return p.returncode,buffers,failure


def walk(value,path=()):
 if type(value) is dict:
  for key,child in value.items():
   if key=='pruning':yield path+(key,),child
   yield from walk(child,path+(key,))
 elif type(value) is list:
  for index,child in enumerate(value):yield from walk(child,path+(index,))


results=[]
for label,enabled in (('off',False),('on',True)):
 root=Path(tempfile.mkdtemp(prefix='authored-inspect-'+label+'-',dir=B))
 (root/'home/.grok').mkdir(parents=True);(root/'workspace').mkdir()
 expected=dict(zip(NAMES,(enabled,3,4000,1500,1500,10)))
 lines=['[cli]','auto_update = false','use_leader = false','[compaction.pruning]']
 lines += [f'{key} = '+('true' if value else 'false') if type(value) is bool else f'{key} = {value}' for key,value in expected.items()]
 for vendor in ('cursor','claude','codex'):
  lines += ['[compat.'+vendor+']']+[key+' = false' for key in ('skills','rules','agents','mcps','hooks','sessions')]
 (root/'home/.grok/config.toml').write_text('\n'.join(lines)+'\n')
 command=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--ro-bind','/usr','/usr','--symlink','usr/lib','/lib','--symlink','usr/bin','/bin','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/etc','--dir','/home','--ro-bind',str(BIN),'/grok','--bind',str(root),'/work','--chdir','/work/workspace','--clearenv','--setenv','HOME','/work/home','--setenv','GROK_HOME','/work/home/.grok','--setenv','PATH','/usr/bin:/bin','--setenv','LANG','C.UTF-8','/grok','inspect','--json']
 rc,buffers,failure=bounded(command)
 row=dict(profile=label,returncode=rc,stdout_bytes=len(buffers['stdout']),stderr_bytes=len(buffers['stderr']),bound_failure=failure,authored_config_sha256=hashlib.sha256((root/'home/.grok/config.toml').read_bytes()).hexdigest(),network_namespace_requested=True,canonical_home_mounted=False,auth_provided=False,json_object=False,target_pruning_object_count=0,exact_six_values_match=False)
 try:
  value=json.loads(buffers['stdout']);row['json_object']=type(value) is dict
  candidates=list(walk(value));row['target_pruning_object_count']=len(candidates)
  if len(candidates)==1:
   path,target=candidates[0]
   # Export only a recognized closed key path, never arbitrary discovery keys.
   approved={('config','compaction','pruning'),('effective_config','compaction','pruning'),('effectiveConfig','compaction','pruning'),('compaction','pruning'),('configuration','compaction','pruning')}
   row['approved_target_path_observed']=path in approved
   if path in approved:
    row['target_path']=list(path)
    values={k:target.get(k) for k in NAMES} if type(target) is dict else {}
    row['target_values']={k:v for k,v in values.items() if type(v) is type(expected[k])}
    row['exact_six_values_match']=set(values)==set(NAMES) and all(type(values[k]) is type(expected[k]) and values[k]==expected[k] for k in NAMES)
 except (ValueError,TypeError,RecursionError):pass
 results.append(row)
 # A namespace refusal gives no native-config evidence; don't repeat identical failure.
 if rc!=0 and not buffers['stdout']:break
out=dict(schema='grok-authored-inspect-qualification@1',binary_sha256=EXPECTED,authored_only=True,canonical_configuration_read=False,credentials_read=False,provider_calls=0,network_namespace_required=True,profiles=results)
with (B/'inspect-authored-result-01.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(out,sort_keys=True))
