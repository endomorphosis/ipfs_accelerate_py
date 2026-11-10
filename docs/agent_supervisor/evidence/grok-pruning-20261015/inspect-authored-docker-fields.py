"""Pinned public-image offline inspection of authored pruning configuration."""
from pathlib import Path
import hashlib,json,os,selectors,signal,subprocess,tempfile,time,shutil,uuid,tomllib
B=Path(__file__).resolve().parent
BIN=Path('/home/barberb/lift_coding/artifacts/source-context-substitution-20261013/live-preparation/grok-1.0.46/grok')
EXPECTED='45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e'
IMAGE='sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3'
NAMES=('enabled','keep_last_n_turns','soft_trim_threshold','soft_trim_head','soft_trim_tail','hard_clear_age_turns')
SAFE_PATH_WORDS={'config','configuration','effective_config','effectiveConfig','merged_config','mergedConfig','effective','merged','compaction','pruning','settings','values','resolved','resolvedConfig','resolved_config','toml','raw','content'}
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


def normalized(value):
 return ''.join(ch for ch in value.lower() if ch.isalnum())


def field_census(value):
 counters={name:0 for name in NAMES};keys={'pruning':0,'compaction':0};six=0
 def visit(node):
  nonlocal six
  if type(node) is dict:
   names={normalized(k):v for k,v in node.items()}
   for label in keys:keys[label]+=sum(label in k for k in names)
   for field in NAMES:counters[field]+=sum(normalized(field) in k for k in names)
   if all(any(normalized(field) in k for k in names) for field in NAMES):six+=1
   for child in node.values():visit(child)
  elif type(node) is list:
   for child in node:visit(child)
 visit(value)
 return dict(normalized_target_key_occurrences=counters,pruning_or_compaction_key_occurrences=keys,six_target_fields_same_object_count=six)


def walk(value,path=()):
 if type(value) is dict:
  for key,child in value.items():
   if key=='pruning':yield path+(key,),child,False
   yield from walk(child,path+(key,))
 elif type(value) is list:
  for index,child in enumerate(value):yield from walk(child,path+(index,))
 elif type(value) is str and '[compaction.pruning]' in value:
  try:
   d=tomllib.loads(value)
   if type(d.get('compaction')) is dict and 'pruning' in d['compaction']:
    yield path+('compaction','pruning'),d['compaction']['pruning'],True
  except (ValueError,TypeError):pass


results=[]
for label,enabled in (('off',False),('on',True)):
 assert shutil.disk_usage(B).free>=34359738368
 root=Path(tempfile.mkdtemp(prefix='authored-docker-inspect-'+label+'-',dir=B))
 (root/'home/.grok').mkdir(parents=True);(root/'workspace').mkdir()
 expected=dict(zip(NAMES,(enabled,3,4000,1500,1500,10)))
 lines=['[cli]','auto_update = false','use_leader = false','[compaction.pruning]']
 lines += [f'{key} = '+('true' if value else 'false') if type(value) is bool else f'{key} = {value}' for key,value in expected.items()]
 for vendor in ('cursor','claude','codex'):
  lines += ['[compat.'+vendor+']']+[key+' = false' for key in ('skills','rules','agents','mcps','hooks','sessions')]
 config=root/'home/.grok/config.toml';config.write_text('\n'.join(lines)+'\n')
 config_hash=hashlib.sha256(config.read_bytes()).hexdigest()
 row=dict(profile=label,authored_config_sha256=config_hash,auth_provided=False,canonical_home_mounted=False,network_mode='none',image_id=IMAGE,json_object=False,target_pruning_object_count=0,exact_six_values_match=False)
 cid=None
 try:
  create=['docker','create','--pull=never','--name','authored-pruning-inspect-'+uuid.uuid4().hex[:12],
    '--network=none','--cap-drop=ALL','--security-opt=no-new-privileges:true','--read-only','--log-driver=none',
    '--pids-limit=128','--memory=512m','--cpus=1','--user',str(os.getuid())+':'+str(os.getgid()),
    '--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=67108864,mode=1777',
    '--mount','type=bind,src='+str(BIN)+',dst=/grok,readonly',
    '--mount','type=bind,src='+str(root)+',dst=/work',
    '--workdir','/work/workspace',IMAGE,'/usr/bin/env','-i','HOME=/work/home','GROK_HOME=/work/home/.grok','PATH=/usr/local/bin:/usr/bin:/bin','LANG=C.UTF-8','/grok','inspect','--json']
  created=subprocess.run(create,env={'PATH':'/usr/bin:/bin'},stdin=subprocess.DEVNULL,capture_output=True,timeout=15)
  row['create_returncode']=created.returncode;row['create_stderr_bytes']=len(created.stderr)
  assert created.returncode==0
  cid=created.stdout.decode().strip();assert len(cid)==64 and set(cid)<=set('0123456789abcdef')
  row['container_id_sha256']=hashlib.sha256(cid.encode()).hexdigest()
  rc,buffers,failure=bounded(['docker','start','--attach',cid])
  row.update(returncode=rc,stdout_bytes=len(buffers['stdout']),stderr_bytes=len(buffers['stderr']),bound_failure=failure)
  try:
   value=json.loads(buffers['stdout']);row['json_object']=type(value) is dict
   row.update(field_census(value));row['native_report_sha256']=hashlib.sha256(buffers['stdout']).hexdigest()
   candidates=list(walk(value));row['target_pruning_object_count']=len(candidates)
   row['recognized_top_level_sections']=[key for key in ('config','configuration','effective_config','effectiveConfig','settings','sources','layers','compat','features','environment','memory','compaction') if type(value) is dict and key in value]
   if len(candidates)==1:
    path,target,parsed_toml=candidates[0]
    allowed=all(type(part) is int and 0<=part<128 or type(part) is str and part in SAFE_PATH_WORDS for part in path)
    row['approved_target_path_observed']=allowed
    if allowed:
     row['target_path']=list(path);row['target_from_embedded_toml']=parsed_toml
     values={k:target.get(k) for k in NAMES} if type(target) is dict else {}
     row['target_values']={k:v for k,v in values.items() if type(v) is type(expected[k])}
     row['exact_six_values_match']=set(values)==set(NAMES) and all(type(values[k]) is type(expected[k]) and values[k]==expected[k] for k in NAMES)
  except (ValueError,TypeError,RecursionError):pass
 except BaseException as e:
  row['operation_failed']=True
  row['error_class']=type(e).__name__ if type(e).__name__ in ('AssertionError','TimeoutExpired','OSError','ValueError') else 'other'
 finally:
  if cid:
   cleanup=subprocess.run(['docker','rm','--force',cid],env={'PATH':'/usr/bin:/bin'},stdin=subprocess.DEVNULL,capture_output=True,timeout=15)
   row['cleanup_returncode']=cleanup.returncode
   absent=[]
   for _ in range(2):
    check=subprocess.run(['docker','container','ls','--all','--quiet','--no-trunc','--filter','id='+cid],env={'PATH':'/usr/bin:/bin'},stdin=subprocess.DEVNULL,capture_output=True,timeout=10)
    absent.append(check.returncode==0 and cid.encode() not in check.stdout.splitlines())
   row['exact_container_absence_observations']=absent
  row['authored_config_unchanged']=hashlib.sha256(config.read_bytes()).hexdigest()==config_hash
 results.append(row)
 if row.get('operation_failed') or row.get('returncode')!=0:break
out=dict(schema='grok-authored-docker-inspect-field-qualification@1',binary_sha256=EXPECTED,authored_only=True,canonical_configuration_read=False,credentials_read=False,provider_calls=0,network_requests_possible=False,profiles=results,native_reports_equal=len(results)==2 and results[0].get('native_report_sha256')==results[1].get('native_report_sha256'),minimum_free_bytes=34359738368,remaining_free_bytes=shutil.disk_usage(B).free)
with (B/'inspect-authored-docker-fields-result-01.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(out,sort_keys=True))
