"""Install and qualify the separate model-worker identity in a Harbor container.

All scripts and the namespace manifest are owned by container root. Model calls
are absent from qualification; the only delegated entry is the pinned router.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
import tempfile
import threading
import zlib
from pathlib import Path

from .terminal_deployment import PYTHON, ROOT, native_codex_exposure_script, runtime_environment

# A root-protected deployment choice is separate from the unchanged @1 Unix
# boundary. It is never inferred from caller flags or a serialized receipt.
PROFILE_SCHEMA = "container-worker-deployment-profile@1"
PROFILE_PATH = ROOT + "/worker-profile.json"
PROFILE_GUARD = r'''
import hashlib,json,os,pathlib,stat
def _profile_read(path):
 path=pathlib.Path(path)
 for ancestor in path.parents:
  info=ancestor.lstat()
  if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_mode&0o022:raise SystemExit('protected worker profile ancestor required')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode) or before.st_uid!=0 or before.st_nlink!=1 or stat.S_IMODE(before.st_mode)!=0o444 or not 0<before.st_size<=16384:raise SystemExit('protected bounded worker profile required')
  raw=os.read(fd,16385)
  witness=lambda info:(info.st_dev,info.st_ino,info.st_mode,info.st_uid,info.st_gid,info.st_nlink,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
  if len(raw)!=before.st_size or witness(before)!=witness(os.fstat(fd)) or witness(before)!=witness(path.lstat()):raise SystemExit('worker profile changed during read')
  return raw
 finally:os.close(fd)
def _profile_json(raw):
 def unique(rows):
  value={}
  for key,item in rows:
   if key in value:raise SystemExit('duplicate worker profile field')
   value[key]=item
  return value
 try:value=json.loads(raw,object_pairs_hook=unique)
 except (ValueError,UnicodeError):raise SystemExit('canonical worker profile required')
 if (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()!=raw:raise SystemExit('canonical worker profile required')
 return value
def _observer_absence(root,role):
 # Worker cannot read owner-private directories. Their absence is checked by
 # root at installation and by the owner before every delegation, never guessed
 # from PermissionError. Public/worker-visible provider paths are checked here.
 directories=[root/'provider-bin']
 if role in ('worker','installation'):directories.append(root/'worker-home/.codex')
 forbidden=[root/'native-codex',root/'native-codex.manifest.json']
 if role!='worker':
  directories += [root/'codex-auth',root/'home/.codex']
  forbidden += [root/'home/.nvm']
 for directory in directories:
  try:
   info=directory.lstat()
   if not stat.S_ISDIR(info.st_mode) or directory.is_symlink() or any(directory.iterdir()):raise SystemExit('observer profile refuses credential or provider directory contents')
  except FileNotFoundError:pass
 for path in forbidden+[pathlib.Path(base)/name for base in ('/usr/local/bin','/usr/bin','/bin') for name in ('codex','codex-code-mode-host')]:
  try:path.lstat()
  except FileNotFoundError:continue
  raise SystemExit('observer profile refuses native provider binaries')
def _worktree_observation_arguments(argv):
 if not argv or not argv[0].startswith('--observe-worker-worktree'):return None
 if len(argv)!=5 or argv[0]!='--observe-worker-worktree':raise SystemExit('closed worker worktree observation required')
 pid,ticks,birth,worktree=argv[1:]
 if (not pid.isascii() or not pid.isdecimal() or str(int(pid))!=pid or not 0<int(pid)<2**31 or not ticks.isascii() or not ticks.isdecimal() or str(int(ticks))!=ticks or not 0<int(ticks)<2**63):raise SystemExit('literal worker birth numbers required')
 if len(birth)!=38 or not birth.startswith('birth:') or any(c not in '0123456789abcdef' for c in birth[6:]):raise SystemExit('exact worker birth digest required')
 path=pathlib.PurePosixPath(worktree)
 if not 0<len(worktree.encode('utf-8'))<=4096 or any(c in worktree for c in ('\x00','\n','\r')) or not path.is_absolute() or str(path)!=worktree or '..' in path.parts or worktree=='/':raise SystemExit('canonical allocated worktree required')
 return int(pid),int(ticks),birth,worktree
def _worker_profile(root,role,argv):
 profile=_profile_json(_profile_read(root/'worker-profile.json'))
 if (type(profile)is not dict or set(profile)!={'schema','observer_only','boundary_sha256'} or profile['schema']!='container-worker-deployment-profile@1' or type(profile['observer_only'])is not bool or type(profile['boundary_sha256'])is not str):raise SystemExit('closed worker deployment profile required')
 boundary=_profile_read(root/'container-boundary.json')
 if hashlib.sha256(boundary).hexdigest()!=profile['boundary_sha256']:raise SystemExit('worker profile boundary differs')
 if role=='worker' and _worktree_observation_arguments(argv)is not None:return profile
 if not profile['observer_only']:return profile
 # Cleanup must remain available after an asset-presence refusal. It makes no
 # provider-absence claim and cannot dispatch caller code or a provider.
 if role=='worker' and argv==['--cleanup']:return profile
 _observer_absence(root,role)
 if role=='validation':raise SystemExit('observer profile refuses candidate validation execution')
 if argv==['--preflight'] or argv==['--preflight','--preflight-sleep','30']:return profile
 values={'--model','--reasoning-effort','--timeout','--max-output-tokens','--purpose','--semantic-repository','--semantic-transport','--intent-requirements-transport','--code-retrieval-codec','--code-retrieval-wire-delivery','--doctor-residual-artifact','--doctor-residual-sha256','--doctor-residual-task-cid','--public-instruction-artifact','--public-instruction-sha256','--public-instruction-task-cid','--checked-context-artifact','--checked-context-sha256','--checked-context-task-cid'}
 parsed={};index=0
 while index<len(argv):
  flag=argv[index]
  if flag in parsed or flag not in values|{'--observe-model-input-only'}:raise SystemExit('observer profile refuses alternate worker route')
  if flag=='--observe-model-input-only':parsed[flag]=True;index+=1;continue
  if index+1>=len(argv) or not argv[index+1] or argv[index+1].startswith('--'):raise SystemExit('observer profile requires complete literal options')
  parsed[flag]=argv[index+1];index+=2
 if (parsed.get('--observe-model-input-only')is not True or parsed.get('--purpose','coding')!='coding' or not all(parsed.get(flag) for flag in ('--semantic-repository','--public-instruction-artifact','--public-instruction-sha256','--public-instruction-task-cid'))):raise SystemExit('observer profile requires bound coding observation')
 checked=('--checked-context-artifact','--checked-context-sha256','--checked-context-task-cid')
 if any(flag in parsed for flag in checked) and (not all(parsed.get(flag) for flag in checked) or parsed.get('--semantic-transport','identifiers')!='native' or parsed.get('--intent-requirements-transport','expanded')!='expanded' or parsed.get('--code-retrieval-codec','expanded')!='expanded' or parsed.get('--code-retrieval-wire-delivery','inline')!='inline' or any(flag.startswith('--doctor-') for flag in parsed)):raise SystemExit('checked context requires exact native expanded coding observation')
 return profile
'''

OWNER_MONITOR = r"""
import signal,subprocess
process=None
def interrupted(signum,frame):
 raise InterruptedError('owner monitor interrupted')
for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)
code=1
try:
 process=subprocess.Popen(command)
 guard=globals().get("proof_query_guard") or globals().get("checked_context_guard")
 if guard is not None:guard.note_spawned(process)
 code=process.wait()
except BaseException:
 if process is not None:
  try:process.terminate()
  except ProcessLookupError:pass
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:process.kill();process.wait()
 raise
finally:
 for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,signal.SIG_IGN)
 cleanup=subprocess.run(['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry','--cleanup'],cwd='/',stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,start_new_session=True,timeout=15)
 if cleanup.returncode:code=1
 guard=globals().get("proof_query_guard") or globals().get("checked_context_guard")
 if guard is not None:guard.abort()
raise SystemExit(code)
"""

OWNER_ENTRY = (
    r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import json,os,pathlib,sys
root=pathlib.Path('/opt/ipfs-supervisor')
""" + PROFILE_GUARD + r"""
profile=_worker_profile(root,'owner',sys.argv[1:])
boundary=json.loads((root/'container-boundary.json').read_text())
if os.getuid()!=boundary['owner_uid'] or os.geteuid()!=boundary['owner_uid']:raise SystemExit('owner identity required')
cwd=pathlib.Path.cwd()
if cwd.resolve()!=cwd or not any(cwd!=pathlib.Path(p) and cwd.is_relative_to(p) for p in boundary['allowed_worktree_roots']):raise SystemExit('allocated candidate required')
marker=cwd/'.git'
if not marker.is_file() or marker.is_symlink() or marker.stat().st_uid!=1000:raise SystemExit('owner allocated Git worktree marker required')
# Allocate group writes only in this exact candidate; canonical and owner state
# remain outside this traversal. Refuse links rather than chmod their targets.
for base,dirs,files in os.walk(cwd,followlinks=False):
 p=pathlib.Path(base)
 if p.is_symlink() or p.stat().st_uid not in (1000,1001):raise SystemExit('candidate directory ownership differs')
 if p.stat().st_uid==1000:p.chmod(0o3770 if p==cwd else 0o2770)
 for name in dirs:
  if (p/name).is_symlink():raise SystemExit('candidate directory symlink refused')
 for name in files:
  q=p/name
  if q.is_symlink() or not q.is_file() or q.stat().st_uid not in (1000,1001) or q.stat().st_nlink!=1:raise SystemExit('candidate input is not one allocated regular file')
  if q.stat().st_uid==1000:q.chmod(0o440 if q==cwd/'.git' else (0o770 if q.stat().st_mode&0o111 else 0o660))
command=['/usr/bin/sudo','-n','-u','benchmarkworker','--',str(root/'bin/worker-entry'),*sys.argv[1:]]
proof_query_guard=None
checked_context_guard=None
if any(part==flag or part.startswith(flag+'=') for part in sys.argv[1:] for flag in ('--finite-proof-query-context','--finite-proof-query-sha256','--finite-proof-query-context-cid')):
 sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
 from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_dispatch import require_finite_proof_query_isolated_worker_dispatch
 proof_query_guard=require_finite_proof_query_isolated_worker_dispatch(command=command,environment=dict(os.environ),worktree=cwd)
if os.environ.get('IPFS_ACCELERATE_CHECKED_CONTEXT_DISPATCH_SOCKET') or os.environ.get('IPFS_ACCELERATE_CHECKED_CONTEXT_DISPATCH_CID') or os.environ.get('IPFS_ACCELERATE_CHECKED_CONTEXT_OWNER_BIRTH') or os.environ.get('IPFS_ACCELERATE_CHECKED_CONTEXT_WRAPPER_TICKET'):
 sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
 from ipfs_accelerate_py.agent_supervisor.runtime.checked_context_dispatch import require_checked_context_isolated_worker_dispatch
 checked_context_guard=require_checked_context_isolated_worker_dispatch(command=command,environment=dict(os.environ),worktree=cwd)
"""
    + OWNER_MONITOR
)

VALIDATION_ENTRY = (
    r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import os,sys,pathlib
root=pathlib.Path('/opt/ipfs-supervisor')
""" + PROFILE_GUARD + r"""
profile=_worker_profile(root,'validation',sys.argv[1:])
if os.getuid()!=1000 or os.geteuid()!=1000:raise SystemExit('owner identity required')
if len(sys.argv)<2:raise SystemExit('literal validation argv required')
command=['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry','--validate',*sys.argv[1:]]
"""
    + OWNER_MONITOR
)

WORKER_WORKTREE_FAILURE = r'''
_worktree_observation_stage='profile'
def _worktree_failure(error):
 frames=[];frame=error.__traceback__
 while frame is not None:
  code=frame.tb_frame.f_code
  clean=lambda value,limit:''.join(c if 32<=ord(c)<127 else '?' for c in value)[:limit]
  frames.append({'file':clean(pathlib.Path(code.co_filename).name,256),'function':clean(code.co_name,128),'line':min(max(frame.tb_lineno,0),2**31-1)})
  frame=frame.tb_next
 value={'schema':'checked-worker-kernel-worktree-refusal@1','stage':_worktree_observation_stage,
  'error_type':type(error).__name__[:128],'message_sha256':hashlib.sha256(str(error).encode('utf-8',errors='replace')).hexdigest(),
  'failure_frames':frames[-16:],'metadata_only':True,'execution_authority':False,'proof_authority':False,
  'completion_authority':False,'source_body_executed':False,'provider_calls':0}
 raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
 if len(raw.encode())>16384:raise SystemExit('bounded worker refusal required')
 print(raw,flush=True)
'''

WORKER_WORKTREE_OBSERVER = r'''
def _observe_worker_worktree(arguments):
 global _worktree_observation_stage
 _worktree_observation_stage='boundary'
 pid,ticks,birth,worktree=arguments
 boundary=json.loads(_profile_read(root/'container-boundary.json'))
 fields={'schema','container_id','image_id','namespaces','owner_uid','worker_uid','allowed_worktree_roots','owner_private_paths','validation_repository_roots','single_worker'}
 if (type(boundary)is not dict or set(boundary)!=fields or boundary['schema']!='supervisor-container-worker-boundary@1' or boundary['owner_uid']!=1000 or type(boundary['owner_uid'])is not int or boundary['worker_uid']!=1001 or type(boundary['worker_uid'])is not int or boundary['single_worker']is not True):raise SystemExit('exact single-worker metadata boundary required')
 namespaces={key:os.readlink('/proc/self/ns/'+key) for key in ('pid','mnt','net')}
 if boundary['namespaces']!=namespaces:raise SystemExit('worktree observer namespace differs')
 _worktree_observation_stage='self_privilege'
 self_status=dict(line.split(':',1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
 if any(int(self_status[key].strip(),16) for key in ('CapEff','CapPrm','CapAmb')) or self_status.get('NoNewPrivs','').strip()!='1':raise SystemExit('worktree observer privilege boundary differs')
 _worktree_observation_stage='worktree_scope'
 roots=boundary['allowed_worktree_roots'];private=boundary['owner_private_paths']
 if type(roots)is not list or roots!=[str(root/'worktrees')] or type(private)is not list or not 1<=len(private)<=16:raise SystemExit('fixed worktree observer scope required')
 expected=pathlib.Path(worktree)
 if expected.resolve(strict=True)!=expected or expected.is_symlink() or not expected.is_dir() or not expected.is_relative_to(root/'worktrees') or expected==root/'worktrees':raise SystemExit('exact allocated observation worktree required')
 for value in roots+private:
  path=pathlib.Path(value)
  if not path.is_absolute() or '..' in path.parts or str(path)=='/' or path.is_symlink() or path.resolve(strict=True)!=path:raise SystemExit('exact worker metadata boundary path required')
  info=path.lstat()
  if value in roots:
   if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0,1000) or os.access(path,os.W_OK):raise SystemExit('protected worktree parent required')
  else:
   if info.st_uid!=1000 or info.st_mode&0o077 or not(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)) or expected.is_relative_to(path) or any(os.access(path,mode) for mode in (os.R_OK,os.W_OK,os.X_OK)):raise SystemExit('protected owner metadata boundary required')
  for parent in path.parents:
   info=parent.lstat()
   if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0,1000) or os.access(parent,os.W_OK):raise SystemExit('protected worker metadata ancestor required')
 def observe():
  global _worktree_observation_stage
  _worktree_observation_stage='target_stat'
  proc=pathlib.Path('/proc')/str(pid)
  raw=(proc/'stat').read_bytes()
  if not 0<len(raw)<=4096 or not raw.startswith((str(pid)+' (').encode()) or b') 'not in raw:raise SystemExit('bounded worker kernel stat required')
  facts=raw[raw.rfind(b') ')+2:].split()
  if len(facts)<20 or facts[0]in(b'Z',b'X'):raise SystemExit('live worker kernel stat required')
  parent=int(facts[1]);start=int(facts[19])
  boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  if len(boot)!=36 or any(c not in '0123456789abcdef-' for c in boot) or parent<=0:raise SystemExit('exact worker boot and parent required')
  _worktree_observation_stage='target_status'
  status_raw=(proc/'status').read_bytes()
  if not 0<len(status_raw)<=16384:raise SystemExit('bounded worker kernel status required')
  status=dict(line.split(':',1) for line in status_raw.decode().splitlines() if ':' in line)
  if status.get('Uid','').split()!=['1001']*4 or any(int(status[key].strip(),16) for key in ('CapEff','CapPrm','CapAmb')):raise SystemExit('exact unprivileged worker UID required')
  readiness=status.get('NoNewPrivs','').strip()
  if readiness not in ('0','1'):raise SystemExit('exact worker readiness state required')
  _worktree_observation_stage='target_command'
  command_raw=(proc/'cmdline').read_bytes()
  if not 0<len(command_raw)<=65536 or not command_raw.endswith(b'\x00'):raise SystemExit('bounded worker kernel command required')
  command=[part.decode('utf-8') for part in command_raw[:-1].split(b'\x00')]
  marker=str(root/'bin/worker-entry')
  if not 1<=len(command)<=80 or any(not part or len(part.encode())>8192 for part in command) or command.count(marker)!=1 or '--observe-worker-worktree'in command:raise SystemExit('original coding worker command marker required')
  _worktree_observation_stage='target_birth_worktree'
  observed_birth='birth:'+hashlib.sha256((str(pid)+':'+str(start)+':'+boot+':'+str(parent)).encode()).hexdigest()[:32]
  if start!=ticks or observed_birth!=birth or os.readlink(str(proc/'cwd'))!=worktree or {key:os.readlink(str(proc/'ns'/key)) for key in ('pid','mnt','net')}!=namespaces:raise SystemExit('worker birth or actual worktree changed')
  if expected.resolve(strict=True)!=expected or expected.is_symlink():raise SystemExit('allocated observation path changed')
  return readiness=='1',{'pid':pid,'uid':1001,'parent_pid':parent,'start_time_ticks':start,'process_birth_id':observed_birth,'boot_id':boot,'worktree':worktree,'command_sha256':hashlib.sha256(json.dumps(command,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),'namespaces':namespaces}
 deadline=time.monotonic()+2.0
 original=None
 while True:
  ready,first=observe()
  if original is None:original=first
  elif first!=original:raise SystemExit('worker metadata changed during readiness observation')
  if ready:break
  _worktree_observation_stage='readiness'
  if time.monotonic()>=deadline:raise SystemExit('worker readiness did not arrive within bound')
  time.sleep(min(.01,max(0,deadline-time.monotonic())))
 _worktree_observation_stage='stable_observation'
 ready,second=observe()
 if not ready:raise SystemExit('worker readiness changed during observation')
 if first!=second:raise SystemExit('worker metadata changed during observation')
 _worktree_observation_stage='output'
 value={'schema':'checked-worker-kernel-worktree-observation@1',**second,'metadata_only':True,'execution_authority':False,'proof_authority':False,'completion_authority':False,'source_body_executed':False,'provider_calls':0}
 raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
 if len(raw.encode())>16384:raise SystemExit('worktree metadata observation exceeds bound')
 return raw
'''

WORKER_ENTRY = r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import argparse,ctypes,hashlib,json,os,pathlib,sys,time
root=pathlib.Path('/opt/ipfs-supervisor')
""" + WORKER_WORKTREE_FAILURE + PROFILE_GUARD + r"""
metadata_mode=len(sys.argv)>1 and sys.argv[1].startswith('--observe-worker-worktree')
try:
 if os.getuid()!=1001 or os.geteuid()!=1001:raise SystemExit('worker identity required')
 profile=_worker_profile(root,'worker',sys.argv[1:])
except BaseException as error:
 if metadata_mode:_worktree_failure(error);raise SystemExit(1)
 raise
# Prevent this external-isolation worker from acquiring any new Unix privilege.
libc=ctypes.CDLL(None,use_errno=True)
try:
 _worktree_observation_stage='privilege'
 if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'no-new-privileges failed')
except BaseException as error:
 if metadata_mode:_worktree_failure(error);raise SystemExit(1)
 raise
os.umask(0o007)
os.environ.clear()
os.environ.update({'HOME':str(root/'worker-home'),'PATH':str(root/'provider-bin')+':/usr/local/bin:/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root/'source')+':'+str(root/'datasets')+':'+str(root/'kit'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
os.environ.update({'PYTHON':str(root/'venv/bin/python'),'PYTHON_BIN':str(root/'venv/bin/python'),'GIT_CONFIG_COUNT':'3','GIT_CONFIG_KEY_0':'safe.directory','GIT_CONFIG_VALUE_0':str(pathlib.Path.cwd()),'GIT_CONFIG_KEY_1':'core.hooksPath','GIT_CONFIG_VALUE_1':'/dev/null','GIT_CONFIG_KEY_2':'core.fsmonitor','GIT_CONFIG_VALUE_2':'false'})
os.environ.update({'IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE':str(root/'venv/bin/python'),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONHASHSEED':'0'})
if profile['observer_only']:os.environ['GIT_OPTIONAL_LOCKS']='0'
""" + WORKER_WORKTREE_OBSERVER + r"""
observation_arguments=_worktree_observation_arguments(sys.argv[1:])
if observation_arguments is not None:
 try:print(_observe_worker_worktree(observation_arguments))
 except BaseException as error:_worktree_failure(error);raise SystemExit(1)
 raise SystemExit(0)
sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
from ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary import verify_container_worker_boundary
artifact=root/'container-boundary.json'
digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
if sys.argv[1:]==['--cleanup']:
 boundary=json.loads(artifact.read_bytes())
 if boundary['namespaces']!={k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')}:raise SystemExit('cleanup namespace differs')
 import signal
 def workers():
  found=[]
  for p in pathlib.Path('/proc').iterdir():
   if not p.name.isdigit() or int(p.name)==os.getpid():continue
   try:
    status=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
    if status['Uid'].split()[0]=='1001' and not status['State'].strip().startswith('Z'):found.append(int(p.name))
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  return found
 for signum in (signal.SIGTERM,signal.SIGKILL):
  for pid in workers():
   try:os.kill(pid,signum)
   except ProcessLookupError:pass
  time.sleep(.2)
 if workers():raise SystemExit('worker cleanup incomplete')
 raise SystemExit(0)
validating=len(sys.argv)>1 and sys.argv[1]=='--validate'
evidence=verify_container_worker_boundary(artifact=artifact,expected_sha256=digest,workspace=pathlib.Path.cwd(),purpose='validation' if validating else 'coding')
if validating:
 if any(part=='--observe-model-input-only' or part.startswith('--observe-model-input-only=') for part in sys.argv[2:]):raise SystemExit('model input observation requires bound router coding')
 if any(part==flag or part.startswith(flag+'=') for part in sys.argv[2:] for flag in ('--checked-context-artifact','--checked-context-sha256','--checked-context-task-cid')):raise SystemExit('checked context requires bound router coding observation')
 if len(sys.argv)<3 or len(sys.argv)>130 or any(len(x)>8192 for x in sys.argv[2:]):raise SystemExit('bounded literal validation argv required')
 os.execvpe(sys.argv[2],sys.argv[2:],dict(os.environ))
parser=argparse.ArgumentParser()
parser.add_argument('--model',default='gpt-5.6-sol');parser.add_argument('--reasoning-effort',choices=['low','medium','high','xhigh','max'],default='high')
parser.add_argument('--timeout',type=int,default=90);parser.add_argument('--max-output-tokens',type=int,default=4096)
parser.add_argument('--purpose',choices=['planning','coding'],default='coding')
parser.add_argument('--observe-model-input-only',action='store_true')
parser.add_argument('--semantic-repository',type=pathlib.Path)
parser.add_argument('--semantic-transport',choices=['identifiers','native'],default='identifiers')
parser.add_argument('--intent-requirements-transport',choices=['expanded','compact'],default='expanded')
parser.add_argument('--code-retrieval-codec',choices=['expanded','compact'],default='expanded')
parser.add_argument('--code-retrieval-wire-delivery',choices=['inline','materialized'],default='inline')
parser.add_argument('--doctor-candidate-artifact',type=pathlib.Path)
parser.add_argument('--doctor-candidate-sha256')
parser.add_argument('--doctor-task-cid')
parser.add_argument('--doctor-contract-artifact',type=pathlib.Path)
parser.add_argument('--doctor-contract-sha256')
parser.add_argument('--doctor-contract-task-cid')
parser.add_argument('--finite-repository-artifact',type=pathlib.Path)
parser.add_argument('--finite-repository-sha256')
parser.add_argument('--finite-repository-task-cid')
parser.add_argument('--finite-proof-query-context',type=pathlib.Path)
parser.add_argument('--finite-proof-query-sha256')
parser.add_argument('--finite-proof-query-context-cid')
parser.add_argument('--doctor-residual-artifact',type=pathlib.Path)
parser.add_argument('--doctor-residual-sha256')
parser.add_argument('--doctor-residual-task-cid')
parser.add_argument('--public-instruction-artifact',type=pathlib.Path)
parser.add_argument('--public-instruction-sha256')
parser.add_argument('--public-instruction-task-cid')
parser.add_argument('--checked-context-artifact',type=pathlib.Path)
parser.add_argument('--checked-context-sha256')
parser.add_argument('--checked-context-task-cid')
parser.add_argument('--preflight',action='store_true');parser.add_argument('--preflight-sleep',type=int,default=0)
args=parser.parse_args()
doctor_values=(args.doctor_candidate_artifact,args.doctor_candidate_sha256,args.doctor_task_cid)
residual_values=(args.doctor_residual_artifact,args.doctor_residual_sha256,args.doctor_residual_task_cid)
contract_values=(args.doctor_contract_artifact,args.doctor_contract_sha256,args.doctor_contract_task_cid)
finite_values=(args.finite_repository_artifact,args.finite_repository_sha256,args.finite_repository_task_cid)
proof_query_values=(args.finite_proof_query_context,args.finite_proof_query_sha256,args.finite_proof_query_context_cid)
instruction_values=(args.public_instruction_artifact,args.public_instruction_sha256,args.public_instruction_task_cid)
checked_values=(args.checked_context_artifact,args.checked_context_sha256,args.checked_context_task_cid)
if any(checked_values) and (not all(checked_values) or not args.observe_model_input_only or args.purpose!='coding' or args.semantic_repository is None or not all(instruction_values) or args.semantic_transport!='native' or args.intent_requirements_transport!='expanded' or args.code_retrieval_codec!='expanded' or args.code_retrieval_wire_delivery!='inline' or args.preflight or any(doctor_values) or any(contract_values) or any(residual_values) or any(finite_values) or any(proof_query_values)):raise SystemExit('checked context requires exact native expanded coding observation')
if any(finite_values) and (not all(finite_values) or any(contract_values) or any(doctor_values) or any(residual_values) or any(instruction_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact finite repository candidate invocation required')
if any(proof_query_values) and (not all(proof_query_values) or not all(finite_values) or any(contract_values) or any(doctor_values) or any(residual_values) or any(instruction_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact finite proof-query context invocation required')
if any(instruction_values) and (not all(instruction_values) or args.preflight or any(contract_values) or any(doctor_values) or args.purpose!='coding'):raise SystemExit('exact public instruction binding requires router coding')
if any(contract_values) and (not all(contract_values) or any(doctor_values) or any(residual_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact local contract candidate invocation required')
if any(doctor_values) and (not all(doctor_values) or args.preflight or args.semantic_repository is not None or any(residual_values)):raise SystemExit('exact Doctor handoff or router invocation required')
if any(residual_values) and (not all(residual_values) or args.preflight or args.semantic_repository is None or args.purpose!='coding'):raise SystemExit('exact Doctor residual context requires semantic coding route')
if args.semantic_transport=='native' and (args.semantic_repository is None or args.purpose!='coding' or args.preflight or any(doctor_values) or any(contract_values) or any(finite_values)):raise SystemExit('native semantic transport requires semantic router coding')
if args.intent_requirements_transport=='compact' and (args.semantic_repository is None or not all(instruction_values) or args.purpose!='coding' or args.preflight or any(doctor_values) or any(contract_values) or any(finite_values)):raise SystemExit('compact Intent requirements transport requires indexed public instruction router coding')
if args.code_retrieval_codec=='compact' and (args.semantic_transport!='native' or args.semantic_repository is None or not all(instruction_values) or args.purpose!='coding' or args.preflight or any(doctor_values) or any(contract_values) or any(finite_values) or any(proof_query_values)):raise SystemExit('compact code retrieval codec requires indexed native public instruction router coding')
if args.code_retrieval_wire_delivery=='materialized' and (args.code_retrieval_codec!='compact' or args.semantic_transport!='native' or args.semantic_repository is None or not all(instruction_values) or args.purpose!='coding' or args.preflight or any(doctor_values) or any(contract_values) or any(finite_values) or any(proof_query_values)):raise SystemExit('materialized code retrieval wire delivery requires indexed native compact public instruction router coding')
if args.observe_model_input_only and (args.semantic_repository is None or not all(instruction_values) or args.purpose!='coding' or args.preflight or any(doctor_values) or any(contract_values) or any(finite_values) or any(proof_query_values)):raise SystemExit('model input observation requires bound router coding')
if not 1<=args.timeout<=300 or not 0<=args.preflight_sleep<=30:raise SystemExit('bounded timeout required')
if args.preflight:
 prompt=sys.stdin.buffer.read(2*1024*1024+1)
 if len(prompt)>2*1024*1024:raise SystemExit('preflight prompt exceeds bound')
 (pathlib.Path.cwd()/'worker-preflight-prompt.txt').write_bytes(prompt)
 target=pathlib.Path.cwd()/'worker-preflight.txt';target.write_text('worker candidate edit\n')
 denied=[]
 for private in (root/'state/preflight-key',root/'state/preflight-owner.duckdb'):
  for mode in ('rb','ab'):
   try:
    with private.open(mode):pass
   except PermissionError:denied.append(str(private)+':'+mode)
   else:raise SystemExit('private owner authority accessible')
 marker=pathlib.Path.cwd()/'.git'
 for action in ('write','unlink'):
  try:
   if action=='write':marker.open('ab').close()
   else:marker.unlink()
  except PermissionError:denied.append('.git:'+action)
  else:raise SystemExit('worker can mutate Git worktree authority')
 import subprocess
 top=subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()
 if pathlib.Path(top)!=pathlib.Path.cwd():raise SystemExit('worker cannot read exact Git worktree identity')
 if args.preflight_sleep:subprocess.Popen(['/bin/sleep','30'],start_new_session=True)
 print(json.dumps({'schema':'container-worker-preflight@1','pid':os.getpid(),'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups(),'boundary':evidence,'candidate_write':target.read_text()=='worker candidate edit\n','private_denials':denied,'workspace':str(pathlib.Path.cwd()),'prompt_bytes':len(prompt),'prompt_sha256':hashlib.sha256(prompt).hexdigest(),'original_task_python':subprocess.check_output(['python3','--version'],text=True).strip(),'runtime_python':sys.version.split()[0],'provider_calls':0}),flush=True)
 if args.preflight_sleep:time.sleep(args.preflight_sleep)
elif args.finite_proof_query_context is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_context import main
 sys.argv=[sys.argv[0],'--artifact',str(args.finite_repository_artifact),'--sha256',args.finite_repository_sha256,'--task-cid',args.finite_repository_task_cid,'--context',str(args.finite_proof_query_context),'--context-sha256',args.finite_proof_query_sha256,'--context-cid',args.finite_proof_query_context_cid]
 raise SystemExit(main())
elif args.finite_repository_artifact is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner import main
 sys.argv=[sys.argv[0],'--artifact',str(args.finite_repository_artifact),'--sha256',args.finite_repository_sha256,'--task-cid',args.finite_repository_task_cid]
 raise SystemExit(main())
elif args.doctor_contract_artifact is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import main
 sys.argv=[sys.argv[0],'--artifact',str(args.doctor_contract_artifact),'--sha256',args.doctor_contract_sha256,'--task-cid',args.doctor_contract_task_cid]
 raise SystemExit(main())
elif args.doctor_candidate_artifact is not None:
 from ipfs_accelerate_py.agent_supervisor.runtime.doctor_candidate_runner import main
 sys.argv=[sys.argv[0],'--artifact',str(args.doctor_candidate_artifact),'--sha256',args.doctor_candidate_sha256,'--task-cid',args.doctor_task_cid]
 raise SystemExit(main())
else:
 from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main
 sys.argv=[sys.argv[0],'--model',args.model,'--reasoning-effort',args.reasoning_effort,'--purpose',args.purpose,'--timeout',str(args.timeout),'--max-output-tokens',str(args.max_output_tokens),'--container-boundary',str(artifact),'--container-boundary-sha256',digest]
 sys.argv+=['--intent-requirements-transport',args.intent_requirements_transport]
 sys.argv+=['--code-retrieval-codec',args.code_retrieval_codec]
 sys.argv+=['--code-retrieval-wire-delivery',args.code_retrieval_wire_delivery]
 if args.semantic_repository is not None:sys.argv+=['--semantic-repository',str(args.semantic_repository),'--semantic-transport',args.semantic_transport]
 if args.doctor_residual_artifact is not None:sys.argv+=['--doctor-residual-artifact',str(args.doctor_residual_artifact),'--doctor-residual-sha256',args.doctor_residual_sha256,'--doctor-residual-task-cid',args.doctor_residual_task_cid]
 if args.public_instruction_artifact is not None:sys.argv+=['--public-instruction-artifact',str(args.public_instruction_artifact),'--public-instruction-sha256',args.public_instruction_sha256,'--public-instruction-task-cid',args.public_instruction_task_cid]
 if args.checked_context_artifact is not None:sys.argv+=['--checked-context-artifact',str(args.checked_context_artifact),'--checked-context-sha256',args.checked_context_sha256,'--checked-context-task-cid',args.checked_context_task_cid]
 if args.observe_model_input_only:sys.argv+=['--observe-model-input-only']
 raise SystemExit(main())
"""


class WorkerWorktreeObservationError(RuntimeError):
    """The fixed same-UID worker metadata observation failed closed."""


_GIT_METADATA_READ_KEY = object()
_GIT_OPTIONAL_QUERY_LOCK = threading.RLock()
_GIT_OPTIONAL_QUERY_ACTIVE = threading.local()


class WorkerGitMetadataReadGuard:
    """Original retained custody of one narrow read permission transition."""
    def __init__(self, rows, receipt, *, _key=None):
        if _key is not _GIT_METADATA_READ_KEY:
            raise WorkerWorktreeObservationError("original Git read-grant factory required")
        self._rows = tuple(rows)
        self._receipt_wire = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self._pid, self._secret, self._closed = os.getpid(), secrets.token_bytes(32), False
        self._query_active, self._query_lock = None, _GIT_OPTIONAL_QUERY_LOCK
        self._seal_state()

    def _state(self):
        return json.dumps({"self": id(self), "pid": self._pid, "rows": self._rows,
            "receipt": self._receipt_wire.decode(), "closed": self._closed,
            "query_active": self._query_active, "query_lock": id(self._query_lock)},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def _seal_state(self):
        self._seal = hmac.new(self._secret, self._state(), hashlib.sha256).digest()

    def _identity(self, *, open_required=True):
        if (type(self) is not WorkerGitMetadataReadGuard or self._pid != os.getpid()
                or not hmac.compare_digest(self._seal, hmac.new(self._secret, self._state(), hashlib.sha256).digest())
                or open_required and self._closed):
            raise WorkerWorktreeObservationError("original active Git read-grant custody required")

    @property
    def receipt(self):
        self._identity()
        return json.loads(self._receipt_wire)

    def readonly_query_environment(self):
        """Prevent optional Git status writes in this original checked query seam.

        The process environment is shared. This lock serializes participating
        contexts; unrelated writers remain subject to every strict FD check.
        """
        from contextlib import contextmanager
        self._identity()
        if (self._query_active is not None or self._query_lock is not _GIT_OPTIONAL_QUERY_LOCK
                or getattr(_GIT_OPTIONAL_QUERY_ACTIVE, "token", None) is not None):
            raise WorkerWorktreeObservationError("original nonnested Git query context required")
        @contextmanager
        def query():
            self._identity()
            if (self._query_active is not None or self._query_lock is not _GIT_OPTIONAL_QUERY_LOCK
                    or getattr(_GIT_OPTIONAL_QUERY_ACTIVE, "token", None) is not None):
                raise WorkerWorktreeObservationError("original nonnested Git query context required")
            with self._query_lock:
                self._identity()
                if self._query_active is not None or self._query_lock is not _GIT_OPTIONAL_QUERY_LOCK:
                    raise WorkerWorktreeObservationError("original nonnested Git query context required")
                self.require_current()
                present = "GIT_OPTIONAL_LOCKS" in os.environ
                previous = os.environ.get("GIT_OPTIONAL_LOCKS")
                token = (os.getpid(), threading.get_ident(), id(self), id(self._query_lock))
                self._query_active = token
                _GIT_OPTIONAL_QUERY_ACTIVE.token = token
                self._seal_state()
                primary = None
                try:
                    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
                    yield
                    self._identity()
                    if (self._query_active is not token or self._query_lock is not _GIT_OPTIONAL_QUERY_LOCK
                            or getattr(_GIT_OPTIONAL_QUERY_ACTIVE, "token", None) is not token):
                        raise WorkerWorktreeObservationError("original Git query context was rebound")
                    if os.environ.get("GIT_OPTIONAL_LOCKS") != "0":
                        raise WorkerWorktreeObservationError("readonly Git query environment changed")
                    self.require_current()
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    cleanup_error = None
                    try:
                        self._identity()
                        if (self._query_active is not token or self._query_lock is not _GIT_OPTIONAL_QUERY_LOCK
                                or getattr(_GIT_OPTIONAL_QUERY_ACTIVE, "token", None) is not token):
                            raise WorkerWorktreeObservationError("original Git query context was rebound")
                        if os.environ.get("GIT_OPTIONAL_LOCKS") != "0":
                            raise WorkerWorktreeObservationError("readonly Git query environment changed")
                    except BaseException as error:
                        cleanup_error = error
                    finally:
                        # Restoration uses the retained entry values even when
                        # the query failed, without promoting mutated custody.
                        if present:
                            os.environ["GIT_OPTIONAL_LOCKS"] = previous
                        else:
                            os.environ.pop("GIT_OPTIONAL_LOCKS", None)
                    try:
                        self._identity()
                        if (self._query_active is not token or self._query_lock is not _GIT_OPTIONAL_QUERY_LOCK
                                or getattr(_GIT_OPTIONAL_QUERY_ACTIVE, "token", None) is not token):
                            raise WorkerWorktreeObservationError("original Git query context was rebound")
                        self._query_active = None
                        del _GIT_OPTIONAL_QUERY_ACTIVE.token
                        self._seal_state()
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
                    if primary is None and cleanup_error is not None:
                        raise cleanup_error
        return query()

    def _drift(self, *, phase, name, kind, expected, retained, path_stat, fresh=None,
               canonical=True, expected_sha256=None, observed_sha256=None):
        """Bounded selected metadata only; this never excuses custody drift."""
        self._identity()
        field_names = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
                       "st_size", "st_mtime_ns", "st_ctime_ns")
        fields = {field_names[index] for actual in (retained, path_stat, fresh) if actual is not None
                  for index, (before, after) in enumerate(zip(expected, actual)) if before != after}
        if not canonical:
            fields.add("canonical_path")
        if expected_sha256 != observed_sha256:
            fields.add("sha256")
        receipt = json.loads(self._receipt_wire)
        if name == receipt["worktree"] + "/.git":
            role, relative = "allocated_git_pointer", ".git"
        else:
            role = "canonical_git_metadata"
            relative = str(Path(name).relative_to(Path(receipt["canonical_repository"])))
        def metadata(value):
            return dict(zip(field_names, value)) if value is not None else None
        diagnostic = {"schema": "checked-worker-git-metadata-drift@1", "phase": phase,
            "path_role": role, "relative_path": relative, "kind": kind,
            "canonical_path_changed": not canonical,
            "changed_fields": sorted(fields), "expected_stat": metadata(expected),
            "current_fd_stat": metadata(retained), "current_path_stat": metadata(path_stat),
            "current_fresh_stat": metadata(fresh), "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256, "metadata_only": True,
            "execution_authority": False, "proof_authority": False, "completion_authority": False,
            "source_body_executed": False, "provider_calls": 0}
        raw = json.dumps(diagnostic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 16384 or not 0 < len(relative.encode()) <= 4096:
            raise WorkerWorktreeObservationError("Git custody diagnostic exceeded bound")
        error = WorkerWorktreeObservationError("retained Git metadata custody changed")
        error.git_metadata_drift = json.loads(raw)
        raise error

    def require_current(self):
        self._identity()
        for name, fd, before, granted, digest, kind in self._rows:
            path = Path(name)
            canonical = path.resolve(strict=True) == path
            retained, path_stat = _git_stat(os.fstat(fd)), _git_stat(path.lstat())
            if not canonical or retained != tuple(granted) or path_stat != tuple(granted):
                self._drift(phase="retained", name=name, kind=kind, expected=granted,
                            retained=retained, path_stat=path_stat, canonical=canonical)
            if kind != "directory":
                observed_digest = hashlib.sha256(os.pread(fd, granted[6] + 1, 0)).hexdigest()
                if observed_digest != digest:
                    self._drift(phase="retained_bytes", name=name, kind=kind, expected=granted,
                        retained=_git_stat(os.fstat(fd)), path_stat=_git_stat(path.lstat()),
                        expected_sha256=digest, observed_sha256=observed_digest)
            fresh = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                            | (os.O_DIRECTORY if kind == "directory" else 0))
            try:
                fresh_first = _git_stat(os.fstat(fresh))
                if fresh_first != tuple(granted):
                    self._drift(phase="fresh", name=name, kind=kind, expected=granted,
                        retained=_git_stat(os.fstat(fd)), path_stat=_git_stat(path.lstat()), fresh=fresh_first)
                observed_digest = hashlib.sha256(os.pread(fresh, granted[6] + 1, 0)).hexdigest() if kind != "directory" else None
                fresh_last = _git_stat(os.fstat(fresh))
                retained, path_stat = _git_stat(os.fstat(fd)), _git_stat(path.lstat())
                if (fresh_first != tuple(granted) or observed_digest != digest
                        or fresh_last != tuple(granted) or retained != tuple(granted) or path_stat != tuple(granted)):
                    self._drift(phase="fresh", name=name, kind=kind, expected=granted,
                        retained=retained, path_stat=path_stat,
                        fresh=fresh_first if fresh_first != tuple(granted) else fresh_last,
                        expected_sha256=digest, observed_sha256=observed_digest)
            finally:
                os.close(fresh)

    __call__ = require_current

    def close(self):
        self._identity(open_required=False)
        if self._query_active is not None:
            raise WorkerWorktreeObservationError("close the original Git query context before custody")
        if self._closed:
            return
        errors = []
        for name, fd, before, granted, digest, kind in reversed(self._rows):
            try:
                current = os.fstat(fd)
                if (current.st_dev, current.st_ino) != (granted[0], granted[1]):
                    raise WorkerWorktreeObservationError("original Git custody descriptor was replaced")
                os.close(fd)
            except OSError as error:
                if error.errno != 9:
                    errors.append(error)
            except WorkerWorktreeObservationError as error:
                errors.append(error)
        self._closed = True
        self._seal_state()
        if errors:
            raise WorkerWorktreeObservationError("Git custody close did not close every original descriptor") from errors[0]


def _git_stat(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def grant_worker_git_metadata_readonly(*, worktree, canonical_repository):
    """Grant group reads to one native registration and current loose object DAG.

    The caller must retain this original guard under its native claim fence.
    No branch/history enumeration, packed-object fallback or group write occurs.
    """
    def require(value, message):
        if not value:
            raise WorkerWorktreeObservationError(message)
    require(os.getuid() == os.geteuid() == 1000, "original supervisor UID required for Git read grant")
    repository, workspace = Path(canonical_repository), Path(worktree)
    for path in (repository, workspace):
        require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir() and not path.is_symlink(),
                "canonical existing repository and allocated worktree required")
    require(repository != workspace and workspace.is_relative_to(Path("/opt/ipfs-supervisor/worktrees"))
            and workspace != Path("/opt/ipfs-supervisor/worktrees"), "fixed native allocated Git worktree required")
    git = repository / ".git"
    rows, by_path, contents = [], {}, {}
    directory_domains = {"metadata": set(), "objects": set()}
    total_object_bytes = 0
    def capture(path, kind, *, maximum=1024*1024, domain="metadata", optional=False):
        if str(path) in by_path:
            return contents.get(str(path))
        if optional and not os.path.lexists(path):
            return None
        require(path.resolve(strict=True) == path, "no symlink in selected Git read metadata")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                     | (os.O_DIRECTORY if kind == "directory" else 0))
        try:
            before = os.fstat(fd)
            mode = stat.S_IMODE(before.st_mode)
            root_readable = before.st_uid == 0 and before.st_gid == 1000 and bool(mode & 0o040)
            require((before.st_uid == 1000 and before.st_gid == 1000) or root_readable,
                    "selected Git metadata must belong to its supervisor or readable deployer")
            if kind == "directory":
                require(stat.S_ISDIR(before.st_mode) and mode in {0o700, 0o750, 0o755},
                        "protected exact Git metadata directory mode required")
                directory_domains[domain].add(str(path))
                require(len(directory_domains[domain]) <= (8 if domain == "metadata" else 129),
                        "selected Git directory bound exceeded")
                raw = None
            else:
                allowed = {0o400, 0o440, 0o444, 0o644} if domain == "objects" else {0o600, 0o640, 0o440, 0o444, 0o644}
                require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and mode in allowed
                        and 0 < before.st_size <= maximum, "protected bounded single-link Git metadata required")
                raw = os.pread(fd, maximum + 1, 0)
                require(len(raw) == before.st_size, "selected Git metadata changed during read")
                require(sum(row[5] != "directory" and row[0] not in object_paths for row in rows) < 16
                        or domain == "objects", "Git registration metadata file bound exceeded")
            require(_git_stat(before) == _git_stat(os.fstat(fd)) == _git_stat(path.lstat()),
                    "selected Git metadata custody changed during capture")
            row = [str(path), fd, _git_stat(before), None,
                   hashlib.sha256(raw).hexdigest() if raw is not None else None,
                   "directory" if kind == "directory" else domain]
            rows.append(row); by_path[str(path)] = row; contents[str(path)] = raw
            return raw
        except BaseException:
            os.close(fd)
            raise
    def directory(path, domain="metadata"):
        capture(path, "directory", domain=domain)
    object_paths, objects, visited_trees = set(), {}, set()
    def oid(value):
        require(type(value) is str and re.fullmatch(r"[0-9a-f]{40}", value), "exact SHA1 Git object identity required")
        return value
    def object_data(identifier, expected_type):
        nonlocal total_object_bytes
        identifier = oid(identifier)
        if identifier in objects:
            require(objects[identifier][0] == expected_type, "selected Git object type disagrees")
            return objects[identifier][1]
        require(len(objects) < 128, "current Git object closure exceeds bound")
        directory(git / "objects", "objects")
        fanout = git / "objects" / identifier[:2]
        directory(fanout, "objects")
        path = fanout / identifier[2:]
        object_paths.add(str(path))
        raw = capture(path, "file", maximum=8*1024*1024, domain="objects")
        inflater = zlib.decompressobj()
        try:
            decoded = inflater.decompress(raw, 8*1024*1024+1)
        except zlib.error as error:
            raise WorkerWorktreeObservationError("malformed selected loose Git object") from error
        require(len(decoded) <= 8*1024*1024 and inflater.eof and not inflater.unused_data
                and not inflater.unconsumed_tail, "bounded complete loose Git object required; packed objects unsupported")
        header, separator, body = decoded.partition(b"\0")
        require(separator and b" " in header and hashlib.sha1(decoded).hexdigest() == identifier,
                "loose Git object identity differs")
        actual_type, size = header.split(b" ", 1)
        require(actual_type == expected_type.encode() and size.isdigit() and str(len(body)).encode() == size,
                "loose Git object type or size differs")
        total_object_bytes += len(decoded)
        require(total_object_bytes <= 64*1024*1024, "current Git object byte closure exceeds bound")
        objects[identifier] = (expected_type, body)
        return body
    tree_entries = 0
    def tree_data(identifier, depth=0):
        nonlocal tree_entries
        require(depth <= 16, "current Git tree depth exceeds bound")
        if identifier in visited_trees:
            return
        visited_trees.add(identifier)
        data = object_data(identifier, "tree")
        offset = 0
        while offset < len(data):
            end = data.find(b"\0", offset)
            require(end > offset and end + 21 <= len(data), "bounded closed Git tree entry required")
            entry = data[offset:end]; mode, separator, name = entry.partition(b" ")
            require(separator and name not in {b"", b".", b"..", b".git"}
                    and not any(char in name for char in (b"/", b"\\", b"\n", b"\r")), "exact Git tree entry path required")
            name.decode("utf-8", errors="strict")
            child = data[end+1:end+21].hex(); offset = end + 21; tree_entries += 1
            require(tree_entries <= 128, "current Git tree entry bound exceeded")
            if mode == b"40000":
                tree_data(child, depth+1)
            else:
                require(mode in {b"100644", b"100755"}, "Git symlink/submodule tree entries unsupported")
                object_data(child, "blob")
    try:
        marker_raw = capture(workspace / ".git", "file", maximum=4096)
        require(marker_raw.startswith(b"gitdir: ") and marker_raw.endswith(b"\n") and marker_raw.count(b"\n") == 1,
                "exact native allocated Git registration pointer required")
        registration = Path(marker_raw[8:-1].decode("utf-8"))
        require(registration.is_absolute() and registration.parent == git / "worktrees"
                and registration.name not in {"", ".", ".."} and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", registration.name),
                "allocated Git pointer outside exact canonical registration")
        directory(git); directory(git / "worktrees"); directory(registration)
        head = capture(registration / "HEAD", "file", maximum=4096)
        backlink = capture(registration / "gitdir", "file", maximum=4096)
        common = capture(registration / "commondir", "file", maximum=4096)
        capture(registration / "index", "file", maximum=2*1024*1024)
        require(backlink == (str(workspace / ".git") + "\n").encode() and common == b"../..\n",
                "exact Git registration backlink and common directory required")
        require(head.startswith(b"ref: refs/heads/implementation/") and head.endswith(b"\n")
                and head.count(b"\n") == 1, "exact native implementation HEAD required")
        reference = head[5:-1].decode("ascii")
        parts = reference.split("/")
        require(len(parts) <= 7 and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                and part not in {".", ".."} and ".." not in part and not part.endswith(".lock") for part in parts),
                "closed native implementation branch reference required")
        parent = git
        for part in parts[:-1]:
            parent /= part; directory(parent)
        branch_raw = capture(git / reference, "file", maximum=128)
        selected_commit = oid(branch_raw.decode("ascii").removesuffix("\n"))
        require(branch_raw == (selected_commit + "\n").encode(), "exact implementation commit reference required")
        canonical_head = capture(git / "HEAD", "file", maximum=4096)
        if canonical_head.startswith(b"ref: ") and canonical_head.endswith(b"\n"):
            canonical_ref = canonical_head[5:-1].decode("ascii")
            require(re.fullmatch(r"refs/heads/[A-Za-z0-9_.-]+", canonical_ref)
                    and ".." not in canonical_ref and not canonical_ref.endswith(".lock"), "bounded canonical baseline reference required")
            canonical_commit = capture(git / canonical_ref, "file", maximum=128)
        else:
            canonical_commit = canonical_head
        require(canonical_commit == (selected_commit + "\n").encode(), "allocated and canonical Git baseline commits differ")
        for name in ("ORIG_HEAD", "config.worktree"):
            capture(registration / name, "file", maximum=65536, optional=True)
        if os.path.lexists(registration / "logs"):
            directory(registration / "logs")
            capture(registration / "logs/HEAD", "file", maximum=1024*1024, optional=True)
        commit = object_data(selected_commit, "commit")
        first_line = commit.split(b"\n", 1)[0]
        require(first_line.startswith(b"tree "), "selected current Git commit lacks its tree")
        root_tree = oid(first_line[5:].decode("ascii"))
        tree_data(root_tree)
        # Capture every selected object before introducing any group read.
        for row in rows:
            path, fd, before, _, digest, kind = row
            require(_git_stat(os.fstat(fd)) == tuple(before) == _git_stat(Path(path).lstat()),
                    "original selected Git custody changed before read grant")
            if kind != "directory":
                require(hashlib.sha256(os.pread(fd, before[6]+1, 0)).hexdigest() == digest,
                        "original selected Git bytes changed before read grant")
        for row in rows:
            path, fd, before, _, digest, kind = row
            mode = stat.S_IMODE(before[2])
            desired = 0o750 if kind == "directory" and mode == 0o700 else mode | 0o040
            require(not desired & 0o022 and desired & 0o700 == mode & 0o700,
                    "Git group read grant must preserve owner rights and forbid group writes")
            if desired != mode:
                require(before[3] == 1000, "cannot alter root-owned Git metadata permissions")
                os.fchmod(fd, desired)
            granted = _git_stat(os.fstat(fd))
            require(granted[:2] == tuple(before[:2]) and granted[3:8] == tuple(before[3:8])
                    and stat.S_IMODE(granted[2]) == desired and stat.S_IFMT(granted[2]) == stat.S_IFMT(before[2]),
                    "Git read permission transition changed original custody")
            row[3] = granted
        receipt = {"schema": "checked-worker-git-metadata-read-grant@1", "worktree": str(workspace),
            "canonical_repository": str(repository), "registration": str(registration),
            "selected_branch": reference.removeprefix("refs/heads/"), "selected_commit": selected_commit,
            "current_tree": root_tree, "scope": "exact_native_worktree_registration_current_commit_tree_blob_closure",
            "parent_history_traversed": False, "packed_objects_supported": False,
            "metadata_directories": len(directory_domains["metadata"]), "object_directories": len(directory_domains["objects"]),
            "current_objects": len(objects), "current_object_bytes": total_object_bytes,
            "files": [{"path": row[0], "bytes": row[3][6], "sha256": row[4], "kind": row[5],
                "original_mode": stat.S_IMODE(row[2][2]), "granted_mode": stat.S_IMODE(row[3][2]),
                "permissions_changed": row[2][2] != row[3][2]} for row in rows if row[5] != "directory"],
            "directories": [{"path": row[0], "original_mode": stat.S_IMODE(row[2][2]),
                "granted_mode": stat.S_IMODE(row[3][2]), "permissions_changed": row[2][2] != row[3][2]}
                for row in rows if row[5] == "directory"],
            "execution_authority": False, "proof_authority": False, "completion_authority": False,
            "source_body_executed": False, "provider_calls": 0}
        guard = WorkerGitMetadataReadGuard(rows, receipt, _key=_GIT_METADATA_READ_KEY)
        guard.require_current()
        return guard
    except BaseException:
        for row in reversed(rows):
            os.close(row[1])
        raise


def _worker_entry_custody():
    """Retain the exact installed root-protected entry, never a caller helper."""
    path = Path("/opt/ipfs-supervisor/bin/worker-entry")
    def witness(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    def parents():
        if path.resolve(strict=True) != path:
            raise WorkerWorktreeObservationError("exact installed worker entry required")
        for ancestor in path.parents:
            info = ancestor.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                raise WorkerWorktreeObservationError("root-protected worker entry ancestor required")
    parents()
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        expected = WORKER_ENTRY.encode("utf-8")
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o555 or before.st_size != len(expected)
                or not 0 < before.st_size <= 128 * 1024):
            raise WorkerWorktreeObservationError("protected exact installed worker entry required")
        original = witness(before)
        def verify():
            parents()
            if (witness(os.fstat(fd)) != original or witness(path.lstat()) != original
                    or os.pread(fd, len(expected) + 1, 0) != expected):
                raise WorkerWorktreeObservationError("installed worker entry custody changed")
            fresh = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            try:
                if (witness(os.fstat(fresh)) != original or os.pread(fresh, len(expected) + 1, 0) != expected
                        or witness(os.fstat(fresh)) != original or witness(os.fstat(fd)) != original
                        or witness(path.lstat()) != original):
                    raise WorkerWorktreeObservationError("fresh installed worker entry custody changed")
            finally:
                os.close(fresh)
        verify()
        return fd, verify
    except BaseException:
        os.close(fd)
        raise


def observe_worker_kernel_worktree(*, worker_identity, expected_worktree, timeout_seconds=3.0):
    """Observe one exact live coding worker's cwd through the existing UID route.

    This returns metadata only. It cannot dispatch input, execute repository code,
    choose another helper or replace the caller's native birth and claim checks.
    """
    def require(value, message):
        if not value:
            raise WorkerWorktreeObservationError(message)
    require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
            and 0 < timeout_seconds <= 3, "finite original-window helper timeout required")
    timeout_seconds = float(timeout_seconds)
    fields = {"pid", "uid", "parent_pid", "start_time_ticks", "process_birth_id", "command"}
    require(type(worker_identity) is dict and set(worker_identity) == fields,
            "closed current worker kernel identity required")
    require(all(type(worker_identity[name]) is int for name in ("pid", "uid", "parent_pid", "start_time_ticks"))
            and type(worker_identity["process_birth_id"]) is str and type(worker_identity["command"]) is list
            and all(type(part) is str for part in worker_identity["command"]),
            "literal closed worker kernel identity fields required")
    identity = json.loads(json.dumps(worker_identity, allow_nan=False))
    require(all(type(identity[name]) is int and 0 < identity[name] < 2**63
                for name in ("pid", "parent_pid", "start_time_ticks"))
            and identity["pid"] < 2**31 and type(identity["uid"]) is int and identity["uid"] == 1001,
            "exact worker PID, UID and birth numbers required")
    require(type(identity["process_birth_id"]) is str
            and re.fullmatch(r"birth:[0-9a-f]{32}", identity["process_birth_id"]), "exact worker birth digest required")
    command = identity["command"]
    marker = "/opt/ipfs-supervisor/bin/worker-entry"
    require(type(command) is list and 1 <= len(command) <= 80
            and all(type(part) is str and 0 < len(part.encode("utf-8")) <= 8192
                    and not any(c in part for c in ("\0", "\n", "\r")) for part in command)
            and command.count(marker) == 1 and "--observe-worker-worktree" not in command,
            "bounded original coding worker command required")
    command_raw = json.dumps(command, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    require(len(command_raw) <= 65536, "worker kernel command exceeds observation bound")
    require(type(expected_worktree) is str and 0 < len(expected_worktree.encode("utf-8")) <= 4096
            and not any(c in expected_worktree for c in ("\0", "\n", "\r")), "literal allocated worktree required")
    workspace = Path(expected_worktree)
    allowed = Path("/opt/ipfs-supervisor/worktrees")
    require(workspace.is_absolute() and str(workspace) == expected_worktree and ".." not in workspace.parts
            and workspace != allowed and workspace.is_relative_to(allowed)
            and workspace.resolve(strict=True) == workspace and workspace.is_dir() and not workspace.is_symlink(),
            "exact fixed-root allocated worktree required")
    fd, guard = _worker_entry_custody()
    try:
        guard()
        def unique(rows):
            value = {}
            for key, item in rows:
                require(key not in value, "duplicate worker metadata field")
                value[key] = item
            return value
        def closed_refusal(raw):
            if type(raw) is not bytes or not 0 < len(raw) <= 16384:
                return None
            try:
                value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
                canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                fields = {"schema", "stage", "error_type", "message_sha256", "failure_frames", "metadata_only",
                    "execution_authority", "proof_authority", "completion_authority", "source_body_executed", "provider_calls"}
                stages = {"profile", "privilege", "boundary", "self_privilege", "worktree_scope", "target_stat",
                    "target_status", "target_command", "target_birth_worktree", "readiness", "stable_observation", "output"}
                if (type(value) is not dict or set(value) != fields or canonical != raw
                        or value["schema"] != "checked-worker-kernel-worktree-refusal@1"
                        or type(value["stage"]) is not str or value["stage"] not in stages
                        or type(value["error_type"]) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}", value["error_type"])
                        or type(value["message_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["message_sha256"])
                        or type(value["failure_frames"]) is not list or len(value["failure_frames"]) > 16
                        or value["metadata_only"] is not True or any(value[name] is not False for name in
                            ("execution_authority", "proof_authority", "completion_authority", "source_body_executed"))
                        or type(value["provider_calls"]) is not int or value["provider_calls"] != 0):
                    return None
                for frame in value["failure_frames"]:
                    if (type(frame) is not dict or set(frame) != {"file", "function", "line"}
                            or type(frame["file"]) is not str or not 0 < len(frame["file"]) <= 256
                            or "/" in frame["file"] or "\\" in frame["file"]
                            or type(frame["function"]) is not str or not 0 < len(frame["function"]) <= 128
                            or any(not 32 <= ord(c) < 127 for c in frame["file"] + frame["function"])
                            or type(frame["line"]) is not int or not 0 <= frame["line"] <= 2**31 - 1):
                        return None
                return value
            except (ValueError, UnicodeError, TypeError, WorkerWorktreeObservationError):
                return None
        def failure(phase, returncode, raw):
            raw = raw if type(raw) is bytes else b""
            value = {"schema": "checked-worker-kernel-worktree-helper-failure@1", "phase": phase,
                "returncode": returncode if type(returncode) is int else None,
                "stdout_bytes": len(raw), "stdout_sha256": hashlib.sha256(raw).hexdigest(),
                "helper_sha256": hashlib.sha256(WORKER_ENTRY.encode("utf-8")).hexdigest(),
                "helper_bytes": len(WORKER_ENTRY.encode("utf-8")), "timeout_seconds": timeout_seconds,
                "helper_refusal": closed_refusal(raw) if phase == "subprocess_response" and returncode != 0 else None,
                "metadata_only": True, "execution_authority": False, "proof_authority": False,
                "completion_authority": False, "source_body_executed": False, "provider_calls": 0}
            error = WorkerWorktreeObservationError("bounded successful worker metadata response required")
            error.worker_worktree_failure = value
            return error
        try:
            result = subprocess.run(["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--", marker,
                "--observe-worker-worktree", str(identity["pid"]), str(identity["start_time_ticks"]),
                identity["process_birth_id"], expected_worktree], cwd="/", env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as error:
            guard()
            raise failure("subprocess_timeout", None, error.output) from error
        # The exact pinned helper emits one closed record bounded before print;
        # reject any unexpected subprocess output as well.
        guard()
        if not (type(result.returncode) is int and result.returncode == 0
                and type(result.stdout) is bytes and 0 < len(result.stdout) <= 16384):
            raise failure("subprocess_response", result.returncode, result.stdout)
        try:
            value = json.loads(result.stdout.decode("utf-8"), object_pairs_hook=unique)
            canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        except (ValueError, UnicodeError, TypeError) as error:
            raise WorkerWorktreeObservationError("canonical worker metadata response required") from error
        expected_fields = {"schema", "pid", "uid", "parent_pid", "start_time_ticks", "process_birth_id", "boot_id",
            "worktree", "command_sha256", "namespaces", "metadata_only", "execution_authority", "proof_authority",
            "completion_authority", "source_body_executed", "provider_calls"}
        require(type(value) is dict and set(value) == expected_fields and canonical == result.stdout
                and value["schema"] == "checked-worker-kernel-worktree-observation@1"
                and value["metadata_only"] is True and all(value[name] is False for name in
                    ("execution_authority", "proof_authority", "completion_authority", "source_body_executed"))
                and type(value["provider_calls"]) is int and value["provider_calls"] == 0,
                "closed data-only worker metadata response required")
        require(all(type(value[name]) is int and value[name] == identity[name]
                    for name in ("pid", "uid", "parent_pid", "start_time_ticks"))
                and value["process_birth_id"] == identity["process_birth_id"]
                and value["command_sha256"] == hashlib.sha256(command_raw).hexdigest()
                and value["worktree"] == expected_worktree and type(value["boot_id"]) is str
                and re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value["boot_id"]),
                "worker metadata response differs from captured birth or worktree")
        namespaces = value["namespaces"]
        require(type(namespaces) is dict and set(namespaces) == {"pid", "mnt", "net"}
                and all(type(namespaces[name]) is str and re.fullmatch(name + r":\[[0-9]+\]", namespaces[name])
                        for name in namespaces), "closed kernel namespace metadata required")
        require(workspace.resolve(strict=True) == workspace and workspace.is_dir() and not workspace.is_symlink(),
                "allocated observation worktree changed")
        guard()
        return value
    finally:
        try:
            guard()
        finally:
            os.close(fd)


async def install_worker_boundary(environment, *, container_name: str, output: Path,
                                  observer_only: bool = False) -> dict:
    """Install the real protected worker scripts, without calling a provider.

    Observer-only setup needs the existing supervisor UID1000, task Git root,
    runtime Python and sudo. It needs neither credentials nor Codex binaries.
    Those provider assets must be absent; this helper never deletes them to
    manufacture absence. The default profile retains the coding deployment.
    """
    if type(observer_only) is not bool:
        raise ValueError("worker deployment profile must be an exact Boolean")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    async def run(label, command, *, user="root", env=None, timeout=180):
        result = await environment.exec(
            command=command, cwd="/app", user=user, env=env, timeout_sec=timeout
        )
        (output / (label + ".log")).write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code:
            raise RuntimeError("worker deployment " + label + " failed")
        return result

    # Select only nonsecret engine identity fields, never Docker's environment.
    inspect = subprocess.check_output(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .Id}} {{json .Image}} {{json .State.Pid}} {{json .State.Running}} {{json .HostConfig.Privileged}}",
            container_name,
        ],
        text=True,
    ).split()
    cid, image, pid, running, privileged = [json.loads(x) for x in inspect]
    if not running or privileged:
        raise ValueError("running ordinary Docker task container required")
    namespaces = json.loads(
        (
            await run(
                "namespaces",
                "python3 -I -c "
                + shlex.quote(
                    "import json,os;print(json.dumps({k:os.readlink('/proc/self/ns/'+k) for k in ('pid','mnt','net')}))"
                ),
            )
        ).stdout
    )
    host_namespaces = {
        k: __import__("os").readlink(f"/proc/{pid}/ns/{k}") for k in ("pid", "mnt", "net")
    }
    if host_namespaces != namespaces:
        raise ValueError("Docker engine and observed namespace identities differ")
    if observer_only:
        await run("observer-only-provider-absence", "python3 -I -c " + shlex.quote(
            "import pathlib\nroot=pathlib.Path('/opt/ipfs-supervisor')\n" + PROFILE_GUARD
            + "\n_observer_absence(root,'installation')\n"))
    credential_setup = ("" if observer_only else
        f"install -o benchmarkworker -g supervisor -m 0600 {ROOT}/codex-auth/auth.json {ROOT}/worker-home/.codex/auth.json && ")
    system_setup = ("test -x /usr/bin/sudo && " if observer_only else
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sudo && ")
    await run(
        "worker-account",
        system_setup +
        f"useradd --uid 1001 --gid supervisor --home-dir {ROOT}/worker-home --shell /usr/sbin/nologin benchmarkworker && "
        f"install -d -o root -g root -m 0755 {ROOT}/bin {ROOT}/provider-bin && "
        f"install -d -o supervisor -g supervisor -m 0750 {ROOT}/worktrees && "
        f"install -d -o benchmarkworker -g supervisor -m 0700 {ROOT}/worker-home {ROOT}/worker-home/.codex {ROOT}/worker-home/context-wires && "
        + credential_setup +
        f"chmod 0700 {ROOT}/home {ROOT}/codex-auth {ROOT}/state",
    )
    wire_directory_check = """import pathlib,stat
root=pathlib.Path('/opt/ipfs-supervisor')
for path in (root, *root.parents):
 info=path.lstat()
 if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_mode&0o022:
  raise ValueError('protected runtime ancestor required')
for path in (root/'worker-home',root/'worker-home/context-wires'):
 info=path.lstat()
 if not stat.S_ISDIR(info.st_mode) or info.st_uid!=1001 or stat.S_IMODE(info.st_mode)!=0o700 or path.resolve(strict=True)!=path:
  raise ValueError('fixed private worker wire directory required')
"""
    await run("worker-wire-directory", "python3 -I -c " + shlex.quote(wire_directory_check))
    if not observer_only:
        await run("native-codex-binary", "python3 -I -c " + shlex.quote(native_codex_exposure_script()))
    boundary = {
        "schema": "supervisor-container-worker-boundary@1",
        "container_id": cid,
        "image_id": image,
        "namespaces": namespaces,
        "owner_uid": 1000,
        "worker_uid": 1001,
        "allowed_worktree_roots": [ROOT + "/worktrees"],
        "owner_private_paths": [ROOT + "/state", ROOT + "/home", ROOT + "/codex-auth"],
        "validation_repository_roots": ["/app"],
        "single_worker": True,
    }
    raw = (json.dumps(boundary, sort_keys=True, indent=2) + "\n").encode()
    profile = {"schema": PROFILE_SCHEMA, "observer_only": observer_only,
               "boundary_sha256": hashlib.sha256(raw).hexdigest()}
    profile_raw = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
    sudoers = (
        "Defaults:supervisor !use_pty\nsupervisor ALL=(benchmarkworker) NOPASSWD: "
        + ROOT
        + "/bin/worker-entry *\n"
    )
    with tempfile.TemporaryDirectory() as temporary:
        for name, data, target, mode in [
            ("boundary", raw, ROOT + "/container-boundary.json", "0444"),
            ("profile", profile_raw, PROFILE_PATH, "0444"),
            ("owner", OWNER_ENTRY.encode(), ROOT + "/bin/router-worker", "0555"),
            ("validation", VALIDATION_ENTRY.encode(), ROOT + "/bin/validation-worker", "0555"),
            ("worker", WORKER_ENTRY.encode(), ROOT + "/bin/worker-entry", "0555"),
            ("sudoers", sudoers.encode(), "/etc/sudoers.d/supervisor-worker", "0440"),
        ]:
            path = Path(temporary) / name
            path.write_bytes(data)
            await environment.upload_file(path, target)
            await run(
                "protect-" + name, "chown root:root " + target + " && chmod " + mode + " " + target
            )
    await run("sudoers-check", "visudo -cf /etc/sudoers.d/supervisor-worker")
    await run(
        "canonical-git-protect",
        "find /app/.git -type d -exec chmod 0750 {} + && find /app/.git -type f -exec chmod 0640 {} +",
    )
    result = {
        "schema": "container-worker-deployment@1",
        "boundary": boundary,
        "boundary_sha256": hashlib.sha256(raw).hexdigest(),
        "deployment_profile": profile,
        "deployment_profile_sha256": hashlib.sha256(profile_raw).hexdigest(),
        "deployment_profile_path": PROFILE_PATH,
        "observer_only": observer_only,
        "provider_assets_required": not observer_only,
        "provider_asset_absence_scope": (
            "root/owner checks of fixed provider directories and binaries; worker checks only accessible provider paths"
            if observer_only else None),
        "implementation_command": ROOT
        + "/bin/router-worker --model gpt-5.6-sol --reasoning-effort high --timeout 90",
        "candidate_runner_argv": [ROOT + "/bin/validation-worker"],
        "worker_worktree_root": ROOT + "/worktrees",
        "provider_calls": 0,
        "qualified": False,
    }
    (output / "installation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


async def deploy_worker_boundary(environment, *, output: Path, observer_only: bool = False) -> dict:
    """Fresh Harbor deployment: install and exercise the actual worker boundary."""
    if type(observer_only) is not bool:
        raise ValueError("worker deployment profile must be an exact Boolean")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Harbor's own service lookup resolves the original running task container;
    # the installer independently compares engine and in-container namespaces.
    container = await environment._platform._resolve_service_container("main")
    installed = await install_worker_boundary(
        environment, container_name=container, output=output / "installation", observer_only=observer_only
    )
    qualified = await qualify_worker_boundary(environment, output=output / "qualification")
    result = {**installed, "qualified": qualified["qualified"], "qualification": qualified}
    (output / "deployment.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


async def qualify_worker_boundary(environment, *, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    probe = r"""import json,os,pathlib,signal,subprocess,time
root=pathlib.Path('/opt/ipfs-supervisor');state=root/'state';candidate=root/'worktrees/provider-free'
subprocess.run(['git','-C','/app','worktree','add','--detach',str(candidate),'HEAD'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
(state/'preflight-key').write_text('synthetic-noncredential-private-check\n');(state/'preflight-key').chmod(0o600)
(state/'preflight-owner.duckdb').write_bytes(b'synthetic-owner-state');(state/'preflight-owner.duckdb').chmod(0o600)
command=[str(root/'bin/router-worker'),'--preflight']
completed=subprocess.run(command,cwd=candidate,text=True,capture_output=True,timeout=30)
if completed.returncode:raise RuntimeError(completed.stderr)
evidence=json.loads(completed.stdout)
if not evidence['candidate_write'] or len(evidence['private_denials'])!=6:raise ValueError('worker permission preflight failed')
if (candidate/'worker-preflight.txt').stat().st_uid!=1001:raise ValueError('actual worker did not write candidate')
# A second preflight has no caller-controlled code. Terminate its actual sudo
# monitor and require both monitor and exact observed worker process to exit.
(candidate/'worker-preflight.txt').unlink()
process=subprocess.Popen(command+['--preflight-sleep','30'],cwd=candidate,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
sleeping=json.loads(process.stdout.readline());worker=sleeping['pid']
os.kill(process.pid,signal.SIGTERM)
try:process.wait(timeout=10)
except subprocess.TimeoutExpired:
 os.killpg(process.pid,signal.SIGKILL);process.wait();raise
for _ in range(50):
 try:
  status=pathlib.Path('/proc')/str(worker)/'stat'
  if not status.exists() or status.read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(.1)
else:raise ValueError('worker survived owner termination')
remaining=[]
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  status=(path/'status').read_text();uid=next(x for x in status.splitlines() if x.startswith('Uid:')).split()[1]
  state_line=next(x for x in status.splitlines() if x.startswith('State:'))
  if uid=='1001' and 'Z' not in state_line:remaining.append(int(path.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):pass
if remaining:raise ValueError('live worker processes remain')
subprocess.run(['git','-C','/app','worktree','remove','--force',str(candidate)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
print(json.dumps({'qualified':True,'permission_evidence':evidence,'termination_worker_pid':worker,'remaining_worker_pids':remaining,'provider_calls':0,'completion_authority':False}))
"""
    result = await environment.exec(
        command=PYTHON + " -I -c " + shlex.quote(probe),
        cwd="/app",
        user="supervisor",
        env=runtime_environment(),
        timeout_sec=60,
    )
    (output / "probe.log").write_text((result.stdout or "") + (result.stderr or ""))
    if result.return_code:
        raise RuntimeError("provider-free worker qualification failed")
    evidence = json.loads(result.stdout)
    (output / "qualification.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return evidence
