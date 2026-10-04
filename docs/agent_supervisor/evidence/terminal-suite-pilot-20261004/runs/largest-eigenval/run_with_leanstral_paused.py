"""User-authorized service pause around one generic full supervisor task."""
from pathlib import Path
import json,os,re,signal,subprocess,sys,time,urllib.request,urllib.error
B=Path(__file__).resolve().parent
UNIT='ipfs-accelerate-leanstral.service'
HEALTH='http://172.17.0.1:8080/health'
log=dict(schema='benchmark-service-pause@1',unit=UNIT,user_authorized=True,
         source='explicit user reply: Temporarily stop and restart it',events=[],restored=False)
def save():
 p=B/'service-pause.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(log,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def event(phase,**values):
 log['events'].append(dict(phase=phase,at=time.time(),**values));save();print(json.dumps(dict(phase=phase,**values)),flush=True)
def state():
 p=subprocess.run(['systemctl','--user','show',UNIT,'--property=ActiveState','--property=SubState','--property=MainPID'],capture_output=True,text=True,timeout=15)
 if p.returncode:raise RuntimeError('service state query failed')
 return dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
def memory():
 data={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  key,value=line.split(':',1)
  if key in {'MemAvailable','MemFree','SwapFree'}:data[key+'_kib']=int(value.split()[0])
 data['memory_psi']=Path('/proc/pressure/memory').read_text().strip()
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 data['gpu_query_returncode']=p.returncode;data['gpu_allocations']=[]
 if p.returncode==0:
  for line in p.stdout.splitlines():
   fields=line.split(',')
   if len(fields)==2 and all(v.strip().isdigit() for v in fields):data['gpu_allocations'].append(dict(pid=int(fields[0]),memory_mib=int(fields[1])))
 return data
def health():
 try:
  with urllib.request.urlopen(HEALTH,timeout=5) as response:
   value=json.loads(response.read(4096));return response.status==200 and value.get('status')=='ok'
 except (OSError,ValueError,urllib.error.URLError):return False
def interrupted(signum,frame):raise InterruptedError('benchmark pause wrapper interrupted')
for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,interrupted)
assert not (B/'service-pause.json').exists()
assert (B/'build-audit.json').is_file() and not (B/'execute-exit.json').exists()
before=state();assert before['ActiveState']=='active'
event('before_pause',service=before,resources=memory(),model_ready=health())
stop_attempted=False;code=1;child=None
try:
 stop_attempted=True
 stopped=subprocess.run(['systemctl','--user','stop',UNIT],capture_output=True,text=True,timeout=105)
 after=state();event('paused',returncode=stopped.returncode,service=after,resources=memory())
 if stopped.returncode or after['ActiveState']!='inactive':raise RuntimeError('service pause failed')
 event('benchmark_execute_started')
 child=subprocess.Popen([sys.executable,str(B/'stage.py'),'execute'],cwd='/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002',start_new_session=True)
 pilot_start=json.loads((B.parent/'terminal-suite-pilot-20261004/service-pause.json').read_text())['events'][0]['at']
 remaining=3600-(time.time()-pilot_start)
 if remaining<120:raise RuntimeError('pilot time budget exhausted')
 result=child.wait(timeout=remaining)
 event('benchmark_execute_finished',returncode=result)
 if result:raise RuntimeError('benchmark execute failed')
 code=0
except BaseException as exc:
 event('failed',error_type=type(exc).__name__)
finally:
 if stop_attempted:
  # Finish restoration even after a cancellation of the benchmark wrapper.
  signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
  try:
   if child is not None and child.poll() is None:
    event('owned_process_cleanup_started')
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
     try:os.killpg(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     child.wait(timeout=15)
    event('owned_process_cleanup_finished',returncode=child.returncode)
   # Exact trial directory names are created by this invocation's Harbor job.
   # Remove only its corresponding container, never a broad task-name match.
   job=B/'largest-eigenval-01/jobs/supervisor-full-largest-eigenval'
   for trial in job.glob('largest-eigenval__*'):
    if not trial.is_dir() or trial.is_symlink() or re.fullmatch(r'largest-eigenval__[A-Za-z0-9]+',trial.name) is None:continue
    name=trial.name.lower()+'__env-main-1'
    found=subprocess.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],capture_output=True,text=True,timeout=15)
    if found.returncode:raise RuntimeError('owned container cleanup discovery failed')
    ids=found.stdout.split()
    if ids:
     cleaned=subprocess.run(['docker','rm','-f',*ids],capture_output=True,text=True,timeout=60)
     event('owned_container_cleanup',returncode=cleaned.returncode,count=len(ids))
  except BaseException as exc:
   code=1;event('owned_cleanup_error',error_type=type(exc).__name__)
  event('restoration_started')
  try:
   restored=subprocess.run(['systemctl','--user','start',UNIT],capture_output=True,text=True,timeout=1210)
   event('restart_returned',returncode=restored.returncode,service=state())
   deadline=time.monotonic()+1200;last_update=0
   while restored.returncode==0 and time.monotonic()<deadline:
    current=state()
    if current.get('ActiveState')=='active' and current.get('SubState')=='running' and int(current.get('MainPID','0'))>0 and health():
     log['restored']=True;event('restored',service=current,resources=memory(),model_ready=True);break
    if time.monotonic()-last_update>=30:
     event('waiting_for_model_readiness',service=state());last_update=time.monotonic()
    time.sleep(2)
   if not log['restored']:event('restoration_unqualified',service=state())
  except BaseException as exc:event('restoration_error',error_type=type(exc).__name__)
  if not log['restored']:code=1
 log['returncode']=code;save()
raise SystemExit(code)
