"""Install and qualify the separate model-worker identity in a Harbor container.

All scripts and the namespace manifest are owned by container root. Model calls
are absent from qualification; the only delegated entry is the pinned router.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from .terminal_deployment import PYTHON, ROOT, native_codex_exposure_script, runtime_environment

OWNER_MONITOR = r"""
import signal,subprocess
process=None
def interrupted(signum,frame):
 raise InterruptedError('owner monitor interrupted')
for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)
code=1
try:
 process=subprocess.Popen(command)
 guard=globals().get("proof_query_guard")
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
 guard=globals().get("proof_query_guard")
 if guard is not None:guard.abort()
raise SystemExit(code)
"""

OWNER_ENTRY = (
    r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import json,os,pathlib,sys
root=pathlib.Path('/opt/ipfs-supervisor')
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
if any(part==flag or part.startswith(flag+'=') for part in sys.argv[1:] for flag in ('--finite-proof-query-context','--finite-proof-query-sha256','--finite-proof-query-context-cid')):
 sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
 from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_dispatch import require_finite_proof_query_isolated_worker_dispatch
 proof_query_guard=require_finite_proof_query_isolated_worker_dispatch(command=command,environment=dict(os.environ),worktree=cwd)
"""
    + OWNER_MONITOR
)

VALIDATION_ENTRY = (
    r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import os,sys
if os.getuid()!=1000 or os.geteuid()!=1000:raise SystemExit('owner identity required')
if len(sys.argv)<2:raise SystemExit('literal validation argv required')
command=['/usr/bin/sudo','-n','-u','benchmarkworker','--','/opt/ipfs-supervisor/bin/worker-entry','--validate',*sys.argv[1:]]
"""
    + OWNER_MONITOR
)

WORKER_ENTRY = r"""#!/opt/ipfs-supervisor/venv/bin/python -I
import argparse,ctypes,hashlib,json,os,pathlib,sys,time
root=pathlib.Path('/opt/ipfs-supervisor')
if os.getuid()!=1001 or os.geteuid()!=1001:raise SystemExit('worker identity required')
# Prevent this external-isolation worker from acquiring any new Unix privilege.
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'no-new-privileges failed')
os.umask(0o007)
os.environ.clear()
os.environ.update({'HOME':str(root/'worker-home'),'PATH':str(root/'provider-bin')+':/usr/local/bin:/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(root/'source')+':'+str(root/'datasets')+':'+str(root/'kit'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
os.environ.update({'PYTHON':str(root/'venv/bin/python'),'PYTHON_BIN':str(root/'venv/bin/python'),'GIT_CONFIG_COUNT':'3','GIT_CONFIG_KEY_0':'safe.directory','GIT_CONFIG_VALUE_0':str(pathlib.Path.cwd()),'GIT_CONFIG_KEY_1':'core.hooksPath','GIT_CONFIG_VALUE_1':'/dev/null','GIT_CONFIG_KEY_2':'core.fsmonitor','GIT_CONFIG_VALUE_2':'false'})
os.environ.update({'IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE':str(root/'venv/bin/python'),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONHASHSEED':'0'})
sys.path[:0]=[str(root/'source'),str(root/'datasets'),str(root/'kit')]
from ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary import verify_container_worker_boundary
artifact=root/'container-boundary.json'
digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
selected_provider=json.loads(artifact.read_bytes()).get('provider','codex_cli')
if selected_provider not in ('codex_cli','grok_cli'):raise SystemExit('unknown deployed provider')
if selected_provider=='grok_cli':
 os.environ['GROK_HOME']=str(root/'worker-home/.grok')
 os.environ['ipfs_accelerate_py_GROK_CLI_CMD']=str(root/'provider-bin/grok')
 os.environ.update({name:'0' for name in ('GROK_CODEX_AGENTS_ENABLED','GROK_CODEX_HOOKS_ENABLED','GROK_CODEX_MCPS_ENABLED','GROK_CODEX_RULES_ENABLED','GROK_CODEX_SESSIONS_ENABLED','GROK_CODEX_SKILLS_ENABLED')})
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
 if len(sys.argv)<3 or len(sys.argv)>130 or any(len(x)>8192 for x in sys.argv[2:]):raise SystemExit('bounded literal validation argv required')
 os.execvpe(sys.argv[2],sys.argv[2:],dict(os.environ))
parser=argparse.ArgumentParser()
parser.add_argument('--provider',choices=['codex_cli','grok_cli'],default=selected_provider)
parser.add_argument('--model',default='gpt-6.1-sol');parser.add_argument('--reasoning-effort',choices=['low','medium','high','xhigh','max'],default='high')
parser.add_argument('--timeout',type=int,default=90);parser.add_argument('--max-output-tokens',type=int,default=4096)
parser.add_argument('--purpose',choices=['planning','coding'],default='coding')
parser.add_argument('--semantic-repository',type=pathlib.Path)
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
parser.add_argument('--preflight',action='store_true');parser.add_argument('--preflight-sleep',type=int,default=0)
args=parser.parse_args()
if args.provider!=selected_provider:raise SystemExit('provider differs from deployed worker route')
doctor_values=(args.doctor_candidate_artifact,args.doctor_candidate_sha256,args.doctor_task_cid)
residual_values=(args.doctor_residual_artifact,args.doctor_residual_sha256,args.doctor_residual_task_cid)
contract_values=(args.doctor_contract_artifact,args.doctor_contract_sha256,args.doctor_contract_task_cid)
finite_values=(args.finite_repository_artifact,args.finite_repository_sha256,args.finite_repository_task_cid)
proof_query_values=(args.finite_proof_query_context,args.finite_proof_query_sha256,args.finite_proof_query_context_cid)
instruction_values=(args.public_instruction_artifact,args.public_instruction_sha256,args.public_instruction_task_cid)
if any(finite_values) and (not all(finite_values) or any(contract_values) or any(doctor_values) or any(residual_values) or any(instruction_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact finite repository candidate invocation required')
if any(proof_query_values) and (not all(proof_query_values) or not all(finite_values) or any(contract_values) or any(doctor_values) or any(residual_values) or any(instruction_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact finite proof-query context invocation required')
if any(instruction_values) and (not all(instruction_values) or args.preflight or any(contract_values) or any(doctor_values) or args.purpose!='coding'):raise SystemExit('exact public instruction binding requires router coding')
if any(contract_values) and (not all(contract_values) or any(doctor_values) or any(residual_values) or args.preflight or args.semantic_repository is not None or args.purpose!='coding'):raise SystemExit('exact local contract candidate invocation required')
if any(doctor_values) and (not all(doctor_values) or args.preflight or args.semantic_repository is not None or any(residual_values)):raise SystemExit('exact Doctor handoff or router invocation required')
if any(residual_values) and (not all(residual_values) or args.preflight or args.semantic_repository is None or args.purpose!='coding'):raise SystemExit('exact Doctor residual context requires semantic coding route')
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
 sys.argv=[sys.argv[0],'--provider',args.provider,'--model',args.model,'--reasoning-effort',args.reasoning_effort,'--purpose',args.purpose,'--timeout',str(args.timeout),'--max-output-tokens',str(args.max_output_tokens),'--container-boundary',str(artifact),'--container-boundary-sha256',digest]
 if args.semantic_repository is not None:sys.argv+=['--semantic-repository',str(args.semantic_repository)]
 if args.doctor_residual_artifact is not None:sys.argv+=['--doctor-residual-artifact',str(args.doctor_residual_artifact),'--doctor-residual-sha256',args.doctor_residual_sha256,'--doctor-residual-task-cid',args.doctor_residual_task_cid]
 if args.public_instruction_artifact is not None:sys.argv+=['--public-instruction-artifact',str(args.public_instruction_artifact),'--public-instruction-sha256',args.public_instruction_sha256,'--public-instruction-task-cid',args.public_instruction_task_cid]
 raise SystemExit(main())
"""


async def install_worker_boundary(environment, *, container_name: str, output: Path, provider: str = "codex_cli") -> dict:
    if provider not in {"codex_cli", "grok_cli"}:
        raise ValueError("explicit supported container provider required")
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
    provider_home = ".grok" if provider == "grok_cli" else ".codex"
    auth_home = "grok-auth" if provider == "grok_cli" else "codex-auth"
    await run(
        "worker-account",
        f"apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sudo && "
        f"useradd --uid 1001 --gid supervisor --home-dir {ROOT}/worker-home --shell /usr/sbin/nologin benchmarkworker && "
        f"install -d -o root -g root -m 0755 {ROOT}/bin {ROOT}/provider-bin && "
        f"install -d -o supervisor -g supervisor -m 0750 {ROOT}/worktrees && "
        f"install -d -o benchmarkworker -g supervisor -m 0700 {ROOT}/worker-home {ROOT}/worker-home/{provider_home} && "
        f"install -o benchmarkworker -g supervisor -m 0600 {ROOT}/{auth_home}/auth.json {ROOT}/worker-home/{provider_home}/auth.json && "
        f"chmod 0700 {ROOT}/home {ROOT}/{auth_home} {ROOT}/state",
    )
    if provider == "codex_cli":
        await run("native-codex-binary", "python3 -I -c " + shlex.quote(native_codex_exposure_script()))
    else:
        # Deployment already exposed and pinned this root-owned executable.
        from .terminal_grok_deployment import require_grok_version_output
        observed = await run("native-grok-version", ROOT + "/provider-bin/grok --version")
        require_grok_version_output(observed.stdout)
    boundary = {
        "schema": "supervisor-container-worker-boundary@1",
        "container_id": cid,
        "image_id": image,
        "namespaces": namespaces,
        "owner_uid": 1000,
        "worker_uid": 1001,
        "allowed_worktree_roots": [ROOT + "/worktrees"],
        "owner_private_paths": [ROOT + "/state", ROOT + "/home", ROOT + "/" + auth_home],
        "provider": provider,
        "validation_repository_roots": ["/app"],
        "single_worker": True,
    }
    raw = (json.dumps(boundary, sort_keys=True, indent=2) + "\n").encode()
    sudoers = (
        "Defaults:supervisor !use_pty\nsupervisor ALL=(benchmarkworker) NOPASSWD: "
        + ROOT
        + "/bin/worker-entry *\n"
    )
    with tempfile.TemporaryDirectory() as temporary:
        for name, data, target, mode in [
            ("boundary", raw, ROOT + "/container-boundary.json", "0444"),
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
        "implementation_command": ROOT
        + "/bin/router-worker --provider " + provider + " --model "
        + ("grok-4.7" if provider == "grok_cli" else "gpt-6.1-sol") + " --reasoning-effort high --timeout 90",
        "candidate_runner_argv": [ROOT + "/bin/validation-worker"],
        "worker_worktree_root": ROOT + "/worktrees",
        "provider_calls": 0,
        "qualified": False,
    }
    (output / "installation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


async def deploy_worker_boundary(environment, *, output: Path, provider: str = "codex_cli") -> dict:
    """Fresh Harbor deployment: install and exercise the actual worker boundary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Harbor's own service lookup resolves the original running task container;
    # the installer independently compares engine and in-container namespaces.
    container = await environment._platform._resolve_service_container("main")
    installed = await install_worker_boundary(
        environment, container_name=container, output=output / "installation", provider=provider
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
