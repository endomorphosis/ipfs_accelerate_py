
import json,os,pathlib,pwd,subprocess,sys
root=pathlib.Path('/opt/ipfs-supervisor')
sys.path[:0]=[str(root/n) for n in ('source','datasets','kit')]
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY,WORKER_ENTRY
assert sys.argv[3] in ('full','selected')
mode_path=root/'fixture-mode.json';mode_path.write_text(json.dumps(dict(mode=sys.argv[3])));os.chown(mode_path,0,0);mode_path.chmod(0o444)
assert pwd.getpwnam('supervisor').pw_uid==1000
assert pwd.getpwnam('benchmarkworker').pw_uid==1001
subprocess.run(['usermod','-a','-G',pwd.getpwuid(1000).pw_name,'benchmarkworker'],check=True)
anchor='else:\n from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main'
assert WORKER_ENTRY.count(anchor)==1
replacement="else:\n if args.model=='numerical-ipc-authored':\n  import runpy\n  runpy.run_path('/qualification/worker_probe.py')['install']()\n from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main"
worker=WORKER_ENTRY.replace(anchor,replacement)
client="#!/opt/ipfs-supervisor/venv/bin/python -I\nimport ctypes,os,sys\nlibc=ctypes.CDLL(None,use_errno=True)\nif libc.prctl(38,1,0,0,0)!=0:raise OSError('no-new-privileges failed')\nos.execv('/opt/ipfs-supervisor/venv/bin/python',['/opt/ipfs-supervisor/venv/bin/python','-I','-B','/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py',*sys.argv[1:]])\n"
for name,text in (('owner-worker',OWNER_ENTRY),('worker-entry',worker),('probe-client',client)):
 path=root/'bin'/name;path.write_text(text);os.chown(path,0,0);path.chmod(0o555)
sudo=pathlib.Path('/etc/sudoers.d/numerical-ipc-qualification')
sudo.write_text('supervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *, /opt/ipfs-supervisor/bin/probe-client *\n')
os.chown(sudo,0,0);sudo.chmod(0o440)
for name,mode in (('state',0o700),('repository',0o755),('worktrees',0o755),('worker-observations',0o750),('fixture-ipc',0o750)):
 path=root/name;path.mkdir(exist_ok=True);os.chown(path,1000,1000);path.chmod(mode)
for name in ('worker-result.json','router-observation.json'):
 path=root/'fixture-ipc'/name;path.touch();os.chown(path,1001,1000);path.chmod(0o660)
for name in ('.local','.duckdb','.ipfs_accelerate'):
 path=pathlib.Path('/home/supervisor')/name;path.mkdir(exist_ok=True);os.chown(path,1000,1000);path.chmod(0o700)
boundary=dict(schema='supervisor-container-worker-boundary@1',container_id=sys.argv[1],image_id=sys.argv[2],owner_uid=1000,worker_uid=1001,single_worker=True,provider='codex_cli',namespaces={name:os.readlink('/proc/self/ns/'+name) for name in ('pid','mnt','net')},allowed_worktree_roots=[str(root/'worktrees')],owner_private_paths=[str(root/'state')])
path=root/'container-boundary.json';path.write_text(json.dumps(boundary,sort_keys=True));os.chown(path,0,0);path.chmod(0o444)
print(json.dumps(dict(schema='numerical-ipc-root-setup@1',boundary=boundary,owner_entry_sha256=__import__('hashlib').sha256(OWNER_ENTRY.encode()).hexdigest(),worker_entry_production_sha256=__import__('hashlib').sha256(WORKER_ENTRY.encode()).hexdigest(),worker_entry_authored_provider_hook_sha256=__import__('hashlib').sha256(worker.encode()).hexdigest(),qualification_hook='Only fixture model inserts authored generate_text before actual router main; real OWNER_ENTRY unchanged')))
