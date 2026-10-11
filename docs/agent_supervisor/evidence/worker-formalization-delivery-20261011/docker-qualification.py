"""Offline privilege qualification, with explicitly authored owner selection.

Executes production OWNER_ENTRY and source-formalization helper. The diagnostic
worker invokes the production guard but replaces the provider and cleanup entry.
Host tests independently exercise real owner indexes/checkpoint and router.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/opt/ipfs-supervisor')
sys.path[:0] = [str(ROOT / name) for name in ('source', 'datasets', 'kit')]
from benchmarks.agent_supervisor.container_coding.container_worker_deployment import OWNER_ENTRY
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references, context_provider_input_payload
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard


def run(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, check=True, timeout=30, **kwargs)


def owned(path, mode, owner=1000, group=1000):
    path.chmod(mode)
    os.chown(path, owner, group)


for name, mode in [('state', 0o700), ('worktrees', 0o750), ('worker-observations', 0o750)]:
    p = ROOT / name
    p.mkdir(exist_ok=True)
    owned(p, mode)
ROOT.chmod(0o755)
os.chown(ROOT, 0, 0)
(ROOT / 'bin').mkdir(exist_ok=True)
owned(ROOT / 'bin', 0o755, 0, 0)
repository = Path('/app')
repository.mkdir()
source = b'def authored(value):\n    return value + 1\n'
(repository / 'source.py').write_bytes(source)
run(['git', 'init', '-q', str(repository)])
run(['git', '-C', str(repository), 'add', 'source.py'])
run(['git', '-C', str(repository), '-c', 'user.name=Authored Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'authored source'])
workspace = ROOT / 'worktrees' / 'candidate'
run(['git', '-C', str(repository), 'worktree', 'add', '--detach', '-q', str(workspace)])
for tree in (repository, workspace):
    for base, dirs, files in os.walk(tree):
        owned(Path(base), 0o750)
        for name in files:
            owned(Path(base) / name, 0o640)
boundary = dict(schema='supervisor-container-worker-boundary@1',
    container_id=os.environ['QUAL_CONTAINER_ID'], image_id=os.environ['QUAL_IMAGE_ID'],
    namespaces={key: os.readlink('/proc/self/ns/' + key) for key in ('pid', 'mnt', 'net')},
    owner_uid=1000, worker_uid=1001, allowed_worktree_roots=[str(ROOT / 'worktrees')],
    owner_private_paths=[str(ROOT / 'state')], provider='codex_cli', single_worker=True)
(ROOT / 'container-boundary.json').write_text(json.dumps(boundary))
owned(ROOT / 'container-boundary.json', 0o444, 0, 0)
hashes = {'source.py': hashlib.sha256(source).hexdigest()}
payload = dict(schema=guard.WORKER_EVIDENCE_SCHEMA, repository=str(repository),
    task_cid='task:authored', task_id='authored-task',
    context_bundle={'artifact': '.runtime/context.json', 'sha256': 'b' * 64},
    contexts=[dict(kind='numerical-formalization-context', summary={'schema':'authored-summary@1'},
        source_hashes_sha256=hashlib.sha256(guard._json(hashes).encode()).hexdigest())],
    proof_authority=False, formalization_authority=False, execution_authority=False,
    completion_authority=False, planning_authority=False, dispatch_authority=False,
    source_semantics_verified=False, whole_program_semantics_verified=False,
    source_obligations_discharged=0)
selection = dict(worker_evidence=payload, numerical_formalization={'source_hashes':hashes})
(ROOT / 'state' / 'authored-selection.json').write_text(json.dumps(selection))
owned(ROOT / 'state' / 'authored-selection.json', 0o600)
refs = build_text_context_references(guard._json(payload), reference_prefix='source-formalization',
    kind=guard.KIND, path=payload['context_bundle']['artifact'], repository_id='repo:authored',
    tree_id='tree:authored', required=True, chunk_bytes=6144)
prompt = guard._json(context_provider_input_payload(repository_id='repo:authored', tree_id='tree:authored',
    objective_id='authored-task', objective_revision='rev:1', policy_id='policy:authored',
    policy_revision='policy:1', caller='coding-worker', stage='implementation', goal='Change source',
    authority={}, scope={}, acceptance={}, evidence=refs))
owner_prelude = '''
import sys,json,pathlib
sys.path[:0]=['/opt/ipfs-supervisor/source','/opt/ipfs-supervisor/datasets','/opt/ipfs-supervisor/kit']
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
def authored_selection(**kwargs):
    value=json.loads(pathlib.Path('/opt/ipfs-supervisor/state/authored-selection.json').read_text())
    assert kwargs['repository']==pathlib.Path('/app') and kwargs['task_id']=='authored-task'
    return value
guard.load_task_context_selection=authored_selection
'''
owner = ROOT / 'bin' / 'qualified-owner'
owner.write_text('#!/opt/ipfs-supervisor/venv/bin/python -I\n' + owner_prelude + OWNER_ENTRY)
owned(owner, 0o555, 0, 0)
worker = ROOT / 'bin' / 'worker-entry'
worker.write_text('''#!/opt/ipfs-supervisor/venv/bin/python -I
import ctypes,fcntl,hashlib,json,os,pathlib,sys
if sys.argv[1:]==['--cleanup']:raise SystemExit(0)
assert os.getuid()==1001 and os.geteuid()==1001
libc=ctypes.CDLL(None,use_errno=True)
assert libc.prctl(38,1,0,0,0)==0
sys.path[:0]=['/opt/ipfs-supervisor/source','/opt/ipfs-supervisor/datasets','/opt/ipfs-supervisor/kit']
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
root=pathlib.Path('/opt/ipfs-supervisor')
try:(root/'state/authored-selection.json').read_bytes()
except PermissionError:private_denied=True
else:raise AssertionError('worker read private owner state')
def refuse(**kwargs):raise AssertionError('worker replayed private owner state')
guard.load_task_context_selection=refuse
mode=sys.argv[sys.argv.index('--qualification-mode')+1]
prompt=sys.stdin.read()
original_sha=hashlib.sha256(prompt.encode()).hexdigest()
seals=fcntl.fcntl(0,fcntl.F_GET_SEALS)
assert seals & (fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL)==15
handoff=None;digest=None
if '--source-formalization-handoff' in sys.argv:
    handoff=pathlib.Path(sys.argv[sys.argv.index('--source-formalization-handoff')+1])
    digest=sys.argv[sys.argv.index('--source-formalization-handoff-sha256')+1]
    assert not os.access(handoff,os.W_OK)
    assert not os.access(handoff.parent,os.W_OK)
if mode=='prompt_drift':prompt+=' changed'
elif mode=='stripped':prompt='ordinary instruction'
elif mode=='source_drift':pathlib.Path('source.py').write_text('changed source')
elif mode=='digest_drift':digest='0'*64
try:
    observed=guard.validate_source_formalization_context(prompt=prompt,repository=pathlib.Path('/app'),
        workspace=pathlib.Path.cwd(),boundary=root/'container-boundary.json',
        boundary_sha256=hashlib.sha256((root/'container-boundary.json').read_bytes()).hexdigest(),
        handoff=handoff,handoff_sha256=digest)
except ValueError as exc:
    assert mode in ('prompt_drift','stripped','source_drift','digest_drift'),str(exc)
    observed={'refused':str(exc)}
else:
    assert mode in ('valid','legacy','preflight')
    if mode=='valid':
        assert observed['owner_validation_inherited'] is True
        assert observed['owner_private_state_replayed_here'] is False
        assert observed['source_obligations_discharged']==0
    else:assert observed is None and handoff is None
print(json.dumps(dict(mode=mode,uid=os.getuid(),private_denied=private_denied,stdin_seals=seals,
    original_prompt_sha256=original_sha,observation=observed,provider_calls=0)))
''')
owned(worker, 0o555, 0, 0)
sudoers = Path('/etc/sudoers.d/supervisor-worker')
sudoers.write_text('Defaults:supervisor !use_pty\nsupervisor ALL=(benchmarkworker) NOPASSWD: /opt/ipfs-supervisor/bin/worker-entry *\n')
owned(sudoers, 0o440, 0, 0)
run(['visudo', '-cf', str(sudoers)])
results = []
for mode in ('valid', 'prompt_drift', 'stripped', 'digest_drift', 'source_drift', 'legacy', 'preflight'):
    (workspace / 'source.py').write_bytes(source)
    owned(workspace / 'source.py', 0o640)
    text = prompt if mode not in ('legacy','preflight') else ('plain instruction λ' if mode=='legacy' else '')
    command=['/usr/bin/sudo','-n','-u','supervisor','--',str(owner),
        '--semantic-repository','/app','--qualification-mode',mode]
    result=run(command,cwd=workspace,input=text.encode())
    rows=[json.loads(row) for row in result.stdout.decode().splitlines() if row.startswith('{')]
    assert len(rows)==1,result.stdout
    assert rows[0]['original_prompt_sha256']==hashlib.sha256(text.encode()).hexdigest()
    results.append(rows[0])
print(json.dumps(dict(schema='authored-docker-formalization-qualification@1',cases=results,
    image_id=boundary['image_id'],container_id=boundary['container_id'],
    namespaces=boundary['namespaces'],owner_entry_sha256=hashlib.sha256(OWNER_ENTRY.encode()).hexdigest(),
    owner_selection='authored fixture; real owner validation covered separately on host',
    worker='diagnostic worker; production guard; production parser covered separately',
    network='none',provider_calls=0,neural_inference_calls=0),sort_keys=True))
