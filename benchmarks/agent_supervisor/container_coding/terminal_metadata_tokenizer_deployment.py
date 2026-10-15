"""Deploy captured local tokenizer bytes into protected container storage.

The host bundle supplies bytes and identity only. Remote paths are fixed here;
the model's editable task and receipt paths never select a deployment target.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile

from .terminal_deployment import ROOT, PYTHON, runtime_environment
from .terminal_metadata_tokenizer_bundle import verify_metadata_tokenizer_bundle_snapshot


def _captured(snapshot):
    snapshot = verify_metadata_tokenizer_bundle_snapshot(snapshot)
    selection, manifest = snapshot.selection(), snapshot.manifest()
    files = dict(snapshot.files())
    return selection, manifest, {name: files[name] for name in ("descriptor.json", "tokenizer.json")}


def metadata_tokenizer_remote_descriptor(snapshot):
    """Derive the sole permitted remote descriptor, without receipt paths."""
    selection, manifest, _ = _captured(snapshot)
    return ROOT + "/metadata-tokenizers/" + selection["manifest_sha256"] + "/descriptor.json", manifest["descriptor_sha256"]


def _root_script(snapshot, phase):
    if phase not in {"create", "publish", "verify"}:
        raise ValueError("explicit tokenizer deployment phase required")
    selection, _, files = _captured(snapshot)
    expected = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                for name, raw in files.items()}
    prefix = ("phase=" + repr(phase) + "\nroot_text=" + repr(ROOT)
              + "\nmanifest_pin=" + repr(selection["manifest_sha256"])
              + "\nexpected=" + repr(expected) + "\n")
    return prefix + r'''
import hashlib,json,os,pathlib,stat
if os.getuid()!=0 or os.geteuid()!=0:raise ValueError('root deployment identity required')
root=pathlib.Path(root_text)
base=root/'metadata-tokenizers'
target=base/manifest_pin
def directory(path,mode=None):
 info=path.lstat()
 if (not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_gid!=0
     or info.st_mode&0o022 or path.resolve(strict=True)!=path
     or mode is not None and stat.S_IMODE(info.st_mode)!=mode):
  raise ValueError('protected tokenizer directory required')
 return info
for parent in reversed((root,*root.parents)):directory(parent)
if phase=='create':
 try:base.mkdir(mode=0o755)
 except FileExistsError:pass
 else:base.chmod(0o755)
 directory(base,0o755)
 target.mkdir(mode=0o700)
 directory(target,0o700)
else:
 directory(base,0o755)
 directory(target,0o700 if phase=='publish' else 0o555)
 names=[]
 with os.scandir(target) as entries:
  for entry in entries:
   names.append(entry.name)
   if len(names)>2:raise ValueError('unexpected tokenizer deployment file')
 if sorted(names)!=sorted(expected):raise ValueError('tokenizer deployment population mismatch')
 flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC
 parent_fd=os.open(target,flags)
 try:
  for name,row in expected.items():
   fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=parent_fd)
   try:
    info=os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size!=row['bytes']:
     raise ValueError('bounded regular tokenizer file required')
    if phase=='publish':
     os.fchown(fd,0,0);os.fchmod(fd,0o444);os.fsync(fd)
    before=os.fstat(fd)
    if before.st_uid!=0 or before.st_gid!=0 or stat.S_IMODE(before.st_mode)!=0o444:
     raise ValueError('protected tokenizer file required')
    digest=hashlib.sha256();count=0
    while count<=row['bytes']:
     chunk=os.read(fd,min(65536,row['bytes']+1-count))
     if not chunk:break
     count+=len(chunk);digest.update(chunk)
    after=os.fstat(fd)
    identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    if (identity(before)!=identity(after) or count!=row['bytes'] or digest.hexdigest()!=row['sha256']
        or identity(os.stat(name,dir_fd=parent_fd,follow_symlinks=False))!=identity(after)):
     raise ValueError('tokenizer deployment byte binding changed')
   finally:os.close(fd)
  if phase=='publish':os.fchmod(parent_fd,0o555);os.fsync(parent_fd)
  held=os.fstat(parent_fd);current=directory(target,0o555)
  if (held.st_dev,held.st_ino)!=(current.st_dev,current.st_ino):
   raise ValueError('tokenizer deployment directory changed')
 finally:os.close(parent_fd)
print(json.dumps({'schema':'terminal-tokenizer-root-verification@1','phase':phase,
                  'manifest_sha256':manifest_pin,'verified':True},sort_keys=True))
'''


def _worker_script(snapshot):
    selection, manifest, _ = _captured(snapshot)
    descriptor, pin = metadata_tokenizer_remote_descriptor(snapshot)
    loader = Path(__file__).resolve().parents[3] / "ipfs_accelerate_py/agent_supervisor/runtime/local_prompt_tokenizer.py"
    if loader.is_symlink() or not loader.is_file() or loader.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("current local tokenizer loader source required")
    loader_pin = hashlib.sha256(loader.read_bytes()).hexdigest()
    return ("descriptor=" + repr(descriptor) + "\npin=" + repr(pin)
        + "\nprovider=" + repr(selection["provider"]) + "\nmodel=" + repr(selection["model"])
        + "\nasset_pin=" + repr(manifest["asset_sha256"]) + "\nroot_text=" + repr(ROOT)
        + "\nloader_pin=" + repr(loader_pin) + "\n" + r'''
import hashlib,json,os,pathlib,stat,sys,types
if os.getuid()!=1001 or os.geteuid()!=1001:raise ValueError('isolated tokenizer worker required')
loader=pathlib.Path(root_text)/'source/ipfs_accelerate_py/agent_supervisor/runtime/local_prompt_tokenizer.py'
for path in (loader,*loader.parents):
 info=path.lstat()
 if info.st_uid!=0 or info.st_mode&0o022 or path.resolve(strict=True)!=path:
  raise ValueError('root protected tokenizer loader source required')
fd=os.open(loader,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
try:
 before=os.fstat(fd)
 if not stat.S_ISREG(before.st_mode) or not 1<=before.st_size<=2*1024*1024:
  raise ValueError('bounded tokenizer loader source required')
 with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(2*1024*1024+1)
 after=os.fstat(fd)
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=loader_pin:
  raise ValueError('tokenizer loader source changed')
finally:os.close(fd)
module=types.ModuleType('owner_local_tokenizer_probe');module.__file__=str(loader)
sys.modules[module.__name__]=module
exec(compile(raw,str(loader),'exec'),module.__dict__)
load_owner_prompt_tokenizer=module.load_owner_prompt_tokenizer
OwnerPromptTokenizer=module.OwnerPromptTokenizer
local=load_owner_prompt_tokenizer(artifact=descriptor,expected_sha256=pin,provider=provider,model=model)
OwnerPromptTokenizer._verify_binding(local)
probe_tokens=OwnerPromptTokenizer.count(local,'Bounded local tokenizer readiness.')
source=OwnerPromptTokenizer.provenance(local)
if source['asset_sha256']!=asset_pin:raise ValueError('deployed tokenizer asset differs')
print(json.dumps({'schema':'terminal-tokenizer-worker-readiness@1','descriptor_sha256':pin,
 'asset_sha256':asset_pin,'tokenizers_version':source['tokenizers_version'],
 'probe_tokens':probe_tokens,'verified':True},sort_keys=True))
''')


async def _checked_exec(environment, *, script, user, expected, timeout_sec=30):
    result = await environment.exec(command=shlex.join([PYTHON, "-I", "-c", script]),
        user=user, cwd="/", env=runtime_environment(), timeout_sec=timeout_sec)
    if type(result.return_code) is not int or result.return_code != 0:
        raise RuntimeError("local tokenizer deployment or readiness verification failed")
    raw = result.stdout
    if type(raw) is not str or len(raw.encode()) > 16384:
        raise ValueError("bounded tokenizer deployment observation required")
    try:
        observed = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValueError("tokenizer deployment observation unavailable") from error
    if type(observed) is not dict or any(type(observed.get(key)) is not type(value)
            or observed.get(key) != value for key, value in expected.items()):
        raise ValueError("tokenizer deployment observation differs")
    return observed


async def verify_deployed_metadata_tokenizer_bundle(environment, *, snapshot):
    """Recheck protected bytes and the worker runtime before coding can start."""
    selection, manifest, _ = _captured(snapshot)
    await _checked_exec(environment, script=_root_script(snapshot, "verify"), user="root",
        expected={"schema": "terminal-tokenizer-root-verification@1", "phase": "verify",
                  "manifest_sha256": selection["manifest_sha256"], "verified": True})
    worker = await _checked_exec(environment, script=_worker_script(snapshot), user="benchmarkworker",
        expected={"schema": "terminal-tokenizer-worker-readiness@1", "verified": True,
                  "descriptor_sha256": manifest["descriptor_sha256"], "asset_sha256": manifest["asset_sha256"]})
    if (type(worker.get("probe_tokens")) is not int or not 1 <= worker["probe_tokens"] <= (1 << 63) - 1
            or worker.get("tokenizers_version") != manifest["tokenizers_version"]):
        raise ValueError("deployed tokenizer runtime observation differs")
    return {"schema": "terminal-metadata-tokenizer-deployment@1", "verified": True,
        "manifest_sha256": selection["manifest_sha256"], "descriptor_sha256": manifest["descriptor_sha256"],
        "asset_sha256": manifest["asset_sha256"], "tokenizers_version": manifest["tokenizers_version"],
        "root_protection_verified": True, "worker_runtime_verified": True,
        "provider_discovery_performed": False, "provider_calls": 0,
        "provider_tokenizer_equivalence_verified": False, "token_savings_established": False}


async def deploy_metadata_tokenizer_bundle(environment, *, snapshot, output: Path):
    """Upload only captured portable bytes; preserve any failed deployment."""
    selection, _, files = _captured(snapshot)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    await _checked_exec(environment, script=_root_script(snapshot, "create"), user="root",
        expected={"schema": "terminal-tokenizer-root-verification@1", "phase": "create",
                  "manifest_sha256": selection["manifest_sha256"], "verified": True})
    remote_parent = str(Path(metadata_tokenizer_remote_descriptor(snapshot)[0]).parent)
    with tempfile.TemporaryDirectory(prefix="captured-", dir=output) as temporary:
        for name, raw in files.items():
            path = Path(temporary) / name
            with path.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o400)
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            await environment.upload_file(path, remote_parent + "/" + name)
    await _checked_exec(environment, script=_root_script(snapshot, "publish"), user="root",
        expected={"schema": "terminal-tokenizer-root-verification@1", "phase": "publish",
                  "manifest_sha256": selection["manifest_sha256"], "verified": True})
    observation = await verify_deployed_metadata_tokenizer_bundle(environment, snapshot=snapshot)
    with (output / "observation.json").open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(observation, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return observation
