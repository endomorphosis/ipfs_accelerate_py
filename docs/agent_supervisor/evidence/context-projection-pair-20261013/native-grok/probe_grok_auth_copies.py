"""Read-only host auth custody; metadata command may mutate only temporary copies."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,shutil,stat,subprocess,tempfile,time
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/metadata-live-20261013')
module=P/'benchmarks/agent_supervisor/container_coding/terminal_grok_deployment.py'
spec=importlib.util.spec_from_file_location('grok_deployment_reviewed',module);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
auth=Path('/home/barberb/.grok/auth.json')
def identity(path):
 s=path.lstat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid,s.st_nlink)
before=identity(auth);assert stat.S_ISREG(before[5]) and not auth.is_symlink() and before[2] in range(1,65537) and before[7]==1
rows=[]
for label,binary in [('pinned-1.0.46',R/'grok-1.0.46/grok'),('installed-1.0.52',Path('/home/barberb/.local/bin/grok').resolve(strict=True))]:
 start=time.monotonic();temp=None
 with tempfile.TemporaryDirectory(prefix='grok-metadata-copy-',dir=R) as directory:
  temp=Path(directory);(temp/'.grok').mkdir(mode=0o700)
  assert identity(auth)==before
  sourcefd=os.open(auth,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
  with os.fdopen(sourcefd,'rb') as src,(temp/'.grok/auth.json').open('xb') as dest:shutil.copyfileobj(src,dest)
  (temp/'.grok/auth.json').chmod(0o600)
  assert identity(auth)==before
  version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,timeout=15,env={'HOME':str(temp),'GROK_HOME':str(temp/'.grok'),'PATH':'/usr/bin:/bin'})
  assert version.returncode==0
  expected='grok 1.0.46' if label=='pinned-1.0.46' else 'grok 1.0.52'
  assert version.stdout.strip().startswith(expected) and len(version.stdout)<256 and not version.stderr
  out,err,code,reason=guard._bounded_catalog_command(binary,temp)
  observation=guard._catalog_observation(out,err,code)
  if reason is not None:observation.update(status='unavailable',reason=reason,models_observed=0,selected_model_observed=False)
  with binary.open('rb') as stream:binarysha=hashlib.file_digest(stream,'sha256').hexdigest()
  row=dict(label=label,binary=str(binary),binary_sha256=binarysha,version=version.stdout.strip(),status=observation['status'],reason=observation['reason'],native_exit_code=code,native_catalog_commands=1,stdout_bytes=len(out),stderr_bytes=len(err),models_observed=observation['models_observed'],selected_model_observed=observation['selected_model_observed'],elapsed_seconds=time.monotonic()-start,host_auth_unchanged=identity(auth)==before,authentication_authority=False,provider_generation_calls=0)
  del out,err
 row['temporary_copy_removed']=not temp.exists();rows.append(row)
 print(json.dumps(row,sort_keys=True),flush=True)
assert identity(auth)==before
result=dict(schema='source-context-grok-isolated-copy-comparison@1',observed_at_utc=datetime.now(timezone.utc).isoformat(),scope='Fresh temporary host HOME per native binary; copied auth only, no Docker qualification',rows=rows,host_auth_unchanged=True,host_auth_write_performed=False,credential_contents_or_digest_retained=False,raw_provider_output_retained=False,login_performed=False,provider_generation_calls=0,models_metadata_commands=len(rows),guard_source_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'grok-isolated-copy-comparison.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
