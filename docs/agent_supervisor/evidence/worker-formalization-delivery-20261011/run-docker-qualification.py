import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parent
r=root/sys.argv[1]
r.mkdir()
p=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
d=p.parent/'source-unit-numerical-links-20261011'
k=p.parent/'publication-main-20261004/kit'
script=root/'docker-qualification.py'
image='ipfs-supervisor-preview-worker:20261005-r3'
image_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
cmd=['docker','create','--user','0:0','--network','none','--memory','3g','--cpus','2','--pids-limit','128']
for src,dst in [(p,'source'),(d,'datasets'),(k,'kit'),(script,'qualification.py')]:
    cmd+=['--mount',f'type=bind,src={src},dst=/opt/ipfs-supervisor/{dst},readonly']
cmd+=['--env','IPFS_KIT_DISABLE=1','--env','PYTHONDONTWRITEBYTECODE=1','--entrypoint','/bin/sleep',image,'300']
start=time.monotonic()
cid=subprocess.check_output(cmd,text=True).strip()
run=['docker','exec','--env','QUAL_CONTAINER_ID='+cid,'--env','QUAL_IMAGE_ID='+image_id,
    cid,'/opt/ipfs-supervisor/venv/bin/python','-I','/opt/ipfs-supervisor/qualification.py']
sources=['ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py',
    'benchmarks/agent_supervisor/container_coding/container_worker_deployment.py']
before={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in sources}
(r/'command.json').write_text(json.dumps(dict(create=cmd,execute=run,
    controller_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),source_before=before),indent=2)+'\n')
try:
    subprocess.run(['docker','start',cid],check=True,capture_output=True)
    with (r/'stdout.log').open('wb') as out:
        result=subprocess.run(run,stdout=out,stderr=subprocess.STDOUT,timeout=240)
    after={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in sources}
    (r/'result.json').write_text(json.dumps(dict(exit_code=result.returncode,seconds=time.monotonic()-start,
        container_id=cid,image_id=image_id,source_before=before,source_after=after),indent=2)+'\n')
    print((r/'stdout.log').read_text()[-9000:])
    print((r/'result.json').read_text())
finally:
    subprocess.run(['docker','rm','-f',cid],capture_output=True,check=True)
