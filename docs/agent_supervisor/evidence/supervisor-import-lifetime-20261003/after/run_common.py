from pathlib import Path
import json,os,subprocess,time
B=Path(__file__).resolve().parent;W=B.parent.parent;A=W/'.worktrees/ir-release-accelerate-20261002';D=A.parent/'ir-release-datasets-20261002'
recipe=json.loads((B/'prepare-command.json').read_text());base=recipe['argv'][3]
insert='''
root=pathlib.Path(sys.argv[2]);root.mkdir()
(root/'authored.py').write_text('def authored_function(value):\\n    return value + 1\\n')
from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify
qualify(root, root/'vectors', ['authored.py'], 'authored function')
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context
source384_repository_context._pins()
from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384
codebase_source_units_384.pins()
'''
program=base.replace('for name in sys.argv[1:]:importlib.import_module(name)','importlib.import_module(sys.argv[1])'+insert)
for tag,name in [('common-prepare','terminal_indexed_preparation'),('common-driver','terminal_container_supervisor')]:
 argv=[recipe['argv'][0],'-B','-c',program,'benchmarks.agent_supervisor.container_coding.'+name,str(B/tag)]
 (B/(tag+'-command.json')).write_text(json.dumps(dict(argv=argv,cwd=str(A),environment_overrides=recipe['environment_overrides'],scope='authored tiny vector index and actual shared producer imports only; not native cgroup admission'),indent=2)+'\n')
 started=time.monotonic()
 with (B/(tag+'.stdout')).open('x') as out,(B/(tag+'.stderr')).open('x') as err:r=subprocess.run(argv,cwd=A,env={**os.environ,**recipe['environment_overrides']},stdout=out,stderr=err,timeout=120)
 (B/(tag+'-exit.json')).write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-started),indent=2)+'\n')
 assert r.returncode==0
 d=json.loads((B/(tag+'.stdout')).read_text().splitlines()[-1]);(B/(tag+'.json')).write_text(json.dumps(d,indent=2)+'\n');print(tag,{k:d[k] for k in ['seconds','rss','peak_rss_kib']})
