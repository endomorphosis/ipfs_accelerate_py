import hashlib,importlib,json,os,shutil,sys
from pathlib import Path
sys.path[:0]=['/code/source','/code/datasets','/code/kit']
os.chdir('/tmp')
for name in ('OPENAI_API_KEY','XAI_API_KEY','ANTHROPIC_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
 assert not os.environ.get(name),name+' must be absent'
import duckdb
con=duckdb.connect(':memory:')
platform=con.execute('PRAGMA platform').fetchone()[0]
cache=Path.home()/'.duckdb/extensions'/('v'+duckdb.__version__)/platform
cache.mkdir(parents=True,exist_ok=False)
extensions={}
for name in ('ducklake','httpfs','quack'):
 original=Path('/code/extensions')/(name+'.duckdb_extension');target=cache/original.name
 shutil.copyfile(original,target);con.execute('LOAD '+name)
 extensions[name]={'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
con.close()
modules={}
for name in ('ipfs_accelerate_py.agent_supervisor.runtime.published_task_context','ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_selection','ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime','ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation','ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context','ipfs_accelerate_py.agent_supervisor.runtime.published_retrieval','ipfs_datasets_py','ipfs_kit_py'):
 module=importlib.import_module(name);path=Path(module.__file__)
 assert path.is_relative_to('/code/source') or path.is_relative_to('/code/datasets') or path.is_relative_to('/code/kit')
 modules[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
Path('/qualification-output/module-origins.json').write_text(json.dumps({'modules':modules,'extensions':extensions,'duckdb':duckdb.__version__,'platform':platform},indent=2)+'\n')
import pytest
raise SystemExit(pytest.main(['-q','--color=no','-x','/code/source/test/api/semantic_state/test_ranked_published_task_context.py','--basetemp=/tmp/ranked-pytest','--junitxml=/qualification-output/junit.xml','-o','cache_dir=/tmp/pytest-cache','-o','junit_family=xunit1']))
