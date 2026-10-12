import hashlib,importlib,json,os,sys
from pathlib import Path
sys.path[:0]=['/code/source','/code/datasets','/code/kit']
os.chdir('/tmp')
from benchmarks.agent_supervisor.container_coding.terminal_semantic_metadata_policy import require_semantic_metadata_archive
manifest=json.loads(Path('/qualification/archive-manifest.json').read_text())
for mode in ('common-bindings@1','common-bindings-instruction@1'):require_semantic_metadata_archive(manifest,mode)
modules={}
for name in ('ipfs_accelerate_py.agent_supervisor.runtime.semantic_instruction_view','ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation','ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner','ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime','benchmarks.agent_supervisor.container_coding.terminal_context_audit'):
 module=importlib.import_module(name)
 path=Path(module.__file__)
 assert path.is_relative_to('/code/source')
 modules[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
Path('/qualification-output/module-origins.json').write_text(json.dumps(modules,indent=2)+'\n')
import pytest
raise SystemExit(pytest.main(['-q','--color=no','-x','/code/source/test/api/test_semantic_instruction_dispatch.py','--basetemp=/tmp/instruction-pytest','--junitxml=/qualification-output/junit.xml','-o','cache_dir=/tmp/pytest-cache','-o','junit_family=xunit1']))
