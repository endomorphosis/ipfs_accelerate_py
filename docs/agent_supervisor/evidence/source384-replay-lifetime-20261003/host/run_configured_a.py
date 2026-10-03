"""Actual on-disk consumer tests as an explicit existing-profile client."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import time

B=Path(__file__).resolve().parent
W=B.parents[1]
A=W/'.worktrees/ir-release-accelerate-20261002'
D=W/'.worktrees/ir-release-datasets-20261002'
from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as consumer
from ipfs_datasets_py.logic.software_contracts import codebase_ir,cache,codebase_source_units_384 as units
assert Path(consumer.__file__).resolve()==A/'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py'
assert Path(codebase_ir.__file__).resolve()==D/'ipfs_datasets_py/logic/software_contracts/codebase_ir.py'
assert Path(units.__file__).resolve()==D/'ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py'
assert codebase_ir._manifest_producer_key() is not None
assert os.environ['PYTHONPATH']==str(A)+os.pathsep+str(D)
import pytest
from configured_client_plugin import ExplicitClientPlugin
name='actual-a-context'
args=['-q',str(A/'benchmarks/agent_supervisor/container_coding/test_terminal_source384_context.py'),
      '--basetemp='+str(B/(name+'-temp')),'--junitxml='+str(B/(name+'.xml'))]
paths=[Path(consumer.__file__),Path(codebase_ir.__file__),Path(cache.__file__),Path(units.__file__),
       Path(__file__),B/'configured_client_plugin.py',*(Path(v) for v in args if v.endswith('.py'))]
pins=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=pins()
environment={k:os.environ.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS',
    'CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
    'CODEBASE384_CHECKPOINT','CODEBASE384_EMBEDDING_SNAPSHOT','IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB',
    'IPFS_DATASETS_PY_SEAL_KEY_STORE_PATH')}
command={'argv':[sys.executable,str(Path(__file__))],'pytest_args':args,'cwd':str(Path.cwd()),
    'environment_overrides':environment,'source_pins':before,'actual_on_disk_sources':True,
    'owner_overlay':False,'native_producer_recognized':True,
    'scheduler_mode':'explicit_configured_client_same_native_ledger',
    'host_default_startup_qualification':False,
    'versions':{n:metadata.version(n) for n in ('torch','numpy','transformers','sentence-transformers','tokenizers')}}
(B/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
started=time.monotonic()
status=int(pytest.main(args,plugins=[ExplicitClientPlugin()]))
after=pins()
receipt={'returncode':status,'elapsed_seconds':time.monotonic()-started,
         'source_pins_after':after,'sources_unchanged':after==before}
(B/(name+'-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
assert before==after
raise SystemExit(status)
