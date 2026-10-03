"""Execute the draft's authored controls using module paths, never live edits."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

B=Path(__file__).resolve().parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
D=Path('/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB']=str(B/'overlay-seals.duckdb')
sys.path[:0]=[str(A),str(D),'/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages']
import benchmarks.agent_supervisor.container_coding as package
package.__path__=[str(B/'proposed/benchmarks/agent_supervisor/container_coding'),*package.__path__]
import pytest
before=json.loads((B/'base-pins.json').read_text())
assert all(hashlib.sha256((A/name).read_bytes()).hexdigest()==pin for name,pin in before.items())
tests=['test_terminal_deployment.py','test_terminal_source384_qualification.py','test_terminal_source384_transport.py','test_terminal_torch_wheel.py','test_full_supervisor_harbor_agent.py']
argv=['-q','--rootdir='+str(B),'--confcutdir='+str(B),*[str(A/'benchmarks/agent_supervisor/container_coding'/name) for name in tests],'--junitxml='+str(B/'final-neighbors-02.xml')]
(B/'final-neighbors-02-command.json').write_text(json.dumps({'argv':[sys.executable,str(__file__)],'pytest_argv':argv,'scope':'draft-only module-path overlay; no production edits, Docker or model calls','producer_pins':'draft-pins.json'},indent=2)+'\n')
started=time.monotonic();code=pytest.main(argv)
after={name:hashlib.sha256((A/name).read_bytes()).hexdigest() for name in before}
loaded={name:str(getattr(value,'__file__','')) for name,value in sys.modules.items() if name.startswith('benchmarks.agent_supervisor.container_coding.') and Path(str(getattr(value,'__file__',''))).name in {'terminal_setup_cache_advice.py','terminal_setup_cache_files.py','terminal_setup_cache_codex.py','terminal_setup_cache_libraries.py','terminal_deployment.py','full_supervisor_benchmark.py','full_supervisor_harbor_agent.py'}}
(B/'final-neighbors-02-exit.json').write_text(json.dumps({'returncode':int(code),'seconds':time.monotonic()-started,'production_pins_unchanged':after==before,'loaded_owners':loaded},indent=2)+'\n')
raise SystemExit(code)
