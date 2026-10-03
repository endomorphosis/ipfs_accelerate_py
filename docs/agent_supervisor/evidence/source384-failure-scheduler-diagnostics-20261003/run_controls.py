"""Pin three proposed A modules under canonical names; no live tree mutation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

B=Path(__file__).resolve().parent
A=B.parents[1]/'.worktrees/ir-release-accelerate-20261002'
selected=['terminal_resource_diagnostics','terminal_source384_qualification','terminal_container_supervisor']
paths={name:B/'proposed/benchmarks/agent_supervisor/container_coding'/f'{name}.py' for name in selected}
for name,path in paths.items():
 full='benchmarks.agent_supervisor.container_coding.'+name
 assert full not in sys.modules
 spec=importlib.util.spec_from_file_location(full,path);module=importlib.util.module_from_spec(spec)
 sys.modules[full]=module;spec.loader.exec_module(module)
import pytest
new=B/'proposed/benchmarks/agent_supervisor/container_coding/test_terminal_resource_diagnostics.py'
old=A/'benchmarks/agent_supervisor/container_coding/test_terminal_failure_diagnostics.py'
args=['-q',str(new),str(old),'-k','not test_actual_canonical_resource_probe_reports_only_finite_resource_values',
 '--junitxml='+str(B/'controls-01.xml')]
files=list(paths.values())+[new,old,Path(__file__)]
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(B/'controls-01-command.json').write_text(json.dumps(dict(argv=[sys.executable,str(Path(__file__))],pytest_args=args,
 source_pins=pins,environment={k:os.environ.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
 scope='controlled scheduler and resource samples only; embedded probe failure and driver primary-result preservation; no native models/Docker or actual host sampling'),indent=2)+'\n')
start=time.monotonic();code=int(pytest.main(args))
(B/'controls-01-exit.json').write_text(json.dumps(dict(exit_code=code,seconds=time.monotonic()-start,
 source_pins_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
raise SystemExit(code)
