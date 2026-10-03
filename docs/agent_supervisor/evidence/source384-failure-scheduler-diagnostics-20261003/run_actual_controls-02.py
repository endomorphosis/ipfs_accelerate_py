"""Qualify actual checked-out owners, without bootstrap or import overrides."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

B=Path(__file__).resolve().parent
A=B.parents[1]/'.worktrees/ir-release-accelerate-20261002'
D=B.parents[1]/'.worktrees/ir-release-datasets-20261002'
from benchmarks.agent_supervisor.container_coding import terminal_resource_diagnostics, terminal_source384_qualification, terminal_container_supervisor
modules=[terminal_resource_diagnostics,terminal_source384_qualification,terminal_container_supervisor]
for module in modules:
 assert Path(module.__file__).resolve()==A/'benchmarks/agent_supervisor/container_coding'/(module.__name__.rsplit('.',1)[-1]+'.py')
for row in json.loads((B/'final-pins.json').read_text())['files']:
 assert hashlib.sha256((A/row['path']).read_bytes()).hexdigest()==row['after_sha256']
import pytest
new=A/'benchmarks/agent_supervisor/container_coding/test_terminal_resource_diagnostics.py'
old=A/'benchmarks/agent_supervisor/container_coding/test_terminal_failure_diagnostics.py'
args=['-q',str(new),str(old),'-k','not test_actual_canonical_resource_probe_reports_only_finite_resource_values',
 '--junitxml='+str(B/'actual-02.xml')]
files=[Path(m.__file__) for m in modules]+[new,old,Path(__file__)]
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(B/'actual-02-command.json').write_text(json.dumps(dict(argv=[sys.executable,str(Path(__file__))],pytest_args=args,
 source_pins=pins,actual_loaded_modules={m.__name__:m.__file__ for m in modules},import_override=False,
 git_heads={str(root):subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() for root in (A,D)},
 environment={k:os.environ.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
 scope='controlled snapshots/resources and actual embedded-probe/driver failure paths; no native model/Docker/actual host sampling'),indent=2)+'\n')
start=time.monotonic();code=int(pytest.main(args))
(B/'actual-02-exit.json').write_text(json.dumps(dict(exit_code=code,seconds=time.monotonic()-start,
 source_pins_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
raise SystemExit(code)
