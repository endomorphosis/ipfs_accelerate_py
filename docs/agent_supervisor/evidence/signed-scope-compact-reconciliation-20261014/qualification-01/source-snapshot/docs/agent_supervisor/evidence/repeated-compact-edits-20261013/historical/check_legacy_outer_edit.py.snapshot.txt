"""Read-only default config/control wire comparison against the frozen parent."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import types
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as current
from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION

Q=Path('/home/barberb/lift_coding/.worktrees/supervisor-token-growth-20261013')
BASE='b606929cdc0ba00902c027994795239a34e50433'
R=Path('/home/barberb/lift_coding/artifacts/repeated-compact-edits-20261013')
def old(name):
    path='benchmarks/agent_supervisor/container_coding/'+name+'.py'
    source=subprocess.check_output(['git','show',BASE+':'+path],cwd=Q)
    module=types.ModuleType('benchmarks.agent_supervisor.container_coding._historical_'+name)
    module.__file__=str(Q/path)
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module,hashlib.sha256(source).hexdigest()
legacy,legacy_sha=old('full_supervisor_benchmark')
legacy_controls,controls_sha=old('benchmark_controls')
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def declared(owner,cfg):
    return owner.build_controls(cfg,task_input_sha256={'instruction.md':'a'*64},task=current.TASK,
        model=current.MODEL,reasoning_effort='high',cli_version=CLI_VERSION)
rows=[]
for arm,numerical,planning in itertools.product(('full','no-index'),(None,'python-numerical-control-structure@3'),('model','declared-contract@1')):
    for profile,mode in ([(p,m) for p in ('disabled','identity@1','python-whitespace@1','python-comments@1') for m in (('disabled','mapped-span@1') if p!='disabled' else ('disabled',))] if arm=='full' else [('disabled','disabled')]):
        args=(Path('/authored/dataset'),Path('/authored/output'),Path('/authored/archive'),arm)
        kwargs=dict(numerical_formalization_profile=numerical,planning_mode=planning,source_view_profile=profile,**({"source_edit_mode":mode} if mode!="disabled" else {}))
        before=legacy.config_for(*args,**kwargs)
        after=current.config_for(*args,**kwargs)
        explicit=current.config_for(*args,**dict(kwargs,source_edit_mode=mode))
        first=declared(legacy_controls,before);second=declared(controls,after)
        assert encoded(before)==encoded(after)==encoded(explicit)
        assert encoded(first)==encoded(second)
        rows.append(dict(arm=arm,numerical=numerical,planning=planning,source_view_profile=profile,source_edit_mode=mode,
            config_sha256=hashlib.sha256(encoded(after)).hexdigest(),controls_sha256=hashlib.sha256(encoded(second)).hexdigest(),
            config_bytes_equal=True,controls_bytes_equal=True))
result=dict(schema='source-edit-legacy-wire-replay@1',baseline=BASE,baseline_source_sha256={'benchmark':legacy_sha,'controls':controls_sha},
    cases=rows,case_count=len(rows),complete=True,provider_calls=0,
    scope='Serialized absent/explicit-disabled and mapped-span@1 config/control bytes compared with the frozen previous commit; selected producer pins intentionally require fresh archives.')
(R/'legacy-outer-wire-replay.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({'complete':True,'cases':len(rows),'provider_calls':0}))
