import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path('/home/barberb/lift_coding')
P = ROOT/'.worktrees/supervisor-comparison-contracts-20261011'
D = ROOT/'.worktrees/formalization-datasets-20261011'
R = ROOT/'artifacts/supervisor-numerical-adapter-20261011'
S = ROOT/'artifacts/supervisor-learned-numerical-20261011'
OUT = P/'docs/agent_supervisor/evidence/numerical-selection-consumer-20261011'
OUT.mkdir(parents=True, exist_ok=True)

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy(source, name):
    target = OUT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return name

runs = [(name, R/name/'command.json', R/name/'qualification.json', R/name/'junit.xml')
    for name in ('datasets-02','datasets-budget-01','merged-02','deployment-final','harbor-census-final')]
runs += [(name, S/name/'qualification-01-command.json', S/name/'qualification-01-exit.json',
    S/name/'qualification-01.xml') for name in ('transport','driver-guards')]
observations, distinct = [], set()
for name, command, result, junit in runs:
    data = json.loads(result.read_text())
    assert data.get('returncode', data.get('exit_code')) == 0, (name, data)
    cases = ET.parse(junit).getroot().findall('.//testcase')
    assert cases and all(not any(c.tag in ('failure','error','skipped') for c in case) for case in cases), name
    ids = {case.attrib['classname']+'::'+case.attrib['name'] for case in cases}
    distinct |= ids
    observations.append(dict(name=name, passed=len(cases), distinct_tests=len(ids),
        command=copy(command,name+'/command.json'), result=copy(result,name+'/result.json'),
        junit=copy(junit,name+'/junit.xml')))

retained = []
reasons = {
 'datasets-01':'Fixed installed Torch metadata versus loaded runtime build suffix mismatch.',
 'consumer-01':'Corrected test expected source map from manifest metadata rows to SHA strings.',
 'consumer-02':'Successful original E2E before added container/empty-population guards; superseded by merged-02.',
 'merged-01':'Collection stopped because host inference interpreter lacks Harbor; split interpreter-specific suites.',
 'deployment-01':'Initial archive-test interpreter/collection failure; superseded by deployment-final.',
 'deployment-02':'Initial archive-test launcher/plugin environment failure; superseded by deployment-final.',
 'deployment-03':'Fixture GTE path layout mismatch; superseded by deployment-final.',
 'deployment-04':'Producer file size bound below existing implementation daemon size; raised source-only cap to4MiB.',
 'deployment-05':'Concurrent source edit invalidated selected archive pins; superseded by stable merged qualification.',
}
for name, reason in reasons.items():
    directory = R/name
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in ('.json','.xml','.log'):
            files[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    retained.append(dict(run=name, local_path=str(directory), reason=reason, files=files))

for name in ('docker-01','docker-02'):
    for item in ('command.json','qualification.json','stdout.log','stderr.log','import_probe.py'):
        path = R/name/item
        if path.exists(): copy(path,name+'/'+item)

sources = {}
for label, repo, base in (('P',P,'b90b3755b51b1ff1b1a461faa96c671344be8762'),
        ('D',D,'4b9e96753')):
    changed = subprocess.check_output(['git','diff','--name-only',base,'HEAD'],cwd=repo,text=True).splitlines()
    sources[label] = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        files={name:sha(repo/name) for name in changed if name.endswith('.py') and (repo/name).is_file()})

copy(Path(__file__), 'consolidate.py')
write(OUT/'qualification.json', dict(schema='supervisor-numerical-selection-qualification@1',
    sources=sources, runs=observations, distinct_passing_tests=len(distinct), tests=sorted(distinct),
    retained_failed_or_superseded_runs=retained,
    inference=dict(checkpoint_file_sha256='26ccc7cf8420e0ed6cbbe22603e09be0e3572fb32a1e489e877fc5081f506e03',
        actual_cached_GTE_CPU_and_learned_head=True, planner_and_provider_replies='authored tests',
        live_provider_calls=0, training_steps=0, downloads=0, source_obligations_discharged=0),
    limits=dict(container_model_execution_qualified=False,
        container_blocker='Cached preview image lacks sentence-transformers, transformers and tokenizers.',
        benchmark_score=None, token_efficiency_score=None, learned_successor_refresh_available=False,
        whole_program_semantics_verified=False, grok_authentication_reprobed=False)))
print(json.dumps(dict(distinct_passing_tests=len(distinct), runs=[(x['name'],x['passed']) for x in observations])))
