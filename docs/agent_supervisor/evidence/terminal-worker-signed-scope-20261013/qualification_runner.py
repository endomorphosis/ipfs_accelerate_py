"""Execute signed Terminal scope producer and consumer regressions once with retained source custody.

No separate collection, provider, model, Docker or official verifier invocation.
Dependency bindings cover the explicit listed files, not a full import closure.
"""
from __future__ import annotations
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
BASE = "f100a21ecb2edc7c827989bf0144478f3911d4d2"
DATASETS = Path('/home/barberb/lift_coding/.worktrees/terminal-worker-scope-datasets-20261013')
DATASETS_REVISION = '59561dd2979e49fd26aa3fcfed6a5edbca1f718f'
KIT = Path('/home/barberb/lift_coding/.worktrees/publication-main-20261004/kit')
KIT_REVISION = '8ac00bb1974b1378e12091d546d8425af34049e0'
TERMINAL_SITE = Path('/home/barberb/lift_coding/.venvs/terminal-bench-harbor/lib/python3.12/site-packages')
B = 'benchmarks/agent_supervisor/container_coding/'
TARGETS = tuple(B + name for name in (
    'test_terminal_worker_signed_scope.py',
    'test_terminal_scope_worker_capability.py',
    'test_terminal_worker_capability.py',
    'test_terminal_deployment.py',
    'test_terminal_context_audit.py',
    'test_terminal_signed_semantic_scope.py',
    'test_source_view_worker_wiring.py',
)) + tuple('test/api/' + name for name in (
    'test_router_signed_terminal_scope.py',
    'test_router_public_instruction.py',
    'test_semantic_instruction_dispatch.py',
    'test_semantic_instruction_view.py',
    'test_semantic_router_integration.py',
    'test_semantic_router_translation.py',
    'test_router_implementation_runner.py',
    'test_router_failure_diagnostics.py',
))
FOCUSED_TARGETS = (TARGETS[0], TARGETS[1], TARGETS[7])




def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def git(root, *args): return subprocess.check_output(['git', *args], cwd=root)
def write(path, data): path.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def bindings():
    # Reuse only the earlier reviewed path inventory, never an earlier result.
    old = json.loads((ROOT/'docs/agent_supervisor/evidence/signed-terminal-source-scope-20261013/qualification-02/commands.json').read_bytes())
    names = {(row['repository'], row['path']) for row in old['before']}
    names.update(('accelerator', name) for name in TARGETS)
    names.add(('accelerator','ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py'))
    names.add(('accelerator', str(Path(__file__).relative_to(ROOT))))
    names.update(('accelerator', name) for name in git(ROOT, 'diff', '--name-only', BASE).decode().splitlines()
                 if name.endswith('.py'))
    module = ast.parse((ROOT/(B+'terminal_worker_capability.py')).read_bytes())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='SOURCE_PATHS' for t in node.targets):
            names.update(('accelerator', name) for name in ast.literal_eval(node.value))
    rows=[]
    for repository, name in sorted(names):
        root = ROOT if repository=='accelerator' else DATASETS
        raw=(root/name).read_bytes()
        row=dict(repository=repository, path=name, bytes=len(raw), sha256=sha(raw))
        if repository!='accelerator':
            original=git(DATASETS, 'show', DATASETS_REVISION+':'+name)
            row['matches_immutable_git_object']=original==raw
            if original!=raw: raise ValueError('Datasets source differs: '+name)
        rows.append(row)
    return rows


def main():
    if len(sys.argv) not in (2,3) or re.fullmatch(r'qualification-[0-9]{2,3}',sys.argv[1]) is None or (len(sys.argv)==3 and sys.argv[2]!='focused'):
        raise SystemExit('Usage: qualification_runner.py qualification-01')
    targets = FOCUSED_TARGETS if len(sys.argv)==3 else TARGETS
    assert git(DATASETS,'rev-parse','HEAD').decode().strip()==DATASETS_REVISION
    assert git(KIT,'rev-parse','HEAD').decode().strip()==KIT_REVISION
    assert TERMINAL_SITE.is_dir()
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    attempt=PHASE/sys.argv[1];attempt.mkdir(exist_ok=False)
    before=bindings()
    changed=set(git(ROOT, 'diff', '--name-only', BASE).decode().splitlines())
    changed.update((B+'test_terminal_worker_signed_scope.py',B+'test_terminal_scope_worker_capability.py','test/api/test_router_signed_terminal_scope.py'))
    snapshots=[]
    for name in sorted(changed):
        if not name.endswith('.py') or not (ROOT/name).is_file(): continue
        raw=(ROOT/name).read_bytes()
        snapshot=attempt/'source-snapshot'/(name+'.snapshot.txt')
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(raw)
        snapshots.append(dict(path=name,bytes=len(raw),sha256=sha(raw),snapshot=str(snapshot.relative_to(ROOT))))
    private=Path(tempfile.mkdtemp(prefix='terminal-worker-signed-scope-'+sys.argv[1]+'-'))
    updates=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI='0',
        IPFS_KIT_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONPATH=os.pathsep.join(map(str,(ROOT,DATASETS,TERMINAL_SITE,KIT))),
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(private/'orchestration'),
        IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(private/'seal.duckdb'))
    removed=('IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET','IPFS_ACCELERATE_AGENT_QUACK_TOKEN',
        'IPFS_ACCELERATE_META_INDEX_DUCKDB','IPFS_ACCELERATE_META_INDEX_DUCKLAKE','PYTEST_ADDOPTS',
        'PYTEST_PLUGINS','IPFS_TEST_PROOF_REUSE_MODE')
    env=os.environ.copy()
    for name in removed: env.pop(name,None)
    env.update(updates)
    argv=[sys.executable,'-B','-m','pytest','-c','pytest.ini',*targets,'--color=no',
        '-o','log_cli=false','-o','cache_dir='+str(private/'pytest-cache'),'--tb=short',
        '--junitxml='+str(attempt/'junit.xml')]
    record=dict(schema='terminal-worker-signed-scope-qualification@1',status='running',
        started_at_utc=utc(),accelerator_base=BASE,accelerator_execution_head=git(ROOT,'rev-parse','HEAD').decode().strip(),
        datasets_execution_revision=DATASETS_REVISION,datasets_execution_path=str(DATASETS),
        kit_execution_revision=KIT_REVISION,kit_execution_path=str(KIT),interpreter=sys.executable,
        targets=list(targets),focused_execution=targets==FOCUSED_TARGETS,pytest_invocations=1,separate_collection=False,before=before,
        source_binding_scope='listed_dependencies_not_complete_import_closure',
        frozen_changed_source_snapshots=snapshots,
        historical_evidence_result_reused=False,argv=argv,cwd=str(ROOT),environment_updates=updates,
        inherited_environment_names_removed=list(removed),private_runtime_directory=str(private),
        provider_calls=0,external_provider_calls=0,authored_provider_stubs_exercised=True,model_calls=0,training_steps=0,solver_calls=0,docker_run=False,
        official_task_result=None,total_session_token_savings_established=False,
        runtime_databases_published=False,proof_authority=False,completion_authority=False)
    write(attempt/'commands.json',record)
    print('Starting '+str(len(targets))+' selected suites; '+str(len(before))+' source bindings.',flush=True)
    started=time.monotonic()
    try:
        with (attempt/'tests.stdout.log').open('wb') as stdout,(attempt/'tests.stderr.log').open('wb') as stderr:
            result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        record['pytest_exit_code']=result.returncode
    except Exception as error:
        record['runner_error']=dict(type=type(error).__name__,message=str(error))
    record.update(completed_at_utc=utc(),elapsed_seconds=round(time.monotonic()-started,3))
    try:
        record['after']=bindings()
        record['source_before_after_equal']=record['before']==record['after']
        record['datasets_head_after']=git(DATASETS,'rev-parse','HEAD').decode().strip()
        record['kit_head_after']=git(KIT,'rev-parse','HEAD').decode().strip()
        record['historical_codec_and_edit_mapper_unchanged']=all(
            (ROOT/name).read_bytes()==git(ROOT,'show',BASE+':'+name) for name in (
                'ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py',
                'ipfs_accelerate_py/agent_supervisor/runtime/source_view_edits.py'))
        for name in ('tests.stdout.log','tests.stderr.log','junit.xml'):
            raw=(attempt/name).read_bytes()
            record[name]=dict(bytes=len(raw),sha256=sha(raw))
        text=(attempt/'tests.stdout.log').read_text(errors='replace')
        counts=re.findall(r'^(?:collecting \.\.\. )?collected ([0-9]+) items?\s*$',text,re.M)
        record['collection_count_observations']=list(map(int,counts))
        record['collected_tests']=int(counts[0]) if len(counts)==1 else None
        document=ET.parse(attempt/'junit.xml').getroot()
        suites=[document] if document.tag=='testsuite' else document.findall('testsuite')
        record['junit']={key:sum(int(suite.get(key,0)) for suite in suites)
            for key in ('tests','failures','errors','skipped')}
        cases=[(case.get('classname'),case.get('name')) for case in document.iter('testcase')]
        record['distinct_junit_cases']=len(set(cases))
        record['collection_execution_agrees']=record['collected_tests']==len(cases)==len(set(cases))==record['junit']['tests']
        record['status']='pass' if (record.get('pytest_exit_code')==0 and record['source_before_after_equal']
            and record['datasets_head_after']==DATASETS_REVISION and record['kit_head_after']==KIT_REVISION
            and record['historical_codec_and_edit_mapper_unchanged'] and record['collection_execution_agrees']
            and record['collected_tests']>0 and record['junit']['errors']==record['junit']['failures']==record['junit']['skipped']==0) else 'fail'
    except Exception as error:
        record['observation_error']=dict(type=type(error).__name__,message=str(error));record['status']='fail'
    write(attempt/'commands.json',record)
    print(json.dumps({key:record.get(key) for key in ('status','pytest_exit_code','collected_tests','junit','elapsed_seconds','observation_error')}),flush=True)
    return 0 if record['status']=='pass' else 1


if __name__=='__main__': raise SystemExit(main())
