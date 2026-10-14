"""Qualify signed complete-source replay after working-set callbacks.

No separate collection or external provider, model, Docker or official verifier invocation.
Authored provider/boundary seams and real inert socket client probes are identified explicitly.
Dependency bindings cover the explicit listed files, not a full import closure.
"""
from __future__ import annotations
import ast
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
BASE = "b184d30229b3b8f23a390ca24580962d44620536"
BASELINE = Path("/home/barberb/lift_coding/.worktrees/repository-proof-context-20261014")
EXECUTION_ROOT = ROOT
DATASETS = Path('/home/barberb/lift_coding/.worktrees/repository-proof-context-datasets-20261014')
DATASETS_REVISION = 'a0f60d42f836571a0023515e92dfb0e5724f0d79'
KIT = Path('/home/barberb/lift_coding/.worktrees/repository-proof-context-kit-20261014')
KIT_REVISION = '40c98c71b95be750509ffdb21e7e5494d91b4e79'
B = 'benchmarks/agent_supervisor/container_coding/'
TARGETS = tuple(json.loads((PHASE/'targets.json').read_bytes()))

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def git(root, *args): return subprocess.check_output(['git', *args], cwd=root)
def write(path, data): path.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def bindings():
    names = {(row['repository'], row['path']) for row in json.loads((PHASE/'source-inventory.json').read_bytes())}
    rows=[]
    for repository, name in sorted(names):
        root = (EXECUTION_ROOT if (EXECUTION_ROOT/name).is_file() else ROOT) if repository=='accelerator' else DATASETS
        raw=(root/name).read_bytes()
        row=dict(repository=repository, path=name, source_checkout=str(root), bytes=len(raw), sha256=sha(raw))
        if repository!='accelerator':
            original=git(DATASETS, 'show', DATASETS_REVISION+':'+name)
            row['matches_immutable_git_object']=original==raw
            if original!=raw: raise ValueError('Datasets source differs: '+name)
        rows.append(row)
    return rows


def main():
    global EXECUTION_ROOT
    if len(sys.argv) != 2 or re.fullmatch(r'(?:baseline|qualification)-[0-9]{2,3}',sys.argv[1]) is None:
        raise SystemExit('Usage: qualification_runner.py baseline-01|qualification-01')
    baseline = sys.argv[1].startswith('baseline-')
    EXECUTION_ROOT = BASELINE if baseline else ROOT
    targets = (str(ROOT / TARGETS[0])+'::test_post_page_drift_refuses_signed_dispatch[canonical-omitted-bytes]',) if baseline else TARGETS
    if baseline: assert git(EXECUTION_ROOT,'rev-parse','HEAD').decode().strip()==BASE
    else: subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=EXECUTION_ROOT,check=True)
    assert git(DATASETS,'rev-parse','HEAD').decode().strip()==DATASETS_REVISION
    assert git(KIT,'rev-parse','HEAD').decode().strip()==KIT_REVISION
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    attempt=PHASE/sys.argv[1];attempt.mkdir(exist_ok=False)
    before=bindings()
    changed=set(git(ROOT, 'diff', '--name-only', BASE).decode().splitlines())
    changed.update((TARGETS[0],'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',str(Path(__file__).relative_to(ROOT))))
    snapshots=[]
    for name in sorted(changed):
        if not name.endswith('.py') or not (ROOT/name).is_file(): continue
        source_root=EXECUTION_ROOT if baseline and name=='ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py' else ROOT
        source=source_root/name
        raw=source.read_bytes()
        snapshot=attempt/'source-snapshot'/(name+'.snapshot.txt')
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(raw)
        snapshots.append(dict(path=name,source_checkout=str(source_root),bytes=len(raw),sha256=sha(raw),snapshot=str(snapshot.relative_to(ROOT))))
    private=Path(tempfile.mkdtemp(prefix='router-signed-scope-closing-'+sys.argv[1]+'-'))
    updates=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI='0',
        IPFS_KIT_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONPATH=os.pathsep.join(map(str,(EXECUTION_ROOT,DATASETS,KIT))),
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(private/'orchestration'),
        IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(private/'seal.duckdb'))
    removed=('IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET','IPFS_ACCELERATE_AGENT_QUACK_TOKEN',
        'IPFS_ACCELERATE_META_INDEX_DUCKDB','IPFS_ACCELERATE_META_INDEX_DUCKLAKE','PYTEST_ADDOPTS',
        'PYTEST_PLUGINS','IPFS_TEST_PROOF_REUSE_MODE')
    env=os.environ.copy()
    for name in removed: env.pop(name,None)
    env.update(updates)
    argv=[sys.executable,'-B','-m','pytest','-c','pytest.ini',*targets,'--color=no',
        '--basetemp='+str(private/'pytest-temp'),'-o','log_cli=false','-o','cache_dir='+str(private/'pytest-cache'),'--tb=short','--maxfail=1',
        '--junitxml='+str(attempt/'junit.xml')]
    if baseline: argv.append('--import-mode=importlib')
    record=dict(schema='router-signed-scope-closing-qualification@1',status='running',baseline_control=baseline,
        started_at_utc=utc(),accelerator_base=BASE,accelerator_execution_head=git(EXECUTION_ROOT,'rev-parse','HEAD').decode().strip(),
        datasets_execution_revision=DATASETS_REVISION,datasets_execution_path=str(DATASETS),
        kit_execution_revision=KIT_REVISION,kit_execution_path=str(KIT),interpreter=sys.executable,
        interpreter_version=sys.version,platform=platform.platform(),
        installed_distribution_versions={name:importlib.metadata.version(name)
            for name in ('pytest','duckdb','cryptography','numpy')},
        targets=list(targets),pytest_invocations=1,separate_collection=False,before=before,
        source_binding_scope='listed_dependencies_not_complete_import_closure',
        frozen_changed_source_snapshots=snapshots,
        historical_evidence_result_reused=False,argv=argv,cwd=str(EXECUTION_ROOT),environment_updates=updates,
        inherited_environment_names_removed=list(removed),private_runtime_directory=str(private),
        provider_calls=0,external_provider_calls=0,authored_provider_stubs_exercised=True,authored_pinned_prover_transcripts=False,native_solver_or_kernel_executions=0,authored_legacy_recording_index_seam=True,authored_listener_and_deployment_metadata_seams=True,model_calls=0,training_steps=0,solver_calls=0,docker_run=False,
        actual_inert_child_probes=not baseline,worker_uid_qualified=False,production_coding_child_adoption=False,
        official_task_result=None,total_session_token_savings_established=False,
        runtime_databases_published=False,proof_authority=False,completion_authority=False)
    write(attempt/'commands.json',record)
    print('Starting '+str(len(targets))+' selected targets; '+str(len(before))+' source bindings.',flush=True)
    started=time.monotonic()
    try:
        with (attempt/'tests.stdout.log').open('wb') as stdout,(attempt/'tests.stderr.log').open('wb') as stderr:
            result=subprocess.run(argv,cwd=EXECUTION_ROOT,env=env,stdout=stdout,stderr=stderr)
        record['pytest_exit_code']=result.returncode
    except Exception as error:
        record['runner_error']=dict(type=type(error).__name__,message=str(error))
    record.update(completed_at_utc=utc(),elapsed_seconds=round(time.monotonic()-started,3))
    text=''
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
    if baseline:
        observations=[]
        for line in text.splitlines():
            if line.startswith('{'):
                try: value=json.loads(line)
                except ValueError: continue
                if type(value) is dict and value.get('schema')=='router-implementation-invocation@1': observations.append(value)
        # The unchanged parent must reach one authored provider after valid
        # paging despite omitted signed-input drift. An unrelated setup error
        # does not qualify this regression control.
        record['baseline_router_observations']=[{key:row.get(key) for key in ('status','router_calls','terminal_semantic_scope_dispatch_rechecked','failure_phase')} for row in observations]
        reproduced=(record.get('pytest_exit_code')==1 and record.get('junit')==dict(tests=1,failures=1,errors=0,skipped=0)
            and 'DID NOT RAISE' in text and any(row.get('status')=='provider_returned' and row.get('router_calls')==1
                and row.get('terminal_semantic_scope_dispatch_rechecked') is True for row in observations)
            and record.get('source_before_after_equal') is True)
        record['status']='expected_regression' if reproduced else 'fail'
        record['unpatched_omitted_source_drift_reaches_authored_provider']=reproduced
    write(attempt/'commands.json',record)
    print(json.dumps({key:record.get(key) for key in ('status','pytest_exit_code','collected_tests','junit','elapsed_seconds','observation_error')}),flush=True)
    return 0 if record['status'] in {'pass','expected_regression'} else 1


if __name__=='__main__': raise SystemExit(main())
