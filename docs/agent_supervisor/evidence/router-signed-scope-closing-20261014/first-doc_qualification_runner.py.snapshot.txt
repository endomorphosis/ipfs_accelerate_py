"""Frozen local workflow blocks and signed closing-replay evidence custody.

No tests, providers, models, training, Docker or official verifier are invoked.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
WORKFLOW = ROOT/'.github/workflows/documentation-gates.yml'
WORKFLOW_SHA = '0a46fa5b8905600cb9f74adaaec8b8aeb5f89878612a43fb7a847e586bde3763'
CHECKER = ROOT/'scripts/docs/check_current_docs_links.py'
CHECKER_SHA = '596a0038b18d592f2e605643890afc85e457558f7b887932a474557d7586819e'
EXPECTED_NAMES = ('Supervisor primary-doc vocabulary','Maintained-surface relative links',
    'Packaging version pin alignment','Closeout navigation markers')
BASE = "b184d30229b3b8f23a390ca24580962d44620536"
PRESERVED_EVIDENCE_REVISION = "b184d30229b3b8f23a390ca24580962d44620536"
DATASETS_REVISION = "a0f60d42f836571a0023515e92dfb0e5724f0d79"
KIT_REVISION = "40c98c71b95be750509ffdb21e7e5494d91b4e79"
ROSTER = ('benchmarks/agent_supervisor/container_coding/test_terminal_signed_working_set_closing.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_worker_signed_scope.py', 'benchmarks/agent_supervisor/container_coding/test_terminal_signed_compact_handoff.py', 'test/api/test_semantic_working_set_dispatch.py', 'test/api/test_semantic_router_integration.py', 'test/api/test_router_implementation_runner.py', 'test/api/test_router_public_instruction.py', 'test/api/test_repository_world_dispatch.py')
DOCUMENTS = (str((PHASE/'README.md').relative_to(ROOT)),
    'docs/agent_supervisor/repository_proof_context.md',
    'docs/agent_supervisor/reversible_codebase_views_plan.md')
NEW_SUITES = {"benchmarks.agent_supervisor.container_coding.test_terminal_signed_working_set_closing"}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def need(value,reason):
    if not value:raise ValueError(reason)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def bind(path):
    raw=path.read_bytes();return dict(path=str(path.relative_to(ROOT)),bytes=len(raw),sha256=sha(raw))

def extract_workflow_blocks():
    raw = WORKFLOW.read_bytes()
    need(sha(raw) == WORKFLOW_SHA, "documentation workflow differs from reviewed bytes")
    lines, blocks, index, name = raw.decode("utf-8").splitlines(), [], 0, None
    while index < len(lines):
        line = lines[index]
        if line.startswith("      - name: "):
            name = line.removeprefix("      - name: ")
        match = re.fullmatch(r"        run: (.+)", line)
        if match:
            value = match.group(1)
            if value == "|":
                body = []
                index += 1
                while index < len(lines) and (not lines[index] or lines[index].startswith("          ")):
                    body.append(lines[index][10:] if lines[index] else "")
                    index += 1
                script, form = "\n".join(body).rstrip() + "\n", "indented_block"
            else:
                need(value and not value.startswith(("|", ">")), "unsupported workflow run scalar")
                script, form = value + "\n", "single_command"
                index += 1
            blocks.append(dict(name=name, form=form, script=script,
                script_sha256=sha(script.encode("utf-8"))))
            continue
        need(not line.lstrip().startswith("run:"), "unrecognized workflow run indentation")
        index += 1
    need(tuple(row["name"] for row in blocks) == EXPECTED_NAMES,
         "exact four reviewed workflow run blocks required")
    need([row["form"] for row in blocks] == ["single_command", "single_command", "indented_block", "indented_block"],
         "expected two single commands and two indented blocks")
    return blocks


def qualification():
    roster=json.loads((PHASE/'targets.json').read_bytes())
    need(type(roster) is list and tuple(roster)==ROSTER, 'exact eight-suite roster required')
    directories=[path for path in PHASE.glob('qualification-*') if path.is_dir()]
    need(all(re.fullmatch(r'qualification-[0-9]{2,3}',path.name) for path in directories),
         'qualification directory name differs')
    records=[]
    for attempt in sorted(directories,key=lambda path:int(path.name.removeprefix('qualification-'))):
        report=json.loads((attempt/'commands.json').read_bytes())
        need(report['accelerator_base']==BASE, 'retained qualification base differs')
        for name in ('tests.stdout.log','tests.stderr.log','junit.xml'):
            raw=(attempt/name).read_bytes()
            need(report[name]==dict(bytes=len(raw),sha256=sha(raw)), 'raw qualification custody differs')
        for row in report['frozen_changed_source_snapshots']:
            path=ROOT/row['snapshot']
            need(path.is_relative_to(attempt/'source-snapshot')
                 and path.resolve(strict=True)==path and not path.is_symlink(),
                 'snapshot must remain in its original qualification directory')
            raw=path.read_bytes()
            need((len(raw),sha(raw))==(row['bytes'],row['sha256']), 'qualification snapshot custody differs')
        records.append((attempt,report))
    full=[(attempt,report) for attempt,report in records if report['targets']==roster]
    need(full and full[-1][1]['status']=='pass', 'latest complete eight-suite qualification must pass')
    attempt,report=full[-1]
    need(report['source_before_after_equal'] is True and report['before']==report['after']
         and report['collection_execution_agrees'] is True, 'source/count custody differs')
    need(type(report['pytest_exit_code']) is int and report['pytest_exit_code']==0,
         'qualification exit differs')
    need(report['datasets_execution_revision']==DATASETS_REVISION
         and report['datasets_head_after']==DATASETS_REVISION
         and report['kit_execution_revision']==KIT_REVISION
         and report['kit_head_after']==KIT_REVISION, 'frozen execution dependency differs')
    need(report['historical_codec_and_edit_mapper_unchanged'] is True,
         'original reversible view/edit mapper changed')
    cases=list(ET.parse(attempt/'junit.xml').getroot().iter('testcase'))
    names=[(case.get('classname'),case.get('name')) for case in cases]
    need(all(type(suite) is str and suite and type(name) is str and name for suite,name in names),
         'exact JUnit identities required')
    tests=len(cases)
    need(type(report['collected_tests']) is int and tests>0
         and len(set(names))==tests==report['collected_tests']==report['distinct_junit_cases'],
         'distinct JUnit identities differ')
    outcome=dict(tests=tests,failures=sum(1 for case in cases for _ in case.iter('failure')),
                 errors=sum(1 for case in cases for _ in case.iter('error')),
                 skipped=sum(1 for case in cases for _ in case.iter('skipped')))
    need(outcome==report['junit']==dict(tests=tests,failures=0,errors=0,skipped=0),
         'final JUnit outcome contains unsuccessful cases')
    suites={path.removesuffix('.py').replace('/','.') for path in roster}
    need({suite for suite,_ in names}==suites, 'JUnit suite population differs')
    new=sum(suite in NEW_SUITES for suite,_ in names)
    need(new>0, 'new signed-closing suite has no qualified cases')
    need(report['authored_pinned_prover_transcripts'] is False
         and report['authored_provider_stubs_exercised'] is True
         and report['authored_legacy_recording_index_seam'] is True
         and report['actual_inert_child_probes'] is True
         and report['worker_uid_qualified'] is False
         and report['production_coding_child_adoption'] is False
         and report['docker_run'] is False and report['official_task_result'] is None
         and report['proof_authority'] is False and report['completion_authority'] is False
         and report['total_session_token_savings_established'] is False,
         'qualified authored/observational evidence limits differ')
    need(all(type(report[field]) is int and report[field]==0 for field in
             ('provider_calls','external_provider_calls','model_calls','training_steps',
              'solver_calls','native_solver_or_kernel_executions')),
         'qualification invoked external proof/model/training execution')
    return attempt,report,new


def guard(*, excluded_outputs):
    attempt,report,new=qualification()
    checked=[]
    datasets=Path(report['datasets_execution_path'])
    for row in report['before']:
        owner=ROOT if row['repository']=='accelerator' else datasets
        raw=(owner/row['path']).read_bytes()
        need((len(raw),sha(raw))==(row['bytes'],row['sha256']),'qualified source differs: '+row['path'])
        if row['repository']!='accelerator':
            need(raw==subprocess.check_output(['git','show',report['datasets_execution_revision']+':'+row['path']],cwd=owner),'immutable datasets object differs')
        checked.append(row)
    inventory=json.loads((PHASE/'source-inventory.json').read_bytes())
    identities=[(row['repository'],row['path']) for row in checked]
    expected=[(row['repository'],row['path']) for row in inventory]
    need(len(set(identities))==len(identities) and len(set(expected))==len(expected)
         and set(identities)==set(expected), 'qualification source inventory differs')
    need(len(checked)==265,'listed dependency population differs')
    need(sum(row['repository']=='accelerator' for row in checked)==214, 'accelerator binding population differs')
    need(sum(row['repository']!='accelerator' for row in checked)==51, 'datasets binding population differs')
    # Preserve every prior evidence object from the published parent source.
    # The new closing qualification is a separate capsule.
    git('merge-base','--is-ancestor',BASE,PRESERVED_EVIDENCE_REVISION)
    git('merge-base','--is-ancestor',PRESERVED_EVIDENCE_REVISION,'HEAD')
    evidence=[]
    for prefix,ref in [('docs/agent_supervisor/evidence',PRESERVED_EVIDENCE_REVISION)]:
        entries=git('ls-tree','-r','-z',ref,'--',prefix).split(b'\0')
        verified=[]
        for entry in entries:
            if not entry:continue
            metadata,name=entry.split(b'\t',1);mode,kind,oid=metadata.decode().split()
            name=name.decode();raw=(ROOT/name).read_bytes()
            actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            need(kind=='blob' and mode in ('100644','100755') and actual==oid,'retained prior evidence changed: '+name)
            verified.append(dict(path=name,git_object=oid,sha256=sha(raw),bytes=len(raw)))
        evidence.append(dict(revision=ref,file_count=len(verified),tree_records_sha256=sha(json.dumps(verified,sort_keys=True,separators=(',',':')).encode()),all_git_objects_match=True))
    baseline_reports=[]
    for baseline in sorted(PHASE.glob('baseline-[0-9][0-9]')):
        record=json.loads((baseline/'commands.json').read_bytes())
        need(record['baseline_control'] is True and record['status']=='expected_regression'
             and record['accelerator_execution_head']==BASE and record['source_before_after_equal'] is True
             and record['unpatched_omitted_source_drift_reaches_authored_provider'] is True,
             'unpatched regression control differs')
        for name in ('tests.stdout.log','tests.stderr.log','junit.xml'):
            raw=(baseline/name).read_bytes()
            need(record[name]==dict(bytes=len(raw),sha256=sha(raw)), 'baseline raw custody differs')
        for row in record['frozen_changed_source_snapshots']:
            raw=(ROOT/row['snapshot']).read_bytes()
            need((len(raw),sha(raw))==(row['bytes'],row['sha256']), 'baseline snapshot custody differs')
            if row['path']=='ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py':
                need(raw==git('show',BASE+':'+row['path']), 'baseline router snapshot differs from published source')
        baseline_reports.append(bind(baseline/'commands.json'))
    need(len(baseline_reports)==1, 'one retained regression control required')
    return dict(qualification_path=str(attempt.relative_to(ROOT)),qualification_sha256=sha((attempt/'commands.json').read_bytes()),
        baseline_controls=baseline_reports,selected_tests=report['collected_tests'],suites=len(report['targets']),new_cases=new,source_bindings=checked,
        preserved_evidence=evidence,documents=[bind(ROOT/name) for name in DOCUMENTS],
        read_guards=[bind(p) for p in (Path(__file__),WORKFLOW,CHECKER)],
        current_evidence=[bind(p) for p in sorted(PHASE.rglob('*')) if p.is_file()
            and p not in excluded_outputs],qualification_attempts=len(list(PHASE.glob('qualification-[0-9][0-9]'))))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',help='optional two- or three-digit documentation attempt; original filenames are the default')
    args=parser.parse_args(argv)
    if args.attempt is None:
        receipt=PHASE/'documentation-checks.json';log=PHASE/'documentation-workflow.log'
    else:
        need(re.fullmatch(r'[0-9]{2,3}',args.attempt) is not None,
             'exact documentation attempt required')
        receipt=PHASE/('documentation-checks-'+args.attempt+'.json')
        log=PHASE/('documentation-workflow-'+args.attempt+'.log')
    excluded_outputs={receipt,log}
    need(not receipt.exists() and not log.exists(),'documentation outputs already exist')
    need(sha(CHECKER.read_bytes())==CHECKER_SHA,'reviewed checker changed')
    before=guard(excluded_outputs=excluded_outputs);blocks=extract_workflow_blocks()
    report=dict(schema='router-signed-scope-closing-documentation@1',started_at_utc=utc(),before=before,
        documentation_attempt=args.attempt,
        exact_workflow_run_blocks=blocks,workflow_sha256=WORKFLOW_SHA,steps=[],
        additional_tests=False,models=False,providers=False,solver_calls=0,docker=False,
        native_solver_or_kernel_executions=0,proof_authority=False,completion_authority=False,
        total_session_token_savings_established=False,complete_import_closure_attested=False,
        hosted_ci_passed_here=False,status='running')
    with receipt.open('x') as stream:
        stream.write(json.dumps(report,indent=2,sort_keys=True)+'\n')
    env=os.environ.copy()
    for key in ('BASH_ENV','ENV','SHELLOPTS','CDPATH','PYTHONPATH','PYTHONSTARTUP'):env.pop(key,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PATH=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH',''))
    with log.open('x') as output:
        for block in blocks:
            argv=['bash','--noprofile','--norc','-euo','pipefail','-c',block['script']]
            result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
            row=dict(name=block['name'],argv=argv,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
            report['steps'].append(row);output.write(json.dumps(row)+'\n');output.flush()
            need(result.returncode==0,'documentation gate failed: '+block['name'])
    spec=importlib.util.spec_from_file_location('router_signed_scope_documentation_checker',CHECKER)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    need(checker.ROOT==ROOT,'link checker root differs')
    links=[]
    for name in DOCUMENTS:
        path=ROOT/name;text=checker.strip_markup_noise(path.read_text())
        matches=[checker.split_target(match.group('target')) for match in checker.LINK_RE.finditer(text)]
        relative=[(target,fragment) for target,fragment in matches if not checker.is_external(target,fragment)]
        errors=checker.check_file(path,check_anchors=True)
        links.append(dict(path=name,errors=errors,relative_links=len(relative),anchors=sum(bool(fragment) for _,fragment in relative)))
    report.update(manual_links=links,relative_links=sum(row['relative_links'] for row in links),anchors=sum(row['anchors'] for row in links))
    need(all(not row['errors'] for row in links),'manual links or anchors failed')
    after=guard(excluded_outputs=excluded_outputs);need(before==after,'documentation read guards changed')
    report.update(after=after,status='pass',completed_at_utc=utc(),log_binding=bind(log))
    receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:report[key] for key in ('status','relative_links','anchors')}))
    return 0

if __name__=='__main__':raise SystemExit(main())
