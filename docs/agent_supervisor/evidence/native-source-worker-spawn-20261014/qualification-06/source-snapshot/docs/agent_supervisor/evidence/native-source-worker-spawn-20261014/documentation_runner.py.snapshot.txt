"""Frozen local workflow blocks plus native broker launch documentation custody.

No tests, providers, models, training, Docker or official verifier are invoked.
"""
from __future__ import annotations
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
BASE = "83ad1ca1a43ae53d3642efe8799d25a9df8cebf3"
DOCUMENTS = ('docs/agent_supervisor/reversible_codebase_views_plan.md',
    'docs/agent_supervisor/current_source_worktree_allocation.md',
    'docs/agent_supervisor/repository_source_scope.md',
    'docs/agent_supervisor/source_lookup_worker.md',
    str((PHASE/'README.md').relative_to(ROOT)))
NEW_TESTS = ('test_native_source_worker_spawn',
    'test_native_source_worker_spawn_custody',
    'test_native_source_worker_spawn_lock_order')

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
    records=[]
    for attempt in sorted(PHASE.glob('qualification-[0-9][0-9]')):
        report=json.loads((attempt/'commands.json').read_bytes())
        for name in ('tests.stdout.log','tests.stderr.log','junit.xml'):
            raw=(attempt/name).read_bytes()
            need(report[name]==dict(bytes=len(raw),sha256=sha(raw)), 'raw qualification custody differs')
        for row in report['frozen_changed_source_snapshots']:
            raw=(ROOT/row['snapshot']).read_bytes()
            need((len(raw),sha(raw))==(row['bytes'],row['sha256']), 'qualification snapshot custody differs')
        records.append((attempt,report))
    need(records and records[-1][1]['status']=='pass','final native source launch qualification must pass')
    attempt,report=records[-1]
    need(report['source_before_after_equal'] and report['collection_execution_agrees'], 'source/count custody differs')
    need(report['pytest_exit_code']==0 and report['accelerator_base']==BASE, 'qualification exit/base differs')
    need(all(report['junit'][key]==0 for key in ('failures','errors','skipped')), 'final qualification contains unsuccessful cases')
    cases=list(ET.parse(attempt/'junit.xml').getroot().iter('testcase'))
    names={(case.get('classname'),case.get('name')) for case in cases}
    need(len(names)==len(cases)==report['collected_tests']==report['junit']['tests'], 'distinct JUnit identities differ')
    need(all(not list(case) or all(child.tag not in {'failure','error','skipped'} for child in case) for case in cases),
         'raw JUnit contains unsuccessful cases')
    new=sum(any((name or '').endswith(suffix) for suffix in NEW_TESTS) for name,_ in names)
    need(new==53, 'new case identities differ')
    need(report['targets']==json.loads((PHASE/'targets.json').read_bytes()) and len(report['targets'])==12,'suite roster differs')
    return attempt,report,new


def guard():
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
    need(len(checked)==218,'listed dependency population differs')
    need(sum(row['repository']=='accelerator' for row in checked)==167, 'accelerator binding population differs')
    need(sum(row['repository']!='accelerator' for row in checked)==51, 'datasets binding population differs')
    evidence=[]
    for prefix,ref in [('docs/agent_supervisor/evidence',BASE)]:
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
    return dict(qualification_path=str(attempt.relative_to(ROOT)),qualification_sha256=sha((attempt/'commands.json').read_bytes()),
        selected_tests=report['collected_tests'],suites=len(report['targets']),new_cases=new,source_bindings=checked,
        preserved_evidence=evidence,documents=[bind(ROOT/name) for name in DOCUMENTS],
        read_guards=[bind(p) for p in (Path(__file__),WORKFLOW,CHECKER)],
        current_evidence=[bind(p) for p in sorted(PHASE.rglob('*')) if p.is_file()
            and p.name not in {'documentation-checks.json','documentation-workflow.log'}],qualification_attempts=len(list(PHASE.glob('qualification-[0-9][0-9]'))))


def main():
    receipt=PHASE/'documentation-checks.json';log=PHASE/'documentation-workflow.log'
    need(not receipt.exists() and not log.exists(),'documentation outputs already exist')
    need(sha(CHECKER.read_bytes())==CHECKER_SHA,'reviewed checker changed')
    before=guard();blocks=extract_workflow_blocks()
    report=dict(schema='native-source-worker-spawn-documentation@1',started_at_utc=utc(),before=before,
        exact_workflow_run_blocks=blocks,workflow_sha256=WORKFLOW_SHA,steps=[],
        additional_tests=False,models=False,providers=False,docker=False,hosted_ci_passed_here=False,status='running')
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
    spec=importlib.util.spec_from_file_location('nominated_allocation_documentation_checker',CHECKER)
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
    after=guard();need(before==after,'documentation read guards changed')
    report.update(after=after,status='pass',completed_at_utc=utc(),log_binding=bind(log))
    receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({key:report[key] for key in ('status','relative_links','anchors')}))
    return 0

if __name__=='__main__':raise SystemExit(main())
