import hashlib,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path

base=Path(__file__).resolve().parent
output=base/'public-component-evidence'
phases=['before','after','before-capture-release','final','diagnostic','diagnostic-wide','baseline-ownership','baseline-capture-ownership','final-corrected','final-fresh']
result=json.loads((base/'final-fresh-exit.json').read_text())
assert result['returncode']==0 and result['source_pins_unchanged']
assert not output.exists()
output.mkdir()
def copy(source,dest):
    source=base/source;target=output/dest
    assert source.is_file() and not source.is_symlink()
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
for phase in phases:
    for suffix in ('-command.json','-exit.json','-stdout.txt','-stderr.txt','.xml'):
        copy(Path(phase+suffix),Path('controls')/(phase+suffix))
for folder in ('before','minimal-generation','first-final-generation'):
    for source in sorted((base/folder).glob('*.py')):
        copy(source.relative_to(base),Path('sources')/folder/source.name)
for source in sorted((base/'final-fresh-sources').rglob('*.py')):
    copy(source.relative_to(base),Path('sources/final')/source.relative_to(base/'final-fresh-sources'))
for name in ('diagnostic-test.py','diagnostic-wide-test.py','baseline-original-plugin.py','before_bind_plugin.py','run_controls.py','build_public_evidence.py'):
    copy(Path(name),Path('diagnostics')/name)
copy(Path('before-source-pins.json'),Path('before-source-pins.json'))
all_sources={}
for path in output.rglob('*.py'):
    all_sources.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(),[]).append(path.relative_to(output).as_posix())
counts={};bindings={}
for phase in phases:
    command=json.loads((base/(phase+'-command.json')).read_text())
    bindings[phase]={}
    for source,digest in {**command['source_sha256'],**command.get('archived_bind_override',{})}.items():
        assert digest in all_sources,(phase,source,digest)
        bindings[phase][source]={'sha256':digest,'retained_sources':all_sources[digest]}
    cases=list(ET.parse(base/(phase+'.xml')).getroot().iter('testcase'))
    counts[phase]={'executions':len(cases),'failures':sum(c.find('failure') is not None for c in cases),'errors':sum(c.find('error') is not None for c in cases),'skips':sum(c.find('skipped') is not None for c in cases)}
final_cases=list(ET.parse(base/'final-fresh.xml').getroot().iter('testcase'))
assert counts['final-fresh']==dict(executions=188,failures=0,errors=0,skips=0)
assert len({(x.attrib.get('classname'),x.attrib['name']) for x in final_cases})==188
for phase in ('baseline-ownership','baseline-capture-ownership'):
    assert counts[phase]==dict(executions=2,failures=2,errors=0,skips=0)
(output/'source-bindings.json').write_text(json.dumps(bindings,indent=2,sort_keys=True)+'\n')
receipt={'schema':'admitted-context-lifetime-controls@1','selected_final_label':'final-fresh','selected_final_tests':188,'selected_final_unique_tests':188,'selected_final_skips':0,'prior_labels_do_not_add_qualification':True,'counts_by_label':counts,'final_exit':result,'checks':['same native world persistence and live-owner verification before gate','same post-capture fresh source/model gate and exact selection comparison','semantic payload/view and capture/task-list/record/contract collectibility at gate','fresh gate refusal prevents task-context result/bundle and Doctor publication'],'test_only_collection':True,'production_gc_added':False,'production_admission_thresholds_changed':False,'rss_reduction_measured':False,'docker_requalification_included':False,'credentials_exported':False,'hidden_verifier_bodies_exported':False}
(output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
(output/'README.txt').write_text("""Admitted-context construction ownership controls, 2026-10-04

The final production change drops completed semantic payload/view/reader, native
world capture, task-list and contract aliases before the existing fresh context
gate. Six scalar identities remain for the unchanged result. World persistence
and live-owner verification still precede the gate. The checked receipt,
descriptor and summaries must still equal the nominated selection before task
context publication. No admission limit, deadline, solver or currentness gate
was relaxed. No production collection call was added.

Selected qualification: final-fresh, 188 unique host tests, zero failures,
errors or skips. This is the previous 15-file A consumer control scope plus
indexed-preparation controls, with two new admitted construction cases.
All 17 selected owner/test source hashes were captured before that run and
checked unchanged afterwards. Exact argv, cwd, environment overrides, exit,
XML and source copies are retained. Seal/key store paths are in commands only;
the private stores and their contents/hashes are excluded from this export.

The new test uses authored local task declarations with real native scanner,
index, intent admission, persistence and world validation. Only the explicit
fresh-gate refusal is authored. It verifies that refusal leaves no published
task-context result/bundle, context result or Doctor eligibility artifact. A
private pre-gate world capture can exist, as before. It has no completion
or execution authority. No provider or hidden benchmark verifier was used.

The initial witness incorrectly required immediate destruction. Read-only GC
referrer diagnostics found a returned test view() frame retaining payload/view
objects in collectable cycles. The final witness calls gc.collect() in the
test before checking weak references. Thus it demonstrates removal of caller
ownership/collectibility, not immediate reclamation, lower RSS, or recovery of
the native run's 9 MiB admission deficit. Native view owns only a root and
Path-backed block reader; it does not retain the complete semantic block graph.

Both final witness cases fail against the archived original bind function after
test-only collection. They also fail on capture aliases against the intermediate
semantic-only release. These baselines load only the archived function AST into
the native test module; they do not mutate checkout source files. Their plugin
bytes and selected source hashes are retained. Original weak-reference failures,
the intermediate 35-pass/1-failure run, the initial 187-pass/1-failure broad run
and referrer diagnostics are prior controls, not extra passing qualification.
The subsequent final-corrected run reused isolated seal stores and therefore
reported 12 passes plus 176 'pytest AST seal matches' skips. It is not qualifying.
Final-fresh uses newly allocated per-label seal, key and scheduler paths;
no old state was deleted. All 188 tests executed there.
The per-label counts and source-bindings file distinguish all generations.

This package contains no new Docker qualification or full benchmark outcome.
The earlier native failed trial predates this owner change and is retained by
the parent separately. A fresh unchanged-limits Docker run is still required
before claiming this change helps its admission boundary. No token-efficiency,
whole-program proof, learned correctness or benchmark-success claim follows.
""")
secret=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b')]
rows=[]
for path in sorted(output.rglob('*')):
    if not path.is_file():continue
    raw=path.read_bytes()
    assert not any(pattern.search(raw) for pattern in secret),path.relative_to(output)
    rows.append({'path':path.relative_to(output).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
manifest={'schema':'public-evidence-manifest@1','files':rows}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(output),'members':len(rows),'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),'receipt_sha256':hashlib.sha256((output/'receipt.json').read_bytes()).hexdigest(),'selected_counts':counts['final-fresh']}))
