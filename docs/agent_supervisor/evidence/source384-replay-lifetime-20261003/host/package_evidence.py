"""Close completed source-free lifetime evidence; never include model reports."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
W=ROOT.parents[1]
A=W/'.worktrees/ir-release-accelerate-20261002'
TARGET=A/'docs/agent_supervisor/evidence/source384-replay-lifetime-20261003'
D_RECEIPTS=Path(sys.argv[1])
D_LABEL=sys.argv[2]
D_RETRY=sys.argv[3]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def counts(path):
    result={'passed':0,'failed':0,'skipped':0}
    for case in ET.parse(path).getroot().iter('testcase'):
        status=('failed' if case.find('failure') is not None or case.find('error') is not None
                else 'skipped' if case.find('skipped') is not None else 'passed')
        result[status]+=1
    return result

assert not TARGET.exists(),'closed evidence destination must be fresh'
assert counts(ROOT/'actual-a-lifetime.xml')==dict(passed=8,failed=0,skipped=0)
assert counts(ROOT/'actual-a-context.xml')==dict(passed=13,failed=0,skipped=0)
d_xml=D_RECEIPTS/(D_LABEL+'.xml')
d_counts=counts(d_xml)
d_retry_xml=D_RECEIPTS/(D_RETRY+'.xml')
d_retry_counts=counts(d_retry_xml)
d_rows=[case for case in ET.parse(d_xml).getroot().iter('testcase')
        if case.attrib.get('classname','').endswith('test_codebase_source_units_lifetime')]
assert len(d_rows)==6 and all(case.find('failure') is None and case.find('error') is None and case.find('skipped') is None for case in d_rows)
assert d_counts==dict(passed=266,failed=7,skipped=0)
assert d_retry_counts==dict(passed=7,failed=0,skipped=0)
distinct={}
for path in (d_xml,d_retry_xml):
    for case in ET.parse(path).getroot().iter('testcase'):
        if case.find('failure') is None and case.find('error') is None and case.find('skipped') is None:
            distinct[(case.attrib.get('classname','').split('.')[-1],case.attrib['name'])]=True
assert len(distinct)==273
for label in ('actual-a-lifetime','actual-a-context'):
    receipt=json.loads((ROOT/(label+'-exit.json')).read_text())
    assert receipt['returncode']==0 and receipt['sources_unchanged']
selected=[]

def copy(src,dest):
    source=Path(src);target=TARGET/dest
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    selected.append({'source':str(source),'path':dest,'sha256':sha(source)})

for name in ['A-candidate.patch','D-candidate.patch','candidate.patch','source-pins.json',
             'qualification.json','independent-preparation-review.json','run_controls.py',
             'run_actual_a.py','bootstrap/sitecustomize.py','package_evidence.py',
             'run_configured_a.py','configured_client_plugin.py','configured-client-provenance.json',
             'actual-a-context-configured-client-before.json','actual-a-context-configured-client-after.json',
             'actual-a-context-launch.json','actual-a-context-controller-exit.json']:
    copy(ROOT/name,'host/'+name)
for label in ['controls-before','controls-proposed','actual-a-lifetime','actual-a-context']:
    for suffix in ['.log','.xml','-command.json','-exit.json']:
        copy(ROOT/(label+suffix),'host/'+label+suffix)
for label in ['before','proposed']:
    for source in sorted((ROOT/label).rglob('*.py')):
        copy(source,'sources/'+str(source.relative_to(ROOT)))
for source in sorted((ROOT/'tests').glob('*.py')):
    copy(source,'tests/'+source.name)
for label in (D_LABEL,D_RETRY):
    for suffix in ['.log','.xml','-command.json','-exit.json']:
        copy(D_RECEIPTS/(label+suffix),'datasets/'+label+suffix)
for name in ['configured_client_plugin.py','run_source_controls_03.py',
             'controls-02-scheduler-diagnosis.json','controls-03-configured-client-before.json',
             'controls-03-configured-client-after.json']:
    copy(D_RECEIPTS/name,'datasets/'+name)

qualification={'schema':'source384-replay-lifetime-final@1',
    'status':'actual_lifetime_and_configured_client_native_controls_passed',
    'changes':['Release consumer raw inference bytes after parsing; later native replay and fresh artifact digest checks stay unchanged.',
               'Release duplicate native replay graph after exact bytes and execution flags validation, before second fresh source observation.'],
    'suites':{'off_tree_before':counts(ROOT/'controls-before.xml'),
              'off_tree_candidate':counts(ROOT/'controls-proposed.xml'),
              'actual_accelerate_lifetime':counts(ROOT/'actual-a-lifetime.xml'),
              'actual_accelerate_context':counts(ROOT/'actual-a-context.xml'),
              'actual_datasets_host_default_attempt':d_counts,
              'actual_datasets_configured_client_retry':d_retry_counts,
              'actual_datasets_distinct_passing_union':len(distinct),
              'actual_datasets_lifetime_subset':6},
    'distinct_lifetime_and_consumer_cases':27,
    'counting':'14 authored lifetime cases (8 A +6 D) and13 existing A integration cases. D six are a subset of the separately counted datasets suite, not added again; off-tree reruns are not additional distinct tests.',
    'native_scope':'Existing A context tests exercise actual pinned checkpoint/GTE inference, nomination into planning/dispatch and replay without a second model call. Provider response is authored; no live provider, benchmark reward or token score.',
    'scheduler_scope':'A13 and D7 are explicit configured clients of the existing native4CPU/9830MiB ledger. A before/after has zero leases/waiters and unchanged config. Sampler, headroom, thresholds, source deadlines and ownership checks remain native. No host-default startup qualification: D seven original fixtures refused conflicting active canonical configuration; final retry selected that same configuration through the supported native API.',
    'limits':['Logical object release verified on this CPython runtime.',
              'No RSS, resource-admission or throughput improvement established.',
              'No complete Docker/full-supervisor qualification in this package.'],
    'source_pins':json.loads((ROOT/'source-pins.json').read_text()),'selected_files':selected}
(TARGET/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
(TARGET/'README.md').write_text('''# Source384 replay object lifetime

The consumer releases the already-parsed inference byte string before native replay. The datasets validator releases its duplicate parsed replay report after exact byte and execution-flag comparison, before its second fresh source observation. Both changes are one-line local-reference deletions. Source/model/producer checks, cancellation/deadlines, final fresh digest checks and detached returned reports remain intact.

Fourteen authored focused controls establish logical lifetime and refusal behavior. With original owners, the two next-boundary lifetime assertions failed while twelve refusal/result controls passed. Both candidate owners passed all fourteen controls. Whole canonical modules were selected off-tree; these were controlled loader/admission seams, not native inference qualification.

After applying the exact patches, actual on-disk A lifetime tests passed8 and the existing A Source384 context suite passed13. The latter uses the real pinned checkpoint and cached GTE, an authored provider response, signed input scope and planning/dispatch replay checks. The datasets suite retained here includes six authored lifetime controls and existing native Source384 replay checks. Its initial actual run passed266 and refused seven fixtures because another daemon owned an incompatible canonical scheduler configuration. Those seven subsequently passed using the supported configured-client selection, yielding273 distinct passing cases across the two retained runs. The six datasets lifetime cases are counted once, within that separate suite. An earlier unrelated collection failure from bare sibling fixture imports is retained in the datasets actual-run artifact directory; only those test imports were corrected before the runs included here.

A13 and the D7 retry are explicitly configured-client qualification, not host-default startup qualification. Their external test plugin selects the existing4CPU/9830MiB scheduler through its native API and same ledger. It changes no sampler, capacity, threshold, deadline, state file or daemon, and restores the local dependency afterward. The foreign lease had naturally ended before A13's fixture; its before/after receipts have zero leases/waiters with identical configuration. Parent and isolated worker use coherent interpreter site roots, with no unrelated Harbor dependency path.

These results establish earlier logical object release and preserved contracts. They do not establish lower resident memory, improved resource admission, a full Docker/supervisor benchmark result or token savings. Native inference reports, benchmark input bodies, weights, vectors, databases and credentials are excluded. The manifest is a closed inventory of this package.
''')
files=[]
for file in sorted(TARGET.rglob('*')):
    if file.is_file():
        files.append({'path':str(file.relative_to(TARGET)),'bytes':file.stat().st_size,'sha256':sha(file)})
(TARGET/'manifest.json').write_text(json.dumps({'schema':'closed-source384-replay-lifetime-evidence@1','files':files},indent=2)+'\n')
ready={'package':str(TARGET),'manifest_sha256':sha(TARGET/'manifest.json'),'members':len(files),
       'bytes':sum(f['bytes'] for f in files),
       'scoped_added_files':[str((TARGET/f['path']).relative_to(A)) for f in files]
                           +[str((TARGET/'manifest.json').relative_to(A))]}
(ROOT/'package-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps({k:v for k,v in ready.items() if k!='scoped_added_files'}))
