"""Seal compact context-snapshot fix evidence without duplicate daemon bodies."""
from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
B=Path(__file__).resolve().parent
W=B.parents[1];A=W/'.worktrees/ir-release-accelerate-20261002'
OUT=A/'docs/agent_supervisor/evidence/worker-context-bundle-snapshot-20261003'
if OUT.exists():raise ValueError('closed destination must be fresh')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins=json.loads((B/'candidate-pins.json').read_text())
owner=A/'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py'
test=A/'test/api/semantic_state/test_worker_context_snapshot.py'
assert sha(owner)==pins['candidate_sha256']=='a576ea51549b4a83ec9f5a6cae132c3a1b2badb826c437295cd2f68cd7b0faf0'
assert sha(test)==sha(B/'test_worker_context_snapshot.py')=='cdd6b1962460af8366ab94c163218a44705b01abd1f1dd8e938f79a21222cf1b'
xml=ET.parse(B/'actual-source-01.xml').getroot();cases=list(xml.iter('testcase'))
assert len(cases)==36 and not any(list(c) for c in cases)
# Verify expected empty testcase children: no failure, skip or captured payload.
exit_receipt=json.loads((B/'actual-source-01-exit.json').read_text());assert exit_receipt['returncode']==0
OUT.mkdir(parents=True)
files=[]
def copy(name):
 p=B/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size>200000:raise ValueError(name)
 q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 files.append({'artifact':str(p),'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
for n in ['candidate.patch','candidate-pins.json','test_worker_context_snapshot.py',
 'actual-source-01-command.json','actual-source-01-exit.json','actual-source-01.log','actual-source-01.xml',
 'independent-preparation-review.json','baseline-collection-import.json','baseline-collection-import.log',
 'native-class-01.log','native-class-01.xml','native-class-01-command.json',
 'native-class-02.log','native-class-02.xml','native-class-02-command.json','test_worker_context_snapshot-02.py',
 'native-class-03.log','native-class-03.xml','native-class-03-command.json','controls-summary.json']:
 copy(n)
write('qualification.json',{'schema':'worker-context-single-snapshot-qualification@1','production_change_applied':True,
 'owner':str(owner.relative_to(A)),'base_owner_sha256':pins['base_sha256'],'final_owner_sha256':sha(owner),
 'patch':'candidate.patch','patch_sha256':sha(B/'candidate.patch'),'focused_test':str(test.relative_to(A)),
 'focused_test_sha256':sha(test),'actual_source_suite':{'passed':36,'failed':0,'skipped':0,'pytest_seconds':71.91,
 'process_seconds':exit_receipt['seconds'],'new_cases':8,'neighbor_cases':28,'xml':'actual-source-01.xml',
 'command':'actual-source-01-command.json','method_injection':False,'ast_slicing':False},
 'behavior':'One fresh normalized immutable metadata snapshot per initial or retry context construction; exact live bundle validation once, same artifact-specific checks and original ordering.',
 'prior_nomination_loads_when_all_three_artifacts_present':{'initial':8,'retry':11},
 'current_nomination_loads_per_construction_when_bundle_selected':1,
 'cross_construction_cache':False,'independent_start_dispatch_publication_fences_changed':False,
 'native_source384_checkpoint_replay_timing_measured':False,'aggregate_speedup_measured':False,
 'source384_validator_in_focused_new_cases':'Controlled seam records or rejects; real bundle parser and imported daemon/context compiler execute.',
 'retry_case_scope':'Actual retry method reaches diagnostic boundary after evidence construction; test stops there before unrelated compiler/diagnostic work.',
 'legacy_suite_blocker':{'artifact':'baseline-collection-import.json','candidate_independent':True,
 'description':'Legacy todo-daemon-port test module imports absent private Copilot timeout constant. Excluded without stubs; not repaired here.'},
 'earlier_off_tree_runs':{'collection_failure':'native-class-01','test_assertion_failure':'native-class-02',
 'reason':'Initial new test assumed compiler retained insertion order; compiler canonicalizes evidence. Final test separately verifies original read order and final membership.',
 'corrected_focused_run':'native-class-03'},
 'benchmark_result':False,'provider_calls':0,'docker_qualification_run':False,
 'large_daemon_bodies_included':False,'authorization_or_semantic_authority_changed':False})
write('omitted-source-descriptors.json',{'schema':'large-owner-body-pins@1','files':[
 {'path':'before.py','bytes':(B/'before.py').stat().st_size,'sha256':sha(B/'before.py')},
 {'path':'candidate.py','bytes':(B/'candidate.py').stat().st_size,'sha256':sha(B/'candidate.py')}],
 'reason':'Full daemon bodies are retained locally; compact exact diff plus before/after hashes avoid duplicate multi-megabyte snapshots.'})
(OUT/'README.md').write_text('''# One live task-context nomination per construction

The worker previously loaded the same selected bundle for every semantic, retrieval and world metadata field: eight live Source384 validations in a complete initial context construction, and eleven in the retry construction including presence checks. Each load traversed the same canonical bundle validation. These are code-path counts, not measured aggregate costs.

The fix copies normalized task metadata, validates the exact task/bundle nomination once, rejects conflicting metadata, and exposes a read-only local snapshot to the existing artifact consumers. Initial read/evidence order remains semantic→retrieval→world; retry order remains semantic→world→retrieval. A new construction and a directly invoked helper validate afresh. No instance/global cache is introduced. The optional snapshot argument is internal plumbing, not an external authority interface. Independent semantic/source, world coherence and retrieval checks are unchanged, as are startup, dispatch and post-publication staleness fences.

The actual on-disk daemon and final focused test passed36 cases with zero failures/skips in71.91 seconds: eight new real-class controls and28 existing bundle, semantic/world, and deterministic-context cases. This final run performs no AST slicing or method injection. New controls use the real bundle parser and context compiler, with native Source384 validation and artifact reads at controlled seams. The retry case executes the actual retry method through evidence construction and then intentionally stops at the diagnostic boundary. No model checkpoint throughput, full daemon Docker execution or benchmark advantage is established here.

Earlier off-tree qualification is retained separately. The broad legacy todo-daemon-port module could not collect because it imports an absent private Copilot timeout constant; the same import failed independently against unchanged runtime bytes, without candidate injection. No stub or unrelated runtime repair was added. In the following off-tree run35 tests passed and one new assertion failed because ContextCompiler canonicalizes final evidence order. The corrected test checks evidence membership and original artifact-read order separately; its final eight cases passed before the36-case actual-source run. The final test also explicitly isolates its orchestration state in the temporary directory.

The compact patch, exact test, source hashes, raw test logs/XML, independent read-only review and baseline import refusal are included. Full multi-megabyte daemon bodies remain local and are represented by hashes; no benchmark input, model weights, embedding arrays, authentication, database or hidden verifier payload is included. Paths in command receipts retain their original local execution layout. The independent review examined the final source patch and did not rerun tests.

Timing is ordinary test duration, with other light development work possible on the host; it is not a performance comparison. The published result establishes the per-construction boundary behavior and preservation of neighboring contracts. The separate setup-cache diagnostics and their closed packages are unchanged.
''')
copy('package_evidence.py')
write('provenance.json',{'schema':'explicit-whitelist-provenance@1','copies':files})
rows=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
write('manifest.json',{'schema':'closed-worker-context-single-snapshot-evidence@1','files':rows})
r={'package':str(OUT),'manifest_sha256':sha(OUT/'manifest.json'),'members':len(rows),'member_bytes':sum(x['bytes'] for x in rows),
 'files':[str((OUT/x['path']).relative_to(A)) for x in rows]+[str((OUT/'manifest.json').relative_to(A))]}
(B/'scoped-added-files.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='files'}))
