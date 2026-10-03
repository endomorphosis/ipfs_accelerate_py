from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
B=Path(__file__).resolve().parent;A=B.parents[1]/'.worktrees/ir-release-accelerate-20261002'
E=A/'docs/agent_supervisor/evidence/source384-failure-scheduler-diagnostics-20261003'
assert not E.exists();E.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(source,relative):
 target=E/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
pins=json.loads((B/'final-pins.json').read_text())
pins['tests']={'passed':39,'skipped':0,'deselected':1,'pytest_seconds':.84,'scope':'actual on-disk A; controlled snapshots/resources; no host sampling or native inference'}
(B/'final-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
for row in pins['files']:
 source=A/row['path'];assert sha(source)==row['after_sha256'];copy(source,'final-sources/'+row['path'])
for source in (B/'before').rglob('*.py'):copy(source,'before/'+str(source.relative_to(B/'before')))
copy(B/'public-evidence/proposed/benchmarks/agent_supervisor/container_coding/terminal_resource_diagnostics.py','previous-helper/terminal_resource_diagnostics.py')
for prefix in ('controls-01','actual-02','actual-03'):
 for suffix in ('.xml','.log','-command.json','-exit.json'):copy(B/(prefix+suffix),'controls/'+prefix+suffix)
for name in ('run_controls.py','run_actual_controls-02.py','run_actual_controls.py','native-shape-review.json','final-pins.json','production.patch','build_actual_evidence.py'):
 copy(B/name,name)
xml=ET.parse(B/'actual-03.xml');assert len(xml.findall('.//testcase'))==39 and not any(xml.findall('.//'+k) for k in ('failure','error','skipped'))
q={'schema':'terminal-failure-scheduler-actual-qualification@1','status':'applied_and_tested_not_committed_by_agent',
'final_source_pins':pins['files'],'actual03':{'passed':39,'skipped':0,'failed':0,'errors':0,'deselected':1,'pytest_seconds':.84,'process_seconds':1.0644948129920522,'import_override':False,'fresh_supported_ast_seal_database':True},
'prior_actual02':{'passed':24,'skipped':15,'skip_reason':'pytest AST seal matches','deselected':1,'retained_without_promoting_cached_skips_to_execution':True},
'prior_off_tree01':{'passed':39,'pytest_seconds':.48,'helper_docstring_before_direct_ledger_clarification':True},
'distinct_control_count':39,'scope':{'controlled_scheduler':True,'controlled_resource_samples':True,'actual_native_sampler_case_explicitly_deselected':True,'native_model_inference':False,'docker':False,'source_payload_exported':False,'exact_admission_decision':False,'causal_proof':False,'resource_budgets_or_policy_changed':False},
'compatibility_review':'native-shape-review.json','previous_off_tree_package_manifest_sha256':'c15e7882e74a33d4b90bf480507b6a6e9a8be44909aec45cc5b34ac073e7dc4a'}
(E/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
(E/'README.md').write_text('''# Actual failure-only scheduler diagnostics

The applied A helper, context probe and full driver passed **39 controlled tests
in 0.84 seconds**, with zero skips and one deliberate deselection of the existing
actual-host-resource-sampler case. The modules loaded from the real checkout,
without bootstrap or import overrides. No native model or Docker run was needed.

The first actual run executed 24 tests and reused 15 existing tests through the
mandatory pytest AST seal. Its raw result is retained. The final run used the
supported fresh per-run seal database and executed all 39 selected tests. The
preceding off-tree 39-pass generation is also retained; these are the same 39
distinct controls, not additive tests. The only helper change after the off-tree
run clarifies the docstring to say no *direct* ledger reads; previous helper bytes
are included for exact generation correspondence.

Failure paths now append a bounded source-free scheduler observation alongside
the existing post-unwind resource sample. Only fixed finite capacity/allocation,
lease/waiter counts, proof backoff and proof recovery fields are exported. Unknown
reason/phase strings map to a fixed enum value. Paths, arbitrary labels, source,
lease capabilities and diagnostic exception messages are excluded. Collection
failures preserve the original task error, failure phase, result and cleanup.

The helper selects exactly one already-imported native facade through the existing
private registry and nonblocking process lock. It never creates or configures an
owner. Missing, busy, ambiguous or changed layouts yield unavailable diagnostics.
It then uses the supported snapshot API, which may recover stale leases normally.
The current D source shape is independently inspected in `native-shape-review.json`;
that is source review, not a live host-pressure test. The helper bounds output to
4096 bytes and retains the native snapshot's existing ledger-locking semantics.

The snapshot is labeled `after_unwind`: the failing call has unwound, but full task
shutdown may still be pending. Its backoff can outlive a request. It is neither
the failed request's exact admission sample nor causal proof of why that request
failed. Resource limits, deadlines, admission thresholds and configuration remain
unchanged. No benchmark score, inference completion or proof authority is claimed.

Command and source paths in the receipts describe the original development
machine. `final-sources/` contains exact current bytes; `before/` and
`previous-helper/` retain preceding producer generations. No state database,
benchmark input, model weights, authentication data or runtime archive is included.
''')
manifest={'schema':'terminal-failure-scheduler-actual-evidence@1','files':[{'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(E.rglob('*')) if p.is_file()]}
(E/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
scope=sorted([row['path'] for row in pins['files']]+[str(p.relative_to(A)) for p in E.rglob('*') if p.is_file()])
(B/'actual-commit-scope.json').write_text(json.dumps({'repository':str(A),'files':[{'path':r,'sha256':sha(A/r),'bytes':(A/r).stat().st_size} for r in scope]},indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(E/'manifest.json'),'members':len(manifest['files']),'member_bytes':sum(r['bytes'] for r in manifest['files']),'scope_files':len(scope)},indent=2))
