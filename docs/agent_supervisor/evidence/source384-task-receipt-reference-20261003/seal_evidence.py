from pathlib import Path
import hashlib,json,subprocess,shutil,xml.etree.ElementTree as E
B=Path(__file__).resolve().parent; W=B.parents[1]; A=W/'.worktrees/ir-release-accelerate-20261002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py','test/api/semantic_state/test_source384_task_context_bundle.py','test/api/semantic_state/test_source384_receipt_reference.py','test/api/semantic_state/test_worker_context_snapshot.py','test/integration/test_admitted_context_refresh.py','benchmarks/agent_supervisor/container_coding/test_terminal_context_rebind.py','benchmarks/agent_supervisor/container_coding/test_terminal_source384_context.py']
rows=[dict(path=f,sha256=sha(A/f),bytes=(A/f).stat().st_size) for f in files]
(B/'frozen-sources.json').write_text(json.dumps(dict(schema='source384-receipt-reference-freeze@1',files=rows),indent=2)+'\n')
test_ids=set();runs=[]
for name in ('focused-01','final-02','neighbors-03'):
 cases=E.parse(B/(name+'.xml')).getroot().findall('.//testcase')
 ids={(x.get('classname').removeprefix('test.'),x.get('name')) for x in cases};test_ids.update(ids)
 assert not [x for x in cases if x.find('error') is not None or x.find('failure') is not None or x.find('skipped') is not None]
 runs.append(dict(prefix=name,tests=len(cases),all_passed=True))
assert len(test_ids)==56
result=dict(schema='source384-receipt-reference-qualification@1',status='component_qualified',runs=runs,distinct_passing_controls=56,
 test_identity_normalization='remove optional pytest-root prefix test. from classname; names retained exactly',
 runtime_changes=[files[0]],actual_inference_test_assertion_updated_not_rerun=files[-1],
 model_calls=0,provider_calls=0,docker_runs=0,retained_transport='retained-native-transport.json',
 legacy_read_schemas=['supervisor-task-context-nominations@1','supervisor-task-context-nominations@2'],new_write_schema='supervisor-task-context-nominations@3',
 bundle_max_bytes=131072,full_receipt_max_bytes=131072,reference_max_bytes=4096,live_validation_unchanged=True,historical_read_grants_currentness=False,benchmark_reward_claim=False,
 limits='Transport/publication/worker controls only; not remaining full-trial deadline, successor preparation, neural inference or task correctness.',source_pins_sha256=sha(B/'frozen-sources.json'))
(B/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
(B/'candidate.patch').write_bytes(subprocess.check_output(['git','-C',str(A),'diff','--',*[f for f in files if 'test_source384_receipt_reference.py' not in f]]))
(B/'scoped-files.json').write_text(json.dumps(files,indent=2)+'\n')
P=B/'public-evidence';P.mkdir()
for f in files:
 target=P/'sources'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(A/f,target)
small=['frozen-sources.json','qualification.json','candidate.patch','scoped-files.json','replay_retained_receipt.py','retained-native-transport.json','retained-native-transport.log','run_controls.py','independent-review.json']
for n in ('focused-01','final-02','neighbors-03'):
 small += [n+s for s in ('-command.json','-exit.json','.log','.xml')]
for f in small:shutil.copyfile(B/f,P/f)
readme='''# Source384 task receipt references

The full supervisor reached indexed context and planning, then failed because its 78,563-byte Source384 receipt exceeded the task bundle's 32,768-byte inline limit. The canonical receipt producer permits 131,072 bytes. Increasing the inline limit would still duplicate the full receipt for each of up to sixteen tasks.

The writer now emits nominations@3: a closed, at-most-4,096-byte reference to the existing external receipt.json, binding its full SHA-256 and byte count. The bundle and complete receipt limits remain 131,072 bytes. The selected reference is resolved with the existing canonical, regular-file, no-follow, singleton, stable-byte reader. The complete receipt then reaches the unchanged current-source/model validator. Historical observation resolves exact bytes without granting freshness or successor availability. Legacy @1 and @2 readers remain supported.

The final runtime owner passed 56 distinct controls: 44 transport/legacy controls, one actual native owner/publication/STOP/historical-consumer lifecycle, and eleven worker-snapshot/warm-rebind controls. Runs contain 44, 45 and 55 overlapping cases, not 144 distinct cases. Native numerical inference is not executed. The real-checkpoint test has an assertion-only update for the new schema and full historical selection; that test was not rerun in this slice.

The retained native receipt transport replay changes only repository/output for an authored temporary filesystem. Its original 78,563 bytes become 78,734 bytes; every other field, including all 220 source hashes and inventory rows, is unchanged. Sixteen task references occupy 5,925 bytes. A recording validator isolates transport from model replay. This neither renews the old producer pins nor qualifies the relocated receipt as current model/source evidence. No task/model/CAS bodies or auth are included.

This fixes a packaging contract. It does not qualify the remaining full-supervisor deadline, repair correctness, benchmark reward, or automatic successor preparation. The failed full run is retained separately. No deadlines, scheduling policy, model architecture or weights changed.
'''
(P/'README.md').write_text(readme)
shutil.copyfile(__file__,P/'seal_evidence.py')
manifest=dict(schema='source384-task-receipt-reference-evidence@1',files=[dict(path=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(P.rglob('*')) if p.is_file()])
(P/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(manifest_sha256=sha(P/'manifest.json'),members=len(manifest['files']),bytes=sum(f['bytes'] for f in manifest['files']),qualification=result)))
