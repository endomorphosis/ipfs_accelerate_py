"""Curate explicit offline evidence; never copy private live/worker payloads."""
from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
A=Path(__file__).parent
Q=Path('/home/barberb/lift_coding/.worktrees/supervisor-token-growth-20261013')
E=Q/'docs/agent_supervisor/evidence/repeated-compact-edits-20261013'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def load(p):return json.loads(p.read_bytes())
assert not E.exists(), 'Use a fresh evidence directory; never overwrite retained publication.'
E.mkdir(parents=True)
copies=[]
def copy(src,dst):
 source=A/src;target=E/dst
 assert source.is_file() and not source.is_symlink()
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 copies.append(dict(source=src,destination=dst,bytes=target.stat().st_size,sha256=sha(target)))
runs={'root':('root-tests-02',349,'exit.json'),'outer':('outer-tests-03',149,'result.json'),'client':('client-tests-01',116,'result.json'),'core':('service-final-tests-03',100,'exit.json')}
ledger={};allnodes=set();pins={}
for label,(name,expected,result_name) in runs.items():
 r=A/name;x=ET.parse(r/'junit.xml');cases=x.findall('.//testcase');nodes=sorted(n.attrib['classname']+'::'+n.attrib['name'] for n in cases)
 assert len(nodes)==len(set(nodes))==expected
 assert not any(n.find(k) is not None for n in cases for k in ('failure','error','skipped'))
 assert not any((n.text or '').strip() for n in x.iter())
 assert not allnodes.intersection(nodes);allnodes.update(nodes)
 result=load(r/result_name);assert result.get('exit_code',result.get('returncode'))==0 and result['source_unchanged'] is True
 command=load(r/'command.json')
 before=load(r/'source-before.json') if (r/'source-before.json').exists() else result.get('source_before') or command['source_before']
 after=load(r/'source-after.json') if (r/'source-after.json').exists() else result['source_after']
 assert before==after
 for path,h in after.items():
  if path.endswith('.py'):
   assert sha(Q/path)==h, path
   assert path not in pins or pins[path]==h,path
   pins[path]=h
 for f in ('command.json',result_name,'junit.xml'):
  copy(name+'/'+f,'tests/'+label+'/'+f)
 for f in ('source-before.json','source-after.json'):
  if (r/f).exists():copy(name+'/'+f,'tests/'+label+'/'+f)
 ledger[label]=dict(run=name,passed=expected,failed=0,errors=0,skipped=0,source_unchanged=True,nodes=nodes,junit_sha256=sha(r/'junit.xml'))
assert len(allnodes)==714
write(E/'test-ledger.json',dict(schema='repeated-compact-edit-test-ledger@1',distinct_passed=714,executed=714,failed=0,errors=0,skipped=0,overlap=0,runs=ledger,legacy_byte_comparison_cases_separate=32))
write(E/'source-bindings.json',dict(schema='repeated-compact-edit-qualified-source-bindings@1',all_current_hashes_verified=True,files=pins))
for src,dst in [
 ('client-qualification.json','qualifications/client.json'),('outer-qualification-02.json','qualifications/outer.json'),
 ('client-legacy-guidance-baseline.json','historical/client-legacy-guidance-baseline.json'),
 ('legacy-outer-wire-replay.json','historical/legacy-outer-wire-replay.json'),
 ('check_legacy_outer_edit.py','historical/check_legacy_outer_edit.py'),
 ('integration-review-01.json','reviews/history/integration-review-01.json'),
 ('core-independent-review.json','reviews/history/core-independent-review.json'),
 ('delivery-fence-independent-review.json','reviews/delivery-fence-independent-review.json'),
 ('docker-byte-accounting-review.json','reviews/docker-byte-accounting-review.json'),
 ('review_docker_byte_accounting.py','reviews/review_docker_byte_accounting.py'),
 ('docker/release-02.json','docker/release.json'),
 ('docker/run_docker_qualification.py','docker/run_docker_qualification.py'),
 ('docker/owner_probe.py','docker/owner_probe.py'),('docker/worker_probe.py','docker/worker_probe.py'),
 ('docker/run-01/qualification.json','docker/qualification.json'),
 ('docker/run-01/results/probe-qualification.json','docker/probe-qualification.json'),
 ('docker/run-01/results/session-observation.json','docker/session-observation.json'),
 ('docker/run-01/container-controls.json','docker/container-controls.json'),
 ('docker/run-01/command.json','docker/command.json'),
 ('environment.json','tests/environment.json'),('qualify.py','tests/qualify.py')]:copy(src,dst)
write(E/'qualifications/core.json',dict(schema='repeated-source-edit-final-core-qualification@1',complete=True,tests=100,failed=0,errors=0,skipped=0,source_unchanged=True,run='service-final-tests-03',source_sha256={p:pins[p] for p in ('ipfs_accelerate_py/agent_supervisor/runtime/worker_source_edit.py','test/api/test_worker_source_edit.py')},junit_sha256=sha(A/'service-final-tests-03/junit.xml'),scope='Complete final current-file core suite; supersedes staged96+4 for final-file coverage, preserves historical runs.'))
history=[]
for name in ('root-tests-01','outer-tests-01','outer-tests-02','service-tests-01','service-matcher-tests-02'):
 r=A/name;x=ET.parse(r/'junit.xml');nodes=x.findall('.//testcase');failed=[]
 for n in nodes:
  if any(n.find(k) is not None for k in ('failure','error')):failed.append(dict(node=n.attrib['classname']+'::'+n.attrib['name'],status='error' if n.find('error') is not None else 'failure'))
 resultfile='exit.json' if (r/'exit.json').exists() else 'result.json'
 history.append(dict(run=name,testcases=len(nodes),failed_or_error_nodes=failed,result=load(r/resultfile),junit_sha256=sha(r/'junit.xml'),original_raw_failure_text_published=False))
write(E/'retained-failures.json',dict(schema='repeated-compact-edit-retained-history@1',runs=history,diagnoses=[
 dict(run='root-tests-01',cases=2,cause='Authored extra-program fixture supplied unsorted program_paths; sorted canonical subset and matching scope corrected.',production_failure=False),
 dict(run='root-tests-01',cases=2,cause='Common-bindings cross-mode tamper correctly raised ValueError; test expected returned checks instead. Expectation corrected for metadata projection; legacy check-result retained.',production_failure=False),
 dict(run='outer-tests-01',cases=2,cause='Cross-mode refusal fixture omitted its required authored runtime archive manifest. Fixture corrected and file rerun.',production_failure=False),
 dict(run='independent source review',cause='Surrounding broker checks or failed send could follow core confirmation clearing. Production owner delivery fence and generic response callback were added, independently reviewed, and qualified in final root349 suite.',production_change=True,not_exposed_by_initial_test_failures=True)],notes=['Initial service96 and additive matcher4 each passed with owned files unchanged; aggregate harness noted unrelated concurrent edits. Final100-file suite now passed with all recorded source hashes unchanged.','Historical runs are not added to the714 distinct final current-file cases.']))
review_files=['ipfs_accelerate_py/agent_supervisor/runtime/'+n for n in ('worker_source_broker.py','worker_source_client.py','worker_source_edit_client.py','numerical_obligation_broker.py','router_implementation_runner.py')]+['benchmarks/agent_supervisor/container_coding/'+n for n in ('terminal_source_edit_policy.py','terminal_source_view_observation.py','terminal_context_audit.py','container_worker_deployment.py')]
review=dict(schema='repeated-source-edit-final-integration-review@1',status='clear',blocking_findings=[],source_sha256={p:sha(Q/p) for p in review_files},resolved_finding='Authenticated active matching repeated confirmation blocks before context checks; only exact confirmed page after owner-observed successful send clears. Refused/missing/failed send remains blocked. Old/foreign ticket cannot consume or clear newer active proposal.',checks=['Generic response callback defaults to no-op, resets per-request locals and stops session on callback exception.','Selected repeated mode is joined across descriptor, runtime selection, guidance, retrospective prompt, configuration and accounting.','Successful confirmation clears advisory candidate context; next reads/formal capsules are separately generation-bound and not upgraded to proof authority.','Read-only working inspection remains possible after blocked edit outcome.','Exact retrospective prompt tests cover legacy and common-bindings metadata plus mode tampering.'],limits=['Successful kernel send does not prove client receipt/decoding. Client unknown outcomes require inspection and never trigger automatic retry.','No live provider calls, formal proof discharge, TerminalBench score, full-supervisor START or token-savings claim in this offline evidence.'],related_test_runs=['root-tests-02','outer-tests-03','client-tests-01','service-final-tests-03'])
write(A/'integration-review-02.json',review);copy('integration-review-02.json','reviews/integration-review.json')
write(E/'qualification.json',dict(schema='repeated-compact-edit-offline-qualification@1',complete=True,mode='mapped-span@2',default_mode_unchanged='disabled',prior_mapped_mode_preserved='mapped-span@1',maximum_successful_preparations_per_session=16,one_outstanding_proposal=True,distinct_passed_tests=714,test_breakdown={k:v[1] for k,v in runs.items()},legacy_byte_comparison_cases_separate=32,all_latest_run_recorded_sources_unchanged=True,all_current_python_bindings_match=True,docker=dict(complete=load(A/'docker/run-01/qualification.json')['complete'],confirmed_edits=3,distinct_working_generations=4,owner_uid=1000,worker_uid=1001,network='none',container_removed=True,authored_router_hook=True),scope=dict(external_model_calls=0,full_supervisor_start_qualified=False,official_benchmark_verifier_executed=False,formal_obligations_discharged=0,live_token_savings_measured=False),evidence_files=dict(tests='test-ledger.json',sources='source-bindings.json',history='retained-failures.json',final_integration_review='reviews/integration-review.json',byte_accounting='reviews/docker-byte-accounting-review.json'),exclusions=['private-state','private-fixture-ipc','private-worker-observations','raw prompts and transcripts','authentication files and values','DuckDB databases','large runtime archives and binaries','live-preparation and live-trial artifacts']))
copy('curate_evidence.py','curate_evidence.py')
write(E/'copy-list.json',dict(schema='repeated-compact-edit-evidence-copy-list@1',copies=copies,generated=['qualification.json','test-ledger.json','source-bindings.json','retained-failures.json','qualifications/core.json','copy-list.json'],manifest_sealing_pending_parent_review=True))
print(json.dumps(dict(files=sum(p.is_file() for p in E.rglob('*')),tests=len(allnodes),source_bindings=len(pins),destination=str(E))))
