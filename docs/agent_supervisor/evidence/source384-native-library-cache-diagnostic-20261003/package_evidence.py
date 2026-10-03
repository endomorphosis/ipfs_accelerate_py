"""Seal this completed seven-library diagnostic using an explicit public whitelist."""
import hashlib
import json
from pathlib import Path
import shutil

B=Path(__file__).resolve().parent
OUT=B/'public-evidence'
if OUT.exists(): raise ValueError('closed package must be fresh')
OUT.mkdir()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v): (OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
files=[]
def put(name):
 p=B/name
 if not p.is_file() or p.is_symlink() or p.stat().st_size>200000:
  raise ValueError('explicit bounded regular evidence required: '+name)
 q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 files.append({'artifact':str(p),'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})

names=['cache_candidate.py','codex_cache_candidate.py','native_library_cache_candidate.py',
 'pin_native_libraries.py','run_diagnostic.py','test_codex_cache_candidate.py','test_native_library_cache_candidate.py',
 'controls-02.json','controls-02.log','final-sources.json','finite-library-source-pins.json',
 'source-pin-reproduction.json','independent-review.json','experiment-provenance.json',
 'host-archive-provenance.json','run-command.json','launch.json','launcher-exit.json','scope.json',
 'before-pins.json','after-pins.json','containers-before.json','containers-after.json','exit.json','failure.json',
 'cache-advice.json','cache-advice-status.json','codex-cache-advice.json','codex-cache-advice-status.json',
 'library-cache-advice.json','library-cache-advice-status.json','post-context-scheduler.json','verdict.json',
 'source-free-outcome.json','independent-outcome-audit.json','collect_outcome.py','audit_outcome.py',
 'docker-01/cache-resources-before.json','docker-01/cache-resources-after.json',
 'docker-01/codex-cache-resources-before.json','docker-01/codex-cache-resources-after.json',
 'docker-01/library-cache-resources-before.json','docker-01/library-cache-resources-after.json',
 'docker-01/resources.json','docker-01/admission-estimate.json','docker-01/detailed-resources.json',
 'docker-01/source384-context.json','docker-01/source384-result.json','docker-01/source384-probe-status.json',
 'docker-01/source384-probe.stdout','docker-01/source384-probe.stderr',
 'docker-01/worker-boundary/installation/native-codex-binary.log']
for n in names: put(n)

outcome=json.loads((B/'source-free-outcome.json').read_text())
audit=json.loads((B/'independent-outcome-audit.json').read_text())
assert audit['outcome_sha256']==sha(B/'source-free-outcome.json')
assert outcome['inference']['verified'] is False
omitted=[]
for n in ['docker-01/deployment/deployment.json','docker-01/qualification.json']:
 p=B/n
 omitted.append({'artifact':str(p),'bytes':p.stat().st_size,'sha256':sha(p),
  'reason':'Duplicate complete source-hash inventories and deployment metadata; selected outcome fields and independent equality checks retained.'})
write('omitted-receipt-metadata.json',{'schema':'omitted-local-receipt-descriptors@1','files':omitted,
 'native_inference_export_present':False,'task_bodies_auth_checkpoint_binary_and_database_payloads_included':False})
write('qualification.json',{'schema':'seven-library-diagnostic-evidence@1',
 'diagnostic_only':True,'qualified':False,'benchmark_result':False,'advantage_claimed':False,
 'provider_calls':0,'official_verifier_executed':False,'model_loads':None,'coverage':None,
 'initial_index_and_worker_output_validation_returned':'Control-flow inference from exact frozen traceback; no published numerical report.',
 'failure_boundary':'Post-worker source observation child resource lease',
 'failure_error':'LeaseTimeoutError','durable_inference_receipt_created':False,
 'post_unwind_backoff_reason':'proof_memory_headroom','post_unwind_available_memory_mib':8637,
 'post_unwind_active_leases_and_waiters':0,'exact_child_admission_sample_retained':False,
 'resource_profile':'source384-5cpu-12gib@1','actual_cpu_max':'500000 100000','actual_memory_max':'12884901888',
 'deadlines_seconds':{'source384_native':90,'qualifier_probe':270,'container_exec':300,'outer':2300},
 'timings_seconds':{'deployment':192.10708819700085,'prepare':10.435561653997866,
  'initial_context_until_failure':97.01350516800449,'native_probe':108.46736702099588,'whole_diagnostic':368.04462550800235},
 'advice':{'archive':{'files':26905,'bytes':3914232094},'native_binary_copies':{'files':4,'bytes':628736528},
  'native_libraries':{'files':7,'bytes':414442268}},
 'advice_bytes_are_freed_bytes':False,'all_advice_scopes_metadata_unchanged':True,
 'controls':{'passed':25,'receipt':'controls-02.json','source_generation':'final-sources.json',
  'scope':'Authored helper/diagnostic controls, not model tests; earlier24-control generation excluded.'},
 'original_files_preserved_across_deployment':218,'container_cleanup_observed':True,
 'archive_sha256':'a83b157f478dbcb869f134e71bfd02b17bdb7f82f3de8d281cc6202f66464d0b',
 'host_deployer_sha256':'bb4e4f845f03026572530b81fde6a540a2cd0ab89b17e606587ddfbc33dfc285',
 'archived_deployer_sha256':'c4e3dcc3aa2fae3d77fa538ec4855778d26a5cd70392ff54d4ae6b399653869d',
 'production_cache_policy_changed':False,'admission_or_deadline_changed':False,
 'prior_three_generation_package_manifest_sha256':'46941ee221c49b67f48c5fff9b5f5925993757186bcd6e6b53640c11884983bf',
 'next_128_wheel_member_experiment_included':False})
(OUT/'README.md').write_text('''# Seven-library setup-cache diagnostic

This separate instrumented run failed after the numerical worker returned and its output passed validation. The next required source observation could not acquire a child lease. No durable inference report, model-load count or decoded-coverage receipt was published. This is not a production qualification, official benchmark result or proof of a repair.

The trace reaches `codebase_source_units_384.py:248` at `observe()`, after `_worker` returned on line247 and `_validate_output` completed earlier on line248. It then fails at `RepositoryCodebaseIndex.observe_current` child acquisition. The frozen consumer's earlier initial-index preparation and root admission therefore returned. Those are control-flow inferences checked against exact owner hashes; they are distinct from a retained model result. `independent-outcome-audit.json` records the checks.

All three advice stages completed without reported errors or metadata changes:26905 archive files/3914232094 bytes, four independently hashed public executable copies/628736528 bytes, and seven exact installed public libraries/414442268 bytes. The seven comprise three archive-pinned DuckDB extensions and four CPU-wheel members. Their exact hashes, sizes and modes matched the reviewed selection metadata. Advice success and logical byte counts do not measure memory reclaimed. No global cache drop, cgroup policy change, body write or credential read was introduced by these helpers.

Observed available memory changed4430→8193 MiB across archive advice,8192→8810 MiB across executable advice, and8810→9216 MiB across library advice. Actual container limits were5 CPU and12288 MiB. The failure-handling sample was8637 MiB **after unwind**. A separate existing-owner scheduler snapshot, also after unwind, retained `proof_memory_headroom` backoff and zero active leases/waiters. Neither observation contains the exact child admission sample or contemporaneous lease state. The retained reason supports a headroom refusal; the later sample must not be substituted for the rejected decision.

Deployment took192.107 seconds, preparation10.436 seconds, initial context failed after97.014 seconds, the native probe lasted108.467 seconds, and the whole diagnostic368.045 seconds. The existing90-second Source384 contract,270-second probe,300-second container exec and2300-second outer bound were unchanged; this run does not qualify successful completion within them.

The final25 helper/diagnostic controls and exact authored sources are included. The earlier24-control generation predates preservation of a primary context error when optional diagnostic persistence fails; it is not counted or included here. The independently reviewed final generation is pinned by `independent-review.json` and `final-sources.json`.

The experiment intentionally reused archivea83b157f with the corrected host UV deployerbb4e4f84; its archived deployerc4e3dcc3 did not install Python. Source and model producer pins stayed unchanged. All218 original input hashes matched across deployment, and the retained cleanup observation found no matching container. Provider calls remain0 and verifier execution false; token counters, model-load count and coverage remain null.

Only an explicit public whitelist is copied. Authored framework/helper code, public library/binary metadata, resource receipts and framework failure stack traces are included. Raw benchmark bodies, authentication, hidden verifier inputs, native inference bodies, embeddings, weights, wheels, binaries, databases and full runtime archives are excluded. Large duplicate deployment/inventory receipts are described by hashes in `omitted-receipt-metadata.json`. Script paths retain the original host execution layout; `provenance.json` maps each copied artifact.

The earlier107-member package remains immutable and separate. The prospective128-wheel-member experiment is not included. No maintained repository document or production cache policy is changed by this staging package.
''')
put('package_evidence.py')
write('provenance.json',{'schema':'explicit-whitelist-provenance@1','copies':files})
rows=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
write('manifest.json',{'schema':'closed-seven-library-cache-diagnostic-evidence@1','files':rows})
result={'staging_package':str(OUT),'manifest_sha256':sha(OUT/'manifest.json'),'members':len(rows),
 'member_bytes':sum(r['bytes'] for r in rows),'files':[r['path'] for r in rows]+['manifest.json']}
(B/'scoped-files.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},sort_keys=True))
