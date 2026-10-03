"""Copy an explicit public diagnostic inventory; no model or state payloads."""
import hashlib,json,shutil,tarfile
from pathlib import Path

B=Path(__file__).parent
ART=B.parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
OUT=A/'docs/agent_supervisor/evidence/source384-docker-stage-diagnostics-20261003'
OUT.mkdir(parents=True,exist_ok=False)

def copy(source,relative):
    assert source.is_file() and not source.is_symlink()
    target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)

common_names=('analyze_phases.py','check_diagnostic.py','cleanup-observation.json','command.json',
    'controls-command.json','controls-native-guard.json','diagnostic-collection.json',
    'diagnostic-result.json','docker-enforced-limits.json','docker.log','exit.json',
    'independent-review.json','instrumented-probe.py','phase-analysis.json','phase-events.jsonl',
    'phase-summary.json','producer-verification.json','run-admission.json','run_diagnostic.py','timing_runtime.py')
generations=(
 ('worker-budget','source384-docker-stage-diagnostic-20261003',
  ('controls-exit.json','controls-final-exit.json','controls-final-review-exit.json',
   'controls-final-review.log','controls-final.log','controls-initial-native-guard.json',
   'controls-reviewed-exit.json','controls-reviewed.log','controls.log')),
 ('cold-preparation','source384-docker-cold-stages-20261003',
  ('controls-01-exit.json','controls-01.log','controls-02-exit.json','controls-02.log',
   'plan.json','projection-persistence-distribution.json')),
)
for label,directory,extra in generations:
    source=ART/directory
    for name in common_names+extra:copy(source/name,label+'/'+name)
    # Only fixed public qualifier/deployment outputs are admissible here.
    for name in ('admission-estimate.json','public-instruction.md','qualification.json',
      'resources.json','resource-probe.stderr','resource-probe.stdout',
      'source384-context.json','source384-probe-status.json','source384-probe.stderr',
      'source384-probe.stdout','source384-result.json'):
        copy(source/'docker-01'/name,label+'/docker-01/'+name)
    deployment=('deployment.json','original-inputs.log','root-create.log','archive-extract.log',
      'source-generation.log','system-install.log','python-runtime-install.log','python-create.log',
      'torch-cpu-install.log','python-install.log','account-create.log','native-extensions.log',
      'native-imports.log','empty-native-start.log','retained-inputs.log')
    for name in deployment:copy(source/'docker-01/deployment'/name,label+'/docker-01/deployment/'+name)

archive=ART/'source384-docker-reconstruction-20261003/bundle'
manifest=json.loads((archive/'manifest.json').read_text())
assert manifest['archive_sha256']=='43ba7c9bc830b9c6c684589db15ddad25a37339014a5287d41a962c10d98f9c3'
assert hashlib.sha256((archive/'runtime.tar.gz').read_bytes()).hexdigest()==manifest['archive_sha256']
copy(archive/'manifest.json','common/archive-manifest.json')
owners=[
 'source/benchmarks/agent_supervisor/container_coding/terminal_deployment.py',
 'source/benchmarks/agent_supervisor/container_coding/terminal_source384_qualification.py',
 'source/benchmarks/agent_supervisor/container_coding/benchmark_resource_profile.py',
 'source/ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
 'source/ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/codebase_ir.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/content.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/duckdb_ast_store.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/duckdb_ingest.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/cache.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/codebase_source_units_384_worker.py',
 'datasets/ipfs_datasets_py/logic/software_contracts/semantic_index/scanner.py',
 'datasets/ipfs_datasets_py/duckdb_control/codebase_catalog.py',
]
indexed={row['path']:row for row in manifest['files']}
remaining=set(owners)
with tarfile.open(archive/'runtime.tar.gz','r|gz') as tar:
    for member in tar:
        name=member.name
        if name not in remaining:continue
        assert member.isfile() and member.size<=1024*1024
        raw=tar.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==indexed[name]['sha256']
        target=OUT/'common/owners'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        remaining.remove(name)
assert not remaining
copy(B/'README.md','README.md')
copy(B/'package_evidence.py','package_evidence.py')
qualification=dict(schema='source384-docker-stage-diagnostics-evidence@1',diagnostic_only=True,
 production_qualification_claimed=False,benchmark_result=False,completed_model_inference_qualified=False,
 source_qualified_proof_claimed=False,provider_calls=0,training_steps=0,official_verifier_executed=False,
 archive_sha256=manifest['archive_sha256'],resource_profile='source384-5cpu-12gib@1',
 original_tracked_inputs=218,native_context_budget_seconds=90,probe_budget_seconds=270,
 generations=[dict(path=label,analysis=json.loads((OUT/label/'phase-analysis.json').read_text()),
   cleanup=json.loads((OUT/label/'cleanup-observation.json').read_text())) for label,_,_ in generations],
 earlier_production_failure_manifest='f3dc59d2cacd6e0375e420384af5fa4ee8870ebe626dd3f574008833a6f72d33')
(OUT/'qualification.json').write_text(json.dumps(qualification,indent=2,sort_keys=True)+'\n')
rows=[]
for file in sorted(OUT.rglob('*')):
    if file.is_file():rows.append(dict(path=file.relative_to(OUT).as_posix(),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
(OUT/'manifest.json').write_text(json.dumps(dict(schema='closed-source384-stage-diagnostics@1',files=rows),indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(OUT),files=len(rows),bytes=sum(r['bytes'] for r in rows),manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())))
