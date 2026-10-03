"""Collect closed external evidence; no runtime, task or model bodies."""
from pathlib import Path
import hashlib,json,shutil
B=Path(__file__).resolve().parent;W=B.parents[1];O=B/'public-evidence'
if O.exists():raise ValueError('existing closed package must remain unchanged')
O.mkdir();selected=[]
def copy(path,relative):
 if path.is_symlink() or not path.is_file():raise ValueError('regular evidence required')
 if path.stat().st_size>4*1024**2:raise ValueError('bounded compact evidence required')
 dest=O/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 selected.append({'path':str(relative),'source':str(path.relative_to(W)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
root_names=['build_fresh.py','run_qualification.py','package_evidence.py','resources-before.json','build-command.json','build-exit.json','build.stdout','build.stderr','build-controller-command.json','build-controller-exit.json','build-controller.stdout','build-controller.stderr','build-audit.json','frozen-production-pins.json','run-admission.json','run-admission.template.json','qualification-command.template.json','launcher-review-02-controls.json','qualification-command.json','qualification-controller-command.json','qualification-controller-exit.json','qualification-controller.stdout','qualification-controller.stderr','qualification-before-pins.json','qualification-after-pins.json','qualification-public-task-pins.json','qualification-containers-before.json','qualification-containers-after.json','qualification-failure.json','qualification-exit.json','setup-progress-01.json','verdict.json','archive-binding-summary.json']
for name in root_names:copy(B/name,Path('run')/name)
for prefix in ('build-attempt-01','builder-review-01','launcher-review-01'):
 for path in sorted((B/prefix).iterdir()):copy(path,Path(prefix)/path.name)
for name in ['qualification.json','resources.json','admission-estimate.json','source384-result.json','source384-context.json','source384-probe-status.json','source384-probe.stdout','source384-probe.stderr','resource-probe.stdout','resource-probe.stderr']:
 copy(B/'qualification-01'/name,Path('native')/name)
copy(B/'bundle/setup-cache-selection.json',Path('native/setup-cache-selection.json'))
A=W/'.worktrees/ir-release-accelerate-20261002';D=W/'.worktrees/ir-release-datasets-20261002'
source_names={'source':[
 'ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
 'benchmarks/agent_supervisor/container_coding/terminal_deployment.py',
 'benchmarks/agent_supervisor/container_coding/terminal_initial_context.py',
 'benchmarks/agent_supervisor/container_coding/terminal_source384_qualification.py',
 'benchmarks/agent_supervisor/container_coding/terminal_setup_cache_advice.py',
 'benchmarks/agent_supervisor/container_coding/terminal_setup_cache_files.py',
 'benchmarks/agent_supervisor/container_coding/terminal_setup_cache_codex.py',
 'benchmarks/agent_supervisor/container_coding/terminal_setup_cache_libraries.py'],
 'datasets':['ipfs_datasets_py/logic/software_contracts/codebase_ir.py','ipfs_datasets_py/logic/software_contracts/cache.py','ipfs_datasets_py/logic/software_contracts/codebase_source_units_384.py','ipfs_datasets_py/logic/software_contracts/codebase_resources.py','ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py']}
manifest=json.loads((B/'bundle/manifest.json').read_text());rows={r['path']:r for r in manifest['files']}
for prefix,names in source_names.items():
 root={'source':A,'datasets':D}[prefix]
 for name in names:
  path=root/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==rows[prefix+'/'+name]['sha256']
  copy(path,Path('runtime-snapshots')/prefix/name)
(O/'README.txt').write_text('''Ordinary Source384 production-cache qualification: retained failure

The fresh local candidate bundle passed full archive verification:26,918 regular files, exactly13 changed and13 added against the previous a83 archive, no removals, unchanged checkpoint/GTE/wheel and nonmutable dependency metadata. The ordinary5CPU/12GiB qualification used production cache advice, with no diagnostic runtime hooks and unchanged90s Source384/270s probe/300s execution/2300s controller bounds. Setup passed, including exact local Torch wheel installation, native imports, empty supervisor START, retained original inputs and all three finite advice populations.

The run did NOT qualify. After10.370s preparation and159.394s initial_context, the probe failed at the second observe inside validate_shared_parent_units, waiting for a nested child resource lease. The probe lasted170.816s; total controller691.304s includes setup. Exact frozen source snapshots and raw source-free trace are included. No full Harbor trial or verifier was executed.

The failing request is4096MiB,1CPU and1child-process slot in SNAPSHOT_EVALUATION, beneath the replay4096MiB child and consumer6144MiB/3CPU/3slot parent. The consumer replay wrapper starts a90s budget and passes remaining time into D. At second observe, left=deadline−monotonic is passed to observe_current and admission is capped at min(30s,left). Exact numerical left is not recorded and cannot be reconstructed from whole-phase elapsed time. The observed exception is admission timeout, not a completed observation followed by its execution deadline check.

Preflight available memory9350MiB exceeded the estimated8602MiB threshold, but this was not admission authority. The later failure sample8667MiB/CPUstall41.31%/IO0.16%/memoryPSI0.0 was captured after unwinding. No actual failed admission sample or persisted backoff reason was exported, so this package does not attribute the refusal to memory, CPU or another specific condition.

All26918 archive files (3,914,491,746 logical bytes), four exact public Codex copies (628,736,528 bytes), and131 exact installed payloads (565,702,269 bytes) were advised with no reported errors and unchanged metadata. Archive advice was metadata-only; public binary/library bodies were explicitly hashed before advice with no body reads afterward. Advice is best effort: these byte totals are not measured freed bytes and do not establish admission. No global cache drop, scheduler threshold change, cancellation bypass or deadline increase occurred.

No immutable native inference export was produced before failure. Model-load and neural-candidate counts therefore remain unknown here; control flow alone is not counted as a published/verified model result. Provider and training calls are zero. The controller verified container cleanup and unchanged runtime/task input hashes. Both local candidate commits remain a measured generation; no inference, proof, benchmark reward or token-saving claim is made.

The first build attempt failed before creating an archive because its subprocess dependency path omitted multiformats. That exact script, command, failure and disk receipt remain under build-attempt-01. Retry used the already successful preparation environment; no runtime source or policy changed. The launcher label's earlier generation is retained separately. Current review explicitly separates passing configured-client components from historical host-default fixture refusals; it does not claim all possible startup modes qualified.

This is an external closed evidence draft, not a published runtime release. It contains no benchmark task source/instruction bodies, weights, database/CAS objects, auth contents or lease capabilities. The large archive/full inventory is excluded; its exact hash, metadata summary, full audit and reproduction command identify the retained local bundle. Model asset references contain only selection metadata and digests. Archived runtime source snapshots were checked against the actual bundle inventory. All package members are listed in manifest.json.
''')
(O/'provenance.json').write_text(json.dumps({'schema':'ordinary-source384-qualification-evidence-provenance@1','external_draft_only':True,'qualified':False,'archive_payload_included':False,'full_archive_manifest_included':False,'task_bodies_included':False,'model_weights_included':False,'db_cas_payloads_included':False,'auth_contents_included':False,'lease_authority_included':False,'selected_files':selected},indent=2,sort_keys=True)+'\n')
files=[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(O.rglob('*')) if p.is_file()]
(O/'manifest.json').write_text(json.dumps({'schema':'closed-ordinary-source384-production-cache-evidence@1','files':files},indent=2,sort_keys=True)+'\n')
print(json.dumps({'manifest_sha256':hashlib.sha256((O/'manifest.json').read_bytes()).hexdigest(),'members':len(files),'bytes':sum(x['bytes'] for x in files)}))
