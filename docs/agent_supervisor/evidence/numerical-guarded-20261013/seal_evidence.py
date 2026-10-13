"""Copy the reviewed qualification allowlist and seal its exact bytes."""
import hashlib,json,shutil
from pathlib import Path
R=Path('/home/barberb/lift_coding/artifacts/numerical-guarded-coverage-20261013')
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
OUT=P/'docs/agent_supervisor/evidence/numerical-guarded-20261013'
OUT.mkdir(parents=True,exist_ok=True)
paths=['baseline-v1.json','baseline.stdout','baseline.stderr','capture_v1_baseline.py',
       'environment.json','authored_evaluation_probe.py','authored-evaluation-observation.json',
       'qualify_public_source.py','run_checks.py','verify_publication_inputs.py','publication-inputs.json',
       'test-ledger.json','passed-node-ids.txt','public-evidence-audit.json','logic-integration-audit.json',
       'guarded-consumer-independent-review.json','check_documentation.py','seal_evidence.py']
paths+=json.loads((R/'docker-evidence-audit.json').read_bytes())['publishable_subset']
for folder in ['datasets-final-01','accelerator-final-01','accelerator-focused-01','accelerator-targeted-02','public-run-01','publication-check-01']:
    paths += [folder+'/'+name for name in ('command.json','stdout','stderr','exit.json','result.json','junit.xml') if (R/folder/name).is_file()]
paths += [str(p.relative_to(R)) for p in (R/'public-01').iterdir() if p.is_file()]
paths += [str(p.relative_to(R)) for p in (R/'frontend').rglob('*') if p.is_file()]
paths += ['logic-tests-01.xml','logic-tests-02.xml','logic-tests-03.xml']
if (R/'documentation-01').exists():
    paths += [str(p.relative_to(R)) for p in (R/'documentation-01').iterdir() if p.is_file()]
for name in sorted(set(paths)):
    src,dst=R/name,OUT/name
    assert src.is_file() and not src.is_symlink()
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    assert src.read_bytes()==dst.read_bytes()
files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
manifest=dict(schema='numerical-guarded-evidence-manifest@1',files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)},raw_results_copied_without_rewriting=True,private_owner_state_included=False)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(files=len(files)+1,bytes=sum(p.stat().st_size for p in files),manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())))
