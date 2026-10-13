"""Copy only the reviewed public allowlist and seal its exact bytes."""
import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
OUT=P/'docs/agent_supervisor/evidence/numerical-loop-20261013'
assert OUT.is_dir()
paths=['baseline.json','capture_baseline.py','environment.json','run_checks.py',
    'qualify_public_source.py','authored_loop_protocol_probe.py','authored-loop-protocol-observation.json',
    'frontend_cross_layer_audit.py','frontend-cross-layer-audit.json','logic-independent-review.json',
    'audit_public_evidence.py','public-evidence-audit.json','audit_docker_evidence.py',
    'docker-fixture-failure-diagnosis.json','test-ledger.json','passed-node-ids.txt','summarize_tests.py',
    'accelerator-frozen.json','historical-summary-replay.json','verify_publication_inputs.py',
    'publication-inputs.json','audit_planning_summary.py','release_docker.py','check_documentation.py','seal_evidence.py','logic-tests-03.xml']
paths+=json.loads((R/'docker-evidence-audit.json').read_bytes())['publishable_subset']
for folder in ('baseline-run','datasets-final-02','accelerator-final-01','accelerator-focused-01',
               'public-run-02','publication-check-01','documentation-01','documentation-02'):
    paths += [folder+'/'+name for name in ('command.json','stdout','stderr','exit.json','result.json',
        'junit.xml','test-nodes.json','gates.stdout','gates.stderr') if (R/folder/name).is_file()]
paths += [str(p.relative_to(R)) for p in (R/'public-02').iterdir() if p.is_file()]
paths += [str(p.relative_to(R)) for p in (R/'frontend').rglob('*') if p.is_file()]
if (R/'consumer-summary-independent-review.json').is_file():paths.append('consumer-summary-independent-review.json')
for name in sorted(set(paths)):
    src,dst=R/name,OUT/name
    assert src.is_file() and not src.is_symlink()
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    assert src.read_bytes()==dst.read_bytes()
files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
expected=set(paths)|{'README.md','.gitattributes'}
assert {str(p.relative_to(OUT)) for p in files}==expected
manifest=dict(schema='numerical-loop-evidence-manifest@1',files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)},raw_results_copied_without_rewriting=True,private_owner_state_included=False)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(files=len(files)+1,bytes=sum(p.stat().st_size for p in files),manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())))
