"""Verify final qualification bytes after preserving concurrent upstream changes."""
import hashlib,json,subprocess
from pathlib import Path
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
D=Path('/home/barberb/lift_coding/.worktrees/indexed-coverage-datasets-20261013')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
release=json.loads((R/'docker-release-03.json').read_bytes())
assert all(sha(p)==v for p,v in release['source_bindings'].items())
assert all(sha(p)==v for p,v in release['runner_bindings'].items())
base=json.loads((R/'baseline.json').read_bytes())
assert structured_source_384._implementation()==base['structured_checkpoint_implementation']
pins=json.loads((R/'accelerator-frozen.json').read_bytes())['source_sha256']
assert all(sha(P/p)==v for p,v in pins.items())
upstream=git(D,'rev-parse','8f3856ee9')
subprocess.run(['git','merge-base','--is-ancestor',upstream,'HEAD'],cwd=D,check=True)
files=git(D,'diff','--name-only',upstream,'HEAD').splitlines();assert len(files)==6
public=json.loads((R/'public-02/qualification.json').read_bytes())
assert public['complete'] and public['historical_reports_unchanged'] and public['structured_checkpoint_pins_unchanged']
for module,digest in public['producer_sha256'].items():
    path=D/(module.replace('.','/')+'.py');assert sha(path)==digest
result=dict(schema='numerical-loop-publication-inputs@1',passed=True,datasets_commit=git(D,'rev-parse','HEAD'),preserved_datasets_upstream=upstream,accelerator_base=git(P,'rev-parse','HEAD'),source_bindings_exact=True,structured_checkpoint_pins_unchanged=True,historical_reports_unchanged=True,changed_datasets_files=files,qualified_source_and_public_input_count=len(release['source_bindings']),qualified_runner_count=len(release['runner_bindings']))
(R/'publication-inputs.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
