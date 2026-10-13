"""Verify qualification bindings after preserving the concurrent upstream change."""
import hashlib,json,subprocess
from pathlib import Path
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384
r=Path(__file__).resolve().parent
d=Path('/home/barberb/lift_coding/.worktrees/indexed-coverage-datasets-20261013')
release=json.loads((r/'docker-release-01.json').read_bytes())
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in release['source_bindings'].items())
base=json.loads((r/'baseline-v1.json').read_bytes())
assert structured_source_384._implementation()==base['structured_checkpoint_implementation']
upstream='18efafc7a28b891edc53a694838d458563490787'
subprocess.run(['git','merge-base','--is-ancestor',upstream,'HEAD'],cwd=d,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d,text=True).strip()
files=subprocess.check_output(['git','diff','--name-only',upstream,'HEAD'],cwd=d,text=True).splitlines()
assert len(files)==6
record=dict(schema='numerical-guarded-publication-inputs@1',passed=True,datasets_commit=head,preserved_upstream=upstream,source_bindings_exact=True,structured_checkpoint_pins_unchanged=True,changed_files=files,qualified_source_count=len(release['source_bindings']))
(r/'publication-inputs.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
