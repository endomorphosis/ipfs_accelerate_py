"""Bind the reviewed public evidence, excluding the manifest from its own hash."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).parent.resolve()
repo = Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
public = repo / 'docs/agent_supervisor/evidence/context-projection-pair-20261013'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
copy_path = public / 'copy-list.json'
copies = json.loads(copy_path.read_bytes())
extra = {
    'check_documentation.py': 'documentation/check_documentation.py',
    'documentation-final/result.json': 'documentation/result.json',
    'documentation-final/gates.stdout': 'documentation/gates.stdout',
    'documentation-final/gates.stderr': 'documentation/gates.stderr',
    'context-projection-final-review.json': 'final-review.json',
    'context-projection-final-recomputation.json': 'native-codex/final/independent-recomputation.json',
    'recompute_final_pair_review.py': 'native-codex/final/recompute_final_pair_review.py',
    'review_final_pair_docs.py': 'review_final_pair_docs.py',
    'seal_evidence.py': 'seal_evidence.py',
}
for source, destination in extra.items():
    src, dst = root / source, public / destination
    assert src.is_file() and not src.is_symlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    row = dict(source_relative=source, evidence_relative=destination,
               bytes=src.stat().st_size, sha256=digest(src))
    copies['copied_files'] = [r for r in copies['copied_files'] if r['evidence_relative'] != destination]
    copies['copied_files'].append(row)
copies['manifest_frozen'] = True
copies['status'] = 'all_offline_and_native_outcomes_retained_and_manifest_frozen'
copy_path.write_text(json.dumps(copies, indent=2, sort_keys=True) + '\n')
for row in copies['copied_files']:
    src, dst = root / row['source_relative'], public / row['evidence_relative']
    assert all(p.is_file() and not p.is_symlink() for p in (src, dst))
    assert digest(src) == digest(dst) == row['sha256']
    assert src.stat().st_size == dst.stat().st_size == row['bytes']
gate = json.loads((public / 'documentation/result.json').read_bytes())
assert gate['passed']
for name, expected in gate['source_sha256'].items():
    assert digest(repo / name) == expected, name
scope_path = public / 'scope.json'
scope = json.loads(scope_path.read_bytes())
scope['manifest_frozen'] = True
scope_path.write_text(json.dumps(scope, indent=2, sort_keys=True) + '\n')
files = {}
for path in sorted(public.rglob('*')):
    assert not path.is_symlink(), str(path)
    if path.is_file() and path != public / 'manifest.json':
        files[str(path.relative_to(public))] = dict(bytes=path.stat().st_size, sha256=digest(path))
manifest = dict(schema='context-projection-evidence-manifest@1',
    self_excluded=True, files=files, file_count=len(files),
    copied_source_artifact_count=len(copies['copied_files']),
    total_bytes=sum(v['bytes'] for v in files.values()),
    authentication_authority=False, proof_authority=False)
(public / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps({k:v for k,v in manifest.items() if k != 'files'}))
