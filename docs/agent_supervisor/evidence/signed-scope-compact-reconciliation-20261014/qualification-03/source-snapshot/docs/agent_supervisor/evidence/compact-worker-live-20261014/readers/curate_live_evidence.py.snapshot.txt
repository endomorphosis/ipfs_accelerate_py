"""Copy an explicit content-free allowlist; seal only after the third readout."""
import argparse, hashlib, json, shutil
from pathlib import Path
A=Path(__file__).resolve().parent
Q=Path('/home/barberb/lift_coding/.worktrees/supervisor-token-growth-20261013')
DEST=Q/'docs/agent_supervisor/evidence/compact-worker-live-20261014'
FILES={
 'readouts/original-pair.json':'live-readout-pair-final-01.json',
 'readouts/native-shape-original-pair.json':'native-shape-pair-final-01.json',
 'readouts/identity-diagnostic.json':'identity-diagnostic-01.json',
 'readouts/identity-control-diagnosis.json':'identity-token-control-audit.json',
 'inputs/original-pair-lock.json':'pair-lock.json',
 'inputs/original-readout-registration.json':'live-readout-input.json',
 'inputs/original-control-audit.json':'live-preparation/control-audit.json',
 'inputs/metadata-release.json':'metadata-treatment-release-03.json',
 'inputs/metadata-execution-lock.json':'metadata-execution-lock-03.json',
 'inputs/metadata-control-audit.json':'live-preparation/metadata-treatment-control-audit-03.json',
 'inputs/initial-prepare-refusal.json':'live-preparation/identity-prepare-result.json',
 'reviews/final-offline-readiness.json':'final-readiness-review.json',
 'reviews/original-readout.json':'live-readout-independent-review-02.json',
 'reviews/metadata-preparation.json':'metadata-treatment-independent-preparation-review-03.json',
 'reviews/metadata-reader.json':'metadata-treatment-reader-independent-review-03.json',
 'reporting/original-readout-freeze.json':'live-readout-freeze-02.json',
 'reporting/original-readout-checks.json':'live-readout-checks-02.json',
 'reporting/native-shape-checks.json':'native-shape-checks.json',
 'reporting/metadata-finalized-checks.json':'metadata-treatment-finalized-checks-01.json',
 'history/superseded-readout-freeze.json':'live-readout-freeze.json',
 'history/superseded-readout-review.json':'live-readout-independent-review.json',
 'history/superseded-readout-checks.json':'live-readout-checks-before-unreceipted-mapping-fix.json',
 'history/collect_live_readout_before_mapping_fix.py':'collect_live_readout-before-unreceipted-mapping-fix.py',
 'readers/collect_live_readout.py':'collect_live_readout.py',
 'readers/collect_native_shape.py':'collect_native_shape.py',
 'readers/collect_metadata_treatment_03.py':'collect_metadata_treatment_03.py',
 'readers/check_live_readout.py':'check_live_readout.py',
 'readers/check_native_shape.py':'check_native_shape.py',
 'readers/check_metadata_treatment_finalized_03.py':'check_metadata_treatment_finalized_03.py',
 'readers/curate_live_evidence.py':'curate_live_evidence.py',
 'launchers/run_locked_trial.py':'run_locked_trial.py',
 'launchers/run_metadata_treatment_03.py':'run_metadata_treatment_03.py',
}
for label,directory in [('identity','grok-identity-02'),('comments','grok-comments-02'),('metadata','grok-comments-metadata-03')]:
 FILES[f'launch/{label}-start.json']=f'live-trials/{directory}/launch-start.json'
 if label!='metadata':FILES[f'launch/{label}-result.json']=f'live-trials/{directory}/launch-result.json'
FINAL_FILES={
 'readouts/metadata-treatment.json':'metadata-treatment-readout-final-01.json',
 'launch/metadata-result.json':'live-trials/grok-comments-metadata-03/launch-result.json',
}
def binding(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
def curate(final=False):
 DEST.mkdir(parents=True,exist_ok=True)
 copies=[]
 for name,source in sorted((FILES|FINAL_FILES if final else FILES).items()):
  src=A/source;dst=DEST/name
  assert src.is_file() and not src.is_symlink()
  raw=src.read_bytes();assert len(raw)<2_000_000
  dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert dst.read_bytes()==raw, 'existing copied evidence changed: '+name
  else:dst.write_bytes(raw)
  copies.append(dict(relative_path=name,source=binding(src),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 manifest=dict(schema='compact-worker-live-public-evidence@1',sealed=final,
  benchmark_source_commit='3f08cc43cbd96c9f8a5114091f65566a5962dbbc',
  runtime_archive_sha256='1fe8bfc4564a916174c4b9df42c7243d6b0e6fd8b1e71593a2385bf7fbe8d157',
  original_pair_lock_sha256='295b12b3f4464bbab8422ce8fdfe0414949fd3295948a54426a95ce48e015fa5',
  collector_sha256='79c0edcc6eda284ac13f2f0eb3d3341a84125523d377530b4f4dbb5cabb19a10',
  third_reader_sha256='0af7025ed58a89e14db28ff14e97249c04db89b295dc48cedf9d3c2a5bc18e4e',
  copied_files=copies, copied_file_count=len(copies),
  raw_full_receipts_published=False,raw_prompts_or_transcripts_published=False,
  private_databases_credentials_or_runtime_archive_published=False,
  original_pair_population_unchanged=True,causal_advantage_claimed=False,
  provider_calls_by_curation=0)
 if final:
  finalreadout=json.loads((A/'metadata-treatment-readout-final-01.json').read_text())
  assert finalreadout['status']=='observed' and finalreadout['frozen_source_pin_count']==327 and finalreadout['frozen_source_pins_verified'] is True
  manifest['final_collection_verified_source_pins']=327
  files=[]
  for p in sorted(DEST.rglob('*')):
   if p.is_file() and p.name!='manifest.json':files.append(dict(relative_path=p.relative_to(DEST).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
  manifest.update(files=files,file_count=len(files))
  path=DEST/'manifest.json'
  with path.open('x')as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
 output=A/('live-curation-final.json'if final else'live-curation-staged.json')
 with output.open('x')as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps(dict(destination=str(DEST),copied_files=len(copies),sealed=final,receipt=binding(output))))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--final',action='store_true');curate(p.parse_args().final)
