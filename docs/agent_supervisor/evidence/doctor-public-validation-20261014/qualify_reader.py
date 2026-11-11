"""Offline pure-reader qualification; no launcher, providers or runtime imports."""
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

A=Path(__file__).resolve().parent
W=Path('/home/barberb/lift_coding/.worktrees/supervisor-public-validation-20261014')
C=Path('/home/barberb/lift_coding/artifacts/doctor-repaired-live-20261014')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pins(paths):return {str(path):dict(sha256=sha(path),bytes=path.stat().st_size) for path in paths}
provenance=json.loads((A/'reader-contract-provenance.json').read_text())
sources=[Path(row['source']) for row in provenance['records']]
source_before=pins(sources)
for row in provenance['records']:
 assert sha(Path(row['source']))==row['source_sha256']
 assert sha(Path(row['output']))==row['output_sha256']
artifacts=[path for path in A.glob('*.py') if path.name!='run-tests.py']+[A/'reader-contract-provenance.json']
artifact_before=pins(artifacts)
prior=json.loads((A/'public-wiring03-source-pins.json').read_text())['files']
owned=[W/'benchmarks/agent_supervisor/container_coding'/name for name in ('benchmark_controls.py',
 'full_supervisor_benchmark.py','full_supervisor_harbor_agent.py','terminal_container_supervisor.py',
 'terminal_doctor_coding_policy.py','terminal_source_view_observation.py','test_terminal_public_validation_policy.py')]
owned_match=all(sha(path)==prior[str(path.relative_to(W))] for path in owned)
old_paths=[C/'readout-final.json',C/'read_pilot.py',C/'safe_diagnostics.py',C/'extract_reader_contracts.py']
old_before=pins(old_paths)
modules=['test_read_pilot','test_safe_diagnostics','test_terminal_rejection_reader','test_public_validation_reader']
started=time.monotonic()
with (A/'reader-public-tests-final.log').open('w') as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
import read_pilot
current=read_pilot.read_run(read_pilot.PRIOR)
expected=json.loads((C/'readout-final.json').read_text())['pilot']
assert current==expected
(A/'retained-c-public-reader-replay.json').write_text(json.dumps(dict(schema='retained-pilot-reader-replay@1',
 prior_readout_sha256=sha(C/'readout-final.json'),projection_identical=True,pilot=current,
 provider_calls=0,new_public_validation_invented=False),indent=2,sort_keys=True)+'\n')
runtime_imports=sorted(name for name in sys.modules if name=='ipfs_accelerate_py' or name.startswith('ipfs_accelerate_py.')
 or name=='ipfs_datasets_py' or name.startswith('ipfs_datasets_py.'))
record=dict(schema='public-validation-reader-qualification@2',tests_run=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful(),
 source_files=source_before,source_pins_unchanged=source_before==pins(sources),
 artifact_files=artifact_before,artifact_pins_unchanged=artifact_before==pins(artifacts),
 benchmark_owned_files=pins(owned),benchmark_owned_files_match_public_wiring03=owned_match,
 retained_c_projection_identical=current==expected,old_c_evidence_unchanged=old_before==pins(old_paths),
 runtime_modules_imported=runtime_imports,provider_calls=0,
 live_launcher_created=any((A/name).exists() for name in ('run.py','recipe.json')),
 seconds=time.monotonic()-started)
(A/'reader-public-qualification.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
assert result.wasSuccessful() and not runtime_imports and record['artifact_pins_unchanged'] and record['source_pins_unchanged'] and owned_match
print(json.dumps({key:record[key] for key in ('tests_run','passed','source_pins_unchanged','artifact_pins_unchanged',
 'benchmark_owned_files_match_public_wiring03','retained_c_projection_identical','old_c_evidence_unchanged',
 'runtime_modules_imported','provider_calls','live_launcher_created')}))
