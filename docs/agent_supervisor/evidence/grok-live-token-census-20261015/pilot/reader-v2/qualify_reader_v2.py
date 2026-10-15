"""Offline reader qualification, byte preservation and retained-report replay."""
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import read_planning_ablation as added
from export_custody_contract import export

A=Path('/home/barberb/lift_coding/artifacts/doctor-public-validation-20261014')
Z=Path('/home/barberb/lift_coding/artifacts/planner-acceptance-20261015')
T=Path('/home/barberb/lift_coding/artifacts/declared-public-planning-20261015')


def bind(path):
    raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def write(name,value):
    with (ROOT/name).open('x') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


started=time.monotonic()
source_paths=sorted(ROOT.glob('*.py'))+[ROOT/name for name in
    ('reporting-freeze.json','copy-provenance.json','custody-export-provenance.json')]
old_freeze=json.loads((T/'reporting-freeze.json').read_text())
preserved=[Path(row['path']) for row in old_freeze['files']]+[T/'reporting-freeze.json',T/'readout-final.json',
    A/'readout-final.json',Z/'readout-final.json',T/'readout-final-v2.json']
before=[bind(path) for path in source_paths+preserved]
for row in old_freeze['files']:assert bind(Path(row['path']))==row
assert export()==json.loads((ROOT/'custody-export-provenance.json').read_text())
with (ROOT/'tests-final.log').open('x') as stream:
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_read_pilot','test_safe_diagnostics','test_terminal_rejection_reader','test_public_validation_reader',
        'test_planning_ablation_reader','test_reader_import_closure']))
reader=added.baseline_reader()
replays=[]
for directory in (A,Z):
    recipe=json.loads((directory/'recipe.json').read_text())
    expected=json.loads((directory/'readout-final.json').read_text())['pilot']
    current=reader.read_run(recipe['output'])
    assert current==expected
    replays.append(dict(prior_readout=bind(directory/'readout-final.json'),baseline_projection_identical=True))
prior=json.loads((Z/'recipe.json').read_text())['output']
current=added.read_comparison(T/'recipe.json',prior_directory=prior)
assert current==json.loads((T/'readout-final-v2.json').read_text())
runtime=[name for name in sys.modules if name.startswith(('ipfs_accelerate_py','ipfs_datasets_py','ipfs_kit_py'))]
passed=result.wasSuccessful() and before==[bind(path) for path in source_paths+preserved] and not runtime
record=dict(schema='reader-import-closure-qualification@1',passed=passed,tests_run=result.testsRun,
    failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    source_and_preserved_pins_unchanged=before==[bind(path) for path in source_paths+preserved],files=before,
    closed_custody_export_identical=True,replays=replays,retained_t_readout=bind(T/'readout-final-v2.json'),
    retained_t_repeat_identical=True,original_empty_t_readout_preserved=(T/'readout-final.json').stat().st_size==0,
    runtime_modules_imported=runtime,provider_calls=0,benchmark_steps_run=0,seconds=time.monotonic()-started,
    tests_log=bind(ROOT/'tests-final.log'))
write('qualification.json',record)
assert passed and result.testsRun==43
print(json.dumps({key:record[key] for key in ('passed','tests_run','source_and_preserved_pins_unchanged',
    'retained_t_repeat_identical','original_empty_t_readout_preserved','runtime_modules_imported','provider_calls')}))
