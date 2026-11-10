"""Authored fixtures and exact old closed-output preservation, no provider calls."""
from pathlib import Path
import hashlib,json,sys,time,unittest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import read_projection as added
X=Path('/home/barberb/lift_coding/artifacts/grok-cap-end-20261015')


def bind(path):
 raw=path.read_bytes();return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


label=sys.argv[1]
assert label.isascii() and label.replace('-','').isalnum()
paths=sorted(ROOT.glob('*.py'))+[ROOT/'upstream-freeze.json']
upstream=json.loads((ROOT/'upstream-freeze.json').read_text())
paths += [Path(row['path']) for row in upstream['files']]+[X/'readout-final.json']
before=[bind(p) for p in paths];start=time.monotonic()
with (ROOT/(label+'.log')).open('x') as f:
 result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_read_projection'))
r=json.loads((X/'recipe.json').read_text())
base=added.baseline_reader().read_comparison(X/'recipe.json',prior_directory=r['previous_run']['output'])
expected=json.loads((X/'readout-final.json').read_text())
unchanged_old=base==expected
# The additive reader can read old reports without changing any known counters or unknowns.
projection=added.read_run(r['output'])
old_projection_absent=bool(projection['trials']) and all(row['observation']['status']=='closed_candidate_observations'
 and all(o['status']=='absent' for o in row['observation']['observations']) for row in projection['trials'])
runtime=[m for m in sys.modules if m.startswith(('ipfs_accelerate_py','ipfs_datasets_py','ipfs_kit_py'))]
unchanged=before==[bind(p) for p in paths]
passed=result.wasSuccessful() and unchanged and unchanged_old and old_projection_absent and not runtime
q=dict(schema='candidate-projection-reader-qualification@1',passed=passed,tests_run=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),files=before,
 source_and_preserved_pins_unchanged=unchanged,old_closed_metrics_and_unknowns_identical=unchanged_old,
 prior_projection_absent=old_projection_absent,runtime_modules_imported=runtime,provider_calls=0,
 benchmark_steps_run=0,raw_prompts_or_provider_text_exported=False,tests_log=bind(ROOT/(label+'.log')),
 seconds=time.monotonic()-start)
write(ROOT/(label+'.json'),q)
print(json.dumps({k:q[k] for k in ('passed','tests_run','failures','errors','skipped','old_closed_metrics_and_unknowns_identical','prior_projection_absent','runtime_modules_imported')}))
raise SystemExit(0 if passed else 1)
