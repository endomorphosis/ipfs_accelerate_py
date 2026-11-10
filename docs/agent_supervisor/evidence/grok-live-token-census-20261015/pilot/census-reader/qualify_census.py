"""Offline authored qualification; does not read live trial reports."""
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SOURCE = Path('/home/barberb/lift_coding/.worktrees/supervisor-grok-cap-end-20261015')


def binding(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


started = time.monotonic()
paths = sorted(ROOT.glob('*.py')) + [SOURCE / ('ipfs_accelerate_py/agent_supervisor/runtime/' + name)
    for name in ('semantic_router_translation.py', 'router_implementation_runner.py')]
before = [binding(path) for path in paths]
with (ROOT / 'tests-final.log').open('x') as stream:
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_read_census'))
runtime = [name for name in sys.modules if name.startswith(('ipfs_accelerate_py', 'ipfs_datasets_py', 'ipfs_kit_py'))]
unchanged = before == [binding(path) for path in paths]
passed = result.wasSuccessful() and unchanged and not runtime
record = dict(schema='closed-semantic-census-reader-qualification@1', passed=passed,
    tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
    source_pins_unchanged=unchanged, files=before, runtime_modules_imported=runtime,
    provider_calls=0, retained_reports_read=0, source_or_prompt_text_exported=False,
    tests_log=binding(ROOT / 'tests-final.log'), seconds=time.monotonic() - started)
with (ROOT / 'qualification.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps({key:record[key] for key in ('passed', 'tests_run', 'failures', 'errors', 'source_pins_unchanged', 'provider_calls')}))
raise SystemExit(0 if passed else 1)
