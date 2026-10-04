"""Export bounded regression evidence; raw failure output stays private/local."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
OUT = B / 'public-regression'
A = Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
OWNERS = [
 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
 'ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py']

def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
 (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

final = json.loads((B / 'regression-02/exit.json').read_text())
assert final['returncode'] == 0 and final['source_pins_unchanged']
assert final['xml_counts'] == dict(tests=249, failures=0, errors=0, skipped=0)
for path in OWNERS:
 assert digest(A / path) == final['after_pins']['source/' + path]
OUT.mkdir(exist_ok=False)
for label in ('regression-02', 'regression-01', 'baseline-stale-01'):
 for name in ('command.json', 'exit.json'):
  shutil.copyfile(B / label / name, OUT / (label + '-' + name))
shutil.copyfile(B / 'regression-02/results.xml', OUT / 'regression-02.xml')
for name in ('run_regressions.py', 'run_stale_baseline.py', 'baseline_overlay.py', 'package_regressions.py'):
 shutil.copyfile(B / name, OUT / name)
failures = []
for label in ('regression-01', 'baseline-stale-01'):
 rows = []
 for case in ET.parse(B / label / 'results.xml').getroot().iter('testcase'):
  for child in case:
   if child.tag in ('failure', 'error'):
    assert case.get('name') == 'test_typed_daemon_promotes_local_attempt_before_provider'
    assert child.get('message') == "AttributeError: 'QuackStateServer' object has no attribute 'revoke_typed_client_grant'"
    rows.append(dict(test=case.get('name'), kind=child.tag, exception_type='AttributeError',
      missing_api='QuackStateServer.revoke_typed_client_grant', body_exported=False))
 assert len(rows) == 1
 failures.append(dict(label=label, failures=rows, raw_xml_sha256=digest(B / label / 'results.xml'), raw_xml_exported=False))
write('known-stale-control.json', dict(schema='source384-deferral-stale-control@1', runs=failures,
 qualification=False, production_fix_scope=False))
write('owner-bindings.json', dict(schema='source384-deferral-owner-bindings@1',
 original={path:digest(B / 'owner-original' / Path(path).name) for path in OWNERS},
 final={path:digest(A / path) for path in OWNERS},
 original_snapshots_exported=False, scope='Three exact runtime owners; other consumed pins appear in each command and exit receipt'))
(OUT / 'README.md').write_text('''# Source384 pre-dispatch deferral: affected existing regressions

The final `regression-02` run passed **249 distinct controls, with no failures, errors, or skips**, against the three recorded final runtime owner hashes. These cover the Portal bridge, candidate budgets, bounded pre-effect retry, Source384 task selections, and selected embedded-owner/provider-custody controls. This is an affected host regression group, not a whole-repository or benchmark qualification.

The earlier `regression-01` had 249 passes and one stale fixture teardown failure: `QuackStateServer.revoke_typed_client_grant` is absent. The identical failure was reproduced by `baseline-stale-01` with all three exact pre-fix owner snapshots loaded at their canonical paths. Only that test was excluded from the final scope. Its failed XML remains local, and the bounded exported failure projection states the missing API without exporting traceback bodies. The independent new actual typed-owner control uses working teardown and is published in the separate component package; do not double-count preliminary runs.

The new runtime behavior keeps the same admitted process/attempt/control receipt after an exact pre-dispatch Source384 resource refusal. It requires successful task/resource-claim cleanup, a native bridge receipt, current custody and unchanged native admission checks. Callback intent is atomically settled to a closed no-dispatch state; a bounded 5-second cooldown permits fresh source verification. The old callback-intent@1 body vocabulary remains unchanged. Generic failures, cancellation, missing native deadline, post-provider states, foreign claims and uncertain effects gain no retry authority.

The child retry deadline is the minimum of its initially captured signed grant expiry and original attempt start plus configured lease duration; monotonic capture prevents renewal of the grant bound. The separate 840-second benchmark work cutoff remains enforced by the existing driver and cancellation. Nomination reading and Source384 validation share the remaining bound, with the original 90-second per-validation maximum. At most 16 resource deferrals are admitted. The primary-gate diagnostic preserves the original bounded aggregate gate/sample; optional pressure attribution is omitted. No threshold or source/proof gate is relaxed.

No Docker benchmark, external provider call, token advantage, host-pressure recovery, task completion, whole-program proof, or completion-RPC deadline extension is established by these controls. Raw stdout, credential stores, model bodies, verifier bodies and source snapshots are not exported. Commands record only explicit environment overrides and selected source pins, not the complete inherited environment. The baseline harness requires the separately retained original owner snapshots; their hashes are recorded in owner-bindings.json.
''')
rows=[]
for path in sorted(OUT.iterdir()):
 rows.append(dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path)))
write('manifest.json', dict(schema='bounded-evidence-package@1', files=rows, member_count=len(rows),
 member_bytes=sum(row['bytes'] for row in rows)))
print(json.dumps(dict(directory=str(OUT), members=len(rows), manifest_sha256=digest(OUT/'manifest.json'))))
