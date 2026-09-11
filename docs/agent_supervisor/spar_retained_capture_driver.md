# Retained SPAR capture driver

`scripts.ops.agent_supervisor.spar_retained_capture_driver` executes the reviewed
capture boundary in one process. It does not advance native source, admit a
launch amendment, start a successor, settle callbacks, release old claims,
establish signing authority or declare completion.

Run the module from its frozen review checkout. Supply the actual accepted
native checkout/configuration and freshly reviewed source commit/tree, the
current fleet configuration, and an absent operation directory outside the
native checkout whose parent is already owned and private. The default CLI
performs inspection and closes its observation descriptors. It does not create
an operation directory, install a unit drop-in or write any stop marker.

```text
/usr/bin/python3 -I scripts/run_spar_retained_capture.py \
  --repository-root <accepted-native-checkout> \
  --config-path <absolute-native-config> \
  --fleet-config <absolute-current-fleet-config> \
  --operation-root <absent-private-operation-directory> \
  --expected-source-commit <fresh-native-commit> \
  --expected-source-tree <fresh-native-tree>
```

The driver eagerly imports the capture/prepare dependencies, verifies the native
DuckDB version profile, opens only a policy-sealed in-memory DuckDB, inventories
it and loads the migration catalog **before** native/fleet inspection or output,
unit or HOLD writes. The exact interpreter and dependency bytes remain bound
and are checked again before arming and each pre-install operation. Missing
DuckDB therefore rejects at `stage=new`, with no native closure.

If the selected isolated interpreter already imports DuckDB, that existing
runtime is observed and retained. Otherwise, explicitly bind the reviewed
installed native runtime; the driver never enables or searches the user site.
Generate a code declaration with the intended native interpreter before the
operation:

```text
python3 -c 'import json; from scripts.ops.agent_supervisor.spar_capture_runtime import observed_runtime; print(json.dumps(observed_runtime(), sort_keys=True))' > reviewed-runtime.json
sha256sum reviewed-runtime.json
```

Review the declaration and bind its exact digest in the prepared invocation:
`--runtime-manifest /absolute/reviewed-runtime.json
--runtime-manifest-sha256 <reviewed-digest>`. This selects code, not process,
task or database authority. It binds the interpreter executable hash, Python
ABI, native DuckDB version, package sources, metadata and compiled extension.
The isolated loader executes only verified package source bytes and a sealed
memfd copy of the extension. It runs no `.pth`/site hooks, adds no site-package
search path, reads no credentials and performs no installation or download.
Unbound imports, changed hashes, wrong interpreter/ABI and mixed already-loaded
DuckDB code reject admission.

Add `--session` to retain the inspected process and its two fleet coordination
locks for explicit JSON-line commands on stdin. The returned `inspection_cid`
selects the observation already retained in that process; a saved JSON document
cannot create a session. Use these commands in order:

```json
{"action":"arm","inspection_cid":"<CID emitted by this same process>"}
{"action":"request_closure"}
{"action":"poll"}
{"action":"capture"}
{"action":"prepare"}
{"action":"install"}
{"action":"finish"}
```

Repeat `poll` until it returns a positive native closure. A pending observation
keeps all handles. `capture` is refused before that closure. `close_inspection`
is available only before arming. Source/launch adoption remains a separate
reviewed operation after successful `finish`.

Inspection opens the existing SPAR `watchdog.lock` and repair `queue.lock`
without creating or unlinking either file. Both must be obtainable. The current
SPAR job must have a recognized non-running status and a finished prior attempt
or an explicitly empty attempt history; a recent repair-worker record must say
waiting/idle; the actual fixed repair-job unit must be positively inactive with
MainPID zero. Running claims, a selected-but-unclaimed job, unknown unit states,
missing records, stale worker status and lock-path replacement all deny entry.
The exact fleet configuration and job bytes remain bound through the operation.

The native capture session independently loads the accepted native operator in
fresh interpreters and retains its exact owner pidfd, cgroup descriptor, source,
bootstrap, broker and stopped-lane evidence. Namespace permission failures stay
unknown. No private reader token or environment is copied into the driver.

`arm` verifies the retained inspection, adds only the dedicated runtime
`95-native-legacy-capture.conf` drop-in and reloads the user manager. The drop-in
sets Restart=no, SendSIGKILL=no, an infinite stop timeout, RefuseManualStart=yes,
Delegate=yes, ExitType=cgroup and the negative HOLD condition. All prior unit
files and drop-ins must remain unchanged. The exact reviewed no-fork sentinel
is then attached while the original owner is still alive. HOLD is still absent.

Only `request_closure` can create this operation's owned HOLD. The accepted
native loop performs its own cleanup; the driver sends no native signal or
systemctl stop. Closure requires exact owner pidfd exit, its own unchanged
stopped identity, closed old lane births and a positive cgroup observation
containing only the retained sentinel. This is explicitly not an empty cgroup.

`capture`, `prepare` and `install` use the existing retained-session capture,
offline clone and origin installer. They preserve complete DB/WAL/receipt/
cursor/opaque-byte inputs and bind the prepared inode, original cursor bytes and
original row inventory. Unknown claims and callbacks stay unknown. Audit files
contain public identities and checksums; the private raw bundle stays in the
owned operation directory. Stage errors retain the native session and fleet
locks. Failed capture and prepare operations can continue only in that same
process with its original retained objects:

```json
{"action":"status"}
{"action":"retry_capture"}
{"action":"retry_prepare"}
```

Use the matching retry action only when `status` reports `capture-failed` or
`prepare-failed`. Retry revalidates the native closed population, original
session, HOLD/inhibition, runtime and retained capture. Native capture retains
and verifies its original flock/OFD descriptors and the first input inventory;
it never reacquires its own locks or replaces its initial input baseline.
Each attempt uses fresh absent output paths (`raw-capture-002`,
`inspection-copy-002` or `prepared-clone-002`), preserving all previous trees.
Attempts are bounded to 32 per stage. No retry is admitted after an install
attempt. A successful in-memory capture/preparation remains admitted if only
its later audit write fails; the status identifies the actual completed stage.

Error diagnostics include bounded exception/module names, a missing-module
name when present, and structural traceback frames. They omit exception
arguments, locals, source text and full filesystem paths. Captured pre-install
failures remain available in the controller's status even if an audit write
fails. Audit JSON cannot reconstruct any retained object or authorize a retry.

`finish` verifies the installed origin marker and retained raw bundle, closes
only the driver's exact sentinel by EOF, observes its normal exit, then releases
the session and fleet locks. HOLD, startup inhibition and the permanent native
profile marker remain. It does not infer an empty cgroup after the last actor
leaves or restore automatic restart.

After arming, input EOF keeps the controller alive with its handles rather than
silently abandoning custody. Keep the original interactive session available.
Terminating that controller loses its in-memory custody and cannot be repaired
by loading its JSON reports; retained markers and partial installation evidence
then require separate reconciliation.

Run ordinary boundary tests with:

```text
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  test/api/semantic_refactoring/test_spar_retained_capture_driver.py -q
```

The opt-in host tests create uniquely named disposable user units, retain real
pidfds/cgroup descriptors and attach the actual sentinel. They exercise normal
HOLD-driven exit, real DuckDB capture/install, changed-unit refusal and prepared
database rejection. Public task/source observations are isolated fixtures, so
these tests do not supply live SPAR native admission. Cleanup touches only each
test's unit and owned drop-ins, and verifies their final absence after exact
test actor exit.

```text
SPAR_RUN_HOST_CAPTURE_TEST=1 PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  test/api/semantic_refactoring/test_spar_retained_capture_driver_host.py -q
```

The exact isolated entry point and its explicit dependency binding are covered
by `test_spar_capture_runtime.py`. The additional opt-in
`test_spar_capture_isolated_host.py` invokes `/usr/bin/python3 -I` controllers
with no user-site path, uses the normal JSON command loop, and completes real
disposable host capture/prepare/install/finish after injected capture and
prepare failures. It verifies that the same queue FDs survive retry and the
sentinel exits normally only after installation. Native task/source/repair
observations remain explicit public fixtures; these tests grant no live SPAR
admission. Run them with the same host-test environment above.
