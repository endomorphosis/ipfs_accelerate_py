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
python3 -m scripts.ops.agent_supervisor.spar_retained_capture_driver \
  --repository-root <accepted-native-checkout> \
  --config-path <absolute-native-config> \
  --fleet-config <absolute-current-fleet-config> \
  --operation-root <absent-private-operation-directory> \
  --expected-source-commit <fresh-native-commit> \
  --expected-source-tree <fresh-native-tree>
```

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
locks; partially completed stages cannot be blindly repeated from audit JSON.

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
