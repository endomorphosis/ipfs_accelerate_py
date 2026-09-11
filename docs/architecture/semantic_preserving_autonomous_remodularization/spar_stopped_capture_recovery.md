# Fresh stopped-owner queue capture

An interrupted capture controller may retain every queue lock after its native
owner has closed. Its exported JSON is not a replacement for its in-memory live
capture capability. `spar_stopped_capture` defines a separate admission protocol
for that state. `CoherentLegacyCapture`, its schema, and the original installer
remain unchanged. The new capture is `spar/native-stopped-queue-capture@1`, its
installed origin is `spar/native-stopped-queue-origin@1`, and native launch uses
`native-stopped-capture@1`.

The protocol first binds the exact failed workflow controller and its exact
no-state helper by process birth, argv, retained pidfds, native cgroup inode,
HOLD bytes, loaded unit semantics, and all unit file bytes. The native owner and
complete broker lane population must be positively dead. Original bootstrap,
stopped owner status, native source forest, and configuration must remain exact.
Actual fdinfo must show the failed controller's six existing fleet/queue locks,
including the canonical queue OFD writer. The complete original queue is hashed
while those locks are retained. No old audit JSON supplies any of these facts.

A new helper runs only the fixed isolated Python sentinel program from `/`,
with three pipe descriptors and no state imports, credentials, owner grants, or
inherited descriptors. It joins the inhibited native unit before the failed
controller is retired. Both helpers must be observed together. Only then can
the exact failed workflow controller receive SIGTERM through its retained pidfd.
The native owner is never signaled or restarted. Positive exits of the failed
controller and old helper are required, while the new helper continuously keeps
the same cgroup populated. A timeout retains the new helper; `retire` can finish
observing the already-admitted retirement without sending another signal.
`abort_overlap` is available only before retirement and only while the original
controller, helper, and all original locks remain positively intact.

The new controller then acquires the existing fleet locks, four task-store
FLOCKs, three queue FLOCKs, and OFD writer fences on both canonical databases.
Every acquisition is nonblocking and checks exact owned regular inodes. The
complete queue must still equal its pre-handover file identities and digests.
A competing holder denies admission. A competing writer that entered and left
between the old and new lock lifetimes is detected by the inventory/digest
check. Failure retains inhibition and the keeper; it does not infer completion
or steal the competitor's locks.

Only under these fresh fences are the canonical task DB and optional WAL copied
to an immutable original directory and a separate inspection directory. The
accepted native source runs in a fresh `/usr/bin/python3 -I -B` process using an
explicit, hashed DuckDB runtime manifest. Eager native dependency probes run
before keeper overlap. The worker opens only the inspection copy, uses native
`IntentRepository`/`DatabaseTaskSource` with schema installation disabled, and
performs the accepted route and launch-source lineage checks without admitting
an amendment. Exact latest generation, stopped server, closed epoch, immutable
bootstrap/plan/task population, all available untruncated native closeout facts,
and an empty active/unknown lease population are required. Unknown non-lease
claims remain visible and unchanged. The worker never opens canonical state.

The full original queue, including opaque files and cursor source bytes, is
copied under those same retained fences. Its existing import planner qualifies
a separate prepared clone. The fresh receipt is written exclusively and fsynced;
its bytes cannot recreate the capture object. The distinct installer requires
the exact fresh producer object, checks all original and prepared content, and
preserves full original copies before canonical replacement. It writes a
separate durable `native-stopped-profile-required.json` marker. A missing or
invalid installed stopped origin, a conflicting legacy marker, or an explicit
wrong launch profile blocks native startup. Any installation failure retains
the marker and custody for independent recovery; there is no installation retry
or completion inference.

The retained command entry point is:

```
/usr/bin/python3 -I scripts/run_spar_stopped_capture.py \
  --request /absolute/reviewed-request.json --request-sha256 REVIEWED_SHA256
```

The request has schema `spar/stopped-capture-driver-request@1` and exact fields:
`repository_root`, `config_path`, `fleet_config`, `controller_birth`,
`helper_birth`, `controller_argv`, `owner_identity`, `hold_sha256`, `source_head`, `source_tree`,
`operation_root`, `runtime_manifest`, `runtime_manifest_sha256`, and
`runtime_helper_sha256`. All nominations are independently checked against
actual retained native and process facts. The request cannot grant task or store
authority. The driver emits one audit JSON response per command. Send
`{"command":"overlap"}`, `retire`, `fence`, `capture`, `prepare`, `install`, and
`finish` in that order. `status` observes retained state; `abort_overlap` is
restricted as above. Failed pre-install captures and preparations keep their
original output directories and may retry under the same actual fences with
new output paths. Changed task, queue, source, or runtime facts deny retry.

`finish` is admitted only after successful installation and exact origin marker
validation. It closes the controller's own locks and lets its own helper exit
normally, retaining the existing unit inhibition and HOLD. Source adoption,
native launch-source amendment CAS, owner startup, task settlement, callbacks,
signing, Git publication, and branch completion remain separate native actions.
A native source seal must include the new producer, consumer, runtime, and
entry points before any source transition.

Qualification separates actual kernel/user-systemd behavior from native
semantic authority. Disposable tests exercise keeper overlap/exit, no-state
helper death, real lock contention, capture retry, task WAL and queue drift,
opaque/cursor preservation, exact isolated runtime startup, fresh versus live
capability rejection, and distinct origin installation. A separately
materialized accepted-native board exercises real plan/task lineage and stopped
store observation under actual fences; fixture task facts are never presented
as production admission evidence.
