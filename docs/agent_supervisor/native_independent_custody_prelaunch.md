# Native independent-work prelaunch under retained custody

An unresolved Portal callback is not quiescent, even when its managed daemon
has exited. A complete native audit may exclude an exactly quarantined task
from independent task selection while continuing to report
`safe_to_restart=False`, `quiesced=False`, and `completion_authorized=False`.
The ordinary supervisor restart predicate remains unchanged.

The native retained-program controller has a separate launch path:

1. Hold the existing Quack owner mutation fence, then the exact managed-daemon
   launch lock. Preserve an already authenticated live child; otherwise require
   the existing managed-child custody check to prove process-tree cleanup.
2. Construct only this lane's native execution and coordination owner. Current
   central quarantine heads select the independent path instead of opening
   peer execution writers for historical recovery. Re-acknowledge this lane's
   retained state from its native records under the new process instance, then
   run the complete remaining-population audit.
3. Verify all current central heads against their immutable whole-root
   filesystem custody records. Malformed roots, failure to establish a current acknowledgement,
   changed execution/coordination snapshots, changed Quack owner binding, or
   incomplete audit population cannot grant launch authority.
4. Enter a private, process-local native launch scope. Bind the imported and
   current control-plane sources, exact command/environment, owner scope,
   lane database paths, sharding, and launch-lock identity. Revalidate before
   consumption. The scope expires after 30 seconds, cannot be serialized as a
   receipt, and can be consumed only once while its native context is active.
5. Close the controller-side execution writer before starting the child, while
   retaining both outer locks through the actual subprocess start. The new
   daemon must refresh its own process-bound acknowledgement and repeat its
   normal native startup audit before claiming or dispatching work.

Preflight may report independent work as admitted and skip ordinary quiescent
maintenance. That diagnostic is never accepted by the actual spawn gate. A
later spawn repeats the native procedure; copying an earlier status mapping
cannot authorize it. Source files are read through a bounded regular-file
reader, so a FIFO cannot stall the source gate before type validation.

The retained conflict and missing-nested-state cases keep their old attempts,
claims, phases, callbacks, leases, and workspaces. No inferred workspace
association, callback settlement, expiry release, replay, completion, merge,
or retry authority is introduced. Independent work continues through the
existing central task exclusion, DAG/CAS checks, resource claims, and fresh
workspace allocation.

The shared filesystem module includes the independently reviewed enclosing
Git-store locks from main commit
`740b5d3a46842db03b72ca63559232ab539c5be8`. Native code deliberately retains its
existing `task_sources.owner_task_quarantine` validation import so that
`QuarantineDenied` has one native identity. This single import line is the
only difference from that main shared module.

Qualification includes four simultaneous native lane writers, one-shot child
launch and successor acknowledgement, a real controller/Quack reconstruction
with both locks proved held at subprocess start, retained-row preservation,
source/command/owner/acknowledgement/custody substitution failures, refusal of
a diagnostic-only admission mapping, unknown-child refusal, and bounded FIFO
rejection. The generic disposable Quack owner does not carry the production
historical manifest pins; those remain separately exercised by the retained
recovery suites.

This source does not itself establish production deployment closure. Every
filesystem mutation owner sharing the relevant Git stores must run compatible
custody guards before the first freeze. Current source/bootstrap seals and the
native managed-process identities must independently admit each deployment.
Foreign execution databases are intentionally not reopened to manufacture an
acknowledgement or to attest another live process's imported source.
