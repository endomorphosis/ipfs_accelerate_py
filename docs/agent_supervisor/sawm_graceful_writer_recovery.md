SAWM native writer-loss recovery
===============================

`writer-recovery-inspect` and `writer-recovery-close` are narrow M70 operator
commands for a verified missing canonical writer lock. The native configuration
is admitted by the existing immutable M70 materialization gate. An expected
manifest binds current Git HEAD/tree/index, configuration, owner generation and
identity, exact process births/argv/cwd/UID and independently owned sessions,
PID markers, canonical inode and both native FLOCKs. A matching canonical POSIX
WRITE lock refuses this operation; inaccessible namespaces remain unknown.
The observation opens no database and reads no credential.

The caller first establishes the configured SAWM HOLD and an inactive global
repair-job boundary. The dispatcher must remain excluded by the operational
caller throughout recovery and subsequent native startup. HOLD does not stop
an already running repair job. These commands do not create HOLD or stop a
service. Every effect rechecks these inputs, current source and owner identity.

The recovery holds each native lane launch fence and uses retained pidfds. It
suspends every controller thread, then every wrapper thread, closes each daemon
with TERM, and checks for remaining descendants. It queues wrapper TERM before
releasing the launch fence and resuming that wrapper. Only after all known
children exit and the scoped marker/process census passes does it TERM and
resume the controller. Every exception attempts CONT for the exact processes
this invocation suspended. Timeouts preserve partial results and never escalate.
The process census includes independently owned groups/sessions and direct
kernel parentage, so changing argv/cwd or reparenting cannot hide an observed
member of those scopes. Unobservable unrelated processes are reported; there is
no claim of global visibility or containment of escaped descendants.

An exclusive nofollow journal preserves each completed observation. The owner
receives one TERM only after lane/controller closure. Success additionally
requires the actual native stopped identity, preserved canonical inode, absence
of its store locks and a refused loopback endpoint. A failure after process exit
leaves an incomplete recovery report; it does not manufacture a stopped receipt.
The native wrapper's existing shutdown reconciliation remains unchanged. This
command does not settle a task, callback, claim, worktree or provider obligation.

For old sealed processes, merely editing source cannot load this command. A
separately reviewed external bootstrap may import the exact qualified mechanism
and invoke the old source's own M70 admission without changing REPO_ROOT. It must
apply the same manifest/journal/exclusion gates to the actual old root. Source
publication and a subsequent Git transition remain separate caller operations.

After independently observed closure and reviewed source transition, the next
native owner command is the existing `quack-start`. Its M70 gate admits
`admitted_stopped_generation_48_reuse`, preserving generation 48 and the UUID;
it does not create a generation-49 authority. Once native owner readiness is
established, the existing `preflight` and `launch` paths perform
current source, configuration, plan and credential admission. A refusal at any
stage stops the continuation and preserves the original task/callback history.

Qualification includes exact native process-method tests with disposable
children, public operator routing with the real M70 configuration, and old25f
native DuckDB/daemon/Portal-wrapper/owner shutdown and same-generation reuse.
The storage fixture preserves an accepted running attempt, missing terminal
provider return, a leased dirty worktree, and all existing execution/coordination
rows. Its extension transport is substituted, so that test does not claim live
sealed-extension or current live plan admission. The full native launch gate is
still required by the operational caller.

The separately qualified phased mechanism suspends the exact controller, all
wrappers and all daemons before its population barrier. Each wrapper retains its
native launch fence. Already spawned helper processes receive no signals or
name-based exemption: the strict census waits only for observed disappearance.
A helper waiting for its paused parent causes a bounded refusal and CONT of all
surviving actors. Daemon TERM/CONT requests precede daemon exit waits; all daemon
and helper closure must be observed before wrapper TERM. Wrapper stop flags are
queued while suspended, then launch fences are released before wrapper CONT so
native cleanup can acquire them. The original sequential helper remains intact.

A scope refusal now retains bounded PID/birth/parent/group/session/cwd and argv
hash evidence. Identity drift remains explicitly unverified. The operator emits
that evidence, and the external bootstrap writes a private failure artifact
only after mechanism finally has attempted CONT. None of these observations
settles a callback, releases a claim or authorizes source adoption by itself.
