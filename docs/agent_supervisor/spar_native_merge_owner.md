# SPAR retained queue owner bootstrap

The native SPAR queue role is separate from the task owner. It serves the
existing legacy queue and its durable recovery cursors and receipts through
the same retained DuckDB handle and exact Quack owner identity. Task tables
stay empty in this role; startup does not run task completion repair or legacy
queue purging.

`scripts.ops.agent_supervisor.spar_merge_owner.prepare_offline_clone` accepts an
explicit offline directory, a complete `spar/legacy-queue-offline-bundle@1`
manifest, and a new private destination. The manifest names the database and
optional matching WAL, every file's size and SHA256, repository/target/store,
declared source IDs, queue policy, admitted scope bindings, and explicit cursor
and receipt imports. It rejects unlisted files, unsupported schemas, missing
import mappings, symlinks, nonregular files, observed input kernel locks and
changing inputs. Kernel lock absence is only a diagnostic veto; it cannot
establish prior consumer closure.

The clone replays any supplied WAL before its logical baseline is recorded.
Native full-schema installation is allowed only when no owner schema exists.
An existing partial schema or malformed UUID/generation is refused. Existing
legacy and foreign rows remain exact; only enumerated owner/recovery relations
may append metadata. New domain relations must be empty. The inventory uses
deterministically ordered bounded pages because this native DuckDB wrapper
does not expose streaming cursors. Its aggregate bound is one million rows
and 64 MiB of encoded row digests across all tables.

Cursor imports require the exact native path
`train/post-merge-recovery-cursors/<binding-sha256>.json`, content identity,
repository, target, attempt root and all eight native stages. Receipt imports
require explicit logical keys, content IDs and complete contiguous versions.
The canonical `train/distributed-publications.json` must explicitly import to
`distributed-publications`, the key read by the owner-backed train. It must
remain the current receipt head; archiving it without importing it, mapping it
to an unrelated key, or superseding it with another supplied snapshot is denied
before clone creation. Prior publication IDs, task fences and duplicate history
therefore remain visible after migration and owner restart. This preservation
check does not validate publication authority or close prior callbacks.
No filename is treated as authority to reconstruct an unknown receipt key.

`start_queue_owner` starts a prepared role and binds both typed services after
their explicit schema/import admission. `qualify_offline_bundle` rehearses
two generations on a disposable clone, reads admitted scopes and historical
receipts through exact-peer typed grants, checks real POSIX writer exclusion
before and after checkpoint, and closes both generations. A failed final check
keeps `qualified=false`. The supplied source IDs are declared inputs, not
source acceptance; the result explicitly sets `source_admission`,
`live_custody_qualified`, and `completion_authority` to false.

The operator must independently bind a coherent capture and prior native
consumer closure to the current admitted launch before handing queue/recovery
credentials to workers. Offline rehearsal does not authorize live migration,
dispatch, source adoption, claim recovery or completion. Unknown processing
claims and existing receipt/cursor evidence remain preserved. The subsequent
versioned broker/daemon handoff supplies the actual native factory; it is a
separate source component from this offline role.

## Fresh-origin production handoff

The native operator has an explicit `supervise --merge-owner-profile
native-fresh-origin@1` path. It runs the existing clean-source, sealed-board
preflight and current `LaunchSourceAmendment` admission before creating either
owner or granting workers access. It initializes only an absent private
configured queue directory. An existing directory or legacy database without
the canonical `legacy_merge_native_origins` record is refused, even if the
queue is empty or a caller supplies a JSON origin/closure claim. The separate
`native-legacy-capture@1` profile requires the installed native migrated origin
described below. A directory, empty queue, audit report or caller closure flag
cannot supply that origin.

Fresh initialization creates the empty legacy queue schema and a new owner
UUID, then records its exact namespace, immutable creation manifest and content
identity in that database. Restarts retain this UUID and origin, validate the
current repository/target/store/path and all five-field recovery scopes, and
refuse an observed active database lock before opening the file. Creation
source head/tree remain historical metadata. They do not authenticate current
source or pin all future restarts to the initial commit: current source
acceptance comes from the existing operator amendment gate. The schema helper
requires this admitted caller; it is not a remote admission or migration API.
A changed config or plan scope requires explicit migration, never an empty
cursor reset.

The controller retains two independent Quack servers with distinct database
UUIDs, stores, sockets, ports, and owner generations. Its existing inherited
listener admits the exact daemon process birth and controller/supervisor/daemon
lineage. A versioned response supplies three different credentials: the
unchanged task-only grant, the exact three-scope queue grant, and the exact
four-scope recovery grant. No queue/recovery credential enters an environment
variable, CLI argument, file or diagnostic. The daemon hardens its process
before receiving the response, verifies the controller kernel peer and both
owner identities, opens the three typed connections, and passes the queue and
recovery pair plus its currently admitted config/plan CIDs into the actual
`bind_database_portal_execution_from_args` factory call.

The client may retry one lost bootstrap response with its original nonce. The
controller returns only the same still-current three grants for that exact
birth and both unchanged owners; revoked grants or a different nonce cannot
revive the bundle. Partial grant issuance is revoked before any response.
Malformed/foreign requests produce bounded rejection and do not restart the
owner. Closing daemon client streams does not release a retained consumer
lease. Unknown prior-generation leases, claims, callback history and all goal
acceptance predicates remain governed by the existing owner recovery API.
There is no filesystem queue fallback and no timeout-only lease settlement.

Qualification includes two actual Quack transports under one controller PID,
real typed sockets and distinct identities, writer-lock survival through
checkpoint, actual supervisor/daemon child lineage, lost-response replay,
malformed-request isolation, and the actual factory recovery call. These are
disposable tests. This source path does not authorize adoption of SPAR's current
legacy queue or claim that any live callback has closed.

## Native import mapping producer

`scripts.ops.agent_supervisor.spar_legacy_import_plan.produce_offline_import_plan`
accepts an explicit offline queue directory, a new private inspection directory,
and the declared repository, target, store, source and recovery scopes. It
preserves every input file and verifies the file population and byte identities
before opening only the new inspection copy with owner discovery disabled.
Committed WAL is replayed in that copy. No input database handle, live owner
credential, or signing-key API is opened.

The producer joins each supported canonical train receipt to its preserved
request ID, task, canonical identity, full candidate object ID and exact target
metadata. Primary receipt keys come from the native train's forward key
function. The literal `quarantine-<request_id>` producer is also supported.
Unknown key forms, orphan receipts, mismatched coordinates and ambiguous scopes
are refused; the tool never guesses a key by removing filename characters.
The canonical publication ledger and complete eight-stage recovery cursors retain
their existing native keys and scopes. Other files, including opaque private
material, are preserved without interpretation or use for signing.

The returned manifest applies to the original supplied offline bundle, not the
inspection copy whose WAL may have been replayed. It can feed the existing
`prepare_offline_clone` importer. Revision one denotes the first imported
snapshot in the new recovery store; it does not invent overwritten historical
versions. Logical row inventory remains exact, including processing claims and
unknown callback evidence.

This producer establishes a mapping of preserved bytes, not original capture
coherence, signature provenance, prior consumer closure, callback settlement,
current source acceptance or completion. Those result fields remain false.
An import plan by itself does not open either native legacy launch gate.

## Retained legacy capture and successor origin

`RetainedNativeLegacySession` in `spar_legacy_capture.py` binds the reviewed
current source/config/forest, actual service/MainPID, authenticated native
closeout snapshot, and all six admitted supervisor/daemon births before the
owner exits. Native observation runs in a fresh interpreter rooted in the
accepted checkout, so candidate imports cannot substitute native source.
The status observation has a separate bounded digest budget; it is not a
queue receipt. Namespace permission failures remain explicitly unknown.

The session opens a pidfd for the exact owner and retains the original cgroup
descriptor. It requires every admitted lane birth to be dead and a complete
native snapshot with no active task lease. These observations establish the
drained native actor set; they do not settle queue claims or callback effects.

The separately reviewed operator workflow is:

1. Arm the session with the expected current commit and tree, while the native
   owner is still ready. Keep the same session alive through installation.
2. Install a unit drop-in with `Restart=no`, `SendSIGKILL=no`,
   `TimeoutStopSec=infinity`, `RefuseManualStart=yes`, `Delegate=yes`,
   `ExitType=cgroup`, and `ConditionPathExists=!<runtime>/HOLD`. Reload its
   configuration and verify the effective properties. The hold is still absent.
3. Call `retain_workflow_sentinel()`. It starts one fixed Python `-I -S` child
   at `/`, with only its private standard streams and a credential-free fixed
   environment. The child prints readiness and waits on stdin. It cannot
   dispatch work, fork, access repository state, or receive owner credentials.
   The manager attaches only that exact retained child to the native unit.
4. Place the native hold. The existing retained-closeout handler exits
   gracefully. Require actual owner pidfd exit, its own unchanged `stopped`
   status, all admitted lane births dead, and a cgroup containing exactly the
   known workflow sentinel. `cgroup_empty_observed` remains false in this
   contract. An absent cgroup is never substituted for a population read.
5. Use `session.capture(destination=..., inspection_destination=...)` to copy
   the complete configured queue tree, including DB/WAL, receipts/cursors,
   processing claims and opaque private bytes. The session derives repository,
   target, store, source, config, sealed plan and lane scopes itself. It holds
   the consumer/queue flocks and an OFD database writer lock throughout; reading
   and closing copied file descriptors cannot silently release that lock.
6. Build a distinct `prepare_offline_clone` using the captured manifest, then
   call `spar_legacy_origin.install_captured_queue(captured, prepared)` while
   retaining the same closure and inhibition. Every original byte remains in
   the complete durable capture. Validation binds the candidate before its
   connection closes and verifies the staged copy again before replacement.
7. Close the session after the held transaction. Only its own sentinel receives
   EOF; its normal exit and exact pidfd exit are recorded. Source adoption,
   complete runtime/Git/worktree preservation, current launch amendment
   admission and the later supervised start remain separate operator steps.

The installer writes a permanent `native-legacy-profile-required.json` marker
before its first canonical effect. An interrupted transaction retains this
requirement and the original capture; it does not roll back, reset leases, or
fall back to the filesystem queue. Normal native startup selects the required
legacy profile and loads the canonical database origin, checking UUID, current
repository/target/store/path/config/plan/lane coordinates and exact imported
receipt/cursor population. The original source head remains history; the
existing current-source amendment gate still controls each launch.

The capture receipt distinguishes native process closure from unknown callback
outcomes. All old queue rows, claim tokens, generations, consumer IDs and
receipt bodies are preserved. Owner-backed train recovery has no abandoned
claim takeover method; migration cannot release an old processing claim.
Original private bytes are copied only within the explicitly authorized queue
bundle. No live vault is inspected, signing key generated, or signing authority
claimed. Native goal/root acceptance and board completion remain unchanged.

Disposable host qualification on systemd 255 confirmed that ordinary exit and
`RemainAfterExit=yes` both race cgroup removal. The retained sentinel contract
was stable in three actual user-service trials, including an independently
qualified update of a running service through its own temporary drop-in.
The deployed native closeout handler also returned `stopped` after exact-pidfd
SIGTERM with a progress-only status fixture. These tests changed only their
uniquely named disposable units; they are not evidence of SPAR live adoption.

### Captured scope to current launch configuration

A native legacy origin remains bound to the captured configuration and its
original cursor files. A current dependency seal changes the launch configuration
CID; treating that as an empty recovery scope would lose the retained positions.
The native entry now permits one explicit transition when the only configuration
change is `dependency_seal_cid`. The current typed launch amendment must already
be admitted by the materializer. The transition verifies the captured Git source,
current committed files and seal, unchanged immutable bootstrap inputs, and every
old/new lane, attempt root, plan and repository coordinate.

The separate queue owner imports the original scopes first. Before issuing any
worker grants it creates the new scopes from the exact retained initial cursor
values and appends a versioned transition receipt. Original scope rows, cursor
history, claims, callback receipts and the captured origin remain unchanged.
Restart replay preserves later successor cursor heads. An advanced original
scope, a second configuration transition, reverting to the captured configuration,
or any change beyond the dependency seal requires a separate admission; none can
silently reseed positions. This operation grants no signing, callback settlement,
semantic completion or Git integration authority.

The current rendered seal/configuration are forward source artifacts. The
original bootstrap receipt and its Git-addressed configuration, board, plan,
objectives and validator remain historical inputs and are never regenerated to
make the new source pass.

## Generic integration and deployed cursor profiles

The generic runtime retains its eight-stage recovery schema, including
`false_completed_requests`, `false_pending_requests`, and
`false_processing_requests`. The earlier deployed SPAR runtime and its
captured origins use five stages. A valid old cursor content identity does
not authorize padding these three stages or restarting their scans. Both the
native import producer and owner provisioner require the current exact stage
population and reject the older profile while preserving its source bytes.
An existing five-stage deployment must stay on its qualified runtime until a
separate state/source transition is implemented and independently qualified.
This integration alone does not admit such a transition.

Generic replica publication retains main's canonical directory and database
anchors and copies via the retained descriptor. The isolated copier remains
available for older native lineage callers and explicit offline qualification;
its presence does not replace the stronger namespace checks in the generic
owner. Typed recovery also retains session-aware grant revalidation under the
owner transaction lock.
