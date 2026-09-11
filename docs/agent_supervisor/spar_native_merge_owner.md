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
repository, target, attempt root and all five native stages. Receipt imports
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
`native-legacy-capture@1` profile remains refused until coherent capture and
old-consumer closure have independently admitted native producers.

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
