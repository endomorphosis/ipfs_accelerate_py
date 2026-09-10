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
