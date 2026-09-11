# Native SPAR source verification component

The SPAR launcher can execute a bounded datasets source verifier and expose its
current result through the existing authenticated closeout snapshot. This is
one independently executed source component. It never issues an accepted root,
completes a task or goal, settles a callback, or authorizes publication.

The first producer is disabled by default. In a separately qualified native
owner launch, `IPFS_ACCELERATE_SPAR_VERIFY_SOURCE=1` runs it after the existing kit
source-forest publication. It has no new RPC writer operation. Unavailability
is reported and does not stop an otherwise healthy owner. This change does not
itself authorize a restart, remove a maintenance hold or change a source pin.

The launcher constructs the request from its sealed bootstrap/profile, current
native task/goal/claim/receipt rows, current source forest and kit source CAS.
It records exact verifier module bytes, installed CID dependency bytes and
versions, interpreter bytes, and the current native owner incarnation. The
child receives sealed memfds, with no DB handle, status token or task credential.
The parent retains its actual pidfd, verifies the birth/boot/parent/executable
handshake, and observes exit before consuming its sealed result. The child
uses the datasets snapshot, scanner, semantic bundle producer and reader, and
required-source byte/span verifier; it does not execute the target program.

Each repository's first deterministic chunk contains at most eight Python
files, eight KiB per file and sixteen KiB total. The complete committed Git
inventory, selection, omissions, snapshot exclusions, opaque entries and source
limitations are retained in content records. This is explicitly partial source
coverage. There is currently no continuation producer for the remaining chunks.
Large semantic bundles remain expensive in the existing Merkle verifier, so
this component keeps the existing verification and a bounded execution deadline.

Kit binds a separate closed namespace to the *same* admitted DuckDB connection,
transaction lock and native identity. Its existing namespace isolation remains
intact. Immutable requests, semantic blocks, reports and execution records are
stored there, followed by a root CAS. Rechecks before execution, after execution
and after persistence prevent changed inputs from becoming a current result.
The status reader rechecks current source and native bindings without writing.
A content record alone cannot invent execution: admission also requires that
the current living producer retained and completed that exact child. A new owner
must execute again. Exact replay in the same incarnation returns the same record.

The following coverage is still unavailable and remains explicit:

- Complete required language and dynamic semantics, including unprocessed source.
- Required mode roots and transitions.
- Differential trace, selection and proof executions.
- Noncompensable safety floors with retained failure denominators.
- Self-hosted capstone and later procedure reuse without general LLM execution.
- The two unchanged epochs over all seven fixed-point slices.
- Native callback, lane and merge settlement.

Expected negative-vector labels such as `network_denied_general_llm` remain
specifications. They are not reclassified as failed positive executions, and
neither a nomination's success boolean nor a CID can supply missing coverage.
The independent accepted-root producer, SPAR goal CAS consumer, accepted final
report and final publication gate remain separate unfinished work.

Qualification uses the actual Quack owner lifecycle, migration, exclusive lease,
authenticated transport and independent verifier processes in new disposable
directories. Tests on tmpfs establish native connection, lock, process and IPC
behavior; they do not establish physical disk durability. No live board source,
owner, hold, queue, task, goal, unit or branch is modified by qualification.
