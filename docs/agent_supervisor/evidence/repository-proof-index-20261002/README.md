# Repository proof/cache local qualification — 2026-10-02

The final combined suite passed **120 tests**, with no skips or failures and four
existing multiprocessing fork warnings, in 80.63 seconds. These are local cache,
identity and discovery controls using real DuckDB and fresh Python processes.
This run did not execute a theorem prover, training, model-provider calls or a
supervisor daemon. It grants no proof, execution or completion authority.

The suite includes 17 existing Doctor tests, 14 new durable-state tests, 22
existing formal-cache tests, nine new pagination tests and 58 new cross-package
identity tests. [tests.log](tests.log) and [tests.xml](tests.xml) retain the exact
results; [summary.json](summary.json) records the counts and scope. The scoped
[source inventory](source-inventory.json) and [provenance](provenance.json) bind
the working sources and command. This is not a full imported-dependency closure.

The qualified changes retain Doctor keys, reverse roots, revocations, tombstones,
quarantines and receipt observations in the existing cache owner's database.
Restart, late key registration, concurrent conflicting receipts, invalidation
between lookup/store and return, malformed state and private-key refusal are
covered. The canonical bridge preserves all 16 datasets dimensions and explicit
supervisor execution context, while refusing marker-only proof admission.
Discovery scans retain cursors and omissions across more than 512 historical
rows and prevent truncated-key/source-basename collisions.

Remaining boundaries are explicit: Quack snapshot transport cannot provide this
adapter's atomic read/write/read transaction; it is refused. The canonical
identity bridge needs a separately verified obligation-linking owner before
positive cross-package proof reuse. Discovery pages are not a consistent
snapshot across calls, and automatic filename discovery retains its separate
16-file cap. No source watcher, distributed invalidation or full CodeEvidencePlane
join is qualified by these tests. All 32 production backlog criteria remain open.

The broader board-neighbor run had **19 passes and two failures**. Both failures
reproduced with the unchanged HEAD board module: configured scheduler arguments
omit `--board-namespace`, and sibling implementation daemons retain an empty
namespace. [board-discovery-diagnostic.log](board-discovery-diagnostic.log) and
[board-baseline-control.log](board-baseline-control.log) preserve those failures.
The baseline loader and original Git blob identity are retained for attribution;
they are historical diagnostics, not part of the final 120 passing cases.

[checksums.json](checksums.json) covers every retained evidence file except itself.
No database, checkpoint, source corpus or private runtime state is packaged.
