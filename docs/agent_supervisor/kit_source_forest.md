# Native kit source-forest persistence

The semantic-preserving remodularization launcher publishes its current Git
source forest through the kit `DuckDBCoordinationStore` bound to the existing
exclusive native owner. It opens no additional database or mutable file store.
The ordinary `authoritative-status` command exposes the admitted component under
`completion_profile.kit_source_forest_persistence`.

The launcher first verifies its sealed task population and bootstrap profile.
It loads only the selected kit coordination modules, requiring their captured
bytes to match the kit commit in the current source forest. The store checks the
current database UUID, generation, fence, process birth, server and namespace.
Immutable manifest and transition bytes remain in the owner's existing DuckDB.
The kit CAS transaction atomically publishes one successor from an exact root
and revision; stale writers and changed idempotency requests remain conflicts.
The producer reuses the exact kit transition validator shared with the semantic
state adapter. It does not construct missing datasets or semantic evidence.

Native status reads do not publish a root. They re-read the stored manifest and
transition inside the existing owner snapshot, compare the exact current source
forest and profile bindings, and expose the transition CID and root revision.
A source change makes an older component unavailable until the launcher performs
a new qualified publication. Repeated publication of the current forest replays
the same operation. Ordinary owner restart preserves the database and namespace;
the new generation must pass native binding checks before accessing the record.

This is acceptance of the kit persistence component only. It does not accept
semantic roots, settle goals, authorize reports, qualify loaded interpreter
bytes, certify runtime or queue settlement, or authorize Git publication. The
existing datasets, required-mode, safety-floor, capstone, goal, source and merge
gates remain required. Nomination-only reports retain their negative status.

To deploy an updated kit producer, publish tested kit code, select its exact
GitHub commit as the parent gitlink, and requalify the parent source through the
existing stopped-owner launch amendment. The native launcher then runs the
producer automatically after sealed status admission. A missing or rejected
producer remains visible as an unresolved component while the retained owner
continues serving its read-only closeout status.
