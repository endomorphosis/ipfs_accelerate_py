# Diagnostics across database attempts

A fresh database attempt creates a new Portal log root and starts local attempt
1. The bridge can now attach bounded diagnostics from the exact prior database
attempt without importing local retry state or scanning sibling directories.

The reader requires the current owner-projected claim and its explicit
`claimed_from_revision` predecessor. Typed `database_attempt_admitted` receipts
must also match the immediately preceding reservation, process attestation and
admission revision. The failed retry receipt must reference the exact prior
claim/attempt/lease/fence identity. Claim, terminal and current task bodies must
retain the same immutable contract. The existing verified bridge binding fixes
the current revision, complete task-contract digest and repository tree; the
actual parsed Portal task payload must remain identical when rendering.
Historical projection files and receipt formats are unchanged.

Only known failure/review codes and integer return codes enter the prompt.
Unknown prose, code-shaped private values, paths, commands, prompt addenda and
acceptance fields are omitted. The text is explicitly diagnostic data. It cannot
settle callbacks, release claims, expand scope, increase retry limits or authorize
completion. A changed binding cannot be installed on a reused Portal daemon.

The typed adapter reuses the existing authenticated
`executor_task_revision_history_page` operation. It reads at most the latest 32
revisions, 256 KiB total, and checks owner generation before and after. The four
second budget is checked between RPCs; an in-flight RPC retains the transport's
existing timeout. A predecessor outside the window, head above 10,000 (the
existing offset admission bound), generation churn, unavailable read or prompt
budget shortage yields no feedback. It never falls back to complete history or
a local database. The legacy local adapter only uses a new standalone tracked
connection and refuses borrowed, remote and read-session handles before SQL.

Qualification includes real DuckDB and the authenticated production typed socket
with owner-issued executor grants, actual daemon reservation/admission and bridge
dispatch, a pre-effect deferral, and its successor's local attempt-1 prompt. The
TCP listener setup and external provider are test doubles; this does not qualify
a live native deployment or unlimited-history/high-volume availability. Historical
PCTDD diagnostic files alone do not establish owner receipt lineage, and missing
structured codes cannot be reconstructed from those files by this reader.
