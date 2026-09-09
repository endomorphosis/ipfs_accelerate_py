# Retained attempt dispatch barrier

A running attempt belonging to the logical owner remains a dispatch barrier when a changed prefix or shard selection excludes it. Missing process metadata, denied process inspection, a dead PID, or a closed daemon record cannot settle the attempt or prove that provider effects are absent.

The daemon first reconciles prepared completions and expired coordination claims across the owner's full attempt population. Only a native terminal transition removes a running attempt from that population. The expiry pass does not dispatch new provider work; a later pass may dispatch once no retained running attempt blocks it. Durable provider or effect evidence continues through the existing blocked or completion reconciliation paths.

Native DuckDB regression coverage verifies the unchanged shared task, claim, and attempt across repeated passes for missing metadata, malformed birth records, denied inspection, closed or dead processes, PID reuse, and a live process. A positive case verifies exact lease expiry, terminalization, and later dispatch in separate passes.

Recovery passes preserve the same fences for every task alias. A shared no-provider fence or an origin-lane recovery saga ends the recovery pass before generic retry eligibility is considered. A failed terminal candidate remains quarantined from generic rearm. A changed process identity, an extra-gate alias, or a missing-workspace path string does not establish a provider-free predecessor or refund an exhausted attempt budget. Unknown durable callbacks remain blocked until the existing native recovery API admits exact evidence. Native DuckDB regression tests assert that task status, revision, and receipt remain unchanged when those proofs are absent.

Qualified daemon source updates use the supervisor's existing source reload gate. It holds the owner mutation fence across an admitted idle projection, exact managed-child quiescence, and a second projection. Active providers defer reload; source maintenance must not turn a process identity or a cached status into permission to stop their effects.
