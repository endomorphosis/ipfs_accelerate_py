Cooperative database-daemon shutdown
===================================

Python signal handlers may run while a native DuckDB call is active. Raising
`SystemExit` from that handler unwinds into the engine, interrupts query or
connection cleanup, and can leave the process spinning with its writer lock and
WAL still held. The owner then cannot close the lane.

`DatabaseDaemonShutdown` is a process-local latch. `SIGTERM` and `SIGINT`
record one first-request signum in memory, including if a second signal arrives
during cleanup. They do not raise. Ordinary Python control flow calls
`checkpoint()` only after native calls return, at claim/provider/effect
boundaries and around the daemon loop. A completed, already-admitted callback
still records its outcome before the next boundary. The latch never settles
retained work, grants a new claim, or retries an unknown attempt.

Native graceful recovery still sends one `SIGTERM` and never escalates. With
this latch, that signal can complete the in-flight engine call and the existing
`close()` path, so the next stall of this class can exit without wedging the
owner. Historical SAWM-016 remains an unresolved UNKNOWN obligation; this path
does not refund, requeue or complete it.
