# Host worker diagnostic

A fresh bounded process consumed the unchanged SecurityIR checkpoint and pinned
GTE assets, using 128 normalized public-source rows retained from the successful
container capture. One model load produced 127 unverified candidates; the
overlength row remained deferred. The bounded process completed in 18.581s;
the surrounding controller interval was 18.664s, and workspace cleanup passed.

The runner adopted the existing shared scheduler configuration without changing
its capacity or another owner's reservation. Its own 4096-MiB, one-CPU/process
request fits beside the existing reservation. A separate full-context attempt
was refused before work because its detected configuration differed from that
active ledger; it was not retried with a smaller source-context envelope.

This is an instrumented host worker-only run with host package versions, not
the deployed container runtime or a live codebase observation. cProfile adds
overhead; cumulative function times overlap and are not additive stage times.
There is no source proof, successful full task or container performance claim.
The initial diagnostic runner's missing cancellation argument was corrected
before the successful run; its failure hash is retained separately.
