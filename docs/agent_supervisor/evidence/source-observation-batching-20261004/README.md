# Source observation component qualification

The final combined generation passes 207 integration controls and seven native
checkpoint/GTE Source384 controls. Additional span tests and the 26-case batch
review are retained. observation.json counts unique XML test identities and
reports repeated executions separately; overlapping tests are not added twice.
The earlier 67-test generation is retained separately and predates the
aggregate-memory correction.

Active AST batches reconstruct canonical payloads and verify every relation.
The later identity fence rereads blob, file and revision rows, detecting owner
invalidation or replacement during CAS work. It does not establish that no
relation-only raw SQL tampering occurred after the full read; such tampering is
rejected by the next full reconstruction. This is not a transactional snapshot
of the source filesystem and every database table through final publication.

Multi-CID reads exceeding the encoded-payload or query-row bounds raise an
explicit batch-limit error. The consumer releases that exception frame and
splits the request; it never retains an oversized scalar fallback tuple.
Singletons preserve existing scalar bounds. These are encoded-input and row
limits, not a hard Python or SQL RSS limit. Cancellation and deadline checks,
independent CAS verification, fresh source observation, and final head checks
remain mandatory.

Source extraction now reuses one immutable, exact-source-local line table.
Every function retains byte-map verification and isolated AST comparison.
The public Bottle pure-host profile preserves complete outputs and reduces
median extraction CPU from 0.400 to 0.201 seconds. That component measurement
does not establish a Docker speedup or recovery of the benchmark deadline.

All retained results are component controls. They grant no proof, source
semantics, production activation, full-task completion or token advantage.
