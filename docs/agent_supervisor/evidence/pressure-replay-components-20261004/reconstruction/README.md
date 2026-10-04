# Fresh AST observation reconstruction evidence

The final focused run executed **18 tests: 18 passed, zero skipped**. Native SQL reads still reconstruct and verify the complete AST projection and relations. The optimized observation reads and verifies fresh CAS bytes, joins them by exact canonical payload identity, and retains all provenance/currentness fences while avoiding a second discarded ASTRecord.

Public Bottle contains one 3,590,561-byte AST payload in this diagnostic. The archived pre-change observe_current implementation constructed two records; the final native implementation constructed one. All six observations returned the same manifest CID and digest. Separate untraced alternating CPU observations were about 17.7% lower in this bounded local component diagnostic. This is not a Terminal-Bench score or an admission-memory result.

The historical loader remains unchanged. Eighteen controls cover Unicode equivalence, bounded CAS corruption and tampering after SQL reconstruction, invalidation/source/producer races, custom and memory stores/loaders, and schema wrappers installed before module import. The final SQL identity fence retains its existing blob/file/revision scope; it does not claim a new atomic snapshot for relation-only raw SQL tampering after the complete initial relation audit.

The two earlier nonqualifying diagnostics are retained separately. focused-01 had two fixture API errors. The first Bottle profile used default 64KiB capture and therefore exercised an opaque entry with zero ASTs; its timings and allocation numbers do not support the AST optimization. The corrected profile explicitly used an isolated 4-entry/256KiB capture and asserted one AST. Optional corrected tracemalloc was skipped by the bounded-time budget; no RSS/cgroup reduction or gate recovery is asserted.

Only public source, source snapshots, exact command/exit records and component receipts are included. Private local stores, keys, credentials, model blobs and hidden benchmark bodies are excluded. Parent-owned combined/native qualification is separate.
