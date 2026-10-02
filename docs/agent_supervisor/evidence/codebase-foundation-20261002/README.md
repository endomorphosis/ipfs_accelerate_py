# Codebase source foundation and bounded staging qualification

The selected source/catalog implementation is integrated into the isolated
release checkout. The current foundation test run passes **203 cases**, the
bounded CAS dependency suite passes **13**, and the additive staging suite
passes **23**: **239 distinct cases**. Earlier runs are not added to this count.

The tests use actual file-backed DuckDB and Git capture. They cover dirty source
at unchanged Git HEAD, staged/untracked/deleted files, parser-context identity,
independent worktree views, immutable history, native process restart, atomic
head/AST publication, ABA generations, corruption, cancellation and exact retry.
The staging tests additionally cover oversized/opaque inventory, source that
must never execute, Unicode/CRLF, failed parses, lost responses, rollback,
changed/deleted cached artifacts, producer drift, classifier-owner pinning, and cache eviction.

A separate native fixture stages **320 Python source files in 20 deterministic
shards of 16 units**, closing/reopening DuckDB after 128 units. Both completed
runs leave the current CodebaseHead and active AST generation unchanged.

| Run | Elapsed | Units/s | Peak owner RSS |
| --- | ---: | ---: | ---: |
| Initial implementation | 670.197 s | 0.477 | 189,404 KiB |
| Final implementation | 26.821 s | 11.931 | 157,996 KiB |

The observed speedup is **24.99× in this paired fixture**. The initial loop
recomputed the entire sealed manifest root for every AST record. The final loop
uses the already verified root and a bounded parsed inventory/AST cache. Every
cache hit still rereads and compares exact CAS bytes and checks current producer
identities; fresh processes fully reconstruct the typed records. The final run
records two manifest parses and fresh AST reconstruction after the reopen.
The earlier completed optimized run is retained as `staging-optimized-previous-*`.
Final qualification additionally pins the direct classifier, malformed-path and
canonicalization owners; a warm-cache regression rejects classifier-owner drift.
The timing samples do not control concurrent host load or establish a distribution.
A cache-only intermediate run was interrupted to fix the repeated root
recomputation; its diagnostic, implementation and log are retained and excluded
from the timing comparison. Its resource journal ended with no live leases or
waiters.

These are structural staging and correctness controls. They execute no target
source, autoencoder training/inference, LLM provider, or theorem backend. This
fixture uses a deterministic healthy admission sampler; it does not qualify
external pressure or hard whole-owner memory limits. It supplies no Terminal-Bench
reward, task token score, generic scaling claim, global dependency closure,
proof authority, or production admission.

The profile remains clean committed source only. Dirty-overlay paging, global
cross-shard relation resolution, inference shards, model residency, DuckLake
publication and promotion to a complete current source/proof head remain open.
All 32 broader RPI production criteria remain open in the
[backlog ledger](../../../architecture/repository_proof_index_backlog_status.json).

The [manifest](manifest.json) binds the retained logs, JUnit results, selected
source port hashes, completed trials, intermediate diagnostic and exact drivers.
The drivers are retained execution artifacts with machine-specific fresh output
paths; they are not installed service entrypoints.
