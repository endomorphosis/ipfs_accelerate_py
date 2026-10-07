# Finite schedule qualification — 2026-10-07

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Supervisor and symbolic-operator developers

**Sources:** [qualification.json](qualification.json) and [operator guide](../../FINITE_SCHEDULE_CONTRACTS.md)

**Last verified:** 2026-10-07

The local frozen-source qualification passed **819 checks**, with no skips,
failures or errors: 373 shared datasets checks and
446 supervisor/retrieval checks. These are authored integration
and regression tests, not a Terminal-Bench reward or token comparison.

## What ran

The shared `ipfs_datasets_py` operator compiles exact finite integer interval
constraints to QF_LIA and invokes the admitted Z3 backend. An independent
endpoint-sweep checker verifies witness feasibility. Tests exercise real
SAT/UNSAT and adversarial input, model, identity and receipt cases. The shared
run took 31.71 seconds wall time.

The supervisor run took 493.35 seconds wall time. It covers signed
Intent @5 admission and planning, immutable source binding, DuckDB/DuckLake
hydration, capability scoping, candidate dispatch and staged publication.
The public scheduling fixture builds lexical and semantic indexes before
planning, reuses them after admission and synthesizes a checked candidate
without a provider. Its canonical public validator remains a structural smoke
check; finite witness feasibility is separately checked.

The separate native scheduling fixture attaches semantic/world context, runs
START, claims the task, creates the output in an allocated worktree, validates
with an independently authored finite checker, publishes, completes, refreshes
context and runs STOP. It ends with zero tracked process members and no model
provider calls. Its test took 20.555 seconds.
It does not build initial vector retrieval or qualify an autoencoder checkpoint.

## Gap found and fixed

The first frozen supervisor run had 400 passes and one failure: the public
instruction shared no terms with indexed symbol names, so lexical preflight
raised `query has no vocabulary terms` before planning. That failed run is
retained and identified in the qualification JSON.

An exact zero cosine query now yields a complete, replayable empty result.
The complete AST/vector index and DuckLake links remain populated. Initial
context and publication refresh use that same rule. Source freshness, row
normalization, nonzero query normalization, query identity and vocabulary pins
remain enforced. Tests reject fabricated hits for a zero query and confirm
that changed source still produces stale context.

Earlier development runs also exposed another process changing the account-wide
proof resource scheduler configuration. Qualification uses isolated owned
`local-benchmark@1` ledgers, asserting no leaked leases or waiters. Production
configuration-drift refusal remains in place; no foreign ledger was reset.

## Evidence and limits

[qualification.json](qualification.json) records commands, counts, elapsed times,
source hashes, implementation commits and hashes of the retained logs/XML.
Both final runs assert unchanged sources during execution; the recorded source
bytes match the committed blobs. Full local artifacts are under
`artifacts/finite-schedule-20261007/qualification` in the parent workspace.

The optional Kit import came from the separately installed checkout recorded
in the JSON. This does not qualify the parent repository's different Kit pin.
GitHub Actions billing is independently blocked; local success does not claim
that required remote checks passed.

The scheduling interpretation is explicitly reviewed and authored. Evidence
establishes finite integer feasibility, not intent equivalence, a Lean/kernel
theorem, optimality or whole-task correctness. UTC/ICS, recurrence, timezones,
preemption and implicit preferences remain outside this profile. No new
Terminal-Bench score, suite completion or matched token-saving result is claimed.
