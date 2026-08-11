# Planner/Doctor live paired benchmark contract

Status: PDR-003 preregistration. This document explains the machine-readable
policy in `config/agent_supervisor_planner_doctor_benchmark.json`; the JSON is
normative.

## Purpose and authority

The benchmark compares the current Planner/Doctor path with a deterministic
symbolic path and a symbolic-first, LLM-residual-only path. It measures real
service execution, not fixture-returned values. PDR-070 will implement the live
runner, PDR-071 the process-tree/provider/GPU telemetry, and PDR-072 the
independent oracle and external promotion corpus.

The tracked repository partition is deliberately classified
**public-conformance-only**. The source commit, tree, algorithm, seed, and CIDs
are public, so a candidate can reconstruct it. CIDs alone establish integrity;
they do not prove that a model or process never saw the content.
These cases can exercise the runner and detect regressions, but cannot promote
a candidate. Promotion remains disabled until PDR-072 supplies a separately
sealed external holdout and oracle.

The policy is self-sealing-forbidden. Its external operator receipt binds the
policy, this document, the focused test, the holdout manifest, the source tree,
partition roots, algorithm, counts, and oracle handle. That receipt grants
benchmark activation only. It grants no mutation, completion, task-status, or
promotion authority. Its operator-approved identity is independently pinned in
the protected scheduler configuration, so changing an artifact and recomputing
the receipt self-hash cannot authenticate a replacement receipt.

## Frozen corpus and overlap firewall

The corpus is every one of the 12,253 records returned by
`git ls-tree -rz --full-tree -r` at commit
`f25e5719cb738a50fb96bac4bea3f66ebca9800b`, tree
`cf6f6450bba0126a9846ae88bf7764f3e1ccb7c3`. Working-tree and untracked
content are excluded.

Records first enter a provenance-family connected component. Family edges are:

- identical Git object identity;
- identical Python AST after parsing UTF-8 and dumping without source
  attributes;
- identical non-empty public top-level function/class contract signature; and
- identical normalized provenance path after declared replica-prefix and
  backup-suffix normalization.

The family CID, rather than an individual path, is hashed with the frozen seed
and assigned modulo five. Buckets zero through three are development and
bucket four is heldout. Thus exact content, normalized AST, normalized
contract, or provenance-family overlap cannot straddle the roots. Any overlap
found by an independent rebuild rejects the entire corpus seal; it is never
silently moved, sampled, or removed after evaluation.

The resulting `RepositoryCorpusIndex@1` commitments are:

| Partition | Members | Families | Blobs | Gitlinks | Index CID |
| --- | ---: | ---: | ---: | ---: | --- |
| development | 9,899 | 8,106 | 9,890 | 9 | `baguqeerajbqnhanydnp2via2erbrg5fxb3j3v6es6pmxr2zpos6nbq2btzta` |
| public holdout | 2,354 | 2,032 | 2,353 | 1 | `baguqeerata5zmzkyuae3hb6flnop6eqxi3tlka2sm4ccwjkea45mdph3ijta` |

The tracked manifest contains commitments, selectors/generator handles, case
denominators, and oracle slot handles. It contains no gold outcome, oracle
body, prompt body, mutation body, or expected patch.

## Non-exposure protocol

The operator runner resolves an opaque case handle and materializes only that
case packet into an isolated arm worktree. It removes corpus-CAS access before
launching the candidate. After the candidate process tree has terminated,
capabilities have been revoked, and its output root is final, the judge may
mount heldout and oracle CAS read-only in a separate operator namespace outside
all candidate worktrees.

Promotion evidence therefore needs mount-table, process-tree termination,
network-denial, CAS-access-log, output-finalization, and post-output judge-mount
receipts. A CID match without these receipts is not non-exposure evidence.

## Paired inputs and arms

Each comparison unit is the same case, cache stratum, requested concurrency,
and repetition. Before execution, an operator-owned run manifest seals the
repository forest and overlay, rendered prompt/directive, mutation, task-source
revision, authority policy, IntentIR, SecurityIR, property catalog, toolchain,
provider, model, tokenizer, context limits, hardware, worker profile, cache
seed, deterministic seed, deadline, budget, and oracle manifest. Those values
are exactly equal across primary arms. Only the code root, Planner/Doctor mode,
and provider-call permission are treatments. Secrets are not hashed; the
credential capability class and stable account alias are bound instead.

The primary arms are:

1. current-mainline production baseline;
2. deterministic symbolic Planner/Doctor, with model and network calls
   forbidden; and
3. symbolic-first hybrid with only a bounded residual packet sent to the fixed
   provider/model.

One-factor diagnostic ablations disable AST/program graphs, knowledge graphs,
BM25/vector/GraphRAG nomination, logic/theorem provers, static formal analysis,
proof caches, or ZKP attestations. Ablations explain effects but cannot promote.

## Cache, concurrency, and denominators

Every arm receives isolated `cold`, `exact-warm`, `delta`, and `restart`
namespaces. Exact hits must rederive assurance. Delta runs bind parent and
mutation roots and invalidate the dependency closure. Restart runs use a fresh
process tree and replay native proofs; they cannot reuse memory state. Cache
sharing across arms, repetitions, or development/heldout partitions rejects
the run.

Requested concurrency is 1, 2, 4, and the configured bootstrap maximum 6.
Effective width is the minimum of request, configured maximum, admitted DAG
width, and resource admission. Requested, ready, admitted, and observed widths
remain distinct. A reduced effective width never rewrites the denominator, and
unequal resource admission requires a paired rerun.

There is one unscored priming run and three scored repetitions per required
cell. All 12 cases, four cache strata, four concurrency settings, and all three
primary arms are mandatory: 1,728 scored executions. Crashes, timeouts, and
cancellations remain failures in the denominator. Skips, outlier trimming, and
post-result exclusions are forbidden.

## Measurements

Clock/parallelism includes makespan, critical path, speedup, efficiency,
occupancy, queue p50/p95, widths, merge serialization, first useful
counterexample, and accepted criteria per hour.

Token efficiency uses provider-native input, output, reused, retry, and
cancelled tokens; call count; exact tokenizer identity; context bytes; cache
reuse; token/cost per accepted criterion and proved obligation; and
deterministic LLM avoidance.

Resource telemetry covers the entire descendant process tree: user/system CPU,
CPU-seconds, peak RSS, GiB-seconds, I/O, artifact growth, process count,
network, quota and provider cost. When an accelerator is present it also
records utilization, peak VRAM, and GPU-seconds. Energy is optional.

Quality covers plan validity and coverage, dependency accuracy, prediction
error and replan locality; Doctor precision/recall, localization, abstention,
analytical repair, convergence, recurrence, blast radius and rollback; and
independent tests, mutation/property/fuzz/differential/metamorphic checks,
proof coverage and kernel reconstruction, SecurityIR/IntentIR conformance,
compatibility, patch minimality, flakes, and post-merge regression.

Every sample is either `measured` with value/unit/sensor receipt or
`unavailable` with a reason. Unavailable is never encoded as numeric zero.
A measured zero still needs a sensor receipt. Required telemetry loss blocks
promotion; GPU telemetry is conditional on accelerator presence and provider
token telemetry is conditional on a provider call.

## Admission and promotion order

Evaluation is fail-closed and ordered:

1. validate the external activation seal, frozen pair, real execution receipt
   chain, exact denominators, and required telemetry;
2. require every raw non-compensable safety count to equal zero, including
   authority/policy/scope/secret/path escapes, stale cache or forged CID/proof
   admission, missed consumers/frontiers, SecurityIR/IntentIR misses, hidden
   oracle/benchmark mutation, partial transaction, false fixed point,
   rollback failure, false completion, and synthetic/skipped promotion input;
3. require paired quality non-inferiority with preregistered zero margins and
   independent oracle evidence; then
4. compare the Pareto resource frontier. A candidate must be non-dominated and
   materially improve at least one preregistered clock, token, CPU, memory, or
   cost metric.

No throughput, token, cost, or resource improvement compensates for a safety
floor or quality failure. Fixture fields, candidate self-reports, task status,
synthetic runs, dry runs, mocks, xfails, and skipped checks have no promotion
authority. Automatic promotion remains false even after a passing comparison.

## Budgets and stops

The JSON fixes per-case and qualifying-run wall, CPU, RSS, GPU, VRAM, I/O,
artifact, network, token, provider-cost, model-call, and process ceilings.
Accounting includes priming, retries, cancellation, failures, oracle work, and
telemetry. Budget exhaustion cancels and observes every descendant, seals
partial receipts, and rejects promotion.

Safety violations, leakage, protected-artifact mutation, forged evidence,
required telemetry/oracle loss, rollback/restoration failure, budget
exhaustion, or the operator kill switch stop immediately. Incomplete cells,
quality regression, lack of material Pareto improvement, mismatched admission,
or two identical residual retries stop without promotion. Early-success
stopping and sequential result peeking are forbidden.

To change any case, root, input binding, denominator, margin, budget, or stop
rule, issue a new policy version, obtain a new external operator seal, and run
a fresh baseline. The unattended controller cannot rewrite this seed contract
or its oracle to make itself pass.
