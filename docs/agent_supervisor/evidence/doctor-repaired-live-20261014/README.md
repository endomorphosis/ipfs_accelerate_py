# Fresh diagnostic run after Doctor observation repairs

This full indexed `largest-eigenval` run used the repaired runtime at
`3ee0d2901b9b61234ebb3efb150513e505ada6ac`, Grok 4.7 through `llm_router`,
the eight-turn coding limit, and `doctor-coding-two-attempts@1`. It retained the
original 1,800-second shared coding deadline, identity source view, disabled
mapped edits, and 5-CPU/20-GiB resource profile. The separate numerical-obligation
broker remained unset. Only the runtime archive and output directory changed
from the earlier eight-turn run. This is a diagnostic follow-up, not a matched
same-runtime savings experiment or a native Codex comparison.

**Official reward: 0.** Eigenpair correctness and dominance passed; speed failed.
Native supervisor completion was false. Supervisor time was 429.84 seconds;
the full benchmark execute command took 585.13 seconds.

The [result summary](result-summary.json) and [frozen reader output](readout-final.json)
show both coding children started, exited 1, and were waited and retired. Only
the first produced a coding router receipt. The observed **22,214 tokens cover
planning only**. Final coding usage, complete native usage and billing totals
remain unknown. No token savings or successful two-model-call continuation
was established.

## What the repaired run established

The Doctor input audit now reports `verified_all_observed_model_inputs` for the
one usable coding receipt. This is the retained in-container comparison, not
an independent host replay or evidence that the unreceipted child had valid
input. The earlier two runs remain historically unknown.

The new bridge observation retains two distinct, closed, **unattributed** errors:
one in `provider_invocation`, and a `ValueError` in `candidate_dispatch_context`.
The latter reaches line 268 of the frozen runtime's
`candidate_dispatch_context.py`, the handoff lifetime/current-clock/original-
deadline predicate. The preceding launch-settings check was passed on that
path. The report narrows the failing gate but does not retain the timestamps
needed to identify the exact failed conjunct. It does not authenticate which
child authored each log record.

Grok still reported `max_turns_reached` followed by one `end`, then exited 1.
The stream remained tainted and final coding usage was unavailable. There were
no malformed, oversized or unknown frames in the bounded progress record.
Eight usage-event observations survived, but their counter semantics remain
unverified and summation is forbidden. The original `end` payload was not retained,
so the run cannot distinguish a rejected field, stop-reason mismatch or another
terminal consistency failure. The preceding parser repair is qualified for its
authored protocol fixtures; this live sequence remains unresolved.

## Preparation and evidence

The runtime archive SHA-256 was
`fcc74dd9af41e6b5c81c39e7aaa6bc97c24141b588157e5323b5dcd87ff02a74`.
All 27,546 archive members were verified; the Doctor collector, Grok parser and
bridge diagnostic modules were explicitly checked against the committed source.
All 623 host source pins remained unchanged through execution and collection.
Fresh container worker qualification passed with zero provider calls; credential
readiness observed the selected model. The container was removed after the run.

The safe reader passed 17 authored tests and independent privacy review before
launch. Its diagnostic contracts are extracted from the pinned source without
importing runtime/provider modules. Its timing projection uses closed field
names and finite nonnegative values. Progress reporting retains event counts
without exporting usage-event values or summing them. Source hashes and extraction
provenance are retained alongside the reader.

The [copy provenance](copy-provenance.json) records exact allowlisted evidence
bytes. Absolute paths inside records are historical provenance, and the launcher
is a one-shot experiment record. Raw provider text, credentials, owner-private
archives and campaign logs are excluded. The readout's `prior` is the historical
eight-turn run, not evidence of a causal token reduction. The stronger public
behavior/performance validation design remains pending in the
[earlier pair report](../doctor-live-pair-20261014/README.md).
