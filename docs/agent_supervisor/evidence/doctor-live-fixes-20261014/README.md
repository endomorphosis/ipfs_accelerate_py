# Repairs following the first live Doctor pair

The [live pair](../doctor-live-pair-20261014/README.md) exposed incomplete audit
and failure observations. These fixes preserve useful evidence for a fresh run;
they do not recover missing historical provider usage or establish benchmark
success.

* Doctor v2 receipt serialization now forwards the original provider, requested
  output-token setting and enforcement flag to the audit subprocess. Missing or
  forged values still fail validation. Historical v1 projection is unchanged.
* Grok streaming accepts one bare terminal `end` marker, or an agreeing final
  end envelope after a failure notification. Failure stays failure. Existing
  identity and spend fields must agree with type-sensitive comparisons; missing
  fields, contradictions, extra payload, repeated trailers and post-terminal
  substantive events remain invalid. Per-event usage stays unverified and cannot
  be summed into a native total.
* The bridge retains two to four distinct, valid closed error records under
  `router-child-failure-observations@1`. Canonical sorting deliberately removes
  log-order attribution. The record grants no dispatch, continuation, settlement
  or completion authority. Duplicates, malformed or conflicting populations,
  and overflow still yield ambiguity. Existing single-error reports are unchanged.

[Qualification](qualification.json) contains **426 distinct passing tests across
12 modules**, no failures or skips. All 22 recorded current source pins match
the qualified runs; pinned datasets/kit imports and dependency distributions
remained unchanged. [Independent review](independent-review.json) found no
remaining blockers after correcting the historical ambiguity claim and
tightening the Grok envelope comparisons.

| Qualification | Passed | Scope |
| --- | ---: | --- |
| Collector | 79 | Actual serialization, independent interpreter reconstruction, invalid original fields |
| Grok stream | 213 | Terminal/end protocol, final usage, native failure classification, contradictory data |
| Continuation | 17 | Two native child/router paths with explicit Grok 8, late inventory drift, turn-bound checks |
| Diagnostics and forwarding | 117 | Real failing child processes through settlement, sidecar and export; worker parser forwarding |

Provider responses and UID boundaries in continuation tests use doubles. Worker
forwarding tests execute the actual installed parser with a stubbed router
entrypoint. These tests do not qualify a fresh Docker owner/worker boundary or
identify the exact second-child exception from the historical live run. No live
provider calls were made during qualification.

The original collector regression failed before its fix. The Grok regression
likewise reproduced six new failures before repair. Those developmental results,
and the earlier 192-test Grok pass superseded by the stricter 213-test run, remain
in the evidence directory. The final count includes only the four runs listed
above; no test case is counted twice.

The public behavioral/performance validation profile and its typed continuation
cause are still pending, as described in the live-pair report. Neither structural
acceptance nor a process exit code is a proof that the task's public performance
obligation was satisfied.
