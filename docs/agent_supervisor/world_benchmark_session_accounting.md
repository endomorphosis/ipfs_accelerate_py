# Session accounting in the supervisor benchmark

Harbor usage export and benchmark comparison now reconcile native observations
through `context_delivery_accounting`. Distinct invocation IDs can refer to the
same resumed Codex session. The reporting path selects its latest consistent
cumulative snapshot once, including observations in different retained trials.
Accurate session accounting is needed to assess whether a smaller semantic view
saves workflow tokens.

## Reporting path

`terminal_supervisor_usage_accounting.py` provides
`summarize_supervisor_usage(report)`, `summarize_supervisor_usage_reports(reports)`
and `summarize_supervisor_usage_summaries(summaries)`. The latter reconciles safe
observations across trials without adding already aggregated trial totals.

Harbor's `measured_usage` delegates to this bridge. Export also checks integer
counters, conservation and cache semantics. `benchmark_comparison` uses the same
bridge within trials and across the retained arm population. Failed attempts
remain present; official verifier reward and usage completeness stay separate.

An authored regression supplies two invocation IDs for one session:

| Cumulative observation | Input | Cached subset | Output | Total |
| --- | ---: | ---: | ---: | ---: |
| Earlier | 100 | 80 | 10 | 110 |
| Later | 150 | 100 | 20 | 170 |

The session contributes 170 tokens. Adding 110 and 170 would charge overlapping
history. Two observed invocation IDs do not establish two physical provider
requests or a complete census of internal retries. Conflicting or regressing
snapshots withhold the affected totals.

## Missing evidence and categories

Missing provider/model, session identity, artifact pin or record census remains
an unsupported observation. Those fields are not synthesized. Unbound counters
remain visible per observation but cannot inflate known aggregate subtotals
alongside an identified session. The runtime core applies the same rule to
earlier paired-accounting consumers and validates native record limits again
when reading sealed observations.

An absent legacy `unreceipted_provider_attempt` field means no missing attempt
was reported in that supplied report. Explicit null remains unknown; a nonempty
failure marker preserves an unreceipted attempt. An empty trial report stays a
gap when combined with other reports. None of these fields proves a complete
campaign or auxiliary-call census.

Codex cached input is a subset of input. Grok uses separate fresh-input,
cached-input and cache-write categories. Mixed-provider raw input counts remain
unavailable; normalized input including cache is reported separately when
supported. A task-success or close event cannot supply missing token counters.

Historical native-baseline reports retain their original session-census
contract. Missing, contradictory or overlapping census data withholds qualified
aggregates; raw reported counters remain visible separately. Counter sums obey
the same numeric bounds as individual counters.

Supervisor USD totals and USD-per-pass ratios are unavailable because the
retained per-invocation amount has no established additive or cumulative scope.
The original amount remains under `observed_reported_cost_usd`, without billing
verification. Session observations involving failed trials can include earlier
work; they do not attribute all of that usage exclusively to failed work.

## Durable attempt handoff

`terminal_collected_supervisor_usage.project_collected_supervisor_report(...)`
validates the independently pinned declaration, slot census, component joins
and optional verifier population before selecting an arm. It returns the
private `provider_invocations` format consumed by Harbor reporting, preserving
missing receipts and global unknown-call information. Its
`world_packet_attempt_collection` metadata retains input pins, slot states,
the durable gate and the combined comparison gate.

```python
report = project_collected_supervisor_report(
    declaration, slot_census, verifier_results, arm="candidate",
    expected_declaration_sha256=declaration_pin,
    expected_slot_census_sha256=census_pin,
    expected_verifier_results_sha256=verifier_pin,
)
usage = measured_usage(report)
```

The report contains full private receipts. Publish the content-free usage
summary where appropriate. A known cost can coexist with a missing terminal or
unverified output: inspect the retained durable and comparison gates before
making an efficiency claim. Projection creates no Harbor execution, official
reward, live source currentness or provider authority.

## Qualification

The final suites pass 633 distinct cases: 66 runtime-core cases, 70 bridge and
handoff cases, 119 Harbor/provider-stop cases, 153 comparison/baseline cases and
225 prior paired-collection/accounting/context-observation cases. Test provider
receipts and counts are explicitly authored.

An offline replay also uses the older actual receipts in the real Harbor
reporting functions and comparison collector. It preserves all three slots and
known token subtotals of 486,151 and 21,251. Complete totals and Harbor exported
counters remain null; the earlier experiment stays unqualified. Its adapter
envelopes are authored, so this stage creates no new model calls, Harbor trials
or official rewards.

The final replay comparison's full canonical SHA-256 is
`1f70587dc681faed6a1236413145fc67d450b20e02ab0e7c20ab1e2f596a327b`.
Private evidence is retained under
`/home/barberb/lift_coding/artifacts/world-benchmark-session-accounting-20261015-01`.
Complete workflow savings and default adoption remain unqualified.

See [durable attempt retention](world_packet_attempt_retention.md) and
[presentation quality evidence](world_packet_quality.md) for preceding contracts
and the actual rate-limited trial. A fresh controlled paired trial remains the
empirical next step when the provider is available.
