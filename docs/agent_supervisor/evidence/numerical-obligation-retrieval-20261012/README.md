# Bounded retrieval of pending numerical obligations — 2026-10-12

The public `largest-eigenval` source probe recovered all **45 pending
obligations** through the supervisor's source-validated owner API. The planning
summary remains below 8,192 bytes and now retains complete counts by obligation
kind even when all individual rows are omitted. This closes an information
access gap, not a source-correctness proof or a measured token-saving claim.

| Current public-source observation | Result |
| --- | ---: |
| Exact pending rows retrieved | 45/45 |
| `eigen.py` rows / pages | 16 / 4 |
| `eval.py` rows / pages | 29 / 8 |
| Largest complete page, UTF-8 bytes | 6,282 |
| Requested page bound, UTF-8 bytes | 8,192 |
| Planning summary, canonical bytes | 8,094 |
| Cold-process pages reproduced exactly | 12/12 |
| Metadata/world DuckLake catalogs / links / records | 3 / 3 / 1 |
| Source obligations discharged | 0 |

The [current public qualification](current-02/qualification.json) and
[cold replay](current-02/cold-replay.json) use only the permitted public
`eigen.py` and `eval.py` source files. All original public inputs and production
source hashes were rechecked unchanged. Exact function/kind/combined filters,
unknown-function empty results, and five stale-source/receipt/continuation
controls passed. No target program, model, provider, solver or Docker trial was
executed. Native provider token usage is unmeasured.

The 12 full-row responses total 57,827 bytes, while the complete reports total
472,546 bytes. These contain different information: reports also hold ProgramIR
and logic projections, whereas pages contain pending rows and provenance.
They are not equivalent encodings, and their size difference is not a measured
compression ratio or session-token reduction. The older probe's summary was
8,096 bytes; both summaries stay within the same limit but contain different
selected details with explicit omissions.

The shared datasets module owns source/report replay, exact filtering and
deterministic pagination. The accelerator wrapper binds the selected owner
receipt, source tree, complete report, query and continuation. It validates
the retained index before following a locator, checks report bytes before
parsing, and checks the full source/artifact/producer snapshot again before
returning. A row that cannot fit raises an error. No rows are partially cut or
silently skipped. Bool/integer substitutions no longer pass exact receipt
replay comparisons.

The three final suites passed **263 tests, zero failures/errors/skips**:

- [Datasets pager: 42](datasets-targeted-01/qualification.json), covering exact
  full-row pagination, Unicode, query binding, stale reports and byte limits.
- [Owner wrapper: 28](owner-tests-03/qualification.json), covering real lake
  receipts, owner binding, all-row recovery, strict types and concurrent edits.
- [Integration: 193](integration-03/qualification.json), covering existing
  context, archive selection, observations, preflight and successor lifecycle,
  plus five new cache/planner/coding delivery cases. An old authored archive
  fixture now contains a valid empty gzip tar; production verification remains
  enabled.

One trailing blank line was removed from the new integration test for
publication. The [byte/AST comparison](publication-test-whitespace.json) binds
both versions; test assertions and all production bytes are unchanged.

The integration uses authored provider responses. The real planner and coding
dispatch paths carry exact kind counts, retrieval discovery and checked context
references. An owner-side test process resolves an omitted row from that
reference. An isolated model/CLI retrieval tool or RPC into owner-private lakes
was **not** exercised or newly qualified. Planning still uses one provider
request. See the [delivery scope](delivery-scope.json) and
[independent code review](independent-final-review.json).

The final consumer SHA-256 is
`e536eb34f9d2a833d0613f5cdd7c24ad1ca4874f3e649286177e1e5048fcef53`;
the shared datasets pager is
`a2c0d7bf3f24bde9287ee06d65fedcb6920e1c1d3d04b520f3949825e5497d8f`,
published in datasets commit `f47161b757190380e9898aff155067c1ad609217`.
The [commands and source bindings](current-02/command.json) identify the exact
tested producers. The earlier successful public probe is retained under
`historical-01`; it predates the locator-read hardening and does not qualify
the final consumer. Earlier failed authored test fixtures remain in the local
run directory; they involved no model calls or benchmark attempts.

The [artifact manifest](artifact-manifest.json) binds the bounded evidence.
Owner databases, full source reports, source text, provider prompts and model
payloads are excluded. The next integration requirement is a task-bound,
owner-mediated retrieval request/response path available to the isolated agent,
followed by a separately controlled complete-run token comparison. The
[usage and proof boundaries](../../numerical_proof_admission_readiness.md#current-structural-obligation-retrieval)
describe the current opt-in profile and API.
