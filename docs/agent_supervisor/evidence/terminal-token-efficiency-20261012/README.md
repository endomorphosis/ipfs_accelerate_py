# Token-efficiency implementation qualification — 2026-10-12

This change implements opt-in instruction deduplication, retrieval-ranked
capsules, and descriptive whole-attempt tokens/cost per official pass. It
retains the full supervisor's planning and `llm_router` path. These are offline
implementation and packaging checks, not a new live Terminal Bench result.

## Measured complete input bytes

The signed dispatch fixtures use actual local admission, indexing, context
compilation, router preparation and historical delivery audit. Only the coding
provider response is authored. Both complete inputs include the workspace
mapping and the complete admitted public instruction; this measurement uses
the legacy reply contract.

| Fixture | Native bytes | Candidate bytes | Selected bytes | Outcome |
| --- | ---: | ---: | ---: | --- |
| Large signed instruction | 31,807 | 27,574 | 27,574 | 4,233 fewer bytes (13.31%) |
| Tiny signed instruction | 18,713 | 18,847 | 18,713 | Keep original input |

The observations come from pytest properties in
[the retained JUnit result](dispatch-metrics-final-01/junit.xml), summarized in
[input-byte-observations.json](dispatch-metrics-final-01/input-byte-observations.json).
They do not measure provider tokens, hidden CLI scaffolding, coding behavior,
benchmark reward or complete-run savings. Workspace path lengths also affect
these fixture input sizes.

## Behavior qualified

- `common-bindings-instruction@1` references only the exact duplicate instruction
  in `semantic.raw_sources`. The complete separately admitted instruction stays
  inline. Restore reproduces the original semantic transport byte-for-byte.
  Missing, stale, foreign or modified source bindings refuse before discovery.
- Selection uses complete rendered inputs. Larger candidates, invalid counts
  and counter failures retain the original prompt. The requested treatment is
  retained on fallback. Census stops before provider discovery. Historical
  replay without a supplied counter reconstructs delivery but leaves counter
  verification false.
- Retrieval ranking replays the exact native query/source population and maps
  hits to source-bound symbols and line spans. Required raw sources, admission,
  freshness and existing count/byte limits still apply. Cold reload verifies
  the same retained retrieval selection. Default lexical behavior is retained.
- Historical ranked replay independently checks retained source hashes, the
  complete AST snapshot and query result. It reproduces the original input
  after source edits/removal and returns historical status, never currentness.
  Current dispatch and operational historical responses still refuse stale
  source use.
- Efficiency ratios include all supplied retained attempts, including failed
  attempts. Only official reward 1 supplies the denominator. Unknown rewards,
  missing required totals and zero passes produce explicit unavailable reasons.
  Provider-reported cost remains distinct from a verified bill.

## Test evidence

Each run directory retains the exact command/environment overrides, JUnit,
pytest log and before/after source hashes. No model call, training run or model
download was performed by these host tests. The pure codec and independent
audit edge suites were recorded by separate reviewers.

| Run | Passing tests | Scope |
| --- | ---: | --- |
| [instruction-codec-02](instruction-codec-02/qualification.json) | 40 | Exact restoration, tamper rejection, Unicode/embedded markers, complete-input selection |
| [instruction-audit-edges-01](instruction-audit-edges-01/qualification.json) | 11 | Historical counter absence, treatment relabeling, foreign census binding |
| [dispatch-metrics-final-01](dispatch-metrics-final-01/qualification.json) | 89 | Signed instruction path (10) and comparison accounting (79) |
| [retrieval-historical-01](retrieval-historical-01/qualification.json) | 120 | Current/historical retrieval, semantic replay, cold reload, successor refusal |
| [harbor-final-01](harbor-final-01/qualification.json) | 237 | Configuration, archive policy, adapter/driver, shipped worker parser, treatment drift |
| [native-final-01](native-final-01/qualification.json) | 289 | Existing metadata/public-instruction/audit regressions and real native census owner/cleanup |

These **786 distinct host tests** passed without skips and with their tracked
sources unchanged; see the combined [qualification](qualification.json) and
final [source manifest](source-manifest.json).
The runner is retained as [run_tests.py](run_tests.py); it uses the recorded
local development interpreter and dependency paths. The command receipts,
rather than ambient shell activation, identify each test environment.

## Actual archive and offline Docker smoke

[Archive qualification](archive-01/qualification.json) checked all 12,266 actual
tar members against the manifest and current source bytes. Both metadata views
passed the required 38-helper policy closure. The local 110,698,194-byte archive
has SHA-256 `25a83f461c081130c482b5b1703f6dceaeb0ca22ea4b99892e1147c928bb397d`.
It contains production source and the three existing DuckDB extensions, with no
model weights, Lean assets, task input or credentials. Archive bytes and the
extracted tree remain local rather than being committed as evidence.

[Docker attempt 02](docker-instruction-02/qualification.json) ran all 10 signed
instruction dispatch tests against those extracted production sources. All 10
passed without skips in the cached image
`ipfs-supervisor-numerical-dev:20261011-r2`, image ID
`sha256:771900c18b663b33a5bf96e066346b2d5c842506523efb167aebe8d3a99b5c07`.
The container used no network, 5 CPUs/12 GiB, a read-only root and read-only
source mounts. All 12,287 tracked source/test/control files remained unchanged;
the container was removed. Imported module locations are retained in
[module-origins.json](docker-instruction-02/results/module-origins.json).
These 10 cases repeat host tests and are not added to the distinct host count.

Docker's shorter temporary workspace paths measured 31,798 to 27,565 bytes for
the large fixture (the same 4,233-byte reduction). The tiny fixture retained
18,705 bytes instead of the 18,837-byte candidate. The first Docker attempt
failed before imports because the generated entry script had an escaping error;
[its receipt](docker-instruction-01/qualification.json) is retained. The corrected
script was compiled before launch. No production change was needed.

This qualifies source packaging and offline router/audit behavior in the cached
image. It does not qualify a complete newly deployed supervisor, optional
runtimes, live provider access or Terminal Bench performance.

## Operational limits and next experiment

The instruction treatment is explicit at benchmark preparation:
`--semantic-metadata-view common-bindings-instruction@1`, full arm, original
`supervisor-semantic-router-input@1`, and a current runtime archive. It requires
the exact duplicated raw instruction and separately admitted verbatim block.
Legacy representation remains the default. The CLI byte proxy is not a
qualified provider tokenizer; an injected counter is separately identified and
replayed where available.

Retrieval ranking is programmatic through
`prepare_initial_context(..., retrieval_ranked_capsules=True)`. It has no new
benchmark CLI flag. Ranked successor publication explicitly refuses before
partial preparation: fresh retrieval must be generated before selecting the
successor's capsules. Original retrieval artifacts/snapshots and semantic
blocks must remain available for historical replay. Similarity and replay do
not establish code correctness, proof coverage or retrieval quality.

Run matched ordinary-full and instruction-deduplicated trials first, retaining
all attempts and official verifier results. Keep provider/model, planning,
task inputs, reply contract and budgets fixed. Compare total native tokens per
official pass, wall time and reported cost. Evaluate ranked capsules separately
after successor refresh is integrated; keep proof-cache reuse and progressive
Quack disclosure as distinct treatments. The previously retained live compact
transport comparison used more total tokens, so smaller input bytes are not
accepted as evidence of total savings.
