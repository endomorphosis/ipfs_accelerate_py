# Compact numerical planning hints and exact obligation retrieval

The selected `python-numerical-control-structure@3` context now keeps compact,
source-bound hints inside its existing 8 KiB planning budget. On the permitted
public `largest-eigenval` inputs, the previous 7,614-byte summary evicted every
inline pending obligation. The new 8,064-byte summary retains eight hints spanning
both source files, all four functions, both loops and all four loop proof phases.
It includes five of the eight loop proof obligations; 97 of the 105 total pending
rows remain omitted inline, with exact counters. No obligation was discharged.

This is selective context delivery. The summary grows by 450 bytes to make
useful references discoverable; the complete source and proof records remain
available through validated retrieval. It does not create a reversible minified
codebase or establish that arbitrary abbreviated source saves model tokens.

## Consumer and shared pager

The supervisor's [context owner](../../ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py)
packs hints deterministically across source paths, functions, loop regions and
proof phases. A shared `numerical-planning-hints@1` contract declares the hints
unverified and requires full context before interpretation. Proof applicability
remains distinct from source-expression evaluation. Full reference lists within
accepted obligation hints are preserved; whole-row and upstream omissions are
counted. Oversized candidates cannot evict every shorter candidate.

The [isolated worker client](../../ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py)
supports an optional exact ID:

```text
/opt/ipfs-supervisor/venv/bin/python -I \
  /opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py \
  --path <source_path> --obligation-id <obligation_id>
```

The existing owner-selected descriptor supplies the session. Optional
`--function-id` and `--kind` filters are conjunctive. The shared datasets pager
performs source/report replay before filtering and returns the complete native
row, including guards, CFG occurrences and proof scope. Exact-ID responses contain
zero or one row and cannot continue onto another page. The client rejects foreign
rows, mismatched filters and contradictory pagination. Source edits invalidate
retrieval. ID selection grants no proof, execution or completion authority.

Omitting the new filter, including an explicit Python `None`, preserves the old
query and cursor bytes. Twelve captured historical pages across v1/v2/v3 replayed
exactly. The v1/v2 summary bytes, all three public report versions, grammar/logic
files and structured-checkpoint implementation pins also replayed unchanged.
The v3 summary and consumer implementation hashes change, so old pinned contexts
must be rebuilt through normal preparation.

## Public-input Docker measurement

Two fresh network-disabled containers used the production owner launcher,
separate owner/worker UIDs, `llm_router` implementation runner and numerical IPC
broker. Actual DuckLake preparation verified three catalogs, three links and a
world record. An authored provider callback checked that the complete numerical
summary survived compiled worker context and model-prompt transport unchanged.
The production planner request compiler separately accepted exactly the same
staged summary. The Docker graph and provider replies were authored; this is not
a live planner or full supervisor START qualification. A regression separately
exercises preparation → planner callback → admitted indexed context.

The five inspection scenarios use only the public instruction and visible hints:
one target-entrypoint hint and one hint for each loop proof phase. Residual hints
were unavailable in the bounded summary and are recorded as unselectable. The
selector does not inspect full reports or source bodies to discover hidden IDs,
and these authored examples do not prove task relevance or adequate coverage.

| Retrieval scenario | Requests | Returned row instances | Request + response JSON bytes | Including common 8,064-byte summary |
| --- | ---: | ---: | ---: | ---: |
| Five scenarios using function/kind filters | 7 | 8 | 42,008 | 50,072 |
| Same five rows using exact IDs | 5 | 5 | 26,954 | 35,018 |
| Separate exhaustive scan of all pending work | 56 | 105 | 404,290 | 412,354 |

Exact-ID selection reduces the first two rows' measured total by **30.1%**,
including the common summary, or **35.8%** for request/response JSON alone.
Function/kind paging stops when the selected row is found. Each scenario runs
independently without a response cache; no batching or deduplication benefit is
assumed. Both selectors return byte-identical complete rows. The exhaustive scan
has a different coverage objective and is a separate transport-cost reference.
The historical summary offered no IDs; its discovery cost was not measured.

The table counts complete canonical JSON envelopes. IPC adds four framing bytes
per request and per response; the CLI adds one output newline per response.
These layers are recorded separately. Reconstructed request hashes and actual
response hashes, sizes and row counts matched the owner ledger for every measured
page. Selected-session diagnostic requests are excluded from the table and
included in the retained full-session accounting.

Both containers refused wrong/unrelated peers, forged/crossed cursors and stale
canonical or allocated source, then recovered after restoration. Five private
owner paths were unreadable to the worker. Worker privilege checks, descriptor
and socket removal, source restoration and container cleanup passed. Production
request and byte quotas remained unchanged. Root setup and a writable container
base support the existing sudo transition; this is bounded custody qualification.

See the [sealed evidence](evidence/numerical-planning-selection-20261013/README.md)
for test results, baseline replay, exact source pins and independent reviews.
**849 distinct tests passed**: 77 shared-library pager tests and 772 accelerator
regressions. Two explicit learned-checkpoint integration tests were skipped
because their development checkpoint and cached GTE snapshot were not supplied.
The 34 focused consumer tests are included in the 772 and are not counted again.
The first selected-container attempt failed before dispatch because the fixture
assumed every planner constraint was a mapping; existing constraints may also be
strings. The fixture was corrected and both subsequent modes completed.

## Remaining qualification

These measurements include no live model usage, benchmark reward, learned
autoencoder inference/training, checkpoint upload or proof discharge. No new
Lean artifacts were generated: the grammar and logic producers are unchanged.
The [earlier native comparison](context_projection_pair.md) remains the live
performance evidence. A fresh live ablation should charge indexing, planning,
summary delivery, retrieval, retries and provider usage, and verify task success
under the same model, task inputs and work budget. Only that run can establish
whether this selective delivery reduces total model tokens while preserving
coding performance.
