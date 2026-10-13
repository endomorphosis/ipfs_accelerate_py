# Matched source-view read/edit measurement

The October 13 Docker qualification measures the same scripted work through the
existing `llm_router`, isolated worker, source broker and default reader CLI.
The identity arm exposed 79,362 UTF-8 bytes; the comment-compacting arm exposed
79,229 bytes. The difference is **133 bytes, or 0.17%**, counting the prepared
router prompt and every actual CLI stdout/stderr once.

This is a source-reading experiment with an authored provider hook. It has zero
model calls, no native token score and no official Terminal-Bench reward. A real
provider can receive earlier history repeatedly; these bytes do not estimate
that total. The [retained evidence](evidence/source-view-read-edit-pair-20261013/README.md)
records the precise source, runner, environment and replay bindings.

## Fixed workload and measured result

Two fresh worker/broker sessions ran sequentially in one fresh container, with
network access disabled, separate owner/worker UIDs, read-only producer mounts,
five CPUs and 12 GiB memory. A single frozen native preflight supplied the same
58,089-byte owner input to both sessions. Reusing that preflight avoids differing
task/world identities in the prepared input. Identity ran first; this experiment
does not compare cold/warm indexing cost or execution speed.

The source inputs are the permitted public `eigen.py` and `eval.py` from
`largest-eigenval`. The admitted catalog also includes three harness support
files. Each session performs 11 matched logical operations in 12 physical CLI
calls: source bootstrap, optional catalog navigation, complete source reads,
exact expansion, an authored `idx` to `i` edit, stale-read refusal, refreshed
reads, exact postimage expansion, restoration and another stale-read refusal.
Both edit original source after expansion. Neither arm invokes the compact-edit
mapper or solves the benchmark's original coding instruction.

| Captured quantity | `identity@1` | `python-comments@1` |
| --- | ---: | ---: |
| Complete prepared router prompt | 53,933 | 53,940 |
| Default CLI stdout, including final newlines | 25,429 | 25,289 |
| CLI stderr | 0 | 0 |
| One-pass prompt plus stdout/stderr | **79,362** | **79,229** |
| Canonical scripted argv JSON, separate | 4,141 | 4,141 |
| Delivered native response JSON, separate | 39,645 | 39,709 |

All values are UTF-8 bytes. Native response JSON excludes socket framing and is
not added to the displayed CLI text. The compact profile removes 35 bytes from
each of four `eigen.py` reads. `eval.py` uses the codec's exact fallback for
unsupported f-strings. Those 140 saved CLI bytes are offset by seven extra
profile-guidance bytes in the prepared prompt. The native transport grows by
64 bytes even as the displayed output shrinks.

Optional complete catalog discovery costs 12,151 displayed bytes per arm. The
paths were already known from the initial manifest and authored fixture. This
cost is retained in both totals; it is not evidence that a model needed the
catalog. Exact expansions and negative requests are also retained.

## Comparator and verification boundary

[`source_view_trace_comparison.py`](../../benchmarks/agent_supervisor/container_coding/source_view_trace_comparison.py)
accepts two literal CLI traces, common controls and independent owner replay
records. It recomputes prompt/stream hashes and lengths, enforces the normal CLI
rendering, assembles complete paginated reads, and compares logical operations
even when physical page counts differ. Generation roles and initial pin policy
must match. Both exact expansions must reproduce the prescribed original-source
edit. Truncated pages, duplicate operations, changed input, profile drift,
inconsistent accounting and proof/token claims refuse.

Owner replay records remain caller-supplied evidence. Their hashes bind records;
the comparator does not authenticate a running owner. In this Docker experiment
the independent owner additionally rebuilt source views and the complete catalog,
joined native session accounting, checked worker cleanup, and restored both the
public source and complete native workspace. Both arms produced postimage
`6ffc01fe089eac6a9331d7bb836519d7857f8d0cf8ef9dc0dd489a39ec132f8b`.

That postimage matches the previously qualified host compact-edit application.
The [lineage check](evidence/source-view-read-edit-pair-20261013/host-mapper-lineage.json)
checks unchanged mapper modules and retained receipts. It does not rerun the
historical numerical cases, demonstrate a worker-to-mapper handoff, or credit
avoided expansion tokens.

## Implication for indexed context

The current optional compact reader changes later source reads; it does not
itself decide initial source admission. A subsequent
[prompt census and metadata comparison](context_projection_pair.md) established
that this fixture already defers both program bodies and retains three support
files inline. This result gives little reason to expect comment removal alone to
reduce complete-session tokens substantially on these small inputs. It does not
measure the value of DuckDB/DuckLake, capsules, target-trained CodebaseIR or any
other IR checkpoint: no checkpoint inference or source proof was performed by
this experiment. The shared preflight did perform static indexing, capsule/world
construction and Doctor diagnostics; automatic repair and the Doctor's planning,
proof, synthesis, impact and transaction stages were not run. This is preflight plus
isolated worker qualification, not full supervisor START.

The next implementation should connect the
[indexed reading plan](reversible_codebase_views_plan.md) to actual context
replacement where source is still repeated: select task-relevant source and small source-bound capsules, keep
maps and full proof artifacts with the owner, and expand only when needed.
Measure initial disclosure and subsequent ordinary/indexed reads together.
Model-driven compact-edit submission still needs its own production contract.
Then compare repeated matched model sessions with all-attempt native usage and
official task verification. A smaller catalog, fewer redundant reads or a
source substitute must earn its saving in that complete measurement.
