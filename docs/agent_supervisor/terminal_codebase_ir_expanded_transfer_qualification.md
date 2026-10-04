# TerminalBench expanded development transfer qualification

The existing intent-conditioned relevance head now has an explicit transfer API
for a larger frozen CodebaseIR corpus. It preserves the historical checkpoint,
training receipt and five original query results exactly. The expanded corpus
contains **375 functions, nine queries and 22 navigation judgments** across six
admitted source files. This is development evidence; supervisor defaults and
symbolic task selection remain unchanged.

The [transfer API](../../benchmarks/agent_supervisor/container_coding/terminal_codebase_intent_ranker_transfer.py)
requires independent old/new corpus, checkpoint and training-receipt pins. It
replays both native corpora and the original fitted head, requires exact old
source/query/judgment/candidate populations, and permits history references only
to grow. The four original training pairs and their ordered 80-dimensional
feature maps remain identical. New codebases and queries may enter validation
or test roles only. No optimizer step or rewritten training receipt is produced.
The distinct transfer receipt binds evaluation lineage while retaining the
original checkpoint and fit lineage. Its pure metadata projector adds a
`ranker_transfer` family and binds every ranking row to both corpora. The older
closed 12-family source-analysis consumer rejects this new profile; integration
requires a separately reviewed profile migration.

The four additional experiment inputs are the public instruction and complete
initial source for `llm-inference-batching-scheduler` (`baseline_packer.py`, six
functions) and `feal-differential-cryptanalysis` (`feal.py`, seven functions).
The nested batching helper is included. FEAL `create_random_keys` remains in
the candidate bank, unjudged and unexecuted. Evaluation, solution, generator,
oracle, `cost_model` and hidden assets are excluded from the experiment inputs.
Admitted source is parsed, never imported or executed as application code.
This does not establish dependency, module-global, whole-program or whole-repo
semantics. Scheduling performance, the global shape budget and cryptanalytic
key recovery remain unproved.

The [frozen publication](../../../../artifacts/codebase_ir_terminal_bench/terminal-codebase-ir-expanded-transfer-20261004-01/evidence/freeze-01/frozen.json)
closes inputs, queries, labels, splits and history before transfer inference.
The historical head and authored source review predate this freeze; global
pretraining exposure is unknown. These are development validation/test banks,
not blind holdouts. The nine query roles are two train, three validation and
four test; all four added queries are outside training. The native split guards
passed, with 13 within-role duplicate groups retained. Both native SourceRefs
and the opaque original requirement survive every query. Navigation labels do
not establish satisfaction of the full instruction.

The four new first-positive ranks are mixed:

| Query | Role / candidates | Trained | Lexical | Zero | Reverse |
| --- | --- | ---: | ---: | ---: | ---: |
| batching-build-plan | validation / 6 | 1 | 2 | 6 | 6 |
| batching-representative | validation / 6 | 2 | 4 | 3 | 5 |
| feal-encryption | test / 7 | 2 | 1 | 1 | 6 |
| feal-round-transform | test / 7 | 5 | 1 | 5 | 3 |

The trained head helps both batching queries and worsens both FEAL queries.
Mean reciprocal first-positive rank over these four development queries is
**0.55 trained versus 0.6875 lexical**. This supports keeping ranking diagnostic;
it does not qualify generalized improvement. Each result retains its entire
bank for trained, lexical, zero and reverse controls. Model-off has an empty
ranking and null metrics. The original five result objects remain unchanged;
the old singleton banks still provide degenerate controls. No labels, queries,
roles or model parameters were adjusted after viewing these scores.

**29 focused cases / 87 phases passed** in the
[captured test run](../../../../artifacts/codebase_ir_terminal_bench/terminal-codebase-ir-expanded-transfer-20261004-01/evidence/tests-02/tests.xml).
Controls exercise mandatory external pins, native train/source/operator drift,
full banks and original guards, history removal, separately replayed scores,
resealed ranking/residual/authority tampering, old/new metadata lineage,
detached outputs, and refusal by the older closed consumer. The first run
retained 13 passes and 16 failures: a production guard required literal `True`
but received a nonempty set for the added-query condition. Converting that
condition explicitly to `bool` fixed it; the tests were unchanged. Both actual
test runs completed one synthetic four-epoch fit, eight unit epochs total.
There was no new public fit or autoencoder fit. The historical 128-step fit
remains finite learning evidence, not an asymptotic or semantic convergence
certificate.

Native DuckDB and isolated DuckLake hydration completed, followed by a fresh
Python process reopening and verifying the complete metadata. The
[native readback](../../../../artifacts/codebase_ir_terminal_bench/terminal-codebase-ir-expanded-transfer-20261004-01/evidence/actual-01/metadata-readback.json)
contains **4,174 rows, 13 families and 67 lake row packets**: six sources,
375 AST records, 397 KG edges, 375 vectors, nine queries, 22 judgments, two
split/history records, one checkpoint, one training receipt, 2,984 rankings,
one transfer record, one coverage record, and zero contracts. The KG records
syntactic containment and authored navigation only. Vectors retain native
44-dimensional AST features; they are not learned autoencoder latents. Lake
packets contain canonical JSON rows, without typed vector columns. Contracts
remain unknown, so this is not complete metadata or proof-index coverage.

The [independent file-only audit](../../../../artifacts/codebase_ir_terminal_bench/terminal-codebase-ir-expanded-transfer-20261004-01/file-only-audit-01.json)
reimplemented the fixed token buckets, all 746 query-candidate feature/score
pairs and all 2,984 ranking rows. It checked every exported wrapper, payload,
family digest and row root against retained metadata inputs, the old checkpoint
and fit equality, freeze/public-input pins, source custody, tests and leases.
The reader imported no producer, performed no SQL and made no fit or prover
call. Native restart success is bound producer evidence, separately from the
reader's export verification; neither check proves program behavior.

All four actual attempts are retained, including the failed first test.
Their measured producer outers total **31.541022538993275 seconds**. Owned
times are nested and are not added again. Every root acquired and released one
CPU, 2,048 MiB and four child slots under the unchanged 2/50/10 pressure policy,
with 30-second admission and 120-second outer limits; all waiters drained.
Selected parent import counts were 91, 91, 85 and 92, matching before/after.
These are bounded observations, not a complete loader or fresh-child census.
Full-session CPU, GPU, memory peaks and process counts are unavailable.
Preparation, audit, documentation and retention are outside the producer sum.
The [review record](../architecture/repository_proof_index_and_codebase_ir.expanded_transfer_review.json)
pins evidence and preserves these scopes. Final sealing and readback are a
separate last writer under the artifact root's `final-retention-01`.

Each query retains two unknown residuals and planning abstains. No new checker,
Lean compilation, verified proof-cache hit, PlanCreate or semantic task/fact
authority is produced. All 32 repository-proof-index criteria remain OPEN.
The next work is to ground original mandatory requirements in reviewed native
AST/contracts/formula evidence, qualify source-to-logic projections and Lean
bridges with explicit supported fragments, and validate cache invalidation
against source/dependency changes. Optimizer and arithmetic convergence
obligations remain separate. Any adaptation of the model needs a new
predeclared training/evaluation corpus and new receipt, preserving this mixed
transfer result. Before ranking affects supervisor evidence, review complete
query coverage and the explicit migration of the existing analysis consumer.
