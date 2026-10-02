# Counterexample repair through the native supervisor — 2026-10-02

The [normal qualification](summary.json) completed an independently admitted
scalar coding task using actual Intent384 and Security384 checkpoint inference,
counterexample-driven operator proposals, fresh Lake checks, and the native
supervisor daemon. The worker published exactly the candidate checked by the
models and kernel. Independent public validation passed, the task reached
`completed`, context and retrieval refreshed, and STOP left zero processes.

The complete run took **38.521 seconds** with **zero LLM-provider calls**.
This is one integration observation, not a throughput estimate, a new holdout
score, a Terminal-Bench reward, or a token-savings comparison. No checkpoint was
changed, trained, or promoted. Exact checkpoint selections and finite domains
are retained in [inputs](inputs.json); invocation and environment overrides are
in [command.json](command.json).

## Executed pipeline

The original instruction was decoded and numerically replayed before the
independently authored goal/task graph. It requires `left > 0` and a returned
result equal to `old(right) * old(left)`. The caller explicitly selected the
source, action, mapping `left → capacity`, `right → threshold`, and inclusive
integer domains `[-1, 1]`. This controlled instruction was already evaluated
with the action checkpoint; it is not a new accuracy sample.

Native AST extraction, lexical symbol vectors, DuckDB persistence/reopening,
DuckLake metadata projection/retrieval, semantic capsules, and world-state
capture ran before repair. The [index report](vector-result.json) identifies
lexical TF-IDF retrieval; GTE-small supplies the separately selected learned
checkpoint input embeddings. This is a one-file permitted scope.

The repair consumer freshly inferred the original and both proposed sources.
Each underwent source qualification, rebuilt Intent association, and an actual
live Lake check. Nine finite inputs were enumerated; three enabled the declared
precondition in each case.

| Source return expression | Fresh decoded/source-qualified candidate | Live bounded effect result |
| --- | --- | --- |
| `capacity + threshold` | Yes | Refuted; original counterexample |
| `capacity - threshold` | Yes | Refuted; proposal rejected |
| `capacity * threshold` | Yes | Satisfied; unique selected proposal |

The shared datasets proposal module changed only the arithmetic token. It did
not construct a new predicted IR or declare its proposed repair successful.
The supervisor's ProgramWorld operator independently reproduced each proposed
source. The repair stage made three fresh Security advisor calls, in addition
to the initial task-context inference. Original instructions, configurations,
checkpoint files, source snapshots, and producer pins were rechecked.

The owner handoff required both candidates checked and exactly one nonvacuous
success. It separately verified signed task admission, instruction binding,
task revision, and source preimage. The worker materialized those pinned bytes
in the allocated worktree. Native validation, publication, task completion,
and post-publication refresh then ran through their existing owners. The
canonical source remained unchanged until native publication.

## Evidence and tests

The [test summary](test-summary.json) records **142 passed, zero skipped**:
37 shared proposal tests, 22 consumer tests, 30 owner/worker tests, and 53
neighboring Doctor/admission regressions. Supervisor tests used fresh seal
catalogs. Proposal and consumer suites use authored numerical controls with
actual Lake; the separate normal qualification supplies the actual checkpoint
inference evidence. Neighboring Doctor checks exercised Lean and Z3.

The complete [candidate evidence](candidate-result.json.gz) and
[lifecycle/context result](result.json.gz) are losslessly gzip-compressed.
Their uncompressed hashes are in the summary. The original
[Intent advice](intent-advice.json), [run log](native-supervision.log), and
[source identities](source-identity.json) are retained separately. The
[manifest](manifest.json) covers these evidence files. No owner credentials,
private database, checkpoint weights, or core dump are included.

## Retained failures and limits

The [run history](run-history.json) retains the unsuccessful attempts:

- The first fixture inherited group-writable `.runtime` permissions. The
  owner handoff refused it; the driver now creates its fresh parent as `0755`.
- A later START never received its required owner heartbeat within 30 seconds.
  The child completed initial owner authentication but had not created its
  execution sidecar. The intermittent blocking operation remains unidentified.
  No lifecycle deadline or health check was weakened, and later success is not
  claimed to fix that timeout.
- An early diagnostic parent segfaulted while printing a shim traceback,
  before worker startup. The [diagnostic log](failed-diagnostic.log.gz) is
  retained; its cause is unresolved. Later diagnostics were limited to native
  children, and the successful run used no diagnostic Python hook.
- The diagnostic worker successfully materialized the candidate, but the
  fixture declared an absolute Python validation path. The sealed runtime
  requires `python` or `python3`. The driver now declares `python3`; the
  [failed validation log](failed-validation-worker.log) remains available.

This closes the bounded counterexample-to-native-worker handoff demonstrated
here. It does not add learned goal decomposition, arbitrary vulnerability
repair, LegalIR/UI/UX constraint composition, or whole-program equivalence.
The source grammar remains one two-parameter integer function with one supported
arithmetic expression. Domains and source/action mappings remain explicit.
Saved proof reports never become live proof or completion authority.

The existing keyword/signature Doctor composition is unchanged; this scalar
path uses its own owner handoff and the native supervisor gates. Native proof
producer checks still require a Git checkout, so complete gitless Docker
repair remains unqualified. No new Codex/indexed/no-index comparison is claimed.
