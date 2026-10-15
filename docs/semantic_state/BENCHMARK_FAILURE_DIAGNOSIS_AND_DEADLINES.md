# Retained benchmark failures and deadline corrections

This guide records the earlier 300-second correction batch. Fresh benchmark
preparations now default to 600 seconds; the current budget and archive checks
are described in [benchmark timeout configuration](BENCHMARK_TIMEOUT_CONFIGURATION.md).

The September 29 full and no-index trials failed for different reasons.
The retained `failure_diagnosis.json` in
`artifacts/supervisor-benchmark-failure-fixes-20261012-01` binds the original
reports, aggregate outcomes and current helper evidence to their hashes.
It exports no hidden verifier assertions or solution source.

| Trial | Observed failure | Limits of the diagnosis |
| --- | --- | --- |
| Full indexed | Cold context took 107.245 seconds. Coding hit its 56-second limit and the task remained in progress. Cleanup succeeded. | Published candidate bytes and the raw coding transcript were not retained; lexical error indicators do not establish a tool-protocol cause. |
| No-index | Coding returned in 50.868 seconds. All 367 repository regressions and five of six task checks passed. The remaining outcome check concerned the CWE identifier. | The exact published report was not retained. We cannot distinguish classification, extra/missing identifiers or formatting. |

Native completion in this harness admits compilation and the declared report
shape. Its acceptance criterion explicitly leaves benchmark correctness
unverified. The public smoke accepts an arbitrary well-formed CWE ID; it
cannot establish that the reported vulnerability classification is correct.
The fix does not hardcode an identifier inferred from a hidden outcome.

Literal public-instruction forwarding, bounded public-file capture and verified
initial-index reuse already existed in current source before this batch.
The old trials did not qualify those current paths. Current cold-context
latency and original-task success still need fresh measurement.

## Driver interruption

`terminal_container_supervisor` now raises `_AgentWorkBudgetExpired`, a
dedicated `BaseException`, from its work alarm and exhausted remaining-budget
checks. Optional components handle ordinary `Exception` failures; they can no
longer swallow the deadline after its timer has been disabled.

The outer report boundary catches this specific interruption. Existing STOP,
runtime/owner close, worker cleanup, receipt collection and report persistence
remain in their finalization paths. `KeyboardInterrupt` and `SystemExit`
retain their existing propagation behavior after finalization. The driver's
40-second cleanup reserve and total budget bounds remain unchanged.

## Adapter budget and receipt binding

`FullSupervisorAgent.run` uses one monotonic 300-second budget from entry.
Instruction and requirement uploads, public-input capture, driver execution,
report/admission downloads and public-output export all consume that budget.
Each coroutine is created only after its remaining-time check.

Immediately before dispatch, the driver's integer budget is the remaining
adapter time minus 30 seconds: ten for execution grace, fifteen for three
bounded exports, and five for final receipt work. It stays within the existing
90–285-second driver limits. A zero-cost fixture gives 270 seconds; real
elapsed transport and setup shrink that amount. Insufficient time refuses
dispatch rather than resetting or extending the original benchmark limit.
The minimum concerns the complete driver, including its cleanup reserve; it
does not promise a minimum coding allocation or eliminate cold-index cost.

Timeout and cancellation still attempt bounded evidence export, and preserve
the original exception. New export operations stop when the shared deadline
can no longer accommodate them. `adapter_deadline` records stage budgets,
elapsed time and exhaustion. Async tests verify cooperative cancellation;
Harbor's outer limit and native worker cleanup remain the process boundaries.

Every run also gets a locally generated 32-character nonce. The driver state,
report and admission exports use that same fresh path beneath the private
state root. An early upload failure cannot import a previous run's completion
or usage from the old fixed path. The three exposed token counters reset to
unknown at entry, including when an `AgentContext` is reused.

## Verification scope

The new tests exercise actual POSIX signals in isolated child processes and
actual adapter coroutines with owned environment/runtime doubles. Regressions
cover cancellation, STOP/finalization, failed usage retention, stale receipts,
public file retention and existing signed command forwarding. A native public
smoke fixture demonstrates the distinction between report shape and CWE
correctness. Test catalogs and source captures stay in the new artifact
namespace, with mandatory AST seals enabled.

These checks do not establish a successful rerun of the original task, exact
CWE correctness, lower current provider usage or Docker cancellation behavior.
The historical counters remain partial observations as described in
[the relation audit](SEMANTIC_SELECTION_RELATION_AUDIT.md). No provider run,
model download or live owner workflow is part of this correction batch.

The final sealed run passes 120 cases: 22 new regressions and 98 compatibility
checks, with no skips and unchanged source guards throughout execution. The
first attempt stopped during collection because workspace `.venv` lacks
Harbor. The final run exposes the existing original Harbor 0.23.0 environment
through a read-only overlay; it installs no packages. Eight unrelated captured
input files remain unchanged. The two deadline modules and one nonce-bound
export assertion are the intentional existing-file edits.
