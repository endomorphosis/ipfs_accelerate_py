# Container coding qualification and live benchmark protocol

The newer joined native integration is documented in
[SUPERVISION_INTEGRATION.md](SUPERVISION_INTEGRATION.md): isolated local profiles,
pending acceptance, semantic capsules, world snapshots, learned retrieval and
actual Doctor transactions. `run_supervision_docker.py` runs its offline native
qualification. The original scripted and live component pilots below remain
separate experiments; none establishes a Terminal-Bench/native Codex advantage.

Fresh general supervisor routes default to `grok-4.7`, with `gpt-6.1-sol`
as the guarded Codex fallback. Existing quota, authentication, evidence and
effect-boundary gates still apply; changing the model does not authorize a
fallback. Newly issued route authorizations bind the new model pair. Old
signed route profiles must be reissued before selecting these defaults.
The container comparison explicitly pins `codex_cli` / `gpt-6.1-sol` in both
the supervisor arm and native Codex baseline. That matched-model profile does
not exercise the general supervisor's Grok route or its fallback.

Full-supervisor Codex deployment installs the pinned CLI through a source-owned
installer with a private npm configuration and cache. Distribution Node.js and
npm have a separate 300-second system dependency deadline, with TERM followed
by KILL after five seconds and a 330-second Harbor execution allowance. APT
retains one retry and 30-second HTTP/HTTPS network timeouts. Node.js must be
version 16 or newer. Unsupported versions fail before authentication or model execution.
The installer supports Linux ARM64 and x86-64, with independently pinned hashes
for both native executables. The native installer retains its separate shared
180-second deadline and 210-second Harbor execution allowance, bounded downloads
and output, and process-group cleanup on timeout or cancellation. A failed
system dependency step stops deployment before native installation or credential
transfer; it does not consume or extend the native installer deadline.
It checks package and executable versions before exposing the protected runtime.
Setup receipts contain versions, hashes and phase results without npm output or
inherited environment values. A failed setup remains a failed campaign attempt;
recovery uses a fresh preparation and retains the original failure.

For a lossless metadata comparison, hold `semantic_context_profile` and the
selected capsule population constant and vary `semantic_metadata_view` between
`legacy` and `common-bindings@1`. Opt-in trial receipts include
`semantic_metadata_dispatch`, which correlates each observed selection receipt
with the retained exact-input audit. Only `effective_treatment_observed: true`
establishes that every observed coding input used the factored representation.
A valid legacy fallback remains a completed attempt in the requested treatment
arm, with its provider usage included. Missing or conflicting observations do
not qualify. Compare repeated trials using native total-token counts and official
task success; prompt bytes and local tokenizer counts alone do not establish
end-to-end savings.

Native Codex receipts also include `interaction_telemetry` by default. This
passive collector records bounded tool input/output sizes, pairing status,
elapsed time between recorded events, and changes between adjacent cumulative
usage checkpoints. It exports no tool arguments, outputs, or call identifiers.
The first checkpoint has an unknown baseline; repeated snapshots add no usage,
and malformed or decreasing counters interrupt the observation sequence. These
checkpoints are not provider requests, and co-occurring tool events do not
establish per-tool token costs. String byte counts measure decoded UTF-8;
structured payloads use canonical JSON. Recorded elapsed time is not measured
tool execution latency.

Receipts retain at most 128 tool pairs and 256 usage checkpoints, with explicit
omission counts; the diagnostic pass examines at most 100,000 records from an existing
32 MB bounded rollout. Oversized payload measurements stay unknown. Native
aggregate usage remains separate from these diagnostic observations. A resumed
session, malformed record, invalid latest cumulative snapshot, or counter reset
cannot inherit an earlier completion signal. Fresh runtime archives are required
to qualify the new observations in live trials.

For an explicit output-preview experiment, prepare the full Codex arm with
`--tool-output-policy artifact-preview@1`. The default is `legacy`; default
configuration and planning inputs retain their existing bytes. The selected
coding advisory offers a bounded local output-artifact helper, with exact
retrieval available when more context is needed. Native CLI tools retain their
normal behavior. The selection is frozen into preparation and requires an
archive containing the current helper, policy, and invocation/audit sources.

`--tool-output-policy artifact-preview@2` selects an experimental compact helper
response. Its coding advisory requests `--response-format compact`; capture and
page responses use version-two schemas while internal capture receipts retain
version one. Compact responses carry one exact replay encoding and omit
retrieval hashes for streams already shown in full. The existing
`artifact-preview@1` selection retains its original helper response format.
For commands expected to return little output, the advisory favors ordinary
native tools so wrapper instructions and metadata do not outweigh the output.

The first version-one pilot did not show a token-saving benefit. Compact output
is a follow-up experiment, and smaller helper responses alone do not establish
lower provider usage. The default remains `legacy`; neither preview policy
intercepts native tools or guarantees that the worker adopts the helper.

A selected policy does not establish worker adoption or token savings. Check
the coding invocation's policy receipt and observed helper use separately from
the declared benchmark selection, and include any retrieval overhead in a
matched repeated comparison. Native provider token totals and official task
success remain the outcome measures; preview bytes alone are insufficient.

The opt-in helper requires Linux process and file primitives. Each explicit
capture runs argv once without stdin or a TTY, and enforces capture, store, and
time limits. Stdout and stderr remain separate byte streams; their original
interleaving is not reconstructed. SIGTERM/SIGINT cleanup is bounded; SIGKILL
or deliberate process-group escape remains covered by the existing worker and
container cleanup boundary. Native interactive tools remain available.

Complete captures support exact retrieval during that coding invocation. The
runner removes the temporary artifact store afterward and retains content-free
counts and binding hashes. These artifacts do not form a persistent proof
cache. Missing usage or incomplete cleanup remains explicit in qualification.

A separate source-representation experiment uses `--source-view-policy
source-view-raw@1` as the matched control and `--source-view-policy
source-view-compact@1` as treatment. Both offer the same bounded Python helper
and hash-checked symbol-edit protocol. Keep the semantic context profile,
metadata representation, tool-output policy, model, resource limits and task
inputs fixed. These policies require a fresh runtime archive; `legacy` remains
the default until provider-token and task-success evidence justifies changing it.

The helper indexes explicit visible Python paths, renders an AST-equivalent
view, and maps an edited definition back into its canonical file. Original
identifiers, strings and docstrings remain. An unchanged-AST round trip preserves
all original bytes; an accepted edit preserves bytes outside the selected
definition, checks the expected whole-file hash, validates syntax, and returns a
refreshed view in the selected format. Cooperating edits are serialized. The
hash check rejects stale snapshots; unrelated editors still need to coordinate
writes. AST equality does not prove behavioral equivalence under source
inspection or other reflective behavior, and edited programs need their normal
tests and acceptance checks.

The initial edit scope is unambiguous top-level functions/classes and direct
class members with a supported standalone layout. Whole-module replacement,
renames, nested functions, conditional definitions, non-Python files, hidden
paths and symlinks are unsupported. Normal tools remain available for required
original detail. This is a source view over the original checkout, not a
persistent obfuscated copy or a whole-repository compiler. Provider-owned tools
are not intercepted. Invocation receipts distinguish the advertised policy
from observed helper operations; local operation counts are advisory evidence,
not a security attestation or task acceptance authority.

Run the offline qualification on reviewed Terminal Bench public inputs with:

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_source_view_experiment \
  --dataset /path/to/terminal-bench-2 --output /path/to/new-evidence-directory
```

It copies the reviewed `llm-inference-batching-scheduler` and `tune-mjcf` inputs,
checks exact no-op round trips, exercises guarded edits and stale rejection,
and counts complete request/response traffic for matched read/edit/refresh
cycles. Public-input edits are harmless probes on disposable copies; an authored
fixture separately checks a deliberate behavior change. It does not read task
solutions or hidden verifier contents, solve the public tasks, run a provider,
or produce official rewards. A cached local tokenizer supplies proxy counts
when available; no tokenizer is downloaded. Smaller views can lose their token
advantage once index, mapping and edit overhead is included. Only paired live
trials with complete native usage and official rewards can establish practical
benefit. Include failed trials and workers that do not adopt the helper.
The indexed payload contains selected capsules, admissions, required raw support
and the context pack. Optional program raw source is already omitted. Some
capsule facets retain normalized AST projections, with nested definitions
elided by the producer; this is not another full copy of the repository. The
helper experiment adds guidance, and live totals include initial context and
later helper traffic. Measure component costs before choosing a replacement.

The separate `--source-input-policy` experiment puts verified code views in the
actual coding input, regardless of helper adoption. Its four opt-in settings
keep source selection fixed:

| Policy | Source rendering | Normalized facet AST in provider input |
| --- | --- | --- |
| `source-input-raw@1` | Raw selected source | Retained |
| `source-input-compact@1` | Compact selected source | Retained |
| `source-input-ast-ref-raw@1` | Raw selected source | Bound reference |
| `source-input-ast-ref-compact@1` | Compact selected source | Bound reference |

Compare the first two to isolate rendering, and the raw policies to isolate AST
replacement. Also include `legacy` to account for the added view instructions
and binding overhead. Compact rendering can be unprofitable. A compact arm must
include at least one actual compact view; empty or all-fallback treatments fail
before the provider call. Indexed selection can nominate only small constants,
so unsupported or unprofitable tasks must remain in the results.

Canonical capsules, CIDs, proof obligations, admissions and required raw sources
remain intact. A separately versioned provider presentation replaces only
allowlisted facet AST slots; retained source and producer evidence reproduce
the exact original transport. A selected module or class read includes nested
bodies that its normalized facet projection can elide; both rendering arms
receive that same source scope, and its added cost remains in the measurement.
Historical replay uses retained blocks and the bound renderer/Python version
after the allocated worktree is removed. Operational dispatch checks the complete captured source
manifest against both canonical and allocated files, including a final check
before the provider call. These checks establish representation and freshness,
not program correctness. Exact current source remains required before editing.

The source block is bounded to eight selected views and 32 KiB including framing,
and 4 MiB of retained source, with the existing 256,000-byte complete-input cap.
Legacy remains the default. The additive settings compose with metadata
factoring and native tokenizer fallback. AST-reference v1 explicitly rejects
metadata factoring and tokenizer selection; it never silently drops treatment.
Preparation freezes the setting and requires an archive containing matching
runtime sources. Existing provider cooldowns and resource limits still apply.

Run a model-free comparison of full captured coding inputs with:

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_source_input_experiment \
  --dataset /path/to/terminal-bench-2 --output /path/to/new-input-evidence-directory
```

This uses the real producer, DuckDB selection, router and retained-evidence
audit, with a local provider stub. It records complete-input bytes and optional
local tokenizer proxy counts across legacy and the four settings, checks stale
source rejection, and replays after worktree removal. It does not execute
benchmark source or measure model reasoning, later tool calls, provider tokens,
task success or official reward. Signed supervisor planning and container setup
are outside this qualification. Live comparisons require fresh frozen sources,
positive inclusion, complete native usage, official rewards and unchanged
resource/backoff controls.

Other public Terminal-Bench tasks can use the source-bound task profiles in
[TERMINAL_SUITE_PROFILES.md](TERMINAL_SUITE_PROFILES.md). The first broader full
supervisor pilot records two official passes and one native-execution timeout,
with source/checkpoint pins, token accounting, and readiness gaps for the rest
of the suite. It is a selected-task pilot with a distinct resource/time profile.

The [symbolic capability assessment](../../../docs/agent_supervisor/terminal_symbolic_capabilities.md)
documents the full arm's per-task gap report, signed separation of generated
harness support from program inputs, and bounded native progress diagnostics.
Re-prepare trials to use the isolated Python structural validator. These changes
are qualified with authored native repair cases; they do not supply new official
Terminal-Bench scores or establish a baseline advantage.

The [IntentIR planning improvement plan](../../../docs/architecture/TERMINAL_BENCH_INTENT_PLANNING_PLAN.md)
traces the current public instruction entry point and defines requirements,
symbolic planning, and admission milestones. The first local coverage milestone
is implemented: a source-bound datasets requirement ledger, independently
authored artifact/validation groundings, checked provider bindings, and signed
coverage retained through native task contracts. A bounded reviewed-operation
symbolic route now joins the existing obligation compiler, candidate planner,
critic and formal compiler. Task-specific worker requirements and read-only
public-check progress with bounded repair nominations are also implemented.
General prompt interpretation and live benchmark evaluation remain in the backlog.

The proposed [repository proof index and codebase IR plan](../../../docs/architecture/REPOSITORY_PROOF_INDEX_AND_CODEBASE_IR_PLAN.md)
adds current-code matching, bounded per-repository autoencoder training, checked
formal properties and DuckDB/DuckLake evidence storage. Its separate backlog
defines the path from IntentIR and source evidence into symbolic planning and
signed admission; this repository-proof pipeline remains proposed.

The first [codebase IR qualification experiment](../../../docs/agent_supervisor/terminal_codebase_ir_qualification.md)
now joins real supervisor preplanning, 358-function public Bottle autoencoder
training, source-bound Z3/Lean model checks and exact native DuckDB/DuckLake
metadata replay. The completed run retains 8,723 records across 28 families.
Finite loss reduction is measured; the declared stability criterion is unmet.
Learned semantic decoding, the complete logic-family matrix and authoritative
IntentIR-to-code planning remain open.

The subsequent [learned candidate experiment](../../../docs/agent_supervisor/terminal_codebase_decoder_qualification.md)
trains a fresh native production head on exact public source, reconstructs three
function ASTs and checks both header candidates through the reviewed native
model route. Its recorded cross-entropy and fixed finite tail criterion compile
in Lean; this is numerical/model evidence, with broader typed semantics and
optimizer convergence still unproved. Source, weights and complete producer
records replay through a separate native catalog referencing the initial run.

The [conditional evidence and IntentIR control experiment](../../../docs/agent_supervisor/terminal_codebase_intent_qualification.md)
adds exact native proof-key lookup, reviewed model nominations with all behavior
still residual, and frozen materials for a separate authored symbolic control.
The full public instruction stays unresolved and is refused by planning;
repository proof facts are not admitted through the administrative profile.

The [current repository catalog lane](../../../docs/agent_supervisor/terminal_codebase_repository_qualification.md)
rebuilds actual AST/KG/contracts and lexical vectors for the signed source forest,
replays complete frozen learned inventories, and consumes the byte-bound indexed
context in a fresh native symbolic selection. A private same-HEAD dirty edit
refuses the old index, evidence and model; a source-only successor agrees with a
cold rebuild. Behavioral requirements remain residual and successor admission
is not qualified.

Pass `--intent-requirement-contract /path/to/requirements.json` to
`full_supervisor_benchmark prepare`, the container supervisor, or its preparation
command, or supply `intent_requirement_contract` in `FullSupervisorAgent` kwargs.
The `intent-plan-requirement-contract@1` artifact
must name `.supervisor-instruction.md` and bind the exact public instruction.
Selecting it requires checked coverage; a rejected interpretation or missing
binding cannot silently use direct planning. The current profile supports atomic
native goals and retains unsupported compound scope. Optional autoencoder
advice remains separate. Local coverage does not establish correct translation,
task repair, or official reward.

A version 2 requirement artifact adds `symbolic_operations`: reviewed exact
native atom matchers bound to the complete signed task population, outputs,
checks and dependencies. It selects `intent_symbolic` and makes zero planning
provider calls. Selection, schedule, critique and formal effects are recomputed
at admission and retained in native pending contracts. Conditions, alternatives
and prohibitions are unsupported in this symbolic profile. The worker still
needs its qualified execution provider. Omitting a contract selects `direct`;
version 1 selects `intent_coverage`.

The coverage preparation command needs both the accelerate and datasets
packages available to the Harbor Python. Use a newly built runtime archive
containing these modules; the earlier source review snapshot identifies the
pre-implementation checkout.

For the current original-task comparison, use the Harbor environment's Python
with `full_supervisor_benchmark`, an explicit archive from `terminal_deployment`,
and a fresh trial directory:

```bash
PYTHONPATH=. python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark prepare \
  --arm full --dataset /path/to/terminal-bench-2 \
  --archive /path/to/full-runtime-archive --output /path/to/fresh-full-trial
PYTHONPATH=. python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark execute \
  --output /path/to/fresh-full-trial
```

Use `--arm no-index` with its matched source archive for the current combined
ablation, which disables both indexes and the symbolic Doctor;
`native_codex_baseline` uses Harbor's native Codex adapter. Each execution checks
the prepared hashes and runs once. The full arm includes cold index costs and
actual worker context; unsupported Doctor repairs abstain. Official reward,
native completion, cleanup, and all observed provider usage remain separate
measurements. See [INDEXED_ABLATIONS.md](INDEXED_ABLATIONS.md) for the larger
planned matrix and unmeasured parallel-worker comparisons.

### Provider cooldown and recovery

`full_supervisor_benchmark execute` enables provider admission by default for
both Codex and Grok profiles. It shares private state between processes of the
same OS user under
`${XDG_STATE_HOME:-$HOME/.local/state}/ipfs_accelerate_py/terminal-provider-backoff`.
The scope is provider-wide, so switching models cannot bypass a stop. Use
`--provider-backoff-directory /path/to/private-state` for an isolated campaign;
all cooperating launchers must use that same directory. Other entry points,
including direct router calls and `native_codex_baseline`, do not use this gate.

A retained rate-limit failure starts a 60-second cooldown, doubling on failed
recovery attempts to a one-hour local cap, with bounded positive jitter.
Structured relative reset hints can extend the wait to seven days. Longer
hints require an operator reset; they are never shortened into an early retry.
Text from a prompt, tool result, or free-form exception cannot set a delay.
The gate does not estimate quota capacity or issue proactive provider requests.

While blocked, execution writes `provider-deferral.json`, returns
`state: provider_deferred`, and exits with status **75** before invoking Harbor
or creating `invocation.json`. `--provider-max-wait-seconds N` optionally waits
up to 60 seconds before fresh work; the default returns immediately. Waiting
holds no state lock and rechecks the current cooldown. After expiry, exactly
one new trial holds the provider's recovery lock through its execution.
Other launchers defer. A verified provider return clears availability; it does
not establish task success, benchmark reward, or complete token accounting.

Planning and coding calls can modify a checkout before failing, so failed
trials are never automatically replayed or routed to another model/provider.
Already admitted healthy trials may finish. Billing, authentication, quota,
invalid stop metadata, and unverified recovery results require explicit
operator intervention. A crashed recovery probe remains blocked even after its
OS lock is released. Caught post-dispatch collection errors and failed state
commits also leave a durable uncertainty marker. Missing provider receipts
after recorded agent execution block admission; setup-only
failures without evidence of a provider attempt do not create a new cooldown.
State corruption or unavailable storage prevents new admission; no
cross-process durability can be guaranteed
if the filesystem cannot persist either the state or its fallback marker.

After resolving the provider or interrupted-trial problem, reset the shared
gate explicitly, then prepare a fresh trial:

```bash
PYTHONPATH=. python -m benchmarks.agent_supervisor.container_coding.terminal_provider_backoff reset \
  --provider codex_cli
```

For an isolated directory, add `--directory /path/to/private-state`. Reset
refuses an active recovery probe and requires readable, valid state. Preserve
and repair corrupt state before resetting; do not discard failed trial
receipts. Native session closure and `usage_accounting_complete` remain
separate observations: missing token counters stay `null`. CPU/memory
admission and external-pressure backoff still apply independently.

Offline regressions cover typed provider failures, the real native adapter
with an authored local CLI, the public benchmark CLI with an authored Harbor
fixture, and 32-process recovery contention. These checks establish dispatch
and recovery behavior without claiming a live provider benchmark or token
savings. Run the provider tests together with router diagnostics and the
existing Harbor, task-binding, and benchmark-comparison suites.

For a matched capsule-budget experiment, prepare two `--arm full` trials with
`--semantic-context-profile indexed-default@1` and
`--semantic-context-profile indexed-compact@1`. Both use current-source DuckDB
selection and a 32 KiB payload bound; their capsule limits are eight and two.
Indexed profiles use a short objective bound to the public instruction's SHA-256.
The complete instruction remains required raw context and the exact retrieval
query. This avoids repeating long instructions inside the objective and context
pack while preserving the payload limit and literal task requirements.
Capsule counts are ceilings: the byte limit can reduce both profiles to the
same count. Check retained counts before treating a task as a budget comparison.
The historical default `legacy-bounded@1` retains lexical selection. These
profiles do not change vector retrieval, intent transport, model settings, or
task instructions. Use one fresh archive and identical task/resource settings
for both trials. Configuration, archive sources, and the observed runtime
profile are checked before a comparison receipt can be complete. Input audits
may use up to 30 seconds of the remaining cleanup allowance; missing or timed
out audits remain unknown. Provider usage across all turns and official reward
remain necessary to assess total tokens per solved task.

Both supervisor model-coding routes now receive the original public instruction
verbatim through an owner-selected, signed source binding, in addition to the
planner's task summary. The router verifies the source and allocated worktree
before calling the provider and records the inclusion outside semantic
minification. This prevents public output requirements from depending on whether
the worker discovers an instruction file. The harness retains bounded before/after
copies of the declared public outputs for diagnosis. Native completion remains
separate from official benchmark correctness.

For requirement-backed manifest version 4, public-instruction version 2 also
delivers exact task requirements, native atoms, source spans, prerequisite
requirements, supported global prohibitions and signed outputs/checks. The
worker verifies the public graph and receipt replay without owner profile keys;
native persistence remains checked by owner launch. The terminal observation
report includes `intent_requirements` and `intent_requirement_repair`: current,
failed, stale, unobserved or unmeasured public evidence, missing outputs, and
validation/repair nominations within signed scope. Nominations require fresh
independent admission before execution. Source meaning and official reward
remain unresolved by these reports.

[SECURITY_CHECKPOINT_PLAN.md](SECURITY_CHECKPOINT_PLAN.md) describes the portable
SecurityIR weight format, Hugging Face publication, pinned ModelManager loading,
formal-candidate training and prover-checked planning milestones. It also defines
separate Doctor/index/autoencoder ablations; the current combined arm cannot
isolate their individual contributions.

The original `run_docker.sh` below is a zero-model-call **component qualification**,
not a Terminal-Bench score or an end-to-end supervisor evaluation. It imports the
production supervisor conflict scheduler, runs seeded Python repairs, checks
their outputs, and asks real Z3 and cvc5 executables to check finite task-state
invariants. The worker is an explicitly labeled scripted oracle. It does not
launch the implementation daemon, call a model, exercise worktree merging,
or demonstrate token savings.

From the ipfs_accelerate repository:

```bash
bash benchmarks/agent_supervisor/container_coding/run_docker.sh
```

The first build downloads a Python base image and Debian solver packages.
Execution has no network, credentials, Docker socket, or writable source mount.
Only the results directory is mounted writable; repair files live in temporary
container storage. The script runs regression checks before 12 trials (three
repetitions of each configuration). Pass a different output directory as the
first argument. Requires Docker; Linux arm64 was used for the initial run.

`artifacts/container_coding_qualification/qualification.json` records:

- Real repair-test results, final source hashes, scheduling waves, elapsed time,
  solver overhead, and individual solver verdicts bound to query hashes.
- Negative controls for unvalidated completion, unsatisfied dependencies, and
  overlapping writes. Unit tests isolate write conflicts from dependencies.
- Scheduler source hash, corpus hash, Python version, and both solver versions.
- Context payload **bytes**, with token savings explicitly `null`. The scripted
  oracle does not consume context, so byte reduction is only a payload-size
  observation, not evidence that a model can solve with the shorter prompt.

The adjacent image ID, source HEAD, and dirty status identify the local run.
The image installs distribution solver versions and uses a moving base tag;
retain the built image by ID for replay. The mounted source is the current
checkout, not a sealed source release. Existing unrelated edits are preserved.

## Experiment to answer the actual efficiency question

The companion `live.py` is a small router-backed coding pilot. It uses
`ipfs_accelerate_py.llm_router` and the efficient-route catalog shared with the
model manager. It selects an available API or authenticated CLI route once and
holds provider/model fixed across the four arms. Codex and Grok CLI adapters
are invoked through `llm_router`; the harness does not invoke either executable
directly. Cross-provider fallback, local model fallback, and response-cache
sharing between calls are excluded. This is still a component harness, not the full implementation daemon.

After building the image above, run from the repository root:

```bash
PYTHONPATH=. python3 benchmarks/agent_supervisor/container_coding/live.py \
  --preflight --provider grok_cli \
  --output artifacts/container_coding_qualification/router-preflight.json
PYTHONPATH=. python3 benchmarks/agent_supervisor/container_coding/live.py \
  --provider grok_cli --output artifacts/container_coding_qualification/router-pilot.json
```

Optional `--provider` and `--model` pin a router route. For example, use
`--provider grok_cli` or `--provider codex_cli` to use existing CLI logins. Configure credentials
through the existing router configuration on the **host**. Neither credentials
nor the host environment are passed to the generated-code containers. The host
calls the router; each arm's generated code executes in a separate networkless
Docker container. No model text is evaluated by the host Python interpreter.
The pilot makes at most 16 router calls and requests at most 512 output tokens
per call with a 90-second provider timeout. CLI adapters may not implement
the output-token cap, so the request count and timeout are the operative bounds. These are request bounds, not a
dollar-budget enforcement mechanism; provider-internal retries may add requests.

Native usage is captured from the selected router provider's chat response
or its thread-local CLI usage receipt when available, including usage from responses that later fail parsing or
validation. Missing usage stays `null` and prevents token-savings claims.
Text-only responses without native usage receipts leave token counts unknown. Savings
are calculated only when both compared arms solve the full corpus and all call
usage is known. The pilot has one repetition, simple tasks, and no repair retry;
use it to validate wiring before running a substantive study. Preflight records
`blocked` and exits 2 if no supported provider is discovered. Installed and
authenticated CLI routes are eligible; mocked/local fallback routes are excluded.

Use the same tasks, model/version, token budget, timeout, CPU allocation, and
initial repository snapshot in every arm. Run several paired repetitions:

| Arm | Workers per task | State checks | Context |
| --- | ---: | --- | --- |
| A | 1 | Existing supervisor validation | Full history |
| B | 3 | Existing supervisor validation | Full history |
| C | 3 | Additional solver-gated state | Full history |
| D | 3 | Additional solver-gated state | Obligation/delta capsule |

Keep benchmark-trial concurrency separate from workers cooperating on one
task. Increasing Harbor trial concurrency alone does not test collaboration.
Do not disable existing supervisor safety/acceptance gates for an ablation.

For live runs, collect provider-native usage across **all** planner, worker,
reviewer, failed, retried, and abandoned calls. Use the supervisor's
`self_improvement/supervisor_token_ledger.py` accounting contract; missing usage
is unknown, not zero. Cached input tokens remain a subset of input usage.
Report success rate, total tokens per successful task, paired wall time,
provider cost, CPU-seconds, solver overhead, retries, conflicts, and duplicate
work. Include failures in each arm's token numerator. Report success alongside
efficiency so dropping difficult tasks cannot look like an improvement.

The supervisor stores authoritative work state. Provers check whether a proposed
transition preserves an explicit invariant; they are not the state database.
Completion still needs the benchmark's independent verifier. The qualification
checks facts in concrete snapshots; it does not prove arbitrary Python programs,
the supervisor implementation, or correctness of a natural-language translation.
Two agreeing SMT solvers are not a kernel-checked Lean proof.

For production state transitions, bind receipts to the task, source tree,
obligation, dependency versions, solver version, and checker policy. Invalidate
on changed dependencies. `unknown`, timeouts, missing tools, stale receipts,
and disagreement must block proof-required transitions. Use existing supervisor
proof routing, receipt-cache, and conformance modules rather than promoting this
small benchmark checker into production authority. Introduce Lean or another
kernel checker only for obligations with a reviewed encoding and measurable
benefit; charge its translation/proof cost to the experiment.

## Terminal-Bench integration boundary

[Harbor](https://docs.harborframework.com/core-concepts/agents/pre-integrated-agents)
supports custom agents and container task environments. Pin a dataset revision
and task subset rather than relying on a moving `latest` alias. The
[Terminal-Bench documentation](https://github.com/harbor-framework/docs/blob/main/examples/terminal-bench.mdx)
describes running it with Harbor.

The next integration is a custom Harbor agent that:

1. Receives the task instruction and a clean task environment; provisions a
   pinned supervisor checkout and the selected provider inside that environment.
2. Uses `SupervisorControlService` to materialize the task workflow, and the
   existing implementation daemon/multi-lane runner for workers, leases,
   isolated edits, validation, and merging. It must not substitute this oracle.
3. Runs under a per-trial timeout and provider budget, retaining durable state
   and usage receipts. It returns the final working tree to Harbor.
4. Lets Harbor's verifier determine task success independently and joins that
   outcome to the supervisor token ledger. It preserves raw trajectories for
   audit and calculates paired comparisons across A–D.

The general installed-supervisor Harbor adapter is **not implemented by the
qualification runner**. A limited two-task workspace adapter is provided below.
The companion router pilot accepts both API configuration and existing Grok/Codex CLI logins; it is not a replacement
for the full daemon/Harbor evaluation. Select a spend limit and enforce it through
the existing usage coordinator before scaling beyond the small pilot.
The four local modes make the scheduling/proof/context plumbing testable now,
but their timings should not be extrapolated to model performance.

## Discovery regression and initial live result

The initial pilot incorrectly intersected router discovery with an API-only
allowlist. The router itself already discovered both authenticated `codex_cli`
and `grok_cli` providers on this machine. The harness now retains those routes,
reports the discovered provider list, and uses the correct CLI allocation path.
A second defect used a prefixed allocation session ID; Grok requires a UUID
because the router forwards this ID to its native session. Each call now gets
a fresh UUID, with regression coverage for both fixes.

CLI usage is reset in the calling thread before each dispatch and read even on
failure, preventing a timeout from inheriting an earlier call's token receipt.
Runs with task failures exit 1; blocked setup exits 2. Preflight readiness is
local discovery evidence, while a completed live run verifies actual dispatch.

Render a live result as Markdown:

```bash
python3 benchmarks/agent_supervisor/container_coding/report.py \
  artifacts/container_coding_qualification/router-grok-pilot-v2.json \
  --output artifacts/container_coding_qualification/router-grok-pilot-v2.md
```

## Native objective and implementation daemon pilot

`prepare_daemon.py` seeds a disposable Git repository with failing tests and
invokes the native objective daemon to create a root goal, two subgoals, and
three task packets. `daemon_run.py` invokes the production implementation
daemon, which owns dispatch, worktrees, validation, and merge processing.

Run from the supervisor source checkout:

```bash
PYTHONPATH=. python3 benchmarks/agent_supervisor/container_coding/prepare_daemon.py \
  --output artifacts/native-daemon-trial
IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok \
IPFS_ACCELERATE_AGENT_GROK_MODEL=grok-4.7 \
python3 benchmarks/agent_supervisor/container_coding/daemon_run.py \
  --repository artifacts/native-daemon-trial/repo \
  --state-root artifacts/native-daemon-trial/state --once
```

Omit `--once` to run continuously; interrupt with Ctrl-C. Each implementation
attempt has a 180-second timeout and each task a two-attempt limit. This is not
a dollar-budget cap. Optional `--plan-with-router` on preparation requests
Grok planning through the LLM router; inspect generation receipts because native
planning can fall back to deterministic plans.

The September 28 pilot generated three goals and three tasks. The native Grok
worker repaired `operations.py`, passed native validation, and merged commit
`9dc779bf47d7ba19950b5e310df9ec3277d99bc7d` through the merge queue. Both independent
Docker unittest cases passed after merging; both failed at baseline. BENCH-002
completed. A subsequent overlapping BENCH-003 attempt timed out and produced
no accepted patch; BENCH-001 and BENCH-003 remain pending. Preparation now tracks
discovery notes and bundles so Git worktrees receive that context. This setup
change was smoke-tested, but its effect on live follow-up completion is unverified.

This pilot uses the explicit legacy-Markdown task-source profile, with the
daemon on the host and a native Docker implementation worker. It is not the
activated Quack-backed prompt facade, a fully containerized supervisor, a
Terminal-Bench score, or a parallel efficiency measurement. Facade initialization
currently fails on an incompatible `repository_root` argument to
`initialize_local_profile`. All three router planning requests in this pilot
fell back to deterministic plans. The bounded daemon passes have exited.

Results and native receipts are under
`artifacts/container_coding_qualification/full_daemon/`, including
`daemon-pass-result.json`, `daemon-third-result.json`, and implementation logs.
The runner now forwards `--max-task-attempts` to its daemon constructors; the
focused regression assertion passes when invoked directly (the repository pytest configuration skips this module). The older runner test module cannot collect
because it imports the unavailable `IMPLEMENTATION_DAEMON_MODULE_SENTINEL`.

## Terminal-Bench workspace adapter

`harbor_agent.py` implements Harbor's `BaseAgent` interface for two selected
Terminal-Bench 2 coding tasks: `cancel-async-tasks` and `polyglot-c-py`.
`terminal_worker.py` generates a native root goal, subgoal, and task packet,
then runs one production implementation-daemon pass with Grok 4.6. Only a
merged deliverable is copied into Harbor's `/app`; Harbor runs the unchanged
official verifier afterward. Native validation uses a separate public-contract
smoke test, which is not benchmark completion authority.

The adapter intentionally accepts only those two tasks and requires an empty
initial `/app`. It does not silently discard files from arbitrary benchmark
environments. The public instruction is preserved, with an explicit mapping
from `/app` to the isolated Git workspace. Official tests and reference solutions
are never supplied to the implementation worker. The benchmark opts into
`IPFS_ACCELERATE_AGENT_GROK_TASK_TOOL_PROFILE=files`, retaining the sealed local
file-tool list and disabling shell, web, MCP, and subagent tools. The default
production terminal-enabled profile is unchanged. Generated context references
are made workspace-relative before board registration and Git commit. The host daemon and its native
Docker worker are separate from the Harbor task container; this profile cannot
measure general interactive terminal capabilities or environment provisioning.

Example from the outer `lift_coding` workspace:

```bash
uv venv .venvs/terminal-bench-harbor --python 3.12
uv pip install --python .venvs/terminal-bench-harbor/bin/python 'harbor==0.23.0'
git clone https://github.com/harbor-framework/terminal-bench-2.git .benchmarks/terminal-bench-2
git -C .benchmarks/terminal-bench-2 checkout 2fd12b88aafdd04a52c298e3940bcb189f9766d6
python3 external/ipfs_accelerate/benchmarks/agent_supervisor/container_coding/terminal_run.py \
  --harbor .venvs/terminal-bench-harbor/bin/harbor \
  --dataset .benchmarks/terminal-bench-2 \
  --jobs-dir artifacts/terminal_bench_supervisor \
  --job-name small-serial-pilot
```

The calling Python needs the supervisor dependencies and the existing Grok
credentials. Harbor uses its separate virtual environment. The launcher uses
`--force-build` because the dataset's published x86-64 images do not execute on
this ARM64 host; it builds the unchanged upstream Dockerfiles for the host.
Each task gets one 300-second implementation attempt; the outer subprocess has
a 480-second limit. These are time/attempt bounds, not a hard dollar cap.

Use a fresh job name for every run. `--concurrency 2` runs independent Harbor
trials concurrently; that measures inter-task concurrency, not the supervisor's
internal multi-agent decomposition. The corrected pilot runs two trials concurrently. Do not claim
speedup or token savings without paired repetitions under matching conditions.
A changed adapter constitutes a different configuration, not another identical
benchmark attempt.

`terminal_report.py JOB_DIRECTORY` creates `summary.json` and `summary.md` with
official rewards, infrastructure errors, environment/agent/verifier timings,
and raw native usage-event counters. Cache/input semantics are preserved rather
than guessed, and normalized tokens/cost remain unavailable. Harbor's zero exit
status can include errored trials; `terminal_run.py` returns 2 for infrastructure
errors, 1 for verifier failures, and 0 only when both tasks pass.

Adapter boundary tests (no model calls):

```bash
PYTHONPATH=external/ipfs_accelerate/benchmarks/agent_supervisor/container_coding \
  .venvs/terminal-bench-harbor/bin/python -m unittest -v test_terminal_adapter
```

An initial unrestricted diagnostic run was invalidated after the async worker
attempted to retrieve upstream tests and a reference solution. Its logs and
`INVALID_FOR_COMPARISON.json` are retained. It must not be used as a baseline
against the file-only rerun. Prior architecture/adapter setup errors also remain
in their original job directories; neither made model calls.

Each trial gets a distinct native goal/task prefix derived from its artifact
path. This avoids reusing the supervisor's persistent attempt counters and
completion receipts across independent trials. A no-dispatch daemon pass is
reported as an infrastructure error, not as a model attempt. Preparation can
be exercised without model calls using `terminal_worker.py --prepare-only`.
