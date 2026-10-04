# Shared datasets and supervisor proof resources

`execute_datasets_native_bundle` sizes a shared reservation from the tasks and
uses the datasets host scheduler by default. Its lower-level bridge,
`open_datasets_prover_lease`, connects the supervisor's existing task executor to
that scheduler. Both entry points use resource safety by default:
one host reservation contains the supervisor, its admitted tasks, their Hammer
portfolios and the native solver children. A supplied datasets parent produces
one nested reservation. CPU, memory and process capacity are not charged to a
second independent host budget.

The task-sized entry point considers at most four workers by default and reduces
that width to fit the declared task costs and configured capacity. The lower-level
bridge's default budget instead allows at most four workers with 512 MiB per
worker, within the scheduler's configured capacity and lane reservations. Larger
tasks such as Isabelle should use the task-sized entry point or an explicit
`MultiProverResourceBudget` with a sufficient bounded envelope. Default host
policy preserves headroom and observes memory, CPU, I/O and kernel-task pressure,
including visible container limits. Each child admission rechecks current
pressure and waits through cooldown until capacity returns, cancellation arrives or admission time
expires. This reduces new work under pressure; it does not suspend all existing
processes. An explicitly injected scheduler retains its supplied policy.

## PID and thread pressure

Default proof admission also checks available kernel-task headroom from visible
cgroup-v2 ancestor `pids.max`/`pids.current` limits and, when available, the host's
`threads-max` and `/proc/loadavg` task count. These counts include OS threads.
The gate conservatively budgets 16 kernel tasks per reserved process slot plus
32 tasks of headroom. It counts root reservations once; nested descendants do
not add another copy of their parent's envelope. Live usage and reservations
can overlap in this estimate, making admission intentionally conservative.

Insufficient headroom records `proof_pid_headroom` and uses the existing shared
cooldown, recovery, cancellation and deadline path. A finite known limit with
malformed or unreadable usage refuses admission through the telemetry backoff;
absent optional PID telemetry retains the other resource checks. The sampler
does not infer free capacity from `RLIMIT_NPROC`, which would require a complete
real-UID thread census and privilege/namespace handling.

The 16-task credit is an estimate, not an OS thread limit for JVM/ML helpers.
This increment leaves configured process caps, explicit budgets and persisted
scheduler configuration unchanged; the qualified host still has 16 process
slots. Existing long-lived clients acquire the new gate after reloading the
updated code. A calibrated larger helper-process policy and a coordinated
persisted-policy migration remain future work.

## Run native SMT tasks

Install both packages and the requested native solvers in the same environment.
The following example checks an authored propositional assertion with Z3 and
CVC5. It does not install solvers or reinterpret the assertion as repository
behavior.

```python
from ipfs_datasets_py.logic.hammers import semantic_routing
from ipfs_accelerate_py.agent_supervisor.proof.datasets_hammer_tasks import (
    make_datasets_smt_task,
)
from ipfs_accelerate_py.agent_supervisor.proof.datasets_native_bundle import (
    execute_datasets_native_bundle,
)

parser = semantic_routing.modal
parsed = parser.parse_modal("p and not p", parser.profile_k())
assert parsed.ok
replay = parser.parse_modal(parsed.printed, parser.profile_k())
target = dict(
    request_id="contradiction", source_construct="authored-assertion",
    logic_family="propositional", ast_format="shared_logic",
    printed=parsed.printed, native_ast=replay.root.to_dict(),
    solver_names=("z3", "cvc5"),
)
task = make_datasets_smt_task(target=target, expected_verdict="unsat")
execution = execute_datasets_native_bundle([task])
receipt = execution.receipt
print(execution.plan.to_dict())
```

This adapter requires an actual bridged child at execution. It reserves 128 MiB
for Python coordination plus 256 MiB per concurrent solver by default. The outer
supervisor distributes independent tasks; each task runs its two solvers
sequentially unless `max_parallel_processes=2` is explicitly requested. Per-task
parallelism must fit the enclosing budget. Target AST/parser/capability replay
precedes execution. Target inputs are copied when the task is prepared.
Complete serialized targets have a 1 MiB cap; the native 4,096-node, 64-level
and 64 KiB printed-source bounds still apply.

Every requested solver must return the explicit `sat` or `unsat` expectation.
Missing executables, refused attempts, cancellation, errors, unknown answers and
opposite verdicts cannot unlock dependent tasks. A failed result retains a
bounded verdict ledger through `ProverTaskFailure`. The adapter disables result
cache bypass and early winner cancellation. Successful task completion means
the expected **raw solver observations** were collected. Source correspondence,
kernel proof, behavioral facts and task-completion authority remain false.

`dependencies` refers to the existing supervisor's AND readiness rule. This is
not a new proof-composition algebra. The adapter deliberately supports only the
checked propositional Z3/CVC5 route. The separate closed Lean and Isabelle
profiles below add kernel execution. ATP conjecture validity, general Lean,
Rocq, general Isabelle, general TLA+, modal/deontic and other families still need
faithful native adapters and evidence gates.

## Dependency-ready execution

`plan_datasets_native_bundle(tasks, max_workers=4)` exposes the task-sized plan
without launching native work. It accepts up to 64 tasks and a requested width
of 1 through 32. For each resource dimension, it reserves the sum of the largest
requests at the selected width, then reduces that width until every possible
subset fits. This conservative calculation can reserve more than a particular
dependency graph needs. It is distinct from the supervisor's adaptive packing
of ready tasks inside the admitted envelope. Plans describe costs; only fresh
admission grants capacity. Provider/model work requires explicit accounting and
is rejected by this convenience entry point.

The supervisor refills its bounded worker queue as tasks finish. A completed
dependency can unlock its successor while an unrelated slow task continues.
Ready-task packing accounts for each task's CPU, memory, process, thread, model,
artifact, disk and provider requirements. Submitted tasks awaiting admission
also count in the forecast, preventing later work from taking their capacity.
Actual local and native admission remain authoritative and observe live pressure.
An in-process task can use free capacity while all native process slots are busy.
Receipt order still follows input order, and failed dependencies block their
dependent branch.

Healthy scheduler reads retain their lock, fresh state read and owner-liveness
checks. An unchanged canonical state skips the durable rewrite; actual changes,
including renewals, cancellation and recovery, retain the fsync/atomic-replace
path. Owner probes are reused only within one locked operation. There is no
cross-call state or liveness cache.

## Run a closed native Lean task

`make_datasets_lean_nat_task` generates and checks only
`forall n : Nat, n + k = k + n`, with an exact integer `k` from 0 through 65,535.
It uses the installed Lean executable and `Init`; install the toolchain before
calling it. A task cannot supply arbitrary Lean source or replace its runner.

```python
from ipfs_accelerate_py.agent_supervisor.proof.datasets_kernel_tasks import (
    make_datasets_lean_nat_task,
)

kernel_task = make_datasets_lean_nat_task(
    task_id="nat-commutativity", offset=7, dependencies=("contradiction",),
)
receipt = execute_datasets_native_bundle([task, kernel_task]).receipt
```

Both the version probe and the fresh kernel invocation use the admitted native
child, cancellation, a remaining wall deadline, CPU limits and bounded output.
The default task reserves 128 MiB for coordination plus a 512 MiB sampled native
process-tree RSS limit. Lean uses one worker and a 4 GiB virtual-address-space
cap so its runtime can reserve thread stacks; this virtual cap is separate from
the lower RSS limit. Sampled enforcement can overshoot between checks and is
not an aggregate cgroup guarantee.

Acceptance requires fresh successful native execution, no `sorry`, an empty
axiom report, and exact source/theorem/request/toolchain/result bindings. A
replayed packet or contradictory typed-result metadata is rejected. The emitted
`kernel_authority` concerns only this generated Nat theorem under the installed
Lean toolchain. Source correspondence, repository behavior, execution permission
and task-completion authority remain false. Scheduling SMT then Lean through an
AND dependency demonstrates shared execution and accounting; it supplies no
logical correspondence between their statements.

## Run a closed native Isabelle task

`make_datasets_isabelle_nat_task` generates only the theorem
`forall n : nat, n + k = k + n`, for an exact integer `k` from 0 through 65,535.
It imports `Main` and uses `by (rule add.commute)`. The reviewed managed Isabelle
runtime and its HOL heap must already be installed; this task accepts no
arbitrary theory, caller-supplied runner or task-time installer.

```python
from ipfs_accelerate_py.agent_supervisor.proof.datasets_isabelle_tasks import (
    make_datasets_isabelle_nat_task,
)
from ipfs_accelerate_py.agent_supervisor.proof.datasets_native_bundle import (
    execute_datasets_native_bundle,
)

isabelle_task = make_datasets_isabelle_nat_task(
    task_id="isabelle-nat-commutativity", offset=7,
)
execution = execute_datasets_native_bundle(
    [isabelle_task], timeout_seconds=180,
)
receipt = execution.receipt
```

The default request reserves 3 CPU slots, 3 computation-thread slots, 12 process
slots and 2,304 MiB: 256 MiB for coordination plus a 2 GiB sampled native
process-tree RSS guard. Each native process has a separate 32 GiB address-space
limit so Poly/ML can reserve virtual heap and stack space. Private settings cap
the JVM heap at 256 MiB and the ML heap at 1,024 MiB, and select single-threaded
ML proof work and garbage collection. Bootstrap JVM options are bounded too.
Computation-thread slots are accounting estimates, not a ceiling on JVM/ML
service threads; process reservations are not a hard OS process-count limit.
Sampled RSS enforcement can overshoot and supplies no aggregate cgroup guarantee.

On the qualified host's 16-process policy, a bundle containing multiple Isabelle
tasks is reduced to width one. A bundle with one Isabelle task and lighter Lean
or TLC tasks can overlap when the other dimensions fit. The conservative
largest-subset planner may still serialize a mixed bundle containing several
Isabelle tasks even when a particular light/heavy pairing could fit.

Version discovery and kernel checking each acquire a fresh native child under
the actual bridged task lease. New external pressure between these phases can
therefore defer the kernel launch. Both phases retain the same cancellation and
remaining task deadline, and recheck them after admission. Private versioned
user-home settings and bootstrap JVM limits are shared with installed-runtime
preparation; neither phase substitutes a separate host reservation. Acceptance
binds the generated source, theorem, runtime and fresh native result, and
requires the theorem's no-oracle audit marker. This is
kernel evidence under the trusted Isabelle/HOL installation. The audit does not
establish that HOL is axiom-free or attest every installed heap and library.
Repository-source correspondence, behavior, cross-family proof composition,
execution permission and task-completion authority remain false.

## Prepare an installed Isabelle runtime

`prepare_isabelle_runtime` observes an existing managed installation with default
shared admission. Its modes distinguish command availability, a no-build HOL
preflight and an optional fixed `True` theorem smoke check:

```python
from ipfs_datasets_py.logic.backends.installers.isabelle_preparation import (
    prepare_isabelle_runtime,
)

preparation = prepare_isabelle_runtime(mode="hol", timeout_seconds=120)
if not preparation.usable:
    raise RuntimeError(f"{preparation.status}: {preparation.reason_code}")
print(preparation.to_dict())
```

The default reservation is 3 CPU slots, 12 process slots and 2,304 MiB, with the
same 2 GiB sampled RSS and 32 GiB per-process virtual limits as the closed Nat
task. One deadline covers static runtime checks, admission and all probes. Each
version/help/HOL/smoke phase acquires its own child and observes live pressure.
Supply an existing `parent_lease` without `scheduler` to reuse its authority;
the parent must contain the complete preparation envelope. An explicit
`install_root` or `executable` selects the installation without a PATH fallback.

The API performs no download or explicit heap build. The `hol` and `smoke` modes
use `build -n` to check the installed HOL heap before proceeding. If the trusted
installation changes afterward, `process_theories` can attempt a bounded private
rebuild. Selected runtime identities are checked again before a usable result
is returned; the observation does not attest every heap/library or grant proof,
repository, behavior or execution authority. Tasks still require fresh checks.

Default setup inspection and smoke now use this admitted preparation route:

```bash
python -m ipfs_datasets_py.logic.external_provers.isabelle_setup --smoke --timeout 120
```

`ensure_isabelle_ready()` uses the same route when neither installation nor an
explicit HOL build is requested. Its setup report distinguishes readiness levels
and retains the bounded preparation receipt; a late failure cannot promote an
earlier successful probe. Explicit `--install` now uses the bounded installation
API below. Explicit `--build-hol` now uses that same owner to rebuild persistent
system heaps in unpublished staging, as described below.

The Isabelle archive installer separately streams downloads and cached-file
hashing with cumulative byte limits, and extracts only bounded gzip/plain tar
streams into private staging. Metadata, decompressed bytes, member counts and
paths have caps; live disk checks and cooperative cancellation apply during
copying. It rejects unsafe links and unsupported member types, validates gzip
trailers, and uses private launcher temporary files. Ordinary publication
failures restore the prior runtime and launcher. These controls do not make
filesystem I/O/cleanup latency hard-bounded, provide aggregate RSS enforcement
for the complete installer, or make multi-path publication crash-atomic.

## Run a closed finite TLC task

`make_datasets_tlc_counter_task` checks the generated finite counter that starts
at zero and increments to `counter_bound` (1 through 64). The requested invariant
is `count <= invariant_max`, where `invariant_max` ranges from zero to the bound.
The task succeeds only when native TLC completes the check with exactly
`counter_bound + 1` distinct states. A smaller invariant produces a retained
counterexample and a failed task; dependent tasks remain blocked.

```python
from ipfs_accelerate_py.agent_supervisor.proof.datasets_tla_tasks import (
    make_datasets_tlc_counter_task,
)

counter_task = make_datasets_tlc_counter_task(
    task_id="finite-counter", counter_bound=16, invariant_max=16,
)
receipt = execute_datasets_native_bundle([counter_task]).receipt
```

This Linux profile requires the reviewed user-local TLC JAR and its managed
manifest selecting an eligible installed Java runtime. It checks the pinned JAR
bytes and uses a private copy; caller-supplied source, launchers and result
packets are not accepted. Java and TLC probes run after admission through the
same bounded runner as the model check. No installation runs inside the task.

The default request reserves 640 MiB, including 128 MiB for coordination. Native
execution uses a 512 MiB sampled RSS guard, a separate finite virtual-address
cap, a small explicit JVM heap, one TLC worker, and the remaining task CPU/wall
budget. The enclosing lease must contain that request. Cancellation and resource
exhaustion suppress success, and cleanup retains the shared reservation until
native execution drains. Exact native/result bindings and state counts reject
replayed, contradictory or incomplete outcomes.

The compiler's generic liveness property is explicitly removed in this versioned
safety-only profile, and terminal deadlock checking is disabled. The result makes
no liveness, fairness, deadlock-freedom, unbounded theorem, repository-behavior or
execution/completion claim. An AND dependency across SMT, Lean and TLC controls
execution order; separate logical correspondence obligations remain necessary.

The generic datasets TLC/Apalache runner also carries request CPU/RSS/virtual
limits into model and version launches, sharing one deadline and cancellation
signal. Legacy JVM launchers must size their heap to fit those limits; an
unconstrained JVM can fail allocation. Constructor discovery and lazy installer
probes remain separate legacy setup operations. The closed adapter avoids them
by supplying its already bounded native capability check.

The state-model installer's downloader now streams 64 KiB chunks with a default
512 MiB artifact cap, cumulative limits during cached-file hashing, and checksum
and length checks before atomic replacement. Failures clean private partial files
and preserve the existing destination. Cancellation and deadlines are cooperative
between reads; DNS, HTTP headers and chunk framing can delay checks. This change
does not provide shared admission for the whole installer, bound its separate
runtime probes, or guarantee crash-durable directory publication.

## Prepare an installed TLC runtime

Use the separate bounded preparation API to check an already installed managed
TLC runtime before scheduling tasks. Default admission is enabled; no explicit
scheduler or resource lease is required.

```python
from ipfs_datasets_py.logic.backends.installers.state_model_preparation import (
    prepare_tlc_runtime,
)

preparation = prepare_tlc_runtime(timeout_seconds=15)
if not preparation.usable:
    raise RuntimeError(f"{preparation.status}: {preparation.reason_code}")
print(preparation.to_dict())
```

The default preparation reservation is one CPU slot, one process slot and
384 MiB, including 128 MiB for coordination and a 256 MiB sampled native RSS
guard. One deadline covers admission, bounded file checks, Java discovery and
TLC help. Each probe acquires a fresh child, so newly arriving external pressure
can trigger admission backoff between probes. Cancellation also interrupts
admission waits. An existing native lease can be supplied as `parent_lease`;
when using a parent, omit `scheduler`.

Preparation checks bounded regular files, the reviewed JAR digest and managed
manifest, then uses a private JAR copy and explicit JVM limits for its probes.
It performs no download, installation or model check. A usable result is a
historical runtime observation, not proof authority or a substitute for the
task's fresh checks. Legacy installer and constructor probes retain their
separate setup limitations.

## Share an existing parent

Pass `parent_lease=existing_datasets_lease` to
`execute_datasets_native_bundle` or the lower-level bridge. The task-sized plan
then fits within that parent's envelope and execution acquires one nested
reservation. For a source scan or a
training owner that already accepts a datasets parent, pass
`lease.datasets_parent_lease` as that operation's parent. It must acquire its
own child and fit the same envelope. An executing bridged task receives
`context.lease.datasets_parent_lease` and `context.lease.datasets_scheduler`.
Use that admitted child for nested native work. A router runner likewise
receives its admitted child as `request.resource_lease` when configured with a
bridged resource lease.

The bridge accepts real native lease objects. Tokens and unrelated supervisor
budget objects do not create authority. Low-level datasets token/keyless APIs
remain trusted local-owner interfaces; the bridge is not a security boundary
against another process able to edit the scheduler's state file. Legacy
standalone supervisor accounting remains available and does not automatically
become host-wide accounting.

## Cancellation and release

The bridge's overall deadline includes root admission; task deadlines include
child admission. Task cancellation also interrupts a pressure wait. After
admission, deadlines and cancellation are checked again before launch.

Closing or expiring an ancestor revokes new work and signals descendants. The
datasets scheduler retains its reservation until live descendants acknowledge
release. An expired but OS-live owner remains charged; a heartbeat cannot
resurrect revoked authority. An OS-dead owner may be recovered after its live
descendants drain.

Supervisor callable and command execution holds prevent early reclamation.
Callables must observe cancellation and join work they start before returning.
Python threads cannot be forcibly killed: an uncooperative live callable can
retain capacity indefinitely. A timeout receipt can therefore precede final
cleanup; it never turns a late result into a cache success. OS-dead-owner
recovery alone does not guarantee that an orphaned native process has exited.

The Linux Hammer lifecycle uses a trusted native `prlimit` executable before
exec, avoiding Python pre-exec hooks in pool threads. Solvers and version probes
receive CPU/address-space limits, bounded wall time and cancellation. Probe time
is deducted from the solver wall budget. Missing native limit setup fails
closed. Reservations for Python coordination remain estimates; address-space
limits are per native process, not a cgroup-enforced aggregate RSS ceiling.

## Qualification

The retained [execution report](../../../ipfs_datasets/workspace/shared-prover-qualification-20261002/qualification.md)
records native checks, controlled pressure injection, lifecycle stress and the
1/2/4-worker benchmark. Pressure tests inject telemetry rather than exhausting
the host. Benchmark timings describe this small fixture on a shared machine.
They do not establish general many-core scaling or mixed-family proving.

The subsequent [scaling and Lean qualification](../../../ipfs_datasets/workspace/shared-prover-scaling-qualification-20261002/qualification.md)
retains the scheduler microprofile, queue race regressions, native mixed SMT/Lean
tests and a deterministic larger SMT benchmark. At 1/2/4 workers, median execution
times for eight Tseitin tasks were 21.57/14.89/8.91 seconds: 2.42× execution speedup
at four workers. Including target preparation, the median totals were
27.36/20.63/14.63 seconds (1.87×). The tiny fixture still slowed as width grew.
These measurements qualify the recorded workloads and resource limits, not
general repository proof throughput or scaling beyond four workers.

The [TLC and mixed-family qualification](../../../ipfs_datasets/workspace/shared-prover-tlc-qualification-20261002/qualification.md)
records 624 passing tests and 144 native checks across Z3/CVC5, Lean and TLC.
The 1/2/4-worker mixed workload takes 11.76/7.77/6.93 seconds in median execution
time, a 1.70× speedup at four workers (1.52× including preparation). Selected
source, Java/JAR, manifest and launcher identities match before and after the
benchmark. Small authored fixtures, sampled resource accounting and distinct
result authorities remain explicit limits; this is shared execution evidence.

The [Isabelle and task-sized execution qualification](../../../ipfs_datasets/workspace/shared-prover-isabelle-qualification-20261002/qualification.md)
retains the native footprint measurement and qualification of the closed
Isabelle profile, task-sized admission and bounded installed-runtime preparation.
It records 716 passing tests and 153 native checks in nine paired trials. Under
one held shared reservation, four workers give 2.44× execution speedup or 2.17×
including task preparation; parent preparation/admission costs are recorded
separately. Two earlier attempts safely deferred launches when competing owners
held process capacity. All owned leases drained after the final population.
The closed profiles do not qualify general mixed-family proof composition or
public service activation. Admission for complete legacy installer workflows and
their separate probes remains open work.

The subsequent [installer and readiness qualification report](../../../ipfs_datasets/workspace/shared-prover-installer-qualification-20261002/REPORT.md)
records **900 passing tests with zero failures, errors or skips** for the
PID-pressure gate, bounded Isabelle readiness, fresh child checks, archive
controls and supervisor regressions described above. One fresh local HTTP installation of retained
official archive bytes completed in 13.228 seconds; warm reuse took 0.891 seconds.
Bounded kernel smoke and cleanup checks passed. Consult that report for final
suite counts, source hashes and retained failed attempts. Complete installer
admission was outside that historical experiment; the subsequent installation
milestone below covers the default Isabelle route. Broader production gates remain open.

The same qualification repaired two supervisor races. Native command results
recheck cancellation and the deadline before publication or success caching,
including a process that exits before the monitor next checks its signal.
The ready queue also reconsiders capacity after an in-flight admission finishes;
it does not wait for that task to finish before starting an eligible sibling.
Actual child grants still enforce the shared envelope. Deterministic timing
regressions and real process cleanup checks cover these boundaries.

The datasets `SolverPortfolio` now has one finite `overall_timeout_seconds`,
defaulting to `policy.hammer_policy.timeout_seconds` (30 seconds by default).
It includes root/child admission, queued attempts, version probes and execution.
The optional `resource_wait_timeout_seconds` caps each wait further; omission
uses the remaining overall time, and zero retains immediate admission attempts.
Per-solver timeouts still cap admitted work, and late conclusive outputs cannot
be promoted. Injected Python callbacks remain cooperative. The bounded
overall deadlines of the bundle and installed-runtime preparation APIs should
not be assumed to apply to every older portfolio entry point.

The subsequent [production installation qualification](../../../ipfs_datasets/workspace/shared-prover-installation-runtime-qualification-20261002/REPORT.md)
uses `ensure_isabelle_installation` directly. Default setup installation, the
ordinary lazy bridge and reviewed registry dispatch now use this API:

```python
from ipfs_datasets_py.logic.backends.installers.isabelle_installation import ensure_isabelle_installation

receipt = ensure_isabelle_installation(yes=True, timeout_seconds=600)
assert receipt.usable, receipt.to_dict()
```

One shared 3-CPU/12-process envelope covers the cancellable installer lock,
archive staging, fresh staged kernel smoke and fresh published kernel smoke.
Warm reuse also runs a fresh bounded smoke. Each worker/native phase receives a
fresh pressure-checked child. Downloads, including unfinished partials, remain
within controller-owned staging; only a verified archive replaces the cache.
The controller retains the previous tree and launcher until final validation,
and restores them on ordinary failure or cancellation. Large cleanup runs in a
bounded child; `cleanup_pending` reports retained paths if cleanup cannot finish.
Publication is not crash-atomic, and rollback filesystem calls can outlast the
deadline. RSS monitoring is sampled; process reservations are not kernel task
ceilings. Direct legacy installer calls, explicit custom installers, lazy outer
lock waits and frontend discovery retain their
separate lifecycle limits. These runtime receipts confer no source/IntentIR or
cross-family proof authority.

Explicit persistent HOL builds use the same controller transaction:

```python
receipt = ensure_isabelle_installation(
    yes=True, install_root="/path/to/isolated/isabelle",
    build_hol=True, allow_download=False, build_memory_mb=6144,
)
assert receipt.usable and receipt.hol_build["persistent_heap_published"]
```

Build-only setup requires the pinned archive at
`<install_root>/downloads/<official archive basename>`; for Linux ARM this is
`Isabelle2025-2_linux_arm.tar.gz`. It does not search workspace caches or fetch
bytes unless installation/download is explicitly permitted. Setup makes one
owner call for `--install` and/or `--build-hol`, with default total deadlines of
600 seconds for installation or 3,600 for a build; inspection remains 120 seconds.
The staged HOL heap is removed while Pure is preserved. Only a successful new
build, fresh staged/published smoke and exact published heap verification allow
the controller to report persistent publication.

The [native build measurement](../../../ipfs_datasets/workspace/isabelle-hol-build-qualification-20261002/official-hol-build-6g/result.json)
used the unchanged shared pool with four CPU credits, twelve process credits and
6,400 MiB. The default build profile gives 6 GiB sampled RSS, a 2 GiB JVM heap,
3 GiB ML heap and 1 GiB other-memory allowance. The native build worker completed
in 286.896 seconds; cold transaction/warm verification took 329.660/10.319 seconds,
and total benchmark time was 347.482 seconds. Sampled descendants peaked at
5,222.87 MiB, twelve processes and 85 OS threads; all observed descendants and
owned leases drained. Controller RSS was measured separately and is not capped
by the 256 MiB coordination allowance. This is one build, not a many-core speedup
or aggregate cgroup guarantee. The
[final qualification](../../../ipfs_datasets/workspace/isabelle-hol-build-qualification-20261002/REPORT.md)
records 1,217 distinct cases with passing latest results and zero latest errors
or skips. Raw execution retains 1,219 runs, including two timeouts that passed
exact unchanged-source retries without relaxing limits. One was confirmed at
30-second root admission; the 60-second preparation deadline has truncated
original diagnostics and no more precise boundary claim. The smaller-JVM
failed attempt and separate recovery are retained. General
source/IntentIR correspondence, proof composition and public activation remain open.

The separate [native cancellation control](../../../ipfs_datasets/workspace/isabelle-hol-build-qualification-20261002/native-cancel/result.json)
passed sixteen checks in 32.087 seconds after observing stable Java and two
Poly/ML processes in three distinct groups/sessions. All fifteen sampled
PID/birth identities stopped before installation returned; no runtime was
published. A separate admitted, locked cleanup with a fresh signal removed the
stage while preserving the original cancelled receipt. Observed ancestry and
birth tracking cover this fixture, not hostile daemon escape or aggregate
cgroup containment.

## Default public Isabelle frontend and reconstruction

`IsabelleFrontend` and `IsabelleReconstructor` now use the admitted installed
runtime owner for capability, capture and reconstruction. Their optional
`parent_lease`, `scheduler` and `cancellation` arguments preserve one shared
authority; omitting them selects the default pressure-aware owner. Capability
discovery never installs. Explicit frontend `auto_install=True` enables its
first-use setup; reconstruction additionally requires the request's
`policy.network_allowed=True`. Supplied theories and installed components remain
trusted native programs.

The [public-adapter benchmark](../../../ipfs_datasets/workspace/isabelle-hammer-execution-qualification-20261002/native-public-input-bounds/result.json)
passed twenty checks in 51.014 seconds. Its native calls captured and reconstructed
a true and a false statement, accepting only the true theorem. An already-set
cancel signal admitted no work; a separate capture was cancelled only after two
samples observed its native theory child and live Poly/ML. All recorded native
workspaces, sampled live descendants and owned leases drained. Twenty-two native
phases used distinct child leases: twenty-one completed and one was cancelled.
The selected run observed no backoff or queued request. The retained earlier
71.395-second run recorded nineteen shared memory-pressure backoff observations
under its prior source generation; neither run changed the shared policy.
These observations are not an independent stress or host-exhaustion experiment.
The selected generation also rejects oversized source before unbounded
instrumentation and cleans observed subprocesses after unexpected controller errors.

The default operation envelope is three CPU credits, twelve process credits and
2,304 MiB, including 2 GiB sampled native RSS and 256 MiB coordination accounting.
The observed descendant peak was 591.09 MiB, eight processes and 74 OS threads;
controller RSS was separately observed at 692.10 MiB. Process credits are
estimates and coordination memory is not a hard controller cap. The named-theorem
no-oracle audit operates under trusted HOL; it does not establish HOL-axiom freedom
or repository behavior. A composite MCP workflow still makes separate adapter
calls with separate deadlines. Lean/Coq legacy routing, aggregate cgroup limits
and shared admission for direct generic canonical-backend callers remain outside
this phase; the closed supervisor task already supplies its admitted probe and
runner. General mixed-family composition remains open. The
[phase report](../../../ipfs_datasets/workspace/isabelle-hammer-execution-qualification-20261002/REPORT.md)
records 1,460 passing latest outcomes and ten PATH-gated legacy Coq skips across
1,470 distinct cases. The skips leave native Coq coverage unqualified and do not
establish that no managed runtime is installed. Raw execution retains 1,480
cases, including one outdated executable-path assertion failure; the complete
corrected ten-case file passed without changing production code. Earlier
measurements remain unchanged.
