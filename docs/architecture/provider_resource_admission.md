# Provider batch resource admission

`runtime/provider_batch_scheduler.py` composes physical provider batches with
`runtime/resource_scheduler.py` through `ResourceSchedulerBatchAdmission`.
The adapter uses the existing scheduler's atomic `acquire` and `release` methods.
It does not create another capacity ledger. A physical call holds one lease;
duplicate subscribers share that execution only when their admission claims match.

Batch compatibility includes input sealing, trace and tokenizer identities,
checkpoint and exclusive-write claims, resource profiles, capabilities, networking,
and numeric resource demands. Unrelated request provenance remains independent.
Queued claims are detached from caller-owned mappings. Strict validation rejects
malformed claims and prevents a caller from lowering configured memory or capability
requirements. Exclusive or mutating work runs as one execution group per batch.

Host CPU, RAM, disk, GPU, worker capacity, provider concurrency, quota, token limits,
and provider cooldown telemetry are checked again on admission attempts. Temporary
pressure leaves the request queued with the configured admission retry interval.
Malformed, stale, or policy-incompatible claims fail without provider dispatch.
Cancellation or expiry before dispatch releases the grant. Cancellation during a
physical callback keeps the lease until that callback returns.

Acquisition and stage transition apply the same finite process, pool, memory, disk,
VRAM, token, quota, and latency budgets. A rejected transition leaves its original
lease unchanged and reports no newly reserved resources. Campaign profile ceilings
also narrow child execution limits; zero byte declarations cannot erase a named
profile's finite limits. Network declarations must be a subset of the scheduler's
trusted policy and do not authorize themselves. Effective resource kinds include
stage, class, and explicit claims, so flags cannot conceal GPU, network, or prover use.
Custom host snapshots must advertise a prover capability or a CPU proof resource
class for prover work. A GPU snapshot must include measured total and available
VRAM. Mixed profiles request optional specialized resources only when the selected
stage or caller declares them.

`schedule` and `simulate_pipeline` produce advisory overlap and fairness receipts.
Each receipt describes the reservations visible at that candidate's evaluation.
Dispatch still requires `acquire`; an advisory snapshot does not reserve resources.
Seals and identity strings here are consistency claims supplied by upstream owners,
not proof certificates or independent execution authorization.

## Qualification and limits

The focused tests cover claim isolation, pressure recovery, immutable refusal,
lease lifetime, profile round trips, transition budgets, and resource-kind bypasses.
The planning-consumer test uses authored repository source and a stub provider with
the real source index, prompt construction, batch scheduler, and resource adapter.
The bounded benchmark emits aggregate counts only; it does not call a real provider,
run the inspected code, or measure provider-billed token savings.

From the repository root, run
`python -m benchmarks.agent_supervisor.provider_resource_admission --output /tmp/provider-admission.json`.
The standalone command first checks live host admission for two worker slots and
256 MiB. It returns a deferred report and exit code 2 when that admission is refused.
The stress phase injects pressure telemetry instead of placing artificial load on
the machine. Its local preflight does not reserve capacity across independent
processes.

## Default shared host admission

`create_leanstral_goal_development_batch_scheduler` enables `ProviderHostAdmission`
by default. It combines supervisor claim checks with the datasets
`GlobalResourceScheduler`, the existing shared CPU/RAM/process owner. Each callback
reserves a bounded envelope (by default one CPU slot, one process slot, and 256 MiB).
A supplied native parent lease delegates its capacity instead of creating another
root reservation. Missing native authority or disabled native safety refuses work.
Sampled host pressure or exhausted capacity leaves work queued; acquisition is
nonblocking so the queue can still cancel or expire subscribers.

The native lease is checked again immediately before dispatch. Cancellation during
the callback keeps the reservation until the callback returns. Custom admission is
composed with host checks, including its lifecycle and release hooks. The default
router's scalar fallback receives an explicit worker limit bounded by the admitted
CPU/process envelope; an environment override cannot expand it. Custom callbacks
must respect their declared envelopes. `host_admission_enabled=False` is an explicit
rollback or authored-test opt-out. The CPU profile refuses GPU demands.

These are admission estimates and child-limit contracts. Runners must enforce
physical limits; admission does not impose kernel memory limits or forcibly preempt
running callbacks when outside pressure rises. Native backend batch internals may
need additional worker limits. Other provider consumers, including direct daemon
calls, still require their own shared host integration.

## Atomic physical batch usage

`runtime/provider_batch_usage.py` supplies `PhysicalBatchUsageAdmission` to the
existing batch scheduler. The real endpoint `UsageCoordinator` is the quota owner;
its CAS reservation, dispatch marker, and commit form one physical lifecycle.
Reservation checks current capacity inside the coordinator's atomic transaction;
admission does not use a separately sampled snapshot as authorization or a clock-
sensitive precondition.
The three retained ASI167 tests now exercise this successor against a real ledger.

Configure the adapter with a trusted provider-to-scope supplier, an aggregate token
ceiling, a reservation TTL, and optionally host admission. Each request must name
that exact `usage_scope_id`, positive input/output bounds, and a bounded timeout.
Scope participates in batch compatibility and singleflight identity, preventing
unrelated scopes from sharing a reservation. The coordinator's configured limits
remain authoritative across all schedulers using its store.

One aggregate reservation covers one callback that makes **one physical provider
request**: one request/concurrency slot, the unique batch items, their input/output
ceilings, and shared input overhead once. Duplicate subscribers share that charge.
Members pruned before dispatch are removed from settlement; cancelling a subscriber
after dispatch cannot refund work already sent. A trusted complete usage extractor
can settle measured input/output/total tokens. Otherwise settlement is explicitly
an estimated ceiling. Receipts contain identities, digests, and quantities, without
prompts or raw responses, and repeat the same physical reservation across subscribers.
Consumers must deduplicate that identity when aggregating physical usage.

Missing dispatch acknowledgement prevents the callback. Unconfirmed settlement
fails the result and retains the reservation for reconciliation. Callback failure
charges the live member ceilings conservatively. Automatic scalar fallback is
disabled for this adapter, because a second provider call needs its own reservation.
Scope-local denial backoff starts at 100 ms and doubles to at most one second;
intermediate queue retries do not acquire host capacity or write another denied
reservation. Coordinator limits and upstream provider cooldown remain separate
authorities.

Reservation expiry is crash recovery, not physical cancellation. The configured TTL
must exceed the actual bounded callback lifetime plus settlement time. Logical
subscriber deadlines alone do not terminate a callback or renew its reservation.
Deployment callbacks must enforce their physical timeouts to avoid a still-running
call outliving its concurrency hold.

For a callback with a genuine one-request batch contract, compose explicitly:

```python
host = ProviderHostAdmission(budget=ProviderHostBudget(max_parallel=2))
admission = PhysicalBatchUsageAdmission(
    coordinator,
    scope_supplier=trusted_scope_for_provider,
    max_batch_tokens=32_768,
    reservation_ttl_ms=120_000,
    host_admission=host,
)
scheduler = ProviderBatchScheduler(dispatch_one_provider_batch, admission=admission)
```

The Leanstral router may dispatch multiple remote calls for one local batch, and
the router can already have its own usage admission. Consequently the factory
enables shared **host** admission but does not automatically install this aggregate
usage adapter. Select exactly one usage owner at the actual physical call boundary.
Gateway migration retains the supplied gateway and replay state, invokes one owner,
and requires an injected coordinator with acknowledged reservation operations for
enforcement. Enforcement without an execution callback is refused before reserving
or charging. Its trusted coordinator protocol remains injectable; the physical
batch adapter specifically requires the concrete endpoint coordinator. Migrated
text-only callbacks retain all declared token estimates in settlement and identify
them as estimates. Usage migration defaults remain unchanged until real
scope/revision/fence configuration is available.

The router preserves explicitly supplied backend/candidate selection instead of
replacing it through automatic model selection. Capacity and semantic failures
propagate to the usage owner without an unaccounted retry under a default model.
Default calls still support automatic selection, and generation traces identify
the backend that actually executed.

## Stable admission revisions

`UsageSnapshot.usage_revision` remains a timestamped observation identity, with its
existing serialization and hashes unchanged. It is unsuitable for a stable state
precondition: time alone changes it, and its identity omits numerical headroom and
caller budgets. `UsageCoordinator.admission_snapshot(scope_id)` instead returns a
versioned `UsageAdmissionSnapshot` containing both a legacy snapshot and a separate
`admission_revision`, captured from the same store read.

The new revision binds scope, ledger generation, writer fence, the complete stored
document identity, and the coordinator's partition configuration. It changes after
any committed mutation, including budget changes, completed usage, administrative
changes, compaction, and other scopes' writes. Clock-only reads leave it unchanged.
Whole-ledger invalidation is intentionally conservative and may require replanning
after unrelated writes; the physical batch adapter continues to use current-state
CAS admission without speculative pins.

Pass the revision explicitly as `reserve(expected_admission_revision=...)`.
The coordinator checks it inside every CAS attempt, then checks capacity against
the current clock. A matching pin does not freeze headroom, grant capacity, prevent
window transitions, or bypass cooldown and reservation expiry. The old
`expected_usage_revision` parameter retains its existing strict observational
semantics. Supplying both preconditions is rejected.

Supervisor scope, bridge, and consumer context accept an optional
`admission_revision`. The pin is inherited through the budget hierarchy and must
match between scope and bridge. Empty fields are omitted from serialized records,
preserving legacy contract identities. New gateways pass the stable precondition
when supplied and otherwise retain the legacy observational precondition. Owners
without the new protocol refuse the request; the gateway never retries unpinned.
An enforced request carrying a stable pin also refuses degraded local execution
when coordination is stale, unknown, or unavailable, even if a local fallback
budget was supplied. That budget cannot validate the endpoint state precondition.
Usage receipts retain the original observation identity and include the admission
pin in their nested scope.

Obtain both identities together when building a migrated call:

```python
view = coordinator.admission_snapshot(endpoint_scope_id)
context = build_consumer_call_context(
    consumer_id="task_proposal_router",
    provider_id=provider_id,
    endpoint_scope_id=endpoint_scope_id,
    usage_revision=view.snapshot.usage_revision,
    admission_revision=view.admission_revision,
    fence_id="1",  # Use the deployment's actual writer fence.
)
```

Exact reservation replay may return the recorded decision after its own write has
advanced the ledger generation. For the new protocol, the coordinator first checks
a stored fingerprint of the full reservation contract, including the original
pin, quantities, caller budget, estimate, owner, lease, fence, and TTL. A changed
contract cannot reuse that idempotency entry. Gateway replay uses its existing
terminal cache; a new gateway seeing a prior reservation refuses reinvocation and
requires reconciliation. Replaying a ledger decision is not dispatch authority.

The offline rollout harness uses the new paired view. Other consumers still need
explicit migration from legacy revision pins; placeholder scope/fence identities
do not become valid deployment configuration. Automatic refresh/replanning of
stale pinned work remains a caller responsibility.

The durable real-clock qualification runs with
`python -m benchmarks.agent_supervisor.provider_admission_revision --requests 16 --output /tmp/provider-revisions.json`.
It first checks live local admission for one worker and 256 MiB, then uses an
isolated durable ledger and authored callbacks. It checks real-clock success,
in-process replay, reconstruction of the gateway/store, stale mutation refusal,
explicit refresh, quota exhaustion, and zero remaining holds. This measures local
accounting behavior; it does not call providers or measure billed token savings.

The 2026-10-13 local qualification passed 498 distinct relevant tests, with four
skips. The 16-request benchmark completed in 8,531 ms: 17 authored callbacks
(including one successful refresh), 16 cached replays, four reconstructed-gateway
reinvocation refusals, and zero duplicate callbacks or remaining usage holds.
These timings describe the isolated durable-ledger workload on this host, not
provider throughput. Fresh/stale pin tests also cover refusal of degraded
execution and changed-contract replay.

The bounded usage benchmark runs with
`python -m benchmarks.agent_supervisor.provider_batch_usage --requests 32 --output /tmp/provider-usage.json`.
Its live local preflight requests four worker slots and 256 MiB before starting.
The campaign uses two real queues sharing an isolated in-memory CAS usage ledger
and local host owner, with authored callbacks and controlled pressure. It exercises
quota denial/backoff, late duplicate subscribers, queued/running cancellation,
conservative failure settlement, and zero remaining holds. The native shared host
owner is covered separately by isolated factory tests. Neither this benchmark nor
its local preflight qualifies cross-process usage storage, hard containment,
provider-billed token savings, or live provider execution.
