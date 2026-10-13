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

This repair qualifies the adapter when installed by a consumer. It does not enable
the optional adapter in every provider consumer. Limits are admission estimates and
child-limit contracts: runners must enforce those limits, and already-running work
is not forcibly preempted when external pressure rises. In-process leases do not
coordinate independent supervisors; account-wide admission remains a separate owner.

Three retained ASI167 tests still reference removed usage-admission APIs. Their
historical implementation inspected usage snapshots and maintained local bookkeeping;
it could bypass usage checks when custom admission was supplied. Its successor must
compose the existing `ProviderExecutionGateway` and usage coordinator with host
admission, reserve once per physical dispatch, and preserve logical-member accounting
through cancellation and fallback. Restoring a second scheduler would not close that
gap. Default consumer adoption and shared usage accounting require that integration.
