# Managed preparation and worker delegation

The finite qualification driver has an explicit `--pipeline-resources` option.
It implies `--full-trial-resources` and composes the existing managed CPU resource
adapter. The older full-trial option retains its original bridge and v1 grant.
Neither option is a Terminal-Bench score or permission to promote a model.

The selected pipeline parent has 3 CPU slots, 3 GiB RAM and 3 process slots. Its
validation/cleanup compartment reserves 1 CPU, 1 GiB and 1 process slot. Ordinary
high-level work uses 1 GiB while a preparation stage can use the other ordinary
1 GiB as a sibling. All descendants use the one canonical datasets host authority.

Disk declarations use observed development workload sizes: the prior finite trial
retained approximately 151 MiB, including a 42.48 MB intent database, 21.51 MB source
database and approximately 50 MB initial/published context. High-level phases now
precharge 128 MiB each; individual preparation stages precharge 16 MiB each. These
are conservative declared ceilings, not exact disk measurements or reclaimed-byte
savings. The parent declares 1536 MiB, with 640 MiB protected and 896 MiB ordinary.
The explicit model-off profile contains:

| Population | Count | Ceiling each | Total |
|---|---:|---:|---:|
| Ordinary high-level phases | 5 | 128 MiB | 640 MiB |
| Ordinary preparation phases | 12 | 16 MiB | 192 MiB |
| Protected high-level phases | 4 | 128 MiB | 512 MiB |
| Protected preparation validation phases | 3 | 16 MiB | 48 MiB |

The resulting ordinary/protected declarations are 832/560 MiB, within their
896/640 MiB compartments. Admission thresholds and actual host pressure checks
remain unchanged. The driver reports these declarations and retains the actual
per-phase receipts; increasing the workload population requires another explicit
budget. Setup remains before admission and consumes the total trial deadline.

`prepare_repository_benchmark` retains its exact native `envelope` argument. Its
optional `pipeline` must own that same envelope. `pipeline_attempt_root` must be an
existing directory in the exact named roots. Source CAS, source SQL, selected proof
cache and selected model output owners must also remain in those roots. Every
stage retains its serialized request payload, creates a unique attempt, forwards
the daemon's native child lease, precharges the full external-write ceiling, and
checks final named-root growth before explicit durable finalization. Failed work
retains its claim. Final growth sampling is not a hard quota or a measurement of
transient write peaks; the complete ceiling remains charged after success.

V2 worker grants bind the actual daemon lease to its protected validation phase
and host root. The private immutable file and digest are verified before closed
v1/v2 dispatch. V2 additionally checks owner PID, boot, deadline, all ancestor
links, native lane, daemon reservation identity and exact CPU/RAM/process demands.
The canonical scheduler authenticates the capability when admitting the worker.
Public lease IDs alone cannot authorize it. V1 validation and root-only behavior
remain unchanged. The worker still verifies independent repository/task/proof
evidence; neither resource grant supplies completion authority.

The v2 receipt has schema `repository-delegated-pipeline-phase@1`. Its
`parent_lease_id` names the daemon; `root_lease_id`, `bridge_phase_lease_id` and
`daemon_reservation_id` identify the complete chain. `lease.released` must be true
before the driver finalizes the worker scope. Private tokens are excluded from
receipts and evidence exports. The protected worker scope stays open through
native STOP and context refresh. Proof-cache inputs receive a fresh phase parent
when moving from preparation to planning. Fixed public checks run through the
existing bounded process owner under protected capacity.

Failure handling retains the primary error, additional resource cleanup errors,
failed receipt observations and `result.json`. A failed finalization can retain
disk/host contexts for explicit recovery; the driver does not claim automatic
reaping of uncooperative external workers. Callers must drain
`pending_pipeline_recoveries()`. Forgotten handles across canonical lease expiry
can accumulate over a long-lived process; a lifetime inventory cap is not qualified.

This remains a cooperative sampled CPU profile. The external supervisor/worker
RSS is not captured by the daemon's direct-child sampler. Deterministic semantic
context preparation creates bounded ambient temporary snapshots outside the named
roots; those transient files are not covered by this disk census. No optional
Security/Intent inference, learned retrieval, model training or provider calls are
selected by this driver. GPU/per-device admission, hard aggregate limits and
gradual pressure recovery remain outside this qualification.

The helper controls use injected host telemetry, real file-backed source/catalog
owners and actual subprocess delegation. The CLI/parser/error controls use explicit
failure doubles and do not establish live admission. Actual native runs and legacy
model preparation checks are reported separately in the retained evidence.
