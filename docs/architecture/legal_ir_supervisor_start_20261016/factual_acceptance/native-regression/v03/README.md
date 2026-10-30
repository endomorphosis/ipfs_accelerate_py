# V03 genuine native fixture entry

This new frozen generation preserves V1 and V02 and uses the new adapter-v02 bundle.
The only capability timing change is a finite900-second process-local handle
lifetime; same-process, one-use, source/provenance checks remain unchanged.
Validation/controller lifecycles are900 seconds to include real ledger locks
and global censuses. The attachment barrier is180 seconds. The productive child
wall deadline remains120 seconds, with explicit POSIX CPU/memory/file/open-FD
limits applied by the source-pinned startup barrier immediately before exec. Parented disk locks use the supported defaultNone;
Validation uses its finite60-second lock timeout. None of these changes modifies
the native scheduler policy or any task attempt cap.

The standalone entry creates and retains all actual SDK objects in one Python
interpreter. It imports the held genuine producer tree, opens the existing
fixed-policy scheduler, acquires real ORCHESTRATION root→phase→parented disk for
source preparation, stages an explicitly synthetic isolated fixture, builds the
supported hermetic runtime, freezes the complete exact SDK validation request,
then runs actual validation/attached child/seal/finalize/handle consumption. It
cannot receive a borrowed native owner object from a parent process.

Author a concrete reviewable source-only entry request with:

```sh
python3 -B prepare_entry_request.py --name legal-ir-native-acceptance-fixture-<owned-run> --run-root /absolute/named-root/legal-ir-native-acceptance-fixture-<owned-run> --mode handles_only --output /absolute/frozen-request.json
```

That helper performs no native/product imports or runtime actions. It binds
every adapter/regression manifest row and actual native/accelerator owner module.
The root owner reviews/finalizes the request, then executes the entry directly
with the printed request SHA; the entry itself obtains actual ownership:

```sh
python3 -B entry.py /absolute/frozen-request.json <exact-request-sha256>
```

Do not launch this inside V05 as a child expecting in-process borrowed leases.
No accepting worker CLI, file attestation, environment authority switch,
credential or provider call exists. All provider/task observations are visibly
synthetic and every real-task acceptance indicator remainsfalse.

The default handles_only mode exercises genuine native issuance and one-use
consumption without any board publication. For the complete supported regression,
select isolated_fixture_publication in the reviewed request. After validation
finalization the entry requests separate genuine ORCHESTRATION root/phase/disk
ownership, respecting real capacity and queue refusal; it then runs the trusted
coordinator on only the isolated fixture board. It writes an intent, completes
only fixture LIR-9001, preserves63blocked fixture tasks, verifies readback, and
finalizes actual controller ownership. The initial preparation owner remains
active to account for the isolated source and frozen controller request.

The preparation, validation and controller attempts are private siblings under
the chosen run root. Preparation's external directory metadata is conservatively
precharged; the controller separately precharges its bounded fixture board
write. All cache/temp/log/event sidecars remain inside the corresponding actual
claims. The real held policy and named roots are checked, never initialized or
reconfigured. A refused admission or missing Bubblewrap/kernel support fails
closed; root records the real failure and genuine retained claims rather than
simulating acceptance.

The pre-closeout qualification artifact explicitly says preparation closeout
is pending. Only the final entry stdout terminal claims actual preparation
release after the genuine API returnedreleased; no productive artifact is
written after releasing its claim. Capture that stdout under the caller's own
admitted logging if durable terminal evidence is required.

36 controlled adapter tests (including finite deadline and owner-change refusal)
are a separate validation requirement. This real integration qualifies API
plumbing only; real LIR-9001 still requires an actual provider candidate and
correct host source/task/attempt/provider binding. See the separate original
PACKET_ADOPTION_ADVISORY for schema/closed-CLI reconciliation, preservation of
canonical identity, and the two consumed durable attempts.

V03 pins the actual redirected child environment before execution using public
dataclasses.replace on HermeticValidationRuntime (runtime_id recomputed). The
actual SDK environment map must match that frozen identity exactly before Popen.
It also permits at most three open tries only for an exact native pre-admission
ledger-lock refusal while public owner metadata still says not_entered, record
None and resource_leaseNone. Any claim, lease, other failure or productive
execution prevents retry and retains actual failure evidence.

The stdlib-only startup barrier validates the exact frozen bounds argument and
applies RLIMIT_CPU/AS/FSIZE/NOFILE immediately before productive exec. It calls no
private SDK helper and uses no threaded-parent preexec callback. NPROC is
deliberately omitted because it counts the entire UID; four actual native
child-group slots remain cooperative sampled enforcement. No hard per-job
process quota is claimed. Fixture paths must be canonical and outside/tmp
because the hermetic command mounts a new tmpfs there.
