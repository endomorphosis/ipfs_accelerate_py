Ordinary trusted runner success handoff: source proposal V01

Status: source proposal under review, not deployed and not qualified for actual
provider execution. No provider/native job or model/auth probe was executed by
the source authors. Existing runtime/frozen source, canonical board/namespace,
the two consumed attempts and all 63 later blockers remain unchanged.

The proposed public Python API is:
run_ordinary_with_success_handoff(args, receipt_fd, *, owner=None,
    watchdog_owner=None, source_sha256=None, declared_external_storage=None,
    on_success=None) -> int

An opt-in call requires on_success(outcome), a trusted same-interpreter callback
that returns None or raises. Callback exceptions and non-None returns prevent
successful return. There is no accepting CLI, serialized outcome loader, copied
native key, environment authority switch or portable token API. With no callback,
the wrapper delegates to the existing runner; the ordinary CLI remains unchanged.

V01 supports only the actual primary Grok 4.7 route without preflight, fallback
or recovery. These unsupported selectors and ambient receipt channels refuse
before _run. The exact original provider route and fresh invocation UUID are
observed, not inferred from configured model preferences or historical receipts.
The user's Codex GPT 6.1 Sol backup is not implemented in this source generation.
Every future productive preflight and fallback process must receive separately
reviewed gating/accounting and actual closeout. Existing prohibition on switching
to Codex after Grok task dispatch remains in force. Pure assembly of the Codex
builder shape does not provide live backup support.

Mandatory actual ownership and source prerequisites:

- Two genuine same-process ParentedPhaseResourceReservation objects in the
  ORCHESTRATION lane: primary owner with at least 1 CPU, 512 MiB and 8 child slots;
  watchdog_owner with at least 1 CPU, 256 MiB and 4 child slots. Reservation/native
  lease IDs must differ, both must have fresh child histories, and both must join
  the same actual orchestration parent phase. The parent must actually admit their
  simultaneous envelopes; an inert dict, fake owner or copied native key cannot.
- Both original native contexts remain active through provider/watcher quiescence,
  callback completion and honest finalization. The actual newly allocated Docker
  cleanup watchdog attaches to watchdog_owner before fallible external-storage,
  daemon/runtime or provider-source checks. Docker transport attaches to the
  primary owner before the fixed marker can release model startup.
- Exact stable SHA256 pins for this hook, proposed runner, process_security,
  four actually loaded native SDK implementations, actual Python/Docker binaries,
  resolved trusted Grok binary and ten direct dispatch helpers. Those helpers are
  llm_router, agent_implementation_route, hash_pressure, provider_command_binding,
  provider_command_environment, provider_failure_policy, validation_runtime,
  lifecycle_orchestrator, worktree_lifecycle and provider_attempt_store. Required
  paths are checked before imports and joined to actually imported module paths.
- Trusted implementation/binary reads allow existing owner-controlled 0664 files
  only with exact pins, root/current ownership, one link, no world write and stable
  full metadata. Reads stream in 1 MiB chunks with a 256 MiB file bound; binaries
  are not copied into memory as one large read. A hash is correspondence evidence,
  not provider or native authority.
- Actual original precharges cover the full candidate workspace and external
  provider temporary root. Declared paths join the existing public SDK hashed
  charge-label convention and exact charged bytes. TMPDIR must be selected before
  tempfile/native imports and stay outside strict private owner attempts. Both
  owners' private outputs and any callback writes are independently charged.
  No storage cap/policy is changed and no grant is created by this hook.

Actual opt-in execution and failure boundary:

1. Check both genuine owners, original ancestry, fresh inventories, exact sources,
   storage declarations and actual primary route. Bind a new UUID to this original
   invocation. Ordinary A91 remains distinct from protected 74; no protected CAS
   lease, fixture grant or benchmark/failure identifier is repurposed.
2. Use the actual ordinary Grok Docker builder. Preserve its original image,
   mounts, network and isolation options; wrap only the exact entrypoint/tail in
   a fixed mounted-Python stdin marker gate. No image is downloaded or acquired.
3. Capture the actual created ordinary cleanup watchdog and attach it immediately
   to the separate watchdog owner. Preserve all allocated actual lease/process
   references on factory, daemon, source or storage failure.
4. Start actual Docker attach as an isolated Popen with a pipe. Before any provider
   input, attach the exact Popen to the primary owner, recheck both owners and
   independently capture the expected daemon/runtime, A91 CID/name/image/provider
   label and container init host PID/birth/namespace/cgroup. The container remains
   behind the fixed marker until these checks pass.
5. Recheck source/ownership and write exactly the complete marker, handling partial
   writes, then close primary stdin. The primary prompt is a bound file; this
   generation does not implement a Codex streamed prompt path. Wrong marker,
   premature EOF, unknown/too-fast birth or changed confinement yields no callback.
6. Monitor through the main-thread select/os.read loop and timed wait, checking
   both native owners at intervals of at most 0.5 seconds between I/O waits. There
   is no monitor thread that could race lease closure or provider input. Original
   transport wait/reap and provider exit 0 precede the runner's Docker finally.
7. After actual runner cleanup, independently verify the same daemon/runtime,
   exact CID/name absence, lease-root absence, provider-home and prompt-path
   absence, reaped watchdog, original kernel-scope quiescence and both native
   attached groups dead. Docker transport PID is distinct from model/container
   host PID; its reap alone does not prove model descendants stopped.
8. Only after all observations and original source/owner rechecks does the required
   callback receive immutable metadata plus actual retained native/process objects,
   actual primary route, fresh execution UUID and original model host birth/scope.
   Report-only candidate creation, task/attempt identity observation, native factual
   VALIDATION and separate one-use controller acceptance remain independent work.

Failure-finally repeats the original same-daemon CID/name and captured kernel
scope checks after any provider gate release. Unknown cleanup, live original
groups, provider-home/prompt remnants or unavailable fence observations do not
produce returned success. OrdinaryCustodyPending exposes actual in-memory owner,
lease and process references for the trusted caller; the exception itself supplies
no CPU, disk or cleanup authority. The caller must catch it inside the original
native contexts, retain both native parents and complete independent stopped
cleanup before unwinding or finalizing. No blind process-group kill, old receipt
adoption, automatic attempt refund or policy relaxation is authorized here.

Source controls and separate native qualification:

test_handoff_memory.py checks normative order/projections, immutable outcome
refusals, default behavior, explicit owner/options/source/storage conditions and
failure boundaries. Inert objects and supplied mappings are unit controls, never
genuine native grants or observed provider success. test_gate_memory.py exercises
the exact fixed marker reader/writer with inert I/O. test_builder_callsites.py
imports actual proposed builders, assembles their generated entrypoint/tail and
forbids process execution; regular inert fixtures and aliases remain in a unique
genuinely precharged external test root.

The exact qualify_ordinary_success_hook_v01.py harness is archived in this bundle
before its source manifest is frozen. It accepts an expected manifest SHA on the
CLI, verifies all rows and its own exact copy, derives the expected unique-case
count from controlled_testcases, and writes only private JUnit/result files.
Successful subtest reports do not duplicate parent case counts. Controls may be
run by root beneath genuine finite native admission, but they do not qualify
actual provider/native execution, deploy the patch or accept a canonical task.

A fresh genuine finite non-model Docker/native control remains required: prove
both actual owner attachments precede gate release, both owners are monitored,
actual PID roles and kernel fence correspond, source or callback refusal prevents
success, and every failure holds custody until all original scopes are quiet.
No model/auth probe is necessary for that control. Genuine quota-only Codex
backup and real report/candidate/acceptance integration remain later reviewed
generations. No statutory fidelity, model accuracy, graph correctness, training
eligibility or Lean compilation is claimed by this source proposal.
