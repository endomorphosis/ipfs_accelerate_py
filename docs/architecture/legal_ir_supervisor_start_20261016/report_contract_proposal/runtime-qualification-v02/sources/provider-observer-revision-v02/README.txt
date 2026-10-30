Ordinary provider handoff, source audit revision V02

This revision corrects the earlier V01 emphasis: the current ordinary daemon
builds a trusted grok_cli_runner command. That runner imports route functions
from llm_router, owns fallback argv/auth, selected route and Docker lease, and
executes the model CLI in a separate process/container. The model does not own
the trusted route-runner callback or fallback selection. A benchmark/shared
router_implementation_runner receipt callback is a different optional API;
its UUID must not be silently adopted as the ordinary provider execution ID.

The trusted runner could carry actual typed route, Popen and Docker observations
in the same interpreter to a future host lifecycle hook. No new portable token
layer is needed. Today there is no such public successful handoff. The gap is
not that every receipt callback is executed inside an untrusted LLM container;
it is that existing public APIs do not join the complete successful lifecycle
with actual task/native attempt authority and actual model process identity.

NativeCandidateContext.provider_worker_pid must identify the actual untrusted
model CLI. For Docker this requires the exact model/container-init host PID and
birth joined to daemon/CID/image/cgroup. The trusted route-runner PID and Docker
attach-client Popen PID have distinct roles and cannot be relabeled. Retain
the actual transport child object separately. Current protected lease internals
can capture container init/cgroup before input, but ordinary legacy helpers
cannot gain that authority by passing fabricated protected objects or JSON.

Minimal reviewable call order for the upstream hook:
1. Genuine same-process admitted host ownership and source pins; actual board,
   task revision and consumed ordinal observed before release.
2. Trusted ordinary route selected; existing execution binding observed, or a
   new explicit host execution convention reviewed and minted before dispatch.
3. Actual child attached; actual model/container PID birth observed; input only
   released after corresponding native/provider gates.
4. Actual wait/reap; independently known process/Docker/watchdog/lease cleanup.
5. Genuine bounded fsynced host-private receipts under actual accounting.
6. Typed successful handoff to independent checker in the same trusted process.
7. Separately owned report-only commit and actual candidate correspondence;
   real native validation and separate coordinator consume one-use handle.

The existing _run returns from many branches inside try and closes its lease
in finally. A callback appended after a return would be unreachable; a callback
inside finally could accidentally run with a pending failure. Refactor the
trusted wrapper so actual success handoff runs only after successful known
cleanup, with retained actual objects. Required hook failures must propagate;
the existing diagnostic receipt_observer swallowing policy is inappropriate
for required ownership/correspondence gates. Do not convert a failed/unknown
provider effect or cleanup into successful observation.

PUBLIC_API_AUDIT.json pins exact reviewed input source and records role/fact
boundaries. UPSTREAM_HANDOFF_TASKS.json contains three supplementary engineering
goals and six dependent tasks. All pending; no canonical board or counters are
changed. Two consumed historical attempts and all63 future blockers are intact.
No effectful observer/token stub is implemented, no native/provider job ran,
no task was accepted, and no model/legal/Lean quality is inferred.

Historical launch03 remains failure evidence only. Exact pins are retained;
it is not a fresh provider outcome. Native fixture qualification is independently
owned by root and is not inferred here. V01 files are preserved. This V02 audit
supersedes its role/UUID emphasis and future portable-token wording. No tests
are warranted for documentation/metadata alone; JSON/hash authoring checks are
not runtime qualification.
