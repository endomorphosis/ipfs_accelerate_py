# World packet presentation and quality evidence

The opt-in router presentation preserves the complete business contents of a
strict repository world packet while using a lossless model view. Historical
invocation replay reconstructs the exact rendered user input and retains the
complete recorded invocation receipt. These are input custody contracts; they
do not establish successful task repair, model quality, provider ingestion,
billing, or a reduction in complete session usage.

## Live dispatch boundary

`router_implementation_runner.run` accepts `world_task_presentation=None`.
The default retains the original world packet and original receipt fields.
An opt-in presentation must be the original issued
`PreparedRepositoryWorldPacketPresentation`, paired with its original issued,
current, strict `PreparedRepositoryWorldPacket` on the supported coding route.
The existing read-only observation packet cannot authorize a provider call.

The live router checks the presentation's original references, full inverse,
and receipt before the original packet's final source and native closing
checks. The codec cannot expand the task's effect scope or write paths.
Combining the new presentation with `world_packet_artifact_dir` is refused:
that older historical artifact format reconstructs the original full prompt.
This opt-in library API does not activate a general Harbor or container CLI
profile and does not change supervisor defaults.

The router receipt distinguishes these inputs:

| Receipt binding | Input represented |
| --- | --- |
| `native_prompt_*` and `prompt_*` | Original task text |
| `world_task_packet.prompt_*` | Original compiled world packet |
| `world_task_presentation` | Original/presented/sidecar/codec receipt identities |
| `router_prompt_*` | Selected presented body |
| `model_prompt_*` | Complete rendered user input, including workspace guidance and any supported coding reply contract |

`dispatch_rechecked` and `provider_attempted` remain recorded router events.
They are separate from task acceptance, provider acknowledgements, native
rollout usage, and owner validation.

## Historical bundle

`runtime/repository_world_presentation_invocation.py` supplies the additive
`repository-world-presentation-invocation-artifact@1` format. Existing packet
and invocation replay formats remain unchanged.

Capture the original packet **before** the provider can edit source. After
the attempt, join its independently retained digest to the complete invocation
receipt and exact presentation custody:

```python
import hashlib
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import freeze_repository_world_packet
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_presentation_invocation import (
    freeze_repository_world_presentation_invocation,
    invocation_receipt_sha256,
    replay_repository_world_presentation_invocation,
)

digest = lambda data: hashlib.sha256(data).hexdigest()
original = freeze_repository_world_packet(packet)  # Before model dispatch.
# Run the supported router with the original packet and issued presentation.
# Retain its actual receipt, including a failed attempt's emitted receipt.
receipt_pin = invocation_receipt_sha256(invocation_receipt)
sidecar_pin = digest(presentation.sidecar_bytes)
bundle = freeze_repository_world_presentation_invocation(
    packet_artifact=original, expected_packet_sha256=digest(original),
    invocation_receipt=invocation_receipt, expected_invocation_sha256=receipt_pin,
    presentation=presentation, expected_sidecar_sha256=sidecar_pin,
)
historical = replay_repository_world_presentation_invocation(
    bundle, expected_sha256=digest(bundle),
    expected_packet_sha256=digest(original),
    expected_invocation_sha256=receipt_pin,
    expected_sidecar_sha256=sidecar_pin,
)
```

An explicit `view_prompt` plus `sidecar_bytes` can replace the issued
presentation argument for historical capture, with a mandatory independent
sidecar pin. Replay always requires the bundle and all three component pins.
The invocation pin covers the full finite canonical JSON receipt, including
unknown fields, duplicate observations, failures, and null usage; it differs
from a pretty-printed receipt file's raw byte digest. Keep both identities when
retaining a file. Store artifacts and pins under private host custody: the
bundle contains original task, context, workspace, and sidecar content.

Replay verifies the original strict packet, exact codec inverse, packet/task
and presentation receipt joins, then uses the production prompt renderer and
coding reply contract for the recorded workspace string. It does not resolve
that path or reopen the original source owner. A valid provider edit can make
the original live packet stale without invalidating historical input replay.
Pre-dispatch failures can be archived only if a complete rendered-input receipt
exists; failures before such a receipt remain missing evidence.

The result exposes `model_prompt`, a detached complete
`recorded_invocation_receipt()`, and a content-free diagnostic `receipt()`.
It provides no live packet, dispatch method, restored signature, or completion
authority. Usage fields are retained observations, without recounting tokens,
validating billing, reconstructing provider-internal history, or proving model
consumption. Bounds refuse oversized data rather than omit fields: 4 MiB total
bundle, 1 MiB full invocation receipt, and the existing 2 MiB packet / 1 MiB
sidecar / bounded model-input profiles.

## Quality and usage comparison

Use the existing Terminal Bench/Harbor reward and acceptance checks with the
same task, source, requirements, provider/model route, and control identities.
Declare the paired arm slots before running; retain failures, retries, fallback
selection, missing outcomes, and every reported attempt. Keep official reward,
source/output validation, observed usage completeness, and scientific comparison
eligibility separate. Public task artifacts are benchmark inputs, not training
data. No hidden verifier content belongs in the model prompt.

Previous live compact-worker trials already warn against equating a smaller
initial prompt with a cheaper session: the three retained largest-eigenval arms
passed but reported about 1.31M, 1.62M, and 2.45M tokens, with complete all-attempt
usage unknown. See [compact_worker_benchmark.md](compact_worker_benchmark.md).
The earlier worker delivery/presentation experiment used an authored client and
local tokenizer proxy, not a model quality trial. This historical replay slice
adds exact input custody for a future actual paired repair trial; execution and
qualification manifests must supply any subsequent measured quality claims.

## Closed paired accounting API

`benchmarks/agent_supervisor/container_coding/terminal_world_packet_quality_accounting.py`
provides `compare_paired_quality_usage(...)` and
`validate_quality_accounting_report(...)`. Supply four separately pinned finite
JSON inputs: a pre-run control and attempt-slot declaration, a post-run
observation inventory, every declared attempt, and the verifier results.
`canonical_input_sha256(...)` hashes the full canonical input, including any
internal seals; `quality_accounting_report_sha256(...)` pins the complete report.
Validation verifies an independently retained report pin and regenerates the
report from all four original inputs.

Declare every case and both arms, exact required check IDs, source/task/
requirements/acceptance identities, provider/model, budgets, cache and execution
policies, and each arm's codec identity. Independent planning and coding calls
can have separate root slots. A retry's parent must precede it in the same
case and arm. Keep refusals, failures, missing receipts and unattempted slots;
amendments must retain their original declaration and prior observations.

Each verifier names its `evaluated_attempt_id`. Its output hash must equal
that dispatched attempt's output hash in the same case and arm. Independent
post-run inventory pins bind the attempt, complete receipt, verifier and output
identities. A result from an unrelated output cannot qualify a cheaper attempt.
The verifier's provenance label and hashes bind supplied evidence; they do not
authenticate the verifier or prove its assertions.

Native usage is selected once per session across **all** attempts and cases in
each arm using `context_delivery_accounting`, rather than summing resumed
Codex cumulative snapshots. Conflicting snapshots, absent receipts and unknown
call coverage withhold complete totals. Tool/retrieval payload bytes stay
separate from native tokens. Unknown controls, including unknown actual cache
state, block comparison. Raw per-attempt observations remain available even
when complete totals are null.

The report separates structural eligibility, complete verifier evidence,
candidate quality non-regression, complete supplied usage, and conditional
scientific eligibility. Quality non-regression does not require universal
correctness or even both arms passing. An authored fixture can exercise the
criteria without becoming observed model evidence. Supplied-call completeness
does not establish complete campaign calls, auxiliary calls, provider internal
retries, billing, model ingestion or permission to change defaults.

## Bounded actual repair trial, 2026-10-15

The root-owned trial used genuine native source/index/compiler owners and the
strict router with Codex CLI 0.160.0, `gpt-6.1-sol`, high reasoning, a 180-second
per-call deadline and unchanged fallback policy. Both arms started with the
same two-file arithmetic fixture. The seven predeclared finite checks covered
public signatures, negative/zero/positive addition, subtraction and its AST.
Independent verification ran retained checker bytes and generated source in
cached containers with networking disabled and physical CPU, memory and PID
limits. This is a small integration trial, not a Terminal Bench result or an
evaluation of learned IR models.

| Observation | Original packet | Lossless view |
| --- | ---: | ---: |
| Complete submitted user-input bytes | 31,755 | 21,048 |
| Receipted native cumulative tokens | 486,151 | 21,251 |
| Provider outcome | Returned repair | Rate limit |
| Independent finite checks passed | 7 / 7 | 4 / 7 |

The smaller view reached the actual provider route and its submitted input
replayed exactly, including the failed call's full receipt. The rate-limited
call left the original source unchanged. Its lower recorded usage cannot
establish a saving because it did not achieve equivalent quality. Native
cached input is a subset of input; the totals above do not add it again.

A preceding original-arm driver attempt used the wrong working directory and
was refused before an invocation receipt existed. It remains an explicit
unknown attempt; a separately retained amendment appended the corrected retry
before launch. Therefore full workflow usage remains unknown. Actual provider
cache state was also unknown. No additional model retry was made after the
rate limit. The final comparison rejects quality non-regression and scientific
qualification, preserves all three slots, and leaves complete totals null.

The final full canonical comparison SHA-256 is
`2104284877741144ad56995aa90e279cbd07221693bbcfc74e5b1672ea3d76bc`.
Private evidence is retained under
`/home/barberb/lift_coding/artifacts/world-packet-quality-20261015-01`.
The final suites passed 207 distinct tests (65 dispatch, 45 historical replay,
97 accounting). Defaults remain unchanged. The next empirical step needs
provider availability and a fresh, predeclared paired trial; this run grants
no default adoption or whole-workflow savings claim.
