# Launch compact-context comparison through Terminal Bench

The standard Harbor → supervisor → signed coding command → worker → router path
accepts the existing versioned codec and semantic role-view options. The default
remains `legacy@1` with projection disabled. This guide covers forwarding;
source, owner, tokenizer and task validation still use their existing checks.

## Select the comparison

For `FullSupervisorAgent`, supply these optional keyword arguments along with
the existing full-arm configuration:

```python
semantic_slot_policy_revision="closed-typed-identifiers@2"
semantic_context_projection=True
```

The equivalent supervisor CLI options are:

```text
--arm full
--semantic-slot-policy-revision closed-typed-identifiers@2
--semantic-context-projection
```

The host `full_supervisor_benchmark prepare` command also accepts these
options and the tokenizer options below. For example, from the
`external/ipfs_accelerate` checkout with the existing Harbor environment:

```bash
../../.venvs/terminal-bench-harbor/bin/python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark prepare --dataset /absolute/original-dataset --archive /absolute/runtime-full --output /absolute/fresh-comparison --arm full --semantic-slot-policy-revision closed-typed-identifiers@2 --semantic-context-projection
```

Without an explicit tokenizer binding, the router resolves an offline counter
for the requested provider and model. Supported installed model mappings use
their local tokenizer; other models use a labelled proxy or a byte estimate.
An already deployed container-local descriptor and its hash override automatic
selection. Invalid explicit bindings still refuse rather than falling back.
Preparation validates option formats before reading the dataset or archive.
It does not read, copy or resolve the descriptor against the host filesystem.
Tokenizer options alone remain available on `--arm no-index`; advanced
semantic options require `--arm full`.

Fresh preparation records `prompt_comparison_options`. Execution refuses a
different configured selection before launch. Collection reports
`prompt_comparison_config_unchanged` for both prepared and retained trial
configuration. Legacy preparations without this separate pin report that
comparison as unknown while retaining the existing configuration SHA guard.
`execute` and `collect` use retained options; fresh CLI options do not rewrite
an existing trial.

For a coding-prompt ablation, the full arm additionally accepts
`--doctor-dispatch-policy model-only`. The equivalent adapter keyword is
`doctor_dispatch_policy="model-only"`. Default `auto` retains automatic Doctor
dispatch; the explicit control skips that selection after normal planning,
indexed-context preparation and source admission. Coding then uses the ordinary
signed model-worker command, public instruction and validation/publication
path. The switch requires the full arm and adds no runtime or token budget.

Preparation pins the policy and execution rejects configuration mismatches.
The driver records `doctor_dispatch_policy` and
`doctor_dispatch_observation.selection_performed`. A skipped selection has
`skip_reason="explicit_model_only_ablation"`; it is not a Doctor residual or
a failed proof. Compare the same policy in both encoding arms to isolate
representation effects. A default-auto run that publishes a Doctor candidate
does not exercise coding-prompt compression. This control supplies no tokenizer
asset, compatibility evidence or measured savings.

The driver and worker-option source change requires a fresh runtime archive.
Rebuild it through the existing bundle workflow before preparing a new trial;
completed trials and their archives remain immutable.

V2 compares native, legacy aliases and full-context V2 aliases, plus a versioned
packed V2 wire when eligible reference rows shrink. Adding projection also
compares the verified role view with V2 aliases and its packed alternative.
The options request alternatives; they do not force a representation to be sent.
The [reference packing guide](REFERENCE_WIRE_PACKING.md) describes its lossless
columns, bounds and unchanged table identities.

The packing addition has its own [237-case guarded validation](../../../../artifacts/supervisor-reference-packing-20261012/README.md).
Its complete-message fixture measured 32,006 bytes for full V2 and 30,122 bytes
for packed V2. Earlier launch-validation results remain tied to their recorded
source snapshot; this separate campaign validates the changed packing/router
paths and does not combine old test counts.

The coding arguments are assembled before the existing signed launch is
created. The worker forwards them to the existing router only for semantic
model coding with a bound source repository. Planning retains its existing
tokenizer/budget forwarding and receives no advanced semantic options.
Provider-free Doctor routes retain their existing commands, including when the
full supervisor has global comparison options. Direct advanced options on a
Doctor or other non-router worker path refuse.

Unknown policies, projection without V2, non-Boolean API switches, missing
semantic repositories and advanced options on the no-index arm refuse. Default
calls add no flags and retain their prior argument lists. These options create
no owner, discover no model and stage no asset.

## Automatic counting and explicit overrides

Automatic resolution never downloads assets or calls a provider. It uses the
installed, supported tiktoken model mapping when both the encoding specification
and authenticated local ranks are available. Otherwise it tries cached
`o200k_base` as a proxy, then `ceil(UTF-8 bytes / 4)` as a coarse fallback. The
resolver preserves the exact requested provider/model and records the backend,
version, rank hash, mapping match, compatibility basis and fallback reason.
It should be resolved again for the actual route if a caller changes models.

Grok 4.7 and `gpt-5.6-sol` currently have no verified mapping in the checked local
installation. Their proxy results have `measurement_status="estimated"`,
`estimated_input_tokens` and `selected_estimated_content_tokens`; measured
`input_tokens` and `selected_content_tokens` remain `None`. This makes offline
comparison possible without claiming remote tokenizer compatibility or savings.
The resolver accepts arbitrary provider/model identities; this Harbor coding
adapter remains qualified for its existing Codex route.

xAI also provides a [model-specific text tokenization endpoint](https://docs.x.ai/developers/rest-api-reference/inference/other).
It is not called by this offline resolver. Its counts cover the submitted text;
the [official SDK documentation](https://github.com/xai-org/xai-sdk-ts#tokenization)
explains that inference can add tokens. Provider receipts therefore remain the
authority for actual request and session usage.

Use an explicitly supplied compatible tokenizer descriptor already deployed at
an absolute container-local path, with its exact hash:

```text
--prompt-tokenizer-artifact /opt/ipfs-supervisor/models/prompt-tokenizer/descriptor.json
--prompt-tokenizer-sha256 <exact-lowercase-sha256>
--max-input-tokens <existing-message-ceiling>
```

The descriptor/hash must be provided together. The optional token budget also
works with automatic counts. An estimated budget records
`token_budget_basis="estimated_content_tokens"` and
`token_budget_guaranteed=false`; it cannot guarantee the model's actual limit.
Local tokenizer counts also cover only the rendered user message, excluding
provider instructions, tool turns and other request framing.
Host paths are not automatically copied into the container;
this implementation adds no tokenizer packaging, runtime installation,
downloads or model-compatibility evidence. See the
[counter guide](TOKEN_USAGE_RECONCILIATION.md) for binding and deployment limits.

The router counts complete rendered user messages, including workspace
instructions, translation overhead and literal suffixes. It selects an
alternative only when the selected counter shows strictly fewer tokens or estimated tokens than
native and the existing ceilings permit it. Equal semantic counts prefer
legacy, then full V2, projected V2, packed V2 and packed projected V2. Missing, failed or inconsistent counts
keep native; a configured token budget requires a working count or estimate.

## Review what was sent

The invocation receipt records:

New selections use `supervisor-model-prompt-selection@2` to distinguish the
quality and budget metadata from historical `@1` receipts. Historical audit
supports the new single-comparison legacy route and preserves the original
`@1` validator. Its existing multi-candidate V2/packing replay limitation
remains; those comparisons do not become audit-qualified by this change.

| Field | Scope |
| --- | --- |
| `semantic_slot_policy_revision` | Requested non-default comparison policy. |
| `semantic_context_projection` | Verification/projection observation when requested. |
| `semantic_slot_policy_selected` | Selected codec policy, or `None` when native wins. |
| `semantic_context_projection_selected` | Whether projected V2 was actually sent. |
| `semantic_reference_packing_selected` | Packing revision actually sent, or `None`. |
| `prompt_selection.comparison_candidates` | Complete-message counts or labelled estimates and availability for each alternative. |
| `prompt_selection.token_count_kind` | `tokenizer` or `estimate`; never provider usage. |
| `prompt_tokenizer_source` | Automatic backend/mapping/fallback provenance, or the explicit descriptor provenance. |
| `prompt_selection.selected_candidate` | Winning semantic alternative, or `None` for native. |
| `native_prompt_sha256` | Original stdin identity, retained independently of the selected message. |
| `model_prompt_sha256` | Exact selected rendered user-message identity. |

Projected context keeps the complete source, selected capsules, admissions,
required coverage, task and freshness caveats. Its initial policy defers only
estimator telemetry, bound to the unchanged original. The
[projection guide](SEMANTIC_CONTEXT_PROJECTION.md) describes restoration,
expansion and scope. Expansion is an API seam; no model tool is registered by
these launch flags.

## Interpret measurements

Byte reduction does not establish provider-token savings. Plain role-view
comparisons can also change reference chunk sizes, so compare matching chunk
policies before attributing a reduction to deferred diagnostics. The new
`test_semantic_projection_chunk_comparison` uses 6,144-byte chunks on both
original and projected context and records complete rendered bytes with token
counts unknown.

In that controlled fixture, native is 40,492 bytes, the plain role view is
43,527 bytes, full-context V2 is 32,006 bytes and projected V2 is 33,324 bytes.
The estimator-only projection therefore adds overhead. It remains an optional
comparison candidate; the router must compare complete-message counts or labelled estimates to
select it. Full-context V2's byte reduction also remains a fixture observation,
with production token counts unknown.

Whole-session comparison must retain all assigned tasks and all planning,
coding, failure, retry and expansion attempts. The existing session reconciler
counts native sessions once and leaves missing usage unknown; local counts are
not additional provider charges. See the [implementation plan](../architecture/terminal_bench_token_reduction.plan.md)
and [validation artifacts](../../../../artifacts/supervisor-compact-launch-20261012/README.md)
for the observed code version and current evidence limits.

No context limit, temperature or task acceptance criterion changes here. Only
an actual `lake build <Lib>` grants a Lean admit. Transport restoration,
projection tests and stored rows do not; the Constitution remains unformalized.
