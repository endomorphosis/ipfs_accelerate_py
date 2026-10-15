# Count complete metadata prompts with a pinned local tokenizer

The container supervisor can explicitly choose its existing compact semantic
metadata view using an owner-provisioned local tokenizer. The original and
candidate include the complete rendered coding prompt, workspace guidance,
public instruction and applicable reply contract. A candidate must be smaller
in both UTF-8 bytes and measured local tokens. Otherwise the original prompt
is retained. Existing defaults and codecs remain in place.

This extends the existing programmatic `SemanticMetadataTokenCounter` path to
the container supervisor and generated router worker. It reuses
`load_owner_prompt_tokenizer`; no tokenizer assets, model weights or new
checkpoints are downloaded or trained.

Use the following arguments together on `terminal_container_supervisor` or
`router_implementation_runner`, alongside the existing nonlegacy metadata view:

```text
--semantic-metadata-view capsule-columns@1
--semantic-metadata-tokenizer-artifact /opt/local-tokenizers/owner-tokenizer.json
--semantic-metadata-tokenizer-sha256 <full SHA-256 of descriptor bytes>
```

The descriptor uses the existing `supervisor-local-prompt-tokenizer@1` schema.
It names the exact provider and model, tokenizer identity, special-token policy,
installed `tokenizers` version, tokenizer JSON file and its full SHA-256. Its
revision binds the asset hash and runtime version. Relative asset paths resolve
against the descriptor directory. The loader rejects symlinks, changed assets,
truncation and padding. Both files must already be available to the deployed
worker; paths on the host are insufficient. Provision them outside the editable
task workspace and retain their pins as experiment controls.

The new pair is independent of `--prompt-tokenizer-artifact`, which measures the
existing general prompt-selection path. It cannot be combined with an injected
metadata counter, a general prompt tokenizer or `--max-input-tokens`; a second
selection must not change the audited metadata decision. The container route
requires the full indexed arm and original semantic transport. The container's
separate metadata-census mode is not supported with this pair in this increment.
Planning and provider-free Doctor routes do not consume a coding tokenizer.

The resulting metadata selection uses the existing `@2` receipt with
`qualification: estimate`. Its counter identity includes the special-token
policy; separate provenance retains descriptor, asset and runtime bindings.
Loading failures refuse dispatch. Failed counts retain the original prompt
without falling back to a byte estimate. Local bindings are checked again
before provider dispatch.

The final context audit receives the descriptor pair from its trusted caller.
It can recompute selection with that counter and compare the complete receipt.
It does not read tokenizer paths supplied by an invocation receipt. An audit
without the explicit pair preserves the existing unknown counter-verification
result. Availability and drift failures withhold verification.

A matching provider/model label is an owner declaration. These counts measure
the two rendered strings; they do not establish the remote model's tokenizer,
provider-added framing, tool output, cached usage, generated reasoning or total
workflow cost. Preserve actual native session receipts and all attempts through
the [session accounting path](world_benchmark_session_accounting.md) for a paired
benchmark. Do not adopt a default based on a smaller initial prompt.

This increment exposes the container and worker command-line path. Harbor
configuration does not yet stage these owner-provisioned assets or expose the
new descriptor pair. A Harbor experiment requires an explicit, pinned deployment
extension and a fresh runtime archive; old archives are historical evidence.

Private qualification artifacts are retained at
`/home/barberb/lift_coding/artifacts/metadata-tokenizer-selection-20261015-01`.
The offline comparison uses authored 4/8-function inputs, genuine deterministic
indexing, the existing lossless codec, the actual router and a cached GTE
tokenizer as an explicitly labelled local proxy. Its provider seam is scripted.
It establishes no Codex tokenizer equivalence, public benchmark quality,
legal-corpus coverage, live model ingestion or complete-workflow token savings.

Qualification passed 660 distinct runtime, worker-routing and historical-audit
tests, with no skipped cases. The six offline pairs selected the compact prompt
four times and retained the original twice; all six restored the original
context exactly and matched the complete prompt captured at the scripted
provider seam. No new benchmark model calls, training, downloads or Harbor
trials were launched. The offline report's full SHA-256 is
`51ec6fe6daf647fe7de335ae50087e40d48f3a5de9315272e0a2a32671258831`.
