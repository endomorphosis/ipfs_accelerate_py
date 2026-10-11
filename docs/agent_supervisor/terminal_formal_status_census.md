# Formal-status availability before context replacement

The formal-status census checks whether the native bindings needed for proof
reuse are available before replacing repeated proof status. The current Terminal
route provides task-bound Source384/numerical candidates and proof references.
It does not provide a complete admitted Doctor proof-owner/key binding. Those
candidates and references therefore cannot establish accepted proof coverage.

## Terminal observation

`prepare_terminal_live_metadata_census(..., include_formal_status_census=True)`
adds a separately versioned `@2` owner report with a closed formal-status
observation. Default `@1` reports remain unchanged. The native prompt, runner
options, model assets and metadata-view selection remain unchanged.

The opt-in hashes the admitted task/revision, source manifest, contract and
current indexed context into a separate Terminal binding. It invokes no proof
owner or lookup: accepted receipt and repeated-status populations are **null**
(unmeasured), not zero. Parent `formal_status_context_binding_current` describes
only the checked Terminal context. The nested formal observation remains
unbound and cannot claim proof-owner currentness. On task/source/context drift
or cleanup failure, the escaped parent report clears that currentness flag
before raising. Existing custody and cleanup checks remain in force.

No proof cache is guessed from a catalog, path, key identifier, candidate IR or
task body. Required facts stay inline and no proof card is appended.

## Programmatic native-owner census

[`formal_status_census.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/formal_status_census.py)
accepts an already instantiated native `DoctorProofCacheGate`, an immutable
bounded tuple of complete `DoctorProofCacheKey` values, independently expected
key identities and a host current-binding probe. The host supplies task/revision,
source/context, candidate-tree and admitted owner-path bindings. These are
caller-admitted inputs; hashing them does not authenticate a source or owner.
There is no owner constructor, arbitrary database-path factory or model-callable
tool in this interface.

Structural validation runs before callbacks or filesystem observation. Partial,
private, foreign, duplicate and oversized key populations refuse. Missing owner,
keys or source bindings leave accepted coverage unmeasured and make no gate
calls. An explicitly admitted empty key population differs from missing keys.

The bound lane checks current host bindings and the existing owner file identity,
uses the native gate's lookup and render revalidation, checks exact returned
key/receipt identities, and rederives verdict and assurance. Visible durable
revocation or quarantine refuses the observation. Source/task/owner changes
during consultation also refuse. The existing metadata-suppression context
prevents opportunistic activation of another metadata owner.

Normal Doctor lookup can register dependencies, observations, refusals and
equivocation in its existing native owner. The report discloses these possible
bookkeeping writes; this lane is **not storage-read-only**. It creates no new
owner and never directly scans cache tables or consults a historical DuckLake
snapshot for acceptance.
Operation counters cover this adapter's calls. The host probe is admitted by
its caller; its arbitrary side effects are not audited by a returned binding.
The zero new-owner count describes constructors in this adapter, without
attesting the host callback.

Positive reusable Doctor hits admit only complete, current **proved** receipts
with sufficient independently derived assurance. Miss, rejected, diagnostic,
quarantined and tombstoned dispositions stay distinct. A miss is not refutation.
Recorded and derived verdicts must retain their own meanings; this positive
cache is not a six-verdict accepted population. Cancelled, error, unsupported,
inconclusive and disproved records cannot become accepted proof cards.

All observations cover the explicitly requested key set at native validation
points. Sequential checks are not an atomic owner snapshot, and eligibility at
future rendering or commit remains unverified. A census report cannot replace
fresh `revalidate_for_render` or `revalidate_for_commit` calls. Repeated-status
coverage and complete-run token savings remain unmeasured in both lanes.

## Next qualification

Provide a reviewed current-owner/key binding through the actual Terminal host
route before measuring real receipt coverage. Then identify exact repeated
status material and its complete required-fact closure. Replacement must retain
property, native verdict, assurance, assumptions, dependency digest and residual
obligations; all six native verdicts remain distinct in the appropriate native
claim/evidence APIs. Qualify bounded source/receipt expansion before moving any
required fact out of the prompt.

Census the resulting complete input and its lookup cost before a matched live
trial. The runtime archive must include the new census helper and updated
preparation/policy source digests; historical frozen archives remain historical
evidence. This implementation does not change decoder checkpoints, embeddings,
family/schema/task inventories or the parallel 8D/384D/768D paths.
