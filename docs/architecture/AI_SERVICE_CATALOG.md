# AI Service Catalog

**Status:** Reference
**Owner:** model-catalog / service-architecture
**Audience:** Developers, operators, and agents that consume catalog identities,
snapshots, resolution, and MCP/MCP++ projections
**Scope:** Schema v1 records, operation taxonomy, lifecycle and operational
state, source precedence, security and non-invocation boundary, and rollout
gates for the information plane
**Non-goals:** Router invocation, streaming, provider client construction, or
endpoint-usage reservation (see [MODEL_SERVICE_ROUTING.md](MODEL_SERVICE_ROUTING.md));
MCP transport and UCAN policy (see [MCP_RUNTIME.md](MCP_RUNTIME.md))
**Sources:** `ipfs_accelerate_py/model_catalog/` (`schema.py`, `catalog.py`,
`registry.py`, `resolver.py`, `snapshot.py`, `sources/`);
`ipfs_accelerate_py/api_integrations/model_registry.py`
(`RUNTIME_SOURCE_PRECEDENCE`);
[MODEL_SERVICE_ROUTING.md](MODEL_SERVICE_ROUTING.md)
**Last verified:** `2bf2cebd3` (2026-08-03); schema version, closed operation
set, lifecycle vocabulary, operational tri-state fields, and default source
precedence constants checked against the tree

The AI Service Catalog is the canonical information and resolution plane for
AI services in `ipfs_accelerate_py`. It answers what is known, which operations
are declared, and which binding satisfies a set of constraints. It is not a
fifth inference router.

The LLM, embeddings, multimodal, and voice routers remain the invocation
plane. They own provider construction, requests, batching, streaming, fallback,
timeouts, and provider-specific errors.

```text
static / persistent / router / deployment / peer sources
                         |
                 AIServiceCatalog
          registry + resolver + snapshot
                         |
                    ModelManager
             /           |          \
          Python         MCP       MCP++ peers

Invocation: llm_router | embeddings_router | multimodal_router | voice_router
```

This separation is a security and lifecycle boundary. Importing, listing,
getting, and resolving catalog records must not install software, start a
process, construct a provider client, load model weights, read a credential
store, probe an endpoint, fetch media, or make a model request. A descriptor
can report whether a credential-shaped environment setting is present without
returning its value. Source refresh is separate, explicit, named, and
policy-gated.

## Versioned schema and identities

The v1 Python schema version is `1.0`. Its immutable records are:

| Record | Meaning | Stable identity inputs |
| --- | --- | --- |
| `ProviderDescriptor` | Provider/service and aggregate capabilities | canonical provider name |
| `ModelDescriptor` | Provider-owned model | provider ID and canonical model name |
| `DeploymentDescriptor` | Served endpoint or local deployment | provider/model IDs, name, normalized endpoint |
| `RouterBinding` | Invocation route for a model or deployment | router, provider, model, and deployment IDs |
| `CapabilityDescriptor` | Operations, modalities, media types, and limits | contained in its parent record |
| `Provenance` | Source, source record, observation, expiry, and issuer | contained in its parent record |
| `CatalogSnapshot` | Sorted immutable record collection | canonical snapshot content |

IDs are derived from canonical, non-secret fields. Aliases are lookup aids and
do not change identity. A snapshot `revision` is the content ID of its
canonical records; record order and `created_at` do not change it. The same
content must therefore produce the same revision in the router, ModelManager,
MCP, and MCP++ projections.

Do not confuse three versions:

- catalog record schema: `1.0`;
- catalog snapshot revision/CID: changes with canonical content;
- MCP++ interface: `ai.catalog.v1`, interface version `1.0.0`, with its own
  interface CID.

An interface CID identifies a method schema. It is not a catalog revision.

## Operation taxonomy

The closed v1 operation set is:

| Operation | Meaning |
| --- | --- |
| `text.generate` | Prompt-to-text generation |
| `text.chat` | Structured conversational text |
| `embedding.generate` | Text or supported input to an embedding |
| `vision.generate` | Multimodal/vision generation |
| `audio.transcribe` | Audio to text |
| `audio.synthesize` | Text to audio |
| `batch` | The primary operation supports a batch request |
| `stream` | The primary operation supports streaming |
| `tool.call` | The primary operation supports tool calls |

`batch` and `stream` qualify an invokable operation; they cannot be the only
operations in a capability or binding. Adding an operation requires a schema
review, MCP and MCP++ schema changes, authorization classification, bounded
request/response rules, and cross-surface conformance fixtures.

## Lifecycle and operational state

Lifecycle describes the stage of a record:

`unknown`, `declared`, `configured`, `starting`, `ready`, `degraded`,
`unavailable`, `stopped`, `deprecated`, or `retired`.

Operational facts are independent tri-state values (`true`, `false`, or
unknown):

- `known`: the identity is known;
- `configured`: local configuration exists;
- `authorized`: required authority or credential state is satisfied;
- `reachable`: an endpoint was observed reachable;
- `healthy`: a health observation succeeded;
- `routable`: policy permits the binding to be selected.

None implies another. A static record may be known but have unknown
configuration and reachability. A reachable deployment may be unauthorized.
A healthy provider may still be non-routable under policy. Consumers must
request the state they require; unknown does not satisfy a `true` constraint.

## Sources, precedence, and conflicts

Sources publish complete immutable snapshots with provenance. Default
precedence is:

| Source class | Default precedence |
| --- | ---: |
| Federated peer | `-100` |
| Packaged static inventory | `10` |
| Persistent ModelManager metadata | `20` |
| Router discovery | `30` |
| Active deployment/backend | `40` |
| Legacy runtime API-model additions | `100` |

Higher precedence wins a field-level claim. This order lets router truth
override descriptive seeds and active deployment truth override a static
claim. Remote peers cannot override trusted local identities. Equal-precedence
incompatible claims fail closed where authority is ambiguous and remain
visible as bounded diagnostics; a source must never silently erase another
source's healthy records.

`AIServiceCatalog.refresh()` accepts explicit source names. Side-effecting
sources require a `RefreshPolicy` with `allow_side_effects=True`, optionally
restricted to an allowlist. A failed refresh retains the last published
generation and reports a bounded source failure.

## Query and resolution

`ModelManager` is the public Python facade:

```python
from ipfs_accelerate_py.model_manager import get_default_model_manager

manager = get_default_model_manager()
services = manager.list_services(operation="text.generate", limit=25)
models = manager.list_catalog_models(provider="openrouter", limit=25)
result = manager.resolve(
    operation="text.generate",
    provider="openrouter",
    routable=True,
)

print(services.snapshot_revision)
if result.found:
    print(result.candidates[0].binding.binding_id)
else:
    print(result.reasons)
```

List cursors are bound to one immutable revision. A cursor from an older
revision fails with a typed revision-mismatch result instead of silently
continuing in changed data. Resolution intersects operation, modality, model,
provider, deployment, policy, device, context, health, locality, and
operational-state constraints, then ranks deterministic candidates. It does
not invoke the selected binding.

## Extension boundaries

To add a provider:

1. implement invocation in its owning router;
2. publish a side-effect-free `ProviderDescriptor` and model hints;
3. register aliases only at that router boundary;
4. emit `RouterBinding` records for invokable models;
5. add fake-provider discovery, resolution, and invocation tests;
6. run the cross-router and cross-surface conformance suites.

To add a source, implement `load()` as a bounded metadata projection. Mark
whether refresh may have side effects, provide explicit precedence and
provenance, and preserve valid rows when another row is malformed. Do not move
provider request logic into a source, `AIServiceCatalog`, or `ModelManager`.

To add an MCP operation, define one bounded local tool schema, add it to
`ai.catalog.v1` when it is canonical, assign read/refresh/invoke authority, and
prove local MCP and MCP++ schema parity. Compatibility aliases remain local
unless deliberately versioned into the IDL.

## Security and federation

Catalog inputs are untrusted. The schema rejects secret-bearing keys and
credential-shaped values, malformed identities, unsupported URIs, excess
records, oversized strings, and unknown fields. MCP responses redact raw
deployment endpoints while preserving stable deployment IDs.

Federated advertisements contain a compact catalog CID/revision, operation
summary, interface CIDs, issuer, signature, and expiry—not an unbounded model
inventory. Peer records are signed over canonical content, TTL-bounded,
stale-aware, replay-checked, page-bounded, and fetched only through an
authorized transport. URL/media access is subject to SSRF and size policy.
Unsigned, expired, malformed, oversized, or revision-mismatched data fails
closed.

MCP++ uses separate authorities:

- `ai.catalog/read` for list, get, resolve, and health;
- `ai.catalog/refresh` for explicit refresh;
- `ai.catalog/invoke` for router invocation.

Selection receipts contain candidate IDs, filters, ranking inputs, chosen
binding, fallback boundaries, provenance, revision, and timestamps. They must
not contain prompts, messages, media, output, request bodies, credentials,
headers, or raw endpoint URIs.

## Caching, invalidation, and metrics

The catalog cache stores metadata only. It is never a prompt, media, provider
response, or inference-output cache. Capability metadata and dynamic health
have independent defaults: 300 seconds for capabilities and 15 seconds for
health. Refresh is single-flight per source/view. Registration, deployment
lifecycle, credential-state, peer revision, and explicit-refresh events
invalidate only affected projections. Stale-on-error is explicit policy and
stale records remain observable.

The dependency-free metrics contract exposes bounded labels for:

- `catalog_source_latency_seconds_count` and `_sum`;
- `catalog_cache_hits_total` and `catalog_cache_misses_total`;
- `catalog_stale_records`;
- `catalog_conflicts_total`;
- `catalog_no_match_total`;
- `catalog_resolutions_total`;
- `catalog_health_transitions_total`.

Keep source names and reason/outcome labels bounded. Never label metrics with a
model name, prompt, endpoint, peer payload, or credential.

## Migration and compatibility

Existing callers may migrate one surface at a time:

| Legacy use | Canonical replacement |
| --- | --- |
| `ModelManager.list_models()` returning `ModelMetadata` | `list_catalog_models()` or `list_models(canonical=True)` |
| Static `APIModelRegistry` / `api_models` inventories | their `catalog` property or ModelManager |
| Tuple backend registrations | deployment descriptors |
| Direct registry selection | `ModelManager.resolve()` |
| Historical MCP model discovery | `model_catalog_*` tools |
| MCP `generate_text` | `llm_generate` |
| MCP `generate_embeddings` / `generate_embedding` | `embeddings_generate` |

Compatibility projections preserve documented names and return shapes, but
canonical IDs and the catalog revision are authoritative. Persist old data
before upgrading, verify the migrated snapshot and aliases, and retain the
previous application release until the parity gate passes. Migration is
idempotent: retrying it must not duplicate canonical identities.

## Testing and live smoke

The release gate is offline and deterministic:

```bash
python -m pytest \
  test/test_ai_service_catalog_schema.py \
  test/test_ai_service_catalog_registry.py \
  test/test_ai_service_catalog_sources.py \
  test/test_model_manager_catalog.py \
  test/test_ai_catalog_conformance.py -q
```

Usage-aware routing (optional overlay; modes `off`/`observe`/`shadow`/`assist`/
`enforce`) has a separate offline gate that must stay green before promotion:

```bash
python -m pytest \
  test/test_endpoint_usage_schema.py \
  test/test_endpoint_usage_adapters.py \
  test/test_endpoint_usage_ledger.py \
  test/test_endpoint_usage_routing.py \
  test/test_model_manager_usage_routing.py \
  test/test_llm_router_usage_routing.py \
  test/test_embeddings_router_usage_routing.py \
  test/test_multimodal_router_usage_routing.py \
  test/test_voice_router_usage_routing.py \
  test/test_ai_router_usage_contract.py \
  test/test_endpoint_usage_controls.py \
  test/test_endpoint_usage_conformance.py \
  test/test_endpoint_usage_faults.py \
  test/test_endpoint_usage_rollout.py -q
```

See [Endpoint usage-aware routing plan](ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md)
for window coverage, fault matrix, and staged rollout exit gates. Default
usage tests never open the network; live usage smokes require
`IPFS_ACCELERATE_PY_ENDPOINT_USAGE_LIVE` and a micro-budget cap.

Default catalog conformance uses fake providers and blocks process, network,
credential-file, and model-load side effects during cold listing. Live smokes
are selected per modality, so an operator does not need every modality:

```bash
IPFS_ACCELERATE_PY_AI_CATALOG_LIVE=text,embeddings \
IPFS_ACCELERATE_PY_AI_CATALOG_LIVE_TEXT_PROVIDER=openrouter \
IPFS_ACCELERATE_PY_AI_CATALOG_LIVE_TEXT_MODEL=openai/gpt-4o-mini \
IPFS_ACCELERATE_PY_AI_CATALOG_LIVE_EMBEDDINGS_PROVIDER=openrouter \
python -m pytest test/test_ai_catalog_conformance.py \
  -k opt_in_live_provider_smoke -q
```

Valid selectors are `text`, `embeddings`, `multimodal`, `transcription`, and
`synthesis`; `all` selects all five. Each accepts
`IPFS_ACCELERATE_PY_AI_CATALOG_LIVE_<MODALITY>_PROVIDER` and `_MODEL`.
Selected operations fail on an invocation error. Unselected modalities skip.

## Rollout and rollback

Use staged gates:

1. pass schema, registry, source, security, observability, compatibility, and
   conformance tests offline;
2. expose read-only catalog queries and compare identities/revisions against
   legacy projections in shadow traffic;
3. enable explicit refresh for named sources under least privilege;
4. run opt-in live canaries for the modalities available in the environment;
5. advertise the revision to MCP++ peers;
6. promote invocation traffic while watching conflict, no-match, stale,
   resolution, latency, and health-transition metrics.

Stop promotion on identity or revision drift, import/list side effects,
compatibility fixture drift, unsigned/stale peer acceptance, unexpected
endpoint disclosure, or elevated no-match/health failures.

Rollback is data-preserving:

1. disable peer federation and side-effecting refresh;
2. stop routing new catalog-selected traffic while keeping router invocation
   entry points available;
3. retain and serve the last known immutable snapshot for diagnosis;
4. withdraw the new MCP++ advertisement/traffic where the deployment supports
   it;
5. pin the previous application release and its compatible schema;
6. restore legacy callers, which remain reversible projections;
7. preserve diagnostics, receipts, revisions, and metrics for the incident;
8. rerun offline parity before re-enabling a stage.

Do not rewrite a bad revision in place. Publish corrected canonical content as
a new revision.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No candidates | Inspect resolution reasons and distinguish unknown from false state. Remove only the constraint that policy permits. |
| Alias is ambiguous | Use the stable provider/model ID; ambiguity intentionally fails closed. |
| Cursor revision mismatch | Restart pagination against the returned current revision. |
| Source is stale | Check source state, expiry, cache view, and last refresh error; authorize an explicit named refresh if permitted. |
| Static and runtime values differ | Inspect claims, provenance, precedence, and equal-precedence conflict diagnostics. |
| Peer data is absent | Check signature, issuer trust, expiry/replay window, authority, page bounds, and advertised revision. |
| MCP refresh is denied | Supply refresh authority and explicit source names; read authority is insufficient. |
| Endpoint is redacted | This is expected in MCP. Use the deployment ID within the trusted invocation plane. |
| Listing starts work | Treat as a release blocker; run the cold conformance test and locate the provider/source that crossed the information-plane boundary. |

## Compatibility sunset policy

Legacy Python registries and MCP names are deprecated compatibility
projections, but removal is not scheduled and deprecation is explicitly
reversible. No date should be inferred from this document.

A future removal requires all of the following:

- a separately announced release and sunset date;
- at least one documented migration window;
- passing public Python and MCP compatibility fixtures until that window ends;
- parity telemetry showing no unexplained identity or revision drift;
- replacement coverage for every supported operation;
- an operator-tested rollback release;
- release notes identifying removed names and equivalent canonical names.

Any failed gate postpones removal. Security fixes may restrict unsafe behavior,
but should preserve a bounded compatibility error where preserving execution
would be unsafe.

See [LLM Router](../LLM_ROUTER.md), [MCP Server](../MCP_SERVER.md),
[router ownership](../ROUTER_OWNERSHIP.md), the
[canonical MCP runtime](../../ipfs_accelerate_py/mcp_server/README.md), and
[MCP++ records](../../mcpplusplus/README.md).
