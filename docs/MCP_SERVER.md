# MCP Server: AI Catalog and Router Tools

The canonical MCP runtime is `ipfs_accelerate_py.mcp_server`. Its AI tools
separate catalog information from router invocation and carry the selected
catalog revision across both planes.

- Catalog tools query the `ModelManager` facade. They never invoke a provider.
- Invocation tools resolve one immutable revision, then call the owning LLM,
  embeddings, multimodal, or voice router.
- MCP++ `ai.catalog.v1` publishes the same eleven canonical operations with
  bounded schemas and separate read, refresh, and invoke authorities.

For server installation and transport startup, see
[MCP setup](guides/MCP_SETUP_GUIDE.md) and the
[runtime README](../ipfs_accelerate_py/mcp_server/README.md).

## Canonical tools

Catalog query tools:

| Tool | Purpose |
| --- | --- |
| `model_catalog_list_services` | Page provider/service descriptors |
| `model_catalog_list_models` | Page canonical model descriptors |
| `model_catalog_get` | Get one provider, model, deployment, or binding |
| `model_catalog_resolve` | Rank bindings for typed constraints |
| `model_catalog_health` | Read already-published source and record health |
| `model_catalog_refresh` | Explicitly refresh named sources with authority |

Router invocation tools:

| Tool | Owner | Operation |
| --- | --- | --- |
| `llm_generate` | `llm_router` | `text.generate` |
| `embeddings_generate` | `embeddings_router` | `embedding.generate` |
| `multimodal_generate` | `multimodal_router` | `vision.generate` |
| `voice_transcribe` | `voice_router` | `audio.transcribe` |
| `voice_synthesize` | `voice_router` | `audio.synthesize` |

Compatibility names `generate_text`, `generate_embeddings`, and
`generate_embedding` remain registered for existing MCP callers. They project
to the canonical text/embedding implementation. Historical `model_*` tools
also remain available while callers migrate to `model_catalog_*`.

## Revisions, pagination, and receipts

Successful catalog and invocation envelopes include `schema_version` and
`catalog_revision`. Invocation output also includes the selected binding and a
bounded receipt. A client that needs reproducibility should send the revision
it resolved; a changed revision fails with a typed mismatch instead of routing
against different content.

List pagination uses revision-bound cursors with at most 1,000 items per page.
If content changes between pages, restart from the first page at the new
revision. Do not combine pages from different revisions.

The selected binding exposes stable IDs, not credentials. Deployment endpoint
URIs are redacted from MCP responses. Receipts include selection facts and
revision but exclude prompts, media, model output, headers, credentials, and
raw endpoints.

## Bounded schemas and errors

Input schemas reject unknown properties and bound free text, arrays, pages,
timeouts, output sizes, embedding dimensions, stream chunks, and media. Media
must use one of the explicit inline, URI, or artifact-reference variants.
Remote media is denied unless the operation and authority explicitly allow it.

Common error codes include:

- `invalid_filter` or `invalid_request` for malformed bounded inputs;
- `no_match` for an empty constraint intersection;
- `ambiguous_identifier` for an alias that maps to multiple canonical IDs;
- `catalog_revision_mismatch` or `cursor_revision_mismatch`;
- `refresh_denied`, `invalid_sources`, or `source_refresh_failed`;
- typed invocation, timeout, media, and output-bound failures.

Errors retain the safe catalog revision when available and do not echo
credential values, private endpoint details, prompts, or media bodies.

## Authority and refresh

MCP++ separates authority by effect:

| Authority | Permitted operations |
| --- | --- |
| `ai.catalog/read` | list, get, resolve, health |
| `ai.catalog/refresh` | named explicit refresh |
| `ai.catalog/invoke` | five modality invocation tools |

Read authority cannot refresh or invoke. Refresh requires a non-empty source
list and an explicit true authority field at the local MCP boundary.
Side-effecting sources are additionally checked by catalog refresh policy.
Invocation authority does not grant catalog refresh.

Remote advertisements and catalog pages are untrusted. Signatures, issuers,
expiry, replay state, catalog revision, page bounds, URL/media policy, and
capabilities are validated before records enter a peer source. A remote record
cannot override a trusted local identity.

## Migration

Migrate discovery first, invocation second:

1. replace historical model enumeration with
   `model_catalog_list_services`/`model_catalog_list_models`;
2. persist stable IDs and `catalog_revision`, not list positions;
3. replace registry heuristics with `model_catalog_resolve`;
4. change `generate_text` to `llm_generate`;
5. change `generate_embeddings` or `generate_embedding` to
   `embeddings_generate`;
6. adopt the canonical multimodal and voice tool names;
7. validate output against the MCP++ `ai.catalog.v1` schema when interoperating
   with MCP++ peers.

Compatibility aliases have no scheduled removal date. See the
[catalog sunset policy](architecture/AI_SERVICE_CATALOG.md#compatibility-sunset-policy)
for the gates required before a future removal.

## Testing and rollout

Default tests use fake routers/providers and require no network:

```bash
python -m pytest \
  test/mcp_server/test_ai_catalog_tools.py \
  test/mcp_server/test_ai_router_text_embedding_tools.py \
  test/mcp_server/test_ai_router_vision_voice_tools.py \
  test/test_mcplusplus_ai_catalog_idl.py \
  test/test_mcplusplus_ai_catalog.py \
  test/test_ai_catalog_conformance.py -q
```

Usage control tools (`model_catalog_usage`, `model_catalog_usage_metrics`,
`route_preview`) share reason codes and authorities with the Python control
service and MCP++ `ai.usage.v1` IDL. Offline parity and rollout proofs:

```bash
python -m pytest \
  test/test_endpoint_usage_controls.py \
  test/test_endpoint_usage_conformance.py \
  test/test_endpoint_usage_faults.py \
  test/test_endpoint_usage_rollout.py -q
```

Read/preview paths are side-effect free and never reserve capacity. Admin
mutations require `ai.usage/admin`, expected revision, lease/fence, and
idempotency. Distributed admission without a fenced coordinator fails closed.

Roll out read-only catalog tools first and compare their stable IDs/revisions
with Python and legacy views. Then enable refresh for named sources, run
per-modality live canaries, publish MCP++ advertisements, and finally promote
invocation traffic. Monitor catalog conflicts, no matches, stale records,
source latency, cache hit/miss rates, resolution outcomes, and health
transitions.

For opt-in live smoke configuration, see
[Testing and live smoke](architecture/AI_SERVICE_CATALOG.md#testing-and-live-smoke).
Unselected modalities skip, so a deployment can test only providers it has.

## Rollback and troubleshooting

On identity/revision drift, unexpected side effects, schema incompatibility,
security failure, or elevated routing errors:

1. stop new catalog-selected invocation traffic;
2. disable peer federation and side-effecting refresh;
3. keep the last immutable snapshot for read-only diagnosis;
4. withdraw or stop routing to the new MCP++ advertisement;
5. pin the previous server release;
6. restore compatible tool names and rerun offline parity tests.

Troubleshooting:

| Symptom | Action |
| --- | --- |
| `no_match` | Inspect the safe resolution reasons and required operational state. |
| `ambiguous_identifier` | Retry with a stable ID rather than an alias. |
| revision mismatch | Re-list or re-resolve and bind the request to the returned revision. |
| refresh denied | Use refresh—not read—authority and specify named sources. |
| stale peer | Check signature, issuer, expiry, replay window, and advertised revision. |
| redacted endpoint | Expected behavior; invoke by binding/deployment ID. |
| oversized media/output | Reduce the bounded request or select a compatible binding. |
| registration resolves a manager/provider | Release blocker; tool registration must remain cold. |

See the [AI Service Catalog architecture](architecture/AI_SERVICE_CATALOG.md),
[LLM Router](LLM_ROUTER.md), and [MCP++ records](../mcpplusplus/README.md).
