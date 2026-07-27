# AI Service Catalog Task Board

This board is the executable projection of the
[AI Service Catalog objective heap](ai_service_catalog.objectives.md). It is
ordered by explicit dependency edges and predicted file ownership so the agent
supervisor can safely use six implementation lanes after the shared schema is
stable.

Program invariants:

- The catalog is the canonical discovery and resolution plane; the four
  modality routers remain the invocation plane.
- Known, configured, authorized, reachable, healthy, and routable are distinct
  states.
- Import, listing, and implicit discovery have no installation, process,
  network, credential, or model-loading side effects.
- Canonical identities and content IDs contain no secrets and are deterministic
  across Python, CLI, MCP, and MCP++ projections.
- Remote catalog data is untrusted, signed, bounded, stale-aware, and
  capability-gated.
- Existing APIs remain available through tested compatibility projections.
- Concurrent work must preserve user changes and re-read the latest merge
  target before modifying shared router or registry files.
- Production launches must pass
  `python scripts/ops/ai_service_catalog_supervisor.py preflight`. The initial
  baseline uses six lanes in a fresh runtime namespace with objective refill
  disabled; low-backlog refill may be enabled only after the seeded graph is
  nearly drained.
- The task board and objective heap are tracked, operator-protected
  control-plane inputs. Refill-generated implementation tasks may cite their
  discovery receipts, but must never claim the receipts, board, or heap as
  editable outputs.

## AICAT-001 Define versioned catalog schemas and stable identities

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: catalog-contracts
- Depends on:
- Goal id: AICAT-G010
- Outputs: ipfs_accelerate_py/model_catalog/__init__.py, ipfs_accelerate_py/model_catalog/schema.py, ipfs_accelerate_py/model_catalog/identity.py, test/test_ai_service_catalog_schema.py
- Validation: python -m pytest test/test_ai_service_catalog_schema.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/contracts
- Parallel lane: catalog-schema
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 14000
- Predicted files: ipfs_accelerate_py/model_catalog/__init__.py, ipfs_accelerate_py/model_catalog/schema.py, ipfs_accelerate_py/model_catalog/identity.py, test/test_ai_service_catalog_schema.py
- Allow concurrent with:
- Conflict policy: This task solely owns the initial model_catalog package contracts. Do not modify routers, ModelManager, backend managers, MCP, or legacy registries.
- Preconditions: Inspect current ModelMetadata, ProviderInfo, backend tuples, router provider records, MCP service records, and API model registries before freezing field names.
- Effects: Add immutable bounded ProviderDescriptor, ModelDescriptor, DeploymentDescriptor, RouterBinding, CapabilityDescriptor, LifecycleState, CatalogSnapshot metadata, and canonical identity helpers.
- Acceptance: Define versioned schemas and the operation taxonomy text.generate, text.chat, embedding.generate, vision.generate, audio.transcribe, audio.synthesize, batch, stream, and tool.call. Keep known, configured, authorized, reachable, healthy, and routable orthogonal. Canonical serialization and IDs must be deterministic, order independent where specified, bounded, round-trippable, and free of credentials. Validate aliases, URIs, timestamps, sizes, capability combinations, schema versions, unknown fields, redaction, collision inputs, and malformed records. Cold import must have no provider, process, network, install, or model-load side effects.

## AICAT-002 Implement the catalog registry, resolver, and snapshots

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: catalog-contracts
- Depends on: AICAT-001
- Goal id: AICAT-G010
- Outputs: ipfs_accelerate_py/model_catalog/registry.py, ipfs_accelerate_py/model_catalog/resolver.py, ipfs_accelerate_py/model_catalog/snapshot.py, test/test_ai_service_catalog_registry.py
- Validation: python -m pytest test/test_ai_service_catalog_registry.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/contracts
- Parallel lane: catalog-registry
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 13000
- Predicted files: ipfs_accelerate_py/model_catalog/registry.py, ipfs_accelerate_py/model_catalog/resolver.py, ipfs_accelerate_py/model_catalog/snapshot.py, test/test_ai_service_catalog_registry.py
- Allow concurrent with: AICAT-003, AICAT-004, AICAT-005, AICAT-006, AICAT-007
- Conflict policy: Consume AICAT-001 contracts without modifying routers or external registries. Any schema gap must be proved by a focused failing test before a narrow contract amendment.
- Preconditions: AICAT-001 schemas and identity rules pass their focused tests.
- Effects: Add deterministic registration, source precedence, alias resolution, filtering, pagination, collision handling, immutable snapshots, revisions, CIDs, and bounded diagnostic records.
- Acceptance: Duplicate canonical records coalesce while incompatible claims remain visible and fail closed where authority is ambiguous. Resolution accepts operation, modality, model, provider, deployment, policy, device, context, health, and locality constraints and returns ranked candidates plus reasons. Pagination is stable under one snapshot. Snapshot CID changes only with canonical content. Registry iteration is deterministic and thread safe. Tests cover aliases, collisions, precedence, stale inputs, filter intersections, no candidates, pagination cursors, concurrent readers, deterministic ranking, and serialization round trips.

## AICAT-003 Publish LLM router provider and model descriptors

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: router-discovery
- Depends on: AICAT-001
- Goal id: AICAT-G020
- Outputs: ipfs_accelerate_py/llm_router.py, test/test_llm_router_catalog_discovery.py
- Validation: python -m pytest test/test_llm_router_catalog_discovery.py test/test_llm_router_integration.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/routers/llm
- Parallel lane: llm-router-catalog
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 15000
- Predicted files: ipfs_accelerate_py/llm_router.py, test/test_llm_router_catalog_discovery.py
- Allow concurrent with: AICAT-002, AICAT-004, AICAT-005, AICAT-006, AICAT-007
- Conflict policy: Own only LLM router discovery and its tests. Preserve all current provider changes, generation behavior, fallback, batching, caching, and provider aliases; re-read the latest target before editing this shared file.
- Preconditions: Shared catalog descriptors are available and current llm_router provider registration and lazy resolution behavior have been mapped.
- Effects: Add side-effect-free list_providers, get_provider_descriptor, list_models, resolve_model, and catalog source projection behavior for text providers.
- Acceptance: Every built-in and dynamically registered LLM provider emits a shared descriptor with operations, aliases, model hints, context limits when known, streaming, batching, tools, locality, device, authorization, and readiness semantics. Listing must not instantiate optional clients, install CLIs, access credentials, probe endpoints, or load models. Explicit model resolution remains behaviorally compatible with generate_text. Unknown capabilities are represented as unknown rather than false. Existing provider order, cache keys, errors, and tests remain compatible.

## AICAT-004 Publish embeddings router provider and model descriptors

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: router-discovery
- Depends on: AICAT-001
- Goal id: AICAT-G020
- Outputs: ipfs_accelerate_py/embeddings_router.py, test/test_embeddings_router_catalog_discovery.py
- Validation: python -m pytest test/test_embeddings_router_catalog_discovery.py test/test_embeddings_router_contract.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/routers/embeddings
- Parallel lane: embeddings-router-catalog
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 13000
- Predicted files: ipfs_accelerate_py/embeddings_router.py, test/test_embeddings_router_catalog_discovery.py
- Allow concurrent with: AICAT-002, AICAT-003, AICAT-005, AICAT-006, AICAT-007
- Conflict policy: Own only embeddings router discovery and focused tests. Preserve current provider, batching, normalization, device, and fallback work; re-read the latest target before editing.
- Preconditions: Shared catalog descriptors exist and embeddings provider resolution and output contracts have been mapped.
- Effects: Add compatible discovery methods and a catalog source projection for embedding providers and models.
- Acceptance: Descriptors report embedding dimensions, input types, maximum inputs or tokens when known, normalization, batching, locality, device, authorization, and readiness without constructing heavy clients or loading weights. Explicit resolution agrees with the provider selected by existing embedding generation for the same constraints. Dynamic registration updates discovery deterministically. Unknown metadata remains unknown. Existing contracts and provider behavior stay green.

## AICAT-005 Publish multimodal router provider and model descriptors

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: router-discovery
- Depends on: AICAT-001
- Goal id: AICAT-G020
- Outputs: ipfs_accelerate_py/multimodal_router.py, test/test_multimodal_router_catalog_discovery.py
- Validation: python -m pytest test/test_multimodal_router_catalog_discovery.py test/test_multimodal_router.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/routers/multimodal
- Parallel lane: multimodal-router-catalog
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 13000
- Predicted files: ipfs_accelerate_py/multimodal_router.py, test/test_multimodal_router_catalog_discovery.py
- Allow concurrent with: AICAT-002, AICAT-003, AICAT-004, AICAT-006, AICAT-007
- Conflict policy: Own only multimodal router discovery and focused tests. Preserve current generation and provider behavior and re-read the latest target before editing this shared file.
- Preconditions: Shared catalog descriptors exist and multimodal registration, modality, and provider selection paths have been mapped.
- Effects: Add compatible discovery methods and a catalog source projection for image, vision, and other multimodal providers.
- Acceptance: Descriptors distinguish accepted and produced modalities, media MIME families, URI versus inline input, size and count limits when known, streaming, batching, locality, device, authorization, and readiness. Listing performs no media fetch, client construction, network probe, or model load. Resolution and existing invocation select compatible providers for identical constraints. Dynamic providers and aliases appear deterministically without changing generation behavior.

## AICAT-006 Publish voice router provider and model descriptors

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: router-discovery
- Depends on: AICAT-001
- Goal id: AICAT-G020
- Outputs: ipfs_accelerate_py/voice_router.py, test/test_voice_router_catalog_discovery.py
- Validation: python -m pytest test/test_voice_router_catalog_discovery.py test/test_voice_router.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/routers/voice
- Parallel lane: voice-router-catalog
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 13000
- Predicted files: ipfs_accelerate_py/voice_router.py, test/test_voice_router_catalog_discovery.py
- Allow concurrent with: AICAT-002, AICAT-003, AICAT-004, AICAT-005, AICAT-007
- Conflict policy: Own only voice router discovery and focused tests. Preserve transcription, synthesis, streaming, and provider behavior; re-read the latest target before editing.
- Preconditions: Shared catalog descriptors exist and voice provider registration and operation dispatch have been mapped.
- Effects: Add compatible discovery methods and a catalog source projection for transcription and synthesis providers.
- Acceptance: Descriptors distinguish transcription from synthesis and report language, voice, audio MIME, sample-rate, duration or size limits, streaming, batching, locality, device, authorization, and readiness when known. Listing does not capture audio, fetch media, instantiate optional clients, probe networks, or load models. Existing invocation and explicit catalog resolution agree for equivalent constraints. Dynamic registration and aliases are deterministic and existing tests remain green.

## AICAT-007 Add persistent and static catalog source adapters

- Status: completed
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: catalog-manager
- Depends on: AICAT-001
- Goal id: AICAT-G030
- Outputs: ipfs_accelerate_py/model_catalog/sources/__init__.py, ipfs_accelerate_py/model_catalog/sources/persistent.py, ipfs_accelerate_py/model_catalog/sources/static.py, test/test_ai_service_catalog_static_sources.py
- Validation: python -m pytest test/test_ai_service_catalog_static_sources.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/sources
- Parallel lane: persistent-static-sources
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 10000
- Predicted files: ipfs_accelerate_py/model_catalog/sources/__init__.py, ipfs_accelerate_py/model_catalog/sources/persistent.py, ipfs_accelerate_py/model_catalog/sources/static.py, test/test_ai_service_catalog_static_sources.py
- Allow concurrent with: AICAT-002, AICAT-003, AICAT-004, AICAT-005, AICAT-006
- Conflict policy: Create source adapters only. Do not modify ModelManager, API registries, backend managers, or routers.
- Preconditions: Shared catalog records and identity helpers pass focused tests; inspect existing ModelManager persistence and static registry schemas.
- Effects: Add pure adapters that project persisted ModelManager metadata and packaged static provider or model inventories into canonical records with explicit provenance.
- Acceptance: Adapters accept injected records or paths, perform no implicit I/O beyond the supplied local source, preserve source revision and timestamps, redact secret-shaped fields, bound counts and field sizes, report malformed rows without discarding valid rows, and never assert dynamic health from static data. Duplicate seeds map to stable canonical identities. Tests cover legacy field names, partial rows, invalid data, precedence metadata, redaction, deterministic ordering, and empty sources.

## AICAT-008 Assemble router and metadata sources into AIServiceCatalog

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: catalog-manager
- Depends on: AICAT-002, AICAT-003, AICAT-004, AICAT-005, AICAT-006, AICAT-007
- Goal id: AICAT-G030
- Outputs: ipfs_accelerate_py/model_catalog/catalog.py, ipfs_accelerate_py/model_catalog/sources/routers.py, test/test_ai_service_catalog_sources.py
- Validation: python -m pytest test/test_ai_service_catalog_sources.py test/test_ai_service_catalog_registry.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/aggregation
- Parallel lane: catalog-aggregation
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 15000
- Predicted files: ipfs_accelerate_py/model_catalog/catalog.py, ipfs_accelerate_py/model_catalog/sources/routers.py, test/test_ai_service_catalog_sources.py
- Allow concurrent with: AICAT-010
- Conflict policy: Own catalog orchestration and router source adapters only. Do not modify router implementations or ModelManager.
- Preconditions: Registry, resolver, snapshots, all four router discovery surfaces, and persistent or static source adapters are complete.
- Effects: Add AIServiceCatalog source registration, deterministic source snapshots, merge precedence, conflict diagnostics, filters, resolution, refresh policy, and aggregate health projection.
- Acceptance: Router truth, persistent metadata, and static capability records merge by canonical identities without one source silently overwriting another. Source precedence is explicit and provenance remains queryable. Listing and normal resolution are side-effect free. Explicit refresh selects named sources and requires policy for side-effecting sources. One failing source cannot erase healthy records from others. Source and output bounds, deterministic ordering, snapshot isolation, concurrent reads, partial failures, and conflict diagnostics have focused tests.

## AICAT-009 Expose the canonical catalog through ModelManager

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: catalog-manager
- Depends on: AICAT-008
- Goal id: AICAT-G030
- Outputs: ipfs_accelerate_py/model_manager.py, test/test_model_manager_catalog.py
- Validation: python -m pytest test/test_model_manager_catalog.py ipfs_accelerate_py/mcp/tests/test_model_manager_improvements.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/model-manager
- Parallel lane: model-manager-catalog
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 16000
- Predicted files: ipfs_accelerate_py/model_manager.py, test/test_model_manager_catalog.py
- Allow concurrent with: AICAT-011
- Conflict policy: Own ModelManager facade and compatibility behavior. Preserve all current persistence, endpoint, GraphRAG, and CRUD changes; do not move provider invocation into ModelManager.
- Preconditions: The aggregate AIServiceCatalog API is stable and existing ModelManager public behavior has compatibility fixtures.
- Effects: Add catalog injection and list_services, list_models, get_service, get_model_descriptor, resolve, health, snapshot, and explicit refresh methods while projecting legacy model metadata.
- Acceptance: ModelManager is the public canonical catalog facade and returns bounded versioned records, stable pagination, catalog revision, source provenance, and typed no-match or ambiguity diagnostics. Read methods have no side effects. Refresh is explicit and policy-gated. Existing add, update, remove, endpoint, load, save, search, and MCP consumers retain return shapes or documented compatibility wrappers. Persistence migration is idempotent and rollback-safe. Tests cover old registries, empty state, concurrent reads, source failure, refresh denial, and no invocation ownership.

## AICAT-010 Represent served endpoints and backends as deployments

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backend-convergence
- Depends on: AICAT-002, AICAT-007
- Goal id: AICAT-G040
- Outputs: ipfs_accelerate_py/model_catalog/sources/deployments.py, test/test_ai_catalog_deployment_sources.py
- Validation: python -m pytest test/test_ai_catalog_deployment_sources.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/backends/deployments
- Parallel lane: deployment-sources
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 11000
- Predicted files: ipfs_accelerate_py/model_catalog/sources/deployments.py, test/test_ai_catalog_deployment_sources.py
- Allow concurrent with: AICAT-008
- Conflict policy: Add pure deployment source adapters and tests only. Do not modify ModelManager or InferenceBackendManager yet.
- Preconditions: Catalog deployment records exist; inspect ModelManager endpoint records, InferenceBackendManager registrations, and MCP++ served-model advertisements.
- Effects: Project served OpenAI-compatible endpoints and registered backend state into deployment descriptors with injected liveness and readiness observations.
- Acceptance: Deployment identity includes canonical service, model, provider, protocol, endpoint identity, and locality without credentials or raw bearer URLs. Configured, reachable, live, ready, healthy, and routable remain distinct. Source adapters never send an inference request and use injected probes only during explicit refresh. Health samples carry timestamp, TTL, provenance, and bounded diagnostics. Tests cover local and remote endpoints, aliases, stopped servers, stale probes, malformed URLs, redaction, duplicate endpoints, and deterministic snapshots.

## AICAT-011 Converge InferenceBackendManager on typed catalog records

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backend-convergence
- Depends on: AICAT-008, AICAT-010
- Goal id: AICAT-G040
- Outputs: ipfs_accelerate_py/inference_backend_manager.py, test/test_inference_backend_catalog.py
- Validation: python -m pytest test/test_inference_backend_catalog.py test/test_inference_backend_manager.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/backends/manager
- Parallel lane: backend-manager-catalog
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 15000
- Predicted files: ipfs_accelerate_py/inference_backend_manager.py, test/test_inference_backend_catalog.py
- Allow concurrent with: AICAT-009
- Conflict policy: Own InferenceBackendManager integration and tests. Preserve current backend and Meta model API changes; consume catalog descriptors without changing router or ModelManager files.
- Preconditions: Aggregate catalog and deployment source contracts are complete; map every existing tuple index and provider key before migration.
- Effects: Replace internal unversioned provider tuple assumptions with typed provider, model, router-binding, and deployment descriptors while retaining compatibility accessors.
- Acceptance: Registration, lookup, provider selection, endpoint lifecycle, status, and invocation delegation use named typed fields. Existing tuple-shaped inputs remain accepted through one deprecation adapter and produce equivalent behavior. The manager publishes deployment changes to the catalog source without making ModelManager an invocation path. Dynamic register and unregister update snapshots deterministically. Tests cover old tuples, new records, malformed registrations, aliases, concurrent updates, liveness separation, delegation, and unchanged provider behavior.

## AICAT-012 Replace duplicate API model registries with compatibility projections

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: backend-convergence
- Depends on: AICAT-009, AICAT-011
- Goal id: AICAT-G040
- Outputs: ipfs_accelerate_py/api_integrations/model_registry.py, ipfs_accelerate_py/api_backends/api_models_registry.py, test/test_ai_catalog_legacy_registry.py
- Validation: python -m pytest test/test_ai_catalog_legacy_registry.py test/test_api_models_registry.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/backends/legacy
- Parallel lane: legacy-registry-projections
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 15000
- Predicted files: ipfs_accelerate_py/api_integrations/model_registry.py, ipfs_accelerate_py/api_backends/api_models_registry.py, test/test_ai_catalog_legacy_registry.py
- Allow concurrent with: AICAT-014, AICAT-015
- Conflict policy: Own only the two duplicate API registries and compatibility tests. Preserve current API model additions and do not delete public imports or return types.
- Preconditions: ModelManager catalog facade and typed backend registration are complete; enumerate all imports and callers of both legacy registries.
- Effects: Make legacy registries seed or query the canonical catalog through explicit adapters and add deprecation metadata without breaking callers.
- Acceptance: Static model knowledge has one canonical seed path and legacy list, search, recommend, validate, get, add, and export behavior remains available as a deterministic projection. Provider and model aliases map to canonical IDs. Runtime additions persist through the supported catalog source rather than forked globals. Cold import performs no network discovery. Generated parity fixtures fail on field drift, duplicate identities, lost provider metadata, or incompatible return shapes. Deprecation is documented and reversible.

## AICAT-013 Add MCP catalog query and resolution tools

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp-ai-services
- Depends on: AICAT-009
- Goal id: AICAT-G050
- Outputs: ipfs_accelerate_py/mcp_server/tools/model_tools/native_model_tools.py, test/mcp_server/test_ai_catalog_tools.py
- Validation: python -m pytest test/mcp_server/test_ai_catalog_tools.py ipfs_accelerate_py/mcp/tests/test_model_manager_improvements.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/mcp/query
- Parallel lane: mcp-catalog-query
- Resource class: mcp-integration
- Token class: large
- Estimated tokens: 14000
- Predicted files: ipfs_accelerate_py/mcp_server/tools/model_tools/native_model_tools.py, test/mcp_server/test_ai_catalog_tools.py
- Allow concurrent with: AICAT-011
- Conflict policy: Own native model query tools and focused tests. Preserve legacy tool names and avoid querying Hugging Face or alternate registries when the ModelManager catalog is authoritative.
- Preconditions: ModelManager exposes the stable catalog facade and native model tool registration behavior is mapped.
- Effects: Add model_catalog_list_services, model_catalog_list_models, model_catalog_get, model_catalog_resolve, model_catalog_health, and privileged model_catalog_refresh tools with compatibility wrappers.
- Acceptance: Query tools delegate to ModelManager, expose schema version and catalog revision, support bounded filters and pagination, return typed ambiguity and no-match errors, and never expose credentials or raw private endpoint secrets. Read tools have no side effects. Refresh requires explicit authority and named sources. Existing search, load, save, and endpoint tools retain compatibility. Python and MCP results have canonical identity parity. Tests cover malformed filters, cursor revision mismatch, large result bounds, denied refresh, source failure, redaction, and cold registration.

## AICAT-014 Route MCP text and embeddings through canonical routers

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp-ai-services
- Depends on: AICAT-003, AICAT-004, AICAT-013
- Goal id: AICAT-G050
- Outputs: ipfs_accelerate_py/mcp_server/tools/ai_router_tools/__init__.py, ipfs_accelerate_py/mcp_server/tools/ai_router_tools/text_embedding.py, test/mcp_server/test_ai_router_text_embedding_tools.py
- Validation: python -m pytest test/mcp_server/test_ai_router_text_embedding_tools.py test/mcp_server/test_ai_catalog_tools.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/mcp/text-embedding
- Parallel lane: mcp-text-embedding
- Resource class: mcp-integration
- Token class: large
- Estimated tokens: 15000
- Predicted files: ipfs_accelerate_py/mcp_server/tools/ai_router_tools/__init__.py, ipfs_accelerate_py/mcp_server/tools/ai_router_tools/text_embedding.py, test/mcp_server/test_ai_router_text_embedding_tools.py
- Allow concurrent with: AICAT-012, AICAT-015
- Conflict policy: Establish the new provider-neutral ai_router_tools package for text and embeddings. Delegate invocation to llm_router and embeddings_router and do not add a fifth routing implementation.
- Preconditions: LLM and embeddings discovery plus ModelManager catalog query tools are complete; inspect current shared inference and embedding MCP tool behavior for compatibility.
- Effects: Add typed llm_generate and embeddings_generate tools with catalog resolution, canonical router dispatch, bounded requests and outputs, streaming descriptors, and selection metadata.
- Acceptance: Text calls invoke llm_router.generate_text through its supported contract and embedding calls invoke embeddings_router rather than the ipfs_datasets implementation or a nonexistent backend-manager generate method. Explicit service, model, provider, policy, and device constraints are resolved against one catalog revision. Return selected router binding and a bounded receipt without prompt text or secrets. Enforce input count, text bytes, dimensions, output bytes, timeout, cancellation, and streaming limits. Compatibility tool aliases delegate to the canonical implementation. Tests use fake routers and cover mismatch, denial, fallback boundaries, batch bounds, cancellation, errors, and Python-to-MCP parity.

## AICAT-015 Route MCP multimodal and voice operations through canonical routers

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcp-ai-services
- Depends on: AICAT-005, AICAT-006, AICAT-013
- Goal id: AICAT-G050
- Outputs: ipfs_accelerate_py/mcp_server/tools/ai_router_tools/vision_voice.py, test/mcp_server/test_ai_router_vision_voice_tools.py
- Validation: python -m pytest test/mcp_server/test_ai_router_vision_voice_tools.py test/mcp_server/test_ai_catalog_tools.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/mcp/vision-voice
- Parallel lane: mcp-vision-voice
- Resource class: mcp-integration
- Token class: large
- Estimated tokens: 15000
- Predicted files: ipfs_accelerate_py/mcp_server/tools/ai_router_tools/vision_voice.py, test/mcp_server/test_ai_router_vision_voice_tools.py
- Allow concurrent with: AICAT-012, AICAT-014
- Conflict policy: Own multimodal and voice tool implementation in a separate module. Do not modify router invocation internals or duplicate media fetching.
- Preconditions: Multimodal and voice discovery plus ModelManager catalog query tools are complete; current media input and output contracts are mapped.
- Effects: Add multimodal_generate, voice_transcribe, and voice_synthesize tools with catalog resolution and canonical router dispatch.
- Acceptance: Each tool resolves the required operation and modality against one catalog snapshot and dispatches only through multimodal_router or voice_router. Schemas distinguish URI, artifact reference, and bounded inline media; remote fetch is delegated to an allowlisted media layer and is disabled by default. Enforce MIME, bytes, item count, duration, sample rate, dimensions, timeout, cancellation, output, and streaming limits. Return selected binding and bounded receipt without media payloads or secrets. Tests cover wrong modality, unsupported MIME, oversized input, SSRF-shaped URI rejection, cancellation, provider errors, and Python-to-MCP parity.

## AICAT-016 Define the ai.catalog.v1 MCP++ interface

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcplusplus-federation
- Depends on: AICAT-013, AICAT-014, AICAT-015
- Goal id: AICAT-G060
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py, test/test_mcplusplus_ai_catalog_idl.py
- Validation: python -m pytest test/test_mcplusplus_ai_catalog_idl.py test/test_mcplusplus_idl_registry.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/mcplusplus/idl
- Parallel lane: mcplusplus-catalog-idl
- Resource class: mcp-integration
- Token class: large
- Estimated tokens: 14000
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py, test/test_mcplusplus_ai_catalog_idl.py
- Allow concurrent with: AICAT-012
- Conflict policy: Own IDL descriptors and conformance tests only. Do not alter service advertisement or transport behavior yet.
- Preconditions: Local MCP catalog and modality invocation schemas are stable and versioned.
- Effects: Register ai.catalog.v1 query, resolution, health, refresh, text, embedding, multimodal, transcription, and synthesis operation descriptors with deterministic interface CIDs.
- Acceptance: IDL input and output records match local MCP schemas, separate read, refresh, and invoke authority, include schema and catalog revisions, declare pagination and streaming semantics, and bound every string, list, byte, media, timeout, and diagnostic field. Interface CIDs are deterministic and change on incompatible schema edits. Unknown versions and operations fail closed with upgrade metadata. Existing MCP++ interfaces retain their CIDs and behavior. Round-trip, CID stability, compatibility, malformed input, and size-bound tests pass.

## AICAT-017 Advertise and federate compact MCP++ catalog snapshots

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcplusplus-federation
- Depends on: AICAT-010, AICAT-016
- Goal id: AICAT-G060
- Outputs: ipfs_accelerate_py/model_catalog/sources/peers.py, ipfs_accelerate_py/mcplusplus_module/service_registry.py, ipfs_accelerate_py/mcplusplus_module/trio/server.py, test/test_mcplusplus_ai_catalog.py
- Validation: python -m pytest test/test_mcplusplus_ai_catalog.py ipfs_accelerate_py/mcplusplus_module/tests/test_trio_server.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/mcplusplus/federation
- Parallel lane: mcplusplus-catalog-federation
- Resource class: network-simulated
- Token class: large
- Estimated tokens: 17000
- Predicted files: ipfs_accelerate_py/model_catalog/sources/peers.py, ipfs_accelerate_py/mcplusplus_module/service_registry.py, ipfs_accelerate_py/mcplusplus_module/trio/server.py, test/test_mcplusplus_ai_catalog.py
- Allow concurrent with: AICAT-012
- Conflict policy: Own peer source and MCP++ service advertisement integration. Preserve current Trio server and service registry behavior and use injected transports for tests.
- Preconditions: Deployment descriptors and ai.catalog.v1 IDL are complete; current advertisement signing and startup snapshot behavior are mapped.
- Effects: Replace startup-only served model lists with compact catalog revision adverts and add bounded peer catalog fetching as an untrusted source.
- Acceptance: Advertisements carry issuer, service identity, catalog CID, revision, operation summary, interface CIDs, endpoint protocol, issued time, expiry, and signature rather than an unbounded model inventory. Dynamic local catalog changes emit a new revision without restarting the server. Peer sources fetch pages only through injected authorized transports, verify canonical content against advertised CIDs, preserve peer provenance, isolate identities by trust domain, and never override trusted local records. Tests cover update propagation, pagination, partial peers, duplicate peers, stale snapshots, CID mismatch, disconnect, restart, and deterministic offline transport.

## AICAT-018 Harden federated catalog signatures, authorization, and input policy

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mcplusplus-federation
- Depends on: AICAT-017
- Goal id: AICAT-G060
- Outputs: ipfs_accelerate_py/mcplusplus_module/service_registry.py, ipfs_accelerate_py/model_catalog/security.py, test/test_ai_catalog_security.py
- Validation: python -m pytest test/test_ai_catalog_security.py test/test_mcplusplus_ai_catalog.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/mcplusplus/security
- Parallel lane: catalog-federation-security
- Resource class: security-review
- Token class: large
- Estimated tokens: 17000
- Predicted files: ipfs_accelerate_py/mcplusplus_module/service_registry.py, ipfs_accelerate_py/model_catalog/security.py, test/test_ai_catalog_security.py
- Allow concurrent with: AICAT-019
- Conflict policy: Serialize service registry edits after AICAT-017. Own federation policy and adversarial tests; preserve transport compatibility.
- Preconditions: Compact advertisements and peer catalog source behavior are complete.
- Effects: Canonically sign full advertisement records, validate trust and expiry, enforce separate read, refresh, and invoke capabilities, and add URL, record, page, media, and diagnostic bounds.
- Acceptance: Signatures cover issuer, service identity, catalog CID, revision, operations, interfaces, endpoint, issue time, expiry, and nonce. Verification checks trusted issuer, replay window, clock skew, expiry, CID, and schema version before records enter the catalog. UCAN or equivalent policy distinguishes catalog read, remote refresh, health probe, and each invocation operation. URLs enforce scheme, host, port, redirect, DNS, loopback, link-local, private-range, and allowlist policy. Unsigned, stale, replayed, oversized, recursive, malformed, secret-bearing, and SSRF-shaped inputs fail closed with bounded errors. Adversarial and property tests require no live network.

## AICAT-019 Add cache invalidation, selection receipts, and observability

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: catalog-observability
- Depends on: AICAT-009, AICAT-011, AICAT-017
- Goal id: AICAT-G070
- Outputs: ipfs_accelerate_py/model_catalog/cache.py, ipfs_accelerate_py/model_catalog/events.py, ipfs_accelerate_py/model_catalog/receipts.py, test/test_ai_catalog_observability.py
- Validation: python -m pytest test/test_ai_catalog_observability.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/observability
- Parallel lane: catalog-observability
- Resource class: cpu-small
- Token class: large
- Estimated tokens: 16000
- Predicted files: ipfs_accelerate_py/model_catalog/cache.py, ipfs_accelerate_py/model_catalog/events.py, ipfs_accelerate_py/model_catalog/receipts.py, test/test_ai_catalog_observability.py
- Allow concurrent with: AICAT-018
- Conflict policy: Own cache, event, receipt, and focused observability modules. Do not add response caching to modality invocations or modify central routers.
- Preconditions: ModelManager facade, typed backend lifecycle, and federated source revisions are available.
- Effects: Add snapshot caches, source-specific TTLs, event-driven invalidation, single-flight refresh, bounded selection receipts, provenance traces, health samples, and low-cardinality metrics.
- Acceptance: Static capabilities and dynamic health use separate TTLs. Unchanged content reuses snapshot CIDs. Registration, deployment lifecycle, credential-state, explicit refresh, and peer revision events invalidate only affected views. Concurrent refresh collapses per source and cancellation does not poison cache state. Selection receipts include candidates, policy filters, ranking inputs, selected binding, fallback boundaries, catalog revision, source provenance, and timestamps without prompts, media, output, credentials, or raw endpoints. Metrics cover source latency, hits, misses, stale records, conflicts, no-match reasons, resolutions, and health transitions with bounded labels. Deterministic clock and concurrency tests pass.

## AICAT-020 Prove conformance, preserve compatibility, and document rollout

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: catalog-rollout
- Depends on: AICAT-012, AICAT-014, AICAT-015, AICAT-016, AICAT-018, AICAT-019
- Goal id: AICAT-G080
- Outputs: test/test_ai_catalog_conformance.py, docs/architecture/AI_SERVICE_CATALOG.md, docs/LLM_ROUTER.md, docs/MCP_SERVER.md, docs/INDEX.md
- Validation: python -m pytest test/test_ai_service_catalog_schema.py test/test_ai_service_catalog_registry.py test/test_ai_service_catalog_sources.py test/test_model_manager_catalog.py test/test_ai_router_catalog_contract.py test/test_inference_backend_catalog.py test/test_ai_catalog_legacy_registry.py test/mcp_server/test_ai_catalog_tools.py test/mcp_server/test_ai_router_text_embedding_tools.py test/mcp_server/test_ai_router_vision_voice_tools.py test/test_mcplusplus_ai_catalog_idl.py test/test_mcplusplus_ai_catalog.py test/test_ai_catalog_security.py test/test_ai_catalog_observability.py test/test_ai_catalog_conformance.py -q
- Board namespace: ai-service-catalog-v1
- Bundle: ai-catalog/rollout
- Parallel lane: catalog-conformance-rollout
- Resource class: test-large
- Token class: large
- Estimated tokens: 18000
- Predicted files: test/test_ai_catalog_conformance.py, docs/architecture/AI_SERVICE_CATALOG.md, docs/LLM_ROUTER.md, docs/MCP_SERVER.md, docs/INDEX.md
- Allow concurrent with:
- Conflict policy: This task owns final generated conformance fixtures and architecture or operator documentation. Re-read all current docs and preserve unrelated provider guidance.
- Preconditions: Legacy projections, all modality MCP tools, MCP++ IDL and security, and observability are complete.
- Effects: Add generated cross-surface drift checks, backward-compatibility fixtures, cold-import side-effect tests, environment-gated live smokes, architecture documentation, migration guidance, rollout gates, and rollback instructions.
- Acceptance: Every declared router binding resolves or carries a typed non-routable reason; router, ModelManager, legacy, MCP, and MCP++ projections agree on canonical identities and revisions; default tests use fake providers and no network; imports and listing trigger no install, process, model load, credential read, or network request; compatibility fixtures cover public Python and MCP names; opt-in live smokes exercise one available provider for text, embeddings, multimodal, transcription, and synthesis without requiring every modality; documentation explains information versus invocation planes, schemas, operation taxonomy, lifecycle states, source precedence, extension boundaries, security, caching, metrics, migration, troubleshooting, rollout, rollback, and compatibility sunset policy.

## AICAT-021 Resolve validation retry-budget failure for AICAT-005

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: AICAT-001
- Outputs: ipfs_accelerate_py/multimodal_router.py, test/test_multimodal_router_catalog_discovery.py, /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/ai-service-catalog-v2/state/discovery
- Validation: python -m pytest test/test_multimodal_router_catalog_discovery.py test/test_multimodal_router.py -q
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in AICAT-005. Use evidence in /home/barberb/.local/share/ipfs_accelerate_py/agent-supervisor/ai-service-catalog-v2/state/discovery/2026-07-27-aicat-021-aicat-005-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release AICAT-005 from strategy blocked_tasks.
