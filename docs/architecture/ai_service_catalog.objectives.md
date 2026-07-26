# AI Service Catalog Objective Heap

This objective heap is the durable source of intent for making every supported
AI service discoverable through one typed catalog while preserving the
invocation ownership of the existing LLM, embeddings, multimodal, and voice
routers. The companion todo board is the executable projection consumed by the
agent supervisor.

The catalog is an information and resolution plane, not a fifth inference
router. A model being known must remain distinct from it being configured,
authorized, reachable, healthy, and routable. Model output and successful
process exit are proposal evidence rather than completion evidence; goals close
only when current-tree implementation and validation receipts cover their
acceptance criteria.

Program invariants:

- Routers retain ownership of invocation, batching, streaming, fallback, and
  provider-specific request semantics.
- ModelManager exposes the canonical catalog facade and does not duplicate
  provider invocation logic.
- Discovery, import, and listing are side-effect free and never install a
  provider, start a model, probe the network, or expose credentials.
- Stable service, model, deployment, and router-binding identities are derived
  from canonical non-secret fields and versioned schemas.
- MCP and MCP++ callers see the same catalog revision and invoke services
  through the canonical routers.
- Remote catalog data is signed, bounded, stale-aware, capability-gated, and
  treated as untrusted input.
- Existing Python, CLI, router, MCP, and ModelManager interfaces remain
  compatible through explicit projections and deprecation paths.

## AICAT-G000 Canonical AI service discovery and invocation

- Status: active
- Parent:
- Fib priority: 1
- Track: ai-service-catalog
- Priority: P0
- Bundle: ai-catalog/root
- Goal: Make all local and federated AI services queryable through a versioned ModelManager catalog and invokable through llm_router, embeddings_router, multimodal_router, and voice_router from Python, CLI, MCP, and MCP++ surfaces.
- Evidence: AICAT-G010, AICAT-G020, AICAT-G030, AICAT-G040, AICAT-G050, AICAT-G060, AICAT-G070, AICAT-G080
- Outputs: ipfs_accelerate_py/model_catalog, ipfs_accelerate_py/model_manager.py, ipfs_accelerate_py/inference_backend_manager.py, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/embeddings_router.py, ipfs_accelerate_py/multimodal_router.py, ipfs_accelerate_py/voice_router.py, ipfs_accelerate_py/mcp_server, ipfs_accelerate_py/mcplusplus_module
- Validation: python -m pytest test/test_ai_service_catalog_schema.py test/test_ai_service_catalog_registry.py test/test_ai_service_catalog_sources.py test/test_model_manager_catalog.py test/test_ai_router_catalog_contract.py test/test_inference_backend_catalog.py test/mcp_server/test_ai_catalog_tools.py test/mcp_server/test_ai_router_tools.py test/test_mcplusplus_ai_catalog.py test/test_ai_catalog_security.py test/test_ai_catalog_conformance.py -q
- Acceptance: One canonical versioned catalog represents provider, model, deployment, router binding, operation, capability, lifecycle, health, provenance, and policy metadata without secrets; all four routers publish discovery contracts and remain invocation owners; ModelManager provides list, get, resolve, health, and explicit refresh operations; duplicate registries become compatibility projections; MCP and MCP++ expose typed query and invocation operations; federated records are signed, bounded, authorized, and stale-aware; selection receipts explain routing decisions; compatibility and cross-surface conformance tests pass.
- Gap task: Implement the highest-priority incomplete child goal with focused deterministic tests and without moving provider invocation into the catalog.
- Refinement: Keep contracts, modality discovery, source aggregation, ModelManager integration, legacy convergence, MCP query, typed invocation, federation, security, observability, and rollout in dependency-aware file-ownership lanes.
- Embedding query: model manager AI service catalog provider registry model deployment router capability MCP MCP++ federation llm embeddings multimodal voice
- AST query: ModelManager InferenceBackendManager LLMProvider EmbeddingProvider MultimodalProvider VoiceProvider ServiceRegistry IDLRegistry

## AICAT-G010 Versioned catalog contracts and stable identities

- Status: active
- Parent: AICAT-G000
- Fib priority: 2
- Track: catalog-contracts
- Priority: P0
- Bundle: ai-catalog/contracts
- Goal: Define immutable bounded schemas for providers, models, deployments, router bindings, catalog snapshots, capabilities, lifecycle states, provenance, and operation names with deterministic identities and content addressing.
- Evidence: ipfs_accelerate_py/model_catalog/schema.py, ipfs_accelerate_py/model_catalog/identity.py, ipfs_accelerate_py/model_catalog/registry.py, ipfs_accelerate_py/model_catalog/resolver.py, ipfs_accelerate_py/model_catalog/snapshot.py, test/test_ai_service_catalog_schema.py, test/test_ai_service_catalog_registry.py
- Outputs: ipfs_accelerate_py/model_catalog/__init__.py, ipfs_accelerate_py/model_catalog/schema.py, ipfs_accelerate_py/model_catalog/identity.py, ipfs_accelerate_py/model_catalog/registry.py, ipfs_accelerate_py/model_catalog/resolver.py, ipfs_accelerate_py/model_catalog/snapshot.py
- Validation: python -m pytest test/test_ai_service_catalog_schema.py test/test_ai_service_catalog_registry.py -q
- Acceptance: The operation taxonomy includes text generation and chat, embeddings, vision, transcription, synthesis, batch, stream, and tool calls; known, configured, authorized, reachable, healthy, and routable states cannot be conflated; canonical serialization and CIDs are deterministic; aliases and collisions resolve predictably; bounds and redaction reject malformed or secret-bearing records; snapshot revisions change only when canonical content changes; imports and enumeration are side-effect free.
- Gap task: Add or repair the smallest missing schema, identity, resolver, or snapshot guarantee and prove it with deterministic tests.
- Refinement: Stabilize immutable records and identity rules before connecting any router or persistent source.
- Embedding query: typed model catalog schema provider descriptor deployment router binding operation lifecycle content identifier redaction
- AST query: ProviderInfo ModelMetadata APIModelInfo ModelEndpoint ProviderDescriptor CatalogSnapshot

## AICAT-G020 Router-owned provider and model discovery

- Status: active
- Parent: AICAT-G000
- Fib priority: 3
- Track: router-discovery
- Priority: P0
- Bundle: ai-catalog/routers
- Goal: Give llm_router, embeddings_router, multimodal_router, and voice_router one compatible, side-effect-free discovery surface while preserving each router's invocation behavior and provider-specific capabilities.
- Evidence: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/embeddings_router.py, ipfs_accelerate_py/multimodal_router.py, ipfs_accelerate_py/voice_router.py, test/test_ai_router_catalog_contract.py
- Outputs: ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/embeddings_router.py, ipfs_accelerate_py/multimodal_router.py, ipfs_accelerate_py/voice_router.py
- Validation: python -m pytest test/test_ai_router_catalog_contract.py test/test_llm_router_integration.py test/test_embeddings_router_contract.py -q
- Acceptance: Every router provides list_providers, get_provider_descriptor, list_models, and resolve_model semantics using shared records; discovery does not instantiate heavy providers, install software, load model weights, or make network calls; operation and streaming, batching, input, output, context, device, and authorization capabilities are accurate; aliases and model overrides are deterministic; existing generation methods and public provider names remain compatible.
- Gap task: Implement the next missing modality discovery adapter and parity tests without changing provider invocation policy.
- Refinement: Use one independent file-ownership lane per router, followed by a cross-router contract test.
- Embedding query: llm embeddings multimodal voice router provider model discovery capabilities aliases context device
- AST query: list_providers get_provider_descriptor list_models resolve_model generate_text generate_embeddings generate_multimodal transcribe synthesize

## AICAT-G030 Catalog aggregation and ModelManager facade

- Status: active
- Parent: AICAT-G000
- Fib priority: 5
- Track: catalog-manager
- Priority: P0
- Bundle: ai-catalog/manager
- Goal: Aggregate router, persistent metadata, static configuration, and live deployment sources into a deterministic catalog exposed through ModelManager without duplicating invocation logic.
- Evidence: ipfs_accelerate_py/model_catalog/sources, ipfs_accelerate_py/model_catalog/catalog.py, ipfs_accelerate_py/model_manager.py, test/test_ai_service_catalog_sources.py, test/test_model_manager_catalog.py
- Outputs: ipfs_accelerate_py/model_catalog/sources, ipfs_accelerate_py/model_catalog/catalog.py, ipfs_accelerate_py/model_manager.py
- Validation: python -m pytest test/test_ai_service_catalog_sources.py test/test_model_manager_catalog.py -q
- Acceptance: Source adapters preserve provenance and precedence; local configured state does not overwrite static capability truth; duplicate records merge by canonical identity; conflicting claims remain inspectable; ModelManager supports list_services, list_models, get, resolve, health, snapshot, and explicit refresh with bounded filters and pagination; refresh side effects require explicit policy; existing ModelManager CRUD and endpoint APIs remain compatible.
- Gap task: Close the next source aggregation or ModelManager facade gap with source-precedence and compatibility tests.
- Refinement: Build source adapters against shared schemas, aggregate them, then expose the catalog through ModelManager.
- Embedding query: ModelManager catalog aggregation source precedence provider model endpoint deployment snapshot refresh
- AST query: ModelManager ModelMetadata add_model list_models get_model discover_models load_registry

## AICAT-G040 Deployment truth and legacy registry convergence

- Status: active
- Parent: AICAT-G000
- Fib priority: 8
- Track: backend-convergence
- Priority: P0
- Bundle: ai-catalog/backends
- Goal: Represent currently served endpoints and inference backends as typed deployments and convert tuple and static model registries into compatibility projections over the catalog.
- Evidence: ipfs_accelerate_py/inference_backend_manager.py, ipfs_accelerate_py/api_integrations/model_registry.py, ipfs_accelerate_py/api_backends/api_models_registry.py, ipfs_accelerate_py/model_catalog/sources/deployments.py, test/test_inference_backend_catalog.py, test/test_ai_catalog_legacy_registry.py
- Outputs: ipfs_accelerate_py/inference_backend_manager.py, ipfs_accelerate_py/api_integrations/model_registry.py, ipfs_accelerate_py/api_backends/api_models_registry.py, ipfs_accelerate_py/model_catalog/sources/deployments.py
- Validation: python -m pytest test/test_inference_backend_catalog.py test/test_ai_catalog_legacy_registry.py -q
- Acceptance: Backend registrations use typed descriptors instead of unversioned tuples; served OpenAI-compatible endpoints appear as deployment records with liveness and readiness separated; static API inventories are seeded once and projected for old callers; compatibility APIs retain names and return shapes; no discovery path performs a model request; drift tests fail when a legacy projection and canonical record disagree.
- Gap task: Migrate the next duplicate backend or static registry path while retaining compatibility and explicit provenance.
- Refinement: Add deployment source adapters before modifying backend registration, then migrate duplicate registries behind projections.
- Embedding query: inference backend manager served endpoint deployment registry static API models compatibility migration drift
- AST query: InferenceBackendManager register_backend APIModelRegistry APIModelsRegistry model_endpoints start_model

## AICAT-G050 MCP catalog queries and typed router invocation

- Status: active
- Parent: AICAT-G000
- Fib priority: 13
- Track: mcp-ai-services
- Priority: P0
- Bundle: ai-catalog/mcp
- Goal: Let MCP clients query the ModelManager catalog and invoke all four modality routers through explicit typed tools with bounded schemas and consistent errors.
- Evidence: ipfs_accelerate_py/mcp_server/tools/model_tools/native_model_tools.py, ipfs_accelerate_py/mcp_server/tools/ai_router_tools, test/mcp_server/test_ai_catalog_tools.py, test/mcp_server/test_ai_router_tools.py
- Outputs: ipfs_accelerate_py/mcp_server/tools/model_tools/native_model_tools.py, ipfs_accelerate_py/mcp_server/tools/ai_router_tools
- Validation: python -m pytest test/mcp_server/test_ai_catalog_tools.py test/mcp_server/test_ai_router_tools.py -q
- Acceptance: MCP exposes catalog list services, list models, get, resolve, health, and privileged refresh; text and embedding tools call llm_router and embeddings_router rather than alternate implementations; multimodal generation, transcription, and synthesis have typed tools backed by their canonical routers; request and response sizes, media references, pagination, timeouts, streaming, and errors are bounded; tool output includes catalog revision and selected binding without credentials; old model tools remain compatible.
- Gap task: Add the next missing catalog query or modality invocation tool and prove Python-to-MCP parity.
- Refinement: Land read-only catalog queries first, then text and embeddings, then multimodal and voice operations.
- Embedding query: MCP model manager catalog query llm embeddings multimodal voice typed tool router invocation
- AST query: register_native_model_tools generate_text generate_embeddings multimodal voice mcp tool

## AICAT-G060 MCP++ interface, federation, and authorization

- Status: active
- Parent: AICAT-G000
- Fib priority: 21
- Track: mcplusplus-federation
- Priority: P0
- Bundle: ai-catalog/mcplusplus
- Goal: Publish a versioned ai.catalog.v1 MCP++ interface, advertise compact catalog revisions, federate bounded remote catalogs, and enforce signed provenance and capability authorization.
- Evidence: ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py, ipfs_accelerate_py/mcplusplus_module/service_registry.py, ipfs_accelerate_py/mcplusplus_module/trio/server.py, test/test_mcplusplus_ai_catalog.py, test/test_ai_catalog_security.py
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py, ipfs_accelerate_py/mcplusplus_module/service_registry.py, ipfs_accelerate_py/mcplusplus_module/trio/server.py, ipfs_accelerate_py/model_catalog/sources/peers.py
- Validation: python -m pytest test/test_mcplusplus_ai_catalog.py test/test_ai_catalog_security.py -q
- Acceptance: ai.catalog.v1 defines read, resolve, health, refresh, and modality invocation operations; service advertisements carry catalog CID, revision, operation summary, interface CIDs, expiry, and issuer rather than an unbounded model list; signatures cover the full canonical record; UCAN or equivalent policy distinguishes read, refresh, and invoke authority; remote URLs, records, media, pages, and TTLs are bounded; SSRF, replay, stale, unsigned, malformed, and oversized inputs fail closed; peer records cannot override trusted local identities.
- Gap task: Close the next IDL, advertisement, federation, signature, authorization, or untrusted-input boundary with adversarial tests.
- Refinement: Define the IDL before changing advertisements, then add federation before tightening authorization and input policy.
- Embedding query: MCP++ ai catalog IDL service advertisement federation signature UCAN capability SSRF TTL replay
- AST query: IDLRegistry ServiceRegistry advertise_service discover_services TrioMCPServer UCAN

## AICAT-G070 Cache, provenance, routing receipts, and observability

- Status: active
- Parent: AICAT-G000
- Fib priority: 34
- Track: catalog-observability
- Priority: P1
- Bundle: ai-catalog/observability
- Goal: Make catalog snapshots efficient and routing decisions explainable through bounded caches, invalidation events, provenance, health samples, metrics, and deterministic selection receipts.
- Evidence: ipfs_accelerate_py/model_catalog/cache.py, ipfs_accelerate_py/model_catalog/events.py, ipfs_accelerate_py/model_catalog/receipts.py, test/test_ai_catalog_observability.py
- Outputs: ipfs_accelerate_py/model_catalog/cache.py, ipfs_accelerate_py/model_catalog/events.py, ipfs_accelerate_py/model_catalog/receipts.py
- Validation: python -m pytest test/test_ai_catalog_observability.py -q
- Acceptance: Static descriptors and dynamic health have separate TTL and invalidation rules; unchanged snapshots reuse content identities; refresh is single-flight and bounded; provider registration, deployment lifecycle, credential-state, and peer updates invalidate only affected projections; selection receipts record candidates, policy filters, ranking inputs, chosen binding, fallback boundaries, catalog revision, and timestamps without prompts or secrets; metrics expose source latency, cache hit ratio, stale records, resolution outcomes, and health transitions with bounded cardinality.
- Gap task: Implement the next cache, invalidation, provenance, receipt, or metric guarantee with deterministic clock and concurrency tests.
- Refinement: Keep cache policy independent of provider invocation and use receipts as the shared debugging contract across Python, MCP, and MCP++.
- Embedding query: catalog cache invalidation event provenance routing selection receipt health metrics single flight
- AST query: cache refresh invalidate resolve selection trace health metrics

## AICAT-G080 Conformance, compatibility, documentation, and rollout

- Status: active
- Parent: AICAT-G000
- Fib priority: 55
- Track: catalog-rollout
- Priority: P1
- Bundle: ai-catalog/rollout
- Goal: Prove that registry, router, ModelManager, MCP, MCP++, and legacy views agree, preserve compatibility, document extension boundaries, and provide an opt-in live smoke and rollback path.
- Evidence: test/test_ai_catalog_conformance.py, docs/architecture/AI_SERVICE_CATALOG.md, docs/LLM_ROUTER.md, docs/MCP_SERVER.md
- Outputs: test/test_ai_catalog_conformance.py, docs/architecture/AI_SERVICE_CATALOG.md, docs/LLM_ROUTER.md, docs/MCP_SERVER.md, docs/INDEX.md
- Validation: python -m pytest test/test_ai_service_catalog_schema.py test/test_ai_service_catalog_registry.py test/test_ai_service_catalog_sources.py test/test_model_manager_catalog.py test/test_ai_router_catalog_contract.py test/test_inference_backend_catalog.py test/test_ai_catalog_legacy_registry.py test/mcp_server/test_ai_catalog_tools.py test/mcp_server/test_ai_router_tools.py test/test_mcplusplus_ai_catalog.py test/test_ai_catalog_security.py test/test_ai_catalog_observability.py test/test_ai_catalog_conformance.py -q
- Acceptance: Generated conformance fixtures prove every declared router binding is resolvable and every compatibility projection matches canonical fields; cold imports and listing have no network, install, process, model-load, or credential side effects; local and federated Python, CLI, MCP, and MCP++ views agree on identity and revision; default tests are offline and deterministic; environment-gated live smokes exercise one service per available modality; documentation explains architecture, operation taxonomy, lifecycle states, extension points, security policy, migration, troubleshooting, metrics, rollout, and rollback.
- Gap task: Add the missing conformance proof, compatibility fixture, documentation, live smoke, or rollback safeguard.
- Refinement: Close with a generated drift matrix and documentation derived from the stable interfaces rather than implementation history.
- Embedding query: AI catalog conformance compatibility documentation migration live smoke rollout rollback router MCP
- AST query: tests provider registry model manager mcp server mcplusplus docs
