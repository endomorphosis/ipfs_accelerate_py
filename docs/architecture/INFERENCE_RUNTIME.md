# Inference runtime and router lifecycle

**Status:** Current  
**Audience:** Developers and agents tracing a single inference request from public entrypoint through catalog/router, backend/worker, and result  
**Scope:** Inference/data-plane request lifecycle: discovery vs invocation, modality routers, provider selection, execution adapters, caching, result and error propagation, and graceful degradation when optional providers or hardware are absent  
**Non-goals:** Agent-supervisor scheduling and objective heaps; MCP transport and tool-policy details (see MCP guides); catalog schema and resolution math in depth (see [AI Service Catalog](AI_SERVICE_CATALOG.md)); endpoint-usage admission product surface (see planned MODEL_SERVICE_ROUTING); package install profiles; P2P mesh topology beyond the task-queue handoff  
**Last verified:** `f279353053fe41593d76a95245416933d08e8999` (2026-08-03); public entrypoints, router modules, backend manager, worker package, and optional unified service checked against the live tree  

This guide is the maintained **InferenceRequestFlow@1** and **RouterFallbackFlow@1** narrative. A developer should follow one request across public entrypoint → catalog/router → backend/worker → result **without assuming optional providers, GPUs, CLIs, or network backends are present**.

---

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Package boundary | `ipfs_accelerate_py/__init__.py` | Lazy optional re-exports; `get_instance()`, `generate_text` |
| Process singleton | `get_instance()` | Process-wide `ipfs_accelerate_py` coordinator |
| Inference coordinator | `ipfs_accelerate_py/ipfs_accelerate.py` (`ipfs_accelerate_py`) | Endpoints, queues, `run_model`, `infer`, `get_capabilities` |
| LLM invocation | `ipfs_accelerate_py/llm_router.py` | `generate_text`, `chat_completions_create`, `get_llm_provider` |
| Embeddings invocation | `ipfs_accelerate_py/embeddings_router.py` | `embed_text`, `embed_texts` |
| Multimodal invocation | `ipfs_accelerate_py/multimodal_router.py` | `generate_multimodal`, `generate_multimodal_text` |
| Voice invocation | `ipfs_accelerate_py/voice_router.py` | `synthesize`, `transcribe` (provider protocol) |
| Router DI / cache hooks | `ipfs_accelerate_py/router_deps.py` | Injected deps and process caches |
| Information plane | `ipfs_accelerate_py/model_catalog/` | Descriptors only; no provider HTTP |
| Catalog facade | `model_manager` (`get_default_model_manager`) | List/resolve; does not invoke |
| Backend registry | `ipfs_accelerate_py/inference_backend_manager.py` | `InferenceBackendManager`, `select_backend_for_task` |
| Optional orchestration | `ipfs_accelerate_py/unified_inference_service.py` | HTTP/WS/P2P stack when installed |
| Local transformers | `ipfs_accelerate_py/transformers_integration.py` | `TransformersModelProvider` |
| API adapters | `ipfs_accelerate_py/api_backends/` | OpenAI, HF TGI/TEI, Ollama, … + queue/circuit breaker |
| Hardware worker | `ipfs_accelerate_py/worker/` | `worker_py`, skillset and backend utils |
| Optional model HTTP | `ipfs_accelerate_py/hf_model_server/` | FastAPI/WebSocket surface when enabled |
| CLI | `ipfs_accelerate_py/cli_entry.py` → `cli` | Console script; does not invent backends |
| MCP (canonical) | `ipfs_accelerate_py/mcp_server/` | Tools call routers; transport out of scope here |
| MCP (compat facade) | `ipfs_accelerate_py/mcp/` | Compatibility only |
| Task queue handoff | `llm_router.submit_task` / `p2p_tasks/` | Optional async/out-of-process path |

Related maintained pages (read, do not treat as this guide’s output):

- [Architecture overview](overview.md) — layers and capability-oriented stance  
- [AI Service Catalog](AI_SERVICE_CATALOG.md) — discovery vs invocation boundary  
- [Guide conventions](GUIDE_CONVENTIONS.md) — writing contract  
- [Unified inference backend (optional)](../INFERENCE_BACKEND_README.md) — backend manager and service ops  
- [MCP server README](../../ipfs_accelerate_py/mcp_server/README.md) — tools and transports  

---

## 1. Context and component map

### 1.1 Planes

Inference lives on the **inference/data plane**. The **supervisor/control plane** (agent supervisor, objective heap, leases) is not on the hot path and must not be required to generate text or embeddings.

```text
Client / app / CLI / MCP tool
        |
   public package API (entrypoint)
        |
   +---- catalog / ModelManager     (information only)
   |              |
   |         resolve binding IDs
   |              |
   +---- modality router            (invocation plane)
                |
        provider construction + selection
                |
        execution adapter (API | local HF | CLI | worker | optional backend)
                |
        result / typed error / optional cache write
```

### 1.2 Why routers stay separate from execution adapters

| Layer | Owns | Must not own |
| --- | --- | --- |
| **Catalog / ModelManager** | Immutable descriptors, resolve ranking, lifecycle and operational facts | HTTP to providers, loading weights, reading secret values |
| **Modality routers** (`llm_router`, `embeddings_router`, `multimodal_router`, `voice_router`) | Provider construction, timeouts, batching, streaming where supported, fallback policy, response caches, generation traces | Durable catalog identity, transport policy for MCP |
| **Execution adapters** (`api_backends/*`, transformers provider, CLI wrappers, `worker/*`, optional GPU backends) | Protocol and hardware I/O, circuit breakers, queues at the wire | Cross-modality policy or catalog schema |
| **Backend manager / unified service** (optional) | Multi-backend registration, health, load-balance among *registered* backends | Being the only way to call `generate_text` |
| **Coordinator** (`ipfs_accelerate_py`) | Endpoint registration, local queues, capability summary, `run_model` / `infer` | Replacing the modality routers for chat/embed APIs |

Collapsing catalog into routers would make discovery side-effecting. Collapsing routers into adapters would duplicate fallback and cache policy per wire protocol. Keeping them separate is a **security and lifecycle boundary**: list/resolve never installs software, starts processes, or spends credentials; only explicit invoke does.

### 1.3 Baseline vs optional

| Always treat as baseline | Optional (probe / configure / extra) |
| --- | --- |
| Importing package modules that load without heavy deps | Remote API keys and paid providers |
| Router entry functions that raise clearly when no provider exists | CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU, Qualcomm |
| Capability and status helpers that report absence | `UnifiedInferenceService`, HF model server, libp2p |
| CPU-oriented local paths when `transformers` (or similar) is installed | CLI binaries (codex, copilot, goose, grok, …) |
| Failures that surface as exceptions or structured error dicts | P2P task mesh and remote peers |

**Import success is never a capability signal.** Prefer `get_instance().get_capabilities(detail=True)`, router `list_providers()`, and explicit provider selection over assuming hardware or network backends.

---

## 2. Public entrypoints

Multiple surfaces reach the same runtime. Pick one for the walkthrough below; the others call into the same routers or coordinator.

| Surface | How a developer enters | What it eventually uses |
| --- | --- | --- |
| **Python API (preferred for text)** | `from ipfs_accelerate_py import generate_text` or `from ipfs_accelerate_py.llm_router import generate_text` | `llm_router` → provider |
| **Chat completions shape** | `chat_completions_create(...)` | Native chat if provider supports it; else `generate_text` |
| **Embeddings** | `embed_text` / `embed_texts` from package or `embeddings_router` | Embeddings providers |
| **Multimodal / voice** | `generate_multimodal*`, voice synthesize/transcribe | Respective routers |
| **Coordinator singleton** | `get_instance()` then `run_model` / `infer` / endpoint APIs | Local transformers, endpoint handlers, `api_backends` |
| **CLI** | `ipfs-accelerate` via `cli_entry.main` | Same package; capability and command paths |
| **MCP tools** | `mcp_server` tools such as `generate_text`, `inference_run_inference` | Routers / coordinator; policy in MCP docs |
| **Optional unified service** | `start_unified_service()` | Backend manager + optional HF server + P2P |

Structural example (no network required for the *shape* of the call):

```python
from ipfs_accelerate_py import generate_text, get_instance

# Invocation plane: text generation with automatic provider discovery.
# Without credentials or local transformers, this raises rather than inventing success.
try:
    text = generate_text("Say hello in one short sentence.")
except Exception as exc:
    print(type(exc).__name__, exc)

# Capability plane: what this process currently exposes (JSON-safe summary).
caps = get_instance().get_capabilities(detail=True)
print(caps.get("task_types"), caps.get("hwtest"))
```

Optional catalog resolution (does **not** run inference):

```python
from ipfs_accelerate_py.model_manager import get_default_model_manager

manager = get_default_model_manager()
result = manager.resolve(operation="text.generate", routable=True)
if result.found:
    print(result.candidates[0].binding.binding_id)
else:
    print(result.reasons)
```

---

## 3. Request lifecycle (one request)

The primary journey for this guide is **LLM text generation** via `generate_text`. Parallel modality routers follow the same control shape with different payload types.

### 3.1 Sequence (happy path and absence-safe path)

```text
Caller
  |  generate_text(prompt, provider=None, model_name=None, ...)
  v
llm_router.generate_text
  |-- clear generation trace / usage admission (thread-local diagnostics)
  |-- if usage_coordinator + enforce policy: usage admission path (optional product)
  |-- if response cache enabled and non-side-effecting:
  |      cache hit? --> return cached str (no remote charge)
  v
get_llm_provider(provider)     # discovery + construction
  |-- preferred name or env IPFS_ACCELERATE_PY_LLM_PROVIDER / aliases
  |-- else first available optional builtin (API/CLI) that constructs cleanly
  |-- else local HuggingFace transformers provider if installed
  |-- else raise RuntimeError / LLMRouterError  (fail closed: no silent mock success)
  v
provider.generate(prompt, model_name=..., **kwargs)
  |-- within-provider model fallbacks (e.g. HF model list) may apply
  |-- side-effecting/agent kwargs disable model retry and cross-provider fallback
  v
str result
  |-- set get_last_generation_trace() (effective provider/model)
  |-- optional response cache write
  |-- return to caller
```

Cross-provider **fallback** (RouterFallbackFlow@1) only runs when the primary provider fails **and** the request is not side-effecting **and** the provider is unpinned or is an optional pin that allows fallback. Explicit `provider=` is a pin by default (no silent switch). Local HF remains a last-resort path when `allow_local_fallback=True` (default) and no hard pin forbids it.

### 3.2 Ownership table

| Stage | Owner | Input | Output / side effects |
| --- | --- | --- | --- |
| Public call | App / CLI / MCP | Prompt, optional provider/model | Error if router missing |
| Cache lookup | `RouterDeps` + router | Stable cache key from prompt/kwargs | Cached string or miss |
| Provider resolve | `get_llm_provider` | Name / env / auto order | `LLMProvider` or hard error |
| Catalog (optional parallel) | `AIServiceCatalog` / ModelManager | Operation constraints | Binding IDs; **no I/O to model** |
| Backend select (optional stack) | `InferenceBackendManager` | Task, model, health filters | `BackendInfo` or `None` |
| Execute | Provider / adapter / worker | Prompt or tensors | Text, embeddings, or error |
| Result | Router | Provider return value | `str` / OpenAI-compat object / vectors |
| Diagnostics | Thread-local traces | Effective names, admission codes | `get_last_generation_trace()` etc. |

### 3.3 Sync / async boundary

| Path | Style | Notes |
| --- | --- | --- |
| `generate_text`, `embed_*`, most provider factories | **Synchronous** | Safe default for library callers; may block on network/CLI |
| `ipfs_accelerate_py.infer`, endpoint consumers, queues | **Async** (`async def`) | Endpoint selection and batch consumers |
| `submit_task` / `wait_task` | Sync API wrapping optional anyio/trio for P2P | Local queue or remote peer; not required for single-process use |
| `UnifiedInferenceService.start`, HF server, WebSocket, libp2p | **Async service** | Optional multi-protocol process |
| `BaseAPIBackend` queue processor | Background **thread** | Per-adapter concurrency and circuit breaker |

Do not assume every entrypoint is async. Prefer the sync router API for application glue unless you already run an event loop and the endpoint/queue APIs.

### 3.4 Parallel modality paths

```text
text.generate / text.chat     --> llm_router
embedding.generate            --> embeddings_router
vision / multimodal generate  --> multimodal_router
audio.transcribe / synthesize --> voice_router
local tensors / endpoints     --> ipfs_accelerate_py.run_model / infer
registered multi-backend task --> InferenceBackendManager.select_backend_for_task
```

Catalog operation names (`text.generate`, …) map to router capabilities; backend manager task labels (`text-generation`, `chat`, `embedding`, …) are translated at the registration boundary into the same catalog operations when publishing deployment sources.

---

## 4. Catalog vs router (information vs invocation)

From [AI Service Catalog](AI_SERVICE_CATALOG.md):

- **Catalog** answers: what is known, which operations are declared, which binding satisfies constraints.
- **Routers** answer: construct a client, send a request, apply fallback, return bytes/text.

```text
static / router / deployment sources
              |
       AIServiceCatalog  --> snapshot revision
              |
         ModelManager.resolve(...)   # no provider HTTP
              |
         (caller chooses to invoke)
              |
         llm_router / embeddings_router / ...
```

Rules developers must keep:

1. `list` / `get` / `resolve` must not install packages, start servers, load weights, or spend API keys.  
2. Credential-shaped env may be reported as **present/absent**, never returned as values.  
3. A resolved binding ID is **not** a guarantee the provider will succeed at invoke time.  
4. Routers may publish catalog descriptors (`list_providers`, `list_models`, `catalog_snapshot`) as **projections** of what they can construct; those projections still do not invoke.

---

## 5. Provider selection and execution adapters

### 5.1 LLM provider resolution order (conceptual)

When `provider` is omitted and no force-env is set, resolution roughly prefers:

1. Accelerate/coordinator hook when enabled and available  
2. First constructible optional API/CLI provider in a fixed discovery order (OpenRouter, OpenAI, HF Inference API, xAI, Meta AI, various CLIs, …) — **only if** credentials/binaries make construction succeed  
3. Local HuggingFace transformers when the stack is importable  
4. Otherwise **raise** (no fake success)

Environment force (examples; see module docstring for full list):

- `IPFS_ACCELERATE_PY_LLM_PROVIDER` / `ipfs_accelerate_py_LLM_PROVIDER` — pin provider name  
- Per-provider keys and model defaults (OpenRouter, xAI, HF, CLI commands, …)  
- Opt-in discovery flags (e.g. Goose CLI discovery) so rare/heavy tools do not install themselves during automatic selection  

### 5.2 Coordinator path (`run_model` / `infer`)

`ipfs_accelerate_py.run_model` loads via `TransformersModelProvider` and runs tensor inference, returning a **dict** with `success` / `error` rather than always raising. `infer` selects an endpoint handler (local, API, libp2p, …) and returns handler results or a `ValueError` payload when no endpoint exists.

Use this path when integrating hardware endpoints and batch queues. Use `generate_text` when you want a single string and provider fallback policy.

### 5.3 API backends and worker

- **`api_backends/`** — protocol adapters (OpenAI-compatible, HF TGI/TEI, Ollama, Groq, …). Shared `BaseAPIBackend` supplies priority queue and **circuit breaker** (CLOSED / OPEN / HALF_OPEN) so failing remotes degrade without tight spin loops.  
- **`worker/`** — hardware-oriented worker (`worker_py`), skillsets, and utils (CUDA, OpenVINO, Apple, Qualcomm, llama.cpp, …). Presence of a skillset module does not prove the host can run it; probe at runtime.  
- **`InferenceBackendManager`** — registers backends with capabilities and health; `select_backend_for_task` filters to **healthy** backends matching task/model/protocol. Returns `None` when nothing matches (caller must handle).  
- **`UnifiedInferenceService`** — optional process that wires backend manager, HF server, WebSocket, and libp2p; dependencies are try/imported and disabled when missing.

### 5.4 Caching

| Cache | What it stores | What it must not store as authority |
| --- | --- | --- |
| Router response cache | Generation/embed results keyed by stable digests | Credentials, raw endpoints, unredacted secrets |
| Provider instance cache | Constructed provider objects in-process | Capability proof across processes |
| Catalog cache | Metadata snapshots only | Prompts, model outputs, or “invoke succeeded” |
| Chat history helpers (LLM) | Session text by CID-like keys when enabled | Cross-tenant secrets |

Side-effecting / agent requests (`agent=True` / `side_effecting=True` style kwargs) **bypass** response caches and automatic cross-provider fallback so tool-using runs do not replay or switch mid-flight.

---

## 6. Failure semantics, fallback, and error taxonomy

### 6.1 Fail-closed conditions

| Condition | Behavior |
| --- | --- |
| No provider can be constructed | Raise `RuntimeError` / `LLMRouterError` (or modality equivalent); do not return empty success text as if a model ran |
| Explicit unknown provider name | Raise `ValueError` / router error |
| Forced provider unavailable | Raise with actionable install/auth message |
| Side effects already started | No model retry and no cross-provider fallback; re-raise |
| Explicit provider pin | No automatic cross-provider fallback (unless optional-pin rules apply) |
| Backend manager: no healthy backend | `select_backend_for_task` → `None`; log warning |
| Catalog resolve: no candidates | Typed “not found” reasons; still not an invoke |
| Protected paths / supervisor leases | Out of scope here; control plane fails closed independently |

### 6.2 Degradation (optional missing)

| Missing piece | Degradation |
| --- | --- |
| No API keys | Skip those providers in auto order |
| No CLI binary | Skip CLI provider; explicit selection may offer install only when coded and allowed |
| No `transformers` | Local HF path unavailable; remote providers may still work |
| No CUDA / GPU libs | Hardware paths report false/unavailable; CPU or remote remains |
| No libp2p / P2P config | `submit_task` stays local queue or errors; mesh not required |
| Unified service deps missing | Components log warning and stay `None`; package API still usable |
| Cache backend failure | Miss-through; generation proceeds without cache |

### 6.3 Fallback taxonomy (RouterFallbackFlow@1)

```text
Level 0  Cache hit
Level 1  Primary provider + requested model
Level 2  Same provider, default/alternate model (within-provider)
Level 3  Cross-provider optional chain (unpinned / allowed optional pins only)
Level 4  Local HF fallback when allow_local_fallback and available
Level F  Raise original or last error (fail closed)
```

**Never** treat fallback as proof that the preferred provider is healthy. Use `get_last_generation_trace()` for the **effective** provider and model after the call.

### 6.4 Error taxonomy (developer-facing)

| Kind | Typical types / signals | Caller action |
| --- | --- | --- |
| Configuration | Missing key, unknown provider name | Set env or pass `provider=` |
| Capacity / quota | `UsageCapacityError`, CLI quota classification | Back off, switch pin, or raise budget |
| Compatibility | HF model incompatibility → model fallback list | Pin a working model or accept fallback list |
| Transport / backend | HTTP/CLI failures, circuit OPEN | Retry later or select another backend |
| Local runtime | Transformers/device errors | Change device, model, or use remote provider |
| Structural | `success: False` dicts from `run_model` | Inspect `error` field |
| Cancellation | usage cancel event / task abortion hooks | Stop work; do not mark success |

### 6.5 Recovery boundaries

- **Automatic:** within-provider model retries; unpinned cross-provider tries; circuit breaker half-open after timeout.  
- **Operator:** install extras, set credentials, start HF server or llama.cpp, enable discovery flags.  
- **Not recovery:** inventing a mock provider as production success; treating catalog `known` as `healthy`; treating board/todo status as inference health.

---

## 7. State and identity

| Artifact | Identity / scope | Authority |
| --- | --- | --- |
| Catalog snapshot revision | Content ID of canonical records | Information plane only |
| Router binding ID | Router + provider + model + deployment inputs | Selection receipt input |
| Generation trace | Thread-local last call | Diagnostic; not admission proof |
| Usage admission payload | Thread-local operational codes | Optional; never contains prompts/secrets |
| Endpoint status maps | Process-local on coordinator | Liveness hints for queues |
| Backend metrics | Process-local on backend manager | Ops; not SLAs |
| Response cache keys | Digest of provider/model/prompt/kwargs | Correctness under same inputs only |

---

## 8. Design rationale

**Rationale.** The runtime separates **what is known** (catalog), **how to invoke** (routers), and **how to talk to a wire or device** (adapters) so that:

1. Discovery stays side-effect free and federatable.  
2. Fallback, caching, and usage admission can be modality-consistent without rewriting every HTTP client.  
3. Optional hardware and commercial providers can be absent without breaking the package import or the mental model of the hot path.  
4. Agent/side-effecting work can opt out of automatic fallback and caches to preserve safety.

**Alternatives** that were rejected or would break invariants:

| Alternative | Breakage |
| --- | --- |
| Single “do everything” facade that resolves and invokes in one call with side effects | Catalog consumers trigger spend/install; harder to reason about authority |
| Only backend manager / unified service as the public API | Heavy optional stack becomes mandatory; simple `generate_text` dies offline |
| Silent mock success when no provider exists | Hides misconfiguration; fails open for agents and CI |
| Collapsing all modalities into one router without protocols | Embeddings/audio contracts and fallback labels collide |
| Trusting LLM output as proof of successful routing | Trace and receipts exist precisely so prose is non-authoritative |

**Consequences.**

- **Positive:** Clear entry for apps (`generate_text`); capability-first ops; modular extras; consistent failure language.  
- **Negative:** Multiple related entrypoints (`run_model` vs routers vs optional HTTP); dual MCP packages (canonical + facade); large router modules; operators must learn which plane they are in. Documentation must keep pointing at **current** symbols rather than historical “unified” diagrams alone.

---

## 9. Extension points

| Goal | Where to extend | Rules |
| --- | --- | --- |
| New LLM provider | `register_llm_provider` / builtin factory in `llm_router` | Publish catalog descriptors; no secrets in descriptors; tests with fakes |
| New embedding/voice/multimodal provider | Matching `register_*` and protocol | Same catalog + fallback label discipline |
| New HTTP API backend | `api_backends/` subclassing `BaseAPIBackend` | Queue + circuit breaker; wire into coordinator or backend manager explicitly |
| New hardware skill | `worker/skillset` or hardware utils | Probe capability; never claim universal support |
| New backend type in manager | `InferenceBackendManager.register_*` | Map tasks to catalog operations at the boundary |
| New public tool | MCP tool registration calling existing routers | Auth classification in MCP docs; do not reimplement routing |
| New catalog source | `model_catalog/sources` | Side-effect-free `load()` unless refresh policy allows |

---

## 10. Operational signals

| Signal | How to read it |
| --- | --- |
| `get_instance().get_capabilities(detail=True)` | Task types, models/endpoints known to coordinator, hwtest booleans, MCP counts |
| `list_providers()` / modality equivalents | Constructible/projected providers (still not live SLAs) |
| `get_last_generation_trace()` | Effective provider and model after last LLM call |
| `get_last_usage_admission()` | Operational admission codes when usage path is wired |
| Backend manager status report | Health, metrics, routing table when service is used |
| Logs | Router and backend loggers; prefer structured reasons over scraping chat |
| Circuit breaker state | Per API backend: OPEN means degrade/skip until reset timeout |

Network-heavy live provider smoke tests are **optional** for architecture verification. Absence of a key or GPU must remain a first-class documented outcome.

---

## 11. Verification

Deterministic checks for this guide (no optional hardware required):

```bash
# File contract (DOC-006 acceptance)
test -f docs/architecture/INFERENCE_RUNTIME.md
rg -q 'Last verified' docs/architecture/INFERENCE_RUNTIME.md
rg -qi 'failure|fallback' docs/architecture/INFERENCE_RUNTIME.md
git diff --check

# Source anchors still present
test -f ipfs_accelerate_py/ipfs_accelerate.py
test -f ipfs_accelerate_py/llm_router.py
test -f ipfs_accelerate_py/inference_backend_manager.py
test -f ipfs_accelerate_py/embeddings_router.py
test -f ipfs_accelerate_py/worker/worker.py

# Import and capability probe (may warn; must not require CUDA)
python - <<'PY'
from ipfs_accelerate_py import get_instance
caps = get_instance().get_capabilities(detail=True)
assert isinstance(caps, dict)
assert "task_types" in caps
print("capabilities_ok", sorted(caps.get("task_types") or []))
PY
```

Review checklist:

- [ ] One request can be traced entrypoint → router → adapter → result  
- [ ] Catalog is described as non-invoking  
- [ ] Fallback levels and fail-closed cases are explicit  
- [ ] Optional providers/hardware are never assumed present  
- [ ] Routers vs execution adapters separation is stated with rationale  

---

## 12. Related guides and plans

| Document | Role |
| --- | --- |
| [overview.md](overview.md) | Maintained one-screen architecture |
| [AI_SERVICE_CATALOG.md](AI_SERVICE_CATALOG.md) | Catalog schema, sources, security |
| [GUIDE_CONVENTIONS.md](GUIDE_CONVENTIONS.md) | Architecture writing contract |
| [ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md](ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md) | Plan: usage-aware admission (non-normative until Current guide) |
| [../INFERENCE_BACKEND_README.md](../INFERENCE_BACKEND_README.md) | Optional unified backend ops |
| [../UNIFIED_INFERENCE_ARCHITECTURE.md](../UNIFIED_INFERENCE_ARCHITECTURE.md) | Compatibility pointer to current design |
| MCP server README under `ipfs_accelerate_py/mcp_server/` | Tools and transports |
| Planned: `SYSTEM_CONTEXT.md`, `MODEL_SERVICE_ROUTING.md`, `MCP_RUNTIME.md`, `DISTRIBUTED_RUNTIME.md` | Sibling architecture guides in the documentation refresh |

---

## Appendix A: Minimal mental model

```text
Ask catalog what can be true.
Ask a router to make it true once.
Adapters speak wires and devices.
Absence is an answer, not a crash of the architecture.
```

## Appendix B: Interfaces claimed by this guide

| Interface | Meaning in this tree |
| --- | --- |
| **InferenceRequestFlow@1** | Public entry → (optional catalog) → modality router → provider/adapter → result/error/trace |
| **RouterFallbackFlow@1** | Cache → primary → within-provider model fallback → optional cross-provider → optional local HF → raise |

These names are documentation contracts for parallel writers and agents. They are not importable Python types.
