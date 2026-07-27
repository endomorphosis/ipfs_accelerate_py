# LLM Router

`ipfs_accelerate_py.llm_router` is the provider boundary for text generation.
It resolves an explicitly selected provider, a registered provider, or the
first available configured provider. External credentials, CLIs, SDKs, model
weights, and local services are optional; importing the router does not install
or start them.

It is also the implementation used by the historical
`ipfs_datasets_py.llm_router` import path. See
[inference router ownership](ROUTER_OWNERSHIP.md) for the shared boundary.

## Basic usage

```python
from ipfs_accelerate_py import generate_text

answer = generate_text(
    "Explain content addressing in one sentence.",
    provider="openrouter",
    model_name="openai/gpt-4o-mini",
    max_tokens=128,
    temperature=0.1,
)
print(answer)
```

When no provider is specified, the router checks its configured provider
resolution path. To make a run reproducible, select a provider explicitly and
record the model, relevant environment, and dependency versions.

## Catalog discovery versus invocation

The router has two deliberately separate surfaces:

| Information plane | Invocation plane |
| --- | --- |
| `list_providers()` | `generate_text()` |
| `get_provider_descriptor(name)` | `generate_text_batch()` |
| `list_models(provider=...)` | `generate_text_mesh()` |
| `resolve_model(..., operation=...)` | provider-specific request handling |
| `get_catalog_snapshot()` / `catalog_snapshot()` | streaming, fallback, and response caching |

Discovery returns the shared versioned `ProviderDescriptor`,
`ModelDescriptor`, and `RouterBinding` records. It does not construct a
provider, install a CLI, start a process, load model weights, read a credential
store, probe an endpoint, or make a model request. `resolve_model()` is
metadata-only and accepts the same explicit provider aliases and model
overrides as invocation.

```python
from ipfs_accelerate_py import llm_router

providers = llm_router.list_providers()
model = llm_router.resolve_model(
    "openai/gpt-4o-mini",
    provider="openrouter",
    operation="text.generate",
)
snapshot = llm_router.get_catalog_snapshot()

print(model.model_id, snapshot.revision)
```

Canonical names determine stable IDs. Aliases such as `hf` or `codex` remain
accepted compatibility inputs but do not create new identities. A provider
being known is not proof that it is configured, authorized, reachable,
healthy, or routable; those are independent tri-state facts.

The aggregate catalog and `ModelManager.resolve()` can rank bindings across
routers and sources. After resolution, invoke the selected text binding through
this module. The catalog never owns `generate_text()`. See the
[AI Service Catalog architecture](architecture/AI_SERVICE_CATALOG.md).

## Provider names

The built-in names currently recognized by the router include:

| Provider | Boundary | Typical prerequisite |
| --- | --- | --- |
| `openrouter` | OpenRouter-compatible HTTP API | API key and network access |
| `codex_cli` | Codex CLI process | `codex` executable and auth |
| `copilot_cli` | GitHub Copilot CLI process | Copilot CLI and auth |
| `copilot_sdk` | Python Copilot SDK | optional SDK and auth |
| `goose_cli` | Block/AAIF Goose CLI process | `goose` executable; default backend is Meta Muse Spark via OpenAI-compatible env |
| `gemini_cli` / `gemini_py` | Gemini CLI or Python wrapper | Gemini tool/SDK and credentials |
| `grok_cli` | Official xAI Grok CLI | `grok` executable and CLI OAuth or `XAI_API_KEY` |
| `claude_code` / `claude_py` | Claude Code CLI or Python wrapper | Claude tool/SDK and credentials |
| `mistral_vibe` | Mistral Vibe CLI | `vibe` and Mistral credentials |
| `xai` | xAI OpenAI-compatible API | `XAI_API_KEY` |
| `meta_ai` | Meta Model API / Muse Spark (HTTP) | Meta API credentials (`meta_ai_api_key` / `MODEL_API_KEY`) |
| `llama_cpp` | Local llama.cpp server | running or auto-startable server |
| `llama_cpp_native` | `llama-cpp-python` binding | local GGUF and binding |
| `local_hf` | Transformers pipeline | `transformers` and model weights |
| `mock` | deterministic test provider | no external dependency |

Aliases such as `codex`, `claude`, `grok`, `xai_cli`, `hf`, `huggingface`,
`vibe`, `goose`, and `accelerate` are accepted where implemented. For text
generation, `grok` prefers the installed CLI and falls back to the xAI REST
provider when the CLI is unavailable; use `grok_cli` or `grok_api` when the
transport must be unambiguous.

`goose_cli` is a peer of `codex_cli` / `copilot_cli`. Ordinary
`generate_text(..., provider="goose_cli")` is **chat-only** (no tools, no
session, no default extensions). The default model backend is Meta Muse Spark
through Goose's OpenAI-compatible transport (`OPENAI_HOST=https://api.meta.ai`
plus the package Meta credential). Direct HTTP Muse Spark without Goose remains
`meta_ai`. Authorized tool-using agent runs pass `agent=True` and an explicit
`workspace` (used by the agent supervisor). Use `get_llm_provider(name)` or the
source module for the exact current alias set.

## Provider selection and registration

```python
from ipfs_accelerate_py import generate_text, register_llm_provider

class ExampleProvider:
    def generate(self, prompt, *, model_name=None, **kwargs):
        return f"response for {prompt}"

register_llm_provider("example", lambda: ExampleProvider())
print(generate_text("hello", provider="example"))
```

Production registrations should also supply a side-effect-free provider
descriptor and model hints. A generated router binding must resolve back to the
same canonical provider/model IDs. Provider construction belongs in the
factory; it must not happen while the descriptor or catalog snapshot is being
built.

The environment variable `ipfs_accelerate_py_LLM_PROVIDER` forces a provider
name. If it names a provider that is not registered or available, the router
fails rather than silently selecting an unrelated provider.

## Configuration

The router reads the following current namespaces. Values are examples, not
secrets to commit:

| Variable | Purpose |
| --- | --- |
| `ipfs_accelerate_py_LLM_PROVIDER` | Force provider selection. |
| `ipfs_accelerate_py_LLM_MODEL` | Default local-HF/model fallback name. |
| `ipfs_accelerate_py_OPENROUTER_API_KEY` or `OPENROUTER_API_KEY` | OpenRouter authentication. |
| `ipfs_accelerate_py_OPENROUTER_MODEL` | OpenRouter model. |
| `ipfs_accelerate_py_CODEX_CLI_MODEL` / `ipfs_accelerate_py_CODEX_MODEL` | Codex model hint. |
| `ipfs_accelerate_py_COPILOT_CLI_CMD` | Copilot CLI command template. |
| `ipfs_accelerate_py_COPILOT_SDK_MODEL` | Copilot SDK model. |
| `ipfs_accelerate_py_GEMINI_CLI_CMD` | Gemini CLI command template. |
| `ipfs_accelerate_py_GROK_CLI_CMD` / `GROK_CLI_CMD` | Grok CLI command or command template. |
| `ipfs_accelerate_py_GROK_CLI_MODEL` / `GROK_CLI_MODEL` | Optional Grok CLI model; otherwise the CLI default is used. |
| `ipfs_accelerate_py_CLAUDE_CODE_CLI_CMD` | Claude Code command template. |
| `IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD` / `ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD` | Mistral Vibe command template. |
| `MISTRAL_API_KEY` or `ipfs_accelerate_py_MISTRAL_API_KEY` | Mistral authentication. |
| `XAI_API_KEY` or `ipfs_accelerate_py_XAI_API_KEY` | xAI authentication. |
| `MODEL_API_KEY`, `META_AI_API_KEY`, or `ipfs_accelerate_py_META_AI_API_KEY` | Meta Model API authentication; the encrypted `meta_ai_api_key` credential is used when these are unset. |
| `ipfs_accelerate_py_META_AI_MODEL` | Meta Model API model; defaults to `muse-spark-1.1`. |
| `ipfs_accelerate_py_META_AI_BASE_URL` | Meta Model API endpoint; defaults to `https://api.meta.ai/v1`. |
| `IPFS_ACCELERATE_LLAMA_CPP_*` | llama.cpp server URL, model, startup, and GPU settings. |
| `IPFS_ACCELERATE_LLAMA_CPP_NATIVE_*` | native llama.cpp model, context, thread, and GPU settings. |
| `ipfs_accelerate_py_ROUTER_CACHE` | Provider-instance cache; enabled unless `0`. |
| `ipfs_accelerate_py_ROUTER_RESPONSE_CACHE` | Response cache; enabled unless `0`. |
| `ipfs_accelerate_py_ROUTER_CACHE_KEY` | `sha256` or `cid` response key strategy. |
| `ipfs_accelerate_py_ROUTER_CACHE_CID_BASE` | CID encoding base. |

The router also accepts legacy/alternate environment names for selected batch
and integration settings. Prefer the names documented in the module docstring
and inspect the current source before relying on an undocumented alias.

### Meta Muse Spark 1.1

Store the credential once in the encrypted package credential manager:

```python
from getpass import getpass
from ipfs_accelerate_py.common.secrets_manager import get_global_secrets_manager

get_global_secrets_manager().set_credential(
    "meta_ai_api_key",
    getpass("Meta Model API key: "),
)
```

Then use the canonical hosted model without exporting the key:

```python
from ipfs_accelerate_py import generate_text

answer = generate_text(
    "Reply with one sentence about content-addressed data.",
    provider="meta_ai",
    model_name="muse-spark-1.1",
    max_completion_tokens=512,
)
print(answer)
```

The compatibility spelling `meta-spark/Spark-1.1` is normalized to
`muse-spark-1.1`. Environment credentials override the encrypted store for a
single process. Set `IPFS_ACCELERATE_PY_DISABLE_SECRET_MANAGER=1` when an
isolated process must not read persistent credentials.

## Caching and dependency injection

```python
from ipfs_accelerate_py import RouterDeps, generate_text

deps = RouterDeps()
first = generate_text("first prompt", provider="mock", deps=deps)
second = generate_text("second prompt", provider="mock", deps=deps)
print(first, second)
```

`RouterDeps` lets applications share caches, backend managers, storage, and
other injected resources. `clear_llm_router_caches()` clears router-local
provider caches. Response caching is useful only when the provider request is
safe to replay; do not cache prompts or outputs containing sensitive data
without an appropriate storage policy.

Router provider/response caches are separate from the AI catalog metadata
cache. The latter stores only descriptors and health samples, uses independent
capability and health TTLs, and never stores prompts, media, or inference
output. Clearing one cache does not promise to invalidate the other.

## Grok CLI

Authenticate the installed CLI once, then select the CLI transport explicitly:

```bash
grok login --device-code
python - <<'PY'
from ipfs_accelerate_py import generate_text

print(generate_text(
    "Explain content addressing in one sentence.",
    provider="grok_cli",
    model_name="grok-4.5",
    max_tokens=128,
))
PY
```

The router invokes Grok in bounded, non-interactive JSON mode. It disables
plan mode, subagents, web search, cross-session memory, and tools by default,
passes the prompt through an owner-only temporary file, and extracts only the
final response text. OAuth credentials from `grok login` take precedence in
the CLI; `XAI_API_KEY` is also supported. Alternate accelerator and datasets
xAI key variables are forwarded to the CLI as `XAI_API_KEY`.

Use `trace=True` with `trace_jsonl_path=...` or `trace_dir=...` to retain
request, session, usage, and cost metadata. Prompt and response text are not
written to that trace by the router.

## Batch and mesh helpers

The module also exposes `generate_text_batch()` for ordered local batches and
`generate_text_mesh()` / `generate_text_mesh_batch()` for the optional P2P
TaskQueue route. Batch helpers use bounded worker counts; they do not make an
external provider infinitely parallel. Configure provider rate limits and
resource leases before increasing concurrency.

## llama.cpp examples

For an already-running compatible server:

```bash
export ipfs_accelerate_py_LLM_PROVIDER=llama_cpp
export IPFS_ACCELERATE_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1
python - <<'PY'
from ipfs_accelerate_py import generate_text
print(generate_text("Give one sentence about formal verification."))
PY
```

Native binding operation uses `llama_cpp_native` and the corresponding
`IPFS_ACCELERATE_LLAMA_CPP_NATIVE_*` model settings. Confirm context size,
GPU layers, threads, and model compatibility on the target host.

## Testing

Use the deterministic mock provider for offline contract tests:

```bash
python -m pytest test/test_llm_router_integration.py -q
python -m pytest test/test_llm_router_llama_cpp.py -q
```

Provider-specific tests may require credentials, CLIs, a running llama.cpp
server, network access, or model files. A provider being registered or
discoverable is not evidence that an end-to-end request will succeed.

The catalog conformance suite is offline by default and uses injected fake
providers:

```bash
python -m pytest \
  test/test_llm_router_catalog_discovery.py \
  test/test_ai_catalog_conformance.py -q
```

An operator may select only the live modalities available in an environment:

```bash
IPFS_ACCELERATE_PY_AI_CATALOG_LIVE=text \
IPFS_ACCELERATE_PY_AI_CATALOG_LIVE_TEXT_PROVIDER=openrouter \
IPFS_ACCELERATE_PY_AI_CATALOG_LIVE_TEXT_MODEL=openai/gpt-4o-mini \
python -m pytest test/test_ai_catalog_conformance.py \
  -k opt_in_live_provider_smoke -q
```

For migration, keep calling `generate_text()` for invocation and replace
duplicate provider/model enumeration with `ModelManager.list_services()`,
`list_catalog_models()`, and `resolve()`. MCP callers should migrate from
`generate_text` to `llm_generate`; the former remains a compatibility alias
with no scheduled removal. Roll back a catalog rollout by disabling catalog
selected traffic and refresh/federation, retaining the last immutable
revision, and continuing to invoke the router through its compatible public
entry points.

## Trust boundary

Router output is provider output. Applications and the agent supervisor must
keep it in a proposal/data tier until schema validation, policy checks, tests,
or authoritative proof/evidence receipts accept it. The router does not turn
generated text into executable code or a merge decision.

See [API overview](api/overview.md), [architecture overview](architecture/overview.md),
[AI Service Catalog](architecture/AI_SERVICE_CATALOG.md),
[MCP Server](MCP_SERVER.md), [testing](development/testing.md), and the
[agent supervisor guide](guides/AGENT_SUPERVISOR_GUIDE.md).
