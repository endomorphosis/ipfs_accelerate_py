# LLM Router

`ipfs_accelerate_py.llm_router` is the provider boundary for text generation.
It resolves an explicitly selected provider, a registered provider, or the
first available configured provider. External credentials, CLIs, SDKs, model
weights, and local services are optional; importing the router does not install
or start them.

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

## Provider names

The built-in names currently recognized by the router include:

| Provider | Boundary | Typical prerequisite |
| --- | --- | --- |
| `openrouter` | OpenRouter-compatible HTTP API | API key and network access |
| `codex_cli` | Codex CLI process | `codex` executable and auth |
| `copilot_cli` | GitHub Copilot CLI process | Copilot CLI and auth |
| `copilot_sdk` | Python Copilot SDK | optional SDK and auth |
| `gemini_cli` / `gemini_py` | Gemini CLI or Python wrapper | Gemini tool/SDK and credentials |
| `claude_code` / `claude_py` | Claude Code CLI or Python wrapper | Claude tool/SDK and credentials |
| `mistral_vibe` | Mistral Vibe CLI | `vibe` and Mistral credentials |
| `xai` | xAI OpenAI-compatible API | `XAI_API_KEY` |
| `meta_ai` | Meta Llama-compatible API | Meta API credentials |
| `llama_cpp` | Local llama.cpp server | running or auto-startable server |
| `llama_cpp_native` | `llama-cpp-python` binding | local GGUF and binding |
| `local_hf` | Transformers pipeline | `transformers` and model weights |
| `mock` | deterministic test provider | no external dependency |

Aliases such as `codex`, `claude`, `hf`, `huggingface`, `vibe`, and
`accelerate` are accepted where implemented. Use `get_llm_provider(name)` or
the source module for the exact current alias set.

## Provider selection and registration

```python
from ipfs_accelerate_py import generate_text, register_llm_provider

class ExampleProvider:
    def generate(self, prompt, *, model_name=None, **kwargs):
        return f"response for {prompt}"

register_llm_provider("example", lambda: ExampleProvider())
print(generate_text("hello", provider="example"))
```

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
| `ipfs_accelerate_py_CLAUDE_CODE_CLI_CMD` | Claude Code command template. |
| `IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD` / `ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD` | Mistral Vibe command template. |
| `MISTRAL_API_KEY` or `ipfs_accelerate_py_MISTRAL_API_KEY` | Mistral authentication. |
| `XAI_API_KEY` or `ipfs_accelerate_py_XAI_API_KEY` | xAI authentication. |
| `META_AI_API_KEY` or `ipfs_accelerate_py_META_AI_API_KEY` | Meta AI authentication. |
| `IPFS_ACCELERATE_LLAMA_CPP_*` | llama.cpp server URL, model, startup, and GPU settings. |
| `IPFS_ACCELERATE_LLAMA_CPP_NATIVE_*` | native llama.cpp model, context, thread, and GPU settings. |
| `ipfs_accelerate_py_ROUTER_CACHE` | Provider-instance cache; enabled unless `0`. |
| `ipfs_accelerate_py_ROUTER_RESPONSE_CACHE` | Response cache; enabled unless `0`. |
| `ipfs_accelerate_py_ROUTER_CACHE_KEY` | `sha256` or `cid` response key strategy. |
| `ipfs_accelerate_py_ROUTER_CACHE_CID_BASE` | CID encoding base. |

The router also accepts legacy/alternate environment names for selected batch
and integration settings. Prefer the names documented in the module docstring
and inspect the current source before relying on an undocumented alias.

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

## Trust boundary

Router output is provider output. Applications and the agent supervisor must
keep it in a proposal/data tier until schema validation, policy checks, tests,
or authoritative proof/evidence receipts accept it. The router does not turn
generated text into executable code or a merge decision.

See [API overview](api/overview.md), [architecture overview](architecture/overview.md),
[testing](development/testing.md), and the [agent supervisor guide](guides/AGENT_SUPERVISOR_GUIDE.md).
