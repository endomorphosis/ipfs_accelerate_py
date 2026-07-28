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
`vibe`, `goose`, `goose-cli`, `block_goose`, `aaif_goose`, and `accelerate` are
accepted where implemented. For text generation, `grok` prefers the installed
CLI and falls back to the xAI REST provider when the CLI is unavailable; use
`grok_cli` or `grok_api` when the transport must be unambiguous.

`goose_cli` is a peer of `codex_cli` / `copilot_cli`. Ordinary
`generate_text(..., provider="goose_cli")` is **chat-only** (no tools, no
session, no default extensions). The default model backend is Meta Muse Spark
through Goose's OpenAI-compatible transport (`OPENAI_HOST=https://api.meta.ai`
plus the package Meta credential). Direct HTTP Muse Spark without Goose remains
`meta_ai`. Authorized tool-using agent runs require an explicit agent policy
and path roots (see [Goose CLI](#goose-cli)). Use `get_llm_provider(name)` or
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
| `ipfs_accelerate_py_GROK_CLI_MODEL` / `GROK_CLI_MODEL` | Grok CLI model; defaults to `grok-4.5` (run `grok models` for the catalog). |
| `ipfs_accelerate_py_CLAUDE_CODE_CLI_CMD` | Claude Code command template. |
| `IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD` / `ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD` | Mistral Vibe command template. |
| `MISTRAL_API_KEY` or `ipfs_accelerate_py_MISTRAL_API_KEY` | Mistral authentication. |
| `XAI_API_KEY` or `ipfs_accelerate_py_XAI_API_KEY` | xAI authentication. |
| `MODEL_API_KEY`, `META_AI_API_KEY`, or `ipfs_accelerate_py_META_AI_API_KEY` | Meta Model API authentication; the encrypted `meta_ai_api_key` credential is used when these are unset. |
| `ipfs_accelerate_py_META_AI_MODEL` | Meta Model API model; defaults to `muse-spark-1.1`. |
| `ipfs_accelerate_py_META_AI_BASE_URL` | Meta Model API endpoint; defaults to `https://api.meta.ai/v1`. |
| `IPFS_ACCELERATE_GOOSE_PATH` (aliases: `IPFS_ACCELERATE_PY_GOOSE_PATH`, `ipfs_accelerate_py_GOOSE_BIN`, `GOOSE_BIN`, `GOOSE_CLI_PATH`) | Explicit Goose binary path (detect-only; never installs). |
| `IPFS_ACCELERATE_GOOSE_DISCOVERY` (aliases: `IPFS_ACCELERATE_PY_GOOSE_DISCOVERY`, `ipfs_accelerate_py_GOOSE_DISCOVERY`) | Opt-in for Goose in *implicit* provider discovery (default off). |
| `IPFS_ACCELERATE_GOOSE_AUTO_INSTALL` (aliases: `IPFS_ACCELERATE_PY_GOOSE_AUTO_INSTALL`, `ipfs_accelerate_py_GOOSE_AUTO_INSTALL`) | Allow explicit `ensure_goose` install path when not set to a falsey value; set `0` to disable. |
| `ipfs_accelerate_py_GOOSE_CLI_MODEL` / `GOOSE_MODEL` | Goose model name (not the underlying provider). |
| `GOOSE_PROVIDER` (aliases: `ipfs_accelerate_py_GOOSE_PROVIDER`, `IPFS_ACCELERATE_PY_GOOSE_PROVIDER`) | Goose underlying provider id (for example `openai`). |
| `GOOSE_PATH_ROOT` | Absolute isolation root for agent sessions and path validation. |
| `IPFS_ACCELERATE_GOOSE_MANAGED_ROOT` | Override managed install base (version dirs live under this root). |
| `IPFS_ACCELERATE_LLAMA_CPP_*` | llama.cpp server URL, model, startup, and GPU settings. |
| `IPFS_ACCELERATE_LLAMA_CPP_NATIVE_*` | native llama.cpp model, context, thread, and GPU settings. |
| `ipfs_accelerate_py_ROUTER_CACHE` | Provider-instance cache; enabled unless `0`. |
| `ipfs_accelerate_py_ROUTER_RESPONSE_CACHE` | Response cache; enabled unless `0`. |
| `ipfs_accelerate_py_ROUTER_CACHE_KEY` | `sha256` or `cid` response key strategy. |
| `ipfs_accelerate_py_ROUTER_CACHE_CID_BASE` | CID encoding base. |

The router also accepts legacy/alternate environment names for selected batch
and integration settings. Prefer the names documented in the module docstring
and inspect the current source before relying on an undocumented alias. Do not
commit API keys or provider tokens; pass credentials through the environment
or the encrypted package secret manager.

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

## Goose CLI

Block/AAIF Goose is integrated as a first-class router provider (`goose_cli`,
alias `goose` and related spellings) through the shared CLI runtime. The same
canonical adapter backs `llm_router`, CLI endpoint registration, MCP tools,
compatibility wrappers, and opt-in P2P workers. **Chat generation and agent
execution are deliberately separate authority surfaces.**

### Provider identity and aliases

| Role | Canonical name | Compatibility aliases |
| --- | --- | --- |
| Safe chat / default router provider | `goose_cli` | `goose`, `goose-cli`, `block_goose`, `block-goose`, `aaif_goose` |
| Explicit remote agent provider name (P2P) | `goose_agent` | `goose-agent` |

`model_name` / `GOOSE_MODEL` select the Goose **model**. `goose_provider` /
`GOOSE_PROVIDER` select Goose's **underlying** provider (for example `openai`
for Muse Spark over the OpenAI-compatible transport). Never collapse model and
underlying provider into one field.

### Canonical environment variables (no secrets)

Values below are names and purposes only. Never commit real keys.

| Canonical variable | Compatibility aliases | Purpose |
| --- | --- | --- |
| `IPFS_ACCELERATE_GOOSE_PATH` | `IPFS_ACCELERATE_PY_GOOSE_PATH`, `ipfs_accelerate_py_GOOSE_BIN`, `IPFS_ACCELERATE_PY_GOOSE_BIN`, `IPFS_ACCELERATE_AGENT_GOOSE_BIN`, `GOOSE_BIN`, `GOOSE_CLI_PATH` | Explicit path to the `goose` executable |
| `IPFS_ACCELERATE_GOOSE_DISCOVERY` | `IPFS_ACCELERATE_PY_GOOSE_DISCOVERY`, `ipfs_accelerate_py_GOOSE_DISCOVERY` | Opt-in so Goose may appear during *implicit* discovery (default **off**) |
| `IPFS_ACCELERATE_GOOSE_AUTO_INSTALL` | `IPFS_ACCELERATE_PY_GOOSE_AUTO_INSTALL`, `ipfs_accelerate_py_GOOSE_AUTO_INSTALL` | When not falsey, explicit selection may install the pinned release; set `0` / `false` / `off` to disable |
| `IPFS_ACCELERATE_GOOSE_MANAGED_ROOT` | `IPFS_ACCELERATE_PY_GOOSE_MANAGED_ROOT` | Base directory for versioned managed installs |
| `IPFS_ACCELERATE_GOOSE_VARIANT` | `IPFS_ACCELERATE_PY_GOOSE_VARIANT`, `GOOSE_LINUX_VARIANT`, `GOOSE_WINDOWS_VARIANT` | Archive variant (`standard`, `vulkan`, `musl`, `cuda` where supported) |
| `ipfs_accelerate_py_GOOSE_CLI_MODEL` | `IPFS_ACCELERATE_PY_GOOSE_CLI_MODEL`, `GOOSE_MODEL` | Goose model (default often `muse-spark-1.1`) |
| `GOOSE_PROVIDER` | `ipfs_accelerate_py_GOOSE_PROVIDER`, `IPFS_ACCELERATE_PY_GOOSE_PROVIDER`, `IPFS_ACCELERATE_GOOSE_PROVIDER` | Underlying provider id for Goose (default `openai` for Muse Spark) |
| `GOOSE_PATH_ROOT` | `IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT` (worker) | Absolute root for agent path isolation |
| `IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI` | — | P2P worker: admit safe chat Goose providers |
| `IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT` | — | P2P worker: admit agent mode (also needs allowlist) |
| `IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_ALLOWED_ROOTS` | — | Extra absolute roots allowed for remote path fields |
| `IPFS_ACCELERATE_GOOSE_LIVE` | — | Gate for the opt-in live smoke test (default suite is offline) |

Backend credentials (for example `OPENAI_API_KEY` with `OPENAI_HOST`, or other
provider keys Goose understands) are required for *ready* chat. The installer
and discovery paths never print or store secret values; they only check whether
a non-empty auth marker is present.

### Discovery does not install (explicit opt-in)

Two planes must stay distinct:

1. **Generic / implicit discovery** (`list_providers()`, automatic provider
   order, unforced resolution) is **detect-only**. It never calls
   `ensure_goose`, never downloads archives, and **does not include Goose
   unless** `IPFS_ACCELERATE_GOOSE_DISCOVERY=1` (or a listed alias) is set.
2. **Explicit selection** (`provider="goose_cli"` / `"goose"`, forced
   `ipfs_accelerate_py_LLM_PROVIDER=goose_cli`) may invoke the pinned lazy
   installer via `ensure_goose` unless auto-install is disabled with
   `IPFS_ACCELERATE_GOOSE_AUTO_INSTALL=0`.

Importing the package, listing providers, and catalog snapshot construction
never install Goose or start a model request.

### Safe chat usage (router)

Ordinary `generate_text` is chat-only: `GOOSE_MODE=chat`, no session, no default
profile, no builtin/external extensions, low max-turn and tool-repetition
bounds, prompt on stdin, structured JSON output. Side-effecting kwargs without
authorization are rejected.

```python
from ipfs_accelerate_py import generate_text

# Explicit provider selection. Does not require discovery opt-in.
# Ensure OPENAI_API_KEY (or package Meta credential) and optional OPENAI_HOST
# are configured in the process environment — do not hard-code secrets.
answer = generate_text(
    "Explain content addressing in one sentence.",
    provider="goose_cli",
    model_name="muse-spark-1.1",
    goose_provider="openai",
    max_tokens=128,
)
print(answer)
```

Compatibility alias:

```python
answer = generate_text("ping", provider="goose", model_name="muse-spark-1.1")
```

To allow Goose only when scanning available providers automatically:

```bash
export IPFS_ACCELERATE_GOOSE_DISCOVERY=1
# still does not install; binary must already be on PATH or IPFS_ACCELERATE_GOOSE_PATH
```

### Authorized agent endpoint usage (separate surface)

Agent mode can run tools, write under bounded roots, and use sessions. It is
**not** the default for `generate_text`. Requirements:

- Explicit agent authorization (`agent=True` / `side_effecting=True` /
  `allow_side_effects=True` and/or a `GooseAgentPolicy` / `agent_policy` mapping)
- Absolute `cwd` / `workspace` under an absolute `path_root` (`GOOSE_PATH_ROOT`)
- Explicit approval mode (not `chat`), extension/builtin allowlists (may be empty)
- Finite `max_turns`, timeouts, and output bounds
- Endpoint package enable policy (`enable_agent=True` on the registered endpoint)
- Agent requests **bypass** response caches, default-model retry, automatic
  provider fallback, and concurrent batch workers

```python
from pathlib import Path
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (
    register_cli_endpoint,
    execute_cli_inference,
    get_cli_endpoint,
)

work = Path("/var/lib/goose-jobs/job-1").resolve()
root = work.parent
work.mkdir(parents=True, exist_ok=True)

register_cli_endpoint(
    tool="goose",
    endpoint_id="goose-agent-local",
    config={"enable_agent": True, "model": "muse-spark-1.1", "goose_provider": "openai"},
)

adapter = get_cli_endpoint("goose-agent-local")
result = adapter.execute(
    "Summarize README.md in one paragraph.",
    execution_mode="agent",
    enable_agent=True,
    allow_side_effects=True,
    cwd=str(work),
    path_root=str(root),
    approval_mode="approve",
    builtins=["developer"],
    extensions=[],
    max_turns=8,
    max_tool_repetitions=3,
    timeout_seconds=120,
    max_output_bytes=65536,
    session_id="job-1",
    allowed_cwd_roots=[str(root)],
)
print(result["status"], result.get("text") or result.get("result"))
# envelopes never echo the prompt or credentials
```

One-shot safe chat on the same endpoint surface (no agent flags):

```python
out = execute_cli_inference(
    "goose-agent-local",
    "Reply with a single short greeting.",
    timeout=60,
)
```

### Managed install location and pinned updates

Lazy installation is **explicit-only**, noninteractive, SHA-256 verified against
the packaged release manifest, and atomically promoted under install locks.

- **Pinned version** (code constant): `PINNED_GOOSE_VERSION` in
  `ipfs_accelerate_py.cli_runtime.providers.goose` (currently `1.44.0`).
- **Default managed root**:
  `$XDG_DATA_HOME/ipfs_accelerate_py/goose/<version>/` or, when `XDG_DATA_HOME`
  is unset, `~/.local/share/ipfs_accelerate_py/goose/<version>/`.
- **Executable path**: `<managed-root>/<version>/bin/goose` (or `goose.exe` on
  Windows). Override the base with `IPFS_ACCELERATE_GOOSE_MANAGED_ROOT`.
- **Discovery order** (never installs): explicit path env → PATH `goose` →
  managed version directory for the pinned release.

**Pinned-version update procedure:**

1. Choose a supported Goose release tag and download assets for each supported
   OS/arch/libc/variant matrix entry.
2. Update `ipfs_accelerate_py/cli_runtime/installers/goose_release_manifest.json`:
   set `pinned_version`, `release_tag`, `minimum_version`, and per-asset
   `asset_name`, `size_bytes`, and `sha256`. Keep `schema_version` and
   `allowed_archive_members` consistent with the installer.
3. Bump `PINNED_GOOSE_VERSION` in the provider module to match the manifest.
4. Run `python -m pytest test/test_goose_installer.py -q` and the Goose router /
   endpoint suites offline.
5. On hosts already running Goose, either point `IPFS_ACCELERATE_GOOSE_PATH` at
   the new managed binary or re-run explicit ensure (for example by selecting
   `provider="goose_cli"` with auto-install enabled) so the new version directory
   is populated. Leave the previous version directory in place until smoke
   checks pass.

**Checksum manifest maintenance:** every published asset must carry a correct
`sha256` of the full archive bytes. Reject digest mismatches fail-closed; do
not weaken verification for convenience. Prefer updating the packaged manifest
in source control rather than pointing at unpinned “latest” URLs.

### Shared versus isolated `GOOSE_PATH_ROOT`

| Mode | Configuration | Implications |
| --- | --- | --- |
| Shared root | One `GOOSE_PATH_ROOT` (or worker `IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT`) for many jobs | Simpler ops; sessions and tool writes share the same confinement tree. Require distinct session ids and careful cwd layout. A path escape under the shared root affects co-tenants. |
| Isolated root | Per-job or per-worker absolute root (recommended for agent / P2P sticky sessions) | Stronger isolation and cleaner rollback (delete the job root). Persisted ACP/agent sessions should stick to one worker with its own absolute root. Do not reuse a root across untrusted tenants. |

All agent path fields (`cwd`, `workspace`, config/recipe/trace/session paths)
must resolve under the configured root(s). Relative paths fail closed.

### Readiness versus liveness, cancellation, and recovery

| Probe | Side effects | Meaning |
| --- | --- | --- |
| **Liveness** / list endpoints | No model request, no install | Process/registry entry exists; adapter is registered |
| **Installed** | Detect-only binary/version probe | Goose executable found; may still lack credentials |
| **Configured** | No secret values logged | Provider/model or endpoint config present |
| **Ready** (readiness) | Still no chat prompt | Installed **and** authenticated/configured for chat |
| **Unsupported version** | Fail closed | Binary lacks required safety flags for this integration |
| **Degraded / missing** | — | Not ready for traffic; do not treat as success |

Cancellation and timeouts kill the Goose child process group (descendants
included). Streaming and ACP session stop paths clean up the same way.
After cancel or uncertain failure:

1. Inspect `side_effects_started` / error classification on the envelope.
2. Do **not** blindly retry agent work; tool activity may have already mutated
   the workspace.
3. Recover by re-opening an authorized session only when the policy and roots
   still apply, or by resetting the isolated job root and starting a new
   session id.

### P2P worker gates and no-replay behavior

Goose is **absent from the default remote provider allowlist**. Enabling remote
Goose is a two-step opt-in:

```bash
# Safe chat only
export IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI=1

# Agent (also require allowlist membership for goose_agent / wildcard under gate)
export IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT=1
export IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS=goose_cli,goose_agent
export IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT=/var/lib/goose-worker/root
```

Wildcard (`*` / `all`) expansion still requires the matching enable gate before
Goose names are admitted. After any Goose attempt—especially agent or uncertain
failures—workers:

- do **not** fall back to Codex, Copilot, or local HF
- do **not** cache or auto-retry side-effecting work
- record delivery keys so **duplicate delivery does not automatically replay**
  uncertain agent tasks in-process
- keep sticky sessions on the assigned worker with an absolute `GOOSE_PATH_ROOT`

### Offline default test suite and opt-in live smoke

The Goose pytest suite is **offline by default**: fakes and mocks only; no live
binary, network install, or provider credentials are required.

```bash
python -m pytest \
  test/test_llm_router_goose.py \
  test/test_goose_cli_endpoint.py \
  test/test_goose_p2p_policy.py \
  test/test_goose_cli_provider.py \
  test/test_goose_installer.py -q
```

Live smoke is gated by an explicit environment variable **and** a configured
provider (binary + backend credentials). Example:

```bash
export IPFS_ACCELERATE_GOOSE_LIVE=1
# binary on PATH or:
# export IPFS_ACCELERATE_GOOSE_PATH=/path/to/goose
# export OPENAI_API_KEY=...   # or Meta package credential; do not commit
# export GOOSE_PROVIDER=openai
# export GOOSE_MODEL=muse-spark-1.1
python -m pytest test/test_llm_router_goose.py -k opt_in_live_goose_smoke -q
```

Without `IPFS_ACCELERATE_GOOSE_LIVE=1`, the live smoke test is skipped and the
default suite remains fully offline.

### Rollback and troubleshooting

**Rollback**

1. Stop routing traffic to `goose_cli` / `goose` (unset forced provider and
   discovery flags; remove Goose from worker allowlists and disable
   `ENABLE_GOOSE_*` gates).
2. Point `IPFS_ACCELERATE_GOOSE_PATH` at the previous managed version directory,
   or remove the new version dir and restore the prior manifest pin in a
   follow-up deploy.
3. Retain isolated job roots for forensic review; delete only after policy
   allows. ACP sessions bound to a failed pin should be stopped, not replayed.
4. Continue using peer providers (`codex_cli`, `copilot_cli`, `meta_ai`, `mock`)
   without Goose in the automatic discovery set.

**Troubleshooting**

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Goose missing from `list_providers()` | Discovery default off | Set `IPFS_ACCELERATE_GOOSE_DISCOVERY=1` or select `provider="goose_cli"` explicitly |
| `not installed` / missing health | No binary on PATH or managed root | Install explicitly, set `IPFS_ACCELERATE_GOOSE_PATH`, or enable auto-install for explicit selection |
| Installed but not ready | No backend credential / provider config | Configure `GOOSE_PROVIDER` and the matching API key env without logging secrets |
| Auto-install no-ops | `IPFS_ACCELERATE_GOOSE_AUTO_INSTALL=0` or implicit discovery path | Use explicit provider selection; flip auto-install only when policy allows |
| Digest / manifest errors | Manifest sha256 or size mismatch | Re-verify archives; repair `goose_release_manifest.json`; do not skip checksums |
| Unsupported version | Old Goose without required safety flags | Upgrade to the pinned release |
| Agent `policy_denied` | Missing `enable_agent`, roots, or `allow_side_effects` | Supply full agent policy with absolute paths |
| P2P `not allowed` / disabled | Enable gates off | Set `ENABLE_GOOSE_CLI` / `ENABLE_GOOSE_AGENT` and allowlist as required |
| Unexpected cross-provider result | Should not happen after Goose attempt | Confirm worker build includes Goose no-fallback policy; treat as a defect |

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

Goose CLI contracts (offline fakes by default; see [Goose CLI](#goose-cli)):

```bash
python -m pytest \
  test/test_llm_router_goose.py \
  test/test_goose_cli_endpoint.py \
  test/test_goose_p2p_policy.py -q
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

Usage-aware admission for `llm_router` is opt-in via `RoutingPolicy.mode`.
Default `off` preserves legacy selection byte-for-byte. Observe/shadow never
change selection; enforce/assist reserve before dispatch. Cross-router contract
and staged rollout proofs:

```bash
python -m pytest \
  test/test_llm_router_usage_routing.py \
  test/test_ai_router_usage_contract.py \
  test/test_endpoint_usage_conformance.py \
  test/test_endpoint_usage_faults.py \
  test/test_endpoint_usage_rollout.py -q
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
