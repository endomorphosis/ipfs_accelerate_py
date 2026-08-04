# HuggingFace Model Server

**Status:** Reference
**Owner:** hf-model-server maintainers
**Audience:** Operators and developers who need a local FastAPI HF serve path
**Scope:** Optional package surface under `ipfs_accelerate_py/hf_model_server/`:
install, CLI (`discover` / `hardware` / `serve`), HTTP routes, configuration,
and contract tests
**Non-goals:** Making the server a default install or a catalog-selected router
backend without explicit projection; claiming production security defaults
**Sources:** `ipfs_accelerate_py/hf_model_server/server.py`,
`config.py`, `cli.py`; `requirements-hf-server.txt`;
`test/test_hf_model_server_endpoint_contract.py`;
`test/api/test_serving_readiness_contracts.py`
**Last verified:** `2bf2cebd3` (2026-08-03); CLI commands, `ServerConfig` /
`HF_SERVER_*` env loading, and registered routes checked against the tree

`ipfs_accelerate_py.hf_model_server` is an optional FastAPI model-serving
surface. It is separate from the package-level accelerator API and from the
canonical MCP server. Its behavior is defined by the code in
`ipfs_accelerate_py/hf_model_server/` and the endpoint contract tests.

## Install and inspect

Install the repository's server requirements or the full feature extra:

```bash
python -m pip install -r requirements-hf-server.txt
# or
python -m pip install "ipfs-accelerate-py[full]"
```

Inspect the live CLI before starting a service:

```bash
python -m ipfs_accelerate_py.hf_model_server.cli --help
python -m ipfs_accelerate_py.hf_model_server.cli hardware
python -m ipfs_accelerate_py.hf_model_server.cli discover
```

## Start the server

```bash
python -m ipfs_accelerate_py.hf_model_server.cli serve \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --log-level INFO
```

The default configuration enables model discovery, batching, response caching,
circuit breaking, health checks, and Prometheus metrics when their optional
dependencies are available. Configure the service through `ServerConfig` or
the `HF_SERVER_*` environment variables in
`ipfs_accelerate_py/hf_model_server/config.py`.

## HTTP surface

The current server registers these routes (conditional routes are noted):

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check. |
| `GET` | `/ready` | Readiness check (503 when skill registry is unset). |
| `GET` | `/status` | Server status payload. |
| `GET` | `/metrics` | Prometheus metrics when metrics are enabled. |
| `GET` | `/v1/models` | List served models. |
| `POST` | `/v1/completions` | Completion requests. |
| `POST` | `/v1/chat/completions` | Chat completion requests. |
| `POST` | `/v1/embeddings` | Embedding requests. |
| `POST` | `/models/load` | Load a model. |
| `POST` | `/models/unload` | Unload a model. |
| `WS` | `/ws/{client_id}` | Optional streaming/status channel when websocket support is present. |
| `POST`/`GET` | `/admin/keys/*` | Optional admin key management when the API-key manager is configured. |

Default `ServerConfig.host` is `0.0.0.0`; the `serve` example above binds
`127.0.0.1` deliberately for local development.

Example health check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/models
```

Request schemas and authentication behavior are defined by the FastAPI server
and should be checked before client code is generated. Do not treat an endpoint
as proof that a model, tokenizer, hardware backend, or provider is available.

## Configuration and security

Important `ServerConfig` controls include host/port/workers, preferred hardware,
batching, caching, circuit breaking, model limits, health/metrics, API keys,
rate limiting, request queues, CORS, and log format. Before remote exposure:

- bind through an authenticated deployment boundary;
- set `HF_SERVER_API_KEY`/admin credentials where appropriate;
- restrict CORS origins instead of keeping the default wildcard;
- set model, queue, memory, and request limits; and
- separate metrics access from public inference access.

The default development configuration is not a production security policy.

## Tests

The maintained contract coverage includes:

```bash
python -m pytest test/test_hf_model_server_endpoint_contract.py -q
python -m pytest test/api/test_serving_readiness_contracts.py -q
```

Hardware, model-download, and network tests require their own optional
dependencies and should be run separately.

## Related documentation

- [Installation](../../guides/getting-started/installation.md)
- [Hardware guide](../../guides/hardware/overview.md)
- [Deployment](../../guides/deployment/README.md)
- [API overview](../../api/overview.md)
- [Current documentation state](../../development/DOCUMENTATION_CURRENT_STATE.md)
