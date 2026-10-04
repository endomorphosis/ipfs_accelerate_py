# Docker Deployment (API, Dashboard, Reverse Proxy)

This stack runs the MCP API, the Performance Dashboard, and a reverse proxy (Caddy) with optional HTTPS.

## Prerequisites
- Docker and Docker Compose v2
- Optional: a DNS A record pointing to this host if you want automatic HTTPS

## Services
- `api` (port 8000): IPFS Accelerate MCP API (binds inside container; published to host 8000)
- `dashboard` (port 8080): Performance Dashboard (published to host 8080)
- `reverse-proxy` (ports 80/443): Routes requests to API and Dashboard
  - `/mcp/*`, `/docs`, `/openapi.json`, `/redoc*` → `api:8000`
  - All other paths → `dashboard:8080`

## Environment
- `SITE_DOMAIN` (default `localhost`): Domain Caddy will serve
- `MCP_CORS_ORIGINS` (default `*`): Comma-separated allowed origins for browser clients
- `FULL_IPFS` (default `0`): Set `1` to enable full IPFS (libp2p, etc.) in API container

## Start
```bash
cd deployments/docker
# Optional: set domain and CORS
export SITE_DOMAIN=your.domain.com
export MCP_CORS_ORIGINS="https://yourapp.example.com"
# Optional: enable full IPFS inside the API container
export FULL_IPFS=1

# Build and run
docker compose up -d --build
```

## Verify
- Reverse proxy (domain):
  - `https://your.domain.com/mcp/resource/ipfs_accelerate/supported_models`
  - `https://your.domain.com/docs`
  - `https://your.domain.com/` (dashboard)
- Direct (host ports):
  - `http://HOST_IP:8000/docs`
  - `http://HOST_IP:8080`

Quick checks:
```bash
# API (resource)
curl -s http://127.0.0.1:8000/mcp/resource/ipfs_accelerate/supported_models | head
# API (tool)
curl -s -X POST http://127.0.0.1:8000/mcp/tool/get_hardware_info -H 'Content-Type: application/json' -d '{}'
# Dashboard
curl -s http://127.0.0.1:8080/api/status
```

## Stop
```bash
docker compose down
```

## Notes
- The API image installs `libgmp-dev` to support libp2p crypto dependencies when `FULL_IPFS=1`.
- For public HTTPS, ensure `SITE_DOMAIN` resolves to this host and ports 80/443 are reachable.