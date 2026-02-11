# Two-Laptop Runbook: P2P GitHub API Cache

This runbook shows how to validate that the GitHub API cache can be shared **between two different machines** over libp2p, so you can avoid hammering the GitHub API.

It is designed to be copy/paste-friendly for a second laptop (or a second ChatGPT instance).

## What you’re testing

- Laptop A performs a GitHub API call via the GitHub CLI wrapper (or writes a synthetic entry).
- The cache entry is broadcast over libp2p.
- Laptop B receives the entry and can read it from its local cache (`peer_hits` increments).

## Prereqs (both laptops)

1. **Python + deps** (same venv on both machines recommended)

```bash
pip install -U pip
pip install "libp2p @ git+https://github.com/libp2p/py-libp2p@main" cryptography
```

Optional (only if you want content-hash validation features):

```bash
pip install py-multiformats-cid
```

2. **GitHub CLI authentication** (required for discovery and for real GitHub API calls)

```bash
gh auth status
```

3. **Open inbound ports**

You need inbound TCP access to the listen ports you choose (examples below use `9100` and `9101`). On Linux with `ufw`:

```bash
sudo ufw allow 9100/tcp
sudo ufw allow 9101/tcp
```

If you’re on different networks / behind strict NAT, inbound dials may fail without port forwarding.

## Key environment variables

Set these consistently across machines:

- `GITHUB_REPOSITORY` — used for cross-host discovery via GitHub (format `owner/repo`).
- `CACHE_P2P_SHARED_SECRET` — recommended: a shared secret so peers can decrypt messages even if they use different GitHub tokens.

Set these per-machine:

- `CACHE_LISTEN_PORT` — pick a unique port per laptop.
- `RUNNER_NAME` — **must be unique** per machine/process when using the local file registry fallback.

Useful knobs:

- `IPFS_ACCELERATE_P2P_FORCE_LOCALHOST=1` — forces loopback advertisement for *same-host* multi-process tests.
- `IPFS_ACCELERATE_P2P_CACHE_DIR` — enables local file-based peer registry (best for local dev / CI, not two separate laptops unless the dir is shared).

## Recommended: two-laptop test using GitHub discovery (real or synthetic)

This uses [tools/github_p2p_cache_smoke.py](tools/github_p2p_cache_smoke.py).

### 0) Pick a discovery repo

Choose a repo both laptops can access (public is fine; private works if both tokens have access). Example:

```bash
export GITHUB_REPOSITORY=owner/repo
```

### 1) Set a shared secret (both laptops)

```bash
export CACHE_P2P_SHARED_SECRET='replace-with-a-random-shared-secret'
```

### 2) Start the reader (Laptop B)

```bash
export CACHE_LISTEN_PORT=9101
python tools/github_p2p_cache_smoke.py --read --target octocat/Hello-World --wait-seconds 120 --verbose
```

Keep this running while you run the writer.

### 3) Run the writer (Laptop A)

Real call (requires `gh` auth):

```bash
export CACHE_LISTEN_PORT=9100
python tools/github_p2p_cache_smoke.py --write --target octocat/Hello-World --verbose
```

Offline/synthetic (no GitHub API call):

```bash
export CACHE_LISTEN_PORT=9100
python tools/github_p2p_cache_smoke.py --write --synthetic --target octocat/Hello-World --verbose
```

### 4) Expected results

On Laptop B (reader):

- It prints `Received cache entry!`.
- `peer_hits` increments in `Final stats`.

On Laptop A (writer):

- `connected_peers` should become `>= 1`.
- If it stays at 0, you’re likely blocked by NAT/firewall or discovery isn’t finding peers.

## Multi-laptop (3+ machines) quickstart

- All machines share the same `GITHUB_REPOSITORY` and `CACHE_P2P_SHARED_SECRET`.
- Each machine uses a unique `CACHE_LISTEN_PORT`.
- Run a reader on each machine, then run a writer on any one machine.

Example port assignment:

- Laptop A: `CACHE_LISTEN_PORT=9100`
- Laptop B: `CACHE_LISTEN_PORT=9101`
- Laptop C: `CACHE_LISTEN_PORT=9102`

## Local-only dev mode (single machine, two processes)

If you want to reproduce the setup on a single machine (two terminals), use the local file registry fallback.

Terminal 1 (reader):

```bash
export IPFS_ACCELERATE_P2P_CACHE_DIR=/tmp/ipfs_accel_p2p_peers
export RUNNER_NAME=local-reader
export CACHE_P2P_SHARED_SECRET='dev-secret'
export CACHE_LISTEN_PORT=9101
export IPFS_ACCELERATE_P2P_FORCE_LOCALHOST=1

python tools/github_p2p_cache_smoke.py --read --synthetic --target octocat/Hello-World --wait-seconds 90
```

Terminal 2 (writer):

```bash
export IPFS_ACCELERATE_P2P_CACHE_DIR=/tmp/ipfs_accel_p2p_peers
export RUNNER_NAME=local-writer
export CACHE_P2P_SHARED_SECRET='dev-secret'
export CACHE_LISTEN_PORT=9100
export IPFS_ACCELERATE_P2P_FORCE_LOCALHOST=1

python tools/github_p2p_cache_smoke.py --write --synthetic --target octocat/Hello-World
```

Notes:

- `RUNNER_NAME` must differ or peers overwrite the same `peer_<runner>.json` file.
- `IPFS_ACCELERATE_P2P_FORCE_LOCALHOST=1` avoids “dialing our public IP” on the same host.

## Troubleshooting

### Reader times out; writer shows `connected_peers: 0`

- Verify inbound firewall rules allow the listen ports.
- Verify both machines are on the same network (for easiest testing).
- Try the connectivity probe test:
  - Server: run [test/integration/test_p2p_remote_host_connectivity.py](test/integration/test_p2p_remote_host_connectivity.py) with `--serve` on Laptop A.
  - Client: run the pytest `test_remote_host_probe` on Laptop B using `P2P_REMOTE_MULTIADDR=...`.

### Writer logs “broadcast may be a no-op”

That means discovery hasn’t connected yet. Keep the reader running and re-run the writer once `connected_peers` becomes `>= 1`.

### Different GitHub tokens can’t decrypt entries

Set the same `CACHE_P2P_SHARED_SECRET` on all machines.
