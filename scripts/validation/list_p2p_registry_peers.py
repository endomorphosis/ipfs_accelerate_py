#!/usr/bin/env python3
"""List active P2P peers from the GitHub Issue-backed registry.

This is the quickest way to validate that another node is online and
heartbeating. It requires GitHub CLI auth:
- Prefer setting GH_TOKEN in the environment (or via systemd EnvironmentFile)
- Or run: gh auth login

Usage:
  python scripts/validation/list_p2p_registry_peers.py
  IPFS_ACCELERATE_GITHUB_REPO=owner/repo GH_TOKEN=... python scripts/validation/list_p2p_registry_peers.py

Exit codes:
  0: command ran successfully (peers may still be 0)
  2: registry could not be queried (auth/repo misconfig)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List active P2P registry peers")
    parser.add_argument(
        "--repo",
        default=os.environ.get("IPFS_ACCELERATE_GITHUB_REPO") or "endomorphosis/ipfs_accelerate_py",
        help="GitHub repo in owner/repo format (default: env IPFS_ACCELERATE_GITHUB_REPO or endomorphosis/ipfs_accelerate_py)",
    )
    parser.add_argument("--max", type=int, default=50, help="Max peers to return")
    args = parser.parse_args(argv)

    repo = (args.repo or "").strip()
    if not repo or "/" not in repo:
        print("ERROR: --repo must be in owner/repo format", file=sys.stderr)
        return 2

    try:
        from ipfs_accelerate_py.github_cli.p2p_peer_registry import P2PPeerRegistry

        registry = P2PPeerRegistry(repo=repo)
        peers = registry.discover_peers(max_peers=int(args.max))
    except Exception as e:
        print(f"ERROR: failed to query peer registry: {e}", file=sys.stderr)
        print("Hint: set GH_TOKEN (preferred) or run 'gh auth login'", file=sys.stderr)
        return 2

    now = _now_utc()

    # Render a compact table-ish view plus JSON.
    print(f"Repo: {repo}")
    print(f"Active peers: {len(peers)}")
    for peer in peers:
        runner = peer.get("runner_name") or "(unknown)"
        peer_id = peer.get("peer_id") or ""
        multiaddr = peer.get("multiaddr") or ""
        last_seen_s = peer.get("last_seen") or ""

        age_s = None
        try:
            # registry uses datetime.utcnow().isoformat() (no tz)
            last_seen = datetime.fromisoformat(last_seen_s)
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            age_s = max(0.0, (now - last_seen).total_seconds())
        except Exception:
            pass

        age_str = "?" if age_s is None else f"{int(age_s)}s"
        peer_short = (peer_id[:16] + "…") if peer_id else "(missing)"
        print(f"- {runner}: {peer_short} last_seen_age={age_str} addr={multiaddr}")

    print("\nJSON:")
    print(json.dumps(peers, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
