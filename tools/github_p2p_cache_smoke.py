#!/usr/bin/env python3

"""P2P GitHub API cache smoke test.

Run this on two different machines to validate:
- public rendezvous/discovery via GitHub issue registry (GITHUB_REPOSITORY)
- libp2p connectivity (best-effort; requires reachability)
- cache entry broadcast/ingestion (peer_hits increments on receiver)

Typical usage:
  # On both machines (same values):
  export GITHUB_REPOSITORY=owner/repo
  export CACHE_P2P_SHARED_SECRET='some-shared-secret'

  # Machine A (writer)
  python tools/github_p2p_cache_smoke.py --write --target octocat/Hello-World

  # Machine B (reader)
  python tools/github_p2p_cache_smoke.py --read --target octocat/Hello-World --wait-seconds 90

Notes:
- For real "public internet" dialing, the listening node must be reachable.
  If you're behind NAT/firewall without port-forwarding, inbound dials may fail.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
import tempfile
import subprocess
from typing import Any, Optional


def _ensure_repo_on_syspath() -> None:
    """Allow running this script without `pip install -e .`.

    When executed from outside the repo root (e.g., `/tmp`), Python won't be
    able to import the local `ipfs_accelerate_py` package unless the repo root
    is on `sys.path`.
    """

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _loopback_multiaddr(*, port: int, peer_id: str) -> str:
    return f"/ip4/127.0.0.1/tcp/{port}/p2p/{peer_id}"


def _run_local_two_node_demo(args: argparse.Namespace) -> int:
    """Run a single-command local demo by spawning a writer subprocess.

    This avoids needing two terminals / two machines for a basic end-to-end test.
    It uses a synthetic cache entry (no GitHub API access needed).
    """

    _ensure_repo_on_syspath()

    if not args.synthetic:
        print("NOTE: --local-two-node forces --synthetic to avoid GitHub API calls.")
        args.synthetic = True

    if args.cache_dir:
        print("NOTE: --local-two-node ignores --cache-dir and uses temp dirs.")

    base_dir = Path(tempfile.mkdtemp(prefix="ipfs_accel_p2p_smoke_"))
    p2p_registry_dir = base_dir / "p2p_peers"
    reader_cache_dir = base_dir / "reader_cache"
    writer_cache_dir = base_dir / "writer_cache"
    p2p_registry_dir.mkdir(parents=True, exist_ok=True)
    reader_cache_dir.mkdir(parents=True, exist_ok=True)
    writer_cache_dir.mkdir(parents=True, exist_ok=True)

    reader_port = int(args.listen_port)
    writer_port = reader_port + 1

    if args.shared_secret:
        os.environ["CACHE_P2P_SHARED_SECRET"] = args.shared_secret

    # Force local file-based peer discovery by ensuring no repo is set.
    os.environ["GITHUB_REPOSITORY"] = ""
    os.environ["IPFS_ACCELERATE_P2P_CACHE_DIR"] = str(p2p_registry_dir)

    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

    _print_banner(label="Local demo: reader starting")
    cache = GitHubAPICache(
        cache_dir=str(reader_cache_dir),
        enable_persistence=True,
        enable_p2p=True,
        p2p_listen_port=reader_port,
        github_repo=None,
        enable_peer_discovery=True,
    )

    try:
        stats = cache.get_stats()
        print("Reader initial stats:\n" + _safe_json(stats))
        peer_id = str(stats.get("peer_id") or "")
        if not peer_id:
            print("ERROR: reader did not report a peer_id; cannot run local demo", file=sys.stderr)
            return 2

        reader_bootstrap = _loopback_multiaddr(port=reader_port, peer_id=peer_id)
        print("Reader bootstrap addr (loopback):\n" + _safe_json([reader_bootstrap]))

        # Start writer subprocess, pointing it at the reader via CACHE_BOOTSTRAP_PEERS.
        writer_env = dict(os.environ)
        writer_env["GITHUB_REPOSITORY"] = ""
        writer_env["IPFS_ACCELERATE_P2P_CACHE_DIR"] = str(p2p_registry_dir)
        writer_env["IPFS_ACCELERATE_CACHE_DIR"] = str(writer_cache_dir)
        writer_env["CACHE_BOOTSTRAP_PEERS"] = reader_bootstrap
        if args.shared_secret:
            writer_env["CACHE_P2P_SHARED_SECRET"] = args.shared_secret

        writer_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--write",
            "--synthetic",
            "--target",
            args.target,
            "--listen-port",
            str(writer_port),
            "--cache-dir",
            str(writer_cache_dir),
            "--registry-repo",
            "",
        ]
        if args.verbose:
            writer_cmd.append("--verbose")

        _print_banner(label="Local demo: spawning writer")
        print("Writer command:\n" + _safe_json(writer_cmd))
        writer = subprocess.Popen(
            writer_cmd,
            env=writer_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.time() + float(args.wait_seconds)
        while time.time() < deadline:
            value = cache.get("p2p_smoke", target=args.target)
            if value is not None:
                print("Received cache entry from writer!")
                print("Value:\n" + _safe_json(value))
                print("Final reader stats:\n" + _safe_json(cache.get_stats()))

                try:
                    out = writer.communicate(timeout=10)[0]
                    if out:
                        print("\n--- Writer output ---\n" + out)
                except Exception:
                    pass

                return 0

            # If the writer exits early, surface its output and fail fast.
            rc = writer.poll()
            if rc is not None:
                out = ""
                try:
                    out = writer.communicate(timeout=5)[0] or ""
                except Exception:
                    pass
                print(f"Writer exited early with code {rc}")
                if out:
                    print("\n--- Writer output ---\n" + out)
                print("Reader stats:\n" + _safe_json(cache.get_stats()))
                return 1

            time.sleep(float(args.poll_interval))

        print("Timed out waiting for cache entry (local demo).")
        print("Final reader stats:\n" + _safe_json(cache.get_stats()))
        try:
            out = writer.communicate(timeout=5)[0]
            if out:
                print("\n--- Writer output ---\n" + out)
        except Exception:
            try:
                writer.terminate()
            except Exception:
                pass
        return 1

    finally:
        try:
            cache.shutdown()
        except Exception:
            pass


def _configure_logging(verbose: bool) -> None:
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _print_banner(*, label: str) -> None:
    hostname = socket.gethostname()
    print(f"\n=== {label} ===")
    print(f"host={hostname} pid={os.getpid()}")


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True, default=str)
    except Exception:
        return str(obj)


def _wait_for_connected_peers(cache: Any, *, min_peers: int, timeout_s: float, poll_s: float = 0.25) -> bool:
    """Best-effort wait until the cache reports at least `min_peers` connected peers."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            stats = cache.get_stats()
            connected = int(stats.get("connected_peers", 0) or 0)
            if connected >= min_peers:
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P2P GitHub API cache smoke test")
    parser.add_argument(
        "--registry-repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help=(
            "GitHub repo used for discovery (owner/repo). Defaults to GITHUB_REPOSITORY. "
            "If omitted, the cache falls back to local file-based peer discovery (see IPFS_ACCELERATE_P2P_CACHE_DIR)."
        ),
    )
    parser.add_argument("--listen-port", type=int, default=int(os.environ.get("CACHE_LISTEN_PORT", "9100")))
    parser.add_argument("--cache-dir", default=os.environ.get("IPFS_ACCELERATE_CACHE_DIR"), help="Optional cache directory (defaults to ~/.cache/github_cli)")
    parser.add_argument("--shared-secret", default=os.environ.get("CACHE_P2P_SHARED_SECRET"), help="Optional shared secret for encryption; sets CACHE_P2P_SHARED_SECRET")
    parser.add_argument(
        "--disable-discovery",
        action="store_true",
        help="Disable peer discovery entirely (no GitHub issue registry and no local file-based discovery).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Perform a GitHub API call and broadcast the resulting cache entry")
    mode.add_argument("--read", action="store_true", help="Wait for a peer-broadcasted cache entry and verify it was received")

    parser.add_argument("--target", default="octocat/Hello-World", help="Target repo for get_repo_info (owner/repo)")
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic cache entry instead of calling gh (avoids GitHub API)")
    parser.add_argument(
        "--local-two-node",
        action="store_true",
        help=(
            "Run a one-command local demo by starting a reader in this process and spawning a synthetic writer subprocess. "
            "Uses loopback bootstrap (no GitHub API required)."
        ),
    )
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    _ensure_repo_on_syspath()

    if args.local_two_node:
        if args.read or args.write:
            # The mutually-exclusive group enforces one; we allow either but
            # local-two-node always behaves like a reader orchestrator.
            pass
        return _run_local_two_node_demo(args)

    if args.shared_secret:
        os.environ["CACHE_P2P_SHARED_SECRET"] = args.shared_secret

    enable_peer_discovery = not args.disable_discovery

    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache

    _print_banner(label="P2P cache node starting")

    cache = GitHubAPICache(
        cache_dir=args.cache_dir,
        enable_persistence=True,
        enable_p2p=True,
        p2p_listen_port=args.listen_port,
        github_repo=args.registry_repo,
        enable_peer_discovery=enable_peer_discovery,
    )

    try:
        stats = cache.get_stats()
        print("Initial stats:\n" + _safe_json(stats))

        # Private helper, but extremely useful for manual bootstrapping.
        try:
            advertised = cache._get_advertised_multiaddrs()  # type: ignore[attr-defined]
            print("Advertised multiaddrs:\n" + _safe_json(advertised))
        except Exception:
            pass

        if args.write:
            _print_banner(label="Writer")

            # The P2P runtime initializes in a background thread and does its
            # initial bootstrap connect asynchronously. For a smoke test, it's
            # useful to wait briefly for at least one peer connection before
            # writing/broadcasting.
            if enable_peer_discovery:
                connected = _wait_for_connected_peers(cache, min_peers=1, timeout_s=15.0)
                if not connected:
                    print("WARNING: No connected peers yet; broadcast may be a no-op.")

            if args.synthetic:
                payload = {
                    "kind": "synthetic",
                    "target": args.target,
                    "ts": time.time(),
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                }
                cache.put("p2p_smoke", payload, ttl=600, target=args.target)
                print("Wrote synthetic cache entry under operation=p2p_smoke")
            else:
                from ipfs_accelerate_py.github_cli.wrapper import GitHubCLI

                gh = GitHubCLI(enable_cache=True, cache=cache)
                info = gh.get_repo_info(args.target, use_cache=True)
                if not info:
                    print("ERROR: gh.get_repo_info returned no data", file=sys.stderr)
                    return 1
                print("Fetched repo info (and cached it).")

            # Give some time for discovery/dial and broadcast.
            time.sleep(5.0)
            print("Post-write stats:\n" + _safe_json(cache.get_stats()))
            print("Writer done.")
            return 0

        _print_banner(label="Reader")
        deadline = time.time() + args.wait_seconds
        last_peer_hits = cache.get_stats().get("peer_hits", 0)

        while time.time() < deadline:
            if args.synthetic:
                value = cache.get("p2p_smoke", target=args.target)
            else:
                value = cache.get("get_repo_info", repo=args.target)

            if value is not None:
                print("Received cache entry!")
                print("Value:\n" + _safe_json(value))
                print("Final stats:\n" + _safe_json(cache.get_stats()))
                return 0

            stats = cache.get_stats()
            peer_hits = stats.get("peer_hits", 0)
            if peer_hits != last_peer_hits:
                last_peer_hits = peer_hits
                print("peer_hits changed:\n" + _safe_json(stats))

            time.sleep(args.poll_interval)

        print("Timed out waiting for cache entry.")
        print("Final stats:\n" + _safe_json(cache.get_stats()))
        return 1

    finally:
        try:
            cache.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
