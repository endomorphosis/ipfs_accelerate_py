#!/usr/bin/env python3
"""
IPFS Accelerate MCP Server CLI

This script provides a command-line interface for starting and managing
the IPFS Accelerate MCP server.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp_cli")


def _default_p2p_queue_path() -> str:
    env_path = os.environ.get("IPFS_ACCELERATE_PY_TASK_QUEUE_PATH") or os.environ.get("IPFS_DATASETS_PY_TASK_QUEUE_PATH")
    if env_path and str(env_path).strip():
        return str(env_path).strip()

    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root and str(cache_root).strip():
        return os.path.join(str(cache_root).strip(), "ipfs_datasets_py", "task_queue.duckdb")

    return "~/.cache/ipfs_datasets_py/task_queue.duckdb"


def _default_task_p2p_announce_file() -> str:
    """Prefer a repo-local announce file when running under MCP/systemd.

    Systemd units often use ProtectHome=read-only, which makes the default
    ~/.cache-based announce file unwritable and can leave stale data around.
    """

    try:
        repo_root = Path(__file__).resolve().parents[2]
        return str(repo_root / "state" / "p2p" / "task_p2p_announce.json")
    except Exception:
        return str(Path(os.getcwd()) / "state" / "p2p" / "task_p2p_announce.json")

def main():
    """Main entry point for the MCP server CLI."""
    parser = argparse.ArgumentParser(
        description="IPFS Accelerate Model Context Protocol (MCP) Server"
    )
    
    # Define command-line arguments
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host address to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=9000, 
        help="Port to listen on (default: 9000)"
    )

    parser.add_argument(
        "--mcp-p2p-port",
        type=int,
        default=int(os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_PORT", "9100")),
        help="libp2p port for MCP-adjacent P2P services (default: 9100)",
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--dev", 
        action="store_true", 
        help="Run in development mode with auto-reload"
    )

    # Optional: also run ipfs_datasets_py P2P task worker/service in-process.
    # This enables a remote machine running the MCP server to pick up libp2p
    # task submissions from other nodes.
    parser.add_argument(
        "--p2p-task-worker",
        action="store_true",
        help="Also start accelerate-owned DuckDB task worker (+ optional libp2p TaskQueue service) in a background thread",
    )
    parser.add_argument(
        "--p2p-queue",
        default=_default_p2p_queue_path(),
        help="DuckDB task queue path for P2P tasks (default: ~/.cache/ipfs_datasets_py/task_queue.duckdb)",
    )
    parser.add_argument(
        "--p2p-worker-id",
        default="accelerate-mcp-worker",
        help="Worker id used when claiming tasks (default: accelerate-mcp-worker)",
    )
    parser.add_argument(
        "--p2p-service",
        action="store_true",
        help=(
            "Start the libp2p TaskQueue RPC service (writes an announce file for zero-config client auto-discovery). "
            "If used with --p2p-task-worker, the worker will host the service; otherwise the service runs standalone."
        ),
    )
    parser.add_argument(
        "--p2p-listen-port",
        type=int,
        default=None,
        help="TCP port for the libp2p TaskQueue service (default: env IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT or 9710)",
    )

    parser.add_argument(
        "--p2p-enable-tools",
        action="store_true",
        help="Enable remote op=call_tool on the TaskQueue p2p service (sets IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=1)",
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Export the canonical MCP-adjacent libp2p port so downstream libraries
    # (TaskQueue client discovery, mDNS, etc.) can reliably operate in
    # single-port MCP mode.
    if getattr(args, "mcp_p2p_port", None):
        try:
            os.environ.setdefault(
                "IPFS_ACCELERATE_PY_MCP_P2P_PORT",
                str(int(args.mcp_p2p_port)),
            )
        except Exception:
            pass

    # When running under MCP/systemd, standardize on a single "MCP p2p port"
    # unless the caller explicitly set a TaskQueue listen port.
    if args.p2p_listen_port is None and getattr(args, "mcp_p2p_port", None):
        # Only apply this when we will actually start the p2p TaskQueue service.
        # (Otherwise leave defaults untouched for pure-HTTP usage.)
        if args.p2p_service or str(os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE") or "").strip().lower() in {"1", "true", "yes", "on"}:
            args.p2p_listen_port = int(args.mcp_p2p_port)

    # Allow systemd to toggle p2p features via env without changing unit args.
    if not args.p2p_service:
        env_p2p_service = os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE")
        if str(env_p2p_service or "").strip().lower() in {"1", "true", "yes", "on"}:
            args.p2p_service = True
    if not args.p2p_task_worker:
        env_p2p_worker = os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_TASK_WORKER")
        if str(env_p2p_worker or "").strip().lower() in {"1", "true", "yes", "on"}:
            args.p2p_task_worker = True
    if not args.p2p_enable_tools:
        env_enable_tools = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS")
        if str(env_enable_tools or "").strip().lower() in {"1", "true", "yes", "on"}:
            args.p2p_enable_tools = True

    if args.p2p_enable_tools:
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"

    if args.p2p_service and args.p2p_listen_port:
        os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT", str(int(args.p2p_listen_port)))
        os.environ.setdefault("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", str(int(args.p2p_listen_port)))

    # Ensure the announce file is writable and consistent across MCP/systemd.
    # Without this, clients may read stale ~/.cache announce data while the
    # service can't update it.
    if args.p2p_service:
        announce_path = _default_task_p2p_announce_file()
        os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE", announce_path)
        os.environ.setdefault("IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE", announce_path)

    # Cache integration: when hosting the TaskQueue p2p service, prefer to share
    # GitHub cache entries via the TaskQueue cache.get/set RPC (single libp2p
    # port) rather than starting a second libp2p host in the GitHub cache.
    if args.p2p_service:
        # Enable discovery-based task-p2p cache usage for the GitHub API cache.
        os.environ.setdefault("IPFS_ACCELERATE_PY_GITHUB_CACHE_TASK_P2P_DISCOVERY", "1")

        # Avoid port conflicts when the GitHub cache would otherwise try to
        # listen on the same port.
        os.environ["CACHE_ENABLE_P2P"] = "false"
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Import MCP components (import here to avoid circular imports)
    from ipfs_accelerate_py import ipfs_accelerate_py
    from ipfs_accelerate_py.mcp.server import create_mcp_server
    
    try:
        # Create IPFS Accelerate instance
        logger.info("Initializing IPFS Accelerate...")
        accelerate = ipfs_accelerate_py()

        # Proactively initialize the GitHub API cache so its P2P subsystem can
        # start (best-effort). This keeps cache sharing active even if no MCP
        # tool calls it immediately.
        try:
            from ipfs_accelerate_py.github_cli.cache import get_global_cache

            _cache = get_global_cache()
            _ = getattr(_cache, "get_stats", lambda: {})()
            logger.info("Initialized GitHub API cache (best-effort)")
        except Exception as exc:
            logger.debug(f"GitHub API cache init skipped: {exc}")

        if args.p2p_task_worker:
            import threading

            # Mark the local worker as enabled for TaskQueue status reporting.
            # The TaskQueue service uses env vars to describe in-process worker
            # configuration in `status(detail=True)`.
            os.environ.setdefault("IPFS_ACCELERATE_PY_MCP_P2P_TASK_WORKER", "1")
            os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER", "1")

            queue_path = os.path.expanduser(str(args.p2p_queue))

            def _run_p2p_worker() -> None:
                try:
                    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

                    run_worker(
                        queue_path=queue_path,
                        worker_id=str(args.p2p_worker_id),
                        poll_interval_s=0.25,
                        once=False,
                        p2p_service=bool(args.p2p_service),
                        p2p_listen_port=args.p2p_listen_port,
                        accelerate_instance=accelerate,
                    )
                except Exception as exc:
                    logger.error(f"Failed to start ipfs_accelerate_py P2P task worker: {exc}")

            t = threading.Thread(
                target=_run_p2p_worker,
                name="ipfs_accelerate_py_p2p_task_worker",
                daemon=True,
            )
            t.start()
            logger.info(
                "Started ipfs_accelerate_py task worker thread "
                f"(queue={queue_path}, worker_id={args.p2p_worker_id}, p2p_service={bool(args.p2p_service)})"
            )

        # If the user asked for the p2p service but did not start the worker,
        # run the TaskQueue libp2p RPC service standalone.
        if (not args.p2p_task_worker) and args.p2p_service:
            import threading

            queue_path = os.path.expanduser(str(args.p2p_queue))

            def _run_p2p_service_only() -> None:
                try:
                    from ipfs_accelerate_py.p2p_tasks.runtime import TaskQueueP2PServiceRuntime

                    rt = TaskQueueP2PServiceRuntime()
                    rt.start(queue_path=queue_path, listen_port=args.p2p_listen_port, accelerate_instance=accelerate)
                    # Keep thread alive as long as the process is alive.
                    while True:
                        import time

                        time.sleep(3600)
                except Exception as exc:
                    logger.error(f"Failed to start standalone ipfs_accelerate_py TaskQueue p2p service: {exc}")

            t2 = threading.Thread(
                target=_run_p2p_service_only,
                name="ipfs_accelerate_py_p2p_task_service",
                daemon=True,
            )
            t2.start()
            logger.info(
                "Started ipfs_accelerate_py TaskQueue p2p service thread "
                f"(queue={queue_path}, listen_port={args.p2p_listen_port or 'env/default'})"
            )
        
        # Create MCP server
        logger.info("Creating MCP server...")
        mcp_server = create_mcp_server(accelerate_instance=accelerate)
        
        # Start the server
        logger.info(f"Starting MCP server on {args.host}:{args.port}...")
        if args.dev:
            logger.info("Running in development mode with auto-reload enabled")
            mcp_server.run(host=args.host, port=args.port, reload=True)
        else:
            mcp_server.run(host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Error starting MCP server: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
