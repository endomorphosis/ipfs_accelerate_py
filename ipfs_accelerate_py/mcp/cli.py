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
import subprocess
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp_cli")


def _default_p2p_queue_path() -> str:
    env_path = os.environ.get("IPFS_ACCELERATE_PY_TASK_QUEUE_PATH") or os.environ.get(
        "IPFS_DATASETS_PY_TASK_QUEUE_PATH"
    )
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

    # Default behavior: run MCP + TaskQueue worker/service.
    # Provide --no-* switches to disable for pure-HTTP usage.
    parser.add_argument(
        "--p2p-task-worker",
        dest="p2p_task_worker",
        action="store_true",
        help="Start a DuckDB task worker alongside MCP (default: on)",
    )
    parser.add_argument(
        "--no-p2p-task-worker",
        dest="p2p_task_worker",
        action="store_false",
        help="Do not start the task worker",
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
        "--p2p-autoscale",
        dest="p2p_autoscale",
        action="store_true",
        help="Autoscale workers based on backlog (default: on)",
    )
    parser.add_argument(
        "--no-p2p-autoscale",
        dest="p2p_autoscale",
        action="store_false",
        help="Disable autoscaling",
    )
    parser.add_argument(
        "--p2p-autoscale-min",
        type=int,
        default=int(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MIN", "1")),
        help="Minimum autoscaled worker count (default: 1)",
    )
    parser.add_argument(
        "--p2p-autoscale-max",
        type=int,
        default=int(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MAX", "4")),
        help="Maximum autoscaled worker count (default: 4)",
    )
    parser.add_argument(
        "--p2p-autoscale-idle-s",
        type=float,
        default=float(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_IDLE_S", "30")),
        help="Seconds idle before scaling down (default: 30)",
    )
    parser.add_argument(
        "--p2p-autoscale-poll-s",
        type=float,
        default=float(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_POLL_S", "2")),
        help="Autoscaler polling interval seconds (default: 2)",
    )
    parser.add_argument(
        "--p2p-autoscale-mesh-children",
        dest="p2p_autoscale_mesh_children",
        action="store_true",
        help="Enable mesh peer-claiming for autoscaled child workers (default: on)",
    )
    parser.add_argument(
        "--no-p2p-autoscale-mesh-children",
        dest="p2p_autoscale_mesh_children",
        action="store_false",
        help="Disable mesh peer-claiming for autoscaled child workers",
    )

    parser.add_argument(
        "--p2p-autoscale-remote",
        dest="p2p_autoscale_remote",
        action="store_true",
        help="Scale workers based on remote peer backlog too (default: on)",
    )
    parser.add_argument(
        "--no-p2p-autoscale-remote",
        dest="p2p_autoscale_remote",
        action="store_false",
        help="Do not scale workers based on remote backlog",
    )
    parser.add_argument(
        "--p2p-autoscale-remote-refresh-s",
        type=float,
        default=float(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE_REFRESH_S", "5")),
        help="Remote backlog poll interval seconds (default: 5)",
    )
    parser.add_argument(
        "--p2p-autoscale-remote-max-peers",
        type=int,
        default=int(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_REMOTE_MAX_PEERS", "10")),
        help="Max peers to poll for remote backlog (default: 10)",
    )
    parser.add_argument(
        "--p2p-service",
        dest="p2p_service",
        action="store_true",
        help="Start the libp2p TaskQueue RPC service (default: on)",
    )
    parser.add_argument(
        "--no-p2p-service",
        dest="p2p_service",
        action="store_false",
        help="Do not start the libp2p TaskQueue RPC service",
    )
    parser.add_argument(
        "--p2p-listen-port",
        type=int,
        default=None,
        help="TCP port for the libp2p TaskQueue service (default: env IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT or 9710)",
    )

    parser.add_argument(
        "--p2p-enable-tools",
        dest="p2p_enable_tools",
        action="store_true",
        help=(
            "Enable remote op=call_tool on the TaskQueue p2p service "
            "(sets IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=1)"
        ),
    )

    parser.set_defaults(
        p2p_task_worker=True,
        p2p_service=True,
        p2p_autoscale=True,
        p2p_autoscale_remote=True,
        p2p_autoscale_mesh_children=True,
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
        env_mcp_p2p_service = os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE")
        env_mcp_p2p_service_enabled = str(env_mcp_p2p_service or "").strip().lower() in {"1", "true", "yes", "on"}
        if args.p2p_service or env_mcp_p2p_service_enabled:
            args.p2p_listen_port = int(args.mcp_p2p_port)

    # Allow env to override behavior without changing args.
    if not args.p2p_enable_tools:
        env_enable_tools = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS")
        if str(env_enable_tools or "").strip().lower() in {"1", "true", "yes", "on"}:
            args.p2p_enable_tools = True

    # Environment may still override these defaults if explicitly set.
    if os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE") is not None:
        args.p2p_service = str(os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_TASK_WORKER") is not None:
        args.p2p_task_worker = str(os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_TASK_WORKER") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if args.p2p_enable_tools:
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS"] = "1"

    if args.p2p_service and args.p2p_listen_port:
        os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT", str(int(args.p2p_listen_port)))
        os.environ.setdefault("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", str(int(args.p2p_listen_port)))

    if args.p2p_service and getattr(args, "mcp_p2p_port", None):
        # Make p2p_tasks.worker default to the same port when spawned.
        os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT", str(int(args.mcp_p2p_port)))
        os.environ.setdefault("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", str(int(args.mcp_p2p_port)))

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

    # P2P task components
    from ipfs_accelerate_py.p2p_tasks.orchestrator import start_orchestrator_in_background
    from ipfs_accelerate_py.p2p_tasks.runtime import TaskQueueP2PServiceRuntime

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

        queue_path = os.path.expanduser(str(args.p2p_queue))
        os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_QUEUE_PATH", queue_path)
        os.environ.setdefault("IPFS_DATASETS_PY_TASK_QUEUE_PATH", queue_path)

        # Start the TaskQueue p2p service in-process (the MCP process owns the mesh).
        rt: TaskQueueP2PServiceRuntime | None = None
        if args.p2p_service:
            rt = TaskQueueP2PServiceRuntime()
            rt.start(queue_path=queue_path, listen_port=args.p2p_listen_port, accelerate_instance=accelerate)

            def _stop_p2p_service() -> None:
                try:
                    rt.stop()  # type: ignore[union-attr]
                except Exception:
                    pass

            atexit.register(_stop_p2p_service)
            logger.info(
                "Started ipfs_accelerate_py TaskQueue p2p service "
                f"(queue={queue_path}, listen_port={args.p2p_listen_port or 'env/default'})"
            )

        # If enabled, start orchestrator loop that spawns thin workers.
        orchestrator = None
        if args.p2p_task_worker:
            os.environ.setdefault("IPFS_ACCELERATE_PY_MCP_P2P_TASK_WORKER", "1")
            os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER", "1")

            # Base worker id used as a prefix by the orchestrator for spawned workers.
            os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ID", str(args.p2p_worker_id))
            os.environ.setdefault("IPFS_DATASETS_PY_TASK_WORKER_ID", str(args.p2p_worker_id))

            # Plumb MCP flags into env so orchestrator can consume them.
            if args.p2p_autoscale_min is not None:
                os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MIN"] = str(int(args.p2p_autoscale_min))
            if args.p2p_autoscale_max is not None:
                os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MAX"] = str(int(args.p2p_autoscale_max))
            if args.p2p_autoscale_poll_s is not None:
                os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_POLL_S"] = str(float(args.p2p_autoscale_poll_s))
            if args.p2p_autoscale_idle_s is not None:
                os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_IDLE_S"] = str(float(args.p2p_autoscale_idle_s))

            # If autoscale is explicitly disabled, clamp min=max=1.
            if not bool(args.p2p_autoscale):
                os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MIN"] = "1"
                os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MAX"] = "1"

            # Mesh draining is owned by orchestrator; allow disabling by setting max_peers=0.
            if not bool(args.p2p_autoscale_remote):
                os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_MAX_PEERS", "0")

            orchestrator = start_orchestrator_in_background(
                queue_path=queue_path,
                accelerate_instance=accelerate,
                supported_task_types=None,
            )

            def _stop_orchestrator() -> None:
                try:
                    if orchestrator is not None:
                        orchestrator.stop(timeout_s=2.0)
                except Exception:
                    pass

            atexit.register(_stop_orchestrator)
            logger.info("Started task orchestrator (spawns thin workers as needed)")

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
