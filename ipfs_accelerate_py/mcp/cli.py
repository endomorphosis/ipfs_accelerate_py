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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp_cli")

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
        default=os.environ.get(
            "IPFS_ACCELERATE_PY_TASK_QUEUE_PATH",
            os.environ.get("IPFS_DATASETS_PY_TASK_QUEUE_PATH", "~/.cache/ipfs_datasets_py/task_queue.duckdb"),
        ),
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
            "When used with --p2p-task-worker, also start the libp2p TaskQueue RPC service "
            "(writes an announce file for zero-config client auto-discovery)"
        ),
    )
    parser.add_argument(
        "--p2p-listen-port",
        type=int,
        default=None,
        help="TCP port for the libp2p TaskQueue service (default: env IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT or 9710)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Import MCP components (import here to avoid circular imports)
    from ipfs_accelerate_py import ipfs_accelerate_py
    from ipfs_accelerate_py.mcp.server import create_mcp_server
    
    try:
        # Create IPFS Accelerate instance
        logger.info("Initializing IPFS Accelerate...")
        accelerate = ipfs_accelerate_py()

        if args.p2p_task_worker:
            import threading

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
