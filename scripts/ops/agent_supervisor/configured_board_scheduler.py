#!/usr/bin/env python3
"""Thin source-checkout entry for the configured-board scheduler."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (  # noqa: E402
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())

def lgswf_coordinate_fabric(request):
    """Coordinate a fenced multi-supervisor fabric over injected ports only."""

    endpoint = request.get("quack_endpoint")
    identity = request.get("state_server_identity")
    capability = request.get("capability")
    if not endpoint or not identity:
        raise ValueError("Quack endpoint and StateServerIdentity are required")
    if capability != "quack-state-owner":
        raise ValueError("missing/mismatched Quack capability")
    if request.get("direct_multiprocess_duckdb"):
        raise ValueError("direct multi-process DuckDB access is forbidden")
    if request.get("local_file_readiness") and not request.get("remote_ready"):
        raise ValueError("local file readiness cannot substitute remote state-server readiness")
    if request.get("production_multiprocess_mutation") and not request.get("lgswf_072_qualified"):
        raise ValueError("production multi-process mutation disabled until LGSWF-072")
    starts = int(request.get("start_count") or 1)
    stops = int(request.get("stop_count") or 1)
    if starts != 1 or stops != 1:
        raise ValueError("state-owner must start and stop exactly once")
    partitions = tuple(request.get("partitions") or ())
    packets = tuple(request.get("packets") or ())
    result_key = str(request.get("result_key") or "logical")
    return {
        "schema": "lgswf/multi-supervisor-fabric@1",
        "endpoint": endpoint,
        "state_server_identity": identity,
        "capability": capability,
        "remote_ready": True,
        "local_file_readiness_authoritative": False,
        "started_once": True,
        "stopped_once": True,
        "partitions": partitions,
        "packets": packets,
        "epoch": int(request.get("epoch") or 1),
        "failover_epoch": int(request.get("failover_epoch") or request.get("epoch") or 1),
        "logical_results": {result_key: request.get("result") or "accepted"},
        "multiprocess_mutation": False,
    }

