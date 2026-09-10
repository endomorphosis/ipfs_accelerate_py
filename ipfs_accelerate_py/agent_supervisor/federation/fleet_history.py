"""Bounded native Quack reads and content-addressed DuckLake history writes."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .fleet_observation import canonical, observation_cid, validate_observation

VIEW_SCHEMA = "ipfs_accelerate_py/agent-supervisor/fleet-aggregate-view@1"


def read_native_view(deployment_path: Path, inventory_path: Path) -> dict[str, Any]:
    deployment = json.loads(deployment_path.read_text())
    if deployment.get("schema") != "ipfs_accelerate_py/agent-supervisor/quack-fleet-topology@1":
        raise ValueError("compiled native fleet deployment required")
    script = Path(deployment["instances"]["aggregate_control"]["start_argv"][1]).with_name("quack_fleet_aggregate.py")
    with tempfile.TemporaryFile() as output:
        subprocess.run([sys.executable, "-P", str(script), "--query", "--deployment", str(deployment_path),
                        "--inventory", str(inventory_path)], stdout=output, stderr=subprocess.DEVNULL, timeout=45, check=True)
        if output.tell() > 16 * 1024 * 1024:
            raise ValueError("native fleet view exceeds export limit")
        output.seek(0)
        return json.load(output)


def project_view(connection: Any, view: dict[str, Any]) -> int:
    """Single-writer replay by observation CID; historical rows grant no authority."""
    if view.get("schema") != VIEW_SCHEMA or view.get("completion_authority") is not False:
        raise ValueError("native observational fleet view required")
    sources = view.get("sources")
    if not isinstance(sources, dict) or len(sources) > 4096:
        raise ValueError("bounded source inventory required")
    rows = []
    for source_id, item in sources.items():
        current = item.get("current")
        if current is None:
            continue
        current = validate_observation(current)
        if current["source_id"] != source_id:
            raise ValueError("source observation shape differs")
        payload = canonical(current)
        cid = observation_cid(current)
        rows.append((cid, source_id, current["observed_at"], current["availability"], payload))
    connection.execute("CREATE TABLE IF NOT EXISTS fleet_lake.fleet_source_observations (observation_cid VARCHAR, source_id VARCHAR, observed_at VARCHAR, availability VARCHAR, payload_json VARCHAR, completion_authority BOOLEAN)")
    connection.execute("BEGIN TRANSACTION")
    try:
        for row in rows:
            connection.execute("INSERT INTO fleet_lake.fleet_source_observations SELECT ?, ?, ?, ?, ?, FALSE WHERE NOT EXISTS (SELECT 1 FROM fleet_lake.fleet_source_observations WHERE observation_cid = ?)", [*row, row[0]])
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return len(rows)

