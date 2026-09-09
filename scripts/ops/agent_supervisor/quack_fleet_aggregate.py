#!/usr/bin/env python3
"""Run one native Quack aggregate owner with bounded authenticated source reads."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_observer import (
        FleetObserver,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_topology import SCHEMA
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )
    deployment = json.loads(args.deployment.read_text())
    if deployment.get('schema') != SCHEMA:
        raise ValueError('compiled fleet topology required')
    owner = deployment['instances']['aggregate_control']
    program = owner['database_program']
    if program['authority_mode'] != 'quack' or program['task_source_kind'] != 'duckdb' or program['failover_policy'] != 'fail_closed':
        raise ValueError('aggregate control must use DuckDB + Quack')
    host, port = program['quack_endpoint'].removeprefix('quack:').rsplit(':', 1)
    if host != '127.0.0.1':
        raise ValueError('aggregate owner must bind loopback')
    database, state = Path(owner['database_path']), Path(owner['state_dir'])
    server = build_server(database_path=database, state_dir=state, repository_root=database.parent.parent,
                          host=host, port=int(port), repository_id='fleet:aggregate_control', store_id=program['store_id'],
                          secret_handle=program['endpoint_secret_handle'], allow_legacy_board_unstall=False)
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    observer = None
    try:
        identity = server.start()
        print(json.dumps(identity.to_dict()), flush=True)
        observer = FleetObserver(server, args.inventory, args.deployment.parent / 'aggregate-view.json', poll_seconds=args.poll_seconds)
        observer.start()
        while not stopping:
            if not observer.thread.is_alive() or time.monotonic() - observer.last_progress > 600:
                raise RuntimeError('fleet observer stopped making bounded progress')
            if server.stop_control_path().is_file():
                break
            server.service_mutation_inbox(max_requests=32)
            time.sleep(0.25)
    finally:
        if observer:
            observer.stop()
        server.stop()


if __name__ == '__main__':
    main()
