#!/usr/bin/env python3
"""Create the DuckDB benchmark schema used by Hallucinate App dashboards.

The schema stores benchmark, compatibility, integration-test, web-platform, and
Hallucinate App <-> mobile handoff evidence in one local database.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import duckdb


COMMON_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS hardware_platforms (
    hardware_id INTEGER PRIMARY KEY,
    hardware_type VARCHAR NOT NULL,
    device_name VARCHAR,
    platform VARCHAR,
    platform_version VARCHAR,
    driver_version VARCHAR,
    memory_gb DOUBLE,
    compute_units INTEGER,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
    model_id INTEGER PRIMARY KEY,
    model_name VARCHAR NOT NULL,
    model_family VARCHAR,
    modality VARCHAR,
    source VARCHAR,
    version VARCHAR,
    parameters_million DOUBLE,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS test_runs (
    run_id INTEGER PRIMARY KEY,
    test_name VARCHAR NOT NULL,
    test_type VARCHAR NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_seconds DOUBLE,
    success BOOLEAN,
    git_commit VARCHAR,
    git_branch VARCHAR,
    command_line VARCHAR,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


BENCHMARK_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS performance_results (
    result_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    hardware_id INTEGER NOT NULL,
    test_case VARCHAR NOT NULL,
    batch_size INTEGER DEFAULT 1,
    sequence_length INTEGER,
    precision VARCHAR DEFAULT 'fp32',
    total_time_seconds DOUBLE,
    average_latency_ms DOUBLE,
    throughput_items_per_second DOUBLE,
    memory_peak_mb DOUBLE,
    power_watts DOUBLE,
    iterations INTEGER,
    warmup_iterations INTEGER,
    metrics JSON,
    version_tag VARCHAR,
    git_commit_hash VARCHAR,
    environment_hash VARCHAR,
    run_group_id VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
);

CREATE TABLE IF NOT EXISTS performance_batch_results (
    batch_id INTEGER PRIMARY KEY,
    result_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    latency_ms DOUBLE,
    memory_usage_mb DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (result_id) REFERENCES performance_results(result_id)
);

CREATE TABLE IF NOT EXISTS hardware_compatibility (
    compatibility_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    hardware_id INTEGER NOT NULL,
    is_compatible BOOLEAN NOT NULL,
    detection_success BOOLEAN NOT NULL,
    initialization_success BOOLEAN NOT NULL,
    error_message VARCHAR,
    error_type VARCHAR,
    suggested_fix VARCHAR,
    workaround_available BOOLEAN,
    compatibility_score DOUBLE,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
);

CREATE TABLE IF NOT EXISTS integration_test_results (
    test_result_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    test_module VARCHAR NOT NULL,
    test_class VARCHAR,
    test_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    execution_time_seconds DOUBLE,
    hardware_id INTEGER,
    model_id INTEGER,
    error_message VARCHAR,
    error_traceback VARCHAR,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

CREATE TABLE IF NOT EXISTS integration_test_assertions (
    assertion_id INTEGER PRIMARY KEY,
    test_result_id INTEGER NOT NULL,
    assertion_name VARCHAR NOT NULL,
    passed BOOLEAN NOT NULL,
    expected_value VARCHAR,
    actual_value VARCHAR,
    message VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (test_result_id) REFERENCES integration_test_results(test_result_id)
);
"""


WEB_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS web_platform_results (
    result_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    model_id INTEGER,
    hardware_id INTEGER,
    platform VARCHAR NOT NULL,
    browser VARCHAR,
    browser_version VARCHAR,
    test_file VARCHAR,
    success BOOLEAN,
    load_time_ms DOUBLE,
    initialization_time_ms DOUBLE,
    inference_time_ms DOUBLE,
    total_time_ms DOUBLE,
    shader_compilation_time_ms DOUBLE,
    memory_usage_mb DOUBLE,
    error_message VARCHAR,
    metrics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
);

CREATE TABLE IF NOT EXISTS webgpu_advanced_features (
    feature_id INTEGER PRIMARY KEY,
    result_id INTEGER NOT NULL,
    compute_shader_support BOOLEAN,
    parallel_compilation BOOLEAN,
    shader_cache_hit BOOLEAN,
    workgroup_size INTEGER,
    compute_pipeline_time_ms DOUBLE,
    pre_compiled_pipeline BOOLEAN,
    memory_optimization_level VARCHAR,
    audio_acceleration BOOLEAN,
    video_acceleration BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (result_id) REFERENCES web_platform_results(result_id)
);
"""


MOBILE_HANDOFF_SQL = """
CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_handoff_events (
    handoff_id VARCHAR PRIMARY KEY,
    objective_id VARCHAR DEFAULT 'VAIOS-G707',
    interface_contract VARCHAR NOT NULL,
    source_surface VARCHAR DEFAULT 'hallucinate_app',
    target_surface VARCHAR DEFAULT 'mobile',
    operation VARCHAR DEFAULT 'dispatch_content_search',
    edge_session_id VARCHAR NOT NULL,
    correlation_id VARCHAR NOT NULL,
    query TEXT,
    filters JSON,
    ipfs_cids JSON,
    libp2p_peer_id VARCHAR,
    libp2p_session_id VARCHAR,
    descriptor_refs JSON,
    mediation_receipt JSON,
    receipt_cid VARCHAR,
    status VARCHAR DEFAULT 'queued',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ha_mobile_handoff_edge
    ON hallucinate_app_mobile_handoff_events(edge_session_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_ha_mobile_handoff_correlation
    ON hallucinate_app_mobile_handoff_events(correlation_id);

CREATE VIEW IF NOT EXISTS hallucinate_app_mobile_handoff_latest AS
SELECT
    handoff_id,
    objective_id,
    interface_contract,
    source_surface,
    target_surface,
    operation,
    edge_session_id,
    correlation_id,
    query,
    receipt_cid,
    status,
    recorded_at
FROM hallucinate_app_mobile_handoff_events
WHERE objective_id = 'VAIOS-G707'
ORDER BY recorded_at DESC;
"""


VIEWS_SQL = """
CREATE OR REPLACE VIEW model_hardware_compatibility AS
SELECT
    m.model_name,
    m.model_family,
    hp.hardware_type,
    hp.device_name,
    COUNT(CASE WHEN hc.is_compatible THEN 1 END) AS compatible_count,
    COUNT(CASE WHEN NOT hc.is_compatible THEN 1 END) AS incompatible_count,
    AVG(CASE
        WHEN hc.compatibility_score IS NOT NULL THEN hc.compatibility_score
        WHEN hc.is_compatible THEN 1.0
        ELSE 0.0
    END) AS avg_compatibility_score,
    MAX(hc.created_at) AS last_tested
FROM hardware_compatibility hc
JOIN models m ON hc.model_id = m.model_id
JOIN hardware_platforms hp ON hc.hardware_id = hp.hardware_id
GROUP BY m.model_name, m.model_family, hp.hardware_type, hp.device_name;

CREATE OR REPLACE VIEW latest_performance_metrics AS
SELECT *
FROM (
    SELECT
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        pr.batch_size,
        pr.precision,
        pr.average_latency_ms,
        pr.throughput_items_per_second,
        pr.memory_peak_mb,
        pr.created_at,
        ROW_NUMBER() OVER (
            PARTITION BY m.model_id, hp.hardware_id
            ORDER BY pr.created_at DESC
        ) AS rn
    FROM performance_results pr
    JOIN models m ON pr.model_id = m.model_id
    JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
)
WHERE rn = 1;

CREATE OR REPLACE VIEW integration_test_status AS
SELECT
    test_module,
    COUNT(*) AS total_tests,
    COUNT(CASE WHEN status = 'pass' THEN 1 END) AS passed,
    COUNT(CASE WHEN status = 'fail' THEN 1 END) AS failed,
    COUNT(CASE WHEN status = 'error' THEN 1 END) AS errors,
    COUNT(CASE WHEN status = 'skip' THEN 1 END) AS skipped,
    MAX(created_at) AS last_run
FROM integration_test_results
GROUP BY test_module;
"""


DROP_ORDER = [
    "hallucinate_app_mobile_handoff_latest",
    "integration_test_status",
    "latest_performance_metrics",
    "model_hardware_compatibility",
    "webgpu_advanced_features",
    "web_platform_results",
    "integration_test_assertions",
    "integration_test_results",
    "hardware_compatibility",
    "performance_batch_results",
    "performance_results",
    "test_runs",
    "models",
    "hardware_platforms",
    "hallucinate_app_mobile_handoff_events",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmark_db.duckdb"),
        help="Path to create or update the DuckDB database.",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Insert deterministic sample rows for smoke testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop known tables and views before recreating the schema.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print schema objects.")
    return parser.parse_args()


def execute_statements(conn: duckdb.DuckDBPyConnection, sql_blocks: Iterable[str]) -> None:
    for block in sql_blocks:
        conn.execute(block)


def connect_to_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    os.makedirs(db_path.expanduser().resolve().parent, exist_ok=True)
    return duckdb.connect(str(db_path))


def drop_existing_objects(conn: duckdb.DuckDBPyConnection) -> None:
    for name in DROP_ORDER:
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def create_schema(conn: duckdb.DuckDBPyConnection, *, force: bool = False) -> None:
    if force:
        drop_existing_objects(conn)
    execute_statements(
        conn,
        [
            COMMON_TABLES_SQL,
            BENCHMARK_TABLES_SQL,
            WEB_TABLES_SQL,
            MOBILE_HANDOFF_SQL,
            VIEWS_SQL,
        ],
    )


def generate_sample_data(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO hardware_platforms (
            hardware_id, hardware_type, device_name, metadata
        ) VALUES (1, 'cpu', 'local-smoke', ?)
        """,
        [json.dumps({"sample": True})],
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO models (
            model_id, model_name, model_family, modality, metadata
        ) VALUES (1, 'bert-base-uncased', 'bert', 'text', ?)
        """,
        [json.dumps({"sample": True})],
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO test_runs (
            run_id, test_name, test_type, success, metadata
        ) VALUES (1, 'schema-smoke', 'integration', TRUE, ?)
        """,
        [json.dumps({"objective_id": "VAIOS-G707"})],
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO hallucinate_app_mobile_handoff_events (
            handoff_id,
            interface_contract,
            edge_session_id,
            correlation_id,
            query,
            ipfs_cids,
            libp2p_peer_id,
            descriptor_refs,
            mediation_receipt,
            receipt_cid,
            status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "sample:vaios-g707:handoff",
            "handsfree.interop.hallucinate_app_mobile/handoff@0.1.0",
            "local:edge-session:handsfree-mobile-orb-edge",
            "sample-correlation",
            "cid:bafy-mobile-proof",
            json.dumps(["bafy-mobile-proof"]),
            "12D3KooWHallucinateMobile",
            json.dumps(
                [
                    "handsfree.meta_glasses.mobile.mobile_orb_bridge@0.1.0",
                    "handsfree.interop.hallucinate_app_mobile.hallucinate_app_mobile_interop@0.1.0",
                ]
            ),
            json.dumps({"receipt_id": "sample:receipt:vaios-g707"}),
            "sample:receipt:vaios-g707",
            "delivered",
        ],
    )


def describe_schema(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("SHOW TABLES").fetchall()
    return [str(row[0]) for row in rows]


def main() -> None:
    args = parse_args()
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, force=args.force)
        if args.sample_data:
            generate_sample_data(conn)
        if args.verbose:
            for name in describe_schema(conn):
                count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                print(f"{name}: {count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
