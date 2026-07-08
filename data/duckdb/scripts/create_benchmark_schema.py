#!/usr/bin/env python3
"""Create the DuckDB benchmark schema used by hallucinate_app.

The schema stores model benchmark data, integration test results, and the
hallucinate_app-to-mobile handoff receipts used by the mobile interop dashboard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable


HALLUCINATE_APP_MOBILE_HANDOFF_CONTRACT = (
    "handsfree.hallucinate_app/mobile-handoff@0.1.0"
)
HALLUCINATE_APP_MOBILE_INTERFACE_DESCRIPTOR = (
    "hallucinate_app.mobile.interface_descriptor.v1"
)


DROP_OBJECTS: tuple[str, ...] = (
    "DROP VIEW IF EXISTS hallucinate_app_mobile_handoff_summary",
    "DROP VIEW IF EXISTS latest_performance_metrics",
    "DROP VIEW IF EXISTS model_hardware_compatibility",
    "DROP TABLE IF EXISTS hallucinate_app_mobile_handoffs",
    "DROP TABLE IF EXISTS integration_test_assertions",
    "DROP TABLE IF EXISTS integration_test_results",
    "DROP TABLE IF EXISTS performance_batch_results",
    "DROP TABLE IF EXISTS webgpu_advanced_features",
    "DROP TABLE IF EXISTS web_platform_results",
    "DROP TABLE IF EXISTS hardware_compatibility",
    "DROP TABLE IF EXISTS performance_results",
    "DROP TABLE IF EXISTS test_runs",
    "DROP TABLE IF EXISTS models",
    "DROP TABLE IF EXISTS hardware_platforms",
)


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
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
    )
    """,
    """
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
    )
    """,
    """
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
    )
    """,
    """
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
        FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
        FOREIGN KEY (model_id) REFERENCES models(model_id),
        FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS performance_batch_results (
        batch_id INTEGER PRIMARY KEY,
        result_id INTEGER NOT NULL,
        batch_index INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        latency_ms DOUBLE,
        memory_usage_mb DOUBLE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (result_id) REFERENCES performance_results(result_id)
    )
    """,
    """
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
    )
    """,
    """
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
    )
    """,
    """
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS web_platform_results (
        result_id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL,
        model_id INTEGER NOT NULL,
        hardware_id INTEGER NOT NULL,
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
    )
    """,
    """
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_handoffs (
        handoff_id INTEGER PRIMARY KEY,
        request_id VARCHAR NOT NULL UNIQUE,
        contract VARCHAR NOT NULL,
        descriptor VARCHAR NOT NULL,
        operation VARCHAR NOT NULL,
        source VARCHAR NOT NULL,
        target VARCHAR NOT NULL,
        payload JSON NOT NULL,
        handoff JSON NOT NULL,
        policy JSON NOT NULL,
        receipt JSON,
        status VARCHAR DEFAULT 'created',
        mobile_session_id VARCHAR,
        created_at TIMESTAMP NOT NULL,
        acknowledged_at TIMESTAMP,
        CHECK (contract = 'handsfree.hallucinate_app/mobile-handoff@0.1.0'),
        CHECK (descriptor = 'hallucinate_app.mobile.interface_descriptor.v1'),
        CHECK (operation IN ('search', 'filter', 'clear', 'module_test', 'benchmark_telemetry'))
    )
    """,
    """
    CREATE OR REPLACE VIEW model_hardware_compatibility AS
    SELECT
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        COUNT(CASE WHEN hc.is_compatible THEN 1 END) AS compatible_count,
        COUNT(CASE WHEN NOT hc.is_compatible THEN 1 END) AS incompatible_count,
        AVG(COALESCE(hc.compatibility_score, CASE WHEN hc.is_compatible THEN 1.0 ELSE 0.0 END))
            AS avg_compatibility_score,
        MAX(hc.created_at) AS last_tested
    FROM hardware_compatibility hc
    JOIN models m ON hc.model_id = m.model_id
    JOIN hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    GROUP BY m.model_name, m.model_family, hp.hardware_type, hp.device_name
    """,
    """
    CREATE OR REPLACE VIEW latest_performance_metrics AS
    SELECT *
    FROM (
        SELECT
            m.model_name,
            m.model_family,
            hp.hardware_type,
            hp.device_name,
            pr.test_case,
            pr.batch_size,
            pr.precision,
            pr.average_latency_ms,
            pr.throughput_items_per_second,
            pr.memory_peak_mb,
            pr.created_at,
            ROW_NUMBER() OVER (
                PARTITION BY m.model_id, hp.hardware_id, pr.test_case
                ORDER BY pr.created_at DESC
            ) AS rn
        FROM performance_results pr
        JOIN models m ON pr.model_id = m.model_id
        JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    ) ranked
    WHERE rn = 1
    """,
    """
    CREATE OR REPLACE VIEW hallucinate_app_mobile_handoff_summary AS
    SELECT
        operation,
        status,
        COUNT(*) AS handoff_count,
        MIN(created_at) AS first_created_at,
        MAX(COALESCE(acknowledged_at, created_at)) AS last_seen_at
    FROM hallucinate_app_mobile_handoffs
    GROUP BY operation, status
    """,
)


INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_perf_results_model_hw ON performance_results(model_id, hardware_id)",
    "CREATE INDEX IF NOT EXISTS idx_perf_results_run_group ON performance_results(run_group_id)",
    "CREATE INDEX IF NOT EXISTS idx_integration_results_module ON integration_test_results(test_module)",
    "CREATE INDEX IF NOT EXISTS idx_mobile_handoffs_request ON hallucinate_app_mobile_handoffs(request_id)",
    "CREATE INDEX IF NOT EXISTS idx_mobile_handoffs_operation ON hallucinate_app_mobile_handoffs(operation)",
    "CREATE INDEX IF NOT EXISTS idx_mobile_handoffs_created ON hallucinate_app_mobile_handoffs(created_at)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hallucinate_app benchmark DuckDB schema")
    parser.add_argument("--output", default="./benchmark_db.duckdb", help="DuckDB file to create or update")
    parser.add_argument("--sample-data", action="store_true", help="Insert a small validation dataset")
    parser.add_argument("--force", action="store_true", help="Drop and recreate managed tables and views")
    parser.add_argument("--verbose", action="store_true", help="Print executed schema steps")
    return parser.parse_args()


def connect_to_db(db_path: str):
    import duckdb

    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def execute_all(conn, statements: Iterable[str], verbose: bool = False) -> None:
    for statement in statements:
        sql = statement.strip()
        if verbose:
            print(sql.splitlines()[0][:100])
        conn.execute(sql)


def create_schema(conn, *, force: bool = False, verbose: bool = False) -> None:
    if force:
        execute_all(conn, DROP_OBJECTS, verbose=verbose)
    execute_all(conn, SCHEMA_STATEMENTS, verbose=verbose)
    execute_all(conn, INDEX_STATEMENTS, verbose=verbose)


def insert_sample_data(conn) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    conn.execute(
        """
        INSERT OR IGNORE INTO hardware_platforms
        (hardware_id, hardware_type, device_name, platform, metadata, created_at)
        VALUES (1, 'cpu', 'local validation CPU', 'local', ?, ?)
        """,
        [json.dumps({"purpose": "schema validation"}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO models
        (model_id, model_name, model_family, modality, source, metadata, created_at)
        VALUES (1, 'bert-base-uncased', 'bert', 'text', 'huggingface', ?, ?)
        """,
        [json.dumps({"sample": True}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO test_runs
        (run_id, test_name, test_type, started_at, completed_at, execution_time_seconds,
         success, git_commit, git_branch, command_line, metadata, created_at)
        VALUES (1, 'hallucinate_app_mobile_interop', 'integration', ?, ?, 1.0,
                TRUE, 'sample', 'main', 'python -m pytest tests/integration -q', ?, ?)
        """,
        [now, now, json.dumps({"contract": HALLUCINATE_APP_MOBILE_HANDOFF_CONTRACT}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO integration_test_results
        (test_result_id, run_id, test_module, test_class, test_name, status,
         execution_time_seconds, metadata, created_at)
        VALUES (1, 1, 'test_hallucinate_app_mobile_interop',
                'TestHallucinateAppMobileInterop',
                'test_mobile_handoff_contract_descriptor', 'pass', 1.0, ?, ?)
        """,
        [json.dumps({"descriptor": HALLUCINATE_APP_MOBILE_INTERFACE_DESCRIPTOR}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO hallucinate_app_mobile_handoffs
        (handoff_id, request_id, contract, descriptor, operation, source, target,
         payload, handoff, policy, receipt, status, mobile_session_id, created_at,
         acknowledged_at)
        VALUES (1, 'sample-hallucinate-mobile-search',
                'handsfree.hallucinate_app/mobile-handoff@0.1.0',
                'hallucinate_app.mobile.interface_descriptor.v1',
                'search', 'hallucinate_app', 'mobile', ?, ?, ?, ?, 'accepted',
                'mobile-sample-session', ?, ?)
        """,
        [
            json.dumps({"query": "sample", "filter": {}}),
            json.dumps({"transport": "local-ipc-or-http", "receipt_required": True}),
            json.dumps({"user_visible": True, "requires_mobile_ack": True}),
            json.dumps({"status": "accepted"}),
            now,
            now,
        ],
    )


def main() -> None:
    args = parse_args()
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, force=args.force, verbose=args.verbose)
        if args.sample_data:
            insert_sample_data(conn)
        if args.verbose:
            tables = conn.execute("SHOW TABLES").fetchall()
            print(f"Created or verified {len(tables)} DuckDB tables/views in {args.output}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
