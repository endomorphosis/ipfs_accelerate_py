#!/usr/bin/env python3
"""Create the DuckDB benchmark schema used by hallucinate_app.

The script intentionally keeps the legacy helper names used by downstream
utilities while adding the MGW-579 Hallucinate App <-> mobile interop table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create benchmark database schema")
    parser.add_argument(
        "--output",
        "--db-path",
        dest="output",
        default="./benchmark_db.duckdb",
        help="Path to create/update the DuckDB database",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Generate sample data to test the schema",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate managed tables before creating the schema",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logging information",
    )
    return parser.parse_args()


def connect_to_db(db_path: str):
    """Connect to a DuckDB database, creating parent directories as needed."""
    import duckdb

    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return duckdb.connect(db_path)


def _drop_tables(conn, tables: Iterable[str]) -> None:
    for table in tables:
        conn.execute(f"DROP VIEW IF EXISTS {table}")
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def create_common_tables(conn, force: bool = False) -> None:
    """Create dimension tables shared by benchmark and integration results."""
    if force:
        _drop_tables(conn, ["test_runs", "models", "hardware_platforms"])

    conn.execute(
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
        """
    )
    conn.execute(
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
        """
    )
    conn.execute(
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
        """
    )


def create_performance_tables(conn, force: bool = False) -> None:
    """Create tables for model and hardware benchmark metrics."""
    if force:
        _drop_tables(conn, ["performance_batch_results", "performance_results"])

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS performance_results (
            result_id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            hardware_id INTEGER NOT NULL,
            test_case VARCHAR NOT NULL,
            batch_size INTEGER DEFAULT 1,
            precision VARCHAR,
            total_time_seconds DOUBLE,
            average_latency_ms DOUBLE,
            throughput_items_per_second DOUBLE,
            memory_peak_mb DOUBLE,
            iterations INTEGER,
            warmup_iterations INTEGER,
            metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """
    )
    conn.execute(
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
        """
    )


def create_hardware_compatibility_tables(conn, force: bool = False) -> None:
    """Create tables for hardware compatibility test results."""
    if force:
        _drop_tables(conn, ["hardware_compatibility"])

    conn.execute(
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
        """
    )


def create_integration_test_tables(conn, force: bool = False) -> None:
    """Create tables for integration test results and assertion details."""
    if force:
        _drop_tables(conn, ["integration_test_assertions", "integration_test_results"])

    conn.execute(
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
        """
    )
    conn.execute(
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
        """
    )


def create_web_platform_tables(conn, force: bool = False) -> None:
    """Create tables for browser/WebNN/WebGPU test results."""
    if force:
        _drop_tables(conn, ["webgpu_advanced_features", "web_platform_results"])

    conn.execute(
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
        """
    )
    conn.execute(
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
        """
    )


def create_hallucinate_app_mobile_interop_tables(conn, force: bool = False) -> None:
    """Create the objective validation repair table for VAIOS-G707 / MGW-579."""
    if force:
        _drop_tables(
            conn,
            [
                "hallucinate_app_mobile_interop_latest",
                "hallucinate_app_mobile_interop_events",
            ],
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_interop_events (
            interop_event_id INTEGER PRIMARY KEY,
            event_name VARCHAR NOT NULL DEFAULT 'hallucinate-app:mobile-interop-handoff',
            contract VARCHAR NOT NULL DEFAULT 'interface contract hallucinate_app mobile',
            descriptor_name VARCHAR NOT NULL DEFAULT 'HALLUCINATE_APP_MOBILE_INTEROP_DESCRIPTOR',
            mobile_interface VARCHAR NOT NULL DEFAULT 'handsfree.meta_glasses.mobile.hallucinate_app_mobile_interop@0.1.0',
            edge_session_id VARCHAR,
            correlation_id VARCHAR,
            action VARCHAR NOT NULL,
            query TEXT,
            filter_json JSON,
            route_json JSON,
            receipt_cid VARCHAR,
            accepted BOOLEAN DEFAULT FALSE,
            source_path VARCHAR NOT NULL DEFAULT 'hallucinate_app/hallucinate_app/node/dashboard/content_browser/search_interface.js',
            mobile_descriptor_path VARCHAR NOT NULL DEFAULT 'mobile/src/orb/metaGlassesOrbDescriptors.js',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_interop_event_name
        ON hallucinate_app_mobile_interop_events(event_name)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_interop_edge_session
        ON hallucinate_app_mobile_interop_events(edge_session_id)
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE VIEW hallucinate_app_mobile_interop_latest AS
        SELECT
            interop_event_id,
            event_name,
            contract,
            descriptor_name,
            mobile_interface,
            edge_session_id,
            correlation_id,
            action,
            query,
            accepted,
            receipt_cid,
            created_at
        FROM hallucinate_app_mobile_interop_events
        ORDER BY created_at DESC
        """
    )


def create_views(conn) -> None:
    """Create reporting views across the benchmark schema."""
    conn.execute(
        """
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
        GROUP BY test_module
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE VIEW latest_performance_metrics AS
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
            pr.created_at
        FROM performance_results pr
        JOIN models m ON pr.model_id = m.model_id
        JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY m.model_id, hp.hardware_id
            ORDER BY pr.created_at DESC
        ) = 1
        """
    )
    conn.execute(
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
        """
    )


def generate_sample_data(conn) -> None:
    """Insert a compact sample dataset, including one mobile interop handoff."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    conn.execute(
        """
        INSERT OR IGNORE INTO hardware_platforms
        (hardware_id, hardware_type, device_name, platform, metadata, created_at)
        VALUES (1, 'mobile', 'Meta glasses display simulator', 'simulator', ?, ?)
        """,
        [json.dumps({"surface": "meta_glasses_display"}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO models
        (model_id, model_name, model_family, modality, source, metadata, created_at)
        VALUES (1, 'hallucinate-app-mobile-interop', 'integration', 'multimodal', 'local', ?, ?)
        """,
        [json.dumps({"objective": "VAIOS-G707"}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO test_runs
        (run_id, test_name, test_type, started_at, completed_at, success, command_line, metadata, created_at)
        VALUES (1, 'MGW-579 Hallucinate App mobile interop', 'integration', ?, ?, TRUE, ?, ?, ?)
        """,
        [
            now,
            now,
            "python -m pytest tests/integration/test_hallucinate_app_mobile_interop.py",
            json.dumps({"task": "MGW-579", "goal": "VAIOS-G707"}),
            now,
        ],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO hallucinate_app_mobile_interop_events
        (interop_event_id, edge_session_id, correlation_id, action, query, filter_json, route_json, receipt_cid, accepted, created_at)
        VALUES (1, 'local:edge-session:handsfree-mobile-orb-edge', 'mgw-579-sample', 'search', ?, ?, ?, ?, TRUE, ?)
        """,
        [
            "show saved model artifacts on glasses",
            json.dumps({"mimetype": "application/json"}),
            json.dumps({"from": "hallucinate_app", "to": "mobile", "transport": "mobile_orb_bridge"}),
            "local:receipt:mgw-579",
            now,
        ],
    )


def create_schema(conn, force: bool = False) -> None:
    """Create the full benchmark schema."""
    if force:
        _drop_tables(
            conn,
            [
                "model_hardware_compatibility",
                "latest_performance_metrics",
                "integration_test_status",
                "hallucinate_app_mobile_interop_latest",
                "integration_test_assertions",
                "integration_test_results",
                "webgpu_advanced_features",
                "web_platform_results",
                "performance_batch_results",
                "performance_results",
                "hardware_compatibility",
                "hallucinate_app_mobile_interop_events",
                "test_runs",
                "models",
                "hardware_platforms",
            ],
        )
    create_common_tables(conn, False)
    create_performance_tables(conn, False)
    create_hardware_compatibility_tables(conn, False)
    create_integration_test_tables(conn, False)
    create_web_platform_tables(conn, False)
    create_hallucinate_app_mobile_interop_tables(conn, False)
    create_views(conn)


def main() -> int:
    args = parse_args()
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, args.force)
        if args.sample_data:
            generate_sample_data(conn)
        if args.verbose:
            tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
            print(f"Created/verified {len(tables)} DuckDB tables and views at {Path(args.output)}")
            for table in tables:
                print(f"  - {table}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
