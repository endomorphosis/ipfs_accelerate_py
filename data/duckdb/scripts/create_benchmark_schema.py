#!/usr/bin/env python
"""Create the DuckDB benchmark schema used by Hallucinate App dashboards.

The schema keeps the historical benchmark/result tables and adds the
VAIOS-G707 ``interface contract hallucinate_app mobile`` event tables consumed
by the mobile interoperability validation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT = "interface contract hallucinate_app mobile"
HALLUCINATE_APP_MOBILE_INTEROP_DESCRIPTOR = {
    "name": "hallucinate_app_mobile_content_browser",
    "namespace": "handsfree.hallucinate_app.mobile",
    "version": "0.1.0",
    "objective_id": "VAIOS-G707",
    "methods": [
        "ingest_content_search",
        "apply_content_filter",
        "open_module_test_interface",
        "record_benchmark_timeseries_sample",
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create benchmark database schema")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmark_db.duckdb"),
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
        help="Drop and recreate managed tables/views",
    )
    parser.add_argument("--verbose", action="store_true", help="Print SQL progress")
    return parser.parse_args(argv)


def connect_to_db(db_path: os.PathLike[str] | str):
    """Connect to DuckDB, importing the dependency only when the CLI is used."""

    import duckdb

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _execute(conn: Any, sql: str, verbose: bool = False) -> None:
    if verbose:
        first_line = next((line.strip() for line in sql.splitlines() if line.strip()), "")
        print(first_line)
    conn.execute(sql)


def drop_managed_objects(conn: Any, verbose: bool = False) -> None:
    for view_name in [
        "hallucinate_app_mobile_interop_timeseries",
        "model_hardware_compatibility",
        "integration_test_summary",
    ]:
        _execute(conn, f"DROP VIEW IF EXISTS {view_name}", verbose)
    for table_name in [
        "hallucinate_app_mobile_benchmark_samples",
        "hallucinate_app_mobile_interop_events",
        "integration_test_assertions",
        "integration_test_results",
        "performance_batch_results",
        "performance_results",
        "hardware_compatibility",
        "test_runs",
        "models",
        "hardware_platforms",
    ]:
        _execute(conn, f"DROP TABLE IF EXISTS {table_name}", verbose)


def create_common_tables(conn: Any, verbose: bool = False) -> None:
    _execute(
        conn,
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
        verbose,
    )
    _execute(
        conn,
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
        verbose,
    )
    _execute(
        conn,
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
        verbose,
    )


def create_performance_tables(conn: Any, verbose: bool = False) -> None:
    _execute(
        conn,
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
        """,
        verbose,
    )
    _execute(
        conn,
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
        verbose,
    )


def create_hardware_compatibility_tables(conn: Any, verbose: bool = False) -> None:
    _execute(
        conn,
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
        verbose,
    )


def create_integration_test_tables(conn: Any, verbose: bool = False) -> None:
    _execute(
        conn,
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
        verbose,
    )
    _execute(
        conn,
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
        verbose,
    )


def hallucinate_app_mobile_interop_schema_sql() -> str:
    return """
    CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_interop_events (
        event_id VARCHAR PRIMARY KEY,
        contract VARCHAR DEFAULT 'interface contract hallucinate_app mobile',
        objective_id VARCHAR DEFAULT 'VAIOS-G707',
        source_surface VARCHAR DEFAULT 'hallucinate_app',
        target_surface VARCHAR DEFAULT 'mobile',
        descriptor_namespace VARCHAR DEFAULT 'handsfree.hallucinate_app.mobile',
        descriptor_name VARCHAR DEFAULT 'hallucinate_app_mobile_content_browser',
        descriptor_version VARCHAR DEFAULT '0.1.0',
        action VARCHAR NOT NULL,
        mobile_route VARCHAR,
        query_text TEXT,
        filter_json JSON,
        handoff_payload JSON,
        accepted BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_benchmark_samples (
        sample_id VARCHAR PRIMARY KEY,
        event_id VARCHAR NOT NULL,
        metric_name VARCHAR NOT NULL,
        metric_value DOUBLE NOT NULL,
        metric_unit VARCHAR,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSON,
        FOREIGN KEY (event_id) REFERENCES hallucinate_app_mobile_interop_events(event_id)
    );

    CREATE VIEW IF NOT EXISTS hallucinate_app_mobile_interop_timeseries AS
    SELECT
        e.event_id,
        e.contract,
        e.objective_id,
        e.action,
        e.mobile_route,
        e.created_at AS event_time,
        s.sample_id,
        s.metric_name,
        s.metric_value,
        s.metric_unit,
        s.recorded_at
    FROM hallucinate_app_mobile_interop_events e
    LEFT JOIN hallucinate_app_mobile_benchmark_samples s ON e.event_id = s.event_id;
    """


def create_hallucinate_app_mobile_interop_tables(
    conn: Any, verbose: bool = False
) -> None:
    for statement in hallucinate_app_mobile_interop_schema_sql().split(";"):
        if statement.strip():
            _execute(conn, f"{statement};", verbose)
    _execute(
        conn,
        """
        CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_interop_contract
            ON hallucinate_app_mobile_interop_events(contract, objective_id)
        """,
        verbose,
    )
    _execute(
        conn,
        """
        CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_interop_created
            ON hallucinate_app_mobile_interop_events(created_at)
        """,
        verbose,
    )
    _execute(
        conn,
        """
        CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_samples_metric
            ON hallucinate_app_mobile_benchmark_samples(metric_name, recorded_at)
        """,
        verbose,
    )


def create_views(conn: Any, verbose: bool = False) -> None:
    _execute(
        conn,
        """
        CREATE OR REPLACE VIEW model_hardware_compatibility AS
        SELECT
            m.model_name,
            m.model_family,
            hp.hardware_type,
            hp.device_name,
            COUNT(CASE WHEN hc.is_compatible THEN 1 END) AS compatible_count,
            COUNT(CASE WHEN NOT hc.is_compatible THEN 1 END) AS incompatible_count,
            AVG(
                CASE
                    WHEN hc.compatibility_score IS NOT NULL THEN hc.compatibility_score
                    WHEN hc.is_compatible THEN 1.0
                    ELSE 0.0
                END
            ) AS avg_compatibility_score,
            MAX(hc.created_at) AS last_tested
        FROM hardware_compatibility hc
        JOIN models m ON hc.model_id = m.model_id
        JOIN hardware_platforms hp ON hc.hardware_id = hp.hardware_id
        GROUP BY m.model_name, m.model_family, hp.hardware_type, hp.device_name
        """,
        verbose,
    )
    _execute(
        conn,
        """
        CREATE OR REPLACE VIEW integration_test_summary AS
        SELECT
            test_module,
            test_class,
            status,
            COUNT(*) AS result_count,
            MAX(created_at) AS last_recorded_at
        FROM integration_test_results
        GROUP BY test_module, test_class, status
        """,
        verbose,
    )


def create_schema(conn: Any, force: bool = False, verbose: bool = False) -> None:
    if force:
        drop_managed_objects(conn, verbose)
    create_common_tables(conn, verbose)
    create_performance_tables(conn, verbose)
    create_hardware_compatibility_tables(conn, verbose)
    create_integration_test_tables(conn, verbose)
    create_hallucinate_app_mobile_interop_tables(conn, verbose)
    create_views(conn, verbose)


def insert_sample_data(conn: Any, verbose: bool = False) -> None:
    now = dt.datetime.now(dt.UTC).isoformat()
    metadata = json.dumps({"source": "create_benchmark_schema.py"})
    _execute(
        conn,
        f"""
        INSERT OR REPLACE INTO hardware_platforms
            (hardware_id, hardware_type, device_name, platform, metadata)
        VALUES (1, 'mobile', 'Meta glasses companion simulator', 'ios/android', '{metadata}')
        """,
        verbose,
    )
    _execute(
        conn,
        f"""
        INSERT OR REPLACE INTO models
            (model_id, model_name, model_family, modality, source, metadata)
        VALUES (1, 'hallucinate-app-mobile-interop', 'interop', 'multimodal', 'local', '{metadata}')
        """,
        verbose,
    )
    _execute(
        conn,
        f"""
        INSERT OR REPLACE INTO test_runs
            (run_id, test_name, test_type, started_at, completed_at, success, metadata)
        VALUES (1, 'VAIOS-G707 sample', 'integration', '{now}', '{now}', TRUE, '{metadata}')
        """,
        verbose,
    )
    payload = json.dumps(
        {
            "contract": HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT,
            "descriptor": HALLUCINATE_APP_MOBILE_INTEROP_DESCRIPTOR,
            "query": "sample",
            "filter": {"mimetype": "text/"},
        }
    )
    _execute(
        conn,
        f"""
        INSERT OR REPLACE INTO hallucinate_app_mobile_interop_events
            (event_id, action, mobile_route, query_text, filter_json, handoff_payload)
        VALUES (
            'sample-vaios-g707',
            'ingest_content_search',
            'mobile://hallucinate_app/content-browser/search',
            'sample',
            '{{"mimetype":"text/"}}',
            '{payload}'
        )
        """,
        verbose,
    )
    _execute(
        conn,
        """
        INSERT OR REPLACE INTO hallucinate_app_mobile_benchmark_samples
            (sample_id, event_id, metric_name, metric_value, metric_unit, metadata)
        VALUES (
            'sample-vaios-g707-latency',
            'sample-vaios-g707',
            'handoff_latency_ms',
            0.0,
            'ms',
            '{"sample":true}'
        )
        """,
        verbose,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, force=args.force, verbose=args.verbose)
        if args.sample_data:
            insert_sample_data(conn, verbose=args.verbose)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
