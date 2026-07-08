#!/usr/bin/env python
"""Create the DuckDB benchmark and interoperability evidence schema."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - exercised by callers without duckdb installed.
    raise SystemExit("duckdb is required to create the benchmark schema") from exc


TABLES_IN_DROP_ORDER = [
    "hallucinate_app_mobile_interop_events",
    "integration_test_assertions",
    "integration_test_results",
    "performance_batch_results",
    "performance_results",
    "hardware_compatibility",
    "test_runs",
    "models",
    "hardware_platforms",
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
        help="Insert a small Hallucinate App/mobile interop sample event.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate managed tables before creating the schema.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print created table names.")
    return parser.parse_args()


def connect_to_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def execute_many(conn: duckdb.DuckDBPyConnection, statements: Iterable[str]) -> None:
    for statement in statements:
        conn.execute(statement)


def drop_tables(conn: duckdb.DuckDBPyConnection) -> None:
    for table in TABLES_IN_DROP_ORDER:
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def create_common_tables(conn: duckdb.DuckDBPyConnection) -> None:
    execute_many(
        conn,
        [
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
        ],
    )


def create_benchmark_tables(conn: duckdb.DuckDBPyConnection) -> None:
    execute_many(
        conn,
        [
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
                latency_ms DOUBLE,
                throughput_items_per_second DOUBLE,
                memory_peak_mb DOUBLE,
                memory_mb DOUBLE,
                power_watts DOUBLE,
                iterations INTEGER,
                warmup_iterations INTEGER,
                metrics JSON,
                version_tag VARCHAR,
                git_commit_hash VARCHAR,
                environment_hash VARCHAR,
                run_group_id VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        ],
    )


def create_hallucinate_app_mobile_interop_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_interop_events (
            event_id VARCHAR PRIMARY KEY,
            descriptor_id VARCHAR NOT NULL,
            interface_contract VARCHAR NOT NULL,
            objective_goal VARCHAR NOT NULL,
            source_surface VARCHAR NOT NULL,
            target_surface VARCHAR NOT NULL,
            event_name VARCHAR NOT NULL,
            operation VARCHAR NOT NULL,
            edge_session_id VARCHAR,
            query TEXT,
            filter JSON,
            render_targets JSON,
            dispatch_operation VARCHAR,
            status VARCHAR DEFAULT 'emitted',
            evidence_term VARCHAR DEFAULT 'objective validation repair',
            emitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CHECK (interface_contract = 'interface contract hallucinate_app mobile'),
            CHECK (objective_goal = 'VAIOS-G707'),
            CHECK (source_surface = 'hallucinate_app'),
            CHECK (target_surface = 'mobile'),
            CHECK (event_name = 'hallucinate-app:mobile-interop-handoff')
        )
        """
    )


def create_views(conn: duckdb.DuckDBPyConnection) -> None:
    execute_many(
        conn,
        [
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
            """,
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
            """,
            """
            CREATE OR REPLACE VIEW hallucinate_app_mobile_interop_latest AS
            SELECT
                event_id,
                descriptor_id,
                interface_contract,
                objective_goal,
                source_surface,
                target_surface,
                event_name,
                operation,
                edge_session_id,
                status,
                evidence_term,
                emitted_at,
                recorded_at
            FROM hallucinate_app_mobile_interop_events
            WHERE objective_goal = 'VAIOS-G707'
            """,
        ],
    )


def insert_sample_data(conn: duckdb.DuckDBPyConnection) -> None:
    emitted_at = datetime(2026, 7, 8, tzinfo=timezone.utc).replace(tzinfo=None)
    conn.execute(
        """
        INSERT OR REPLACE INTO hallucinate_app_mobile_interop_events (
            event_id,
            descriptor_id,
            interface_contract,
            objective_goal,
            source_surface,
            target_surface,
            event_name,
            operation,
            edge_session_id,
            query,
            filter,
            render_targets,
            dispatch_operation,
            status,
            evidence_term,
            emitted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "event:hallucinate-app-mobile:vai-674",
            "hallucinate-app-mobile-interop@0.1.0",
            "interface contract hallucinate_app mobile",
            "VAIOS-G707",
            "hallucinate_app",
            "mobile",
            "hallucinate-app:mobile-interop-handoff",
            "search_handoff",
            "local:edge-session:handsfree-mobile-orb-edge",
            "objective validation repair",
            json.dumps({"source": "VAI-674"}),
            json.dumps(["mobile_card", "meta_glasses_display", "audio_summary"]),
            "dispatch_glasses_response",
            "validated",
            "objective validation repair",
            emitted_at,
        ],
    )


def create_schema(conn: duckdb.DuckDBPyConnection, *, force: bool = False) -> None:
    if force:
        drop_tables(conn)
    create_common_tables(conn)
    create_benchmark_tables(conn)
    create_hallucinate_app_mobile_interop_tables(conn)
    create_views(conn)


def main() -> None:
    args = parse_args()
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, force=args.force)
        if args.sample_data:
            insert_sample_data(conn)
        if args.verbose:
            table_names = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
            for table_name in table_names:
                print(table_name)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
