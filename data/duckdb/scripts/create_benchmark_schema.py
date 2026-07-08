#!/usr/bin/env python3
"""Create the DuckDB benchmark schema used by hallucinate_app test surfaces."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT = (
    "handsfree.hallucinate_app/mobile-search-handoff@0.1.0"
)
HALLUCINATE_APP_MOBILE_ACTION_ID = "mobile_hallucinate_app_search"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create benchmark database schema")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmark_db.duckdb"),
        help="Path to create or update the DuckDB database",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Generate sample data that exercises the schema",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop known tables and views before recreating the schema",
    )
    parser.add_argument("--verbose", action="store_true", help="Print schema creation details")
    return parser.parse_args()


def connect_to_db(db_path: str | Path):
    import duckdb

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _execute_many(conn: Any, statements: list[str]) -> None:
    for statement in statements:
        conn.execute(statement)


def drop_schema(conn: Any) -> None:
    _execute_many(
        conn,
        [
            "DROP VIEW IF EXISTS hallucinate_app_mobile_interop_status",
            "DROP VIEW IF EXISTS integration_test_status",
            "DROP VIEW IF EXISTS latest_performance_metrics",
            "DROP TABLE IF EXISTS hallucinate_app_mobile_handoff_assertions",
            "DROP TABLE IF EXISTS hallucinate_app_mobile_handoffs",
            "DROP TABLE IF EXISTS integration_test_assertions",
            "DROP TABLE IF EXISTS integration_test_results",
            "DROP TABLE IF EXISTS performance_batch_results",
            "DROP TABLE IF EXISTS performance_results",
            "DROP TABLE IF EXISTS web_platform_results",
            "DROP TABLE IF EXISTS hardware_compatibility",
            "DROP TABLE IF EXISTS test_runs",
            "DROP TABLE IF EXISTS models",
            "DROP TABLE IF EXISTS hardware_platforms",
        ],
    )


def create_common_tables(conn: Any) -> None:
    _execute_many(
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


def create_performance_tables(conn: Any) -> None:
    _execute_many(
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
            "CREATE INDEX IF NOT EXISTS idx_performance_results_run ON performance_results(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_results_model_hw ON performance_results(model_id, hardware_id)",
        ],
    )


def create_hardware_compatibility_tables(conn: Any) -> None:
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


def create_integration_test_tables(conn: Any) -> None:
    _execute_many(
        conn,
        [
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


def create_web_platform_tables(conn: Any) -> None:
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


def create_hallucinate_app_mobile_tables(conn: Any) -> None:
    _execute_many(
        conn,
        [
            """
            CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_handoffs (
                handoff_id INTEGER PRIMARY KEY,
                run_id INTEGER,
                request_id VARCHAR NOT NULL,
                correlation_id VARCHAR,
                contract VARCHAR NOT NULL,
                action_id VARCHAR NOT NULL,
                profile VARCHAR,
                source_surface VARCHAR NOT NULL,
                target_surface VARCHAR NOT NULL,
                query VARCHAR,
                filters JSON,
                cid VARCHAR,
                path VARCHAR,
                mobile_payload JSON NOT NULL,
                control_plane JSON,
                handoff JSON,
                policy JSON,
                receipts JSON,
                status VARCHAR DEFAULT 'queued',
                receipt_cid VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_handoff_assertions (
                assertion_id INTEGER PRIMARY KEY,
                handoff_id INTEGER NOT NULL,
                assertion_name VARCHAR NOT NULL,
                passed BOOLEAN NOT NULL,
                expected_value VARCHAR,
                actual_value VARCHAR,
                message VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (handoff_id) REFERENCES hallucinate_app_mobile_handoffs(handoff_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_handoff_request ON hallucinate_app_mobile_handoffs(request_id)",
            "CREATE INDEX IF NOT EXISTS idx_handoff_contract ON hallucinate_app_mobile_handoffs(contract, action_id)",
            "CREATE INDEX IF NOT EXISTS idx_handoff_created ON hallucinate_app_mobile_handoffs(created_at)",
        ],
    )


def create_views(conn: Any) -> None:
    _execute_many(
        conn,
        [
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
                pr.created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY m.model_id, hp.hardware_id
                    ORDER BY pr.created_at DESC
                ) AS rn
            FROM performance_results pr
            JOIN models m ON pr.model_id = m.model_id
            JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            QUALIFY rn = 1
            """,
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
            CREATE OR REPLACE VIEW hallucinate_app_mobile_interop_status AS
            SELECT
                h.request_id,
                h.contract,
                h.action_id,
                h.source_surface,
                h.target_surface,
                h.query,
                h.status,
                h.receipt_cid,
                COUNT(a.assertion_id) AS assertion_count,
                COUNT(CASE WHEN a.passed THEN 1 END) AS passed_assertions,
                h.created_at
            FROM hallucinate_app_mobile_handoffs h
            LEFT JOIN hallucinate_app_mobile_handoff_assertions a
                ON h.handoff_id = a.handoff_id
            GROUP BY
                h.request_id,
                h.contract,
                h.action_id,
                h.source_surface,
                h.target_surface,
                h.query,
                h.status,
                h.receipt_cid,
                h.created_at
            """,
        ],
    )


def create_schema(conn: Any, *, force: bool = False) -> None:
    if force:
        drop_schema(conn)
    create_common_tables(conn)
    create_performance_tables(conn)
    create_hardware_compatibility_tables(conn)
    create_integration_test_tables(conn)
    create_web_platform_tables(conn)
    create_hallucinate_app_mobile_tables(conn)
    create_views(conn)


def _insert_or_ignore(conn: Any, sql: str, values: tuple[Any, ...]) -> None:
    try:
        conn.execute(sql, values)
    except Exception as exc:  # pragma: no cover - duplicate sample rows are harmless.
        if "duplicate" not in str(exc).lower() and "constraint" not in str(exc).lower():
            raise


def generate_sample_data(conn: Any) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    _insert_or_ignore(
        conn,
        """
        INSERT INTO hardware_platforms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "mobile",
            "Handsfree Mobile App",
            "expo",
            "54",
            None,
            0.0,
            0,
            json.dumps({"surface": "mobile.results"}),
            now,
        ),
    )
    _insert_or_ignore(
        conn,
        "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "hallucinate-app-content-search",
            "content-browser",
            "multimodal",
            "hallucinate_app",
            "0.1.0",
            0.0,
            json.dumps({"contract": HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT}),
            now,
        ),
    )
    _insert_or_ignore(
        conn,
        "INSERT INTO test_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "hallucinate_app_mobile_interop",
            "integration",
            now,
            now,
            0.01,
            True,
            None,
            "local",
            "python -m pytest tests/integration/test_hallucinate_app_mobile_interop.py",
            json.dumps({"objective": "VAIOS-G707"}),
            now,
        ),
    )
    mobile_payload = {
        "type": HALLUCINATE_APP_MOBILE_ACTION_ID,
        "contract": HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT,
        "request_id": "hallucinate-app-mobile-sample",
        "query": "cid:QmDemo hallucinate_app mobile interoperability",
        "filters": {"source": "sample"},
        "result_limit": 20,
        "source_surface": "hallucinate_app.content_browser",
        "target_surface": "mobile.results",
    }
    _insert_or_ignore(
        conn,
        """
        INSERT INTO hallucinate_app_mobile_handoffs (
            handoff_id,
            run_id,
            request_id,
            correlation_id,
            contract,
            action_id,
            profile,
            source_surface,
            target_surface,
            query,
            filters,
            cid,
            path,
            mobile_payload,
            control_plane,
            handoff,
            policy,
            receipts,
            status,
            receipt_cid
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            1,
            1,
            "hallucinate-app-mobile-sample",
            "hallucinate-app-mobile-sample",
            HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT,
            HALLUCINATE_APP_MOBILE_ACTION_ID,
            "swissknife.mcp++/event-envelope@0.1.0",
            "hallucinate_app.content_browser",
            "mobile.results",
            mobile_payload["query"],
            json.dumps(mobile_payload["filters"]),
            None,
            None,
            json.dumps(mobile_payload),
            json.dumps({"route": "hallucinate_app.mobile.search_handoff"}),
            json.dumps({"ipfs_cids": []}),
            json.dumps({"outcome": "allow"}),
            json.dumps([]),
            "delivered",
            "sha256:hallucinate-app-mobile:sample",
        ),
    )


def main() -> None:
    args = parse_args()
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, force=args.force)
        if args.sample_data:
            generate_sample_data(conn)
        if args.verbose:
            tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
            print(f"Created {len(tables)} DuckDB tables/views in {args.output}")
            for table in tables:
                print(f"  - {table}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
