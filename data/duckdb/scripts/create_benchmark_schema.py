#!/usr/bin/env python3
"""Create the benchmark DuckDB schema used by Hallucinate App tests.

The script is intentionally importable: integration tests load it directly to
verify the schema API and only require DuckDB when a real database connection is
created.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT = (
    "handsfree.hallucinate_app/mobile-search-handoff@0.1.0"
)
HALLUCINATE_APP_MOBILE_ACTION_ID = "mobile_hallucinate_app_search"
OBJECTIVE_ID = "VAIOS-G707"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create benchmark database schema")
    parser.add_argument("--output", type=Path, default=Path("./benchmark_db.duckdb"))
    parser.add_argument("--sample-data", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def connect_to_db(db_path: str | os.PathLike[str]):
    """Connect to a DuckDB database, importing DuckDB lazily."""

    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("duckdb is required to create the benchmark schema") from exc

    path = Path(db_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _execute_many(conn: Any, statements: list[str]) -> None:
    for statement in statements:
        conn.execute(statement)


def create_common_tables(conn: Any, force: bool = False) -> None:
    if force:
        _execute_many(
            conn,
            [
                "DROP VIEW IF EXISTS hallucinate_app_mobile_interop_status",
                "DROP VIEW IF EXISTS integration_test_status",
                "DROP VIEW IF EXISTS latest_performance_metrics",
                "DROP VIEW IF EXISTS model_hardware_compatibility",
                "DROP TABLE IF EXISTS hallucinate_app_mobile_handoff_assertions",
                "DROP TABLE IF EXISTS hallucinate_app_mobile_handoffs",
                "DROP TABLE IF EXISTS integration_test_assertions",
                "DROP TABLE IF EXISTS integration_test_results",
                "DROP TABLE IF EXISTS performance_batch_results",
                "DROP TABLE IF EXISTS performance_results",
                "DROP TABLE IF EXISTS hardware_compatibility",
                "DROP TABLE IF EXISTS test_runs",
                "DROP TABLE IF EXISTS models",
                "DROP TABLE IF EXISTS hardware_platforms",
            ],
        )

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
                memory_gb FLOAT,
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
                parameters_million FLOAT,
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
                execution_time_seconds FLOAT,
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


def create_performance_tables(conn: Any, force: bool = False) -> None:
    if force:
        _execute_many(
            conn,
            [
                "DROP TABLE IF EXISTS performance_batch_results",
                "DROP TABLE IF EXISTS performance_results",
            ],
        )

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
                total_time_seconds FLOAT,
                average_latency_ms FLOAT,
                throughput_items_per_second FLOAT,
                memory_peak_mb FLOAT,
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
                latency_ms FLOAT,
                memory_usage_mb FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (result_id) REFERENCES performance_results(result_id)
            )
            """,
        ],
    )


def create_hardware_compatibility_tables(conn: Any, force: bool = False) -> None:
    if force:
        conn.execute("DROP TABLE IF EXISTS hardware_compatibility")

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
            compatibility_score FLOAT,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """
    )


def create_integration_test_tables(conn: Any, force: bool = False) -> None:
    if force:
        _execute_many(
            conn,
            [
                "DROP TABLE IF EXISTS integration_test_assertions",
                "DROP TABLE IF EXISTS integration_test_results",
            ],
        )

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
                execution_time_seconds FLOAT,
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
                FOREIGN KEY (test_result_id)
                    REFERENCES integration_test_results(test_result_id)
            )
            """,
        ],
    )


def create_hallucinate_app_mobile_tables(conn: Any, force: bool = False) -> None:
    """Create VAIOS-G707 Hallucinate App <-> mobile handoff tables."""

    if force:
        _execute_many(
            conn,
            [
                "DROP VIEW IF EXISTS hallucinate_app_mobile_interop_status",
                "DROP TABLE IF EXISTS hallucinate_app_mobile_handoff_assertions",
                "DROP TABLE IF EXISTS hallucinate_app_mobile_handoffs",
            ],
        )

    _execute_many(
        conn,
        [
            f"""
            CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_handoffs (
                handoff_id INTEGER PRIMARY KEY,
                request_id VARCHAR NOT NULL,
                objective_id VARCHAR DEFAULT '{OBJECTIVE_ID}',
                interface_contract VARCHAR DEFAULT '{HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT}',
                action_id VARCHAR DEFAULT '{HALLUCINATE_APP_MOBILE_ACTION_ID}',
                event_type VARCHAR DEFAULT 'transport.handoff',
                source_surface VARCHAR DEFAULT 'hallucinate_app.content_browser',
                target_surface VARCHAR DEFAULT 'mobile.results',
                query TEXT,
                filter JSON,
                handoff JSON,
                mobile_payload JSON,
                control_plane JSON,
                receipts JSON,
                mediation_receipt JSON,
                receipt_cid VARCHAR,
                edge_session_id VARCHAR,
                correlation_id VARCHAR,
                ipfs_cids JSON,
                libp2p_peer_id VARCHAR,
                status VARCHAR DEFAULT 'queued',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_handoff_assertions (
                assertion_id INTEGER PRIMARY KEY,
                handoff_id INTEGER,
                assertion_name VARCHAR NOT NULL,
                assertion_status VARCHAR NOT NULL,
                expected_value VARCHAR,
                actual_value VARCHAR,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (handoff_id)
                    REFERENCES hallucinate_app_mobile_handoffs(handoff_id)
            )
            """,
            """
            CREATE OR REPLACE VIEW hallucinate_app_mobile_interop_status AS
            SELECT
                request_id,
                objective_id,
                interface_contract,
                action_id,
                source_surface,
                target_surface,
                edge_session_id,
                correlation_id,
                receipt_cid,
                status,
                created_at,
                completed_at
            FROM hallucinate_app_mobile_handoffs
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_handoffs_request
                ON hallucinate_app_mobile_handoffs(request_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_handoffs_contract
                ON hallucinate_app_mobile_handoffs(interface_contract)
            """,
        ],
    )


def create_views(conn: Any) -> None:
    _execute_many(
        conn,
        [
            """
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
            GROUP BY m.model_name, m.model_family, hp.hardware_type, hp.device_name
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
        ],
    )


def create_indexes(conn: Any) -> None:
    _execute_many(
        conn,
        [
            "CREATE INDEX IF NOT EXISTS idx_performance_results_run ON performance_results(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_results_model_hw ON performance_results(model_id, hardware_id)",
            "CREATE INDEX IF NOT EXISTS idx_integration_results_module ON integration_test_results(test_module)",
            "CREATE INDEX IF NOT EXISTS idx_hardware_compatibility_model_hw ON hardware_compatibility(model_id, hardware_id)",
        ],
    )


def insert_sample_data(conn: Any) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR IGNORE INTO hardware_platforms (
            hardware_id, hardware_type, device_name, platform, metadata
        ) VALUES (1, 'cpu', 'local', 'test', ?)
        """,
        [json.dumps({"sample": True})],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO models (
            model_id, model_name, model_family, modality, source, metadata
        ) VALUES (1, 'sample-model', 'sample', 'text', 'local', ?)
        """,
        [json.dumps({"sample": True})],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO test_runs (
            run_id, test_name, test_type, started_at, completed_at, success, metadata
        ) VALUES (1, 'sample-run', 'integration', ?, ?, TRUE, ?)
        """,
        [now, now, json.dumps({"sample": True})],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO hallucinate_app_mobile_handoffs (
            handoff_id, request_id, query, ipfs_cids, status
        ) VALUES (1, 'sample-handoff', 'cid:QmDemo mobile', ?, 'acknowledged')
        """,
        [json.dumps(["QmDemo"])],
    )


def create_schema(conn: Any, force: bool = False, sample_data: bool = False) -> None:
    create_common_tables(conn, force=force)
    create_performance_tables(conn, force=force)
    create_hardware_compatibility_tables(conn, force=force)
    create_integration_test_tables(conn, force=force)
    create_hallucinate_app_mobile_tables(conn, force=force)
    create_views(conn)
    create_indexes(conn)
    if sample_data:
        insert_sample_data(conn)


def main() -> int:
    args = parse_args()
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, force=args.force, sample_data=args.sample_data)
        if args.verbose:
            print(f"Benchmark schema created at {args.output}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
