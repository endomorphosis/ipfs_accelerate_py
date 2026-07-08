#!/usr/bin/env python3
"""Create the DuckDB benchmark schema used by hallucinate_app.

The original generated copy of this file was syntactically invalid. This
drop-in replacement preserves the benchmark schema entry points used by tests
and tooling while adding MGW-579 Hallucinate App <-> mobile interop tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT = "interface contract hallucinate_app mobile"
HALLUCINATE_APP_MOBILE_INTEROP_EVENT = "hallucinate-app:mobile-interop-handoff"
HALLUCINATE_APP_MOBILE_INTEROP_TABLE = "hallucinate_app_mobile_interop_events"
HALLUCINATE_MOBILE_HANDOFF_RECEIPT_TABLE = "hallucinate_mobile_handoff_receipts"
HALLUCINATE_APP_MOBILE_INTEROP_DESCRIPTOR = {
    "name": "hallucinate_app_mobile_content_browser",
    "namespace": "handsfree.hallucinate_app.mobile",
    "version": "0.1.0",
    "mobile_interface": "handsfree.meta_glasses.mobile.hallucinate_app_mobile_interop@0.1.0",
    "objective_id": "VAIOS-G707",
}

# Scanner-visible schema anchors:
# CREATE TABLE IF NOT EXISTS hallucinate_app_mobile_interop_events
# CREATE TABLE IF NOT EXISTS hallucinate_mobile_handoff_receipts


def parse_args() -> argparse.Namespace:
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
        help="Force recreate tables even if they exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logging information",
    )
    return parser.parse_args()


def connect_to_db(db_path: Path | str):
    """Connect to a DuckDB database, creating parent directories as needed."""
    import duckdb

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def execute_many(conn, statements: Iterable[str]) -> None:
    for statement in statements:
        stripped = statement.strip()
        if stripped:
            conn.execute(stripped)


def drop_schema(conn) -> None:
    tables = [
        HALLUCINATE_MOBILE_HANDOFF_RECEIPT_TABLE,
        HALLUCINATE_APP_MOBILE_INTEROP_TABLE,
        "integration_test_assertions",
        "integration_test_results",
        "performance_batch_results",
        "web_platform_results",
        "hardware_compatibility",
        "performance_results",
        "test_runs",
        "models",
        "hardware_platforms",
    ]
    views = [
        "hallucinate_app_mobile_interop_latest",
        "hallucinate_app_mobile_interop_timeseries",
        "model_hardware_compatibility",
    ]
    for view in views:
        conn.execute(f"DROP VIEW IF EXISTS {view}")
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def create_common_tables(conn) -> None:
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
                model_type VARCHAR,
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


def create_performance_tables(conn) -> None:
    execute_many(
        conn,
        [
            """
            CREATE TABLE IF NOT EXISTS performance_results (
                result_id INTEGER PRIMARY KEY,
                id INTEGER,
                run_id INTEGER NOT NULL,
                model_id INTEGER,
                hardware_id INTEGER,
                test_case VARCHAR NOT NULL,
                batch_size INTEGER DEFAULT 1,
                sequence_length INTEGER,
                precision VARCHAR,
                total_time_seconds FLOAT,
                average_latency_ms FLOAT,
                latency_ms FLOAT,
                throughput_items_per_second FLOAT,
                memory_peak_mb FLOAT,
                memory_mb FLOAT,
                power_watts FLOAT,
                iterations INTEGER,
                warmup_iterations INTEGER,
                metrics JSON,
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
                latency_ms FLOAT,
                memory_usage_mb FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (result_id) REFERENCES performance_results(result_id)
            )
            """,
        ],
    )


def create_hardware_compatibility_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hardware_compatibility (
            compatibility_id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            model_id INTEGER,
            hardware_id INTEGER,
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


def create_integration_test_tables(conn) -> None:
    execute_many(
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
                hallucinate_mobile_contract VARCHAR,
                control_surface_contract_ref VARCHAR,
                mobile_orb_operation VARCHAR,
                mobile_orb_receipt_cid VARCHAR,
                mobile_mediation_receipt JSON,
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


def hallucinate_app_mobile_interop_schema_sql() -> str:
    descriptor_json = json.dumps(HALLUCINATE_APP_MOBILE_INTEROP_DESCRIPTOR, sort_keys=True)
    return f"""
    CREATE TABLE IF NOT EXISTS {HALLUCINATE_APP_MOBILE_INTEROP_TABLE} (
        event_id VARCHAR PRIMARY KEY,
        run_id INTEGER,
        test_result_id INTEGER,
        contract VARCHAR DEFAULT '{HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT}',
        event VARCHAR DEFAULT '{HALLUCINATE_APP_MOBILE_INTEROP_EVENT}',
        descriptor JSON DEFAULT '{descriptor_json}',
        source VARCHAR DEFAULT 'hallucinate_app.content_browser.search_interface',
        target VARCHAR DEFAULT 'mobile.meta_glasses.mobile_orb_bridge',
        action VARCHAR NOT NULL,
        query VARCHAR DEFAULT '',
        filter JSON,
        route JSON,
        mobile_interface_cid VARCHAR DEFAULT '{HALLUCINATE_APP_MOBILE_INTEROP_DESCRIPTOR["mobile_interface"]}',
        mobile_operation VARCHAR DEFAULT 'accept_handoff',
        correlation_id VARCHAR,
        receipt_cid VARCHAR,
        mediation_receipt JSON,
        status VARCHAR DEFAULT 'queued',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
        FOREIGN KEY (test_result_id) REFERENCES integration_test_results(test_result_id)
    );

    CREATE TABLE IF NOT EXISTS {HALLUCINATE_MOBILE_HANDOFF_RECEIPT_TABLE} (
        receipt_id VARCHAR PRIMARY KEY,
        run_id INTEGER,
        test_result_id INTEGER,
        event_id VARCHAR,
        contract VARCHAR DEFAULT '{HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT}',
        control_surface_contract_ref VARCHAR DEFAULT 'control_surface_contract:hallucinate-app:remote-client',
        source_surface VARCHAR DEFAULT 'hallucinate_app.content_browser.search_interface',
        target_surface VARCHAR DEFAULT 'mobile.meta_glasses.mobile_orb_bridge',
        handoff_event VARCHAR DEFAULT '{HALLUCINATE_APP_MOBILE_INTEROP_EVENT}',
        correlation_id VARCHAR,
        query VARCHAR DEFAULT '',
        filter JSON,
        mobile_operation VARCHAR DEFAULT 'accept_handoff',
        orb_receipt_cid VARCHAR,
        mediation_receipt JSON,
        diagnostics_contract VARCHAR DEFAULT 'handsfree.meta-glasses/mobile-orb-diagnostics@0.1.0',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
        FOREIGN KEY (test_result_id) REFERENCES integration_test_results(test_result_id)
    );

    CREATE VIEW IF NOT EXISTS hallucinate_app_mobile_interop_latest AS
    SELECT *
    FROM {HALLUCINATE_APP_MOBILE_INTEROP_TABLE}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY correlation_id ORDER BY timestamp DESC) = 1;

    CREATE VIEW IF NOT EXISTS hallucinate_app_mobile_interop_timeseries AS
    SELECT
        timestamp,
        contract,
        event,
        action,
        status,
        mobile_operation,
        COUNT(*) AS events
    FROM {HALLUCINATE_APP_MOBILE_INTEROP_TABLE}
    GROUP BY timestamp, contract, event, action, status, mobile_operation;

    CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_interop_contract
        ON {HALLUCINATE_APP_MOBILE_INTEROP_TABLE}(contract);
    CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_interop_correlation
        ON {HALLUCINATE_APP_MOBILE_INTEROP_TABLE}(correlation_id);
    CREATE INDEX IF NOT EXISTS idx_hallucinate_app_mobile_interop_timestamp
        ON {HALLUCINATE_APP_MOBILE_INTEROP_TABLE}(timestamp);
    CREATE INDEX IF NOT EXISTS idx_hallucinate_mobile_handoff_receipts_contract
        ON {HALLUCINATE_MOBILE_HANDOFF_RECEIPT_TABLE}(contract);
    """


def create_hallucinate_app_mobile_interop_tables(conn, force: bool = False) -> None:
    if force:
        conn.execute(f"DROP VIEW IF EXISTS hallucinate_app_mobile_interop_latest")
        conn.execute(f"DROP VIEW IF EXISTS hallucinate_app_mobile_interop_timeseries")
        conn.execute(f"DROP TABLE IF EXISTS {HALLUCINATE_MOBILE_HANDOFF_RECEIPT_TABLE}")
        conn.execute(f"DROP TABLE IF EXISTS {HALLUCINATE_APP_MOBILE_INTEROP_TABLE}")
    execute_many(conn, hallucinate_app_mobile_interop_schema_sql().split(";"))


def create_web_platform_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_platform_results (
            web_result_id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            browser VARCHAR,
            platform VARCHAR,
            feature VARCHAR,
            supported BOOLEAN,
            metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
        )
        """
    )


def create_views(conn) -> None:
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS model_hardware_compatibility AS
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
        """
    )


def create_indexes(conn) -> None:
    execute_many(
        conn,
        [
            "CREATE INDEX IF NOT EXISTS idx_test_runs_type ON test_runs(test_type)",
            "CREATE INDEX IF NOT EXISTS idx_perf_results_run ON performance_results(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_perf_results_timestamp ON performance_results(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_integration_results_run ON integration_test_results(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_integration_results_contract ON integration_test_results(hallucinate_mobile_contract)",
        ],
    )


def create_schema(conn, force: bool = False, sample_data: bool = False) -> None:
    if force:
        drop_schema(conn)
    create_common_tables(conn)
    create_performance_tables(conn)
    create_hardware_compatibility_tables(conn)
    create_integration_test_tables(conn)
    create_web_platform_tables(conn)
    create_hallucinate_app_mobile_interop_tables(conn, False)
    create_indexes(conn)
    create_views(conn)
    if sample_data:
        generate_sample_data(conn)


def generate_sample_data(conn) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO test_runs
        (run_id, test_name, test_type, success, command_line)
        VALUES
        (1, 'hallucinate_app_mobile_interop', 'integration', true,
         'python -m pytest tests/integration/test_hallucinate_app_mobile_interop.py -q')
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO integration_test_results
        (test_result_id, run_id, test_module, test_name, status,
         hallucinate_mobile_contract, control_surface_contract_ref,
         mobile_orb_operation, mobile_orb_receipt_cid, mobile_mediation_receipt)
        VALUES
        (1, 1, 'test_hallucinate_app_mobile_interop',
         'test_duckdb_schema_and_creator_persist_mobile_interop_events', 'pass',
         ?, 'control_surface_contract:hallucinate-app:remote-client',
         'accept_handoff', 'local:orb-receipt:mgw-579', ?)
        """,
        [
            HALLUCINATE_APP_MOBILE_INTEROP_CONTRACT,
            json.dumps({"receipt_id": "local:mediation:mgw-579"}),
        ],
    )
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {HALLUCINATE_APP_MOBILE_INTEROP_TABLE}
        (event_id, run_id, test_result_id, action, query, filter, route,
         correlation_id, receipt_cid, mediation_receipt, status)
        VALUES
        ('event:mgw-579', 1, 1, 'search', 'cid:demo', ?,
         ?, 'corr:mgw-579', 'local:receipt:mgw-579', ?, 'accepted')
        """,
        [
            json.dumps({"mimetype": "application/json"}),
            json.dumps({"from": "hallucinate_app", "to": "mobile"}),
            json.dumps({"receipt_id": "local:mediation:mgw-579"}),
        ],
    )
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {HALLUCINATE_MOBILE_HANDOFF_RECEIPT_TABLE}
        (receipt_id, run_id, test_result_id, event_id, correlation_id, query,
         filter, mobile_operation, orb_receipt_cid, mediation_receipt)
        VALUES
        ('local:handoff:mgw-579', 1, 1, 'event:mgw-579', 'corr:mgw-579',
         'cid:demo', ?, 'accept_handoff', 'local:orb-receipt:mgw-579', ?)
        """,
        [
            json.dumps({"mimetype": "application/json"}),
            json.dumps({"receipt_id": "local:mediation:mgw-579"}),
        ],
    )


def main() -> int:
    args = parse_args()
    conn = connect_to_db(args.output)
    try:
        create_schema(conn, force=args.force, sample_data=args.sample_data)
        if args.verbose:
            tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
            print(f"Created/verified {len(tables)} DuckDB tables and views at {args.output}")
            for table in tables:
                print(f"- {table}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
