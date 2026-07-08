#!/usr/bin/env python3
"""Create the DuckDB benchmark schema used by Hallucinate App integration gates.

The schema stores benchmark results, hardware compatibility checks, web-platform
results, integration-test assertions, and VAIOS-G707 Hallucinate App <-> mobile
handoff receipts.  It is intentionally dependency-light so integration tests can
import and execute it in a temporary database.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable

import duckdb


HALLUCINATE_MOBILE_HANDOFF_CONTRACT = (
    "handsfree.hallucinate-app/mobile-search-handoff@0.1.0"
)
HALLUCINATE_MOBILE_CONTROL_SURFACE_CONTRACT_REF = (
    "control_surface_contract:hallucinate-app:remote-client"
)
HALLUCINATE_MOBILE_DIAGNOSTICS_CONTRACT = (
    "handsfree.meta-glasses/mobile-orb-diagnostics@0.1.0"
)


DROP_ORDER = [
    "hallucinate_mobile_handoff_receipts",
    "integration_test_assertions",
    "integration_test_results",
    "webgpu_advanced_features",
    "web_platform_results",
    "hardware_compatibility",
    "performance_batch_results",
    "performance_results",
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
        help="Insert a compact sample dataset after creating the schema.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop known schema objects before recreating them.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print schema progress.")
    return parser.parse_args()


def connect_to_db(db_path: Path | str) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def execute_many(conn: duckdb.DuckDBPyConnection, statements: Iterable[str]) -> None:
    for statement in statements:
        conn.execute(statement)


def drop_schema(conn: duckdb.DuckDBPyConnection) -> None:
    execute_many(conn, ["DROP VIEW IF EXISTS hallucinate_mobile_handoff_history"])
    for table in DROP_ORDER:
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


def create_performance_tables(conn: duckdb.DuckDBPyConnection) -> None:
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
                precision VARCHAR,
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
        ],
    )


def create_hardware_compatibility_tables(conn: duckdb.DuckDBPyConnection) -> None:
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


def create_integration_test_tables(conn: duckdb.DuckDBPyConnection) -> None:
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
                execution_time_seconds DOUBLE,
                hardware_id INTEGER,
                model_id INTEGER,
                error_message VARCHAR,
                error_traceback VARCHAR,
                metadata JSON,
                hallucinate_mobile_contract VARCHAR,
                control_surface_contract_ref VARCHAR,
                mobile_orb_operation VARCHAR,
                mobile_orb_receipt_cid VARCHAR,
                mobile_mediation_receipt JSON,
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
            f"""
            CREATE TABLE IF NOT EXISTS hallucinate_mobile_handoff_receipts (
                receipt_id VARCHAR PRIMARY KEY,
                run_id INTEGER,
                test_result_id INTEGER,
                contract VARCHAR NOT NULL DEFAULT '{HALLUCINATE_MOBILE_HANDOFF_CONTRACT}',
                control_surface_contract_ref VARCHAR NOT NULL DEFAULT '{HALLUCINATE_MOBILE_CONTROL_SURFACE_CONTRACT_REF}',
                source_surface VARCHAR NOT NULL,
                target_surface VARCHAR NOT NULL,
                handoff_event VARCHAR NOT NULL DEFAULT 'hallucinate-app:mobile-search-handoff',
                correlation_id VARCHAR NOT NULL,
                query TEXT,
                filter JSON,
                mobile_operation VARCHAR NOT NULL,
                orb_receipt_cid VARCHAR,
                mediation_receipt JSON,
                diagnostics_contract VARCHAR DEFAULT '{HALLUCINATE_MOBILE_DIAGNOSTICS_CONTRACT}',
                status VARCHAR NOT NULL DEFAULT 'recorded',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
                FOREIGN KEY (test_result_id) REFERENCES integration_test_results(test_result_id)
            )
            """,
        ],
    )


def create_web_platform_tables(conn: duckdb.DuckDBPyConnection) -> None:
    execute_many(
        conn,
        [
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
        ],
    )


def create_indexes_and_views(conn: duckdb.DuckDBPyConnection) -> None:
    execute_many(
        conn,
        [
            "CREATE INDEX IF NOT EXISTS idx_perf_results_created_at ON performance_results(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_perf_results_version ON performance_results(version_tag)",
            "CREATE INDEX IF NOT EXISTS idx_perf_results_model_hw_time ON performance_results(model_id, hardware_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_perf_results_run_group ON performance_results(run_group_id)",
            "CREATE INDEX IF NOT EXISTS idx_integration_results_module ON integration_test_results(test_module, test_name)",
            "CREATE INDEX IF NOT EXISTS idx_hallucinate_mobile_receipts_correlation ON hallucinate_mobile_handoff_receipts(correlation_id)",
            "CREATE INDEX IF NOT EXISTS idx_hallucinate_mobile_receipts_contract ON hallucinate_mobile_handoff_receipts(contract, control_surface_contract_ref)",
            "CREATE INDEX IF NOT EXISTS idx_hallucinate_mobile_receipts_operation ON hallucinate_mobile_handoff_receipts(mobile_operation)",
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
            CREATE OR REPLACE VIEW hallucinate_mobile_handoff_history AS
            SELECT
                receipt_id,
                contract,
                control_surface_contract_ref,
                source_surface,
                target_surface,
                handoff_event,
                correlation_id,
                mobile_operation,
                orb_receipt_cid,
                diagnostics_contract,
                status,
                created_at
            FROM hallucinate_mobile_handoff_receipts
            ORDER BY created_at DESC
            """,
        ],
    )


def create_schema(
    conn: duckdb.DuckDBPyConnection,
    *,
    force: bool = False,
) -> None:
    if force:
        drop_schema(conn)
    create_common_tables(conn)
    create_performance_tables(conn)
    create_hardware_compatibility_tables(conn)
    create_integration_test_tables(conn)
    create_web_platform_tables(conn)
    create_indexes_and_views(conn)


def generate_sample_data(conn: duckdb.DuckDBPyConnection) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    conn.execute(
        """
        INSERT OR IGNORE INTO hardware_platforms
        (hardware_id, hardware_type, device_name, platform, metadata, created_at)
        VALUES (1, 'mobile', 'Meta glasses handset bridge', 'ios/android', ?, ?)
        """,
        [json.dumps({"surface": "mobile"}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO models
        (model_id, model_name, model_family, modality, source, metadata, created_at)
        VALUES (1, 'hallucinate-mobile-interop', 'contract', 'multimodal', 'hallucinate_app', ?, ?)
        """,
        [json.dumps({"goal": "VAIOS-G707"}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO test_runs
        (run_id, test_name, test_type, started_at, completed_at, execution_time_seconds,
         success, git_branch, command_line, metadata, created_at)
        VALUES (1, 'hallucinate_app_mobile_interop', 'integration', ?, ?, 0.1, true,
                'local', 'python -m pytest tests/integration -q', ?, ?)
        """,
        [now, now, json.dumps({"objective": "VAIOS-G707"}), now],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO integration_test_results
        (test_result_id, run_id, test_module, test_class, test_name, status,
         hardware_id, model_id, hallucinate_mobile_contract,
         control_surface_contract_ref, mobile_orb_operation,
         mobile_orb_receipt_cid, mobile_mediation_receipt, metadata, created_at)
        VALUES (1, 1, 'test_hallucinate_app_mobile_interop',
                'TestHallucinateAppMobileInterop',
                'test_duckdb_schema_records_handoff_receipts', 'pass',
                1, 1, ?, ?, 'render_widget',
                'sha256:sample-mobile-orb-receipt', ?, ?, ?)
        """,
        [
            HALLUCINATE_MOBILE_HANDOFF_CONTRACT,
            HALLUCINATE_MOBILE_CONTROL_SURFACE_CONTRACT_REF,
            json.dumps({"receipt_id": "sha256:sample-mediation-receipt"}),
            json.dumps({"objective": "VAIOS-G707"}),
            now,
        ],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO hallucinate_mobile_handoff_receipts
        (receipt_id, run_id, test_result_id, source_surface, target_surface,
         correlation_id, query, filter, mobile_operation, orb_receipt_cid,
         mediation_receipt, created_at)
        VALUES ('sha256:sample-hallucinate-mobile-handoff', 1, 1,
                'hallucinate_app.content_browser.search_interface',
                'mobile.meta_glasses.mobile_orb_bridge',
                'sample-correlation', 'cid:sample', ?, 'render_widget',
                'sha256:sample-mobile-orb-receipt', ?, ?)
        """,
        [
            json.dumps({"mimetype": "application/json"}),
            json.dumps({"receipt_id": "sha256:sample-mediation-receipt"}),
            now,
        ],
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
            print(f"Created {len(tables)} DuckDB tables/views in {args.output}:")
            for table in tables:
                print(f"  - {table}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
